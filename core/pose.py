"""Body keypoints for tracked people.

What this module is, precisely: it takes a person whose box and identity are
already known, crops that box out of the frame, and asks a pose model where
that person's joints are. Nothing more. It does not decide who the person is,
it does not decide whether anything is wrong, and it does not detect people.
Detection says WHERE, tracking says WHICH, pose says WHAT SHAPE. Behaviour and
threat stay where they already are, working over time, and pose is one more
input available to them rather than a replacement for them.

Why crops rather than whole frames
----------------------------------
Running a pose model over the full frame would be a second detector: it would
find its own people, with its own boxes and its own ids, and then something
would have to reconcile those with ByteTrack's. That is duplicated work and a
new class of bug, where the pose model and the tracker disagree about how many
people are present. Cropping to a box the tracker already owns keeps one source
of identity. The crop is also small, so the model sees the person larger than
it would in a downscaled full frame, which helps on distant figures.

Why YOLO pose rather than MediaPipe
-----------------------------------
The project already runs Ultralytics for detection and tracking. A YOLO pose
model loads through the same library, takes the same device selection, honours
the same thread limit, and returns results in a shape the codebase already
parses. MediaPipe would add a second inference runtime, a second device story
and a second set of failure modes for the same 17 points. The keypoints
themselves are COCO-17 either way.

Cost control
------------
Pose is optional work, so it is bounded rather than best-effort:

* only confirmed person tracks are eligible, so a one-frame flicker never
  costs an inference;
* boxes below a minimum height are skipped, because a 20 pixel figure has no
  recoverable joints and the model will return noise with high confidence;
* a per-frame budget caps how many people are processed, largest first, so a
  crowd degrades predictably instead of stalling the pipeline;
* an interval lets pose run every Nth analysed frame. Posture changes over
  hundreds of milliseconds, not tens, so this loses very little.

Every one of those is configuration, not a constant buried in the code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from core.inference_gate import inference_gate

logger = logging.getLogger(__name__)


#: COCO-17 keypoint order, as produced by the Ultralytics pose models.
KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)
KEYPOINT_COUNT: int = len(KEYPOINT_NAMES)
KEYPOINT_INDEX: dict[str, int] = {n: i for i, n in enumerate(KEYPOINT_NAMES)}

#: Pairs to draw as bones. Purely for rendering.
SKELETON: tuple[tuple[int, int], ...] = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
)


class PoseError(RuntimeError):
    """Raised when the pose model cannot be loaded."""


@dataclass
class PoseResult:
    """Body keypoints for one tracked person, in frame coordinates.

    Coordinates are normalised 0-1 against the FULL frame, not the crop, so a
    consumer never needs to know a crop happened. ``visible`` is per keypoint:
    a joint the model could not see is present in the array but flagged, so
    downstream code can tell "arm is down" from "arm is hidden".
    """

    track_number: int
    ts: float
    #: (17, 2) normalised x, y against the full frame.
    keypoints: np.ndarray
    #: (17,) model confidence per keypoint.
    scores: np.ndarray
    #: (17,) True where the keypoint's score clears the threshold.
    visible: np.ndarray
    #: Mean score over the visible keypoints, 0 when none are visible.
    confidence: float = 0.0

    @property
    def visible_count(self) -> int:
        return int(np.count_nonzero(self.visible))

    def point(self, name: str) -> Optional[tuple[float, float]]:
        """A named keypoint, or None when it is not visible."""
        index = KEYPOINT_INDEX.get(name)
        if index is None or not bool(self.visible[index]):
            return None
        return float(self.keypoints[index][0]), float(self.keypoints[index][1])

    def to_dict(self) -> dict:
        """Compact, JSON-safe form for the database and the API."""
        return {
            "track_number": int(self.track_number),
            "ts": float(self.ts),
            "confidence": round(float(self.confidence), 3),
            "visible_count": self.visible_count,
            "keypoints": {
                name: {
                    "x": round(float(self.keypoints[i][0]), 4),
                    "y": round(float(self.keypoints[i][1]), 4),
                    "score": round(float(self.scores[i]), 3),
                    "visible": bool(self.visible[i]),
                }
                for i, name in enumerate(KEYPOINT_NAMES)
            },
        }


#: Posture labels. UNKNOWN is a real answer, not a failure.
POSTURE_UNKNOWN = "UNKNOWN"
POSTURE_UPRIGHT = "UPRIGHT"
POSTURE_BENT = "BENT"
POSTURE_HORIZONTAL = "HORIZONTAL"


@dataclass
class Posture:
    """A geometric description of one pose. NOT a behaviour, and not a verdict.

    This is trigonometry on a skeleton: how far the torso leans from vertical,
    and how tall the figure is relative to its width. It says the body is bent
    over. It does not say the person is hiding, digging, crawling or injured,
    and it must never be presented as if it did. Deciding what a posture means
    needs it held over time and read against the scene, which is the behaviour
    layer's job, not this one's.

    Two honest limits. One camera with no calibration cannot separate a person
    lying down from a person walking directly towards the lens, because both
    foreshorten the torso the same way. And a label is only offered when
    enough of the relevant joints are actually visible; otherwise it stays
    UNKNOWN rather than guessing from two points.
    """

    label: str = POSTURE_UNKNOWN
    #: Degrees away from vertical, 0 upright and 90 flat. None when unknown.
    torso_angle: Optional[float] = None
    #: Skeleton height divided by width. None when unknown.
    aspect: Optional[float] = None
    #: How many of the keypoints this reading is based on.
    support: int = 0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "torso_angle": (round(float(self.torso_angle), 1)
                            if self.torso_angle is not None else None),
            "aspect": (round(float(self.aspect), 2)
                       if self.aspect is not None else None),
            "support": int(self.support),
        }


def estimate_posture(result: PoseResult,
                     bent_degrees: float = 40.0,
                     horizontal_degrees: float = 65.0) -> Posture:
    """Describe the geometry of one pose.

    The torso vector runs from the midpoint of the shoulders to the midpoint
    of the hips. Its angle from vertical is the measurement everything else
    rests on, because it survives partial occlusion of the limbs, which is the
    normal case in surveillance footage.
    """
    def midpoint(a: str, b: str) -> Optional[tuple[float, float]]:
        pa, pb = result.point(a), result.point(b)
        if pa and pb:
            return ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
        return pa or pb

    shoulders = midpoint("left_shoulder", "right_shoulder")
    hips = midpoint("left_hip", "right_hip")
    support = result.visible_count

    aspect: Optional[float] = None
    if support >= 2:
        xs = result.keypoints[result.visible][:, 0]
        ys = result.keypoints[result.visible][:, 1]
        width = float(xs.max() - xs.min())
        height = float(ys.max() - ys.min())
        if width > 1e-6:
            aspect = height / width

    if shoulders is None or hips is None:
        return Posture(POSTURE_UNKNOWN, None, aspect, support)

    dx = hips[0] - shoulders[0]
    dy = hips[1] - shoulders[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return Posture(POSTURE_UNKNOWN, None, aspect, support)

    # Angle away from the vertical axis, 0 to 90.
    angle = float(np.degrees(np.arctan2(abs(dx), abs(dy))))

    if angle >= horizontal_degrees:
        label = POSTURE_HORIZONTAL
    elif angle >= bent_degrees:
        label = POSTURE_BENT
    else:
        label = POSTURE_UPRIGHT
    return Posture(label, angle, aspect, support)


@dataclass
class PoseStats:
    """Counters, so the cost of pose is visible rather than assumed."""

    frames_considered: int = 0
    crops_run: int = 0
    poses_found: int = 0
    skipped_too_small: int = 0
    skipped_invalid: int = 0
    skipped_budget: int = 0
    total_ms: float = 0.0
    last_ms: float = 0.0
    _recent: list[float] = field(default_factory=list)

    def record(self, ms: float, crops: int, found: int) -> None:
        self.last_ms = ms
        self.total_ms += ms
        self.crops_run += crops
        self.poses_found += found
        self._recent.append(ms)
        if len(self._recent) > 60:
            self._recent.pop(0)

    @property
    def avg_ms(self) -> float:
        return sum(self._recent) / len(self._recent) if self._recent else 0.0

    @property
    def ms_per_crop(self) -> float:
        return self.total_ms / self.crops_run if self.crops_run else 0.0


class PoseEstimator:
    """Run a pose model on crops of already-tracked people.

    Camera-agnostic by construction: nothing here knows a camera id exists.
    Each pipeline owns an instance built from the same config, so every camera
    behaves the same way and differs only in what its lens sees.
    """

    _ALIASES = {
        "yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt",
        "yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt",
    }

    def __init__(
        self,
        model_path: str | Path = "models/yolov8n-pose.pt",
        enabled: bool = True,
        device: str = "auto",
        image_size: int = 192,
        keypoint_threshold: float = 0.30,
        min_box_height: float = 0.12,
        min_crop_pixels: int = 48,
        max_persons_per_frame: int = 4,
        interval_frames: int = 2,
        allow_download: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.model_path = Path(model_path)
        self.image_size = int(image_size)
        self.keypoint_threshold = float(keypoint_threshold)
        self.min_box_height = float(min_box_height)
        self.min_crop_pixels = int(min_crop_pixels)
        self.max_persons_per_frame = max(1, int(max_persons_per_frame))
        self.interval_frames = max(1, int(interval_frames))
        self.stats = PoseStats()

        self._model: Any = None
        self._device = device
        self._device_label = "not loaded"
        self._error: Optional[str] = None

        if self.enabled:
            try:
                self._load(allow_download)
            except PoseError as exc:
                # A missing pose model disables pose. It must never take the
                # rest of the pipeline down with it, because detection,
                # tracking, zones and alerts do not depend on it.
                self._error = str(exc)
                self.enabled = False
                logger.error("Pose estimation disabled: %s", exc)

    # -- loading ---------------------------------------------------------

    def _load(self, allow_download: bool) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise PoseError(
                "The 'ultralytics' package is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc

        from core.detector import detect_device

        if self.model_path.exists():
            target = str(self.model_path)
        elif self.model_path.name in self._ALIASES and allow_download:
            logger.warning(
                "%s not found on disk. Ultralytics will download the stock "
                "'%s' weights (needs internet, once).",
                self.model_path, self.model_path.name,
            )
            target = self.model_path.name
        else:
            raise PoseError(
                f"Pose weights not found: {self.model_path}. Put a pose .pt "
                f"file in models/ and set pose.model_path in "
                f"config/application.yaml, or set pose.enabled to false."
            )

        try:
            model = YOLO(target)
        except Exception as exc:
            raise PoseError(f"Failed to load {target}: {exc}") from exc

        # A detection model loaded here would silently produce no keypoints
        # for ever. Refuse it rather than look broken later.
        task = getattr(model, "task", None)
        if task != "pose":
            raise PoseError(
                f"{target} is a '{task}' model, not a pose model. It would "
                f"never return keypoints."
            )

        info = detect_device(self._device)
        self._model = model
        self._device_label = info.label
        self._resolved_device = info.device
        logger.info(
            "Pose estimator ready | %s | %s | imgsz=%d | "
            "min box height %.0f%% of frame | max %d people per frame | "
            "every %d analysed frame(s)",
            self.model_path.name, info.label, self.image_size,
            self.min_box_height * 100, self.max_persons_per_frame,
            self.interval_frames,
        )

    # -- eligibility -----------------------------------------------------

    def should_run(self, analysed_frame_count: int) -> bool:
        """Whether this analysed frame is due for pose.

        Counts ANALYSED frames, not source frames. The distinction is not
        cosmetic: the pipeline already drops frames via ``frame_stride``, so
        testing the source index means the two dividers interact. With
        ``frame_stride: 2`` and ``interval_frames: 2``, every analysed frame
        has an even source index and the interval matched all of them, which
        made the knob do nothing at all. Counting analysed frames keeps the
        two settings independent, which is how they read in the config file.
        """
        if not self.enabled:
            return False
        return analysed_frame_count % self.interval_frames == 0

    @staticmethod
    def _valid_box(box_norm: Any) -> bool:
        """Whether a box is well formed, before asking whether it is big."""
        if not box_norm or len(box_norm) != 4:
            return False
        x1, y1, x2, y2 = box_norm
        if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
            return False
        return x2 > x1 and y2 > y1

    def eligible(self, box_norm: tuple[float, float, float, float],
                 frame_height: int) -> bool:
        """Whether one box is worth a pose inference.

        A box shorter than ``min_box_height`` of the frame, or smaller than
        ``min_crop_pixels`` once measured in pixels, is skipped. Both matter:
        the fraction keeps the rule resolution-independent, the pixel floor
        stops a 4K frame from passing a 30 pixel figure on a technicality.
        """
        _, y1, _, y2 = box_norm
        height_norm = max(0.0, y2 - y1)
        if height_norm < self.min_box_height:
            return False
        if height_norm * frame_height < self.min_crop_pixels:
            return False
        return True

    # -- inference -------------------------------------------------------

    def estimate(
        self,
        frame: np.ndarray,
        tracks: Iterable[Any],
        ts: float,
    ) -> dict[int, PoseResult]:
        """Estimate poses for the given tracks. Returns results by track id.

        ``tracks`` must already be filtered to confirmed people by the caller:
        this module does not decide what a person is, the detector's class
        names do. Tracks are processed largest first so that when the budget
        bites it drops the figures with least recoverable detail.
        """
        if not self.enabled or self._model is None:
            return {}

        frame_height, frame_width = frame.shape[:2]
        self.stats.frames_considered += 1

        candidates: list[tuple[float, Any, tuple[int, int, int, int]]] = []
        for track in tracks:
            box_norm = getattr(track, "box", None)
            # Shape first, then size. An inverted or non-finite box is
            # malformed, not small, and counting it as small would hide a
            # real fault behind a threshold that looks like it is working.
            if not self._valid_box(box_norm):
                self.stats.skipped_invalid += 1
                continue
            if not self.eligible(box_norm, frame_height):
                self.stats.skipped_too_small += 1
                continue

            pixels = self._to_pixels(box_norm, frame_width, frame_height)
            if pixels is None:
                self.stats.skipped_invalid += 1
                continue
            x1, y1, x2, y2 = pixels
            candidates.append((float((y2 - y1) * (x2 - x1)), track, pixels))

        if not candidates:
            return {}

        candidates.sort(key=lambda item: -item[0])
        if len(candidates) > self.max_persons_per_frame:
            self.stats.skipped_budget += (
                len(candidates) - self.max_persons_per_frame
            )
            candidates = candidates[:self.max_persons_per_frame]

        crops = []
        meta = []
        for _, track, (x1, y1, x2, y2) in candidates:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                self.stats.skipped_invalid += 1
                continue
            crops.append(crop)
            meta.append((track, x1, y1, x2 - x1, y2 - y1))

        if not crops:
            return {}

        started = time.perf_counter()
        try:
            # One batched call rather than one call per person: the same
            # number of crops, a single trip through the model.
            # Same gate as the detector, and the same reason: on MPS two
            # threads encoding Metal work at once aborts the process.
            with inference_gate(self._resolved_device):
                outputs = self._model.predict(
                    crops,
                    imgsz=self.image_size,
                    device=self._resolved_device,
                    verbose=False,
                )
        except Exception as exc:
            logger.error("Pose inference failed: %s", exc)
            self._error = str(exc)
            return {}
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        results: dict[int, PoseResult] = {}
        for output, (track, off_x, off_y, crop_w, crop_h) in zip(outputs, meta):
            parsed = self._parse(
                output, track, ts, off_x, off_y, crop_w, crop_h,
                frame_width, frame_height,
            )
            if parsed is not None:
                results[int(track.track_number)] = parsed

        self.stats.record(elapsed_ms, len(crops), len(results))
        return results

    # -- parsing ---------------------------------------------------------

    def _parse(
        self, output: Any, track: Any, ts: float,
        off_x: int, off_y: int, crop_w: int, crop_h: int,
        frame_width: int, frame_height: int,
    ) -> Optional[PoseResult]:
        """Turn one Ultralytics result into frame-normalised keypoints."""
        keypoints = getattr(output, "keypoints", None)
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return None

        # A crop can still contain more than one person, for example when two
        # people overlap. Take the one the model is most confident about,
        # which is the figure the box was drawn around.
        confidences = keypoints.conf
        if confidences is None:
            best = 0
        else:
            per_person = np.asarray(confidences).reshape(
                len(keypoints.xy), -1).mean(axis=1)
            best = int(np.argmax(per_person))

        xy = np.asarray(keypoints.xy[best], dtype=np.float32)
        if xy.shape != (KEYPOINT_COUNT, 2):
            return None
        scores = (np.asarray(confidences[best], dtype=np.float32)
                  if confidences is not None
                  else np.zeros(KEYPOINT_COUNT, dtype=np.float32))

        # Crop pixels -> full-frame pixels -> normalised full-frame.
        normalised = np.empty_like(xy)
        normalised[:, 0] = (xy[:, 0] + off_x) / float(frame_width)
        normalised[:, 1] = (xy[:, 1] + off_y) / float(frame_height)
        np.clip(normalised, 0.0, 1.0, out=normalised)

        visible = scores >= self.keypoint_threshold
        # A keypoint the model placed at the origin of the crop is a
        # placeholder for "not found", not a joint at the top-left corner.
        placeholder = (xy[:, 0] <= 0.0) & (xy[:, 1] <= 0.0)
        visible = visible & ~placeholder

        confidence = float(scores[visible].mean()) if visible.any() else 0.0

        return PoseResult(
            track_number=int(track.track_number),
            ts=float(ts),
            keypoints=normalised,
            scores=scores,
            visible=visible,
            confidence=confidence,
        )

    @staticmethod
    def _to_pixels(
        box_norm: tuple[float, float, float, float],
        width: int, height: int,
        pad: float = 0.04,
    ) -> Optional[tuple[int, int, int, int]]:
        """Normalised box to integer pixel bounds, padded and clamped.

        The pad matters. A tracker box is usually drawn tight, and a pose
        model given a tight crop tends to lose the hands and feet that sit on
        the boundary. A few percent of margin recovers them.
        """
        x1, y1, x2, y2 = box_norm
        if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
            return None
        if x2 <= x1 or y2 <= y1:
            return None

        pad_x = (x2 - x1) * pad
        pad_y = (y2 - y1) * pad
        px1 = int(round(max(0.0, x1 - pad_x) * width))
        py1 = int(round(max(0.0, y1 - pad_y) * height))
        px2 = int(round(min(1.0, x2 + pad_x) * width))
        py2 = int(round(min(1.0, y2 + pad_y) * height))

        if px2 - px1 < 8 or py2 - py1 < 8:
            return None
        return px1, py1, px2, py2

    # -- reporting -------------------------------------------------------

    def status_text(self) -> str:
        """One honest line for the status panels."""
        if self._error and not self.enabled:
            return f"Unavailable: {self._error.splitlines()[0]}"
        if not self.enabled:
            return "Disabled in config"
        if self.stats.crops_run == 0:
            return f"Ready ({self.model_path.name}, {self._device_label})"
        return (f"{self.stats.poses_found} poses from "
                f"{self.stats.crops_run} crops, "
                f"{self.stats.ms_per_crop:.0f} ms each")

    def describe(self) -> dict:
        return {
            "enabled": self.enabled,
            "model": self.model_path.name,
            "device": self._device_label,
            "image_size": self.image_size,
            "min_box_height": self.min_box_height,
            "max_persons_per_frame": self.max_persons_per_frame,
            "interval_frames": self.interval_frames,
            "error": self._error,
        }

    @classmethod
    def from_config(cls, config) -> "PoseEstimator":
        """Build from the ``pose`` section of application.yaml."""
        models_dir = config.path("models_dir")
        name = str(config.get("pose.model_file", "yolov8n-pose.pt"))
        return cls(
            model_path=models_dir / name,
            enabled=bool(config.get("pose.enabled", True)),
            device=str(config.get("pose.device",
                                  config.get("detection.device", "auto"))),
            image_size=int(config.get("pose.image_size", 192)),
            keypoint_threshold=float(
                config.get("pose.keypoint_threshold", 0.30)),
            min_box_height=float(config.get("pose.min_box_height", 0.12)),
            min_crop_pixels=int(config.get("pose.min_crop_pixels", 48)),
            max_persons_per_frame=int(
                config.get("pose.max_persons_per_frame", 4)),
            interval_frames=int(config.get("pose.interval_frames", 2)),
        )
