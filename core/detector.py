"""YOLO object detection and identity assignment.

Two responsibilities, deliberately kept together because Ultralytics couples
them: run the network, and ask its built-in tracker (ByteTrack or BoT-SORT)
to attach a stable id to each box. The richer notion of a track - trajectory,
speed, dwell time, zone history - lives in :mod:`core.tracker`, which consumes
what this module produces.

Two rules this module follows strictly:

1. **Class names come from the loaded model, never from a hard-coded table.**
   The previous generation of this project mapped class ids through a fixed
   7-entry dictionary. Load a different model and a COCO ``car`` silently
   became a ``crowd``. Here, names are read from ``model.names``, and a class
   requested in the config that the model does not provide is reported at
   startup and dropped. The application never claims to detect something the
   weights cannot detect.

2. **A missing model file is an error, not a silent substitution.** Falling
   back to different weights without saying so produces confident nonsense.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from core.inference_gate import inference_gate
from database.models import Detection

logger = logging.getLogger(__name__)


class DetectorError(RuntimeError):
    """Raised when the model cannot be loaded or inference fails fatally."""


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

@dataclass
class DeviceInfo:
    """What hardware inference will actually run on, and why."""

    device: str
    label: str
    reason: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.label} ({self.device})"


def detect_device(preference: str = "auto") -> DeviceInfo:
    """Choose an inference device.

    ``auto`` prefers CUDA, then Apple Silicon MPS, then CPU. An explicit
    preference is honoured when it is genuinely available and falls back with
    an explanation when it is not. CUDA is never required.
    """
    pref = (preference or "auto").strip().lower()

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return DeviceInfo("cpu", "CPU", "PyTorch is not installed")

    cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    mps_ok = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )

    def cuda_info() -> DeviceInfo:
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover
            name = "CUDA GPU"
        return DeviceInfo("cuda:0", name, "CUDA GPU available")

    if pref.startswith("cuda"):
        if cuda_ok:
            return cuda_info()
        return DeviceInfo("cpu", "CPU", "CUDA was requested but is not available")

    if pref == "mps":
        if mps_ok:
            return DeviceInfo("mps", "Apple Silicon GPU (MPS)", "MPS requested")
        return DeviceInfo("cpu", "CPU", "MPS was requested but is not available")

    if pref == "cpu":
        return DeviceInfo("cpu", "CPU", "CPU explicitly selected")

    # auto
    if cuda_ok:
        return cuda_info()
    if mps_ok:
        return DeviceInfo("mps", "Apple Silicon GPU (MPS)",
                          "Apple Silicon detected, MPS backend available")
    return DeviceInfo("cpu", "CPU", "no GPU backend available")


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

@dataclass
class DetectorStats:
    """Rolling inference statistics for the HUD and camera health panel."""

    frames: int = 0
    detections: int = 0
    total_inference_ms: float = 0.0
    last_inference_ms: float = 0.0
    _recent: list[float] = field(default_factory=list)

    def record(self, inference_ms: float, count: int) -> None:
        self.frames += 1
        self.detections += count
        self.total_inference_ms += inference_ms
        self.last_inference_ms = inference_ms
        self._recent.append(inference_ms)
        if len(self._recent) > 60:
            self._recent.pop(0)

    @property
    def avg_inference_ms(self) -> float:
        if not self._recent:
            return 0.0
        return sum(self._recent) / len(self._recent)

    @property
    def inference_fps(self) -> float:
        avg = self.avg_inference_ms
        return 1000.0 / avg if avg > 0 else 0.0


class ObjectDetector:
    """Thin, honest wrapper around an Ultralytics YOLO model.

    Parameters
    ----------
    model_path:
        Weights file. Must exist, unless it is a bare Ultralytics alias such
        as ``yolov8n.pt`` that the library is allowed to fetch.
    confidence, iou, image_size:
        Standard YOLO inference parameters.
    device:
        ``auto``, ``cpu``, ``mps`` or ``cuda:0``.
    class_filter:
        Class *names* to keep. Names the model does not know are dropped with
        a warning and listed in :attr:`unsupported_classes`.
    tracker:
        ``bytetrack`` or ``botsort``.
    allow_download:
        Whether a missing bare alias may be downloaded. Off means fully
        offline operation, and a missing file is a clear error.
    """

    _ALIASES = {
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    }

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.35,
        iou: float = 0.50,
        image_size: int = 640,
        device: str = "auto",
        class_filter: Optional[Iterable[str]] = None,
        tracker: str = "bytetrack",
        half: bool = False,
        allow_download: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)
        self.tracker_name = str(tracker).lower()
        self.half = bool(half)
        self.stats = DetectorStats()

        self.device_info = detect_device(device)
        self.device = self.device_info.device

        self._model = self._load(allow_download)
        self.class_names: dict[int, str] = dict(self._model.names)

        self.requested_classes = [str(c) for c in (class_filter or [])]
        self.class_ids, self.unsupported_classes = self._resolve_classes(
            self.requested_classes
        )
        self._tracking_started = False

        logger.info(
            "Detector ready | %s | %s | conf=%.2f iou=%.2f imgsz=%d | "
            "%d classes known, %d active",
            self.model_path.name, self.device_info, self.confidence,
            self.iou, self.image_size, len(self.class_names),
            len(self.class_ids) if self.class_ids else len(self.class_names),
        )
        if self.unsupported_classes:
            logger.warning(
                "These configured classes are NOT supported by %s and have "
                "been dropped: %s. The model knows: %s",
                self.model_path.name,
                ", ".join(self.unsupported_classes),
                ", ".join(sorted(self.class_names.values())[:25]),
            )

    # -- loading ---------------------------------------------------------

    @staticmethod
    def _limit_threads() -> int:
        """Stop PyTorch from taking every core.

        Inference is CPU-bound and PyTorch defaults to one thread per core. On
        a laptop that starves the interface thread, and the symptom is a
        window that stops repainting while a video is being analysed. Leaving
        one core free costs a little throughput and buys a usable application.
        """
        import os

        try:
            import torch
        except ImportError:  # pragma: no cover
            return 0

        cores = os.cpu_count() or 2
        threads = max(1, cores - 1)
        try:
            torch.set_num_threads(threads)
        except Exception:  # pragma: no cover - platform dependent
            return 0
        logger.info("PyTorch limited to %d of %d CPU threads so the interface "
                    "keeps a core", threads, cores)
        return threads

    def _load(self, allow_download: bool) -> Any:
        self._limit_threads()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError(
                "The 'ultralytics' package is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc

        target: str
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
            raise DetectorError(
                f"Model weights not found: {self.model_path}\n"
                f"Put a .pt file in the models/ folder and set "
                f"detection.model_path in config/application.yaml, or choose "
                f"one from the Settings screen. No substitute model will be "
                f"loaded silently, because that would relabel every detection."
            )

        try:
            model = YOLO(target)
        except Exception as exc:
            raise DetectorError(
                f"Failed to load {target}: {exc}\n"
                f"The file may be corrupt, or built for an incompatible "
                f"Ultralytics version."
            ) from exc

        if not getattr(model, "names", None):
            raise DetectorError(
                f"{target} loaded but exposes no class names. It does not look "
                f"like a detection model."
            )
        return model

    def _resolve_classes(
        self, requested: list[str]
    ) -> tuple[Optional[list[int]], list[str]]:
        """Map requested class names to ids, reporting any the model lacks."""
        if not requested:
            return None, []

        lookup = {name.lower(): idx for idx, name in self.class_names.items()}
        ids: list[int] = []
        missing: list[str] = []
        for name in requested:
            idx = lookup.get(name.strip().lower())
            if idx is None:
                missing.append(name)
            else:
                ids.append(idx)
        return (sorted(set(ids)) or None), missing

    # -- introspection ---------------------------------------------------

    @property
    def supported_class_names(self) -> list[str]:
        return [self.class_names[k] for k in sorted(self.class_names)]

    @property
    def active_class_names(self) -> list[str]:
        if self.class_ids is None:
            return self.supported_class_names
        return [self.class_names[i] for i in self.class_ids]

    def supports(self, name: str) -> bool:
        return name.strip().lower() in {n.lower() for n in self.class_names.values()}

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_path.name,
            "model_path": str(self.model_path),
            "device": self.device,
            "device_label": self.device_info.label,
            "device_reason": self.device_info.reason,
            "confidence": self.confidence,
            "iou": self.iou,
            "image_size": self.image_size,
            "tracker": self.tracker_name,
            "class_count": len(self.class_names),
            "active_classes": self.active_class_names,
            "unsupported_requested": self.unsupported_classes,
        }

    # -- inference -------------------------------------------------------

    def _parse(self, result: Any) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)
        ids = (
            boxes.id.cpu().numpy().astype(int)
            if getattr(boxes, "id", None) is not None
            else None
        )

        out: list[Detection] = []
        for i in range(len(xyxy)):
            class_id = int(clses[i])
            out.append(
                Detection(
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, f"class_{class_id}"),
                    confidence=float(confs[i]),
                    x1=float(xyxy[i][0]),
                    y1=float(xyxy[i][1]),
                    x2=float(xyxy[i][2]),
                    y2=float(xyxy[i][3]),
                    track_number=int(ids[i]) if ids is not None else None,
                )
            )
        return out

    def _common_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.image_size,
            "device": self.device,
            "verbose": False,
        }
        if self.class_ids:
            kwargs["classes"] = self.class_ids
        # Half precision is a CUDA-only optimisation; forcing it elsewhere
        # produces dtype errors rather than speed.
        if self.half and self.device.startswith("cuda"):
            kwargs["half"] = True
        return kwargs

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect objects in one frame. No identities are assigned."""
        started = time.perf_counter()
        try:
            # Same gate as track(). This path is the fallback used when
            # tracking raises, so it can run from a camera thread too.
            with inference_gate(self.device):
                results = self._model.predict(image, **self._common_kwargs())
        except Exception as exc:
            logger.error("Inference failed: %s", exc, exc_info=True)
            raise DetectorError(f"Inference failed: {exc}") from exc

        elapsed = (time.perf_counter() - started) * 1000.0
        detections = self._parse(results[0]) if results else []
        self.stats.record(elapsed, len(detections))
        return detections

    def track(self, image: np.ndarray) -> list[Detection]:
        """Detect and assign persistent ids using the configured tracker.

        ``persist=True`` keeps tracker state between calls, which is what
        stops a new identity being created for the same person every frame.
        """
        started = time.perf_counter()
        kwargs = self._common_kwargs()
        kwargs["persist"] = True
        kwargs["tracker"] = f"{self.tracker_name}.yaml"

        try:
            # Serialised on MPS only. Two threads encoding Metal work at
            # once aborts the process; see core/inference_gate.py.
            with inference_gate(self.device):
                results = self._model.track(image, **kwargs)
        except Exception as exc:
            logger.error(
                "Tracking failed (%s); falling back to detection without ids "
                "for this frame", exc,
            )
            return self.detect(image)

        elapsed = (time.perf_counter() - started) * 1000.0
        detections = self._parse(results[0]) if results else []
        self.stats.record(elapsed, len(detections))
        self._tracking_started = True
        return detections

    def reset_tracker(self) -> None:
        """Forget tracker state, so a new video starts numbering from 1."""
        try:
            predictor = getattr(self._model, "predictor", None)
            trackers = getattr(predictor, "trackers", None) if predictor else None
            if trackers:
                for tracker in trackers:
                    if hasattr(tracker, "reset"):
                        tracker.reset()
                logger.debug("Tracker state reset")
        except Exception as exc:  # pragma: no cover - version dependent
            logger.debug("Could not reset tracker state: %s", exc)
        self._tracking_started = False
