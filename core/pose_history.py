"""Remember what each tracked person's body has been doing.

A single pose is a shape. A behaviour is a shape changing over time, so
classifying one from one frame is guesswork dressed as analysis: a person
mid-stride looks momentarily like a person standing on one leg, and a person
bending to pick something up passes through the same silhouette as a person
crouching to hide. Only the sequence separates them.

This module is the memory the rest of the chain reads from:

    PoseEstimator  ->  PoseHistory  ->  MovementAnalyzer  ->  BehaviourClassifier

It stores samples per Track ID, keyed by the identity ByteTrack already
assigned, so there is one id space and no reconciliation step. It holds a short
window rather than everything, because behaviour is a recent-past question and
an unbounded buffer is a slow memory leak on a long-running camera.

It also keeps a per-track reference height. A person's size in frame depends on
how far they are from the camera, so a raw measurement in frame units cannot be
compared between two people or between the same person at two distances.
Dividing by that person's own standing height makes every measurement here
roughly scale-free, which is what lets one set of thresholds work on any
camera without being tuned per camera.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np

from core.pose import KEYPOINT_INDEX

#: Pose could not be attempted or produced nothing usable this frame.
POSE_UNAVAILABLE = "UNAVAILABLE"
#: Pose ran and returned landmarks.
POSE_TRACKED = "TRACKED"
#: Pose ran recently but not on this frame; the last landmarks still stand.
POSE_STALE = "STALE"


def _midpoint(result, a: str, b: str) -> Optional[tuple[float, float]]:
    """Midpoint of two named keypoints, or whichever one is visible."""
    pa, pb = result.point(a), result.point(b)
    if pa and pb:
        return ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
    return pa or pb


@dataclass
class PoseSample:
    """One moment in a person's history, reduced to what behaviour needs.

    The raw 17 keypoints are kept, but the derived scalars are computed once
    here rather than repeatedly downstream. Every geometric value is in
    normalised frame coordinates, y increasing downward.
    """

    ts: float
    #: (17, 2) normalised keypoints, and (17,) visibility.
    keypoints: np.ndarray
    visible: np.ndarray
    #: Normalised box of the track at this moment.
    box: tuple[float, float, float, float]

    head: Optional[tuple[float, float]] = None
    shoulders: Optional[tuple[float, float]] = None
    hips: Optional[tuple[float, float]] = None
    knees: Optional[tuple[float, float]] = None
    ankles: Optional[tuple[float, float]] = None
    wrists: Optional[tuple[float, float]] = None

    #: Vertical extent from the highest confident upper point to the feet.
    body_height: Optional[float] = None
    #: Degrees the shoulder-to-hip vector leans from vertical, 0 to 90.
    torso_angle: Optional[float] = None
    #: Hip-knee-ankle angle in degrees, 180 straight, smaller = more bent.
    knee_angle: Optional[float] = None
    #: How many keypoints this sample is based on.
    support: int = 0

    @classmethod
    def from_result(cls, result, box, ts: float) -> "PoseSample":
        head = _midpoint(result, "left_ear", "right_ear") or result.point("nose")
        shoulders = _midpoint(result, "left_shoulder", "right_shoulder")
        hips = _midpoint(result, "left_hip", "right_hip")
        knees = _midpoint(result, "left_knee", "right_knee")
        ankles = _midpoint(result, "left_ankle", "right_ankle")
        wrists = _midpoint(result, "left_wrist", "right_wrist")

        top = head or shoulders
        bottom = ankles or knees
        body_height = None
        if top and bottom:
            body_height = abs(bottom[1] - top[1])
            if body_height < 1e-4:
                body_height = None

        torso_angle = None
        if shoulders and hips:
            dx = hips[0] - shoulders[0]
            dy = hips[1] - shoulders[1]
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                torso_angle = float(math.degrees(math.atan2(abs(dx), abs(dy))))

        knee_angle = None
        if hips and knees and ankles:
            knee_angle = _angle(hips, knees, ankles)

        return cls(
            ts=float(ts),
            keypoints=result.keypoints,
            visible=result.visible,
            box=tuple(box),
            head=head, shoulders=shoulders, hips=hips,
            knees=knees, ankles=ankles, wrists=wrists,
            body_height=body_height,
            torso_angle=torso_angle,
            knee_angle=knee_angle,
            support=result.visible_count,
        )

    @property
    def anchor(self) -> Optional[tuple[float, float]]:
        """The point used to measure this person's movement.

        Hips, because they sit near the centre of mass and move less than the
        limbs, so horizontal speed measured from them is not dominated by the
        swing of an arm.
        """
        return self.hips or self.shoulders or self.head


def _angle(a: tuple[float, float], b: tuple[float, float],
           c: tuple[float, float]) -> Optional[float]:
    """Interior angle at b, in degrees."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return float(math.degrees(math.acos(cos)))


class TrackPoseHistory:
    """The recent pose history of ONE tracked person."""

    def __init__(self, track_number: int, window_seconds: float = 3.0,
                 max_samples: int = 90) -> None:
        self.track_number = int(track_number)
        self.window_seconds = float(window_seconds)
        self.samples: Deque[PoseSample] = deque(maxlen=int(max_samples))
        #: Robust estimate of this person's standing height in frame units.
        self._height_samples: Deque[float] = deque(maxlen=int(max_samples))
        self._reference_height: Optional[float] = None

    def add(self, sample: PoseSample) -> None:
        self.samples.append(sample)
        if sample.body_height:
            self._height_samples.append(sample.body_height)
            self._reference_height = None  # recompute lazily
        self._prune(sample.ts)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.samples and self.samples[0].ts < cutoff:
            self.samples.popleft()

    @property
    def reference_height(self) -> Optional[float]:
        """This person's standing height in frame units.

        The 80th percentile of every height seen, not the maximum. The maximum
        is one bad frame away from being wrong: a single mis-placed ankle
        keypoint stretches the skeleton and would inflate the reference for
        the rest of the track's life, after which every genuine crouch would
        measure as normal.
        """
        if self._reference_height is None and self._height_samples:
            self._reference_height = float(
                np.percentile(np.asarray(self._height_samples), 80))
        return self._reference_height

    @property
    def duration(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].ts - self.samples[0].ts

    @property
    def latest(self) -> Optional[PoseSample]:
        return self.samples[-1] if self.samples else None

    def since(self, seconds: float) -> list[PoseSample]:
        if not self.samples:
            return []
        cutoff = self.samples[-1].ts - seconds
        return [s for s in self.samples if s.ts >= cutoff]


class PoseHistory:
    """Every tracked person's pose history, keyed by Track ID."""

    def __init__(self, window_seconds: float = 3.0,
                 max_samples: int = 90) -> None:
        self.window_seconds = float(window_seconds)
        self.max_samples = int(max_samples)
        self._tracks: dict[int, TrackPoseHistory] = {}

    def record(self, track_number: int, result, box, ts: float) -> PoseSample:
        history = self._tracks.get(track_number)
        if history is None:
            history = TrackPoseHistory(track_number, self.window_seconds,
                                       self.max_samples)
            self._tracks[track_number] = history
        sample = PoseSample.from_result(result, box, ts)
        history.add(sample)
        return sample

    def get(self, track_number: int) -> Optional[TrackPoseHistory]:
        return self._tracks.get(track_number)

    def forget(self, track_number: int) -> None:
        """Drop a finished track. Called when the tracker retires it."""
        self._tracks.pop(track_number, None)

    def prune(self, live_track_numbers: set[int]) -> None:
        """Drop histories for tracks that no longer exist.

        Without this the dictionary grows for the life of the process, which
        on a camera running for days is a slow leak.
        """
        for number in [n for n in self._tracks if n not in live_track_numbers]:
            del self._tracks[number]

    def __len__(self) -> int:
        return len(self._tracks)
