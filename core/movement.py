"""Turn a person's pose history into measurable movement signals.

This layer computes numbers. It does not decide what they mean: that is the
classifier's job, and keeping the two apart is what makes either of them
testable. Every signal here can be checked against a hand-built history
without loading a model.

The central idea is that raw frame units are useless for comparing people.
A person ten metres from the camera is half the height of the same person at
five metres, so "moved 0.02 of the frame this second" says nothing about how
fast they were going. Every distance here is therefore divided by that
person's own standing height, measured from their own history. The unit is
body heights per second, and it means roughly the same thing at any distance
and on any camera:

    walking   ~0.6 to 1.2 body heights per second
    running   ~2 and above

Those are physical facts about human gait (a walk is about 1.4 m/s against a
1.7 m frame, a run comfortably over 3 m/s), not thresholds tuned to one clip,
which is why the same numbers work without per-camera calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.pose_history import PoseSample, TrackPoseHistory


@dataclass
class MovementSignals:
    """Everything measured about one person's recent movement.

    ``ready`` is the honest gate: with too little history the answer is "not
    enough evidence yet", not a guess.
    """

    track_number: int
    ts: float
    ready: bool = False
    reason: str = "no history"

    #: Seconds of history these signals are based on.
    duration: float = 0.0
    samples: int = 0

    #: Standing height of this person in normalised frame units.
    reference_height: Optional[float] = None
    #: Current vertical extent divided by that reference. 1.0 = upright.
    height_ratio: Optional[float] = None
    #: Hip-to-ankle distance divided by its own upright reference.
    hip_ratio: Optional[float] = None

    #: Degrees from vertical, 0 upright and 90 flat.
    torso_angle: Optional[float] = None
    #: Hip-knee-ankle angle, 180 straight.
    knee_angle: Optional[float] = None

    #: Horizontal speed in body heights per second.
    speed: float = 0.0
    #: Vertical speed of the hips, body heights per second, POSITIVE = falling.
    vertical_speed: float = 0.0
    #: Largest downward vertical speed seen in the recent window.
    peak_fall_speed: float = 0.0
    #: Largest upward vertical speed seen in the recent window.
    peak_rise_speed: float = 0.0
    #: Change in height_ratio across the window, negative = getting lower.
    height_trend: float = 0.0

    #: Compass direction of travel, or None when effectively stationary.
    direction: Optional[str] = None

    #: Wrists below hip level.
    hands_low: bool = False
    #: Wrists above shoulder level.
    hands_raised: bool = False
    #: Ankles and wrists at a similar height, as when on all fours.
    limbs_level: bool = False

    #: How many keypoints the latest sample had.
    support: int = 0

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "reason": self.reason,
            "duration": round(self.duration, 2),
            "height_ratio": _round(self.height_ratio),
            "hip_ratio": _round(self.hip_ratio),
            "torso_angle": _round(self.torso_angle, 1),
            "knee_angle": _round(self.knee_angle, 1),
            "speed": round(self.speed, 2),
            "vertical_speed": round(self.vertical_speed, 2),
            "direction": self.direction,
            "support": self.support,
        }


def _round(value: Optional[float], places: int = 3) -> Optional[float]:
    return None if value is None else round(float(value), places)


def _direction(dx: float, dy: float) -> Optional[str]:
    """Compass direction of a movement vector. y is DOWN in image space."""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    # Screen up (negative dy) is north.
    angle = math.degrees(math.atan2(dx, -dy)) % 360.0
    names = ("NORTH", "NORTH-EAST", "EAST", "SOUTH-EAST",
             "SOUTH", "SOUTH-WEST", "WEST", "NORTH-WEST")
    return names[int((angle + 22.5) % 360.0 // 45.0)]


def _median_filter(points: list) -> list:
    """Width-3 median on the anchor positions, endpoints left as they are."""
    if len(points) < 3:
        return points
    smoothed = [points[0]]
    for i in range(1, len(points) - 1):
        ts = points[i][0]
        xs = sorted(p[1][0] for p in points[i - 1:i + 2])
        ys = sorted(p[1][1] for p in points[i - 1:i + 2])
        smoothed.append((ts, (xs[1], ys[1])))
    smoothed.append(points[-1])
    return smoothed


class MovementAnalyzer:
    """Measure one person's movement from their pose history."""

    def __init__(
        self,
        min_samples: int = 4,
        min_duration: float = 0.25,
        window_seconds: float = 1.0,
        stationary_speed: float = 0.15,
        min_support: int = 6,
        min_event_seconds: float = 0.15,
        min_box_height: float = 0.14,
    ) -> None:
        self.min_samples = int(min_samples)
        self.min_duration = float(min_duration)
        self.window_seconds = float(window_seconds)
        self.stationary_speed = float(stationary_speed)
        self.min_support = int(min_support)
        # Shortest span a vertical event is measured over. Below this a single
        # bad frame is indistinguishable from real motion.
        self.min_event_seconds = float(min_event_seconds)
        # Slightly above the pose estimator's own gate, so a track sitting
        # right on that gate does not feed marginal readings into behaviour.
        self.min_box_height = float(min_box_height)

    # ------------------------------------------------------------------

    def analyse(self, history: TrackPoseHistory) -> MovementSignals:
        latest = history.latest
        if latest is None:
            return MovementSignals(history.track_number, 0.0,
                                   reason="no pose samples")

        signals = MovementSignals(
            track_number=history.track_number,
            ts=latest.ts,
            samples=len(history.samples),
            duration=history.duration,
            support=latest.support,
            torso_angle=latest.torso_angle,
            knee_angle=latest.knee_angle,
        )

        if len(history.samples) < self.min_samples:
            signals.reason = (f"only {len(history.samples)} samples, "
                              f"need {self.min_samples}")
            return signals
        if history.duration < self.min_duration:
            signals.reason = (f"only {history.duration:.2f}s of history, "
                              f"need {self.min_duration:.2f}s")
            return signals
        if latest.support < self.min_support:
            signals.reason = (f"only {latest.support} keypoints visible, "
                              f"need {self.min_support}")
            return signals

        # Keypoint count is not enough on its own. The pose estimator has a
        # minimum box size, but a track that hovers around it gets pose on
        # the frames where it briefly exceeds the gate, and those readings
        # then flow in here as though they were trustworthy.
        #
        # Measured on real footage: a person whose box was below the gate for
        # 67% of the clip reported a hip ratio of 0.44 - deep crouch
        # territory - while their torso angle said upright and their knees
        # said straight. Internally contradictory, because the ankle
        # keypoints on a 119-pixel figure are guesses. Only the walking-speed
        # gate stopped it being classified as CROUCHING, which was luck.
        box_height = abs(latest.box[3] - latest.box[1]) if latest.box else 0.0
        if box_height < self.min_box_height:
            signals.reason = (f"box only {box_height:.2f} of frame, "
                              f"need {self.min_box_height:.2f} to trust pose")
            return signals

        reference = history.reference_height
        if not reference or reference < 1e-4:
            signals.reason = "no reliable standing height yet"
            return signals
        signals.reference_height = reference

        # -- posture ratios ---------------------------------------------
        if latest.body_height:
            signals.height_ratio = latest.body_height / reference
        signals.hip_ratio = self._hip_ratio(history, latest, reference)

        # -- speeds ------------------------------------------------------
        window = history.since(self.window_seconds)
        self._speeds(signals, window, reference)

        # -- limb geometry ----------------------------------------------
        self._limbs(signals, latest, reference)

        signals.ready = True
        signals.reason = "ok"
        return signals

    # ------------------------------------------------------------------

    @staticmethod
    def _hip_ratio(history: TrackPoseHistory, latest: PoseSample,
                   reference: float) -> Optional[float]:
        """How high the hips sit above the feet, against this person's normal.

        This separates a crouch from a person simply being cropped at the
        bottom of frame: a crouch lowers the hips towards the ankles, whereas
        a crop removes the ankles entirely and leaves this undefined.
        """
        if not latest.hips or not latest.ankles:
            return None
        current = abs(latest.ankles[1] - latest.hips[1])
        uprights = [abs(s.ankles[1] - s.hips[1])
                    for s in history.samples
                    if s.hips and s.ankles]
        if len(uprights) < 3:
            return None
        upright = float(np.percentile(np.asarray(uprights), 80))
        if upright < 1e-4:
            return None
        return current / upright

    def _speeds(self, signals: MovementSignals, window: list[PoseSample],
                reference: float) -> None:
        points = [(s.ts, s.anchor) for s in window if s.anchor]
        if len(points) < 2:
            return

        # Median-filter the anchor track before measuring anything from it.
        #
        # This is the only thing that actually removes a single-frame spike.
        # Averaging pairs does not: a 1-frame outlier still leaves two pairs
        # each carrying half of it. Measuring net displacement over a longer
        # span does not either, because the outlier can BE the endpoint of
        # the span. A width-3 median cannot be moved by one bad sample at
        # all, while a real jump or fall lasting several frames passes
        # through untouched. Found by a test that produced JUMPING from a
        # person standing still.
        points = _median_filter(points)

        first_ts, first = points[0]
        last_ts, last = points[-1]
        span = last_ts - first_ts
        if span <= 1e-6:
            return

        dx = last[0] - first[0]
        dy = last[1] - first[1]
        signals.speed = math.hypot(dx, dy) / reference / span
        signals.vertical_speed = (dy / reference) / span
        if signals.speed >= self.stationary_speed:
            signals.direction = _direction(dx, dy)

        # Vertical rates between consecutive samples. A jump or a fall shows
        # up here and an average over the whole window would smear a
        # half-second event into nothing.
        #
        # But the peak of single-frame rates is the wrong statistic, and this
        # was a real bug: one bad frame where a hip keypoint jumps produces a
        # huge rate, and because the peak is taken over a one-second window
        # that single outlier dominated for a full second - long enough to
        # outlast the debounce and confirm a JUMPING event from a person
        # standing perfectly still. A test caught it.
        #
        # The fix is to measure NET displacement over a minimum span rather
        # than a rate between two adjacent frames. A spike goes up and comes
        # straight back, so across a 0.15s span its net displacement is
        # almost zero and it scores nothing. A real jump or fall keeps
        # moving in one direction for several frames and survives.
        rates = []
        for i, (t0, p0) in enumerate(points):
            for t1, p1 in points[i + 1:]:
                span = t1 - t0
                if span < self.min_event_seconds:
                    continue
                rates.append(((p1[1] - p0[1]) / reference) / span)
                break

        falls = [r for r in rates if r > 0]
        rises = [-r for r in rates if r < 0]
        signals.peak_fall_speed = max(falls) if falls else 0.0
        signals.peak_rise_speed = max(rises) if rises else 0.0

        heights = [(s.body_height or 0.0) / reference for s in window
                   if s.body_height]
        if len(heights) >= 2:
            signals.height_trend = heights[-1] - heights[0]

    @staticmethod
    def _limbs(signals: MovementSignals, latest: PoseSample,
               reference: float) -> None:
        if latest.wrists and latest.hips:
            signals.hands_low = latest.wrists[1] > latest.hips[1]
        if latest.wrists and latest.shoulders:
            signals.hands_raised = latest.wrists[1] < latest.shoulders[1]
        if latest.wrists and latest.ankles:
            gap = abs(latest.wrists[1] - latest.ankles[1]) / reference
            signals.limbs_level = gap < 0.25
