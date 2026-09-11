"""Behaviour analysis: loitering, and the feature vector for anomaly detection.

Loitering here is a genuine temporal test, not a synonym for "detected". An
object is loitering only when **all** of the following hold:

1. it has been continuously inside one zone for longer than the configured
   threshold, and
2. its recent movement radius is smaller than the configured radius, so it is
   dwelling rather than passing through, and
3. it has been observed enough times for that judgement to be worth making.

Each condition is reported separately in :class:`LoiteringEvent` so the
operator can see why the call was made.

The eight behavioural features produced here are what the anomaly engine
learns from. They describe *what an object did*, not *what it looks like*,
which is the difference between "unusual behaviour" and "unusual pixels".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.tracker import TrackState

logger = logging.getLogger(__name__)

#: Names of the behavioural features, in vector order. Exposed so the GUI can
#: label them without duplicating the list.
FEATURE_NAMES: tuple[str, ...] = (
    "avg_speed",            # frame diagonals per second
    "max_speed",
    "dwell_seconds",        # time in the current zone
    "distance_norm",        # total path length
    "direction_changes",
    "zone_transitions",
    "restricted_seconds",   # time spent inside alarming zones
    "track_age_seconds",
)
FEATURE_DIM: int = len(FEATURE_NAMES)


@dataclass
class LoiteringEvent:
    """An object has been judged to be loitering."""

    track_number: int
    class_name: str
    zone_name: str
    ts: float
    duration_seconds: float
    movement_radius: float
    entered_ts: float

    @property
    def duration_text(self) -> str:
        minutes, seconds = divmod(int(self.duration_seconds), 60)
        return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"

    def describe(self) -> str:
        return (
            f"{self.class_name.upper()} #{self.track_number:02d} loitering in "
            f"{self.zone_name} for {self.duration_text}"
        )

    def to_dict(self) -> dict:
        return {
            "track_number": self.track_number,
            "class_name": self.class_name,
            "zone_name": self.zone_name,
            "duration_seconds": round(self.duration_seconds, 1),
            "duration_text": self.duration_text,
            "movement_radius": round(self.movement_radius, 4),
            "entered_ts": self.entered_ts,
        }


class BehaviourEngine:
    """Compute behavioural signals from track state."""

    def __init__(
        self,
        loitering_seconds: float = 12.0,
        loitering_radius: float = 0.06,
        min_samples_for_loitering: int = 8,
        high_speed_threshold: float = 0.35,
    ) -> None:
        self.loitering_seconds = float(loitering_seconds)
        self.loitering_radius = float(loitering_radius)
        self.min_samples_for_loitering = int(min_samples_for_loitering)
        self.high_speed_threshold = float(high_speed_threshold)

    # -- loitering --------------------------------------------------------

    def check_loitering(
        self, track: TrackState, now: float
    ) -> Optional[LoiteringEvent]:
        """Return an event the first time *track* qualifies as loitering.

        Returns ``None`` on every later call for the same dwell, so one person
        standing still produces one event rather than one per frame. The flag
        resets when the object changes zone (see
        :meth:`TrackState.enter_zone`).
        """
        if not track.zone_name:
            return None
        if track.frames_seen < self.min_samples_for_loitering:
            return None

        dwell = track.dwell_seconds(now)
        if dwell < self.loitering_seconds:
            track.loitering = False
            return None

        radius = track.movement_radius(window_seconds=self.loitering_seconds)
        if radius > self.loitering_radius:
            # Present for a long time, but moving through. Not loitering.
            track.loitering = False
            return None

        was_loitering = track.loitering
        track.loitering = True
        if track.loitering_since is None:
            track.loitering_since = now

        if was_loitering or track.loitering_reported:
            return None

        track.loitering_reported = True
        return LoiteringEvent(
            track_number=track.track_number,
            class_name=track.class_name,
            zone_name=track.zone_name,
            ts=now,
            duration_seconds=dwell,
            movement_radius=radius,
            entered_ts=track.zone_entered_ts,
        )

    def is_high_speed(self, track: TrackState) -> bool:
        """Whether the object is moving unusually fast for the scene."""
        return track.speed >= self.high_speed_threshold

    # -- features ---------------------------------------------------------

    def features(self, track: TrackState, now: float | None = None) -> np.ndarray:
        """The behavioural feature vector for one track.

        Order matches :data:`FEATURE_NAMES`. Always length
        :data:`FEATURE_DIM`, always finite, so the anomaly model never sees a
        NaN.
        """
        reference = now if now is not None else track.last_ts
        raw = [
            track.avg_speed,
            track.max_speed,
            track.dwell_seconds(reference),
            track.distance_norm,
            float(track.direction_changes),
            float(track.zone_transitions),
            track.restricted_seconds,
            track.age_seconds,
        ]
        vector = np.asarray(raw, dtype=np.float32)
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def features_from_row(row: dict) -> np.ndarray:
        """Rebuild a feature vector from a persisted ``tracks`` row.

        Lets the anomaly engine warm-start from previous sessions instead of
        needing to relearn the scene every time the application opens.
        """
        raw = [
            float(row.get("avg_speed") or 0.0),
            float(row.get("max_speed") or 0.0),
            float(row.get("dwell_seconds") or 0.0),
            float(row.get("distance_norm") or 0.0),
            float(row.get("direction_changes") or 0.0),
            float(row.get("zone_transitions") or 0.0),
            float(row.get("restricted_seconds") or 0.0),
            float(row.get("frames_seen") or 0.0),
        ]
        vector = np.asarray(raw, dtype=np.float32)
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    def describe_features(self, track: TrackState,
                          now: float | None = None) -> dict[str, float]:
        """Named features, for tooltips and incident detail blobs."""
        values = self.features(track, now)
        return {
            name: round(float(value), 4)
            for name, value in zip(FEATURE_NAMES, values)
        }
