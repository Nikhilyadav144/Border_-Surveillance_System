"""Persistent track state and kinematics.

Ultralytics gives us a stable integer id per object. That number on its own is
not a track: it says "this is the same thing" but nothing about what the thing
did. This module turns the stream of (id, box, timestamp) observations into a
:class:`TrackState` carrying trajectory, speed, heading, distance travelled,
direction changes and zone history - the raw material for loitering,
behavioural anomaly detection and threat scoring.

Units
-----
Positions are normalised to 0-1. Distances and speeds are therefore expressed
in *fractions of the frame diagonal*, and speed in fractions per second. That
is honest: without a camera calibration there is no way to convert to metres.
If ``pixels_per_metre`` is configured, :attr:`TrackState.speed_mps` becomes
available and the GUI shows m/s instead.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Iterable, Optional

from core.geometry import angle_difference, compass_direction, heading_degrees

logger = logging.getLogger(__name__)

Point = tuple[float, float]


@dataclass
class TrackSample:
    """One observation of a track."""

    ts: float
    x: float                 # normalised centre / reference x
    y: float                 # normalised reference y
    box: tuple[float, float, float, float]   # normalised x1,y1,x2,y2
    confidence: float


@dataclass
class TrackState:
    """Everything known about one tracked identity."""

    track_number: int
    class_name: str
    camera_id: str
    first_ts: float
    last_ts: float

    frames_seen: int = 0
    samples: Deque[TrackSample] = field(default_factory=lambda: deque(maxlen=240))

    confidence_sum: float = 0.0
    max_confidence: float = 0.0

    distance_norm: float = 0.0
    max_speed: float = 0.0
    _speed_window: Deque[float] = field(default_factory=lambda: deque(maxlen=5))

    heading: float = 0.0
    _last_heading: Optional[float] = None
    direction_changes: int = 0

    zone_name: Optional[str] = None
    zone_entered_ts: float = 0.0
    zones_visited: list[str] = field(default_factory=list)
    zone_transitions: int = 0
    restricted_seconds: float = 0.0

    line_crossings: int = 0
    loitering: bool = False
    loitering_since: Optional[float] = None
    loitering_reported: bool = False

    anomaly_score: Optional[float] = None
    anomaly_status: str = "INSUFFICIENT_DATA"

    plate_text: Optional[str] = None
    plate_attempts: int = 0
    plate_last_attempt_ts: float = 0.0

    # -- pose -------------------------------------------------------------
    # Body keypoints live on the track, keyed by the identity ByteTrack
    # already assigned. No parallel id space, no reconciliation step: if the
    # tracker says this is person 7, these are person 7's joints.
    #: Latest core.pose.PoseResult, or None if pose has never run on this
    #: track (too small, not due this frame, or pose disabled).
    pose: Optional[Any] = None
    #: Latest core.pose.Posture derived from that result.
    posture: Optional[Any] = None
    #: Wall/media time of the most recent successful pose reading.
    pose_ts: float = 0.0
    #: How many successful readings this track has had.
    pose_count: int = 0
    #: UNAVAILABLE / TRACKED / STALE. Never fabricated: a track whose box was
    #: too small or too occluded for pose says UNAVAILABLE and stops there.
    pose_status: str = "UNAVAILABLE"

    # -- activity ---------------------------------------------------------
    # Behaviour is temporal, so it lives on the track rather than on a frame.
    #: Current debounced behaviour name, e.g. WALKING.
    behaviour: str = "UNKNOWN"
    #: How sure the classifier is, 0-1.
    behaviour_confidence: float = 0.0
    #: Seconds the current behaviour has held.
    behaviour_duration: float = 0.0
    #: Compass direction of travel, or None when effectively stationary.
    movement_direction: Optional[str] = None
    #: Latest core.movement.MovementSignals, for display and explanation.
    movement: Optional[Any] = None

    # -- boundary relationship --------------------------------------------
    #: NONE / APPROACHING / AT_BOUNDARY / INTERACTING / CROSSED / UNCERTAIN.
    #: Says how this person relates to the boundary configured for THEIR
    #: camera, not whether a fence is visible anywhere in frame.
    boundary_state: str = "NONE"
    #: Which configured boundary, when one was selected.
    boundary_line: Optional[str] = None
    #: Distance to that boundary in this person's own body heights.
    boundary_distance: Optional[float] = None

    # -- threat -----------------------------------------------------------
    # The latest verdict the ThreatEngine reached FOR THIS TRACK, kept here
    # beside the behaviour and boundary state rather than in a side table,
    # because it is the same kind of fact: something the pipeline concluded
    # about this person.
    #
    # None until the engine has actually scored them. A person nobody has
    # assessed reports nothing, not LOW - LOW is the verdict that looks
    # harmless, so defaulting to it would be the dangerous invention.
    threat_level: Optional[str] = None
    threat_score: Optional[float] = None
    #: Machine-readable factor codes, e.g. ["boundary_crossed", "high_speed"].
    threat_signals: list = field(default_factory=list)
    #: The same factors as operator-readable text.
    threat_reasons: list = field(default_factory=list)

    db_id: Optional[int] = None
    missed_frames: int = 0
    closed: bool = False

    # -- identity ---------------------------------------------------------

    @property
    def key(self) -> str:
        return f"{self.camera_id}:{self.class_name}:{self.track_number}"

    @property
    def label(self) -> str:
        return f"{self.class_name.upper()} #{self.track_number:02d}"

    # -- kinematics -------------------------------------------------------

    @property
    def age_seconds(self) -> float:
        return max(0.0, self.last_ts - self.first_ts)

    @property
    def avg_confidence(self) -> float:
        return self.confidence_sum / self.frames_seen if self.frames_seen else 0.0

    @property
    def position(self) -> Point:
        if not self.samples:
            return (0.0, 0.0)
        last = self.samples[-1]
        return (last.x, last.y)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.samples[-1].box if self.samples else (0.0, 0.0, 0.0, 0.0)

    @property
    def speed(self) -> float:
        """Smoothed current speed, frame diagonals per second."""
        if not self._speed_window:
            return 0.0
        return sum(self._speed_window) / len(self._speed_window)

    @property
    def avg_speed(self) -> float:
        age = self.age_seconds
        return self.distance_norm / age if age > 0.5 else 0.0

    @property
    def direction(self) -> str:
        return compass_direction(self.heading)

    @property
    def trajectory(self) -> list[Point]:
        return [(s.x, s.y) for s in self.samples]

    def trajectory_rows(self) -> list[tuple[float, float, float]]:
        """Trajectory as (t, x, y) triples, for persistence."""
        return [(round(s.ts, 3), round(s.x, 5), round(s.y, 5)) for s in self.samples]

    def movement_radius(self, window_seconds: float = 10.0) -> float:
        """Radius of the smallest circle containing recent positions.

        This is the loitering test: a person walking through has a large
        radius, a person waiting around has a small one, regardless of how
        much they fidget.
        """
        if len(self.samples) < 2:
            return 0.0
        cutoff = self.last_ts - window_seconds
        recent = [s for s in self.samples if s.ts >= cutoff]
        if len(recent) < 2:
            return 0.0
        cx = sum(s.x for s in recent) / len(recent)
        cy = sum(s.y for s in recent) / len(recent)
        return max(math.hypot(s.x - cx, s.y - cy) for s in recent)

    def dwell_seconds(self, now: float | None = None) -> float:
        """Time spent in the current zone."""
        if not self.zone_name:
            return 0.0
        return max(0.0, (now if now is not None else time.time())
                   - self.zone_entered_ts)

    def speed_mps(self, pixels_per_metre: float | None,
                  frame_diagonal_px: float) -> Optional[float]:
        """Speed in metres per second, when a calibration is available."""
        if not pixels_per_metre or pixels_per_metre <= 0 or frame_diagonal_px <= 0:
            return None
        return self.speed * frame_diagonal_px / pixels_per_metre

    # -- updates ----------------------------------------------------------

    def observe(
        self,
        ts: float,
        point: Point,
        box: tuple[float, float, float, float],
        confidence: float,
        direction_change_degrees: float = 45.0,
    ) -> None:
        """Record one observation and update every derived quantity."""
        previous = self.samples[-1] if self.samples else None

        self.samples.append(
            TrackSample(ts=ts, x=point[0], y=point[1], box=box,
                        confidence=confidence)
        )
        self.frames_seen += 1
        self.last_ts = ts
        self.missed_frames = 0
        self.confidence_sum += confidence
        self.max_confidence = max(self.max_confidence, confidence)

        if previous is None:
            return

        dt = ts - previous.ts
        step = math.hypot(point[0] - previous.x, point[1] - previous.y)
        self.distance_norm += step

        if dt > 1e-4:
            instant = step / dt
            self._speed_window.append(instant)
            self.max_speed = max(self.max_speed, self.speed)

        # Heading only means something once the object has actually moved.
        # Below this threshold the "direction" is detector jitter.
        if step > 0.004:
            new_heading = heading_degrees((previous.x, previous.y), point)
            if self._last_heading is not None:
                if angle_difference(self._last_heading, new_heading) \
                        >= direction_change_degrees:
                    self.direction_changes += 1
            self._last_heading = new_heading
            self.heading = new_heading

    def enter_zone(self, zone_name: Optional[str], ts: float,
                   was_restricted: bool = False, previous_dwell: float = 0.0) -> None:
        """Record a zone change already decided by the zone engine."""
        if was_restricted and previous_dwell > 0:
            self.restricted_seconds += previous_dwell
        self.zone_name = zone_name
        self.zone_entered_ts = ts
        if zone_name:
            if zone_name not in self.zones_visited:
                self.zones_visited.append(zone_name)
            self.zone_transitions += 1
        # A zone change resets loitering: the object moved somewhere new.
        self.loitering = False
        self.loitering_since = None
        self.loitering_reported = False

    def mark_missed(self) -> None:
        self.missed_frames += 1

    def summary(self) -> dict:
        """Compact view for the GUI track table."""
        return {
            "track": self.track_number,
            "label": self.label,
            "class": self.class_name,
            "age": round(self.age_seconds, 1),
            "frames": self.frames_seen,
            "confidence": round(self.avg_confidence, 3),
            "speed": round(self.speed, 4),
            "distance": round(self.distance_norm, 4),
            "direction": self.direction,
            "direction_changes": self.direction_changes,
            "zone": self.zone_name or "-",
            "dwell": round(self.dwell_seconds(self.last_ts), 1),
            "loitering": self.loitering,
            "anomaly": self.anomaly_status,
            "anomaly_score": (round(self.anomaly_score, 4)
                              if self.anomaly_score is not None else None),
            "plate": self.plate_text or "-",
        }


class TrackManager:
    """Own the set of live tracks for one camera."""

    def __init__(
        self,
        camera_id: str,
        max_age_frames: int = 30,
        min_hits: int = 3,
        max_trajectory_points: int = 240,
        direction_change_degrees: float = 45.0,
        coast_frames: int = 5,
    ) -> None:
        self.camera_id = camera_id
        self.max_age_frames = int(max_age_frames)
        self.min_hits = int(min_hits)
        #: How many processed frames a track may go unmatched and still be
        #: PUBLISHED as a live object. See `confirmed`.
        self.coast_frames = max(0, int(coast_frames))
        self.max_trajectory_points = int(max_trajectory_points)
        self.direction_change_degrees = float(direction_change_degrees)

        self.tracks: dict[int, TrackState] = {}
        self.closed_tracks: list[TrackState] = []
        self._total_created = 0

    # -- lifecycle --------------------------------------------------------

    def update(
        self,
        observations: Iterable[tuple[int, str, Point,
                                     tuple[float, float, float, float], float]],
        ts: float,
    ) -> tuple[list[TrackState], list[TrackState]]:
        """Apply this frame's observations.

        Parameters
        ----------
        observations:
            Tuples of ``(track_number, class_name, point, box_norm, confidence)``.
            ``point`` is the reference point chosen by the zone engine, so
            kinematics and zone membership agree on where the object is.

        Returns
        -------
        (active, closed)
            ``active`` is every track seen this frame. ``closed`` is the
            tracks retired this frame, which the caller persists and feeds to
            the anomaly engine.
        """
        seen: set[int] = set()
        active: list[TrackState] = []

        for track_number, class_name, point, box, confidence in observations:
            seen.add(track_number)
            state = self.tracks.get(track_number)
            if state is None:
                state = TrackState(
                    track_number=track_number,
                    class_name=class_name,
                    camera_id=self.camera_id,
                    first_ts=ts,
                    last_ts=ts,
                )
                state.samples = deque(maxlen=self.max_trajectory_points)
                self.tracks[track_number] = state
                self._total_created += 1
            elif state.class_name != class_name:
                # The detector changed its mind about the class. Keep the
                # identity but adopt the newer, higher-confidence label.
                state.class_name = class_name

            state.observe(
                ts=ts,
                point=point,
                box=box,
                confidence=confidence,
                direction_change_degrees=self.direction_change_degrees,
            )
            active.append(state)

        closed: list[TrackState] = []
        for number, state in list(self.tracks.items()):
            if number in seen:
                continue
            state.mark_missed()
            if state.missed_frames > self.max_age_frames:
                state.closed = True
                closed.append(state)
                self.closed_tracks.append(state)
                del self.tracks[number]

        return active, closed

    def close_all(self) -> list[TrackState]:
        """Retire every live track. Called when a session ends."""
        closed = list(self.tracks.values())
        for state in closed:
            state.closed = True
        self.closed_tracks.extend(closed)
        self.tracks.clear()
        return closed

    # -- queries ----------------------------------------------------------

    def get(self, track_number: int) -> Optional[TrackState]:
        return self.tracks.get(track_number)

    def confirmed(self) -> list[TrackState]:
        """Tracks that are trusted AND currently believed to be in view.

        Two separate conditions, and keeping them separate is the point.

        `frames_seen >= min_hits` says the track has been observed often
        enough to be more than a flicker.

        `missed_frames <= coast_frames` says it is still plausibly there. A
        track that stops matching detections keeps its LAST box, speed,
        posture and behaviour, because nothing updates them; publishing it
        indefinitely means publishing a frozen assertion about a person who
        may have left. On CAM-01 that is exactly what happened: PERSON #01
        walked out of frame at f149 and his track went on reporting the same
        box at 3.72 body heights per second, labelled RUNNING, for another
        60 frames - two seconds of a person who was not there. Of the 62
        frames the system called him RUNNING, 61 were that.

        A short coast is still wanted, which is why this is not zero: a
        detector that misses one frame should not make a person blink out of
        the operator's table and back in. The track itself also stays in
        `self.tracks` until `max_age_frames`, so the tracker can still
        re-attach the same identity after a longer occlusion. What ends
        early is the CLAIM that the object is visible right now.
        """
        return [t for t in self.tracks.values()
                if t.frames_seen >= self.min_hits
                and t.missed_frames <= self.coast_frames]

    def count_class(self, names: set[str]) -> int:
        """How many CONFIRMED live tracks carry one of these class names.

        Counting every entry in ``self.tracks`` counts one-frame flickers as
        people. A synthetic perimeter clip produced a track from two
        detections that then never appeared again; it raised no incident,
        because incident analysis already requires ``min_hits``, but it was
        still reported to the operator as a person standing at the fence.
        The count an operator reads and the set of tracks trusted to raise an
        incident should be the same set.
        """
        lowered = {n.lower() for n in names}
        return sum(1 for t in self.confirmed()
                   if t.class_name.lower() in lowered)

    @property
    def live_count(self) -> int:
        return len(self.tracks)

    @property
    def total_created(self) -> int:
        return self._total_created

    def reset(self) -> None:
        self.tracks.clear()
        self.closed_tracks.clear()
        self._total_created = 0
