"""Does this person have anything to do with that boundary?

A camera can see several fences. A person can be near one without touching
it. A tracked box can jitter across a line without anybody moving. None of
those are security events, and a system that treats them as such teaches its
operator to ignore it.

This module answers a narrower, answerable question: for one tracked person
and the boundaries configured on that camera, which boundary are they
actually engaging with, and how far has that engagement gone.

    NONE -> APPROACHING -> AT_BOUNDARY -> INTERACTING -> CROSSED

Each step needs evidence held over time. Nothing here fires from one frame.

What this module does NOT do
---------------------------
It does not find fences. There is no fence class in the detector's COCO
vocabulary, and finding linear structure in the image with edge or Hough
methods returns the fence, the handrail, the court markings, the kerb and the
roofline with equal confidence. Nothing in the pixels says which of those a
person is not allowed to cross - that is a fact about the site, not about the
image. So the boundaries come from ``config/zones.yaml``, configured per
camera by an operator who knows the ground.

That is a deliberate choice of a true answer over an impressive one. The AI
capability - detect, track, pose, measure, decide - is identical on every
camera. Only the geometry differs, because geometry describes a physical
place.

Choosing which boundary
-----------------------
When several are configured, the relevant one is the nearest, measured to the
line SEGMENT rather than an infinite line, and only if the person is within
an engagement radius. Distances are in the person's own body heights, so the
same radius means the same real-world distance at any camera distance.

If the second-nearest boundary is not meaningfully further away than the
nearest - judged as a ratio, so the test means the same thing at any camera
distance - the answer is UNCERTAIN and no interaction is reported, because
picking one at random would be inventing a fact.

Interaction evidence
--------------------
Proximity alone is not interaction. Somebody walking a path beside a fence is
near it for the whole clip. What separates interaction from proximity is the
body:

* a hand reaching towards the boundary - nearest wrist much closer than the
  body's own reference point;
* feet leaving their normal level - climbing or vaulting lifts them;
* the body rising while the person stays put horizontally.

Confirming a crossing
---------------------
A crossing is an EVENT, not a condition. The LineCrossingEngine reports it on
the single frame where the segments intersect and then falls silent under its
cooldown, so a state machine that waits for the signal to persist waits
forever. An early version of this module did exactly that: CROSSED was
debounced like every other state, needed the crossing signal on consecutive
frames to survive the hold, and therefore could not be reached at all outside
a unit test that fed it an unrealistic sustained signal.

What confirms a crossing is not repetition of the signal but corroboration
from the person's own trajectory:

* they were already engaged with THAT boundary - approaching it, at it, or
  interacting with it - when the crossing was reported. A crossing claimed by
  a track that had nothing to do with the boundary is a tracker artefact.
* they are subsequently observed on the OPPOSITE side of it.

The claim is held pending until the side flip is seen, and dropped if it
never is. That tolerates a gap in the readings, which matters: during the
vault on CAM-01 the person's box shrank below the size at which pose can be
trusted, so there are eleven frames in the middle of the event with no
measurements at all. The side flip is confirmed when they land.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Optional

from core.geometry import line_side, point_segment_distance

logger = logging.getLogger(__name__)

# -- interaction states -----------------------------------------------------
NONE = "NONE"
APPROACHING = "APPROACHING"
AT_BOUNDARY = "AT_BOUNDARY"
INTERACTING = "INTERACTING"
CROSSED = "CROSSED"
UNCERTAIN = "UNCERTAIN"

STATES: tuple[str, ...] = (NONE, APPROACHING, AT_BOUNDARY, INTERACTING,
                           CROSSED, UNCERTAIN)


@dataclass
class BoundaryReading:
    """One person measured against one boundary, this frame."""

    line_name: str
    #: Distance from the person's reference point to the segment, in body
    #: heights. None when the person's height is not yet known.
    distance: Optional[float] = None
    #: Sign of the side the person is on. Stable across frames.
    side: float = 0.0
    #: Nearest wrist distance, body heights. None when wrists not visible.
    hand_distance: Optional[float] = None
    #: Nearest ankle distance, body heights.
    foot_distance: Optional[float] = None
    #: How far the feet have risen above this person's own normal standing
    #: level, in body heights. Positive means lifted.
    foot_elevation: float = 0.0
    #: Negative when the person is getting closer over the recent window.
    approach_rate: float = 0.0
    #: How far the nearest hand has moved TOWARDS the boundary compared with
    #: where this person's hands normally sit, in body heights. Positive
    #: means extended.
    hand_extension: float = 0.0
    #: Horizontal speed, body heights per second. Needed to tell a climber
    #: from a runner: both lift their feet, only one stays put.
    speed: float = 0.0

    def to_dict(self) -> dict:
        return {
            "line": self.line_name,
            "distance": _round(self.distance),
            "hand_distance": _round(self.hand_distance),
            "hand_extension": round(self.hand_extension, 3),
            "foot_elevation": round(self.foot_elevation, 3),
            "speed": round(self.speed, 2),
            "approach_rate": round(self.approach_rate, 3),
            "side": round(self.side, 4),
        }


def _round(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(float(v), 3)


def _snapshot(state: "BoundaryInteraction") -> "BoundaryInteraction":
    """A frozen copy of the state at the moment of a transition.

    ``update`` used to hand back the live object it keeps per track, so a
    caller that collected the events - the acceptance script does - ended up
    with a list of references to one record that kept changing underneath it.
    Every entry then described the person's LATEST state rather than the one
    that fired. A transition is a historical fact and has to stop moving.
    """
    return replace(state)


def _reference_point(track: Any) -> Optional[tuple[float, float]]:
    """The track's own reference point, if it has one.

    This is the point the tracker and the line-crossing engine agree on -
    the bottom-centre of the box for a person. Returns None for a track
    object that does not expose one, so callers can fall back to the pose.
    """
    position = getattr(track, "position", None)
    try:
        x, y = position  # type: ignore[misc]
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


@dataclass
class BoundaryInteraction:
    """The debounced verdict for one person against one boundary."""

    track_id: int
    line_name: Optional[str]
    state: str = NONE
    confidence: float = 0.0
    since: float = 0.0
    last_ts: float = 0.0
    reading: Optional[BoundaryReading] = None
    #: Set once the tracker has confirmed a genuine segment crossing.
    crossing_seen: bool = False
    #: Side the person was on before the crossing, for the "opposite side"
    #: confirmation.
    side_before: Optional[float] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.last_ts - self.since)

    def describe(self) -> str:
        where = f" at {self.line_name}" if self.line_name else ""
        return (f"PERSON #{self.track_id:02d} {self.state}{where} "
                f"({self.confidence:.0%})")

    def to_dict(self) -> dict:
        return {
            "track_id": int(self.track_id),
            "line": self.line_name,
            "state": self.state,
            "confidence": round(float(self.confidence), 2),
            "duration": round(self.duration, 2),
            "evidence": self.reading.to_dict() if self.reading else {},
        }


@dataclass
class _Candidate:
    state: str = NONE
    line_name: Optional[str] = None
    since: float = 0.0
    confidence: float = 0.0
    reading: Optional[BoundaryReading] = None


@dataclass
class _PendingCrossing:
    """A crossing the tracker reported, not yet corroborated."""

    line_name: str
    #: Which side of the line the person was on immediately before it.
    side_before: float
    claimed_ts: float
    #: The state they were in when it was claimed, kept for the log line.
    state_before: str


class BoundaryInteractionAnalyzer:
    """Track how each person relates to the boundaries on this camera.

    Camera-agnostic: it is handed whatever lines that camera has configured
    and behaves identically for all of them. A camera with no lines produces
    no interactions, which is the correct answer, not a degraded one.
    """

    def __init__(
        self,
        engage_distance: float = 1.6,
        near_distance: float = 0.55,
        hand_reach: float = 0.35,
        hand_extension: float = 0.08,
        climb_elevation: float = 0.18,
        climb_max_speed: float = 0.60,
        approach_rate: float = -0.15,
        ambiguous_ratio: float = 1.35,
        hold_seconds: float = 0.35,
        crossed_hold_seconds: float = 0.50,
        crossing_confirm_seconds: float = 2.5,
    ) -> None:
        # All distances are in the person's own body heights.
        self.engage_distance = float(engage_distance)
        self.near_distance = float(near_distance)
        self.hand_reach = float(hand_reach)
        self.hand_extension = float(hand_extension)
        self.climb_elevation = float(climb_elevation)
        self.climb_max_speed = float(climb_max_speed)
        self.approach_rate = float(approach_rate)
        self.ambiguous_ratio = float(ambiguous_ratio)
        self.hold_seconds = float(hold_seconds)
        self.crossed_hold_seconds = float(crossed_hold_seconds)
        #: How long a reported crossing stays pending while waiting to see
        #: the person on the other side. Long enough to survive the gap in
        #: readings while somebody is actually going over - measured at 11
        #: frames, 0.37s, on CAM-01 - and short enough that an uncorroborated
        #: claim is forgotten rather than confirmed by an unrelated later
        #: movement.
        self.crossing_confirm_seconds = float(crossing_confirm_seconds)

        self._states: dict[int, BoundaryInteraction] = {}
        self._candidates: dict[int, _Candidate] = {}
        self._pending: dict[int, _PendingCrossing] = {}
        #: track id -> line name -> side on the previous measured frame.
        self._last_side: dict[int, dict[str, float]] = {}
        self._distance_window: dict[tuple[int, str], list] = {}
        self._hand_window: dict[tuple[int, str], list] = {}

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure(self, track: Any, lines: Iterable[Any], history: Any,
                ts: float, speed: float = 0.0) -> list[BoundaryReading]:
        """Measure this person against every configured boundary."""
        readings: list[BoundaryReading] = []
        if history is None:
            return readings
        reference = history.reference_height
        latest = history.latest
        if not reference or reference < 1e-4 or latest is None:
            return readings

        anchor = latest.anchor
        feet = latest.ankles
        point = feet or anchor
        if point is None:
            return readings

        # WHICH SIDE is measured from the track's own reference point - the
        # bottom-centre of its box - not from the pose. This is not a detail.
        # The LineCrossingEngine tests that same point for intersection, so
        # any other point can disagree with it, and on real footage it did:
        # a person standing against the railing had ankle keypoints landing a
        # few pixels beyond the line while their box bottom was still short
        # of it, so the pose said they were already across before they
        # climbed, and no side change was ever observed for the vault that
        # followed. Distances, hands and feet still come from the pose; only
        # the side comes from the point the crossing was declared on.
        side_point = _reference_point(track) or point

        for line in lines:
            raw = point_segment_distance(point[0], point[1],
                                         line.start, line.end)
            reading = BoundaryReading(
                line_name=line.name,
                distance=raw / reference,
                side=line_side(side_point[0], side_point[1],
                               line.start, line.end),
            )

            if latest.wrists:
                reading.hand_distance = point_segment_distance(
                    latest.wrists[0], latest.wrists[1],
                    line.start, line.end) / reference
            if feet:
                reading.foot_distance = point_segment_distance(
                    feet[0], feet[1], line.start, line.end) / reference

            reading.hand_extension = self._hand_extension(
                track.track_number, line.name, reading.hand_distance, ts)
            reading.foot_elevation = self._foot_elevation(history, reference)
            reading.speed = float(speed)
            reading.approach_rate = self._approach_rate(
                track.track_number, line.name, reading.distance, ts)
            readings.append(reading)

        return readings

    def _hand_extension(self, track_id: int, line_name: str,
                        hand_distance: Optional[float], ts: float) -> float:
        """How far the hand has moved towards the boundary, body heights.

        Absolute hand distance cannot be used on its own, and finding that
        out cost a test. A configured boundary is often a roughly horizontal
        line in image space with the person standing below it, and a standing
        person's hands are at chest height while their feet are on the
        ground. So the hands are ALWAYS nearer that line than the feet are,
        and "hand closer than body" reports every passer-by as reaching for
        the fence.

        What actually distinguishes reaching is movement: the wrist leaves
        where that person's wrists normally sit and travels towards the
        boundary. Measured against their own recent history, so it carries no
        assumption about which way the boundary runs.
        """
        if hand_distance is None:
            return 0.0
        key = (track_id, line_name)
        window = self._hand_window.setdefault(key, [])
        window.append((ts, hand_distance))
        cutoff = ts - 2.0
        while len(window) > 2 and window[0][0] < cutoff:
            window.pop(0)
        if len(window) < 6:
            return 0.0
        import numpy as np
        typical = float(np.median([d for _, d in window]))
        return max(0.0, typical - hand_distance)

    @staticmethod
    def _foot_elevation(history: Any, reference: float) -> float:
        """How far the feet sit above their own normal level, body heights.

        Compared against this person's own history rather than a fixed image
        row, so it means the same thing wherever they are standing and on any
        camera. Climbing or vaulting lifts the feet; walking does not.
        """
        samples = [s for s in history.samples if s.ankles]
        if len(samples) < 5:
            return 0.0
        # Normal foot level is the LOWEST the feet usually reach, taken as a
        # high percentile of y (y grows downward), which ignores the frames
        # where they are already off the ground.
        import numpy as np
        ys = np.asarray([s.ankles[1] for s in samples])
        ground = float(np.percentile(ys, 80))
        current = samples[-1].ankles[1]
        return max(0.0, (ground - current) / reference)

    def _approach_rate(self, track_id: int, line_name: str,
                       distance: Optional[float], ts: float) -> float:
        """Change in distance per second. Negative means closing in."""
        if distance is None:
            return 0.0
        key = (track_id, line_name)
        window = self._distance_window.setdefault(key, [])
        window.append((ts, distance))
        cutoff = ts - 1.0
        while len(window) > 2 and window[0][0] < cutoff:
            window.pop(0)
        if len(window) < 3:
            return 0.0
        span = window[-1][0] - window[0][0]
        if span <= 1e-6:
            return 0.0
        return (window[-1][1] - window[0][1]) / span

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, readings: list[BoundaryReading]
               ) -> tuple[Optional[BoundaryReading], bool]:
        """Pick the boundary this person is engaging with.

        Returns (reading, ambiguous). Ambiguous means two boundaries are too
        close together to choose between, and the honest answer is UNCERTAIN
        rather than an arbitrary pick.
        """
        usable = [r for r in readings
                  if r.distance is not None and r.distance <= self.engage_distance]
        if not usable:
            return None, False
        usable.sort(key=lambda r: r.distance)
        if len(usable) >= 2:
            nearest, second = usable[0].distance, usable[1].distance
            # RATIO, not absolute gap. An absolute margin is scale-dependent
            # in exactly the way this codebase avoids everywhere else: a
            # person far from the camera is small, so every gap measured in
            # their body heights is small, and they would be permanently
            # UNCERTAIN while a near person never was.
            #
            # Measured on real footage: someone walking between two fences at
            # 0.28 and 0.42 body heights was called ambiguous because the
            # absolute gap was 0.14 against a 0.15 margin - even though one
            # fence was half again as far as the other, which is not
            # ambiguous at all.
            if nearest > 1e-6 and (second / nearest) < self.ambiguous_ratio:
                return usable[0], True
        return usable[0], False

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _instant_state(self, reading: BoundaryReading, ambiguous: bool
                       ) -> tuple[str, float]:
        """The reading for this frame, before debouncing.

        CROSSED is deliberately not producible here. It is not something a
        single frame's geometry can show - it needs the tracker's crossing
        event plus a side change - so it is decided in ``update`` instead.
        """
        if ambiguous:
            return UNCERTAIN, 0.4

        # `or` would be wrong here: a distance of exactly 0.0 is falsy, so a
        # person standing precisely ON the boundary would be treated as
        # infinitely far from it. Found by a test where someone climbing
        # passed straight through the line.
        distance = reading.distance if reading.distance is not None else 999.0

        # Interaction needs the BODY involved, not just proximity. A person
        # walking a path beside a fence is near it the whole time.
        # Reaching means the hand MOVED towards the boundary and ended up
        # close to it. Both halves are required: the movement alone could be
        # someone gesturing, and the closeness alone is where a horizontal
        # boundary above a standing person produces false positives.
        reaching = (reading.hand_distance is not None
                    and reading.hand_distance < self.hand_reach
                    and reading.hand_extension >= self.hand_extension)
        # Feet leaving the ground is NOT enough on its own, and real footage
        # proved it: a person running past the fence lifted their feet 0.47
        # of a body height as part of their stride and was reported as
        # INTERACTING at 1.55 body heights per second. Running lifts the
        # feet; so does climbing. What separates them is that a climber has
        # stopped going anywhere.
        climbing = (reading.foot_elevation >= self.climb_elevation
                    and reading.speed < self.climb_max_speed)
        if distance <= self.near_distance and (reaching or climbing):
            margin = max(
                reading.hand_extension / max(self.hand_extension, 1e-6) - 1.0,
                (reading.foot_elevation - self.climb_elevation) / 0.3,
            )
            return INTERACTING, min(0.95, 0.6 + 0.35 * max(0.0, min(1.0, margin)))

        if distance <= self.near_distance:
            return AT_BOUNDARY, 0.6 + 0.3 * (
                1.0 - distance / max(self.near_distance, 1e-6))

        if (distance <= self.engage_distance
                and reading.approach_rate <= self.approach_rate):
            return APPROACHING, 0.55 + 0.3 * min(
                1.0, abs(reading.approach_rate) / 0.6)

        return NONE, 0.0

    #: States that count as being engaged with a boundary. A crossing
    #: reported for a track in any other state has no approach behind it and
    #: is treated as a tracker artefact rather than a security event.
    _ENGAGED: tuple[str, ...] = (APPROACHING, AT_BOUNDARY, INTERACTING)

    def _arm_crossing(self, track_id: int, state: BoundaryInteraction,
                      line_names: list[str], previous_sides: dict[str, float],
                      ts: float) -> None:
        """Record a reported crossing as pending, if it has an approach."""
        if state.state not in self._ENGAGED or state.line_name is None:
            return
        if line_names and state.line_name not in line_names:
            # The tracker crossed a different line from the one this person
            # is engaged with. Saying nothing is better than attributing the
            # crossing to whichever boundary happened to be nearest.
            return
        side_before = previous_sides.get(state.line_name)
        if side_before is None or abs(side_before) < 1e-9:
            return
        self._pending[track_id] = _PendingCrossing(
            line_name=state.line_name, side_before=side_before,
            claimed_ts=ts, state_before=state.state)

    def _confirm_crossing(self, track_id: int,
                          readings: dict[str, BoundaryReading],
                          ts: float) -> Optional[BoundaryReading]:
        """Has a pending crossing been corroborated by a side change?"""
        pending = self._pending.get(track_id)
        if pending is None:
            return None
        if ts - pending.claimed_ts > self.crossing_confirm_seconds:
            logger.debug("PERSON #%02d crossing of %s expired uncorroborated",
                         track_id, pending.line_name)
            self._pending.pop(track_id, None)
            return None
        reading = readings.get(pending.line_name)
        if reading is None or abs(reading.side) < 1e-9:
            return None
        if (reading.side > 0) == (pending.side_before > 0):
            return None
        self._pending.pop(track_id, None)
        return reading

    def update(self, track: Any, lines: Iterable[Any], history: Any,
               ts: float, crossing_seen: bool = False, speed: float = 0.0,
               crossed_lines: Optional[Iterable[str]] = None
               ) -> Optional[BoundaryInteraction]:
        """Fold one frame into this person's boundary state.

        ``crossing_seen`` comes from the existing LineCrossingEngine, which
        already does the segment-intersection test. This module does not
        duplicate that; it decides whether the crossing had the supporting
        evidence of a real approach, and whether the person is afterwards
        actually on the other side. ``crossed_lines`` names which boundaries
        the tracker reported, so a crossing of one line is not attributed to
        another; omit it and the person's currently selected boundary is
        assumed.

        Returns an interaction only when the state CHANGES and has held.
        """
        track_id = int(track.track_number)
        lines = list(lines)

        state = self._states.get(track_id)
        if state is None:
            state = BoundaryInteraction(track_id=track_id, line_name=None,
                                        since=ts, last_ts=ts)
            self._states[track_id] = state
        state.last_ts = ts

        if not lines:
            return None

        readings = self.measure(track, lines, history, ts, speed=speed)
        by_name = {r.line_name: r for r in readings}
        previous_sides = self._last_side.get(track_id, {})
        if readings:
            self._last_side[track_id] = {r.line_name: r.side for r in readings}

        # -- crossing: arm on the event, confirm on the side change --------
        if crossing_seen:
            self._arm_crossing(track_id, state, list(crossed_lines or []),
                               previous_sides, ts)
        confirmed = self._confirm_crossing(track_id, by_name, ts)
        if confirmed is not None and state.state != CROSSED:
            state.state = CROSSED
            state.line_name = confirmed.line_name
            state.confidence = 0.9
            state.since = ts
            state.reading = confirmed
            state.crossing_seen = True
            state.side_before = previous_sides.get(confirmed.line_name)
            self._candidates.pop(track_id, None)
            return _snapshot(state)

        reading, ambiguous = self.select(readings)
        if reading is None:
            self._candidates.pop(track_id, None)
            return None

        instant, confidence = self._instant_state(reading, ambiguous)

        # CROSSED is the headline state and must not be silently downgraded
        # the frame after it is confirmed - a person who has just gone over a
        # fence is still standing next to it, so the geometry alone would read
        # AT_BOUNDARY again immediately. It holds for `crossed_hold_seconds`
        # and then gives way to whatever is true now. The crossing itself is
        # already recorded as an incident and does not depend on this hold.
        if (state.state == CROSSED and instant != CROSSED
                and (ts - state.since) < self.crossed_hold_seconds):
            state.reading = reading
            return None

        if instant == NONE:
            self._candidates.pop(track_id, None)
            return None

        if instant == state.state and reading.line_name == state.line_name:
            state.confidence = max(state.confidence, confidence)
            state.reading = reading
            self._candidates.pop(track_id, None)
            return None

        candidate = self._candidates.get(track_id)
        if (candidate is None or candidate.state != instant
                or candidate.line_name != reading.line_name):
            self._candidates[track_id] = _Candidate(
                state=instant, line_name=reading.line_name, since=ts,
                confidence=confidence, reading=reading)
            return None

        candidate.confidence = max(candidate.confidence, confidence)
        candidate.reading = reading

        if ts - candidate.since < self.hold_seconds:
            return None

        state.state = instant
        state.line_name = candidate.line_name
        state.confidence = candidate.confidence
        state.since = candidate.since
        state.reading = candidate.reading
        self._candidates.pop(track_id, None)
        return _snapshot(state)

    # ------------------------------------------------------------------

    def state(self, track_id: int) -> Optional[BoundaryInteraction]:
        return self._states.get(track_id)

    def prune(self, live: set[int]) -> None:
        for track_id in [t for t in self._states if t not in live]:
            del self._states[track_id]
        for track_id in [t for t in self._candidates if t not in live]:
            del self._candidates[track_id]
        for track_id in [t for t in self._pending if t not in live]:
            del self._pending[track_id]
        for track_id in [t for t in self._last_side if t not in live]:
            del self._last_side[track_id]
        for key in [k for k in self._distance_window if k[0] not in live]:
            del self._distance_window[key]
        for key in [k for k in self._hand_window if k[0] not in live]:
            del self._hand_window[key]

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self._states.values():
            if s.state != NONE:
                counts[s.state] = counts.get(s.state, 0) + 1
        return counts

    @classmethod
    def from_config(cls, config) -> "BoundaryInteractionAnalyzer":
        section = (config.section("activity").get("boundary", {}) or {})
        return cls(**{k: float(v) for k, v in section.items()})
