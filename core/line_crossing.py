"""Virtual boundary line crossing detection.

A crossing is detected when the segment joining an object's previous and
current position intersects a configured line. That is more reliable than
comparing which side the object is on, because a fast object can jump the
line between two processed frames and a side comparison would still catch it
only if both samples land on opposite sides - which segment intersection
tests directly.

Direction is reported two ways:

* ``entering`` / ``exiting`` - relative to the side the operator declared as
  "in" via ``entering_direction`` in ``zones.yaml``
* a compass word - ``north`` / ``south`` / ``east`` / ``west`` - derived from
  the object's actual heading

Each track can cross the same line more than once, so a cooldown prevents an
object loitering exactly on the boundary from producing a stream of events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from core.geometry import (
    Point,
    compass_direction,
    heading_degrees,
    line_side,
    segments_intersect,
)

logger = logging.getLogger(__name__)

_VALID_DIRECTIONS = ("north", "south", "east", "west")


@dataclass(frozen=True)
class VirtualLine:
    """One configured boundary."""

    name: str
    start: Point
    end: Point
    colour: str = "#00b4d8"
    entering_direction: str = "north"
    #: Whether crossing this line is an event worth reporting.
    #
    # Not every barrier in view is a security boundary, and the two needs
    # pull in opposite directions. To decide which fence a person is engaged
    # with, the system has to know where the OTHER fences are - otherwise it
    # is choosing from a list of one and cannot be said to have chosen at
    # all. But a line the crossing engine knows about is a line it reports
    # crossings of.
    #
    # `alerting: false` separates the two. The boundary analyzer still sees
    # the line, measures distance to it and can select it as the one a
    # person is nearest; the crossing engine measures it and stays quiet.
    #
    # It also covers a real limitation. A fence in the BACKGROUND cannot be
    # used as a crossing line from a low viewpoint: a person's reference
    # point is the bottom of their box, and someone standing well in front
    # of that fence - or lifting their feet in front of it - sweeps that
    # point across the line in image space while never going near it in the
    # world. A 2-D line test has no depth information with which to tell
    # those apart, so the fence is declared non-alerting rather than
    # producing crossings nobody made.
    alerting: bool = True

    def side_of(self, point: Point) -> float:
        return line_side(point[0], point[1], self.start, self.end)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start": list(self.start),
            "end": list(self.end),
            "colour": self.colour,
            "entering_direction": self.entering_direction,
            "alerting": self.alerting,
        }


@dataclass
class CrossingEvent:
    """One object crossed one line."""

    line: VirtualLine
    track_number: int
    class_name: str
    ts: float
    from_point: Point
    to_point: Point
    compass: str
    direction: str            # "entering" | "exiting"
    confidence: float = 0.0

    def describe(self) -> str:
        return (
            f"{self.class_name.upper()} #{self.track_number:02d} crossed "
            f"{self.line.name} heading {self.compass} ({self.direction})"
        )

    def to_dict(self) -> dict:
        return {
            "line": self.line.name,
            "track_number": self.track_number,
            "class_name": self.class_name,
            "compass": self.compass,
            "direction": self.direction,
            "from": list(self.from_point),
            "to": list(self.to_point),
            "ts": self.ts,
        }


@dataclass
class _TrackLineState:
    #: Timestamp of the first sample ever fed for this track. Used for the
    #: warm-up window, which is why it is recorded even before the track has
    #: moved far enough to be tested against a line.
    first_ts: Optional[float] = None
    last_point: Optional[Point] = None
    #: line name -> timestamp of the last crossing. A line absent from this
    #: mapping has never been crossed, which is not the same as having been
    #: crossed at time zero: treating them the same swallowed the first
    #: crossing of every track whenever timestamps started near the epoch.
    last_crossing_ts: dict[str, float] = field(default_factory=dict)
    crossing_count: int = 0


class LineCrossingEngine:
    """Detect crossings of configured virtual lines."""

    def __init__(
        self,
        lines: Iterable[dict[str, Any]] | Iterable[VirtualLine] = (),
        cooldown_seconds: float = 3.0,
        min_travel: float = 0.004,
        warmup_seconds: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        cooldown_seconds:
            Minimum gap between two crossings of the same line by the same
            track. Stops a target standing on the line from spamming events.
        min_travel:
            Movement below this (fraction of frame) is treated as jitter and
            cannot trigger a crossing.
        warmup_seconds:
            A track's first ``warmup_seconds`` are not tested against any
            line. A newly detected object's box is still converging: it grows
            as the person emerges from occlusion or from the frame edge, and
            the reference point - the bottom-centre for a person - sweeps
            across the frame for a reason that is not motion. On CAM-01 that
            sweep was 0.13 of frame height in six frames and produced two
            crossings of the perimeter railing before the person had taken a
            step, which then swallowed the real vault thirty frames later
            under the cooldown.

            The cost is real and is the reason this is a short window rather
            than a long one: a genuine crossing inside a track's first half
            second is missed. That trade is deliberate - a missed crossing is
            recoverable when the person is still in view, an invented one is
            not.
        """
        self.lines: list[VirtualLine] = [
            ln if isinstance(ln, VirtualLine) else self._from_dict(ln)
            for ln in lines
        ]
        self.cooldown_seconds = float(cooldown_seconds)
        self.min_travel = float(min_travel)
        self.warmup_seconds = max(0.0, float(warmup_seconds))
        self._state: dict[int, _TrackLineState] = {}

        if self.lines:
            logger.info(
                "Line crossing engine ready with %d lines: %s",
                len(self.lines), ", ".join(ln.name for ln in self.lines),
            )

    @staticmethod
    def _from_dict(data: dict[str, Any]) -> VirtualLine:
        direction = str(data.get("entering_direction", "north")).lower()
        if direction not in _VALID_DIRECTIONS:
            logger.warning(
                "Line '%s': entering_direction '%s' is not one of %s; "
                "using 'north'.",
                data.get("name"), direction, ", ".join(_VALID_DIRECTIONS),
            )
            direction = "north"
        start = data.get("start", [0.0, 0.5])
        end = data.get("end", [1.0, 0.5])
        return VirtualLine(
            name=str(data.get("name", "Unnamed Line")),
            start=(float(start[0]), float(start[1])),
            end=(float(end[0]), float(end[1])),
            colour=str(data.get("colour", "#00b4d8")),
            entering_direction=direction,
            alerting=bool(data.get("alerting", True)),
        )

    # -- per-frame update ------------------------------------------------

    def update(
        self,
        track_number: int,
        class_name: str,
        point: Point,
        ts: float,
        confidence: float = 0.0,
    ) -> list[CrossingEvent]:
        """Feed one normalised position. Returns any crossings it caused."""
        if not self.lines:
            return []

        state = self._state.setdefault(track_number, _TrackLineState())
        if state.first_ts is None:
            state.first_ts = ts
        previous = state.last_point
        state.last_point = point

        if previous is None:
            return []

        # Warm-up: the box is still converging, so movement of the reference
        # point cannot be told apart from growth of the box.
        if (ts - state.first_ts) < self.warmup_seconds:
            return []

        travelled = ((point[0] - previous[0]) ** 2
                     + (point[1] - previous[1]) ** 2) ** 0.5
        if travelled < self.min_travel:
            return []

        events: list[CrossingEvent] = []
        for line in self.lines:
            # Reference-only geometry: measured by the boundary analyzer,
            # never reported as a crossing. See VirtualLine.alerting.
            if not line.alerting:
                continue
            if not segments_intersect(previous, point, line.start, line.end):
                continue

            last = state.last_crossing_ts.get(line.name)
            if last is not None and (ts - last) < self.cooldown_seconds:
                continue
            state.last_crossing_ts[line.name] = ts
            state.crossing_count += 1

            bearing = heading_degrees(previous, point)
            compass = compass_direction(bearing)
            direction = (
                "entering" if compass == line.entering_direction else "exiting"
            )
            events.append(
                CrossingEvent(
                    line=line,
                    track_number=track_number,
                    class_name=class_name,
                    ts=ts,
                    from_point=previous,
                    to_point=point,
                    compass=compass,
                    direction=direction,
                    confidence=confidence,
                )
            )
        return events

    # -- queries ---------------------------------------------------------

    def crossing_count(self, track_number: int) -> int:
        state = self._state.get(track_number)
        return state.crossing_count if state else 0

    def total_crossings(self) -> int:
        return sum(s.crossing_count for s in self._state.values())

    def forget(self, track_number: int) -> None:
        self._state.pop(track_number, None)

    def as_dicts(self) -> list[dict]:
        return [ln.to_dict() for ln in self.lines]

    def __len__(self) -> int:
        return len(self.lines)
