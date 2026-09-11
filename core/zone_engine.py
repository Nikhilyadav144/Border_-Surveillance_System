"""Polygon zone occupancy and intrusion events.

The engine keeps, per tracked object, which zone it currently occupies and
when it entered. From that it produces two kinds of event:

``ZoneEvent(kind="enter")``
    The object crossed into a zone this frame. This is the intrusion signal.
``ZoneEvent(kind="exit")``
    The object left a zone. Used to close out dwell timing.

Two design choices worth knowing about:

* **The reference point is the bottom-centre of the box, not its centre.**
  For a person standing in a scene, the feet are where they actually are; the
  box centre floats at chest height and enters a ground-plane zone early.
  Vehicles use the box centre, which better matches their footprint.

* **Membership is debounced.** A detection box jitters by a few pixels every
  frame, so a target standing exactly on a boundary would otherwise generate
  an enter/exit pair every frame. A change of zone must persist for
  ``debounce_frames`` consecutive observations before it is believed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from core.geometry import Point, bounding_box, point_in_polygon, polygon_centroid

logger = logging.getLogger(__name__)


class ZoneKind:
    RESTRICTED = "RESTRICTED"
    BORDER = "BORDER"
    NO_VEHICLE = "NO_VEHICLE"
    PATROL = "PATROL"
    OBSERVATION = "OBSERVATION"

    #: Kinds where the mere presence of any object is an event worth scoring.
    ALARMING = {RESTRICTED, BORDER}
    #: Kinds where only vehicles matter.
    VEHICLE_ONLY = {NO_VEHICLE}
    ALL = (RESTRICTED, BORDER, NO_VEHICLE, PATROL, OBSERVATION)


@dataclass(frozen=True)
class Zone:
    """One configured polygon."""

    name: str
    kind: str
    polygon: tuple[Point, ...]
    colour: str = "#e5484d"

    @property
    def centroid(self) -> Point:
        return polygon_centroid(self.polygon)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return bounding_box(self.polygon)

    def contains(self, x: float, y: float) -> bool:
        return point_in_polygon(x, y, self.polygon)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "polygon": [list(p) for p in self.polygon],
            "colour": self.colour,
        }


@dataclass
class ZoneEvent:
    """An object entered or left a zone."""

    kind: str                    # "enter" | "exit"
    zone: Zone
    track_number: int
    class_name: str
    ts: float
    point: Point
    confidence: float = 0.0
    dwell_seconds: float = 0.0   # populated on exit

    @property
    def is_alarming(self) -> bool:
        """Whether this entry should feed the threat engine."""
        if self.kind != "enter":
            return False
        if self.zone.kind in ZoneKind.ALARMING:
            return True
        return False

    def describe(self) -> str:
        verb = "entered" if self.kind == "enter" else "left"
        return (f"{self.class_name.upper()} #{self.track_number:02d} "
                f"{verb} {self.zone.name}")


@dataclass
class _Occupancy:
    """Per-track zone state."""

    zone_name: Optional[str] = None
    entered_ts: float = 0.0
    pending_zone: Optional[str] = None
    pending_count: int = 0
    visited: list[str] = field(default_factory=list)
    transitions: int = 0


class ZoneEngine:
    """Track which zone each object is in and emit enter/exit events."""

    def __init__(
        self,
        zones: Iterable[dict[str, Any]] | Iterable[Zone] = (),
        vehicle_classes: Iterable[str] = (),
        debounce_frames: int = 2,
    ) -> None:
        self.zones: list[Zone] = [
            z if isinstance(z, Zone) else self._from_dict(z) for z in zones
        ]
        self.vehicle_classes = {c.lower() for c in vehicle_classes}
        self.debounce_frames = max(1, int(debounce_frames))
        self._state: dict[int, _Occupancy] = {}

        if self.zones:
            logger.info(
                "Zone engine ready with %d zones: %s",
                len(self.zones),
                ", ".join(f"{z.name} [{z.kind}]" for z in self.zones),
            )
        else:
            logger.info("Zone engine ready with no zones configured")

    # -- construction ----------------------------------------------------

    @staticmethod
    def _from_dict(data: dict[str, Any]) -> Zone:
        polygon = tuple(
            (float(p[0]), float(p[1])) for p in (data.get("polygon") or [])
        )
        kind = str(data.get("kind", ZoneKind.OBSERVATION)).upper()
        if kind not in ZoneKind.ALL:
            logger.warning(
                "Zone '%s' has unknown kind '%s'; treating it as OBSERVATION. "
                "Valid kinds: %s",
                data.get("name"), kind, ", ".join(ZoneKind.ALL),
            )
            kind = ZoneKind.OBSERVATION
        return Zone(
            name=str(data.get("name", "Unnamed Zone")),
            kind=kind,
            polygon=polygon,
            colour=str(data.get("colour", "#e5484d")),
        )

    # -- reference point -------------------------------------------------

    def reference_point(
        self,
        box_norm: tuple[float, float, float, float],
        class_name: str,
    ) -> Point:
        """Where the object 'is', in normalised coordinates."""
        x1, y1, x2, y2 = box_norm
        cx = (x1 + x2) / 2.0
        if class_name.lower() in self.vehicle_classes:
            return (cx, (y1 + y2) / 2.0)
        # People stand on the ground plane: use the bottom edge.
        return (cx, y2)

    def zone_at(self, x: float, y: float) -> Optional[Zone]:
        """The first zone containing the point.

        Zones are tested in configuration order, so a small high-priority
        polygon listed before a large observation area wins the overlap.
        """
        for zone in self.zones:
            if zone.contains(x, y):
                return zone
        return None

    # -- per-frame update ------------------------------------------------

    def update(
        self,
        track_number: int,
        class_name: str,
        box_norm: tuple[float, float, float, float],
        ts: float,
        confidence: float = 0.0,
    ) -> list[ZoneEvent]:
        """Feed one observation. Returns any enter/exit events it produced."""
        if not self.zones:
            return []

        point = self.reference_point(box_norm, class_name)
        current = self.zone_at(*point)
        current_name = current.name if current else None

        state = self._state.setdefault(track_number, _Occupancy())
        events: list[ZoneEvent] = []

        if current_name == state.zone_name:
            # Stable. Any pending change was noise.
            state.pending_zone = None
            state.pending_count = 0
            return events

        # A change is proposed. Require it to persist before believing it.
        if current_name == state.pending_zone:
            state.pending_count += 1
        else:
            state.pending_zone = current_name
            state.pending_count = 1

        if state.pending_count < self.debounce_frames:
            return events

        previous_name = state.zone_name
        if previous_name:
            previous = self._zone_by_name(previous_name)
            if previous is not None:
                events.append(
                    ZoneEvent(
                        kind="exit",
                        zone=previous,
                        track_number=track_number,
                        class_name=class_name,
                        ts=ts,
                        point=point,
                        confidence=confidence,
                        dwell_seconds=max(0.0, ts - state.entered_ts),
                    )
                )

        state.zone_name = current_name
        state.entered_ts = ts
        state.pending_zone = None
        state.pending_count = 0

        if current is not None:
            if current.name not in state.visited:
                state.visited.append(current.name)
            state.transitions += 1
            events.append(
                ZoneEvent(
                    kind="enter",
                    zone=current,
                    track_number=track_number,
                    class_name=class_name,
                    ts=ts,
                    point=point,
                    confidence=confidence,
                )
            )
        return events

    # -- queries ---------------------------------------------------------

    def _zone_by_name(self, name: str) -> Optional[Zone]:
        for zone in self.zones:
            if zone.name == name:
                return zone
        return None

    def current_zone(self, track_number: int) -> Optional[str]:
        state = self._state.get(track_number)
        return state.zone_name if state else None

    def dwell_seconds(self, track_number: int, now: float | None = None) -> float:
        state = self._state.get(track_number)
        if not state or not state.zone_name:
            return 0.0
        return max(0.0, (now if now is not None else time.time()) - state.entered_ts)

    def visited_zones(self, track_number: int) -> list[str]:
        state = self._state.get(track_number)
        return list(state.visited) if state else []

    def transition_count(self, track_number: int) -> int:
        state = self._state.get(track_number)
        return state.transitions if state else 0

    def is_violation(self, zone: Zone, class_name: str) -> bool:
        """Whether this class being in this zone is a scoreable violation."""
        if zone.kind in ZoneKind.ALARMING:
            return True
        if zone.kind in ZoneKind.VEHICLE_ONLY:
            return class_name.lower() in self.vehicle_classes
        return False

    def forget(self, track_number: int) -> None:
        """Drop state for a track that has ended."""
        self._state.pop(track_number, None)

    def occupancy_counts(self) -> dict[str, int]:
        """How many tracked objects are currently inside each zone."""
        counts = {zone.name: 0 for zone in self.zones}
        for state in self._state.values():
            if state.zone_name in counts:
                counts[state.zone_name] += 1
        return counts

    def as_dicts(self) -> list[dict]:
        return [z.to_dict() for z in self.zones]

    def __len__(self) -> int:
        return len(self.zones)
