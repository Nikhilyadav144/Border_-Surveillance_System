"""Typed records that move between the engines, the database and the GUI.

These are plain dataclasses, not an ORM. The application is small enough that
an ORM would add more indirection than it removes, and hand-written SQL keeps
the schema in one readable file.

Everything here is JSON-serialisable through :meth:`to_row`, which is what the
database layer stores.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional


def now_ts() -> float:
    """Current Unix time in seconds."""
    return time.time()


def ts_text(ts: float | None = None) -> str:
    """Human-readable local timestamp for a Unix time."""
    return datetime.fromtimestamp(ts if ts is not None else time.time()).isoformat(
        sep=" ", timespec="seconds"
    )


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def _loads(value: Any, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


# ---------------------------------------------------------------------------
# Enumerated values, kept as plain strings for SQLite friendliness
# ---------------------------------------------------------------------------

class ThreatLevel:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    ORDER = (LOW, MEDIUM, HIGH, CRITICAL)

    @classmethod
    def rank(cls, level: str) -> int:
        try:
            return cls.ORDER.index(str(level).upper())
        except ValueError:
            return 0

    @classmethod
    def at_least(cls, level: str, minimum: str) -> bool:
        return cls.rank(level) >= cls.rank(minimum)


class IncidentStatus:
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    ALL = (NEW, ACKNOWLEDGED, RESOLVED)


class CameraStatus:
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    ERROR = "ERROR"
    CONFIGURED = "CONFIGURED"
    CONNECTING = "CONNECTING"


class IncidentType:
    RESTRICTED_INTRUSION = "restricted_zone_intrusion"
    BORDER_INTRUSION = "border_zone_intrusion"
    NO_VEHICLE_VIOLATION = "no_vehicle_zone_violation"
    LINE_CROSSING = "line_crossing"
    LOITERING = "loitering"
    ANOMALY = "behavioural_anomaly"
    WATCHLIST = "watchlist_match"

    LABELS = {
        RESTRICTED_INTRUSION: "Restricted zone intrusion",
        BORDER_INTRUSION: "Border zone entry",
        NO_VEHICLE_VIOLATION: "Vehicle in no-vehicle zone",
        LINE_CROSSING: "Virtual boundary crossing",
        LOITERING: "Loitering",
        ANOMALY: "Behavioural anomaly",
        WATCHLIST: "Watchlist match",
    }

    @classmethod
    def label(cls, key: str) -> str:
        return cls.LABELS.get(key, key.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """One object found in one frame, before tracking."""

    class_id: int
    class_name: str
    confidence: float
    # Pixel coordinates in the source frame.
    x1: float
    y1: float
    x2: float
    y2: float
    track_number: Optional[int] = None

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def normalised(self, frame_w: int, frame_h: int) -> tuple[float, float, float, float]:
        """Box as fractions of frame size, clamped to 0-1."""
        w = max(frame_w, 1)
        h = max(frame_h, 1)
        return (
            min(max(self.x1 / w, 0.0), 1.0),
            min(max(self.y1 / h, 0.0), 1.0),
            min(max(self.x2 / w, 0.0), 1.0),
            min(max(self.y2 / h, 0.0), 1.0),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameResult:
    """Everything the pipeline learned about one processed frame."""

    camera_id: str
    frame_index: int
    ts: float
    frame_width: int
    frame_height: int
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)

    def count_of(self, names: set[str]) -> int:
        return sum(1 for d in self.detections if d.class_name in names)


@dataclass
class TrackRecord:
    """A tracked identity as persisted."""

    camera_id: str
    track_number: int
    class_name: str
    first_ts: float
    last_ts: float
    frames_seen: int = 0
    avg_confidence: float = 0.0
    max_confidence: float = 0.0
    distance_norm: float = 0.0
    avg_speed: float = 0.0
    max_speed: float = 0.0
    direction_changes: int = 0
    dwell_seconds: float = 0.0
    zone_transitions: int = 0
    zones_visited: list[str] = field(default_factory=list)
    restricted_seconds: float = 0.0
    anomaly_score: Optional[float] = None
    anomaly_status: str = "INSUFFICIENT_DATA"
    trajectory: list[tuple[float, float, float]] = field(default_factory=list)
    session_id: Optional[int] = None
    db_id: Optional[int] = None

    @property
    def track_key(self) -> str:
        return f"{self.camera_id}:{self.class_name}:{self.track_number}"

    @property
    def label(self) -> str:
        return f"{self.class_name.upper()} #{self.track_number:02d}"

    def to_row(self) -> dict:
        return {
            "session_id": self.session_id,
            "camera_id": self.camera_id,
            "track_key": self.track_key,
            "track_number": self.track_number,
            "class_name": self.class_name,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "first_text": ts_text(self.first_ts),
            "frames_seen": self.frames_seen,
            "avg_confidence": self.avg_confidence,
            "max_confidence": self.max_confidence,
            "distance_norm": self.distance_norm,
            "avg_speed": self.avg_speed,
            "max_speed": self.max_speed,
            "direction_changes": self.direction_changes,
            "dwell_seconds": self.dwell_seconds,
            "zone_transitions": self.zone_transitions,
            "zones_visited": _dumps(self.zones_visited),
            "restricted_seconds": self.restricted_seconds,
            "anomaly_score": self.anomaly_score,
            "anomaly_status": self.anomaly_status,
            "trajectory": _dumps(self.trajectory[-240:]),
        }


@dataclass
class ThreatReason:
    """One explainable contribution to a threat score."""

    code: str
    label: str
    points: float

    def to_dict(self) -> dict:
        return {"code": self.code, "label": self.label, "points": round(self.points, 2)}


@dataclass
class Incident:
    """An operator-facing event."""

    camera_id: str
    incident_type: str
    ts: float
    threat_score: float = 0.0
    threat_level: str = ThreatLevel.LOW
    reasons: list[ThreatReason] = field(default_factory=list)
    track_number: Optional[int] = None
    track_db_id: Optional[int] = None
    object_class: Optional[str] = None
    zone_name: Optional[str] = None
    detail: dict = field(default_factory=dict)
    status: str = IncidentStatus.NEW
    session_id: Optional[int] = None
    db_id: Optional[int] = None
    escalation_count: int = 0

    @property
    def object_label(self) -> str:
        if self.track_number is None or not self.object_class:
            return "-"
        return f"{self.object_class.upper()} #{self.track_number:02d}"

    @property
    def type_label(self) -> str:
        return IncidentType.label(self.incident_type)

    def summary(self) -> str:
        parts = [self.type_label]
        if self.zone_name:
            parts.append(f"in {self.zone_name}")
        if self.track_number is not None and self.object_class:
            parts.append(f"({self.object_label})")
        return " ".join(parts)

    def to_row(self) -> dict:
        return {
            "session_id": self.session_id,
            "camera_id": self.camera_id,
            "track_id": self.track_db_id,
            "track_number": self.track_number,
            "object_class": self.object_class,
            "incident_type": self.incident_type,
            "threat_score": round(self.threat_score, 2),
            "threat_level": self.threat_level,
            "reasons": _dumps([r.to_dict() for r in self.reasons]),
            "zone_name": self.zone_name,
            "detail": _dumps(self.detail),
            "ts": self.ts,
            "ts_text": ts_text(self.ts),
            "status": self.status,
            "escalation_count": self.escalation_count,
        }

    @staticmethod
    def reasons_from_row(value: Any) -> list[ThreatReason]:
        return [
            ThreatReason(
                code=item.get("code", ""),
                label=item.get("label", ""),
                points=float(item.get("points", 0)),
            )
            for item in _loads(value, [])
        ]


@dataclass
class Alert:
    """The notification wrapper around an incident."""

    incident_id: int
    camera_id: str
    severity: str
    alert_type: str
    message: str
    ts: float
    object_label: Optional[str] = None
    status: str = IncidentStatus.NEW
    db_id: Optional[int] = None

    def to_row(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "camera_id": self.camera_id,
            "severity": self.severity,
            "alert_type": self.alert_type,
            "message": self.message,
            "object_label": self.object_label,
            "ts": self.ts,
            "ts_text": ts_text(self.ts),
            "status": self.status,
        }


@dataclass
class PlateRead:
    """One ANPR attempt. ``text`` is None when OCR could not read anything."""

    camera_id: str
    track_number: Optional[int]
    text: Optional[str]
    ocr_confidence: float
    backend: str
    ts: float
    validated: bool = False
    evidence_path: Optional[str] = None
    watchlist_hit: bool = False

    @property
    def display(self) -> str:
        return self.text or "UNREADABLE"

    def to_row(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "track_number": self.track_number,
            "plate_text": self.text,
            "ocr_confidence": round(self.ocr_confidence, 4),
            "validated": int(self.validated),
            "backend": self.backend,
            "evidence_path": self.evidence_path,
            "watchlist_hit": int(self.watchlist_hit),
            "ts": self.ts,
            "ts_text": ts_text(self.ts),
        }


@dataclass
class CameraHealth:
    """Live health of one camera, published to the GUI."""

    camera_id: str
    status: str = CameraStatus.OFFLINE
    fps: float = 0.0
    latency_ms: float = 0.0
    frames_seen: int = 0
    last_frame_ts: Optional[float] = None
    error_message: Optional[str] = None

    @property
    def is_online(self) -> bool:
        return self.status == CameraStatus.ONLINE
