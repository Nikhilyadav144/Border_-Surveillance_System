"""Incident lifecycle and deduplication.

An incident is the operator-facing unit of work. It moves through::

    NEW  ->  ACKNOWLEDGED  ->  RESOLVED

The important behaviour here is **deduplication**. One person standing inside
a restricted zone triggers the zone rule on every processed frame. Without
suppression that is hundreds of incidents for a single event, which is exactly
what makes an alert list useless. So:

* a repeat of the same incident type for the same track within
  ``dedupe_seconds`` does not open a new incident
* instead, if the new assessment scores higher, the existing open incident is
  escalated in place and its escalation counter increments
* a *different* incident type for the same track is always a new incident,
  because "loitering" and "crossed the boundary" are different facts

The manager works with or without a database. Passing ``db=None`` keeps
everything in memory, which is what the unit tests use.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core.threat_engine import ThreatAssessment
from database.models import (
    Incident,
    IncidentStatus,
    ThreatLevel,
    ThreatReason,
)

logger = logging.getLogger(__name__)


@dataclass
class _DedupeKey:
    camera_id: str
    track_number: Optional[int]
    incident_type: str

    def as_tuple(self) -> tuple[str, Optional[int], str]:
        return (self.camera_id, self.track_number, self.incident_type)


@dataclass
class IncidentOutcome:
    """What happened when an incident was submitted."""

    incident: Incident
    created: bool
    escalated: bool = False
    suppressed: bool = False

    @property
    def is_actionable(self) -> bool:
        """Whether this should reach the operator as a fresh alert."""
        return self.created


class IncidentManager:
    """Create, deduplicate and transition incidents."""

    def __init__(
        self,
        db: Any = None,
        dedupe_seconds: float = 20.0,
        escalate_existing: bool = True,
    ) -> None:
        self.db = db
        self.dedupe_seconds = float(dedupe_seconds)
        self.escalate_existing = bool(escalate_existing)

        self._recent: dict[tuple[str, Optional[int], str], Incident] = {}
        self._counter = 0
        self.created_count = 0
        self.escalated_count = 0
        self.suppressed_count = 0

        #: Called with every newly created incident. The pipeline uses this
        #: to raise an alert and capture evidence.
        self.on_incident: Optional[Callable[[Incident], None]] = None

    # -- creation ---------------------------------------------------------

    def submit(
        self,
        camera_id: str,
        incident_type: str,
        assessment: ThreatAssessment,
        ts: float | None = None,
        track_number: int | None = None,
        track_db_id: int | None = None,
        object_class: str | None = None,
        zone_name: str | None = None,
        detail: dict | None = None,
        session_id: int | None = None,
    ) -> IncidentOutcome:
        """Offer an event to the manager.

        Returns an :class:`IncidentOutcome` describing whether a new incident
        was opened, an existing one escalated, or the event suppressed.
        """
        moment = ts if ts is not None else time.time()
        key = _DedupeKey(camera_id, track_number, incident_type).as_tuple()

        existing = self._find_open(key, moment)
        if existing is not None:
            return self._handle_repeat(existing, assessment, detail, moment)

        incident = Incident(
            camera_id=camera_id,
            incident_type=incident_type,
            ts=moment,
            threat_score=assessment.score,
            threat_level=assessment.level,
            reasons=list(assessment.reasons),
            track_number=track_number,
            track_db_id=track_db_id,
            object_class=object_class,
            zone_name=zone_name,
            detail=dict(detail or {}),
            status=IncidentStatus.NEW,
            session_id=session_id,
        )

        if self.db is not None:
            try:
                self.db.insert_incident(incident)
            except Exception as exc:
                logger.error("Could not persist incident: %s", exc, exc_info=True)
        else:
            self._counter += 1
            incident.db_id = self._counter

        self._recent[key] = incident
        self.created_count += 1
        logger.info(
            "INCIDENT #%s %s | %s | %s | score %.0f",
            incident.db_id, incident.threat_level, camera_id,
            incident.summary(), incident.threat_score,
        )

        if self.on_incident is not None:
            try:
                self.on_incident(incident)
            except Exception as exc:  # pragma: no cover - callback safety
                logger.error("Incident callback failed: %s", exc, exc_info=True)

        return IncidentOutcome(incident=incident, created=True)

    def _find_open(
        self,
        key: tuple[str, Optional[int], str],
        now: float,
    ) -> Optional[Incident]:
        """The still-open incident this event would duplicate, if any."""
        if key[1] is None:
            # Without a track id there is nothing stable to deduplicate on.
            return None

        cached = self._recent.get(key)
        if cached is not None:
            if now - cached.ts <= self.dedupe_seconds \
                    and cached.status != IncidentStatus.RESOLVED:
                return cached
            del self._recent[key]

        if self.db is None:
            return None

        try:
            row = self.db.find_open_incident(
                camera_id=key[0],
                track_number=key[1],
                incident_type=key[2],
                since_ts=now - self.dedupe_seconds,
            )
        except Exception as exc:
            logger.error("Deduplication lookup failed: %s", exc)
            return None
        if row is None:
            return None

        incident = Incident(
            camera_id=row["camera_id"],
            incident_type=row["incident_type"],
            ts=float(row["ts"]),
            threat_score=float(row["threat_score"]),
            threat_level=row["threat_level"],
            reasons=Incident.reasons_from_row(row["reasons"]),
            track_number=row["track_number"],
            track_db_id=row["track_id"],
            object_class=row["object_class"],
            zone_name=row["zone_name"],
            status=row["status"],
            escalation_count=int(row["escalation_count"] or 0),
        )
        incident.db_id = int(row["id"])
        self._recent[key] = incident
        return incident

    def _handle_repeat(
        self,
        existing: Incident,
        assessment: ThreatAssessment,
        detail: dict | None,
        now: float,
    ) -> IncidentOutcome:
        """Escalate or suppress a repeat of an open incident."""
        if not self.escalate_existing or assessment.score <= existing.threat_score:
            self.suppressed_count += 1
            return IncidentOutcome(
                incident=existing, created=False, suppressed=True
            )

        existing.threat_score = assessment.score
        existing.threat_level = assessment.level
        existing.reasons = list(assessment.reasons)
        existing.escalation_count += 1
        if detail:
            existing.detail.update(detail)

        if self.db is not None and existing.db_id:
            try:
                self.db.escalate_incident(
                    incident_id=existing.db_id,
                    threat_score=existing.threat_score,
                    threat_level=existing.threat_level,
                    reasons_json=json.dumps(
                        [r.to_dict() for r in existing.reasons]
                    ),
                    detail_json=json.dumps(existing.detail, default=str),
                )
            except Exception as exc:
                logger.error("Could not escalate incident: %s", exc)

        self.escalated_count += 1
        logger.info(
            "INCIDENT #%s escalated to %s (score %.0f)",
            existing.db_id, existing.threat_level, existing.threat_score,
        )
        return IncidentOutcome(
            incident=existing, created=False, escalated=True
        )

    # -- lifecycle --------------------------------------------------------

    def acknowledge(self, incident_id: int, operator: str = "operator") -> bool:
        """Move an incident to ACKNOWLEDGED."""
        return self._transition(incident_id, IncidentStatus.ACKNOWLEDGED, operator)

    def resolve(
        self,
        incident_id: int,
        operator: str = "operator",
        note: str | None = None,
    ) -> bool:
        """Move an incident to RESOLVED and stop deduplicating against it."""
        ok = self._transition(incident_id, IncidentStatus.RESOLVED, operator, note)
        if ok:
            for key, incident in list(self._recent.items()):
                if incident.db_id == incident_id:
                    del self._recent[key]
        return ok

    def _transition(
        self,
        incident_id: int,
        status: str,
        operator: str,
        note: str | None = None,
    ) -> bool:
        if status not in IncidentStatus.ALL:
            logger.error("Refusing unknown incident status %r", status)
            return False
        if self.db is not None:
            try:
                self.db.set_incident_status(incident_id, status, operator, note)
            except Exception as exc:
                logger.error("Could not update incident %s: %s", incident_id, exc)
                return False
        for incident in self._recent.values():
            if incident.db_id == incident_id:
                incident.status = status
        logger.info("Incident #%s -> %s by %s", incident_id, status, operator)
        return True

    # -- helpers ----------------------------------------------------------

    def forget_track(self, camera_id: str, track_number: int) -> None:
        """Drop dedupe state for a track that has ended."""
        for key in list(self._recent):
            if key[0] == camera_id and key[1] == track_number:
                del self._recent[key]

    def reset(self) -> None:
        self._recent.clear()

    def stats(self) -> dict[str, int]:
        return {
            "created": self.created_count,
            "escalated": self.escalated_count,
            "suppressed": self.suppressed_count,
        }
