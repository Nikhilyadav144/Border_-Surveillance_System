"""Alert generation and notification dispatch.

An alert is the notification wrapper around an incident: the thing that
appears in the operator's feed and, above a configured severity, pops a toast.

Notification channels are pluggable. The desktop channel is the only one
implemented, because a prototype must not depend on a paid cloud service, but
adding email, SMS, webhook or radio dispatch means writing one
:class:`NotificationChannel` subclass and registering it. Nothing else changes.
"""

from __future__ import annotations

import abc
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Iterable, Optional

from database.models import Alert, Incident, IncidentStatus, ThreatLevel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Notification channels
# ---------------------------------------------------------------------------

class NotificationChannel(abc.ABC):
    """Somewhere an alert can be delivered."""

    name = "channel"

    @abc.abstractmethod
    def send(self, alert: Alert, incident: Incident) -> bool:
        """Deliver one alert. Return True on success."""

    def is_available(self) -> bool:
        return True


class DesktopChannel(NotificationChannel):
    """In-application delivery.

    Holds a callback the GUI installs. The manager never imports Qt, so the
    core stays testable headless.
    """

    name = "desktop"

    def __init__(self, callback: Optional[Callable[[Alert, Incident], None]] = None):
        self.callback = callback

    def send(self, alert: Alert, incident: Incident) -> bool:
        if self.callback is None:
            return False
        try:
            self.callback(alert, incident)
            return True
        except Exception as exc:  # pragma: no cover - GUI safety
            logger.error("Desktop notification failed: %s", exc, exc_info=True)
            return False

    def is_available(self) -> bool:
        return self.callback is not None


class LogChannel(NotificationChannel):
    """Always-available fallback that writes the alert to the log."""

    name = "log"

    def send(self, alert: Alert, incident: Incident) -> bool:
        logger.warning(
            "ALERT [%s] %s | %s | %s",
            alert.severity, alert.camera_id, alert.message,
            incident.summary(),
        )
        return True


# ---------------------------------------------------------------------------
# Alert manager
# ---------------------------------------------------------------------------

@dataclass
class AlertStats:
    total: int = 0
    by_severity: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.by_severity is None:
            self.by_severity = {level: 0 for level in ThreatLevel.ORDER}


class AlertManager:
    """Turn incidents into alerts and dispatch them."""

    def __init__(
        self,
        db: Any = None,
        toast_min_level: str = "HIGH",
        feed_size: int = 200,
        channels: Iterable[NotificationChannel] | None = None,
    ) -> None:
        self.db = db
        self.toast_min_level = str(toast_min_level).upper()
        self.feed: Deque[tuple[Alert, Incident]] = deque(maxlen=int(feed_size))
        self.channels: list[NotificationChannel] = list(channels or [LogChannel()])
        self.stats = AlertStats()

    # -- channels ---------------------------------------------------------

    def add_channel(self, channel: NotificationChannel) -> None:
        self.channels.append(channel)
        logger.info("Notification channel registered: %s", channel.name)

    def channel(self, name: str) -> Optional[NotificationChannel]:
        for channel in self.channels:
            if channel.name == name:
                return channel
        return None

    # -- raising ----------------------------------------------------------

    def raise_for_incident(self, incident: Incident) -> Optional[Alert]:
        """Create, persist and dispatch the alert for a new incident."""
        if incident.db_id is None:
            logger.debug("Incident has no id yet; skipping alert")
            return None

        alert = Alert(
            incident_id=incident.db_id,
            camera_id=incident.camera_id,
            severity=incident.threat_level,
            alert_type=incident.incident_type,
            message=self._message(incident),
            ts=incident.ts,
            object_label=incident.object_label,
            status=IncidentStatus.NEW,
        )

        if self.db is not None:
            try:
                self.db.insert_alert(alert)
            except Exception as exc:
                logger.error("Could not persist alert: %s", exc, exc_info=True)

        self.feed.appendleft((alert, incident))
        self.stats.total += 1
        self.stats.by_severity[alert.severity] = (
            self.stats.by_severity.get(alert.severity, 0) + 1
        )

        # Every available channel receives every alert. Whether an alert
        # becomes a visible pop-up is a presentation decision, made by the GUI
        # using should_toast(), not a dispatch decision made here. Filtering at
        # this level would hide alerts from the feed as well as the toast.
        for channel in self.channels:
            if channel.is_available():
                channel.send(alert, incident)

        return alert

    def _message(self, incident: Incident) -> str:
        """One human sentence describing the incident."""
        parts: list[str] = []
        if incident.object_class and incident.track_number is not None:
            parts.append(incident.object_label)
        parts.append(incident.type_label.lower())
        if incident.zone_name:
            parts.append(f"in {incident.zone_name}")
        sentence = " ".join(parts).strip()
        sentence = sentence[0].upper() + sentence[1:] if sentence else "Incident"
        if incident.reasons:
            top = max(incident.reasons, key=lambda r: r.points)
            if top.label.lower() not in sentence.lower():
                sentence += f" - {top.label}"
        return sentence

    def should_toast(self, severity: str) -> bool:
        return ThreatLevel.at_least(severity, self.toast_min_level)

    # -- workflow ---------------------------------------------------------

    def acknowledge(self, incident_id: int) -> None:
        for alert, _ in self.feed:
            if alert.incident_id == incident_id:
                alert.status = IncidentStatus.ACKNOWLEDGED

    def resolve(self, incident_id: int) -> None:
        for alert, _ in self.feed:
            if alert.incident_id == incident_id:
                alert.status = IncidentStatus.RESOLVED

    # -- queries ----------------------------------------------------------

    def recent(self, limit: int = 50) -> list[tuple[Alert, Incident]]:
        return list(self.feed)[:limit]

    def active_count(self) -> int:
        return sum(1 for alert, _ in self.feed if alert.status == IncidentStatus.NEW)

    def highest_active_severity(self) -> str:
        highest = ThreatLevel.LOW
        for alert, _ in self.feed:
            if alert.status == IncidentStatus.RESOLVED:
                continue
            if ThreatLevel.rank(alert.severity) > ThreatLevel.rank(highest):
                highest = alert.severity
        return highest

    def clear_feed(self) -> None:
        self.feed.clear()
