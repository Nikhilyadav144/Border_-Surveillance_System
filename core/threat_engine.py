"""Transparent, configurable threat scoring.

The score is a plain weighted sum. Every factor that fired is recorded with
its label and its point contribution, so an incident can always answer the
question "why is this a 82 and not a 40?".

    INCIDENT #0042
    Threat Score: 82/100   Threat Level: HIGH
    Reasons:
      + 30  Restricted zone intrusion
      + 20  Loitering
      + 10  ... 4m 21s in zone
      + 15  Behavioural anomaly
      + 10  Night-time activity

Positioning
-----------
This is a CONFIGURABLE PROTOTYPE RISK MODEL. The weights in
``config/threat_rules.yaml`` are engineering defaults picked so the demo
behaves sensibly. They are not derived from any military, police or
international security standard and have not been validated against real
incident data. The output ranks events for a human operator. It is not an
authoritative threat assessment and must not be presented as one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from database.models import ThreatLevel, ThreatReason

logger = logging.getLogger(__name__)

#: Shown in the GUI wherever a score is displayed.
DISCLAIMER = (
    "Prototype risk model. Weights are configurable engineering defaults, "
    "not a validated security standard. Always requires human review."
)


@dataclass
class ThreatContext:
    """Everything the scorer is allowed to look at.

    Assembled by the pipeline for one candidate incident. Keeping it explicit
    means the scoring function has no hidden inputs and is trivial to test.
    """

    incident_type: str
    class_name: str = ""
    zone_kind: Optional[str] = None
    zone_name: Optional[str] = None

    in_restricted_zone: bool = False
    in_border_zone: bool = False
    vehicle_in_no_vehicle_zone: bool = False
    crossed_line: bool = False

    loitering: bool = False
    loitering_seconds: float = 0.0

    anomaly_detected: bool = False
    anomaly_score: Optional[float] = None

    watchlist_match: bool = False
    watchlist_value: Optional[str] = None

    unauthorised_vehicle: bool = False
    group_size: int = 0
    speed: float = 0.0

    #: Measured from the frames by :class:`core.low_light.LowLightDetector`,
    #: not read off the clock. The pipeline sets it on every context it builds.
    low_light: bool = False

    #: The person's relationship to a CONFIGURED security boundary, from
    #: :mod:`core.boundary_interaction`. Not the same fact as ``crossed_line``:
    #: that one says the tracker's reference point intersected a line segment
    #: on this frame, which is a geometric event that a jitter can fake. This
    #: one says the crossing had an approach behind it and the person was
    #: afterwards seen on the far side. A crossing carrying that evidence is
    #: worth more than a bare intersection, and this is where the difference
    #: is paid.
    boundary_state: str = "NONE"

    #: The person was travelling with their torso folded well off vertical.
    #: Context, not an incident: it raises the score of whatever else is
    #: happening rather than firing on its own. Somebody jogging bent into
    #: the wind produces this too, which is exactly why it does not alert by
    #: itself.
    low_profile_movement: bool = False
    #: The behaviour label behind that flag, for the operator-facing reason.
    behaviour: Optional[str] = None

    ts: Optional[float] = None


@dataclass
class ThreatAssessment:
    """The scored result."""

    score: float
    level: str
    reasons: list[ThreatReason] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    @property
    def reason_labels(self) -> list[str]:
        return [r.label for r in self.reasons]

    def explain(self) -> str:
        """Multi-line explanation, as shown in the incident detail panel."""
        lines = [f"Threat Score: {self.score:.0f}/100",
                 f"Threat Level: {self.level}", "", "Reasons:"]
        if not self.reasons:
            lines.append("  (no scoring factors fired)")
        for reason in self.reasons:
            lines.append(f"  + {reason.points:>4.0f}  {reason.label}")
        lines.extend(["", self.disclaimer])
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "level": self.level,
            "reasons": [r.to_dict() for r in self.reasons],
        }


class ThreatEngine:
    """Score a :class:`ThreatContext` against the configured rules."""

    def __init__(
        self,
        rules: dict[str, Any] | None = None,
        night_start_hour: int = 20,
        night_end_hour: int = 6,
    ) -> None:
        rules = rules or {}
        self.factors: dict[str, Any] = rules.get("factors") or {}
        self.levels: dict[str, int] = rules.get("levels") or {
            "LOW": 0, "MEDIUM": 25, "HIGH": 50, "CRITICAL": 75
        }
        self.minimum_levels: dict[str, str] = rules.get("minimum_levels") or {}
        self.night_start_hour = int(night_start_hour)
        self.night_end_hour = int(night_end_hour)

        if not self.factors:
            logger.warning(
                "No threat factors are configured. Every incident will score 0 "
                "and be recorded at LOW. Check config/threat_rules.yaml."
            )

    # -- rule access ------------------------------------------------------

    def _weight(self, key: str) -> float:
        factor = self.factors.get(key)
        if isinstance(factor, dict):
            return float(factor.get("weight", 0))
        if isinstance(factor, (int, float)):
            return float(factor)
        return 0.0

    def _label(self, key: str, fallback: str) -> str:
        factor = self.factors.get(key)
        if isinstance(factor, dict):
            return str(factor.get("label", fallback))
        return fallback

    def _setting(self, key: str, name: str, default: Any) -> Any:
        factor = self.factors.get(key)
        if isinstance(factor, dict):
            return factor.get(name, default)
        return default

    def _scalar(self, key: str, default: float = 0.0) -> float:
        value = self.factors.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def is_night(self, ts: float | None = None) -> bool:
        """Whether *ts* falls inside the configured night window.

        NOT used for scoring. The ``night_time`` factor now fires on measured
        scene luminance (``ThreatContext.low_light``), because the clock is a
        poor proxy for darkness: a floodlit compound at 02:00 is not a
        low-light scene, an unlit bay at noon is, and recorded footage
        replayed the next morning is night video with a daytime timestamp.

        Kept as a public utility for schedule-driven features that genuinely
        want the wall clock, such as "was this incident out of hours".
        """
        moment = datetime.fromtimestamp(ts) if ts else datetime.now()
        hour = moment.hour
        start, end = self.night_start_hour, self.night_end_hour
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        # Window wraps past midnight, e.g. 20:00 to 06:00
        return hour >= start or hour < end

    # -- scoring ----------------------------------------------------------

    def assess(self, ctx: ThreatContext) -> ThreatAssessment:
        """Score one context. Pure: same input, same output."""
        reasons: list[ThreatReason] = []

        def add(code: str, fallback_label: str, points: float | None = None,
                suffix: str = "") -> None:
            weight = self._weight(code) if points is None else float(points)
            if weight <= 0:
                return
            label = self._label(code, fallback_label)
            reasons.append(
                ThreatReason(code=code, label=label + suffix, points=weight)
            )

        if ctx.in_restricted_zone:
            add("restricted_zone_intrusion", "Restricted zone intrusion",
                suffix=f" ({ctx.zone_name})" if ctx.zone_name else "")

        if ctx.in_border_zone:
            add("border_zone_intrusion", "Border zone entry",
                suffix=f" ({ctx.zone_name})" if ctx.zone_name else "")

        if ctx.vehicle_in_no_vehicle_zone:
            add("no_vehicle_zone_violation", "Vehicle in a no-vehicle zone",
                suffix=f" ({ctx.zone_name})" if ctx.zone_name else "")

        if ctx.crossed_line:
            add("line_crossing", "Virtual boundary crossed")

        # A crossing that came with an approach, an engagement and a confirmed
        # appearance on the far side is a different claim from a segment
        # intersection, so it scores separately and on top. Both factors are
        # in threat_rules.yaml and either can be set to 0.
        if ctx.boundary_state == "CROSSED":
            add("boundary_crossed", "Security boundary crossed with "
                                    "approach and landing confirmed")
        elif ctx.boundary_state == "INTERACTING":
            add("boundary_interacting", "Physically engaging with the "
                                        "security boundary")

        if ctx.low_profile_movement:
            suffix = f" ({ctx.behaviour})" if ctx.behaviour else ""
            add("low_profile_movement", "Moving while bent over",
                suffix=suffix)

        if ctx.loitering:
            add("loitering", "Loitering",
                suffix=f" ({self._duration_text(ctx.loitering_seconds)})")
            bonus = self._loitering_bonus(ctx.loitering_seconds)
            if bonus > 0:
                reasons.append(ThreatReason(
                    code="loitering_duration",
                    label=f"Extended dwell time "
                          f"({self._duration_text(ctx.loitering_seconds)})",
                    points=bonus,
                ))

        if ctx.anomaly_detected:
            suffix = (f" (score {ctx.anomaly_score:+.3f})"
                      if ctx.anomaly_score is not None else "")
            add("behavioural_anomaly", "Behavioural anomaly", suffix=suffix)

        if ctx.unauthorised_vehicle:
            add("unauthorised_vehicle", "Unauthorised vehicle")

        if ctx.watchlist_match:
            suffix = f" ({ctx.watchlist_value})" if ctx.watchlist_value else ""
            add("watchlist_match", "Watchlist match", suffix=suffix)

        # Measured darkness, not the hour of day. The factor key stays
        # `night_time` so existing threat_rules.yaml files keep working.
        if ctx.low_light:
            add("night_time", "Low-light conditions")

        min_group = int(self._setting("group_presence", "min_group_size", 3))
        if ctx.group_size >= min_group:
            add("group_presence", "Group presence",
                suffix=f" ({ctx.group_size} people)")

        speed_threshold = float(self._setting("high_speed", "threshold", 0.35))
        if ctx.speed >= speed_threshold > 0:
            add("high_speed", "Unusually fast movement")

        total = min(100.0, sum(r.points for r in reasons))
        level = self._level_for(total)

        floor = self.minimum_levels.get(ctx.incident_type)
        if floor and ThreatLevel.rank(floor) > ThreatLevel.rank(level):
            level = str(floor).upper()

        return ThreatAssessment(score=total, level=level, reasons=reasons)

    def _loitering_bonus(self, seconds: float) -> float:
        per_minute = self._scalar("loitering_duration_per_minute", 0.0)
        cap = self._scalar("loitering_duration_max", 0.0)
        if per_minute <= 0 or seconds <= 0:
            return 0.0
        minutes = int(seconds // 60)
        bonus = minutes * per_minute
        return min(bonus, cap) if cap > 0 else bonus

    def _level_for(self, score: float) -> str:
        """Map a score to a level using the configured thresholds."""
        ordered = sorted(
            ((str(name).upper(), float(threshold))
             for name, threshold in self.levels.items()),
            key=lambda kv: kv[1],
        )
        level = ordered[0][0] if ordered else ThreatLevel.LOW
        for name, threshold in ordered:
            if score >= threshold:
                level = name
        return level

    @staticmethod
    def _duration_text(seconds: float) -> str:
        minutes, secs = divmod(int(max(0.0, seconds)), 60)
        return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"

    # -- introspection ----------------------------------------------------

    def describe_rules(self) -> list[tuple[str, str, float]]:
        """(code, label, weight) for the Settings screen."""
        out: list[tuple[str, str, float]] = []
        for code, factor in self.factors.items():
            if isinstance(factor, dict):
                out.append((code, str(factor.get("label", code)),
                            float(factor.get("weight", 0))))
            elif isinstance(factor, (int, float)):
                out.append((code, code.replace("_", " ").title(), float(factor)))
        return sorted(out, key=lambda item: -item[2])

    def set_weight(self, code: str, weight: float) -> None:
        """Change a weight at runtime, from the Settings screen."""
        factor = self.factors.get(code)
        if isinstance(factor, dict):
            factor["weight"] = float(weight)
        else:
            self.factors[code] = {"weight": float(weight), "label": code}
