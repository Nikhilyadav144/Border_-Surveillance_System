"""Anomaly detection, threat scoring, incident lifecycle and alerts."""

from __future__ import annotations

import numpy as np
import pytest

from core.alert_manager import AlertManager, LogChannel, NotificationChannel
from core.anomaly_engine import (
    STATUS_ANOMALOUS,
    STATUS_DISABLED,
    STATUS_INSUFFICIENT,
    STATUS_NORMAL,
    AnomalyEngine,
)
from core.behaviour_engine import FEATURE_DIM
from core.incident_manager import IncidentManager
from core.threat_engine import ThreatContext, ThreatEngine
from core.watchlist import WatchlistEntry, WatchlistMatcher
from database.models import Incident, IncidentStatus, IncidentType, ThreatLevel

RULES = {
    "factors": {
        "restricted_zone_intrusion": {"weight": 30, "label": "Restricted zone intrusion"},
        "border_zone_intrusion": {"weight": 25, "label": "Border zone entry"},
        "no_vehicle_zone_violation": {"weight": 20, "label": "Vehicle in a no-vehicle zone"},
        "line_crossing": {"weight": 20, "label": "Virtual boundary crossed"},
        "boundary_crossed": {"weight": 40, "label": "Security boundary crossed"},
        "boundary_interacting": {"weight": 25,
                                 "label": "Engaging with the boundary"},
        "low_profile_movement": {"weight": 15,
                                 "label": "Moving while bent over"},
        "loitering": {"weight": 20, "label": "Loitering"},
        "loitering_duration_per_minute": 5,
        "loitering_duration_max": 15,
        "behavioural_anomaly": {"weight": 15, "label": "Behavioural anomaly"},
        "unauthorised_vehicle": {"weight": 20, "label": "Unauthorised vehicle"},
        "watchlist_match": {"weight": 30, "label": "Watchlist match"},
        "night_time": {"weight": 10, "label": "Night-time activity"},
        "group_presence": {"weight": 10, "label": "Group presence", "min_group_size": 3},
        "high_speed": {"weight": 10, "label": "Unusually fast movement", "threshold": 0.35},
    },
    "levels": {"LOW": 0, "MEDIUM": 25, "HIGH": 50, "CRITICAL": 75},
    "minimum_levels": {"watchlist_match": "HIGH"},
}

# 09:00, comfortably outside the 20:00-06:00 night window.
DAYTIME = 1_757_000_000.0


def daytime_engine() -> ThreatEngine:
    engine = ThreatEngine(RULES, night_start_hour=20, night_end_hour=6)
    engine.is_night = lambda ts=None: False   # keep the tests deterministic
    return engine


def walking_features(seed: int = 0, n: int = 40) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        np.array([
            0.08 + rng.normal(0, 0.01),    # avg_speed
            0.15 + rng.normal(0, 0.02),    # max_speed
            2.0 + rng.normal(0, 0.5),      # dwell
            1.2 + rng.normal(0, 0.1),      # distance
            1.0 + rng.integers(0, 2),      # direction changes
            1.0,                           # zone transitions
            0.0,                           # restricted seconds
            14.0 + rng.normal(0, 1.0),     # age
        ], dtype=np.float32)
        for _ in range(n)
    ]


class TestAnomalyEngine:
    def test_reports_insufficient_data_before_it_has_learned(self):
        engine = AnomalyEngine(min_samples=25)
        result = engine.score(np.zeros(FEATURE_DIM, dtype=np.float32))
        assert result.status == STATUS_INSUFFICIENT
        assert result.score is None
        assert result.samples_needed == 25
        assert "Not enough data" in result.describe()

    def test_never_fabricates_a_score_while_learning(self):
        engine = AnomalyEngine(min_samples=25)
        for vector in walking_features(n=10):
            engine.add_sample(vector)
        for _ in range(20):
            assert engine.score(walking_features(n=1)[0]).score is None

    def test_fits_once_enough_samples_arrive(self):
        engine = AnomalyEngine(min_samples=25, contamination=0.1)
        fitted = False
        for vector in walking_features(n=30):
            fitted = engine.add_sample(vector) or fitted
        assert fitted
        assert engine.is_fitted

    def test_typical_behaviour_scores_normal(self):
        engine = AnomalyEngine(min_samples=25, contamination=0.05)
        for vector in walking_features(n=60):
            engine.add_sample(vector)
        result = engine.score(walking_features(seed=99, n=1)[0])
        assert result.is_decided
        assert result.status == STATUS_NORMAL

    def test_wildly_different_behaviour_scores_anomalous(self):
        engine = AnomalyEngine(min_samples=25, contamination=0.05)
        for vector in walking_features(n=60):
            engine.add_sample(vector)
        outlier = np.array([2.5, 4.0, 300.0, 40.0, 25.0, 12.0, 250.0, 400.0],
                           dtype=np.float32)
        result = engine.score(outlier)
        assert result.status == STATUS_ANOMALOUS
        assert result.score is not None and result.score < 0
        assert result.contributions

    def test_disabled_engine_says_so(self):
        engine = AnomalyEngine(enabled=False)
        assert engine.score(np.zeros(FEATURE_DIM)).status == STATUS_DISABLED

    def test_wrong_length_vector_is_rejected_not_padded(self):
        engine = AnomalyEngine(min_samples=5)
        assert engine.add_sample([1.0, 2.0]) is False
        assert engine.sample_count == 0

    def test_save_and_load_round_trip(self, tmp_path):
        path = tmp_path / "anomaly.joblib"
        engine = AnomalyEngine(min_samples=25, model_path=path)
        for vector in walking_features(n=40):
            engine.add_sample(vector)
        assert engine.save() == path

        reloaded = AnomalyEngine(min_samples=25, model_path=path)
        assert reloaded.load()
        assert reloaded.is_fitted

    def test_model_from_a_different_feature_set_is_refused(self, tmp_path):
        import joblib
        path = tmp_path / "stale.joblib"
        joblib.dump({"model": object(), "scaler": object(),
                     "feature_names": ["something", "else"]}, path)
        engine = AnomalyEngine(model_path=path)
        assert engine.load() is False
        assert not engine.is_fitted


class TestThreatEngine:
    def test_intrusion_alone_scores_its_weight(self):
        assessment = daytime_engine().assess(ThreatContext(
            incident_type=IncidentType.RESTRICTED_INTRUSION,
            class_name="person", in_restricted_zone=True,
            zone_name="Compound", ts=DAYTIME,
        ))
        assert assessment.score == 30
        assert assessment.level == ThreatLevel.MEDIUM
        assert len(assessment.reasons) == 1

    def test_a_confirmed_boundary_crossing_outweighs_a_bare_intersection(self):
        """The two boundary factors answer different questions.

        `crossed_line` says the reference point intersected a line segment
        between two frames, which a growing or jittering box can produce.
        `boundary_state == CROSSED` says core/boundary_interaction.py had the
        evidence: an approach to THAT boundary, then the person observed on
        the far side. The second is worth more, and this is where it is paid.
        """
        bare = daytime_engine().assess(ThreatContext(
            incident_type=IncidentType.LINE_CROSSING, class_name="person",
            crossed_line=True, ts=DAYTIME,
        ))
        confirmed = daytime_engine().assess(ThreatContext(
            incident_type=IncidentType.LINE_CROSSING, class_name="person",
            crossed_line=True, boundary_state="CROSSED", ts=DAYTIME,
        ))
        assert bare.score == 20
        assert bare.level == ThreatLevel.LOW
        assert confirmed.score == 60
        assert confirmed.level == ThreatLevel.HIGH

    def test_moving_while_bent_cannot_raise_an_incident_on_its_own(self):
        """Context, not an event.

        Somebody jogging bent into the wind looks the same to a camera as
        somebody keeping their head below a fence line. The factor lifts the
        score of an event that is already happening; on its own it must stay
        below the MEDIUM threshold so it can never be the whole reason for
        an alert.
        """
        alone = daytime_engine().assess(ThreatContext(
            incident_type=IncidentType.LINE_CROSSING, class_name="person",
            low_profile_movement=True, behaviour="BENT_RUN", ts=DAYTIME,
        ))
        assert alone.score == 15
        assert alone.level == ThreatLevel.LOW
        assert "BENT_RUN" in alone.reasons[0].label

    def test_moving_while_bent_adds_to_a_real_event(self):
        without = daytime_engine().assess(ThreatContext(
            incident_type=IncidentType.RESTRICTED_INTRUSION,
            class_name="person", in_restricted_zone=True, ts=DAYTIME,
        ))
        with_it = daytime_engine().assess(ThreatContext(
            incident_type=IncidentType.RESTRICTED_INTRUSION,
            class_name="person", in_restricted_zone=True,
            low_profile_movement=True, behaviour="BENT_RUN", ts=DAYTIME,
        ))
        assert with_it.score == without.score + 15

    def test_factors_accumulate(self):
        assessment = daytime_engine().assess(ThreatContext(
            incident_type=IncidentType.LOITERING, class_name="person",
            in_restricted_zone=True, loitering=True, loitering_seconds=90,
            anomaly_detected=True, anomaly_score=-0.2, ts=DAYTIME,
        ))
        # 30 intrusion + 20 loitering + 5 duration + 15 anomaly
        assert assessment.score == 70
        assert assessment.level == ThreatLevel.HIGH

    def test_score_is_capped_at_100(self):
        assessment = daytime_engine().assess(ThreatContext(
            incident_type=IncidentType.WATCHLIST, class_name="car",
            in_restricted_zone=True, in_border_zone=True,
            vehicle_in_no_vehicle_zone=True, crossed_line=True,
            loitering=True, loitering_seconds=600, anomaly_detected=True,
            unauthorised_vehicle=True, watchlist_match=True,
            group_size=8, speed=1.0, ts=DAYTIME,
        ))
        assert assessment.score == 100
        assert assessment.level == ThreatLevel.CRITICAL

    def test_every_reason_is_explainable(self):
        assessment = daytime_engine().assess(ThreatContext(
            incident_type=IncidentType.LOITERING, class_name="person",
            in_restricted_zone=True, loitering=True, loitering_seconds=125,
            ts=DAYTIME,
        ))
        explanation = assessment.explain()
        assert "Threat Score" in explanation
        assert "Restricted zone intrusion" in explanation
        assert "Loitering" in explanation
        assert "prototype" in explanation.lower()
        assert sum(r.points for r in assessment.reasons) == assessment.score

    def test_loitering_bonus_is_capped(self):
        engine = daytime_engine()
        long_dwell = engine.assess(ThreatContext(
            incident_type=IncidentType.LOITERING, class_name="person",
            loitering=True, loitering_seconds=3600, ts=DAYTIME,
        ))
        bonus = next(r for r in long_dwell.reasons
                     if r.code == "loitering_duration")
        assert bonus.points == 15

    def test_minimum_level_floor_is_applied(self):
        engine = daytime_engine()
        engine.factors["watchlist_match"]["weight"] = 1
        assessment = engine.assess(ThreatContext(
            incident_type=IncidentType.WATCHLIST, class_name="car",
            watchlist_match=True, watchlist_value="DL01AB1234", ts=DAYTIME,
        ))
        assert assessment.score == 1
        assert assessment.level == ThreatLevel.HIGH   # floored, not computed

    def test_zero_weight_factor_never_fires(self):
        engine = daytime_engine()
        engine.set_weight("line_crossing", 0)
        assessment = engine.assess(ThreatContext(
            incident_type=IncidentType.LINE_CROSSING, class_name="person",
            crossed_line=True, ts=DAYTIME,
        ))
        assert assessment.score == 0
        assert assessment.reasons == []

    def test_no_rules_configured_scores_zero(self):
        assessment = ThreatEngine({}).assess(ThreatContext(
            incident_type=IncidentType.RESTRICTED_INTRUSION,
            in_restricted_zone=True, ts=DAYTIME,
        ))
        assert assessment.score == 0
        assert assessment.level == ThreatLevel.LOW

    @pytest.mark.parametrize("hour,expected", [
        (22, True), (2, True), (5, True), (6, False), (12, False), (19, False),
    ])
    def test_night_window_wraps_past_midnight(self, hour, expected):
        import datetime
        engine = ThreatEngine(RULES, night_start_hour=20, night_end_hour=6)
        moment = datetime.datetime(2026, 9, 4, hour, 30).timestamp()
        assert engine.is_night(moment) is expected

    def test_group_threshold_is_configurable(self):
        engine = daytime_engine()
        assert engine.assess(ThreatContext(
            incident_type=IncidentType.RESTRICTED_INTRUSION,
            group_size=2, ts=DAYTIME)).score == 0
        assert engine.assess(ThreatContext(
            incident_type=IncidentType.RESTRICTED_INTRUSION,
            group_size=3, ts=DAYTIME)).score == 10


class TestIncidentManager:
    @staticmethod
    def _assessment():
        return daytime_engine().assess(ThreatContext(
            incident_type=IncidentType.RESTRICTED_INTRUSION,
            class_name="person", in_restricted_zone=True, ts=DAYTIME,
        ))

    def test_first_submission_creates_an_incident(self):
        manager = IncidentManager(dedupe_seconds=20)
        outcome = manager.submit(
            "CAM-01", IncidentType.RESTRICTED_INTRUSION, self._assessment(),
            ts=100.0, track_number=1, object_class="person",
        )
        assert outcome.created
        assert outcome.incident.threat_level == ThreatLevel.MEDIUM

    def test_repeat_within_the_window_is_suppressed(self):
        manager = IncidentManager(dedupe_seconds=20)
        manager.submit("CAM-01", IncidentType.RESTRICTED_INTRUSION,
                       self._assessment(), ts=100.0, track_number=1)
        for step in range(1, 60):
            outcome = manager.submit(
                "CAM-01", IncidentType.RESTRICTED_INTRUSION,
                self._assessment(), ts=100.0 + step * 0.3, track_number=1,
            )
            assert not outcome.created
        assert manager.created_count == 1
        assert manager.suppressed_count == 59

    def test_a_higher_score_escalates_in_place(self):
        manager = IncidentManager(dedupe_seconds=20, escalate_existing=True)
        manager.submit("CAM-01", IncidentType.RESTRICTED_INTRUSION,
                       self._assessment(), ts=100.0, track_number=1)
        worse = daytime_engine().assess(ThreatContext(
            incident_type=IncidentType.RESTRICTED_INTRUSION,
            in_restricted_zone=True, loitering=True, loitering_seconds=120,
            anomaly_detected=True, ts=DAYTIME,
        ))
        outcome = manager.submit("CAM-01", IncidentType.RESTRICTED_INTRUSION,
                                 worse, ts=105.0, track_number=1)
        assert not outcome.created
        assert outcome.escalated
        # 30 intrusion + 20 loitering + 10 duration + 15 anomaly = 75
        assert outcome.incident.threat_score == 75
        assert outcome.incident.threat_level == ThreatLevel.CRITICAL
        assert outcome.incident.escalation_count == 1
        assert manager.created_count == 1

    def test_a_different_type_is_always_a_new_incident(self):
        manager = IncidentManager(dedupe_seconds=60)
        manager.submit("CAM-01", IncidentType.RESTRICTED_INTRUSION,
                       self._assessment(), ts=100.0, track_number=1)
        outcome = manager.submit("CAM-01", IncidentType.LOITERING,
                                 self._assessment(), ts=101.0, track_number=1)
        assert outcome.created
        assert manager.created_count == 2

    def test_a_different_track_is_always_a_new_incident(self):
        manager = IncidentManager(dedupe_seconds=60)
        manager.submit("CAM-01", IncidentType.RESTRICTED_INTRUSION,
                       self._assessment(), ts=100.0, track_number=1)
        outcome = manager.submit("CAM-01", IncidentType.RESTRICTED_INTRUSION,
                                 self._assessment(), ts=101.0, track_number=2)
        assert outcome.created

    def test_after_the_window_a_new_incident_opens(self):
        manager = IncidentManager(dedupe_seconds=20)
        manager.submit("CAM-01", IncidentType.RESTRICTED_INTRUSION,
                       self._assessment(), ts=100.0, track_number=1)
        outcome = manager.submit("CAM-01", IncidentType.RESTRICTED_INTRUSION,
                                 self._assessment(), ts=200.0, track_number=1)
        assert outcome.created

    def test_callback_fires_for_new_incidents_only(self):
        seen: list = []
        manager = IncidentManager(dedupe_seconds=60)
        manager.on_incident = seen.append
        manager.submit("CAM-01", IncidentType.RESTRICTED_INTRUSION,
                       self._assessment(), ts=100.0, track_number=1)
        manager.submit("CAM-01", IncidentType.RESTRICTED_INTRUSION,
                       self._assessment(), ts=101.0, track_number=1)
        assert len(seen) == 1

    def test_lifecycle_with_a_database(self, db):
        manager = IncidentManager(db=db, dedupe_seconds=20)
        db.upsert_camera("CAM-01", "Test", "Alpha", "video", "x.mp4")
        outcome = manager.submit("CAM-01", IncidentType.RESTRICTED_INTRUSION,
                                 self._assessment(), ts=100.0, track_number=1,
                                 object_class="person")
        incident_id = outcome.incident.db_id
        assert db.get_incident(incident_id)["status"] == IncidentStatus.NEW

        assert manager.acknowledge(incident_id)
        assert db.get_incident(incident_id)["status"] == IncidentStatus.ACKNOWLEDGED

        assert manager.resolve(incident_id, note="checked")
        row = db.get_incident(incident_id)
        assert row["status"] == IncidentStatus.RESOLVED
        assert row["resolution_note"] == "checked"

    def test_resolving_stops_deduplication_against_it(self, db):
        manager = IncidentManager(db=db, dedupe_seconds=600)
        db.upsert_camera("CAM-01", "Test", "Alpha", "video", "x.mp4")
        first = manager.submit("CAM-01", IncidentType.RESTRICTED_INTRUSION,
                               self._assessment(), ts=100.0, track_number=1)
        manager.resolve(first.incident.db_id)
        second = manager.submit("CAM-01", IncidentType.RESTRICTED_INTRUSION,
                                self._assessment(), ts=110.0, track_number=1)
        assert second.created


class TestAlertManager:
    def test_alert_is_created_and_dispatched(self, db):
        db.upsert_camera("CAM-01", "Test", "Alpha", "video", "x.mp4")
        incident = Incident(camera_id="CAM-01",
                            incident_type=IncidentType.LOITERING,
                            ts=100.0, threat_score=60,
                            threat_level=ThreatLevel.HIGH,
                            track_number=1, object_class="person",
                            zone_name="Compound")
        db.insert_incident(incident)

        received: list = []

        class Capture(NotificationChannel):
            name = "capture"

            def send(self, alert, incident):
                received.append((alert, incident))
                return True

        manager = AlertManager(db=db, channels=[LogChannel(), Capture()])
        alert = manager.raise_for_incident(incident)

        assert alert is not None
        assert alert.severity == ThreatLevel.HIGH
        assert len(received) == 1
        assert len(db.get_alerts()) == 1

    def test_message_names_the_object_and_the_zone(self):
        manager = AlertManager()
        incident = Incident(camera_id="CAM-01",
                            incident_type=IncidentType.LOITERING,
                            ts=1.0, track_number=7, object_class="person",
                            zone_name="Compound")
        incident.db_id = 1
        alert = manager.raise_for_incident(incident)
        assert "PERSON #07" in alert.message
        assert "Compound" in alert.message

    @pytest.mark.parametrize("severity,expected", [
        ("LOW", False), ("MEDIUM", False), ("HIGH", True), ("CRITICAL", True),
    ])
    def test_toast_threshold(self, severity, expected):
        assert AlertManager(toast_min_level="HIGH").should_toast(severity) is expected

    def test_every_alert_reaches_every_channel(self):
        """Filtering belongs in the GUI, not the dispatcher."""
        received: list = []

        class Capture(NotificationChannel):
            name = "capture"

            def send(self, alert, incident):
                received.append(alert.severity)
                return True

        manager = AlertManager(toast_min_level="CRITICAL",
                               channels=[Capture()])
        for index, level in enumerate(ThreatLevel.ORDER):
            incident = Incident(camera_id="CAM-01",
                                incident_type=IncidentType.LOITERING,
                                ts=float(index), threat_level=level)
            incident.db_id = index + 1
            manager.raise_for_incident(incident)
        assert received == list(ThreatLevel.ORDER)


class TestWatchlist:
    def _matcher(self, near: int = 0) -> WatchlistMatcher:
        return WatchlistMatcher([
            WatchlistEntry(1, "plate", "DL01AB1234", "Stolen vehicle"),
            WatchlistEntry(2, "plate", "HR26XY9876", None),
        ], near_match_distance=near)

    def test_exact_match(self):
        match = self._matcher().match("DL01AB1234")
        assert match is not None and match.exact

    def test_separators_and_case_are_ignored(self):
        assert self._matcher().match("dl 01 ab 1234") is not None

    def test_unknown_plate_does_not_match(self):
        assert self._matcher().match("XX99ZZ0000") is None

    def test_near_matching_is_off_by_default(self):
        assert self._matcher().match("DL01AB1235") is None

    def test_near_match_is_flagged_as_not_exact(self):
        match = self._matcher(near=1).match("DL01AB1235")
        assert match is not None
        assert not match.exact
        assert "needs operator confirmation" in match.describe()

    def test_empty_input_never_matches(self):
        assert self._matcher().match("") is None
        assert self._matcher().match("---") is None

    def test_inactive_entries_are_excluded(self):
        matcher = WatchlistMatcher([
            WatchlistEntry(1, "plate", "DL01AB1234", active=False),
        ])
        assert matcher.match("DL01AB1234") is None
