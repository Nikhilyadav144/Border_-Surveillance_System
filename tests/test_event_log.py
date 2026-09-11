"""The recorded event log behind the Detection Logs page.

Detection Logs is not a second opinion about what happened - it is a reading
of what the pipeline already wrote down. These tests hold that line: the
camera an event belongs to, the time it happened, the level the ThreatEngine
gave it and the words an operator reads are all fixed in the database at the
moment of the event, and the endpoint only formats them.

Nothing here loads a model or reads a video. Rows are written by hand so the
correct answer is known in advance.
"""

from __future__ import annotations

import json

import pytest

from api.server import (
    _event_row,
    _headline,
    camera_events as camera_events_endpoint,
    events as events_endpoint,
)
from database.models import (
    Alert,
    Incident,
    IncidentType,
    ThreatLevel,
    ThreatReason,
)

# 2026-09-11, a few seconds apart, so ordering is unambiguous.
T0 = 1_789_110_874.0


def record(db, *, camera_id, ts, incident_type, level, score, track=1,
           reasons=(), detail=None, zone=None, session_id=1, message=None):
    """Write one incident and, unless told otherwise, its alert.

    Mirrors what the pipeline does: IncidentManager inserts the incident,
    AlertManager inserts the alert that quotes it.
    """
    incident = Incident(
        camera_id=camera_id,
        incident_type=incident_type,
        ts=ts,
        threat_score=score,
        threat_level=level,
        reasons=[ThreatReason(code=c, label=l, points=p)
                 for c, l, p in reasons],
        track_number=track,
        object_class="person",
        zone_name=zone,
        detail=detail or {},
        session_id=session_id,
    )
    db.insert_incident(incident)
    if message is not None:
        db.insert_alert(Alert(
            incident_id=incident.db_id,
            camera_id=camera_id,
            severity=level,
            alert_type=incident_type,
            message=message,
            ts=ts,
            object_label=incident.object_label,
        ))
    return incident


@pytest.fixture
def logged(db):
    """Three cameras, five events, deliberately out of time order."""
    record(db, camera_id="CAM-01", ts=T0 + 14,
           incident_type=IncidentType.SUSPICIOUS_BEHAVIOUR,
           level=ThreatLevel.HIGH, score=50, track=1,
           reasons=[("behaviour_crouching", "Crouching detected", 30)],
           detail={"behaviour": "CROUCHING", "boundary_state": "AT_BOUNDARY"},
           message="PERSON #01 suspicious behaviour - Crouching detected")
    record(db, camera_id="CAM-01", ts=T0 + 123,
           incident_type=IncidentType.RESTRICTED_INTRUSION,
           level=ThreatLevel.HIGH, score=70, track=2,
           reasons=[("restricted_zone_intrusion",
                     "Restricted zone intrusion", 30)],
           detail={"behaviour": "WALKING"}, zone="Restricted - Sports Court",
           message="PERSON #02 restricted zone intrusion")
    record(db, camera_id="CAM-01", ts=T0 + 27,
           incident_type=IncidentType.SUSPICIOUS_BEHAVIOUR,
           level=ThreatLevel.HIGH, score=50, track=2,
           reasons=[("behaviour_jumping", "Jumping detected", 50)],
           detail={"behaviour": "JUMPING"},
           message="PERSON #02 suspicious behaviour - Jumping detected")
    record(db, camera_id="CAM-02", ts=T0 + 41, session_id=2,
           incident_type=IncidentType.LINE_CROSSING,
           level=ThreatLevel.MEDIUM, score=30, track=4,
           detail={"behaviour": "RUNNING"},
           message="PERSON #04 virtual boundary crossing")
    record(db, camera_id="CAM-03", ts=T0 + 60, session_id=3,
           incident_type=IncidentType.LOITERING,
           level=ThreatLevel.LOW, score=20, track=7,
           detail={"behaviour": "UNKNOWN"},
           message="PERSON #07 loitering")
    return db


def rows(db, **kwargs):
    return [_event_row(r) for r in db.get_events(**kwargs)]


class TestCameraFiltering:
    """The requirement that matters most: CAM-02 events are never CAM-01's."""

    def test_one_camera_returns_only_its_own_events(self, logged):
        for camera_id, expected in (("CAM-01", 3), ("CAM-02", 1),
                                    ("CAM-03", 1)):
            got = rows(logged, camera_id=camera_id)
            assert len(got) == expected, camera_id
            assert {r["camera_id"] for r in got} == {camera_id}

    def test_every_event_names_its_camera(self, logged):
        """So a consolidated view can never lose the attribution."""
        assert all(r["camera_id"] for r in rows(logged))

    def test_all_cameras_is_the_only_mixed_view(self, logged):
        got = rows(logged)
        assert len(got) == 5
        assert {r["camera_id"] for r in got} == {"CAM-01", "CAM-02", "CAM-03"}

    def test_a_camera_with_nothing_recorded_returns_nothing(self, logged):
        """Not an error, and above all not somebody else's rows."""
        assert rows(logged, camera_id="CAM-04") == []

    def test_a_set_of_sessions_is_the_consolidated_view(self, logged):
        """ALL CAMERAS is every camera's CURRENT run, not the archive.

        Scoped the same way the single-camera view is, so the two never
        disagree about which run they are describing.
        """
        got = rows(logged, session_ids=[1, 2, 3])
        assert len(got) == 5
        assert {r["camera_id"] for r in got} == {"CAM-01", "CAM-02", "CAM-03"}

    def test_an_old_run_is_left_out_of_the_consolidated_view(self, logged):
        record(logged, camera_id="CAM-01", ts=T0 + 900, session_id=9,
               incident_type=IncidentType.LOITERING,
               level=ThreatLevel.LOW, score=20, message="old run")
        got = rows(logged, session_ids=[9, 2, 3])
        assert {r["session_id"] for r in got} == {9, 2, 3}
        assert len(got) == 3

    def test_no_camera_has_run_returns_nothing_not_everything(self, logged):
        """The dangerous failure mode: an empty scope quietly widening to
        the whole database and presenting old events as this run's."""
        assert rows(logged, session_ids=[]) == []

    def test_a_session_narrows_to_one_run_of_one_camera(self, logged):
        """Re-running a camera must not show the old run's events beside
        the new run's totals."""
        record(logged, camera_id="CAM-01", ts=T0 + 900, session_id=9,
               incident_type=IncidentType.LOITERING,
               level=ThreatLevel.LOW, score=20, track=1,
               message="PERSON #01 loitering")
        first = rows(logged, camera_id="CAM-01", session_id=1)
        second = rows(logged, camera_id="CAM-01", session_id=9)
        assert len(first) == 3
        assert len(second) == 1
        assert {r["session_id"] for r in second} == {9}


class TestOrdering:

    def test_newest_first(self, logged):
        stamps = [r["ts"] for r in rows(logged)]
        assert stamps == sorted(stamps, reverse=True)

    def test_the_timestamp_is_the_event_time_not_the_read_time(self, logged):
        """The log is read long after the run. Every stamp must still be
        the moment the frame was processed."""
        for row in rows(logged):
            assert T0 <= row["ts"] <= T0 + 1000
            assert row["ts_text"].startswith("20")
            assert row["time"].count(":") == 2


class TestWhatTheOperatorReads:

    def test_a_behaviour_event_is_named_by_its_behaviour(self, logged):
        """"Suspicious behaviour" tells nobody anything they can act on."""
        jump = next(r for r in rows(logged, camera_id="CAM-01")
                    if r["behaviour"] == "JUMPING")
        assert jump["headline"] == "Jumping / abnormal movement"
        crouch = next(r for r in rows(logged, camera_id="CAM-01")
                      if r["behaviour"] == "CROUCHING")
        assert crouch["headline"] == "Crouching / suspicious movement"

    def test_other_events_are_named_by_their_type(self, logged):
        assert _headline("restricted_zone_intrusion", None) == \
            "Restricted zone intrusion"
        assert _headline("line_crossing", None) == "Boundary crossing"
        assert _headline("loitering", None) == "Loitering detected"

    def test_an_unnamed_incident_type_is_visibly_unnamed(self):
        """Better a plain reading of the type than a comfortable default
        that quietly mislabels a new incident kind."""
        assert _headline("meteor_strike", None) == "Meteor strike"

    def test_an_unlisted_behaviour_still_reads_as_english(self):
        assert _headline("suspicious_behaviour", "CARTWHEELING") == \
            "Cartwheeling detected"

    def test_the_row_carries_the_sentence_the_operator_was_shown(
            self, logged):
        """One source of truth. The alert text is not rebuilt here."""
        row = next(r for r in rows(logged, camera_id="CAM-02"))
        assert row["message"] == "PERSON #04 virtual boundary crossing"

    def test_scoring_reasons_travel_with_the_event(self, logged):
        row = next(r for r in rows(logged, camera_id="CAM-01")
                   if r["behaviour"] == "JUMPING")
        assert row["reasons"] == ["Jumping detected"]

    def test_no_technical_scores_in_the_headline(self, logged):
        for row in rows(logged):
            assert "=" not in row["headline"]
            assert "score" not in row["headline"].lower()


class TestThreatLevelComesFromPython:

    def test_the_stored_level_is_returned_unchanged(self, logged):
        levels = {r["headline"]: r["threat_level"] for r in rows(logged)}
        assert levels["Jumping / abnormal movement"] == ThreatLevel.HIGH
        assert levels["Crouching / suspicious movement"] == ThreatLevel.HIGH
        assert levels["Boundary crossing"] == ThreatLevel.MEDIUM
        assert levels["Loitering detected"] == ThreatLevel.LOW

    def test_the_score_travels_with_the_level(self, logged):
        """So a client can show the level WITH the number behind it and
        never has to infer one from the other."""
        for row in rows(logged):
            assert row["threat_level"] in ThreatLevel.ORDER
            assert isinstance(row["threat_score"], (int, float))


class TestHonesty:

    def test_an_unknown_behaviour_is_reported_as_absent(self, logged):
        """UNKNOWN means the classifier concluded nothing. Sending the word
        on would invite the interface to display it as a behaviour."""
        row = rows(logged, camera_id="CAM-03")[0]
        assert row["behaviour"] is None

    def test_an_incident_with_no_alert_is_still_an_event(self, db):
        """It happened and it was recorded. Dropping it would make the log
        disagree with the incident count beside it."""
        record(db, camera_id="CAM-01", ts=T0,
               incident_type=IncidentType.LOITERING,
               level=ThreatLevel.LOW, score=10, message=None)
        got = rows(db, camera_id="CAM-01")
        assert len(got) == 1
        assert got[0]["message"] is None
        assert got[0]["headline"] == "Loitering detected"

    def test_a_corrupt_detail_blob_does_not_lose_the_event(self, db):
        """Enrichment failing must cost a field, not a row."""
        record(db, camera_id="CAM-01", ts=T0,
               incident_type=IncidentType.LOITERING,
               level=ThreatLevel.LOW, score=10, message="x")
        db.execute("UPDATE incidents SET detail='{not json', reasons='['")
        got = rows(db, camera_id="CAM-01")
        assert len(got) == 1
        assert got[0]["behaviour"] is None
        assert got[0]["reasons"] == []


class TestOneEventOneRow:

    def test_a_persisting_threat_is_one_row_not_one_per_frame(self, logged):
        """The log does no deduplication of its own, and must not need to:
        IncidentManager already folded the repeats into one incident with an
        escalation count. One row per incident is one row per event."""
        got = rows(logged, camera_id="CAM-01")
        assert len(got) == len(
            logged.get_incidents(camera_id="CAM-01", limit=100))

    def test_an_escalated_incident_reports_how_often_it_repeated(self, db):
        record(db, camera_id="CAM-01", ts=T0,
               incident_type=IncidentType.LOITERING,
               level=ThreatLevel.LOW, score=10, message="x")
        db.execute("UPDATE incidents SET escalation_count=4")
        assert rows(db, camera_id="CAM-01")[0]["escalation_count"] == 4


class FakeRuntime:
    """A camera runtime with only the field this endpoint reads."""

    def __init__(self, session_id):
        self.session_id = session_id


class TestCameraScopedEndpoint:
    """/api/cameras/{id}/events - the camera filter lives in the URL.

    A parameter can be forgotten; a path segment cannot. This endpoint also
    decides the session server-side, so the interface never has to work out
    which run it is looking at.
    """

    def _wire(self, monkeypatch, logged, sessions):
        import api.server as server
        monkeypatch.setattr(server, "db", logged)
        monkeypatch.setattr(server, "runtimes", {
            cam: FakeRuntime(sid) for cam, sid in sessions.items()
        })

    def test_it_returns_only_that_cameras_events(self, logged, monkeypatch):
        self._wire(monkeypatch, logged,
                   {"CAM-01": 1, "CAM-02": 2, "CAM-03": 3})
        result = camera_events_endpoint("CAM-01")
        assert result["camera_id"] == "CAM-01"
        assert result["count"] == 3
        assert {e["camera_id"] for e in result["events"]} == {"CAM-01"}

    def test_it_scopes_to_the_cameras_current_run(self, logged, monkeypatch):
        """The summary and the timeline must describe the same run."""
        record(logged, camera_id="CAM-01", ts=T0 + 900, session_id=9,
               incident_type=IncidentType.LOITERING,
               level=ThreatLevel.LOW, score=20, message="new run")
        self._wire(monkeypatch, logged, {"CAM-01": 9})
        result = camera_events_endpoint("CAM-01")
        assert result["session_id"] == 9
        assert result["count"] == 1

    def test_a_camera_that_never_ran_returns_no_events(self, logged,
                                                       monkeypatch):
        """Not the archive. Old events under a NOT YET RUN heading is the
        exact contradiction this endpoint exists to prevent."""
        self._wire(monkeypatch, logged, {"CAM-01": None})
        result = camera_events_endpoint("CAM-01")
        assert result == {"camera_id": "CAM-01", "scope": "session",
                          "session_id": None, "count": 0, "events": []}

    def test_scope_all_reaches_the_whole_history(self, logged, monkeypatch):
        self._wire(monkeypatch, logged, {"CAM-01": None})
        result = camera_events_endpoint("CAM-01", scope="all")
        assert result["count"] == 3

    def test_an_unknown_camera_is_a_404_not_an_empty_list(self, logged,
                                                          monkeypatch):
        """An empty list would read as "this camera recorded nothing",
        which is a different and wrong answer."""
        from fastapi import HTTPException
        self._wire(monkeypatch, logged, {"CAM-01": 1})
        with pytest.raises(HTTPException) as caught:
            camera_events_endpoint("CAM-99")
        assert caught.value.status_code == 404


class TestAlertCountMatchesEventCount:
    """The contradiction the operator reported: 4 alerts, 0 events.

    Structurally these cannot differ: IncidentManager creates one incident,
    AlertManager raises exactly one alert for it, and this log returns one
    row per incident. These tests pin that chain so a future change that
    breaks it fails here rather than on a dashboard.
    """

    def test_one_incident_yields_one_alert_and_one_row(self, db):
        from core.alert_manager import AlertManager
        from core.incident_manager import IncidentManager
        from core.threat_engine import ThreatAssessment

        incidents = IncidentManager(db=db, dedupe_seconds=20)
        alerts = AlertManager(db=db)
        incidents.on_incident = alerts.raise_for_incident

        assessment = ThreatAssessment(
            score=50, level=ThreatLevel.HIGH,
            reasons=[ThreatReason(code="behaviour_jumping",
                                  label="Jumping detected", points=50)],
        )
        for index in range(5):
            incidents.submit(
                camera_id="CAM-01",
                incident_type=IncidentType.SUSPICIOUS_BEHAVIOUR,
                assessment=assessment,
                ts=T0 + index * 0.2,      # the same event, five frames
                track_number=2,
                object_class="person",
                detail={"behaviour": "JUMPING"},
                session_id=1,
            )

        got = rows(db, camera_id="CAM-01")
        assert len(db.get_alerts(limit=100)) == 1
        assert len(got) == 1
        assert got[0]["headline"] == "Jumping / abnormal movement"
        assert got[0]["threat_level"] == ThreatLevel.HIGH

    def test_a_later_separate_event_is_its_own_record(self, db):
        from core.alert_manager import AlertManager
        from core.incident_manager import IncidentManager
        from core.threat_engine import ThreatAssessment

        incidents = IncidentManager(db=db, dedupe_seconds=20)
        alerts = AlertManager(db=db)
        incidents.on_incident = alerts.raise_for_incident
        assessment = ThreatAssessment(score=50, level=ThreatLevel.HIGH)

        for ts in (T0, T0 + 60):      # well outside the dedupe window
            incidents.submit(
                camera_id="CAM-01",
                incident_type=IncidentType.SUSPICIOUS_BEHAVIOUR,
                assessment=assessment, ts=ts, track_number=2,
                object_class="person", detail={"behaviour": "JUMPING"},
                session_id=1,
            )

        assert len(rows(db, camera_id="CAM-01")) == 2
        assert len(db.get_alerts(limit=100)) == 2


class TestEndpointContract:
    """The endpoint itself, not just the row mapper."""

    def test_the_endpoint_filters_by_camera(self, logged, monkeypatch):
        import api.server as server
        monkeypatch.setattr(server, "db", logged)
        assert {r["camera_id"] for r in events_endpoint(camera_id="CAM-01")} \
            == {"CAM-01"}
        assert len(events_endpoint()) == 5

    def test_the_endpoint_parses_a_session_set(self, logged, monkeypatch):
        import api.server as server
        monkeypatch.setattr(server, "db", logged)
        assert len(events_endpoint(session_ids="1,2,3")) == 5
        assert len(events_endpoint(session_ids="2")) == 1
        assert events_endpoint(session_ids="") == []

    def test_a_malformed_session_set_is_refused_not_ignored(
            self, logged, monkeypatch):
        """Ignoring it would silently widen the query to every camera's
        entire history - the one wrong answer that still looks right."""
        import api.server as server
        from fastapi import HTTPException
        monkeypatch.setattr(server, "db", logged)
        with pytest.raises(HTTPException):
            events_endpoint(session_ids="1,banana")

    def test_the_limit_is_bounded(self, logged, monkeypatch):
        import api.server as server
        monkeypatch.setattr(server, "db", logged)
        assert len(events_endpoint(limit=1)) == 1
        assert len(events_endpoint(limit=0)) == 1        # floored, not empty
        assert len(events_endpoint(limit=10_000)) == 5   # capped, not refused

    def test_every_field_the_interface_needs_is_present(self, logged,
                                                        monkeypatch):
        import api.server as server
        monkeypatch.setattr(server, "db", logged)
        row = events_endpoint(camera_id="CAM-01")[0]
        for field in ("camera_id", "track_id", "object_type", "behaviour",
                      "threat_level", "threat_score", "headline", "message",
                      "reasons", "ts", "ts_text", "time", "incident_type"):
            assert field in row, field
        assert json.dumps(row)   # must survive the wire
