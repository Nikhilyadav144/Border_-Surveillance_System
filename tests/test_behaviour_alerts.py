"""A behaviour may be an event in its own right.

Until this existed, a behaviour could only ever ADD to the score of an
incident something else had raised. A person could be classified JUMPING
beside a fence and, if they happened not to cross a configured line or enter
a configured zone, the system produced no incident, no alert and no record:
the box said JUMPING and nothing downstream cared.

These tests run the real pipeline with a stub detector - no weights, no
video, no frames - and call the behaviour path directly, so what is asserted
is the policy and the arithmetic rather than whether a particular clip
happens to contain a jump.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from core.behaviour_classifier import BehaviourEvent
from core.pipeline import SurveillancePipeline
from core.tracker import TrackState
from database.models import IncidentType, ThreatLevel
from tests.test_activity import box as skeleton_box
from tests.test_activity import result as pose_result
from tests.test_activity import skeleton


class StubDetector:
    """Stands in for YOLO so the pipeline builds in milliseconds.

    Only the members the pipeline touches during construction and reporting.
    It is never asked to detect anything: these tests drive the behaviour
    path directly rather than through frames.
    """

    model_path = pathlib.Path("stub.pt")
    device = "cpu"

    class stats:
        last_inference_ms = 0.0

    def track(self, frame):
        return []

    def detect(self, frame):
        return []

    def describe(self):
        return {"model": "stub"}


@pytest.fixture
def pipeline(config, db, tmp_path):
    config.set("paths.evidence_dir", str(tmp_path / "evidence"))
    config.ensure_directories()
    camera = {
        "id": "CAM-BEH", "name": "Behaviour", "location": "Test",
        # No geometry on purpose. A behaviour alert must not need a zone or
        # a line to exist, which is the whole point of this path.
        "zones_profile": "none",
        "source": {"type": "video", "path": "unused.mp4"},
    }
    db.upsert_camera("CAM-BEH", "Behaviour", "Test", "video", "unused.mp4")
    return SurveillancePipeline(config=config, camera=camera, database=db,
                                detector=StubDetector())


def person(now: float = 1_757_000_000.0) -> TrackState:
    track = TrackState(track_number=17, class_name="person",
                       camera_id="CAM-BEH", first_ts=now - 2.0, last_ts=now)
    track.frames_seen = 30
    track.samples.append(type("S", (), {
        "ts": now, "point": (0.5, 0.8), "box": (0.45, 0.6, 0.55, 0.9),
        "confidence": 0.9,
    })())
    return track


def event(behaviour: str, confidence: float = 0.9,
          now: float = 1_757_000_000.0) -> BehaviourEvent:
    return BehaviourEvent(track_id=17, behavior=behaviour,
                          confidence=confidence, start_time=now - 0.4,
                          duration=0.4, evidence={"hip_ratio": 0.5})


FRAME = np.zeros((540, 960, 3), dtype=np.uint8)
NOW = 1_757_000_000.0


class TestBehaviourRaisesItsOwnIncident:

    def test_a_jump_alone_produces_an_incident(self, pipeline):
        """No zone, no line, no anomaly. Just a person going over something."""
        incident = pipeline._raise_behaviour_incident(
            person(), event("JUMPING"), FRAME, NOW, person_count=1)
        assert incident is not None
        assert incident.incident_type == IncidentType.SUSPICIOUS_BEHAVIOUR
        assert incident.camera_id == "CAM-BEH"
        assert incident.track_number == 17

    def test_the_jump_is_scored_HIGH_by_the_engine_not_by_a_label(
            self, pipeline):
        """The level has to come out of the arithmetic.

        Asserting the level alone would pass just as happily if somebody had
        written the string HIGH into the incident, so the score is asserted
        beside it: 50 is JUMPING's weight in threat_rules.yaml and 50 is
        where HIGH begins.
        """
        incident = pipeline._raise_behaviour_incident(
            person(), event("JUMPING"), FRAME, NOW, person_count=1)
        assert incident.threat_score >= 50
        assert incident.threat_level == ThreatLevel.HIGH

    def test_the_incident_says_what_the_person_was_doing(self, pipeline):
        """An operator must not have to guess why they were woken."""
        incident = pipeline._raise_behaviour_incident(
            person(), event("CROUCHING"), FRAME, NOW, person_count=1)
        assert any("Crouching" in r.label for r in incident.reasons)
        assert incident.detail["behaviour"] == "CROUCHING"
        assert incident.detail["behaviour_event"]["behavior"] == "CROUCHING"

    def test_the_track_carries_the_verdict_for_the_overlay(self, pipeline):
        """The box the operator looks at and the alert must agree.

        React is not allowed to compute a threat level of its own, so the
        level drawn on the box has to be the one the engine produced here.
        """
        track = person()
        pipeline._raise_behaviour_incident(
            track, event("JUMPING"), FRAME, NOW, person_count=1)
        assert track.threat_level == ThreatLevel.HIGH
        assert track.threat_score >= 50


class TestWhatDoesNotAlert:

    def test_ordinary_behaviour_raises_nothing(self, pipeline):
        for behaviour in ("WALKING", "STANDING", "SITTING", "BENDING",
                          "UNKNOWN"):
            assert pipeline._raise_behaviour_incident(
                person(), event(behaviour), FRAME, NOW, person_count=1
            ) is None, behaviour

    def test_moving_while_bent_now_raises_its_own_event(self, pipeline):
        """A decision that was reversed on evidence, and why.

        Asked early on whether bent-over movement should alert or only add
        to the score, the operator chose the score. That held until CAM-01
        was measured: the classifier never outputs CROUCHING or JUMPING on
        that footage at all. The person the operator describes as "running
        while crouching" is classified BENT_RUN, and the person who goes
        over the fence is caught by the boundary analyser, not by a JUMPING
        label. Under the old list, the one camera with real suspicious
        movement in it produced no behaviour event whatsoever.

        So these two now alert - at the weights they already had, which is
        the point: BENT_WALK is 15 (LOW) and BENT_RUN 25 (MEDIUM), so they
        report without shouting. The original concern was that bent
        movement should not be treated as a serious threat, and the score
        is what enforces that, not silence.
        """
        for behaviour, expected in (("BENT_WALK", ThreatLevel.LOW),
                                    ("BENT_RUN", ThreatLevel.MEDIUM)):
            assert behaviour in pipeline.alerting_behaviours
            track = person()
            track.track_number = 40 + len(behaviour)
            incident = pipeline._raise_behaviour_incident(
                track, event(behaviour), FRAME, NOW, person_count=1)
            assert incident is not None, behaviour
            assert incident.threat_level == expected, behaviour

    def test_bent_movement_is_reported_not_escalated(self, pipeline):
        """It must never outrank a confirmed crossing on its own."""
        bent = pipeline._raise_behaviour_incident(
            person(), event("BENT_RUN"), FRAME, NOW, person_count=1)
        assert bent.threat_score < 50      # below the HIGH threshold

    def test_a_low_confidence_reading_does_not_wake_anybody(self, pipeline):
        assert pipeline._raise_behaviour_incident(
            person(), event("JUMPING", confidence=0.2), FRAME, NOW,
            person_count=1,
        ) is None

    def test_the_alerting_set_is_configuration_not_code(self, pipeline):
        """Emptying the list must switch the whole path off."""
        pipeline.alerting_behaviours = frozenset()
        assert pipeline._raise_behaviour_incident(
            person(), event("JUMPING"), FRAME, NOW, person_count=1) is None


class TestOneEventOneAlert:

    def test_the_same_behaviour_repeating_does_not_alert_twice(
            self, pipeline):
        """The requirement: one behavioural event, one alert.

        A crouch spans dozens of frames. The classifier's debounce, feeding
        only freshly-settled events to this method, and the incident
        manager's dedupe window each guard against a per-frame siren; this
        asserts the last of the three, which is the one that survives a
        person crouching, standing and crouching again inside a second.
        """
        track = person()
        first = pipeline._raise_behaviour_incident(
            track, event("CROUCHING"), FRAME, NOW, person_count=1)
        repeat = pipeline._raise_behaviour_incident(
            track, event("CROUCHING"), FRAME, NOW + 0.5, person_count=1)
        assert first is not None
        assert repeat is None

    def test_a_different_person_doing_the_same_thing_is_its_own_alert(
            self, pipeline):
        first = pipeline._raise_behaviour_incident(
            person(), event("CROUCHING"), FRAME, NOW, person_count=2)
        other = person()
        other.track_number = 18
        second = pipeline._raise_behaviour_incident(
            other, event("CROUCHING", now=NOW), FRAME, NOW + 0.5,
            person_count=2)
        assert first is not None and second is not None
        assert second.track_number == 18


class TestEscalation:

    def test_crouching_in_a_restricted_zone_outranks_crouching_alone(
            self, pipeline, monkeypatch):
        """Escalation must be arithmetic, not a special case.

        30 for the crouch on its own is MEDIUM; the same crouch inside a
        restricted zone adds 30 more and becomes HIGH, with no rule anywhere
        that names the combination.
        """
        plain = pipeline._raise_behaviour_incident(
            person(), event("CROUCHING"), FRAME, NOW, person_count=1)

        monkeypatch.setattr(pipeline, "_in_kind",
                            lambda track, kind: kind == "RESTRICTED")
        in_zone_track = person()
        in_zone_track.track_number = 21
        escalated = pipeline._raise_behaviour_incident(
            in_zone_track, event("CROUCHING"), FRAME, NOW + 60,
            person_count=1)

        assert plain.threat_level == ThreatLevel.MEDIUM
        assert escalated.threat_score > plain.threat_score
        assert escalated.threat_level == ThreatLevel.HIGH


class TestCameraAttribution:

    def test_every_behaviour_incident_names_its_own_camera(self, pipeline):
        """CAM-02 events must never appear as CAM-01 events."""
        incident = pipeline._raise_behaviour_incident(
            person(), event("JUMPING"), FRAME, NOW, person_count=1)
        assert incident.camera_id == pipeline.camera_id == "CAM-BEH"

    def test_a_vehicle_never_produces_a_behaviour_alert(self, pipeline):
        """Behaviour is a pose question, and a car has no pose."""
        car = person()
        car.class_name = "car"
        assert pipeline._raise_behaviour_incident(
            car, event("JUMPING"), FRAME, NOW, person_count=1) is None


class StubPose:
    """A pose estimator that returns a scripted body instead of running one.

    The real estimator needs weights, a GPU decision and a person-shaped crop.
    None of that is what these tests are about: they are about whether a
    behaviour the classifier settles on actually reaches the incident path.
    """

    enabled = True

    def __init__(self, frames):
        self._frames = list(frames)
        self._index = 0

    def should_run(self, analysed_frame_count):
        return True

    def estimate(self, frame, tracks, ts):
        points = self._frames[min(self._index, len(self._frames) - 1)]
        self._index += 1
        return {
            track.track_number: pose_result(points, track.track_number, ts)
            for track in tracks
        }


class TestTheChainIsActuallyWired:
    """Proof the path exists from pose through to an incident.

    The tests above call _raise_behaviour_incident directly, which checks the
    policy but would still pass if nothing ever called it. This drives
    _update_pose with a scripted crouch and asserts an incident comes out the
    far end - the wiring, not just the rule.
    """

    # Standing first, THEN crouching. Not decoration: every measurement in
    # this pipeline is scale-free, expressed in the person's own body
    # heights, so the analyser has to see somebody upright before "lower
    # than usual" means anything. A sequence that opens mid-crouch makes the
    # crouch the baseline and reads as an ordinary stance - correctly.
    STAND_THEN_CROUCH = (
        [skeleton()] * 15
        + [skeleton(knee_bend=min(1.0, i / 5.0)) for i in range(6)]
        + [skeleton(knee_bend=1.0)] * 25
    )

    def _drive(self, pipeline, frames, fps=20.0):
        pipeline.pose = StubPose(frames)
        track = person()
        track.frames_seen = pipeline.tracks.min_hits
        # The track has to be one the tracker knows about. _update_pose
        # prunes pose history for identities the tracker has retired, so a
        # track that exists only in the test is pruned every frame and never
        # accumulates the samples behaviour needs.
        pipeline.tracks.tracks[track.track_number] = track
        raised = []
        for index in range(len(frames)):
            ts = NOW + index / fps
            track.last_ts = ts
            fresh = pipeline._update_pose(FRAME, [track], index, ts)
            for _, ev in fresh:
                incident = pipeline._raise_behaviour_incident(
                    track, ev, FRAME, ts, person_count=1)
                if incident is not None:
                    raised.append(incident)
        return track, raised

    def _crouch(self, pipeline):
        return self._drive(pipeline, self.STAND_THEN_CROUCH)

    def test_update_pose_reports_the_events_that_settled_this_frame(
            self, pipeline):
        """The return value is the contract the incident path depends on.

        `behaviour_events` accumulates for the whole run, so reading it per
        frame would re-raise every past event on every frame. This method
        hands back only what is new.
        """
        track, _ = self._crouch(pipeline)
        assert track.behaviour == "CROUCHING", track.behaviour

    def test_a_scripted_crouch_produces_exactly_one_incident(self, pipeline):
        """40 frames of crouching. One alert, not forty."""
        _, raised = self._crouch(pipeline)
        assert len(raised) == 1
        assert raised[0].incident_type == IncidentType.SUSPICIOUS_BEHAVIOUR
        assert raised[0].threat_level == ThreatLevel.MEDIUM
        assert any("Crouching" in r.label for r in raised[0].reasons)

    def test_standing_still_for_the_same_length_raises_nothing(
            self, pipeline):
        """The control. Without it the test above proves only that the
        method runs, not that it discriminates."""
        track, raised = self._drive(
            pipeline, [skeleton()] * len(self.STAND_THEN_CROUCH))
        assert track.behaviour == "STANDING", track.behaviour
        assert raised == []
