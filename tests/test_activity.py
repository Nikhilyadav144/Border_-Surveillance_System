"""Person activity: history, movement signals, behaviour classification.

These build skeletons by hand and move them, so every behaviour is checked
against input whose correct answer is known in advance. No model is loaded and
no video is read, which is what makes the thresholds testable at all: with
real footage nobody can assert what the answer should be.

The synthetic person is 0.4 of the frame tall standing, which is a typical
figure in surveillance video.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.behaviour_classifier import (
    BENT_RUN,
    BENT_WALK,
    CRAWLING,
    CROUCHING,
    FALLING,
    JUMPING,
    RUNNING,
    SITTING,
    STANDING,
    UNKNOWN,
    WALKING,
    BehaviourClassifier,
)
from core.movement import MovementAnalyzer
from core.pose import KEYPOINT_COUNT, KEYPOINT_INDEX, PoseResult
from core.pose_history import PoseHistory


HEIGHT = 0.40   # standing height as a fraction of the frame


def skeleton(x: float = 0.5, feet_y: float = 0.9, height: float = HEIGHT,
             torso_lean: float = 0.0, knee_bend: float = 0.0,
             hands_down: bool = False) -> dict[str, tuple[float, float]]:
    """An upright figure, optionally folded.

    ``height`` is the head-to-foot extent. ``knee_bend`` from 0 to 1 raises
    the feet towards the hips, which is what crouching looks like from a
    fixed camera: the head comes down, the feet stay put.
    """
    head_y = feet_y - height
    hip_y = feet_y - height * (0.5 - 0.25 * knee_bend)
    shoulder_y = feet_y - height * (0.82 - 0.1 * knee_bend)
    knee_y = feet_y - height * (0.25 - 0.1 * knee_bend)
    # A bent knee travels FORWARD, not straight down. Without this the hip,
    # knee and ankle stay collinear and the knee angle reads 180 degrees no
    # matter how deep the crouch, which is not what a real body does.
    knee_forward = knee_bend * height * 0.35
    lean = torso_lean * height * 1.4

    points = {
        "nose": (x, head_y),
        "left_ear": (x - 0.01, head_y + 0.005),
        "right_ear": (x + 0.01, head_y + 0.005),
        "left_shoulder": (x - 0.03 + lean, shoulder_y),
        "right_shoulder": (x + 0.03 + lean, shoulder_y),
        "left_elbow": (x - 0.05 + lean, shoulder_y + height * 0.15),
        "right_elbow": (x + 0.05 + lean, shoulder_y + height * 0.15),
        "left_hip": (x - 0.025, hip_y),
        "right_hip": (x + 0.025, hip_y),
        "left_knee": (x - 0.025 + knee_forward, knee_y),
        "right_knee": (x + 0.025 + knee_forward, knee_y),
        "left_ankle": (x - 0.025, feet_y),
        "right_ankle": (x + 0.025, feet_y),
    }
    wrist_y = feet_y - height * 0.05 if hands_down else shoulder_y + height * 0.3
    points["left_wrist"] = (x - 0.06 + lean, wrist_y)
    points["right_wrist"] = (x + 0.06 + lean, wrist_y)
    return points


def result(points: dict[str, tuple[float, float]], track_id: int = 1,
           ts: float = 0.0) -> PoseResult:
    xy = np.zeros((KEYPOINT_COUNT, 2), dtype=np.float32)
    scores = np.zeros(KEYPOINT_COUNT, dtype=np.float32)
    for name, (px, py) in points.items():
        index = KEYPOINT_INDEX[name]
        xy[index] = (px, py)
        scores[index] = 0.9
    visible = scores >= 0.3
    return PoseResult(track_number=track_id, ts=ts, keypoints=xy,
                      scores=scores, visible=visible, confidence=0.9)


def box(points) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points.values()]
    ys = [p[1] for p in points.values()]
    return (min(xs), min(ys), max(xs), max(ys))


def feed(frames, fps: float = 20.0, track_id: int = 1):
    """Push a sequence of skeletons through history and analysis.

    Returns (history, analyzer, list of signals) so a test can assert on any
    stage without rebuilding the chain.
    """
    history = PoseHistory(window_seconds=3.0)
    analyzer = MovementAnalyzer()
    signals = []
    for index, points in enumerate(frames):
        ts = index / fps
        history.record(track_id, result(points, track_id, ts), box(points), ts)
        signals.append(analyzer.analyse(history.get(track_id)))
    return history, analyzer, signals


def classify_sequence(frames, fps: float = 20.0,
                      classifier: BehaviourClassifier | None = None):
    """Run a sequence and return (final state, every event raised)."""
    classifier = classifier or BehaviourClassifier()
    _, _, signals = feed(frames, fps)
    events = []
    for s in signals:
        event = classifier.update(s)
        if event is not None:
            events.append(event)
    return classifier.state(1), events


# ---------------------------------------------------------------------------


class TestHistory:
    def test_samples_are_kept_per_track_id(self):
        history = PoseHistory()
        for track_id in (1, 2):
            for index in range(5):
                points = skeleton(x=0.2 * track_id)
                history.record(track_id, result(points, track_id, index / 20),
                               box(points), index / 20)
        assert len(history) == 2
        assert len(history.get(1).samples) == 5
        assert len(history.get(2).samples) == 5
        # Two people, two separate histories, no bleed between them.
        assert history.get(1).latest.hips[0] != history.get(2).latest.hips[0]

    def test_window_drops_old_samples(self):
        history = PoseHistory(window_seconds=1.0)
        for index in range(60):
            points = skeleton()
            history.record(1, result(points, 1, index / 20), box(points),
                           index / 20)
        assert history.get(1).duration <= 1.05

    def test_reference_height_is_robust_to_one_bad_frame(self):
        # A single frame where the ankle keypoint flies off must not become
        # the person's standing height for the rest of the track.
        history = PoseHistory()
        for index in range(20):
            points = skeleton()
            history.record(1, result(points, 1, index / 20), box(points),
                           index / 20)
        good = history.get(1).reference_height
        bad = skeleton()
        bad["left_ankle"] = (0.5, 0.99)
        bad["right_ankle"] = (0.5, 0.99)
        history.record(1, result(bad, 1, 1.05), box(bad), 1.05)
        assert history.get(1).reference_height == pytest.approx(good, abs=0.02)

    def test_prune_releases_finished_tracks(self):
        history = PoseHistory()
        for track_id in (1, 2, 3):
            points = skeleton()
            history.record(track_id, result(points, track_id, 0.0),
                           box(points), 0.0)
        history.prune({1, 3})
        assert len(history) == 2
        assert history.get(2) is None


class TestMovementSignals:
    def test_not_ready_without_enough_history(self):
        _, _, signals = feed([skeleton()] * 2)
        assert signals[-1].ready is False
        assert "samples" in signals[-1].reason

    def test_not_ready_when_too_few_keypoints_are_visible(self):
        # Only a face. Nothing about the legs can be measured.
        frames = []
        for _ in range(10):
            full = skeleton()
            frames.append({k: full[k] for k in ("nose", "left_ear",
                                                "right_ear")})
        _, _, signals = feed(frames)
        assert signals[-1].ready is False

    def test_speed_is_measured_in_body_heights_per_second(self):
        # Move exactly one body height horizontally in exactly one second.
        frames = [skeleton(x=0.2 + HEIGHT * (i / 20.0)) for i in range(21)]
        _, _, signals = feed(frames, fps=20.0)
        assert signals[-1].speed == pytest.approx(1.0, abs=0.15)

    def test_direction_is_reported_only_when_actually_moving(self):
        still = [skeleton()] * 20
        _, _, signals = feed(still)
        assert signals[-1].direction is None

        moving = [skeleton(x=0.2 + 0.02 * i) for i in range(20)]
        _, _, signals = feed(moving)
        assert signals[-1].direction == "EAST"


class TestPhase1Behaviours:
    def test_standing_still(self):
        state, _ = classify_sequence([skeleton()] * 30)
        assert state.behavior == STANDING

    def test_walking(self):
        # ~0.8 body heights per second.
        frames = [skeleton(x=0.15 + HEIGHT * 0.8 * (i / 20.0))
                  for i in range(30)]
        state, _ = classify_sequence(frames)
        assert state.behavior == WALKING

    def test_running(self):
        # ~2.6 body heights per second.
        frames = [skeleton(x=0.05 + HEIGHT * 2.6 * (i / 20.0))
                  for i in range(25)]
        state, _ = classify_sequence(frames)
        assert state.behavior == RUNNING

    def test_crouching(self):
        frames = [skeleton()] * 15
        frames += [skeleton(knee_bend=min(1.0, i / 5.0)) for i in range(6)]
        frames += [skeleton(knee_bend=1.0)] * 20
        state, _ = classify_sequence(frames)
        assert state.behavior == CROUCHING

    def test_crawling(self):
        # Lower to the ground over about three quarters of a second, then
        # crawl forward. Dropping from upright to prone in a single frame is
        # a FALL, not a crawl, and the classifier is right to say so - an
        # earlier version of this test made that mistake.
        frames = [skeleton()] * 12
        for i in range(1, 16):
            p = i / 15.0
            frames.append(skeleton(height=HEIGHT * (1 - 0.65 * p),
                                   torso_lean=p, knee_bend=p,
                                   hands_down=p > 0.5))
        frames += [skeleton(height=HEIGHT * 0.35, torso_lean=1.0,
                            knee_bend=1.0, hands_down=True,
                            x=0.5 + 0.004 * i) for i in range(30)]
        state, _ = classify_sequence(frames)
        assert state.behavior == CRAWLING

    def test_jumping(self):
        frames = [skeleton()] * 12
        # Up fast, then back down.
        frames += [skeleton(feet_y=0.9 - 0.05 * i) for i in range(1, 4)]
        frames += [skeleton(feet_y=0.9 - 0.05 * i) for i in (3, 2, 1)]
        frames += [skeleton()] * 6
        _, events = classify_sequence(frames)
        assert any(e.behavior == JUMPING for e in events)

    def test_falling(self):
        frames = [skeleton()] * 12
        # Hips drop fast while the torso goes over.
        frames += [skeleton(height=HEIGHT * (1 - 0.18 * i),
                            torso_lean=min(1.0, 0.4 * i),
                            knee_bend=min(1.0, 0.3 * i))
                   for i in range(1, 5)]
        frames += [skeleton(height=HEIGHT * 0.35, torso_lean=1.0,
                            knee_bend=1.0)] * 6
        _, events = classify_sequence(frames)
        assert any(e.behavior == FALLING for e in events)

    def test_sitting_is_between_standing_and_crouching(self):
        frames = [skeleton()] * 15
        frames += [skeleton(knee_bend=0.45)] * 25
        state, _ = classify_sequence(frames)
        assert state.behavior in (SITTING, CROUCHING)


class TestTemporalNotFrameByFrame:
    def test_one_noisy_frame_does_not_change_the_behaviour(self):
        classifier = BehaviourClassifier()
        frames = [skeleton()] * 25
        # One frame where the skeleton collapses, as a bad detection would.
        frames.insert(20, skeleton(height=HEIGHT * 0.3, knee_bend=1.0))
        frames += [skeleton()] * 10
        state, events = classify_sequence(frames, classifier=classifier)
        assert state.behavior == STANDING
        assert not any(e.behavior == CROUCHING for e in events), (
            "a single frame must not be enough to change a person's state"
        )

    def test_a_sustained_change_does_change_it(self):
        frames = [skeleton()] * 20 + [skeleton(knee_bend=1.0)] * 25
        state, events = classify_sequence(frames)
        assert state.behavior == CROUCHING
        assert any(e.behavior == CROUCHING for e in events)

    def test_one_event_per_change_not_one_per_frame(self):
        frames = [skeleton()] * 60
        _, events = classify_sequence(frames)
        assert len(events) <= 1, (
            f"standing still for 3 seconds produced {len(events)} events"
        )

    def test_duration_accumulates_while_the_behaviour_holds(self):
        frames = [skeleton()] * 60
        state, _ = classify_sequence(frames, fps=20.0)
        assert state.duration > 1.5

    def test_event_carries_the_required_fields(self):
        frames = [skeleton()] * 20 + [skeleton(knee_bend=1.0)] * 25
        _, events = classify_sequence(frames)
        event = next(e for e in events if e.behavior == CROUCHING)
        payload = event.to_dict()
        for key in ("track_id", "behavior", "confidence", "duration"):
            assert key in payload
        assert 0.0 < payload["confidence"] <= 1.0


class TestHonesty:
    def test_unknown_when_there_is_not_enough_evidence(self):
        classifier = BehaviourClassifier()
        _, _, signals = feed([skeleton()] * 2)
        name, confidence, evidence = classifier.classify(signals[-1])
        assert name == UNKNOWN
        assert confidence == 0.0
        assert "reason" in evidence

    def test_no_landmarks_are_invented_for_a_partial_body(self):
        frames = []
        for _ in range(15):
            full = skeleton()
            frames.append({k: full[k] for k in
                           ("nose", "left_shoulder", "right_shoulder")})
        _, _, signals = feed(frames)
        assert signals[-1].ready is False
        assert signals[-1].height_ratio is None

    def test_classification_reports_the_numbers_behind_it(self):
        frames = [skeleton(x=0.15 + HEIGHT * 0.8 * (i / 20.0))
                  for i in range(30)]
        classifier = BehaviourClassifier()
        _, _, signals = feed(frames)
        name, _, evidence = classifier.classify(signals[-1])
        assert name == WALKING
        assert "speed" in evidence, "a behaviour must carry its evidence"


class TestMultiplePeople:
    def test_two_people_get_separate_histories_and_behaviours(self):
        history = PoseHistory()
        analyzer = MovementAnalyzer()
        classifier = BehaviourClassifier()

        for index in range(40):
            ts = index / 20.0
            # Person 1 walks. Person 2 stands still.
            walker = skeleton(x=0.15 + HEIGHT * 0.9 * ts)
            stander = skeleton(x=0.8)
            for track_id, points in ((1, walker), (2, stander)):
                history.record(track_id, result(points, track_id, ts),
                               box(points), ts)
                classifier.update(analyzer.analyse(history.get(track_id)))

        assert classifier.state(1).behavior == WALKING
        assert classifier.state(2).behavior == STANDING
        assert classifier.summary() == {WALKING: 1, STANDING: 1}

    def test_forgetting_one_person_leaves_the_other_alone(self):
        history = PoseHistory()
        classifier = BehaviourClassifier()
        analyzer = MovementAnalyzer()
        for track_id in (1, 2):
            for index in range(20):
                ts = index / 20.0
                points = skeleton(x=0.3 * track_id)
                history.record(track_id, result(points, track_id, ts),
                               box(points), ts)
                classifier.update(analyzer.analyse(history.get(track_id)))

        history.prune({1})
        classifier.prune({1})
        assert history.get(2) is None
        assert classifier.state(2) is None
        assert classifier.state(1) is not None


class TestMovingWhileBent:
    """Posture and gait together, which is where the suspicious cases live.

    Every posture rule in the classifier ends with `speed < walk_speed`,
    because crouching, bending and sitting were defined as things you do
    standing still. The consequence was that the moment a person moved,
    their posture was thrown away and only gait was left, so travelling fast
    while folded double came out as WALKING.
    """

    def test_bent_over_and_travelling_is_not_reported_as_walking(self):
        # Torso well over, moving at a brisk walk.
        frames = [skeleton(x=0.10 + HEIGHT * 0.9 * (i / 20.0),
                           torso_lean=0.55) for i in range(40)]
        state, _ = classify_sequence(frames)
        assert state.behavior in (BENT_WALK, BENT_RUN), state.behavior
        assert state.behavior != WALKING

    def test_travelling_fast_while_bent_is_a_bent_run(self):
        frames = [skeleton(x=0.05 + HEIGHT * 2.6 * (i / 20.0),
                           torso_lean=0.55) for i in range(40)]
        state, _ = classify_sequence(frames)
        assert state.behavior == BENT_RUN, state.behavior

    def test_upright_running_is_still_plain_running(self):
        """The new rule must not swallow ordinary running."""
        frames = [skeleton(x=0.05 + HEIGHT * 2.6 * (i / 20.0))
                  for i in range(40)]
        state, _ = classify_sequence(frames)
        assert state.behavior == RUNNING, state.behavior

    def test_bent_over_and_stationary_is_still_bending(self):
        """Standing still and bent is a different thing, and stays so."""
        frames = [skeleton(torso_lean=0.55)] * 40
        state, _ = classify_sequence(frames)
        assert state.behavior not in (BENT_WALK, BENT_RUN), state.behavior
