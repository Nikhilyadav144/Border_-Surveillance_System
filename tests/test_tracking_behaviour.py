"""Track kinematics, loitering and behavioural features."""

from __future__ import annotations

import numpy as np
import pytest

from core.behaviour_engine import FEATURE_DIM, FEATURE_NAMES, BehaviourEngine
from core.tracker import TrackManager, TrackState
from tests.conftest import make_box


def observe(manager: TrackManager, track_id: int, x: float, y: float,
            ts: float, class_name: str = "person", conf: float = 0.9):
    return manager.update(
        [(track_id, class_name, (x, y), make_box(x, y), conf)], ts=ts
    )


class TestTrackManager:
    def test_first_observation_creates_a_track(self):
        manager = TrackManager("CAM-01")
        active, closed = observe(manager, 1, 0.5, 0.5, 1.0)
        assert len(active) == 1
        assert closed == []
        assert manager.total_created == 1
        assert active[0].label == "PERSON #01"

    def test_same_id_does_not_create_a_second_track(self):
        manager = TrackManager("CAM-01")
        for step in range(10):
            observe(manager, 1, 0.5, 0.5 - step * 0.01, 1.0 + step)
        assert manager.total_created == 1
        assert manager.live_count == 1

    def test_track_is_closed_after_max_age(self):
        manager = TrackManager("CAM-01", max_age_frames=3)
        observe(manager, 1, 0.5, 0.5, 1.0)
        closed: list = []
        for step in range(1, 6):
            _, batch = manager.update([], ts=1.0 + step)
            closed += batch
        assert len(closed) == 1
        assert manager.live_count == 0
        assert closed[0].closed

    def test_confirmed_requires_min_hits(self):
        manager = TrackManager("CAM-01", min_hits=3)
        observe(manager, 1, 0.5, 0.5, 1.0)
        assert manager.confirmed() == []
        observe(manager, 1, 0.5, 0.49, 2.0)
        observe(manager, 1, 0.5, 0.48, 3.0)
        assert len(manager.confirmed()) == 1

    def test_a_track_stops_being_published_once_it_stops_matching(self):
        """Regression from real footage.

        PERSON #01 on CAM-01 walked out of frame at f149. His track went on
        being published with the SAME box, the SAME speed of 3.72 body
        heights per second and the SAME label RUNNING for another 60 frames,
        because nothing updates a track that no detection matched. Of the 62
        frames the system called him RUNNING, 61 were that phantom.
        """
        manager = TrackManager("CAM-01", min_hits=3, coast_frames=5,
                               max_age_frames=30)
        for step in range(5):
            observe(manager, 1, 0.5, 0.5 + 0.01 * step, float(step))
        assert len(manager.confirmed()) == 1

        # A short gap is tolerated: one missed detection must not make a
        # person blink out of the operator's table and back in.
        for step in range(5):
            manager.update([], ts=10.0 + step)
        assert len(manager.confirmed()) == 1, "a brief gap should not drop it"

        # Past the coast window it is no longer claimed to be visible.
        manager.update([], ts=20.0)
        assert manager.confirmed() == []
        # But the identity is still alive for re-attachment, because the
        # re-identification window is much longer than the coast window.
        assert manager.get(1) is not None
        assert manager.live_count == 1

    def test_the_person_count_matches_what_is_published(self):
        manager = TrackManager("CAM-01", min_hits=3, coast_frames=2)
        for step in range(4):
            observe(manager, 1, 0.5, 0.5, float(step))
        assert manager.count_class({"person"}) == 1
        for step in range(6):
            manager.update([], ts=10.0 + step)
        assert manager.count_class({"person"}) == 0
        assert len(manager.confirmed()) == manager.count_class({"person"})

    def test_distance_accumulates_along_the_path(self):
        manager = TrackManager("CAM-01")
        for step in range(11):
            observe(manager, 1, 0.1 + step * 0.05, 0.5, 1.0 + step)
        track = manager.get(1)
        assert track.distance_norm == pytest.approx(0.5, abs=1e-3)

    def test_speed_is_distance_over_time(self):
        manager = TrackManager("CAM-01")
        for step in range(6):
            observe(manager, 1, 0.1 + step * 0.1, 0.5, float(step))
        track = manager.get(1)
        # 0.1 of the frame per second.
        assert track.speed == pytest.approx(0.1, abs=0.02)

    def test_heading_is_reported_as_a_compass_word(self):
        manager = TrackManager("CAM-01")
        for step in range(5):
            observe(manager, 1, 0.5, 0.8 - step * 0.05, float(step))
        assert manager.get(1).direction == "north"

    def test_direction_changes_are_counted(self):
        manager = TrackManager("CAM-01", direction_change_degrees=45)
        path = [(0.3, 0.5), (0.4, 0.5), (0.5, 0.5),      # east
                (0.5, 0.4), (0.5, 0.3),                   # turn north
                (0.4, 0.3), (0.3, 0.3)]                   # turn west
        for step, (x, y) in enumerate(path):
            observe(manager, 1, x, y, float(step))
        assert manager.get(1).direction_changes >= 2

    def test_jitter_does_not_count_as_a_direction_change(self):
        manager = TrackManager("CAM-01", direction_change_degrees=45)
        for step in range(40):
            wobble = 0.001 if step % 2 else -0.001
            observe(manager, 1, 0.5 + wobble, 0.5 + wobble, float(step))
        assert manager.get(1).direction_changes == 0

    def test_movement_radius_small_when_stationary(self):
        manager = TrackManager("CAM-01")
        for step in range(30):
            observe(manager, 1, 0.5 + 0.002 * (step % 3), 0.5, float(step))
        assert manager.get(1).movement_radius(30.0) < 0.01

    def test_movement_radius_large_when_walking(self):
        manager = TrackManager("CAM-01")
        for step in range(30):
            observe(manager, 1, 0.05 + step * 0.03, 0.5, float(step))
        assert manager.get(1).movement_radius(30.0) > 0.2

    def test_class_relabel_keeps_the_identity(self):
        manager = TrackManager("CAM-01")
        observe(manager, 1, 0.5, 0.5, 1.0, class_name="car")
        observe(manager, 1, 0.5, 0.5, 2.0, class_name="truck")
        assert manager.total_created == 1
        assert manager.get(1).class_name == "truck"

    def test_close_all_retires_everything(self):
        manager = TrackManager("CAM-01")
        observe(manager, 1, 0.5, 0.5, 1.0)
        observe(manager, 2, 0.2, 0.2, 1.0)
        assert len(manager.close_all()) == 2
        assert manager.live_count == 0


class TestLoitering:
    @staticmethod
    def _standing_track(seconds: float, engine: BehaviourEngine,
                        radius: float = 0.005) -> TrackState:
        manager = TrackManager("CAM-01")
        steps = int(seconds * 5)
        for step in range(steps + 1):
            ts = step / 5.0
            observe(manager, 1, 0.5 + radius * ((step % 3) - 1), 0.5, ts)
        track = manager.get(1)
        track.zone_name = "Restricted Compound"
        track.zone_entered_ts = 0.0
        return track

    def test_short_dwell_is_not_loitering(self):
        engine = BehaviourEngine(loitering_seconds=10.0)
        track = self._standing_track(4.0, engine)
        assert engine.check_loitering(track, now=4.0) is None
        assert not track.loitering

    def test_long_stationary_dwell_is_loitering(self):
        engine = BehaviourEngine(loitering_seconds=10.0)
        track = self._standing_track(15.0, engine)
        event = engine.check_loitering(track, now=15.0)
        assert event is not None
        assert event.zone_name == "Restricted Compound"
        assert event.duration_seconds == pytest.approx(15.0, abs=0.5)
        assert track.loitering

    def test_loitering_is_reported_once_not_every_frame(self):
        engine = BehaviourEngine(loitering_seconds=10.0)
        track = self._standing_track(15.0, engine)
        assert engine.check_loitering(track, now=15.0) is not None
        for extra in range(1, 30):
            assert engine.check_loitering(track, now=15.0 + extra) is None

    def test_walking_through_for_a_long_time_is_not_loitering(self):
        """The point of the test: presence is not loitering."""
        engine = BehaviourEngine(loitering_seconds=10.0, loitering_radius=0.06)
        manager = TrackManager("CAM-01")
        for step in range(80):
            observe(manager, 1, 0.05 + step * 0.011, 0.5, step / 5.0)
        track = manager.get(1)
        track.zone_name = "Restricted Compound"
        track.zone_entered_ts = 0.0
        assert engine.check_loitering(track, now=16.0) is None
        assert not track.loitering

    def test_no_zone_means_no_loitering(self):
        engine = BehaviourEngine(loitering_seconds=1.0)
        track = self._standing_track(15.0, engine)
        track.zone_name = None
        assert engine.check_loitering(track, now=15.0) is None

    def test_too_few_samples_defers_the_judgement(self):
        engine = BehaviourEngine(loitering_seconds=1.0,
                                 min_samples_for_loitering=20)
        track = self._standing_track(2.0, engine)
        assert engine.check_loitering(track, now=2.0) is None

    def test_changing_zone_resets_loitering(self):
        engine = BehaviourEngine(loitering_seconds=10.0)
        track = self._standing_track(15.0, engine)
        engine.check_loitering(track, now=15.0)
        assert track.loitering
        track.enter_zone("Observation Area", ts=16.0)
        assert not track.loitering
        assert not track.loitering_reported


class TestFeatures:
    def test_vector_has_the_documented_shape(self):
        engine = BehaviourEngine()
        manager = TrackManager("CAM-01")
        observe(manager, 1, 0.5, 0.5, 1.0)
        vector = engine.features(manager.get(1))
        assert vector.shape == (FEATURE_DIM,)
        assert vector.dtype == np.float32
        assert len(FEATURE_NAMES) == FEATURE_DIM

    def test_vector_is_always_finite(self):
        engine = BehaviourEngine()
        manager = TrackManager("CAM-01")
        # Two samples at the same instant would divide by zero if unguarded.
        observe(manager, 1, 0.5, 0.5, 1.0)
        observe(manager, 1, 0.9, 0.9, 1.0)
        assert np.all(np.isfinite(engine.features(manager.get(1))))

    def test_named_features_match_the_vector(self):
        engine = BehaviourEngine()
        manager = TrackManager("CAM-01")
        for step in range(5):
            observe(manager, 1, 0.3 + step * 0.05, 0.5, float(step))
        track = manager.get(1)
        named = engine.describe_features(track)
        vector = engine.features(track)
        assert list(named.keys()) == list(FEATURE_NAMES)
        for index, name in enumerate(FEATURE_NAMES):
            assert named[name] == pytest.approx(float(vector[index]), abs=1e-3)

    def test_features_from_row_round_trips(self):
        row = {
            "avg_speed": 0.1, "max_speed": 0.3, "dwell_seconds": 12.0,
            "distance_norm": 1.4, "direction_changes": 3,
            "zone_transitions": 2, "restricted_seconds": 8.0,
            "frames_seen": 90,
        }
        vector = BehaviourEngine.features_from_row(row)
        assert vector.shape == (FEATURE_DIM,)
        assert vector[0] == pytest.approx(0.1)
        assert vector[4] == pytest.approx(3.0)

    def test_missing_row_values_become_zero_not_nan(self):
        vector = BehaviourEngine.features_from_row({"avg_speed": None})
        assert np.all(np.isfinite(vector))
        assert vector[0] == 0.0
