"""Pose estimation: gating, coordinate mapping, and posture geometry.

These tests do not load the real pose weights. The parts worth testing are the
decisions around the model, not the model itself: which crops are worth an
inference, how crop coordinates get back to frame coordinates, and what the
geometry says. A stub model makes those deterministic and keeps the suite fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.pose import (
    KEYPOINT_COUNT,
    KEYPOINT_INDEX,
    KEYPOINT_NAMES,
    POSTURE_BENT,
    POSTURE_HORIZONTAL,
    POSTURE_UNKNOWN,
    POSTURE_UPRIGHT,
    PoseEstimator,
    PoseResult,
    estimate_posture,
)


class FakeTrack:
    """The minimum of a TrackState that the estimator actually touches."""

    def __init__(self, track_number: int, box, class_name: str = "person"):
        self.track_number = track_number
        self.box = box
        self.class_name = class_name


class FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = xy
        self.conf = conf


class FakeOutput:
    def __init__(self, xy, conf):
        self.keypoints = FakeKeypoints(xy, conf)


class FakeModel:
    """Returns one person per crop, joints at fixed fractions of the crop."""

    task = "pose"

    def __init__(self):
        self.calls = 0
        self.batch_sizes = []

    def predict(self, crops, **kwargs):
        self.calls += 1
        self.batch_sizes.append(len(crops))
        outputs = []
        for crop in crops:
            h, w = crop.shape[:2]
            # Every joint at the centre of the crop, full confidence.
            xy = np.tile(np.array([[w / 2.0, h / 2.0]], dtype=np.float32),
                         (KEYPOINT_COUNT, 1))
            conf = np.ones(KEYPOINT_COUNT, dtype=np.float32)
            outputs.append(FakeOutput(xy[None, ...], conf[None, ...]))
        return outputs


def build(**kwargs) -> PoseEstimator:
    """An estimator with the model injected, so nothing is downloaded."""
    estimator = PoseEstimator.__new__(PoseEstimator)
    estimator.enabled = True
    estimator.model_path = __import__("pathlib").Path("models/fake-pose.pt")
    estimator.image_size = kwargs.get("image_size", 192)
    estimator.keypoint_threshold = kwargs.get("keypoint_threshold", 0.30)
    estimator.min_box_height = kwargs.get("min_box_height", 0.12)
    estimator.min_crop_pixels = kwargs.get("min_crop_pixels", 48)
    estimator.max_persons_per_frame = kwargs.get("max_persons_per_frame", 4)
    estimator.interval_frames = kwargs.get("interval_frames", 2)
    from core.pose import PoseStats
    estimator.stats = PoseStats()
    estimator._model = FakeModel()
    estimator._device = "cpu"
    estimator._resolved_device = "cpu"
    estimator._device_label = "CPU (test)"
    estimator._error = None
    return estimator


def frame(width: int = 640, height: int = 480) -> np.ndarray:
    return np.full((height, width, 3), 120, dtype=np.uint8)


class TestInterval:
    def test_interval_counts_analysed_frames(self):
        # Regression. This used to be tested against the SOURCE frame index,
        # which with frame_stride=2 and interval_frames=2 matched every
        # analysed frame and made the setting do nothing.
        estimator = build(interval_frames=2)
        due = [n for n in range(1, 11) if estimator.should_run(n)]
        assert due == [2, 4, 6, 8, 10]

    def test_interval_of_one_runs_every_analysed_frame(self):
        estimator = build(interval_frames=1)
        assert all(estimator.should_run(n) for n in range(1, 10))

    def test_disabled_never_runs(self):
        estimator = build()
        estimator.enabled = False
        assert not any(estimator.should_run(n) for n in range(1, 10))


class TestEligibility:
    def test_a_tall_person_is_eligible(self):
        estimator = build(min_box_height=0.12, min_crop_pixels=48)
        assert estimator.eligible((0.4, 0.2, 0.5, 0.8), frame_height=480)

    def test_a_short_box_is_skipped_by_fraction(self):
        estimator = build(min_box_height=0.12)
        assert not estimator.eligible((0.4, 0.50, 0.42, 0.55), 480)

    def test_a_short_box_is_skipped_by_pixels(self):
        # Passes the fraction rule on a tiny frame but is only 24 px tall.
        estimator = build(min_box_height=0.10, min_crop_pixels=48)
        assert not estimator.eligible((0.4, 0.4, 0.5, 0.6), frame_height=120)

    def test_a_zero_height_box_is_not_eligible(self):
        estimator = build()
        assert not estimator.eligible((0.4, 0.5, 0.5, 0.5), 480)


class TestPixelMapping:
    def test_box_is_padded_and_clamped(self):
        pixels = PoseEstimator._to_pixels((0.0, 0.0, 0.5, 0.5), 640, 480)
        assert pixels is not None
        x1, y1, x2, y2 = pixels
        assert x1 == 0 and y1 == 0, "padding must not go negative"
        assert x2 <= 640 and y2 <= 480

    def test_inverted_box_is_rejected(self):
        assert PoseEstimator._to_pixels((0.6, 0.6, 0.4, 0.4), 640, 480) is None

    def test_non_finite_box_is_rejected(self):
        assert PoseEstimator._to_pixels(
            (0.1, float("nan"), 0.4, 0.5), 640, 480) is None

    def test_a_sliver_is_rejected(self):
        assert PoseEstimator._to_pixels(
            (0.500, 0.500, 0.502, 0.502), 640, 480) is None


class TestEstimate:
    def test_keypoints_come_back_in_frame_coordinates(self):
        # The stub puts every joint at the centre of the CROP. The box spans
        # x 0.4-0.6 and y 0.2-0.8, so the centre of the crop must land at
        # roughly (0.5, 0.5) of the FRAME once mapped back.
        estimator = build()
        track = FakeTrack(7, (0.4, 0.2, 0.6, 0.8))
        results = estimator.estimate(frame(), [track], ts=100.0)

        assert set(results) == {7}
        result = results[7]
        assert result.keypoints.shape == (KEYPOINT_COUNT, 2)
        assert result.keypoints[:, 0] == pytest.approx(0.5, abs=0.02)
        assert result.keypoints[:, 1] == pytest.approx(0.5, abs=0.02)
        assert result.visible.all()
        assert result.visible_count == KEYPOINT_COUNT

    def test_no_tracks_is_not_an_error(self):
        estimator = build()
        assert estimator.estimate(frame(), [], ts=1.0) == {}
        assert estimator._model.calls == 0, "must not invoke the model"

    def test_small_person_is_skipped_and_counted(self):
        estimator = build(min_box_height=0.12)
        tiny = FakeTrack(3, (0.50, 0.50, 0.52, 0.54))
        assert estimator.estimate(frame(), [tiny], ts=1.0) == {}
        assert estimator.stats.skipped_too_small == 1
        assert estimator._model.calls == 0

    def test_invalid_box_is_skipped_and_counted(self):
        estimator = build()
        broken = FakeTrack(4, (0.6, 0.6, 0.4, 0.4))
        assert estimator.estimate(frame(), [broken], ts=1.0) == {}
        assert estimator.stats.skipped_invalid == 1

    def test_track_with_no_box_attribute_is_skipped(self):
        estimator = build()
        nothing = FakeTrack(5, None)
        assert estimator.estimate(frame(), [nothing], ts=1.0) == {}
        assert estimator.stats.skipped_invalid == 1

    def test_budget_caps_the_crowd_largest_first(self):
        estimator = build(max_persons_per_frame=2)
        tracks = [
            FakeTrack(1, (0.00, 0.10, 0.10, 0.90)),   # tallest
            FakeTrack(2, (0.20, 0.20, 0.28, 0.85)),
            FakeTrack(3, (0.40, 0.40, 0.45, 0.75)),   # smallest
        ]
        results = estimator.estimate(frame(), tracks, ts=1.0)
        assert set(results) == {1, 2}, "should keep the two largest"
        assert estimator.stats.skipped_budget == 1
        assert estimator._model.batch_sizes == [2]

    def test_one_batched_call_not_one_per_person(self):
        estimator = build(max_persons_per_frame=4)
        tracks = [FakeTrack(i, (0.1 * i, 0.2, 0.1 * i + 0.08, 0.8))
                  for i in range(1, 4)]
        estimator.estimate(frame(), tracks, ts=1.0)
        assert estimator._model.calls == 1
        assert estimator._model.batch_sizes == [3]

    def test_disabled_estimator_does_nothing(self):
        estimator = build()
        estimator.enabled = False
        track = FakeTrack(1, (0.4, 0.2, 0.6, 0.8))
        assert estimator.estimate(frame(), [track], ts=1.0) == {}


class TestPoseResult:
    @staticmethod
    def make(points: dict[str, tuple[float, float]]) -> PoseResult:
        xy = np.zeros((KEYPOINT_COUNT, 2), dtype=np.float32)
        scores = np.zeros(KEYPOINT_COUNT, dtype=np.float32)
        for name, (x, y) in points.items():
            index = KEYPOINT_INDEX[name]
            xy[index] = (x, y)
            scores[index] = 0.9
        visible = scores >= 0.3
        return PoseResult(track_number=1, ts=0.0, keypoints=xy,
                          scores=scores, visible=visible,
                          confidence=float(scores[visible].mean())
                          if visible.any() else 0.0)

    def test_point_returns_none_for_invisible_joints(self):
        result = self.make({"nose": (0.5, 0.2)})
        assert result.point("nose") == pytest.approx((0.5, 0.2))
        assert result.point("left_ankle") is None

    def test_to_dict_names_every_keypoint(self):
        result = self.make({"nose": (0.5, 0.2)})
        payload = result.to_dict()
        assert set(payload["keypoints"]) == set(KEYPOINT_NAMES)
        assert payload["visible_count"] == 1


class TestPosture:
    make = staticmethod(TestPoseResult.make)

    def test_standing_person_is_upright(self):
        # Shoulders above hips, vertically aligned.
        result = self.make({
            "left_shoulder": (0.48, 0.30), "right_shoulder": (0.52, 0.30),
            "left_hip": (0.48, 0.55), "right_hip": (0.52, 0.55),
        })
        posture = estimate_posture(result)
        assert posture.label == POSTURE_UPRIGHT
        assert posture.torso_angle == pytest.approx(0.0, abs=5)

    def test_person_bent_over_is_bent(self):
        # Torso leaning about 45 degrees.
        result = self.make({
            "left_shoulder": (0.40, 0.40), "right_shoulder": (0.42, 0.40),
            "left_hip": (0.60, 0.60), "right_hip": (0.62, 0.60),
        })
        assert estimate_posture(result).label == POSTURE_BENT

    def test_person_lying_flat_is_horizontal(self):
        result = self.make({
            "left_shoulder": (0.30, 0.70), "right_shoulder": (0.30, 0.72),
            "left_hip": (0.60, 0.71), "right_hip": (0.60, 0.73),
        })
        assert estimate_posture(result).label == POSTURE_HORIZONTAL

    def test_missing_torso_gives_unknown_not_a_guess(self):
        # Only a face. There is no torso vector, so there is no answer.
        result = self.make({"nose": (0.5, 0.3), "left_eye": (0.49, 0.29)})
        posture = estimate_posture(result)
        assert posture.label == POSTURE_UNKNOWN
        assert posture.torso_angle is None

    def test_posture_reports_how_much_it_saw(self):
        result = self.make({
            "left_shoulder": (0.48, 0.30), "right_shoulder": (0.52, 0.30),
            "left_hip": (0.48, 0.55), "right_hip": (0.52, 0.55),
        })
        assert estimate_posture(result).support == 4


class TestStatusReporting:
    def test_status_before_any_run(self):
        estimator = build()
        assert "Ready" in estimator.status_text()

    def test_status_after_runs_reports_real_counts(self):
        estimator = build()
        track = FakeTrack(1, (0.4, 0.2, 0.6, 0.8))
        estimator.estimate(frame(), [track], ts=1.0)
        text = estimator.status_text()
        assert "1 poses from 1 crops" in text

    def test_disabled_says_so(self):
        estimator = build()
        estimator.enabled = False
        assert estimator.status_text() == "Disabled in config"
