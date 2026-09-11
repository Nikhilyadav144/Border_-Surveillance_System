"""Low-light detection: measured from the frame, stable over time.

The point of these tests is that the verdict follows the picture and nothing
else. The clock is never consulted, so none of these tests can be made to pass
or fail by running them at a different hour, which is exactly the property the
old clock-based night factor lacked.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.low_light import LowLightDetector, LowLightState
from core.threat_engine import ThreatContext, ThreatEngine


def frame(value: int, width: int = 320, height: int = 180) -> np.ndarray:
    """A flat BGR frame of one grey value, so mean luminance is that value."""
    return np.full((height, width, 3), value, dtype=np.uint8)


class TestMeasurement:
    def test_flat_frame_measures_its_own_value(self):
        detector = LowLightDetector()
        assert detector.measure_frame(frame(0)) == pytest.approx(0.0, abs=1.0)
        assert detector.measure_frame(frame(255)) == pytest.approx(255.0,
                                                                   abs=1.0)
        assert detector.measure_frame(frame(128)) == pytest.approx(128.0,
                                                                   abs=1.0)

    def test_downscaling_preserves_the_mean(self):
        # Half black, half white. The mean is 127 whatever the sample width,
        # which is what makes the cheap downscale trustworthy.
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        image[:, 160:] = 255
        for width in (16, 64, 256):
            detector = LowLightDetector(sample_width=width)
            assert detector.measure_frame(image) == pytest.approx(127.5, abs=3)

    def test_accepts_a_single_channel_frame(self):
        detector = LowLightDetector()
        grey = np.full((180, 320), 40, dtype=np.uint8)
        assert detector.measure_frame(grey) == pytest.approx(40.0, abs=1.0)


class TestColdStart:
    def test_says_measuring_until_it_has_enough_samples(self):
        detector = LowLightDetector(min_samples=5, window_frames=15)
        for _ in range(4):
            state = detector.update(frame(10))
            assert state.ready is False
            assert state.low_light is False
            assert state.luminance is None
            assert "Measuring" in state.status_text()

        state = detector.update(frame(10))
        assert state.ready is True
        assert state.low_light is True

    def test_disabled_detector_reports_disabled_and_never_fires(self):
        detector = LowLightDetector(enabled=False)
        state = detector.update(frame(0))
        assert state.disabled is True
        assert state.low_light is False
        assert state.status_text() == "Disabled in config"


class TestHysteresis:
    def test_dark_scene_becomes_low_light(self):
        detector = LowLightDetector(enter_below=60, exit_above=80,
                                    min_samples=3, window_frames=5)
        for _ in range(5):
            state = detector.update(frame(20))
        assert state.low_light is True
        assert state.luminance == pytest.approx(20, abs=2)

    def test_bright_scene_does_not(self):
        detector = LowLightDetector(enter_below=60, exit_above=80,
                                    min_samples=3, window_frames=5)
        for _ in range(5):
            state = detector.update(frame(200))
        assert state.low_light is False

    def test_the_gap_between_thresholds_holds_the_state(self):
        # 70 sits between enter_below and exit_above. Whatever the state was,
        # it stays. A single-threshold design would oscillate here.
        detector = LowLightDetector(enter_below=60, exit_above=80,
                                    min_samples=3, window_frames=3)
        for _ in range(3):
            detector.update(frame(20))
        assert detector.state.low_light is True

        for _ in range(3):
            state = detector.update(frame(70))
        assert state.low_light is True, "should not leave low light at 70"

        for _ in range(3):
            state = detector.update(frame(200))
        assert state.low_light is False

        for _ in range(3):
            state = detector.update(frame(70))
        assert state.low_light is False, "should not enter low light at 70"

    def test_one_dark_frame_does_not_flip_a_lit_scene(self):
        detector = LowLightDetector(enter_below=60, exit_above=80,
                                    min_samples=5, window_frames=15)
        for _ in range(15):
            detector.update(frame(180))
        state = detector.update(frame(0))
        assert state.low_light is False, "a single black frame is not night"

    def test_a_sustained_change_does_flip_it(self):
        detector = LowLightDetector(enter_below=60, exit_above=80,
                                    min_samples=5, window_frames=15)
        for _ in range(15):
            detector.update(frame(180))
        assert detector.state.low_light is False
        for _ in range(15):
            state = detector.update(frame(15))
        assert state.low_light is True

    def test_reset_clears_the_window(self):
        detector = LowLightDetector(min_samples=3, window_frames=5)
        for _ in range(5):
            detector.update(frame(10))
        assert detector.state.low_light is True
        detector.reset()
        assert detector.state.ready is False
        assert detector.state.low_light is False


class TestConfigRepair:
    def test_thresholds_without_a_gap_are_repaired(self):
        # A config with exit <= enter would flap. The detector fixes it rather
        # than obeying it, because obeying it breaks the feature silently.
        detector = LowLightDetector(enter_below=60, exit_above=60)
        assert detector.exit_above > detector.enter_below

    def test_min_samples_cannot_exceed_the_window(self):
        detector = LowLightDetector(window_frames=5, min_samples=50)
        assert detector.min_samples <= detector.window_frames


class TestThreatIntegration:
    """The night_time factor must follow the frame, not the hour."""

    RULES = {
        "factors": {
            "restricted_zone_intrusion": {"weight": 30},
            "night_time": {"weight": 10, "label": "Low-light conditions"},
        },
        "levels": {"LOW": 0, "MEDIUM": 25, "HIGH": 50, "CRITICAL": 75},
    }

    def test_low_light_context_adds_the_factor(self):
        engine = ThreatEngine(self.RULES)
        lit = engine.assess(ThreatContext(incident_type="restricted_zone_intrusion",
                                          in_restricted_zone=True,
                                          low_light=False))
        dark = engine.assess(ThreatContext(incident_type="restricted_zone_intrusion",
                                           in_restricted_zone=True,
                                           low_light=True))
        assert lit.score == 30
        assert dark.score == 40
        assert any(r.code == "night_time" for r in dark.reasons)
        assert not any(r.code == "night_time" for r in lit.reasons)

    def test_the_clock_no_longer_changes_the_score(self):
        # Same context, two timestamps: 03:00 and 15:00 on the same day. The
        # old implementation scored these differently. Nothing should move.
        import datetime as dt

        engine = ThreatEngine(self.RULES, night_start_hour=20, night_end_hour=6)
        night = dt.datetime(2026, 1, 15, 3, 0).timestamp()
        noon = dt.datetime(2026, 1, 15, 15, 0).timestamp()

        a = engine.assess(ThreatContext(incident_type="restricted_zone_intrusion",
                                        in_restricted_zone=True, ts=night))
        b = engine.assess(ThreatContext(incident_type="restricted_zone_intrusion",
                                        in_restricted_zone=True, ts=noon))
        assert a.score == b.score == 30

    def test_low_light_alone_raises_no_incident(self):
        # Context is a signal for scoring. Darkness on its own is not an
        # event, and must not become one.
        engine = ThreatEngine(self.RULES)
        result = engine.assess(ThreatContext(incident_type="none",
                                             low_light=True))
        assert result.score == 10
        assert result.level == "LOW"


class TestStateReporting:
    def test_to_dict_shape_is_stable(self):
        state = LowLightState(low_light=True, luminance=42.345, ready=True,
                              samples=15)
        assert state.to_dict() == {
            "low_light": True,
            "luminance": 42.3,
            "ready": True,
            "samples": 15,
        }

    def test_status_text_reports_the_number(self):
        state = LowLightState(low_light=True, luminance=31.0, ready=True,
                              samples=15)
        assert "LOW LIGHT" in state.status_text()
        assert "31" in state.status_text()
