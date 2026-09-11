"""Geometry, zone occupancy and line crossing."""

from __future__ import annotations

import pytest

from core.geometry import (
    angle_difference,
    compass_direction,
    heading_degrees,
    iou,
    line_side,
    point_in_polygon,
    polygon_area,
    polygon_centroid,
    segments_intersect,
)
from core.line_crossing import LineCrossingEngine
from core.zone_engine import ZoneEngine, ZoneKind
from tests.conftest import make_box

SQUARE = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
# An L shape, to prove the test is not secretly a bounding-box check.
L_SHAPE = [(0.0, 0.0), (0.6, 0.0), (0.6, 0.4), (0.3, 0.4),
           (0.3, 1.0), (0.0, 1.0)]


class TestPointInPolygon:
    def test_centre_is_inside(self):
        assert point_in_polygon(0.5, 0.5, SQUARE)

    @pytest.mark.parametrize("px,py", [
        (0.1, 0.5), (0.9, 0.5), (0.5, 0.1), (0.5, 0.9), (0.0, 0.0),
    ])
    def test_outside_points(self, px, py):
        assert not point_in_polygon(px, py, SQUARE)

    def test_concave_notch_is_outside(self):
        # Inside the L's bounding box, outside the L itself.
        assert not point_in_polygon(0.45, 0.7, L_SHAPE)

    def test_concave_arms_are_inside(self):
        assert point_in_polygon(0.15, 0.7, L_SHAPE)
        assert point_in_polygon(0.45, 0.2, L_SHAPE)

    def test_degenerate_polygon_contains_nothing(self):
        assert not point_in_polygon(0.5, 0.5, [(0.0, 0.0), (1.0, 1.0)])
        assert not point_in_polygon(0.5, 0.5, [])


class TestPolygonMaths:
    def test_area_of_unit_square(self):
        assert polygon_area([(0, 0), (1, 0), (1, 1), (0, 1)]) == pytest.approx(1.0)

    def test_centroid_of_square(self):
        cx, cy = polygon_centroid(SQUARE)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)

    def test_iou_identical_boxes(self):
        assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_iou_disjoint_boxes(self):
        assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_iou_half_overlap(self):
        assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


class TestHeading:
    @pytest.mark.parametrize("dx,dy,expected", [
        (0.0, -0.1, "north"),   # image y decreases upward
        (0.0, 0.1, "south"),
        (0.1, 0.0, "east"),
        (-0.1, 0.0, "west"),
    ])
    def test_compass(self, dx, dy, expected):
        bearing = heading_degrees((0.5, 0.5), (0.5 + dx, 0.5 + dy))
        assert compass_direction(bearing) == expected

    def test_no_movement_is_zero(self):
        assert heading_degrees((0.5, 0.5), (0.5, 0.5)) == 0.0

    def test_angle_difference_wraps(self):
        assert angle_difference(350, 10) == pytest.approx(20)
        assert angle_difference(10, 350) == pytest.approx(20)


class TestSegments:
    def test_crossing_segments(self):
        assert segments_intersect((0, 0), (1, 1), (0, 1), (1, 0))

    def test_parallel_segments(self):
        assert not segments_intersect((0, 0), (1, 0), (0, 1), (1, 1))

    def test_touching_endpoint_counts(self):
        assert segments_intersect((0, 0), (1, 1), (1, 1), (2, 0))

    def test_line_side_signs_are_opposite(self):
        above = line_side(0.5, 0.2, (0.0, 0.5), (1.0, 0.5))
        below = line_side(0.5, 0.8, (0.0, 0.5), (1.0, 0.5))
        assert above * below < 0


class TestZoneEngine:
    def test_reference_point_for_person_is_feet(self, zone_dicts):
        engine = ZoneEngine(zone_dicts, vehicle_classes={"car"})
        point = engine.reference_point((0.4, 0.2, 0.6, 0.8), "person")
        assert point == (pytest.approx(0.5), pytest.approx(0.8))

    def test_reference_point_for_vehicle_is_centre(self, zone_dicts):
        engine = ZoneEngine(zone_dicts, vehicle_classes={"car"})
        point = engine.reference_point((0.4, 0.2, 0.6, 0.8), "car")
        assert point == (pytest.approx(0.5), pytest.approx(0.5))

    def test_entering_a_zone_raises_one_enter_event(self, zone_dicts):
        engine = ZoneEngine(zone_dicts, debounce_frames=1)
        events = engine.update(1, "person", make_box(0.5, 0.42), ts=10.0)
        assert len(events) == 1
        assert events[0].kind == "enter"
        assert events[0].zone.name == "Restricted Compound"
        assert events[0].is_alarming

    def test_staying_put_raises_nothing_further(self, zone_dicts):
        engine = ZoneEngine(zone_dicts, debounce_frames=1)
        engine.update(1, "person", make_box(0.5, 0.42), ts=10.0)
        for step in range(1, 20):
            assert engine.update(1, "person", make_box(0.5, 0.42),
                                 ts=10.0 + step) == []

    def test_leaving_raises_exit_with_dwell(self, zone_dicts):
        engine = ZoneEngine(zone_dicts, debounce_frames=1)
        engine.update(1, "person", make_box(0.5, 0.42), ts=10.0)
        events = engine.update(1, "person", make_box(0.05, 0.42), ts=25.0)
        exits = [e for e in events if e.kind == "exit"]
        assert len(exits) == 1
        assert exits[0].dwell_seconds == pytest.approx(15.0)

    def test_debounce_suppresses_single_frame_flicker(self, zone_dicts):
        engine = ZoneEngine(zone_dicts, debounce_frames=3)
        # One frame just inside is not enough to be believed.
        assert engine.update(1, "person", make_box(0.5, 0.42), ts=1.0) == []
        assert engine.update(1, "person", make_box(0.05, 0.42), ts=2.0) == []
        assert engine.current_zone(1) is None

    def test_debounce_accepts_a_persistent_change(self, zone_dicts):
        engine = ZoneEngine(zone_dicts, debounce_frames=3)
        events = []
        for step in range(4):
            events += engine.update(1, "person", make_box(0.5, 0.42),
                                    ts=float(step))
        assert any(e.kind == "enter" for e in events)
        assert engine.current_zone(1) == "Restricted Compound"

    def test_vehicle_zone_violation_only_applies_to_vehicles(self, zone_dicts):
        engine = ZoneEngine(zone_dicts, vehicle_classes={"car"},
                            debounce_frames=1)
        no_vehicle = next(z for z in engine.zones if z.kind == ZoneKind.NO_VEHICLE)
        assert engine.is_violation(no_vehicle, "car")
        assert not engine.is_violation(no_vehicle, "person")

    def test_zones_are_tested_in_configuration_order(self, zone_dicts):
        engine = ZoneEngine(zone_dicts, debounce_frames=1)
        zone = engine.zone_at(0.5, 0.1)
        assert zone is not None and zone.name == "Border Strip"

    def test_tracks_are_independent(self, zone_dicts):
        engine = ZoneEngine(zone_dicts, debounce_frames=1)
        engine.update(1, "person", make_box(0.5, 0.42), ts=1.0)
        engine.update(2, "person", make_box(0.9, 0.9), ts=1.0)
        assert engine.current_zone(1) == "Restricted Compound"
        assert engine.current_zone(2) is None

    def test_forget_clears_state(self, zone_dicts):
        engine = ZoneEngine(zone_dicts, debounce_frames=1)
        engine.update(1, "person", make_box(0.5, 0.42), ts=1.0)
        engine.forget(1)
        assert engine.current_zone(1) is None

    def test_no_zones_configured_is_silent(self):
        engine = ZoneEngine([], debounce_frames=1)
        assert engine.update(1, "person", make_box(0.5, 0.5), ts=1.0) == []

    def test_unknown_kind_falls_back_to_observation(self):
        engine = ZoneEngine([{
            "name": "Odd", "kind": "TELEPORTER",
            "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
        }])
        assert engine.zones[0].kind == ZoneKind.OBSERVATION


class TestLineCrossing:
    def test_crossing_upward_is_detected(self, line_dicts):
        engine = LineCrossingEngine(line_dicts)
        assert engine.update(1, "person", (0.5, 0.6), ts=1.0) == []
        events = engine.update(1, "person", (0.5, 0.4), ts=2.0)
        assert len(events) == 1
        assert events[0].compass == "north"
        assert events[0].direction == "entering"

    def test_crossing_downward_is_exiting(self, line_dicts):
        engine = LineCrossingEngine(line_dicts)
        engine.update(1, "person", (0.5, 0.4), ts=1.0)
        events = engine.update(1, "person", (0.5, 0.6), ts=2.0)
        assert events[0].compass == "south"
        assert events[0].direction == "exiting"

    def test_moving_parallel_never_crosses(self, line_dicts):
        engine = LineCrossingEngine(line_dicts)
        engine.update(1, "person", (0.2, 0.6), ts=1.0)
        for step, x in enumerate((0.4, 0.6, 0.8), start=2):
            assert engine.update(1, "person", (x, 0.6), ts=float(step)) == []

    def test_cooldown_stops_repeat_events(self, line_dicts):
        engine = LineCrossingEngine(line_dicts, cooldown_seconds=10.0)
        engine.update(1, "person", (0.5, 0.6), ts=1.0)
        assert engine.update(1, "person", (0.5, 0.4), ts=2.0)
        assert engine.update(1, "person", (0.5, 0.6), ts=3.0) == []

    def test_jitter_below_min_travel_is_ignored(self, line_dicts):
        engine = LineCrossingEngine(line_dicts, min_travel=0.05)
        engine.update(1, "person", (0.5, 0.501), ts=1.0)
        assert engine.update(1, "person", (0.5, 0.499), ts=2.0) == []

    def test_first_observation_cannot_cross(self, line_dicts):
        engine = LineCrossingEngine(line_dicts)
        assert engine.update(1, "person", (0.5, 0.1), ts=1.0) == []

    def test_crossing_count_is_tracked(self, line_dicts):
        engine = LineCrossingEngine(line_dicts, cooldown_seconds=0.0)
        engine.update(1, "person", (0.5, 0.6), ts=1.0)
        engine.update(1, "person", (0.5, 0.4), ts=2.0)
        engine.update(1, "person", (0.5, 0.6), ts=3.0)
        assert engine.crossing_count(1) == 2

    def test_no_lines_configured_is_silent(self):
        engine = LineCrossingEngine([])
        engine.update(1, "person", (0.5, 0.6), ts=1.0)
        assert engine.update(1, "person", (0.5, 0.4), ts=2.0) == []

    def test_a_brand_new_track_cannot_cross(self, line_dicts):
        """Regression from real footage.

        A person emerging from behind a railing had his box grow downwards
        by 0.13 of frame height in six frames while he stood still. The
        reference point is the bottom of the box, so it swept across the
        boundary and registered two crossings before he had taken a step -
        and those then swallowed his real vault thirty frames later under
        the cooldown.
        """
        engine = LineCrossingEngine(line_dicts, warmup_seconds=0.5)
        engine.update(1, "person", (0.5, 0.6), ts=0.00)
        assert engine.update(1, "person", (0.5, 0.4), ts=0.20) == []
        # Back to the near side, still inside the window, still silent - and
        # crucially without arming the cooldown, which is what made the real
        # vault invisible.
        assert engine.update(1, "person", (0.5, 0.6), ts=0.30) == []
        # The same movement once the box has had time to settle.
        assert len(engine.update(1, "person", (0.5, 0.4), ts=0.60)) == 1

    def test_warmup_can_be_switched_off(self, line_dicts):
        engine = LineCrossingEngine(line_dicts, warmup_seconds=0.0)
        engine.update(1, "person", (0.5, 0.6), ts=0.00)
        assert len(engine.update(1, "person", (0.5, 0.4), ts=0.05)) == 1

    def test_a_reference_only_line_never_reports_a_crossing(self):
        """`alerting: false` is geometry the analyzer reads and the crossing
        engine ignores.

        A camera needs to know where the OTHER fences are to be able to say
        which one a person is engaged with. But a background fence cannot be
        used as a crossing line from a low viewpoint: a person well in front
        of it sweeps their reference point across it without going near it,
        and a 2-D line test has no depth with which to tell those apart.
        """
        lines = [
            {"name": "Security Fence", "start": [0.0, 0.5], "end": [1.0, 0.5]},
            {"name": "Background Fence", "start": [0.0, 0.3],
             "end": [1.0, 0.3], "alerting": False},
        ]
        engine = LineCrossingEngine(lines, warmup_seconds=0.0)
        # Both lines are configured and visible to anything that asks.
        assert [ln.name for ln in engine.lines] == ["Security Fence",
                                                    "Background Fence"]
        engine.update(1, "person", (0.5, 0.9), ts=0.0)
        # One movement that crosses BOTH lines.
        events = engine.update(1, "person", (0.5, 0.1), ts=1.0)
        assert [e.line.name for e in events] == ["Security Fence"]

    def test_alerting_defaults_to_true(self):
        engine = LineCrossingEngine([
            {"name": "Fence", "start": [0.0, 0.5], "end": [1.0, 0.5]},
        ])
        assert engine.lines[0].alerting is True
        assert engine.lines[0].to_dict()["alerting"] is True


class TestPersonIntelOverlay:
    """The stack of behaviour, boundary and threat drawn under the id.

    Regression: for a track's first second and a half the pipeline has not
    concluded anything - behaviour is UNKNOWN, boundary NONE, threat None -
    and the overlay drew the identity and nothing else. Measured on CAM-01
    that was 52 of 246 person frames, 21%, and it read as a broken overlay
    rather than as a system still working it out.
    """

    @staticmethod
    def _stub(**values):
        import types
        base = dict(behaviour="UNKNOWN", behaviour_confidence=0.0,
                    movement_direction=None, boundary_state="NONE",
                    boundary_line=None, threat_level=None,
                    pose_status="TRACKED")
        base.update(values)
        return types.SimpleNamespace(**base)

    def _draw(self, track):
        import numpy as np
        from core.annotator import Annotator
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        Annotator()._intel_stack(image, track, 20, 20, 300, 380,
                                 (120, 220, 90), 280)
        return image

    def test_a_track_with_nothing_known_still_says_something(self):
        image = self._draw(self._stub())
        assert image.any(), "a person with no conclusions drew nothing at all"

    def test_a_track_with_no_pose_says_so_rather_than_analysing(self):
        # Two different honest statements: the body could not be measured,
        # versus it is being measured and nothing is decided yet.
        no_pose = self._draw(self._stub(pose_status="UNAVAILABLE"))
        analysing = self._draw(self._stub(pose_status="TRACKED"))
        assert no_pose.any() and analysing.any()
        assert not (no_pose == analysing).all(), "both states drew the same text"

    def test_known_behaviour_is_drawn(self):
        known = self._draw(self._stub(behaviour="BENT_RUN",
                                      behaviour_confidence=0.92,
                                      movement_direction="WEST"))
        unknown = self._draw(self._stub())
        assert not (known == unknown).all()

    def test_a_crossing_and_a_threat_both_appear(self):
        plain = self._draw(self._stub(behaviour="WALKING",
                                      behaviour_confidence=0.8))
        loud = self._draw(self._stub(behaviour="WALKING",
                                     behaviour_confidence=0.8,
                                     boundary_state="CROSSED",
                                     boundary_line="Perimeter Railing",
                                     threat_level="CRITICAL"))
        assert loud.sum() > plain.sum(), "crossing and threat added no lines"
