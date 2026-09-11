"""Security geometry is optional and per-camera. AI capability is not.

These guard one invariant: a camera that has not been given a polygon must
never behave as though it had one. The regression they exist to prevent is
real and shipped once - the pipeline's fallback profile was named "default",
so every unconfigured camera silently inherited polygons drawn for one demo
clip, and an ordinary person standing mid-frame scored a HIGH intrusion on a
camera nobody had configured.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.annotator import Annotator
from core.config import load_config
from core.line_crossing import LineCrossingEngine
from core.zone_engine import ZoneEngine


@pytest.fixture(scope="module")
def cfg():
    from core.paths import PROJECT_ROOT
    return load_config(PROJECT_ROOT / "config")


class TestShippedConfiguration:
    def test_there_is_no_profile_called_default(self, cfg):
        # The name is the bug. A profile called "default" invites exactly the
        # inheritance this whole module exists to prevent.
        assert "default" not in cfg.zone_profile_names()

    def test_the_empty_profile_exists_and_is_empty(self, cfg):
        profile = cfg.zone_profile("none")
        assert profile == {"zones": [], "lines": []}

    def test_every_shipped_camera_declares_its_geometry(self, cfg):
        for camera in cfg.cameras:
            assert "zones_profile" in camera, (
                f"{camera['id']} does not declare zones_profile, so it would "
                f"fall back rather than state its intent"
            )

    def test_cameras_do_not_share_geometry(self, cfg):
        counts = {}
        for camera in cfg.cameras:
            profile = cfg.zone_profile(str(camera["zones_profile"]))
            counts[camera["id"]] = (len(profile["zones"]),
                                    len(profile["lines"]))
        # At least one camera with geometry and one without, which is the
        # whole point: geometry is a property of the camera, not the system.
        assert any(z == 0 and ln == 0 for z, ln in counts.values())
        assert any(z > 0 for z, _ in counts.values())

    def test_the_demo_geometry_still_exists_under_its_own_name(self, cfg):
        # Renaming it must not have deleted the ability to configure geometry.
        profile = cfg.zone_profile("demo_border_clip")
        assert profile["zones"] and profile["lines"]


class TestEnginesWithNoGeometry:
    """The same engines, given nothing, must produce nothing."""

    def test_zone_engine_with_no_zones_reports_no_zone(self):
        engine = ZoneEngine(zones=[], vehicle_classes={"car"})
        assert engine.zones == []
        # A box anywhere in the frame, over several frames, produces nothing.
        for step, box in enumerate(((0.4, 0.4, 0.6, 0.6),
                                    (0.0, 0.0, 0.2, 0.3),
                                    (0.8, 0.7, 0.99, 0.99),
                                    (0.2, 0.6, 0.4, 0.9))):
            assert engine.update(track_number=1, class_name="person",
                                 box_norm=box, ts=float(step)) == []

    def test_line_engine_with_no_lines_reports_no_crossing(self):
        engine = LineCrossingEngine(lines=[])
        assert engine.lines == []
        # Walk a track right across the frame. Nothing to cross.
        for step, y in enumerate((0.9, 0.7, 0.5, 0.3, 0.1)):
            assert engine.update(track_number=1, class_name="person",
                                 point=(0.5, y), ts=float(step)) == []

    def test_annotator_draws_nothing_for_empty_geometry(self):
        blank = np.full((240, 320, 3), 100, dtype=np.uint8)
        drawn = Annotator().render(frame=blank, tracks=[], zones=[], lines=[],
                                   hud=None)
        assert np.array_equal(drawn, blank), (
            "an empty profile must draw nothing at all, not draw something "
            "faint or hide it in the interface"
        )


class TestPipelineDefault:
    def test_a_camera_without_zones_profile_gets_no_geometry(self, cfg):
        # Read the fallback the way the pipeline does, without building a
        # pipeline (which would load the models).
        camera: dict = {"id": "CAM-X"}
        profile_name = str(camera.get("zones_profile", "none"))
        assert profile_name == "none"
        profile = cfg.zone_profile(profile_name)
        assert profile["zones"] == []
        assert profile["lines"] == []

    def test_the_fallback_in_the_source_is_none_not_default(self):
        # A string check, because this single word is the entire defect.
        from core.paths import PROJECT_ROOT
        source = (PROJECT_ROOT / "core" / "pipeline.py").read_text()
        assert 'camera.get("zones_profile", "none")' in source
        assert 'camera.get("zones_profile", "default")' not in source

    def test_gui_fallbacks_match_the_pipeline(self):
        from core.paths import PROJECT_ROOT
        root = PROJECT_ROOT
        for relative in ("app/gui/main_window.py", "app/gui/pages/cameras.py"):
            source = (root / relative).read_text()
            assert 'zones_profile", "default"' not in source, (
                f"{relative} still falls back to the demo geometry"
            )
