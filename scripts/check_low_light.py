"""Prove that low-light context comes from the video frame, not the clock.

The old implementation read the hour off the system clock. The only way to
show it has really been replaced is to hold the clock still and change the
picture, then hold the picture still and change the clock. The state must
follow the first and ignore the second.

This runs a real project video through the real pipeline twice: once as
filmed, once with the same frames darkened. Same file, same wall clock, same
timestamps. If the verdict differs between the two runs, it came from the
pixels.

Usage, from the project root with the virtualenv active:

    python3 scripts/check_low_light.py
    python3 scripts/check_low_light.py data/input/sample_02.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_config              # noqa: E402
from core.low_light import LowLightDetector      # noqa: E402
from core.pipeline import SurveillancePipeline   # noqa: E402
from core.threat_engine import ThreatContext, ThreatEngine  # noqa: E402
from inputs.video_source import FileVideoSource  # noqa: E402

DEFAULT_CLIP = ROOT / "data" / "input" / "demo_border.mp4"
MAX_FRAMES = 120


def run(config, clip: Path, darken: float | None) -> dict:
    """Drive `clip` through the pipeline, optionally dimming every frame."""
    camera = {"id": "CAM-01", "name": "check", "zones_profile": "default"}
    pipeline = SurveillancePipeline(config, camera, database=None,
                                    annotate=False)
    source = FileVideoSource(clip, source_id="CAM-01", loop=False,
                             target_fps=None)
    source.open()

    readings: list[float] = []
    verdicts: list[bool] = []
    frames = 0
    while frames < MAX_FRAMES:
        frame = source.read()
        if frame is None:
            break
        image = frame.image
        if darken is not None:
            # Scale brightness down. Same scene, same geometry, same
            # timestamps: only the light level changes.
            image = cv2.convertScaleAbs(image, alpha=darken, beta=0)
        frames += 1
        pipeline.process(image, frame.index, frame.ts)
        state = pipeline.low_light_state
        if state.ready:
            readings.append(state.luminance)
            verdicts.append(state.low_light)
    source.release()

    return {
        "frames": frames,
        "final": pipeline.low_light_state,
        "mean_luminance": float(np.mean(readings)) if readings else None,
        "low_light_fraction": (sum(verdicts) / len(verdicts)
                               if verdicts else None),
    }


def main() -> int:
    clip = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CLIP
    if not clip.is_absolute():
        clip = ROOT / clip
    if not clip.exists():
        print(f"MISSING: {clip}")
        return 1

    config = load_config(ROOT / "config")
    print(f"clip: {clip.name}")
    print(f"thresholds: enter below {config.get('low_light.enter_below')}, "
          f"exit above {config.get('low_light.exit_above')}, "
          f"window {config.get('low_light.window_frames')} frames\n")

    print("1. As filmed")
    lit = run(config, clip, darken=None)
    print(f"   {lit['final'].status_text()}")
    print(f"   mean luminance over the run: {lit['mean_luminance']:.1f}")

    print("\n2. Same clip, same clock, frames dimmed to 18%")
    dark = run(config, clip, darken=0.18)
    print(f"   {dark['final'].status_text()}")
    print(f"   mean luminance over the run: {dark['mean_luminance']:.1f}")

    print("\n3. The clock, held against a fixed picture")
    # Two timestamps twelve hours apart, identical context otherwise.
    import datetime as dt
    engine = ThreatEngine(config.threat_rules, night_start_hour=20,
                          night_end_hour=6)
    night = dt.datetime(2026, 1, 15, 3, 0).timestamp()
    noon = dt.datetime(2026, 1, 15, 15, 0).timestamp()
    base = dict(incident_type="restricted_zone_intrusion",
                in_restricted_zone=True, class_name="person")
    at_03 = engine.assess(ThreatContext(**base, ts=night)).score
    at_15 = engine.assess(ThreatContext(**base, ts=noon)).score
    dark_ctx = engine.assess(ThreatContext(**base, ts=noon,
                                           low_light=True)).score
    print(f"   same incident at 03:00 -> score {at_03:.0f}")
    print(f"   same incident at 15:00 -> score {at_15:.0f}")
    print(f"   same incident at 15:00 but measured dark -> score {dark_ctx:.0f}")

    print()
    checks = [
        ("bright run is not low light",
         lit["final"].ready and not lit["final"].low_light),
        ("dimmed run IS low light",
         dark["final"].ready and dark["final"].low_light),
        ("luminance dropped with the picture",
         dark["mean_luminance"] < lit["mean_luminance"]),
        ("clock alone changes nothing", at_03 == at_15),
        ("measured darkness does change the score", dark_ctx > at_15),
    ]
    for label, passed in checks:
        print(f"   [{'PASS' if passed else 'FAIL'}] {label}")

    ok = all(passed for _, passed in checks)
    print("\nRESULT: " + ("LOW-LIGHT CONTEXT IS FRAME-DRIVEN"
                          if ok else "UNEXPECTED - see the failures above"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
