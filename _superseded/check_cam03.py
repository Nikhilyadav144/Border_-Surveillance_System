"""Verify CAM-03: real footage, and the system correctly stays quiet.

The synthetic perimeter clip proves the detector FIRES when it should. This
one proves it does not fire when it should not, which is the harder half and
the one a false-alarm-prone system fails.

The footage shows a uniformed officer walking a patrol track alongside a
border wall. He never crosses to the wall side. A system that raises an
incident here is producing exactly the kind of false alarm that makes an
operator stop trusting it, so a correct run raises NOTHING while still
tracking him continuously and reading his body keypoints.

The clip is a Getty Images preview, watermark and asset id burned in. It is
here for evaluation only. See the note in config/cameras.yaml.

Usage, from the project root with the virtualenv active:

    python3 scripts/check_cam03.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_config              # noqa: E402
from core.pipeline import SurveillancePipeline   # noqa: E402
from inputs.video_source import FileVideoSource  # noqa: E402


def main() -> int:
    config = load_config(ROOT / "config")

    camera = next((c for c in config.cameras
                   if str(c.get("id")) == "CAM-03"), None)
    if camera is None:
        print("CAM-03 is not defined in config/cameras.yaml")
        return 1

    clip = config.path("input_dir") / str(camera["source"].get("path", ""))
    if not clip.exists():
        print(f"MISSING: {clip}")
        print("CAM-03's evaluation footage is not in data/input/.")
        return 1

    profile = str(camera.get("zones_profile", "none"))
    print(f"CAM-03 zones_profile: {profile}")
    print(f"clip: {clip.name}")
    if profile != "border_wall_diagonal":
        print("Expected 'border_wall_diagonal'. Fix config/cameras.yaml.")
        return 1

    pipeline = SurveillancePipeline(config, camera, database=None,
                                    annotate=False)
    source = FileVideoSource(clip, source_id="CAM-03", loop=False,
                             target_fps=None)
    source.open()
    info = source.info
    print(f"      {info.width}x{info.height} @ {info.fps:.0f} fps, "
          f"{info.frame_count} frames\n")

    frames = 0
    tracks: dict[int, str] = {}
    zones: dict[str, int] = {}
    incidents = []
    keypoints: list[int] = []
    epoch = None

    while True:
        frame = source.read()
        if frame is None:
            break
        frames += 1
        if epoch is None:
            epoch = frame.ts
        out = pipeline.process(frame.image, frame.index, frame.ts)
        for track in out.tracks:
            tracks[track.track_number] = track.class_name
            if track.zone_name:
                zones[track.zone_name] = zones.get(track.zone_name, 0) + 1
            if track.pose is not None:
                keypoints.append(track.pose.visible_count)
        for incident in out.incidents:
            incidents.append((frame.ts - epoch, incident))
    source.release()

    stats = pipeline.pose.stats
    print(f"frames processed : {frames}")
    print(f"tracks           : {sorted(tracks.items())}")
    print(f"zones occupied   : {zones or 'none'}")
    print(f"light            : {pipeline.low_light_state.status_text()}")
    print(f"pose             : {stats.poses_found} poses from "
          f"{stats.crops_run} crops, "
          f"{np.mean(keypoints):.1f} of 17 keypoints visible"
          if keypoints else "pose             : no keypoints")
    print(f"incidents        : {len(incidents)}")
    for offset, incident in incidents:
        print(f"   {offset:5.1f}s  {incident.threat_level:<8} "
              f"{incident.incident_type:<26} {incident.object_label:<11} "
              f"{incident.zone_name}")

    print()
    checks = [
        ("the officer was tracked", len(tracks) > 0),
        ("he stayed on the patrol side", "Patrol Track" in zones),
        ("he never entered the restricted wall side",
         "Restricted - Wall Side" not in zones),
        ("NO incident was raised on legitimate patrol activity",
         len(incidents) == 0),
        ("pose still read his body", stats.poses_found > 0),
    ]
    for label, passed in checks:
        print(f"   [{'PASS' if passed else 'FAIL'}] {label}")

    ok = all(passed for _, passed in checks)
    print("\nRESULT: " + ("CAM-03 OK - real footage, no false alarms"
                          if ok else "UNEXPECTED - see the failures above"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
