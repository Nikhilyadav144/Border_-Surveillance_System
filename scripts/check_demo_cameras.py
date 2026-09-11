"""Run every enabled demo camera against its real footage and report.

This is not a pass/fail test and it does not pretend to be one. Real footage
has no answer key: nobody can assert in advance how many incidents a clip of
people doing ordinary things should produce. What this does is show, per
camera, what the pipeline actually saw, so you can look at it and judge.

The one thing it does assert is the architectural invariant: geometry differs
per camera and AI capability does not.

For a real pass/fail check, use scripts/check_fence_fixture.py, which runs
scripted footage with a known answer.

Usage, from the project root with the virtualenv active:

    python3 scripts/check_demo_cameras.py
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


def run(config, camera: dict) -> dict:
    clip = config.path("input_dir") / str(camera["source"].get("path", ""))
    if not clip.exists():
        return {"error": f"missing footage: {clip.name}"}

    pipeline = SurveillancePipeline(config, camera, database=None,
                                    annotate=False)
    source = FileVideoSource(clip, source_id=str(camera["id"]), loop=False,
                             target_fps=None)
    source.open()

    frames = 0
    tracks: dict[int, str] = {}
    zones: dict[str, int] = {}
    incidents: list = []
    keypoints: list[int] = []
    behaviours: dict[str, int] = {}
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
            if track.behaviour and track.behaviour != "UNKNOWN":
                behaviours[track.behaviour] = behaviours.get(
                    track.behaviour, 0) + 1
        for incident in out.incidents:
            incidents.append((frame.ts - epoch, incident))
    source.release()

    by_class: dict[str, int] = {}
    for name in tracks.values():
        by_class[name] = by_class.get(name, 0) + 1

    return {
        "clip": clip.name,
        "frames": frames,
        "geometry": pipeline.geometry_state,
        "tracks": tracks,
        "by_class": by_class,
        "zones": zones,
        "incidents": incidents,
        "light": pipeline.low_light_state.status_text(),
        "behaviours": behaviours,
        "pose": pipeline.pose.stats,
        "keypoints": float(np.mean(keypoints)) if keypoints else 0.0,
    }


def main() -> int:
    config = load_config(ROOT / "config")
    enabled = [c for c in config.cameras if c.get("enabled")]
    if not enabled:
        print("No cameras are enabled in config/cameras.yaml")
        return 1

    results = []
    for camera in enabled:
        source = camera.get("source", {})
        if source.get("type") != "video":
            print(f"\n### {camera['id']}  {camera.get('name')}")
            print(f"    skipped: source type is '{source.get('type')}', "
                  f"not a file")
            continue

        result = run(config, camera)
        results.append((camera, result))
        print(f"\n### {camera['id']}  {camera.get('name')}")
        if "error" in result:
            print(f"    {result['error']}")
            continue

        geometry = result["geometry"]
        print(f"    footage    : {result['clip']}  ({result['frames']} frames)")
        print(f"    geometry   : {geometry['summary']}")
        print(f"    light      : {result['light']}")
        print(f"    tracked    : {result['by_class'] or 'nothing'}")
        if result["zones"]:
            print(f"    zones used : {result['zones']}")
        if result["behaviours"]:
            top = sorted(result["behaviours"].items(), key=lambda kv: -kv[1])
            print(f"    behaviour  : "
                  + ", ".join(f"{n} ({c} frames)" for n, c in top))
        stats = result["pose"]
        if stats.crops_run:
            print(f"    pose       : {stats.poses_found} poses from "
                  f"{stats.crops_run} crops, "
                  f"{result['keypoints']:.1f} of 17 keypoints")
        else:
            print("    pose       : no eligible people")
        print(f"    incidents  : {len(result['incidents'])}")
        for offset, incident in result["incidents"]:
            print(f"       {offset:6.1f}s  {incident.threat_level:<8} "
                  f"{incident.incident_type:<26} "
                  f"{incident.object_label:<12} "
                  f"{incident.zone_name or '-'}")

    print("\n=== architectural invariant ===")
    ok = []
    geometries = {c["id"]: r["geometry"]["zones"]
                  for c, r in results if "error" not in r}
    detected = {c["id"]: bool(r["tracks"])
                for c, r in results if "error" not in r}
    print(f"  zones per camera   : {geometries}")
    print(f"  detection ran      : {detected}")
    ok.append(("every camera declares its own geometry",
               all(c.get("zones_profile") is not None for c, _ in results)))
    ok.append(("detection ran on every camera regardless of geometry",
               all(detected.values())))
    ok.append(("no camera raised an incident from a zone it does not have",
               all(not r["zones"] or r["geometry"]["zones"] > 0
                   for _, r in results if "error" not in r)))
    print()
    for label, passed in ok:
        print(f"   [{'PASS' if passed else 'FAIL'}] {label}")

    print("\nThis is a report, not a verdict. Real footage has no answer key.")
    print("For pass/fail, run: python3 scripts/check_fence_fixture.py")
    return 0 if all(p for _, p in ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
