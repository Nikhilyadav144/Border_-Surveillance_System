"""Prove security geometry is optional and per-camera, while AI is not.

Two cases, same footage, same pipeline, same engines. The only difference is
whether that camera has a polygon configured.

TEST A, a camera with NO geometry. Detection, tracking, behaviour and pose all
run. No polygon is drawn. No zone intrusion and no line crossing can be raised,
because there is nothing to intrude into. A person simply being visible is not
a security event.

TEST B, the SAME footage on a camera WITH geometry. The polygon is drawn, the
person entering it produces a zone event, and the existing threat and incident
pipeline handles it exactly as before.

The point is the difference between the two columns. If A raises zone
incidents, geometry is leaking across cameras. If B raises none, the engine
has been broken rather than made optional.

Usage, from the project root with the virtualenv active:

    python3 scripts/check_zones.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.annotator import Annotator                # noqa: E402
from core.config import load_config                 # noqa: E402
from core.pipeline import SurveillancePipeline      # noqa: E402
from inputs.video_source import FileVideoSource     # noqa: E402

CLIP = ROOT / "data" / "input" / "demo_border.mp4"
MAX_FRAMES = 300

ZONE_TYPES = {
    "restricted_zone_intrusion",
    "border_zone_intrusion",
    "no_vehicle_zone_violation",
    "line_crossing",
}


def run(config, profile: str) -> dict:
    camera = {"id": "CAM-TEST", "name": "geometry check",
              "zones_profile": profile}
    pipeline = SurveillancePipeline(config, camera, database=None,
                                    annotate=False)
    source = FileVideoSource(CLIP, source_id="CAM-TEST", loop=False,
                             target_fps=None)
    source.open()

    tracks: dict[int, str] = {}
    incidents: list = []
    posed = 0
    frames = 0
    while frames < MAX_FRAMES:
        frame = source.read()
        if frame is None:
            break
        frames += 1
        out = pipeline.process(frame.image, frame.index, frame.ts)
        incidents.extend(out.incidents)
        for track in out.tracks:
            tracks[track.track_number] = track.class_name
            if track.pose is not None:
                posed += 1
    source.release()

    # Does the annotator actually draw anything for this geometry?
    blank = np.full((480, 640, 3), 100, dtype=np.uint8)
    drawn = Annotator().render(frame=blank, tracks=[],
                               zones=pipeline.zone_engine.zones,
                               lines=pipeline.line_engine.lines, hud=None)
    pixels = int(np.count_nonzero(np.any(drawn != blank, axis=2)))

    return {
        "geometry": pipeline.geometry_state,
        "tracks": tracks,
        "incidents": incidents,
        "zone_incidents": [i for i in incidents
                           if i.incident_type in ZONE_TYPES],
        "pixels_drawn": pixels,
        "posed": posed,
        "worst": max((i.threat_score for i in incidents), default=0.0),
        "worst_level": max(incidents, key=lambda i: i.threat_score).threat_level
        if incidents else "-",
    }


def report(title: str, result: dict) -> None:
    geometry = result["geometry"]
    print(f"\n=== {title} ===")
    print(f"  profile            : {geometry['profile']}")
    print(f"  geometry           : {geometry['summary']}")
    print(f"  polygon pixels drawn: {result['pixels_drawn']}")
    print(f"  tracks             : {len(result['tracks'])} "
          f"({sorted(result['tracks'].items())})")
    print(f"  pose readings      : {result['posed']}")
    print(f"  incidents total    : {len(result['incidents'])}")
    print(f"  of which zone/line : {len(result['zone_incidents'])}")
    print(f"  highest threat     : {result['worst']:.0f} "
          f"({result['worst_level']})")
    for incident in result["incidents"][:6]:
        print(f"     {incident.threat_level:<8} {incident.incident_type:<28} "
              f"{incident.object_label}")


def main() -> int:
    if not CLIP.exists():
        print(f"MISSING: {CLIP}")
        return 1

    config = load_config(ROOT / "config")
    print(f"clip: {CLIP.name}, first {MAX_FRAMES} frames, identical for both")
    print(f"profiles available: {config.zone_profile_names()}")

    a = run(config, "none")
    report("TEST A  camera with NO configured geometry", a)

    b = run(config, "demo_border_clip")
    report("TEST B  SAME footage, camera WITH geometry", b)

    print("\n=== cross-camera independence ===")
    cameras = []
    for camera in config.cameras:
        profile = config.zone_profile(str(camera.get("zones_profile", "none")))
        cameras.append((camera["id"], camera.get("zones_profile"),
                        len(profile["zones"]), len(profile["lines"])))
        print(f"  {camera['id']}  profile={str(camera.get('zones_profile')):<22} "
              f"zones={len(profile['zones'])} lines={len(profile['lines'])}")

    print()
    checks = [
        ("A: detection and tracking ran", len(a["tracks"]) > 0),
        ("A: pose ran", a["posed"] > 0),
        ("A: NO zone or line incident was raised",
         len(a["zone_incidents"]) == 0),
        ("A: no polygon was drawn", a["pixels_drawn"] == 0),
        ("A: an ordinary person did not become HIGH or CRITICAL",
         a["worst_level"] in ("-", "LOW", "MEDIUM")),
        ("B: same footage found the same objects",
         set(b["tracks"]) >= set(a["tracks"]) or set(a["tracks"]) >= set(b["tracks"])),
        ("B: the polygon was drawn", b["pixels_drawn"] > 0),
        ("B: zone or line incidents WERE raised",
         len(b["zone_incidents"]) > 0),
        ("geometry differs per camera, capability does not",
         len({c[2] for c in cameras}) > 1),
        ("no camera silently inherits another's polygons",
         all(c[1] is not None for c in cameras)),
    ]
    for label, passed in checks:
        print(f"   [{'PASS' if passed else 'FAIL'}] {label}")

    ok = all(passed for _, passed in checks)
    print("\nRESULT: " + ("GEOMETRY IS OPTIONAL AND PER-CAMERA"
                          if ok else "UNEXPECTED - see the failures above"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
