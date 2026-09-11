"""Verify CAM-02: run the perimeter clip through the real pipeline, headless.

This is a check, not a demo. It opens no window, writes to no database and
starts no server. It drives ``data/input/sample_02.mp4`` through the same
SurveillancePipeline the GUI and the API use, with CAM-02's ``perimeter_fence``
zone profile, and prints every incident the engines raise.

What a correct run looks like:

  * exactly TWO person tracks for the whole clip - the intruder and the
    patrol guard
  * exactly THREE incidents, all raised against the SAME track:
      line crossing        around t = 8 s
      restricted intrusion around t = 8.5 s
      loitering            around t = 16.5 s
  * nothing at all raised against the other track - the guard walks the
    patrol lane and never crosses the fence, so a run that reports incidents
    for both tracks has a false positive, not a working detector

Scores vary legitimately between runs. Threat scoring is additive and two of
its factors depend on context rather than on the footage: ``night_time`` adds
10 when the clock is inside the configured night window, and ``group_presence``
adds 10 when three or more people are tracked at once. A daytime run with two
people scores 20 / 30 / 50; the same clip at 21:00 scores 30 / 40 / 60.

Usage, from the project root with the virtualenv active:

    python3 scripts/check_cam02.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_config              # noqa: E402
from core.pipeline import SurveillancePipeline   # noqa: E402
from inputs.video_source import FileVideoSource  # noqa: E402

CLIP = ROOT / "data" / "input" / "sample_02.mp4"


def main() -> int:
    if not CLIP.exists():
        print(f"MISSING: {CLIP}")
        print("Build it with:  python3 scripts/make_perimeter_video.py")
        return 1

    config = load_config(ROOT / "config")

    # Read CAM-02's own entry rather than assuming its profile, so this check
    # fails loudly if cameras.yaml was not switched over.
    camera = next((c for c in config.cameras
                   if str(c.get("id")) == "CAM-02"), None)
    if camera is None:
        print("CAM-02 is not defined in config/cameras.yaml")
        return 1
    profile = str(camera.get("zones_profile", "none"))
    print(f"CAM-02 zones_profile: {profile}")
    if profile != "perimeter_fence":
        print("Expected 'perimeter_fence'. Fix config/cameras.yaml first.")
        return 1

    pipeline = SurveillancePipeline(config, camera, database=None,
                                    annotate=False)
    source = FileVideoSource(CLIP, source_id="CAM-02", loop=False,
                             target_fps=None)
    source.open()
    info = source.info
    print(f"clip: {info.width}x{info.height} @ {info.fps:.0f} fps, "
          f"{info.frame_count} frames")

    epoch: float | None = None
    frames = 0
    seen: dict[int, str] = {}
    incidents: list[tuple[float, object]] = []

    while True:
        frame = source.read()
        if frame is None:
            break
        frames += 1
        if epoch is None:
            epoch = frame.ts
        out = pipeline.process(frame.image, frame.index, frame.ts)
        for track in out.tracks:
            seen.setdefault(track.track_number, track.class_name)
        for incident in out.incidents:
            incidents.append((frame.ts - epoch, incident))
    source.release()

    print(f"\nframes processed: {frames}")
    print(f"tracks published: "
          f"{sorted((n, c) for n, c in seen.items())}")

    print(f"\n{'#':>3} {'t(s)':>6}  {'LEVEL':<9} {'TYPE':<28} "
          f"{'OBJECT':<12} {'ZONE / LINE':<22} SCORE")
    for index, (offset, incident) in enumerate(incidents, 1):
        print(f"{index:>3} {offset:>6.1f}  {incident.threat_level:<9} "
              f"{incident.incident_type:<28} {incident.object_label:<12} "
              f"{str(incident.zone_name or '-'):<22} "
              f"{incident.threat_score:.0f}")

    culprits = {i.object_label for _, i in incidents}
    print(f"\nincidents: {len(incidents)}   raised against: "
          f"{sorted(culprits) or ['nothing']}")

    ok = (len(seen) == 2 and len(incidents) == 3 and len(culprits) == 1)
    print("\nRESULT: " + ("CAM-02 PERIMETER FENCE OK" if ok else
                          "UNEXPECTED - see the notes at the top of this file"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
