"""Regression fixture: the scripted perimeter clip, with a known answer.

This is a check, not a demo, and it is deliberately NOT tied to any camera in
cameras.yaml. The demo cameras now run real footage, and real footage has no
correct answer anyone can assert in advance - you can see whether it looks
right, but you cannot fail a build on it.

``data/input/sample_02.mp4`` is synthetic and scripted, which is exactly why
it is kept. Two actors follow paths written in
``scripts/make_perimeter_video.py``, so the events the engines should produce
are known before the run starts. That makes this the one place a change to
zones, lines, tracking or loitering can be caught automatically.

It builds its own camera definition rather than reading one, so reconfiguring
the demo cameras can never silently disable it.

The anomaly engine is switched OFF for this run, deliberately. It is a
learned model: its output depends on how many completed tracks it has seen
across every previous run, so the same clip can score differently on Tuesday
than it did on Monday. That is correct behaviour for an anomaly detector and
useless behaviour for a regression fixture. What is being asserted here is the
geometry and behaviour chain - zones, lines, dwell - which is deterministic.

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

    python3 scripts/check_fence_fixture.py
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

    # Self-contained. The geometry this fixture needs is named here, not
    # looked up from whichever camera happens to be configured today.
    camera = {
        "id": "FIXTURE",
        "name": "Perimeter fence regression fixture",
        "zones_profile": "perimeter_fence",
    }
    profile = config.zone_profile(camera["zones_profile"])
    if not profile["zones"] or not profile["lines"]:
        print("The 'perimeter_fence' profile is missing from config/zones.yaml")
        return 1
    print(f"profile: perimeter_fence "
          f"({len(profile['zones'])} zones, {len(profile['lines'])} lines)")

    pipeline = SurveillancePipeline(config, camera, database=None,
                                    annotate=False)
    # See the note at the top: a learned model cannot be part of a fixture
    # whose whole value is giving the same answer every time.
    pipeline.anomaly.enabled = False
    source = FileVideoSource(CLIP, source_id="FIXTURE", loop=False,
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
    print("\nRESULT: " + ("PERIMETER FIXTURE OK" if ok else
                          "UNEXPECTED - see the notes at the top of this file"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
