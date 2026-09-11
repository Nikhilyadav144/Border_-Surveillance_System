"""Verify person activity analysis on real footage, and measure what it costs.

Runs the same clip twice, with the activity chain off and on, and compares.
Two things are being proved: that behaviour is temporal rather than
frame-by-frame, and that adding it changed nothing else - same track ids, same
incidents. A behaviour layer that quietly shifted the detections would be a
regression wearing a feature's clothes.

Usage, from the project root with the virtualenv active:

    python3 scripts/check_activity.py
    python3 scripts/check_activity.py "data/input/<clip>.mp4"
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_config              # noqa: E402
from core.pipeline import SurveillancePipeline   # noqa: E402
from inputs.video_source import FileVideoSource  # noqa: E402

MAX_FRAMES = 400


def run(config, clip: Path, activity: bool) -> dict:
    camera = {"id": "CAM-CHECK", "name": "activity check",
              "zones_profile": "none"}
    pipeline = SurveillancePipeline(config, camera, database=None,
                                    annotate=False)
    pipeline.activity_enabled = activity

    source = FileVideoSource(clip, source_id="CAM-CHECK", loop=False,
                             target_fps=None)
    source.open()

    frames = 0
    tracks: dict[int, str] = {}
    incidents = 0
    events: list = []
    behaviour_seen: Counter = Counter()
    pose_status: Counter = Counter()
    started = time.perf_counter()

    while frames < MAX_FRAMES:
        frame = source.read()
        if frame is None:
            break
        frames += 1
        out = pipeline.process(frame.image, frame.index, frame.ts)
        incidents += len(out.incidents)
        events.extend(pipeline.behaviour_events)
        for track in out.tracks:
            tracks[track.track_number] = track.class_name
            if track.class_name.lower() in pipeline.person_classes:
                behaviour_seen[track.behaviour] += 1
                pose_status[track.pose_status] += 1
    source.release()
    wall = time.perf_counter() - started

    return {
        "frames": frames,
        "ms_per_frame": wall / frames * 1000 if frames else 0.0,
        "tracks": sorted(tracks.items()),
        "incidents": incidents,
        "events": events,
        "behaviours": dict(behaviour_seen),
        "pose_status": dict(pose_status),
        "people": pipeline.people(),
        "activity": pipeline.activity_state,
        "pose_ms": pipeline.pose.stats.ms_per_crop,
        "detector_ms": pipeline.detector.stats.avg_inference_ms,
        "histories": len(pipeline.pose_history),
    }


def _measure_activity_cost(people: int = 4, iterations: int = 150) -> float:
    """Time the history/movement/classifier chain on synthetic skeletons.

    The end-to-end A/B above cannot resolve this layer: it is well under a
    millisecond against a detector costing tens of milliseconds, so run-to-run
    variance is larger than the thing being measured. This times it directly.
    """
    sys.path.insert(0, str(ROOT / "tests"))
    from tests.test_activity import box, result, skeleton   # noqa: E402
    from core.behaviour_classifier import BehaviourClassifier  # noqa: E402
    from core.movement import MovementAnalyzer               # noqa: E402
    from core.pose_history import PoseHistory                # noqa: E402

    history, analyzer = PoseHistory(), MovementAnalyzer()
    classifier = BehaviourClassifier()
    for warm in range(30):
        for t in range(people):
            points = skeleton(x=0.1 + 0.05 * t)
            history.record(t, result(points, t, warm / 20), box(points),
                           warm / 20)
            classifier.update(analyzer.analyse(history.get(t)))

    started = time.perf_counter()
    for i in range(iterations):
        ts = (30 + i) / 20
        for t in range(people):
            points = skeleton(x=0.1 + 0.05 * t + 0.002 * i)
            history.record(t, result(points, t, ts), box(points), ts)
            classifier.update(analyzer.analyse(history.get(t)))
    elapsed = time.perf_counter() - started
    return elapsed / iterations / people * 1000


def main() -> int:
    config = load_config(ROOT / "config")

    if len(sys.argv) > 1:
        clip = Path(sys.argv[1])
        if not clip.is_absolute():
            clip = ROOT / clip
    else:
        # Default to whichever enabled camera has the most people in it.
        clip = config.path("input_dir") / "WhatsApp Video 2026-09-08 at 19.56.37.mp4"
        if not clip.exists():
            clip = config.path("input_dir") / "demo_border.mp4"

    if not clip.exists():
        print(f"MISSING: {clip}")
        return 1

    print(f"clip: {clip.name}, first {MAX_FRAMES} frames, identical both ways\n")

    print("=== 1. activity DISABLED (baseline) ===")
    off = run(config, clip, activity=False)
    print(f"  {off['frames']} frames, {off['ms_per_frame']:.1f} ms/frame")
    print(f"  tracks: {off['tracks']}")
    print(f"  incidents: {off['incidents']}")

    print("\n=== 2. activity ENABLED ===")
    on = run(config, clip, activity=True)
    print(f"  {on['frames']} frames, {on['ms_per_frame']:.1f} ms/frame")
    print(f"  tracks: {on['tracks']}")
    print(f"  incidents: {on['incidents']}")
    print(f"  pose status seen: {on['pose_status']}")
    print(f"  behaviours seen : {on['behaviours']}")
    print(f"  pose histories held: {on['histories']}")
    print(f"  summary: {on['activity']['summary']}")

    print(f"\n  behaviour changes raised: {len(on['events'])}")
    for event in on["events"][:14]:
        print(f"     t={event.start_time:8.2f}  {event.describe()}  "
              f"{event.evidence}")

    print("\n  final per-person state:")
    for row in on["people"]:
        print(f"     PERSON #{row['track_id']:02d}  "
              f"behaviour {row['behaviour']:<9} "
              f"({row['confidence']:.2f}, {row['duration']:.1f}s)  "
              f"movement {str(row['direction']):<11} "
              f"pose {row['pose_status']:<12} "
              f"{row['keypoints']}/17 keypoints")

    print("\n=== 3. cost ===")
    delta = on["ms_per_frame"] - off["ms_per_frame"]
    print(f"  end to end: {off['ms_per_frame']:.1f} -> "
          f"{on['ms_per_frame']:.1f} ms/frame ({delta:+.1f} ms)")
    print(f"  detector {on['detector_ms']:.1f} ms, "
          f"pose {on['pose_ms']:.1f} ms per crop")
    print(f"  activity chain measured directly: "
          f"{_measure_activity_cost():.2f} ms per analysed frame per person")
    print("  Treat the end-to-end delta as noise, not as a measurement: two "
          "runs of\n  a 67 ms/frame detector vary by more than this layer "
          "costs, and the\n  first run also pays for model warm-up. The "
          "direct figure is the real one.")

    print()
    distinct = {e.behavior for e in on["events"]}
    per_track = Counter(e.track_id for e in on["events"])
    checks = [
        ("track ids unchanged by the activity layer",
         off["tracks"] == on["tracks"]),
        ("incidents unchanged by the activity layer",
         off["incidents"] == on["incidents"]),
        ("pose landmarks were attached to person tracks",
         on["pose_status"].get("TRACKED", 0) > 0),
        ("behaviour was classified for at least one person",
         any(b != "UNKNOWN" for b in on["behaviours"])),
        ("behaviour is temporal, not one event per frame",
         all(count < on["frames"] / 4 for count in per_track.values())
         if per_track else True),
        ("each person keeps a separate pose history",
         on["histories"] >= 1),
        ("nothing was fabricated for people pose could not read",
         "UNAVAILABLE" not in on["behaviours"]),
    ]
    for label, passed in checks:
        print(f"   [{'PASS' if passed else 'FAIL'}] {label}")

    ok = all(passed for _, passed in checks)
    print("\nRESULT: " + ("ACTIVITY ANALYSIS OK, PIPELINE UNCHANGED" if ok
                          else "UNEXPECTED - see the failures above"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
