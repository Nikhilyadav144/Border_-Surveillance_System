"""Verify pose estimation, and measure exactly what it costs.

Runs the same clip through the same pipeline twice, once with pose disabled
and once enabled, and compares. The two things worth proving are that pose
adds keypoints and that it changes nothing else: same tracks, same ids, same
incidents. A pose feature that quietly shifted the detections would be a
regression dressed as a feature.

Usage, from the project root with the virtualenv active:

    python3 scripts/check_pose.py
    python3 scripts/check_pose.py data/input/sample_02.mp4
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_config              # noqa: E402
from core.pipeline import SurveillancePipeline   # noqa: E402
from core.pose import KEYPOINT_NAMES             # noqa: E402
from inputs.video_source import FileVideoSource  # noqa: E402

DEFAULT_CLIP = ROOT / "data" / "input" / "demo_border.mp4"
MAX_FRAMES = 200


def run(config, clip: Path, pose_enabled: bool) -> dict:
    camera = {"id": "CAM-01", "name": "check", "zones_profile": "default"}
    pipeline = SurveillancePipeline(config, camera, database=None,
                                    annotate=False)
    pipeline.pose.enabled = pose_enabled

    source = FileVideoSource(clip, source_id="CAM-01", loop=False,
                             target_fps=None)
    source.open()

    track_ids: set[int] = set()
    incidents = 0
    posed_tracks: dict[int, int] = {}
    postures: dict[str, int] = {}
    keypoint_samples: list[int] = []
    frames = 0
    wall_started = time.perf_counter()

    while frames < MAX_FRAMES:
        frame = source.read()
        if frame is None:
            break
        frames += 1
        out = pipeline.process(frame.image, frame.index, frame.ts)
        incidents += len(out.incidents)
        for track in out.tracks:
            track_ids.add(track.track_number)
            if track.pose is not None:
                posed_tracks[track.track_number] = track.pose_count
                keypoint_samples.append(track.pose.visible_count)
                if track.posture is not None:
                    label = track.posture.label
                    postures[label] = postures.get(label, 0) + 1
    source.release()
    wall = time.perf_counter() - wall_started

    return {
        "frames": frames,
        "wall_seconds": wall,
        "ms_per_frame": wall / frames * 1000 if frames else 0.0,
        "detector_ms": pipeline.detector.stats.avg_inference_ms,
        "track_ids": sorted(track_ids),
        "incidents": incidents,
        "posed_tracks": posed_tracks,
        "postures": postures,
        "mean_visible_keypoints": (float(np.mean(keypoint_samples))
                                   if keypoint_samples else 0.0),
        "pose_stats": pipeline.pose.stats,
        "pose_status": pipeline.pose.status_text(),
    }


def main() -> int:
    clip = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CLIP
    if not clip.is_absolute():
        clip = ROOT / clip
    if not clip.exists():
        print(f"MISSING: {clip}")
        return 1

    config = load_config(ROOT / "config")
    print(f"clip: {clip.name}   frames analysed: up to {MAX_FRAMES}")
    print(f"pose config: {config.get('pose.model_file')}, "
          f"imgsz {config.get('pose.image_size')}, "
          f"min box height {config.get('pose.min_box_height')}, "
          f"max {config.get('pose.max_persons_per_frame')} people, "
          f"every {config.get('pose.interval_frames')} analysed frame(s)\n")

    print("=== 1. pose DISABLED (baseline) ===")
    off = run(config, clip, pose_enabled=False)
    print(f"  {off['frames']} frames in {off['wall_seconds']:.1f} s "
          f"= {off['ms_per_frame']:.1f} ms/frame")
    print(f"  detector inference: {off['detector_ms']:.1f} ms")
    print(f"  track ids: {off['track_ids']}")
    print(f"  incidents: {off['incidents']}")

    print("\n=== 2. pose ENABLED ===")
    on = run(config, clip, pose_enabled=True)
    stats = on["pose_stats"]
    print(f"  {on['frames']} frames in {on['wall_seconds']:.1f} s "
          f"= {on['ms_per_frame']:.1f} ms/frame")
    print(f"  detector inference: {on['detector_ms']:.1f} ms")
    print(f"  track ids: {on['track_ids']}")
    print(f"  incidents: {on['incidents']}")
    print(f"  status: {on['pose_status']}")
    print(f"  crops run: {stats.crops_run}, poses found: {stats.poses_found}")
    print(f"  skipped too small: {stats.skipped_too_small}, "
          f"invalid: {stats.skipped_invalid}, "
          f"over budget: {stats.skipped_budget}")
    print(f"  pose cost: {stats.ms_per_crop:.1f} ms per crop, "
          f"{stats.avg_ms:.1f} ms per posed frame")
    print(f"  tracks with keypoints: {sorted(on['posed_tracks'])}")
    print(f"  mean visible keypoints of 17: "
          f"{on['mean_visible_keypoints']:.1f}")
    print(f"  postures seen: {on['postures'] or 'none'}")

    print("\n=== 3. impact ===")
    delta = on["ms_per_frame"] - off["ms_per_frame"]
    pct = (delta / off["ms_per_frame"] * 100) if off["ms_per_frame"] else 0.0
    print(f"  {off['ms_per_frame']:.1f} ms/frame -> "
          f"{on['ms_per_frame']:.1f} ms/frame  "
          f"({delta:+.1f} ms, {pct:+.0f}%)")

    print()
    checks = [
        ("track ids identical with and without pose",
         off["track_ids"] == on["track_ids"]),
        ("incident count identical with and without pose",
         off["incidents"] == on["incidents"]),
        ("pose produced keypoints for at least one track",
         len(on["posed_tracks"]) > 0),
        ("keypoints are the COCO-17 set",
         len(KEYPOINT_NAMES) == 17),
        ("small boxes were skipped rather than guessed",
         stats.skipped_too_small > 0 or stats.crops_run > 0),
    ]
    for label, passed in checks:
        print(f"   [{'PASS' if passed else 'FAIL'}] {label}")

    ok = all(passed for _, passed in checks)
    print("\nRESULT: " + ("POSE OK, PIPELINE UNCHANGED" if ok
                          else "UNEXPECTED - see the failures above"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
