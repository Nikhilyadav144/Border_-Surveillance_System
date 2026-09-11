"""Measure what the pipeline costs, with pose and activity off and on.

The point of this script is that the cost of the activity chain should be a
MEASURED number rather than an assumption. It runs each enabled file-backed
camera twice through the identical pipeline, changing one thing between the
runs: ``pose.enabled`` and ``activity.enabled``.

What is reported, per camera and in total:

* wall-clock frames per second, which is what decides how many cameras a
  machine can carry;
* detector inference milliseconds, from the detector's own counters;
* pose inference milliseconds, from the pose estimator's own counters;
* total pipeline milliseconds per frame, measured around ``process()`` so it
  includes zones, lines, behaviour, boundary, threat and annotation, not just
  the two model calls;
* the largest number of person tracks alive at once, because per-frame cost
  depends on how many people are in view.

Two numbers are reported for cost because the pipeline can skip frames.
``detection.frame_stride`` of 2 means every second frame is analysed and the
rest are cheap, so per-SOURCE-frame and per-ANALYSED-frame are different
questions. Both are shown.

The numbers are specific to the machine that runs it. Run it on your own
hardware rather than quoting these; on a Mac that means the project venv.

Usage, from the project root:

    python3 scripts/check_performance.py
    python3 scripts/check_performance.py --camera CAM-01
    python3 scripts/check_performance.py --max-frames 200
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_config                # noqa: E402
from core.pipeline import SurveillancePipeline     # noqa: E402
from inputs.video_source import FileVideoSource    # noqa: E402


#: Frames run through a freshly built pipeline before the clock starts.
#
# Not a nicety. The first measurement of this script reported the activity
# chain as costing MINUS 20 percent on CAM-01, which is impossible: pose is
# extra work and cannot make the pipeline faster. The cause was that the
# first pass paid for loading the model, allocating tensors and warming the
# kernels, and the second pass inherited all of it. The detector's own
# average inference time differed by 90.0 ms against 67.6 ms between two
# passes over the SAME frames with the SAME model, which is the giveaway.
WARMUP_FRAMES = 24


@dataclass
class Run:
    """One pass over one clip."""

    label: str
    frames: int = 0
    analysed: int = 0
    wall_seconds: float = 0.0
    detect_ms_total: float = 0.0
    pose_ms_total: float = 0.0
    pose_crops: int = 0
    peak_person_tracks: int = 0
    incidents: int = 0
    #: Split, because the pipeline skips frames and the two populations are
    #: an order of magnitude apart. Averaging them together produces a
    #: number that describes neither.
    analysed_ms: list[float] = field(default_factory=list)
    skipped_ms: list[float] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frames / self.wall_seconds if self.wall_seconds else 0.0

    @property
    def ms_per_source_frame(self) -> float:
        return 1000.0 * self.wall_seconds / self.frames if self.frames else 0.0

    @property
    def detect_ms(self) -> float:
        return self.detect_ms_total / self.analysed if self.analysed else 0.0

    @property
    def pose_ms(self) -> float:
        return self.pose_ms_total / self.analysed if self.analysed else 0.0

    @staticmethod
    def _pct(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    def analysed_median(self) -> float:
        return self._pct(self.analysed_ms, 0.5)

    def analysed_p95(self) -> float:
        return self._pct(self.analysed_ms, 0.95)

    def skipped_median(self) -> float:
        return self._pct(self.skipped_ms, 0.5)


def measure(config, camera, clip: Path, activity: bool,
            max_frames: int | None) -> Run:
    """One full pass, with the activity chain either off or on."""
    config.set("pose.enabled", activity)
    config.set("activity.enabled", activity)

    pipeline = SurveillancePipeline(config, camera, database=None,
                                    annotate=True)
    source = FileVideoSource(clip, source_id=str(camera["id"]), loop=False,
                             target_fps=None)
    source.open()

    run = Run(label="activity ON" if activity else "activity OFF")

    # -- warm up, then throw the numbers away --------------------------
    for _ in range(WARMUP_FRAMES):
        frame = source.read()
        if frame is None:
            break
        pipeline.process(frame.image, frame.index, frame.ts)
    pipeline.detector.stats.total_inference_ms = 0.0
    pipeline.detector.stats.frames = 0
    if pipeline.pose is not None:
        pipeline.pose.stats.total_ms = 0.0
        pipeline.pose.stats.crops_run = 0

    started = time.perf_counter()
    while True:
        frame = source.read()
        if frame is None:
            break
        if max_frames is not None and run.frames >= max_frames:
            break
        began = time.perf_counter()
        out = pipeline.process(frame.image, frame.index, frame.ts)
        elapsed = 1000.0 * (time.perf_counter() - began)
        run.frames += 1
        if out.skipped:
            run.skipped_ms.append(elapsed)
        else:
            run.analysed += 1
            run.analysed_ms.append(elapsed)
        run.incidents += len(out.incidents)
        people = sum(1 for t in out.tracks
                     if t.class_name.lower() in pipeline.person_classes)
        run.peak_person_tracks = max(run.peak_person_tracks, people)
    run.wall_seconds = time.perf_counter() - started
    source.release()

    # Straight from the engines' own counters, not re-derived here.
    run.detect_ms_total = pipeline.detector.stats.total_inference_ms
    if activity and pipeline.pose is not None:
        run.pose_ms_total = pipeline.pose.stats.total_ms
        run.pose_crops = pipeline.pose.stats.crops_run
    return run


def best_of(config, camera, clip: Path, activity: bool,
            max_frames: int | None, repeats: int) -> Run:
    """Run the pass several times and keep the FASTEST.

    The minimum, not the mean. Contention on a shared machine can only ever
    ADD time, so the fastest observed pass is the one least polluted by
    whatever else was running. Averaging measures the neighbours.

    This matters more than it sounds. Before repeats were added, CAM-02 -
    which contains no people at all, so the activity chain does nothing and
    runs zero pose crops - measured as 14 percent FASTER with the chain
    switched on. The detector's own inference time moved 81.3 ms to 72.9 ms
    across two passes over identical frames with an identical model. That is
    the noise floor, and any per-camera figure smaller than it is not a
    measurement of anything.
    """
    runs = [measure(config, camera, clip, activity, max_frames)
            for _ in range(max(1, repeats))]
    return min(runs, key=lambda r: r.analysed_median() or float("inf"))


def report(camera_id: str, clip: Path, off: Run, on: Run) -> None:
    print(f"\n{'=' * 74}")
    print(f"  {camera_id}   {clip.name}")
    print(f"{'=' * 74}")
    print(f"  frames {off.frames} source, {off.analysed} analysed"
          f"   peak people in view: {on.peak_person_tracks}")
    print()
    print(f"  {'':<32}{'activity OFF':>18}{'activity ON':>18}")
    print(f"  {'-' * 68}")
    rows = [
        ("wall-clock FPS", f"{off.fps:.1f}", f"{on.fps:.1f}"),
        ("ms per source frame (mean)", f"{off.ms_per_source_frame:.1f}",
         f"{on.ms_per_source_frame:.1f}"),
        ("ANALYSED frame ms (median)", f"{off.analysed_median():.1f}",
         f"{on.analysed_median():.1f}"),
        ("ANALYSED frame ms (95th pct)", f"{off.analysed_p95():.1f}",
         f"{on.analysed_p95():.1f}"),
        ("  detector inference ms", f"{off.detect_ms:.1f}",
         f"{on.detect_ms:.1f}"),
        ("  pose inference ms", "-", f"{on.pose_ms:.1f}"),
        ("SKIPPED frame ms (median)", f"{off.skipped_median():.1f}",
         f"{on.skipped_median():.1f}"),
        ("incidents raised", str(off.incidents), str(on.incidents)),
    ]
    for name, a, b in rows:
        print(f"  {name:<32}{a:>18}{b:>18}")

    # The comparison that means something is the median ANALYSED frame.
    # Wall-clock totals mix in the skipped frames and the long tail, and on
    # a shared machine that tail is mostly other people's work.
    base = off.analysed_median()
    if base > 0:
        delta = 100.0 * (on.analysed_median() - base) / base
        print(f"\n  Activity chain costs {delta:+.0f}% on the median "
              f"analysed frame.")
    if on.pose_crops:
        print(f"  {on.pose_crops} pose crops, "
              f"{on.pose_ms_total / on.pose_crops:.1f} ms each.")
    # Incidents must not depend on how fast the machine is. They can
    # legitimately differ between the two runs, because the activity chain
    # supplies boundary evidence that the threat engine scores - but a wild
    # difference means something other than pose changed.
    if off.incidents != on.incidents:
        print(f"  NOTE: incident count differs ({off.incidents} vs "
              f"{on.incidents}). The activity chain feeds boundary evidence "
              f"into scoring, so some difference is expected.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", help="Only this camera id, e.g. CAM-01")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop each pass after this many frames")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Passes per configuration; the fastest is kept")
    args = parser.parse_args()

    config = load_config(ROOT / "config")
    print(f"machine   : {platform.platform()}")
    print(f"python    : {platform.python_version()}")
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available() else "cpu")
        print(f"torch     : {torch.__version__} on {device}")
    except Exception:
        print("torch     : not importable")
    print(f"stride    : detection.frame_stride = "
          f"{config.get('detection.frame_stride', 1)}")

    cameras = [c for c in config.cameras
               if c.get("enabled", True)
               and str(c.get("source", {}).get("type")) == "video"
               and (args.camera is None or str(c.get("id")) == args.camera)]
    if not cameras:
        print("No enabled file-backed cameras matched.")
        return 1

    totals: list[tuple[str, Run, Run]] = []
    for camera in cameras:
        clip = config.path("input_dir") / str(camera["source"].get("path", ""))
        if not clip.exists():
            print(f"\nMISSING: {clip}")
            continue
        off = best_of(config, camera, clip, False, args.max_frames,
                      args.repeats)
        on = best_of(config, camera, clip, True, args.max_frames,
                     args.repeats)
        report(str(camera["id"]), clip, off, on)
        totals.append((str(camera["id"]), off, on))

    if len(totals) > 1:
        print(f"\n{'=' * 74}")
        print("  ACROSS ALL CAMERAS")
        print(f"{'=' * 74}")
        for camera_id, off, on in totals:
            print(f"  {camera_id}  median analysed frame "
                  f"{off.analysed_median():6.1f} ms off "
                  f"-> {on.analysed_median():6.1f} ms on")

        # Capacity from the median analysed-frame cost and the stride, not
        # from wall-clock FPS, which on a shared machine measures the
        # neighbours as much as the pipeline.
        stride = max(1, int(load_config(ROOT / "config")
                            .get("detection.frame_stride", 1)))
        worst = max(on.analysed_median() for _, _, on in totals)
        per_source = worst / stride
        print(f"\n  Worst median analysed frame with activity on: "
              f"{worst:.1f} ms.")
        print(f"  At frame_stride {stride} that is {per_source:.1f} ms per "
              f"source frame, so one process sustains about "
              f"{1000.0 / per_source:.0f} source fps on this machine.")
        print(f"  Feeding 25 fps cameras, that is roughly "
              f"{max(0, int((1000.0 / per_source) // 25))} camera(s) in one "
              f"process; at 15 fps, {max(0, int((1000.0 / per_source) // 15))}.")
        print("  Single-process estimate. It ignores decode and display cost "
              "elsewhere in the app, and this container is CPU-only.")

    print("\nThese numbers belong to THIS machine. Run the script on the "
          "hardware you intend to demo on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
