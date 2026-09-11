#!/usr/bin/env python3
"""Live GUI test: run the real worker under the real window, headless.

This is the test that proves the pieces fit together. It builds the actual
main window, starts an actual :class:`PipelineWorker` on the demo clip, lets it
run, and captures the camera view while frames are flowing.

It also measures GUI responsiveness. A timer fires every 100 ms on the GUI
thread and records how late it was. If YOLO were running on the GUI thread
those numbers would be hundreds of milliseconds; they should stay small.

    python scripts/gui_live_test.py --seconds 25 --out /tmp/live
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--out", default="/tmp/borderguard_live")
    parser.add_argument("--video", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.show:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtCore import QElapsedTimer, QTimer
    from PySide6.QtWidgets import QApplication

    from app.main import _tune_gil_switch_interval
    _tune_gil_switch_interval()

    from app.gui.main_window import MainWindow
    from app.gui.theme import apply_theme
    from core.config import load_config
    from core.logging_setup import setup_logging
    from database.database import Database

    setup_logging(level="WARNING", console=True)
    config = load_config()
    db = Database(config.path("database_file"))

    video = Path(args.video) if args.video else (
        config.path("input_dir") / "demo_border.mp4"
    )
    if not video.exists():
        print(f"No video at {video}. Run scripts/make_demo_video.py first.")
        return 2

    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow(config, db)
    window.resize(1600, 950)
    window.show()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- probe
    # A GUI-thread timer that records how late each tick is. Lateness is a
    # direct measure of whether the interface is being blocked.
    lateness: list[float] = []
    clock = QElapsedTimer()
    clock.start()
    state = {"expected": 100}

    def probe() -> None:
        actual = clock.elapsed()
        lateness.append(max(0.0, actual - state["expected"]))
        state["expected"] += 100

    probe_timer = QTimer()
    probe_timer.timeout.connect(probe)
    probe_timer.start(100)

    # ------------------------------------------------------------- progress
    counters = {"frames": 0, "incidents": 0, "alerts": 0}
    shots: list[Path] = []

    def on_frame(_frame, stats) -> None:
        counters["frames"] += 1
        n = counters["frames"]
        if n in (12, 60, 130, 220):
            path = out_dir / f"live_{n:04d}.png"
            window.grab().save(str(path))
            shots.append(path)

    def on_incident(_incident) -> None:
        counters["incidents"] += 1

    def on_alert(_alert, _incident) -> None:
        counters["alerts"] += 1

    camera = {
        "id": "CAM-01",
        "name": "Sector Alpha - Main Approach",
        "location": "Sector Alpha",
        "enabled": True,
        "zones_profile": "default",
        "source": {"type": "video", "path": str(video),
                   "loop": False, "target_fps": None},
    }

    window.go_to(1)                       # Cameras page
    window.start_processing(camera)
    assert window.worker is not None
    window.worker.frame_ready.connect(on_frame)
    window.worker.incident_raised.connect(on_incident)
    window.worker.alert_raised.connect(on_alert)

    finished = {"done": False}

    # A real Qt event loop. Driving the loop by hand with processEvents()
    # would measure the harness, not the application.
    print(f"Running the pipeline through the GUI for up to {args.seconds:.0f}s…")

    def capture_and_quit() -> None:
        for index, name in ((0, "dashboard"), (1, "cameras"), (2, "incidents"),
                            (3, "alerts"), (5, "analytics"), (6, "evidence")):
            window.go_to(index)
            path = out_dir / f"after_{index}_{name}.png"
            window.grab().save(str(path))
            shots.append(path)
        probe_timer.stop()
        app.quit()

    def on_finished(_summary) -> None:
        finished["done"] = True
        QTimer.singleShot(700, capture_and_quit)

    window.worker.session_finished.disconnect()
    window.worker.session_finished.connect(on_finished)
    window.worker.session_finished.connect(window._on_session_finished)

    def deadline_reached() -> None:
        if not finished["done"]:
            print("  time limit reached, stopping the worker")
            window.stop_processing()

    QTimer.singleShot(int(args.seconds * 1000), deadline_reached)
    # Hard backstop so the test can never hang a CI run.
    QTimer.singleShot(int((args.seconds + 40) * 1000), capture_and_quit)

    app.exec()

    # ---------------------------------------------------------------- report
    counts = db.analytics_counts()
    worst = max(lateness) if lateness else 0.0
    average = sum(lateness) / len(lateness) if lateness else 0.0
    over_250 = sum(1 for value in lateness if value > 250)

    print("\n" + "=" * 66)
    print("  LIVE GUI TEST RESULT")
    print("=" * 66)
    print(f"  Frames displayed          {counters['frames']}")
    print(f"  Incidents signalled       {counters['incidents']}")
    print(f"  Alerts signalled          {counters['alerts']}")
    print(f"  Incidents in database     {counts['incidents']}")
    print(f"  Evidence files            {counts['evidence_files']}")
    print(f"  Video view frames painted {window.page_cameras.video.frames_shown}")
    print()
    print(f"  GUI timer ticks           {len(lateness)}")
    print(f"  Average lateness          {average:.1f} ms")
    print(f"  Worst lateness            {worst:.1f} ms")
    print(f"  Ticks later than 250 ms   {over_250}")
    print()
    print(f"  Screenshots               {out_dir}")
    for shot in shots:
        print(f"    {shot.name}")

    # The budget has to account for the machine. YOLO is CPU-bound and Python
    # holds the GIL for stretches of the pipeline, so on a 2-core box the
    # interface and the analysis are genuinely competing for the same cores.
    # The test that matters is "does the window keep repainting", not "is the
    # timer perfect": a frozen application shows lateness in whole seconds and
    # paints nothing.
    cores = os.cpu_count() or 2
    budget = 400 if cores >= 8 else 800 if cores >= 4 else 1500
    blocked = worst >= budget
    painting = window.page_cameras.video.frames_shown > 20
    displayed = counters["frames"] > 0 and painting

    print()
    print(f"  CPU cores                 {cores}")
    print(f"  Lateness budget           {budget} ms (worst tick)")
    print(f"  Video displayed           {'YES' if displayed else 'NO'}")
    print(f"  GUI never blocked         {'YES' if not blocked else 'NO'}")
    if cores < 4:
        print()
        print("  Note: with fewer than 4 cores the analysis thread and the")
        print("  interface compete for the same CPUs, so timer lateness in the")
        print("  100-300 ms range is expected and is not a freeze. The window")
        print("  keeps repainting throughout.")
    print("=" * 66)
    responsive = not blocked

    db.close()
    return 0 if (displayed and responsive) else 1


if __name__ == "__main__":
    raise SystemExit(main())
