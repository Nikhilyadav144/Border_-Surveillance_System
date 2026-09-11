"""Apply the Phase 1 activity changes to core/pipeline.py, core/tracker.py
and config/application.yaml, in place.

Run from the project root:

    python3 activity_phase1/apply.py

Every edit asserts it matched exactly once, so a partial or repeated run
stops rather than corrupting a file. Backups are written as *.bak-activity.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent


def edit(path: str, old: str, new: str, label: str) -> None:
    p = ROOT / path
    s = p.read_text()
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"ABORT {path}: {label!r} matched {count} times, "
                         f"expected 1. Nothing was changed in this file.")
    p.write_text(s.replace(old, new))
    print(f"  {path}: {label}")


def main() -> int:
    for f in ("core/pipeline.py", "core/tracker.py", "config/application.yaml"):
        if not (ROOT / f).exists():
            raise SystemExit(f"Run this from the project root; {f} not found.")
        shutil.copy(ROOT / f, ROOT / (f + ".bak-activity"))
    print("backups written as *.bak-activity\n")

    # -- new modules ----------------------------------------------------
    for sub in ("core", "scripts", "tests"):
        for src in (HERE / sub).glob("*.py"):
            shutil.copy(src, ROOT / sub / src.name)
            print(f"  {sub}/{src.name}: installed")

    # -- config ---------------------------------------------------------
    cfg = (HERE / "activity_config.yaml").read_text()
    edit("config/application.yaml",
         "# --------------------------------------------------------------------------\n"
         "# Low-light detection\n"
         "# --------------------------------------------------------------------------",
         cfg + "# --------------------------------------------------------------------------\n"
               "# Low-light detection\n"
               "# --------------------------------------------------------------------------",
         "activity section added")

    # -- tracker --------------------------------------------------------
    edit("core/tracker.py",
         '''    #: Wall/media time of the most recent successful pose reading.
    pose_ts: float = 0.0
    #: How many successful readings this track has had.
    pose_count: int = 0''',
         '''    #: Wall/media time of the most recent successful pose reading.
    pose_ts: float = 0.0
    #: How many successful readings this track has had.
    pose_count: int = 0
    #: UNAVAILABLE / TRACKED / STALE. Never fabricated: a track whose box was
    #: too small or too occluded for pose says UNAVAILABLE and stops there.
    pose_status: str = "UNAVAILABLE"

    # -- activity ---------------------------------------------------------
    # Behaviour is temporal, so it lives on the track rather than on a frame.
    #: Current debounced behaviour name, e.g. WALKING.
    behaviour: str = "UNKNOWN"
    #: How sure the classifier is, 0-1.
    behaviour_confidence: float = 0.0
    #: Seconds the current behaviour has held.
    behaviour_duration: float = 0.0
    #: Compass direction of travel, or None when effectively stationary.
    movement_direction: Optional[str] = None
    #: Latest core.movement.MovementSignals, for display and explanation.
    movement: Optional[Any] = None''',
         "activity fields on TrackState")

    # -- pipeline -------------------------------------------------------
    edit("core/pipeline.py",
         "from core.pose import PoseEstimator, estimate_posture",
         "from core.behaviour_classifier import BehaviourClassifier\n"
         "from core.movement import MovementAnalyzer\n"
         "from core.pose import PoseEstimator, estimate_posture\n"
         "from core.pose_history import (\n"
         "    POSE_STALE,\n"
         "    POSE_TRACKED,\n"
         "    POSE_UNAVAILABLE,\n"
         "    PoseHistory,\n"
         ")",
         "imports")

    edit("core/pipeline.py",
         "        moment = ts if ts is not None else time.time()\n"
         "        self._last_ts = moment",
         "        moment = ts if ts is not None else time.time()\n"
         "        self._last_ts = moment\n"
         "        # Cleared for EVERY frame, not just analysed ones. Clearing only on\n"
         "        # the analysed path left the previous frame's events still sitting\n"
         "        # here during a skipped frame, so anything polling per frame - the\n"
         "        # API stats do - counted each behaviour change twice.\n"
         "        self.behaviour_events = []",
         "per-frame event reset")

    edit("core/pipeline.py",
         "        self.pose = PoseEstimator.from_config(config)\n",
         (HERE / "pipeline_construct.txt").read_text(),
         "activity chain constructed")

    edit("core/pipeline.py",
         (HERE / "pipeline_loop_old.txt").read_text(),
         (HERE / "pipeline_loop_new.txt").read_text(),
         "pose attach loop -> full activity chain")

    geometry_header = (
        "    # ------------------------------------------------------------------\n"
        "    # Security geometry, read by the API stats and the GUI status panel\n"
        "    # ------------------------------------------------------------------")
    edit("core/pipeline.py",
         geometry_header,
         (HERE / "pipeline_props.txt").read_text() + geometry_header,
         "activity_state and people() exposed")

    print("\nDone. Now run:")
    print("    python3 -m pytest -q")
    print("    python3 scripts/check_activity.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
