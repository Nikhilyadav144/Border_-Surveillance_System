#!/usr/bin/env python3
"""Headless GUI smoke test.

Builds the real main window under Qt's ``offscreen`` platform, visits every
page, and writes a PNG of each one. It proves three things that a unit test
cannot:

* every screen constructs without raising
* every screen renders against the real database
* the navigation and the refresh cycle work

    python scripts/gui_smoke_test.py --out /tmp/shots

On a machine with a display it runs the same way; pass ``--show`` to open the
window instead of capturing it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/borderguard_shots")
    parser.add_argument("--show", action="store_true",
                        help="Open the window instead of capturing it")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=950)
    args = parser.parse_args()

    if not args.show:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow
    from app.gui.theme import apply_theme
    from core.config import load_config
    from core.logging_setup import setup_logging
    from database.database import Database

    setup_logging(level="WARNING", console=True)
    config = load_config()
    db = Database(config.path("database_file"))

    app = QApplication(sys.argv)
    apply_theme(app)

    window = MainWindow(config, db)
    window.resize(args.width, args.height)
    window.show()

    if args.show:
        return app.exec()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for index, page in enumerate(window.pages):
        name = page.title.lower().replace(" ", "_")
        try:
            window.go_to(index)
            app.processEvents()
            for _ in range(4):
                app.processEvents()
            path = out_dir / f"{index}_{name}.png"
            window.grab().save(str(path))
            print(f"  ok    {page.title:<14} -> {path}")
        except Exception as exc:
            failures.append(f"{page.title}: {exc}")
            print(f"  FAIL  {page.title:<14} {exc}")

    db.close()

    if failures:
        print(f"\n{len(failures)} page(s) failed:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print(f"\nAll {len(window.pages)} pages rendered. Screenshots in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
