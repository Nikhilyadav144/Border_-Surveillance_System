"""Application entry point.

Start-up order matters and is deliberate:

1. logging, so that a failure in step 2 or 3 is recorded
2. configuration, which can refuse to start on a malformed file
3. the database, which the log handler then also writes to
4. the GUI

Any failure before the GUI exists is reported on the console *and*, if Qt is
already available, in a dialog. A desktop application that dies with a
traceback and no window is a bug, not an error message.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

# Make the project importable no matter where the interpreter was launched.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import ConfigError, load_config          # noqa: E402
from core.logging_setup import (                          # noqa: E402
    attach_database_handler,
    setup_logging,
)
from database.database import Database, DatabaseError     # noqa: E402

logger = logging.getLogger("borderguard")


def _fatal(title: str, message: str) -> int:
    """Report a start-up failure through a dialog when we can, stderr always."""
    print(f"\n{title}\n{'-' * len(title)}\n{message}\n", file=sys.stderr)
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, title, message)
    except Exception:
        pass
    return 1


def _tune_gil_switch_interval() -> None:
    """Let the interface thread get scheduled more often.

    CPython lets a thread hold the GIL for up to `sys.getswitchinterval()`
    seconds (5 ms by default) before it is asked to yield. The pipeline thread
    runs long stretches of pure Python - tracking, zone maths, threat scoring -
    and at 5 ms grants the interface visibly loses its place, especially on a
    machine with few cores.

    1 ms costs a little throughput and buys a window that keeps repainting.
    """
    try:
        if sys.getswitchinterval() > 0.001:
            sys.setswitchinterval(0.001)
    except (ValueError, AttributeError):  # pragma: no cover
        pass


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv)
    _tune_gil_switch_interval()

    # ---------------------------------------------------------------- 1. logs
    log_path = setup_logging(level="INFO")
    logger.info("=" * 62)
    logger.info("Border Guard AI starting")

    # -------------------------------------------------------------- 2. config
    try:
        config = load_config()
    except ConfigError as exc:
        logger.critical("Configuration error: %s", exc)
        return _fatal(
            "Configuration error",
            f"{exc}\n\nFix the file in config/ and start again.\n"
            f"A log of this run is at:\n{log_path}",
        )
    except Exception as exc:
        logger.critical("Unexpected configuration failure: %s", exc,
                        exc_info=True)
        return _fatal("Configuration error", str(exc))

    setup_logging(
        level=str(config.get("logging.level", "INFO")),
        log_file=config.get("logging.file", "logs/borderguard.log"),
        max_bytes=int(config.get("logging.max_bytes", 5_242_880)),
        backup_count=int(config.get("logging.backup_count", 5)),
        console=bool(config.get("logging.console", True)),
    )
    for warning in config.warnings:
        logger.warning("Configuration: %s", warning)

    # ------------------------------------------------------------ 3. database
    try:
        db = Database(config.path("database_file"))
        attach_database_handler(db, level=str(config.get("logging.db_level",
                                                         "INFO")))
    except DatabaseError as exc:
        logger.critical("Database error: %s", exc)
        return _fatal(
            "Database error",
            f"{exc}\n\nCheck that data/database/ exists and is writable.",
        )

    logger.info("Configuration and database ready")
    logger.info("Model: %s", config.model_path())
    logger.info("Database: %s", config.path("database_file"))

    # ----------------------------------------------------------------- 4. GUI
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        from app.gui.main_window import MainWindow
        from app.gui.theme import apply_theme
    except ImportError as exc:
        logger.critical("PySide6 is not available: %s", exc)
        return _fatal(
            "PySide6 is not installed",
            "The desktop interface needs PySide6.\n\n"
            "Install it with:\n    pip install -r requirements.txt",
        )

    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(argv)
    app.setApplicationName(str(config.get("application.name", "IBVAP")))
    app.setApplicationVersion(str(config.get("application.version", "1.0.0")))
    app.setOrganizationName(str(config.get("application.team",
                                           "Vision Vigilants")))

    # Dock and window icon. Optional: a missing file is not a reason to
    # refuse to start.
    from pathlib import Path as _Path
    icon_path = _Path(__file__).resolve().parents[1] / "app" / "resources" \
        / "app-icon.png"
    if icon_path.exists():
        from PySide6.QtGui import QIcon
        app.setWindowIcon(QIcon(str(icon_path)))
    apply_theme(app)

    try:
        window = MainWindow(config, db)
    except Exception as exc:
        logger.critical("Could not build the main window: %s", exc,
                        exc_info=True)
        return _fatal(
            "Could not start the interface",
            f"{exc}\n\n{traceback.format_exc(limit=3)}",
        )

    window.show()
    logger.info("Interface shown, entering the event loop")

    try:
        code = app.exec()
    finally:
        db.close()
        logger.info("Border Guard AI stopped")
        logger.info("=" * 62)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
