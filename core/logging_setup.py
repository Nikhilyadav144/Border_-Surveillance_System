"""Structured application logging.

Three sinks:

* a rotating file under ``logs/`` - the durable record
* the console - useful when running from a terminal
* SQLite, via :class:`DatabaseLogHandler` - what the System Logs screen reads

The database handler is attached separately by the application once the
database exists, because logging has to work before the database does (the
first thing worth logging is often a database failure).
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.paths import ensure_dir, resolve

if TYPE_CHECKING:  # pragma: no cover
    from database.database import Database

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class DatabaseLogHandler(logging.Handler):
    """Mirror log records into the ``system_logs`` table.

    Failures here are swallowed on purpose. A logging handler that raises
    during an error path turns a recoverable problem into a crash, and the
    file and console sinks still have the record.
    """

    def __init__(self, database: "Database", level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self._db = database
        self._lock = threading.Lock()
        self._failed = False

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        if self._failed:
            return
        try:
            with self._lock:
                self._db.insert_system_log(
                    level=record.levelname,
                    logger=record.name,
                    message=self.format(record),
                )
        except Exception:  # pragma: no cover - defensive
            # Stop trying after the first failure so we do not spam.
            self._failed = True


def setup_logging(
    level: str = "INFO",
    log_file: str | Path = "logs/borderguard.log",
    max_bytes: int = 5_242_880,
    backup_count: int = 5,
    console: bool = True,
) -> Path:
    """Configure the root logger. Returns the resolved log file path."""
    numeric = getattr(logging, str(level).upper(), logging.INFO)

    path = resolve(log_file)
    ensure_dir(path.parent)

    root = logging.getLogger()
    root.setLevel(numeric)

    # Remove handlers we installed previously so repeated calls (tests, or a
    # settings reload) do not duplicate every line.
    for handler in list(root.handlers):
        if getattr(handler, "_borderguard", False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric)
    file_handler._borderguard = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        stream.setLevel(numeric)
        stream._borderguard = True  # type: ignore[attr-defined]
        root.addHandler(stream)

    # Third-party libraries are chatty and their INFO output is not ours.
    for noisy in ("ultralytics", "PIL", "matplotlib", "urllib3", "easyocr", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return path


def attach_database_handler(database: Any, level: str = "INFO") -> DatabaseLogHandler:
    """Start mirroring log records into SQLite."""
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    handler = DatabaseLogHandler(database, level=numeric)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._borderguard = True  # type: ignore[attr-defined]
    logging.getLogger().addHandler(handler)
    return handler
