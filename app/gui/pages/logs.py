"""System Logs: what the application has been doing.

Reads the ``system_logs`` table, which mirrors the rotating file log. Startup,
model loading, camera connections, inference errors, incident creation,
database problems and shutdown all land here.

Errors are never swallowed. If something failed, it is in this table with the
module that reported it.
"""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
)

from app.gui.pages.base import Page
from app.gui.theme import Palette, mono_font
from app.gui.widgets.common import Card, KpiCard

LEVEL_COLOURS = {
    "DEBUG": Palette.TEXT_DIM,
    "INFO": Palette.TEXT_MUTED,
    "WARNING": Palette.MEDIUM,
    "ERROR": Palette.CRITICAL,
    "CRITICAL": Palette.CRITICAL,
}


class LogsPage(Page):
    title = "System Logs"
    subtitle = "Application events, mirrored from the rotating log file"

    def build(self) -> None:
        kpis = QHBoxLayout()
        kpis.setSpacing(10)
        self.kpi_total = KpiCard("Log Entries", "0", "stored in the database")
        self.kpi_warnings = KpiCard("Warnings", "0", "", Palette.MEDIUM)
        self.kpi_errors = KpiCard("Errors", "0", "", Palette.CRITICAL)
        self.kpi_file = KpiCard("Log File", "-", "")
        for card in (self.kpi_total, self.kpi_warnings, self.kpi_errors,
                     self.kpi_file):
            kpis.addWidget(card)
        self.content.addLayout(kpis)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.filter_level = QComboBox()
        self.filter_level.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR"])
        self.filter_level.setCurrentText("ALL")
        self.filter_level.currentIndexChanged.connect(self.refresh)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by text…")
        self.search.textChanged.connect(self.refresh)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh)
        btn_trim = QPushButton("Trim Old Entries")
        btn_trim.clicked.connect(self._trim)

        controls.addWidget(self.filter_level)
        controls.addWidget(self.search, 1)
        controls.addWidget(btn_refresh)
        controls.addWidget(btn_trim)
        self.content.addLayout(controls)

        card = Card("Log")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["TIME", "LEVEL", "SOURCE",
                                              "MESSAGE"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        card.add(self.table, 1)
        self.content.addWidget(card, 1)

        self.path_label = QLabel("")
        self.path_label.setObjectName("Dim")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.content.addWidget(self.path_label)

    # ------------------------------------------------------------------

    def refresh(self) -> None:
        db = self.ctx.db
        if db is None:
            return

        level = self.filter_level.currentText()
        rows = db.get_system_logs(level=level, limit=800)

        needle = self.search.text().strip().lower()
        if needle:
            rows = [r for r in rows
                    if needle in str(r["message"]).lower()
                    or needle in str(r["logger"]).lower()]

        all_rows = db.get_system_logs(limit=20000)
        warnings = sum(1 for r in all_rows if r["level"] == "WARNING")
        errors = sum(1 for r in all_rows
                     if r["level"] in ("ERROR", "CRITICAL"))

        self.kpi_total.set_value(len(all_rows), f"{len(rows)} shown")
        self.kpi_warnings.set_value(
            warnings, accent=Palette.MEDIUM if warnings else Palette.TEXT_DIM)
        self.kpi_errors.set_value(
            errors, accent=Palette.CRITICAL if errors else Palette.LOW)

        log_file = Path(str(self.ctx.config.get("logging.file",
                                                "logs/borderguard.log")))
        self.kpi_file.set_value(log_file.name, str(
            self.ctx.config.get("logging.level", "INFO")))
        self.path_label.setText(
            f"Rotating file log: {self.ctx.config.path('logs_dir') / log_file.name}"
        )

        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = [
                time.strftime("%Y-%m-%d %H:%M:%S",
                              time.localtime(float(row["ts"]))),
                str(row["level"]),
                str(row["logger"] or ""),
                str(row["message"]),
            ]
            colour = LEVEL_COLOURS.get(str(row["level"]), Palette.TEXT)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFont(mono_font(9))
                item.setForeground(Palette.qcolor(
                    colour if column in (1, 3) else Palette.TEXT_DIM))
                self.table.setItem(index, column, item)

    def _trim(self) -> None:
        db = self.ctx.db
        if db is None:
            return
        if QMessageBox.question(
            self, "Trim log table",
            "Keep only the most recent 5,000 entries in the database?\n\n"
            "The rotating file log on disk is not touched.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        db.trim_system_logs(keep=5000)
        self.refresh()
