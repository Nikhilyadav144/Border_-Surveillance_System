"""Watchlist: locally managed identifiers of interest.

Everything on this screen stays on this machine. No external service is
consulted and nothing is uploaded.

The bottom half shows every ANPR read the system has made, including the ones
where OCR failed. Those appear as UNREADABLE with the saved crop still on
disk, because an honest "I could not read this" is more useful than a guess.
"""

from __future__ import annotations

import time
from typing import Optional

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
    QVBoxLayout,
    QWidget,
)

from app.gui.pages.base import Page
from app.gui.theme import Palette, mono_font
from app.gui.widgets.common import Badge, Card, KpiCard
from database.models import ThreatLevel


class WatchlistPage(Page):
    title = "Watchlist"
    subtitle = "Local list of vehicles and identifiers of interest"

    def build(self) -> None:
        kpis = QHBoxLayout()
        kpis.setSpacing(10)
        self.kpi_entries = KpiCard("Watchlist Entries", "0", "active")
        self.kpi_hits = KpiCard("Matches", "0", "exact plate matches",
                                Palette.CRITICAL)
        self.kpi_reads = KpiCard("Plates Read", "0", "OCR succeeded",
                                 Palette.LOW)
        self.kpi_unread = KpiCard("Unreadable", "0", "crop saved for review",
                                  Palette.TEXT_DIM)
        for card in (self.kpi_entries, self.kpi_hits, self.kpi_reads,
                     self.kpi_unread):
            kpis.addWidget(card)
        self.content.addLayout(kpis)

        # -- add entry -----------------------------------------------------
        add_card = Card("Add Entry")
        form = QHBoxLayout()
        form.setSpacing(8)

        self.input_kind = QComboBox()
        self.input_kind.addItems(["plate", "tag"])
        self.input_kind.setFixedWidth(90)
        self.input_value = QLineEdit()
        self.input_value.setPlaceholderText("Identifier, e.g. DL01AB1234")
        self.input_label = QLineEdit()
        self.input_label.setPlaceholderText("Label, e.g. Stolen vehicle report")
        self.input_severity = QComboBox()
        self.input_severity.addItems(list(ThreatLevel.ORDER))
        self.input_severity.setCurrentText(ThreatLevel.HIGH)
        self.input_severity.setFixedWidth(110)

        btn_add = QPushButton("Add")
        btn_add.setObjectName("Primary")
        btn_add.clicked.connect(self._add_entry)
        self.input_value.returnPressed.connect(self._add_entry)

        form.addWidget(self.input_kind)
        form.addWidget(self.input_value, 2)
        form.addWidget(self.input_label, 3)
        form.addWidget(self.input_severity)
        form.addWidget(btn_add)
        add_card.add_layout(form)

        note = QLabel(
            "A match is raised only when OCR actually reads a plate that "
            "equals an entry here, ignoring spaces and case. Near matches are "
            "never reported as confirmed hits."
        )
        note.setObjectName("Dim")
        note.setWordWrap(True)
        add_card.add(note)
        self.content.addWidget(add_card)

        # -- entries -------------------------------------------------------
        entries_card = Card("Watchlist")
        self.entries_table = QTableWidget(0, 7)
        self.entries_table.setHorizontalHeaderLabels(
            ["KIND", "VALUE", "LABEL", "SEVERITY", "ACTIVE", "HITS", "LAST HIT"]
        )
        self.entries_table.verticalHeader().setVisible(False)
        self.entries_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.entries_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.entries_table.setAlternatingRowColors(True)
        header = self.entries_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        entries_card.add(self.entries_table, 1)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        btn_toggle = QPushButton("Enable / Disable")
        btn_toggle.clicked.connect(self._toggle_entry)
        btn_remove = QPushButton("Remove")
        btn_remove.setObjectName("Danger")
        btn_remove.clicked.connect(self._remove_entry)
        actions.addStretch(1)
        actions.addWidget(btn_toggle)
        actions.addWidget(btn_remove)
        entries_card.add_layout(actions)
        self.content.addWidget(entries_card, 2)

        # -- plate reads ---------------------------------------------------
        reads_card = Card("ANPR Reads")
        self.reads_table = QTableWidget(0, 7)
        self.reads_table.setHorizontalHeaderLabels(
            ["TIME", "CAMERA", "TRACK", "PLATE", "CONFIDENCE", "FORMAT",
             "WATCHLIST"]
        )
        self.reads_table.verticalHeader().setVisible(False)
        self.reads_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.reads_table.setAlternatingRowColors(True)
        self.reads_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        reads_card.add(self.reads_table, 1)

        self.anpr_status = QLabel("ANPR status unknown")
        self.anpr_status.setObjectName("Dim")
        reads_card.add(self.anpr_status)
        self.content.addWidget(reads_card, 2)

    # ------------------------------------------------------------------

    def refresh(self) -> None:
        db = self.ctx.db
        if db is None:
            return

        entries = db.get_watchlist()
        active = [e for e in entries if e["active"]]
        self.kpi_entries.set_value(len(active), f"of {len(entries)} total")

        self.entries_table.setRowCount(len(entries))
        for index, row in enumerate(entries):
            values = [
                row["kind"],
                row["value"],
                row["label"] or "-",
                "",
                "yes" if row["active"] else "no",
                str(int(row["hit_count"] or 0)),
                time.strftime("%Y-%m-%d %H:%M",
                              time.localtime(float(row["last_hit_ts"])))
                if row["last_hit_ts"] else "-",
            ]
            for column, value in enumerate(values):
                if column == 3:
                    continue
                item = QTableWidgetItem(str(value))
                if column in (1, 5, 6):
                    item.setFont(mono_font(10))
                if column == 0:
                    item.setData(Qt.UserRole, int(row["id"]))
                if not row["active"]:
                    item.setForeground(Palette.qcolor(Palette.TEXT_DIM))
                self.entries_table.setItem(index, column, item)

            badge = Badge()
            badge.set_threat(row["severity"])
            holder = QWidget()
            layout = QHBoxLayout(holder)
            layout.setContentsMargins(4, 2, 4, 2)
            layout.addWidget(badge)
            self.entries_table.setCellWidget(index, 3, holder)

        self._refresh_reads(db)

    def _refresh_reads(self, db) -> None:
        reads = db.get_plates(limit=200)
        readable = [r for r in reads if r["plate_text"]]
        hits = [r for r in reads if r["watchlist_hit"]]

        self.kpi_reads.set_value(len(readable), f"of {len(reads)} attempts")
        self.kpi_unread.set_value(len(reads) - len(readable))
        self.kpi_hits.set_value(
            len(hits),
            accent=Palette.CRITICAL if hits else Palette.TEXT_DIM,
        )

        self.reads_table.setRowCount(len(reads))
        for index, row in enumerate(reads):
            text = row["plate_text"] or "UNREADABLE"
            values = [
                time.strftime("%H:%M:%S", time.localtime(float(row["ts"]))),
                row["camera_id"],
                f"#{int(row['track_number'] or 0):02d}",
                text,
                f"{float(row['ocr_confidence'] or 0):.2f}"
                if row["plate_text"] else "-",
                ("matches format" if row["validated"]
                 else "unvalidated" if row["plate_text"] else "-"),
                "MATCH" if row["watchlist_hit"] else "-",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFont(mono_font(10))
                if column == 3 and not row["plate_text"]:
                    item.setForeground(Palette.qcolor(Palette.TEXT_DIM))
                elif column == 3:
                    item.setForeground(Palette.qcolor(Palette.TEXT))
                if column == 6 and row["watchlist_hit"]:
                    item.setForeground(Palette.qcolor(Palette.CRITICAL))
                self.reads_table.setItem(index, column, item)

        self.anpr_status.setText(
            f"ANPR: {self.ctx.anpr_status or 'not started'}.  "
            f"Unreadable plates keep their cropped image in the evidence "
            f"folder so a human can check them. Plate text is never guessed."
        )

    # ------------------------------------------------------------------

    def _add_entry(self) -> None:
        value = self.input_value.text().strip()
        if not value:
            return
        db = self.ctx.db
        if db is None:
            return
        db.add_watchlist_entry(
            value=value,
            kind=self.input_kind.currentText(),
            label=self.input_label.text().strip() or None,
            severity=self.input_severity.currentText(),
        )
        self.input_value.clear()
        self.input_label.clear()
        self.ctx.watchlist_changed()
        self.refresh()

    def _selected_entry_id(self) -> Optional[int]:
        items = self.entries_table.selectedItems()
        if not items:
            return None
        cell = self.entries_table.item(items[0].row(), 0)
        if cell is None:
            return None
        value = cell.data(Qt.UserRole)
        return int(value) if value is not None else None

    def _toggle_entry(self) -> None:
        entry_id = self._selected_entry_id()
        if entry_id is None or self.ctx.db is None:
            return
        row = next((r for r in self.ctx.db.get_watchlist()
                    if int(r["id"]) == entry_id), None)
        if row is None:
            return
        self.ctx.db.set_watchlist_active(entry_id, not bool(row["active"]))
        self.ctx.watchlist_changed()
        self.refresh()

    def _remove_entry(self) -> None:
        entry_id = self._selected_entry_id()
        if entry_id is None or self.ctx.db is None:
            return
        if QMessageBox.question(
            self, "Remove entry", "Remove this watchlist entry?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self.ctx.db.remove_watchlist_entry(entry_id)
        self.ctx.watchlist_changed()
        self.refresh()
