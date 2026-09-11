"""Incidents: the operator's work queue.

A filterable table on the left, a detail panel on the right showing the full
threat-score breakdown and the evidence images. Acknowledge and Resolve move
an incident through its lifecycle and write straight to SQLite.

The score breakdown is the important part of this screen. An operator should
never see a number without being able to ask why, so every scoring factor that
fired is listed with the points it contributed, along with the standing
disclaimer that this is a configurable prototype model.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.gui.pages.base import Page
from app.gui.theme import Palette, mono_font, ui_font
from app.gui.widgets.common import Badge, Card, StatRow, divider
from core.threat_engine import DISCLAIMER
from database.models import Incident, IncidentStatus, IncidentType, ThreatLevel


class IncidentsPage(Page):
    title = "Incidents"
    subtitle = "Detected events, threat breakdown and operator workflow"

    def build(self) -> None:
        self._current_id: Optional[int] = None

        # -- filters -------------------------------------------------------
        filters = QHBoxLayout()
        filters.setSpacing(8)

        self.filter_status = QComboBox()
        self.filter_status.addItems(["All statuses", "NEW", "ACKNOWLEDGED",
                                     "RESOLVED"])
        self.filter_level = QComboBox()
        self.filter_level.addItems(["All levels", *ThreatLevel.ORDER])
        self.filter_type = QComboBox()
        self.filter_type.addItem("All types", None)
        for key, label in IncidentType.LABELS.items():
            self.filter_type.addItem(label, key)
        self.filter_camera = QComboBox()
        self.filter_camera.addItem("All cameras", None)

        for widget in (self.filter_status, self.filter_level,
                       self.filter_type, self.filter_camera):
            widget.currentIndexChanged.connect(self.refresh)
            filters.addWidget(widget)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh)
        filters.addStretch(1)
        filters.addWidget(refresh_button)
        self.content.addLayout(filters)

        # -- split view ----------------------------------------------------
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_table())
        splitter.addWidget(self._build_detail())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([900, 470])
        self.content.addWidget(splitter, 1)

    # ------------------------------------------------------------------

    def _build_table(self) -> QWidget:
        card = Card("Incident Log")
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["ID", "TIME", "CAMERA", "TYPE", "OBJECT", "ZONE", "SCORE", "LEVEL"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_selection)
        # Fixed widths for the narrow columns, and let ZONE take the slack.
        # The incident type is what an operator reads first, so it gets enough
        # room to be read rather than elided to "Virtual...".
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        for column, width in ((0, 72), (1, 94), (2, 78), (3, 180),
                              (4, 120), (6, 64), (7, 102)):
            self.table.setColumnWidth(column, width)
        card.add(self.table, 1)
        return card

    def _build_detail(self) -> QWidget:
        card = Card("Incident Detail")

        self.detail_title = QLabel("Select an incident")
        self.detail_title.setFont(ui_font(14, bold=True))
        self.detail_badge = Badge("-", Palette.TEXT_DIM)

        title_row = QHBoxLayout()
        title_row.addWidget(self.detail_title, 1)
        title_row.addWidget(self.detail_badge)
        card.add_layout(title_row)
        card.add(divider())

        self.row_camera = StatRow("Camera", "-")
        self.row_time = StatRow("Time", "-")
        self.row_object = StatRow("Object", "-")
        self.row_zone = StatRow("Zone", "-")
        self.row_score = StatRow("Threat score", "-")
        self.row_status = StatRow("Status", "-")
        self.row_escalations = StatRow("Escalations", "0")
        for row in (self.row_camera, self.row_time, self.row_object,
                    self.row_zone, self.row_score, self.row_status,
                    self.row_escalations):
            card.add(row)

        card.add(divider())
        card.add(QLabel("SCORE BREAKDOWN"))
        card.body().itemAt(card.body().count() - 1).widget().setObjectName(
            "SectionTitle")

        self.reasons_box = QTextEdit()
        self.reasons_box.setReadOnly(True)
        self.reasons_box.setFont(mono_font(10))
        self.reasons_box.setMaximumHeight(150)
        self.reasons_box.setPlaceholderText(
            "Select an incident to see how its score was built up.")
        card.add(self.reasons_box)

        self.disclaimer = QLabel(DISCLAIMER)
        self.disclaimer.setWordWrap(True)
        self.disclaimer.setObjectName("Dim")
        card.add(self.disclaimer)

        card.add(divider())
        evidence_label = QLabel("EVIDENCE")
        evidence_label.setObjectName("SectionTitle")
        card.add(evidence_label)

        self.evidence_area = QScrollArea()
        self.evidence_area.setWidgetResizable(True)
        self.evidence_area.setFixedHeight(160)
        self.evidence_holder = QWidget()
        self.evidence_layout = QHBoxLayout(self.evidence_holder)
        self.evidence_layout.setContentsMargins(0, 0, 0, 0)
        self.evidence_layout.setSpacing(8)
        self.evidence_area.setWidget(self.evidence_holder)
        card.add(self.evidence_area)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.btn_ack = QPushButton("Acknowledge")
        self.btn_ack.setObjectName("Primary")
        self.btn_ack.setEnabled(False)
        self.btn_ack.clicked.connect(self._acknowledge)
        self.btn_resolve = QPushButton("Resolve")
        self.btn_resolve.setEnabled(False)
        self.btn_resolve.clicked.connect(self._resolve)
        actions.addWidget(self.btn_ack)
        actions.addWidget(self.btn_resolve)
        actions.addStretch(1)
        card.add_layout(actions)

        return card

    # ------------------------------------------------------------------

    def refresh(self) -> None:
        db = self.ctx.db
        if db is None:
            return

        # Keep the camera filter in step with what exists.
        known = {self.filter_camera.itemData(i)
                 for i in range(self.filter_camera.count())}
        for camera in db.get_cameras():
            if camera["id"] not in known:
                self.filter_camera.addItem(camera["id"], camera["id"])

        status = self.filter_status.currentText()
        level = self.filter_level.currentText()
        rows = db.get_incidents(
            status=None if status.startswith("All") else status,
            level=None if level.startswith("All") else level,
            incident_type=self.filter_type.currentData(),
            camera_id=self.filter_camera.currentData(),
            limit=500,
        )

        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = [
                f"#{int(row['id']):04d}",
                time.strftime("%H:%M:%S", time.localtime(float(row["ts"]))),
                row["camera_id"],
                IncidentType.label(row["incident_type"]),
                (f"{str(row['object_class'] or '').upper()} "
                 f"#{int(row['track_number'] or 0):02d}"
                 if row["track_number"] is not None else "-"),
                row["zone_name"] or "-",
                f"{float(row['threat_score']):.0f}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                # Narrow columns elide; the tooltip keeps the full text
                # reachable without widening the table.
                item.setToolTip(str(value))
                if column in (0, 1, 6):
                    item.setFont(mono_font(10))
                if column == 0:
                    item.setData(Qt.UserRole, int(row["id"]))
                if str(row["status"]) == IncidentStatus.RESOLVED:
                    item.setForeground(Palette.qcolor(Palette.TEXT_DIM))
                self.table.setItem(index, column, item)

            badge = Badge()
            badge.set_threat(row["threat_level"])
            holder = QWidget()
            layout = QHBoxLayout(holder)
            layout.setContentsMargins(4, 2, 4, 2)
            layout.addWidget(badge)
            self.table.setCellWidget(index, 7, holder)

        self.set_subtitle(
            f"{len(rows)} incidents shown  ·  "
            f"{db.count_incidents(IncidentStatus.NEW)} new  ·  "
            f"{db.count_incidents()} total"
        )

    # ------------------------------------------------------------------

    def _on_selection(self) -> None:
        items = self.table.selectedItems()
        if not items:
            return
        row = items[0].row()
        id_item = self.table.item(row, 0)
        if id_item is None:
            return
        incident_id = id_item.data(Qt.UserRole)
        if incident_id is None:
            return
        self.show_incident(int(incident_id))

    def show_incident(self, incident_id: int) -> None:
        db = self.ctx.db
        if db is None:
            return
        row = db.get_incident(incident_id)
        if row is None:
            return

        self._current_id = incident_id
        level = str(row["threat_level"])
        self.detail_title.setText(
            f"INCIDENT #{incident_id:04d}   "
            f"{IncidentType.label(row['incident_type'])}"
        )
        self.detail_badge.set_threat(level)

        self.row_camera.set_value(row["camera_id"])
        self.row_time.set_value(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(row["ts"])))
        )
        self.row_object.set_value(
            f"{str(row['object_class'] or '').upper()} "
            f"#{int(row['track_number'] or 0):02d}"
            if row["track_number"] is not None else "-"
        )
        self.row_zone.set_value(row["zone_name"] or "-")
        self.row_score.set_value(f"{float(row['threat_score']):.0f} / 100",
                                 Palette.threat(level))
        status = str(row["status"])
        self.row_status.set_value(status, {
            IncidentStatus.NEW: Palette.HIGH,
            IncidentStatus.ACKNOWLEDGED: Palette.MEDIUM,
            IncidentStatus.RESOLVED: Palette.LOW,
        }.get(status, Palette.TEXT))
        self.row_escalations.set_value(int(row["escalation_count"] or 0))

        self._render_reasons(row)
        self._render_evidence(db, incident_id)

        self.btn_ack.setEnabled(status == IncidentStatus.NEW)
        self.btn_resolve.setEnabled(status != IncidentStatus.RESOLVED)

    def _render_reasons(self, row) -> None:
        reasons = Incident.reasons_from_row(row["reasons"])
        lines = []
        if reasons:
            for reason in reasons:
                lines.append(f"  + {reason.points:>4.0f}   {reason.label}")
            lines.append("")
            lines.append(f"  = {float(row['threat_score']):>4.0f}   TOTAL "
                         f"({row['threat_level']})")
        else:
            lines.append("  No scoring factors fired for this incident.")

        detail = row["detail"]
        if detail:
            try:
                data = json.loads(detail)
                interesting = {
                    k: v for k, v in data.items()
                    if k in ("duration_text", "plate", "compass", "direction",
                             "ocr_confidence", "zone_kind", "watchlist_label")
                }
                if interesting:
                    lines.append("")
                    lines.append("  DETAIL")
                    for key, value in interesting.items():
                        lines.append(f"    {key:<18} {value}")
            except (TypeError, ValueError):
                pass

        self.reasons_box.setPlainText("\n".join(lines))

    def _render_evidence(self, db, incident_id: int) -> None:
        while self.evidence_layout.count():
            item = self.evidence_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        rows = db.get_evidence(incident_id=incident_id, limit=12)
        if not rows:
            label = QLabel("No evidence captured for this incident.\n"
                           "Evidence is only written at or above the level set "
                           "in Settings.")
            label.setObjectName("Dim")
            label.setWordWrap(True)
            self.evidence_layout.addWidget(label)
            return

        for row in rows:
            holder = QWidget()
            layout = QVBoxLayout(holder)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(3)

            thumb = QLabel()
            thumb.setFixedSize(190, 108)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setStyleSheet(
                f"background: #05070b; border: 1px solid {Palette.BORDER};"
                f"border-radius: 5px;"
            )
            pixmap = QPixmap(str(row["file_path"]))
            if pixmap.isNull():
                thumb.setText("file missing")
                thumb.setStyleSheet(
                    thumb.styleSheet() + f"color: {Palette.ERROR};")
            else:
                thumb.setPixmap(pixmap.scaled(190, 108, Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation))
            caption = QLabel(str(row["kind"]).replace("_", " "))
            caption.setObjectName("Dim")
            caption.setAlignment(Qt.AlignCenter)

            layout.addWidget(thumb)
            layout.addWidget(caption)
            self.evidence_layout.addWidget(holder)

        self.evidence_layout.addStretch(1)

    # ------------------------------------------------------------------

    def _acknowledge(self) -> None:
        if self._current_id is None:
            return
        self.ctx.acknowledge_incident(self._current_id)
        self.refresh()
        self.show_incident(self._current_id)

    def _resolve(self) -> None:
        if self._current_id is None:
            return
        confirm = QMessageBox.question(
            self, "Resolve incident",
            f"Mark incident #{self._current_id:04d} as resolved?\n\n"
            f"The record and its evidence are kept. Resolving stops the "
            f"incident being escalated by further activity from the same "
            f"track.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if confirm != QMessageBox.Yes:
            return
        self.ctx.resolve_incident(self._current_id)
        self.refresh()
        self.show_incident(self._current_id)
