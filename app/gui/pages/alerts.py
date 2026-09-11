"""Alerts: the live notification feed.

Alerts and incidents are one-to-one, but they serve different purposes. The
incident is the record; the alert is the interruption. This screen is the
running feed an operator watches, ordered newest first, with acknowledge and
resolve reachable without leaving it.
"""

from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.pages.base import Page
from app.gui.theme import Palette, mono_font
from app.gui.widgets.common import Badge, Card, KpiCard
from database.models import IncidentStatus, ThreatLevel


class AlertsPage(Page):
    title = "Alerts"
    subtitle = "Live notification feed and operator acknowledgement"

    open_incident = Signal(int)

    def build(self) -> None:
        kpis = QHBoxLayout()
        kpis.setSpacing(10)
        self.kpi_new = KpiCard("New", "0", "unacknowledged", Palette.CRITICAL)
        self.kpi_ack = KpiCard("Acknowledged", "0", "in progress", Palette.MEDIUM)
        self.kpi_resolved = KpiCard("Resolved", "0", "closed", Palette.LOW)
        self.kpi_total = KpiCard("Total", "0", "all time")
        for card in (self.kpi_new, self.kpi_ack, self.kpi_resolved,
                     self.kpi_total):
            kpis.addWidget(card)
        self.content.addLayout(kpis)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.filter_status = QComboBox()
        self.filter_status.addItems(["All statuses", "NEW", "ACKNOWLEDGED",
                                     "RESOLVED"])
        self.filter_status.currentIndexChanged.connect(self.refresh)
        self.filter_severity = QComboBox()
        self.filter_severity.addItems(["All severities", *ThreatLevel.ORDER])
        self.filter_severity.currentIndexChanged.connect(self.refresh)

        self.btn_ack = QPushButton("Acknowledge Selected")
        self.btn_ack.setObjectName("Primary")
        self.btn_ack.clicked.connect(self._acknowledge_selected)
        self.btn_resolve = QPushButton("Resolve Selected")
        self.btn_resolve.clicked.connect(self._resolve_selected)
        self.btn_ack_all = QPushButton("Acknowledge All New")
        self.btn_ack_all.clicked.connect(self._acknowledge_all)
        self.btn_open = QPushButton("Open Incident")
        self.btn_open.clicked.connect(self._open_selected)

        controls.addWidget(self.filter_status)
        controls.addWidget(self.filter_severity)
        controls.addStretch(1)
        controls.addWidget(self.btn_open)
        controls.addWidget(self.btn_ack)
        controls.addWidget(self.btn_resolve)
        controls.addWidget(self.btn_ack_all)
        self.content.addLayout(controls)

        card = Card("Alert Feed")
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["ID", "TIME", "SEVERITY", "CAMERA", "OBJECT", "MESSAGE", "STATUS"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.doubleClicked.connect(self._open_selected)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        for column, width in ((0, 72), (1, 94), (2, 102), (3, 82),
                              (4, 122), (6, 128)):
            self.table.setColumnWidth(column, width)
        card.add(self.table, 1)
        self.content.addWidget(card, 1)

    # ------------------------------------------------------------------

    def refresh(self) -> None:
        db = self.ctx.db
        if db is None:
            return

        status = self.filter_status.currentText()
        severity = self.filter_severity.currentText()
        rows = db.get_alerts(
            status=None if status.startswith("All") else status,
            severity=None if severity.startswith("All") else severity,
            limit=400,
        )

        all_alerts = db.get_alerts(limit=5000)
        by_status: dict[str, int] = {}
        for alert in all_alerts:
            key = str(alert["status"])
            by_status[key] = by_status.get(key, 0) + 1

        self.kpi_new.set_value(by_status.get(IncidentStatus.NEW, 0))
        self.kpi_ack.set_value(by_status.get(IncidentStatus.ACKNOWLEDGED, 0))
        self.kpi_resolved.set_value(by_status.get(IncidentStatus.RESOLVED, 0))
        self.kpi_total.set_value(len(all_alerts))

        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = [
                f"#{int(row['id']):04d}",
                time.strftime("%H:%M:%S", time.localtime(float(row["ts"]))),
                "",
                row["camera_id"],
                row["object_label"] or "-",
                row["message"],
                row["status"],
            ]
            for column, value in enumerate(values):
                if column == 2:
                    continue
                item = QTableWidgetItem(str(value))
                # Narrow columns elide; the tooltip keeps the full text
                # reachable without widening the table.
                item.setToolTip(str(value))
                if column in (0, 1):
                    item.setFont(mono_font(10))
                if column == 0:
                    item.setData(Qt.UserRole, int(row["incident_id"]))
                if column == 6:
                    item.setForeground(Palette.qcolor({
                        IncidentStatus.NEW: Palette.HIGH,
                        IncidentStatus.ACKNOWLEDGED: Palette.MEDIUM,
                        IncidentStatus.RESOLVED: Palette.TEXT_DIM,
                    }.get(str(value), Palette.TEXT)))
                self.table.setItem(index, column, item)

            badge = Badge()
            badge.set_threat(row["severity"])
            holder = QWidget()
            layout = QHBoxLayout(holder)
            layout.setContentsMargins(4, 2, 4, 2)
            layout.addWidget(badge)
            self.table.setCellWidget(index, 2, holder)

        self.set_subtitle(
            f"{by_status.get(IncidentStatus.NEW, 0)} alerts awaiting "
            f"acknowledgement  ·  {len(all_alerts)} total"
        )

    # ------------------------------------------------------------------

    def _selected_incident(self) -> Optional[int]:
        items = self.table.selectedItems()
        if not items:
            return None
        cell = self.table.item(items[0].row(), 0)
        if cell is None:
            return None
        value = cell.data(Qt.UserRole)
        return int(value) if value is not None else None

    def _acknowledge_selected(self) -> None:
        incident_id = self._selected_incident()
        if incident_id is not None:
            self.ctx.acknowledge_incident(incident_id)
            self.refresh()

    def _resolve_selected(self) -> None:
        incident_id = self._selected_incident()
        if incident_id is not None:
            self.ctx.resolve_incident(incident_id)
            self.refresh()

    def _acknowledge_all(self) -> None:
        db = self.ctx.db
        if db is None:
            return
        for alert in db.get_alerts(status=IncidentStatus.NEW, limit=1000):
            self.ctx.acknowledge_incident(int(alert["incident_id"]))
        self.refresh()

    def _open_selected(self) -> None:
        incident_id = self._selected_incident()
        if incident_id is not None:
            self.open_incident.emit(incident_id)
