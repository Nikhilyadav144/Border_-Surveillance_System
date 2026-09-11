"""Dashboard: the at-a-glance state of the whole system.

Every number on this screen is read from SQLite or from the live worker. There
is no demo data and no placeholder: with an empty database the counters read
zero and the panels say so, which is the honest state of a system that has not
processed anything yet.
"""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.pages.base import Page
from app.gui.theme import Palette, mono_font
from app.gui.widgets.common import (
    Badge,
    BarChart,
    Card,
    KpiCard,
    Sparkline,
    StatRow,
    StatusDot,
)
from database.models import IncidentType, ThreatLevel


class DashboardPage(Page):
    title = "Dashboard"
    subtitle = "System overview and current operational picture"

    def build(self) -> None:
        # -- KPI row -------------------------------------------------------
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(10)

        self.kpi_cameras = KpiCard("Active Cameras", "0", "of 0 configured")
        self.kpi_people = KpiCard("People Detected", "0", "tracked identities",
                                  Palette.PERSON)
        self.kpi_vehicles = KpiCard("Vehicles Detected", "0",
                                    "tracked identities", Palette.VEHICLE)
        self.kpi_alerts = KpiCard("Active Alerts", "0", "awaiting action",
                                  Palette.HIGH)
        self.kpi_incidents = KpiCard("Open Incidents", "0", "not yet resolved",
                                     Palette.MEDIUM)
        self.kpi_threat = KpiCard("Threat Level", "LOW", "highest open",
                                  Palette.LOW)

        for card in (self.kpi_cameras, self.kpi_people, self.kpi_vehicles,
                     self.kpi_alerts, self.kpi_incidents, self.kpi_threat):
            kpi_row.addWidget(card)
        self.content.addLayout(kpi_row)

        # -- middle band ---------------------------------------------------
        middle = QHBoxLayout()
        middle.setSpacing(12)

        # Recent incidents
        self.incidents_card = Card("Recent Incidents")
        self.incident_table = QTableWidget(0, 6)
        self.incident_table.setHorizontalHeaderLabels(
            ["TIME", "CAMERA", "TYPE", "OBJECT", "SCORE", "LEVEL"]
        )
        self.incident_table.verticalHeader().setVisible(False)
        self.incident_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.incident_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.incident_table.setAlternatingRowColors(True)
        # Explicit widths beat ResizeToContents here: the incident type is the
        # column an operator actually reads, and content-sizing squeezes it.
        header = self.incident_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        for column, width in ((0, 86), (1, 78), (3, 116), (4, 62), (5, 100)):
            self.incident_table.setColumnWidth(column, width)
        self.incidents_card.add(self.incident_table, 1)
        middle.addWidget(self.incidents_card, 3)

        # Camera health
        self.health_card = Card("Camera Health")
        self.health_container = QVBoxLayout()
        self.health_container.setSpacing(7)
        self.health_card.add_layout(self.health_container)
        self.health_card.body().addStretch(1)
        middle.addWidget(self.health_card, 2)

        # System status
        self.status_card = Card("System Status")
        self.row_model = StatRow("Detection model", "-")
        self.row_device = StatRow("Inference device", "-")
        self.row_classes = StatRow("Active classes", "-")
        self.row_anomaly = StatRow("Anomaly engine", "-")
        self.row_anpr = StatRow("ANPR", "-")
        self.row_watchlist = StatRow("Watchlist entries", "0")
        self.row_evidence = StatRow("Evidence files", "0")
        self.row_database = StatRow("Database", "-")
        for row in (self.row_model, self.row_device, self.row_classes,
                    self.row_anomaly, self.row_anpr, self.row_watchlist,
                    self.row_evidence, self.row_database):
            self.status_card.add(row)
        self.status_card.body().addStretch(1)
        middle.addWidget(self.status_card, 2)

        self.content.addLayout(middle, 3)

        # -- bottom band ---------------------------------------------------
        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        self.trend_card = Card("Incidents Per Day")
        self.trend_chart = Sparkline(colour=Palette.ACCENT)
        self.trend_card.add(self.trend_chart, 1)
        bottom.addWidget(self.trend_card, 2)

        self.type_card = Card("Incidents By Type")
        self.type_chart = BarChart(accent=Palette.ACCENT)
        self.type_card.add(self.type_chart, 1)
        bottom.addWidget(self.type_card, 2)

        self.level_card = Card("Threat Distribution")
        self.level_chart = BarChart()
        self.level_card.add(self.level_chart, 1)
        bottom.addWidget(self.level_card, 2)

        self.content.addLayout(bottom, 2)

        self._health_widgets: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------

    def refresh(self) -> None:
        db = self.ctx.db
        if db is None:
            return

        counts = db.analytics_counts()
        cameras = db.get_cameras()
        online = sum(1 for c in cameras if c["status"] == "ONLINE")

        self.kpi_cameras.set_value(
            online, f"of {len(cameras)} configured",
            Palette.ONLINE if online else Palette.TEXT_DIM,
        )
        self.kpi_people.set_value(counts["people_tracked"])
        self.kpi_vehicles.set_value(counts["vehicles_tracked"])
        self.kpi_alerts.set_value(
            counts["active_alerts"],
            accent=Palette.HIGH if counts["active_alerts"] else Palette.TEXT_DIM,
        )
        self.kpi_incidents.set_value(
            counts["open_incidents"],
            accent=Palette.MEDIUM if counts["open_incidents"]
            else Palette.TEXT_DIM,
        )
        level = db.highest_open_threat()
        self.kpi_threat.set_value(level, accent=Palette.threat(level))

        self._refresh_incidents(db)
        self._refresh_health(cameras)
        self._refresh_status(counts, db)
        self._refresh_charts(db)

    # -- sections ---------------------------------------------------------

    def _refresh_incidents(self, db) -> None:
        rows = db.get_incidents(limit=12)
        table = self.incident_table
        table.setRowCount(len(rows))

        for index, row in enumerate(rows):
            stamp = time.strftime("%H:%M:%S", time.localtime(float(row["ts"])))
            values = [
                stamp,
                row["camera_id"],
                IncidentType.label(row["incident_type"]),
                (f"{str(row['object_class'] or '').upper()} "
                 f"#{int(row['track_number'] or 0):02d}"
                 if row["track_number"] is not None else "-"),
                f"{float(row['threat_score']):.0f}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                # Narrow columns elide; the tooltip keeps the full text
                # reachable without widening the table.
                item.setToolTip(str(value))
                if column in (0, 4):
                    item.setFont(mono_font(10))
                if column == 0:
                    item.setForeground(Palette.qcolor(Palette.TEXT_MUTED))
                table.setItem(index, column, item)

            badge = Badge()
            badge.set_threat(row["threat_level"])
            holder = QWidget()
            layout = QHBoxLayout(holder)
            layout.setContentsMargins(4, 2, 4, 2)
            layout.addWidget(badge)
            table.setCellWidget(index, 5, holder)

        if not rows:
            table.setRowCount(1)
            item = QTableWidgetItem("No incidents recorded yet")
            item.setForeground(Palette.qcolor(Palette.TEXT_DIM))
            table.setItem(0, 0, item)
            table.setSpan(0, 0, 1, 6)

    def _refresh_health(self, cameras) -> None:
        while self.health_container.count():
            item = self.health_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._health_widgets.clear()

        if not cameras:
            empty = QLabel("No cameras configured. Add one on the Cameras "
                           "screen or in config/cameras.yaml.")
            empty.setObjectName("Dim")
            empty.setWordWrap(True)
            self.health_container.addWidget(empty)
            return

        for camera in cameras:
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 2, 0, 2)
            layout.setSpacing(8)

            status = str(camera["status"])
            dot = StatusDot(Palette.status(status))
            name = QLabel(f"{camera['id']}")
            name.setFont(mono_font(10, bold=True))
            name_text = str(camera["name"])
            label = QLabel(name_text if len(name_text) <= 24
                           else name_text[:23] + "\u2026")
            label.setObjectName("Dim")

            detail = QLabel(self._health_detail(camera))
            detail.setFont(mono_font(9))
            detail.setStyleSheet(f"color: {Palette.status(status)};")
            detail.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            layout.addWidget(dot)
            layout.addWidget(name)
            layout.addWidget(label, 1)
            layout.addWidget(detail)
            self.health_container.addWidget(row)

    @staticmethod
    def _health_detail(camera) -> str:
        status = str(camera["status"])
        if status == "ONLINE":
            return (f"{float(camera['fps'] or 0):.0f} FPS  "
                    f"{float(camera['latency_ms'] or 0):.0f} ms")
        if status == "ERROR":
            return "ERROR"
        if camera["last_frame_ts"]:
            return time.strftime("last %H:%M:%S",
                                 time.localtime(float(camera["last_frame_ts"])))
        return status

    def _refresh_status(self, counts: dict, db) -> None:
        # When nothing is running we still report what IS configured, marked
        # as configured rather than loaded. Silence would look like a fault.
        info = self.ctx.detector_info or {}
        cfg = self.ctx.config
        if info:
            self.row_model.set_value(info.get("model", "-"), Palette.ONLINE)
            self.row_device.set_value(info.get("device_label", "-"),
                                      Palette.ONLINE)
            classes = info.get("active_classes") or []
        else:
            model_path = cfg.model_path()
            self.row_model.set_value(
                f"{model_path.name} (configured)",
                Palette.TEXT_MUTED if model_path.exists() else Palette.ERROR,
            )
            from core.detector import detect_device
            device = detect_device(str(cfg.get("detection.device", "auto")))
            self.row_device.set_value(f"{device.label} (available)",
                                      Palette.TEXT_MUTED)
            classes = list(cfg.get("detection.classes", []) or [])
        self.row_classes.set_value(
            (f"{len(classes)}: " + ", ".join(classes[:4]) +
             ("\u2026" if len(classes) > 4 else "")) if classes else "all"
        )
        self.row_anomaly.set_value(self.ctx.anomaly_status or "not started")
        self.row_anpr.set_value(self.ctx.anpr_status or "not started")
        self.row_watchlist.set_value(len(db.get_watchlist(active_only=True)))
        self.row_evidence.set_value(counts["evidence_files"])
        self.row_database.set_value(self.ctx.config.path("database_file").name,
                                    Palette.ONLINE)

    def _refresh_charts(self, db) -> None:
        self.trend_chart.set_data(db.analytics_incidents_by_day(days=14))
        self.type_chart.set_data([
            (IncidentType.label(name), count)
            for name, count in db.analytics_by_type()[:8]
        ])
        levels = db.analytics_by_level()
        self.level_chart.set_data([
            (level, levels.get(level, 0), Palette.threat(level))
            for level in ThreatLevel.ORDER
        ])
