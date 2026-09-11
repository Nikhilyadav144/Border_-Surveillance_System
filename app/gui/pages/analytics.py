"""Analytics: aggregate views over what has actually been recorded.

Every chart on this screen is a SQL query against the local database. If the
database is empty the charts say so. There are no illustrative numbers, no
sample series and no placeholder percentages anywhere on this page.
"""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout

from app.gui.pages.base import Page
from app.gui.theme import Palette
from app.gui.widgets.common import (
    BarChart,
    Card,
    ColumnChart,
    KpiCard,
    Sparkline,
    StatRow,
)
from database.models import IncidentType, ThreatLevel


class AnalyticsPage(Page):
    title = "Analytics"
    subtitle = "Aggregates computed from the local incident database"

    def build(self) -> None:
        kpis = QHBoxLayout()
        kpis.setSpacing(10)
        self.kpi_incidents = KpiCard("Total Incidents", "0", "all time")
        self.kpi_intrusions = KpiCard("Zone Intrusions", "0",
                                      "restricted and border", Palette.CRITICAL)
        self.kpi_loitering = KpiCard("Loitering Events", "0",
                                     "temporal detections", Palette.HIGH)
        self.kpi_crossings = KpiCard("Line Crossings", "0",
                                     "virtual boundaries", Palette.MEDIUM)
        self.kpi_vehicles = KpiCard("Vehicle Alerts", "0",
                                    "no-vehicle zone", Palette.VEHICLE)
        self.kpi_watchlist = KpiCard("Watchlist Matches", "0",
                                     "confirmed", Palette.CRITICAL)
        for card in (self.kpi_incidents, self.kpi_intrusions,
                     self.kpi_loitering, self.kpi_crossings,
                     self.kpi_vehicles, self.kpi_watchlist):
            kpis.addWidget(card)
        self.content.addLayout(kpis)

        grid = QGridLayout()
        grid.setSpacing(12)

        self.card_daily = Card("Incidents Per Day (last 14 days)")
        self.chart_daily = Sparkline(colour=Palette.ACCENT)
        self.card_daily.add(self.chart_daily, 1)
        grid.addWidget(self.card_daily, 0, 0, 1, 2)

        self.card_hourly = Card("Hourly Distribution")
        self.chart_hourly = ColumnChart(colour=Palette.ACCENT)
        self.card_hourly.add(self.chart_hourly, 1)
        grid.addWidget(self.card_hourly, 0, 2)

        self.card_camera = Card("Incidents By Camera")
        self.chart_camera = BarChart()
        self.card_camera.add(self.chart_camera, 1)
        grid.addWidget(self.card_camera, 1, 0)

        self.card_type = Card("Incidents By Type")
        self.chart_type = BarChart()
        self.card_type.add(self.chart_type, 1)
        grid.addWidget(self.card_type, 1, 1)

        self.card_level = Card("Threat Level Distribution")
        self.chart_level = BarChart()
        self.card_level.add(self.chart_level, 1)
        grid.addWidget(self.card_level, 1, 2)

        self.card_summary = Card("Summary")
        self.row_busiest = StatRow("Busiest camera", "-")
        self.row_tracks = StatRow("Tracked identities", "0")
        self.row_people = StatRow("People tracked", "0")
        self.row_vehicles = StatRow("Vehicles tracked", "0")
        self.row_detections = StatRow("Detection samples stored", "0")
        self.row_evidence = StatRow("Evidence files", "0")
        self.row_plates = StatRow("Plates read", "0")
        self.row_sessions = StatRow("Processing sessions", "0")
        for row in (self.row_busiest, self.row_tracks, self.row_people,
                    self.row_vehicles, self.row_detections, self.row_evidence,
                    self.row_plates, self.row_sessions):
            self.card_summary.add(row)
        self.card_summary.body().addStretch(1)
        grid.addWidget(self.card_summary, 2, 0, 1, 3)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        self.content.addLayout(grid, 1)

        self.empty_note = QLabel(
            "No incidents have been recorded yet. Process a video on the "
            "Cameras screen and these charts will populate from the database."
        )
        self.empty_note.setObjectName("Dim")
        self.empty_note.setWordWrap(True)
        self.content.addWidget(self.empty_note)

    # ------------------------------------------------------------------

    def refresh(self) -> None:
        db = self.ctx.db
        if db is None:
            return

        counts = db.analytics_counts()
        self.kpi_incidents.set_value(counts["incidents"])
        self.kpi_intrusions.set_value(counts["intrusions"])
        self.kpi_loitering.set_value(counts["loitering"])
        self.kpi_crossings.set_value(counts["line_crossings"])
        self.kpi_vehicles.set_value(counts["vehicle_alerts"])
        self.kpi_watchlist.set_value(counts["watchlist_matches"])

        self.chart_daily.set_data(db.analytics_incidents_by_day(days=14))
        self.chart_hourly.set_data([
            (f"{hour:02d}", value) for hour, value in db.analytics_hourly()
        ])
        self.chart_camera.set_data(db.analytics_incidents_by_camera())
        self.chart_type.set_data([
            (IncidentType.label(name), value)
            for name, value in db.analytics_by_type()
        ])
        levels = db.analytics_by_level()
        self.chart_level.set_data([
            (level, levels.get(level, 0), Palette.threat(level))
            for level in ThreatLevel.ORDER
        ])

        busiest = db.analytics_busiest_camera()
        self.row_busiest.set_value(
            f"{busiest[0]} ({busiest[1]} incidents)" if busiest else "-"
        )
        self.row_tracks.set_value(counts["tracks"])
        self.row_people.set_value(counts["people_tracked"])
        self.row_vehicles.set_value(counts["vehicles_tracked"])
        self.row_detections.set_value(db.count_detections())
        self.row_evidence.set_value(counts["evidence_files"])
        self.row_plates.set_value(counts["plates_read"])
        self.row_sessions.set_value(len(db.get_sessions(limit=1000)))

        self.empty_note.setVisible(counts["incidents"] == 0)
        self.set_subtitle(
            f"{counts['incidents']} incidents across "
            f"{len(db.analytics_incidents_by_camera())} cameras"
            if counts["incidents"] else
            "Aggregates computed from the local incident database"
        )
