"""Evidence: browse and review captured images.

A filterable gallery backed by the ``evidence`` table. Clicking a thumbnail
opens the full image with the metadata of the incident it belongs to.

Files are never deleted from here except through the explicit retention sweep,
and the sweep reports exactly what it removed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.gui.pages.base import Page
from app.gui.theme import Palette, mono_font
from app.gui.widgets.common import Badge, Card, KpiCard, StatRow
from database.models import Incident, IncidentType, ThreatLevel


class EvidenceThumb(QWidget):
    """One clickable evidence tile."""

    def __init__(self, row, on_click, parent=None) -> None:
        super().__init__(parent)
        self.row = row
        self._on_click = on_click

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self.image = QLabel()
        self.image.setFixedSize(208, 118)
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setStyleSheet(
            f"background: #05070b; border: 1px solid {Palette.BORDER};"
            f"border-radius: 6px;"
        )
        path = Path(str(row["file_path"]))
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.image.setText("file missing")
            self.image.setStyleSheet(
                self.image.styleSheet() + f"color: {Palette.ERROR};"
            )
        else:
            self.image.setPixmap(pixmap.scaled(208, 118, Qt.KeepAspectRatio,
                                               Qt.SmoothTransformation))

        caption = QLabel(
            f"{time.strftime('%H:%M:%S', time.localtime(float(row['ts'])))}  "
            f"{row['camera_id']}  {str(row['kind']).replace('_', ' ')}"
        )
        caption.setObjectName("Dim")
        caption.setFont(mono_font(8))

        layout.addWidget(self.image)
        layout.addWidget(caption)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._on_click(self.row)
        super().mousePressEvent(event)


class EvidenceViewer(QDialog):
    """Full-size view of one evidence image."""

    def __init__(self, row, incident_row, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Evidence - {Path(str(row['file_path'])).name}")
        self.resize(1080, 760)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel(Path(str(row["file_path"])).name)
        title.setFont(mono_font(11, bold=True))
        header.addWidget(title, 1)
        if incident_row is not None:
            badge = Badge()
            badge.set_threat(incident_row["threat_level"])
            header.addWidget(badge)
        layout.addLayout(header)

        image = QLabel()
        image.setAlignment(Qt.AlignCenter)
        image.setStyleSheet("background: #05070b; border-radius: 6px;")
        pixmap = QPixmap(str(row["file_path"]))
        if pixmap.isNull():
            image.setText("The image file is no longer on disk.")
            image.setStyleSheet(image.styleSheet() + f"color: {Palette.ERROR};")
        else:
            image.setPixmap(pixmap.scaled(1040, 580, Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation))
        layout.addWidget(image, 1)

        meta = QHBoxLayout()
        meta.setSpacing(16)
        left = QVBoxLayout()
        left.addWidget(StatRow("Camera", str(row["camera_id"])))
        left.addWidget(StatRow("Captured", time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(float(row["ts"])))))
        left.addWidget(StatRow("Kind", str(row["kind"])))
        left.addWidget(StatRow("Size", f"{int(row['bytes'] or 0) / 1024:.0f} KB"
                                       f"  ({row['width']}x{row['height']})"))
        meta.addLayout(left, 1)

        right = QVBoxLayout()
        if incident_row is not None:
            right.addWidget(StatRow(
                "Incident", f"#{int(incident_row['id']):04d}"))
            right.addWidget(StatRow(
                "Type", IncidentType.label(incident_row["incident_type"])))
            right.addWidget(StatRow(
                "Score", f"{float(incident_row['threat_score']):.0f} / 100",
                Palette.threat(incident_row["threat_level"])))
            right.addWidget(StatRow("Status", str(incident_row["status"])))
        else:
            note = QLabel("Not linked to an incident. Plate crops are stored "
                          "for every ANPR attempt, including failed reads.")
            note.setObjectName("Dim")
            note.setWordWrap(True)
            right.addWidget(note)
        meta.addLayout(right, 1)
        layout.addLayout(meta)

        path_label = QLabel(str(row["file_path"]))
        path_label.setObjectName("Dim")
        path_label.setFont(mono_font(9))
        path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(path_label)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)


class EvidencePage(Page):
    title = "Evidence"
    subtitle = "Captured images, indexed and filterable"

    def build(self) -> None:
        kpis = QHBoxLayout()
        kpis.setSpacing(10)
        self.kpi_files = KpiCard("Evidence Files", "0", "on disk")
        self.kpi_size = KpiCard("Storage Used", "0 MB", "in data/evidence")
        self.kpi_incidents = KpiCard("Incidents With Evidence", "0", "")
        self.kpi_retention = KpiCard("Retention", "-", "configured in Settings")
        for card in (self.kpi_files, self.kpi_size, self.kpi_incidents,
                     self.kpi_retention):
            kpis.addWidget(card)
        self.content.addLayout(kpis)

        filters = QHBoxLayout()
        filters.setSpacing(8)
        self.filter_camera = QComboBox()
        self.filter_camera.addItem("All cameras", None)
        self.filter_kind = QComboBox()
        self.filter_kind.addItem("All kinds", None)
        for kind in ("full_frame", "crop", "plate"):
            self.filter_kind.addItem(kind.replace("_", " "), kind)
        self.filter_level = QComboBox()
        self.filter_level.addItem("All levels", None)
        for level in ThreatLevel.ORDER:
            self.filter_level.addItem(level, level)
        self.filter_days = QComboBox()
        for label, days in (("All time", 0), ("Today", 1), ("Last 7 days", 7),
                            ("Last 30 days", 30)):
            self.filter_days.addItem(label, days)
        for widget in (self.filter_camera, self.filter_kind,
                       self.filter_level, self.filter_days):
            widget.currentIndexChanged.connect(self.refresh)
            filters.addWidget(widget)

        filters.addStretch(1)
        btn_purge = QPushButton("Run Retention Sweep")
        btn_purge.clicked.connect(self._purge)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh)
        filters.addWidget(btn_purge)
        filters.addWidget(btn_refresh)
        self.content.addLayout(filters)

        card = Card("Gallery")
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.gallery_holder = QWidget()
        self.gallery = QGridLayout(self.gallery_holder)
        self.gallery.setContentsMargins(4, 4, 4, 4)
        self.gallery.setSpacing(10)
        self.gallery.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.gallery_holder)
        card.add(self.scroll, 1)
        self.content.addWidget(card, 1)

    # ------------------------------------------------------------------

    def refresh(self) -> None:
        db = self.ctx.db
        if db is None:
            return

        known = {self.filter_camera.itemData(i)
                 for i in range(self.filter_camera.count())}
        for camera in db.get_cameras():
            if camera["id"] not in known:
                self.filter_camera.addItem(camera["id"], camera["id"])

        days = int(self.filter_days.currentData() or 0)
        date_from = time.time() - days * 86400 if days else None

        rows = db.get_evidence(
            camera_id=self.filter_camera.currentData(),
            date_from=date_from,
            limit=400,
        )

        kind = self.filter_kind.currentData()
        if kind:
            rows = [r for r in rows if r["kind"] == kind]

        level = self.filter_level.currentData()
        incident_cache: dict[int, object] = {}
        if level:
            filtered = []
            for row in rows:
                incident_id = row["incident_id"]
                if incident_id is None:
                    continue
                incident = incident_cache.get(int(incident_id))
                if incident is None:
                    incident = db.get_incident(int(incident_id))
                    incident_cache[int(incident_id)] = incident
                if incident is not None and incident["threat_level"] == level:
                    filtered.append(row)
            rows = filtered

        all_rows = db.get_evidence(limit=10000)
        total_bytes = sum(int(r["bytes"] or 0) for r in all_rows)
        with_incident = len({r["incident_id"] for r in all_rows
                             if r["incident_id"] is not None})

        self.kpi_files.set_value(len(all_rows), f"{len(rows)} shown")
        self.kpi_size.set_value(f"{total_bytes / 1_048_576:.1f} MB")
        self.kpi_incidents.set_value(with_incident)
        retention = int(self.ctx.config.get("evidence.retention_days", 30))
        self.kpi_retention.set_value(
            f"{retention}d" if retention > 0 else "keep all",
            "files older than this are removed by the sweep"
            if retention > 0 else "nothing is deleted automatically",
        )

        self._render_gallery(rows)
        self.set_subtitle(
            f"{len(rows)} files shown of {len(all_rows)} captured  ·  "
            f"{self.ctx.config.path('evidence_dir')}"
        )

    def _render_gallery(self, rows) -> None:
        while self.gallery.count():
            item = self.gallery.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not rows:
            empty = QLabel(
                "No evidence matches these filters.\n\n"
                "Evidence is captured automatically when an incident is opened "
                "at or above the level set in Settings, and for every ANPR "
                "attempt."
            )
            empty.setObjectName("Dim")
            empty.setWordWrap(True)
            self.gallery.addWidget(empty, 0, 0)
            return

        columns = 5
        for index, row in enumerate(rows):
            thumb = EvidenceThumb(row, self._open_viewer)
            self.gallery.addWidget(thumb, index // columns, index % columns)

    def _open_viewer(self, row) -> None:
        db = self.ctx.db
        incident = None
        if db is not None and row["incident_id"] is not None:
            incident = db.get_incident(int(row["incident_id"]))
        EvidenceViewer(row, incident, self).exec()

    def _purge(self) -> None:
        db = self.ctx.db
        if db is None:
            return
        retention = int(self.ctx.config.get("evidence.retention_days", 30))
        if retention <= 0:
            QMessageBox.information(
                self, "Retention disabled",
                "evidence.retention_days is 0, so nothing is ever deleted "
                "automatically. Change it in Settings to enable the sweep.",
            )
            return

        cutoff = time.time() - retention * 86400
        paths = db.delete_evidence_before(cutoff)
        removed = self.ctx.purge_evidence(paths)
        QMessageBox.information(
            self, "Retention sweep complete",
            f"Removed {removed} file(s) older than {retention} days.\n"
            f"{len(paths) - removed} database rows had no file on disk.",
        )
        self.refresh()
