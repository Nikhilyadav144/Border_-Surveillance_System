"""Settings: edit the configuration without leaving the application.

Changes are written back to ``config/application.yaml`` and
``config/threat_rules.yaml``. Anything that affects how frames are processed
takes effect the next time processing starts, which is stated on the screen
rather than left for the operator to discover.

Nothing important is hard-coded, so everything important is editable here.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.gui.pages.base import Page
from app.gui.theme import Palette
from app.gui.widgets.common import Card, StatRow, divider
from core.detector import detect_device
from core.threat_engine import DISCLAIMER
from database.models import ThreatLevel


class SettingsPage(Page):
    title = "Settings"
    subtitle = "Operational parameters, written back to config/"

    def build(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_detection(), "Detection")
        self.tabs.addTab(self._tab_behaviour(), "Behaviour")
        self.tabs.addTab(self._tab_threat(), "Threat Scoring")
        self.tabs.addTab(self._tab_anpr(), "ANPR")
        self.tabs.addTab(self._tab_evidence(), "Evidence")
        self.tabs.addTab(self._tab_system(), "System")
        self.content.addWidget(self.tabs, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.status = QLabel("")
        self.status.setObjectName("Dim")
        btn_reload = QPushButton("Discard Changes")
        btn_reload.clicked.connect(self.refresh)
        btn_save = QPushButton("Save To config/")
        btn_save.setObjectName("Primary")
        btn_save.clicked.connect(self._save)
        buttons.addWidget(self.status, 1)
        buttons.addWidget(btn_reload)
        buttons.addWidget(btn_save)
        self.content.addLayout(buttons)

        self.refresh()

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap(inner: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setWidget(inner)
        return area

    def _tab_detection(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        card = Card("Model")
        form = QFormLayout()
        form.setSpacing(9)

        model_row = QHBoxLayout()
        self.in_model = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_model)
        model_row.addWidget(self.in_model, 1)
        model_row.addWidget(browse)
        form.addRow("Weights file", model_row)

        self.in_confidence = QDoubleSpinBox()
        self.in_confidence.setRange(0.01, 0.99)
        self.in_confidence.setSingleStep(0.05)
        self.in_confidence.setDecimals(2)
        form.addRow("Confidence threshold", self.in_confidence)

        self.in_iou = QDoubleSpinBox()
        self.in_iou.setRange(0.01, 0.99)
        self.in_iou.setSingleStep(0.05)
        self.in_iou.setDecimals(2)
        form.addRow("NMS IoU threshold", self.in_iou)

        self.in_imgsz = QSpinBox()
        self.in_imgsz.setRange(160, 1920)
        self.in_imgsz.setSingleStep(32)
        form.addRow("Inference image size", self.in_imgsz)

        self.in_device = QComboBox()
        self.in_device.addItems(["auto", "cpu", "mps", "cuda:0"])
        form.addRow("Device", self.in_device)

        self.in_stride = QSpinBox()
        self.in_stride.setRange(1, 10)
        form.addRow("Process every Nth frame", self.in_stride)

        self.in_classes = QLineEdit()
        self.in_classes.setPlaceholderText(
            "person, car, motorcycle, bus, truck   (blank = every class)")
        form.addRow("Classes to process", self.in_classes)

        self.in_tracker = QComboBox()
        self.in_tracker.addItems(["bytetrack", "botsort"])
        form.addRow("Tracker", self.in_tracker)

        card.add_layout(form)
        layout.addWidget(card)

        info = Card("Detected Hardware")
        device = detect_device("auto")
        info.add(StatRow("Best available device", str(device),
                         Palette.ONLINE))
        info.add(StatRow("Reason", device.reason))
        note = QLabel(
            "A class listed above that the loaded model does not support is "
            "reported at startup and ignored. The application never claims to "
            "detect something the weights cannot detect."
        )
        note.setWordWrap(True)
        note.setObjectName("Dim")
        info.add(note)
        layout.addWidget(info)
        layout.addStretch(1)
        return self._wrap(page)

    def _tab_behaviour(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        card = Card("Loitering and Movement")
        form = QFormLayout()
        form.setSpacing(9)

        self.in_loiter_seconds = QDoubleSpinBox()
        self.in_loiter_seconds.setRange(1.0, 3600.0)
        self.in_loiter_seconds.setSuffix(" s")
        form.addRow("Loitering threshold", self.in_loiter_seconds)

        self.in_loiter_radius = QDoubleSpinBox()
        self.in_loiter_radius.setRange(0.005, 0.5)
        self.in_loiter_radius.setSingleStep(0.01)
        self.in_loiter_radius.setDecimals(3)
        form.addRow("Maximum movement radius", self.in_loiter_radius)

        self.in_direction_change = QDoubleSpinBox()
        self.in_direction_change.setRange(5.0, 180.0)
        self.in_direction_change.setSuffix(" °")
        form.addRow("Direction change angle", self.in_direction_change)

        self.in_max_age = QSpinBox()
        self.in_max_age.setRange(1, 300)
        form.addRow("Drop track after N missed frames", self.in_max_age)

        self.in_min_hits = QSpinBox()
        self.in_min_hits.setRange(1, 60)
        form.addRow("Minimum hits before incidents", self.in_min_hits)

        card.add_layout(form)
        note = QLabel(
            "Loitering requires BOTH a dwell longer than the threshold AND a "
            "movement radius smaller than the limit. An object that is present "
            "for a long time but walking through is not loitering.\n\n"
            "Time is measured in footage seconds, not processing seconds, so a "
            "clip analysed slowly still reports the dwell it really shows."
        )
        note.setWordWrap(True)
        note.setObjectName("Dim")
        card.add(note)
        layout.addWidget(card)

        anomaly = Card("Behavioural Anomaly Detection")
        aform = QFormLayout()
        aform.setSpacing(9)
        self.in_anomaly_enabled = QCheckBox("Enabled")
        aform.addRow("", self.in_anomaly_enabled)
        self.in_anomaly_min = QSpinBox()
        self.in_anomaly_min.setRange(5, 5000)
        aform.addRow("Tracks needed before scoring", self.in_anomaly_min)
        self.in_anomaly_contamination = QDoubleSpinBox()
        self.in_anomaly_contamination.setRange(0.001, 0.5)
        self.in_anomaly_contamination.setSingleStep(0.01)
        self.in_anomaly_contamination.setDecimals(3)
        aform.addRow("Expected anomaly rate", self.in_anomaly_contamination)
        anomaly.add_layout(aform)
        anote = QLabel(
            "Below the sample threshold the engine reports INSUFFICIENT_DATA "
            "and scores nothing. It never fabricates a value to fill the gap."
        )
        anote.setWordWrap(True)
        anote.setObjectName("Dim")
        anomaly.add(anote)
        layout.addWidget(anomaly)
        layout.addStretch(1)
        return self._wrap(page)

    def _tab_threat(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        warning = Card("Positioning")
        text = QLabel(DISCLAIMER)
        text.setWordWrap(True)
        text.setStyleSheet(f"color: {Palette.MEDIUM};")
        warning.add(text)
        layout.addWidget(warning)

        card = Card("Factor Weights")
        self.threat_inputs: dict[str, QSpinBox] = {}
        form = QFormLayout()
        form.setSpacing(9)
        factors = self.ctx.config.threat_rules.get("factors", {})
        for code, factor in factors.items():
            if not isinstance(factor, dict) or "weight" not in factor:
                continue
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setSuffix(" pts")
            self.threat_inputs[code] = spin
            form.addRow(str(factor.get("label", code)), spin)
        card.add_layout(form)
        layout.addWidget(card)

        levels = Card("Level Thresholds")
        self.level_inputs: dict[str, QSpinBox] = {}
        lform = QFormLayout()
        lform.setSpacing(9)
        for level in ThreatLevel.ORDER:
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setSuffix(" pts and above")
            self.level_inputs[level] = spin
            lform.addRow(level, spin)
        levels.add_layout(lform)
        layout.addWidget(levels)
        layout.addStretch(1)
        return self._wrap(page)

    def _tab_anpr(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        card = Card("Number Plate Recognition")
        form = QFormLayout()
        form.setSpacing(9)

        self.in_anpr_enabled = QCheckBox("Enabled")
        form.addRow("", self.in_anpr_enabled)

        self.in_anpr_backend = QComboBox()
        self.in_anpr_backend.addItems(["auto", "easyocr", "tesseract", "none"])
        form.addRow("OCR backend", self.in_anpr_backend)

        self.in_anpr_confidence = QDoubleSpinBox()
        self.in_anpr_confidence.setRange(0.0, 1.0)
        self.in_anpr_confidence.setSingleStep(0.05)
        self.in_anpr_confidence.setDecimals(2)
        form.addRow("Minimum OCR confidence", self.in_anpr_confidence)

        self.in_anpr_height = QSpinBox()
        self.in_anpr_height.setRange(20, 2000)
        self.in_anpr_height.setSuffix(" px")
        form.addRow("Minimum vehicle height", self.in_anpr_height)

        self.in_anpr_regex = QLineEdit()
        self.in_anpr_regex.setPlaceholderText("blank = accept any text")
        form.addRow("Plate format (regex)", self.in_anpr_regex)

        self.in_anpr_unvalidated = QCheckBox(
            "Accept reads that do not match the format")
        form.addRow("", self.in_anpr_unvalidated)

        card.add_layout(form)
        note = QLabel(
            "Below the confidence threshold a plate is recorded as UNREADABLE "
            "and the crop is still saved. Plate text is never guessed or "
            "auto-completed.\n\n"
            "Accuracy on real traffic footage has not been measured for this "
            "build. Measure it on your own footage before quoting a figure."
        )
        note.setWordWrap(True)
        note.setObjectName("Dim")
        card.add(note)
        layout.addWidget(card)

        status = Card("Backend Status")
        self.anpr_status_row = StatRow("Current backend", "not started")
        status.add(self.anpr_status_row)
        install = QLabel(
            "No backend installed? Either:\n"
            "    pip install easyocr        (better on plates, ~100 MB of weights)\n"
            "    pip install pytesseract    (plus 'brew install tesseract' on macOS)\n"
            "With neither, ANPR switches itself off and everything else keeps "
            "working."
        )
        install.setWordWrap(True)
        install.setObjectName("Dim")
        status.add(install)
        layout.addWidget(status)
        layout.addStretch(1)
        return self._wrap(page)

    def _tab_evidence(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        card = Card("Capture")
        form = QFormLayout()
        form.setSpacing(9)

        self.in_ev_enabled = QCheckBox("Enabled")
        form.addRow("", self.in_ev_enabled)

        self.in_ev_level = QComboBox()
        self.in_ev_level.addItems(list(ThreatLevel.ORDER))
        form.addRow("Capture at or above", self.in_ev_level)

        self.in_ev_full = QCheckBox("Save the annotated full frame")
        form.addRow("", self.in_ev_full)
        self.in_ev_crop = QCheckBox("Save a close-up crop of the object")
        form.addRow("", self.in_ev_crop)

        self.in_ev_quality = QSpinBox()
        self.in_ev_quality.setRange(50, 100)
        form.addRow("JPEG quality", self.in_ev_quality)

        self.in_ev_cooldown = QDoubleSpinBox()
        self.in_ev_cooldown.setRange(0.0, 600.0)
        self.in_ev_cooldown.setSuffix(" s")
        form.addRow("Cooldown per track", self.in_ev_cooldown)

        self.in_ev_retention = QSpinBox()
        self.in_ev_retention.setRange(0, 3650)
        self.in_ev_retention.setSuffix(" days  (0 = keep forever)")
        form.addRow("Retention", self.in_ev_retention)

        card.add_layout(form)
        layout.addWidget(card)

        incidents = Card("Incident Handling")
        iform = QFormLayout()
        iform.setSpacing(9)
        self.in_dedupe = QDoubleSpinBox()
        self.in_dedupe.setRange(0.0, 3600.0)
        self.in_dedupe.setSuffix(" s")
        iform.addRow("Deduplication window", self.in_dedupe)
        self.in_escalate = QCheckBox(
            "Escalate an open incident instead of opening a new one")
        iform.addRow("", self.in_escalate)
        self.in_toast_level = QComboBox()
        self.in_toast_level.addItems(list(ThreatLevel.ORDER))
        iform.addRow("Pop-up alerts at or above", self.in_toast_level)
        incidents.add_layout(iform)
        inote = QLabel(
            "The deduplication window is what stops one person in a restricted "
            "zone producing hundreds of identical incidents."
        )
        inote.setWordWrap(True)
        inote.setObjectName("Dim")
        incidents.add(inote)
        layout.addWidget(incidents)
        layout.addStretch(1)
        return self._wrap(page)

    def _tab_system(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        card = Card("Logging and Display")
        form = QFormLayout()
        form.setSpacing(9)

        self.in_log_level = QComboBox()
        self.in_log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        form.addRow("Log level", self.in_log_level)

        self.in_display_fps = QSpinBox()
        self.in_display_fps.setRange(1, 60)
        self.in_display_fps.setSuffix(" fps")
        form.addRow("Video display rate", self.in_display_fps)

        self.in_night_start = QSpinBox()
        self.in_night_start.setRange(0, 23)
        form.addRow("Night starts at hour", self.in_night_start)
        self.in_night_end = QSpinBox()
        self.in_night_end.setRange(0, 23)
        form.addRow("Night ends at hour", self.in_night_end)

        card.add_layout(form)
        layout.addWidget(card)

        overlays = Card("Video Overlays")
        self.overlay_boxes: dict[str, QCheckBox] = {}
        for key, label in (
            ("boxes", "Bounding boxes"), ("labels", "Labels"),
            ("track_ids", "Track identifiers"), ("trajectories", "Trajectories"),
            ("zones", "Zone polygons"), ("lines", "Virtual lines"),
            ("hud", "Heads-up display"),
        ):
            box = QCheckBox(label)
            self.overlay_boxes[key] = box
            overlays.add(box)
        layout.addWidget(overlays)

        paths = Card("Paths")
        cfg = self.ctx.config
        for key in ("input_dir", "evidence_dir", "database_file",
                    "models_dir", "logs_dir"):
            paths.add(StatRow(key.replace("_", " "), str(cfg.path(key))))
        layout.addWidget(paths)
        layout.addStretch(1)
        return self._wrap(page)

    # ------------------------------------------------------------------
    # Load and save
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        cfg = self.ctx.config

        self.in_model.setText(str(cfg.get("detection.model_path", "")))
        self.in_confidence.setValue(float(cfg.get("detection.confidence", 0.35)))
        self.in_iou.setValue(float(cfg.get("detection.iou", 0.5)))
        self.in_imgsz.setValue(int(cfg.get("detection.image_size", 640)))
        self.in_device.setCurrentText(str(cfg.get("detection.device", "auto")))
        self.in_stride.setValue(int(cfg.get("detection.frame_stride", 2)))
        self.in_classes.setText(", ".join(cfg.get("detection.classes", []) or []))
        self.in_tracker.setCurrentText(str(cfg.get("tracking.tracker",
                                                   "bytetrack")))

        self.in_loiter_seconds.setValue(
            float(cfg.get("behaviour.loitering_seconds", 8)))
        self.in_loiter_radius.setValue(
            float(cfg.get("behaviour.loitering_radius", 0.06)))
        self.in_direction_change.setValue(
            float(cfg.get("behaviour.direction_change_degrees", 45)))
        self.in_max_age.setValue(int(cfg.get("tracking.max_age_frames", 30)))
        self.in_min_hits.setValue(int(cfg.get("tracking.min_hits", 3)))

        self.in_anomaly_enabled.setChecked(bool(cfg.get("anomaly.enabled", True)))
        self.in_anomaly_min.setValue(int(cfg.get("anomaly.min_samples", 25)))
        self.in_anomaly_contamination.setValue(
            float(cfg.get("anomaly.contamination", 0.1)))

        factors = cfg.threat_rules.get("factors", {})
        for code, spin in self.threat_inputs.items():
            factor = factors.get(code, {})
            spin.setValue(int(factor.get("weight", 0))
                          if isinstance(factor, dict) else 0)
        levels = cfg.threat_rules.get("levels", {})
        for level, spin in self.level_inputs.items():
            spin.setValue(int(levels.get(level, 0)))

        self.in_anpr_enabled.setChecked(bool(cfg.get("anpr.enabled", True)))
        self.in_anpr_backend.setCurrentText(
            str(cfg.get("anpr.ocr_backend", "auto")))
        self.in_anpr_confidence.setValue(
            float(cfg.get("anpr.min_ocr_confidence", 0.45)))
        self.in_anpr_height.setValue(int(cfg.get("anpr.min_vehicle_height", 90)))
        self.in_anpr_regex.setText(str(cfg.get("anpr.plate_regex") or ""))
        self.in_anpr_unvalidated.setChecked(
            bool(cfg.get("anpr.accept_unvalidated", True)))
        self.anpr_status_row.set_value(self.ctx.anpr_status or "not started")

        self.in_ev_enabled.setChecked(bool(cfg.get("evidence.enabled", True)))
        self.in_ev_level.setCurrentText(
            str(cfg.get("evidence.min_threat_level", "MEDIUM")))
        self.in_ev_full.setChecked(bool(cfg.get("evidence.save_full_frame", True)))
        self.in_ev_crop.setChecked(bool(cfg.get("evidence.save_crop", True)))
        self.in_ev_quality.setValue(int(cfg.get("evidence.jpeg_quality", 90)))
        self.in_ev_cooldown.setValue(
            float(cfg.get("evidence.cooldown_seconds", 10)))
        self.in_ev_retention.setValue(int(cfg.get("evidence.retention_days", 30)))

        self.in_dedupe.setValue(float(cfg.get("incidents.dedupe_seconds", 20)))
        self.in_escalate.setChecked(
            bool(cfg.get("incidents.escalate_existing", True)))
        self.in_toast_level.setCurrentText(
            str(cfg.get("alerts.toast_min_level", "HIGH")))

        self.in_log_level.setCurrentText(str(cfg.get("logging.level", "INFO")))
        self.in_display_fps.setValue(int(cfg.get("gui.display_fps", 25)))
        self.in_night_start.setValue(int(cfg.get("schedule.night_start_hour", 20)))
        self.in_night_end.setValue(int(cfg.get("schedule.night_end_hour", 6)))

        overlays = cfg.get("gui.overlays", {}) or {}
        for key, box in self.overlay_boxes.items():
            box.setChecked(bool(overlays.get(key, True)))

        self.status.setText("Loaded from config/")

    def _browse_model(self) -> None:
        start = str(self.ctx.config.path("models_dir"))
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO weights", start,
            "PyTorch weights (*.pt);;All files (*)",
        )
        if path:
            models_dir = self.ctx.config.path("models_dir")
            candidate = Path(path)
            try:
                relative = candidate.relative_to(models_dir)
                self.in_model.setText(str(relative))
            except ValueError:
                self.in_model.setText(str(candidate))

    def _save(self) -> None:
        cfg = self.ctx.config
        try:
            cfg.set("detection.model_path", self.in_model.text().strip())
            cfg.set("detection.confidence", round(self.in_confidence.value(), 3))
            cfg.set("detection.iou", round(self.in_iou.value(), 3))
            cfg.set("detection.image_size", self.in_imgsz.value())
            cfg.set("detection.device", self.in_device.currentText())
            cfg.set("detection.frame_stride", self.in_stride.value())
            classes = [c.strip() for c in self.in_classes.text().split(",")
                       if c.strip()]
            cfg.set("detection.classes", classes)
            cfg.set("tracking.tracker", self.in_tracker.currentText())

            cfg.set("behaviour.loitering_seconds",
                    round(self.in_loiter_seconds.value(), 2))
            cfg.set("behaviour.loitering_radius",
                    round(self.in_loiter_radius.value(), 4))
            cfg.set("behaviour.direction_change_degrees",
                    round(self.in_direction_change.value(), 1))
            cfg.set("tracking.max_age_frames", self.in_max_age.value())
            cfg.set("tracking.min_hits", self.in_min_hits.value())

            cfg.set("anomaly.enabled", self.in_anomaly_enabled.isChecked())
            cfg.set("anomaly.min_samples", self.in_anomaly_min.value())
            cfg.set("anomaly.contamination",
                    round(self.in_anomaly_contamination.value(), 4))

            cfg.set("anpr.enabled", self.in_anpr_enabled.isChecked())
            cfg.set("anpr.ocr_backend", self.in_anpr_backend.currentText())
            cfg.set("anpr.min_ocr_confidence",
                    round(self.in_anpr_confidence.value(), 3))
            cfg.set("anpr.min_vehicle_height", self.in_anpr_height.value())
            cfg.set("anpr.plate_regex",
                    self.in_anpr_regex.text().strip() or None)
            cfg.set("anpr.accept_unvalidated",
                    self.in_anpr_unvalidated.isChecked())

            cfg.set("evidence.enabled", self.in_ev_enabled.isChecked())
            cfg.set("evidence.min_threat_level",
                    self.in_ev_level.currentText())
            cfg.set("evidence.save_full_frame", self.in_ev_full.isChecked())
            cfg.set("evidence.save_crop", self.in_ev_crop.isChecked())
            cfg.set("evidence.jpeg_quality", self.in_ev_quality.value())
            cfg.set("evidence.cooldown_seconds",
                    round(self.in_ev_cooldown.value(), 2))
            cfg.set("evidence.retention_days", self.in_ev_retention.value())

            cfg.set("incidents.dedupe_seconds", round(self.in_dedupe.value(), 2))
            cfg.set("incidents.escalate_existing", self.in_escalate.isChecked())
            cfg.set("alerts.toast_min_level", self.in_toast_level.currentText())

            cfg.set("logging.level", self.in_log_level.currentText())
            cfg.set("gui.display_fps", self.in_display_fps.value())
            cfg.set("schedule.night_start_hour", self.in_night_start.value())
            cfg.set("schedule.night_end_hour", self.in_night_end.value())
            cfg.set("gui.overlays", {
                key: box.isChecked() for key, box in self.overlay_boxes.items()
            })

            path = cfg.save()

            # Threat rules live in their own file.
            factors = cfg.threat_rules.setdefault("factors", {})
            for code, spin in self.threat_inputs.items():
                factor = factors.get(code)
                if isinstance(factor, dict):
                    factor["weight"] = spin.value()
            levels = cfg.threat_rules.setdefault("levels", {})
            for level, spin in self.level_inputs.items():
                levels[level] = spin.value()

            rules_path = Path(str(cfg.get("threat.rules_file",
                                          "config/threat_rules.yaml")))
            if not rules_path.is_absolute():
                rules_path = cfg.source_file.parent.parent / rules_path
            with rules_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(cfg.threat_rules, handle, sort_keys=False,
                               default_flow_style=False)

        except Exception as exc:
            QMessageBox.critical(
                self, "Could not save settings",
                f"{exc}\n\nCheck that config/ is writable."
            )
            self.status.setText("Save failed")
            return

        self.status.setText(f"Saved to {path.name} and {rules_path.name}")
        QMessageBox.information(
            self, "Settings saved",
            f"Written to:\n  {path}\n  {rules_path}\n\n"
            f"Detection, tracking and threat settings take effect the next "
            f"time you press Start on the Cameras screen. Display settings "
            f"apply immediately.",
        )
        self.ctx.settings_changed()
