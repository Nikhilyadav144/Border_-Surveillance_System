"""Database operations, configuration loading, video sources and ANPR helpers."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from core.anpr import AnprEngine
from core.config import ConfigError, load_config
from core.evidence import EvidenceManager
from database.database import Database
from database.models import (
    Alert,
    CameraStatus,
    Incident,
    IncidentStatus,
    IncidentType,
    PlateRead,
    ThreatLevel,
    TrackRecord,
)
from inputs.base import VideoSourceError
from inputs.factory import build_source, describe_source
from inputs.rtsp_source import RtspSource, redact_url
from inputs.video_source import FileVideoSource


def write_video(path: Path, frames: int = 30, size=(320, 240)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             20.0, size)
    for index in range(frames):
        frame = np.full((size[1], size[0], 3), index * 4 % 255, np.uint8)
        writer.write(frame)
    writer.release()
    return path


class TestDatabase:
    def test_schema_is_created(self, db):
        tables = {row["name"] for row in db.query(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for expected in ("cameras", "tracks", "incidents", "alerts",
                         "evidence", "plates", "watchlist", "system_logs",
                         "zones", "settings", "detections", "sessions"):
            assert expected in tables

    def test_reopening_an_existing_database_is_safe(self, tmp_path):
        path = tmp_path / "twice.db"
        first = Database(path)
        first.upsert_camera("CAM-01", "A", "L", "video", "x.mp4")
        first.close()
        second = Database(path)
        assert second.get_camera("CAM-01") is not None
        second.close()

    def test_camera_upsert_updates_rather_than_duplicates(self, db):
        db.upsert_camera("CAM-01", "First", "L", "video", "a.mp4")
        db.upsert_camera("CAM-01", "Second", "L", "video", "b.mp4")
        assert len(db.get_cameras()) == 1
        assert db.get_camera("CAM-01")["name"] == "Second"

    def test_camera_health_round_trip(self, db):
        db.upsert_camera("CAM-01", "A", "L", "video", "x.mp4")
        db.update_camera_health("CAM-01", CameraStatus.ONLINE, fps=24.5,
                                latency_ms=41.0, frames_seen=120,
                                last_frame_ts=time.time())
        row = db.get_camera("CAM-01")
        assert row["status"] == CameraStatus.ONLINE
        assert row["fps"] == pytest.approx(24.5)

    def test_track_upsert_is_idempotent(self, db):
        record = TrackRecord(camera_id="CAM-01", track_number=5,
                             class_name="person", first_ts=1.0, last_ts=2.0,
                             frames_seen=10)
        first = db.upsert_track(record)
        record.frames_seen = 20
        second = db.upsert_track(record)
        assert first == second
        assert len(db.get_tracks()) == 1
        assert db.get_tracks()[0]["frames_seen"] == 20

    def test_incident_and_alert_persist(self, db):
        db.upsert_camera("CAM-01", "A", "L", "video", "x.mp4")
        incident = Incident(camera_id="CAM-01",
                            incident_type=IncidentType.LOITERING,
                            ts=time.time(), threat_score=60,
                            threat_level=ThreatLevel.HIGH,
                            track_number=3, object_class="person")
        incident_id = db.insert_incident(incident)
        db.insert_alert(Alert(incident_id=incident_id, camera_id="CAM-01",
                              severity=ThreatLevel.HIGH,
                              alert_type=IncidentType.LOITERING,
                              message="test", ts=time.time()))
        assert db.count_incidents() == 1
        assert db.count_active_alerts() == 1
        assert db.highest_open_threat() == ThreatLevel.HIGH

    def test_status_change_cascades_to_the_alert(self, db):
        db.upsert_camera("CAM-01", "A", "L", "video", "x.mp4")
        incident = Incident(camera_id="CAM-01",
                            incident_type=IncidentType.LOITERING,
                            ts=time.time(), threat_level=ThreatLevel.HIGH)
        incident_id = db.insert_incident(incident)
        db.insert_alert(Alert(incident_id=incident_id, camera_id="CAM-01",
                              severity=ThreatLevel.HIGH, alert_type="x",
                              message="m", ts=time.time()))
        db.set_incident_status(incident_id, IncidentStatus.ACKNOWLEDGED)
        assert db.get_alerts()[0]["status"] == IncidentStatus.ACKNOWLEDGED

    def test_resolved_incidents_leave_the_threat_banner(self, db):
        db.upsert_camera("CAM-01", "A", "L", "video", "x.mp4")
        incident = Incident(camera_id="CAM-01", incident_type="x",
                            ts=time.time(), threat_level=ThreatLevel.CRITICAL)
        incident_id = db.insert_incident(incident)
        assert db.highest_open_threat() == ThreatLevel.CRITICAL
        db.set_incident_status(incident_id, IncidentStatus.RESOLVED)
        assert db.highest_open_threat() == ThreatLevel.LOW

    def test_incident_filters(self, db):
        db.upsert_camera("CAM-01", "A", "L", "video", "x.mp4")
        db.upsert_camera("CAM-02", "B", "L", "video", "y.mp4")
        now = time.time()
        for camera, level, kind in (
            ("CAM-01", ThreatLevel.HIGH, IncidentType.LOITERING),
            ("CAM-01", ThreatLevel.LOW, IncidentType.LINE_CROSSING),
            ("CAM-02", ThreatLevel.HIGH, IncidentType.LOITERING),
        ):
            db.insert_incident(Incident(camera_id=camera, incident_type=kind,
                                        ts=now, threat_level=level))
        assert len(db.get_incidents(camera_id="CAM-01")) == 2
        assert len(db.get_incidents(level=ThreatLevel.HIGH)) == 2
        assert len(db.get_incidents(incident_type=IncidentType.LOITERING)) == 2

    def test_watchlist_crud(self, db):
        db.add_watchlist_entry("DL01AB1234", label="Stolen")
        assert len(db.get_watchlist(active_only=True)) == 1
        db.add_watchlist_entry("dl01ab1234", label="Updated")   # same, upper-cased
        assert len(db.get_watchlist()) == 1
        entry_id = db.get_watchlist()[0]["id"]
        db.set_watchlist_active(entry_id, False)
        assert len(db.get_watchlist(active_only=True)) == 0
        db.remove_watchlist_entry(entry_id)
        assert len(db.get_watchlist()) == 0

    def test_watchlist_hits_are_counted(self, db):
        db.add_watchlist_entry("DL01AB1234")
        db.record_watchlist_hit("DL01AB1234")
        db.record_watchlist_hit("dl01ab1234")
        assert db.get_watchlist()[0]["hit_count"] == 2

    def test_unreadable_plates_are_stored_as_null(self, db):
        db.insert_plate(PlateRead(camera_id="CAM-01", track_number=1,
                                  text=None, ocr_confidence=0.0,
                                  backend="tesseract", ts=time.time()))
        assert db.get_plates()[0]["plate_text"] is None
        assert db.analytics_counts()["plates_read"] == 0

    def test_analytics_queries_on_an_empty_database(self, db):
        counts = db.analytics_counts()
        assert counts["incidents"] == 0
        assert db.analytics_by_level() == {}
        assert db.analytics_by_type() == []
        assert db.analytics_busiest_camera() is None
        assert len(db.analytics_hourly()) == 24

    def test_settings_round_trip(self, db):
        db.set_setting("selected", {"camera": "CAM-01", "zoom": 2})
        assert db.get_setting("selected")["camera"] == "CAM-01"
        assert db.get_setting("absent", "fallback") == "fallback"

    def test_system_logs_can_be_filtered_and_trimmed(self, db):
        for index in range(30):
            db.insert_system_log("ERROR" if index % 5 == 0 else "INFO",
                                 "test", f"message {index}")
        assert len(db.get_system_logs(level="ERROR")) == 6
        db.trim_system_logs(keep=10)
        assert len(db.get_system_logs()) == 10


class TestConfig:
    def test_shipped_configuration_is_valid(self, config):
        assert config.warnings == []
        assert config.get("detection.confidence") > 0
        assert config.cameras
        # The shipped profiles are named for what they describe. There is
        # deliberately no profile called "default": geometry is opt-in.
        assert "none" in config.zone_profile_names()
        assert "default" not in config.zone_profile_names()

    def test_dotted_access(self, config):
        assert config.get("detection.model_path")
        assert config.get("nothing.here", "fallback") == "fallback"

    def test_paths_are_absolute(self, config):
        assert config.path("evidence_dir").is_absolute()
        assert config.model_path().is_absolute()

    def test_zone_profile_returns_zones_and_lines(self, config):
        profile = config.zone_profile("demo_border_clip")
        assert profile["zones"] and profile["lines"]

    def test_unknown_profile_warns_and_returns_empty(self, config):
        profile = config.zone_profile("does-not-exist")
        assert profile == {"zones": [], "lines": []}
        assert any("does-not-exist" in w for w in config.warnings)

    def test_out_of_range_value_is_rejected(self, tmp_path):
        (tmp_path / "application.yaml").write_text(
            yaml.safe_dump({"detection": {"confidence": 5.0}}))
        with pytest.raises(ConfigError, match="confidence"):
            load_config(tmp_path)

    def test_malformed_polygon_is_rejected(self, tmp_path):
        (tmp_path / "application.yaml").write_text(yaml.safe_dump({}))
        (tmp_path / "zones.yaml").write_text(yaml.safe_dump({
            "profiles": {"default": {"zones": [
                {"name": "Bad", "kind": "RESTRICTED",
                 "polygon": [[0, 0], [1, 1]]},
            ]}}
        }))
        with pytest.raises(ConfigError, match="at least 3 points"):
            load_config(tmp_path)

    def test_denormalised_coordinates_are_rejected(self, tmp_path):
        (tmp_path / "application.yaml").write_text(yaml.safe_dump({}))
        (tmp_path / "zones.yaml").write_text(yaml.safe_dump({
            "profiles": {"default": {"zones": [
                {"name": "Pixels", "kind": "RESTRICTED",
                 "polygon": [[0, 0], [640, 0], [640, 480]]},
            ]}}
        }))
        with pytest.raises(ConfigError, match="normalised"):
            load_config(tmp_path)

    def test_unknown_camera_source_type_is_rejected(self, tmp_path):
        (tmp_path / "application.yaml").write_text(yaml.safe_dump({}))
        (tmp_path / "cameras.yaml").write_text(yaml.safe_dump({
            "cameras": [{"id": "CAM-01", "source": {"type": "telepathy"}}]
        }))
        with pytest.raises(ConfigError, match="unknown source type"):
            load_config(tmp_path)

    def test_duplicate_camera_ids_are_rejected(self, tmp_path):
        (tmp_path / "application.yaml").write_text(yaml.safe_dump({}))
        (tmp_path / "cameras.yaml").write_text(yaml.safe_dump({
            "cameras": [
                {"id": "CAM-01", "source": {"type": "video", "path": "a.mp4"}},
                {"id": "CAM-01", "source": {"type": "video", "path": "b.mp4"}},
            ]
        }))
        with pytest.raises(ConfigError, match="duplicate camera id"):
            load_config(tmp_path)

    def test_missing_files_degrade_with_warnings(self, tmp_path):
        config = load_config(tmp_path)
        assert config.warnings
        assert config.get("detection.confidence") is not None

    def test_save_and_reload(self, tmp_path, config):
        target = tmp_path / "application.yaml"
        config.set("detection.confidence", 0.61)
        config.save(target)
        reloaded = yaml.safe_load(target.read_text())
        assert reloaded["detection"]["confidence"] == 0.61


class TestVideoSources:
    def test_reading_a_file_yields_frames(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4", frames=20)
        with FileVideoSource(path, source_id="CAM-T") as source:
            frames = list(source)
        assert len(frames) == 20
        assert frames[0].index == 1
        assert frames[0].image.shape[:2] == (240, 320)

    def test_timestamps_follow_media_time_not_the_wall_clock(self, tmp_path):
        """Dwell must mean seconds of footage, not seconds of CPU."""
        path = write_video(tmp_path / "clip.mp4", frames=40)
        with FileVideoSource(path, source_id="CAM-T") as source:
            frames = list(source)
        span = frames[-1].ts - frames[0].ts
        # 39 frames at 20 fps is 1.95 s of footage, whatever the CPU did.
        assert span == pytest.approx(1.95, abs=0.05)

    def test_looping_restarts_the_file(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4", frames=10)
        source = FileVideoSource(path, loop=True)
        source.open()
        collected = [source.read() for _ in range(25)]
        source.release()
        assert all(frame is not None for frame in collected)
        assert collected[-1].index == 25

    def test_missing_file_explains_itself(self, tmp_path):
        with pytest.raises(VideoSourceError, match="not found"):
            FileVideoSource(tmp_path / "nope.mp4").open()

    def test_empty_file_is_rejected(self, tmp_path):
        path = tmp_path / "empty.mp4"
        path.write_bytes(b"")
        with pytest.raises(VideoSourceError):
            FileVideoSource(path).open()

    def test_reading_before_open_raises(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4", frames=5)
        with pytest.raises(VideoSourceError, match="before open"):
            FileVideoSource(path).read()

    def test_factory_builds_each_source_type(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4", frames=5)
        video = build_source({"id": "C1", "source": {
            "type": "video", "path": str(path)}})
        assert isinstance(video, FileVideoSource)

        from inputs.webcam_source import WebcamSource
        webcam = build_source({"id": "C2", "source": {
            "type": "webcam", "index": 0}})
        assert isinstance(webcam, WebcamSource)

        rtsp = build_source({"id": "C3", "source": {
            "type": "rtsp", "url": "rtsp://host/stream"}})
        assert isinstance(rtsp, RtspSource)

    def test_factory_rejects_an_unknown_type(self):
        with pytest.raises(VideoSourceError, match="unknown source type"):
            build_source({"id": "C1", "source": {"type": "carrier-pigeon"}})

    def test_rtsp_credentials_are_never_logged(self):
        redacted = redact_url("rtsp://admin:hunter2@10.0.0.5:554/Streaming/1")
        assert "hunter2" not in redacted
        assert "admin" not in redacted
        assert "10.0.0.5" in redacted

    def test_describe_source_hides_credentials(self):
        described = describe_source({"id": "C", "source": {
            "type": "rtsp", "url": "rtsp://u:p@host/s"}})
        assert "p@host" not in described


class TestEvidenceManager:
    def test_capture_is_gated_by_threat_level(self, tmp_path):
        manager = EvidenceManager(tmp_path, min_threat_level="HIGH")
        assert not manager.should_capture("MEDIUM", "k", now=0.0)
        assert manager.should_capture("HIGH", "k", now=0.0)
        assert manager.should_capture("CRITICAL", "k", now=0.0)

    def test_cooldown_prevents_a_flood(self, tmp_path):
        manager = EvidenceManager(tmp_path, min_threat_level="LOW",
                                  cooldown_seconds=10.0)
        assert manager.should_capture("HIGH", "k", now=0.0)
        manager.note_capture("k", now=0.0)
        assert not manager.should_capture("HIGH", "k", now=5.0)
        assert manager.should_capture("HIGH", "k", now=11.0)

    def test_files_are_written_and_indexed(self, tmp_path, blank_frame):
        manager = EvidenceManager(tmp_path, min_threat_level="LOW")
        files = manager.capture(frame=blank_frame, camera_id="CAM-01",
                                incident_id=42, track_number=7,
                                box=(100, 100, 260, 400), ts=time.time())
        assert len(files) == 2
        for item in files:
            assert item.path.exists()
            assert item.size_bytes > 0
        assert {f.kind for f in files} == {"full_frame", "crop"}

    def test_disabled_manager_writes_nothing(self, tmp_path, blank_frame):
        manager = EvidenceManager(tmp_path, enabled=False)
        assert manager.capture(blank_frame, "CAM-01", 1, 1) == []

    def test_purge_removes_files(self, tmp_path, blank_frame):
        manager = EvidenceManager(tmp_path, min_threat_level="LOW")
        files = manager.capture(blank_frame, "CAM-01", 1, 1, ts=time.time())
        assert manager.purge([str(f.path) for f in files]) == len(files)
        assert not any(f.path.exists() for f in files)


class TestAnprHelpers:
    def test_text_cleaning_strips_punctuation(self):
        engine = AnprEngine(enabled=False)
        assert engine.clean_text("dl-01 ab*1234") == "DL01AB1234"

    def test_validation_against_the_indian_format(self):
        engine = AnprEngine(
            enabled=False,
            plate_regex="^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$")
        assert engine.validate("DL01AB1234")
        assert not engine.validate("12345")
        assert not engine.validate("")

    def test_confusion_correction_only_when_it_fixes_the_format(self):
        engine = AnprEngine(
            enabled=False,
            plate_regex="^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$")
        assert engine.correct_confusions("DLO1AB1234") == "DL01AB1234"
        # Already valid: left alone.
        assert engine.correct_confusions("DL01AB1234") == "DL01AB1234"
        # Unfixable: returned untouched rather than mangled into a guess.
        assert engine.correct_confusions("ZZZZ") == "ZZZZ"

    def test_unavailable_backend_returns_unreadable_never_a_guess(self,
                                                                 blank_frame):
        engine = AnprEngine(enabled=True, ocr_backend="none")
        result = engine.read_plate(blank_frame, (100, 100, 300, 400))
        assert result["text"] is None
        assert result["confidence"] == 0.0
        assert "no OCR backend" in result["reason"]

    def test_small_vehicles_are_skipped_with_a_reason(self, blank_frame):
        engine = AnprEngine(enabled=True, ocr_backend="none",
                            min_vehicle_height=200)
        result = engine.read_plate(blank_frame, (10, 10, 60, 60))
        assert result["text"] is None

    def test_preprocessing_produces_a_binary_image(self):
        plate = np.random.default_rng(0).integers(
            0, 255, (40, 160, 3), dtype=np.uint8)
        prepared = AnprEngine.preprocess(plate)
        assert prepared.ndim == 2
        assert set(np.unique(prepared)).issubset({0, 255})
        assert prepared.shape[0] >= 96

    def test_region_proposal_returns_plate_shaped_boxes(self):
        # A light panel with dark bars: the shape a plate makes to the
        # morphological detector.
        crop = np.full((260, 380, 3), 70, np.uint8)
        cv2.rectangle(crop, (90, 170), (300, 225), (235, 235, 235), -1)
        for x in range(100, 295, 22):
            cv2.rectangle(crop, (x, 182), (x + 11, 214), (25, 25, 25), -1)

        engine = AnprEngine(enabled=False)
        candidates = engine.find_plate_regions(crop)
        assert candidates
        assert any(1.8 <= c.aspect <= 7.0 for c in candidates)

    def test_status_text_is_honest_when_unavailable(self):
        engine = AnprEngine(enabled=True, ocr_backend="none")
        assert not engine.available
        assert "No OCR backend" in engine.status_text()
