"""SQLite persistence layer.

One :class:`Database` instance owns one connection. The connection is shared
between the Qt GUI thread and the pipeline worker thread, so every statement
runs under a re-entrant lock and the connection is opened with
``check_same_thread=False``. WAL journalling lets readers proceed while a
write is in flight, which is what keeps the Analytics screen from stalling
the pipeline.

No engine in ``core/`` imports this module. Persistence is called from the
worker and from the GUI, never from the computer-vision code, so the engines
stay unit-testable without a database.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from core.paths import ensure_dir
from database.models import (
    Alert,
    CameraStatus,
    Incident,
    IncidentStatus,
    PlateRead,
    TrackRecord,
    now_ts,
    ts_text,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"


class DatabaseError(RuntimeError):
    """Raised when the database cannot be opened or initialised."""


class Database:
    """Thread-safe SQLite wrapper with an explicit, hand-written API."""

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._open()
        self._apply_schema()

    # -- lifecycle -------------------------------------------------------

    def _open(self) -> None:
        try:
            ensure_dir(self.path.parent)
            self._conn = sqlite3.connect(
                str(self.path), check_same_thread=False, timeout=15.0
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            logger.info("Database opened at %s", self.path)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot open database at {self.path}: {exc}") from exc

    def _apply_schema(self) -> None:
        try:
            sql = _SCHEMA_FILE.read_text(encoding="utf-8")
        except OSError as exc:
            raise DatabaseError(f"Cannot read schema file: {exc}") from exc
        try:
            with self._lock:
                self._conn.executescript(sql)  # type: ignore[union-attr]
                self._conn.execute(  # type: ignore[union-attr]
                    "INSERT OR IGNORE INTO schema_version (version, applied_at) "
                    "VALUES (?, ?)",
                    (SCHEMA_VERSION, ts_text()),
                )
                self._conn.commit()  # type: ignore[union-attr]
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot initialise schema: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None
                logger.info("Database closed")

    # -- primitives ------------------------------------------------------

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        if self._conn is None:
            raise DatabaseError("Database is closed")
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except sqlite3.Error:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Run a write statement. Returns ``lastrowid``."""
        with self._cursor() as cur:
            cur.execute(sql, params)
            return int(cur.lastrowid or 0)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self._cursor() as cur:
            cur.executemany(sql, list(rows))

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Run a read statement and return every row."""
        if self._conn is None:
            raise DatabaseError("Database is closed")
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute(sql, params)
                return cur.fetchall()
            finally:
                cur.close()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params: Sequence[Any] = (), default: Any = 0) -> Any:
        row = self.query_one(sql, params)
        if row is None or row[0] is None:
            return default
        return row[0]

    @staticmethod
    def _insert(table: str, row: dict[str, Any]) -> tuple[str, list[Any]]:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        return f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values())

    # ==================================================================
    # cameras
    # ==================================================================

    def upsert_camera(
        self,
        camera_id: str,
        name: str,
        location: str | None,
        source_type: str,
        source_detail: str,
        zones_profile: str = "default",
        enabled: bool = True,
        status: str = CameraStatus.CONFIGURED,
    ) -> None:
        stamp = ts_text()
        self.execute(
            """
            INSERT INTO cameras (id, name, location, source_type, source_detail,
                                 zones_profile, enabled, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                location=excluded.location,
                source_type=excluded.source_type,
                source_detail=excluded.source_detail,
                zones_profile=excluded.zones_profile,
                enabled=excluded.enabled,
                updated_at=excluded.updated_at
            """,
            (camera_id, name, location, source_type, source_detail,
             zones_profile, int(enabled), status, stamp, stamp),
        )

    def update_camera_health(
        self,
        camera_id: str,
        status: str,
        fps: float = 0.0,
        latency_ms: float = 0.0,
        frames_seen: int = 0,
        last_frame_ts: float | None = None,
        error_message: str | None = None,
    ) -> None:
        self.execute(
            """
            UPDATE cameras
               SET status=?, fps=?, latency_ms=?, frames_seen=?,
                   last_frame_ts=?, error_message=?, updated_at=?
             WHERE id=?
            """,
            (status, fps, latency_ms, frames_seen, last_frame_ts,
             error_message, ts_text(), camera_id),
        )

    def get_cameras(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM cameras ORDER BY id")

    def get_camera(self, camera_id: str) -> Optional[sqlite3.Row]:
        return self.query_one("SELECT * FROM cameras WHERE id=?", (camera_id,))

    def delete_camera(self, camera_id: str) -> None:
        self.execute("DELETE FROM cameras WHERE id=?", (camera_id,))

    # ==================================================================
    # sessions
    # ==================================================================

    def start_session(
        self,
        camera_id: str,
        source_detail: str,
        model_path: str,
        device: str,
    ) -> int:
        ts = now_ts()
        return self.execute(
            """
            INSERT INTO sessions (camera_id, started_ts, started_text,
                                  source_detail, model_path, device)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (camera_id, ts, ts_text(ts), source_detail, model_path, device),
        )

    def end_session(
        self,
        session_id: int,
        frames_read: int,
        frames_processed: int,
        detections: int,
        incidents: int,
        notes: str | None = None,
    ) -> None:
        self.execute(
            """
            UPDATE sessions
               SET ended_ts=?, frames_read=?, frames_processed=?,
                   detections=?, incidents=?, notes=?
             WHERE id=?
            """,
            (now_ts(), frames_read, frames_processed, detections,
             incidents, notes, session_id),
        )

    def get_sessions(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM sessions ORDER BY started_ts DESC LIMIT ?", (limit,)
        )

    # ==================================================================
    # zones
    # ==================================================================

    def save_zones(self, camera_id: str, profile: str, zones: list[dict]) -> None:
        stamp = ts_text()
        for zone in zones:
            self.execute(
                """
                INSERT INTO zones (camera_id, profile, name, kind, polygon,
                                   colour, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(camera_id, profile, name) DO UPDATE SET
                    kind=excluded.kind, polygon=excluded.polygon,
                    colour=excluded.colour, updated_at=excluded.updated_at
                """,
                (camera_id, profile, zone.get("name", "?"),
                 zone.get("kind", "OBSERVATION"),
                 json.dumps(zone.get("polygon", [])),
                 zone.get("colour"), stamp),
            )

    def get_zones(self, camera_id: str) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM zones WHERE camera_id=? ORDER BY name", (camera_id,)
        )

    # ==================================================================
    # tracks
    # ==================================================================

    def upsert_track(self, track: TrackRecord) -> int:
        row = track.to_row()
        existing = self.query_one(
            """
            SELECT id FROM tracks
             WHERE camera_id=? AND track_number=? AND class_name=?
               AND IFNULL(session_id, -1)=IFNULL(?, -1)
            """,
            (track.camera_id, track.track_number, track.class_name, track.session_id),
        )
        if existing is not None:
            track_id = int(existing["id"])
            assignments = ", ".join(f"{k}=?" for k in row if k != "session_id")
            values = [v for k, v in row.items() if k != "session_id"]
            self.execute(
                f"UPDATE tracks SET {assignments} WHERE id=?", values + [track_id]
            )
            track.db_id = track_id
            return track_id

        sql, values = self._insert("tracks", row)
        track_id = self.execute(sql, values)
        track.db_id = track_id
        return track_id

    def close_track(self, track_db_id: int) -> None:
        self.execute("UPDATE tracks SET closed=1 WHERE id=?", (track_db_id,))

    def get_tracks(self, camera_id: str | None = None, limit: int = 500
                   ) -> list[sqlite3.Row]:
        if camera_id:
            return self.query(
                "SELECT * FROM tracks WHERE camera_id=? ORDER BY last_ts DESC LIMIT ?",
                (camera_id, limit),
            )
        return self.query("SELECT * FROM tracks ORDER BY last_ts DESC LIMIT ?", (limit,))

    def get_track_feature_rows(self, limit: int = 2000) -> list[sqlite3.Row]:
        """Completed tracks with the columns the anomaly engine trains on."""
        return self.query(
            """
            SELECT avg_speed, max_speed, dwell_seconds, distance_norm,
                   direction_changes, zone_transitions, restricted_seconds,
                   frames_seen
              FROM tracks
             WHERE closed=1
             ORDER BY last_ts DESC
             LIMIT ?
            """,
            (limit,),
        )

    # ==================================================================
    # detections
    # ==================================================================

    def insert_detections(self, rows: list[dict]) -> None:
        if not rows:
            return
        sql = (
            "INSERT INTO detections (session_id, camera_id, track_id, frame_index, "
            "ts, ts_text, class_name, confidence, x1, y1, x2, y2, zone_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        self.executemany(
            sql,
            [
                (r.get("session_id"), r["camera_id"], r.get("track_id"),
                 r["frame_index"], r["ts"], ts_text(r["ts"]), r["class_name"],
                 r["confidence"], r["x1"], r["y1"], r["x2"], r["y2"],
                 r.get("zone_name"))
                for r in rows
            ],
        )

    def count_detections(self) -> int:
        return int(self.scalar("SELECT COUNT(*) FROM detections"))

    # ==================================================================
    # incidents
    # ==================================================================

    def insert_incident(self, incident: Incident) -> int:
        sql, values = self._insert("incidents", incident.to_row())
        incident.db_id = self.execute(sql, values)
        return incident.db_id

    def escalate_incident(
        self,
        incident_id: int,
        threat_score: float,
        threat_level: str,
        reasons_json: str,
        detail_json: str,
    ) -> None:
        self.execute(
            """
            UPDATE incidents
               SET threat_score=?, threat_level=?, reasons=?, detail=?,
                   escalation_count=escalation_count+1
             WHERE id=?
            """,
            (round(threat_score, 2), threat_level, reasons_json,
             detail_json, incident_id),
        )

    def set_incident_status(
        self,
        incident_id: int,
        status: str,
        operator: str = "operator",
        note: str | None = None,
    ) -> None:
        ts = now_ts()
        if status == IncidentStatus.ACKNOWLEDGED:
            self.execute(
                "UPDATE incidents SET status=?, acknowledged_ts=?, acknowledged_by=? "
                "WHERE id=?",
                (status, ts, operator, incident_id),
            )
            self.execute(
                "UPDATE alerts SET status=?, acknowledged_ts=? WHERE incident_id=?",
                (status, ts, incident_id),
            )
        elif status == IncidentStatus.RESOLVED:
            self.execute(
                "UPDATE incidents SET status=?, resolved_ts=?, resolved_by=?, "
                "resolution_note=? WHERE id=?",
                (status, ts, operator, note, incident_id),
            )
            self.execute(
                "UPDATE alerts SET status=? WHERE incident_id=?",
                (status, incident_id),
            )
        else:
            self.execute("UPDATE incidents SET status=? WHERE id=?",
                         (status, incident_id))
            self.execute("UPDATE alerts SET status=? WHERE incident_id=?",
                         (status, incident_id))

    def find_open_incident(
        self,
        camera_id: str,
        track_number: int | None,
        incident_type: str,
        since_ts: float,
    ) -> Optional[sqlite3.Row]:
        """Most recent still-open incident of this type for this track."""
        if track_number is None:
            return None
        return self.query_one(
            """
            SELECT * FROM incidents
             WHERE camera_id=? AND track_number=? AND incident_type=?
               AND ts >= ? AND status != 'RESOLVED'
             ORDER BY ts DESC LIMIT 1
            """,
            (camera_id, track_number, incident_type, since_ts),
        )

    def get_incidents(
        self,
        status: str | None = None,
        camera_id: str | None = None,
        level: str | None = None,
        incident_type: str | None = None,
        date_from: float | None = None,
        date_to: float | None = None,
        limit: int = 500,
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status=?")
            params.append(status)
        if camera_id:
            clauses.append("camera_id=?")
            params.append(camera_id)
        if level:
            clauses.append("threat_level=?")
            params.append(level)
        if incident_type:
            clauses.append("incident_type=?")
            params.append(incident_type)
        if date_from is not None:
            clauses.append("ts>=?")
            params.append(date_from)
        if date_to is not None:
            clauses.append("ts<=?")
            params.append(date_to)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        return self.query(
            f"SELECT * FROM incidents {where} ORDER BY ts DESC LIMIT ?", params
        )

    def get_incident(self, incident_id: int) -> Optional[sqlite3.Row]:
        return self.query_one("SELECT * FROM incidents WHERE id=?", (incident_id,))

    def count_incidents(self, status: str | None = None) -> int:
        if status:
            return int(self.scalar(
                "SELECT COUNT(*) FROM incidents WHERE status=?", (status,)))
        return int(self.scalar("SELECT COUNT(*) FROM incidents"))

    def highest_open_threat(self) -> str:
        row = self.query_one(
            """
            SELECT threat_level FROM incidents
             WHERE status != 'RESOLVED'
             ORDER BY CASE threat_level
                        WHEN 'CRITICAL' THEN 4 WHEN 'HIGH' THEN 3
                        WHEN 'MEDIUM'   THEN 2 ELSE 1 END DESC,
                      ts DESC
             LIMIT 1
            """
        )
        return row["threat_level"] if row else "LOW"

    # ==================================================================
    # alerts
    # ==================================================================

    def insert_alert(self, alert: Alert) -> int:
        sql, values = self._insert("alerts", alert.to_row())
        alert.db_id = self.execute(sql, values)
        return alert.db_id

    def get_alerts(
        self,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 300,
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status=?")
            params.append(status)
        if severity:
            clauses.append("severity=?")
            params.append(severity)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        return self.query(
            f"SELECT * FROM alerts {where} ORDER BY ts DESC LIMIT ?", params
        )

    def count_active_alerts(self) -> int:
        return int(self.scalar("SELECT COUNT(*) FROM alerts WHERE status='NEW'"))

    # ==================================================================
    # evidence
    # ==================================================================

    def insert_evidence(
        self,
        incident_id: int | None,
        camera_id: str,
        track_number: int | None,
        kind: str,
        file_path: str,
        width: int,
        height: int,
        size_bytes: int,
        ts: float | None = None,
    ) -> int:
        stamp = ts if ts is not None else now_ts()
        return self.execute(
            """
            INSERT INTO evidence (incident_id, camera_id, track_number, kind,
                                  file_path, width, height, bytes, ts, ts_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (incident_id, camera_id, track_number, kind, file_path,
             width, height, size_bytes, stamp, ts_text(stamp)),
        )

    def get_evidence(
        self,
        incident_id: int | None = None,
        camera_id: str | None = None,
        date_from: float | None = None,
        date_to: float | None = None,
        limit: int = 500,
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[Any] = []
        if incident_id is not None:
            clauses.append("incident_id=?")
            params.append(incident_id)
        if camera_id:
            clauses.append("camera_id=?")
            params.append(camera_id)
        if date_from is not None:
            clauses.append("ts>=?")
            params.append(date_from)
        if date_to is not None:
            clauses.append("ts<=?")
            params.append(date_to)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        return self.query(
            f"SELECT * FROM evidence {where} ORDER BY ts DESC LIMIT ?", params
        )

    def get_evidence_for_incidents(self, incident_ids: list[int]) -> list[sqlite3.Row]:
        if not incident_ids:
            return []
        marks = ",".join("?" for _ in incident_ids)
        return self.query(
            f"SELECT * FROM evidence WHERE incident_id IN ({marks}) ORDER BY ts DESC",
            incident_ids,
        )

    def delete_evidence_before(self, cutoff_ts: float) -> list[str]:
        """Remove evidence rows older than *cutoff_ts*, returning their paths."""
        rows = self.query("SELECT file_path FROM evidence WHERE ts < ?", (cutoff_ts,))
        paths = [r["file_path"] for r in rows]
        self.execute("DELETE FROM evidence WHERE ts < ?", (cutoff_ts,))
        return paths

    # ==================================================================
    # plates and watchlist
    # ==================================================================

    def insert_plate(self, plate: PlateRead) -> int:
        sql, values = self._insert("plates", plate.to_row())
        return self.execute(sql, values)

    def get_plates(self, limit: int = 200) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM plates ORDER BY ts DESC LIMIT ?", (limit,))

    def add_watchlist_entry(
        self,
        value: str,
        kind: str = "plate",
        label: str | None = None,
        severity: str = "HIGH",
        notes: str | None = None,
    ) -> int:
        return self.execute(
            """
            INSERT INTO watchlist (kind, value, label, severity, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(kind, value) DO UPDATE SET
                label=excluded.label, severity=excluded.severity,
                notes=excluded.notes, active=1
            """,
            (kind, value.strip().upper(), label, severity, notes, ts_text()),
        )

    def remove_watchlist_entry(self, entry_id: int) -> None:
        self.execute("DELETE FROM watchlist WHERE id=?", (entry_id,))

    def set_watchlist_active(self, entry_id: int, active: bool) -> None:
        self.execute("UPDATE watchlist SET active=? WHERE id=?",
                     (int(active), entry_id))

    def get_watchlist(self, active_only: bool = False) -> list[sqlite3.Row]:
        if active_only:
            return self.query(
                "SELECT * FROM watchlist WHERE active=1 ORDER BY value")
        return self.query("SELECT * FROM watchlist ORDER BY value")

    def record_watchlist_hit(self, value: str, kind: str = "plate") -> None:
        self.execute(
            "UPDATE watchlist SET hit_count=hit_count+1, last_hit_ts=? "
            "WHERE kind=? AND value=?",
            (now_ts(), kind, value.strip().upper()),
        )

    # ==================================================================
    # system logs
    # ==================================================================

    def insert_system_log(self, level: str, logger: str, message: str) -> None:
        ts = now_ts()
        self.execute(
            "INSERT INTO system_logs (ts, ts_text, level, logger, message) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts, ts_text(ts), level, logger, message),
        )

    def get_system_logs(
        self, level: str | None = None, limit: int = 500
    ) -> list[sqlite3.Row]:
        if level and level != "ALL":
            return self.query(
                "SELECT * FROM system_logs WHERE level=? ORDER BY ts DESC LIMIT ?",
                (level, limit),
            )
        return self.query(
            "SELECT * FROM system_logs ORDER BY ts DESC LIMIT ?", (limit,))

    def trim_system_logs(self, keep: int = 20000) -> None:
        self.execute(
            "DELETE FROM system_logs WHERE id NOT IN "
            "(SELECT id FROM system_logs ORDER BY ts DESC LIMIT ?)",
            (keep,),
        )

    # ==================================================================
    # settings
    # ==================================================================

    def set_setting(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "updated_at=excluded.updated_at",
            (key, json.dumps(value), ts_text()),
        )

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value FROM settings WHERE key=?", (key,))
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (TypeError, ValueError):
            return default

    # ==================================================================
    # analytics
    # ==================================================================

    def analytics_incidents_by_day(self, days: int = 14) -> list[tuple[str, int]]:
        rows = self.query(
            """
            SELECT date(ts, 'unixepoch', 'localtime') AS day, COUNT(*) AS n
              FROM incidents
             GROUP BY day ORDER BY day DESC LIMIT ?
            """,
            (days,),
        )
        return [(r["day"], int(r["n"])) for r in reversed(rows)]

    def analytics_incidents_by_camera(self) -> list[tuple[str, int]]:
        rows = self.query(
            "SELECT camera_id, COUNT(*) AS n FROM incidents "
            "GROUP BY camera_id ORDER BY n DESC"
        )
        return [(r["camera_id"], int(r["n"])) for r in rows]

    def analytics_by_level(self) -> dict[str, int]:
        rows = self.query(
            "SELECT threat_level, COUNT(*) AS n FROM incidents GROUP BY threat_level"
        )
        return {r["threat_level"]: int(r["n"]) for r in rows}

    def analytics_by_type(self) -> list[tuple[str, int]]:
        rows = self.query(
            "SELECT incident_type, COUNT(*) AS n FROM incidents "
            "GROUP BY incident_type ORDER BY n DESC"
        )
        return [(r["incident_type"], int(r["n"])) for r in rows]

    def analytics_hourly(self) -> list[tuple[int, int]]:
        rows = self.query(
            """
            SELECT CAST(strftime('%H', ts, 'unixepoch', 'localtime') AS INTEGER) AS hr,
                   COUNT(*) AS n
              FROM incidents GROUP BY hr ORDER BY hr
            """
        )
        counts = {int(r["hr"]): int(r["n"]) for r in rows}
        return [(h, counts.get(h, 0)) for h in range(24)]

    def analytics_busiest_camera(self) -> tuple[str, int] | None:
        by_camera = self.analytics_incidents_by_camera()
        return by_camera[0] if by_camera else None

    def analytics_counts(self) -> dict[str, int]:
        """Headline counters used by the Dashboard and Analytics screens."""
        def count_type(name: str) -> int:
            return int(self.scalar(
                "SELECT COUNT(*) FROM incidents WHERE incident_type=?", (name,)))

        return {
            "incidents": self.count_incidents(),
            "open_incidents": int(self.scalar(
                "SELECT COUNT(*) FROM incidents WHERE status!='RESOLVED'")),
            "active_alerts": self.count_active_alerts(),
            "intrusions": (count_type("restricted_zone_intrusion")
                           + count_type("border_zone_intrusion")),
            "loitering": count_type("loitering"),
            "line_crossings": count_type("line_crossing"),
            "anomalies": count_type("behavioural_anomaly"),
            "vehicle_alerts": count_type("no_vehicle_zone_violation"),
            "watchlist_matches": count_type("watchlist_match"),
            "tracks": int(self.scalar("SELECT COUNT(*) FROM tracks")),
            "people_tracked": int(self.scalar(
                "SELECT COUNT(*) FROM tracks WHERE class_name='person'")),
            "vehicles_tracked": int(self.scalar(
                "SELECT COUNT(*) FROM tracks WHERE class_name IN "
                "('car','truck','bus','motorcycle','bicycle')")),
            "evidence_files": int(self.scalar("SELECT COUNT(*) FROM evidence")),
            "plates_read": int(self.scalar(
                "SELECT COUNT(*) FROM plates WHERE plate_text IS NOT NULL")),
        }
