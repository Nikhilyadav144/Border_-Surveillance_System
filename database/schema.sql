-- ==========================================================================
-- Border Guard AI - SQLite schema
-- --------------------------------------------------------------------------
-- Applied by database/database.py on first run and on every start-up. Every
-- statement is idempotent, so upgrading an existing database is safe.
--
-- Timestamps are stored twice on purpose:
--   ts        REAL   - Unix epoch seconds, for arithmetic and sorting
--   ts_text   TEXT   - ISO-8601 local time, so the file is readable in any
--                      SQLite browser without conversion
-- ==========================================================================

PRAGMA foreign_keys = ON;

-- --------------------------------------------------------------------------
-- schema_version: lets a future release migrate an old database in place
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL
);

-- --------------------------------------------------------------------------
-- cameras
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cameras (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    location       TEXT,
    source_type    TEXT NOT NULL,           -- video | webcam | rtsp
    source_detail  TEXT,                    -- path, index or url
    zones_profile  TEXT DEFAULT 'default',
    enabled        INTEGER NOT NULL DEFAULT 1,
    status         TEXT NOT NULL DEFAULT 'OFFLINE',  -- ONLINE|OFFLINE|ERROR|CONFIGURED
    last_frame_ts  REAL,
    fps            REAL DEFAULT 0,
    latency_ms     REAL DEFAULT 0,
    frames_seen    INTEGER DEFAULT 0,
    error_message  TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

-- --------------------------------------------------------------------------
-- sessions: one processing run of one camera
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id      TEXT NOT NULL,
    started_ts     REAL NOT NULL,
    started_text   TEXT NOT NULL,
    ended_ts       REAL,
    source_detail  TEXT,
    model_path     TEXT,
    device         TEXT,
    frames_read    INTEGER DEFAULT 0,
    frames_processed INTEGER DEFAULT 0,
    detections     INTEGER DEFAULT 0,
    incidents      INTEGER DEFAULT 0,
    notes          TEXT,
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
);

-- --------------------------------------------------------------------------
-- tracks: one row per tracked object identity
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tracks (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER,
    camera_id        TEXT NOT NULL,
    track_key        TEXT NOT NULL,          -- "CAM-01:person:17"
    track_number     INTEGER NOT NULL,       -- the tracker's own id
    class_name       TEXT NOT NULL,
    first_ts         REAL NOT NULL,
    last_ts          REAL NOT NULL,
    first_text       TEXT NOT NULL,
    frames_seen      INTEGER DEFAULT 0,
    avg_confidence   REAL DEFAULT 0,
    max_confidence   REAL DEFAULT 0,
    distance_norm    REAL DEFAULT 0,         -- travelled, fraction of diagonal
    avg_speed        REAL DEFAULT 0,         -- diagonal fractions per second
    max_speed        REAL DEFAULT 0,
    direction_changes INTEGER DEFAULT 0,
    dwell_seconds    REAL DEFAULT 0,
    zone_transitions INTEGER DEFAULT 0,
    zones_visited    TEXT,                   -- JSON list
    restricted_seconds REAL DEFAULT 0,
    anomaly_score    REAL,
    anomaly_status   TEXT DEFAULT 'INSUFFICIENT_DATA',
    trajectory       TEXT,                   -- JSON list of [t, x, y]
    closed           INTEGER DEFAULT 0,
    UNIQUE (camera_id, session_id, track_number, class_name)
);
CREATE INDEX IF NOT EXISTS idx_tracks_camera  ON tracks(camera_id);
CREATE INDEX IF NOT EXISTS idx_tracks_session ON tracks(session_id);
CREATE INDEX IF NOT EXISTS idx_tracks_last    ON tracks(last_ts);

-- --------------------------------------------------------------------------
-- detections: sampled, not every frame. See evidence.* config for why.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS detections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER,
    camera_id    TEXT NOT NULL,
    track_id     INTEGER,
    frame_index  INTEGER NOT NULL,
    ts           REAL NOT NULL,
    ts_text      TEXT NOT NULL,
    class_name   TEXT NOT NULL,
    confidence   REAL NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,      -- normalised 0-1
    zone_name    TEXT,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_detections_ts     ON detections(ts);
CREATE INDEX IF NOT EXISTS idx_detections_camera ON detections(camera_id);

-- --------------------------------------------------------------------------
-- zones: the polygons that were active during a session
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS zones (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT NOT NULL,
    profile     TEXT NOT NULL,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    polygon     TEXT NOT NULL,               -- JSON list of [x, y]
    colour      TEXT,
    updated_at  TEXT NOT NULL,
    UNIQUE (camera_id, profile, name)
);

-- --------------------------------------------------------------------------
-- incidents: the operator-facing record
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS incidents (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     INTEGER,
    camera_id      TEXT NOT NULL,
    track_id       INTEGER,
    track_number   INTEGER,
    object_class   TEXT,
    incident_type  TEXT NOT NULL,            -- intrusion|loitering|line_crossing|...
    threat_score   REAL NOT NULL DEFAULT 0,
    threat_level   TEXT NOT NULL DEFAULT 'LOW',
    reasons        TEXT,                     -- JSON list of {code,label,points}
    zone_name      TEXT,
    detail         TEXT,                     -- JSON blob of type-specific data
    ts             REAL NOT NULL,
    ts_text        TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'NEW',  -- NEW|ACKNOWLEDGED|RESOLVED
    acknowledged_ts REAL,
    acknowledged_by TEXT,
    resolved_ts    REAL,
    resolved_by    TEXT,
    resolution_note TEXT,
    escalation_count INTEGER DEFAULT 0,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_incidents_ts     ON incidents(ts);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_incidents_camera ON incidents(camera_id);
CREATE INDEX IF NOT EXISTS idx_incidents_level  ON incidents(threat_level);
CREATE INDEX IF NOT EXISTS idx_incidents_type   ON incidents(incident_type);

-- --------------------------------------------------------------------------
-- alerts: one per incident, carrying operator workflow state
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id    INTEGER NOT NULL,
    camera_id      TEXT NOT NULL,
    severity       TEXT NOT NULL,
    alert_type     TEXT NOT NULL,
    message        TEXT NOT NULL,
    object_label   TEXT,
    ts             REAL NOT NULL,
    ts_text        TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'NEW',
    acknowledged_ts REAL,
    FOREIGN KEY (incident_id) REFERENCES incidents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts     ON alerts(ts);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);

-- --------------------------------------------------------------------------
-- evidence: files on disk, indexed here
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evidence (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id  INTEGER,
    camera_id    TEXT NOT NULL,
    track_number INTEGER,
    kind         TEXT NOT NULL,              -- full_frame | crop | plate
    file_path    TEXT NOT NULL,
    width        INTEGER,
    height       INTEGER,
    bytes        INTEGER,
    ts           REAL NOT NULL,
    ts_text      TEXT NOT NULL,
    FOREIGN KEY (incident_id) REFERENCES incidents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_evidence_incident ON evidence(incident_id);
CREATE INDEX IF NOT EXISTS idx_evidence_ts       ON evidence(ts);

-- --------------------------------------------------------------------------
-- plates: ANPR results. Never populated with invented text.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS plates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id     TEXT NOT NULL,
    track_number  INTEGER,
    plate_text    TEXT,                      -- NULL when OCR failed
    ocr_confidence REAL,
    validated     INTEGER DEFAULT 0,         -- 1 = matched the plate_regex
    backend       TEXT,                      -- easyocr | tesseract
    evidence_path TEXT,
    watchlist_hit INTEGER DEFAULT 0,
    ts            REAL NOT NULL,
    ts_text       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plates_ts   ON plates(ts);
CREATE INDEX IF NOT EXISTS idx_plates_text ON plates(plate_text);

-- --------------------------------------------------------------------------
-- watchlist: local, operator-managed
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS watchlist (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL DEFAULT 'plate',   -- plate | tag
    value       TEXT NOT NULL,
    label       TEXT,
    severity    TEXT NOT NULL DEFAULT 'HIGH',
    notes       TEXT,
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL,
    hit_count   INTEGER DEFAULT 0,
    last_hit_ts REAL,
    UNIQUE (kind, value)
);

-- --------------------------------------------------------------------------
-- system_logs: mirrors the file log for the System Logs screen
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS system_logs (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    ts_text  TEXT NOT NULL,
    level    TEXT NOT NULL,
    logger   TEXT,
    message  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_ts    ON system_logs(ts);
CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs(level);

-- --------------------------------------------------------------------------
-- settings: runtime key/value overrides set from the GUI
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT NOT NULL
);
