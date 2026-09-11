import { useEffect, useMemo, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

const EMPTY_STATS = {
  persons: 0,
  vehicles: 0,
  tracks: 0,
  detections: 0,
  incidents: 0,
  alerts: 0,
  evidence: 0,
  plates: 0,
  fps: 0,
  inference_ms: 0,
  anomaly: "READY",
  anpr: "READY",
  error: null,
  status: "OFFLINE",
};

function App() {
  const [activeTab, setActiveTab] = useState("Dashboard");

  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);

  const [systemStatus, setSystemStatus] = useState(null);

  // Backend reachability is its OWN fact, decided ONLY by /api/health.
  // It used to be inferred from whether the camera list happened to load,
  // so one slow stats call painted the entire app "Cannot connect to Python
  // backend" while cameras were plainly running. Three separate states now:
  //   backendOnline  - can we reach the API at all
  //   camera.status  - OFFLINE / STARTING / CONNECTING / ONLINE / ERROR
  //   camera.stream  - IDLE / WAITING / LIVE / STALLED
  const [backendOnline, setBackendOnline] = useState(null);

  // Per-camera live track intelligence from /api/cameras/{id}/tracks.
  // Keyed by camera id so one camera's failure cannot blank another's.
  const [tracksByCamera, setTracksByCamera] = useState({});

  const [loadingCamera, setLoadingCamera] = useState(null);

  const [apiError, setApiError] = useState("");

  const selected = useMemo(
    () =>
      cameras.find(
        (camera) => camera.id === selectedCamera
      ),
    [cameras, selectedCamera]
  );

  // ============================================================
  // LOAD CAMERAS + LIVE STATS
  // ============================================================

  const fetchCameras = async () => {
    try {
      const response = await fetch(
        `${API}/api/cameras`
      );

      if (!response.ok) {
        throw new Error("Unable to load cameras");
      }

      const cameraList = await response.json();

      if (!Array.isArray(cameraList)) {
        throw new Error("Invalid camera response");
      }

      const updatedCameras = await Promise.all(
        cameraList.map(async (camera) => {
          try {
            const statsResponse = await fetch(
              `${API}/api/cameras/${camera.id}/stats`
            );

            if (!statsResponse.ok) {
              return {
                ...camera,
                running:
                  camera.status === "ONLINE",
                starting:
                  camera.status === "STARTING" ||
                  camera.status === "CONNECTING",
                stats: EMPTY_STATS,
              };
            }

            const stats =
              await statsResponse.json();

            return {
              ...camera,

              running:
                stats.status === "ONLINE",

              starting:
                stats.status === "STARTING" ||
                stats.status === "CONNECTING",

              stats,
            };
          } catch {
            return {
              ...camera,

              running:
                camera.status === "ONLINE",

              starting:
                camera.status === "STARTING" ||
                camera.status === "CONNECTING",

              stats: EMPTY_STATS,
            };
          }
        })
      );

      setCameras(updatedCameras);

      // Fed the list we just built rather than component state, which has
      // not re-rendered yet at this point.
      fetchTracks(updatedCameras);


      // Automatically select first camera
      setSelectedCamera((current) => {
        if (
          current &&
          updatedCameras.some(
            (camera) => camera.id === current
          )
        ) {
          return current;
        }

        return updatedCameras.length > 0
          ? updatedCameras[0].id
          : null;
      });
    } catch (error) {
      console.error(
        "Camera loading error:",
        error
      );

      // Deliberately NOT setting a global banner. Whether the backend is
      // reachable is decided by /api/health alone; a failure here means the
      // camera list did not load this tick, and the previous list stays on
      // screen rather than the whole app going red.
    }
  };

  // ============================================================
  // SYSTEM HEALTH
  // ============================================================

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch(
        `${API}/api/health`
      );

      if (!response.ok) {
        setBackendOnline(false);
        setSystemStatus(null);
        return;
      }

      const data = await response.json();

      setSystemStatus(data);
      setBackendOnline(true);
      setApiError("");
    } catch {
      setSystemStatus(null);
      setBackendOnline(false);
    }
  };

  // ============================================================
  // LIVE TRACK INTELLIGENCE
  // ============================================================
  //
  // Only asked of cameras that are actually ONLINE. Polling a stopped
  // camera for its people is a request that can only ever return nothing.

  const fetchTracks = async (list) => {
    const live = list.filter(
      (camera) => camera.stats?.status === "ONLINE"
    );

    if (live.length === 0) {
      setTracksByCamera({});
      return;
    }

    const results = await Promise.all(
      live.map(async (camera) => {
        try {
          const response = await fetch(
            `${API}/api/cameras/${camera.id}/tracks`
          );

          if (!response.ok) {
            return [camera.id, []];
          }

          const data = await response.json();

          return [
            camera.id,
            Array.isArray(data.tracks)
              ? data.tracks
              : [],
          ];
        } catch {
          // One camera failing must not blank the others.
          return [camera.id, []];
        }
      })
    );

    setTracksByCamera(
      Object.fromEntries(results)
    );
  };

  // ============================================================
  // POLLING
  // ============================================================

  useEffect(() => {
    fetchCameras();
    fetchSystemStatus();

    const interval = setInterval(() => {
      fetchCameras();
      fetchSystemStatus();
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // ============================================================
  // START CAMERA
  // ============================================================

  const startCamera = async (cameraId) => {
    setLoadingCamera(cameraId);

    try {
      const response = await fetch(
        `${API}/api/cameras/${cameraId}/start`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        throw new Error(
          "Failed to start camera"
        );
      }

      setSelectedCamera(cameraId);

      setApiError("");

      // Give backend a moment to start
      setTimeout(() => {
        fetchCameras();
      }, 800);
    } catch (error) {
      console.error(error);

      setApiError(
        `Unable to start ${cameraId}`
      );
    } finally {
      setTimeout(() => {
        setLoadingCamera(null);
      }, 700);
    }
  };

  // ============================================================
  // STOP CAMERA
  // ============================================================

  const stopCamera = async (cameraId) => {
    setLoadingCamera(cameraId);

    try {
      const response = await fetch(
        `${API}/api/cameras/${cameraId}/stop`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        throw new Error(
          "Failed to stop camera"
        );
      }

      setApiError("");

      setTimeout(() => {
        fetchCameras();
      }, 300);
    } catch (error) {
      console.error(error);

      setApiError(
        `Unable to stop ${cameraId}`
      );
    } finally {
      setTimeout(() => {
        setLoadingCamera(null);
      }, 500);
    }
  };

  // ============================================================
  // STOP ALL CAMERAS
  // ============================================================

  const stopAllCameras = async () => {
    try {
      const activeCameras = cameras.filter(
        (camera) => camera.running
      );

      await Promise.all(
        activeCameras.map((camera) =>
          fetch(
            `${API}/api/cameras/${camera.id}/stop`,
            {
              method: "POST",
            }
          )
        )
      );

      fetchCameras();
    } catch (error) {
      console.error(
        "Stop all error:",
        error
      );
    }
  };

  // ============================================================
  // DASHBOARD CALCULATIONS
  // ============================================================

  const activeCameras = cameras.filter(
    (camera) => camera.running
  ).length;

  const totalPersons = cameras.reduce(
    (total, camera) =>
      total +
      Number(
        camera.stats?.persons || 0
      ),
    0
  );

  const totalVehicles = cameras.reduce(
    (total, camera) =>
      total +
      Number(
        camera.stats?.vehicles || 0
      ),
    0
  );

  const totalAlerts = cameras.reduce(
    (total, camera) =>
      total +
      Number(
        camera.stats?.alerts || 0
      ),
    0
  );

  const totalDetections = cameras.reduce(
    (total, camera) =>
      total +
      Number(
        camera.stats?.detections || 0
      ),
    0
  );

  // ============================================================
  // NAVIGATION
  // ============================================================

  const navigation = [
    {
      name: "Dashboard",
      icon: "⌂",
    },
    {
      name: "Live Monitoring",
      icon: "◉",
    },
    {
      name: "Camera Management",
      icon: "▣",
    },
    {
      name: "Alerts",
      icon: "!",
    },
    {
      name: "Detection Logs",
      icon: "≡",
    },
    {
      name: "Analytics",
      icon: "▥",
    },
    {
      name: "System Settings",
      icon: "⚙",
    },
  ];

  // ============================================================
  // PAGE RENDERER
  // ============================================================

  const renderPage = () => {
    switch (activeTab) {
      case "Live Monitoring":
        return (
          <LiveMonitoring
            cameras={cameras}
            selected={selected}
            selectedCamera={selectedCamera}
            setSelectedCamera={
              setSelectedCamera
            }
            startCamera={startCamera}
            stopCamera={stopCamera}
            loadingCamera={
              loadingCamera
            }
            tracksByCamera={tracksByCamera}
            backendOnline={backendOnline}
          />
        );

      case "Camera Management":
        return (
          <CameraManagement
            cameras={cameras}
            selected={selected}
            selectedCamera={selectedCamera}
            setSelectedCamera={
              setSelectedCamera
            }
            startCamera={startCamera}
            stopCamera={stopCamera}
            loadingCamera={
              loadingCamera
            }
          />
        );

      case "Alerts":
        return (
          <AlertsPage
            cameras={cameras}
          />
        );

      case "Detection Logs":
        return (
          <DetectionLogs
            cameras={cameras}
          />
        );

      case "Analytics":
        return (
          <Analytics
            cameras={cameras}
          />
        );

      case "System Settings":
        return (
          <Settings
            cameras={cameras}
          />
        );

      default:
        return (
          <Dashboard
            cameras={cameras}
            selected={selected}
            activeCameras={
              activeCameras
            }
            totalPersons={
              totalPersons
            }
            totalVehicles={
              totalVehicles
            }
            totalAlerts={
              totalAlerts
            }
            totalDetections={
              totalDetections
            }
            setSelectedCamera={
              setSelectedCamera
            }
            startCamera={startCamera}
            stopCamera={stopCamera}
            loadingCamera={
              loadingCamera
            }
          />
        );
    }
  };

  // ============================================================
  // MAIN UI
  // ============================================================

  return (
    <div className="app-shell">

      {/* ======================================================
          SIDEBAR
      ====================================================== */}

      <aside className="sidebar">

        <div className="brand">

          <div className="brand-logo">
            <img
              src="/logo-mark.png"
              alt="Vision Vigilants"
            />
          </div>

          <div>
            <div className="brand-title">
              VISION<span>VIGILANTS</span>
            </div>

            {/* Kept short on purpose: at 8px with 2.4px letter-spacing
                anything longer wraps to a second line in a 270px sidebar. */}
            <div className="brand-subtitle">
              IBVAP SURVEILLANCE
            </div>
          </div>

        </div>

        <div className="sidebar-divider" />

        <nav className="navigation">

          {navigation.map((item) => (
            <button
              key={item.name}
              className={`nav-item ${
                activeTab === item.name
                  ? "active"
                  : ""
              }`}
              onClick={() =>
                setActiveTab(item.name)
              }
            >
              <span className="nav-icon">
                {item.icon}
              </span>

              <span>
                {item.name}
              </span>
            </button>
          ))}

        </nav>

        <div className="sidebar-bottom">

          <div className="system-indicator">

            <span className="pulse-dot" />

            <div>

              <div className="indicator-title">
                SYSTEM ONLINE
              </div>

              <div className="indicator-text">
                AI Engine Operational
              </div>

            </div>

          </div>

          <button
            className="shutdown-button"
            onClick={
              stopAllCameras
            }
          >
            STOP ALL CAMERAS
          </button>

        </div>

      </aside>

      {/* ======================================================
          MAIN AREA
      ====================================================== */}

      <main className="main-content">

        <header className="topbar">

          <div>

            <div className="topbar-label">
              BORDER SECURITY COMMAND CENTER
            </div>

            <h1>
              {activeTab}
            </h1>

          </div>

          <div className="topbar-right">

            <div className="topbar-time">
              {new Date().toLocaleTimeString(
                "en-IN",
                {
                  hour12: false,
                }
              )}
            </div>

            <div
              className={
                backendOnline === false
                  ? "topbar-status is-down"
                  : "topbar-status"
              }
            >

              {backendOnline !== false && (
                <span className="pulse-dot" />
              )}

              {backendOnline === null
                ? "CONNECTING"
                : backendOnline
                ? "BACKEND ONLINE"
                : "BACKEND DOWN"}

            </div>

          </div>

        </header>

        {backendOnline === false && (
          <div className="api-error">
            Backend not reachable at {API}. The dashboard is showing the
            last values it received.
          </div>
        )}

        {backendOnline !== false && apiError && (
          <div className="api-warning">
            {apiError}
          </div>
        )}

        {backendOnline !== false &&
          cameras
            .filter((camera) => camera.stats?.error)
            .map((camera) => (
              <div className="api-warning" key={camera.id}>
                {camera.id}: {camera.stats.error}
              </div>
            ))}

        {renderPage()}

      </main>

    </div>
  );
}


// ============================================================
// DASHBOARD
// ============================================================

function Dashboard({
  cameras,
  selected,
  activeCameras,
  totalPersons,
  totalVehicles,
  totalAlerts,
  totalDetections,
  setSelectedCamera,
  startCamera,
  stopCamera,
  loadingCamera,
}) {
  return (
    <div className="page">

      <div className="page-intro">

        <div>

          <div className="section-label">
            SURVEILLANCE OVERVIEW
          </div>

          <h2>
            AI Command Dashboard
          </h2>

          <p>
            Real-time monitoring of the
            intelligent border surveillance
            network.
          </p>

        </div>

      </div>

      {/* ======================================================
          STAT CARDS
      ====================================================== */}

      <div className="dashboard-stats">

        <DashboardStat
          label="ACTIVE CAMERAS"
          value={activeCameras}
          detail={`${cameras.length} configured`}
        />

        <DashboardStat
          label="PERSONS"
          value={totalPersons}
          detail="Live detection"
        />

        <DashboardStat
          label="VEHICLES"
          value={totalVehicles}
          detail="Live detection"
        />

        <DashboardStat
          label="ALERTS"
          value={totalAlerts}
          detail={`${totalDetections} detections`}
        />

      </div>

      {/* ======================================================
          CAMERA NETWORK
      ====================================================== */}

      <section className="panel">

        <div className="panel-header">

          <div>

            <div className="section-label">
              SURVEILLANCE NETWORK
            </div>

            <h3>
              Camera Nodes
            </h3>

          </div>

          <div className="panel-count">
            {activeCameras} ACTIVE /{" "}
            {cameras.length} TOTAL
          </div>

        </div>

        <div className="camera-grid">

          {cameras.map((camera) => (

            <CameraCard
              key={camera.id}
              camera={camera}
              selected={
                selected?.id === camera.id
              }
              onSelect={() =>
                setSelectedCamera(
                  camera.id
                )
              }
              onStart={() =>
                startCamera(
                  camera.id
                )
              }
              onStop={() =>
                stopCamera(
                  camera.id
                )
              }
              loading={
                loadingCamera ===
                camera.id
              }
            />

          ))}

        </div>

      </section>

      {/* ======================================================
          LIVE CAMERA
      ====================================================== */}

      <section className="panel">

        <div className="panel-header">

          <div>

            <div className="section-label">
              LIVE FEED
            </div>

            <h3>
              {selected?.name ||
                "No Camera Selected"}
            </h3>

          </div>

          {selected?.running && (
            <div className="live-badge">
              <span className="pulse-dot" />
              LIVE
            </div>
          )}

        </div>

        <CameraFeed
          camera={selected}
        />

      </section>

    </div>
  );
}


// ============================================================
// DASHBOARD STAT
// ============================================================

function DashboardStat({
  label,
  value,
  detail,
}) {
  return (
    <div className="stat-card">

      <span className="stat-label">
        {label}
      </span>

      <strong className="stat-value">
        {value}
      </strong>

      <small className="stat-detail">
        {detail}
      </small>

    </div>
  );
}


// ============================================================
// CAMERA MANAGEMENT
// ============================================================

function CameraManagement({
  cameras,
  selected,
  selectedCamera,
  setSelectedCamera,
  startCamera,
  stopCamera,
  loadingCamera,
}) {
  const activeCount =
    cameras.filter(
      (camera) => camera.running
    ).length;

  return (
    <div className="page">

      <div className="page-intro">

        <div>

          <div className="section-label">
            SURVEILLANCE NETWORK
          </div>

          <h2>
            Camera Management
          </h2>

          <p>
            Manage connected surveillance
            sources and monitor their
            real-time AI processing status.
          </p>

        </div>

        <div className="big-counter">

          {activeCount}

          <span>
            ACTIVE
          </span>

        </div>

        <div className="big-counter">

          {cameras.length}

          <span>
            TOTAL
          </span>

        </div>

      </div>

      <div className="management-layout">

        {/* CAMERA LIST */}

        <section className="panel camera-list-panel">

          <div className="panel-header">

            <div>

              <div className="section-label">
                CAMERA NODES
              </div>

              <h3>
                Connected Cameras
              </h3>

            </div>

          </div>

          <div className="management-list">

            {cameras.map((camera) => (

              <CameraManagementCard
                key={camera.id}
                camera={camera}
                selected={
                  selectedCamera ===
                  camera.id
                }
                onSelect={() =>
                  setSelectedCamera(
                    camera.id
                  )
                }
                onStart={() =>
                  startCamera(
                    camera.id
                  )
                }
                onStop={() =>
                  stopCamera(
                    camera.id
                  )
                }
                loading={
                  loadingCamera ===
                  camera.id
                }
              />

            ))}

          </div>

        </section>

        {/* SELECTED CAMERA */}

        <section className="panel selected-camera-panel">

          <div className="panel-header">

            <div>

              <div className="section-label">
                SELECTED NODE
              </div>

              <h3>
                {selected?.id ||
                  "NO CAMERA"}
              </h3>

            </div>

            {selected?.running && (
              <div className="live-badge">

                <span className="pulse-dot" />

                LIVE

              </div>
            )}

          </div>

          <CameraFeed
            camera={selected}
          />

          {selected && (
            <div className="selected-info">

              <div>
                <span>NAME</span>

                <strong>
                  {selected.name}
                </strong>
              </div>

              <div>
                <span>LOCATION</span>

                <strong>
                  {selected.location}
                </strong>
              </div>

              <div>
                <span>SOURCE</span>

                <strong>
                  {formatSource(
                    selected.source
                  )}
                </strong>
              </div>

            </div>
          )}

        </section>

      </div>

    </div>
  );
}


// ============================================================
// LIVE MONITORING
// ============================================================

function LiveMonitoring({
  cameras,
  selected,
  selectedCamera,
  setSelectedCamera,
  startCamera,
  stopCamera,
  loadingCamera,
  tracksByCamera,
  backendOnline,
}) {
  // Chrome allows roughly six simultaneous connections per origin over
  // HTTP/1.1, and an MJPEG feed holds one open permanently. Past that the
  // ordinary polling requests queue behind the video and never run, which
  // looks exactly like the backend having died. Four tiles leaves headroom
  // for the stats and health polls.
  const MAX_TILE_STREAMS = 4;

  const live = cameras.filter(
    (camera) => camera.stats?.status === "ONLINE"
  );

  const streaming = live.slice(0, MAX_TILE_STREAMS);

  const [expanded, setExpanded] = useState(null);

  const expandedCamera = cameras.find(
    (camera) => camera.id === expanded
  );

  if (expandedCamera) {
    return (
      <ExpandedCamera
        camera={expandedCamera}
        tracks={tracksByCamera[expandedCamera.id] || []}
        onClose={() => setExpanded(null)}
        startCamera={startCamera}
        stopCamera={stopCamera}
        loadingCamera={loadingCamera}
      />
    );
  }

  return (
    <div className="page">

      <div className="wall-toolbar">

        <div>
          <div className="section-label">
            REAL-TIME SURVEILLANCE
          </div>
          <h2>Surveillance Wall</h2>
        </div>

        <div className="wall-counts">
          <span className="wall-count">
            <strong>{live.length}</strong> LIVE
          </span>
          <span className="wall-count muted">
            <strong>{cameras.length}</strong> CONFIGURED
          </span>
        </div>

      </div>

      {live.length > MAX_TILE_STREAMS && (
        <div className="wall-notice">
          Showing video for the first {MAX_TILE_STREAMS} live cameras.
          A browser holds only about six connections per origin and each
          feed keeps one open, so beyond this the status polling would
          stall behind the video.
        </div>
      )}

      {cameras.length === 0 && (
        <div className="wall-empty">
          {backendOnline === false
            ? "No camera list. The backend is not answering."
            : "No cameras configured."}
        </div>
      )}

      <div className="camera-wall">
        {cameras.map((camera) => (
          <WallTile
            key={camera.id}
            camera={camera}
            tracks={tracksByCamera[camera.id] || []}
            streamAllowed={streaming.some(
              (c) => c.id === camera.id
            )}
            onExpand={() => {
              setSelectedCamera(camera.id);
              setExpanded(camera.id);
            }}
            onStart={() => startCamera(camera.id)}
            onStop={() => stopCamera(camera.id)}
            loading={loadingCamera === camera.id}
          />
        ))}
      </div>

    </div>
  );
}


// ============================================================
// CAMERA STATUS PILL
// ============================================================
//
// Camera state and stream state are different questions and get
// different pills. A camera can be ONLINE and its picture not yet
// flowing; showing one value for both is how "it says online but I see
// nothing" happens.

function StatusPill({ status }) {
  const value = status || "UNKNOWN";
  return (
    <span className={`pill pill-${value.toLowerCase()}`}>
      {(value === "ONLINE" || value === "LIVE") && (
        <span className="pulse-dot" />
      )}
      {value}
    </span>
  );
}


// ============================================================
// ONE WALL TILE
// ============================================================

function WallTile({
  camera,
  tracks,
  streamAllowed,
  onExpand,
  onStart,
  onStop,
  loading,
}) {
  const stats = camera.stats || EMPTY_STATS;
  const status = stats.status || "OFFLINE";
  const stream = stats.stream || "IDLE";
  const online = status === "ONLINE";

  // The <img> is mounted ONLY for a camera the backend says is running.
  // Loading a stream must never be what starts a camera; the lifecycle
  // belongs to the backend and the Start button.
  const showVideo = online && streamAllowed;

  return (
    <article
      className={`wall-tile ${online ? "is-live" : "is-idle"}`}
      onClick={onExpand}
    >

      <header className="tile-head">
        <div className="tile-id">
          <strong>{camera.id}</strong>
          <span>{camera.location || "—"}</span>
        </div>
        <StatusPill status={status} />
      </header>

      <div className="tile-video">
        {showVideo ? (
          <img
            src={`${API}/api/cameras/${camera.id}/stream`}
            alt={`${camera.id} live feed`}
          />
        ) : (
          <div className="tile-placeholder">
            <span className="tile-placeholder-icon">
              {status === "ERROR" ? "!" : "\u25cc"}
            </span>
            <span>
              {status === "ERROR"
                ? stats.error || "CAMERA ERROR"
                : online
                ? "STREAM CAPPED"
                : status}
            </span>
          </div>
        )}

        {showVideo && stream !== "LIVE" && (
          <div className="tile-stream-state">{stream}</div>
        )}

        {tracks.length > 0 && (
          <div className="tile-tracks">
            {tracks.slice(0, 3).map((track) => (
              <TrackChip
                key={track.track_id}
                track={track}
              />
            ))}
            {tracks.length > 3 && (
              <span className="track-chip more">
                +{tracks.length - 3}
              </span>
            )}
          </div>
        )}
      </div>

      <footer className="tile-foot">
        <span>
          PERSONS <strong>{stats.persons ?? 0}</strong>
        </span>
        <span>
          FPS <strong>{Number(stats.fps || 0).toFixed(1)}</strong>
        </span>
        <span>
          {stats.inference_ms
            ? `${Math.round(stats.inference_ms)} MS`
            : "— MS"}
        </span>

        <button
          className={online ? "stop-btn tiny" : "start-btn tiny"}
          disabled={loading}
          onClick={(event) => {
            event.stopPropagation();
            online ? onStop() : onStart();
          }}
        >
          {loading
            ? "..."
            : online
            ? "STOP"
            : "START"}
        </button>
      </footer>

    </article>
  );
}


// ============================================================
// PERSON CHIP
// ============================================================
//
// Every value comes from /api/cameras/{id}/tracks. A field the pipeline
// has not concluded shows its own honest state rather than a default:
// behaviour reads UNKNOWN while the movement layer is still gathering
// pose samples, and threat is absent entirely until the ThreatEngine has
// actually scored that person.

function TrackChip({ track }) {
  const threat = track.threat_level;
  const behaviour =
    track.behaviour && track.behaviour !== "UNKNOWN"
      ? track.behaviour
      : track.pose_status === "UNAVAILABLE"
      ? "NO POSE"
      : "ANALYSING";

  return (
    <span
      className={`track-chip ${
        threat ? `threat-${threat.toLowerCase()}` : ""
      }`}
    >
      <b>#{String(track.track_id).padStart(2, "0")}</b>
      {behaviour}
      {threat && <i>{threat}</i>}
    </span>
  );
}


// ============================================================
// EXPANDED CAMERA
// ============================================================

function ExpandedCamera({
  camera,
  tracks,
  onClose,
  startCamera,
  stopCamera,
  loadingCamera,
}) {
  const stats = camera.stats || EMPTY_STATS;
  const online = stats.status === "ONLINE";

  const [focus, setFocus] = useState(null);

  const focused =
    tracks.find((t) => t.track_id === focus) || tracks[0] || null;

  return (
    <div className="page">

      <div className="wall-toolbar">
        <div>
          <div className="section-label">
            {camera.location || "CAMERA"}
          </div>
          <h2>
            {camera.id} — {camera.name}
          </h2>
        </div>

        <div className="wall-counts">
          <StatusPill status={stats.status} />
          <StatusPill status={stats.stream} />
          <button className="ghost-btn" onClick={onClose}>
            BACK TO WALL
          </button>
        </div>
      </div>

      <div className="expanded-layout">

        <section className="panel expanded-feed">
          <CameraFeed camera={camera} />

          <div className="camera-control-row">
            {online ? (
              <button
                className="stop-btn"
                onClick={() => stopCamera(camera.id)}
                disabled={loadingCamera === camera.id}
              >
                {loadingCamera === camera.id
                  ? "STOPPING..."
                  : "STOP CAMERA"}
              </button>
            ) : (
              <button
                className="start-btn"
                onClick={() => startCamera(camera.id)}
                disabled={loadingCamera === camera.id}
              >
                {loadingCamera === camera.id
                  ? "STARTING..."
                  : "START CAMERA"}
              </button>
            )}
          </div>
        </section>

        <section className="panel track-panel">
          <div className="panel-header">
            <div>
              <div className="section-label">
                TRACK INTELLIGENCE
              </div>
              <h3>
                {tracks.length}{" "}
                {tracks.length === 1 ? "PERSON" : "PEOPLE"} IN VIEW
              </h3>
            </div>
          </div>

          {tracks.length === 0 && (
            <div className="track-empty">
              {online
                ? "No confirmed person tracks right now."
                : "Camera is not running."}
            </div>
          )}

          <div className="track-list">
            {tracks.map((track) => (
              <button
                key={track.track_id}
                className={`track-row ${
                  focused && focused.track_id === track.track_id
                    ? "is-focused"
                    : ""
                }`}
                onClick={() => setFocus(track.track_id)}
              >
                <TrackChip track={track} />
                <span className="track-row-meta">
                  {track.direction || "STATIONARY"}
                </span>
              </button>
            ))}
          </div>

          {focused && <TrackDetail track={focused} />}
        </section>

      </div>

    </div>
  );
}


// ============================================================
// TRACK DETAIL
// ============================================================
//
// Only fields the backend actually sends. Anything the pipeline has not
// established renders as its own state word, never as a plausible value.

function TrackDetail({ track }) {
  const rows = [
    ["Track ID", `#${track.track_id}`],
    ["Detection", `person ${track.detection_confidence ?? "—"}`],
    [
      "Behaviour",
      track.behaviour === "UNKNOWN"
        ? "UNKNOWN"
        : `${track.behaviour} ${
            track.confidence ? `(${Math.round(track.confidence * 100)}%)` : ""
          }`,
    ],
    ["Movement", track.direction || "STATIONARY"],
    [
      "Pose",
      track.pose_status === "TRACKED"
        ? `TRACKED (${track.keypoints}/17 keypoints)`
        : track.pose_status,
    ],
    ["Posture", track.posture || "NOT AVAILABLE"],
    [
      "Boundary",
      track.boundary_state === "NONE"
        ? "NONE"
        : `${track.boundary_state}${
            track.boundary_line ? ` — ${track.boundary_line}` : ""
          }`,
    ],
    ["Zone", track.zone || "NONE"],
    ["Crossings", track.crossings ?? 0],
    ["Anomaly", track.anomaly_status || "NOT AVAILABLE"],
    [
      "Threat",
      track.threat_level
        ? `${track.threat_level} (${track.threat_score})`
        : "NOT ASSESSED",
    ],
  ];

  return (
    <div className="track-detail">
      <dl>
        {rows.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>

      {track.threat_reasons && track.threat_reasons.length > 0 && (
        <div className="track-reasons">
          <div className="section-label">WHY</div>
          <ul>
            {track.threat_reasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}


// ============================================================
// CAMERA CARD
// ============================================================

function CameraCard({
  camera,
  selected,
  onSelect,
  onStart,
  onStop,
  loading,
}) {
  return (
    <div
      className={`camera-card ${
        selected
          ? "selected"
          : ""
      }`}
      onClick={onSelect}
    >

      <div className="camera-card-top">

        <div className="camera-icon">
          ▣
        </div>

        <div className="camera-status">

          <span
            className={`status-dot ${
              camera.running
                ? "running"
                : camera.starting
                ? "starting"
                : ""
            }`}
          />

          {camera.running
            ? "ONLINE"
            : camera.starting
            ? "STARTING"
            : "OFFLINE"}

        </div>

      </div>

      <div className="camera-id">
        {camera.id}
      </div>

      <div className="camera-name">
        {camera.name}
      </div>

      <div className="camera-location">
        {camera.location}
      </div>

      <div className="camera-actions">

        {camera.running ? (

          <button
            className="stop-btn"
            onClick={(event) => {
              event.stopPropagation();
              onStop();
            }}
            disabled={loading}
          >
            {loading
              ? "..."
              : "STOP"}
          </button>

        ) : (

          <button
            className="start-btn"
            onClick={(event) => {
              event.stopPropagation();
              onStart();
            }}
            disabled={loading}
          >
            {loading
              ? "..."
              : "START"}
          </button>

        )}

      </div>

    </div>
  );
}


// ============================================================
// CAMERA MANAGEMENT CARD
// ============================================================

function CameraManagementCard({
  camera,
  selected,
  onSelect,
  onStart,
  onStop,
  loading,
}) {
  return (
    <div
      className={`management-camera ${
        selected
          ? "selected"
          : ""
      }`}
      onClick={onSelect}
    >

      <div className="management-camera-icon">
        ▣
      </div>

      <div className="management-camera-info">

        <div className="management-camera-heading">

          <strong>
            {camera.id}
          </strong>

          <span
            className={`status-label ${
              camera.running
                ? "online"
                : camera.starting
                ? "starting"
                : "offline"
            }`}
          >

            <span className="status-dot" />

            {camera.running
              ? "ONLINE"
              : camera.starting
              ? "STARTING"
              : "OFFLINE"}

          </span>

        </div>

        <div className="management-camera-name">
          {camera.name}
        </div>

        <div className="management-camera-location">
          {camera.location}
        </div>

      </div>

      <div>

        {camera.running ? (

          <button
            className="stop-btn"
            onClick={(event) => {
              event.stopPropagation();
              onStop();
            }}
            disabled={loading}
          >
            {loading
              ? "..."
              : "STOP"}
          </button>

        ) : (

          <button
            className="start-btn"
            onClick={(event) => {
              event.stopPropagation();
              onStart();
            }}
            disabled={loading}
          >
            {loading
              ? "..."
              : "START"}
          </button>

        )}

      </div>

    </div>
  );
}


// ============================================================
// CAMERA FEED
// ============================================================

function CameraFeed({ camera }) {

  if (!camera) {
    return (
      <div className="feed-placeholder">

        <div className="placeholder-icon">
          ▣
        </div>

        <strong>
          NO CAMERA SELECTED
        </strong>

        <span>
          Select a camera from the
          surveillance network.
        </span>

      </div>
    );
  }

  if (!camera.running) {
    return (
      <div className="feed-placeholder">

        <div className="placeholder-icon">
          ◌
        </div>

        <strong>
          CAMERA OFFLINE
        </strong>

        <span>
          Start {camera.id} to begin
          live AI surveillance.
        </span>

      </div>
    );
  }

  return (
    <div className="camera-feed">

      <img
        src={`${API}/api/cameras/${camera.id}/stream`}
        alt={`${camera.id} live surveillance feed`}
      />

      <div className="feed-overlay top-left">
        {camera.id}
      </div>

      <div className="feed-overlay top-right">
        AI PROCESSING
      </div>

      <div className="feed-overlay bottom-left">
        {camera.location}
      </div>

      <div className="feed-overlay bottom-right">
        {Number(
          camera.stats?.fps || 0
        ).toFixed(1)}{" "}
        FPS
      </div>

    </div>
  );
}


// ============================================================
// STATS GRID
// ============================================================

function StatsGrid({ stats }) {

  const metrics = [
    [
      "PERSONS",
      stats.persons,
      "human",
    ],
    [
      "VEHICLES",
      stats.vehicles,
      "vehicle",
    ],
    [
      "TRACKS",
      stats.tracks,
      "track",
    ],
    [
      "DETECTIONS",
      stats.detections,
      "detect",
    ],
    [
      "INCIDENTS",
      stats.incidents,
      "incident",
    ],
    [
      "ALERTS",
      stats.alerts,
      "alert",
    ],
    [
      "PLATES",
      stats.plates,
      "plate",
    ],
    [
      "INFERENCE",
      `${Number(
        stats.inference_ms || 0
      ).toFixed(1)} ms`,
      "speed",
    ],
  ];

  return (
    <div className="stats-grid">

      {metrics.map(
        ([label, value, type]) => (

          <div
            className={`metric-card ${type}`}
            key={label}
          >

            <div className="metric-label">
              {label}
            </div>

            <div className="metric-value">
              {value}
            </div>

          </div>

        )
      )}

      <div className="status-metric">

        <span>
          ANOMALY ENGINE
        </span>

        <strong>
          {stats.anomaly ||
            "NO DATA"}
        </strong>

      </div>

      <div className="status-metric">

        <span>
          ANPR ENGINE
        </span>

        <strong>
          {stats.anpr ||
            "NO DATA"}
        </strong>

      </div>

      {/* Scene light level, measured from the video frames by the
          backend. Never inferred in the browser. */}
      <div className="status-metric">

        <span>
          SCENE LIGHT
        </span>

        <strong>
          {stats.night_context ||
            "NO DATA"}
        </strong>

      </div>

      {/* Pose. The count is tracked people currently carrying keypoints,
          straight from the backend. Landmarks only: posture is geometry,
          not a behaviour verdict. */}
      <div className="status-metric">

        <span>
          POSE ENGINE
        </span>

        <strong>
          {stats.pose ||
            "NO DATA"}
        </strong>

      </div>

      <div className="status-metric">

        <span>
          TRACKS WITH POSE
        </span>

        <strong>
          {stats.poses === undefined ||
          stats.poses === null
            ? "NO DATA"
            : `${stats.poses}`}
        </strong>

      </div>

      {/* Security geometry configured for THIS camera. Optional: a camera
          with no polygon reports "No geometry configured" and cannot raise
          zone or line incidents. The AI pipeline runs either way. */}
      <div className="status-metric">

        <span>
          ZONE GEOMETRY
        </span>

        <strong>
          {stats.geometry ||
            "NO DATA"}
        </strong>

      </div>

    </div>
  );
}


// ============================================================
// ALERTS
// ============================================================

function AlertsPage({ cameras }) {

  const totalAlerts =
    cameras.reduce(
      (total, camera) =>
        total +
        Number(
          camera.stats?.alerts || 0
        ),
      0
    );

  const totalIncidents =
    cameras.reduce(
      (total, camera) =>
        total +
        Number(
          camera.stats?.incidents ||
            0
        ),
      0
    );

  return (
    <div className="page">

      <div className="page-intro">

        <div>

          <div className="section-label">
            SECURITY EVENTS
          </div>

          <h2>
            Alerts & Incidents
          </h2>

          <p>
            Real-time security events
            generated by the AI
            surveillance pipeline.
          </p>

        </div>

        <div className="big-counter">

          {totalAlerts}

          <span>
            ALERTS
          </span>

        </div>

        <div className="big-counter">

          {totalIncidents}

          <span>
            INCIDENTS
          </span>

        </div>

      </div>

      <section className="panel">

        <div className="panel-header">

          <div>

            <div className="section-label">
              EVENT MONITOR
            </div>

            <h3>
              Camera Alert Status
            </h3>

          </div>

        </div>

        <div className="alert-list">

          {cameras.map(
            (camera) => (

              <div
                className="alert-row"
                key={camera.id}
              >

                <div className="alert-severity">

                  {Number(
                    camera.stats?.alerts ||
                      0
                  ) > 0
                    ? "!"
                    : "✓"}

                </div>

                <div className="alert-info">

                  <strong>
                    {camera.id}
                  </strong>

                  <span>
                    {Number(
                      camera.stats?.alerts ||
                        0
                    )}{" "}
                    alerts detected
                  </span>

                </div>

                <div className="alert-count">
                  {Number(
                    camera.stats?.incidents ||
                      0
                  )}
                </div>

              </div>

            )
          )}

        </div>

      </section>

    </div>
  );
}


// ============================================================
// DETECTION LOGS
// ============================================================

function DetectionLogs({
  cameras,
}) {
  return (
    <div className="page">

      <div className="page-intro">

        <div>

          <div className="section-label">
            AI ACTIVITY
          </div>

          <h2>
            Detection Logs
          </h2>

          <p>
            Current detection statistics
            from all camera processing
            nodes.
          </p>

        </div>

      </div>

      <section className="panel">

        <div className="table-container">

          <table>

            <thead>

              <tr>
                <th>CAMERA</th>
                <th>PERSONS</th>
                <th>VEHICLES</th>
                <th>TRACKS</th>
                <th>DETECTIONS</th>
                <th>FPS</th>
                <th>INFERENCE</th>
              </tr>

            </thead>

            <tbody>

              {cameras.map(
                (camera) => (

                  <tr
                    key={camera.id}
                  >

                    <td>

                      <strong>
                        {camera.id}
                      </strong>

                      <small>
                        {camera.name}
                      </small>

                    </td>

                    <td>
                      {camera.stats?.persons ||
                        0}
                    </td>

                    <td>
                      {camera.stats?.vehicles ||
                        0}
                    </td>

                    <td>
                      {camera.stats?.tracks ||
                        0}
                    </td>

                    <td>
                      {camera.stats?.detections ||
                        0}
                    </td>

                    <td>
                      {Number(
                        camera.stats?.fps ||
                          0
                      ).toFixed(1)}
                    </td>

                    <td>

                      {Number(
                        camera.stats
                          ?.inference_ms ||
                          0
                      ).toFixed(1)}

                      ms

                    </td>

                  </tr>

                )
              )}

            </tbody>

          </table>

        </div>

      </section>

    </div>
  );
}


// ============================================================
// ANALYTICS
// ============================================================

function Analytics({
  cameras,
}) {
  const totalPersons =
    cameras.reduce(
      (sum, camera) =>
        sum +
        Number(
          camera.stats?.persons ||
            0
        ),
      0
    );

  const totalVehicles =
    cameras.reduce(
      (sum, camera) =>
        sum +
        Number(
          camera.stats?.vehicles ||
            0
        ),
      0
    );

  const totalDetections =
    cameras.reduce(
      (sum, camera) =>
        sum +
        Number(
          camera.stats?.detections ||
            0
        ),
      0
    );

  const activeCameras =
    cameras.filter(
      (camera) =>
        camera.running
    ).length;

  return (
    <div className="page">

      <div className="page-intro">

        <div>

          <div className="section-label">
            OPERATIONAL ANALYTICS
          </div>

          <h2>
            Surveillance Analytics
          </h2>

          <p>
            Aggregated AI telemetry
            across the surveillance
            network.
          </p>

        </div>

      </div>

      <div className="analytics-cards">

        <AnalyticsCard
          title="PERSONS DETECTED"
          value={totalPersons}
        />

        <AnalyticsCard
          title="VEHICLES DETECTED"
          value={totalVehicles}
        />

        <AnalyticsCard
          title="TOTAL DETECTIONS"
          value={totalDetections}
        />

        <AnalyticsCard
          title="ACTIVE CAMERAS"
          value={activeCameras}
        />

      </div>

      <section className="panel">

        <div className="panel-header">

          <div>

            <div className="section-label">
              NETWORK PERFORMANCE
            </div>

            <h3>
              Camera Performance
            </h3>

          </div>

        </div>

        <div className="performance-list">

          {cameras.map(
            (camera) => (

              <div
                className="performance-row"
                key={camera.id}
              >

                <div className="performance-name">

                  <strong>
                    {camera.id}
                  </strong>

                  <span>
                    {camera.name}
                  </span>

                </div>

                <div className="performance-stat">

                  <span>
                    FPS
                  </span>

                  <strong>
                    {Number(
                      camera.stats?.fps ||
                        0
                    ).toFixed(1)}
                  </strong>

                </div>

                <div className="performance-stat">

                  <span>
                    INFERENCE
                  </span>

                  <strong>

                    {Number(
                      camera.stats
                        ?.inference_ms ||
                        0
                    ).toFixed(1)}

                    ms

                  </strong>

                </div>

                <div className="performance-stat">

                  <span>
                    DETECTIONS
                  </span>

                  <strong>
                    {Number(
                      camera.stats
                        ?.detections ||
                        0
                    )}
                  </strong>

                </div>

              </div>

            )
          )}

        </div>

      </section>

    </div>
  );
}


// ============================================================
// ANALYTICS CARD
// ============================================================

function AnalyticsCard({
  title,
  value,
}) {
  return (
    <div className="analytics-card">

      <span>
        {title}
      </span>

      <strong>
        {value}
      </strong>

      <small>
        REAL-TIME
      </small>

    </div>
  );
}


// ============================================================
// SETTINGS
// ============================================================

function Settings({
  cameras,
}) {

  // Engine status is derived from what the running cameras actually report.
  // With every camera stopped there is no live state to show, and these rows
  // say so instead of claiming ACTIVE.
  const live =
    cameras.filter(
      (camera) =>
        camera.stats?.status === "ONLINE"
    );

  const first = live[0];

  const engineStatus = {
    online: live.length > 0,
    tracking: live.length
      ? `ACTIVE (${live.length} camera${
          live.length === 1 ? "" : "s"
        })`
      : "IDLE - no camera running",
    anomaly:
      first?.stats?.anomaly ||
      "IDLE - no camera running",
    anpr:
      first?.stats?.anpr ||
      "IDLE - no camera running",
    lowLight:
      first?.stats?.night_context ||
      "IDLE - no camera running",
    pose:
      first?.stats?.pose ||
      "IDLE - no camera running",
    geometry:
      first?.stats?.geometry ||
      "IDLE - no camera running",
  };

  return (
    <div className="page">

      <div className="page-intro">

        <div>

          <div className="section-label">
            SYSTEM CONFIGURATION
          </div>

          <h2>
            System Settings
          </h2>

          <p>
            Current surveillance system
            configuration.
          </p>

        </div>

      </div>

      <div className="settings-grid">

        <section className="panel">

          <div className="panel-header">

            <div>

              <div className="section-label">
                AI ENGINE
              </div>

              <h3>
                Detection Configuration
              </h3>

            </div>

          </div>

          <SettingRow
            label="Object Detection"
            value="YOLO"
            active
          />

          {/* These read live backend state. A hard-coded "ACTIVE" would
              still say ACTIVE with every camera stopped. */}
          <SettingRow
            label="Tracking"
            value={engineStatus.tracking}
            active={engineStatus.online}
          />

          <SettingRow
            label="ANPR"
            value={engineStatus.anpr}
          />

          <SettingRow
            label="Anomaly Detection"
            value={engineStatus.anomaly}
            active={engineStatus.online}
          />

          <SettingRow
            label="Low-light Detection"
            value={engineStatus.lowLight}
            active={engineStatus.online}
          />

          <SettingRow
            label="Pose Estimation"
            value={engineStatus.pose}
            active={engineStatus.online}
          />

          <SettingRow
            label="Zone Geometry"
            value={engineStatus.geometry}
          />

        </section>

        <section className="panel">

          <div className="panel-header">

            <div>

              <div className="section-label">
                NETWORK
              </div>

              <h3>
                Camera Network
              </h3>

            </div>

          </div>

          <SettingRow
            label="Configured Cameras"
            value={cameras.length}
            active
          />

          <SettingRow
            label="Active Cameras"
            value={
              cameras.filter(
                (camera) =>
                  camera.running
              ).length
            }
            active
          />

          <SettingRow
            label="API"
            value="FASTAPI"
            active
          />

          <SettingRow
            label="Frontend"
            value="REACT"
            active
          />

        </section>

      </div>

    </div>
  );
}


// ============================================================
// SETTING ROW
// ============================================================

function SettingRow({
  label,
  value,
  active,
}) {
  return (
    <div className="setting-row">

      <span>
        {label}
      </span>

      <strong
        className={
          active
            ? "setting-active"
            : ""
        }
      >
        {value}
      </strong>

    </div>
  );
}


// ============================================================
// SOURCE FORMATTER
// ============================================================

function formatSource(source) {
  if (!source) {
    return "Unknown";
  }

  if (typeof source === "string") {
    return source;
  }

  if (source.type === "video") {
    return source.path || "Video";
  }

  if (source.type === "webcam") {
    return `Webcam ${source.index ?? 0}`;
  }

  if (source.type === "rtsp") {
    return "RTSP Stream";
  }

  return source.type || "Unknown";
}


// ============================================================
// EXPORT
// ============================================================

export default App;