import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import "./App.css";

// WHERE THE BACKEND IS
// --------------------
// There are two ways this page gets loaded, and they need different answers.
//
//   Vite dev server (port 5173)  the backend is a *different* origin, so every
//                                request needs the absolute address and has to
//                                clear the CORS allowlist in api/server.py.
//   FastAPI itself (port 8000)   the backend served this very file, so it is
//                                the *same* origin. An empty prefix makes every
//                                request relative: no CORS, no second port,
//                                and no second process to go stale.
//
// Deciding this at runtime rather than at build time means one bundle works in
// both places, so the production build cannot drift from the dev build.
const API = window.location.port === "5173" ? "http://127.0.0.1:8000" : "";

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

// ============================================================
// AUTHENTICATION
// ============================================================
//
// Everything below the line is a gate in front of the existing dashboard,
// not a change to it. No page, no camera call and no rendering logic was
// rewritten: requests gained a header, the navigation gained a filter, and
// the whole app gained a sign-in screen in front of it.
//
// The important thing this file does NOT do is decide permissions. React
// hides what a normal user may not use, because showing somebody a button
// that will refuse them is bad interface design - but hiding is all it is.
// Every admin route is refused by the backend independently, so typing a URL
// or editing this bundle in devtools gains nothing. The frontend is a
// convenience over the rule; the backend IS the rule.

//: Where the session token lives.
//
// sessionStorage, not localStorage: it is cleared when the tab closes, so a
// session cannot outlive the browsing session on a shared terminal. Not a
// plain variable either, because that would sign the operator out on every
// page refresh, which during a demonstration is worse than useless.
//
// Wrapped in try/catch throughout: a private window or blocked site data
// makes these throw, and an inaccessible store should cost a convenience,
// not crash the application.
//: Whether the signed-in user may control cameras, shared with the camera
//: cards without threading a prop through every component between them.
//
// Presentation only. Hiding a START button does not stop the request; the
// backend refuses it for a normal user whatever this says.
const AccessContext = createContext({ canControl: false });
const useAccess = () => useContext(AccessContext);

const TOKEN_KEY = "ibvap.session";

function readToken() {
  try {
    return sessionStorage.getItem(TOKEN_KEY) || null;
  } catch {
    return memoryToken;
  }
}

// The fallback for when sessionStorage is unavailable. The app still works;
// a refresh just returns you to the login screen.
let memoryToken = null;

function writeToken(token) {
  memoryToken = token;
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    // Held in memoryToken above.
  }
}

//: Called when the backend says the session is gone. Installed by the auth
//: gate so that any request, anywhere in the app, can trigger a sign-out
//: without every component needing to know how.
let onSessionLost = () => {};

/**
 * fetch, with the session token attached and 401 handled once.
 *
 * Every API call in this file goes through here. That is the point: a
 * request that forgot the header would get a 401 and look like a backend
 * outage, and the bug would be invisible until a demo.
 */
async function apiFetch(url, options = {}) {
  const token = readToken();
  const headers = { ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;

  const response = await fetch(url, { ...options, headers });

  // 401 means the session is over - expired, revoked, or the account was
  // disabled while it was open. Handled centrally so the operator gets one
  // clear "session expired" screen instead of every panel failing on its own.
  if (response.status === 401 && token) {
    writeToken(null);
    onSessionLost("SESSION_EXPIRED");
  }
  return response;
}

/** The stream URL, which cannot carry a header. */
function streamUrl(cameraId) {
  // An <img> tag will not send an Authorization header, so the one route
  // that a browser loads as an image takes the token as a query parameter.
  // The backend validates it exactly as it validates the header.
  const token = readToken();
  const base = `${API}/api/cameras/${cameraId}/stream`;
  return token ? `${base}?token=${encodeURIComponent(token)}` : base;
}

// ============================================================
// WHAT EACH ROLE MAY SEE
// ============================================================
//
// The single source of truth for navigation. A page absent from a role's
// list is never rendered and never reachable from the sidebar - and, far
// more importantly, the endpoints behind it refuse that role anyway.

const ADMIN_ONLY_PAGES = new Set([
  "Camera Management",
  "User Management",
  "System Settings",
  "Audit Logs",
]);

function pagesForRole(isAdmin) {
  const all = [
    { name: "Dashboard", icon: "⌂" },
    { name: "Camera Management", icon: "▣" },
    { name: "Alerts", icon: "!" },
    { name: "Detection Logs", icon: "≡" },
    { name: "Analytics", icon: "▥" },
    { name: "User Management", icon: "◍" },
    { name: "Audit Logs", icon: "⎙" },
    { name: "System Settings", icon: "⚙" },
  ];
  return isAdmin ? all : all.filter((p) => !ADMIN_ONLY_PAGES.has(p.name));
}


// ============================================================
// LOGIN
// ============================================================

function Login({ onSignedIn }) {
  const [userId, setUserId] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [backendUp, setBackendUp] = useState(null);

  // /api/health is the one route that stays open, precisely so this screen
  // can tell "wrong password" apart from "the backend is not running".
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const r = await fetch(`${API}/api/health`);
        if (!cancelled) setBackendUp(r.ok);
      } catch {
        if (!cancelled) setBackendUp(false);
      }
    };
    check();
    const t = setInterval(check, 5000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  const submit = async (event) => {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);

    try {
      const response = await fetch(`${API}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, password }),
      });

      let body = null;
      try {
        body = await response.json();
      } catch {
        body = null;
      }

      if (!response.ok) {
        // A failed sign-in and a missing endpoint are completely different
        // problems, and calling both "INVALID CREDENTIALS" sends somebody
        // hunting for a password mistake that was never there. 404 here has
        // exactly one cause worth naming.
        if (response.status === 404) {
          setError({
            title: "BACKEND OUT OF DATE",
            message:
              "The API is running but has no /api/auth/login route, so it " +
              "is serving an older build. uvicorn does not reload on its " +
              "own: stop it and start it again.",
          });
          return;
        }

        // Below this line the backend decided what the failure means and
        // said so in one sentence. React does not second-guess it, and in
        // particular does not try to tell "no such user" from "wrong
        // password" - the backend deliberately does not distinguish them.
        setError({
          title:
            response.status === 403
              ? "ACCOUNT DISABLED"
              : response.status === 429
              ? "TOO MANY ATTEMPTS"
              : response.status === 401
              ? "INVALID CREDENTIALS"
              : `SIGN-IN FAILED (${response.status})`,
          message:
            body?.detail || "Sign-in failed. Please try again.",
        });
        setPassword("");
        return;
      }

      writeToken(body.token);
      onSignedIn(body);
    } catch {
      setError({
        title: "BACKEND UNREACHABLE",
        message: `No response from ${API}. The API process is not running.`,
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-shell">
      <div className="login-panel">
        <div className="login-brand">
          <img src="/logo-mark.png" alt="" />
          <div>
            <div className="brand-title">
              VISION<span>VIGILANTS</span>
            </div>
            <div className="brand-sub">IBVAP SURVEILLANCE</div>
          </div>
        </div>

        <div className="login-head">
          <div className="section-label">SECURE COMMAND ACCESS</div>
          <p>
            Sign in with the credentials issued by your Super Administrator.
          </p>
        </div>

        {error && (
          <div className="login-error">
            <strong>{error.title}</strong>
            <span>{error.message}</span>
          </div>
        )}

        <form onSubmit={submit}>
          <label className="field">
            <span>USER ID</span>
            <input
              type="text"
              value={userId}
              autoFocus
              autoComplete="username"
              spellCheck="false"
              placeholder="ADMIN-001"
              onChange={(e) => setUserId(e.target.value)}
            />
          </label>

          <label className="field">
            <span>PASSWORD</span>
            <input
              type="password"
              value={password}
              autoComplete="current-password"
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>

          <button
            className="login-btn"
            type="submit"
            disabled={busy || !userId || !password}
          >
            {busy ? "VERIFYING…" : "SIGN IN"}
          </button>
        </form>

        <div className="login-foot">
          <span className="login-warn">Authorized personnel only</span>
          <span
            className={`login-health ${
              backendUp === false ? "down" : backendUp ? "up" : ""
            }`}
          >
            {backendUp === null
              ? "CHECKING BACKEND"
              : backendUp
              ? "BACKEND ONLINE"
              : "BACKEND OFFLINE"}
          </span>
        </div>
      </div>
    </div>
  );
}


// ============================================================
// ACCESS DENIED
// ============================================================
//
// Reached only if a normal user's interface somehow routes them to an admin
// page. It is a backstop, not the security boundary - the request behind the
// page is refused by the backend whether or not this screen ever renders.

function AccessDenied({ required = "SUPER ADMIN", onReturn }) {
  return (
    <div className="page">
      <div className="denied">
        <div className="denied-mark">✕</div>
        <h2>ACCESS DENIED</h2>
        <p>You don&rsquo;t have permission to access this section.</p>
        <div className="denied-role">
          <span className="section-label">REQUIRED ROLE</span>
          <strong>{required}</strong>
        </div>
        <button className="login-btn" onClick={onReturn}>
          RETURN TO DASHBOARD
        </button>
      </div>
    </div>
  );
}


// ============================================================
// SESSION EXPIRY WARNING
// ============================================================

function SessionWarning({ secondsLeft, onContinue, onSignOut }) {
  return (
    <div className="session-overlay">
      <div className="session-dialog">
        <div className="section-label">SESSION EXPIRING</div>
        <h3>Your session will expire due to inactivity.</h3>
        <div className="session-count">{Math.max(0, secondsLeft)}s</div>
        <p>
          Continuing sends one authenticated request, which resets the idle
          timer on the server.
        </p>
        <div className="session-actions">
          <button className="login-btn" onClick={onContinue}>
            CONTINUE SESSION
          </button>
          <button className="session-out" onClick={onSignOut}>
            SIGN OUT NOW
          </button>
        </div>
      </div>
    </div>
  );
}


// ============================================================
// AUTH GATE
// ============================================================
//
// Decides whether the dashboard renders at all. Holds the signed-in user,
// runs the idle countdown, and hands both down to App.

function AuthGate() {
  const [auth, setAuth] = useState(null);
  const [booting, setBooting] = useState(true);
  const [notice, setNotice] = useState(null);

  // Seconds since the last authenticated request, kept here so the whole app
  // shares one clock.
  const [idleSeconds, setIdleSeconds] = useState(0);

  const signOutLocal = (reason) => {
    writeToken(null);
    setAuth(null);
    setIdleSeconds(0);
    if (reason === "SESSION_EXPIRED") {
      setNotice({
        title: "SESSION EXPIRED",
        message: "Please sign in again.",
      });
    }
  };

  // Installed once so apiFetch anywhere in the app can end the session.
  useEffect(() => {
    onSessionLost = signOutLocal;
    return () => {
      onSessionLost = () => {};
    };
  }, []);

  // On load, ask the backend who we are. The answer comes from the database
  // via the session token - never from anything the browser stored about
  // itself - so a tampered local copy of the role changes nothing.
  useEffect(() => {
    let cancelled = false;
    const restore = async () => {
      const token = readToken();
      if (!token) {
        if (!cancelled) setBooting(false);
        return;
      }
      try {
        const response = await fetch(`${API}/api/auth/me`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (response.ok) {
          const body = await response.json();
          if (!cancelled) setAuth(body);
        } else {
          writeToken(null);
        }
      } catch {
        writeToken(null);
      } finally {
        if (!cancelled) setBooting(false);
      }
    };
    restore();
    return () => {
      cancelled = true;
    };
  }, []);

  // The idle clock. Any activity in the app resets it, and so does any
  // authenticated request, because the server's idle timer is refreshed by
  // the same traffic.
  useEffect(() => {
    if (!auth) return undefined;
    const reset = () => setIdleSeconds(0);
    const events = ["mousedown", "keydown", "wheel", "touchstart"];
    events.forEach((e) => window.addEventListener(e, reset, { passive: true }));
    const tick = setInterval(() => setIdleSeconds((s) => s + 1), 1000);
    return () => {
      events.forEach((e) => window.removeEventListener(e, reset));
      clearInterval(tick);
    };
  }, [auth]);

  const signOut = async () => {
    try {
      await apiFetch(`${API}/api/auth/logout`, { method: "POST" });
    } catch {
      // The local session ends either way. A failed logout call must not
      // leave somebody stuck signed in on the screen in front of them.
    }
    signOutLocal();
  };

  if (booting) {
    return (
      <div className="login-shell">
        <div className="login-boot">RESTORING SESSION…</div>
      </div>
    );
  }

  if (!auth) {
    return (
      <>
        {notice && (
          <div className="login-shell-notice">
            <strong>{notice.title}</strong> {notice.message}
          </div>
        )}
        <Login
          onSignedIn={(body) => {
            setNotice(null);
            setIdleSeconds(0);
            setAuth(body);
          }}
        />
      </>
    );
  }

  const idleLimit = auth.session_idle_seconds || 1800;
  const warnAt = auth.warn_before_expiry_seconds || 120;
  const secondsLeft = Math.ceil(idleLimit - idleSeconds);
  const warning = secondsLeft <= warnAt;

  return (
    <>
      <App auth={auth} onSignOut={signOut} />
      {warning && (
        <SessionWarning
          secondsLeft={secondsLeft}
          onContinue={async () => {
            // One real request. Tapping a button that only reset a local
            // counter would leave the SERVER's idle timer running, and the
            // session would die anyway a moment later.
            setIdleSeconds(0);
            await apiFetch(`${API}/api/auth/me`);
          }}
          onSignOut={signOut}
        />
      )}
    </>
  );
}


function App({ auth, onSignOut }) {
  const [activeTab, setActiveTab] = useState("Dashboard");

  // The role the BACKEND assigned, read from /api/auth/login or
  // /api/auth/me. Never taken from anything the browser stored about
  // itself, and never used as the only thing standing between a normal
  // user and an admin endpoint.
  const isAdmin = Boolean(auth?.user?.is_admin);

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
      const response = await apiFetch(
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
            const statsResponse = await apiFetch(
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
      const response = await apiFetch(
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
      const response = await apiFetch(
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
      const response = await apiFetch(
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
          apiFetch(
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

  // Fleet totals sum ONLY cameras that are running. A stopped camera has no
  // live state to contribute, and counting its last session produced
  // "0 cameras active, 3 alerts" - a sentence that cannot be true.
  //
  // These are the FLEET view. The per-camera view below is what the selected
  // camera's tiles read, and the two are deliberately different numbers.
  const runningCameras = cameras.filter(
    (camera) => camera.stats?.status === "ONLINE"
  );

  const sumLive = (key) =>
    runningCameras.reduce(
      (total, camera) => total + Number(camera.stats?.[key] || 0),
      0
    );

  const totalPersons = sumLive("persons");
  const totalVehicles = sumLive("vehicles");
  const totalAlerts = sumLive("alerts");
  const totalDetections = sumLive("detections");

  // ============================================================
  // NAVIGATION
  // ============================================================

  // Built from the role the backend reported. A page a normal user may
  // not use is not rendered in the sidebar - but the endpoints behind it
  // refuse that role independently, which is what actually protects it.
  const navigation = pagesForRole(isAdmin);

  // ============================================================
  // PAGE RENDERER
  // ============================================================

  const renderPage = () => {
    switch (activeTab) {
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

      case "User Management":
        return isAdmin ? (
          <UserManagement auth={auth} />
        ) : (
          <AccessDenied onReturn={() => setActiveTab("Dashboard")} />
        );

      case "Audit Logs":
        return isAdmin ? (
          <AuditLogs />
        ) : (
          <AccessDenied onReturn={() => setActiveTab("Dashboard")} />
        );

      case "Camera Management":
        return isAdmin ? (
          <CameraManagement
            cameras={cameras}
            selected={selected}
            selectedCamera={selectedCamera}
            setSelectedCamera={setSelectedCamera}
            startCamera={startCamera}
            stopCamera={stopCamera}
            loadingCamera={loadingCamera}
          />
        ) : (
          <AccessDenied onReturn={() => setActiveTab("Dashboard")} />
        );

      case "System Settings":
        return isAdmin ? (
          <Settings
            cameras={cameras}
          />
        ) : (
          <AccessDenied onReturn={() => setActiveTab("Dashboard")} />
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
            runningCount={runningCameras.length}
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
    <AccessContext.Provider value={{ canControl: isAdmin }}>
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

              {/* The access level in the operator's own words, so what a
                  person can do is stated rather than inferred from which
                  menu items happen to be present. */}
              <div className="indicator-text">
                {auth.user.role_tagline}
              </div>

            </div>

          </div>

          {/* Stopping every camera is camera management, and the backend
              refuses it for a normal user. Showing an operator a button
              that will only ever answer 403 is worse than not showing it:
              it reads as a broken system rather than a restricted one. */}
          {isAdmin && (
            <button
              className="shutdown-button"
              onClick={
                stopAllCameras
              }
            >
              STOP ALL CAMERAS
            </button>
          )}

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

            {/* Who is signed in, and the way out. The role shown here is
                the one the backend reported; it is a label, not a
                permission, and changing it in devtools would change
                nothing about what the API will answer. */}
            <div className="topbar-user">
              <div className="topbar-who">
                <div className="topbar-name">{auth.user.username}</div>
                <div className={`topbar-role ${auth.user.role.toLowerCase()}`}>
                  {auth.user.role_label}
                </div>
              </div>
              <button className="logout-btn" onClick={onSignOut}>
                LOGOUT
              </button>
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
    </AccessContext.Provider>
  );
}


// ============================================================
// DASHBOARD
// ============================================================

// ============================================================
// DASHBOARD — THE COMMAND WALL
// ============================================================
//
// Every panel on this page is fed by one of three sources, and they mean
// different things. Mixing them up is the easiest way to make a dashboard
// that lies, so each panel says which one it is reading:
//
//   LIVE      the running pipeline, right now, via /api/cameras/*/stats.
//             Zero when a camera is stopped, because a stopped camera has
//             no live state. This is the "4/4 ACTIVE, 6 PERSONS" row.
//
//   RECORDED  what the analysis concluded and wrote to the database, via
//             /api/events. It survives a restart and it is the only
//             honest source for a timeline.
//
//   EVIDENCE  the JPEGs the pipeline captured at the moment an incident
//             fired, via /api/evidence. A picture of a real detection,
//             not a stock image.
//
// Nothing on this page is generated in React. If a panel is empty it is
// because the pipeline has not produced that thing yet, and it says so.


/** How often the recorded feeds refresh. The live stats poll on their own
 *  clock in App; these are database reads and do not need to be as quick. */
const DASH_FEED_INTERVAL_MS = 4000;

/** Threat levels in severity order, which is also the order they are
 *  counted and displayed in. Kept here so the badge colours, the counts
 *  and the legend cannot drift apart. */
const THREAT_LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];


/**
 * Recorded events and captured evidence, polled together.
 *
 * The three failure states are kept apart on purpose. A previous version
 * of this project collapsed every non-200 into one "unavailable" message,
 * and the result was that a perfectly healthy backend running an older
 * build reported the same thing as a dead network. They are different
 * problems with different fixes, so they get different words.
 */
function useRecordedFeed() {
  const [events, setEvents] = useState([]);
  const [evidence, setEvidence] = useState([]);
  const [vehicles, setVehicles] = useState([]);
  const [state, setState] = useState("loading");
  const [problem, setProblem] = useState(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const [eventsResponse, evidenceResponse, vehiclesResponse] =
          await Promise.all([
            apiFetch(`${API}/api/events?limit=60`),
            apiFetch(`${API}/api/evidence?kind=crop&limit=12`),
            apiFetch(`${API}/api/vehicles?limit=20&readable_only=false`),
          ]);

        if (cancelled) return;

        // An HTTP error is the backend answering, not failing to answer.
        // Say which route and which code, because "404 on /api/evidence"
        // means the server predates the route and needs restarting, while
        // "500" means it broke. Those are not the same message.
        const failed = [
          !eventsResponse.ok ? ["/api/events", eventsResponse.status] : null,
          !evidenceResponse.ok
            ? ["/api/evidence", evidenceResponse.status]
            : null,
          !vehiclesResponse.ok
            ? ["/api/vehicles", vehiclesResponse.status]
            : null,
        ].filter(Boolean);

        if (failed.length > 0) {
          setState("http-error");
          setProblem(
            failed
              .map(([route, code]) => `${route} returned ${code}`)
              .join(", ")
          );
          return;
        }

        const [eventRows, evidenceRows, vehicleRows] = await Promise.all([
          eventsResponse.json(),
          evidenceResponse.json(),
          vehiclesResponse.json(),
        ]);

        if (cancelled) return;

        setEvents(Array.isArray(eventRows) ? eventRows : []);
        setEvidence(Array.isArray(evidenceRows) ? evidenceRows : []);
        setVehicles(Array.isArray(vehicleRows) ? vehicleRows : []);
        setState("ready");
        setProblem(null);
      } catch (error) {
        if (cancelled) return;
        // The request never completed: the server is not listening, or
        // the browser could not reach it at all.
        setState("unreachable");
        setProblem(String(error?.message || error));
      }
    };

    load();
    const timer = setInterval(load, DASH_FEED_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return { events, evidence, vehicles, state, problem };
}


function Dashboard({
  cameras,
  selected,
  activeCameras,
  totalPersons,
  totalVehicles,
  totalAlerts,
  runningCount,
  setSelectedCamera,
  startCamera,
  stopCamera,
  loadingCamera,
}) {
  const { events, evidence, vehicles, state, problem } = useRecordedFeed();

  // Counts by threat level, over exactly the events that are on screen.
  // Not "all time" and not invented: the panel says what it counted, so
  // the number and its meaning travel together.
  const levelCounts = useMemo(() => {
    const counts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
    for (const event of events) {
      const level = String(event.threat_level || "").toUpperCase();
      if (level in counts) counts[level] += 1;
    }
    return counts;
  }, [events]);

  // The badge counts every row shown beneath it. An earlier version
  // excluded LOW, so the badge read 16 over a list that summed to 17 - a
  // number disagreeing with the list directly below it.
  const countedEvents = THREAT_LEVELS.reduce(
    (total, level) => total + levelCounts[level],
    0
  );

  // WHY THIS PANEL ASKS THE CAMERAS ABOUT ANPR
  // ------------------------------------------
  // An empty plate list has three completely different causes, and showing
  // one message for all three hides the only one that needs acting on:
  //
  //   * no OCR engine is installed, so plates CANNOT be read;
  //   * vehicles were seen but no plate has been captured yet;
  //   * no vehicles have passed at all.
  //
  // The first of those was being reported as "NO VEHICLES TRACKED" while
  // the backend log said "No OCR backend is available, so ANPR is switched
  // off" six times over. The pipeline already knows; it reports its ANPR
  // state on every camera's stats. So the panel asks, instead of guessing
  // from an empty list.
  const runningCameras = cameras.filter((camera) => camera.running);

  const anprUnavailable =
    runningCameras.length > 0 &&
    runningCameras.every((camera) =>
      /no ocr backend/i.test(String(camera.stats?.anpr || ""))
    );

  const vehiclesSeen = runningCameras.reduce(
    (total, camera) => total + Number(camera.stats?.vehicles || 0),
    0
  );

  return (
    <div className="page wall">

      {/* ==================================================
          LEFT: the video wall, then the lower panels
      ================================================== */}

      <div className="wall-main">

        <section className="panel">

          <div className="panel-header">
            <div className="panel-title">
              <span className="panel-glyph">▣</span>
              LIVE CAMERA FEEDS
            </div>
            <div className="panel-count">
              {activeCameras} / {cameras.length} ONLINE
            </div>
          </div>

          <div className="feed-wall">
            {cameras.map((camera) => (
              <FeedTile
                key={camera.id}
                camera={camera}
                selected={selected?.id === camera.id}
                loading={loadingCamera === camera.id}
                onSelect={() => setSelectedCamera(camera.id)}
                onStart={() => startCamera(camera.id)}
                onStop={() => stopCamera(camera.id)}
              />
            ))}
          </div>

        </section>

        <div className="wall-lower">

          {/* ---- threat level tally ---- */}
          <section className="panel">
            <div className="panel-header">
              <div className="panel-title">
                <span className="panel-glyph alarm">⚠</span>
                ALERTS &amp; INCIDENTS
              </div>
              <div className="panel-count">{countedEvents}</div>
            </div>

            <div className="level-tally">
              {THREAT_LEVELS.map((level) => (
                <div className="level-row" key={level}>
                  <span className={`level-dot ${level.toLowerCase()}`} />
                  <span className="level-name">
                    {level === "CRITICAL"
                      ? "Critical"
                      : level === "HIGH"
                        ? "High Threat"
                        : level === "MEDIUM"
                          ? "Medium Threat"
                          : "Low Threat"}
                  </span>
                  <strong className="level-count">
                    {levelCounts[level]}
                  </strong>
                </div>
              ))}
            </div>

            {/* The scope, stated. Without this the numbers look like
                all-time totals and quietly disagree with every other
                count on the page. */}
            <div className="panel-foot">
              across the {events.length} most recent recorded events
            </div>
          </section>

          {/* ---- captured snapshots ---- */}
          <section className="panel">
            <div className="panel-header">
              <div className="panel-title">
                <span className="panel-glyph">◨</span>
                RECENT DETECTIONS
              </div>
            </div>

            <FeedState state={state} problem={problem}
              empty={evidence.length === 0}
              emptyTitle="NO SNAPSHOTS CAPTURED"
              emptyBody="The pipeline saves an image when an incident reaches MEDIUM or above. None have fired yet.">
              <div className="thumb-strip">
                {evidence.map((item) => (
                  <EvidenceThumb key={item.id} item={item} />
                ))}
              </div>
            </FeedState>
          </section>

          {/* ---- timeline ---- */}
          <section className="panel">
            <div className="panel-header">
              <div className="panel-title">
                <span className="panel-glyph">☰</span>
                EVENT TIMELINE
              </div>
            </div>

            <FeedState state={state} problem={problem}
              empty={events.length === 0}
              emptyTitle="NO EVENTS RECORDED"
              emptyBody="Start a camera. Events appear here as the analysis concludes them.">
              <ol className="timeline">
                {events.slice(0, 12).map((event) => (
                  <TimelineRow
                    key={`${event.incident_id}-${event.ts}`}
                    event={event}
                  />
                ))}
              </ol>
            </FeedState>
          </section>

        </div>

      </div>

      {/* ==================================================
          RIGHT RAIL
      ================================================== */}

      <aside className="wall-rail">

        <section className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <span className="panel-glyph alarm">⚠</span>
              CAMERA STATUS
            </div>
          </div>

          <div className="status-list">
            {cameras.map((camera) => (
              <button
                type="button"
                key={camera.id}
                className={
                  "status-row" +
                  (selected?.id === camera.id ? " is-selected" : "")
                }
                onClick={() => setSelectedCamera(camera.id)}
              >
                <span
                  className={
                    "status-dot" +
                    (camera.running
                      ? " online"
                      : camera.starting
                        ? " starting"
                        : " offline")
                  }
                />
                <span className="status-id">{camera.id}</span>
                <span className="status-place">
                  {camera.location || camera.sector || ""}
                </span>
                <span
                  className={
                    "status-state" + (camera.running ? " online" : "")
                  }
                >
                  {camera.running
                    ? "LIVE"
                    : camera.starting
                      ? "STARTING"
                      : "OFFLINE"}
                </span>
              </button>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <span className="panel-glyph">▤</span>
              QUICK STATS
            </div>
          </div>

          {/* All four read the LIVE pipeline and sum only cameras that are
              actually running. A stopped camera contributes nothing, so
              "0 cameras active, 3 persons" cannot appear. */}
          <div className="quick-stats">
            <QuickStat
              glyph="▣"
              label="ACTIVE CAMERAS"
              value={`${activeCameras} / ${cameras.length}`}
            />
            <QuickStat glyph="◉" label="PERSONS" value={totalPersons} />
            <QuickStat glyph="▭" label="VEHICLES" value={totalVehicles} />
            <QuickStat
              glyph="⚠"
              label="ALERTS"
              value={totalAlerts}
              tone={totalAlerts > 0 ? "alarm" : undefined}
            />
          </div>

          <div className="panel-foot">
            live, from {runningCount} running camera
            {runningCount === 1 ? "" : "s"}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <span className="panel-glyph">▭</span>
              VEHICLE PLATES (ANPR)
            </div>
            <div className={"panel-count" + (anprUnavailable ? " warn" : "")}>
              {anprUnavailable
                ? "ENGINE OFF"
                : `${vehicles.filter((v) => v.readable).length} / ${vehicles.length} READ`}
            </div>
          </div>

          {anprUnavailable ? (
            // Not an empty state. The system cannot do this job at all
            // right now, and the fix is one command, so it says both.
            <div className="feed-note error">
              <strong>ANPR UNAVAILABLE</strong>
              <span>
                No OCR engine is installed, so number plates cannot be
                read. Detection, tracking and threat analysis are all
                unaffected and still running.
              </span>
              {/* The bundled installer, not a pip command. Everything it
                  needs already ships with the project, so this works
                  with no internet - which a bare `pip install` does not. */}
              <code>./scripts/enable_anpr.sh</code>
            </div>
          ) : (
            <FeedState state={state} problem={problem}
              empty={vehicles.length === 0}
              emptyTitle={
                vehiclesSeen > 0
                  ? "NO PLATE CAPTURED YET"
                  : "NO VEHICLES TRACKED"
              }
              emptyBody={
                vehiclesSeen > 0
                  ? `${vehiclesSeen} vehicle${vehiclesSeen === 1 ? "" : "s"} ` +
                    "in view. A plate is only attempted once a vehicle is " +
                    "close enough for one to be legible."
                  : "Start a camera that sees vehicles. Plates are read as they pass."
              }>
              <div className="plate-list">
                {vehicles.map((vehicle) => (
                  <PlateRow
                    key={`${vehicle.camera_id}-${vehicle.track_id}-${vehicle.ts}`}
                    vehicle={vehicle}
                  />
                ))}
              </div>
            </FeedState>
          )}

          <div className="panel-foot">
            state and district decoded from the registration prefix
          </div>
        </section>

        <section className="panel rail-events">
          <div className="panel-header">
            <div className="panel-title">
              <span className="panel-glyph">◈</span>
              LIVE THREAT EVENTS
            </div>
          </div>

          <FeedState state={state} problem={problem}
            empty={events.length === 0}
            emptyTitle="NO EVENTS RECORDED"
            emptyBody="Nothing has been detected and concluded yet.">
            <div className="threat-events">
              {events.slice(0, 8).map((event) => (
                <ThreatEventRow
                  key={`${event.incident_id}-${event.ts}`}
                  event={event}
                />
              ))}
            </div>
          </FeedState>
        </section>

      </aside>

    </div>
  );
}


// ============================================================
// ONE VEHICLE, ITS PLATE, AND WHERE IT IS REGISTERED
// ============================================================

function PlateRow({ vehicle }) {
  // A vehicle whose plate could not be read is still shown. On a
  // checkpoint camera "a vehicle passed and we did not get its plate" is
  // a finding an operator needs, not a blank to be tidied away.
  if (!vehicle.readable) {
    return (
      <div className="plate-row is-unread">
        <div className="plate-shot empty">▭</div>
        <div className="plate-body">
          <strong className="plate-text unread">PLATE NOT READ</strong>
          <span className="plate-origin">
            {vehicle.attempts} attempt{vehicle.attempts === 1 ? "" : "s"},
            none legible
          </span>
          <span className="plate-meta">
            {vehicle.camera_id} · {vehicle.time}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="plate-row">
      {vehicle.image_url ? (
        <img
          className="plate-shot"
          src={`${API}${vehicle.image_url}?token=${encodeURIComponent(readToken() || "")}`}
          alt={`Plate crop for ${vehicle.plate_text}`}
          loading="lazy"
        />
      ) : (
        <div className="plate-shot empty">▭</div>
      )}

      <div className="plate-body">
        <div className="plate-line">
          <strong className="plate-text">{vehicle.plate_text}</strong>

          {/* In India the plate's background colour is a legal class, not
              decoration: yellow means the vehicle is for hire. Shown only
              when the classifier was sure - an unreadable or shadowed
              crop carries no chip rather than a guessed class. */}
          {vehicle.vehicle_class && (
            <span
              className={
                "plate-class " +
                String(vehicle.plate_colour || "").toLowerCase()
              }
              title={vehicle.vehicle_class_note || ""}
            >
              {vehicle.vehicle_class}
            </span>
          )}
        </div>

        {/* The decoded origin. District and state are shown as separate
            pieces rather than as one sentence, because the rail is narrow
            and a single line truncated to "Shimla Urban (tourist buses &
            ta..." hides the state, which is the part that always exists.

            Register entries carry a qualifier in brackets - the RTO's own
            wording, such as "(tourist buses & taxis)". It is trimmed off
            the compact line and kept in the tooltip, so nothing is
            invented and nothing is lost.

            When the district is not in the table the backend's own
            sentence is shown verbatim instead, so a partial answer reads
            as partial rather than as a blank. */}
        {vehicle.district_known ? (
          <span className="plate-origin is-full" title={vehicle.origin}>
            <span className="plate-district">
              {String(vehicle.district).split(" (")[0]}
            </span>
            <span className="plate-state">{vehicle.state}</span>
          </span>
        ) : (
          <span className="plate-origin" title={vehicle.origin}>
            {vehicle.origin}
          </span>
        )}

        <div className="plate-meta">
          <span>{vehicle.camera_id}</span>
          <span>{vehicle.time}</span>
          {/* How the answer was reached, so a contested plate looks
              contested rather than certain. */}
          <span className="plate-reads">
            {vehicle.reads}/{vehicle.attempts} reads
          </span>
          <span
            className={
              "plate-conf" + (vehicle.validated ? " ok" : "")
            }
          >
            {vehicle.validated ? "FORMAT OK" : "UNVERIFIED"}{" "}
            {Math.round((vehicle.confidence || 0) * 100)}%
          </span>
        </div>
      </div>
    </div>
  );
}


// ============================================================
// ONE CAMERA IN THE WALL
// ============================================================

function FeedTile({
  camera,
  selected,
  loading,
  onSelect,
  onStart,
  onStop,
}) {
  const stats = camera.stats || {};

  return (
    <div
      className={"feed-tile" + (selected ? " is-selected" : "")}
      onClick={onSelect}
    >

      <div className="tile-head">
        <div className="tile-id">
          {camera.id}
          <span
            className={
              "tile-badge" +
              (camera.running
                ? " live"
                : camera.starting
                  ? " starting"
                  : "")
            }
          >
            {camera.running
              ? "● LIVE"
              : camera.starting
                ? "● STARTING"
                : "● OFFLINE"}
          </span>
        </div>

        <button
          type="button"
          className={"tile-action" + (camera.running ? " stop" : "")}
          disabled={loading}
          onClick={(clickEvent) => {
            // The tile itself selects the camera; the button must not do
            // that too, or starting a camera would also move the
            // selection under the operator's hands.
            clickEvent.stopPropagation();
            camera.running ? onStop() : onStart();
          }}
        >
          {loading ? "···" : camera.running ? "STOP" : "START"}
        </button>
      </div>

      <div className="tile-name">{camera.name || camera.id}</div>

      <div className="tile-view">
        {camera.running ? (
          // The boxes, person numbers, behaviour and threat labels are
          // drawn into the frame by core/annotator.py before it leaves
          // the backend. React never draws an overlay, so what is on
          // screen is exactly what the analysis saw.
          <img
            src={streamUrl(camera.id)}
            alt={`${camera.id} live feed with detection overlay`}
          />
        ) : (
          <div className="tile-dark">
            <span className="tile-dark-glyph">
              {camera.starting ? "◌" : "▣"}
            </span>
            <strong>
              {camera.starting ? "CONNECTING" : "OFFLINE"}
            </strong>
            <span className="tile-dark-hint">
              {camera.starting
                ? "Opening the video source"
                : "Press START to begin analysis"}
            </span>
          </div>
        )}
      </div>

      {/* Live counters for THIS camera. A stopped camera shows dashes
          rather than zeros, because "no data" and "counted zero" are
          different claims. */}
      <div className="tile-stats">
        <span>
          PERSONS{" "}
          <strong>{camera.running ? Number(stats.persons || 0) : "--"}</strong>
        </span>
        <span>
          VEHICLES{" "}
          <strong>{camera.running ? Number(stats.vehicles || 0) : "--"}</strong>
        </span>
        <span className={camera.running && stats.alerts > 0 ? "alarm" : ""}>
          ALERTS{" "}
          <strong>{camera.running ? Number(stats.alerts || 0) : "--"}</strong>
        </span>
      </div>

    </div>
  );
}


// ============================================================
// SMALL PARTS
// ============================================================

function QuickStat({ glyph, label, value, tone }) {
  return (
    <div className={"quick-stat" + (tone ? ` ${tone}` : "")}>
      <span className="quick-glyph">{glyph}</span>
      <span className="quick-label">{label}</span>
      <strong className="quick-value">{value}</strong>
    </div>
  );
}


function ThreatBadge({ level }) {
  const text = String(level || "").toUpperCase();
  if (!text) return null;
  return (
    <span className={`threat-badge ${text.toLowerCase()}`}>{text}</span>
  );
}


/** The label for a person or vehicle, as the operator refers to it. */
function subjectLabel(event) {
  const type = String(event.object_type || "object");
  const noun = type.charAt(0).toUpperCase() + type.slice(1);
  return event.track_id ? `${noun} #${event.track_id}` : noun;
}


/** What happened, in the pipeline's own words.
 *
 *  `headline` comes from the incident type; `behaviour` from the
 *  classifier. Neither is invented here. When the classifier reached no
 *  conclusion the behaviour is simply left out rather than padded with
 *  "unknown", which would read as a finding. */
function describeEvent(event) {
  const parts = [];
  if (event.headline) parts.push(event.headline);
  if (event.behaviour) {
    parts.push(String(event.behaviour).toLowerCase().replace(/_/g, " "));
  }
  return parts.join(" — ") || "Recorded event";
}


function ThreatEventRow({ event }) {
  return (
    <div className="threat-event">
      <span className={`event-tick ${String(event.threat_level || "")
        .toLowerCase()}`} />
      <span className="event-time">{event.time}</span>
      <ThreatBadge level={event.threat_level} />
      <div className="event-body">
        <strong>{subjectLabel(event)}</strong>
        <span>{describeEvent(event)}</span>
      </div>
    </div>
  );
}


function TimelineRow({ event }) {
  return (
    <li className="timeline-row">
      <span className={`timeline-tick ${String(event.threat_level || "")
        .toLowerCase()}`} />
      <span className="timeline-when">
        <strong>{event.time}</strong>
        {/* The date is shown as well as the time. A timeline that shows
            only a clock reads as "today" even when the newest entry is
            from last week. */}
        <small>{event.date}</small>
      </span>
      <span className="timeline-camera">{event.camera_id}</span>
      <span className="timeline-subject">{subjectLabel(event)}</span>
      <span className="timeline-what">{describeEvent(event)}</span>
      <ThreatBadge level={event.threat_level} />
    </li>
  );
}


function EvidenceThumb({ item }) {
  return (
    <figure className="thumb">
      {/* The token goes in the query because a browser will not put an
          Authorization header on an <img> request. Same trade-off, and
          same reasoning, as the camera stream. */}
      <img
        src={`${API}${item.image_url}?token=${encodeURIComponent(readToken() || "")}`}
        alt={`${subjectLabel(item)} captured at ${item.time}`}
        loading="lazy"
      />
      <figcaption>
        <strong>{subjectLabel(item)}</strong>
        <span>{item.headline}</span>
        <div className="thumb-foot">
          <ThreatBadge level={item.threat_level} />
          <small>{item.time}</small>
        </div>
        <small className="thumb-cam">{item.camera_id} · {item.date}</small>
      </figcaption>
    </figure>
  );
}


/**
 * Loading, failure and emptiness, told apart.
 *
 * "Nothing has happened yet" is not a failure and must never be shown as
 * one, and a backend that answered with an error is not the same as a
 * backend that could not be reached. Each gets its own words, so the
 * message points at the actual fix.
 */
function FeedState({
  state,
  problem,
  empty,
  emptyTitle,
  emptyBody,
  children,
}) {
  if (state === "loading") {
    return <div className="feed-note">Loading…</div>;
  }

  if (state === "unreachable") {
    return (
      <div className="feed-note error">
        <strong>BACKEND UNREACHABLE</strong>
        <span>
          The request did not complete. The server may have stopped.
        </span>
        {problem && <code>{problem}</code>}
      </div>
    );
  }

  if (state === "http-error") {
    return (
      <div className="feed-note error">
        <strong>BACKEND ERROR</strong>
        <span>
          The server answered, but with an error. A 404 here means it is
          running an older build and needs restarting.
        </span>
        {problem && <code>{problem}</code>}
      </div>
    );
  }

  if (empty) {
    return (
      <div className="feed-note">
        <strong>{emptyTitle}</strong>
        <span>{emptyBody}</span>
      </div>
    );
  }

  return children;
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
  // Camera control is a Super Admin action. The button is hidden rather
  // than disabled for a normal user: the backend will refuse the request
  // either way, and an interface full of dead controls reads as broken
  // rather than restricted.
  const { canControl } = useAccess();

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

              {/* The CONCLUDED result of this camera's last completed run, not
            the current frame. `final` is written by the backend at the
            moment the run ends and survives the camera going OFFLINE;
            `result_status` says whether the numbers can be trusted yet,
            which the live status cannot - a camera reads OFFLINE both
            before its first run and after a finished one. */}
        {(() => {
          const result = camera.stats?.final;
          const state = camera.stats?.result_status || "NEVER_RUN";
          const done = state === "COMPLETED" || state === "STOPPED";

          return (
            <>
              <div className={`card-result-state state-${state.toLowerCase()}`}>
                {state === "PROCESSING" && (
                  <span className="pulse-dot" />
                )}
                {state === "NEVER_RUN" ? "NOT YET RUN" : state}
              </div>

              <div className="card-live">
                <span>
                  PERSONS{" "}
                  <strong>
                    {done && result ? result.persons : "--"}
                  </strong>
                </span>
                <span>
                  VEHICLES{" "}
                  <strong>
                    {done && result ? result.vehicles : "--"}
                  </strong>
                </span>
                <span>
                  ALERTS{" "}
                  <strong>
                    {done && result ? result.alerts : "--"}
                  </strong>
                </span>
              </div>

              {done && result && (
                <div className="card-result-note">
                  unique over {result.frames_processed} analysed frames
                  {result.completed ? "" : " (stopped early)"}
                </div>
              )}
            </>
          );
        })()}

<div className="camera-actions">

        {!canControl ? (
          <span className="control-locked">VIEW ONLY</span>
        ) : camera.running ? (

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
        src={streamUrl(camera.id)}
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

  // Live counts, from RUNNING cameras only, consistent with the dashboard.
  // A stopped camera has no live alerts; its past ones live in the database
  // with their camera_id and belong to the history view, not this total.
  const running = cameras.filter(
    (camera) => camera.stats?.status === "ONLINE"
  );

  const totalAlerts = running.reduce(
    (total, camera) => total + Number(camera.stats?.alerts || 0),
    0
  );

  const totalIncidents = running.reduce(
    (total, camera) => total + Number(camera.stats?.incidents || 0),
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
//
// The recorded event log for one camera, or for all of them.
//
// This page shows what the pipeline CONCLUDED, not what it is seeing now.
// That is the whole difference between it and the Dashboard wall, and it is
// why every number and every word below comes from the backend:
//
//   summary counts   <- camera.stats.final, written by the API when the
//                       run ended (distinct confirmed identities, deduped
//                       incidents), never a count of the last frame
//   events           <- /api/events, one row per incident, already
//                       deduplicated by IncidentManager
//   threat level     <- the ThreatEngine's stored verdict
//   headline / text  <- composed in Python, from the incident type and the
//                       behaviour the classifier recorded
//
// Nothing on this page is derived in React. If the backend has not
// concluded something, the interface says so rather than filling the gap.

const THREAT_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];

function threatClass(level) {
  const upper = String(level || "").toUpperCase();
  return THREAT_ORDER.includes(upper)
    ? `threat-${upper.toLowerCase()}`
    : "threat-unknown";
}

// How the run ended, in the operator's words. A camera reads OFFLINE both
// before its first run and after a finished one, so the live status cannot
// answer this; result_status can.
const SESSION_WORDS = {
  NEVER_RUN: "NOT YET RUN",
  PROCESSING: "PROCESSING",
  COMPLETED: "COMPLETED",
  STOPPED: "STOPPED / INCOMPLETE",
  ERROR: "ERROR",
};

function DetectionLogs({ cameras }) {
  // "ALL" or a camera id. Nothing else is a valid selection, and the whole
  // page reads from this one value.
  const [selection, setSelection] = useState("ALL");
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  // WHY the list is empty, when it is empty. Three different facts that
  // must never wear each other's words:
  //   null            - the list is the truth; zero events means zero events
  //   {kind:"network"}- the backend could not be reached at all
  //   {kind:"http"}   - the backend answered, and refused
  // Collapsing these is what produced "Could not reach the backend event
  // log" while the backend was up and answering every other request.
  const [problem, setProblem] = useState(null);
  const [openEvent, setOpenEvent] = useState(null);
  const [newestFirst, setNewestFirst] = useState(true);

  const camera = useMemo(
    () => cameras.find((item) => item.id === selection) || null,
    [cameras, selection]
  );

  // Which RUN the events come from: the session this camera is on, or was
  // last on. Scoping to it is what keeps the two halves of this page
  // describing the same thing - re-running a camera replaces both the
  // summary and the event list, instead of showing new totals over old
  // events.
  //
  // null when the camera has not run in this backend process. That is NOT
  // the same as "no filter": leaving the scope off returns everything the
  // camera ever recorded, which would put yesterday's events under a "NOT
  // YET RUN" heading. An empty scope returns nothing, which is the true
  // answer.
  //
  // The backend decides this for a single camera - /api/cameras/{id}/events
  // reads its own runtime's session - so the only place this value is
  // needed here is the consolidated view and the empty-state wording.
  const sessionId =
    camera?.stats?.session_id ?? camera?.stats?.final?.session_id ?? null;

  // Has this camera concluded a run at all? Used only to word the empty
  // state, never to decide what to show.
  const everRan =
    (camera?.stats?.result_status || "NEVER_RUN") !== "NEVER_RUN";

  // The same rule across the fleet, for ALL CAMERAS. Joined into a string
  // so the effect re-runs when the SET changes, not on every render as a
  // freshly built array would cause.
  const allSessions = useMemo(
    () =>
      cameras
        .map(
          (item) =>
            item.stats?.session_id ?? item.stats?.final?.session_id ?? null
        )
        .filter((value) => value != null)
        .join(","),
    [cameras]
  );

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      // One camera asks the camera's own endpoint, which scopes to that
      // camera's current run server-side. The consolidated view asks the
      // fleet endpoint with the set of current runs. Either way the filter
      // is applied in SQL: the interface never receives another camera's
      // events and so cannot display them by mistake.
      const url =
        selection === "ALL"
          ? `${API}/api/events?session_ids=${encodeURIComponent(
              allSessions
            )}&limit=300`
          : `${API}/api/cameras/${encodeURIComponent(
              selection
            )}/events?limit=300`;

      let response;
      try {
        response = await apiFetch(url);
      } catch {
        // The request never completed. The backend is down, the port is
        // wrong, or CORS refused it before a status existed.
        if (cancelled) return;
        setEvents([]);
        setProblem({ kind: "network" });
        setLoading(false);
        return;
      }

      if (!response.ok) {
        // The backend IS reachable and answered. Say what it answered,
        // because 404 here means one specific, fixable thing: the running
        // process predates this endpoint and has to be restarted.
        if (cancelled) return;
        setEvents([]);
        setProblem({ kind: "http", status: response.status });
        setLoading(false);
        return;
      }

      try {
        const data = await response.json();
        if (cancelled) return;
        // The fleet endpoint returns a bare list; the camera endpoint
        // returns {camera_id, events}. Accept both rather than assuming.
        const list = Array.isArray(data) ? data : data?.events;
        setEvents(Array.isArray(list) ? list : []);
        setProblem(null);
      } catch {
        if (cancelled) return;
        setEvents([]);
        setProblem({ kind: "http", status: response.status });
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    setLoading(true);
    setOpenEvent(null);
    load();

    // Refreshed so a run finishing while this page is open shows up.
    const timer = setInterval(load, 4000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [selection, sessionId, allSessions]);

  const ordered = useMemo(() => {
    // Sorted, not recomputed. The backend already returns newest first;
    // this only honours the operator's choice of direction.
    const copy = [...events];
    copy.sort((a, b) =>
      newestFirst ? (b.ts ?? 0) - (a.ts ?? 0) : (a.ts ?? 0) - (b.ts ?? 0)
    );
    return copy;
  }, [events, newestFirst]);

  const result = camera?.stats?.final || null;
  const state = camera?.stats?.result_status || "NEVER_RUN";
  const concluded = state === "COMPLETED" || state === "STOPPED";

  return (
    <div className="page">

      <div className="page-intro">
        <div>
          <div className="section-label">RECORDED ACTIVITY</div>
          <h2>Detection Logs</h2>
          <p>
            Concluded detections and recorded alert events from
            completed camera processing sessions.
          </p>
        </div>
      </div>

      {/* -------------------------------------------------- */}
      {/* CAMERA SELECTOR                                     */}
      {/* -------------------------------------------------- */}

      <section className="panel log-selector-panel">
        <div className="log-selector">
          <div className="section-label">SELECT CAMERA</div>
          <div className="log-selector-row">
            <button
              className={`log-cam-btn ${
                selection === "ALL" ? "active" : ""
              }`}
              onClick={() => setSelection("ALL")}
            >
              ALL CAMERAS
            </button>

            {cameras.map((item) => (
              <button
                key={item.id}
                className={`log-cam-btn ${
                  selection === item.id ? "active" : ""
                }`}
                onClick={() => setSelection(item.id)}
              >
                {item.id}
                <span
                  className={`log-cam-dot state-${String(
                    item.stats?.result_status || "NEVER_RUN"
                  ).toLowerCase()}`}
                />
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* -------------------------------------------------- */}
      {/* CAMERA SUMMARY                                       */}
      {/* -------------------------------------------------- */}

      {selection !== "ALL" && camera && (
        <section className="panel log-summary-panel">
          <div className="log-summary-head">
            <div>
              <div className="log-summary-id">{camera.id}</div>
              <div className="log-summary-name">
                {camera.name}
                {camera.location ? ` — ${camera.location}` : ""}
              </div>
            </div>
            <div
              className={`log-session-state state-${state.toLowerCase()}`}
            >
              <div className="section-label">SESSION</div>
              {SESSION_WORDS[state] || state}
            </div>
          </div>

          <div className="log-summary-grid">
            <div className="log-metric">
              <div className="log-metric-label">PERSONS DETECTED</div>
              <div className="log-metric-value">
                {concluded && result ? result.persons : "--"}
              </div>
            </div>
            <div className="log-metric">
              <div className="log-metric-label">VEHICLES DETECTED</div>
              <div className="log-metric-value">
                {concluded && result ? result.vehicles : "--"}
              </div>
            </div>
            <div className="log-metric">
              <div className="log-metric-label">TOTAL ALERTS</div>
              <div className="log-metric-value">
                {concluded && result ? result.alerts : "--"}
              </div>
            </div>
          </div>

          {concluded && result ? (
            <div className="log-summary-note">
              unique identities over {result.frames_processed} analysed
              frames
              {result.completed ? "" : " — session stopped before the source ended"}
            </div>
          ) : (
            <div className="log-summary-note">
              {state === "PROCESSING"
                ? "run in progress — final figures are published when it ends"
                : "this camera has not completed a processing session yet"}
            </div>
          )}

          {/* TOTAL ALERTS and the timeline below are produced one for one:
              every incident raises exactly one alert and appears as exactly
              one row. If they ever disagree, something is broken and the
              operator is told, rather than being shown "4" above "0" and
              left to work it out. */}
          {concluded &&
            result &&
            problem == null &&
            !loading &&
            result.alerts !== events.length && (
              <div className="log-count-mismatch">
                This session concluded with {result.alerts} alert
                {result.alerts === 1 ? "" : "s"} but {events.length} event
                record{events.length === 1 ? "" : "s"} can be read back.
                The two are written together, so this gap means records
                were lost between the pipeline and the database.
              </div>
            )}
        </section>
      )}

      {/* -------------------------------------------------- */}
      {/* EVENT TIMELINE                                       */}
      {/* -------------------------------------------------- */}

      <section className="panel">
        <div className="panel-header">
          <h3>Alert &amp; Event Timeline</h3>
          <div className="log-timeline-tools">
            <span className="panel-count">
              {loading
                ? "LOADING"
                : `${ordered.length} EVENT${
                    ordered.length === 1 ? "" : "S"
                  }`}
            </span>
            <button
              className="log-sort-btn"
              onClick={() => setNewestFirst((value) => !value)}
            >
              {newestFirst ? "NEWEST FIRST" : "OLDEST FIRST"}
            </button>
          </div>
        </div>

        {problem?.kind === "network" ? (
          <div className="log-empty log-empty-error">
            <strong>BACKEND UNREACHABLE</strong>
            <span>
              No response from {API}. The API process is not running, or
              is not listening on that port.
            </span>
          </div>
        ) : problem?.kind === "http" ? (
          <div className="log-empty log-empty-error">
            <strong>EVENT LOG REFUSED ({problem.status})</strong>
            <span>
              {problem.status === 404
                ? "The backend is running but has no event endpoint. It is serving an older build of api/server.py — uvicorn does not reload on its own, so restart it."
                : `The backend answered ${problem.status} for the event log. Check its console output.`}
            </span>
          </div>
        ) : loading ? (
          <div className="log-empty">
            <span>Loading recorded events…</span>
          </div>
        ) : ordered.length === 0 ? (
          <div className="log-empty">
            <strong>NO EVENTS RECORDED</strong>
            <span>
              {selection === "ALL"
                ? "No camera has recorded an event in its current session."
                : !everRan
                ? `${selection} has not been run yet, so it has no recorded events.`
                : `${selection} recorded no events in this session.`}
            </span>
          </div>
        ) : (
          <div className="log-timeline">
            {ordered.map((event) => (
              <LogEventRow
                key={event.incident_id}
                event={event}
                showCamera={selection === "ALL"}
                expanded={openEvent === event.incident_id}
                onToggle={() =>
                  setOpenEvent(
                    openEvent === event.incident_id
                      ? null
                      : event.incident_id
                  )
                }
              />
            ))}
          </div>
        )}
      </section>

    </div>
  );
}


// ============================================================
// ONE RECORDED EVENT
// ============================================================
//
// Every field is printed as the backend sent it. A field the backend did
// not conclude - a behaviour it never classified, an incident that raised
// no alert - reads as its own state word rather than a plausible value.

function LogEventRow({ event, showCamera, expanded, onToggle }) {
  return (
    <div className={`log-event ${expanded ? "open" : ""}`}>

      <button className="log-event-head" onClick={onToggle}>

        <span className="log-event-time">
          {event.time || "--:--:--"}
        </span>

        <span
          className={`log-level ${threatClass(event.threat_level)}`}
        >
          {event.threat_level || "UNKNOWN"}
        </span>

        {showCamera && (
          <span className="log-event-cam">{event.camera_id}</span>
        )}

        <span className="log-event-actor">
          {event.object_label ||
            (event.track_id != null
              ? `Person #${String(event.track_id).padStart(2, "0")}`
              : "UNTRACKED")}
        </span>

        <span className="log-event-what">{event.headline}</span>

        <span className="log-event-chevron">
          {expanded ? "−" : "+"}
        </span>

      </button>

      {expanded && (
        <div className="log-event-detail">
          <dl>
            <div>
              <dt>CAMERA</dt>
              <dd>{event.camera_id}</dd>
            </div>
            <div>
              <dt>TRACK</dt>
              <dd>
                {event.track_id != null
                  ? `#${String(event.track_id).padStart(2, "0")}`
                  : "NOT TRACKED"}
              </dd>
            </div>
            <div>
              <dt>OBJECT</dt>
              <dd>{event.object_type || "UNKNOWN"}</dd>
            </div>
            <div>
              <dt>BEHAVIOUR</dt>
              <dd>{event.behaviour || "NOT CLASSIFIED"}</dd>
            </div>
            <div>
              <dt>THREAT LEVEL</dt>
              <dd className={threatClass(event.threat_level)}>
                {event.threat_level || "UNKNOWN"}
                {event.threat_score != null
                  ? ` (${event.threat_score})`
                  : ""}
              </dd>
            </div>
            <div>
              <dt>INCIDENT TYPE</dt>
              <dd>{event.incident_type}</dd>
            </div>
            <div>
              <dt>ZONE / LINE</dt>
              <dd>{event.zone_name || "NONE"}</dd>
            </div>
            <div>
              <dt>BOUNDARY</dt>
              <dd>{event.boundary_state || "NOT ASSESSED"}</dd>
            </div>
            <div>
              <dt>TIME</dt>
              <dd>{event.ts_text || "UNKNOWN"}</dd>
            </div>
            <div>
              <dt>STATUS</dt>
              <dd>{event.status}</dd>
            </div>
          </dl>

          {/* The sentence the operator was shown when this fired, not a
              second version of it written here. */}
          {event.message && (
            <div className="log-event-message">{event.message}</div>
          )}

          {/* Why the ThreatEngine scored it as it did, in its own words. */}
          {event.reasons?.length > 0 && (
            <div className="log-event-reasons">
              <div className="section-label">SCORING FACTORS</div>
              <ul>
                {event.reasons.map((reason, index) => (
                  <li key={index}>{reason}</li>
                ))}
              </ul>
            </div>
          )}

          {event.escalation_count > 0 && (
            <div className="log-event-repeat">
              this event recurred {event.escalation_count} time
              {event.escalation_count === 1 ? "" : "s"} and was folded into
              one record
            </div>
          )}
        </div>
      )}
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
// USER MANAGEMENT  (Super Admin only)
// ============================================================
//
// Every row here comes from GET /api/admin/users, which a normal user's
// token cannot call. Rendering this page without the role would therefore
// show an empty table and an error, not somebody else's user list - the
// page is not where the protection lives.

function UserManagement({ auth }) {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [problem, setProblem] = useState(null);
  const [creating, setCreating] = useState(false);
  const [editing, setEditing] = useState(null);

  const load = async () => {
    try {
      const response = await apiFetch(`${API}/api/admin/users`);
      if (!response.ok) {
        setProblem(
          response.status === 403
            ? "Super Admin authorization required."
            : `The backend answered ${response.status}.`
        );
        setUsers([]);
        return;
      }
      setUsers(await response.json());
      setProblem(null);
    } catch {
      setProblem(`No response from ${API}.`);
      setUsers([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const act = async (url, options) => {
    const response = await apiFetch(url, options);
    if (!response.ok) {
      let detail = `Request failed (${response.status}).`;
      try {
        detail = (await response.json())?.detail || detail;
      } catch {
        /* keep the status-only message */
      }
      return { ok: false, detail };
    }
    await load();
    return { ok: true };
  };

  const admins = users.filter((u) => u.role === "SUPER_ADMIN" && u.is_active);

  return (
    <div className="page">
      <div className="page-intro">
        <div>
          <div className="section-label">ACCESS CONTROL</div>
          <h2>User Management</h2>
          <p>
            Accounts that may sign in to this system, and what each one is
            allowed to do.
          </p>
        </div>
      </div>

      {problem && (
        <div className="api-error">{problem}</div>
      )}

      <section className="panel">
        <div className="panel-header">
          <h3>Accounts</h3>
          <div className="log-timeline-tools">
            <span className="panel-count">
              {loading ? "LOADING" : `${users.length} ACCOUNT${users.length === 1 ? "" : "S"}`}
            </span>
            <button className="log-sort-btn" onClick={() => setCreating(true)}>
              + CREATE USER
            </button>
          </div>
        </div>

        <div className="user-table">
          <div className="user-row user-head">
            <span>USER ID</span>
            <span>NAME</span>
            <span>ROLE</span>
            <span>STATUS</span>
            <span>LAST SIGN-IN</span>
            <span />
          </div>

          {users.map((user) => {
            const isSelf = user.user_id === auth.user.user_id;
            const lastAdmin =
              user.role === "SUPER_ADMIN" && user.is_active && admins.length <= 1;
            return (
              <div className="user-row" key={user.user_id}>
                <span className="mono">{user.user_id}</span>
                <span>
                  {user.username}
                  {isSelf && <em className="user-you"> you</em>}
                </span>
                <span className={`user-role ${user.role.toLowerCase()}`}>
                  {user.role === "SUPER_ADMIN" ? "SUPER ADMIN" : "NORMAL USER"}
                </span>
                <span
                  className={`user-status ${user.is_active ? "on" : "off"}`}
                >
                  <i />
                  {user.status}
                  {user.locked && <em className="user-locked">LOCKED</em>}
                </span>
                <span className="mono dim">
                  {user.last_login_ts
                    ? new Date(user.last_login_ts * 1000).toLocaleString()
                    : "never"}
                </span>
                <span className="user-actions">
                  <button onClick={() => setEditing(user)}>EDIT</button>
                  {user.locked && (
                    <button
                      onClick={() =>
                        act(`${API}/api/admin/users/${user.user_id}/unlock`, {
                          method: "POST",
                        })
                      }
                    >
                      UNLOCK
                    </button>
                  )}
                  <button
                    className={user.is_active ? "danger" : ""}
                    disabled={lastAdmin}
                    title={
                      lastAdmin
                        ? "The only active Super Admin cannot be disabled."
                        : ""
                    }
                    onClick={async () => {
                      const result = await act(
                        `${API}/api/admin/users/${user.user_id}`,
                        {
                          method: "PUT",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ is_active: !user.is_active }),
                        }
                      );
                      if (!result.ok) setProblem(result.detail);
                    }}
                  >
                    {user.is_active ? "DISABLE" : "ENABLE"}
                  </button>
                </span>
              </div>
            );
          })}

          {!loading && users.length === 0 && !problem && (
            <div className="log-empty">
              <strong>NO ACCOUNTS</strong>
              <span>Create the first operator account.</span>
            </div>
          )}
        </div>
      </section>

      {(creating || editing) && (
        <UserDialog
          user={editing}
          onClose={() => {
            setCreating(false);
            setEditing(null);
          }}
          onSubmit={async (payload) => {
            const result = editing
              ? await act(`${API}/api/admin/users/${editing.user_id}`, {
                  method: "PUT",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(payload),
                })
              : await act(`${API}/api/admin/users`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(payload),
                });
            if (result.ok) {
              setCreating(false);
              setEditing(null);
            }
            return result;
          }}
        />
      )}
    </div>
  );
}


function UserDialog({ user, onClose, onSubmit }) {
  const editing = Boolean(user);
  const [userId, setUserId] = useState(user?.user_id || "");
  const [username, setUsername] = useState(user?.username || "");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [role, setRole] = useState(user?.role || "NORMAL_USER");
  const [active, setActive] = useState(user ? user.is_active : true);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const mismatch = password !== confirm;

  const submit = async (event) => {
    event.preventDefault();
    if (busy) return;
    if (password && mismatch) {
      setError("The two passwords do not match.");
      return;
    }
    setBusy(true);
    setError(null);

    // On edit, only the fields that were actually changed are sent. An
    // empty password field means "leave the password alone", never "set the
    // password to empty".
    const payload = editing
      ? {
          username,
          role,
          is_active: active,
          ...(password ? { password } : {}),
        }
      : { user_id: userId, username, password, role, is_active: active };

    const result = await onSubmit(payload);
    setBusy(false);
    if (!result.ok) setError(result.detail);
  };

  return (
    <div className="session-overlay" onClick={onClose}>
      <div
        className="session-dialog user-dialog"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="section-label">
          {editing ? `EDIT ${user.user_id}` : "CREATE USER"}
        </div>

        {error && <div className="login-error"><span>{error}</span></div>}

        <form onSubmit={submit}>
          {!editing && (
            <label className="field">
              <span>USER ID</span>
              <input
                value={userId}
                autoFocus
                spellCheck="false"
                placeholder="USER-002"
                onChange={(e) => setUserId(e.target.value.toUpperCase())}
              />
            </label>
          )}

          <label className="field">
            <span>USERNAME</span>
            <input
              value={username}
              placeholder="Operator-01"
              onChange={(e) => setUsername(e.target.value)}
            />
          </label>

          <label className="field">
            <span>
              PASSWORD{editing && <em> — leave blank to keep the current one</em>}
            </span>
            <input
              type="password"
              value={password}
              autoComplete="new-password"
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>

          <label className="field">
            <span>CONFIRM PASSWORD</span>
            <input
              type="password"
              value={confirm}
              autoComplete="new-password"
              onChange={(e) => setConfirm(e.target.value)}
            />
            {password && mismatch && (
              <em className="field-warn">The passwords do not match.</em>
            )}
          </label>

          <div className="field-row">
            <label className="field">
              <span>ROLE</span>
              <select value={role} onChange={(e) => setRole(e.target.value)}>
                <option value="NORMAL_USER">NORMAL USER</option>
                <option value="SUPER_ADMIN">SUPER ADMIN</option>
              </select>
            </label>

            <label className="field">
              <span>STATUS</span>
              <select
                value={active ? "ACTIVE" : "DISABLED"}
                onChange={(e) => setActive(e.target.value === "ACTIVE")}
              >
                <option value="ACTIVE">ACTIVE</option>
                <option value="DISABLED">DISABLED</option>
              </select>
            </label>
          </div>

          {role === "SUPER_ADMIN" && (
            <div className="user-warn">
              A Super Admin can manage every user, every camera and every
              setting, and can read the audit log. Create one only when
              somebody genuinely needs that.
            </div>
          )}

          <div className="session-actions">
            <button className="login-btn" type="submit" disabled={busy}>
              {busy ? "SAVING…" : editing ? "SAVE CHANGES" : "CREATE USER"}
            </button>
            <button className="session-out" type="button" onClick={onClose}>
              CANCEL
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}


// ============================================================
// AUDIT LOG  (Super Admin only)
// ============================================================

const AUDIT_TONE = {
  LOGIN_SUCCESS: "ok",
  LOGOUT: "ok",
  USER_ENABLED: "ok",
  USER_CREATED: "ok",
  LOGIN_FAILED: "warn",
  LOGIN_BLOCKED: "bad",
  ACCESS_DENIED: "bad",
  USER_DISABLED: "warn",
  USER_DELETED: "bad",
  PASSWORD_RESET: "warn",
  SESSION_EXPIRED: "dim",
};

function AuditLogs() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [problem, setProblem] = useState(null);
  const [only, setOnly] = useState("ALL");

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await apiFetch(
          `${API}/api/admin/audit-logs?limit=300`
        );
        if (!response.ok) {
          if (!cancelled) {
            setProblem(
              response.status === 403
                ? "Super Admin authorization required."
                : `The backend answered ${response.status}.`
            );
            setRows([]);
          }
          return;
        }
        const data = await response.json();
        if (!cancelled) {
          setRows(Array.isArray(data) ? data : []);
          setProblem(null);
        }
      } catch {
        if (!cancelled) setProblem(`No response from ${API}.`);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    const timer = setInterval(load, 6000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const shown =
    only === "ALL"
      ? rows
      : rows.filter((r) =>
          only === "SECURITY"
            ? r.status !== "SUCCESS"
            : r.action.startsWith("USER_") || r.action === "PASSWORD_RESET"
        );

  return (
    <div className="page">
      <div className="page-intro">
        <div>
          <div className="section-label">SECURITY RECORD</div>
          <h2>Audit Logs</h2>
          <p>
            Every sign-in, every refusal and every privileged action, in the
            order it happened. A refused request is recorded as carefully as
            a successful one.
          </p>
        </div>
      </div>

      {problem && <div className="api-error">{problem}</div>}

      <section className="panel">
        <div className="panel-header">
          <h3>Recorded actions</h3>
          <div className="log-timeline-tools">
            <span className="panel-count">
              {loading ? "LOADING" : `${shown.length} RECORDS`}
            </span>
            {["ALL", "SECURITY", "ACCOUNTS"].map((f) => (
              <button
                key={f}
                className={`log-sort-btn ${only === f ? "active" : ""}`}
                onClick={() => setOnly(f)}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        <div className="audit-table">
          <div className="audit-row audit-head">
            <span>TIME</span>
            <span>USER</span>
            <span>ACTION</span>
            <span>RESOURCE</span>
            <span>RESULT</span>
          </div>

          {shown.map((row) => (
            <div className="audit-row" key={row.id}>
              <span className="mono">{row.time}</span>
              <span className="mono user">{row.user_id || "—"}</span>
              <span
                className={`audit-action ${AUDIT_TONE[row.action] || ""}`}
              >
                {row.action}
              </span>
              <span className="mono dim">{row.resource || "—"}</span>
              <span className={`audit-status ${row.status.toLowerCase()}`}>
                {row.status}
              </span>
            </div>
          ))}

          {!loading && shown.length === 0 && !problem && (
            <div className="log-empty">
              <strong>NO RECORDS</strong>
              <span>Nothing matching this filter has been recorded.</span>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}



// ============================================================
// EXPORT
// ============================================================
//
// AuthGate, not App. Nothing in the dashboard renders until the backend has
// confirmed who is asking.

export default AuthGate;