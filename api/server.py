from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import anyio
import cv2
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api import deps
from api.auth_routes import admin_router, auth_router
from api.deps import AuthSettings
from core.auth import AuditAction, Principal
from core.config import load_config
from core.pipeline import SurveillancePipeline
from database.database import Database
from inputs.factory import build_source, describe_source


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Border Guard AI API",
    version="1.0.0",
)

#: Worker threads available to synchronous endpoints.
#
# anyio defaults to 40. Every `def` endpoint borrows one for the length of
# the call, so the ones that touch SQLite - alerts, incidents, plates - are
# what really need them. The MJPEG streams used to borrow one EACH,
# permanently, which is what made 40 the number that mattered. That is fixed
# by making the stream async; this raise is belt and braces, not the fix.
THREADPOOL_THREADS = 80

#: How long a stream waits for the camera to produce its first frame before
#: giving up. Without a bound, a stream opened against a camera that is not
#: running waits for ever.
STREAM_FIRST_FRAME_TIMEOUT = 10.0

#: Target pacing for the MJPEG stream. The camera thread produces frames at
#: its own rate; this only decides how often the stream checks for a new one.
STREAM_INTERVAL = 0.03

#: Stats that describe THIS RUN of a camera, and nothing else.
#
# They are zeroed when a camera starts and again when it stops, so a stopped
# camera reads zero and a restarted one begins from zero rather than
# inheriting the last session. Without this, CAM-01 sat at OFFLINE while
# still reporting "alerts 3, incidents 3, detections 120" from a run that had
# already finished - measured, not hypothetical.
#
# The cumulative history is NOT lost: every alert, incident and evidence file
# is in the database with its camera_id, and /api/alerts, /api/incidents and
# the `database` block of /api/dashboard still serve it. Live state and
# historical state are different questions and now have different sources.
LIVE_COUNTER_KEYS = {
    "frame_index": 0,
    "persons": 0,
    "vehicles": 0,
    "tracks": 0,
    "detections": 0,
    "incidents": 0,
    "alerts": 0,
    "evidence": 0,
    "plates": 0,
    "poses": 0,
    "fps": 0.0,
    "inference_ms": 0.0,
    "pose_ms": 0.0,
    # Scene measurements, not configuration. A camera that is not running has
    # not measured anything, so it says so instead of reporting the light it
    # saw last time.
    "low_light": False,
    "luminance": None,
    "night_context": "Not started",
    "pose": "Not started",
}


@app.on_event("startup")
async def _widen_threadpool() -> None:
    try:
        anyio.to_thread.current_default_thread_limiter().total_tokens = (
            THREADPOOL_THREADS
        )
    except Exception as exc:  # pragma: no cover - defensive only
        print(f"Could not raise the thread limiter: {exc}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS / CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAMERAS_FILE = PROJECT_ROOT / "config" / "cameras.yaml"

config = load_config()

db = Database(
    config.path("database_file")
)

# ============================================================
# AUTHENTICATION
# ============================================================
#
# Wired here, at the edge, and nowhere else. `core/` is untouched: no engine
# knows a user exists, and the pipeline behaves identically whether the
# users table has one row or none. Authentication decides who may call the
# API; it has no opinion about what the API then does.
#
# The routers are mounted at the bottom of this module, after every existing
# endpoint has been defined.

auth_settings = AuthSettings.from_config(config)
deps.configure(db, auth_settings)


# ============================================================
# CAMERA RUNTIME
# ============================================================

class CameraRuntime:
    def __init__(self, camera: dict[str, Any]):
        self.camera = camera
        self.camera_id = str(camera["id"])

        self.source = None
        self.pipeline = None
        self.session_id = None

        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()

        #: Guards `latest_frame` AND `stats`. It used to guard only the
        #: frame, so a request thread could read a half-applied stats
        #: update while the camera thread was in the middle of `.update()`.
        self.lock = threading.Lock()
        #: Serialises start/stop. `start` used to check `thread.is_alive()`
        #: and then create a thread, which two concurrent POSTs could both
        #: pass, giving one runtime two processing threads on one source.
        self.control_lock = threading.RLock()

        self.latest_frame: bytes | None = None
        #: Wall time the newest frame was encoded, so a client can tell a
        #: live feed from a frozen one.
        self.latest_frame_ts: float = 0.0

        #: The CONCLUDED result of the last completed run of this camera.
        #:
        #: None until this camera has finished a run. Survives the camera
        #: going OFFLINE - that is the point of it - and is replaced, never
        #: added to, when a new run completes. Belongs to this runtime only,
        #: so CAM-02 finishing cannot touch CAM-01's result.
        self.final_stats: dict[str, Any] | None = None
        #: NEVER_RUN / PROCESSING / COMPLETED / STOPPED / ERROR.
        #: Answers "can I trust the numbers on this card yet", which the
        #: live status cannot: a camera is OFFLINE both before its first run
        #: and after a finished one, and those mean different things.
        self.result_status: str = "NEVER_RUN"

        self.stats = {
            "camera_id": self.camera_id,
            "status": "OFFLINE",
            "frame_index": 0,
            "persons": 0,
            "vehicles": 0,
            "tracks": 0,
            "detections": 0,
            "incidents": 0,
            "alerts": 0,
            "evidence": 0,
            "plates": 0,
            "fps": 0.0,
            "inference_ms": 0.0,
            "anomaly": "disabled",
            "anpr": "disabled",
            # Nothing has been measured before the camera starts, so these
            # say so rather than defaulting to a comfortable "normal light".
            "low_light": False,
            "luminance": None,
            "night_context": "Not started",
            "pose": "Not started",
            "poses": 0,
            "pose_ms": 0.0,
            "zones": 0,
            "lines": 0,
            "zones_profile": None,
            "geometry": "Not started",
            "error": None,
            # Additive, for the operations view. `stream` is what a client
            # should believe about the picture, which is not the same as
            # whether the pipeline thread is alive: a camera can be ONLINE
            # and still have produced no frame yet.
            "stream": "IDLE",
            "frame_age": None,
            "source_ended": False,
        }

    # --------------------------------------------------------
    # Stats access
    # --------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """A consistent copy of the stats for a request thread."""
        with self.lock:
            data = dict(self.stats)
            age = (
                None if self.latest_frame_ts == 0.0
                else round(time.time() - self.latest_frame_ts, 2)
            )
        data["frame_age"] = age
        data["result_status"] = self.result_status
        data["final"] = self.final_stats
        # WHICH RUN this camera is on, or was last on. None until it has
        # started once. Detection Logs scopes its event list to this, so a
        # camera that has never run in this process shows no events rather
        # than events from a session somebody ran yesterday - which would
        # sit under a "NOT YET RUN" heading and contradict it.
        data["session_id"] = self.session_id
        if data["status"] in ("OFFLINE", "ERROR"):
            data["stream"] = "IDLE"
        elif age is None:
            data["stream"] = "WAITING"
        elif age > 2.0:
            data["stream"] = "STALLED"
        else:
            data["stream"] = "LIVE"
        return data

    def _set(self, **values: Any) -> None:
        with self.lock:
            self.stats.update(values)

    def reset_live_stats(self) -> None:
        """Zero everything that describes a single run of this camera.

        Called on START so a restart never inherits the previous session,
        and on STOP so a camera that is not running reports zero. Only this
        runtime is touched, so stopping CAM-01 cannot disturb CAM-02.
        """
        with self.lock:
            self.stats.update(LIVE_COUNTER_KEYS)
            self.latest_frame = None
            self.latest_frame_ts = 0.0

    def _finalise_run(self) -> None:
        """Conclude the run and keep the result for the dashboard card.

        Reads `SurveillancePipeline.session_totals()`, which counts DISTINCT
        confirmed identities rather than what happened to be in the last
        frame. A clip where ten people walk through and two remain at the
        end concludes with ten, not two.

        Only replaces a previous result, never accumulates onto it: running
        a camera twice reports the second run's numbers, not their sum.
        """
        pipeline = self.pipeline
        if pipeline is None:
            # Never got as far as building a pipeline; nothing was analysed,
            # so there is no result to report and the old one stands.
            return
        try:
            totals = pipeline.session_totals()
        except Exception as exc:
            print(f"[{self.camera_id}] could not finalise: {exc}")
            return

        with self.lock:
            ended = bool(self.stats.get("source_ended"))
            errored = self.stats.get("status") == "ERROR"

        self.final_stats = {
            "camera_id": self.camera_id,
            **totals,
            # WHICH run these numbers describe. Detection Logs uses it to
            # pull exactly this session's events, so re-running a camera
            # shows the new run's events beside the new run's totals rather
            # than every event the camera has ever produced.
            "session_id": self.session_id,
            "finished_ts": time.time(),
            # How the run ended, because "3 people" from a clip that played
            # out means something different from "3 people" from a clip the
            # operator cut short after two seconds.
            "completed": ended,
        }
        self.result_status = (
            "ERROR" if errored else "COMPLETED" if ended else "STOPPED"
        )
        print(f"[{self.camera_id}] run concluded: "
              f"{totals['persons']} unique person(s), "
              f"{totals['vehicles']} vehicle(s), "
              f"{totals['alerts']} alert(s) "
              f"over {totals['frames_processed']} analysed frames")

    @property
    def is_live(self) -> bool:
        """Is this runtime still expected to produce frames?"""
        return (
            self.thread is not None
            and self.thread.is_alive()
            and not self.stop_event.is_set()
        )

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    def start(self):
        with self.control_lock:
            if self.thread and self.thread.is_alive():
                return

            self.stop_event.clear()

            # A restart must inherit nothing from the last run: not the
            # picture, or the stream shows a stale frame a client cannot
            # tell from a live one, and not the counters, or CAM-01 appears
            # to have already raised three alerts the moment it starts.
            self.reset_live_stats()
            # A new run is under way, so the previous conclusion no longer
            # describes what is happening. The card shows PROCESSING rather
            # than last run's numbers, which would read as this run's.
            self.result_status = "PROCESSING"

            self._set(status="STARTING", error=None, source_ended=False)

            self.thread = threading.Thread(
                target=self._run,
                daemon=True,
                name=f"camera-{self.camera_id}",
            )

            self.thread.start()

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    def stop(self):
        with self.control_lock:
            self.stop_event.set()

            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5)

            # Reset first, then set the status, so a client polling in
            # between never sees OFFLINE next to last session's numbers.
            self.reset_live_stats()
            self._set(status="OFFLINE")

    # --------------------------------------------------------
    # PROCESS CAMERA
    # --------------------------------------------------------

    def _run(self):

        try:
            self._set(status="CONNECTING", error=None)

            # ------------------------------------------------
            # Open video source
            # ------------------------------------------------

            self.source = build_source(
                self.camera,
                input_dir=config.path("input_dir"),
            )

            self.source.open()

            # ------------------------------------------------
            # Database session
            # ------------------------------------------------

            self.session_id = db.start_session(
                camera_id=self.camera_id,
                source_detail=describe_source(self.camera),
                model_path=str(config.model_path()),
                device=str(
                    config.get(
                        "detection.device",
                        "auto",
                    )
                ),
            )

            # ------------------------------------------------
            # Existing AI pipeline
            # ------------------------------------------------

            self.pipeline = SurveillancePipeline(
                config=config,
                camera=self.camera,
                database=db,
                session_id=self.session_id,
            )

            self.pipeline.persist_zones()

            self._set(status="ONLINE")

            print(
                f"[{self.camera_id}] AI pipeline started"
            )

            # ------------------------------------------------
            # Main processing loop
            # ------------------------------------------------

            while not self.stop_event.is_set():

                frame = self.source.read()

                if frame is None:
                    print(
                        f"[{self.camera_id}] Video source ended"
                    )
                    # Tell anything attached that this camera is finished.
                    # Without it the loop exited, the status went OFFLINE,
                    # and every MJPEG generator kept running against the
                    # frozen last frame because it only ever checked
                    # `stop_event`.
                    self._set(source_ended=True)
                    self.stop_event.set()
                    break

                try:
                    output = self.pipeline.process(
                        frame.image,
                        frame.index,
                        frame.ts,
                    )

                except Exception as exc:
                    print(
                        f"[{self.camera_id}] "
                        f"Frame processing error: {exc}"
                    )
                    continue

                # --------------------------------------------
                # Statistics
                # --------------------------------------------

                self._set(
                    **{
                        "status": "ONLINE",
                        "frame_index": output.frame_index,
                        "persons": output.persons,
                        "vehicles": output.vehicles,
                        "tracks": self.pipeline.tracks.live_count,
                        "detections": self.pipeline.stats.detections,
                        "incidents": self.pipeline.stats.incidents,
                        "alerts": self.pipeline.stats.alerts,
                        "evidence": self.pipeline.stats.evidence_files,
                        "plates": self.pipeline.stats.plates_read,
                        "fps": self.pipeline.stats.fps,
                        "inference_ms":
                            self.pipeline.detector.stats.avg_inference_ms,
                        "anomaly":
                            self.pipeline.anomaly.status_text(),
                        "anpr":
                            self.pipeline.anpr.status_text(),
                        # Scene context, measured from the frames by
                        # core/low_light.py. Three flat keys alongside the
                        # existing ones; the stats shape is unchanged.
                        "low_light":
                            self.pipeline.low_light_state.low_light,
                        "luminance": (
                            round(self.pipeline.low_light_state.luminance, 1)
                            if self.pipeline.low_light_state.luminance
                            is not None else None
                        ),
                        "night_context":
                            self.pipeline.low_light_state.status_text(),
                        # Pose: real counters from core/pose.py. `poses` is
                        # how many tracked people currently carry keypoints,
                        # not how many the model was asked about.
                        "pose": self.pipeline.pose.status_text(),
                        "poses": sum(
                            1 for t in
                            self.pipeline.tracks.tracks.values()
                            if t.pose is not None
                        ),
                        "pose_ms": round(
                            self.pipeline.pose.stats.ms_per_crop, 1
                        ),
                        # Security geometry for THIS camera. Read from the
                        # engines, so the dashboard shows what the pipeline is
                        # really evaluating against. A camera with no polygon
                        # reports zero and says so.
                        "zones":
                            self.pipeline.geometry_state["zones"],
                        "lines":
                            self.pipeline.geometry_state["lines"],
                        "zones_profile":
                            self.pipeline.geometry_state["profile"],
                        "geometry":
                            self.pipeline.geometry_state["summary"],
                        "error": None,
                    }
                )

                # --------------------------------------------
                # Latest annotated frame
                # --------------------------------------------

                image = (
                    output.annotated
                    if output.annotated is not None
                    else output.raw
                )

                if image is not None:

                    success, encoded = cv2.imencode(
                        ".jpg",
                        image,
                        [
                            cv2.IMWRITE_JPEG_QUALITY,
                            80,
                        ],
                    )

                    if success:

                        with self.lock:
                            self.latest_frame = encoded.tobytes()
                            self.latest_frame_ts = time.time()

                # Small yield so the API remains responsive
                time.sleep(0.001)

        except Exception as exc:

            self._set(status="ERROR", error=str(exc))
            # An errored camera will never produce another frame, so any
            # stream attached to it must be released rather than left
            # waiting on a picture that is not coming.
            self.stop_event.set()

            print(
                f"[{self.camera_id}] ERROR: {exc}"
            )

        finally:

            # ------------------------------------------------
            # Close video source
            # ------------------------------------------------

            if self.source is not None:
                try:
                    self.source.release()
                except Exception:
                    pass

            # ------------------------------------------------
            # Finalise AI pipeline
            # ------------------------------------------------

            if self.pipeline is not None:

                try:
                    self.pipeline.finalise()

                    stats = self.pipeline.stats

                    if self.session_id is not None:
                        db.end_session(
                            session_id=self.session_id,
                            frames_read=stats.frames_read,
                            frames_processed=stats.frames_processed,
                            detections=stats.detections,
                            incidents=stats.incidents,
                        )

                except Exception as exc:
                    print(
                        f"[{self.camera_id}] "
                        f"Finalisation error: {exc}"
                    )

            # THE COMPLETION POINT. The source ended, the operator stopped
            # the camera, or the thread is unwinding after an error. Before
            # the live counters are cleared, conclude the run: this is the
            # only moment at which the whole session is still in memory.
            self._finalise_run()

            # The run is over, so its live counters are over with it.
            self.reset_live_stats()
            self._set(status="OFFLINE")
            self.stop_event.set()


# ============================================================
# LOAD CAMERAS
# ============================================================

def load_cameras():

    with CAMERAS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = yaml.safe_load(file) or {}

    return data.get("cameras", [])


cameras = load_cameras()


# ============================================================
# CAMERA RUNTIMES
# ============================================================

runtimes: dict[str, CameraRuntime] = {}


for camera in cameras:

    camera_id = str(camera["id"])

    runtimes[camera_id] = CameraRuntime(
        camera
    )

    # Keep database camera list synchronized
    source = camera.get("source", {})

    source_type = str(
        source.get("type", "")
    )

    if source_type == "video":
        source_detail = str(
            source.get("path", "")
        )

    elif source_type == "webcam":
        source_detail = (
            f"Camera {source.get('index', 0)}"
        )

    else:
        source_detail = str(
            source.get("url", "")
        )

    db.upsert_camera(
        camera_id=camera_id,
        name=str(
            camera.get(
                "name",
                camera_id,
            )
        ),
        location=camera.get("location"),
        source_type=source_type,
        source_detail=source_detail,
        zones_profile=str(
            camera.get(
                "zones_profile",
                "default",
            )
        ),
        enabled=bool(
            camera.get(
                "enabled",
                False,
            )
        ),
    )


# ============================================================
# HEALTH
# ============================================================

# The compiled dashboard, if it has been built. `npm run build` writes it;
# `npm run dev` does not. Resolved once, at import, because the answer cannot
# change while the process is alive.
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"


@app.get("/")
async def root():
    """The dashboard when it has been built, otherwise a status document.

    WHY THE BACKEND SERVES THE FRONTEND
    -----------------------------------
    Running `uvicorn` and `npm run dev` side by side means two processes, two
    ports and two chances for one of them to be stale. That is not a
    hypothetical: a uvicorn left running from before an endpoint existed has
    already produced two separate "the app is broken" reports on this project,
    both of which were really "the old server is still up". One process cannot
    disagree with itself.

    The JSON fallback is kept for the case where nobody has built the frontend
    yet, so a fresh clone still answers on `/` instead of 404-ing, and says
    what to do about it.
    """
    if FRONTEND_INDEX.is_file():
        # no-store: the HTML names hashed asset files. If a browser keeps an
        # old copy of *this* document it will ask for assets that a rebuild has
        # already renamed. The assets themselves are content-hashed and cached
        # hard by the mount below; only this one entry point must stay fresh.
        return FileResponse(FRONTEND_INDEX, headers={"Cache-Control": "no-store"})

    return {
        "name": "Border Guard AI",
        "status": "online",
        "version": "1.0.0",
        "dashboard": "not built",
        "hint": "Run `npm run build` in frontend/ to serve the dashboard here.",
    }


@app.get("/api/health")
async def health():

    return {
        "status": "online",
        "service": "Border Guard AI API",
    }


# ============================================================
# CAMERAS
# ============================================================

@app.get("/api/cameras")
async def get_cameras(_user: Principal = Depends(deps.require_user)):

    result = []

    for camera in cameras:

        camera_id = str(camera["id"])
        snap = runtimes[camera_id].snapshot()

        result.append(
            {
                "id": camera_id,
                "name": camera.get(
                    "name",
                    camera_id,
                ),
                "location": camera.get(
                    "location",
                    "",
                ),
                "enabled": camera.get(
                    "enabled",
                    False,
                ),
                "source": camera.get(
                    "source",
                    {},
                ),
                "status": snap["status"],
                "stream": snap["stream"],
                "frame_age": snap["frame_age"],
                "error": snap["error"],
            }
        )

    return result


# ============================================================
# START CAMERA
# ============================================================

@app.post("/api/cameras/{camera_id}/start")
async def start_camera(
    camera_id: str,
    _user: Principal = Depends(deps.require_camera_control),
):

    if camera_id not in runtimes:
        raise HTTPException(
            status_code=404,
            detail="Camera not found",
        )

    runtime = runtimes[camera_id]

    runtime.start()

    return {
        "success": True,
        "camera_id": camera_id,
        "status": "STARTING",
    }


# ============================================================
# STOP CAMERA
# ============================================================

@app.post("/api/cameras/{camera_id}/stop")
async def stop_camera(
    camera_id: str,
    _user: Principal = Depends(deps.require_camera_control),
):

    if camera_id not in runtimes:
        raise HTTPException(
            status_code=404,
            detail="Camera not found",
        )

    runtime = runtimes[camera_id]

    runtime.stop()

    return {
        "success": True,
        "camera_id": camera_id,
        "status": "OFFLINE",
    }


# ============================================================
# CAMERA STATS
# ============================================================

@app.get("/api/cameras/{camera_id}/stats")
async def camera_stats(camera_id: str, _user: Principal = Depends(deps.require_user)):

    if camera_id not in runtimes:
        raise HTTPException(
            status_code=404,
            detail="Camera not found",
        )

    return runtimes[camera_id].snapshot()


# ============================================================
# LIVE PER-TRACK INTELLIGENCE
# ============================================================

@app.get("/api/cameras/{camera_id}/tracks")
async def camera_tracks(camera_id: str, _user: Principal = Depends(deps.require_user)):
    """One row per tracked person, with what the pipeline concluded.

    Small on purpose. This is polled alongside the stats, so it carries the
    current state of the people in view and nothing historical - no
    trajectories, no keypoint arrays. The heavy per-frame data stays in the
    annotated MJPEG frame where it is already being drawn.

    Every field comes from the running pipeline. A person the engines know
    nothing about yet reports UNKNOWN, UNAVAILABLE or null; none of these are
    filled in with a plausible default. `threat_level` in particular is null
    until the ThreatEngine has actually scored that person, because the level
    that would look natural to default to - LOW - is the one that says
    "nothing to see here".
    """
    if camera_id not in runtimes:
        raise HTTPException(
            status_code=404,
            detail="Camera not found",
        )

    runtime = runtimes[camera_id]
    pipeline = runtime.pipeline

    if pipeline is None:
        return {
            "camera_id": camera_id,
            "status": runtime.snapshot()["status"],
            "tracks": [],
        }

    try:
        rows = pipeline.people()
    except Exception as exc:
        # Enrichment must never take the camera list down with it.
        print(f"[{camera_id}] track intelligence failed: {exc}")
        rows = []

    return {
        "camera_id": camera_id,
        "status": runtime.snapshot()["status"],
        "tracks": rows,
    }


# ============================================================
# LIVE MJPEG STREAM
# ============================================================

async def generate_stream(runtime: CameraRuntime):
    """Yield MJPEG parts for one camera.

    ASYNC, and that is the whole point of this function.

    It used to be a plain generator with `time.sleep`. Starlette drives a
    synchronous iterator through `iterate_in_threadpool`, which borrows one
    anyio worker thread for every `next()` call. That was survivable while
    every `next()` returned quickly - but when the camera had produced no
    frame yet, the old body looped `time.sleep(0.05); continue` INSIDE a
    single `next()` and never returned, so the worker was gone for good.
    Starlette also never got a chance to notice the client had hung up,
    because it was still waiting for a value to send.

    Measured on the old code, with streams opened against cameras that were
    not running: 38 of them left `/api/health` answering in 0.00s; 45 made
    every endpoint time out, and it never recovered - 30 seconds after
    every client had disconnected the API was still dead. anyio's default
    limiter is exactly 40. That is the whole of "the dashboard says BACKEND
    NOT CONNECTED while FastAPI is running", and most of "one camera works
    and another fails", since a start request also needs a worker.

    As a coroutine it runs on the event loop, borrows no worker at all, and
    a client disconnect raises on the next send so it unwinds cleanly.

    It is also bounded now. Three ways out:
      * the runtime is no longer live - stopped, errored, or the file ended;
      * the camera produced no first frame within the timeout;
      * the client went away, which raises out of the yield.
    """
    waited_for_first = 0.0
    sent_any = False

    try:
        while True:
            with runtime.lock:
                frame = runtime.latest_frame

            if frame is None:
                # Nothing to send yet. Either the camera is still warming
                # up, or it is not running at all and never will be.
                if not runtime.is_live and not sent_any:
                    break
                if not sent_any:
                    waited_for_first += STREAM_INTERVAL
                    if waited_for_first >= STREAM_FIRST_FRAME_TIMEOUT:
                        break
                elif not runtime.is_live:
                    break
                await asyncio.sleep(STREAM_INTERVAL)
                continue

            if not runtime.is_live and sent_any:
                # The camera has finished. Stop rather than republishing a
                # frozen last frame for ever.
                break

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
            sent_any = True

            await asyncio.sleep(STREAM_INTERVAL)

    except asyncio.CancelledError:
        # Normal: the client closed the tab or React unmounted the <img>.
        raise


@app.get("/api/cameras/{camera_id}/stream")
async def camera_stream(camera_id: str, request: Request,
                        token: str | None = None):
    """The live MJPEG feed. Authenticated, but not by a header.

    A browser loads this through an `<img src="...">`, and it will not put an
    Authorization header on an image request. The choice is a query token or
    an unauthenticated video feed, and an unauthenticated feed is not a
    choice: the URL is guessable and would hand anybody the cameras.

    The token is the same session token and is validated exactly the same
    way. The cost is that it appears in the server's access log, which is
    why this is the ONLY route that accepts one this way - everything else
    requires the header.
    """
    principal = deps.resolve_principal(request)
    if principal is None and token:
        principal = deps.principal_from_token(token)
    if principal is None:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if camera_id not in runtimes:
        raise HTTPException(
            status_code=404,
            detail="Camera not found",
        )

    runtime = runtimes[camera_id]

    return StreamingResponse(
        generate_stream(runtime),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            # An MJPEG feed must never be cached, and a proxy must not try
            # to buffer it into a single response.
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/api/dashboard")
async def dashboard(_user: Principal = Depends(deps.require_user)):

    active_cameras = 0
    persons = 0
    vehicles = 0
    tracks = 0
    detections = 0
    incidents = 0
    alerts = 0

    # Per-camera rows, so a client never has to reconstruct which camera
    # contributed what. The fleet totals below are a SUM of exactly these
    # rows and nothing else.
    per_camera = {}

    for runtime in runtimes.values():

        stats = runtime.snapshot()
        running = stats["status"] == "ONLINE"

        # Only a RUNNING camera contributes to the live totals. A stopped
        # camera has no live state, and counting its last session made the
        # dashboard read "0 cameras active, 3 alerts, 120 detections", which
        # is a sentence that cannot be true.
        if running:
            active_cameras += 1
            persons += int(stats["persons"])
            vehicles += int(stats["vehicles"])
            tracks += int(stats["tracks"])
            detections += int(stats["detections"])
            incidents += int(stats["incidents"])
            alerts += int(stats["alerts"])

        per_camera[runtime.camera_id] = {
            "camera_id": runtime.camera_id,
            "status": stats["status"],
            "stream": stats["stream"],
            "running": running,
            "persons": int(stats["persons"]),
            "vehicles": int(stats["vehicles"]),
            "tracks": int(stats["tracks"]),
            "detections": int(stats["detections"]),
            "incidents": int(stats["incidents"]),
            "alerts": int(stats["alerts"]),
        }

    database_counts = db.analytics_counts()

    return {
        # LIVE: the cameras running right now, and nothing else.
        "cameras_active": active_cameras,
        "cameras_total": len(runtimes),
        "persons": persons,
        "vehicles": vehicles,
        "tracks": tracks,
        "detections": detections,
        "incidents": incidents,
        "alerts": alerts,
        "per_camera": per_camera,
        # HISTORICAL: everything ever recorded, from the database, with its
        # own camera_id on every row. Deliberately a separate block so the
        # two can never be mistaken for one another.
        "database": database_counts,
    }


# ============================================================
# ALERTS
# ============================================================

@app.get("/api/alerts")
def alerts(_user: Principal = Depends(deps.require_user)):

    rows = db.get_alerts(
        limit=100
    )

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# INCIDENTS
# ============================================================

@app.get("/api/incidents")
def incidents(_user: Principal = Depends(deps.require_user)):

    rows = db.get_incidents(
        limit=100
    )

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# EVENTS
# ============================================================
#
# The recorded event log behind the Detection Logs page.
#
# Every value on a row was written by the Python pipeline at the moment the
# event happened and read back out of SQLite here. Nothing on this endpoint
# is computed for display: the threat level is the ThreatEngine's verdict,
# the timestamp is the processing time of the frame that raised it, the
# behaviour is what the classifier had concluded for that person on that
# frame, and `message` is the exact sentence AlertManager showed the
# operator. React renders these; it does not derive them.

#: Behaviour label -> the short sentence an operator reads. The classifier's
#: labels are enum-shaped (BENT_RUN) and an operator should not have to
#: decode them at 3am. Only formatting - no behaviour is invented here, and
#: a label with no entry falls back to its own words rather than a guess.
BEHAVIOUR_HEADLINES = {
    "JUMPING": "Jumping / abnormal movement",
    "CRAWLING": "Crawling detected",
    "CROUCHING": "Crouching / suspicious movement",
    "FALLING": "Person fell",
    "RUNNING": "Running detected",
    # The operator's own phrase for this is "running while crouching". It is
    # what CAM-01's first person actually does and what the classifier
    # actually calls it, so the sentence says both.
    "BENT_RUN": "Running while crouched / low movement",
    "BENT_WALK": "Moving while crouched / low movement",
    "BENDING": "Bending detected",
}

#: Incident type -> the short sentence, for events that are not about a
#: behaviour. Mirrors database.models.IncidentType.LABELS in operator
#: language; the type itself still travels on every row so a client can tell
#: exactly which incident this was.
INCIDENT_HEADLINES = {
    "restricted_zone_intrusion": "Restricted zone intrusion",
    "border_zone_intrusion": "Border zone entry",
    "no_vehicle_zone_violation": "Vehicle in a no-vehicle zone",
    "line_crossing": "Boundary crossing",
    "loitering": "Loitering detected",
    "behavioural_anomaly": "Abnormal movement pattern",
    "watchlist_match": "Watchlist match",
    "suspicious_behaviour": "Suspicious behaviour",
}


def _headline(incident_type: str, behaviour: str | None) -> str:
    """One short line naming what happened.

    A behaviour incident is named by the behaviour, because "Suspicious
    behaviour" tells an operator nothing they can act on. Everything else is
    named by its type. Falls back to the raw type rather than to a
    comfortable-sounding default, so an incident type nobody has written a
    sentence for is visibly unnamed instead of silently mislabelled.
    """
    if incident_type == "suspicious_behaviour" and behaviour:
        upper = str(behaviour).upper()
        return BEHAVIOUR_HEADLINES.get(
            upper, f"{upper.replace('_', ' ').capitalize()} detected")
    return INCIDENT_HEADLINES.get(
        incident_type, str(incident_type).replace("_", " ").capitalize())


def _event_row(row: Any) -> dict[str, Any]:
    """One database row as the operator log wants to read it."""
    data = dict(row)

    detail = {}
    raw_detail = data.get("detail")
    if raw_detail:
        try:
            detail = json.loads(raw_detail)
        except (TypeError, ValueError):
            detail = {}

    reasons = []
    raw_reasons = data.get("reasons")
    if raw_reasons:
        try:
            reasons = [
                str(item.get("label", ""))
                for item in json.loads(raw_reasons)
                if isinstance(item, dict) and item.get("label")
            ]
        except (TypeError, ValueError):
            reasons = []

    behaviour = detail.get("behaviour")
    if behaviour == "UNKNOWN":
        # The classifier never concluded anything for this person. Say so,
        # rather than sending UNKNOWN for React to dress up as a behaviour.
        behaviour = None

    ts_text = str(data.get("ts_text") or "")
    # "2026-09-11 12:31:27" -> "12:31:27". The date is kept on the row for a
    # client that wants it; the timeline shows the time of day.
    time_only = ts_text.split(" ")[-1] if ts_text else ""

    return {
        "incident_id": data.get("incident_id"),
        "session_id": data.get("session_id"),
        "camera_id": data.get("camera_id"),
        "track_id": data.get("track_number"),
        "object_type": data.get("object_class"),
        "object_label": data.get("object_label"),
        "incident_type": data.get("incident_type"),
        "behaviour": behaviour,
        # The ThreatEngine's verdict, stored at the moment of the event.
        "threat_level": data.get("threat_level"),
        "threat_score": data.get("threat_score"),
        "headline": _headline(str(data.get("incident_type") or ""), behaviour),
        # The sentence the operator was actually shown when this fired.
        # None when the incident raised no alert.
        "message": data.get("message"),
        # Why it scored what it scored, in the threat model's own words.
        "reasons": reasons,
        "zone_name": data.get("zone_name"),
        "boundary_state": detail.get("boundary_state"),
        "status": data.get("status"),
        "escalation_count": data.get("escalation_count") or 0,
        "ts": data.get("ts"),
        "ts_text": ts_text,
        "time": time_only,
    }


@app.get("/api/events")
def events(
    camera_id: str | None = None,
    session_id: int | None = None,
    session_ids: str | None = None,
    limit: int = 200,
    _user: Principal = Depends(deps.require_user),
):
    """Recorded events, newest first.

    Filtering happens in SQL, not in the client, so CAM-01's events cannot
    be shown under CAM-02 by a mistake in the interface.

    * `camera_id` - one camera's events. Every row still carries its own
      camera_id, so nothing downstream has to remember what was asked for.
    * `session_id` - one run of one camera. This is what makes the event
      list agree with the concluded totals shown beside it: re-running a
      camera shows the new run's events, not every event it ever produced.
    * `session_ids` - a comma-separated set, for the consolidated ALL
      CAMERAS timeline: the current concluded run of every camera, mixed
      into one list. An empty value returns nothing, because "no camera has
      concluded a run" is a different answer from "here is the entire
      history of the database".
    * neither - everything recorded, for a client that wants the raw log.
    """
    parsed: list[int] | None = None
    if session_ids is not None:
        parsed = []
        for piece in session_ids.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                parsed.append(int(piece))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"session_ids must be integers, got {piece!r}",
                )

    rows = db.get_events(
        camera_id=camera_id,
        session_id=session_id,
        session_ids=parsed,
        limit=max(1, min(int(limit), 1000)),
    )
    return [_event_row(row) for row in rows]


@app.get("/api/cameras/{camera_id}/events")
def camera_events(
    camera_id: str,
    session_id: int | None = None,
    scope: str = "session",
    limit: int = 200,
    _user: Principal = Depends(deps.require_user),
):
    """One camera's recorded events.

    Alongside /api/cameras/{id}/stats and /api/cameras/{id}/tracks, so a
    client asking about one camera asks the same shape of question every
    time and the camera filter is part of the URL rather than a parameter
    that can be forgotten.

    `scope` decides WHICH run:

    * "session" (default) - the run this camera is on or was last on. This
      is what makes the event list agree with the concluded totals: a
      camera that has never run in this process returns nothing, rather
      than events from a session somebody ran days ago.
    * "all" - everything this camera has ever recorded.

    An explicit `session_id` overrides both.
    """
    runtime = runtimes.get(camera_id)
    if runtime is None:
        raise HTTPException(status_code=404,
                            detail=f"Unknown camera {camera_id}")

    if session_id is None and scope != "all":
        session_id = runtime.session_id
        if session_id is None:
            # Configured, never run in this process. No run means no events
            # from this run - and saying so is not the same as silently
            # widening the query to the camera's whole history.
            return {
                "camera_id": camera_id,
                "scope": scope,
                "session_id": None,
                "count": 0,
                "events": [],
            }

    rows = db.get_events(
        camera_id=camera_id,
        session_id=session_id,
        limit=max(1, min(int(limit), 1000)),
    )
    events_out = [_event_row(row) for row in rows]
    return {
        "camera_id": camera_id,
        "scope": scope,
        "session_id": session_id,
        "count": len(events_out),
        "events": events_out,
    }


# ============================================================
# PLATES
# ============================================================

@app.get("/api/plates")
def plates(_user: Principal = Depends(deps.require_user)):

    rows = db.get_plates(
        limit=100
    )

    return [
        dict(row)
        for row in rows
    ]

# ============================================================
# AUTHENTICATION ROUTES
# ============================================================
#
# Mounted last, after every surveillance endpoint above has been defined, so
# the ordering of this file reads the way the system is layered: the
# pipeline's API first, the access control around it afterwards.
#
# `/api/health` and `/` stay open deliberately. The login screen needs to say
# whether the backend is reachable BEFORE anybody has a token, and a health
# check that requires a session cannot answer the one question it exists for.
# Neither route exposes anything about the cameras, the footage or the users.

app.include_router(auth_router)
app.include_router(admin_router)


# ============================================================
# THE COMPILED DASHBOARD
# ============================================================
#
# Mounted after EVERY route above, and that ordering is load-bearing. Starlette
# matches routes in registration order, so a catch-all mounted at "/" put any
# earlier would swallow "/api/..." and serve a 404 page where JSON belongs.
# Declared last, it only ever sees requests that matched nothing else.
#
# html=True makes the mount serve index.html for a bare directory request.
#
# If the dashboard has not been built there is nothing to mount, and the JSON
# fallback on "/" above explains that instead.

if FRONTEND_DIST.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_DIST, html=True),
        name="dashboard",
    )
