from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import anyio
import cv2
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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

            # A restart must not inherit the last run's picture, or the
            # stream shows a stale frame from the previous session and a
            # client cannot tell it apart from a live one.
            with self.lock:
                self.latest_frame = None
                self.latest_frame_ts = 0.0

            self._set(status="STARTING", error=None, source_ended=False,
                      frame_index=0)

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

@app.get("/")
async def root():

    return {
        "name": "Border Guard AI",
        "status": "online",
        "version": "1.0.0",
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
async def get_cameras():

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
async def start_camera(camera_id: str):

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
async def stop_camera(camera_id: str):

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
async def camera_stats(camera_id: str):

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
async def camera_tracks(camera_id: str):
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
async def camera_stream(camera_id: str):

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
async def dashboard():

    active_cameras = 0
    persons = 0
    vehicles = 0
    tracks = 0
    detections = 0
    incidents = 0
    alerts = 0

    for runtime in runtimes.values():

        stats = runtime.snapshot()

        if stats["status"] == "ONLINE":
            active_cameras += 1

        persons += int(stats["persons"])
        vehicles += int(stats["vehicles"])
        tracks += int(stats["tracks"])
        detections += int(stats["detections"])
        incidents += int(stats["incidents"])
        alerts += int(stats["alerts"])

    database_counts = db.analytics_counts()

    return {
        "cameras_active": active_cameras,
        "persons": persons,
        "vehicles": vehicles,
        "tracks": tracks,
        "detections": detections,
        "incidents": incidents,
        "alerts": alerts,
        "database": database_counts,
    }


# ============================================================
# ALERTS
# ============================================================

@app.get("/api/alerts")
def alerts():

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
def incidents():

    rows = db.get_incidents(
        limit=100
    )

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# PLATES
# ============================================================

@app.get("/api/plates")
def plates():

    rows = db.get_plates(
        limit=100
    )

    return [
        dict(row)
        for row in rows
    ]