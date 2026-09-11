"""The processing worker thread.

Everything expensive happens here: decoding, YOLO inference, tracking, zone
maths, database writes. The GUI thread only ever receives finished results
through Qt signals, which is what keeps the interface responsive while a
video is being analysed.

Threading contract
------------------
* The worker owns the :class:`SurveillancePipeline` and the video source. The
  GUI must never touch either directly.
* Communication out is by signal only. Signals are queued across threads by
  Qt, so the GUI slots run on the GUI thread.
* Communication in is by :meth:`stop`, :meth:`pause` and :meth:`resume`, which
  set flags the run loop checks. No blocking calls into the worker.
* Frames are emitted at most ``display_fps`` times a second. Processing is not
  throttled: dropping a *display* frame costs nothing, dropping an *analysis*
  frame loses evidence.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QMutex, QThread, Signal

from core.config import AppConfig
from core.pipeline import SurveillancePipeline
from database.models import CameraHealth, CameraStatus
from inputs.base import VideoSource, VideoSourceError
from inputs.factory import build_source, describe_source

logger = logging.getLogger(__name__)


class PipelineWorker(QThread):
    """Run one camera's pipeline on a background thread."""

    # -- signals ---------------------------------------------------------
    #: Annotated frame plus a dict of live statistics.
    frame_ready = Signal(object, dict)
    #: A newly opened incident.
    incident_raised = Signal(object)
    #: An alert and the incident behind it.
    alert_raised = Signal(object, object)
    #: Camera health changed.
    health_changed = Signal(object)
    #: Human-readable status transitions for the status bar.
    status_changed = Signal(str, str)
    #: Something went wrong. The message is safe to show an operator.
    error_raised = Signal(str)
    #: The run ended, with a summary dict.
    session_finished = Signal(dict)
    #: Playback progress 0-100 for file sources, -1 for live ones.
    progress_changed = Signal(int)
    #: Emitted once the pipeline is built, carrying pipeline.describe().
    pipeline_ready = Signal(dict)

    def __init__(
        self,
        config: AppConfig,
        camera: dict[str, Any],
        database: Any,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.camera = dict(camera)
        self.camera_id = str(camera.get("id", "CAM-01"))
        self.db = database

        self._stop_requested = False
        self._paused = False
        self._mutex = QMutex()

        self.pipeline: Optional[SurveillancePipeline] = None
        self.source: Optional[VideoSource] = None
        self.session_id: Optional[int] = None

        self._display_interval = 1.0 / max(
            1.0, float(config.get("gui.display_fps", 25))
        )
        self._last_emit = 0.0
        # The live track table is expensive to rebuild and nobody reads it 25
        # times a second. Refreshing it a few times a second is plenty.
        self._table_interval = 0.25
        self._last_table = 0.0
        self._health = CameraHealth(camera_id=self.camera_id)

    # -- control ---------------------------------------------------------

    def stop(self) -> None:
        """Ask the worker to finish after the current frame."""
        self._mutex.lock()
        self._stop_requested = True
        self._paused = False
        self._mutex.unlock()

    def pause(self) -> None:
        self._mutex.lock()
        self._paused = True
        self._mutex.unlock()
        self.status_changed.emit(self.camera_id, "PAUSED")

    def resume(self) -> None:
        self._mutex.lock()
        self._paused = False
        self._mutex.unlock()
        self.status_changed.emit(self.camera_id, "ONLINE")

    @property
    def is_paused(self) -> bool:
        self._mutex.lock()
        value = self._paused
        self._mutex.unlock()
        return value

    def _should_stop(self) -> bool:
        self._mutex.lock()
        value = self._stop_requested
        self._mutex.unlock()
        return value

    # -- run loop --------------------------------------------------------

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            self._run()
        except Exception as exc:  # pragma: no cover - last line of defence
            logger.critical("Worker crashed: %s\n%s", exc, traceback.format_exc())
            self.error_raised.emit(
                f"The processing thread stopped unexpectedly: {exc}"
            )
            self._set_status(CameraStatus.ERROR, str(exc))

    def _run(self) -> None:
        self.status_changed.emit(self.camera_id, "CONNECTING")
        self._set_status(CameraStatus.CONNECTING)

        # -- open the source ----------------------------------------------
        try:
            self.source = build_source(
                self.camera, input_dir=self.config.path("input_dir")
            )
            self.source.open()
        except VideoSourceError as exc:
            logger.error("Cannot open %s: %s", self.camera_id, exc)
            self.error_raised.emit(str(exc))
            self._set_status(CameraStatus.ERROR, str(exc))
            self.session_finished.emit({"camera_id": self.camera_id,
                                        "error": str(exc)})
            return
        except Exception as exc:
            logger.error("Unexpected error opening %s: %s", self.camera_id, exc)
            self.error_raised.emit(f"Could not open the video source: {exc}")
            self._set_status(CameraStatus.ERROR, str(exc))
            self.session_finished.emit({"camera_id": self.camera_id,
                                        "error": str(exc)})
            return

        # -- open the session and build the pipeline -----------------------
        try:
            if self.db is not None:
                self.session_id = self.db.start_session(
                    camera_id=self.camera_id,
                    source_detail=describe_source(self.camera),
                    model_path=str(self.config.model_path()),
                    device=str(self.config.get("detection.device", "auto")),
                )
            self.pipeline = SurveillancePipeline(
                config=self.config,
                camera=self.camera,
                database=self.db,
                session_id=self.session_id,
            )
            self.pipeline.persist_zones()
            self.pipeline.desktop_channel.callback = self._on_alert
            self.pipeline_ready.emit(self.pipeline.describe())
        except Exception as exc:
            logger.error("Could not build the pipeline: %s", exc, exc_info=True)
            self.error_raised.emit(
                f"Could not start analysis: {exc}\n\n"
                f"Check the model path in Settings and that the weights file "
                f"exists."
            )
            self._set_status(CameraStatus.ERROR, str(exc))
            self.source.release()
            self.session_finished.emit({"camera_id": self.camera_id,
                                        "error": str(exc)})
            return

        self._set_status(CameraStatus.ONLINE)
        self.status_changed.emit(self.camera_id, "ONLINE")
        logger.info("Processing started for %s", self.camera_id)

        total_frames = self.source.info.frame_count
        last_health = 0.0
        consecutive_failures = 0

        # -- the loop -------------------------------------------------------
        while not self._should_stop():
            if self.is_paused:
                self.msleep(80)
                continue

            try:
                frame = self.source.read()
            except Exception as exc:
                logger.error("Frame read failed on %s: %s", self.camera_id, exc)
                consecutive_failures += 1
                if consecutive_failures > 10:
                    self.error_raised.emit(f"Video source failed: {exc}")
                    break
                self.msleep(120)
                continue

            if frame is None:
                logger.info("%s: source ended", self.camera_id)
                break

            consecutive_failures = 0

            try:
                output = self.pipeline.process(frame.image, frame.index, frame.ts)
            except Exception as exc:
                # One bad frame must not end the session.
                logger.error("Processing failed on frame %d: %s",
                             frame.index, exc, exc_info=True)
                self.error_raised.emit(
                    f"Frame {frame.index} could not be processed: {exc}"
                )
                continue

            for incident in output.incidents:
                self.incident_raised.emit(incident)

            now = time.perf_counter()
            if now - self._last_emit >= self._display_interval:
                self._last_emit = now
                image = output.annotated if output.annotated is not None \
                    else output.raw
                if image is not None:
                    with_tracks = (now - self._last_table) >= self._table_interval
                    if with_tracks:
                        self._last_table = now
                    self.frame_ready.emit(
                        image, self._snapshot(output, frame, with_tracks)
                    )

            # Hand the GIL back. Without this the interpreter can keep this
            # compute-heavy thread running long enough for the interface to
            # visibly stall between repaints, especially on a 2-core machine.
            self.msleep(1)

            if now - last_health >= 1.0:
                last_health = now
                self._publish_health(frame.wall_ts)
                if total_frames > 0:
                    self.progress_changed.emit(
                        int(100 * min(1.0, frame.index / total_frames))
                    )
                else:
                    self.progress_changed.emit(-1)

        self._shutdown()

    # -- helpers ---------------------------------------------------------

    def _snapshot(self, output: Any, frame: Any,
                  with_tracks: bool = True) -> dict:
        """The live statistics the GUI paints beside the video."""
        pipeline = self.pipeline
        assert pipeline is not None
        return {
            "camera_id": self.camera_id,
            "frame_index": output.frame_index,
            "ts": output.ts,
            "persons": output.persons,
            "vehicles": output.vehicles,
            "tracks": pipeline.tracks.live_count,
            "total_tracks": pipeline.tracks.total_created,
            "detections": pipeline.stats.detections,
            "incidents": pipeline.stats.incidents,
            "alerts": pipeline.stats.alerts,
            "evidence": pipeline.stats.evidence_files,
            "plates": pipeline.stats.plates_read,
            "inference_ms": pipeline.detector.stats.avg_inference_ms,
            "fps": pipeline.stats.fps,
            "frames_read": pipeline.stats.frames_read,
            "frames_processed": pipeline.stats.frames_processed,
            "anomaly": pipeline.anomaly.status_text(),
            "anpr": pipeline.anpr.status_text(),
            "zone_occupancy": pipeline.zone_engine.occupancy_counts(),
            # None means "no change": the GUI keeps what it already shows.
            "track_rows": ([t.summary() for t in output.tracks]
                           if with_tracks else None),
        }

    def _on_alert(self, alert: Any, incident: Any) -> None:
        """Called by the pipeline's desktop notification channel."""
        self.alert_raised.emit(alert, incident)

    def _publish_health(self, wall_ts: float) -> None:
        pipeline = self.pipeline
        source = self.source
        if pipeline is None or source is None:
            return

        self._health.status = (
            CameraStatus.ONLINE if not self.is_paused else CameraStatus.CONNECTING
        )
        self._health.fps = pipeline.stats.fps
        self._health.latency_ms = pipeline.detector.stats.avg_inference_ms
        self._health.frames_seen = source.frames_read
        self._health.last_frame_ts = wall_ts
        self._health.error_message = source.last_error

        self.health_changed.emit(self._health)
        if self.db is not None:
            try:
                self.db.update_camera_health(
                    camera_id=self.camera_id,
                    status=self._health.status,
                    fps=self._health.fps,
                    latency_ms=self._health.latency_ms,
                    frames_seen=self._health.frames_seen,
                    last_frame_ts=self._health.last_frame_ts,
                    error_message=self._health.error_message,
                )
            except Exception as exc:
                logger.debug("Health update failed: %s", exc)

    def _set_status(self, status: str, error: str | None = None) -> None:
        self._health.status = status
        self._health.error_message = error
        self.health_changed.emit(self._health)
        if self.db is not None:
            try:
                self.db.update_camera_health(
                    camera_id=self.camera_id, status=status,
                    fps=self._health.fps, latency_ms=self._health.latency_ms,
                    frames_seen=self._health.frames_seen,
                    last_frame_ts=self._health.last_frame_ts,
                    error_message=error,
                )
            except Exception:
                pass

    def _shutdown(self) -> None:
        """Close the source, finalise the pipeline, report the summary."""
        summary: dict[str, Any] = {"camera_id": self.camera_id}

        if self.source is not None:
            try:
                self.source.release()
            except Exception as exc:
                logger.debug("Source release failed: %s", exc)

        if self.pipeline is not None:
            try:
                self.pipeline.finalise()
                stats = self.pipeline.stats
                summary.update({
                    "frames_read": stats.frames_read,
                    "frames_processed": stats.frames_processed,
                    "detections": stats.detections,
                    "incidents": stats.incidents,
                    "alerts": stats.alerts,
                    "evidence": stats.evidence_files,
                    "plates": stats.plates_read,
                    "tracks": self.pipeline.tracks.total_created,
                    "elapsed": stats.elapsed,
                })
                if self.db is not None and self.session_id is not None:
                    self.db.end_session(
                        session_id=self.session_id,
                        frames_read=stats.frames_read,
                        frames_processed=stats.frames_processed,
                        detections=stats.detections,
                        incidents=stats.incidents,
                    )
            except Exception as exc:
                logger.error("Shutdown failed: %s", exc, exc_info=True)

        self._set_status(CameraStatus.OFFLINE)
        self.status_changed.emit(self.camera_id, "OFFLINE")
        self.progress_changed.emit(100)
        self.session_finished.emit(summary)
        logger.info("Processing stopped for %s", self.camera_id)
