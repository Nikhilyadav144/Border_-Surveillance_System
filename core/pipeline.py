"""The surveillance pipeline.

One class wires every engine together for one camera:

    frame
      -> detector.track()            YOLO + ByteTrack, real identities
      -> TrackManager                trajectory, speed, heading, dwell
      -> ZoneEngine                  polygon occupancy, intrusion events
      -> LineCrossingEngine          virtual boundary crossings
      -> BehaviourEngine             loitering, behavioural features
      -> AnomalyEngine               Isolation Forest over those features
      -> ThreatEngine                explainable score and level
      -> IncidentManager             dedupe, escalate, persist
      -> EvidenceManager             snapshot on disk
      -> AlertManager                notify the operator
      -> Database                    everything above, queryable

This module contains no Qt. It runs identically inside the GUI worker thread
and in a headless script, which is what makes the acceptance test in
``scripts/run_acceptance.py`` meaningful.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from core.alert_manager import AlertManager, DesktopChannel, LogChannel
from core.annotator import Annotator
from core.anomaly_engine import STATUS_ANOMALOUS, AnomalyEngine
from core.anpr import AnprEngine
from core.behaviour_engine import BehaviourEngine
from core.config import AppConfig
from core.detector import ObjectDetector
from core.evidence import EvidenceManager
from core.incident_manager import IncidentManager
from core.line_crossing import LineCrossingEngine
from core.low_light import LowLightDetector
from core.behaviour_classifier import BehaviourClassifier
from core.boundary_interaction import (
    CROSSED,
    BoundaryInteractionAnalyzer,
)
from core.movement import MovementAnalyzer
from core.pose import PoseEstimator, estimate_posture
from core.pose_history import (
    POSE_STALE,
    POSE_TRACKED,
    POSE_UNAVAILABLE,
    PoseHistory,
)
from core.threat_engine import ThreatContext, ThreatEngine
from core.tracker import TrackManager, TrackState
from core.watchlist import WatchlistMatcher, entries_from_rows
from core.zone_engine import ZoneEngine, ZoneKind
from database.models import (
    Detection,
    Incident,
    IncidentType,
    PlateRead,
    ThreatLevel,
    TrackRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Counters for the HUD, the camera health panel and the session record."""

    frames_read: int = 0
    frames_processed: int = 0
    detections: int = 0
    incidents: int = 0
    alerts: int = 0
    evidence_files: int = 0
    plates_read: int = 0
    started_ts: float = field(default_factory=time.time)
    last_frame_ts: float = 0.0
    _fps_window: list[float] = field(default_factory=list)

    def note_frame(self, ts: float) -> None:
        if self.last_frame_ts:
            delta = ts - self.last_frame_ts
            if delta > 0:
                self._fps_window.append(delta)
                if len(self._fps_window) > 30:
                    self._fps_window.pop(0)
        self.last_frame_ts = ts

    @property
    def fps(self) -> float:
        if not self._fps_window:
            return 0.0
        mean = sum(self._fps_window) / len(self._fps_window)
        return 1.0 / mean if mean > 0 else 0.0

    @property
    def elapsed(self) -> float:
        return max(0.0, time.time() - self.started_ts)


@dataclass
class FrameOutput:
    """What the pipeline produced for one frame."""

    frame_index: int
    ts: float
    annotated: Optional[np.ndarray]
    raw: Optional[np.ndarray]
    tracks: list[TrackState] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    incidents: list[Incident] = field(default_factory=list)
    persons: int = 0
    vehicles: int = 0
    inference_ms: float = 0.0
    skipped: bool = False


class SurveillancePipeline:
    """Own every engine for one camera and drive them frame by frame."""

    def __init__(
        self,
        config: AppConfig,
        camera: dict[str, Any],
        database: Any = None,
        detector: Optional[ObjectDetector] = None,
        session_id: Optional[int] = None,
        annotate: bool = True,
    ) -> None:
        self.config = config
        self.camera = camera
        self.camera_id = str(camera.get("id", "CAM-01"))
        self.db = database
        self.session_id = session_id
        self.annotate = annotate

        det_cfg = config.section("detection")
        self.vehicle_classes = {
            c.lower() for c in det_cfg.get("vehicle_classes", [])
        }
        self.person_classes = {
            c.lower() for c in det_cfg.get("person_classes", ["person"])
        }
        self.frame_stride = max(1, int(det_cfg.get("frame_stride", 1)))

        # -- detector ----------------------------------------------------
        self.detector = detector or ObjectDetector(
            model_path=config.model_path(),
            confidence=float(det_cfg.get("confidence", 0.35)),
            iou=float(det_cfg.get("iou", 0.5)),
            image_size=int(det_cfg.get("image_size", 640)),
            device=str(det_cfg.get("device", "auto")),
            class_filter=det_cfg.get("classes") or [],
            tracker=str(config.get("tracking.tracker", "bytetrack")),
            half=bool(det_cfg.get("half", False)),
        )

        # -- zones and lines ---------------------------------------------
        # Security geometry is per-camera and OPTIONAL, because a polygon
        # describes a physical region of one camera's view and means nothing
        # in another's. A camera that does not name a profile therefore gets
        # NO geometry, not a borrowed one.
        #
        # This default used to be "default", which quietly handed every
        # unconfigured camera the polygons drawn for the demo clip. Those
        # cover the middle of the frame, so an ordinary person standing in an
        # ordinary room scored a HIGH intrusion on a camera nobody had
        # configured. The AI runs on every camera; the geometry does not.
        profile_name = str(camera.get("zones_profile", "none"))
        profile = config.zone_profile(profile_name)
        self.profile_name = profile_name
        self.zone_engine = ZoneEngine(
            zones=profile["zones"],
            vehicle_classes=self.vehicle_classes,
        )
        self.line_engine = LineCrossingEngine(
            lines=profile["lines"],
            cooldown_seconds=float(
                config.get("lines.cooldown_seconds", 3.0)),
            min_travel=float(config.get("lines.min_travel", 0.004)),
            warmup_seconds=float(config.get("lines.warmup_seconds", 0.5)),
        )
        #: Seconds a track must exist before its position is used for zone
        #: or line decisions. Shares the line warm-up value: it is the same
        #: fact about the same reference point, so one number governs both.
        self._track_warmup_seconds = float(
            config.get("lines.warmup_seconds", 0.5))
        # Say out loud what this camera can and cannot raise. Silence here is
        # how a missing polygon turns into a mystery about missing alerts.
        if self.zone_engine.zones or self.line_engine.lines:
            logger.info(
                "%s geometry | profile '%s' | %d zone(s), %d line(s)",
                self.camera_id, profile_name,
                len(self.zone_engine.zones), len(self.line_engine.lines),
            )
        else:
            logger.info(
                "%s geometry | profile '%s' | 0 zones, 0 lines - no geometry "
                "configured, so this camera cannot raise zone-intrusion or "
                "line-crossing incidents. Detection, tracking, pose, "
                "behaviour, anomaly and threat scoring all still run.",
                self.camera_id, profile_name,
            )

        # -- tracking and behaviour --------------------------------------
        self.tracks = TrackManager(
            camera_id=self.camera_id,
            max_age_frames=int(config.get("tracking.max_age_frames", 30)),
            min_hits=int(config.get("tracking.min_hits", 3)),
            max_trajectory_points=int(
                config.get("tracking.max_trajectory_points", 240)),
            direction_change_degrees=float(
                config.get("behaviour.direction_change_degrees", 45.0)),
            coast_frames=int(config.get("tracking.coast_frames", 5)),
        )
        self.behaviour = BehaviourEngine(
            loitering_seconds=float(config.get("behaviour.loitering_seconds", 12)),
            loitering_radius=float(config.get("behaviour.loitering_radius", 0.06)),
            high_speed_threshold=float(
                (config.threat_rules.get("factors", {})
                 .get("high_speed", {}) or {}).get("threshold", 0.35)),
        )

        # -- anomaly ------------------------------------------------------
        anomaly_cfg = config.section("anomaly")
        model_file = config.path("models_dir") / str(
            anomaly_cfg.get("model_file", "behaviour_anomaly.joblib"))
        self.anomaly = AnomalyEngine(
            min_samples=int(anomaly_cfg.get("min_samples", 25)),
            contamination=float(anomaly_cfg.get("contamination", 0.10)),
            n_estimators=int(anomaly_cfg.get("n_estimators", 150)),
            random_state=int(anomaly_cfg.get("random_state", 42)),
            refit_every=int(anomaly_cfg.get("refit_every", 25)),
            decision_threshold=float(anomaly_cfg.get("decision_threshold", 0.0)),
            model_path=model_file,
            enabled=bool(anomaly_cfg.get("enabled", True)),
        )
        self.anomaly.load()
        self._warm_start_anomaly()

        # -- scene context -------------------------------------------------
        # Every camera builds its own detector from the same settings. The
        # answer differs per camera only because the pictures differ, which
        # is the whole point: capability is identical, scene is not.
        self.low_light = LowLightDetector.from_config(config)
        self._low_light_state = self.low_light.state

        # -- pose ----------------------------------------------------------
        # Built from the same config for every camera. Whether it runs on a
        # given frame depends on what the detector found, never on which
        # camera the frame came from.
        self.pose = PoseEstimator.from_config(config)

        # -- person activity analysis --------------------------------------
        # Four separate stages, in order, each doing one job:
        #   pose        -> where this person's joints are, this frame
        #   history     -> where they have been, keyed by Track ID
        #   movement    -> measurable signals derived from that history
        #   classifier  -> a named behaviour, debounced over time
        # Nothing here knows a camera id exists. What differs between cameras
        # is only what their lenses can see.
        self.activity_enabled = bool(config.get("activity.enabled", True))
        self.pose_history = PoseHistory(
            window_seconds=float(
                config.get("activity.history.window_seconds", 3.0)),
            max_samples=int(config.get("activity.history.max_samples", 90)),
        )
        self.movement = MovementAnalyzer(
            min_samples=int(config.get("activity.movement.min_samples", 4)),
            min_duration=float(
                config.get("activity.movement.min_duration", 0.25)),
            window_seconds=float(
                config.get("activity.movement.window_seconds", 1.0)),
            stationary_speed=float(
                config.get("activity.movement.stationary_speed", 0.15)),
            min_support=int(config.get("activity.movement.min_support", 6)),
        )
        behaviour_cfg = dict(
            config.section("activity").get("behaviour", {}) or {})

        # WHICH behaviours may raise an incident on their own, and how sure
        # the classifier must be before one does. These two are alerting
        # policy, not classifier geometry, so they are lifted out before the
        # rest is handed to BehaviourClassifier - which takes floats only and
        # would choke on a list.
        self.alerting_behaviours: frozenset[str] = frozenset(
            str(name).upper()
            for name in (behaviour_cfg.pop("alerting_behaviours", None) or [])
        )
        self.behaviour_alert_min_confidence = float(
            behaviour_cfg.pop("alert_min_confidence", 0.55))

        self.behaviour_classifier = BehaviourClassifier(
            **{k: float(v) for k, v in behaviour_cfg.items()}
        )
        # Person-to-boundary relationship. Handed whatever lines THIS camera
        # has configured; a camera with none produces no interactions, which
        # is the correct answer rather than a degraded one.
        self.boundary = BoundaryInteractionAnalyzer.from_config(config)

        #: Behaviour changes raised this frame, for the GUI and the API.
        self.behaviour_events: list = []
        #: Boundary state changes raised this frame.
        self.boundary_events: list = []

        # -- threat, incidents, alerts, evidence --------------------------
        self.threat = ThreatEngine(
            rules=config.threat_rules,
            night_start_hour=int(config.get("schedule.night_start_hour", 20)),
            night_end_hour=int(config.get("schedule.night_end_hour", 6)),
        )
        self.incidents = IncidentManager(
            db=database,
            dedupe_seconds=float(config.get("incidents.dedupe_seconds", 20)),
            escalate_existing=bool(config.get("incidents.escalate_existing", True)),
        )
        # Alerts are created here, not in the GUI, so a headless run produces
        # exactly the same records as an interactive one.
        self.desktop_channel = DesktopChannel()
        self.alerts = AlertManager(
            db=database,
            toast_min_level=str(config.get("alerts.toast_min_level", "HIGH")),
            feed_size=int(config.get("alerts.feed_size", 200)),
            channels=[LogChannel(), self.desktop_channel],
        )

        ev_cfg = config.section("evidence")
        self.evidence = EvidenceManager(
            evidence_dir=config.path("evidence_dir"),
            enabled=bool(ev_cfg.get("enabled", True)),
            min_threat_level=str(ev_cfg.get("min_threat_level", "MEDIUM")),
            save_full_frame=bool(ev_cfg.get("save_full_frame", True)),
            save_crop=bool(ev_cfg.get("save_crop", True)),
            crop_padding=float(ev_cfg.get("crop_padding", 0.25)),
            jpeg_quality=int(ev_cfg.get("jpeg_quality", 90)),
            cooldown_seconds=float(ev_cfg.get("cooldown_seconds", 10)),
            retention_days=int(ev_cfg.get("retention_days", 30)),
        )

        # -- ANPR and watchlist -------------------------------------------
        anpr_cfg = config.section("anpr")
        self.anpr = AnprEngine(
            enabled=bool(anpr_cfg.get("enabled", True)),
            ocr_backend=str(anpr_cfg.get("ocr_backend", "auto")),
            min_ocr_confidence=float(anpr_cfg.get("min_ocr_confidence", 0.45)),
            min_vehicle_height=int(anpr_cfg.get("min_vehicle_height", 90)),
            plate_regex=anpr_cfg.get("plate_regex"),
            accept_unvalidated=bool(anpr_cfg.get("accept_unvalidated", True)),
        )
        self.anpr_cooldown = float(anpr_cfg.get("cooldown_seconds", 3.0))
        self.anpr_max_attempts = int(anpr_cfg.get("max_attempts_per_track", 6))
        self.watchlist = WatchlistMatcher()
        self.refresh_watchlist()

        # -- rendering ----------------------------------------------------
        self.annotator = Annotator(
            overlays=config.get("gui.overlays", {}),
            vehicle_classes=self.vehicle_classes,
        )

        self.stats = PipelineStats()
        self._last_ts: float = time.time()
        self._detection_sample_every = 15   # persist 1 detection row in 15
        self._anomaly_check_every = 20      # score live tracks every N frames
        self._alert_tracks: set[int] = set()

        #: Installed by the worker so alerts reach the GUI.
        self.on_incident: Optional[Callable[[Incident], None]] = None
        self.incidents.on_incident = self._handle_new_incident

        logger.info(
            "Pipeline ready for %s | model=%s | device=%s | zones=%d | lines=%d",
            self.camera_id, self.detector.model_path.name,
            self.detector.device, len(self.zone_engine), len(self.line_engine),
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _warm_start_anomaly(self) -> None:
        """Seed the anomaly baseline from previous sessions in SQLite."""
        if self.db is None or not self.anomaly.enabled or self.anomaly.is_fitted:
            return
        try:
            rows = self.db.get_track_feature_rows(limit=2000)
        except Exception as exc:
            logger.debug("Could not read historical tracks: %s", exc)
            return
        if not rows:
            return
        added = self.anomaly.add_samples(
            BehaviourEngine.features_from_row(dict(row)) for row in rows
        )
        if added >= self.anomaly.min_samples:
            self.anomaly.fit()

    def refresh_watchlist(self) -> None:
        """Reload the watchlist after the operator edits it."""
        if self.db is None:
            return
        try:
            self.watchlist.refresh(
                entries_from_rows(self.db.get_watchlist(active_only=True))
            )
        except Exception as exc:
            logger.warning("Could not load watchlist: %s", exc)

    def persist_zones(self) -> None:
        """Record the active zone geometry alongside the session."""
        if self.db is None:
            return
        try:
            self.db.save_zones(
                self.camera_id, self.profile_name, self.zone_engine.as_dicts()
            )
        except Exception as exc:
            logger.warning("Could not persist zones: %s", exc)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray, frame_index: int,
                ts: float | None = None) -> FrameOutput:
        """Run one frame through the whole chain."""
        moment = ts if ts is not None else time.time()
        self._last_ts = moment
        # Cleared for EVERY frame, not just analysed ones. Clearing only on
        # the analysed path left the previous frame's events still sitting
        # here during a skipped frame, so anything polling per frame - the
        # API stats do - counted each behaviour change twice.
        self.behaviour_events = []
        self.boundary_events = []
        self.stats.frames_read += 1
        self.stats.note_frame(moment)

        # Frame skipping. The skipped frame is still returned so the operator
        # sees smooth video; it simply carries no new analysis.
        if frame_index % self.frame_stride != 0:
            return FrameOutput(
                frame_index=frame_index, ts=moment,
                annotated=self._render(frame, frame_index, moment)
                if self.annotate else None,
                # Confirmed tracks only, to match what an analysed frame
                # publishes. Handing over the raw dict here meant odd frames
                # showed tracks that even frames did not: unconfirmed
                # flickers, and tracks already missing but not yet retired.
                raw=frame, tracks=self.tracks.confirmed(),
                persons=self.tracks.count_class(self.person_classes),
                vehicles=self.tracks.count_class(self.vehicle_classes),
                skipped=True,
            )

        self.stats.frames_processed += 1
        height, width = frame.shape[:2]

        # -- scene context -------------------------------------------------
        # Measured once here and reused by every context built below, so one
        # frame cannot be scored as both lit and dark. Runs on analysed frames
        # only: a skipped frame carries the state forward unchanged, which is
        # right because the scene does not change in 40 milliseconds.
        self._low_light_state = self.low_light.update(frame)

        # -- detection + identity ----------------------------------------
        detections = self.detector.track(frame)
        self.stats.detections += len(detections)

        observations = []
        for det in detections:
            if det.track_number is None:
                continue
            box_norm = det.normalised(width, height)
            point = self.zone_engine.reference_point(box_norm, det.class_name)
            observations.append(
                (det.track_number, det.class_name, point, box_norm, det.confidence)
            )

        active, closed = self.tracks.update(observations, moment)

        # -- pose on confirmed people ------------------------------------
        # Placed here on purpose: after tracking, so every crop belongs to an
        # identity that already exists, and before analysis, so a track that
        # raises an incident this frame carries its keypoints with it.
        # Nothing below depends on pose, so pose being off, unavailable or
        # skipped changes nothing about zones, lines, behaviour, anomaly,
        # threat, incidents, evidence or alerts.
        # Counted in ANALYSED frames so pose.interval_frames stays
        # independent of detection.frame_stride.
        fresh_behaviour = self._update_pose(
            frame, active, self.stats.frames_processed, moment)

        # -- zone, line, behaviour, threat -------------------------------
        new_incidents: list[Incident] = []
        self._alert_tracks.clear()

        person_count = self.tracks.count_class(self.person_classes)
        vehicle_count = self.tracks.count_class(self.vehicle_classes)

        for track in active:
            if track.frames_seen < self.tracks.min_hits:
                continue
            # A track's reference point - the bottom-centre of its box - is
            # not trustworthy while the box is still converging. An object
            # entering the frame, or emerging from behind something, has a
            # box that GROWS, and that growth moves the reference point
            # without the object moving. Zone entries and line crossings
            # read from that point, so both are held back for the same short
            # window rather than each engine guessing separately.
            if track.age_seconds < self._track_warmup_seconds:
                continue
            new_incidents.extend(
                self._analyse_track(track, frame, moment, person_count)
            )

        # -- behaviour on its own ----------------------------------------
        # Runs AFTER _analyse_track so that a person who is both crouching
        # and inside a restricted zone has their zone incident raised first;
        # the incident manager's dedupe window then folds the behaviour into
        # it instead of producing a second card for the same moment.
        for track, event in fresh_behaviour:
            incident = self._raise_behaviour_incident(
                track, event, frame, moment, person_count)
            if incident is not None:
                new_incidents.append(incident)

        # -- anomaly on live tracks --------------------------------------
        if self.stats.frames_processed % self._anomaly_check_every == 0:
            for track in active:
                if track.frames_seen < max(self.tracks.min_hits, 10):
                    continue
                result = self.anomaly.score(self.behaviour.features(track, moment))
                track.anomaly_status = result.status
                track.anomaly_score = result.score
                if result.status == STATUS_ANOMALOUS and not track.loitering:
                    incident = self._raise_anomaly_incident(
                        track, result, frame, moment
                    )
                    if incident is not None:
                        new_incidents.append(incident)

        # -- retire finished tracks --------------------------------------
        for track in closed:
            self._close_track(track, moment)

        # -- ANPR ---------------------------------------------------------
        if self.anpr.available:
            for track in active:
                incident = self._try_anpr(track, frame, moment)
                if incident is not None:
                    new_incidents.append(incident)

        # -- sampled detection rows ---------------------------------------
        self._persist_detection_sample(detections, frame_index, moment,
                                       width, height)

        self.stats.incidents += len(new_incidents)

        annotated = (
            self._render(frame, frame_index, moment)
            if self.annotate else None
        )

        return FrameOutput(
            frame_index=frame_index,
            ts=moment,
            annotated=annotated,
            raw=frame,
            # Same definition as the skipped-frame path above: confirmed live
            # tracks. `active` is only what matched THIS frame, so publishing
            # it made a track blink out of the operator's table whenever the
            # detector missed one frame, and blink back the next.
            tracks=self.tracks.confirmed(),
            detections=detections,
            incidents=new_incidents,
            persons=person_count,
            vehicles=vehicle_count,
            inference_ms=self.detector.stats.last_inference_ms,
        )

    # ------------------------------------------------------------------
    # Per-track analysis
    # ------------------------------------------------------------------

    def _assess(self, ctx: ThreatContext,
                track: TrackState | None = None):
        """Score a context and remember the result against its track.

        The ThreatEngine is a pure function and stays that way. This only
        keeps the latest verdict per track so the operator overlay can show
        a level with the reasons behind it instead of a bare number.
        """
        assessment = self.threat.assess(ctx)
        if track is not None:
            track.threat_level = assessment.level
            track.threat_score = round(float(assessment.score), 1)
            track.threat_signals = [r.code for r in assessment.reasons]
            track.threat_reasons = [r.label for r in assessment.reasons]
        return assessment

    def _event_detail(self, track: TrackState,
                      extra: dict[str, Any]) -> dict[str, Any]:
        """The stored detail blob for one incident.

        Whatever the incident type already recorded, plus WHAT THE PERSON
        WAS DOING when it fired. Without this, a zone intrusion in the
        detection log could say where somebody was but never what they were
        doing there, while the Live Monitoring overlay beside it showed the
        behaviour plainly - the same event described two different ways
        depending on which page you looked at.

        The behaviour is read off the track at the moment of the incident,
        so it is the same reading the overlay drew on that person's box on
        that frame. Nothing is recomputed and nothing is invented: a person
        whose behaviour is still UNKNOWN is recorded as UNKNOWN.
        """
        detail = dict(extra)
        if track.class_name.lower() in self.person_classes:
            detail.setdefault("behaviour", track.behaviour)
            detail.setdefault("behaviour_confidence",
                              round(float(track.behaviour_confidence), 2))
            detail.setdefault("boundary_state", track.boundary_state)
        return detail

    def _person_context(self, track: TrackState) -> dict[str, Any]:
        """Person-specific threat inputs shared by every incident type.

        Kept in one place so that adding a factor is one edit rather than
        four identical ones, where the fourth always gets forgotten. Returns
        nothing for a vehicle: neither of these facts is meaningful for one.
        """
        if track.class_name.lower() not in self.person_classes:
            return {}
        behaviour = track.behaviour
        return {
            "boundary_state": track.boundary_state,
            "behaviour": None if behaviour == "UNKNOWN" else behaviour,
        }

    def _analyse_track(
        self,
        track: TrackState,
        frame: np.ndarray,
        now: float,
        person_count: int,
    ) -> list[Incident]:
        """Zone, line and loitering rules for one track."""
        raised: list[Incident] = []
        point = track.position
        box_norm = track.box

        # -- zone occupancy ------------------------------------------------
        zone_events = self.zone_engine.update(
            track_number=track.track_number,
            class_name=track.class_name,
            box_norm=box_norm,
            ts=now,
            confidence=track.avg_confidence,
        )
        for event in zone_events:
            if event.kind == "exit":
                was_alarming = event.zone.kind in ZoneKind.ALARMING
                track.enter_zone(None, now, was_alarming, event.dwell_seconds)
                continue

            track.enter_zone(event.zone.name, now)
            if not self.zone_engine.is_violation(event.zone, track.class_name):
                continue

            incident_type, ctx = self._zone_context(event.zone, track, now,
                                                    person_count)
            assessment = self._assess(ctx, track)
            outcome = self.incidents.submit(
                camera_id=self.camera_id,
                incident_type=incident_type,
                assessment=assessment,
                ts=now,
                track_number=track.track_number,
                track_db_id=track.db_id,
                object_class=track.class_name,
                zone_name=event.zone.name,
                detail=self._event_detail(track, {
                    "zone_kind": event.zone.kind,
                    "entry_point": list(event.point),
                    "confidence": round(track.avg_confidence, 3),
                    "features": self.behaviour.describe_features(track, now),
                }),
                session_id=self.session_id,
            )
            self._alert_tracks.add(track.track_number)
            if outcome.created:
                self._after_incident(outcome.incident, track, frame, now)
                raised.append(outcome.incident)

        # -- line crossing --------------------------------------------------
        # The engine below already does the segment-intersection test. The
        # boundary analyzer that follows does not repeat it; it decides
        # whether the crossing had the supporting evidence of a real approach
        # and interaction, or arrived out of nowhere as a jitter would.
        crossings = list(self.line_engine.update(
            track_number=track.track_number,
            class_name=track.class_name,
            point=point,
            ts=now,
            confidence=track.avg_confidence,
        ))
        crossed_this_frame = bool(crossings)
        crossed_line_names = [c.line.name for c in crossings]
        track.line_crossings += len(crossings)

        # -- boundary relationship -------------------------------------------
        # This runs BEFORE the crossing incident is scored, and the order is
        # load-bearing. The analyzer is what decides whether the intersection
        # above had an approach behind it and ended with the person on the far
        # side. Scoring the incident first would price every crossing as a
        # bare geometric event and never as a confirmed one.
        #
        # Persons only: a vehicle has no hands to reach with and no feet to
        # lift, so the interaction evidence this layer relies on is
        # meaningless for one.
        if (self.activity_enabled
                and track.class_name.lower() in self.person_classes):
            self._update_boundary(track, now, crossed_this_frame,
                                  crossed_line_names)

        for crossing in crossings:
            ctx = ThreatContext(
                # Scene darkness, measured from this frame.
                low_light=self._low_light_state.low_light,
                incident_type=IncidentType.LINE_CROSSING,
                class_name=track.class_name,
                crossed_line=True,
                in_restricted_zone=self._in_kind(track, ZoneKind.RESTRICTED),
                in_border_zone=self._in_kind(track, ZoneKind.BORDER),
                anomaly_detected=track.anomaly_status == STATUS_ANOMALOUS,
                anomaly_score=track.anomaly_score,
                group_size=person_count,
                speed=track.speed,
                zone_name=track.zone_name,
                ts=now,
                **self._person_context(track),
            )
            assessment = self._assess(ctx, track)
            outcome = self.incidents.submit(
                camera_id=self.camera_id,
                incident_type=IncidentType.LINE_CROSSING,
                assessment=assessment,
                ts=now,
                track_number=track.track_number,
                track_db_id=track.db_id,
                object_class=track.class_name,
                zone_name=crossing.line.name,
                detail=self._event_detail(track, crossing.to_dict()),
                session_id=self.session_id,
            )
            self._alert_tracks.add(track.track_number)
            if outcome.created:
                self._after_incident(outcome.incident, track, frame, now)
                raised.append(outcome.incident)

        # -- loitering -------------------------------------------------------
        loiter = self.behaviour.check_loitering(track, now)
        if loiter is not None:
            ctx = ThreatContext(
                # Scene darkness, measured from this frame.
                low_light=self._low_light_state.low_light,
                incident_type=IncidentType.LOITERING,
                class_name=track.class_name,
                loitering=True,
                loitering_seconds=loiter.duration_seconds,
                in_restricted_zone=self._in_kind(track, ZoneKind.RESTRICTED),
                in_border_zone=self._in_kind(track, ZoneKind.BORDER),
                anomaly_detected=track.anomaly_status == STATUS_ANOMALOUS,
                anomaly_score=track.anomaly_score,
                group_size=person_count,
                zone_name=track.zone_name,
                ts=now,
                **self._person_context(track),
            )
            assessment = self._assess(ctx, track)
            outcome = self.incidents.submit(
                camera_id=self.camera_id,
                incident_type=IncidentType.LOITERING,
                assessment=assessment,
                ts=now,
                track_number=track.track_number,
                track_db_id=track.db_id,
                object_class=track.class_name,
                zone_name=track.zone_name,
                detail=self._event_detail(track, loiter.to_dict()),
                session_id=self.session_id,
            )
            self._alert_tracks.add(track.track_number)
            if outcome.created:
                self._after_incident(outcome.incident, track, frame, now)
                raised.append(outcome.incident)

        return raised

    def _zone_context(
        self, zone: Any, track: TrackState, now: float, person_count: int
    ) -> tuple[str, ThreatContext]:
        """Map a zone entry to an incident type and a threat context."""
        is_vehicle = track.class_name.lower() in self.vehicle_classes

        if zone.kind == ZoneKind.RESTRICTED:
            incident_type = IncidentType.RESTRICTED_INTRUSION
        elif zone.kind == ZoneKind.BORDER:
            incident_type = IncidentType.BORDER_INTRUSION
        else:
            incident_type = IncidentType.NO_VEHICLE_VIOLATION

        ctx = ThreatContext(
            # Scene darkness, measured from this frame.
            low_light=self._low_light_state.low_light,
            incident_type=incident_type,
            class_name=track.class_name,
            zone_kind=zone.kind,
            zone_name=zone.name,
            in_restricted_zone=zone.kind == ZoneKind.RESTRICTED,
            in_border_zone=zone.kind == ZoneKind.BORDER,
            vehicle_in_no_vehicle_zone=(
                zone.kind == ZoneKind.NO_VEHICLE and is_vehicle),
            unauthorised_vehicle=(
                is_vehicle and zone.kind in ZoneKind.ALARMING),
            loitering=track.loitering,
            loitering_seconds=track.dwell_seconds(now),
            anomaly_detected=track.anomaly_status == STATUS_ANOMALOUS,
            anomaly_score=track.anomaly_score,
            group_size=person_count,
            speed=track.speed,
            ts=now,
            **self._person_context(track),
        )
        return incident_type, ctx

    def _in_kind(self, track: TrackState, kind: str) -> bool:
        if not track.zone_name:
            return False
        zone = self.zone_engine._zone_by_name(track.zone_name)
        return zone is not None and zone.kind == kind

    def _raise_behaviour_incident(
        self, track: TrackState, event: Any, frame: np.ndarray, now: float,
        person_count: int,
    ) -> Optional[Incident]:
        """Raise an incident for a behaviour that is an event by itself.

        Only for the labels named in `activity.behaviour.alerting_behaviours`.
        Everything else the classifier produces still reaches the threat
        engine through :meth:`_person_context` and still raises the score of
        whatever else is happening - it simply does not start an incident of
        its own.

        Three separate things keep this from becoming a per-frame siren:

        * the classifier's own debounce, which will not report a behaviour
          until it has held for `hold_seconds`;
        * this method being fed only the events that SETTLED this frame,
          never the accumulated list;
        * the incident manager's dedupe window, which folds a repeat of the
          same behaviour on the same track into the existing incident.

        One crouch is therefore one alert, however many frames it spans.
        """
        behaviour = str(getattr(event, "behavior", "") or "").upper()
        if not behaviour or behaviour not in self.alerting_behaviours:
            return None
        if track.class_name.lower() not in self.person_classes:
            return None
        confidence = float(getattr(event, "confidence", 0.0) or 0.0)
        if confidence < self.behaviour_alert_min_confidence:
            return None

        ctx = ThreatContext(
            # Scene darkness, measured from this frame.
            low_light=self._low_light_state.low_light,
            incident_type=IncidentType.SUSPICIOUS_BEHAVIOUR,
            class_name=track.class_name,
            in_restricted_zone=self._in_kind(track, ZoneKind.RESTRICTED),
            in_border_zone=self._in_kind(track, ZoneKind.BORDER),
            anomaly_detected=track.anomaly_status == STATUS_ANOMALOUS,
            anomaly_score=track.anomaly_score,
            group_size=person_count,
            speed=track.speed,
            zone_name=track.zone_name,
            ts=now,
            **{
                # The behaviour scored here is the one that just fired, not
                # whatever the track happens to be doing by the time the
                # context is built. They are normally the same; when a person
                # changes posture on the same frame they are not, and the
                # alert must describe the event it was raised for.
                **self._person_context(track),
                "behaviour": behaviour,
            },
        )
        assessment = self._assess(ctx, track)
        outcome = self.incidents.submit(
            camera_id=self.camera_id,
            incident_type=IncidentType.SUSPICIOUS_BEHAVIOUR,
            assessment=assessment,
            ts=now,
            track_number=track.track_number,
            track_db_id=track.db_id,
            object_class=track.class_name,
            zone_name=track.zone_name,
            detail=self._event_detail(track, {
                # The behaviour that FIRED, which is what this incident is
                # about. _event_detail fills in `behaviour` as the plain
                # label; this keeps the full event beside it - confidence,
                # duration and the geometry the classifier decided on.
                "behaviour": behaviour,
                "behaviour_event": event.to_dict(),
                "features": self.behaviour.describe_features(track, now),
            }),
            session_id=self.session_id,
        )
        if outcome.created:
            self._alert_tracks.add(track.track_number)
            self._after_incident(outcome.incident, track, frame, now)
            return outcome.incident
        return None

    def _raise_anomaly_incident(
        self, track: TrackState, result: Any, frame: np.ndarray, now: float
    ) -> Optional[Incident]:
        ctx = ThreatContext(
            # Scene darkness, measured from this frame.
            low_light=self._low_light_state.low_light,
            incident_type=IncidentType.ANOMALY,
            class_name=track.class_name,
            anomaly_detected=True,
            anomaly_score=result.score,
            in_restricted_zone=self._in_kind(track, ZoneKind.RESTRICTED),
            in_border_zone=self._in_kind(track, ZoneKind.BORDER),
            zone_name=track.zone_name,
            speed=track.speed,
            ts=now,
            **self._person_context(track),
        )
        assessment = self._assess(ctx, track)
        outcome = self.incidents.submit(
            camera_id=self.camera_id,
            incident_type=IncidentType.ANOMALY,
            assessment=assessment,
            ts=now,
            track_number=track.track_number,
            track_db_id=track.db_id,
            object_class=track.class_name,
            zone_name=track.zone_name,
            detail=self._event_detail(track, {
                "anomaly": result.to_dict(),
                "features": self.behaviour.describe_features(track, now),
            }),
            session_id=self.session_id,
        )
        if outcome.created:
            self._alert_tracks.add(track.track_number)
            self._after_incident(outcome.incident, track, frame, now)
            return outcome.incident
        return None

    # ------------------------------------------------------------------
    # ANPR
    # ------------------------------------------------------------------

    def _try_anpr(
        self, track: TrackState, frame: np.ndarray, now: float
    ) -> Optional[Incident]:
        """Attempt a plate read on a vehicle track, subject to cooldown."""
        if track.class_name.lower() not in self.vehicle_classes:
            return None
        if track.plate_text:
            return None
        if track.plate_attempts >= self.anpr_max_attempts:
            return None
        if now - track.plate_last_attempt_ts < self.anpr_cooldown:
            return None

        track.plate_last_attempt_ts = now
        track.plate_attempts += 1

        height, width = frame.shape[:2]
        box_norm = track.box
        box_px = (box_norm[0] * width, box_norm[1] * height,
                  box_norm[2] * width, box_norm[3] * height)

        result = self.anpr.read_plate(frame, box_px)
        crop = result.get("crop")
        if crop is None:
            return None

        evidence_file = self.evidence.capture_plate(
            crop, self.camera_id, track.track_number, now
        )
        evidence_path = str(evidence_file.path) if evidence_file else None
        if evidence_file:
            self.stats.evidence_files += 1

        text = result.get("text")
        match = self.watchlist.match(text) if text else None

        plate = PlateRead(
            camera_id=self.camera_id,
            track_number=track.track_number,
            text=text,
            ocr_confidence=float(result.get("confidence", 0.0)),
            backend=str(result.get("backend", "none")),
            ts=now,
            validated=bool(result.get("validated", False)),
            evidence_path=evidence_path,
            watchlist_hit=bool(match and match.exact),
        )
        if self.db is not None:
            try:
                self.db.insert_plate(plate)
                if evidence_file:
                    self.db.insert_evidence(
                        incident_id=None, camera_id=self.camera_id,
                        track_number=track.track_number, kind="plate",
                        file_path=str(evidence_file.path),
                        width=evidence_file.width, height=evidence_file.height,
                        size_bytes=evidence_file.size_bytes, ts=now,
                    )
            except Exception as exc:
                logger.error("Could not persist plate read: %s", exc)

        if not text:
            logger.debug("%s plate UNREADABLE (%s)", track.label,
                         result.get("reason", ""))
            return None

        track.plate_text = text
        self.stats.plates_read += 1
        logger.info("%s plate read: %s (confidence %.2f, %s)",
                    track.label, text, plate.ocr_confidence,
                    "validated" if plate.validated else "unvalidated")

        if match is None:
            return None
        if not match.exact:
            logger.warning(match.describe())
            return None

        if self.db is not None:
            try:
                self.db.record_watchlist_hit(match.entry.value)
            except Exception as exc:
                logger.error("Could not record watchlist hit: %s", exc)

        ctx = ThreatContext(
            # Scene darkness, measured from this frame.
            low_light=self._low_light_state.low_light,
            incident_type=IncidentType.WATCHLIST,
            class_name=track.class_name,
            watchlist_match=True,
            watchlist_value=text,
            in_restricted_zone=self._in_kind(track, ZoneKind.RESTRICTED),
            in_border_zone=self._in_kind(track, ZoneKind.BORDER),
            unauthorised_vehicle=True,
            zone_name=track.zone_name,
            ts=now,
        )
        assessment = self.threat.assess(ctx)
        outcome = self.incidents.submit(
            camera_id=self.camera_id,
            incident_type=IncidentType.WATCHLIST,
            assessment=assessment,
            ts=now,
            track_number=track.track_number,
            track_db_id=track.db_id,
            object_class=track.class_name,
            zone_name=track.zone_name,
            detail={
                "plate": text,
                "ocr_confidence": round(plate.ocr_confidence, 3),
                "validated": plate.validated,
                "watchlist_label": match.entry.label,
                "evidence": evidence_path,
            },
            session_id=self.session_id,
        )
        if outcome.created:
            self._alert_tracks.add(track.track_number)
            self._after_incident(outcome.incident, track, frame, now)
            return outcome.incident
        return None

    # ------------------------------------------------------------------
    # Incident side effects
    # ------------------------------------------------------------------

    def _handle_new_incident(self, incident: Incident) -> None:
        """Raise the alert for a newly opened incident, then notify listeners."""
        alert = self.alerts.raise_for_incident(incident)
        if alert is not None:
            self.stats.alerts += 1
        if self.on_incident is not None:
            try:
                self.on_incident(incident)
            except Exception as exc:  # pragma: no cover
                logger.error("Incident handler failed: %s", exc, exc_info=True)

    def _after_incident(
        self, incident: Incident, track: TrackState,
        frame: np.ndarray, now: float,
    ) -> None:
        """Capture evidence for an incident that has just been opened."""
        track_key = f"{self.camera_id}:{track.track_number}"
        if not self.evidence.should_capture(incident.threat_level, track_key, now):
            return

        height, width = frame.shape[:2]
        box_norm = track.box
        box_px = (box_norm[0] * width, box_norm[1] * height,
                  box_norm[2] * width, box_norm[3] * height)

        annotated = self._render(frame, self.stats.frames_processed, now)
        files = self.evidence.capture(
            frame=frame,
            camera_id=self.camera_id,
            incident_id=incident.db_id,
            track_number=track.track_number,
            box=box_px,
            annotated=annotated,
            ts=now,
        )
        self.evidence.note_capture(track_key, now)
        self.stats.evidence_files += len(files)

        if self.db is not None:
            for item in files:
                try:
                    self.db.insert_evidence(
                        incident_id=incident.db_id,
                        camera_id=self.camera_id,
                        track_number=track.track_number,
                        kind=item.kind,
                        file_path=str(item.path),
                        width=item.width, height=item.height,
                        size_bytes=item.size_bytes, ts=item.ts,
                    )
                except Exception as exc:
                    logger.error("Could not index evidence: %s", exc)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _close_track(self, track: TrackState, now: float) -> None:
        """Retire a track: feed the anomaly baseline and persist it."""
        features = self.behaviour.features(track, now)
        self.anomaly.add_sample(features)

        self.zone_engine.forget(track.track_number)
        self.line_engine.forget(track.track_number)
        self.incidents.forget_track(self.camera_id, track.track_number)

        if self.db is None:
            return
        try:
            record = self._to_record(track)
            track_id = self.db.upsert_track(record)
            self.db.close_track(track_id)
        except Exception as exc:
            logger.error("Could not persist track %s: %s", track.label, exc)

    def _to_record(self, track: TrackState) -> TrackRecord:
        return TrackRecord(
            camera_id=self.camera_id,
            track_number=track.track_number,
            class_name=track.class_name,
            first_ts=track.first_ts,
            last_ts=track.last_ts,
            frames_seen=track.frames_seen,
            avg_confidence=track.avg_confidence,
            max_confidence=track.max_confidence,
            distance_norm=track.distance_norm,
            avg_speed=track.avg_speed,
            max_speed=track.max_speed,
            direction_changes=track.direction_changes,
            dwell_seconds=track.dwell_seconds(track.last_ts),
            zone_transitions=track.zone_transitions,
            zones_visited=list(track.zones_visited),
            restricted_seconds=track.restricted_seconds,
            anomaly_score=track.anomaly_score,
            anomaly_status=track.anomaly_status,
            trajectory=track.trajectory_rows(),
            session_id=self.session_id,
            db_id=track.db_id,
        )

    def _persist_detection_sample(
        self, detections: list[Detection], frame_index: int,
        ts: float, width: int, height: int,
    ) -> None:
        """Store a sample of detections, not every one.

        Writing every box of every frame would put tens of thousands of rows
        into SQLite for a two minute clip and tell you nothing the tracks do
        not already say.
        """
        if self.db is None or not detections:
            return
        if frame_index % self._detection_sample_every != 0:
            return
        rows = []
        for det in detections:
            x1, y1, x2, y2 = det.normalised(width, height)
            rows.append({
                "session_id": self.session_id,
                "camera_id": self.camera_id,
                "track_id": None,
                "frame_index": frame_index,
                "ts": ts,
                "class_name": det.class_name,
                "confidence": det.confidence,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "zone_name": (self.zone_engine.current_zone(det.track_number)
                              if det.track_number is not None else None),
            })
        try:
            self.db.insert_detections(rows)
        except Exception as exc:
            logger.error("Could not persist detections: %s", exc)

    # ------------------------------------------------------------------
    # Pose
    # ------------------------------------------------------------------

    def _update_pose(
        self, frame: np.ndarray, active: list[TrackState],
        analysed_frame_count: int, moment: float,
    ) -> list[tuple[TrackState, Any]]:
        """Attach body keypoints to confirmed person tracks.

        Two gates before any inference happens. The interval gate asks whether
        this frame is due at all. The eligibility gate, inside the estimator,
        drops boxes too small to hold recoverable joints. Between them, a
        quiet scene costs nothing and a busy one costs a bounded amount.

        Failures here are swallowed deliberately. Pose is an enrichment: if it
        breaks, the operator should lose keypoints, not lose the camera.

        Returns the ``(track, BehaviourEvent)`` pairs the classifier settled
        on DURING THIS FRAME. ``self.behaviour_events`` accumulates across the
        run and cannot answer "what is new", which is the only question an
        alert may be raised from - reading the accumulated list per frame is
        how a single crouch turns into one alert per frame.
        """
        fresh: list[tuple[TrackState, Any]] = []
        if not self.pose.should_run(analysed_frame_count):
            return fresh

        people = [
            track for track in active
            if track.class_name.lower() in self.person_classes
            and track.frames_seen >= self.tracks.min_hits
        ]
        if not people:
            return fresh

        try:
            results = self.pose.estimate(frame, people, moment)
        except Exception as exc:
            logger.error("Pose estimation failed on %s: %s",
                         self.camera_id, exc)
            return fresh

        for track in people:
            result = results.get(track.track_number)
            if result is None:
                # Pose was attempted and produced nothing usable for this
                # person. Say so rather than leaving a stale reading looking
                # current, and never invent landmarks.
                if track.pose is None:
                    track.pose_status = POSE_UNAVAILABLE
                elif moment - track.pose_ts > 1.0:
                    track.pose_status = POSE_STALE
                continue
            track.pose = result
            track.posture = estimate_posture(result)
            track.pose_ts = moment
            track.pose_count += 1
            track.pose_status = POSE_TRACKED

            if not self.activity_enabled:
                continue

            # History -> signals -> behaviour, all keyed by the Track ID the
            # tracker already assigned.
            self.pose_history.record(track.track_number, result,
                                     track.box, moment)
            history = self.pose_history.get(track.track_number)
            if history is None:
                continue
            signals = self.movement.analyse(history)
            track.movement = signals
            track.movement_direction = signals.direction

            event = self.behaviour_classifier.update(signals)
            state = self.behaviour_classifier.state(track.track_number)
            if state is not None:
                track.behaviour = state.behavior
                track.behaviour_confidence = state.confidence
                track.behaviour_duration = state.duration
            if event is not None:
                self.behaviour_events.append(event)
                fresh.append((track, event))
                logger.info("%s %s", self.camera_id, event.describe())

        # Histories and behaviour states outlive a single frame, so they must
        # be released when the tracker retires an identity. Otherwise both
        # dictionaries grow for the life of the process.
        live = set(self.tracks.tracks)
        self.pose_history.prune(live)
        self.behaviour_classifier.prune(live)
        self.boundary.prune(live)
        return fresh

    def _update_boundary(self, track: TrackState, now: float,
                         crossed: bool,
                         crossed_lines: list[str] | None = None) -> None:
        """Update how this person relates to THIS camera's boundaries.

        Failures are swallowed: this is enrichment, and if it breaks the
        operator should lose a relationship reading, not lose the camera.
        """
        try:
            history = self.pose_history.get(track.track_number)
            signals = track.movement
            # If the movement layer does not trust this person's pose, the
            # boundary layer must not either. Everything it measures -
            # distance, hand extension, foot elevation - comes from the same
            # keypoints.
            #
            # Getting this wrong once was instructive: gating movement on box
            # size but still calling the boundary analyzer left speed at 0.0
            # while the untrustworthy foot_elevation of 0.47 stood, so a
            # person too small to analyse was reported as a STATIONARY
            # CLIMBER at the fence. A partial gate was worse than none.
            if signals is None or not signals.ready:
                return
            event = self.boundary.update(
                track, self.line_engine.lines, history, now,
                crossing_seen=crossed,
                speed=signals.speed,
                crossed_lines=crossed_lines,
            )
            state = self.boundary.state(track.track_number)
            if state is not None:
                track.boundary_state = state.state
                track.boundary_line = state.line_name
                track.boundary_distance = (
                    state.reading.distance if state.reading else None)
            if event is not None:
                self.boundary_events.append(event)
                logger.info("%s %s", self.camera_id, event.describe())
        except Exception as exc:
            logger.error("Boundary analysis failed on %s: %s",
                         self.camera_id, exc)

    # ------------------------------------------------------------------
    # Person activity, read by the API stats and the GUI
    # ------------------------------------------------------------------

    @property
    def activity_state(self) -> dict[str, Any]:
        """What the people on screen are currently doing.

        Counts come from the classifier's own debounced states, so what the
        dashboard shows is the same thing that would raise an event, never a
        separate reading that could disagree with it.
        """
        summary = self.behaviour_classifier.summary()
        posed = sum(1 for t in self.tracks.tracks.values()
                    if t.pose_status == POSE_TRACKED)
        if not self.activity_enabled:
            text = "Disabled in config"
        elif not summary:
            text = "No people tracked"
        else:
            text = ", ".join(f"{count} {name.lower()}"
                             for name, count in sorted(summary.items()))
        return {
            "enabled": self.activity_enabled,
            "summary": text,
            "behaviours": summary,
            "tracked_poses": posed,
            "histories": len(self.pose_history),
            "boundary": self.boundary.summary(),
        }

    def people(self) -> list[dict[str, Any]]:
        """One row per tracked person, for the operator's table.

        Only fields the backend actually has. A person whose box was too
        small for pose reports UNAVAILABLE and UNKNOWN rather than a
        plausible-looking default.
        """
        rows = []
        for track in self.tracks.confirmed():
            if track.class_name.lower() not in self.person_classes:
                continue
            rows.append({
                "track_id": track.track_number,
                "label": track.label,
                "position": [round(v, 4) for v in track.position],
                "behaviour": track.behaviour,
                "confidence": round(float(track.behaviour_confidence), 2),
                "duration": round(float(track.behaviour_duration), 2),
                "direction": track.movement_direction,
                "pose_status": track.pose_status,
                "keypoints": (track.pose.visible_count
                              if track.pose is not None else 0),
                "posture": (track.posture.label
                            if track.posture is not None else None),
                "boundary_state": track.boundary_state,
                "boundary_line": track.boundary_line,
                "boundary_distance": (
                    round(track.boundary_distance, 2)
                    if track.boundary_distance is not None else None),
                "anomaly_status": track.anomaly_status,
                "anomaly_score": (round(float(track.anomaly_score), 4)
                                  if track.anomaly_score is not None else None),
                "zone": track.zone_name,
                "crossings": int(track.line_crossings),
                "detection_confidence": round(float(track.avg_confidence), 2),
                "box": [round(v, 4) for v in track.box],
                "loitering": bool(track.loitering),
                # Threat is None until the engine has actually scored this
                # person. Reporting LOW for someone nobody assessed would be
                # inventing a verdict, and LOW is the one that looks harmless.
                "threat_level": track.threat_level,
                "threat_score": track.threat_score,
                "threat_signals": list(track.threat_signals),
                "threat_reasons": list(track.threat_reasons),
            })
        return rows

    def session_totals(self) -> dict[str, Any]:
        """What this camera CONCLUDED over the whole run.

        Session totals, not a live snapshot, and the distinction is the
        whole point. ``persons`` here is how many DIFFERENT people were
        confirmed at any moment during the run; the ``persons`` in the live
        stats is how many are in view right now. On a clip where two people
        walk through and leave, the live figure ends at 0 and this one ends
        at 2. Reporting the last frame's count as the result of the run
        would report the emptiness of the final frame.

        Every number comes from a component that already existed:

        * persons / vehicles - distinct identities that reached ``min_hits``
          in ``TrackManager``, so a person present for 500 frames counts
          once and a two-frame false positive counts not at all;
        * alerts / incidents - the pipeline's own counters, which are
          incremented only when ``IncidentManager`` actually creates an
          incident, so a threat that persists for 100 frames is one alert
          and not a hundred. That deduplication is not reimplemented here.
        """
        return {
            "persons": self.tracks.unique_seen(self.person_classes),
            "vehicles": self.tracks.unique_seen(self.vehicle_classes),
            "alerts": int(self.stats.alerts),
            "incidents": int(self.stats.incidents),
            "detections": int(self.stats.detections),
            "evidence": int(self.stats.evidence_files),
            "plates": int(self.stats.plates_read),
            "frames_read": int(self.stats.frames_read),
            "frames_processed": int(self.stats.frames_processed),
            "unique_by_class": self.tracks.unique_breakdown(),
        }

    # ------------------------------------------------------------------
    # Security geometry, read by the API stats and the GUI status panel
    # ------------------------------------------------------------------

    @property
    def geometry_state(self) -> dict[str, Any]:
        """What geometry this camera actually has, for display.

        Reported from the engines rather than from the config file, so what
        the dashboard shows is what the pipeline is really evaluating
        against. A camera with no geometry says so plainly instead of the
        interface guessing or staying silent.
        """
        zones = len(self.zone_engine.zones)
        lines = len(self.line_engine.lines)
        if zones or lines:
            summary = (f"{self.profile_name}: {zones} zone"
                       f"{'' if zones == 1 else 's'}, {lines} line"
                       f"{'' if lines == 1 else 's'}")
        else:
            summary = "No geometry configured"
        return {
            "profile": self.profile_name,
            "zones": zones,
            "lines": lines,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Scene context, read by the API stats and the GUI status panel
    # ------------------------------------------------------------------

    @property
    def low_light_state(self):
        """The current measured low-light verdict for this camera.

        Whatever reads this gets the same object the threat contexts were
        built from on the last analysed frame, so the number on screen and
        the number that scored the incident cannot disagree.
        """
        return self._low_light_state

    # ------------------------------------------------------------------
    # Rendering and shutdown
    # ------------------------------------------------------------------

    def _render(
        self, frame: np.ndarray, frame_index: int, ts: float,
        tracks: Optional[list[TrackState]] = None,
    ) -> np.ndarray:
        """Draw the overlay. ``tracks`` is accepted but deliberately unused.

        See the comment on ``live`` below: what gets drawn is decided here,
        from ``confirmed()``, and not by the caller.
        """
        # CONFIRMED tracks only, always. The overlay must show the same set
        # of objects the rest of the system believes in.
        #
        # It did not, and that was the whole of the "a pole is detected as a
        # person" bug. This function used to be handed `active` (every track
        # matched THIS frame, including ones below min_hits) or the raw
        # `self.tracks.tracks` dict (every live track, including ones that
        # stopped matching detections a second ago), while counts, incidents,
        # threat scoring, the API and the database all used `confirmed()`.
        #
        # Measured on CAM-01: one 0.35-confidence detection on a fence post,
        # seen TWICE and never again, was drawn as "PERSON #03" for 34
        # consecutive frames from frame 222 onward - `frames_seen=2` against
        # `min_hits=3`, `missed_frames=30`. The HUD beside it correctly read
        # PERSONS 1. Detection, tracking and confirmation were all behaving;
        # only the picture disagreed with them.
        #
        # Passing a wider list in is now ignored on purpose rather than
        # trusted, so no future call site can reintroduce the mismatch.
        live = self.tracks.confirmed()
        hud = {
            "camera_id": self.camera_id,
            "timestamp": time.strftime("%H:%M:%S", time.localtime(ts)),
            "frame_index": frame_index,
            "fps": self.stats.fps,
            "inference_ms": self.detector.stats.last_inference_ms,
            "persons": self.tracks.count_class(self.person_classes),
            "vehicles": self.tracks.count_class(self.vehicle_classes),
            "tracks": self.tracks.live_count,
            "incidents": self.stats.incidents,
            "threat_level": self._current_threat_level(),
        }
        return self.annotator.render(
            frame=frame,
            tracks=live,
            zones=self.zone_engine.zones,
            lines=self.line_engine.lines,
            hud=hud,
            alert_tracks=set(self._alert_tracks),
        )

    def _current_threat_level(self) -> str:
        if self.db is not None:
            try:
                return self.db.highest_open_threat()
            except Exception:
                pass
        return ThreatLevel.LOW

    def finalise(self) -> None:
        """Close out a session: retire tracks, save the anomaly model."""
        now = self._last_ts
        for track in self.tracks.close_all():
            self._close_track(track, now)
        self.anomaly.save()
        logger.info(
            "Pipeline finished for %s | frames %d/%d | detections %d | "
            "incidents %d | evidence %d | plates %d",
            self.camera_id, self.stats.frames_processed, self.stats.frames_read,
            self.stats.detections, self.stats.incidents,
            self.stats.evidence_files, self.stats.plates_read,
        )

    def describe(self) -> dict[str, Any]:
        """Everything the Settings and Dashboard screens want to display."""
        return {
            "camera_id": self.camera_id,
            "detector": self.detector.describe(),
            "zones": len(self.zone_engine),
            "lines": len(self.line_engine),
            "zone_profile": self.profile_name,
            "anomaly": self.anomaly.status_text(),
            "anpr": self.anpr.status_text(),
            "watchlist_entries": len(self.watchlist),
            "threat_rules": len(self.threat.factors),
        }
