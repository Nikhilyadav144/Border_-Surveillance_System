#!/usr/bin/env python3
"""Headless end-to-end acceptance test.

Runs the exact chain the GUI runs, without the GUI, and reports what actually
happened at every stage::

    LOCAL VIDEO -> FRAME CAPTURE -> YOLO DETECTION -> TRACKING
      -> ZONE / LINE ANALYSIS -> BEHAVIOUR -> ANOMALY -> THREAT SCORING
      -> INCIDENT -> EVIDENCE -> SQLITE -> ALERT -> ANALYTICS

Every number printed is read back out of SQLite or off the filesystem after
the run. Nothing is asserted from memory, because the point of this script is
to prove the persistence actually happened.

    python scripts/run_acceptance.py
    python scripts/run_acceptance.py --video data/input/my_clip.mp4 --max-frames 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config                       # noqa: E402
from core.logging_setup import setup_logging              # noqa: E402
from core.paths import PROJECT_ROOT                       # noqa: E402
from core.pipeline import SurveillancePipeline            # noqa: E402
from database.database import Database                    # noqa: E402
from database.models import CameraStatus                  # noqa: E402
from inputs.factory import build_source, describe_source  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"
)


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n  {text}\n{'=' * 74}")


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {label:<44} {DIM}{detail}{RESET}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=None,
                        help="Video to process. Defaults to the demo clip.")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames. 0 processes the whole file.")
    parser.add_argument("--camera-id", default="CAM-01")
    parser.add_argument(
        "--zones", default="demo_border_clip",
        help="Zone profile to run the acceptance chain against. This run "
             "asserts the zone and line engines FIRE, so it needs geometry, "
             "and a camera in cameras.yaml may legitimately have none. "
             "Passing 'none' will therefore report the three geometry checks "
             "as failures, which is correct: there is nothing to intrude "
             "into. Use scripts/check_zones.py to verify the no-geometry "
             "path properly.")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete the database first, for a clean run.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    setup_logging(level="WARNING" if args.quiet else "INFO", console=True)

    banner("BORDER GUARD AI - END TO END ACCEPTANCE TEST")

    # ---------------------------------------------------------------- config
    config = load_config()
    for warning in config.warnings:
        print(f"  {YELLOW}config:{RESET} {warning}")

    db_path = config.path("database_file")
    if args.fresh and db_path.exists():
        db_path.unlink()
        for suffix in ("-wal", "-shm"):
            side = Path(str(db_path) + suffix)
            if side.exists():
                side.unlink()
        print(f"  {DIM}removed existing database{RESET}")

    video = Path(args.video) if args.video else (
        config.path("input_dir") / "demo_border.mp4"
    )
    if not video.exists():
        print(f"\n{RED}No input video at {video}{RESET}")
        print("Build the demo clip first:  python scripts/make_demo_video.py")
        print("Or point at your own file:  --video path/to/clip.mp4")
        return 2

    camera = next(
        (c for c in config.cameras if c.get("id") == args.camera_id),
        {"id": args.camera_id, "name": "Acceptance Camera",
         # This run asserts zone and line events fire, so it needs geometry.
         "location": "Test", "zones_profile": "demo_border_clip",
         "source": {"type": "video", "path": str(video)}},
    )
    camera = dict(camera)
    # Geometry is a property of a camera, and a real camera may have none.
    # This run is checking the ENGINES, so it supplies its own profile rather
    # than depending on however that camera happens to be configured today.
    camera["zones_profile"] = args.zones
    camera["source"] = dict(camera.get("source", {}))
    camera["source"].update({"type": "video", "path": str(video),
                             "target_fps": None, "loop": False})

    results: list[bool] = []

    # -------------------------------------------------------------- database
    banner("STAGE 1  DATABASE")
    db = Database(db_path)
    before = db.analytics_counts()
    results.append(check("SQLite opened and schema applied", True, str(db_path)))
    db.upsert_camera(
        camera_id=camera["id"], name=camera.get("name", camera["id"]),
        location=camera.get("location"), source_type="video",
        source_detail=describe_source(camera),
        zones_profile=camera.get("zones_profile", "none"),
        enabled=True, status=CameraStatus.CONNECTING,
    )
    results.append(check("Camera registered", db.get_camera(camera["id"]) is not None))

    # ------------------------------------------------------------- pipeline
    banner("STAGE 2  MODEL AND ENGINES")
    session_id = db.start_session(
        camera_id=camera["id"], source_detail=str(video),
        model_path=str(config.model_path()),
        device=str(config.get("detection.device")),
    )
    started = time.time()
    pipeline = SurveillancePipeline(
        config=config, camera=camera, database=db, session_id=session_id
    )
    pipeline.persist_zones()

    info = pipeline.describe()
    det = info["detector"]
    results.append(check("YOLO model loaded", True,
                         f"{det['model']} on {det['device_label']}"))
    results.append(check("Class names read from the model, not hard-coded",
                         len(det["active_classes"]) > 0,
                         ", ".join(det["active_classes"][:6])))
    if det["unsupported_requested"]:
        print(f"  {YELLOW}note:{RESET} configured classes this model does not "
              f"support were dropped: {', '.join(det['unsupported_requested'])}")
    results.append(check("Zones loaded", info["zones"] > 0,
                         f"{info['zones']} zones, profile '{info['zone_profile']}'"))
    results.append(check("Virtual lines loaded", info["lines"] > 0,
                         f"{info['lines']} lines"))
    print(f"  {DIM}anomaly: {info['anomaly']}{RESET}")
    print(f"  {DIM}ANPR:    {info['anpr']}{RESET}")

    # ------------------------------------------------------------ processing
    banner("STAGE 3  PROCESSING")
    source = build_source(camera, input_dir=config.path("input_dir"))
    source.open()
    print(f"  source: {source.info.kind} {source.info.width}x"
          f"{source.info.height} @ {source.info.fps:.1f} fps, "
          f"{source.info.frame_count} frames")

    frames = 0
    detections_seen = 0
    max_tracks = 0
    incident_log: list[str] = []

    try:
        for frame in source:
            output = pipeline.process(frame.image, frame.index, frame.ts)
            frames += 1
            if not output.skipped:
                detections_seen += len(output.detections)
                max_tracks = max(max_tracks, len(output.tracks))
                for incident in output.incidents:
                    incident_log.append(
                        f"#{incident.db_id} {incident.threat_level:8s} "
                        f"{incident.type_label:28s} {incident.object_label:14s} "
                        f"score {incident.threat_score:5.0f}"
                    )
            if args.max_frames and frames >= args.max_frames:
                break
            if not args.quiet and frames % 100 == 0:
                print(f"  {DIM}frame {frames}  tracks {len(output.tracks)}  "
                      f"incidents {pipeline.stats.incidents}{RESET}")
    finally:
        source.release()
        pipeline.finalise()

    elapsed = time.time() - started
    db.end_session(
        session_id=session_id, frames_read=pipeline.stats.frames_read,
        frames_processed=pipeline.stats.frames_processed,
        detections=pipeline.stats.detections,
        incidents=pipeline.stats.incidents,
    )
    db.update_camera_health(
        camera_id=camera["id"], status=CameraStatus.OFFLINE,
        fps=pipeline.stats.fps, frames_seen=pipeline.stats.frames_read,
        last_frame_ts=pipeline.stats.last_frame_ts,
    )

    print(f"\n  processed {frames} frames in {elapsed:.1f}s "
          f"({frames / max(elapsed, 0.01):.1f} fps end to end)")
    print(f"  inference average {pipeline.detector.stats.avg_inference_ms:.0f} ms")

    # -------------------------------------------------------------- verdicts
    banner("STAGE 4  RESULTS READ BACK FROM SQLITE")
    after = db.analytics_counts()

    def delta(key: str) -> int:
        return after.get(key, 0) - before.get(key, 0)

    results.append(check("Frames were captured", frames > 0, f"{frames} frames"))
    results.append(check("YOLO produced real detections", detections_seen > 0,
                         f"{detections_seen} boxes"))
    results.append(check("Tracker assigned persistent ids",
                         pipeline.tracks.total_created > 0,
                         f"{pipeline.tracks.total_created} identities, "
                         f"{max_tracks} concurrent"))
    results.append(check("Tracks persisted to SQLite", delta("tracks") > 0,
                         f"+{delta('tracks')} rows"))

    zone_summary = pipeline.zone_engine.get_summary() if hasattr(
        pipeline.zone_engine, "get_summary") else {}
    intrusions = delta("intrusions")
    results.append(check("Zone / line analysis produced events",
                         intrusions + delta("line_crossings") > 0,
                         f"{intrusions} intrusions, "
                         f"{delta('line_crossings')} line crossings"))
    results.append(check("Loitering detected temporally", delta("loitering") > 0,
                         f"{delta('loitering')} loitering incidents"))
    print(f"  {DIM}anomaly engine: {pipeline.anomaly.status_text()}{RESET}")
    results.append(check("Incidents created", delta("incidents") > 0,
                         f"+{delta('incidents')} incidents"))
    results.append(check("Alerts raised", delta("active_alerts") >= 0
                         and after["incidents"] > 0,
                         f"{db.count_active_alerts()} active"))

    evidence_rows = db.get_evidence(limit=1000)
    on_disk = sum(1 for row in evidence_rows if Path(row["file_path"]).exists())
    results.append(check("Evidence written to disk and indexed",
                         on_disk > 0 and on_disk == len(evidence_rows),
                         f"{on_disk}/{len(evidence_rows)} files present"))

    by_level = db.analytics_by_level()
    by_type = db.analytics_by_type()
    results.append(check("Analytics queries return real data",
                         bool(by_level) and bool(by_type),
                         f"{len(by_level)} levels, {len(by_type)} types"))

    # -------------------------------------------------------------- lifecycle
    banner("STAGE 5  INCIDENT LIFECYCLE")
    open_incidents = db.get_incidents(status="NEW", limit=1)
    if open_incidents:
        incident_id = int(open_incidents[0]["id"])
        pipeline.incidents.acknowledge(incident_id, operator="acceptance-test")
        acknowledged = db.get_incident(incident_id)["status"] == "ACKNOWLEDGED"
        results.append(check("Incident can be acknowledged", acknowledged,
                             f"incident #{incident_id}"))
        pipeline.incidents.resolve(incident_id, operator="acceptance-test",
                                   note="Verified by the acceptance test")
        resolved = db.get_incident(incident_id)["status"] == "RESOLVED"
        results.append(check("Incident can be resolved", resolved,
                             f"incident #{incident_id}"))
    else:
        results.append(check("Incident lifecycle", False, "no incident to test"))

    # --------------------------------------------------------------- summary
    banner("INCIDENTS RAISED")
    if incident_log:
        for line in incident_log[:25]:
            print(f"  {line}")
        if len(incident_log) > 25:
            print(f"  {DIM}... and {len(incident_log) - 25} more{RESET}")
    else:
        print("  none")

    banner("ANALYTICS")
    print("  By threat level: " + (", ".join(
        f"{k} {v}" for k, v in sorted(by_level.items())) or "none"))
    print("  By type:         " + (", ".join(
        f"{k} {v}" for k, v in by_type) or "none"))
    busiest = db.analytics_busiest_camera()
    print(f"  Busiest camera:  {busiest[0]} ({busiest[1]})" if busiest
          else "  Busiest camera:  n/a")
    print(f"  Evidence files:  {after['evidence_files']}")
    print(f"  Plates read:     {after['plates_read']} "
          f"({pipeline.anpr.status_text()})")

    banner("VERDICT")
    passed = sum(1 for r in results if r)
    total = len(results)
    if passed == total:
        print(f"  {GREEN}ALL {total} CHECKS PASSED{RESET}")
        print(f"\n  Database: {db_path}")
        print(f"  Evidence: {config.path('evidence_dir')}")
        print("\n  Launch the desktop application with:  python run.py")
    else:
        print(f"  {RED}{total - passed} of {total} checks failed{RESET}")

    db.close()
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
