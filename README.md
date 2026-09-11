# Border Guard AI

A local desktop application for AI-assisted border and perimeter surveillance.
It runs entirely on your computer, opens a proper native window, and needs no
browser and no cloud service.

Point it at a video file, a webcam, or (once configured) an RTSP camera. It
detects people and vehicles, follows each one with a stable identity, watches
configurable zones and virtual boundaries, measures how objects actually
behave over time, scores what it sees against an editable rule set, opens
incidents, captures evidence, and files everything in a local SQLite database
you can query yourself.

```
┌──────────────────────────────────────────────────────────────────────┐
│  BORDER GUARD AI            THREAT LEVEL HIGH   ● PROCESSING  13:50  │
├──────────────┬───────────────────────────────────────────────────────┤
│  Dashboard   │  ▶ Start   ⏸ Pause   ⏹ Stop      ● ONLINE             │
│  Cameras     │  ┌─────────────────────────────────────────────────┐  │
│  Incidents   │  │  CAM-01  13:50:40   FRAME 258   20 FPS   61 ms  │  │
│  Alerts      │  │  ┌──── Border Strip [BORDER] ───────────────┐   │  │
│  Watchlist   │  │  │                                          │   │  │
│  Analytics   │  │  │      ┌─ Restricted Compound ──┐          │   │  │
│  Evidence    │  │  │      │  [PERSON #01 0.78 LOITER]         │  │  │
│  Settings    │  │  │      └────────────────────────┘          │   │  │
│  System Logs │  │  └──────────────────────────────────────────┘   │  │
│              │  └─────────────────────────────────────────────────┘  │
│  2 alerts    │  FRAME 258 │ FPS 20 │ PERSONS 5 │ TRACKS 6 │ INC 5     │
└──────────────┴───────────────────────────────────────────────────────┘
```

---

## What this is, and what it is not

It **is** a working prototype that demonstrates how an AI-assisted
surveillance system can be built: real detection, real tracking, real
behavioural analysis, real persistence, and a real operator workflow.

It is **not** ready for deployment on an actual border. Before anything like
this could be trusted in the field it would need site trials on real footage,
a measured false-positive and false-negative rate, hardware validation, a
security audit, a legal and privacy review, defined human oversight, and
operational approval. None of that has been done.

Three rules the code follows without exception:

1. **Nothing is fabricated.** No hard-coded statistics, no placeholder
   detections, no invented plate numbers, no random threat scores. When the
   system does not know something it says so: an unreadable plate is
   `UNREADABLE`, and an anomaly engine that has not seen enough data reports
   `INSUFFICIENT_DATA` rather than guessing a number.
2. **Class names come from the loaded model.** Configure a class the weights
   do not support and it is reported at startup and dropped, never silently
   remapped onto something else.
3. **A missing model is an error.** The application will not quietly
   substitute different weights and carry on producing confident nonsense.

---

## Quick start

```bash
# 1. set up
python3.11 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. build the demo clip (optional, ~10 seconds)
python scripts/make_demo_video.py

# 3. run
python run.py
```

Then: **Cameras → CAM-01 → Start Processing**.

To check everything works without opening the window:

```bash
python scripts/run_acceptance.py --fresh
```

That runs the whole chain headless and prints a pass/fail line for each stage,
reading the results back out of SQLite.

---

## Where to put your videos

```
data/input/
```

Drop `.mp4`, `.avi`, `.mov` or `.mkv` files there. They appear in the file
picker behind **Open Video…** on the Cameras screen. To wire a file to a named
camera permanently, edit `config/cameras.yaml`.

The demo clip is a synthetic composite built for testing. Use your own footage
for anything meaningful. **[docs/DATA.md](docs/DATA.md)** explains where to get
suitable video and what scenarios to record.

---

## The nine screens

| Screen | What it does |
|---|---|
| **Dashboard** | Live counters, recent incidents, camera health, system status, trend charts |
| **Cameras** | The camera view: start and stop processing, live video with overlays, per-track table |
| **Incidents** | Filterable incident log with the full threat-score breakdown, evidence, acknowledge and resolve |
| **Alerts** | Newest-first notification feed with bulk acknowledge |
| **Watchlist** | Local list of plates of interest, plus every ANPR read including the failed ones |
| **Analytics** | Aggregates over the database: by day, by hour, by camera, by type, by level |
| **Evidence** | Filterable gallery of captured images with full-size viewing and metadata |
| **Settings** | Every operational parameter, written back to `config/` |
| **System Logs** | Application events mirrored from the rotating log file |

---

## How it works

```
     video file / webcam / RTSP
                │
        ┌───────▼────────┐
        │  VideoSource   │   one interface, three implementations
        └───────┬────────┘
                │  Frame(image, media timestamp)
        ┌───────▼────────┐
        │   Detector     │   YOLO + ByteTrack -> boxes with stable ids
        └───────┬────────┘
        ┌───────▼────────┐
        │  TrackManager  │   trajectory, speed, heading, dwell, turns
        └───────┬────────┘
     ┌──────────┼──────────┐
┌────▼────┐ ┌───▼────┐ ┌───▼──────┐
│  Zones  │ │ Lines  │ │Behaviour │  intrusion, crossing, loitering
└────┬────┘ └───┬────┘ └───┬──────┘
     └──────────┼──────────┘
        ┌───────▼────────┐
        │ Anomaly engine │   Isolation Forest over 8 behavioural features
        └───────┬────────┘
        ┌───────▼────────┐
        │ Threat engine  │   explainable weighted score -> LOW..CRITICAL
        └───────┬────────┘
        ┌───────▼────────┐
        │   Incidents    │   dedupe, escalate, lifecycle
        └───────┬────────┘
     ┌──────────┼──────────┐
┌────▼────┐ ┌───▼────┐ ┌───▼─────┐
│Evidence │ │ SQLite │ │ Alerts  │
└─────────┘ └────────┘ └─────────┘
```

All of the analysis runs on a worker thread, so the window never stops
responding. On the 2-core development machine, GUI event-loop lateness while
YOLO was running measured 170-280 ms on average across five runs (worst tick
340-540 ms) - noticeably less crisp than an idle application, but repainting
continuously throughout. `scripts/gui_live_test.py` measures this on demand.

**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** covers the design decisions.

---

## Configuration

Nothing operational is hard-coded. Four YAML files under `config/`:

| File | Contains |
|---|---|
| `application.yaml` | Detection, tracking, behaviour, anomaly, ANPR, evidence, alerts, logging, GUI |
| `cameras.yaml` | Camera definitions, including an RTSP template |
| `zones.yaml` | Zone polygons and virtual lines, in normalised 0-1 coordinates |
| `threat_rules.yaml` | Threat factor weights and level thresholds |

Edit them by hand, or use the Settings screen, which writes them back.

### Zones and lines

Coordinates are normalised (0.0 to 1.0), so a profile survives a change of
resolution. Zone kinds:

| Kind | Meaning |
|---|---|
| `RESTRICTED` | Any object inside is an intrusion |
| `BORDER` | The line-of-control strip; entry is an intrusion |
| `NO_VEHICLE` | Only vehicles inside are a violation |
| `PATROL` | Friendly area; presence logged, not scored |
| `OBSERVATION` | General monitoring; presence logged, not scored |

Zones are arbitrary polygons, not just rectangles, and the containment test
handles concave shapes correctly.

### Threat scoring

A transparent weighted sum. Every factor that fires is recorded with its label
and points, so an incident can always answer "why this number?":

```
INCIDENT #0042
Threat Score: 75/100        Threat Level: CRITICAL

Reasons:
  +  30  Restricted zone intrusion (Restricted Compound)
  +  20  Loitering (2m 05s)
  +  10  Extended dwell time (2m 05s)
  +  15  Behavioural anomaly (score -0.184)

Prototype risk model. Weights are configurable engineering defaults,
not a validated security standard. Always requires human review.
```

The weights in `config/threat_rules.yaml` are engineering defaults chosen so
the demo behaves sensibly. They are not derived from any security standard and
have not been validated against real incident data.

---

## Number plate recognition

ANPR is optional and off unless an OCR backend is installed:

```bash
pip install easyocr        # better on plates, ~100 MB of weights on first run
# or
pip install pytesseract && brew install tesseract     # lighter
```

The pipeline is: vehicle box → morphological plate-region proposal → upscale,
CLAHE, denoise, binarise → OCR → optional format check.

If OCR fails, returns nothing, or falls below the confidence threshold, the
result is `UNREADABLE` and the cropped image is still saved to the evidence
folder for a human to read. Plate text is never guessed.

Region proposal is classical computer vision, not a trained plate detector. It
works on reasonably sharp, roughly frontal plates and misses small, angled,
blurred or badly lit ones. Accuracy on real traffic footage has not been
measured for this build; measure it on your own footage before quoting a
figure.

---

## Project layout

```
AI-Border-Surveillance/
├── run.py                     launcher with dependency checks
├── app/
│   ├── main.py                start-up sequence
│   ├── gui/
│   │   ├── main_window.py     navigation shell, worker orchestration
│   │   ├── theme.py           the whole visual language
│   │   ├── pages/             the nine screens
│   │   └── widgets/           reusable components, video surface
│   └── workers/
│       └── pipeline_worker.py the processing thread
├── core/                      all the computer vision and logic, Qt-free
│   ├── detector.py            YOLO + tracker, device selection
│   ├── tracker.py             track state and kinematics
│   ├── zone_engine.py         polygon occupancy, intrusion events
│   ├── line_crossing.py       virtual boundary crossings
│   ├── behaviour_engine.py    loitering, behavioural features
│   ├── anomaly_engine.py      Isolation Forest
│   ├── threat_engine.py       explainable scoring
│   ├── anpr.py                plate detection and OCR
│   ├── incident_manager.py    dedupe, escalation, lifecycle
│   ├── alert_manager.py       notification dispatch
│   ├── evidence.py            event-driven capture and retention
│   ├── pipeline.py            the orchestrator
│   ├── annotator.py           overlay rendering
│   ├── watchlist.py           local matching
│   ├── geometry.py            polygon and vector maths
│   ├── config.py              loading and validation
│   ├── logging_setup.py       file, console and database sinks
│   └── paths.py               path resolution
├── inputs/                    video sources behind one interface
├── database/                  schema, records, data access
├── config/                    the four YAML files
├── data/{input,output,evidence,database}/
├── models/                    YOLO weights
├── logs/
├── scripts/                   demo clip, acceptance test, GUI tests
├── tests/                     185 tests
└── docs/                      ARCHITECTURE, DATA, DEMO
```

---

## Commands

```bash
python run.py                              # start the application
python scripts/make_demo_video.py          # build the demo clip
python scripts/run_acceptance.py --fresh   # headless end-to-end check
python scripts/gui_smoke_test.py           # render every screen to PNG
python scripts/gui_live_test.py            # run the worker under the GUI

pytest                                     # all 185 tests
pytest -m "not slow"                       # skip the detector tests (3 s)
pytest tests/test_geometry_zones_lines.py  # one file
```

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No module named 'PySide6'` | `pip install -r requirements.txt` |
| "Model weights not found" | Put a `.pt` file in `models/`, set it in Settings. The application refuses to substitute a different model. |
| "No OCR backend installed" | Expected. ANPR is optional; see above. |
| Video will not open | Codec your OpenCV build cannot decode. Re-encode to H.264 MP4. |
| Playback stutters | Raise `detection.frame_stride`, or lower `detection.image_size`, in Settings. |
| Webcam fails on macOS | Grant Camera access in System Settings → Privacy & Security, then restart. |
| `libGL.so.1` error on Linux | `sudo apt install libgl1 libglib2.0-0` |
| Zones look wrong on your footage | The defaults assume a border along the top of the frame. Edit `config/zones.yaml` or switch the camera to the `simple_centre` profile. |
| Everything reports "learning baseline" | Correct. The anomaly engine needs 25 completed tracks before it will score anything. |

---

## Licence and responsible use

This is educational and demonstration software. Surveillance systems have real
consequences for real people. Before deploying anything derived from this,
check what the law where you are requires about recording in public spaces,
data retention, and automated decision-making, and keep a human in the loop
for every consequential decision.
