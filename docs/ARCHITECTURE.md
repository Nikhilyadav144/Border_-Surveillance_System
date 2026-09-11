# Architecture

Why the system is put together the way it is. For how to run it, see the
README; for how to demonstrate it, see [DEMO.md](DEMO.md).

---

## 1. The four layers

```
┌───────────────────────────────────────────────────────────────┐
│  app/          Qt. Windows, screens, widgets, the worker.     │
│                Knows about the database and the config.       │
│                Knows nothing about OpenCV or YOLO.            │
├───────────────────────────────────────────────────────────────┤
│  core/         All the computer vision and all the decisions. │
│                Imports no Qt, ever. Testable headless.        │
├───────────────────────────────────────────────────────────────┤
│  inputs/       Video acquisition behind one interface.        │
│                Knows nothing about what happens to a frame.   │
├───────────────────────────────────────────────────────────────┤
│  database/     Schema, typed records, hand-written SQL.       │
│                Imports nothing from core or app.              │
└───────────────────────────────────────────────────────────────┘
```

The rule that keeps this honest: **`core/` must never import from `app/`.** It
is what lets `scripts/run_acceptance.py` run the entire pipeline headlessly,
and what makes the 185 tests possible without a display.

`core/` does import `database/models`, but only for the dataclasses. The
persistence layer is passed in, so every engine works with `database=None`.

---

## 2. Video input

```python
class VideoSource(abc.ABC):
    def open(self) -> None: ...
    def _grab(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...
    def _on_read_failure(self) -> bool: ...   # retry, or stop?
```

Three implementations: `FileVideoSource`, `WebcamSource`, `RtspSource`. Retry
policy, throttling, frame counting and health accounting live in the base
class, so all three behave the same way.

`_on_read_failure` is where the types genuinely differ. A file that stops
producing frames has ended. A webcam that stops producing frames dropped one,
and will produce another. An RTSP stream that stops needs reconnecting. One
method, three answers.

### Media time, not wall time

`Frame.ts` is the **analysis clock**, and which clock it is depends on the
source:

* live camera: wall time, because the camera and the wall agree
* video file: media time, derived from frame index and file frame rate

This matters more than it looks. Dwell, speed and loitering are all measured
against this clock. Using wall time on a file would mean a clip analysed at
half speed reports everyone loitering twice as long as they really did, and a
clip analysed faster than real time would never detect loitering at all. The
first version of this application had exactly that bug: the demo clip's
12-second pause was measured as 30 seconds because the CPU was slower than
real time.

`Frame.wall_ts` keeps the real arrival time, for FPS and latency accounting.

### RTSP without an RTSP camera

`config/cameras.yaml` ships a fully-formed RTSP entry with `enabled: false`.
It appears in the interface as CONFIGURED and is never dialled. Set the flag
and fill in the URL and nothing downstream changes.

`RtspSource` adds four things a bare `cv2.VideoCapture(url)` does not: a
bounded open timeout so a wrong address fails in seconds instead of blocking
the worker; forced TCP transport, because UDP RTSP smears frames on anything
but a quiet LAN; a one-frame buffer, so the operator sees the present rather
than a growing backlog; and automatic reconnection. Credentials are redacted
everywhere a URL is logged or displayed.

---

## 3. Detection and identity

`ObjectDetector` wraps Ultralytics. Two rules it enforces:

**Class names come from the model.** `model.names` is read at load time. A
class named in the config that the model does not provide is reported and
dropped, and listed in `unsupported_classes` so the interface can say so. The
previous generation of this project mapped class ids through a fixed 7-entry
table; load different weights and a COCO `car` (class 2) silently became a
`crowd` with threat level HIGH.

**A missing model is an error.** No silent substitution. Falling back to
different weights without saying so produces confident nonsense that looks
exactly like a working system.

Tracking uses Ultralytics' built-in ByteTrack via `model.track(persist=True)`.
Writing a bespoke IoU tracker would be worse: ByteTrack handles low-confidence
recovery and short occlusions, and it is already in the dependency tree.

### Device selection

`auto` prefers CUDA, then Apple Silicon MPS, then CPU. CUDA is never required.
An explicit preference that is not available falls back with an explanation
rather than crashing.

PyTorch defaults to one thread per core. On a laptop that starves the
interface thread, and the symptom is a window that stops repainting. The
detector caps it at `cpu_count - 1`. This was the single largest contributor
to interface responsiveness on the 2-core development machine; section 13 has
the measurements.

---

## 4. Tracking versus a track

A tracker id says "this is the same thing". It says nothing about what the
thing did. `TrackManager` turns the stream of `(id, box, timestamp)`
observations into `TrackState`: trajectory, distance travelled, smoothed
speed, heading, direction changes, zone history, dwell.

Two details that stop the numbers being noise:

* **Heading is only updated when the object has actually moved.** Below a
  small threshold the apparent direction is detector jitter, and counting it
  would produce hundreds of phantom "direction changes" for a stationary
  person.
* **Loitering uses a movement radius, not a displacement.** A person pacing
  back and forth has a large total distance and a small radius. Radius is the
  question worth asking.

### Units

Positions are normalised 0-1. Distances and speeds are therefore in
*fractions of the frame diagonal*, and speed in fractions per second. That is
honest: without a camera calibration there is no way to convert to metres. Set
`tracking.pixels_per_metre` after measuring a known distance in the scene and
speeds become available in m/s.

---

## 5. Zones

Normalised polygons, ray-cast containment, arbitrary shapes including concave
ones. Three decisions worth explaining:

**The reference point is not the box centre.** For a person, it is the
bottom-centre of the box: the feet are where the person actually is, while the
centre floats at chest height and enters a ground-plane zone early. Vehicles
use the centre, which matches their footprint better.

**Membership is debounced.** A detection box jitters every frame. A target
standing on a boundary would otherwise generate an enter/exit pair per frame.
A change of zone must persist for `debounce_frames` observations before it is
believed.

**Only boundary crossings emit events.** The engine tracks occupancy and emits
`enter` and `exit`, not "still inside". This is the first line of defence
against alert floods; incident deduplication is the second.

---

## 6. Line crossing

Detected by segment intersection between the object's previous and current
position and the configured line, not by comparing which side it is on. A fast
object can jump the line between two processed frames, and segment
intersection catches that directly.

Direction is reported twice: `entering`/`exiting` relative to the side the
operator declared as "in", and a compass word derived from actual heading.

---

## 7. Behavioural anomaly detection

An Isolation Forest over eight features that describe *what an object did*,
not what it looks like:

```
avg_speed  max_speed  dwell_seconds  distance_norm
direction_changes  zone_transitions  restricted_seconds  track_age
```

The engine learns from completed tracks in this scene, and scores new tracks
against that baseline.

**The cold start is the important part.** Below `min_samples` completed tracks
the engine returns `INSUFFICIENT_DATA` and a score of `None`. It never invents
a number to fill the gap.

This is a direct response to how the previous generation failed. It shipped a
model fitted on one video's frame-level statistics; on new footage it scored
92% of frames as anomalous, including 39 frames that contained no detections
at all. Two changes fix the class of problem: features describe *tracks*, not
frames, so they mean something across scenes; and the baseline is learned
here, not shipped.

A saved model is refused on load if its feature names do not match, so a stale
model from an older build cannot silently produce meaningless scores.

Isolation Forest is not additive, so exact attribution is not available. The
engine reports the four features furthest from the baseline mean instead,
which answers a narrower but still useful question.

---

## 8. Threat scoring

A plain weighted sum over an editable rule set. Every factor that fires is
recorded with its label and its points, and the reasons always add up to the
score — asserted by a test that reads the persisted JSON back and re-adds it.

Scoring is a pure function of a `ThreatContext` dataclass. It has no hidden
inputs and no access to the pipeline, which is why it is straightforward to
test and why an operator can trust the breakdown.

`minimum_levels` lets a rule floor an incident's level regardless of score, so
a watchlist hit can never be buried at LOW because its weight was turned down.

**Positioning.** These weights are engineering defaults chosen so the demo
behaves sensibly. They are not derived from any military, police or
international standard and have not been validated against real incident data.
The disclaimer is displayed next to every score in the interface, not just
written here.

---

## 9. Incidents: why deduplication is the whole game

One person standing in a restricted zone triggers the zone rule on every
processed frame. Without suppression that is hundreds of incidents for one
event, which is exactly what makes an alert list useless. So:

* a repeat of the same type for the same track inside `dedupe_seconds` does
  not open a new incident
* if the repeat scores higher, the existing incident is escalated in place and
  its escalation counter increments
* a *different* type for the same track is always new: "loitering" and
  "crossed the boundary" are different facts
* resolving an incident stops it being deduplicated against

Measured on the demo clip: around 240 frames of a person inside the restricted
zone produce exactly one intrusion incident, which later escalates when
loitering is confirmed. A test asserts that no track holds more than two
incidents of any one type.

---

## 10. ANPR

```
vehicle box → plate region proposal → preprocessing → OCR → validation
```

Region proposal is morphological: blackhat highlights dark-on-light text, a
Sobel pass finds the dense vertical edges characters make, a closing operation
merges them into one blob, and candidates are filtered on aspect ratio and
relative area. No extra model to download.

Preprocessing upscales, applies CLAHE for local contrast, bilateral-filters to
kill noise while keeping character edges, and binarises with Otsu, inverting
if the result came out light-on-dark.

OCR backends sit behind one interface. EasyOCR first, then Tesseract, then
disabled. With none installed the engine reports `available=False` and the rest
of the application is unaffected.

**The rule:** a plate number is never invented. If OCR returns nothing, returns
something below the confidence threshold, or the region does not look like a
plate, the result is `text=None` and the interface shows `UNREADABLE`. The
crop is saved regardless, because an unreadable plate is exactly the case
where a human needs the picture.

Character-confusion correction (O/0, I/1, S/5, B/8) is applied only when a
plate format is configured and the correction actually makes the text match
it. Without a known format there is no basis for the correction, so none is
made.

---

## 11. Persistence

SQLite, hand-written SQL, no ORM. The application is small enough that an ORM
would add more indirection than it removes, and the schema stays in one
readable file.

Twelve tables: `cameras`, `sessions`, `tracks`, `detections`, `zones`,
`incidents`, `alerts`, `evidence`, `plates`, `watchlist`, `system_logs`,
`settings`.

Thread safety: one connection shared between the GUI and worker threads, under
a re-entrant lock, opened with `check_same_thread=False`, journalling in WAL
mode so readers proceed while a write is in flight. That is what stops the
Analytics screen stalling the pipeline.

Timestamps are stored twice on purpose: `ts` as epoch seconds for arithmetic,
`ts_text` as ISO-8601 local time so the file is readable in any SQLite browser
without conversion.

Detections are **sampled**, not stored per frame. Writing every box of every
frame would put tens of thousands of rows into the database for a two-minute
clip and tell you nothing the tracks do not already say.

---

## 12. Evidence

Event-driven, never continuous. Three gates: a minimum threat level, a
per-track cooldown, and an optional retention sweep.

```
data/evidence/2026-09-04/CAM-01_0042_full_143012.jpg
                         CAM-01_0042_crop_143012.jpg
                         CAM-01_0007_plate_143014.jpg
```

The date folder makes review and retention easy; the incident id in the
filename means a stray file can always be traced back to its record.

The full-frame shot uses the **annotated** frame, because it shows what the
system saw and why it reacted, which is what a reviewer needs.

---

## 13. Threading

```
    GUI thread                          Worker thread (QThread)
    ──────────                          ───────────────────────
    window, widgets, painting           decode → YOLO → track → zones
    reads SQLite for the screens        → behaviour → threat → incident
                                        → evidence → SQLite
         ▲                                        │
         └───────── Qt signals (queued) ──────────┘
```

The contract:

* the worker owns the pipeline and the source; the GUI never touches either
* communication out is by signal only, queued across threads by Qt
* communication in is by `stop()`, `pause()`, `resume()`, which set flags the
  run loop checks; no blocking calls into the worker
* frames are emitted at most `gui.display_fps` times a second; **processing is
  not throttled**, because dropping a display frame costs nothing while
  dropping an analysis frame loses evidence
* the live track table refreshes at 4 Hz, not 25 Hz: nobody reads a table 25
  times a second, and rebuilding it is not free
* the worker calls `msleep(1)` after each frame to hand the GIL back

That last point is not cosmetic. Python's GIL means a compute-heavy background
thread can starve the interface.

`scripts/gui_live_test.py` measures it: it runs the real worker under the real
window and times a 100 ms GUI-thread timer, reporting how late each tick was.

Measured on the development machine, which has **two shared cores** and is the
worst realistic case:

| | before the fixes | after |
|---|---|---|
| average GUI event-loop lateness | 642 ms | **170-280 ms** (varies with load) |
| worst tick | 1007 ms | **340-540 ms** |

The range is honest: five runs of the same test on the same machine produced
averages of 172, 205, 217, 242 and 248 ms. Contention for two cores dominates,
and a single flattering measurement would not be representative.

What the numbers mean in practice: the window keeps repainting several times a
second throughout, and never stops responding. It is not as crisp as an idle
application. On a machine with four or more cores — any recent laptop — the
competition largely disappears, but that has not been measured here and no
figure for it is quoted.

Three changes produced the improvement, in order of how much they contributed:
capping PyTorch at `cpu_count - 1` threads, throttling the live track table to
4 Hz, and `msleep(1)` per frame in the worker. Lowering the CPython GIL switch
interval to 1 ms is also applied; on this machine its effect was inside the
run-to-run variance.

---

## 14. Error handling

Failures are surfaced, never swallowed:

* **Configuration**: a malformed file is a hard error with the file and the
  reason. A *missing* optional file degrades to a documented default and the
  degradation is reported through `AppConfig.warnings`.
* **Video source**: every failure carries an actionable message ("the file may
  be corrupt, or your OpenCV build may lack a decoder for this codec").
* **Model**: a missing model is refused explicitly, with the reason it is
  refused.
* **One bad frame**: logged, reported to the interface, and the session
  continues. Cameras produce corrupt frames; a session must survive them.
* **Database**: a failed write is logged and the pipeline carries on. Losing
  one row is better than losing the run.
* **The database log handler**: swallows its own failures and stops after the
  first, because a logging handler that raises during an error path turns a
  recoverable problem into a crash.
* **The worker thread**: wrapped in a catch-all that reports through
  `error_raised` rather than dying silently.

---

## 15. What was reused from the previous project

| Reused | Where it went |
|---|---|
| Ray-cast point-in-polygon | `core/geometry.py`, extended and tested against concave shapes |
| Normalised zone coordinates | Kept: the right decision, so profiles survive resolution changes |
| Zone kinds and priority mapping | `core/zone_engine.py`, extended with `NO_VEHICLE` and `PATROL` |
| Loitering and approach concepts | `core/behaviour_engine.py`, rebuilt on real tracks |
| Isolation Forest for anomalies | `core/anomaly_engine.py`, moved from frame features to track features |
| Four-tier priority model | `core/threat_engine.py`, made additive and explainable |
| Cooldown-based alert suppression | `core/incident_manager.py`, extended with escalation |
| Optical flow | Dropped. Per-track speed answers the same question better. |
| The 7-class aerial model | Kept in `models/` and selectable, but not the default |

| Replaced | Why |
|---|---|
| Streamlit dashboard | Requirement: a native desktop application |
| Hard-coded `CLASS_NAMES` table | Silently relabelled COCO classes |
| Silent model fallback | Produced confident nonsense from the wrong weights |
| Frame-level anomaly features | Not portable between videos; 92% false-positive rate |
| Shipped pre-fitted anomaly model | Calibrated to a different scene |
| JSON files as the database | No queries, no integrity, a 500-record cap that silently lost history |
| Demo data mixed with real results | The single biggest credibility problem in the original |
| IoU-only tracker | Fragments on any occlusion or frame skip |

---

## 16. Known limitations

* **One camera at a time.** The architecture supports more — nothing in
  `core/` is a singleton and the worker is per-camera — but running several
  YOLO streams on a laptop CPU makes all of them useless. Multi-camera is a
  scheduling problem, not a design problem.
* **Not real time on CPU.** Roughly 60-120 ms per frame with yolov8n. With
  `frame_stride: 2` that is around 8-16 analysed frames per second.
* **Zone geometry is edited in YAML.** There is no click-to-draw editor yet.
* **No re-identification.** An object that leaves the frame and returns gets a
  new identity.
* **Detection quality is the model's, not the application's.** With stock COCO
  weights, expect COCO-level performance. Small, distant, blurred or heavily
  occluded objects are missed.
* **No night-vision or thermal handling.** The night-time threat factor is a
  clock check, not an image analysis. It is labelled as such.
* **False-positive and false-negative rates are unmeasured.** Measuring them
  needs labelled footage from the actual deployment scene.
