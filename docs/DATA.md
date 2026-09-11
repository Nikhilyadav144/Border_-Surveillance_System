# Test data: where to get it and where to put it

This application ships with **no real surveillance footage**, and no claim of
access to any. This document explains how to get video you can legitimately
use, what scenarios are worth recording, and where the files go.

---

## Where files go

```
data/input/
```

Put `.mp4`, `.avi`, `.mov` or `.mkv` files there. They then appear:

* behind **Open Video…** on the Cameras screen (any file on the computer works
  too, this is just the default folder), or
* wired to a named camera in `config/cameras.yaml`:

```yaml
cameras:
  - id: "CAM-01"
    name: "Sector Alpha - Main Approach"
    location: "Sector Alpha"
    enabled: true
    source:
      type: video
      path: "my_footage.mp4"     # relative to data/input, or absolute
      loop: false                # true replays a short clip indefinitely
      target_fps: 25             # null runs as fast as the CPU allows
    zones_profile: "default"
```

---

## Six legitimate sources

### 1. Record it yourself

The best option, and the only one where you own the rights outright.

Put a phone on a tripod overlooking a courtyard, car park, garden or corridor,
and act out the scenarios below. Fifteen minutes of deliberately staged
footage tells you more about whether the system works than an hour of
downloaded video, because you know the ground truth: you know exactly when the
person entered the zone, so you can check whether the system agreed.

Practical notes:

* Get the camera high, looking down at 30-45 degrees. A camera at eye level
  makes people overlap constantly.
* 1080p at 25-30 fps is plenty. 4K just slows inference down.
* Frame it so a person is at least 60 pixels tall. Smaller than that and
  detection recall drops sharply.
* Keep the camera still. A moving camera makes zones meaningless.
* Record the same scene in daylight and at dusk. The difference will surprise
  you.

**Get consent from anyone you record**, and do not publish footage of people
who did not agree to appear in it.

### 2. Public research datasets

Free for research, but **read each licence** — several are non-commercial or
require registration.

| Dataset | Content | Fits |
|---|---|---|
| **VIRAT Video Dataset** | Ground-level surveillance of car parks and campuses, people and vehicles, annotated | Closest match to this application |
| **MOT Challenge** (MOT17, MOT20) | Pedestrian tracking benchmarks, crowded | Testing the tracker |
| **PETS 2009** | Multi-view crowd surveillance | Crowd and group behaviour |
| **UCSD Anomaly Detection** | Pedestrian walkways with labelled anomalies | Anomaly detection |
| **Avenue / ShanghaiTech** | Campus anomaly detection | Anomaly detection |
| **VisDrone** | Drone footage of people and vehicles | Aerial and elevated views |
| **DOTA** | Aerial imagery, oriented boxes | Aerial views, matches `border_yolo.pt` |
| **UA-DETRAC** | Traffic surveillance, vehicles | Vehicle tracking and ANPR practice |

Search each name plus "dataset" to find the official page. Do not take copies
from mirrors that strip the licence.

### 3. Openly licensed stock video

Pexels, Pixabay, Videvo and similar sites carry CC0 or similarly licensed
clips. Search for "security camera", "surveillance", "cctv footage", "people
walking", "parking lot". Check the licence on each individual clip; site-wide
statements are not reliable.

### 4. Publicly accessible live cameras

Many transport authorities and municipalities publish traffic and public-space
camera feeds. Some are RTSP or MJPEG and can be pointed at directly by
configuring an RTSP camera.

Check the terms of use for each feed before recording from it, and do not
access any camera you have not been given permission to use.

### 5. Simulated scenarios

Game engines and simulators (CARLA, AirSim, Unity, Unreal, even Blender)
render synthetic surveillance scenes with perfect ground truth, at any time of
day, with no privacy concerns at all. If you need night footage and cannot
film at night, this is the practical route.

### 6. Webcam

Your laptop camera works as a live source. Set `enabled: true` on `CAM-03` in
`config/cameras.yaml`, or add:

```yaml
  - id: "CAM-WEBCAM"
    name: "Desk Camera"
    enabled: true
    source:
      type: webcam
      index: 0
      target_fps: 20
    zones_profile: "simple_centre"
```

Use the `simple_centre` zone profile: the default profile assumes a border
along the top of the frame, which makes no sense for a desk.

macOS asks for Camera permission the first time. Grant it in System Settings →
Privacy & Security and restart the application.

---

## Scenarios worth recording

Each one exercises a different part of the pipeline. Recording all seven takes
about twenty minutes and gives you a proper regression set.

### 1. Normal pedestrian movement — the false-positive baseline
One or two people walking through at a steady pace, not entering any zone.
**Expect:** detections, tracks, no incidents. If this produces incidents, your
zones or thresholds are wrong.

### 2. Restricted zone intrusion
A person walks into the polygon marked RESTRICTED and out again.
**Expect:** one `restricted_zone_intrusion` incident at entry. One, not
hundreds.

### 3. Loitering
A person enters a zone and stays for longer than
`behaviour.loitering_seconds` (8 by default), standing roughly still.
**Expect:** an intrusion incident, then a `loitering` incident, and the
intrusion escalating as the score rises.

**Also record the negative case**: someone who stays in the zone the same
length of time but keeps walking around. That should *not* be loitering, and
it is the test that proves loitering is a real temporal measurement rather
than a synonym for "detected".

### 4. Line crossing
Someone walks across the virtual line in both directions.
**Expect:** one `line_crossing` incident each way, with `compass` and
`direction` reported correctly.

### 5. Vehicle movement
A car drives through, including through the NO_VEHICLE zone.
**Expect:** vehicle tracks and a `no_vehicle_zone_violation`. If ANPR is
installed and the plate is legible, a plate read. If the plate cannot be read,
`UNREADABLE` with the crop saved — that is the correct outcome, not a failure.

### 6. Multiple people and vehicles
Three or more people in frame at once, ideally crossing paths.
**Expect:** stable identities. Watch for id swaps when people overlap — that is
the honest limit of the tracker and worth knowing before a demo.

### 7. Night-time or low light
The same scene after dark.
**Expect:** noticeably worse detection. This is the most useful clip you can
record, because it shows you the real operating envelope. The night-time
threat factor is a clock check, not image analysis, so it does not compensate.

---

## Getting good results

**Match the model to the viewpoint.** `models/yolov8n.pt` is the stock COCO
model and knows person, car, motorcycle, bus, truck at ground level — the
right default. `models/border_yolo.pt` is a 7-class model trained on aerial
imagery (DOTA, xView, VisDrone) with a measured person recall of 0.28; use it
only for genuinely top-down aerial footage, and expect it to miss most people
in ground-level video.

**Draw the zones to match your scene.** The default profile assumes a border
along the top of the frame. On your own footage it will be meaningless until
you edit `config/zones.yaml`. To find coordinates: open a frame in any image
viewer, note the pixel position of each corner, divide x by the frame width
and y by the frame height.

**Tune the confidence threshold.** Start at 0.35. Too many false boxes: raise
it. Missing obvious people: lower it to 0.25 and accept more noise.

**Let the anomaly engine learn.** It needs 25 completed tracks before it
scores anything, and until then it reports `INSUFFICIENT_DATA`. That is
correct behaviour, not a fault. Process ten minutes of ordinary activity
first, so "unusual" means unusual *for this scene*.

---

## The demo clip

`scripts/make_demo_video.py` builds a 20-second composite: photographic
cut-outs of a person and a vehicle, taken from the sample images that ship
with Ultralytics, moved across a plain synthetic background along a scripted
path.

**The detections it produces are real.** YOLO genuinely finds the person and
the vehicle, the tracker genuinely follows them, and the zone, loitering and
threat logic genuinely fire. Only the scene is synthetic.

It exists so that a fresh clone can be verified end to end without first
hunting for footage. It is a test fixture. Never present output from it as a
real-world result, and never quote accuracy figures measured on it.

---

## Things not to do

**Do not use copyrighted footage without permission.** News clips, film,
YouTube video and commercial stock all have owners.

**Do not record people without consent**, and do not publish footage of people
who did not agree to appear in it. Filming in public is legal in many places;
publishing recognisable faces often is not, and the rules differ by country.

**Do not use real security footage you were not authorised to access.** That
is a criminal offence in most jurisdictions regardless of what you do with it.

**Do not claim access to real border security data.** If you are presenting
this, say plainly what your test data is. An evaluator will respect "I tested
on VIRAT and on footage I recorded myself" far more than a vague claim.

**Do not quote accuracy figures you have not measured.** If you need a number,
label a hundred frames of your own footage by hand, run the system against
them, and report what you actually got.
