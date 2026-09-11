# Demonstration guide

A twelve-minute walkthrough that shows the whole chain working, and the
answers to the questions a technical evaluator will actually ask.

---

## Before you start

```bash
source venv/bin/activate
python scripts/make_demo_video.py          # if you have not already
python scripts/run_acceptance.py --fresh   # confirms everything works
python run.py
```

The acceptance script is worth running beforehand even if you are not going to
show it. It takes about thirty seconds and it tells you whether anything is
broken before an audience finds out for you.

**Start with a clean database** so the numbers on screen are from this run:

```bash
rm -f data/database/borderguard.db*
```

---

## The walkthrough

### 1. Open with the honest framing (30 seconds)

> "This is a local desktop application. No browser, no cloud, no internet.
> Everything you are about to see runs on this laptop, and everything on
> screen comes from a SQLite database on this disk.
>
> It is a prototype. It shows how a system like this can be built. It is not
> field-ready, and I will be specific about what is missing."

Saying the limitation first makes everything after it more credible.

### 2. Dashboard (1 minute)

Open on **Dashboard**. It is empty, and that is the point.

> "Zero incidents, zero alerts, threat level LOW. There is no demo data in
> this application anywhere. If nothing has been processed, it says nothing
> has been processed."

Point at **System Status**: the model, the device it will use, the classes it
will look for, and the anomaly engine reporting that it has not learned
anything yet.

### 3. Cameras: start processing (2 minutes)

Go to **Cameras**. Four cameras are configured. Note CAM-04:

> "That is an RTSP camera, defined and disabled. The architecture handles
> network cameras today; I just do not have one on this desk. Turning it on is
> a flag in a YAML file, not a code change."

Select **CAM-01** and press **Start Processing**.

Then narrate what appears, in this order:

**Detection.** Boxes on the person and the vehicle.
> "Real YOLO detections. The class names come from the model file itself, not
> from a table in my code. If I configure a class the model does not support,
> it tells me at startup and ignores it."

**Tracking.** The `PERSON #01` labels.
> "Those numbers are stable. The same person keeps the same identity across
> frames. That is what makes everything downstream possible: you cannot
> measure how long someone has been somewhere if you cannot tell it is the
> same someone."

**Zones and lines.** The coloured polygons and the cyan line.
> "Configurable polygons in normalised coordinates, so they survive a change
> of resolution or camera. The red strip is the border, the orange box is the
> restricted compound, the yellow area excludes vehicles only."

**The track table.** Point at DWELL, SPEED, HEADING, TURNS, BEHAVIOUR.
> "This is what separates detection from surveillance. Not just 'there is a
> person', but how fast, which way, how long, how many times they have changed
> direction."

### 4. The incident chain (3 minutes)

Watch for the incidents to appear, and call them as they happen:

**Around 4 seconds in:** the person crosses the boundary line.
> "Line crossing, heading north, entering."

**Immediately after:** the person enters the restricted compound.
> "Intrusion. One incident. Not one per frame — the system tracks zone
> occupancy and only reacts to the crossing."

**Around 12 seconds in:** loitering fires, and a HIGH toast appears.
> "Now the interesting one. The person has been inside that zone for eight
> seconds of footage without moving more than six per cent of the frame. That
> is loitering, and it is a genuine temporal measurement — both conditions
> have to hold. Someone walking through for the same eight seconds is not
> loitering, and the system says so."

Point at the threat banner going from LOW to HIGH.

### 5. Incidents: the explainable score (2 minutes)

Stop processing. Go to **Incidents** and click the loitering incident.

This is the strongest screen. Read the breakdown aloud:

```
+  30  Restricted zone intrusion (Restricted Compound)
+  20  Loitering (12s)
=  60  TOTAL (HIGH)
```

> "The score is a weighted sum, and every factor that contributed is listed
> with its points. There is no black box here. An operator can always ask why,
> and the weights are editable in Settings — I will show you."

Then read the disclaimer under it out loud, deliberately:

> "'Prototype risk model. Weights are configurable engineering defaults, not a
> validated security standard.' That sentence is in the interface, not just in
> the documentation, because an operator needs to see it every time."

Point at the evidence thumbnails.
> "Captured automatically when the incident opened. Annotated, so it shows
> what the system saw and why. Indexed in the database, on disk under
> `data/evidence/` by date."

Press **Acknowledge**, then **Resolve**. Show the status changing.

### 6. Alerts and Analytics (1 minute)

**Alerts**: the feed, newest first, with the acknowledge and resolve workflow.

**Analytics**: every chart populated from the run that just happened.
> "All of this is SQL against the local database. Incidents by day, by hour,
> by camera, by type, by threat level. Nothing here is hard-coded — before I
> pressed Start, every one of these was empty."

### 7. Watchlist and ANPR (1 minute)

Go to **Watchlist**. Show the ANPR reads table.

If plates came back UNREADABLE, that is the moment to make the point:

> "The vehicle in this clip has no legible plate, so the system reports
> UNREADABLE and saves the crop for a human to look at. It does not guess.
> A surveillance system that invents plate numbers is worse than one with no
> ANPR at all, because you cannot tell which readings to trust."

Add a watchlist entry to show the mechanism, and be clear about it:

> "A match is raised only when OCR actually reads a plate that equals an entry
> here. There is no fuzzy matching by default, because a false watchlist hit
> is the most damaging false positive this system can produce."

### 8. Settings (1 minute)

Open **Settings** and flick through the tabs.

> "Nothing operational is hard-coded. Detection thresholds, the model, the
> device, loitering duration and radius, every threat weight, the level
> thresholds, evidence rules, retention, the deduplication window. It writes
> back to the YAML files in `config/`."

Show the Threat Scoring tab specifically — the editable weights make the
"configurable prototype" claim concrete rather than rhetorical.

### 9. Close on the limitations (1 minute)

> "What this does not do. It handles one camera at a time — the architecture
> supports more, but running several YOLO streams on a laptop CPU makes all of
> them useless. It is not real time on CPU: about 60 to 120 milliseconds a
> frame. It has no night-vision handling; the night-time threat factor is a
> clock check and it is labelled as one. It has no person re-identification, so
> someone who leaves and comes back gets a new identity. And I have not
> measured its false-positive rate, because doing that honestly needs labelled
> footage from the actual site.
>
> What I would do next is exactly that: get real footage, label a few hundred
> frames, and measure it."

---

## Questions you will be asked

**"Is the AI real, or is this a mock-up?"**
Open **System Logs** and show the model loading, the device selection, the
inference times. Or run `python scripts/run_acceptance.py` in a terminal: it
prints a pass/fail line for every stage and reads the results back out of
SQLite. Or change the confidence threshold in Settings and rerun — the number
of detections changes, which a mock-up could not do.

**"How do you know it is not just detecting, and calling everything an
intrusion?"**
The negative case. Process a clip where someone walks through the zone without
stopping. No loitering incident. The system distinguishes presence from
dwelling.

**"Why did it not flag that person?"**
Be honest. Either the detector missed them (COCO weights, small or occluded
target) or they never entered a zone. Both are visible on screen: the track
table shows what was tracked, the video shows what was boxed.

**"What is the accuracy?"**
> "I have not measured it, so I am not going to quote a number. Measuring it
> honestly needs labelled footage from the scene it would be deployed in. What
> I can tell you is which model is loaded and what its published metrics are."

Refusing to invent a number is the most credible thing you can do in this
conversation.

**"Could this be deployed?"**
> "No, and not because of the software. It would need site trials, a measured
> error rate, hardware validation, a security audit, a legal and privacy
> review, defined human oversight and operational approval. The software is
> the easy part."

**"What is the anomaly detection actually doing?"**
An Isolation Forest over eight behavioural features — speed, dwell, distance,
direction changes, zone transitions, time in restricted areas, track age. It
learns what ordinary movement looks like in that scene and flags tracks unlike
it. It needs 25 completed tracks first, and until then it says
`INSUFFICIENT_DATA` rather than guessing.

**"Does the interface freeze while it is processing?"**
No, and it is measurable: `python scripts/gui_live_test.py` runs the real
worker under the real window and times a GUI-thread timer. On a 2-core machine
under load, ticks ran 170-280 ms late on average while YOLO was running — the
window keeps repainting throughout, it is simply sharing two cores with the
detector. A blocked interface shows lateness in whole seconds and paints
nothing at all.

---

## Running on your own footage

Live, in front of an audience:

1. **Cameras → Open Video…**, pick the file.
2. Press **Start Processing**.

The default zone profile assumes a border along the top of the frame. If your
footage does not look like that, switch the camera to the `simple_centre`
profile, or edit `config/zones.yaml` beforehand. Do this *before* the demo,
not during it.

---

## If something goes wrong live

**Nothing is detected.** Wrong model for the viewpoint — `border_yolo.pt` is
aerial-trained and misses most people at ground level. Switch to `yolov8n.pt`
in Settings. Or the confidence threshold is too high; drop it to 0.25.

**Detections but no incidents.** The zones do not match the scene. Say so,
switch to `simple_centre`, restart processing.

**Everything says "learning baseline".** Correct and expected: the anomaly
engine needs 25 completed tracks. Explain that this is the system refusing to
guess, and move on. It is a good look, not a bad one.

**Playback is slow.** Raise `detection.frame_stride` in Settings. Say why:
*"I am processing every third frame to keep this responsive; the trade-off is
explicit and configurable."*

**The application will not start.** Run `python run.py` from a terminal so the
audience sees the dependency check tell you exactly what is missing.
