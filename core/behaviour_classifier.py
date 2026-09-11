"""Name what a person is doing, from measured movement signals.

Explicitly rule-based, and said out loud rather than implied: there is no
trained action-recognition model in this project, and pretending otherwise
would be the exact failure this codebase exists to avoid. Every decision below
is a comparison against a documented threshold, and every classification can
be traced back to the numbers that produced it.

That is a real limitation and also a real advantage. A rule that says "the
hips dropped to under 60% of their normal height above the feet, and the knees
bent past 100 degrees, for at least a third of a second" can be argued with,
tuned, and explained to somebody who asks why an alert fired. A black-box
classifier at this stage could not be.

Two things keep it honest:

Debouncing. A candidate must hold for a minimum time before it becomes the
person's state. Pose estimation is noisy, and a single frame where an ankle
keypoint jumps can otherwise produce a "FALLING" event from someone standing
still. The cost is latency of a few tenths of a second, which is the right
trade for an alert an operator is meant to trust.

Confidence that means something. It is the margin by which the signals cleared
the thresholds, not a number chosen to look plausible. A borderline crouch
reports lower confidence than a deep one.

What this cannot do is worth stating plainly.

Sitting versus crouching. Both lower the hips and bend the knees, and the
difference is mostly what the person is resting on, which a skeleton does not
show. These two sit either side of one threshold and report modest confidence.

Lying down versus walking towards the lens. Both foreshorten the torso the
same way on one uncalibrated camera.

WALKING AND RUNNING ALONG THE CAMERA AXIS. This is the important one and it
is measured, not guessed. Speed here is image-plane speed, so a person walking
directly towards or away from the camera barely moves in the frame and reads
as STANDING. On the border-wall clip the officer is plainly walking, yet his
hips travel 0.30 of the frame in 16 seconds while he is 0.46 of the frame
tall: about 0.04 body heights per second, far below any honest walking
threshold. Lowering the threshold to catch him would classify genuinely
stationary people as walkers, which is worse.

The obvious fix is to read the gait itself - a walking person's ankles
separate and close even when their body does not translate. It was measured
before being built, and the data does not yet support it. Across nine tracks
the standard deviation of ankle separation was 0.013 to 0.023 of body height
for stationary people and 0.045 to 0.068 for moving ones, which looks
separable until you notice the highest reading of all, 0.149, belongs to a
track that never moved: a partly occluded figure whose ankle keypoints were
being estimated badly. A threshold drawn through those numbers would call that
person a walker. So gait is a real follow-up, not a shipped feature, and it
needs a keypoint-confidence gate and a periodicity test rather than plain
variance before it can be trusted.

Until then, axial walking reads as STANDING. That is a known limitation of
the measurement, stated here rather than hidden behind a tuned threshold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from core.movement import MovementSignals

# -- behaviour names --------------------------------------------------------
UNKNOWN = "UNKNOWN"
STANDING = "STANDING"
WALKING = "WALKING"
RUNNING = "RUNNING"
SITTING = "SITTING"
CROUCHING = "CROUCHING"
JUMPING = "JUMPING"
FALLING = "FALLING"
CRAWLING = "CRAWLING"
BENDING = "BENDING"
BENT_WALK = "BENT_WALK"
BENT_RUN = "BENT_RUN"

#: Phase 1 set, in the order the classifier tests them. Order matters: the
#: dramatic, short-lived events are tested before the postural ones, because a
#: person mid-fall is momentarily also a person who is bent over.
BEHAVIOURS: tuple[str, ...] = (
    FALLING, JUMPING, CRAWLING, CROUCHING, BENDING, SITTING,
    BENT_RUN, BENT_WALK, RUNNING, WALKING, STANDING,
)

#: Behaviours that combine a folded posture with travel. Kept as a named set
#: so the threat rules can ask "was this person moving low?" without
#: hard-coding label strings in three places.
LOW_PROFILE_MOVEMENT: frozenset[str] = frozenset({BENT_WALK, BENT_RUN})


@dataclass
class BehaviourEvent:
    """A behaviour that held long enough to be worth reporting."""

    track_id: int
    behavior: str
    confidence: float
    start_time: float
    duration: float = 0.0
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "track_id": int(self.track_id),
            "behavior": self.behavior,
            "confidence": round(float(self.confidence), 2),
            "duration": round(float(self.duration), 2),
            "start_time": round(float(self.start_time), 3),
            "evidence": self.evidence,
        }

    def describe(self) -> str:
        return (f"PERSON #{self.track_id:02d} {self.behavior} "
                f"({self.confidence:.0%}, {self.duration:.1f}s)")


@dataclass
class _Candidate:
    """A behaviour being considered, not yet confirmed."""

    name: str = UNKNOWN
    since: float = 0.0
    confidence: float = 0.0
    evidence: dict = field(default_factory=dict)


@dataclass
class BehaviourState:
    """What one person is currently doing, and for how long."""

    track_id: int
    behavior: str = UNKNOWN
    confidence: float = 0.0
    since: float = 0.0
    last_ts: float = 0.0
    evidence: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.last_ts - self.since)

    def to_dict(self) -> dict:
        return {
            "track_id": int(self.track_id),
            "behavior": self.behavior,
            "confidence": round(float(self.confidence), 2),
            "duration": round(self.duration, 2),
        }


def _confidence(*margins: float) -> float:
    """Confidence from how far the signals cleared their thresholds.

    Each margin is 0 at the threshold and 1 well past it. The result is the
    mean, floored at 0.5 for anything that passed at all, because a rule that
    fired did fire, and capped below 1.0 because a rule is not proof.
    """
    if not margins:
        return 0.5
    clipped = [max(0.0, min(1.0, m)) for m in margins]
    return round(0.5 + 0.45 * (sum(clipped) / len(clipped)), 3)


class BehaviourClassifier:
    """Classify and debounce one behaviour per Track ID."""

    def __init__(
        self,
        # Postural thresholds, all as fractions of the person's own height.
        crouch_hip_ratio: float = 0.62,
        sit_hip_ratio: float = 0.78,
        crawl_height_ratio: float = 0.45,
        crawl_torso_degrees: float = 55.0,
        bend_torso_degrees: float = 50.0,
        # Gait, in body heights per second.
        walk_speed: float = 0.45,
        run_speed: float = 1.9,
        # Vertical event rates, body heights per second.
        jump_rise_speed: float = 1.2,
        fall_speed: float = 1.6,
        fall_torso_degrees: float = 50.0,
        # Knee bend, degrees. 180 is a straight leg.
        bent_knee_degrees: float = 125.0,
        # Debounce, seconds a candidate must hold before it is believed.
        hold_seconds: float = 0.30,
        event_hold_seconds: float = 0.15,
    ) -> None:
        self.crouch_hip_ratio = float(crouch_hip_ratio)
        self.sit_hip_ratio = float(sit_hip_ratio)
        self.crawl_height_ratio = float(crawl_height_ratio)
        self.crawl_torso_degrees = float(crawl_torso_degrees)
        self.bend_torso_degrees = float(bend_torso_degrees)
        self.walk_speed = float(walk_speed)
        self.run_speed = float(run_speed)
        self.jump_rise_speed = float(jump_rise_speed)
        self.fall_speed = float(fall_speed)
        self.fall_torso_degrees = float(fall_torso_degrees)
        self.bent_knee_degrees = float(bent_knee_degrees)
        self.hold_seconds = float(hold_seconds)
        self.event_hold_seconds = float(event_hold_seconds)

        self._states: dict[int, BehaviourState] = {}
        self._candidates: dict[int, _Candidate] = {}

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def classify(self, s: MovementSignals) -> tuple[str, float, dict]:
        """The instantaneous reading, before any debouncing.

        Returns the behaviour, a confidence, and the numbers that decided it,
        so an operator asking "why did it say that" gets an answer.
        """
        if not s.ready:
            return UNKNOWN, 0.0, {"reason": s.reason}

        height = s.height_ratio
        hip = s.hip_ratio
        torso = s.torso_angle
        knee = s.knee_angle

        # -- FALLING: fast downward hip motion ending with the torso off
        #    vertical. Speed alone is not enough, or sitting down quickly
        #    would read as a fall.
        if (s.peak_fall_speed >= self.fall_speed
                and torso is not None and torso >= self.fall_torso_degrees):
            return FALLING, _confidence(
                (s.peak_fall_speed - self.fall_speed) / self.fall_speed,
                (torso - self.fall_torso_degrees) / 40.0,
            ), {"peak_fall_speed": round(s.peak_fall_speed, 2),
                "torso_angle": round(torso, 1)}

        # -- JUMPING: fast upward hip motion. Both feet leaving the ground is
        #    not measurable from a skeleton alone, so this is the honest
        #    proxy and it is why the threshold is high.
        if s.peak_rise_speed >= self.jump_rise_speed:
            return JUMPING, _confidence(
                (s.peak_rise_speed - self.jump_rise_speed)
                / self.jump_rise_speed,
            ), {"peak_rise_speed": round(s.peak_rise_speed, 2)}

        # -- CRAWLING: very low overall, torso well off vertical, and hands
        #    down near foot level. All three, because any one alone is
        #    something else: low alone is a crouch, torso alone is a bend.
        if (height is not None and height <= self.crawl_height_ratio
                and torso is not None and torso >= self.crawl_torso_degrees
                and s.limbs_level):
            return CRAWLING, _confidence(
                (self.crawl_height_ratio - height) / self.crawl_height_ratio,
                (torso - self.crawl_torso_degrees) / 35.0,
            ), {"height_ratio": round(height, 2),
                "torso_angle": round(torso, 1),
                "limbs_level": True}

        # -- CROUCHING: hips well down towards the ankles with the knees
        #    bent, and not moving off anywhere.
        if (hip is not None and hip <= self.crouch_hip_ratio
                and s.speed < self.walk_speed):
            margins = [(self.crouch_hip_ratio - hip) / self.crouch_hip_ratio]
            if knee is not None:
                margins.append((self.bent_knee_degrees - knee) / 60.0)
            return CROUCHING, _confidence(*margins), {
                "hip_ratio": round(hip, 2),
                "knee_angle": round(knee, 1) if knee is not None else None,
            }

        # -- BENDING: the torso goes over while the hips stay UP. That is
        #    what separates it from crouching, where the hips come down
        #    towards the ankles, and from crawling, where the whole body is
        #    low and the hands are near the ground.
        #
        #    Added after real footage: a person bending deeply to reach the
        #    ground measured a torso angle of 80 degrees from vertical with a
        #    hip ratio of 1.09, meaning their hips had not dropped at all.
        #    The crouch rule tests hip height, so it correctly said no - and
        #    nothing else described what they were plainly doing.
        if (torso is not None and torso >= self.bend_torso_degrees
                and (hip is None or hip > self.crouch_hip_ratio)
                and s.speed < self.walk_speed):
            return BENDING, _confidence(
                (torso - self.bend_torso_degrees) / 35.0,
            ), {"torso_angle": round(torso, 1),
                "hip_ratio": round(hip, 2) if hip is not None else None}

        # -- SITTING: hips lowered but not as far as a crouch, knees bent,
        #    stationary. Genuinely hard to separate from crouching on one
        #    camera, which is why it sits between two thresholds and reports
        #    modest confidence.
        if (hip is not None and hip <= self.sit_hip_ratio
                and knee is not None and knee <= self.bent_knee_degrees
                and s.speed < self.walk_speed):
            return SITTING, _confidence(
                (self.sit_hip_ratio - hip) / 0.2,
                (self.bent_knee_degrees - knee) / 60.0,
            ), {"hip_ratio": round(hip, 2), "knee_angle": round(knee, 1)}

        # -- BENT_WALK / BENT_RUN: folded over AND travelling.
        #
        #    Every posture rule above ends with `speed < walk_speed`, because
        #    crouching, bending and sitting were all defined as things you do
        #    while standing still. That had a consequence nobody intended:
        #    the moment a person moved, their posture was discarded and only
        #    their gait was left, so the most suspicious combination in the
        #    footage - travelling fast while folded double, which is how a
        #    person moves when they do not want to be seen over a barrier -
        #    came out as the blandest label available, WALKING.
        #
        #    Measured on CAM-01: PERSON #01 covered the length of the fence
        #    with his torso 65 to 80 degrees off vertical, accelerating from
        #    0.76 to 3.00 body heights per second. Hips stayed UP the whole
        #    way (hip ratio 1.0 to 1.66), so this is a bend at the waist, not
        #    a crouch, and the crouch rule was right to refuse it.
        #
        #    LIMITATION, stated because it decides where this can be trusted:
        #    the torso angle is measured in the image plane from one camera.
        #    A person travelling directly towards or away from the lens is
        #    foreshortened, and their shoulder-to-hip vector can read as a
        #    large angle while they are in fact upright. This behaviour is
        #    therefore reliable for movement ACROSS the view and unreliable
        #    along the camera axis - the same limitation the walking and
        #    running rules carry, documented at the top of this module.
        if (torso is not None and torso >= self.bend_torso_degrees
                and s.speed >= self.walk_speed):
            label = BENT_RUN if s.speed >= self.run_speed else BENT_WALK
            return label, _confidence(
                (torso - self.bend_torso_degrees) / 35.0,
                (s.speed - self.walk_speed)
                / max(self.run_speed - self.walk_speed, 1e-6),
            ), {"torso_angle": round(torso, 1),
                "speed": round(s.speed, 2),
                "hip_ratio": round(hip, 2) if hip is not None else None}

        # -- Gait, by speed in body heights per second.
        if s.speed >= self.run_speed:
            return RUNNING, _confidence(
                (s.speed - self.run_speed) / self.run_speed,
            ), {"speed": round(s.speed, 2)}

        if s.speed >= self.walk_speed:
            return WALKING, _confidence(
                (s.speed - self.walk_speed) / (self.run_speed - self.walk_speed),
            ), {"speed": round(s.speed, 2)}

        return STANDING, _confidence(
            (self.walk_speed - s.speed) / self.walk_speed,
        ), {"speed": round(s.speed, 2)}

    # ------------------------------------------------------------------
    # Debounced state
    # ------------------------------------------------------------------

    def update(self, signals: MovementSignals) -> Optional[BehaviourEvent]:
        """Fold one reading into a track's state.

        Returns an event only when the behaviour actually CHANGES and has
        held long enough to be believed, so a person standing still for a
        minute produces one event, not fifteen hundred.
        """
        track_id = signals.track_number
        reading, confidence, evidence = self.classify(signals)
        now = signals.ts

        state = self._states.get(track_id)
        if state is None:
            state = BehaviourState(track_id=track_id, since=now, last_ts=now)
            self._states[track_id] = state
        state.last_ts = now

        if reading == UNKNOWN:
            self._candidates.pop(track_id, None)
            return None

        if reading == state.behavior:
            # Same as before: refresh the confidence, keep the start time.
            state.confidence = max(state.confidence, confidence)
            state.evidence = evidence
            self._candidates.pop(track_id, None)
            return None

        candidate = self._candidates.get(track_id)
        if candidate is None or candidate.name != reading:
            self._candidates[track_id] = _Candidate(
                name=reading, since=now, confidence=confidence,
                evidence=evidence)
            return None

        candidate.confidence = max(candidate.confidence, confidence)
        candidate.evidence = evidence

        # Short, dramatic events get a shorter hold, or a half-second fall
        # would be filtered out by the same window that steadies a stance.
        hold = (self.event_hold_seconds if reading in (FALLING, JUMPING)
                else self.hold_seconds)
        if now - candidate.since < hold:
            return None

        state.behavior = reading
        state.confidence = candidate.confidence
        state.since = candidate.since
        state.evidence = candidate.evidence
        self._candidates.pop(track_id, None)

        return BehaviourEvent(
            track_id=track_id,
            behavior=reading,
            confidence=candidate.confidence,
            start_time=candidate.since,
            duration=0.0,
            evidence=candidate.evidence,
        )

    # ------------------------------------------------------------------

    def state(self, track_id: int) -> Optional[BehaviourState]:
        return self._states.get(track_id)

    def forget(self, track_id: int) -> None:
        self._states.pop(track_id, None)
        self._candidates.pop(track_id, None)

    def prune(self, live: set[int]) -> None:
        for track_id in [t for t in self._states if t not in live]:
            del self._states[track_id]
        for track_id in [t for t in self._candidates if t not in live]:
            del self._candidates[track_id]

    def summary(self) -> dict[str, int]:
        """How many people are currently in each behaviour."""
        counts: dict[str, int] = {}
        for state in self._states.values():
            counts[state.behavior] = counts.get(state.behavior, 0) + 1
        return counts
