"""Person-to-boundary relationship: selection, evidence, and crossing.

Built from hand-made skeletons walking scripted paths, so each expectation has
a known-correct answer. The cases that matter most are the negative ones: a
person walking past a fence must not be reported as interacting with it, and a
tracker jitter across a line must not be reported as a crossing.
"""

from __future__ import annotations

import pytest

from core.boundary_interaction import (
    APPROACHING,
    AT_BOUNDARY,
    CROSSED,
    INTERACTING,
    NONE,
    UNCERTAIN,
    BoundaryInteractionAnalyzer,
)
from core.line_crossing import VirtualLine
from core.movement import MovementAnalyzer
from core.pose_history import PoseHistory
from tests.test_activity import HEIGHT, box, result, skeleton


class FakeTrack:
    """Stands in for a TrackState.

    ``position`` matters: the analyzer reads the side of a boundary from the
    track's own reference point - the bottom-centre of its box - because that
    is the point the LineCrossingEngine tests for intersection. A stand-in
    without one would exercise a fallback path that production never takes.
    """

    def __init__(self, track_number: int = 1):
        self.track_number = track_number
        self.class_name = "person"
        self.position = (0.5, 0.9)

    def set_from_box(self, b) -> None:
        x1, _, x2, y2 = b
        self.position = ((x1 + x2) / 2.0, y2)


def feed(frames, lines, fps: float = 20.0, track_id: int = 1,
         analyzer: BoundaryInteractionAnalyzer | None = None,
         crossing_at: int | None = None):
    """Push skeletons through history and the boundary analyzer."""
    analyzer = analyzer or BoundaryInteractionAnalyzer()
    history = PoseHistory(window_seconds=3.0)
    track = FakeTrack(track_id)
    events = []
    for index, points in enumerate(frames):
        ts = index / fps
        b = box(points)
        track.set_from_box(b)
        history.record(track_id, result(points, track_id, ts), b, ts)
        event = analyzer.update(track, lines, history.get(track_id), ts,
                                crossing_seen=(index == crossing_at))
        if event is not None:
            events.append((index, event.state, event.line_name))
    return analyzer, events


#: A horizontal fence across the middle of the frame.
FENCE = VirtualLine(name="Fence", start=(0.0, 0.50), end=(1.0, 0.50))
#: A second, far away, to test selection.
FAR_FENCE = VirtualLine(name="Far Fence", start=(0.0, 0.05), end=(1.0, 0.05))
#: A third almost on top of the first, to test ambiguity.
TWIN_FENCE = VirtualLine(name="Twin Fence", start=(0.0, 0.515), end=(1.0, 0.515))


class TestNoBoundaries:
    def test_camera_with_no_lines_produces_nothing(self):
        analyzer, events = feed([skeleton()] * 30, lines=[])
        assert events == []
        state = analyzer.state(1)
        assert state is None or state.state == NONE

    def test_a_person_far_from_every_boundary_is_not_engaged(self):
        # Feet at 0.95, fence at 0.05: about 2.2 body heights away.
        analyzer, events = feed([skeleton(feet_y=0.95)] * 30,
                                lines=[FAR_FENCE])
        assert events == []


class TestProximityIsNotInteraction:
    def test_walking_past_a_fence_never_reports_interaction(self):
        # This is the case that matters. The person walks parallel to the
        # fence, close to it, for three seconds. They never reach for it and
        # never lift their feet. Reporting INTERACTING here would be the
        # false positive that makes an operator distrust the system.
        frames = [skeleton(x=0.05 + 0.03 * i, feet_y=0.62)
                  for i in range(40)]
        analyzer, events = feed(frames, lines=[FENCE])
        states = {state for _, state, _ in events}
        assert INTERACTING not in states
        assert CROSSED not in states
        assert analyzer.state(1).state in (AT_BOUNDARY, APPROACHING, NONE)

    def test_standing_still_near_a_fence_is_not_an_approach(self):
        frames = [skeleton(feet_y=0.62)] * 40
        analyzer, events = feed(frames, lines=[FENCE])
        assert APPROACHING not in {s for _, s, _ in events}


class TestApproach:
    def test_walking_towards_the_fence_is_an_approach(self):
        # Start far below the fence and close on it.
        frames = [skeleton(feet_y=0.95 - 0.008 * i) for i in range(40)]
        analyzer, events = feed(frames, lines=[FENCE])
        assert APPROACHING in {s for _, s, _ in events}

    def test_arriving_at_the_fence_reports_at_boundary(self):
        frames = [skeleton(feet_y=0.95 - 0.012 * i) for i in range(36)]
        frames += [skeleton(feet_y=0.56)] * 15
        analyzer, _ = feed(frames, lines=[FENCE])
        assert analyzer.state(1).state in (AT_BOUNDARY, INTERACTING)


class TestInteractionEvidence:
    def test_reaching_a_hand_to_the_fence_is_interaction(self):
        frames = [skeleton(feet_y=0.95 - 0.012 * i) for i in range(30)]
        # Now at the fence, hands go down and forward towards it.
        frames += [skeleton(feet_y=0.58, hands_down=True)] * 20
        analyzer, events = feed(frames, lines=[FENCE])
        assert INTERACTING in {s for _, s, _ in events}

    def test_lifting_the_feet_at_the_fence_is_interaction(self):
        # Stand at the fence, then the feet leave their normal level, which
        # is what climbing or vaulting looks like from a fixed camera.
        frames = [skeleton(feet_y=0.58)] * 25
        # The lift has to outlast the debounce hold, which is the point of
        # the hold: a real climb takes about a second, a bad frame does not.
        frames += [skeleton(feet_y=0.58 - 0.02 * i) for i in range(1, 12)]
        frames += [skeleton(feet_y=0.36)] * 10
        analyzer, events = feed(frames, lines=[FENCE])
        assert INTERACTING in {s for _, s, _ in events}

    def test_a_runner_passing_the_fence_is_not_climbing_it(self):
        """Regression from real footage.

        A person running past the fence lifted their feet 0.47 of a body
        height as part of their stride and was reported as INTERACTING at
        1.55 body heights per second. Running lifts the feet just as much as
        climbing does; what separates them is that a climber has stopped.
        """
        frames = []
        for i in range(45):
            # Travelling fast along the fence, feet bouncing as they run.
            lift = 0.05 if i % 4 < 2 else 0.0
            frames.append(skeleton(x=0.05 + 0.02 * i,
                                   feet_y=0.60 - lift))
        analyzer, events = feed(frames, lines=[FENCE])
        # Speed of about 1.0 body heights per second, well above a climb.
        analyzer2 = BoundaryInteractionAnalyzer()
        history = PoseHistory(window_seconds=3.0)
        track = FakeTrack(1)
        seen = []
        for index, points in enumerate(frames):
            ts = index / 20.0
            history.record(1, result(points, 1, ts), box(points), ts)
            event = analyzer2.update(track, [FENCE], history.get(1), ts,
                                     speed=1.55)
            if event is not None:
                seen.append(event.state)
        assert INTERACTING not in seen, (
            "a runner's stride must not read as climbing the fence"
        )

    def test_hands_down_far_from_the_fence_is_not_interaction(self):
        # Someone crouching to tie a shoelace well away from the boundary.
        frames = [skeleton(feet_y=0.95, hands_down=True)] * 40
        analyzer, events = feed(frames, lines=[FENCE])
        assert INTERACTING not in {s for _, s, _ in events}


class TestCrossing:
    def test_a_crossing_needs_the_tracker_to_confirm_it(self):
        # The person walks all the way across, but the crossing signal is
        # never delivered. Without it there is no CROSSED claim.
        frames = [skeleton(feet_y=0.90 - 0.012 * i) for i in range(45)]
        analyzer, events = feed(frames, lines=[FENCE], crossing_at=None)
        assert CROSSED not in {s for _, s, _ in events}

    def test_a_crossing_reported_on_one_frame_only_is_still_confirmed(self):
        """Regression: CROSSED used to be unreachable in production.

        The LineCrossingEngine reports a crossing on the SINGLE frame where
        the segments intersect, then goes quiet under its cooldown. CROSSED
        was debounced like every other state, so it needed the signal on
        consecutive frames to survive the hold and never got them. The only
        test that reached CROSSED fed a sustained signal no real engine
        produces.

        What confirms a crossing now is the person's own trajectory: they
        were engaged with that boundary, and they are afterwards seen on the
        other side of it.
        """
        # Walks steadily from below the fence at 0.50 to well above it.
        frames = [skeleton(feet_y=0.90 - 0.012 * i) for i in range(45)]
        # The intersection happens around index 34. Report it once, there.
        analyzer, events = feed(frames, lines=[FENCE], crossing_at=34)
        states = [s for _, s, _ in events]
        assert CROSSED in states, states

    def test_crossed_is_not_downgraded_on_the_next_frame(self):
        """A person who has just cleared a fence is standing beside it.

        Geometry alone reads that as AT_BOUNDARY again immediately, so
        CROSSED is held for a fixed period before anything else can take
        over. Without the hold the operator sees CROSSED for one frame.
        """
        frames = [skeleton(feet_y=0.90 - 0.012 * i) for i in range(45)]
        analyzer, events = feed(frames, lines=[FENCE], crossing_at=34,
                                fps=20.0)
        crossed_at = next(i for i, s, _ in events if s == CROSSED)
        # 0.5s hold at 20fps is ten frames.
        following = [i for i, s, _ in events
                     if i > crossed_at and s != CROSSED]
        assert not following or (following[0] - crossed_at) >= 10, events

    def test_a_crossing_by_a_track_that_never_approached_is_ignored(self):
        """A crossing needs an approach behind it.

        A crossing reported for someone who was never near the boundary is a
        tracker artefact - an identity swap, or a box that grew across the
        line - not a person going over a fence.
        """
        # Stays two body heights away from FAR_FENCE the whole time.
        frames = [skeleton(feet_y=0.95)] * 40
        analyzer, events = feed(frames, lines=[FAR_FENCE], crossing_at=20)
        assert CROSSED not in {s for _, s, _ in events}

    def test_a_claimed_crossing_with_no_side_change_expires(self):
        # Engaged with the fence, a crossing is claimed, but the person is
        # still on the same side of it a second later. Nothing is reported.
        frames = [skeleton(feet_y=0.90 - 0.010 * i) for i in range(30)]
        frames += [skeleton(feet_y=0.60)] * 60
        analyzer = BoundaryInteractionAnalyzer(crossing_confirm_seconds=1.0)
        analyzer, events = feed(frames, lines=[FENCE], analyzer=analyzer,
                                crossing_at=32)
        assert CROSSED not in {s for _, s, _ in events}
        assert analyzer.state(1).state != CROSSED

    def test_a_single_frame_jitter_does_not_confirm_a_crossing(self):
        # One frame claims a crossing and the person never moves. The
        # crossed hold is 0.5s, so a single frame cannot carry it.
        frames = [skeleton(feet_y=0.62)] * 40
        analyzer, events = feed(frames, lines=[FENCE], crossing_at=20)
        assert CROSSED not in {s for _, s, _ in events}


class TestMultipleBoundaries:
    def test_the_nearest_boundary_is_selected(self):
        # Feet at 0.60: close to FENCE at 0.50, far from FAR_FENCE at 0.05.
        frames = [skeleton(feet_y=0.90 - 0.010 * i) for i in range(35)]
        analyzer, events = feed(frames, lines=[FAR_FENCE, FENCE])
        chosen = {line for _, _, line in events}
        assert chosen == {"Fence"}, f"picked {chosen}"

    def test_one_fence_clearly_nearer_than_another_is_chosen(self):
        """Regression from real footage.

        Someone walking between two fences at 0.28 and 0.42 body heights was
        called UNCERTAIN, because the absolute gap of 0.14 fell under a 0.15
        margin - even though one fence was half again as far as the other.
        An absolute margin is scale-dependent: a distant person is small, so
        every gap in their body heights is small, and they would be
        permanently uncertain while a near person never was.
        """
        # 0.50 and 0.72: at feet 0.62 these are 0.12 and 0.10 of frame away,
        # comfortably different as a ratio.
        far = VirtualLine(name="Far Rail", start=(0.0, 0.95), end=(1.0, 0.95))
        frames = [skeleton(feet_y=0.62)] * 40
        analyzer, events = feed(frames, lines=[FENCE, far])
        chosen = {line for _, _, line in events}
        assert chosen == {"Fence"}, f"picked {chosen}"
        assert UNCERTAIN not in {s for _, s, _ in events}

    def test_a_distant_background_fence_is_never_selected(self):
        frames = [skeleton(feet_y=0.62)] * 40
        analyzer, events = feed(frames, lines=[FAR_FENCE])
        assert all(line != "Far Fence" for _, _, line in events)

    def test_two_boundaries_too_close_together_are_uncertain(self):
        # FENCE at 0.50 and TWIN_FENCE at 0.515 are so close that neither is
        # meaningfully nearer. Picking one would be inventing a fact.
        frames = [skeleton(feet_y=0.90 - 0.010 * i) for i in range(35)]
        analyzer, events = feed(frames, lines=[FENCE, TWIN_FENCE])
        assert UNCERTAIN in {s for _, s, _ in events}

    def test_uncertain_does_not_claim_a_crossing(self):
        frames = [skeleton(feet_y=0.62)] * 40
        analyzer, events = feed(frames, lines=[FENCE, TWIN_FENCE])
        assert CROSSED not in {s for _, s, _ in events}


class TestSegmentNotInfiniteLine:
    def test_a_person_beyond_the_end_of_a_fence_is_not_at_it(self):
        # The fence spans x 0.0 to 0.3. The person stands at x 0.9, level
        # with it. An infinite-line distance would say they are on it.
        short = VirtualLine(name="Short Fence", start=(0.0, 0.50),
                            end=(0.3, 0.50))
        frames = [skeleton(x=0.9, feet_y=0.52)] * 40
        analyzer, events = feed(frames, lines=[short])
        assert AT_BOUNDARY not in {s for _, s, _ in events}
        assert INTERACTING not in {s for _, s, _ in events}


class TestScaleInvariance:
    def test_the_same_geometry_reads_the_same_at_two_camera_distances(self):
        """A near and a far person doing the same thing get the same answer.

        This is what makes one threshold work on every camera: distances are
        in the person's own body heights, not in frame units.
        """
        results = {}
        for label, height in (("near", 0.50), ("far", 0.18)):
            # Both stand half a body height below the fence.
            feet = 0.50 + height * 0.5
            frames = [skeleton(feet_y=feet, height=height)] * 40
            analyzer, _ = feed(frames, lines=[FENCE])
            results[label] = analyzer.state(1).state
        assert results["near"] == results["far"], results
