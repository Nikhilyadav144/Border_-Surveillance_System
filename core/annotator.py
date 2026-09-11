"""Frame overlay drawing.

Pure rendering: it reads state and returns a new image, never mutating the
frame it was given and never touching the database. Keeping it separate means
the pipeline can run headless (tests, batch processing) by simply not calling
it.

Every overlay is individually switchable through ``gui.overlays`` in
``application.yaml``, because an operator reviewing footage often wants the
boxes off.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np

from core.tracker import TrackState
from core.zone_engine import Zone, ZoneKind

logger = logging.getLogger(__name__)

# BGR palette, matched to the GUI theme.
COLOUR_PERSON = (120, 220, 90)
COLOUR_VEHICLE = (250, 170, 40)
COLOUR_OTHER = (200, 200, 200)
COLOUR_ALERT = (60, 70, 240)
COLOUR_LOITER = (40, 190, 255)
COLOUR_TEXT = (245, 245, 245)
COLOUR_HUD_BG = (24, 18, 14)

ZONE_COLOURS = {
    ZoneKind.RESTRICTED: (60, 70, 240),
    ZoneKind.BORDER: (60, 70, 240),
    ZoneKind.NO_VEHICLE: (40, 180, 250),
    ZoneKind.PATROL: (120, 200, 120),
    ZoneKind.OBSERVATION: (140, 180, 90),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def hex_to_bgr(value: str, fallback: tuple[int, int, int] = (200, 200, 200)):
    """Convert ``#rrggbb`` to an OpenCV BGR tuple."""
    try:
        text = value.lstrip("#")
        if len(text) != 6:
            return fallback
        r, g, b = (int(text[i:i + 2], 16) for i in (0, 2, 4))
        return (b, g, r)
    except (ValueError, AttributeError):
        return fallback


class Annotator:
    """Draw detections, tracks, zones, lines and a HUD onto frames."""

    def __init__(
        self,
        overlays: dict[str, bool] | None = None,
        vehicle_classes: Iterable[str] = (),
    ) -> None:
        defaults = {
            "boxes": True, "labels": True, "track_ids": True,
            "trajectories": True, "zones": True, "lines": True, "hud": True,
        }
        defaults.update(overlays or {})
        self.overlays = defaults
        self.vehicle_classes = {c.lower() for c in vehicle_classes}

    # -- helpers ----------------------------------------------------------

    def _class_colour(self, class_name: str) -> tuple[int, int, int]:
        lowered = class_name.lower()
        if lowered == "person":
            return COLOUR_PERSON
        if lowered in self.vehicle_classes:
            return COLOUR_VEHICLE
        return COLOUR_OTHER

    @staticmethod
    def _to_pixels(box_norm: Sequence[float], w: int, h: int
                   ) -> tuple[int, int, int, int]:
        return (
            int(box_norm[0] * w), int(box_norm[1] * h),
            int(box_norm[2] * w), int(box_norm[3] * h),
        )

    @staticmethod
    def _label_box(image: np.ndarray, text: str, x: int, y: int,
                   colour: tuple[int, int, int], scale: float = 0.45,
                   max_width: int | None = None) -> None:
        """Filled label chip with the text above a box.

        Two details keep a crowded scene readable: the chip is clamped to the
        width of the object it belongs to, so labels on neighbouring objects
        do not overlap each other, and when there is no room above the box the
        chip drops inside it instead of being clipped off the top of the frame.
        """
        if max_width is not None and max_width > 24:
            # Trim the text until the chip fits the object it labels.
            while text and cv2.getTextSize(text, _FONT, scale, 1)[0][0] + 8 \
                    > max_width:
                text = text[:-1]
            if not text:
                return

        (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, 1)
        chip_h = th + baseline + 4
        top = y - chip_h
        if top < 0:                       # no room above: put it inside
            top = y
        bottom = top + chip_h
        right = min(image.shape[1], x + tw + 8)
        left = max(0, min(x, image.shape[1] - (tw + 8)))

        cv2.rectangle(image, (left, top), (right, bottom), colour, -1)
        cv2.putText(image, text, (left + 4, bottom - baseline - 1),
                    _FONT, scale, (20, 20, 20), 1, cv2.LINE_AA)

    # -- layers -----------------------------------------------------------

    def draw_zones(self, image: np.ndarray, zones: Iterable[Zone],
                   alpha: float = 0.16) -> np.ndarray:
        """Translucent zone polygons with a solid outline and a name tag."""
        if not self.overlays.get("zones", True):
            return image

        h, w = image.shape[:2]
        overlay = image.copy()
        outlines: list[tuple[np.ndarray, tuple[int, int, int], str]] = []

        for zone in zones:
            if len(zone.polygon) < 3:
                continue
            colour = hex_to_bgr(zone.colour, ZONE_COLOURS.get(
                zone.kind, COLOUR_OTHER))
            points = np.array(
                [[int(px * w), int(py * h)] for px, py in zone.polygon],
                dtype=np.int32,
            )
            cv2.fillPoly(overlay, [points], colour)
            outlines.append((points, colour, f"{zone.name} [{zone.kind}]"))

        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        for points, colour, label in outlines:
            cv2.polylines(image, [points], True, colour, 2, cv2.LINE_AA)
            anchor = points[points[:, 1].argmin()]
            self._label_box(image, label, int(anchor[0]),
                            max(14, int(anchor[1])), colour, scale=0.4)
        return image

    def draw_lines(self, image: np.ndarray, lines: Iterable[Any]) -> np.ndarray:
        """Virtual boundaries, with an arrow showing the 'entering' side."""
        if not self.overlays.get("lines", True):
            return image

        h, w = image.shape[:2]
        for line in lines:
            colour = hex_to_bgr(line.colour, (216, 180, 0))
            start = (int(line.start[0] * w), int(line.start[1] * h))
            end = (int(line.end[0] * w), int(line.end[1] * h))
            # A reference-only boundary is drawn dashed and labelled REF, so
            # the operator can see at a glance which lines will raise an
            # alert and which are only there for the system to reason with.
            # Two lines that look identical but behave differently is how an
            # overlay misleads the person relying on it.
            alerting = bool(getattr(line, "alerting", True))
            if alerting:
                cv2.line(image, start, end, colour, 2, cv2.LINE_AA)
            else:
                self._dashed_line(image, start, end, colour)

            mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            if not alerting:
                self._label_box(image, f"{line.name} [REF]", start[0] + 6,
                                max(14, start[1] - 4), colour, scale=0.4)
                continue
            arrow = {
                "north": (mid[0], mid[1] - 34),
                "south": (mid[0], mid[1] + 34),
                "east": (mid[0] + 34, mid[1]),
                "west": (mid[0] - 34, mid[1]),
            }.get(line.entering_direction, (mid[0], mid[1] - 34))
            cv2.arrowedLine(image, mid, arrow, colour, 2, cv2.LINE_AA,
                            tipLength=0.35)
            self._label_box(image, line.name, start[0] + 6,
                            max(14, start[1] - 4), colour, scale=0.4)
        return image

    @staticmethod
    def _dashed_line(image: np.ndarray, start: tuple[int, int],
                     end: tuple[int, int], colour: tuple[int, int, int],
                     dash: int = 14, thickness: int = 2) -> None:
        """A dashed segment, for boundaries that never raise an alert."""
        x1, y1 = start
        x2, y2 = end
        length = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        if length <= 0:
            return
        steps = max(1, length // dash)
        for i in range(0, steps, 2):
            a = i / steps
            b = min(1.0, (i + 1) / steps)
            pa = (int(x1 + (x2 - x1) * a), int(y1 + (y2 - y1) * a))
            pb = (int(x1 + (x2 - x1) * b), int(y1 + (y2 - y1) * b))
            cv2.line(image, pa, pb, colour, thickness, cv2.LINE_AA)

    def draw_tracks(
        self,
        image: np.ndarray,
        tracks: Iterable[TrackState],
        alert_tracks: Optional[set[int]] = None,
    ) -> np.ndarray:
        """Boxes, identity labels and trajectory tails."""
        h, w = image.shape[:2]
        alerts = alert_tracks or set()

        for track in tracks:
            box = track.box
            if box == (0.0, 0.0, 0.0, 0.0):
                continue
            x1, y1, x2, y2 = self._to_pixels(box, w, h)

            if track.track_number in alerts:
                colour = COLOUR_ALERT
                thickness = 3
            elif track.loitering:
                colour = COLOUR_LOITER
                thickness = 3
            else:
                colour = self._class_colour(track.class_name)
                thickness = 2

            if self.overlays.get("boxes", True):
                cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)

            if self.overlays.get("labels", True):
                pieces: list[str] = []
                if self.overlays.get("track_ids", True):
                    pieces.append(track.label)
                else:
                    pieces.append(track.class_name.upper())
                pieces.append(f"{track.avg_confidence:.2f}")
                if track.loitering:
                    pieces.append("LOITER")
                if track.plate_text:
                    pieces.append(track.plate_text)
                width = max(96, (x2 - x1) + 40)
                # Allow a little overhang so a narrow box still shows its id.
                self._label_box(image, "  ".join(pieces), x1, y1, colour,
                                max_width=width)

                # The intelligence stack. Every line is omitted when the
                # backend has nothing to put in it, so an empty stack means
                # the pipeline genuinely knows nothing yet rather than that
                # the overlay forgot to ask.
                if self.overlays.get("person_intel", True):
                    self._intel_stack(image, track, x1, y1, x2, y2,
                                      colour, width)

            if self.overlays.get("trajectories", True) and len(track.samples) > 2:
                points = np.array(
                    [[int(s.x * w), int(s.y * h)] for s in list(track.samples)[-60:]],
                    dtype=np.int32,
                )
                cv2.polylines(image, [points], False, colour, 2, cv2.LINE_AA)
                cv2.circle(image, tuple(points[-1]), 4, colour, -1, cv2.LINE_AA)
        return image

    #: Colours for the threat badge. Keys are ThreatEngine level names.
    _THREAT_COLOURS = {
        "LOW": (120, 200, 120),
        "MEDIUM": (36, 178, 255),
        "HIGH": (8, 104, 247),
        "CRITICAL": (77, 72, 229),
    }

    def _intel_stack(self, image: np.ndarray, track: Any,
                     x1: int, y1: int, x2: int, y2: int,
                     colour: tuple[int, int, int], width: int) -> None:
        """Behaviour, movement, boundary and threat, under the id line.

        Deliberately compact and deliberately incomplete. Two or three short
        lines sit under the box header; the full record belongs in the
        operator's side panel, not burned into the picture. An overlay that
        covers the person defeats the point of watching them.
        """
        lines: list[tuple[str, tuple[int, int, int]]] = []

        behaviour = getattr(track, "behaviour", "UNKNOWN")
        if behaviour and behaviour != "UNKNOWN":
            confidence = float(getattr(track, "behaviour_confidence", 0.0))
            text = behaviour
            if confidence > 0:
                text = f"{behaviour} {confidence:.0%}"
            direction = getattr(track, "movement_direction", None)
            if direction:
                text = f"{text}  {str(direction).upper()[:2]}"
            lines.append((text, colour))

        state = getattr(track, "boundary_state", "NONE")
        if state and state != "NONE":
            line_name = getattr(track, "boundary_line", None)
            text = state if not line_name else f"{state}  {line_name}"
            lines.append((text, COLOUR_ALERT if state == "CROSSED" else colour))

        threat = getattr(track, "threat_level", None)
        if threat:
            lines.append((f"THREAT {threat}",
                          self._THREAT_COLOURS.get(str(threat).upper(),
                                                   COLOUR_ALERT)))

        if not lines:
            # Nothing concluded YET. Say that, rather than drawing a bare box
            # that looks exactly like a box the system is ignoring.
            #
            # This is not cosmetic. Measured on CAM-01, 52 of 246 person
            # frames - 21% - reach this state, almost all of them in a
            # track's first second and a half while the movement layer waits
            # for enough pose samples to trust. During that window an
            # operator saw an identity number and no sign that anything was
            # being worked out, which reads as a broken overlay.
            #
            # Both messages are honest about a different thing: one says the
            # body could not be measured, the other says it is being measured
            # and no conclusion has been reached. Neither guesses a
            # behaviour.
            pose_status = str(getattr(track, "pose_status", "") or "")
            if pose_status == "UNAVAILABLE":
                lines.append(("NO POSE", COLOUR_OTHER))
            else:
                lines.append(("ANALYSING", COLOUR_OTHER))

        # Under the header when there is room below the box top, otherwise
        # above it, so the stack never lands on the person's head.
        y = y1 + 34
        for text, line_colour in lines:
            self._label_box(image, text, x1, y, line_colour, scale=0.42,
                            max_width=width)
            y += 20

    def draw_detections(
        self, image: np.ndarray, detections: Iterable[Any]
    ) -> np.ndarray:
        """Raw detections, for when tracking is unavailable."""
        if not self.overlays.get("boxes", True):
            return image
        for det in detections:
            colour = self._class_colour(det.class_name)
            p1 = (int(det.x1), int(det.y1))
            p2 = (int(det.x2), int(det.y2))
            cv2.rectangle(image, p1, p2, colour, 2)
            if self.overlays.get("labels", True):
                self._label_box(
                    image, f"{det.class_name} {det.confidence:.2f}",
                    p1[0], p1[1], colour,
                )
        return image

    def draw_hud(self, image: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """Top-left status strip: camera, frame, fps, counts, threat level."""
        if not self.overlays.get("hud", True):
            return image

        lines = [
            f"{info.get('camera_id', '')}  {info.get('timestamp', '')}",
            f"FRAME {info.get('frame_index', 0)}   "
            f"{info.get('fps', 0.0):.1f} FPS   "
            f"{info.get('inference_ms', 0.0):.0f} ms",
            f"PERSONS {info.get('persons', 0)}   "
            f"VEHICLES {info.get('vehicles', 0)}   "
            f"TRACKS {info.get('tracks', 0)}",
        ]
        level = str(info.get("threat_level", "LOW"))
        lines.append(f"THREAT {level}   INCIDENTS {info.get('incidents', 0)}")

        pad = 8
        widths = [cv2.getTextSize(t, _FONT, 0.5, 1)[0][0] for t in lines]
        box_w = max(widths) + pad * 2
        box_h = len(lines) * 20 + pad * 2

        panel = image[0:box_h, 0:box_w].copy()
        cv2.rectangle(panel, (0, 0), (box_w, box_h), COLOUR_HUD_BG, -1)
        cv2.addWeighted(panel, 0.72, image[0:box_h, 0:box_w], 0.28, 0,
                        image[0:box_h, 0:box_w])

        level_colour = {
            "CRITICAL": (60, 70, 240), "HIGH": (40, 130, 250),
            "MEDIUM": (40, 190, 255), "LOW": (120, 200, 120),
        }.get(level, COLOUR_TEXT)

        for i, text in enumerate(lines):
            colour = level_colour if i == len(lines) - 1 else COLOUR_TEXT
            cv2.putText(image, text, (pad, pad + 15 + i * 20),
                        _FONT, 0.5, colour, 1, cv2.LINE_AA)
        return image

    # -- composite --------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        tracks: Iterable[TrackState] = (),
        zones: Iterable[Zone] = (),
        lines: Iterable[Any] = (),
        hud: dict[str, Any] | None = None,
        alert_tracks: Optional[set[int]] = None,
    ) -> np.ndarray:
        """Draw every enabled layer onto a copy of *frame*."""
        image = frame.copy()
        self.draw_zones(image, zones)
        self.draw_lines(image, lines)
        self.draw_tracks(image, tracks, alert_tracks)
        if hud:
            self.draw_hud(image, hud)
        return image
