"""Read the COLOUR of an Indian number plate, and say what it means.

WHY COLOUR IS WORTH READING
---------------------------
In India the background colour of a plate is not decoration. It is a legal
classification, and it tells an operator something the digits do not:

    white  + black text    private vehicle, not for hire
    yellow + black text    commercial: taxi, cab, truck, bus
    green  + white text    electric, private
    green  + yellow text   electric, commercial
    black  + yellow text   self-drive rental
    red    + white text    temporary registration, or a government vehicle
    blue   + white text    foreign diplomatic mission

At a border post "that is a commercial vehicle" is an operationally
different fact from "that is a private car", and it is available from the
plate crop the pipeline already saves. Reading it costs one small image
and no model.

HOW
---
The plate is mostly background with characters over it. So the background
colour is the DOMINANT colour of the crop, and the text colour is what the
minority pixels are. Both are worked out in HSV rather than RGB, because
hue survives the shadow, glare and white balance that a real camera puts
on a plate, and RGB does not.

WHAT IT REFUSES TO DO
---------------------
It never guesses. A crop that is not confidently one of the seven schemes
comes back as UNKNOWN, and the caller shows nothing rather than inventing
a vehicle class. A plate that is dirty, half in shadow, or simply not a
plate at all is exactly the case where a confident wrong answer would be
worse than no answer, because "commercial vehicle" is the kind of claim
somebody might act on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

__all__ = ["PlateColour", "classify", "classify_file"]


#: What each scheme means, in the words an operator would use.
CATEGORY = {
    "WHITE":        ("Private", "Private vehicle, not for hire"),
    "YELLOW":       ("Commercial", "Taxi, cab, truck or bus"),
    "GREEN_WHITE":  ("Electric", "Electric vehicle, private"),
    "GREEN_YELLOW": ("Electric commercial", "Electric taxi or transport"),
    "BLACK":        ("Rental", "Self-drive rental vehicle"),
    "RED":          ("Temporary", "Temporary registration or government"),
    "BLUE":         ("Diplomatic", "Foreign diplomatic mission"),
}

#: Below this share of the crop, the dominant colour is not dominant enough
#: to be called the plate's background.
MIN_SHARE = 0.34

#: Below this the crop is too small to judge.
MIN_PIXELS = 240


@dataclass(frozen=True)
class PlateColour:
    """What colour the plate is, and what that classifies it as."""

    scheme: str                     # WHITE, YELLOW, ... or UNKNOWN
    category: Optional[str]         # "Commercial", "Private", ...
    meaning: Optional[str]          # the sentence for an operator
    share: float                    # how much of the crop agreed
    reason: str                     # why, when it could not decide

    @property
    def known(self) -> bool:
        return self.scheme != "UNKNOWN"

    @property
    def is_commercial(self) -> bool:
        """The question an operator actually asks of a yellow plate."""
        return self.scheme in ("YELLOW", "GREEN_YELLOW")


def _masks(hsv: np.ndarray) -> dict[str, np.ndarray]:
    """One boolean mask per colour scheme.

    The bands are deliberately wide. A plate photographed outdoors is
    never the colour it is on paper: yellow drifts towards orange in low
    sun and towards grey in shade, and white is rarely whiter than about
    200. Narrow textbook bands classify nothing in the field.
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    return {
        # Hue 15-43 covers plate yellow from mustard through to lemon.
        "YELLOW": (h >= 15) & (h <= 43) & (s >= 75) & (v >= 70),
        # White is defined by the ABSENCE of colour, not by a hue.
        "WHITE":  (s <= 55) & (v >= 135),
        "GREEN":  (h >= 44) & (h <= 92) & (s >= 55) & (v >= 45),
        "RED":    (((h <= 9) | (h >= 170)) & (s >= 90) & (v >= 65)),
        "BLUE":   (h >= 96) & (h <= 136) & (s >= 80) & (v >= 45),
        "BLACK":  (v <= 70),
    }


def classify(image: Optional[np.ndarray]) -> PlateColour:
    """Classify a plate crop. BGR in, classification out."""
    if image is None or getattr(image, "size", 0) == 0:
        return PlateColour("UNKNOWN", None, None, 0.0, "no image")

    if image.ndim == 2:                       # greyscale: no colour to read
        return PlateColour("UNKNOWN", None, None, 0.0, "greyscale crop")

    if image.shape[0] * image.shape[1] < MIN_PIXELS:
        return PlateColour("UNKNOWN", None, None, 0.0, "crop too small")

    # Trim a border. The crop comes from a box around the plate, so its
    # outer few percent is usually bumper, not plate, and bumper colour
    # would otherwise vote.
    h, w = image.shape[:2]
    my, mx = max(1, int(h * 0.12)), max(1, int(w * 0.06))
    inner = image[my:h - my, mx:w - mx] if h > 2 * my and w > 2 * mx else image

    blurred = cv2.GaussianBlur(inner, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    total = float(hsv.shape[0] * hsv.shape[1])

    masks = _masks(hsv)
    shares = {name: float(np.count_nonzero(m)) / total for name, m in masks.items()}

    # Characters. Plate text in a dull photograph is not truly black, so
    # this band is more generous than the BLACK background band.
    dark = hsv[..., 2] <= 105
    dark_share = float(np.count_nonzero(dark)) / total

    scheme = max(shares, key=lambda k: shares[k])
    share = shares[scheme]

    # ------------------------------------------------------------------
    # A colour scheme is BACKGROUND PLUS TEXT, and checking both is what
    # separates a plate from a photograph of something that happens to be
    # that colour. Measured on this project's own crops:
    #
    #   a wheel arch in shadow      BLACK 0.46, yellow text 0.00
    #   a plate crop of tarmac      BLACK 0.57, yellow text 0.00
    #   a real white plate          WHITE 0.33, dark text 0.05
    #
    # Judged on background alone, the first two came back as "Rental
    # vehicle", which is a confident wrong answer about a shadow. Every
    # scheme below therefore also has to show the text it is defined by.
    # ------------------------------------------------------------------

    if scheme == "BLACK":
        # Black plates are black WITH YELLOW characters. Without them this
        # is a shadow, a tyre or a dark window - not a plate.
        not_black = ~masks["BLACK"]
        outside = max(1, int(np.count_nonzero(not_black)))
        yellow_text = float(np.count_nonzero(masks["YELLOW"] & not_black)) / outside
        if yellow_text < 0.06:
            return PlateColour(
                "UNKNOWN", None, None, round(share, 3),
                "dark crop with no yellow lettering, so not a black plate")

    elif scheme == "WHITE":
        # A white plate carries dark characters. A blown-out sky or a
        # white bumper panel does not.
        if dark_share < 0.04:
            return PlateColour(
                "UNKNOWN", None, None, round(share, 3),
                "light crop with no dark lettering, so not a white plate")
        if share < 0.26:
            return PlateColour(
                "UNKNOWN", None, None, round(share, 3),
                f"too little white to call it a white plate ({share:.0%})")
        category, meaning = CATEGORY["WHITE"]
        return PlateColour("WHITE", category, meaning, round(share, 3), "")

    elif scheme == "GREEN":
        # Green splits by TEXT colour, and that split is the difference
        # between a private electric car and an electric taxi.
        not_bg = ~masks["GREEN"]
        if np.count_nonzero(not_bg) < 20:
            return PlateColour("UNKNOWN", None, None, round(share, 3),
                               "green crop with no lettering")
        text_yellow = float(np.count_nonzero(masks["YELLOW"] & not_bg))
        text_white = float(np.count_nonzero(masks["WHITE"] & not_bg))
        scheme = "GREEN_YELLOW" if text_yellow > text_white else "GREEN_WHITE"

    if share < MIN_SHARE:
        return PlateColour(
            "UNKNOWN", None, None, round(share, 3),
            f"no colour covers enough of the crop (best {scheme} {share:.0%})")

    category, meaning = CATEGORY[scheme]
    return PlateColour(scheme, category, meaning, round(share, 3), "")


def classify_file(path: str) -> PlateColour:
    """Classify a plate crop that is already on disk."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return PlateColour("UNKNOWN", None, None, 0.0, "could not read file")
    return classify(image)
