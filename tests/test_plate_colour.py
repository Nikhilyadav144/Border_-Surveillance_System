"""Tests for plate colour classification.

Two kinds of test here. Synthetic plates prove the seven Indian schemes
are recognised and told apart. Then a set of NEGATIVE tests proves the
thing that actually matters in the field: a crop that is not a plate must
come back UNKNOWN rather than be given a confident vehicle class.

That second group exists because the first version of this module called
a wheel arch in shadow a "self-drive rental vehicle" - a confident wrong
answer about a shadow, and exactly the sort of claim an operator might
act on.
"""

import numpy as np
import pytest

from core.plate_colour import classify


def plate(bg_bgr, text_bgr, width=160, height=40):
    """A synthetic plate: a coloured rectangle with character bars on it."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = bg_bgr
    # Six character-shaped bars, as a real plate's lettering.
    for i in range(6):
        x = 12 + i * 24
        img[10:height - 10, x:x + 11] = text_bgr
    return img


WHITE = (235, 235, 235)
BLACK = (18, 18, 18)
YELLOW = (25, 200, 235)      # BGR
GREEN = (70, 140, 40)
RED = (40, 40, 200)
BLUE = (170, 60, 30)


# --------------------------------------------------------------------------
# The seven schemes
# --------------------------------------------------------------------------

def test_white_plate_is_a_private_vehicle():
    result = classify(plate(WHITE, BLACK))
    assert result.scheme == "WHITE"
    assert result.category == "Private"
    assert not result.is_commercial


def test_yellow_plate_is_commercial():
    """The one the operator cares about: yellow means taxi, cab, truck
    or bus, and it is the difference between a private car and a vehicle
    for hire."""
    result = classify(plate(YELLOW, BLACK))
    assert result.scheme == "YELLOW"
    assert result.category == "Commercial"
    assert result.is_commercial


def test_green_with_white_text_is_a_private_electric_vehicle():
    result = classify(plate(GREEN, WHITE))
    assert result.scheme == "GREEN_WHITE"
    assert not result.is_commercial


def test_green_with_yellow_text_is_an_electric_taxi():
    """Same background, different lettering, different vehicle class."""
    result = classify(plate(GREEN, YELLOW))
    assert result.scheme == "GREEN_YELLOW"
    assert result.is_commercial


def test_black_with_yellow_text_is_a_rental():
    result = classify(plate(BLACK, YELLOW))
    assert result.scheme == "BLACK"
    assert result.category == "Rental"


def test_red_plate_is_temporary_or_government():
    assert classify(plate(RED, WHITE)).scheme == "RED"


def test_blue_plate_is_diplomatic():
    assert classify(plate(BLUE, WHITE)).scheme == "BLUE"


# --------------------------------------------------------------------------
# It must refuse when it cannot tell
# --------------------------------------------------------------------------

def test_a_dark_crop_with_no_yellow_lettering_is_not_a_black_plate():
    """A wheel arch, a tyre or a shadow. Dark, but not a plate.

    This is the exact false positive the first version produced: it
    reported "self-drive rental vehicle" for a photograph of a shadow.
    """
    shadow = np.full((40, 160, 3), 20, dtype=np.uint8)
    result = classify(shadow)
    assert result.scheme == "UNKNOWN"
    assert "yellow lettering" in result.reason


def test_a_blank_light_crop_is_not_a_white_plate():
    """A bright bumper panel or an overexposed sky has no characters."""
    panel = np.full((40, 160, 3), 230, dtype=np.uint8)
    result = classify(panel)
    assert result.scheme == "UNKNOWN"
    assert "dark lettering" in result.reason


@pytest.mark.parametrize("bad,why", [
    (None, "no image"),
    (np.zeros((0, 0, 3), dtype=np.uint8), "no image"),
    (np.zeros((4, 6, 3), dtype=np.uint8), "too small"),
    (np.zeros((40, 160), dtype=np.uint8), "greyscale"),
])
def test_unusable_input_returns_unknown(bad, why):
    result = classify(bad)
    assert result.scheme == "UNKNOWN"
    assert result.category is None
    assert why in result.reason


def test_a_multicoloured_crop_has_no_dominant_scheme():
    """Random noise: no colour covers enough of it to be a background."""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, size=(40, 160, 3), dtype=np.uint8)
    assert classify(noise).scheme == "UNKNOWN"


# --------------------------------------------------------------------------
# The contract the interface relies on
# --------------------------------------------------------------------------

def test_unknown_never_offers_a_category():
    """The interface shows `category` directly, so UNKNOWN must carry
    nothing rather than something that renders as a vehicle class."""
    result = classify(np.full((40, 160, 3), 20, dtype=np.uint8))
    assert not result.known
    assert result.category is None
    assert result.meaning is None
