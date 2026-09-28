"""Tests for grammar-based repair of an OCR plate reading.

The repair is only safe because an Indian plate's character classes are
fixed: two letters, district digits, series letters, four digits. Every
test here is either "this real misread is repaired correctly" or "this
must NOT be forced into a plate".
"""

import pytest

from core.plate_grammar import normalise


# --------------------------------------------------------------------------
# Real misreads, from this project's own footage
# --------------------------------------------------------------------------

@pytest.mark.parametrize("misread,expected", [
    # CAM-04's lead vehicle is HR26BQ0008. EasyOCR reads B as 8 and Q as
    # 0, because the series position is where those confusions live.
    ("HR26800008", "HR26BO0008"),
    ("HR26800000", "HR26BO0000"),
    # G read for 6 in the district position.
    ("HR2G800008", "HR26BO0008"),
])
def test_real_misreads_are_repaired(misread, expected):
    result = normalise(misread)
    assert result.text == expected
    assert result.repaired
    assert result.valid


def test_the_state_and_district_survive_the_repair():
    """The operationally important part. Even when the series letters
    stay wrong, the first four characters decide the state and district,
    and those are what the repair gets right."""
    assert normalise("HR26800008").prefix == "HR26"
    assert normalise("HR2G800008").prefix == "HR26"


def test_a_correct_reading_is_left_alone():
    result = normalise("HR26BQ0008")
    assert result.text == "HR26BQ0008"
    assert not result.repaired
    assert result.valid


def test_another_correct_reading_is_left_alone():
    result = normalise("PB39N0203")
    assert result.text == "PB39N0203"
    assert not result.repaired


# --------------------------------------------------------------------------
# It must not manufacture plates
# --------------------------------------------------------------------------

@pytest.mark.parametrize("junk", [
    "JOH020J",        # too short, and not plate-shaped
    "",
    "AB",
    "1234",
    "XXXXXXXXXXXXXXXX",
])
def test_non_plates_are_returned_untouched_and_invalid(junk):
    result = normalise(junk)
    assert not result.valid
    assert not result.repaired


def test_a_repair_that_does_not_produce_a_plate_is_discarded():
    """If forcing the character classes still does not yield a plate,
    the original is handed back. A half-repaired string would be worse
    than the raw reading, because it looks corrected."""
    result = normalise("!!!!!!!!!!")
    assert not result.repaired
    assert not result.valid


def test_length_is_never_changed():
    for raw in ("HR26800008", "PB39N0203", "HR2G800008"):
        assert len(normalise(raw).text) == len(raw)


def test_only_known_confusions_are_substituted():
    """X is not confusable with any digit, so a letter position holding
    an X is left as an X rather than being forced into something."""
    result = normalise("HR26XY0008")
    assert result.text == "HR26XY0008"
    assert not result.repaired


# --------------------------------------------------------------------------
# The state letters are deliberately left alone
# --------------------------------------------------------------------------

def test_a_misread_state_is_not_repaired_into_a_different_state():
    """The regression this rule exists for.

    On CAM-04 one reading of the lead vehicle came back as "0R26800000",
    the H of HR misread as a zero. Repairing 0 to O yields "OR26BO0000",
    a perfectly valid plate that decodes to Nuapada, ODISHA - for a car
    registered in Haryana. The repair was grammatically correct and
    operationally wrong.

    A reading that did not get the state right must stay broken, so it
    fails the plausibility check and drops out of the vote, leaving the
    answer to readings that did read the state correctly.
    """
    result = normalise("0R26800000")
    assert result.text == "0R26800000"
    assert not result.repaired
    assert not result.valid


def test_everything_after_the_state_is_still_repaired():
    """A wrong series letter does not move the vehicle to another state,
    so those repairs stay."""
    result = normalise("HR2G800008")
    assert result.text == "HR26BO0008"
    assert result.prefix == "HR26"
    assert result.repaired
