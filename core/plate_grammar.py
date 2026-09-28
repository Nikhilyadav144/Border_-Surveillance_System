"""Repair an OCR plate reading using the grammar of an Indian plate.

THE OBSERVATION
---------------
An Indian registration number has a fixed shape:

    HR 26 BQ 0008
    ^^ ^^ ^^ ^^^^
    |  |  |  four digits, the vehicle's serial number
    |  |  one to three letters, the series
    |  one or two digits, the RTO district
    two letters, the state

Every position is either a LETTER or a DIGIT, and which it is never
varies. So when OCR returns a digit where a letter must be, that is not
an alternative reading - it is a misread, and the correct character is
knowable.

MEASURED ON THIS PROJECT'S OWN FOOTAGE
--------------------------------------
The lead vehicle on CAM-04 carries HR26BQ0008. EasyOCR, given a 4x
upscaled crop, returns:

    HR26800008     B read as 8, Q read as 0
    HR26800000     the same, plus a trailing digit lost

Both are unreadable as plates and both fail the format check. Repaired
positionally they become HR26BO0008 and HR26BO0000: valid plates, and -
this is the part that matters operationally - the first four characters
are exactly right, so the vehicle decodes to Gurgaon, Haryana.

WHAT IT WILL NOT DO
-------------------
It only substitutes within the known confusion pairs, and only where the
grammar forces the class. It never invents a character, never changes
length, and returns the input unchanged when the string cannot be a
plate at all. A repair that could not have been the reading is not a
repair.

And it never touches the two state letters. See the note in normalise():
repairing those turned a Haryana car into an Odisha one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["PlateReading", "normalise"]


#: Digit shapes that an OCR engine returns for a letter. Only the pairs
#: that genuinely look alike in the fonts used on plates.
TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "4": "A",
    "5": "S", "6": "G", "8": "B",
}

#: And the reverse, for the positions that must be digits.
TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "U": "0",
    "I": "1", "L": "1", "J": "1",
    "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8",
}

#: The shape the repaired string has to end up in: two letters, one or
#: two digits, one to three letters, then four digits.
FULL_PLATE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

#: Some older and some commercial plates carry no series letters.
SHORT_PLATE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[0-9]{4}$")


@dataclass(frozen=True)
class PlateReading:
    """A repaired plate, and what had to be changed to get there."""

    text: str            # the repaired string, or the input if unrepairable
    original: str        # exactly what OCR returned, cleaned of punctuation
    repaired: bool       # whether anything was substituted
    changes: int         # how many characters were substituted
    valid: bool          # whether the result matches the plate grammar

    @property
    def prefix(self) -> str:
        """State and district: the part that decides where it is from."""
        return self.text[:4]


def _clean(raw: str) -> str:
    return "".join(ch for ch in str(raw or "").upper() if ch.isalnum())


def normalise(raw: str) -> PlateReading:
    """Repair a reading against the plate grammar.

    The string is anchored from BOTH ends, because those are the parts
    whose class is certain: the first two characters are always letters
    and the last four are always digits. Whatever sits between them is
    the series, which is always letters.
    """
    text = _clean(raw)

    if len(text) < 8 or len(text) > 11:
        # Too short to be a plate, or long enough that the character
        # positions cannot be trusted to line up.
        return PlateReading(text, text, False, 0,
                            bool(FULL_PLATE.match(text)))

    chars = list(text)
    changes = 0

    def force(index: int, table: dict[str, str]) -> None:
        nonlocal changes
        ch = chars[index]
        if ch in table:
            chars[index] = table[ch]
            changes += 1

    # THE STATE LETTERS ARE NEVER REPAIRED.
    # -------------------------------------
    # They are always letters, so repairing them is grammatically sound
    # and operationally wrong. Measured on CAM-04: one reading of the
    # lead vehicle came back as "0R26800000", the H misread as a zero.
    # Mapping 0 to O produces "OR26BO0000" - a perfectly valid plate
    # that decodes to Nuapada, ODISHA, for a car registered in Haryana.
    #
    # The state code is the most consequential two characters on the
    # plate, and a repair there cannot be checked against anything. So a
    # reading that did not get the state right is left broken, fails the
    # plausibility check, and is dropped from the vote - leaving the
    # answer to the readings that did read the state correctly.
    #
    # Everything after the state is repaired, because a wrong series
    # letter does not move the vehicle to another state.

    # The last four: the serial number. Always digits.
    for i in range(len(chars) - 4, len(chars)):
        force(i, TO_DIGIT)

    # Position 2 is always a district digit. Position 3 is a district
    # digit only when what follows can still make a whole plate - most
    # districts are two digits, a few are one.
    force(2, TO_DIGIT)
    two_digit_district = len(chars) >= 9
    if two_digit_district:
        force(3, TO_DIGIT)

    # Everything between the district and the serial is the series, and
    # the series is letters.
    series_start = 4 if two_digit_district else 3
    for i in range(series_start, len(chars) - 4):
        force(i, TO_LETTER)

    repaired = "".join(chars)
    valid = bool(FULL_PLATE.match(repaired) or SHORT_PLATE.match(repaired))

    # A repair that does not produce a plate is not a repair. Hand back
    # what OCR actually said, so nothing downstream treats a mangled
    # string as if it had been corrected.
    if not valid:
        return PlateReading(text, text, False, 0, False)

    return PlateReading(repaired, text, changes > 0, changes, True)
