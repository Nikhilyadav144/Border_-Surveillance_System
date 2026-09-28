"""Tests for the RTO origin decoder.

The point of these is not that the table is complete - it is not, and the
module says so - but that the module never INVENTS an answer. Every test
below is either "this real plate decodes correctly" or "this input must
produce nothing rather than a guess".
"""

import pytest

from core import rto


# --------------------------------------------------------------------------
# Real plates, read by the project's own OCR from CAM-04's footage
# --------------------------------------------------------------------------

@pytest.mark.parametrize("plate,state,district", [
    ("PB39N0203", "Punjab", "Rajpura"),
    ("PB12AH9880", "Punjab", "Ropar"),
    ("HR26BQ0008", "Haryana", "Gurgaon"),
])
def test_plates_from_the_footage_decode(plate, state, district):
    origin = rto.decode(plate)
    assert origin.state == state
    assert origin.district == district
    assert origin.known and origin.district_known


def test_summary_reads_as_a_place():
    assert rto.decode("HR26BQ0008").summary() == "Gurgaon, Haryana"


# --------------------------------------------------------------------------
# OCR does not produce tidy input
# --------------------------------------------------------------------------

@pytest.mark.parametrize("written", [
    "PB39N0203", "PB 39 N 0203", "pb39n0203", "  PB-39-N-0203  ",
])
def test_spacing_and_case_do_not_change_the_answer(written):
    assert rto.decode(written).district == "Rajpura"


# --------------------------------------------------------------------------
# Never invent
# --------------------------------------------------------------------------

@pytest.mark.parametrize("junk", [None, "", "   ", "!!!", "1234", "ZZ99XX1111"])
def test_unrecognised_input_returns_nothing(junk):
    origin = rto.decode(junk)
    assert origin.state is None
    assert origin.district is None
    assert not origin.known
    assert origin.summary() == "Origin unknown"


def test_known_state_with_unknown_district_says_so():
    """The exact case the table cannot cover.

    Telangana was created in 2014 and the register this table came from
    predates it, so TS decodes to the state and to no district. The module
    must report that rather than leaving a blank that looks like a bug.
    """
    origin = rto.decode("TS09EA1234")
    assert origin.state == "Telangana"
    assert origin.district is None
    assert origin.known
    assert not origin.district_known
    assert "not in table" in origin.summary()


def test_ladakh_decodes_to_the_state_only():
    origin = rto.decode("LA01A1234")
    assert origin.state == "Ladakh"
    assert origin.district is None


# --------------------------------------------------------------------------
# Shapes that break the usual rule
# --------------------------------------------------------------------------

def test_bharat_series_is_national_and_has_no_district():
    """22BH1234AA puts the year first, so "first two letters are the state"
    would misread it. It is nationwide by design, so a missing district is
    the correct answer and not a gap."""
    origin = rto.decode("22BH1234AA")
    assert origin.state_code == "BH"
    assert origin.district is None
    assert origin.summary() == "Bharat series (all-India)"


def test_delhi_single_digit_codes_resolve():
    """Delhi writes DL1 to DL18, unpadded, unlike every other state."""
    assert rto.decode("DL3CAB1234").state == "Delhi"
    assert rto.decode("DL3CAB1234").district is not None


def test_both_spellings_of_renamed_states_decode():
    """Odisha was OR before 2011, Uttarakhand was UA before 2007. Vehicles
    carrying the old prefix are still on the road."""
    assert rto.decode("OR02AB1234").state == "Odisha"
    assert rto.decode("OD02AB1234").state == "Odisha"
    assert rto.decode("UA07AB1234").state == "Uttarakhand"
    assert rto.decode("UK07AB1234").state == "Uttarakhand"


# --------------------------------------------------------------------------
# The table itself
# --------------------------------------------------------------------------

def test_every_district_code_starts_with_a_known_state():
    """A district row whose prefix is not in STATES would be unreachable."""
    unknown = sorted({code[:2] for code in rto.DISTRICTS
                      if code[:2] not in rto.STATES})
    assert unknown == []


def test_coverage_is_reported_honestly():
    cov = rto.coverage()
    assert cov["district_codes"] == len(rto.DISTRICTS)
    assert cov["state_codes"] == len(rto.STATES)


# --------------------------------------------------------------------------
# Letter-for-digit misreads in the district position
# --------------------------------------------------------------------------

@pytest.mark.parametrize("misread,district", [
    ("HPO14844131", "Shimla Urban (tourist buses & taxis)"),  # O read for 0
    ("PBI2AH9880", "Ropar"),                                   # I read for 1
    ("HRZ6BQ0008", "Gurgaon"),                                 # Z read for 2
    ("PBl2AH9880", "Ropar"),                                   # lowercase l
])
def test_digit_confusions_in_the_district_position_are_repaired(misread, district):
    """An Indian plate's district position is digits only, so a letter
    there is a misread rather than an alternative spelling. That is what
    makes repairing it safe - the string could not have been valid as read."""
    assert rto.decode(misread).district == district


def test_repair_does_not_force_a_district_out_of_nonsense():
    """XX cannot be digits even after correction, so the district must
    come back empty rather than being bent into a number."""
    origin = rto.decode("HPXX1234")
    assert origin.state == "Himachal Pradesh"
    assert origin.district is None


def test_repair_never_touches_the_series_letters():
    """Only the district position is corrected. The series letters after it
    are genuinely letters, and rewriting them would turn a correct read
    into a wrong plate."""
    assert rto.decode("PB39N0203").plate == "PB39N0203"
    assert rto.decode("PB39O0203").district == "Rajpura"
