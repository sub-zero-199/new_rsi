import pytest
from anpr.normalize import (
    clean_and_correct_text, get_canonical_plate, validate_plate_format,
    is_bh_series, is_valid_state_code, normalize_raw,
)


def test_normalize_raw_strips():
    assert normalize_raw("mh 12 ab-1234.") == "MH12AB1234"


def test_clean_standard_plate():
    assert clean_and_correct_text("MH12AB1234") == "MH12AB1234"
    assert clean_and_correct_text("mh 12 ab 1234") == "MH12AB1234"


def test_clean_rejects_bad_state():
    assert clean_and_correct_text("ZZ12AB1234") == ""


def test_clean_corrects_digit_in_series_position():
    # '8' in a series-letter position should be corrected to 'B'
    assert clean_and_correct_text("MH12A81234") == "MH12AB1234"


def test_bh_series():
    assert is_bh_series("22BH6517A")
    assert get_canonical_plate("22BH6517A") == "22BH6517A"


def test_validate_format():
    assert validate_plate_format("MH12AB1234")
    assert not validate_plate_format("MH1")
    assert not validate_plate_format("TOOLONGPLATE123")


def test_valid_state_code():
    assert is_valid_state_code("MH12AB1234")
    assert not is_valid_state_code("ZZ12AB1234")


# ---- the structure gate: a read must be SHAPED like a plate, not merely start like one ----

@pytest.mark.parametrize("read,expected", [
    ("HP240751",    "HP240751"),     # 8 chars, no series letters at all — a real plate
    ("MH120751",    "MH120751"),
    ("GJ12B2455",   "GJ12B2455"),    # 9, one series letter
    ("GJ12BY2455",  "GJ12BY2455"),   # 10, two
    ("DL1CAJ1993",  "DL1CAJ1993"),   # 10, THREE — Delhi
    ("DL10CAJ1993", "DL10CAJ1993"),  # 11, three + two-digit district
    ("22BH1234AA",  "22BH1234AA"),   # BH series
    ("AGJ39TA2660", "GJ39TA2660"),   # stray leading char is stripped, then matches
])
def test_real_plate_shapes_are_accepted(read, expected):
    assert clean_and_correct_text(read) == expected


@pytest.mark.parametrize("read", [
    "CGJ1ZBM9353",    # stray C makes 'CG' a real state code — reached the portal before
    "CGJ1ZFB1526",
    "GAJ1ZBW866",     # stray -> 'GA'
    "GJ12BY24",       # truncation
    "GJ03BVW906801",  # 13-char noise
    "GJ03WVN950M06",
    "0J03V59081",     # '0J' is not a state code
])
def test_noise_shaped_like_a_plate_is_rejected(read):
    """These all carried a valid 2-letter state code and an allowed length, so the old
    `match_plate_patterns(text) or text` fallback let them through as real vehicles."""
    assert clean_and_correct_text(read) == ""


def test_patterns_are_a_full_match_not_a_substring_hunt():
    """re.search accepted anything CONTAINING a plate, so junk rode along on both sides."""
    from anpr.normalize import match_plate_patterns
    assert match_plate_patterns("GJ12BY2455") == "GJ12BY2455"
    assert match_plate_patterns("XGJ12BY2455X") is None
