"""One glyph must produce one character — and the label it beat is worth keeping.

The detector routinely emits two labels for a character it cannot settle. They used to be
concatenated, so a Z read as both 'A' and 'Z' turned Z8089 into AZ8089 and a 9-character
plate was recorded as a 10-character one. Measured on 303 real crops, 46% of reads
contained such a pair.
"""
from anpr.models.ocr import (CharBox, CharSlot, assemble_plate, resolve_slots,
                             spelling_candidates, SAME_GLYPH_FRAC)

# the real boxes behind GJ14 Z8089, 19 Aug 16:08 (x, y, conf, height as measured)
GJ14Z8089 = [
    CharBox("G", 155.1, 33.0, 0.941, 45.0), CharBox("J", 186.2, 33.5, 0.892, 47.2),
    CharBox("1", 217.5, 34.1, 0.944, 46.5), CharBox("4", 247.2, 34.8, 0.912, 47.5),
    CharBox("A", 138.0, 89.9, 0.691, 47.7), CharBox("Z", 138.1, 89.9, 0.547, 48.0),
    CharBox("8", 169.5, 90.6, 0.932, 46.1), CharBox("0", 201.8, 90.9, 0.939, 46.3),
    CharBox("8", 233.1, 92.1, 0.940, 46.2), CharBox("9", 264.3, 92.9, 0.909, 46.8),
]


def test_two_labels_on_one_glyph_become_one_character():
    text, confs = assemble_plate(GJ14Z8089)
    assert text == "GJ14A8089"          # was 'GJ14AZ8089' — the A/Z pair is 0.1px apart
    assert len(confs) == 9


def test_neighbouring_characters_are_never_merged():
    """Real neighbours sit ~30px apart; the threshold is 15% of ~47px height = ~7px."""
    slots = resolve_slots(GJ14Z8089)
    assert [s.best.label for s in slots] == list("GJ14A8089")
    assert sum(len(s.alternatives) for s in slots) == 1


def test_the_beaten_label_is_kept_as_the_second_spelling():
    """The model knew: 'A or Z'. That is the lead the registry can settle."""
    assert spelling_candidates(GJ14Z8089, limit=4) == ["GJ14A8089", "GJ14Z8089"]


def test_a_far_weaker_label_is_not_offered():
    """A near-tie is a real doubt; a 0.94-vs-0.12 gap is noise and must not reach the API."""
    chars = [CharBox("8", 100.0, 50.0, 0.94, 40.0), CharBox("0", 100.1, 50.0, 0.12, 40.0)]
    slot = resolve_slots(chars)[0]
    assert slot.alternatives and not slot.credible_alternatives
    assert spelling_candidates(chars, limit=4) == ["8"]


def test_every_combination_is_offered_when_two_positions_are_in_doubt():
    chars = [
        CharBox("A", 100.0, 50.0, 0.60, 40.0), CharBox("M", 100.1, 50.0, 0.55, 40.0),
        CharBox("P", 140.0, 50.0, 0.68, 40.0), CharBox("R", 140.1, 50.0, 0.60, 40.0),
    ]
    out = spelling_candidates(chars, limit=4)
    assert out[0] == "AP"                       # both winners first
    assert set(out) == {"AP", "AR", "MP", "MR"}  # a plate may need BOTH corrections


def test_clean_reads_are_untouched():
    chars = [CharBox("G", 100.0, 50.0, 0.95, 40.0), CharBox("J", 130.0, 50.0, 0.93, 40.0)]
    assert assemble_plate(chars)[0] == "GJ"
    assert spelling_candidates(chars, limit=4) == ["GJ"]


def test_threshold_sits_in_the_measured_gap():
    """Duplicates measure <0.05 of char height, real neighbours >=0.30 — nothing between."""
    assert 0.05 < SAME_GLYPH_FRAC < 0.30
