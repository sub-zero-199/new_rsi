from anpr.similarity import (
    calculate_plate_similarity, is_similar, best_match, generate_ocr_variations, are_similar_chars,
)


def test_identical():
    assert calculate_plate_similarity("MH12AB1234", "MH12AB1234") == 1.0


def test_ocr_error_high_similarity():
    assert calculate_plate_similarity("MH12AB1234", "MH12A81234") >= 0.8
    assert calculate_plate_similarity("DL1LAE0896", "DL4LAE0896") >= 0.8


def test_different_state_zero():
    assert calculate_plate_similarity("MH12AB1234", "KA05CD9999") == 0.0


def test_is_similar_threshold():
    assert is_similar("MH12AB1234", "MH12AB1284", 0.8)
    assert not is_similar("MH12AB1234", "MH99ZZ0000", 0.8)


def test_best_match_snaps():
    active = ["HR38AJ0543", "AP39UM5260", "MH12AB1234"]
    cand, score = best_match("MH12A81234", active, 0.8)
    assert cand == "MH12AB1234" and score >= 0.8
    cand, score = best_match("TN09XY4321", active, 0.8)
    assert cand is None


def test_variations_nonempty_and_exclude_original():
    vs = generate_ocr_variations("MH12AB1234", 4)
    assert vs and "MH12AB1234" not in vs


def test_are_similar_chars():
    assert are_similar_chars("8", "B")
    assert are_similar_chars("0", "O")
    assert not are_similar_chars("A", "Z")
