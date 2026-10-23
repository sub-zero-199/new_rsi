"""OCR-aware plate similarity + OCR-variation generation.

Ported from v11_1 src/send_vehicle_number.py (calculate_plate_similarity, are_similar_chars,
generate_ocr_variations). Similarity is an edit distance with OCR-confusable character costs
plus targeted boosts for the common Indian-plate misread patterns.
"""
from __future__ import annotations

from anpr.normalize import get_canonical_plate, VALID_STATE_CODES

_SIMILAR_GROUPS = [
    {"O", "0", "Q", "D"},
    {"I", "1", "L", "|"},
    {"S", "5"},
    {"Z", "2"},
    {"B", "8"},
    {"G", "6"},
    {"E", "3"},
    {"1", "4", "7", "I"},
    {"4", "1", "9", "A"},
]


def are_similar_chars(c1: str, c2: str) -> bool:
    return any(c1 in g and c2 in g for g in _SIMILAR_GROUPS)


def _char_similarity(c1: str, c2: str) -> float:
    if c1 == c2:
        return 1.0
    base_groups = [
        {"O", "0", "Q", "D"}, {"I", "1", "L", "|"}, {"S", "5"}, {"Z", "2"},
        {"B", "8"}, {"G", "6"}, {"C", "G"}, {"4", "A"},
    ]
    for g in base_groups:
        if c1 in g and c2 in g:
            return 0.8
    return 0.0


def calculate_plate_similarity(plate1: str, plate2: str) -> float:
    """Similarity in [0,1]. 1.0 == same canonical plate. Reject across state codes."""
    if not plate1 or not plate2:
        return 0.0
    if plate1 == plate2:
        return 1.0
    c1, c2 = get_canonical_plate(plate1), get_canonical_plate(plate2)
    if c1 == c2:
        return 1.0
    if len(c1) >= 2 and len(c2) >= 2 and c1[:2] != c2[:2]:
        return 0.0

    # missing-first-char case (e.g. "LACBE2791" vs "DL4CBE2791")
    if abs(len(c1) - len(c2)) == 1:
        longer, shorter = (c1, c2) if len(c1) > len(c2) else (c2, c1)
        if longer[1:] == shorter and longer[0] in {"D", "I", "1", "L"}:
            return 0.9

    m, n = len(c1), len(c2)
    if abs(m - n) > 2:
        return 0.0

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if c1[i - 1] == c2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1] + (1 - _char_similarity(c1[i - 1], c2[j - 1]))
                dp[i][j] = min(sub, dp[i][j - 1] + 1, dp[i - 1][j] + 1)

    max_len = max(m, n)
    if max_len == 0:
        return 1.0
    edit = dp[m][n]
    sim = 1.0 - (edit / max_len)

    # boost: missing/OCR digit in the number part
    if abs(m - n) <= 2 and max_len >= 9:
        sn = (c1 if m < n else c2)[-4:]
        ln = (c2 if m < n else c1)[-4:]
        if sn in ln or ln in sn:
            sim = max(sim, 0.85)
        elif len(sn) == len(ln):
            diffs = [(a, b) for a, b in zip(sn, ln) if a != b]
            if len(diffs) <= 2 and all(are_similar_chars(a, b) for a, b in diffs):
                sim = max(sim, 0.85)

    # boost: same-length, <=2 OCR-confusable diffs
    if m == n and edit <= 2 and max_len >= 9:
        diffs = [(c1[i], c2[i]) for i in range(m) if c1[i] != c2[i]]
        if len(diffs) <= 2 and all(are_similar_chars(a, b) for a, b in diffs):
            sim = max(sim, 0.85)

    # penalty: too many differences
    if max_len >= 9 and edit > 2:
        sim *= 0.5
    return max(0.0, sim)


def is_similar(plate1: str, plate2: str, threshold: float = 0.80) -> bool:
    return calculate_plate_similarity(plate1, plate2) >= threshold


def best_match(plate: str, candidates, threshold: float = 0.80):
    """Return (best_candidate, score) over an iterable of candidate plate strings,
    or (None, 0.0) if none reaches `threshold`.
    """
    best, best_score = None, 0.0
    for cand in candidates:
        s = calculate_plate_similarity(plate, cand)
        if s > best_score:
            best, best_score = cand, s
    return (best, best_score) if best_score >= threshold else (None, 0.0)


def generate_ocr_variations(vehicle_number: str, max_variations: int = 5) -> list[str]:
    """Generate position-aware OCR variations of a plate (to fill the candidate set)."""
    if not vehicle_number or len(vehicle_number) < 8:
        return []

    letter_subs = {
        "O": ["D", "Q", "C", "0"], "D": ["O", "Q", "0"], "Q": ["O", "D", "0"],
        "I": ["L", "1", "T"], "L": ["I", "1"], "S": ["5", "8"], "B": ["8", "R"],
        "G": ["6", "C"], "E": ["3", "F"], "Z": ["2", "7"], "C": ["G", "O"],
        "T": ["I", "7", "1"], "A": ["4", "H"], "R": ["8", "B"],
    }
    digit_subs = {
        "0": ["O", "Q", "D"], "1": ["I", "L", "7"], "2": ["Z", "7"], "3": ["E", "8"],
        "4": ["A", "9", "1"], "5": ["S", "6"], "6": ["G", "8", "5"], "7": ["1", "T", "2"],
        "8": ["B", "3", "6"], "9": ["P", "4"],
    }

    up = vehicle_number.upper()
    plate_len = len(up)
    variations: list[str] = []
    seen = {up}

    def expected_type(pos: int) -> str:
        if pos < 2:
            return "letter"
        if pos < 4:
            return "digit"
        if pos < plate_len - 4:
            return "letter"
        return "digit"

    # high-priority O/D confusion in series positions
    high = {"O": "D", "D": "O", "0": "D"}
    for i in range(4, plate_len - 4):
        ch = up[i]
        if ch in high:
            v = up[:i] + high[ch] + up[i + 1:]
            if v not in seen:
                variations.append(v); seen.add(v)
                if len(variations) >= max_variations:
                    return variations

    scored = []
    for i, ch in enumerate(up):
        exp = expected_type(i)
        mism = (exp == "letter" and ch.isdigit()) or (exp == "digit" and ch.isalpha())
        subs = letter_subs if exp == "letter" else digit_subs
        if mism or ch in subs:
            scored.append((i, ch, exp, 2 if mism else 1))
    scored.sort(key=lambda x: -x[3])

    for pos, ch, exp, _ in scored:
        subs = letter_subs if exp == "letter" else digit_subs
        for sub in subs.get(ch, []):
            v = up[:pos] + sub + up[pos + 1:]
            if v in seen:
                continue
            if pos < 2 and v[:2] not in VALID_STATE_CODES:
                continue
            variations.append(v); seen.add(v)
            if len(variations) >= max_variations:
                return variations
    return variations
