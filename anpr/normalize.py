"""Indian license-plate normalization, OCR-error correction, and validation.

Ported from v11_1 src/extract_number.py (ExtractVehicleNumber.*) and
src/send_vehicle_number.py (get_canonical_plate). Kept as pure functions — no OCR engine,
no torch — so the new YOLO OCR stage just feeds its raw string in here.
"""
from __future__ import annotations

import re

VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "CT", "DL", "DN", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB", "JL",
}

OCR_DIGIT_CORRECTIONS = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "|": "1", "!": "1",
    "Z": "2", "z": "2", "E": "3", "A": "4", "h": "4",
    "S": "5", "s": "5", "G": "6", "b": "6", "T": "7", "t": "7", "B": "8", "R": "8",
}

OCR_LETTER_CORRECTIONS = {
    "0": "O", "6": "G", "8": "B", "1": "I", "5": "S", "2": "Z",
    "3": "E", "4": "A", "7": "T", "9": "P",
}

STATE_CODE_CORRECTIONS = {
    "0L": "DL", "DI": "DL", "D1": "DL", "0D": "OD",
    "M4": "MH", "MN": "MH", "K4": "KA", "KH": "KA",
    "G1": "GJ", "6J": "GJ", "GI": "GJ", "GG": "GJ",
    "U4": "UP", "WR": "WB", "T5": "TS", "R1": "RJ",
    "H4": "HR", "P8": "PB", "GE": "GJ",
}

_VALID_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

_FORMATS = [
    r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$",
    r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$",
    r"^[A-Z]{2}TEMP\d{4}$",
    r"^\d{2}BH\d{4}[A-HJ-NP-Z]{1,2}$",
]


def normalize_raw(text: str) -> str:
    """Uppercase and strip spaces/dashes/dots — the lightweight normalization."""
    if not text:
        return ""
    return text.upper().replace(" ", "").replace("-", "").replace(".", "")


def is_bh_series(text: str) -> bool:
    return len(text) >= 4 and text[2:4].upper() == "BH" and text[0:2].isdigit()


def validate_plate_format(text: str) -> bool:
    if not text or len(text) < 9 or len(text) > 11:
        return False
    return any(re.match(p, text) for p in _FORMATS)


def apply_position_based_corrections(text: str) -> str:
    if len(text) < 6:
        return text
    text_upper = text.upper()
    if is_bh_series(text_upper):
        return text_upper  # BH series: leave digits alone

    result = list(text_upper)
    # state code (0-1) -> letters
    for i in range(min(2, len(result))):
        if result[i] in OCR_LETTER_CORRECTIONS:
            result[i] = OCR_LETTER_CORRECTIONS[result[i]]

    # district code boundary detection
    district_end = 2
    for i in range(2, min(5, len(result))):
        if result[i].isalpha() and i > 2:
            district_end = i
            break
        if i == 4:
            district_end = 4
            break
    for i in range(2, district_end):
        if i < len(result) and result[i] in OCR_DIGIT_CORRECTIONS:
            result[i] = OCR_DIGIT_CORRECTIONS[result[i]]

    # series letters -> letters
    for i in range(district_end, len(result) - 4):
        if i < len(result) and result[i] in OCR_LETTER_CORRECTIONS:
            result[i] = OCR_LETTER_CORRECTIONS[result[i]]

    # last 4 -> digits
    for i in range(max(0, len(result) - 4), len(result)):
        if result[i] in OCR_DIGIT_CORRECTIONS:
            result[i] = OCR_DIGIT_CORRECTIONS[result[i]]

    return "".join(result)


def _char_similarity(s1: str, s2: str) -> float:
    if len(s1) != len(s2):
        return 0.0
    return sum(1 for a, b in zip(s1, s2) if a == b) / len(s1)


def correct_state_code(text: str) -> str:
    if len(text) < 2:
        return text
    sc = text[:2]
    if sc in STATE_CODE_CORRECTIONS:
        return STATE_CODE_CORRECTIONS[sc] + text[2:]
    corrected = "".join(OCR_LETTER_CORRECTIONS.get(c, c) for c in sc)
    if corrected in VALID_STATE_CODES:
        return corrected + text[2:]
    for valid in VALID_STATE_CODES:
        if _char_similarity(sc, valid) > 0.5:
            return valid + text[2:]
    return text


def detect_and_fix_plate_alignment(text: str) -> str:
    if len(text) < 9:
        return text
    if text[:2] in VALID_STATE_CODES:
        return text
    if len(text) > 9 and text[1:3] in VALID_STATE_CODES:
        return text[1:]
    if len(text) > 2 and text[1:3] in VALID_STATE_CODES:
        return text[1:]
    return text


# Loose pre-filter around the structure gate. 8 is the shortest real plate (HP240751);
# 13 leaves room for a couple of stray characters that alignment fixing can still strip.
_MIN_RAW_LEN = 8
_MAX_RAW_LEN = 13

# Every shape a real Indian plate can take. The structure IS the length rule: matching one
# of these fixes the length at 8, 9 or 10, so no separate length check is needed after this.
#   state(2 letters, a real code) district(1-2 digits) series(0-3 letters) number(4 digits)
# Delhi uses 3 series letters (DL1CAJ1993), so 11 characters is a legitimate plate.
_PLATE_PATTERNS = (
    r"([A-Z]{2})(\d{2})([A-Z]{3})(\d{4})",          # DL10CAJ1993 (11) — 3-letter series
    r"([A-Z]{2})(\d{1})([A-Z]{3})(\d{4})",          # DL1CAJ1993  (10) — 3-letter series
    r"([A-Z]{2})(\d{2})([A-Z]{2})(\d{4})",          # GJ12BY2455  (10)
    r"([A-Z]{2})(\d{2})([A-Z]{1})(\d{4})",          # GJ12B2455   (9)
    r"([A-Z]{2})(\d{1})([A-Z]{2})(\d{4})",          # GJ1BY2455   (9)
    r"(\d{2})(BH)(\d{4})([A-HJ-NP-Z]{1,2})",        # 22BH1234A   (9-10)
    r"([A-Z]{2})(TEMP)(\d{4})",                     # GJTEMP1234  (10)
    r"([A-Z]{2})(\d{2})(\d{4})",                    # HP240751    (8) — no series letters
)


def match_plate_patterns(text: str) -> str | None:
    """The plate's structure, or None. A FULL match — not a substring hunt.

    `re.search` used to accept any string that merely CONTAINED a plate, so a stray
    character rode along ('AGJ39TA2660' matched on the GJ39TA2660 inside it). Alignment
    fixing already strips leading junk before we get here, so anything still carrying extra
    characters is noise, not a plate.
    """
    for pat in _PLATE_PATTERNS:
        m = re.fullmatch(pat, text)
        if m:
            g = m.groups()
            if g[0] in VALID_STATE_CODES or g[0].isdigit():
                return "".join(g)
    return None


def clean_and_correct_text(text: str) -> str:
    """Full cleanup chain for a raw OCR string -> a best-effort canonical plate, or ''."""
    if not text:
        return ""
    text = normalize_raw(text)
    text = "".join(c for c in text if c in _VALID_CHARS)
    # Loose window only: wide enough that a read carrying a stray character still reaches
    # the repair steps below. The structure gate at the end is what actually decides.
    if len(text) < _MIN_RAW_LEN or len(text) > _MAX_RAW_LEN:
        return ""
    text = detect_and_fix_plate_alignment(text)
    text = apply_position_based_corrections(text)
    text = correct_state_code(text)
    if not is_bh_series(text) and len(text) >= 2 and text[:2] not in VALID_STATE_CODES:
        return ""
    # Structure gate. Previously this was `match_plate_patterns(text) or text` — the
    # fallback meant a string that matched NO plate shape was still accepted as long as it
    # started with a real state code, which is how CGJ1ZBM9353 and GAJ1ZBW866 became trips.
    return match_plate_patterns(text) or ""


def get_canonical_plate(plate: str) -> str:
    """Canonical form for similarity/grouping. Ported from send_vehicle_number.get_canonical_plate.

    Indian format ST##AA#### : state(letters) district(digits) series(letters) number(digits).
    Applies position-aware OCR corrections conservatively.
    """
    if not plate:
        return ""
    clean = re.sub(r"[^A-Z0-9]", "", plate.upper())
    if len(clean) < 8:
        return clean

    bh = re.match(r"^(\d{2})(BH)(\d{4})([A-Z]{1,2})$", clean)
    if bh:
        return clean

    letter_to_number = {"O": "0", "I": "1", "S": "5", "Z": "2", "B": "8", "G": "6"}
    number_to_letter = {"0": "O", "1": "I", "5": "S", "8": "B", "6": "G"}

    if len(clean) < 9:
        return clean

    state_part, district_part, remaining = clean[:2], clean[2:4], clean[4:]
    if len(remaining) >= 4:
        number_part, series_part = remaining[-4:], remaining[:-4]
    else:
        number_part, series_part = remaining, ""

    norm_state = "".join(number_to_letter.get(c, c) if c.isdigit() else c for c in state_part)
    norm_district = "".join(letter_to_number.get(c, c) if c.isalpha() else c for c in district_part)

    norm_series = ""
    for c in series_part:
        if c.isdigit() and c in number_to_letter and len(series_part) <= 3:
            if c == "0" and len(series_part) <= 2:
                norm_series += number_to_letter[c]
            elif c == "1" and series_part.count("1") == 1 and len(series_part) <= 2:
                norm_series += number_to_letter[c]
            else:
                norm_series += c
        else:
            norm_series += c

    norm_number = "".join(letter_to_number.get(c, c) if c.isalpha() else c for c in number_part)

    canonical = norm_state + norm_district + norm_series + norm_number
    if len(canonical) >= 8 and not canonical[:2].isalpha():
        fixed = "".join(number_to_letter.get(c, c) if c.isdigit() else c for c in canonical[:2])
        canonical = fixed + canonical[2:]
    return canonical


def is_valid_state_code(plate: str) -> bool:
    return bool(plate) and len(plate) >= 2 and plate[:2].upper() in VALID_STATE_CODES
