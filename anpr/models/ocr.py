"""Stage-3 assembly: turn YOLO character detections into a plate string.

Clusters character boxes into rows by vertical center (handles 1- and 2-row Indian
plates), suppresses boxes describing the SAME glyph, then orders rows top->bottom and
characters left->right. Pure logic — no torch — so it's unit-testable.

The suppression matters: the detector routinely emits TWO labels for one character when
it cannot decide, and they were being concatenated. Measured on 303 real plate crops,
46% of reads contained such a pair — e.g. a Z read as both 'A' (0.69) and 'Z' (0.55)
0.1px apart, turning Z8089 into AZ8089. Genuine neighbouring characters sit ~30px apart
(0.6+ of character height) and duplicates below 0.05, with nothing in between, so the
0.15 threshold has wide margin on both sides.

The losing label is not thrown away: it is the model's own second opinion, which makes a
far better candidate than blindly swapping look-alike characters.
"""
from __future__ import annotations

from dataclasses import dataclass


# Two boxes are the same glyph when their centres are closer than this fraction of the
# median character height. Duplicates measure <0.05, real neighbours >=0.30.
SAME_GLYPH_FRAC = 0.15
# A suppressed label is only worth offering as an alternative spelling if the model rated
# it at least this fraction of the winner. 'A' 0.69 vs 'Z' 0.55 is a real coin-flip (0.80);
# '8' 0.94 vs '0' 0.12 is noise (0.13).
ALT_MIN_RATIO = 0.60


@dataclass
class CharBox:
    label: str          # '0'-'9' or 'A'-'Z'
    x_center: float
    y_center: float
    conf: float
    height: float = 0.0


@dataclass
class CharSlot:
    """One physical character: the winning box, plus any labels it beat at that position."""
    best: CharBox
    alternatives: list[CharBox]

    @property
    def credible_alternatives(self) -> list[CharBox]:
        """The runner-ups the model rated close enough to be a genuine doubt."""
        floor = self.best.conf * ALT_MIN_RATIO
        return [c for c in self.alternatives if c.conf >= floor]


def cluster_rows(chars: list[CharBox]) -> list[list[CharBox]]:
    """Group characters into rows by y-center. Two chars are in the same row if their
    y-centers differ by less than ~60% of the median character height.
    """
    if not chars:
        return []
    heights = sorted(c.height for c in chars if c.height > 0)
    med_h = heights[len(heights) // 2] if heights else 0.0
    thresh = (med_h * 0.6) if med_h > 0 else 1e9  # if no heights, single row

    rows: list[list[CharBox]] = []
    for c in sorted(chars, key=lambda c: c.y_center):
        placed = False
        for row in rows:
            row_y = sum(rc.y_center for rc in row) / len(row)
            if abs(c.y_center - row_y) <= thresh:
                row.append(c)
                placed = True
                break
        if not placed:
            rows.append([c])
    rows.sort(key=lambda row: sum(rc.y_center for rc in row) / len(row))
    return rows


def resolve_slots(chars: list[CharBox]) -> list[CharSlot]:
    """One CharSlot per physical character, in reading order (rows top->bottom, left->right).

    Boxes whose centres fall within SAME_GLYPH_FRAC of the median character height are the
    same glyph seen twice; the highest-confidence label wins and the others are kept as
    alternatives rather than being concatenated into the plate.
    """
    if not chars:
        return []
    heights = sorted(c.height for c in chars if c.height > 0)
    med_h = heights[len(heights) // 2] if heights else 0.0
    thresh = med_h * SAME_GLYPH_FRAC if med_h > 0 else 0.0

    slots: list[CharSlot] = []
    for row in cluster_rows(chars):
        groups: list[list[CharBox]] = []
        for c in sorted(row, key=lambda c: c.x_center):
            for g in groups:
                if abs(c.x_center - g[0].x_center) <= thresh:
                    g.append(c)
                    break
            else:
                groups.append([c])
        for g in groups:
            g.sort(key=lambda c: c.conf, reverse=True)
            slots.append(CharSlot(best=g[0], alternatives=g[1:]))
    return slots


def assemble_plate(chars: list[CharBox]) -> tuple[str, list[float]]:
    """Return (plate_text, per_char_confidences) ordered rows top->bottom, left->right."""
    slots = resolve_slots(chars)
    return "".join(s.best.label for s in slots), [s.best.conf for s in slots]


def spelling_candidates(chars: list[CharBox], limit: int = 4) -> list[str]:
    """Plate spellings the model itself considered, best first.

    Every credible alternative is a position the detector could not settle, so the true
    plate is usually one of these combinations. Ordered by total confidence lost, so the
    closest coin-flips come first and the caller can stop at the first the registry knows.

    Far better than swapping look-alike characters blindly: on a real read of GJ14Z8089
    the detector reported 'A'(0.69)/'Z'(0.55) at one position, so the correct plate is the
    second entry here — while the blind generator never produced it at all.
    """
    slots = resolve_slots(chars)
    if not slots:
        return []
    # (label, confidence-lost) options per position, winner first
    options = [[(s.best.label, 0.0)]
               + [(a.label, s.best.conf - a.conf) for a in s.credible_alternatives]
               for s in slots]

    combos = [("", 0.0)]
    for opts in options:
        combos = [(text + lab, cost + c) for text, cost in combos for lab, c in opts]
        if len(combos) > 4096:      # pathological read: keep the cheapest branches only
            combos = sorted(combos, key=lambda t: t[1])[:512]
    combos.sort(key=lambda t: t[1])

    out: list[str] = []
    for text, _ in combos:
        if text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out
