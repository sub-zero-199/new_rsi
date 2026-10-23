"""Resolve a finalized detection to a final plate string.

Implements the agreed flow:
  1. clean/normalize the OCR text; apply doppelganger correction; flag whitelist.
  2. Match in local known_plates (SQLite) at >= fuzzy_threshold.
  3. If matched -> snap to that plate (correction). If not matched -> it's new.

Matching here is for plate **correction**.
The push/skip decision itself is made later by the in/out dedup in journey.PointProcessor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from anpr.log import logger
from anpr.normalize import clean_and_correct_text, normalize_raw, get_canonical_plate
from anpr.similarity import best_match
from anpr.aggregation import FinalizedDetection


@dataclass
class MatchResult:
    resolved: str            # final plate to use (snapped/corrected)
    original: str            # raw best OCR text
    matched: bool            # found in local known_plates
    source: Optional[str]    # 'local' | None
    score: float
    whitelisted: bool
    valid: bool              # passed cleanup/format (or whitelisted)


class PlateMatcher:
    def __init__(self, cfg, store, whitelist, doppelganger):
        self.cfg = cfg
        self.store = store
        self.whitelist = whitelist
        self.doppelganger = doppelganger
        self.threshold = cfg.matching.fuzzy_threshold

    def _clean(self, text: str) -> tuple[str, bool]:
        """Return (plate, valid). Whitelisted plates bypass strict cleanup."""
        raw = normalize_raw(text)
        if self.whitelist.is_whitelisted(raw):
            return raw, True
        cleaned = clean_and_correct_text(text)
        if cleaned:
            return cleaned, True
        # keep the raw normalized form but mark invalid (callers may still match it)
        return raw, False

    def resolve(self, det: FinalizedDetection) -> MatchResult:
        plate, valid = self._clean(det.best_text)
        plate = self.doppelganger.check_and_replace(plate)
        whitelisted = self.whitelist.is_whitelisted(plate)

        local_pool = self.store.all_known_plates()
        cand, score = best_match(plate, local_pool, self.threshold)
        if cand:
            logger.info(f"Match(local): {plate} -> {cand} ({score:.2f})")
            return MatchResult(cand, det.best_text, True, "local", score, whitelisted, valid)

        return MatchResult(plate, det.best_text, False, None, 0.0, whitelisted, valid)
