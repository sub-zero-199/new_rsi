"""Detection-window aggregation: collapse many reads of one vehicle into a single
decision (the "best read over the ~window" requirement).

Ported/adapted from v11_1 process_vehicle_number + get_best_detection_from_window. Reads
are grouped per (camera, canonical-plate) window; when a window's time elapses with enough
detections, we emit the top-N candidate plate strings (scored by confidence + canonical
shape + recency) padded with OCR variations. Cooldown/dedup live downstream, not here.

Uses v11_1's canonical-window grouping instead of a separate ByteTrack model — same
"one decision per vehicle" effect at much lower GPU cost (matters on a 4 GB card).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from anpr.normalize import get_canonical_plate
from anpr.similarity import calculate_plate_similarity
from anpr.models.pipeline import PlateCandidate
from anpr.log import logger


@dataclass
class _Read:
    text: str
    canonical: str
    mean_conf: float
    plate_img: np.ndarray
    full_frame: np.ndarray
    ts: float
    spellings: list = field(default_factory=list)   # what the detector itself considered


@dataclass
class _Window:
    point: str
    start_time: float
    reads: list[_Read] = field(default_factory=list)


@dataclass
class FinalizedDetection:
    camera_id: str
    point: str
    best_text: str                 # highest-scoring raw plate text
    candidates: list[str]          # top-N candidate texts (for sequential matching)
    plate_img: np.ndarray          # best read's rectified plate crop
    full_frame: np.ndarray
    mean_conf: float
    ts: int
    detection_count: int


class WindowAggregator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fuzzy_threshold = cfg.matching.fuzzy_threshold
        # windows keyed by (camera_id, canonical)
        self._windows: dict[tuple[str, str], _Window] = {}

    def _window_seconds(self, point: str) -> float:
        return self.cfg.detection_for(point).window_seconds

    def _find_key(self, camera_id: str, canonical: str) -> Optional[tuple[str, str]]:
        exact = (camera_id, canonical)
        if exact in self._windows:
            return exact
        for (cam, key) in self._windows:
            if cam == camera_id and calculate_plate_similarity(canonical, key) >= self.fuzzy_threshold:
                return (cam, key)
        return None

    def add(self, camera_id: str, point: str, cand: PlateCandidate,
            full_frame: np.ndarray, ts: Optional[float] = None) -> None:
        ts = ts if ts is not None else time.time()
        canonical = get_canonical_plate(cand.text)
        if not canonical:
            return
        read = _Read(cand.text, canonical, cand.mean_conf, cand.plate_img, full_frame, ts,
                     list(getattr(cand, 'spellings', []) or []))
        key = self._find_key(camera_id, canonical)
        if key is None:
            self._windows[(camera_id, canonical)] = _Window(point=point, start_time=ts, reads=[read])
            return
        win = self._windows[key]
        win.reads.append(read)
        cap = self.cfg.detection_for(point).max_detections_per_window
        if len(win.reads) > cap:  # keep most recent (vehicle closer = better)
            win.reads = sorted(win.reads, key=lambda r: r.ts, reverse=True)[:cap]

    def sweep(self, now: Optional[float] = None) -> list[FinalizedDetection]:
        """Return finalized detections for windows whose time has elapsed."""
        now = now if now is not None else time.time()
        out: list[FinalizedDetection] = []
        for key in [k for k, w in self._windows.items() if now - w.start_time > self._window_seconds(w.point)]:
            win = self._windows.pop(key)
            det = self._finalize(key[0], win)
            if det is not None:
                out.append(det)
        return out

    def drop_point(self, point: str) -> int:
        """Discard every in-flight window for a point; returns how many were dropped.

        Used when a point stops detecting: a half-collected window must not finalize later
        as though it were a complete read of the vehicle.
        """
        keys = [k for k, w in self._windows.items() if w.point == point]
        for k in keys:
            self._windows.pop(k, None)
        return len(keys)

    def flush_all(self) -> list[FinalizedDetection]:
        out = []
        for key in list(self._windows):
            win = self._windows.pop(key)
            d = self._finalize(key[0], win)
            if d:
                out.append(d)
        return out

    def _finalize(self, camera_id: str, win: _Window) -> Optional[FinalizedDetection]:
        dconf = self.cfg.detection_for(win.point)
        if len(win.reads) < dconf.min_detections:
            logger.debug(f"window {camera_id}/{win.point} dropped: "
                         f"{len(win.reads)} reads < min {dconf.min_detections}")
            return None
        top = self._top_reads(win, dconf.top_n_candidates)
        if not top:
            return None
        # Best read first, then the spellings the DETECTOR itself was torn between for that
        # crop — where it could not settle between two labels on one glyph, the true plate is
        # usually the runner-up. Only then other distinct reads from the window. Model doubts
        # beat blind look-alike swaps: they are evidence, not invention.
        best = top[0]
        candidates: list[str] = []
        for text in list(best.spellings) + [r.text for r in top]:
            if text and text not in candidates:
                candidates.append(text)
        candidates = candidates[:dconf.top_n_candidates]
        return FinalizedDetection(
            camera_id=camera_id, point=win.point, best_text=best.text,
            candidates=candidates, plate_img=best.plate_img, full_frame=best.full_frame,
            mean_conf=best.mean_conf, ts=int(best.ts), detection_count=len(win.reads),
        )

    @staticmethod
    def _top_reads(win: _Window, top_n: int) -> list[_Read]:
        duration = max(1e-6, max(r.ts for r in win.reads) - win.start_time)
        scored = []
        for r in win.reads:
            score = r.mean_conf
            if len(r.text) >= 9:
                score += 0.1
            if r.text[:2].isalpha():
                score += 0.1
            if r.text[-4:].isdigit():
                score += 0.1
            recency = ((r.ts - win.start_time) / duration) ** 2 * 0.30  # later = closer = better
            scored.append((r, score + recency))
        scored.sort(key=lambda x: x[1], reverse=True)
        out, seen = [], set()
        for r, _ in scored:
            k = r.text.upper().strip()
            if k not in seen:
                seen.add(k)
                out.append(r)
                if len(out) >= top_n:
                    break
        return out
