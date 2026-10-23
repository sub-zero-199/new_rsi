"""The plate pipeline contract + a GPU-free MockEngine + the engine factory.

Contract (3 stages, mirroring the trained models):
    detect(frame)  -> [Bbox]                 # m1 plate detection
    rectify(crop)  -> warped image | None    # m2 4-corner keypoints + perspective warp
    read(warped)   -> PlateCandidate | None  # m3 char OCR -> text + confidences

`process(frame)` chains them. Swap the engine without touching the rest of the system.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Bbox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float = 1.0

    def clip(self, w: int, h: int) -> "Bbox":
        return Bbox(max(0, int(self.x1)), max(0, int(self.y1)),
                    min(w, int(self.x2)), min(h, int(self.y2)), self.conf)

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass
class PlateCandidate:
    text: str                       # raw OCR string (pre-normalization)
    mean_conf: float
    bbox: Bbox
    plate_img: np.ndarray           # rectified plate crop (saved as plate.jpg)
    char_confs: list[float] = field(default_factory=list)
    # Spellings the DETECTOR itself considered for this crop, best first (see
    # ocr.spelling_candidates). Where it could not settle between two labels for one
    # glyph, the alternative is here — a far better lead than a blind look-alike swap.
    spellings: list[str] = field(default_factory=list)


class PlatePipeline(abc.ABC):
    """Abstract 3-stage pipeline. Implementations override detect/rectify/read."""

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> list[Bbox]:
        ...

    @abc.abstractmethod
    def rectify(self, crop: np.ndarray) -> Optional[np.ndarray]:
        ...

    @abc.abstractmethod
    def read(self, warped: np.ndarray) -> Optional[PlateCandidate]:
        ...

    def process(self, frame: np.ndarray) -> list[PlateCandidate]:
        """Run all 3 stages over a frame; return one candidate per readable plate."""
        out: list[PlateCandidate] = []
        if frame is None or getattr(frame, "size", 0) == 0:
            return out
        h, w = frame.shape[:2]
        dets = []  # (box, (ox,oy), corners, cand, crop, warped) for the debug overlay
        min_w = w * getattr(self, "min_plate_width_frac", 0.0)
        for box in self.detect(frame):
            box = box.clip(w, h)
            if box.area < 100:
                continue
            # Too far away to read. The `area` guard above only rejects degenerate boxes
            # (~17px wide); this is the readability gate. Measured on 62 real frames, NO
            # read below ~4.2% of frame width ever produced a valid plate, while 85% did
            # above it — so these frames only ever contributed noise and spurious windows.
            if min_w and (box.x2 - box.x1) < min_w:
                continue
            # Pad the detect box by 5px (clamped to the frame) before the keypoint model, so
            # plate corners sitting right at the tight detect edge aren't clipped.
            px1, py1 = max(0, box.x1 - 5), max(0, box.y1 - 5)
            px2, py2 = min(w, box.x2 + 5), min(h, box.y2 + 5)
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                continue
            warped = self.rectify(crop)
            corners = getattr(self, "_last_corners", None)  # keypoint corners (padded-crop coords)
            cand = self.read(warped) if (warped is not None and warped.size) else None
            dets.append((box, (px1, py1), corners, cand, crop, warped))
            if cand is None or not cand.text:
                continue
            cand.bbox = box
            out.append(cand)
        self._debug_save(frame, dets)  # testing hook (no-op unless debug_dir set)
        return out

    def _debug_save(self, frame, dets) -> None:
        """Testing: save debug images. No-op unless `debug_dir` is set (build_pipeline sets
        it from save_rectified). Writes two things per frame:

        1. `debug_dir` (debug/rectified): ONE annotated full-frame image with the detect box
           (green) and the 4 keypoint corners drawn as 4 distinctly-colored dots.
        2. `crops_dir` (debug/crops): per detection, the raw detect cutout (the plate cut from
           the full image) as `*_detect.jpg`, and the keypoint-rectified plate that gets sent
           to OCR (`*_keypoint.jpg`), so both stages' inputs can be eyeballed side by side.
        """
        d = getattr(self, "debug_dir", None)
        if not d or not dets or getattr(self, "_debug_saved", 0) >= getattr(self, "debug_limit", 500):
            return
        try:
            import os
            import time
            import cv2

            # One BGR color per keypoint corner (drawn in the model's corner order):
            #   0 blue, 1 green, 2 red, 3 yellow.
            corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
            crops_dir = getattr(self, "crops_dir", None)

            os.makedirs(d, exist_ok=True)
            if crops_dir:
                os.makedirs(crops_dir, exist_ok=True)
            img = frame.copy()
            label = "NOREAD"
            ms = int(time.time() * 1000)
            for i_det, (box, (ox, oy), corners, cand, crop, warped) in enumerate(dets):
                cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
                txt = cand.text if (cand and cand.text) else "NOREAD"
                if cand and cand.text and label == "NOREAD":
                    label = cand.text
                cv2.putText(img, txt, (box.x1, max(12, box.y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # corners are in the PADDED crop's coords -> map with the padded origin (ox,oy)
                if corners is not None and len(corners) == 4:
                    for i, (cx, cy) in enumerate(corners):
                        px, py = int(cx) + ox, int(cy) + oy
                        cv2.circle(img, (px, py), 5, corner_colors[i], -1)
                if crops_dir:
                    tag = txt if txt != "NOREAD" else "NOREAD"
                    stem = f"{ms}_{i_det}_{tag}"
                    if crop is not None and getattr(crop, "size", 0):
                        cv2.imwrite(os.path.join(crops_dir, f"{stem}_detect.jpg"), crop)
                    if warped is not None and getattr(warped, "size", 0):
                        cv2.imwrite(os.path.join(crops_dir, f"{stem}_keypoint.jpg"), warped)
            cv2.imwrite(os.path.join(d, f"{ms}_{label}.jpg"), img)
            self._debug_saved = getattr(self, "_debug_saved", 0) + 1
        except Exception:
            pass

    def warmup(self) -> None:
        """Optional: run a dummy inference so first real frame isn't slow."""
        return None


class MockEngine(PlatePipeline):
    """GPU-free engine for development/testing.

    Emits a configurable sequence of plates so the downstream pipeline (aggregation,
    matching, journey, portal, S3) can be exercised deterministically. Each plate is
    emitted for `frames_per_plate` consecutive frames (so the detection window reaches
    its min-detections threshold), then it advances to the next.
    """

    def __init__(self, plates: Optional[list[str]] = None, frames_per_plate: int = 5,
                 conf: float = 0.92):
        self.plates = plates or ["MH12AB1234"]
        self.frames_per_plate = max(1, frames_per_plate)
        self.conf = conf
        self._n = 0

    @property
    def current_plate(self) -> str:
        idx = (self._n // self.frames_per_plate) % len(self.plates)
        return self.plates[idx]

    def detect(self, frame: np.ndarray) -> list[Bbox]:
        h, w = frame.shape[:2]
        # a centered "plate" region ~ half width, fifth height
        bw, bh = w // 2, max(20, h // 5)
        x1, y1 = (w - bw) // 2, (h - bh) // 2
        return [Bbox(x1, y1, x1 + bw, y1 + bh, self.conf)]

    def rectify(self, crop: np.ndarray) -> Optional[np.ndarray]:
        return crop  # mock plates are already "flat"

    def read(self, warped: np.ndarray) -> Optional[PlateCandidate]:
        plate = self.current_plate
        self._n += 1
        return PlateCandidate(
            text=plate,
            mean_conf=self.conf,
            bbox=Bbox(0, 0, warped.shape[1], warped.shape[0], self.conf),
            plate_img=warped,
            char_confs=[self.conf] * len(plate),
        )


def build_pipeline(cfg) -> PlatePipeline:
    """Factory: returns MockEngine or the real UltralyticsEngine per inference.engine.

    The Ultralytics import is deferred to here so `mock` runs with zero ML deps.
    """
    inf = cfg.inference
    engine = getattr(inf, "engine", "mock")
    if engine == "mock":
        pipe: PlatePipeline = MockEngine()
    elif engine == "ultralytics":
        from anpr.models.engine import UltralyticsEngine

        pipe = UltralyticsEngine(inf)
    else:
        raise ValueError(f"unknown inference.engine: {engine!r}")

    if getattr(inf, "save_rectified", False):
        import os

        pipe.debug_dir = inf.rectified_dir
        pipe.debug_limit = inf.rectified_limit
        pipe._debug_saved = 0
        # detect cutouts + keypoint-rectified plates -> a `crops/` sibling of the rectified dir
        pipe.crops_dir = os.path.join(os.path.dirname(inf.rectified_dir) or ".", "crops")

    pipe.min_plate_width_frac = getattr(cfg.detection, "min_plate_width_frac", 0.0)
    return pipe
