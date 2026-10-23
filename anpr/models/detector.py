"""Stage-1 helpers: bbox padding around a detected plate before rectify/OCR.
Pure NumPy — no torch.
"""
from __future__ import annotations

from anpr.models.pipeline import Bbox


def pad_bbox(box: Bbox, w: int, h: int, ratio: float = 0.05) -> Bbox:
    """Expand a bbox by `ratio` of its own size on each side, clipped to the frame.
    A little context around the plate helps the keypoint/OCR stages.
    """
    bw = box.x2 - box.x1
    bh = box.y2 - box.y1
    px = int(bw * ratio)
    py = int(bh * ratio)
    return Bbox(box.x1 - px, box.y1 - py, box.x2 + px, box.y2 + py, box.conf).clip(w, h)
