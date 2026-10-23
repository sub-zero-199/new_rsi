"""Stage-2 geometry: order the 4 plate-corner keypoints and perspective-warp the crop
to a flat rectangle ("stretch fit"). Pure OpenCV/NumPy — no torch — so it's unit-testable
and shared by the real engine.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left].

    Uses the standard sum/diff trick: TL has min(x+y), BR has max(x+y),
    TR has min(y-x), BL has max(y-x).
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError(f"expected 4 corners, got {pts.shape[0]}")
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def warp_plate(image: np.ndarray, corners: np.ndarray,
               out_size: tuple[int, int] = (192, 96),
               pad_color: tuple[int, int, int] = (255, 255, 255)) -> Optional[np.ndarray]:
    """Perspective-warp the quadrilateral given by `corners` (in `image` coords) to a flat
    plate, preserving its TRUE aspect ratio, then center it on an `out_size` (w, h) canvas
    padded with `pad_color` (white by default).

    Unlike a plain stretch-fit (which forces the plate into out_size and distorts plates
    whose aspect differs), this measures the plate's natural width/height from the corner
    edge lengths, warps to that size, then scales-to-fit + letterbox-pads with white. This
    keeps characters undistorted before OCR (and in the S3 plate.jpg).
    """
    if image is None or image.size == 0:
        return None
    try:
        src = order_corners(corners)  # TL, TR, BR, BL
    except ValueError:
        return None
    tl, tr, br, bl = src
    # Natural plate size = the longer of each opposing edge pair.
    nat_w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    nat_h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if nat_w < 1 or nat_h < 1:
        return None
    dst = np.array([[0, 0], [nat_w - 1, 0], [nat_w - 1, nat_h - 1], [0, nat_h - 1]],
                   dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, m, (nat_w, nat_h))  # aspect preserved, no stretch

    # Scale-to-fit inside out_size and center on a white canvas (letterbox with pad_color).
    W, H = out_size
    scale = min(W / nat_w, H / nat_h)
    sw, sh = max(1, int(round(nat_w * scale))), max(1, int(round(nat_h * scale)))
    resized = cv2.resize(warped, (sw, sh), interpolation=cv2.INTER_AREA)
    canvas = np.full((H, W, 3), pad_color, dtype=np.uint8)
    x0, y0 = (W - sw) // 2, (H - sh) // 2
    canvas[y0:y0 + sh, x0:x0 + sw] = resized
    return canvas
