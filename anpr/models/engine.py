"""Real 3-model YOLOv11m engine (m1 detect -> m2 keypoint warp -> m3 char OCR).

Ultralytics/torch are imported lazily in __init__, so the rest of the system runs on a
GPU-free box with the MockEngine. One shared instance of each model is held and every
GPU call is serialized behind a lock (the 4 GB GPU runs one inference at a time anyway).

Weights + their I/O specs are an open item; this wires against the standard Ultralytics
result shapes (boxes.xyxy/conf/cls, keypoints.xy, model.names) and degrades gracefully
(falls back to a plain resize if the keypoint model yields no 4 corners).
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

from anpr.log import logger
from anpr.models.pipeline import Bbox, PlateCandidate, PlatePipeline
from anpr.models.rectifier import warp_plate
from anpr.models.ocr import CharBox, assemble_plate, spelling_candidates


def _load_model(weights: str, device: str, use_tensorrt: bool = False, task: str | None = None):
    """Load a YOLO model.

    With use_tensorrt (the deployed default), load ONLY the TensorRT `.engine` sibling and
    fail loudly if it's missing — the `.pt` is never used for inference. Without it, load
    the `.pt` directly.

    `task` must be passed for a TensorRT `.engine` whose task can't be inferred from the file
    (a `.pt` carries its task; the exported `.engine` does not). The keypoint/rectifier model
    is a pose model, so it needs task="pose" — otherwise Ultralytics assumes task=detect and
    never populates result.keypoints (the corners come back empty).
    """
    from ultralytics import YOLO

    if use_tensorrt:
        path = os.path.splitext(weights)[0] + ".engine"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"TensorRT engine not found: {path}. Build it first with "
                f"./tools/build_engines.sh (inference.use_tensorrt is true, so the .pt is "
                f"not used for inference)."
            )
    else:
        path = weights
    model = YOLO(path, task=task) if task else YOLO(path)
    try:
        model.to(device)
    except Exception:
        pass
    logger.info(f"Loaded model {path} on {device} (task={task or 'auto'})")
    return model


def export_tensorrt(weights: str, imgsz: int, half: bool = True) -> str:
    """Export a .pt to a TensorRT .engine (run once on the target GPU). Returns the path."""
    from ultralytics import YOLO

    logger.info(f"Exporting {weights} -> TensorRT (imgsz={imgsz}, half={half})")
    return YOLO(weights).export(format="engine", imgsz=imgsz, half=half)


class UltralyticsEngine(PlatePipeline):
    def __init__(self, inf_cfg):
        self.cfg = inf_cfg
        self._lock = threading.Lock()
        for name, attr in [("detector", "detector_weights"),
                           ("rectifier", "rectifier_weights"),
                           ("ocr", "ocr_weights")]:
            if not getattr(inf_cfg, attr, None):
                raise ValueError(f"inference.{attr} is required for the ultralytics engine")
        trt = bool(getattr(inf_cfg, "use_tensorrt", False))
        self.detector = _load_model(inf_cfg.detector_weights, inf_cfg.device, trt, task="detect")
        # rectifier is a POSE model — must declare task so the .engine returns keypoints.
        self.rectifier = _load_model(inf_cfg.rectifier_weights, inf_cfg.device, trt, task="pose")
        self.ocr = _load_model(inf_cfg.ocr_weights, inf_cfg.device, trt, task="detect")
        self.ocr_names = dict(getattr(self.ocr, "names", {}) or {})

    def _predict(self, model, img, imgsz, conf=0.25):
        kwargs = dict(imgsz=imgsz, verbose=False, device=self.cfg.device)
        if self.cfg.half:
            kwargs["half"] = True
        with self._lock:
            return model.predict(img, conf=conf, **kwargs)

    def detect(self, frame: np.ndarray) -> list[Bbox]:
        res = self._predict(self.detector, frame, self.cfg.detector_imgsz,
                            conf=self.cfg.detector_confidence if hasattr(self.cfg, "detector_confidence") else 0.25)
        boxes: list[Bbox] = []
        if not res:
            return boxes
        b = res[0].boxes
        if b is None or b.xyxy is None:
            return boxes
        xyxy = b.xyxy.cpu().numpy()
        confs = b.conf.cpu().numpy() if b.conf is not None else np.ones(len(xyxy))
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            boxes.append(Bbox(int(x1), int(y1), int(x2), int(y2), float(c)))
        boxes.sort(key=lambda bb: bb.conf, reverse=True)
        return boxes

    def rectify(self, crop: np.ndarray) -> Optional[np.ndarray]:
        out_w, out_h = self.cfg.rectified_size
        self._last_corners = None  # keypoint corners (crop coords) for the debug overlay
        res = self._predict(self.rectifier, crop, self.cfg.rectifier_imgsz)
        kpts = getattr(res[0], "keypoints", None) if res else None
        if kpts is not None and kpts.xy is not None and len(kpts.xy) > 0:
            corners = kpts.xy[0].cpu().numpy()
            if corners.shape[0] == 4:
                self._last_corners = corners
                warped = warp_plate(crop, corners, (out_w, out_h))
                if warped is not None:
                    return warped
        # Fallback: no usable 4 corners -> plain resize (no de-skew).
        import cv2

        return cv2.resize(crop, (out_w, out_h))

    def read(self, warped: np.ndarray) -> Optional[PlateCandidate]:
        res = self._predict(self.ocr, warped, self.cfg.ocr_imgsz)
        if not res:
            return None
        b = res[0].boxes
        if b is None or b.xywh is None or len(b.xywh) == 0:
            return None
        xywh = b.xywh.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int) if b.cls is not None else np.zeros(len(xywh), int)
        confs = b.conf.cpu().numpy() if b.conf is not None else np.ones(len(xywh))
        chars = [
            CharBox(label=str(self.ocr_names.get(int(k), "?")),
                    x_center=float(x), y_center=float(y), conf=float(cf), height=float(h))
            for (x, y, w, h), k, cf in zip(xywh, cls, confs)
        ]
        text, char_confs = assemble_plate(chars)
        if not text:
            return None
        mean_conf = sum(char_confs) / len(char_confs) if char_confs else 0.0
        hh, ww = warped.shape[:2]
        return PlateCandidate(text=text, mean_conf=mean_conf,
                              bbox=Bbox(0, 0, ww, hh, mean_conf),
                              plate_img=warped, char_confs=char_confs,
                              spellings=spelling_candidates(chars, limit=8))

    def warmup(self) -> None:
        dummy = np.zeros((self.cfg.detector_imgsz, self.cfg.detector_imgsz, 3), dtype=np.uint8)
        try:
            self.process(dummy)
        except Exception as e:
            logger.warning(f"warmup failed: {e}")
