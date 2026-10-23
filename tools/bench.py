#!/usr/bin/env python3
"""Benchmark .pt (PyTorch/FP16) vs .engine (TensorRT) inference speed for each model.

Loads each model, warms up, times N inferences on a dummy frame, and reports
milliseconds/inference, FPS, and the .engine speedup over .pt.

  uv run python tools/bench.py            # uses config.yaml
For clean numbers, stop the service first so the GPU is exclusive:
  sudo systemctl stop cwc-anpr.service && uv run python tools/bench.py && sudo systemctl start cwc-anpr.service
"""
from __future__ import annotations
import os, time, gc
import numpy as np
from anpr.config import load_config


def bench(path: str, imgsz: int, n: int = 50, warmup: int = 8):
    from ultralytics import YOLO
    import torch
    model = YOLO(path)
    img = (np.random.rand(imgsz, imgsz, 3) * 255).astype("uint8")
    kw = dict(imgsz=imgsz, device=0, verbose=False)
    if path.endswith(".pt"):
        kw["half"] = True  # match the engine (FP16)
    for _ in range(warmup):
        model.predict(img, **kw)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        model.predict(img, **kw)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return (dt / n) * 1000.0, n / dt


def main() -> int:
    inf = load_config("config.yaml").inference
    models = [
        ("detect", inf.detector_weights, inf.detector_imgsz),
        ("keypoint", inf.rectifier_weights, inf.rectifier_imgsz),
        ("ocr", inf.ocr_weights, inf.ocr_imgsz),
    ]
    print(f"{'model':<10}{'.pt ms':>9}{'.engine ms':>12}{'.pt fps':>9}{'.engine fps':>12}{'speedup':>9}")
    print("-" * 61)
    for name, w, sz in models:
        eng = os.path.splitext(w)[0] + ".engine"
        try:
            pt_ms, pt_fps = bench(w, sz)
        except Exception as e:
            print(f"{name:<10} .pt error: {str(e)[:40]}")
            continue
        try:
            en_ms, en_fps = bench(eng, sz)
        except Exception as e:
            print(f"{name:<10} .engine error: {str(e)[:40]}")
            continue
        print(f"{name:<10}{pt_ms:>9.1f}{en_ms:>12.1f}{pt_fps:>9.1f}{en_fps:>12.1f}{pt_ms/en_ms:>8.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
