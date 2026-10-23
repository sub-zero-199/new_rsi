"""CWC ANPR v2 — unified single-process ANPR for gate-in / weighbridge / gate-out.

A clean rewrite of the v11_1 system:
  * 3-model YOLOv11m cascade (detect -> keypoint stretch-fit -> char OCR) behind a
    single shared GPU inference worker (fits a 4 GB GPU);
  * client-side matching against local known-plates + the per-camera active-vehicle list;
  * Redis (work queue + 24h 3-bucket tracker) + SQLite (durable records);
  * online/offline RSI portal flow and S3 image upload.

The heavy ML stack (torch/ultralytics) is optional: with `inference.engine = "mock"`
the whole system runs and is testable without a GPU.
"""

__version__ = "0.1.0"
