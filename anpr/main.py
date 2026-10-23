"""CLI entrypoint.

  cwc-anpr -c config.yaml            # run the service
  cwc-anpr -c config.yaml --dry-run  # run, but log API/S3 calls instead of sending
  cwc-anpr export -c config.yaml     # export the 3 .pt models to TensorRT (run on the GPU box)
"""
from __future__ import annotations

import argparse
import os
import signal
import threading

from anpr.config import load_config
from anpr.log import logger


def _run(args) -> int:
    from anpr.service import AnprService

    cfg = load_config(args.config)
    if args.dry_run:
        cfg.dry_run = True
    # tell the logging module how many days of logs to keep (before any log file is opened)
    os.environ.setdefault("ANPR_LOG_RETENTION_DAYS", str(cfg.retention.max_age_days))

    svc = AnprService(cfg)
    stop = threading.Event()

    def _handle(signum, frame):
        logger.info(f"signal {signum} received, shutting down")
        stop.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    svc.start()
    try:
        while not stop.wait(30):
            logger.info(f"status: {svc.status()}")
    finally:
        svc.stop()
    return 0


def _export(args) -> int:
    from anpr.models.engine import export_tensorrt

    cfg = load_config(args.config)
    inf = cfg.inference
    for weights, imgsz in [
        (inf.detector_weights, inf.detector_imgsz),
        (inf.rectifier_weights, inf.rectifier_imgsz),
        (inf.ocr_weights, inf.ocr_imgsz),
    ]:
        if weights:
            out = export_tensorrt(weights, imgsz, half=inf.half)
            logger.info(f"exported -> {out}")
        else:
            logger.warning("skipping: weights path not set")
    return 0


def _add_common_args(p) -> None:
    # default=SUPPRESS so a subcommand's copy doesn't clobber a value given before it;
    # the base defaults are set once via ap.set_defaults below.
    p.add_argument("-c", "--config", default=argparse.SUPPRESS, help="path to config.yaml")
    p.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS,
                   help="log API/S3 calls instead of sending")


def main() -> int:
    ap = argparse.ArgumentParser(prog="cwc-anpr", description="CWC ANPR v2 service")
    _add_common_args(ap)
    sub = ap.add_subparsers(dest="cmd")
    _add_common_args(sub.add_parser("run", help="run the service (default)"))
    _add_common_args(sub.add_parser("export", help="export .pt models to TensorRT (GPU box only)"))
    ap.set_defaults(config="config.yaml", dry_run=False)

    args = ap.parse_args()
    if args.cmd == "export":
        return _export(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
