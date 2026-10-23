"""The single shared-GPU inference worker.

One thread samples the latest frame from every camera at the configured target fps,
runs the 3-stage pipeline (serialized — the 4 GB GPU does one inference at a time),
feeds reads into the window aggregator, and emits FinalizedDetection objects (one best
read per vehicle per window) through `on_finalized`.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from anpr.log import logger, point_logger
from anpr.aggregation import WindowAggregator, FinalizedDetection
from anpr.capture import CaptureManager
from anpr.models.pipeline import PlatePipeline


class InferenceWorker:
    def __init__(self, cfg, pipeline: PlatePipeline, captures: CaptureManager,
                 on_finalized: Callable[[FinalizedDetection], None]):
        self.cfg = cfg
        self.pipeline = pipeline
        self.captures = captures
        self.aggregator = WindowAggregator(cfg)
        self.on_finalized = on_finalized
        self.interval = 1.0 / max(0.1, cfg.inference.target_fps)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_processed: dict[str, float] = {}
        self._last_frame_ts: dict[str, float] = {}
        self.frames_processed = 0
        # per-point processing heartbeat (so each point log shows the pipeline is alive)
        self._points = sorted({c.cam.point for c in captures.captures.values()})
        self._hb_frames: dict[str, int] = {}
        self._hb_reads: dict[str, int] = {}
        self._hb_interval = 2.0  # per-point "processing:" heartbeat cadence (idle liveness)
        self._hb_last = 0.0
        # Per-point detection settings, resolved once — config does not change at runtime and
        # detection_for() copies a model on every call.
        self._dconf = {p: cfg.detection_for(p) for p in self._points}
        # Camera gating (detection.allow_degraded_cameras): when a camera was first seen
        # down, and which points are currently halted because of it.
        self._down_since: dict[str, float] = {}
        self._halted: set[str] = set()

    def start(self) -> None:
        try:
            self.pipeline.warmup()
        except Exception:
            pass
        self._thread = threading.Thread(target=self._run, name="inference-worker", daemon=True)
        self._thread.start()
        logger.info(f"Inference worker started (target {self.cfg.inference.target_fps} fps/camera)")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        for det in self.aggregator.flush_all():
            self._emit(det)

    def _emit(self, det: FinalizedDetection) -> None:
        try:
            self.on_finalized(det)
        except Exception as e:
            logger.error(f"on_finalized error for {det.best_text}: {e}")

    # ---------------- camera gating ----------------
    def _down_cameras(self, point: str) -> list[str]:
        return [f"{cam_id} ({cap.status})"
                for cam_id, cap in self.captures.captures.items()
                if cap.cam.point == point and cap.status != "running"]

    def _halted_points(self, now: float) -> set[str]:
        """Points that must stop detecting because an enabled camera has been down too long.

        Only cameras the config enabled are considered — CaptureManager is built from
        `enabled_cameras()` — so disabling a damaged camera leaves a working one-camera
        point instead of one that can never run.
        """
        halted: set[str] = set()
        for cam_id, cap in self.captures.captures.items():
            if cap.status == "running":
                self._down_since.pop(cam_id, None)
                continue
            down_since = self._down_since.setdefault(cam_id, now)
            dconf = self._dconf.get(cap.cam.point)
            if dconf is None or dconf.allow_degraded_cameras:
                continue
            if now - down_since >= dconf.camera_down_grace_seconds:
                halted.add(cap.cam.point)
        return halted

    def _apply_halts(self, now: float) -> set[str]:
        """Update which points are halted, logging each transition once."""
        halted = self._halted_points(now)
        for point in sorted(halted - self._halted):
            dropped = self.aggregator.drop_point(point)
            down = ", ".join(self._down_cameras(point)) or "a camera"
            point_logger(point).warning(
                f"detection HALTED — {down} down, allow_degraded_cameras is false"
                + (f"; discarded {dropped} in-flight window(s)" if dropped else ""))
        for point in sorted(self._halted - halted):
            point_logger(point).info("detection RESUMED — every enabled camera is up")
        self._halted = halted
        return halted

    def _run(self) -> None:
        self._hb_last = time.time()
        while not self._stop.is_set():
            now = time.time()
            halted = self._apply_halts(now)
            for cam_id, cap in self.captures.captures.items():
                if cap.cam.point in halted:
                    continue
                if now - self._last_processed.get(cam_id, 0.0) < self.interval:
                    continue
                latest = cap.read_latest()
                if latest is None:
                    continue
                frame, ts = latest
                if ts == self._last_frame_ts.get(cam_id):
                    continue  # no new frame since last poll
                self._last_frame_ts[cam_id] = ts
                self._last_processed[cam_id] = now
                point = cap.cam.point
                try:
                    cands = self.pipeline.process(frame)
                except Exception as e:
                    logger.error(f"pipeline error on {cam_id}: {e}")
                    point_logger(point).error(f"pipeline error on {cam_id}: {e}")
                    continue
                self.frames_processed += 1
                self._hb_frames[point] = self._hb_frames.get(point, 0) + 1
                if cands:
                    self._hb_reads[point] = self._hb_reads.get(point, 0) + len(cands)
                for c in cands:
                    self.aggregator.add(cam_id, point, c, frame, ts)

            for det in self.aggregator.sweep(now):
                self._emit(det)

            self._maybe_heartbeat(now)
            time.sleep(0.02)

    def _maybe_heartbeat(self, now: float) -> None:
        """Every ~2s, log a per-point processing summary so each point log shows the
        pipeline is alive even when no plates are being read."""
        if now - self._hb_last < self._hb_interval:
            return
        secs = int(now - self._hb_last)
        self._hb_last = now
        for point in self._points:
            frames = self._hb_frames.pop(point, 0)
            reads = self._hb_reads.pop(point, 0)
            if point in self._halted:
                down = ", ".join(self._down_cameras(point)) or "a camera"
                point_logger(point).warning(
                    f"HALTED: not detecting — waiting for {down} (last {secs}s)")
            else:
                point_logger(point).info(
                    f"processing: {frames} frames, {reads} plate reads (last {secs}s)")
        self._hb_frames.clear()
        self._hb_reads.clear()
