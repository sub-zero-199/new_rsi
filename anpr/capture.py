"""Per-camera frame capture. One thread per camera keeps only the *latest* frame
(dropping stale frames so we never lag the stream); the inference worker samples it at
the configured target fps. Ported from v11_1 multi_camera_manager._capture_frames, with
file/video sources added for offline testing.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Optional

# Low-latency RTSP for OpenCV/FFmpeg — set BEFORE cv2 opens any stream:
#   tcp transport, no input buffering, low delay, cap the reorder/latency to ~0.5s, and a
#   5s socket timeout so a slow/flaky camera can't hang the open for ~20s. This prevents
#   the RTSP latency from creeping up over long runs (the "40s to detect" symptom).
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000|reorder_queue_size;0|stimeout;5000000",
)

import cv2
import numpy as np

from anpr.log import logger, camera_event, camera_info
from anpr.config import CameraConfig

# Mask the password in an rtsp://user:pass@host URL before logging it.
_CRED_RE = re.compile(r"(://[^:/@\s]+:)[^@/\s]+(@)")


def mask_source(source: str) -> str:
    return _CRED_RE.sub(r"\1***\2", str(source))


class CameraCapture:
    def __init__(self, cam: CameraConfig, source: str, reconnect_delay: float = 3.0,
                 max_reconnect: int = 5):
        self.cam = cam
        self.source = source
        self.is_file = not str(source).lower().startswith("rtsp")
        self.reconnect_delay = reconnect_delay
        self.max_reconnect = max_reconnect

        self._latest: Optional[tuple[np.ndarray, float]] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.status = "init"
        self._down_logged = False   # log an outage once, not every retry attempt
        self._opened_once = False
        # live stream details (refreshed on each connect) for the periodic status report
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.codec = ""
        self.open_ms = 0
        self.reconnects = 0

    @property
    def camera_id(self) -> str:
        return self.cam.id

    @property
    def safe_source(self) -> str:
        """The source with any RTSP password masked — safe to log."""
        return mask_source(self.source)

    @property
    def tag(self) -> str:
        """Concise identifier for log lines: 'id [point] ip'."""
        return f"{self.cam.id} [{self.cam.point}] {self.cam.address or 'file'}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"capture-{self.cam.id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def read_latest(self) -> Optional[tuple[np.ndarray, float]]:
        with self._lock:
            return self._latest

    def _set_latest(self, frame: np.ndarray, ts: float) -> None:
        with self._lock:
            self._latest = (frame, ts)

    def _read_stream_props(self, cap) -> None:
        """Best-effort resolution / fps / codec from the open stream (cheap, no ffprobe)."""
        try:
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w and h:
                self.width, self.height = w, h
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            if fps > 0:
                self.fps = round(fps, 1)
            fc = int(cap.get(cv2.CAP_PROP_FOURCC))
            if fc:
                codec = "".join(chr((fc >> 8 * i) & 0xFF) for i in range(4))
                codec = "".join(ch for ch in codec if ch.isprintable()).strip()
                if codec:
                    self.codec = codec
        except Exception:
            pass

    def snapshot(self) -> dict:
        """Current status + stream details for the periodic human-readable report."""
        latest = self.read_latest()
        age = (time.time() - latest[1]) if latest else None
        return {"id": self.cam.id, "point": self.cam.point, "ip": self.cam.address or "file",
                "status": self.status, "w": self.width, "h": self.height, "fps": self.fps,
                "codec": self.codec, "open_ms": self.open_ms, "reconnects": self.reconnects,
                "frame_age": age}

    def _run(self) -> None:
        attempts = 0
        while not self._stop.is_set():
            t0 = time.time()
            cap = cv2.VideoCapture(self.source)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                attempts += 1
                self.status = "disconnected"
                if not self._down_logged:  # log the outage once, not every retry
                    camera_event(self.cam.point,
                                 f"{self.tag}: DOWN — cannot open (retrying every {self.reconnect_delay:g}s)",
                                 "warning")
                    self._down_logged = True
                cap.release()
                if self.is_file:
                    self.status = "error"
                    return
                if attempts >= self.max_reconnect:
                    attempts = 0
                time.sleep(self.reconnect_delay)
                continue

            self.status = "running"
            attempts = 0
            self.open_ms = int((time.time() - t0) * 1000)
            self._read_stream_props(cap)
            if self._opened_once:
                self.reconnects += 1
            camera_event(self.cam.point,
                         f"{self.tag}: {'reconnected' if self._opened_once else 'connected'}")
            self._opened_once = True
            self._down_logged = False

            # For file sources, pace to the clip's native fps so the worker samples across it.
            frame_interval = 0.0
            if self.is_file:
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                frame_interval = (1.0 / fps) if fps and fps > 0 else (1.0 / 25.0)

            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret or frame is None or getattr(frame, "size", 0) == 0:
                    if self.is_file:
                        self.status = "ended"
                        cap.release()
                        camera_event(self.cam.point, f"{self.tag}: file source ended")
                        return
                    self.status = "reconnecting"
                    camera_event(self.cam.point, f"{self.tag}: bad frame, reconnecting", "warning")
                    break
                self._set_latest(frame, time.time())
                if frame_interval:
                    time.sleep(frame_interval)

            cap.release()
            if not self.is_file:
                time.sleep(self.reconnect_delay)


class CaptureManager:
    """Owns all camera capture threads and resolves each camera's source from config."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.captures: dict[str, CameraCapture] = {}
        for cam in cfg.enabled_cameras():
            source = cam.build_source(cfg.rtsp_default_username, cfg.rtsp_default_password)
            self.captures[cam.id] = CameraCapture(cam, source)

    def start(self) -> None:
        self.log_camera_map()
        for c in self.captures.values():
            c.start()
        camera_info(f"Started {len(self.captures)} camera captures")
        self._report_stop = threading.Event()
        self._report_thread = threading.Thread(target=self._report_loop, name="cam-status-report",
                                               daemon=True)
        self._report_thread.start()

    def _report_loop(self, interval: float = 60.0) -> None:
        # periodic full snapshot (every `interval`s) so cs.log always shows current detail
        while not self._report_stop.wait(interval):
            try:
                self.status_report()
            except Exception as e:
                logger.error(f"camera status report failed: {e}")

    def status_report(self) -> None:
        """Human-readable snapshot of every camera: state + resolution/codec/fps/liveness/reconnects."""
        caps = list(self.captures.values())
        if not caps:
            return
        id_w = max(len(c.cam.id) for c in caps)
        pt_w = max(len(c.cam.point) for c in caps)
        up = sum(1 for c in caps if c.status == "running")
        camera_info(f"Camera status - {up}/{len(caps)} up:")
        for c in caps:
            s = c.snapshot()
            if s["status"] == "running":
                res = f'{s["w"]}x{s["h"]}' if s["w"] else "?"
                fps = f'{s["fps"]:g}fps' if s["fps"] else "?fps"
                age = f'{s["frame_age"]:.1f}s ago' if s["frame_age"] is not None else "no frame"
                detail = (f'UP    {res} {s["codec"] or "?"} @{fps}  '
                          f'frame {age}  open {s["open_ms"]}ms  reconn {s["reconnects"]}')
            else:
                detail = f'{s["status"].upper():<7} reconn {s["reconnects"]}'
            camera_info(f'  {s["id"]:<{id_w}}  {s["point"]:<{pt_w}}  {str(s["ip"]):<15}  {detail}')

    def log_camera_map(self) -> None:
        """Log each camera's id -> point -> mac -> source so the config wiring can be
        eyeballed at startup (passwords masked). The header + full list go to
        camera_status.log; each camera's line also goes to its own point log."""
        if not self.captures:
            camera_info("No enabled cameras configured", "warning")
            return
        id_w = max(len(cid) for cid in self.captures)
        pt_w = max(len(c.cam.point) for c in self.captures.values())
        camera_info(f"Camera map ({len(self.captures)} enabled):")
        for cid, c in self.captures.items():
            camera_event(c.cam.point,
                         f"  {cid:<{id_w}}  ->  {c.cam.point:<{pt_w}}  "
                         f"mac={c.cam.mac}  src={c.safe_source}")

    def stop(self) -> None:
        rs = getattr(self, "_report_stop", None)
        if rs:
            rs.set()
        for c in self.captures.values():
            c.stop()

    def status(self) -> dict[str, str]:
        return {cid: c.status for cid, c in self.captures.items()}
