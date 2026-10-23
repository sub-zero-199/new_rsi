"""Auto gate-out after 24h. Ported from v11_1 auto_gateout_processor.py.

Every 5 minutes, scans the tracker's pending-auto-gateout queue; for any vehicle whose
gate-in is >=24h old and has no recorded gate-out, it pushes a gate-out trip and records it.
Pending queue is Redis-only, so this is a no-op when running on the local fallback.
"""
from __future__ import annotations

import threading
import time

from anpr.log import logger
from anpr.s3uploader import plate_s3_key


class AutoGateOutProcessor:
    def __init__(self, cfg, tracker, portal, point, check_interval: float = 300.0):
        self.cfg = cfg
        self.tracker = tracker
        self.portal = portal
        self.point = point
        self.check_interval = check_interval
        self._stop = threading.Event()
        self._thread = None
        gate_out_cams = [c for c in cfg.cameras if c.point == "gate_out"]
        self._camera_mac = gate_out_cams[0].mac if gate_out_cams else ""

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="auto-gateout", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._stop.wait(self.check_interval):
                break
            try:
                self._process()
                self.tracker.cleanup_old_entries()
            except Exception as e:
                logger.error(f"auto-gateout loop error: {e}")

    def _process(self) -> None:
        for entry in self.tracker.get_pending_auto_gateouts():
            vehicle = entry["vehicle_number"]
            if self.tracker.check_in_last_24h("gate_out", vehicle):
                self.tracker.remove_pending_auto_gateout(entry["key"])
                continue
            ts = int(time.time())
            anpr_key = (entry.get("anpr_image_path")
                        or plate_s3_key(self.cfg.s3.prefix, "gate_out", vehicle, ts))
            logger.info(f"Auto gate-out for {vehicle} (24h elapsed)")
            payload = self.portal.build_payload(camera_mac=self._camera_mac, vehicle_number=vehicle,
                                                anpr_image_path=anpr_key, detection_time=ts)
            res = self.portal.create_triplog(payload)
            if res.success:
                self.point.record_success(point="gate_out", plate=vehicle, plate_raw=vehicle, ts=ts,
                                          camera_id="auto", s3_key=anpr_key, api_response=res.body,
                                          continuity_status="auto")
                self.tracker.remove_pending_auto_gateout(entry["key"])
