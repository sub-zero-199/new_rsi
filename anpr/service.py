"""Unified ANPR service — wires every layer together in one process.

capture (per camera) -> single GPU inference worker -> window aggregation
  -> match (local then active, >=80%) -> point rules / cooldown
  -> the three-call RSI flow:
       step 1  RSI create-triplogs-for-rsi          (skipped when offline)
       step 2  local ANPR Data API, with the image  (works without internet)
       step 3  RSI again, enriched with step 2's result
  -> on success: record journey/bucket/known-plate + queue the images for S3
  -> offline: steps 1 and 3 are deferred — the queued detection carries step 2's result and
     the original payload, and the uploader fires step 3 once the images land.
"""
from __future__ import annotations

import json
import os
import pathlib
import queue
import sys
import threading
import time
from typing import Optional

# camera_status_service sits alongside the anpr package inside new_rsi/
_STATUS_DIR = pathlib.Path(__file__).parent.parent / "camera_status_service"
if str(_STATUS_DIR) not in sys.path:
    sys.path.insert(0, str(_STATUS_DIR))
try:
    from client import CameraStatusClient as _CameraStatusClient  # type: ignore
    _HAVE_STATUS = True
except Exception as _e:
    _HAVE_STATUS = False
    _CameraStatusClient = None  # type: ignore

from anpr.log import logger, point_logger
from anpr.config import Config
from anpr.models import build_pipeline
from anpr.capture import CaptureManager
from anpr.worker import InferenceWorker
from anpr.aggregation import FinalizedDetection
from anpr.storage_sqlite import SqliteStore
from anpr.storage_redis import RedisQueue
from anpr.tracker import SharedTracker, canonical_plate
from anpr.whitelist import WhitelistManager
from anpr.doppelganger import DoppelgangerManager
from anpr.matching import PlateMatcher
from anpr.journey import PointProcessor
from anpr.portal import PortalClient
from anpr.anpr_data_api import AnprDataApiClient
from anpr.internet import InternetManager
from anpr.s3uploader import ImageBackgroundProcessor, compress_and_save_image, plate_s3_key
from anpr.auto_gateout import AutoGateOutProcessor
from anpr.retention import RetentionJanitor


class AnprService:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.internet = InternetManager(cfg.api.remote_base_url)
        self.store = SqliteStore(cfg.sqlite.path)
        self.queue = RedisQueue(cfg.redis.host, cfg.redis.port, cfg.redis.queue_db)
        self.tracker = SharedTracker(cfg.redis.host, cfg.redis.port, cfg.redis.tracker_db)
        self.whitelist = WhitelistManager(cfg.whitelist_path)
        self.doppelganger = DoppelgangerManager(cfg.doppelganger_path)
        self.matcher = PlateMatcher(cfg, self.store, self.whitelist, self.doppelganger)
        self.point = PointProcessor(cfg, self.store, self.tracker)
        self.portal = PortalClient(cfg)
        self.anpr_api = AnprDataApiClient(cfg)
        self.uploader = ImageBackgroundProcessor(cfg, self.queue, self.internet,
                                                 portal=self.portal)
        self.pipeline = build_pipeline(cfg)
        self.captures = CaptureManager(cfg)
        self.worker = InferenceWorker(cfg, self.pipeline, self.captures, self._on_finalized)
        self.auto_gateout = AutoGateOutProcessor(cfg, self.tracker, self.portal, self.point)

        self.cameras_by_id = {c.id: c for c in cfg.cameras}
        self._tmp_root = "temp_uploads"
        self._stop = threading.Event()
        self._outbox_thread: Optional[threading.Thread] = None
        # Finalized detections are handed off here so the GPU worker never waits on the
        # network. One consumer thread keeps them in order, so the dedup check always sees
        # the previous vehicle's record.
        self._decisions: queue.Queue = queue.Queue(maxsize=500)
        self._decision_thread: Optional[threading.Thread] = None

        # Camera status + internet speed reporting to the central dashboard server
        self.camera_status = _CameraStatusClient() if _HAVE_STATUS else None
        if not _HAVE_STATUS:
            logger.warning(f"camera_status_service unavailable — status reporting disabled: {_e}")

        # Disk retention over the generated-data dirs (logs, debug images, upload staging).
        self.retention: Optional[RetentionJanitor] = None
        if cfg.retention.enabled:
            log_dir = os.environ.get("ANPR_LOG_DIR", "logs")
            debug_dir = os.path.dirname(cfg.inference.rectified_dir) or cfg.inference.rectified_dir
            self.retention = RetentionJanitor(
                dirs=[log_dir, debug_dir, self._tmp_root],
                max_age_days=cfg.retention.max_age_days,
                max_total_gb=cfg.retention.max_total_gb,
                check_interval_seconds=cfg.retention.check_interval_seconds,
            )

    # ---------------- lifecycle ----------------
    def start(self) -> None:
        self.internet.check(force=True)
        self.uploader.start()
        self._decision_thread = threading.Thread(target=self._decision_loop, name="decisions",
                                                 daemon=True)
        self._decision_thread.start()
        self.captures.start()
        self.worker.start()
        self.auto_gateout.start()
        self._outbox_thread = threading.Thread(target=self._outbox_loop, name="outbox-retry", daemon=True)
        self._outbox_thread.start()
        if self.retention:
            self.retention.start()
        if self.camera_status:
            self.camera_status.connect()
            logger.info("Camera status reporting started")
        logger.info(f"ANPR service started ({len(self.cameras_by_id)} cameras, "
                    f"engine={self.cfg.inference.engine}, mode={self.cfg.api.mode})")

    def stop(self) -> None:
        self._stop.set()
        self.worker.stop()  # flushes its last windows into the decision queue
        if self._decision_thread:
            self._decisions.put(None)
            self._decision_thread.join(timeout=60)
        self.captures.stop()
        self.auto_gateout.stop()
        self.uploader.stop()
        if self.retention:
            self.retention.stop()
        if self.camera_status:
            self.camera_status.cleanup()
        logger.info("ANPR service stopped")

    def status(self) -> dict:
        return {
            "cameras": self.captures.status(),
            "queue_size": self.queue.get_queue_size(),
            "online": self.internet.is_online(),
            "frames_processed": self.worker.frames_processed,
        }

    # ---------------- core decision ----------------
    def _on_finalized(self, det: FinalizedDetection) -> None:
        """Called on the GPU worker thread: hand off and return immediately, so a slow or
        unreachable API never stops the cameras from being read."""
        try:
            self._decisions.put_nowait(det)
        except queue.Full:
            logger.error(f"decision queue full — dropping {det.best_text} ({det.point})")

    def _decision_loop(self) -> None:
        while True:
            det = self._decisions.get()
            if det is None:
                return
            self._handle_finalized(det)

    def _handle_finalized(self, det: FinalizedDetection) -> None:
        try:
            cam = self.cameras_by_id.get(det.camera_id)
            camera_mac = cam.mac if cam else det.camera_id
            plog = point_logger(det.point)
            plog.info(f"detected {det.best_text!r} (conf {det.mean_conf:.2f}, "
                      f"{det.detection_count} reads)")
            match = self.matcher.resolve(det)
            plate = match.resolved
            if not (match.valid or match.matched or match.whitelisted):
                plog.info(f"skip invalid plate {det.best_text!r}")
                return
            if match.matched:
                plog.info(f"matched -> {plate} via {match.source} ({match.score:.2f})")
            else:
                plog.info(f"new plate {plate} (no local match)")
            # The portal always creates the trip, so the in/out dedup here is what prevents
            # duplicate trips (gate-in while inside / gate-out while already out).
            decision = self.point.evaluate(det.point, plate, det.ts)
            if not decision.allow:
                plog.info(f"discard {plate}: {decision.reason}")
                return
            plog.info(f"accept {plate} -> create-trip")
            self._push_and_record(det, plate, camera_mac, decision.continuity_status)
        except Exception as e:
            logger.exception(f"on_finalized error: {e}")

    def _push_and_record(self, det: FinalizedDetection, plate: str, camera_mac: str,
                         continuity: str) -> None:
        """The three-call RSI flow for one accepted detection.

        Online, all three run inline. Offline, step 1 is skipped entirely (no point spending
        its timeouts on a link we already know is down) and step 3 is deferred to the
        uploader, which fires it once the images are actually in S3 — using the original
        detection time, so the cloud records when the vehicle passed, not when we caught up.
        """
        plog = point_logger(det.point)
        anpr_key = plate_s3_key(self.cfg.s3.prefix, det.point, plate, det.ts, "plate.jpg")
        offline = self.cfg.api.mode == "offline" or not self.internet.check()
        plate_path, full_path = self._save_images(det, plate)

        payload = self.portal.build_payload(
            camera_mac=camera_mac, vehicle_number=plate, anpr_image_path=anpr_key,
            detection_time=det.ts, whitelisted=self.whitelist.is_whitelisted(plate),
        )

        # ---- step 1: the RSI cloud (online only) ----
        if offline:
            plog.info(f"offline — skipping RSI step-1 for {plate}")
            step1 = None
        else:
            step1 = self.portal.create_triplog(payload)
            body = json.dumps(step1.body) if isinstance(step1.body, (dict, list)) else str(step1.body)
            plog.info(f"RSI step-1 request: {json.dumps(payload)}")
            plog.info(f"RSI step-1 response: HTTP {step1.status_code} | {body}")

        # A timeout means nobody knows anything yet — fall back to the local server and
        # defer the cloud, exactly as if we had been offline from the start.
        deferred = offline or (step1 is not None and not step1.reachable)
        if step1 is not None and not step1.reachable:
            plog.warning(f"RSI unreachable for {plate} — falling back to the local ANPR Data API")

        # ---- step 2: the local ANPR Data API (always, it carries the image) ----
        anpr_details = self.anpr_api.send(
            camera_mac=camera_mac, vehicle_number=plate, plate_img=det.plate_img,
            status="offline" if deferred else "online",
        )
        plog.info(f"ANPR Data API ({'offline' if deferred else 'online'}): "
                  f"{json.dumps(anpr_details['response'])}")

        # ---- did anything accept this vehicle? ----
        cloud_ok = bool(step1 and step1.success)
        local_ok = bool(anpr_details["success"])
        if not (cloud_ok or local_ok):
            # Neither the cloud nor the local server took it. Keep it for the outbox so the
            # trip is not simply lost — this is the one path rsi_current had no answer for.
            # A vehicle waiting at the point is re-read every few seconds; queue it once, not
            # once per read, or every copy becomes its own trip when connectivity returns.
            waiting = self._recent_outbox_id(det.point, plate, det.ts)
            if waiting is not None:
                ImageBackgroundProcessor._cleanup(plate_path, full_path)
                plog.warning(f"push failed for {plate} (cloud and local both refused); "
                             f"already waiting in outbox (#{waiting}), not queued again")
                return
            self.store.enqueue_outbox({
                "camera_mac": camera_mac, "plate": plate, "plate_raw": det.best_text,
                "point": det.point, "ts": det.ts, "anpr_key": anpr_key,
                "plate_path": plate_path, "full_path": full_path, "continuity": continuity,
            })
            plog.warning(f"push failed for {plate} (cloud and local both refused); "
                         f"queued to outbox (will retry)")
            return

        # ---- record it ----
        self.point.record_success(
            point=det.point, plate=plate, plate_raw=det.best_text, ts=det.ts,
            camera_id=det.camera_id, s3_key=anpr_key,
            api_response=(step1.body if step1 else anpr_details["response"]),
            continuity_status=continuity,
        )

        # ---- step 3: tell the cloud what the local server said ----
        entry = {
            "vehicle_number": plate, "plate_img_path": plate_path, "full_img_path": full_path,
            "detection_time": det.ts, "point": det.point, "camera_id": det.camera_id,
            "offline_mode": deferred,
        }
        if deferred:
            # Carry everything step 3 will need; the uploader sends it after the images land.
            entry["rsi_payload"] = json.dumps(payload)
            entry["anpr_details"] = json.dumps(anpr_details)
            plog.info(f"RSI step-3 deferred for {plate} until its images reach S3")
        else:
            self.portal.send_enriched(payload, anpr_details)
        self.queue.store_detection(entry)

    def _save_images(self, det: FinalizedDetection, plate: str) -> tuple[str, str]:
        folder = os.path.abspath(os.path.join(self._tmp_root, det.point, plate, str(det.ts)))
        plate_path = os.path.join(folder, "plate.jpg")
        full_path = os.path.join(folder, "vehicleno.jpg")
        try:
            compress_and_save_image(det.plate_img, plate_path, is_plate=True)
            if self.cfg.s3.upload_full_frame:
                compress_and_save_image(det.full_frame, full_path, is_plate=False)
        except Exception as e:
            logger.error(f"save images failed for {plate}: {e}")
        return plate_path, full_path

    # ---------------- outbox retry ----------------
    def _outbox_window(self, point: str) -> int:
        """How long a queued outbox entry absorbs repeat reads of the same vehicle. Matches the
        gap after which a repeat counts as a genuine new visit at that point."""
        if point == "weighbridge":
            return self.cfg.cooldown.weighbridge_second_gap_seconds
        return self.cfg.cooldown.reentry_gap_seconds

    def _recent_outbox_id(self, point: str, plate: str, ts: int) -> Optional[int]:
        """Id of a pending outbox entry for this exact plate at this point queued within the
        window before `ts`, else None. Exact plate only — never merges two different trucks."""
        canon = canonical_plate(plate)
        window = self._outbox_window(point)
        for item in self.store.pending_outbox(500):
            p = item["payload"]
            if (p.get("point") == point and canonical_plate(p.get("plate", "")) == canon
                    and 0 <= int(ts) - int(p.get("ts", 0)) < window):
                return item["id"]
        return None

    def _outbox_loop(self) -> None:
        while not self._stop.wait(15):
            try:
                if self.cfg.api.mode != "offline" and not self.internet.is_online():
                    continue
                self._drain_outbox()
            except Exception as e:
                logger.error(f"outbox loop error: {e}")

    def _drain_outbox(self) -> None:
        for item in self.store.pending_outbox(20):
            p = item["payload"]
            # Safety net: apply the same rules a live read gets. An earlier entry for
            # this vehicle may have just been sent and recorded (e.g. it is now
            # "already inside"), in which case this one is a repeat, not a new trip.
            decision = self.point.evaluate(p["point"], p["plate"], int(p["ts"]))
            if not decision.allow:
                self.store.mark_outbox(item["id"], "duplicate")
                ImageBackgroundProcessor._cleanup(p.get("plate_path", ""),
                                                  p.get("full_path", ""))
                point_logger(p["point"]).info(
                    f"outbox: {p['plate']} not sent — {decision.reason}")
                continue
            payload = self.portal.build_payload(
                camera_mac=p["camera_mac"], vehicle_number=p["plate"],
                anpr_image_path=p["anpr_key"], detection_time=p["ts"],
                whitelisted=self.whitelist.is_whitelisted(p["plate"]),
            )
            res = self.portal.create_triplog(payload)
            if res.success:
                self.point.record_success(
                    point=p["point"], plate=p["plate"], plate_raw=p["plate_raw"],
                    ts=p["ts"], camera_id="outbox", s3_key=p["anpr_key"],
                    api_response=res.body, continuity_status=p.get("continuity", "ok"),
                )
                self.queue.store_detection({
                    "vehicle_number": p["plate"], "plate_img_path": p["plate_path"],
                    "full_img_path": p["full_path"], "detection_time": p["ts"],
                    "point": p["point"], "camera_id": "outbox",
                })
                self.store.mark_outbox(item["id"], "sent")
            elif res.finalized:
                # A definitive answer on retry: the cloud will never take it, so stop.
                self.store.mark_outbox(item["id"], "rejected")
                point_logger(p["point"]).warning(
                    f"outbox: push rejected for {p['plate']}: {res.message} — dropping")
            else:
                self.store.mark_outbox(item["id"], "pending")
