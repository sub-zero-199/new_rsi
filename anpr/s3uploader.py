"""Image persistence + S3 upload. Ported from v11_1 upload_to_s3.py (compress_and_save_image)
and background_processor.py (the queue-draining uploader), with AWS creds taken from env/config
instead of being hardcoded.

Each detection lands in S3 four times, matching the layout the RSI backend already reads:

    ANPR/{point}/{vehicle}/{detectionTime}/plate.jpg      <- what `anprImagePath` points at
    ANPR/{point}/{vehicle}/{detectionTime}/vehicleno.jpg
    ANPR/{point}/{vehicle}/plate.jpg                      <- un-timestamped "latest"
    ANPR/{point}/{vehicle}/vehicleno.jpg

Local files are deleted only after every upload is verified; when offline they're preserved
and retried. A detection queued while the cloud was unreachable also carries the local ANPR
Data API's result, and once its images land this is what fires the step-3 enriched RSI call
— with the ORIGINAL detection time, not the time the upload happened.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np

from anpr.log import logger


def compress_and_save_image(image: np.ndarray, output_path: str, *, jpeg_quality: int = 85,
                            max_w: int = 1920, max_h: int = 1080, is_plate: bool = False) -> bool:
    """Save an image as a (resized, quality-capped) JPEG. Returns True on a verified write."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0 or image.ndim < 2:
        logger.error(f"cannot save image to {output_path}: invalid array")
        return False
    img = image
    if is_plate:
        jpeg_quality = max(70, jpeg_quality - 5)
        max_w, max_h = min(max_w, 800), min(max_h, 600)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    h, w = img.shape[:2]
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ok = cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    if not ok:
        logger.error(f"cv2.imwrite failed for {output_path}")
        return False
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


# The RSI backend spells the point without the underscore. Keep these exact so the bucket
# layout stays identical to what it has been reading all along.
_S3_POINT = {"gate_in": "gatein", "gate_out": "gateout", "weighbridge": "weighbridge"}


def s3_point(point: str) -> str:
    return _S3_POINT.get(point, point)


def plate_s3_key(prefix: str, point: str, vehicle: str, detection_time: int,
                 name: str = "plate.jpg") -> str:
    """The timestamped key — this is what `anprImagePath` in the RSI payload points at."""
    return f"{prefix.strip('/')}/{s3_point(point)}/{vehicle}/{int(detection_time)}/{name}"


def latest_s3_key(prefix: str, point: str, vehicle: str, name: str = "plate.jpg") -> str:
    """The un-timestamped copy: whatever reads this path wants "this vehicle's newest image"."""
    return f"{prefix.strip('/')}/{s3_point(point)}/{vehicle}/{name}"


class S3Uploader:
    def __init__(self, cfg_s3):
        self.cfg = cfg_s3
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3  # lazy

            kwargs = {}
            if self.cfg.access_key_id and self.cfg.secret_access_key:
                kwargs.update(aws_access_key_id=self.cfg.access_key_id,
                              aws_secret_access_key=self.cfg.secret_access_key)
            if self.cfg.region:
                kwargs["region_name"] = self.cfg.region
            self._client = boto3.client("s3", **kwargs)
            logger.info("S3 client initialized")
        return self._client

    def upload(self, local_path: str, key: str) -> bool:
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            logger.error(f"S3 upload skipped, missing/empty: {local_path}")
            return False
        try:
            self.client.upload_file(local_path, self.cfg.bucket, key)
            self.client.head_object(Bucket=self.cfg.bucket, Key=key)  # verify
            logger.info(f"S3 upload ok: s3://{self.cfg.bucket}/{key}")
            return True
        except Exception as e:
            logger.error(f"S3 upload error {key}: {e}")
            return False


class ImageBackgroundProcessor:
    """Drains the Redis work queue: upload images to S3, delete local files on success.
    Runs on its own thread; pauses while offline so files are preserved for retry.
    """

    def __init__(self, cfg, queue, internet, uploader: Optional[S3Uploader] = None,
                 portal=None):
        self.cfg = cfg
        self.queue = queue
        self.internet = internet
        self.uploader = uploader or S3Uploader(cfg.s3)
        # Needed for the step-3 enriched call on a detection that was queued while offline.
        self.portal = portal
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="s3-uploader", daemon=True)
        self._thread.start()
        logger.info("Image background processor started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.cfg.s3.enabled and not self.internet.is_online():
                    self.internet.check()
                    if not self.internet.is_online():
                        time.sleep(3)
                        continue
                batch = self.queue.get_pending_detections(batch_size=5)
                if not batch:
                    time.sleep(1)
                    continue
                for key, data in batch:
                    if self._stop.is_set():
                        break
                    self._process_one(key, data)
            except Exception as e:
                logger.error(f"uploader loop error: {e}")
                time.sleep(5)

    def _process_one(self, detection_key: str, data: dict) -> None:
        vehicle = str(data.get("vehicle_number", ""))
        point = str(data.get("point", ""))
        plate_path = str(data.get("plate_img_path", ""))
        full_path = str(data.get("full_img_path", ""))
        dt = int(data.get("detection_time", time.time()))

        if not self.cfg.s3.enabled or self.cfg.dry_run:
            logger.info(f"[skip-upload] {vehicle} (s3 disabled/dry-run)")
            self.queue.mark_detection_processed(detection_key, success=True)
            self._send_enriched(data, vehicle)
            self._cleanup(plate_path, full_path)
            return

        prefix = self.cfg.s3.prefix
        jobs = []
        if plate_path and os.path.exists(plate_path):
            jobs.append((plate_path, plate_s3_key(prefix, point, vehicle, dt, "plate.jpg")))
            jobs.append((plate_path, latest_s3_key(prefix, point, vehicle, "plate.jpg")))
        if self.cfg.s3.upload_full_frame and full_path and os.path.exists(full_path):
            jobs.append((full_path, plate_s3_key(prefix, point, vehicle, dt, "vehicleno.jpg")))
            jobs.append((full_path, latest_s3_key(prefix, point, vehicle, "vehicleno.jpg")))

        ok = True
        for local, key in jobs:
            ok &= self.uploader.upload(local, key)

        self.queue.mark_detection_processed(detection_key, success=ok)
        if ok:
            # Images are in place, so the cloud can now resolve anprImagePath. Only now is
            # the enriched call worth making.
            self._send_enriched(data, vehicle)
            self._cleanup(plate_path, full_path)
        else:
            logger.warning(f"upload incomplete for {vehicle}; files preserved for retry")

    def _send_enriched(self, data: dict, vehicle: str) -> None:
        """Step 3 for a detection that was queued while the cloud was unreachable.

        Best-effort by design: the vehicle is already recorded locally, so a failure here
        is logged and dropped rather than re-queued.
        """
        if self.portal is None:
            return
        raw_payload = data.get("rsi_payload")
        raw_details = data.get("anpr_details")
        if not raw_payload or not raw_details:
            return
        try:
            payload = json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload
            details = json.loads(raw_details) if isinstance(raw_details, str) else raw_details
        except Exception as e:
            logger.warning(f"could not read queued RSI payload for {vehicle}: {e}")
            return
        logger.info(f"sending deferred RSI step-3 for {vehicle} "
                    f"(detectionTime {payload.get('detectionTime')})")
        self.portal.send_enriched(payload, details)

    @staticmethod
    def _cleanup(*paths: str) -> None:
        for p in paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
