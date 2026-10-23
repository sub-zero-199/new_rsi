"""Step 2 of the RSI flow: the local ANPR Data API.

This is the one call that carries the plate image itself — base64 in the JSON body, not an
S3 reference — and the one call that still works with the internet down, because the server
sits on the site LAN. That makes it the fallback path: when the RSI cloud is unreachable,
this is what records the vehicle locally, and its result is what step 3 reports upward once
connectivity returns.

Every call returns the same dict shape whether it succeeded or failed, because the caller
embeds it verbatim into the step-3 payload — a failure here is information the cloud wants,
not something to swallow.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
import requests

from anpr.log import logger


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


class AnprDataApiClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.api = cfg.anpr_data_api

    @staticmethod
    def _encode(plate_img: Optional[np.ndarray]) -> str:
        if plate_img is None or not isinstance(plate_img, np.ndarray) or plate_img.size == 0:
            return ""
        try:
            ok, buf = cv2.imencode(".jpg", plate_img)
            return base64.b64encode(buf).decode("utf-8") if ok else ""
        except Exception as e:
            logger.warning(f"could not encode plate image for the ANPR Data API: {e}")
            return ""

    def _result(self, success: bool, body: dict, response: dict, t0: str) -> dict:
        return {"success": success, "url": self.api.url, "request_body": body,
                "response": response, "request_time": t0, "response_time": _now_iso()}

    def send(self, *, camera_mac: str, vehicle_number: str, plate_img: Optional[np.ndarray],
             status: str = "online") -> dict:
        """POST one detection. Never raises — the error becomes the `response` field."""
        # The image is deliberately left out of request_body: that dict is echoed into the
        # step-3 payload, and a base64 frame there would bloat every cloud call.
        request_body = {"cameraId": camera_mac, "vehicleNumber": vehicle_number, "status": status}
        t0 = _now_iso()

        if not self.api.enabled:
            return self._result(False, request_body,
                                {"error": True, "message": "anpr_data_api disabled", "statuscode": 0}, t0)
        if self.cfg.dry_run:
            logger.info(f"[dry-run] ANPR Data API {vehicle_number} -> {self.api.url}")
            return self._result(True, request_body,
                                {"error": False, "message": "dry-run", "statuscode": 200}, t0)

        payload = {**request_body, "image": self._encode(plate_img)}
        try:
            r = requests.post(self.api.url, headers={"Content-Type": "application/json"},
                              data=json.dumps(payload), timeout=self.api.timeout)
            try:
                body = r.json()
            except Exception:
                body = {}
            if r.status_code == 200 and isinstance(body, dict) and body.get("error") is False:
                logger.info(f"ANPR Data API success for {vehicle_number}: {body.get('message', '')}")
                return self._result(True, request_body,
                                    {"error": False, "message": body.get("message", ""),
                                     "statuscode": 200}, t0)
            msg = body.get("message", r.text) if isinstance(body, dict) else r.text
            logger.warning(f"ANPR Data API rejected {vehicle_number} (HTTP {r.status_code}): {msg}")
            return self._result(False, request_body,
                                {"error": True, "message": msg, "statuscode": r.status_code}, t0)
        except Exception as e:
            logger.error(f"ANPR Data API unreachable for {vehicle_number}: {e}")
            return self._result(False, request_body,
                                {"error": True, "message": str(e), "statuscode": 0}, t0)
