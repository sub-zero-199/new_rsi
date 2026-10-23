"""The RSI cloud client — steps 1 and 3 of the three-call flow.

Both steps POST to the same endpoint (`create-triplogs-for-rsi`); step 3 re-sends step 1's
payload with the local ANPR Data API's result attached, so the cloud learns what the local
server did even when the cloud itself was unreachable at detection time.

    step 1  create_triplog()  -> the detection
    step 2  anpr_data_api.py  -> the local server, carries the plate image
    step 3  send_enriched()   -> step 1's payload + step 2's request/response/timings

Two failure kinds, handled very differently:

  * An HTTP REPLY — any status, including a business rejection — is final. It is never
    retried, and the flow proceeds to steps 2 and 3 regardless, because the vehicle did
    pass the gate and the local server still needs telling.
  * A TIMEOUT or network error means nobody knows anything yet. That retries
    (`api.remote_retries`), and if it never lands the caller falls back to the local API
    and queues the detection so step 3 fires when connectivity returns.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from anpr.log import logger


@dataclass
class PortalResult:
    success: bool            # step 1 returned the configured success message
    store: bool              # whether to add the plate to known_plates
    reachable: bool          # False => timeout/network error, never got a reply
    status_code: Optional[int] = None
    body: object = None
    message: str = ""
    payload: dict = field(default_factory=dict)
    # True => the cloud gave a definitive answer (don't retry it via the outbox).
    finalized: bool = False


class PortalClient:
    def __init__(self, cfg):
        self.cfg = cfg

    # ---------------- shared bits ----------------
    def _url(self) -> str:
        return self.cfg.api.remote_base_url.rstrip("/") + self.cfg.api.rsi_endpoint

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.cfg.api.secret_key:
            h["secret_key"] = self.cfg.api.secret_key
        return h

    @staticmethod
    def build_payload(*, camera_mac: str, vehicle_number: str, anpr_image_path: str,
                      detection_time: int, whitelisted: bool = False) -> dict:
        payload = {
            "cameraId": camera_mac,
            "vehicleNumber": vehicle_number,
            "anprImagePath": anpr_image_path,
            "detectionTime": int(detection_time),
            "status": "online",
        }
        if whitelisted:
            # Whitelisted vehicles skip the registry check the cloud would otherwise run.
            payload["disableVahanVerification"] = True
        return payload

    # ---------------- step 1 ----------------
    def create_triplog(self, payload: dict) -> PortalResult:
        """Post the detection. Retries only when there was no reply at all."""
        vehicle = payload.get("vehicleNumber", "?")

        if self.cfg.dry_run:
            logger.info(f"[dry-run] RSI step-1 {json.dumps(payload)}")
            return PortalResult(True, True, True, 200, {"dry_run": True}, "dry-run", payload, True)

        body_json = json.dumps(payload)
        attempts = max(1, self.cfg.api.remote_retries)
        for attempt in range(1, attempts + 1):
            logger.info(f"RSI step-1 request {vehicle} attempt={attempt}/{attempts} {body_json}")
            try:
                r = requests.post(self._url(), headers=self._headers(), data=body_json,
                                  timeout=self.cfg.api.remote_timeout)
            except requests.RequestException as e:
                logger.error(f"RSI step-1 unreachable for {vehicle} "
                             f"(attempt {attempt}/{attempts}): {e}")
                if attempt < attempts:
                    time.sleep(self.cfg.api.remote_retry_delay)
                    continue
                return PortalResult(False, False, False, None, None, str(e), payload, False)

            try:
                body = r.json()
            except Exception:
                body = r.text

            success = False
            if r.status_code == 200 and isinstance(body, dict):
                success = (body.get("error") is False
                           and body.get("message", "") == self.cfg.api.success_message)
            msg = body.get("message", "") if isinstance(body, dict) else str(body)

            if success:
                logger.info(f"RSI step-1 accepted {vehicle}: {msg}")
            else:
                logger.warning(f"RSI step-1 non-success for {vehicle} "
                               f"(HTTP {r.status_code}): {msg}")
            # A reply of any kind is final — the caller carries on to steps 2 and 3.
            return PortalResult(success, success, True, r.status_code, body, msg, payload, True)

        return PortalResult(False, False, False, None, None, "no attempts made", payload, False)

    # ---------------- step 3 ----------------
    def send_enriched(self, payload: dict, anpr_details: dict) -> bool:
        """Re-post the detection with the local API's result attached. Best-effort.

        Never raises and is never queued for retry: by the time it runs the vehicle is
        already recorded, and a stale enrichment arriving hours later would be misleading.
        """
        vehicle = payload.get("vehicleNumber", "?")
        enriched = {
            **payload,
            "type": "rsi-local-api",
            "rsi_request": {"url": anpr_details.get("url"),
                            "body": anpr_details.get("request_body")},
            "rsi_response": anpr_details.get("response"),
            "rsi_request_time": anpr_details.get("request_time"),
            "rsi_response_time": anpr_details.get("response_time"),
        }
        if self.cfg.dry_run:
            logger.info(f"[dry-run] RSI step-3 {json.dumps(enriched)}")
            return True
        try:
            body_json = json.dumps(enriched)
            logger.info(f"RSI step-3 request {vehicle} {body_json}")
            r = requests.post(self._url(), headers=self._headers(), data=body_json,
                              timeout=self.cfg.api.remote_timeout)
            logger.info(f"RSI step-3 response {vehicle} HTTP {r.status_code} {r.text}")
            return r.status_code == 200
        except Exception as e:
            logger.warning(f"RSI step-3 failed for {vehicle}: {e}")
            return False
