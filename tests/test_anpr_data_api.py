"""Step 2 — the local ANPR Data API.

It is the only call carrying the plate image, and the only one that still works with the
internet down. Its result is embedded verbatim into step 3, so a failure has to come back
as data rather than as an exception: the cloud wants to know the local server was down.
"""
import base64
import json

import numpy as np
import pytest
import requests

from anpr.anpr_data_api import AnprDataApiClient


def _send(cfg, plate_img=None, status="online"):
    return AnprDataApiClient(cfg).send(camera_mac="90f7", vehicle_number="MH12AB1234",
                                       plate_img=plate_img, status=status)


def test_success(cfg, requests_mock, img):
    cfg.dry_run = False
    requests_mock.post(cfg.anpr_data_api.url, json={"error": False, "message": "saved"})

    res = _send(cfg, img)

    assert res["success"] is True
    assert res["response"] == {"error": False, "message": "saved", "statuscode": 200}
    assert res["url"] == cfg.anpr_data_api.url


def test_the_plate_image_is_sent_as_base64(cfg, requests_mock, img):
    cfg.dry_run = False
    requests_mock.post(cfg.anpr_data_api.url, json={"error": False})

    _send(cfg, img)

    sent = requests_mock.last_request.json()
    assert sent["image"]
    assert base64.b64decode(sent["image"])[:2] == b"\xff\xd8"   # a real JPEG


def test_the_echoed_request_body_leaves_the_image_out(cfg, requests_mock, img):
    """request_body is copied into every step-3 payload — a base64 frame there would bloat
    every cloud call for no benefit."""
    cfg.dry_run = False
    requests_mock.post(cfg.anpr_data_api.url, json={"error": False})

    res = _send(cfg, img)

    assert "image" not in res["request_body"]
    assert res["request_body"] == {"cameraId": "90f7", "vehicleNumber": "MH12AB1234",
                                   "status": "online"}


def test_a_missing_image_still_sends(cfg, requests_mock):
    cfg.dry_run = False
    requests_mock.post(cfg.anpr_data_api.url, json={"error": False})

    assert _send(cfg, None)["success"] is True
    assert requests_mock.last_request.json()["image"] == ""


# ---- failures come back as data, never as an exception ----

def test_a_rejection_is_reported_not_raised(cfg, requests_mock, img):
    cfg.dry_run = False
    requests_mock.post(cfg.anpr_data_api.url, json={"error": True, "message": "bad plate"})

    res = _send(cfg, img)

    assert res["success"] is False
    assert res["response"]["error"] is True and res["response"]["message"] == "bad plate"


def test_an_unreachable_server_is_reported_not_raised(cfg, requests_mock, img):
    """This is the case step 3 exists for — the cloud learns the local server was down."""
    cfg.dry_run = False
    requests_mock.post(cfg.anpr_data_api.url, exc=requests.exceptions.ConnectTimeout)

    res = _send(cfg, img, status="offline")

    assert res["success"] is False
    assert res["response"]["statuscode"] == 0
    assert res["request_body"]["status"] == "offline"
    assert res["request_time"] and res["response_time"]      # timings are always present


def test_disabled_is_a_reported_failure_not_a_silent_skip(cfg, img):
    cfg.dry_run = False
    cfg.anpr_data_api.enabled = False

    res = _send(cfg, img)

    assert res["success"] is False
    assert "disabled" in res["response"]["message"]


def test_dry_run_sends_nothing(cfg, img):
    cfg.dry_run = True
    res = _send(cfg, img)
    assert res["success"] is True and res["response"]["message"] == "dry-run"
