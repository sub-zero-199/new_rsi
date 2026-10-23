"""The RSI cloud calls — step 1 (the detection) and step 3 (the enrichment).

The distinction that matters everywhere else in the service: a REPLY of any kind is final
and stops the retrying, while a timeout means nobody knows anything yet and is worth
another go. Step 1 returns both facts separately (`success` and `reachable`) because the
caller treats them very differently.
"""
import json

import pytest
import requests

from anpr.portal import PortalClient


def _url(cfg):
    return cfg.api.remote_base_url.rstrip("/") + cfg.api.rsi_endpoint


def _payload(cfg, **kw):
    base = dict(camera_mac="90f7", vehicle_number="MH12AB1234",
                anpr_image_path="ANPR/gateout/MH12AB1234/1/plate.jpg", detection_time=1)
    base.update(kw)
    return PortalClient.build_payload(**base)


# ---- the payload the cloud receives ----

def test_payload_carries_the_agreed_fields(cfg):
    p = _payload(cfg)
    assert p == {"cameraId": "90f7", "vehicleNumber": "MH12AB1234",
                 "anprImagePath": "ANPR/gateout/MH12AB1234/1/plate.jpg",
                 "detectionTime": 1, "status": "online"}


def test_a_whitelisted_vehicle_skips_the_registry_check(cfg):
    assert _payload(cfg, whitelisted=True)["disableVahanVerification"] is True
    assert "disableVahanVerification" not in _payload(cfg)


# ---- step 1 ----

def test_step1_success_requires_the_exact_message(cfg, requests_mock):
    """Any other 200 body is a rejection, not an acceptance."""
    cfg.dry_run = False
    requests_mock.post(_url(cfg), json={"error": False, "message": cfg.api.success_message})

    r = PortalClient(cfg).create_triplog(_payload(cfg))

    assert r.success and r.store and r.reachable and r.finalized
    assert requests_mock.last_request.json()["vehicleNumber"] == "MH12AB1234"


def test_step1_sends_the_secret_key_header(cfg, requests_mock):
    cfg.dry_run = False
    cfg.api.secret_key = "s3cr3t"
    requests_mock.post(_url(cfg), json={"error": False, "message": cfg.api.success_message})

    PortalClient(cfg).create_triplog(_payload(cfg))

    assert requests_mock.last_request.headers["secret_key"] == "s3cr3t"


def test_a_business_rejection_is_final_and_never_retried(cfg, requests_mock):
    """The cloud answered. Retrying cannot change its mind, and the flow still has to carry
    on to the local server — the vehicle did pass the gate."""
    cfg.dry_run = False
    requests_mock.post(_url(cfg), json={"error": True, "message": "Vehicle ID is Required!"})

    r = PortalClient(cfg).create_triplog(_payload(cfg))

    assert not r.success
    assert r.reachable and r.finalized          # answered -> no retry
    assert requests_mock.call_count == 1
    assert "Vehicle ID is Required!" in r.message


def test_an_http_error_is_also_a_reply_so_it_is_not_retried(cfg, requests_mock):
    cfg.dry_run = False
    requests_mock.post(_url(cfg), status_code=500, text="gateway down")

    r = PortalClient(cfg).create_triplog(_payload(cfg))

    assert not r.success and r.reachable and r.finalized
    assert requests_mock.call_count == 1


def test_a_timeout_retries_then_reports_unreachable(cfg, requests_mock):
    """No reply means nobody knows anything — that is the one case worth repeating."""
    cfg.dry_run = False
    cfg.api.remote_retries = 3
    cfg.api.remote_retry_delay = 0
    requests_mock.post(_url(cfg), exc=requests.exceptions.ConnectTimeout)

    r = PortalClient(cfg).create_triplog(_payload(cfg))

    assert not r.success
    assert not r.reachable                       # -> caller falls back to the local API
    assert not r.finalized                       # -> eligible for the outbox
    assert requests_mock.call_count == 3


def test_dry_run_sends_nothing(cfg):
    cfg.dry_run = True
    r = PortalClient(cfg).create_triplog(_payload(cfg))
    assert r.success and r.store and r.reachable


# ---- step 3 ----

_DETAILS = {
    "url": "http://192.168.1.153:5000/api/ANPRData",
    "request_body": {"cameraId": "90f7", "vehicleNumber": "MH12AB1234", "status": "offline"},
    "response": {"error": True, "message": "connection refused", "statuscode": 0},
    "request_time": "2026-09-23T10:00:00.000Z",
    "response_time": "2026-09-23T10:00:10.000Z",
}


def test_step3_reposts_step1s_payload_with_the_local_result_attached(cfg, requests_mock):
    cfg.dry_run = False
    requests_mock.post(_url(cfg), json={"error": False})
    payload = _payload(cfg)

    assert PortalClient(cfg).send_enriched(payload, _DETAILS)

    sent = requests_mock.last_request.json()
    assert sent["type"] == "rsi-local-api"
    for k, v in payload.items():
        assert sent[k] == v                      # the original detection, unchanged
    assert sent["rsi_request"] == {"url": _DETAILS["url"], "body": _DETAILS["request_body"]}
    assert sent["rsi_response"] == _DETAILS["response"]
    assert sent["rsi_request_time"] == _DETAILS["request_time"]


def test_step3_reports_a_local_failure_rather_than_hiding_it(cfg, requests_mock):
    """A failed local call is exactly what the cloud needs to hear about."""
    cfg.dry_run = False
    requests_mock.post(_url(cfg), json={"error": False})

    PortalClient(cfg).send_enriched(_payload(cfg), _DETAILS)

    assert requests_mock.last_request.json()["rsi_response"]["error"] is True


def test_step3_keeps_the_original_detection_time(cfg, requests_mock):
    """A deferred enrichment must date the trip to when the vehicle passed, not to now."""
    cfg.dry_run = False
    requests_mock.post(_url(cfg), json={"error": False})

    PortalClient(cfg).send_enriched(_payload(cfg, detection_time=1_700_000_000), _DETAILS)

    assert requests_mock.last_request.json()["detectionTime"] == 1_700_000_000


def test_step3_never_raises(cfg, requests_mock):
    """It is best-effort: the vehicle is already recorded by the time it runs."""
    cfg.dry_run = False
    requests_mock.post(_url(cfg), exc=requests.exceptions.ConnectTimeout)

    assert PortalClient(cfg).send_enriched(_payload(cfg), _DETAILS) is False
