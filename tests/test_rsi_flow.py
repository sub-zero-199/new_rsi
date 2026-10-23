"""The three-call RSI flow at the service level.

    step 1  RSI cloud            — skipped when we already know we are offline
    step 2  local ANPR Data API  — always, and it carries the plate image
    step 3  RSI cloud, enriched  — inline when online, deferred to the uploader when not

The rule that shapes all of it: the vehicle physically passed the gate, so a refusal from
the cloud must not stop us telling the local server, and a cloud outage must not lose the
event. Something has to accept it, or it goes to the outbox.
"""
import json
import time

import pytest

from anpr.service import AnprService


class _Portal:
    """Stands in for the RSI cloud, recording the order calls arrive in."""

    def __init__(self, step1=None):
        self.step1_result = step1
        self.step1_calls, self.step3_calls, self.order = [], [], []

    # the real build_payload is a staticmethod with no I/O — reuse it verbatim
    from anpr.portal import PortalClient as _P
    build_payload = staticmethod(_P.build_payload)

    def create_triplog(self, payload):
        self.step1_calls.append(payload)
        self.order.append("step1")
        return self.step1_result

    def send_enriched(self, payload, details):
        self.step3_calls.append((payload, details))
        self.order.append("step3")
        return True


class _LocalApi:
    def __init__(self, success=True):
        self.success, self.calls, self.order = success, [], None

    def send(self, *, camera_mac, vehicle_number, plate_img, status="online"):
        self.calls.append({"vehicle": vehicle_number, "status": status,
                           "had_image": plate_img is not None})
        if self.order is not None:
            self.order.append("step2")
        return {"success": self.success, "url": "http://local/api/ANPRData",
                "request_body": {"cameraId": camera_mac, "vehicleNumber": vehicle_number,
                                 "status": status},
                "response": {"error": not self.success,
                             "message": "saved" if self.success else "refused",
                             "statuscode": 200 if self.success else 0},
                "request_time": "t0", "response_time": "t1"}


def _result(**kw):
    from anpr.portal import PortalResult
    base = dict(success=True, store=True, reachable=True, status_code=200,
                body={"error": False}, message="ok", payload={}, finalized=True)
    base.update(kw)
    return PortalResult(**base)


def _svc(cfg, tmp_path, monkeypatch, *, step1=None, local_ok=True, online=True):
    monkeypatch.chdir(tmp_path)
    cfg.dry_run = False
    cfg.s3.enabled = False
    svc = AnprService(cfg)
    svc.portal = _Portal(step1 if step1 is not None else _result())
    svc.anpr_api = _LocalApi(local_ok)
    svc.portal.order = svc.anpr_api.order = []      # one shared list, to assert ordering
    svc.anpr_api.order = svc.portal.order
    svc.internet.check = lambda force=False: online
    svc.internet.is_online = lambda: online
    return svc


def _queued(svc):
    return [data for _k, data in svc.queue.get_pending_detections(10)]


# ---- online: all three, in order ----

def test_online_runs_all_three_calls_in_order(cfg, img, frame, tmp_path, monkeypatch, make_finalized):
    svc = _svc(cfg, tmp_path, monkeypatch)

    svc._handle_finalized(make_finalized("MH12AB1234", "gate_in", int(time.time()), img, frame))

    assert svc.portal.order == ["step1", "step2", "step3"]
    assert svc.anpr_api.calls[0]["status"] == "online"
    assert svc.anpr_api.calls[0]["had_image"] is True      # step 2 is the one with the image
    assert len(svc.store.events_for_point("gate_in")) == 1


def test_the_image_path_sent_to_the_cloud_matches_the_s3_layout(cfg, img, frame, tmp_path,
                                                                monkeypatch, make_finalized):
    svc = _svc(cfg, tmp_path, monkeypatch)
    t = 1_700_000_000

    svc._handle_finalized(make_finalized("MH12AB1234", "gate_out", t, img, frame))

    assert svc.portal.step1_calls[0]["anprImagePath"] == f"ANPR/gateout/MH12AB1234/{t}/plate.jpg"


def test_a_cloud_rejection_still_reaches_the_local_server(cfg, img, frame, tmp_path,
                                                          monkeypatch, make_finalized):
    """The truck passed the gate. A 'Vehicle ID is Required!' from the cloud does not undo
    that, so steps 2 and 3 must still run and the event must still be recorded."""
    svc = _svc(cfg, tmp_path, monkeypatch,
               step1=_result(success=False, store=False, message="Vehicle ID is Required!"))

    svc._handle_finalized(make_finalized("MH12AB1234", "gate_in", int(time.time()), img, frame))

    assert svc.portal.order == ["step1", "step2", "step3"]
    assert len(svc.store.events_for_point("gate_in")) == 1
    assert svc.store.pending_outbox(10) == []              # local accepted it, nothing to retry


# ---- offline / unreachable: step 1 skipped, step 3 deferred ----

def test_offline_skips_step1_and_defers_step3(cfg, img, frame, tmp_path, monkeypatch, make_finalized):
    svc = _svc(cfg, tmp_path, monkeypatch, online=False)
    t = 1_700_000_000

    svc._handle_finalized(make_finalized("MH12AB1234", "gate_in", t, img, frame))

    assert svc.portal.step1_calls == []                    # no point spending its timeouts
    assert svc.portal.step3_calls == []                    # deferred, not sent now
    assert svc.anpr_api.calls[0]["status"] == "offline"

    entry = _queued(svc)[0]
    assert json.loads(entry["rsi_payload"])["detectionTime"] == t
    assert json.loads(entry["anpr_details"])["response"]["message"] == "saved"


def test_an_unreachable_cloud_behaves_like_offline(cfg, img, frame, tmp_path, monkeypatch,
                                                   make_finalized):
    """A timeout means nobody knows anything — fall back to local and defer the cloud."""
    svc = _svc(cfg, tmp_path, monkeypatch,
               step1=_result(success=False, store=False, reachable=False, finalized=False,
                             status_code=None, body=None, message="timeout"))

    svc._handle_finalized(make_finalized("MH12AB1234", "gate_in", int(time.time()), img, frame))

    assert len(svc.portal.step1_calls) == 1                # tried
    assert svc.portal.step3_calls == []                    # then deferred
    assert svc.anpr_api.calls[0]["status"] == "offline"
    assert "rsi_payload" in _queued(svc)[0]
    assert len(svc.store.events_for_point("gate_in")) == 1  # local accepted -> recorded


def test_online_does_not_defer(cfg, img, frame, tmp_path, monkeypatch, make_finalized):
    svc = _svc(cfg, tmp_path, monkeypatch)

    svc._handle_finalized(make_finalized("MH12AB1234", "gate_in", int(time.time()), img, frame))

    entry = _queued(svc)[0]
    assert "rsi_payload" not in entry or not entry.get("rsi_payload")
    assert len(svc.portal.step3_calls) == 1                # already sent inline


# ---- nothing accepted it ----

def test_when_both_cloud_and_local_refuse_the_event_goes_to_the_outbox(cfg, img, frame, tmp_path,
                                                                       monkeypatch, make_finalized):
    """rsi_current had no answer for this case and simply dropped the vehicle."""
    svc = _svc(cfg, tmp_path, monkeypatch,
               step1=_result(success=False, store=False, reachable=False, finalized=False),
               local_ok=False)

    svc._handle_finalized(make_finalized("MH12AB1234", "gate_in", int(time.time()), img, frame))

    assert len(svc.store.pending_outbox(10)) == 1
    assert svc.store.events_for_point("gate_in") == []     # not recorded as done
    assert _queued(svc) == []                              # and not queued for upload


def test_a_local_failure_alone_does_not_lose_the_event(cfg, img, frame, tmp_path, monkeypatch,
                                                       make_finalized):
    """The cloud took it, so the trip exists even though the local server was down."""
    svc = _svc(cfg, tmp_path, monkeypatch, local_ok=False)

    svc._handle_finalized(make_finalized("MH12AB1234", "gate_in", int(time.time()), img, frame))

    assert len(svc.store.events_for_point("gate_in")) == 1
    assert svc.store.pending_outbox(10) == []
    # step 3 still goes, carrying the local failure so the cloud hears about it
    assert svc.portal.step3_calls[0][1]["response"]["error"] is True


# ---- whitelist ----

def test_a_whitelisted_vehicle_tells_the_cloud_to_skip_vahan(cfg, img, frame, tmp_path,
                                                             monkeypatch, make_finalized):
    svc = _svc(cfg, tmp_path, monkeypatch)
    svc.whitelist.is_whitelisted = lambda plate: True

    svc._handle_finalized(make_finalized("MH12AB1234", "gate_in", int(time.time()), img, frame))

    assert svc.portal.step1_calls[0]["disableVahanVerification"] is True


# ---- outbox: a vehicle waiting at the point is queued once, not once per read ----

def _both_down(cfg, tmp_path, monkeypatch):
    return _svc(cfg, tmp_path, monkeypatch,
                step1=_result(success=False, store=False, reachable=False, finalized=False),
                local_ok=False)


def test_repeat_reads_while_both_down_are_queued_once(cfg, img, frame, tmp_path, monkeypatch,
                                                      make_finalized):
    svc = _both_down(cfg, tmp_path, monkeypatch)
    t = 1_700_000_000
    for dt in (0, 12, 25):                                 # truck sits at the gate, re-read
        svc._handle_finalized(make_finalized("MH12AB1234", "gate_in", t + dt, img, frame))

    assert len(svc.store.pending_outbox(10)) == 1
    # the repeats' images are not left behind in the upload staging folder
    assert not (tmp_path / "temp_uploads" / "gate_in" / "MH12AB1234" / str(t + 12)).joinpath(
        "plate.jpg").exists()


def test_different_plates_are_never_merged(cfg, img, frame, tmp_path, monkeypatch, make_finalized):
    svc = _both_down(cfg, tmp_path, monkeypatch)
    t = 1_700_000_000
    svc._handle_finalized(make_finalized("MH12AB1234", "gate_in", t, img, frame))
    svc._handle_finalized(make_finalized("MH12AB1235", "gate_in", t + 5, img, frame))
    assert len(svc.store.pending_outbox(10)) == 2


def test_a_genuine_second_weighing_while_down_is_kept(cfg, img, frame, tmp_path, monkeypatch,
                                                      make_finalized):
    svc = _both_down(cfg, tmp_path, monkeypatch)
    t = 1_700_000_000
    gap = cfg.cooldown.weighbridge_second_gap_seconds
    svc._handle_finalized(make_finalized("MH12AB1234", "weighbridge", t, img, frame))
    svc._handle_finalized(make_finalized("MH12AB1234", "weighbridge", t + gap + 60, img, frame))
    assert len(svc.store.pending_outbox(10)) == 2


def test_outbox_sends_once_and_drops_a_repeat_on_recovery(cfg, img, frame, tmp_path, monkeypatch,
                                                          make_finalized):
    """Past the window the repeat is queued; the send-time check still stops a second trip."""
    svc = _both_down(cfg, tmp_path, monkeypatch)
    t = int(time.time()) - 3600
    svc._handle_finalized(make_finalized("MH12AB1234", "gate_in", t, img, frame))
    svc._handle_finalized(make_finalized("MH12AB1234", "gate_in",
                                         t + cfg.cooldown.reentry_gap_seconds + 60, img, frame))
    assert len(svc.store.pending_outbox(10)) == 2

    svc.portal.step1_result = _result()                    # connectivity is back
    svc._drain_outbox()

    assert len(svc.portal.step1_calls) == 3                # 2 failed live tries + 1 send
    assert len(svc.store.events_for_point("gate_in")) == 1  # one trip, not two
    assert svc.store.pending_outbox(10) == []
