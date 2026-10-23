import time
from anpr.service import AnprService


def _svc(cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # isolate temp_uploads / local_queue_* into tmp
    svc = AnprService(cfg)
    return svc


def test_new_vehicle_pushed_and_recorded(cfg, img, frame, tmp_path, monkeypatch, make_finalized):
    svc = _svc(cfg, tmp_path, monkeypatch)
    t = int(time.time())
    svc._handle_finalized(make_finalized("MH12AB1234", "gate_in", t, img, frame))
    assert svc.store.all_known_plates() == ["MH12AB1234"]
    assert (svc.store.active_journey("MH12AB1234") or {}).get("current_point") == "gate_in"
    assert svc.queue.get_queue_size() >= 1
    assert len(svc.store.events_for_point("gate_in")) == 1


def test_gate_in_not_repushed_while_inside(cfg, img, frame, tmp_path, monkeypatch, make_finalized):
    svc = _svc(cfg, tmp_path, monkeypatch)
    t = int(time.time())
    svc._handle_finalized(make_finalized("MH12AB1234", "gate_in", t, img, frame))
    # already inside -> a second gate-in (well past the 60s cooldown) is skipped, no dup trip
    svc._handle_finalized(make_finalized("MH12AB1234", "gate_in", t + 120, img, frame))
    assert len(svc.store.events_for_point("gate_in")) == 1


def test_gate_out_recorded_then_duplicate_skipped(cfg, img, frame, tmp_path, monkeypatch, make_finalized):
    svc = _svc(cfg, tmp_path, monkeypatch)
    t = int(time.time())
    svc._handle_finalized(make_finalized("MH12AB1234", "gate_in", t, img, frame))
    svc._handle_finalized(make_finalized("MH12AB1234", "gate_out", t + 900, img, frame))
    assert len(svc.store.events_for_point("gate_out")) == 1
    assert svc.store.active_journey("MH12AB1234") is None  # journey completed
    # already out -> a later gate-out (past the re-exit gap) is still skipped as a duplicate
    svc._handle_finalized(make_finalized("MH12AB1234", "gate_out", t + 2000, img, frame))
    assert len(svc.store.events_for_point("gate_out")) == 1


def test_invalid_plate_skipped(cfg, img, frame, tmp_path, monkeypatch, make_finalized):
    svc = _svc(cfg, tmp_path, monkeypatch)
    svc._handle_finalized(make_finalized("ZZ99!!", "gate_in", int(time.time()), img, frame))
    assert svc.store.all_known_plates() == []


def test_slow_api_does_not_block_the_gpu_worker(cfg, img, frame, tmp_path, monkeypatch, make_finalized):
    svc = _svc(cfg, tmp_path, monkeypatch)
    handled = []

    def slow_handle(det):
        time.sleep(1.0)  # stands in for a slow/unreachable RSI cloud
        handled.append(det.best_text)

    monkeypatch.setattr(svc, "_handle_finalized", slow_handle)
    import threading
    svc._decision_thread = threading.Thread(target=svc._decision_loop, daemon=True)
    svc._decision_thread.start()

    t0 = time.time()
    for plate in ("MH12AB1234", "KA05CD9999"):
        svc._on_finalized(make_finalized(plate, "gate_in", int(t0), img, frame))
    assert time.time() - t0 < 0.2          # the GPU thread handed off and moved on

    svc._decisions.put(None)
    svc._decision_thread.join(timeout=5)
    assert handled == ["MH12AB1234", "KA05CD9999"]  # processed in order, none lost
