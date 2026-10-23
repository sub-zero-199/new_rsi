import time
from anpr.storage_sqlite import SqliteStore
from anpr.tracker import SharedTracker
from anpr.journey import PointProcessor

P = "MH12AB1234"


def _proc(cfg):
    store = SqliteStore(cfg.sqlite.path)
    tracker = SharedTracker()  # no redis -> local fallback
    return PointProcessor(cfg, store, tracker), store


def test_gate_in_then_cooldown(cfg):
    pp, store = _proc(cfg)
    t = int(time.time())
    assert pp.evaluate("gate_in", P, t).allow
    pp.record_success(point="gate_in", plate=P, plate_raw=P, ts=t, camera_id="c", s3_key="k")
    # within 60s same-place cooldown -> discard
    assert not pp.evaluate("gate_in", P, t + 30).allow


def test_weighbridge_second_gap(cfg):
    pp, store = _proc(cfg)
    t = int(time.time())
    pp.record_success(point="weighbridge", plate=P, plate_raw=P, ts=t, camera_id="c", s3_key="k")
    assert not pp.evaluate("weighbridge", P, t + 300).allow   # < 600s
    assert pp.evaluate("weighbridge", P, t + 800).allow       # > 600s


def test_gate_out_reexit_and_journey_complete(cfg):
    pp, store = _proc(cfg)
    t = int(time.time())
    pp.record_success(point="gate_in", plate=P, plate_raw=P, ts=t, camera_id="c", s3_key="k")
    assert pp.evaluate("gate_out", P, t + 900).allow          # >300s dwell
    pp.record_success(point="gate_out", plate=P, plate_raw=P, ts=t + 900, camera_id="c", s3_key="k")
    assert not pp.evaluate("gate_out", P, t + 1000).allow     # re-exit <300s
    assert store.active_journey(P) is None                    # completed


def test_mid_flow_join_flagged(cfg):
    pp, store = _proc(cfg)
    t = int(time.time())
    d = pp.evaluate("gate_out", "KA05CD9999", t)
    assert d.allow and d.continuity_status == "mid_flow"


def test_duplicate_gate_in_skipped_while_inside(cfg):
    pp, store = _proc(cfg)
    t = int(time.time())
    pp.record_success(point="gate_in", plate=P, plate_raw=P, ts=t, camera_id="c", s3_key="k")
    # inside -> another gate-in well past the 60s same-place cooldown is still skipped
    assert not pp.evaluate("gate_in", P, t + 120).allow


def test_duplicate_gate_out_skipped_when_already_out(cfg):
    pp, store = _proc(cfg)
    t = int(time.time())
    pp.record_success(point="gate_in", plate=P, plate_raw=P, ts=t, camera_id="c", s3_key="k")
    pp.record_success(point="gate_out", plate=P, plate_raw=P, ts=t + 900, camera_id="c", s3_key="k")
    # already out; a later gate-out past the re-exit gap is still skipped as a duplicate
    assert not pp.evaluate("gate_out", P, t + 2000).allow


def test_weighbridge_after_gateout_reestablishes_inside(cfg):
    pp, store = _proc(cfg)
    t = int(time.time())
    # vehicle exits, then later shows at the weighbridge -> it's on the premises again
    pp.record_success(point="gate_out", plate=P, plate_raw=P, ts=t, camera_id="c", s3_key="k")
    pp.record_success(point="weighbridge", plate=P, plate_raw=P, ts=t + 1000, camera_id="c", s3_key="k")
    # so a subsequent gate-out must be ALLOWED (not "already out"), since WB came after the exit
    assert pp.evaluate("gate_out", P, t + 2000).allow


def test_weighbridge_after_gateout_default_allows_immediately(cfg):
    # default gap is 0 -> a weighbridge read right after gate-out is allowed
    assert cfg.cooldown.weighbridge_after_gateout_gap_seconds == 0
    pp, store = _proc(cfg)
    t = int(time.time())
    pp.record_success(point="gate_out", plate=P, plate_raw=P, ts=t, camera_id="c", s3_key="k")
    assert pp.evaluate("weighbridge", P, t + 5).allow


def test_weighbridge_after_gateout_gap_configurable(cfg):
    cfg.cooldown.weighbridge_after_gateout_gap_seconds = 120
    pp, store = _proc(cfg)
    t = int(time.time())
    pp.record_success(point="gate_out", plate=P, plate_raw=P, ts=t, camera_id="c", s3_key="k")
    assert not pp.evaluate("weighbridge", P, t + 60).allow   # within the 120s gap -> discard
    assert pp.evaluate("weighbridge", P, t + 200).allow      # past the gap -> allow
