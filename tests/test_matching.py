from anpr.storage_sqlite import SqliteStore
from anpr.whitelist import WhitelistManager
from anpr.doppelganger import DoppelgangerManager
from anpr.matching import PlateMatcher


def _matcher(cfg, known=()):
    store = SqliteStore(cfg.sqlite.path)
    for p in known:
        store.upsert_known_plate(p)
    wl = WhitelistManager(cfg.whitelist_path)
    dg = DoppelgangerManager(cfg.doppelganger_path)
    return PlateMatcher(cfg, store, wl, dg), store


def test_local_match_first(cfg, img, frame, make_finalized):
    m, _ = _matcher(cfg, known=["MH12AB1234"])
    r = m.resolve(make_finalized("MH12A81234", "gate_in", 1, img, frame))
    assert r.matched and r.source == "local" and r.resolved == "MH12AB1234"


def test_new_vehicle_not_matched(cfg, img, frame, make_finalized):
    m, _ = _matcher(cfg, known=["MH12AB1234"])
    r = m.resolve(make_finalized("TN09XY4321", "gate_in", 1, img, frame))
    assert not r.matched and r.source is None
