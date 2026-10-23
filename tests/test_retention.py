"""Retention janitor: age-prune + size-cap, protecting active .log files."""
import os
import time

from anpr.retention import RetentionJanitor


def _write(path: str, size: int, mtime: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size)
    os.utime(path, (mtime, mtime))


def test_age_prune_deletes_old_keeps_recent_and_protects_logs(tmp_path):
    now = time.time()
    logs = tmp_path / "logs"
    debug = tmp_path / "debug"
    _write(str(debug / "old.jpg"), 10, now - 20 * 86400)      # 20 days old -> delete
    _write(str(debug / "fresh.jpg"), 10, now - 1 * 86400)     # 1 day old  -> keep
    _write(str(logs / "gi.log"), 10, now - 30 * 86400)        # old but active .log -> keep
    _write(str(logs / "gi.log.2026-01-01"), 10, now - 30 * 86400)  # rotated backup -> delete

    j = RetentionJanitor([str(logs), str(debug)], max_age_days=14, max_total_gb=200)
    j.sweep(now)

    assert not (debug / "old.jpg").exists()
    assert (debug / "fresh.jpg").exists()
    assert (logs / "gi.log").exists()                         # protected
    assert not (logs / "gi.log.2026-01-01").exists()


def test_size_cap_deletes_oldest_first(tmp_path):
    now = time.time()
    d = tmp_path / "debug"
    # 3 files, all within age; total 300 bytes, cap is 150 bytes -> must drop oldest two
    _write(str(d / "a.jpg"), 100, now - 300)   # oldest
    _write(str(d / "b.jpg"), 100, now - 200)
    _write(str(d / "c.jpg"), 100, now - 100)   # newest

    cap_gb = 150 / (1024 ** 3)
    j = RetentionJanitor([str(d)], max_age_days=14, max_total_gb=cap_gb)
    j.sweep(now)

    assert not (d / "a.jpg").exists()
    assert not (d / "b.jpg").exists()
    assert (d / "c.jpg").exists()              # newest survives under the cap
