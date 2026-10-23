"""Disk retention janitor — keeps generated data bounded on the workstation.

Two rules, applied over the managed dirs (logs, debug images, temp-upload staging):

  1. AGE  — delete files older than `max_age_days` (default 14) by mtime.
  2. SIZE — if the managed dirs together still exceed `max_total_gb` (default 200 GB),
            delete the OLDEST files first until back under the cap.

The currently-open ``*.log`` files are never deleted (a rotating handler owns them; their
older days are dropped by the handler + the AGE rule on rotated backups). Runs in a daemon
thread every `check_interval_seconds`, plus one sweep ~30s after boot.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Iterable

from anpr.log import logger

_GB = 1024 ** 3


class RetentionJanitor:
    def __init__(self, dirs: Iterable[str], max_age_days: int = 14,
                 max_total_gb: float = 200.0, check_interval_seconds: int = 3600):
        # de-dup + keep only truthy, existing-or-future dirs
        self.dirs = list(dict.fromkeys(d for d in dirs if d))
        self.max_age_seconds = max(1, int(max_age_days)) * 86400
        self.max_total_bytes = int(max(0.0, float(max_total_gb)) * _GB)
        self.interval = max(60, int(check_interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------------- lifecycle ----------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="retention", daemon=True)
        self._thread.start()
        logger.info(f"retention janitor: {self.max_age_seconds // 86400}d age / "
                    f"{self.max_total_bytes / _GB:.0f} GB cap over {self.dirs} "
                    f"(every {self.interval}s)")

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        if self._stop.wait(30):  # let the service settle before the first sweep
            return
        while True:
            try:
                self.sweep(time.time())
            except Exception as e:
                logger.error(f"retention sweep failed: {e}")
            if self._stop.wait(self.interval):
                return

    # ---------------- sweep ----------------
    def sweep(self, now: float) -> None:
        n_age, freed_age = self._prune_age(now)
        files = list(self._iter_files())
        total = sum(sz for _, sz, _ in files)
        n_size, freed_size = self._enforce_size(files, total)
        self._prune_empty_dirs()
        if n_age or n_size:
            logger.info(
                f"retention: removed {n_age} aged + {n_size} over-cap files, "
                f"freed {(freed_age + freed_size) / _GB:.2f} GB; total now "
                f"{self._total() / _GB:.2f}/{self.max_total_bytes / _GB:.0f} GB")

    @staticmethod
    def _is_protected(path: str) -> bool:
        # never delete an active log file the rotating handler is writing to
        return path.endswith(".log")

    def _iter_files(self):
        for d in self.dirs:
            if not os.path.isdir(d):
                continue
            for root, _dirs, files in os.walk(d):
                for fn in files:
                    p = os.path.join(root, fn)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    yield p, st.st_size, st.st_mtime

    def _prune_age(self, now: float) -> tuple[int, int]:
        n = freed = 0
        for p, sz, mt in list(self._iter_files()):
            if self._is_protected(p):
                continue
            if now - mt > self.max_age_seconds and self._rm(p):
                n += 1
                freed += sz
        return n, freed

    def _enforce_size(self, files, total: int) -> tuple[int, int]:
        n = freed = 0
        if self.max_total_bytes <= 0 or total <= self.max_total_bytes:
            return n, freed
        for p, sz, _mt in sorted(files, key=lambda t: t[2]):  # oldest first
            if total <= self.max_total_bytes:
                break
            if self._is_protected(p):
                continue
            if self._rm(p):
                n += 1
                freed += sz
                total -= sz
        return n, freed

    def _total(self) -> int:
        return sum(sz for _, sz, _ in self._iter_files())

    @staticmethod
    def _rm(path: str) -> bool:
        try:
            os.remove(path)
            return True
        except OSError as e:
            logger.warning(f"retention: could not delete {path}: {e}")
            return False

    def _prune_empty_dirs(self) -> None:
        for d in self.dirs:
            if not os.path.isdir(d):
                continue
            for root, _dirs, _files in os.walk(d, topdown=False):
                if os.path.abspath(root) == os.path.abspath(d):
                    continue  # keep the managed root itself
                try:
                    if not os.listdir(root):
                        os.rmdir(root)
                except OSError:
                    pass
