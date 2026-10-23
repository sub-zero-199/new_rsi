"""Internet connectivity manager (cached). Ported/simplified from v11_1 internet_check.py.

Decides online/offline so the portal client knows whether to try the remote API or go
straight to the local fallback. DNS check + a HEAD to the portal, with short timeouts and
a small result cache so we don't probe on every detection.
"""
from __future__ import annotations

import socket
import threading
import time

import requests

from anpr.log import logger


class InternetManager:
    def __init__(self, api_endpoint: str = "https://cwc.lynkid.io", timeout: float = 1.0):
        self.api_endpoint = api_endpoint
        self.timeout = timeout
        self._online = False
        self._last_check = 0.0
        self._failed = 0
        self._lock = threading.Lock()
        self.cache_online = 5.0
        self.cache_offline = 1.0

    def is_online(self) -> bool:
        return self._online

    def check(self, force: bool = False) -> bool:
        now = time.time()
        with self._lock:
            window = self.cache_online if self._online else self.cache_offline
            if not force and (now - self._last_check) < window:
                return self._online
            self._last_check = now

        online = self._probe()
        with self._lock:
            if online:
                if not self._online:
                    logger.info("Internet connectivity restored")
                self._online, self._failed = True, 0
            else:
                self._failed += 1
                if self._online and self._failed >= 2:
                    logger.warning("Internet connectivity lost")
                    self._online = False
                elif self._failed >= 2:
                    self._online = False
        return self._online

    def _probe(self) -> bool:
        try:
            socket.gethostbyname("google.com")
        except Exception:
            return False
        try:
            r = requests.head(self.api_endpoint, timeout=self.timeout)
            return r.status_code < 500
        except Exception:
            pass
        for url in ("http://www.google.com", "http://1.1.1.1"):
            try:
                r = requests.get(url, timeout=self.timeout)
                if r.status_code in (200, 204, 301, 302, 429):
                    return True
            except Exception:
                continue
        return False
