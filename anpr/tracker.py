"""Shared 24h vehicle tracker with the three buckets (gate_in / weighbridge / gate_out).

Ported from v11_1 shared_vehicle_tracker.py and collapsed to a single instance for the
unified process. Redis sorted-sets give fast 24h windowed lookups + the pending
auto-gate-out queue; a local-dict fallback keeps it working without redis-server.
"""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime
from typing import Optional

from anpr.log import logger

try:
    import redis as _redis
except Exception:
    _redis = None

_DAY = 86400
BUCKETS = ("gate_in", "weighbridge", "gate_out")


def canonical_plate(vehicle_number: str) -> str:
    if not vehicle_number:
        return ""
    return re.sub(r"[^A-Z0-9]", "", vehicle_number.upper())


class SharedTracker:
    def __init__(self, host="localhost", port=6379, db=1):
        self.use_redis = False
        self.redis = None
        if _redis is not None:
            try:
                self.redis = _redis.Redis(host=host, port=port, db=db, decode_responses=True,
                                          socket_connect_timeout=2, socket_timeout=2)
                self.redis.ping()
                self.use_redis = True
                logger.info(f"SharedTracker connected to Redis (db={db})")
            except Exception as e:
                logger.warning(f"SharedTracker Redis unavailable ({e}); local fallback")
        if not self.use_redis:
            self._local: dict[str, list[dict]] = {}
            self._local_lock = threading.Lock()

        self.bucket_24h_key = {
            "gate_in": "anpr:shared:gatein:24h",
            "gate_out": "anpr:shared:gateout:24h",
            "weighbridge": "anpr:shared:weighbridge:24h",
        }
        self.pending_auto_gateout_key = "anpr:shared:pending_auto_gateout"

    # ---------------- store ----------------
    def store_entry(self, bucket: str, vehicle_number: str, detection_time: int,
                    api_response=None, anpr_image_path: str = "") -> bool:
        canon = canonical_plate(vehicle_number)
        if not canon or bucket not in BUCKETS:
            return False
        entry = {
            "vehicle_number": vehicle_number, "canonical": canon,
            "detection_time": int(detection_time), "timestamp": datetime.now().isoformat(),
            "api_response": json.dumps(api_response) if isinstance(api_response, dict) else str(api_response or ""),
            "anpr_image_path": anpr_image_path, "service": bucket,
        }
        if self.use_redis:
            try:
                key = f"{bucket}:{canon}:{int(detection_time)}"
                self.redis.hset(key, mapping=entry)
                self.redis.zadd(self.bucket_24h_key[bucket], {key: detection_time})
                self.redis.expire(key, _DAY + 3600)
                if bucket == "gate_in":
                    self.redis.zadd(self.pending_auto_gateout_key, {key: detection_time + _DAY})
                elif bucket == "gate_out":
                    for gk in self._find_keys("gate_in", canon):
                        self.redis.zrem(self.pending_auto_gateout_key, gk)
                logger.info(f"Stored {bucket} entry: {vehicle_number}")
                return True
            except Exception as e:
                logger.error(f"Redis store {bucket} error: {e}")
                return self._store_local(canon, entry)
        return self._store_local(canon, entry)

    def _store_local(self, canon: str, entry: dict) -> bool:
        with self._local_lock:
            self._local.setdefault(canon, []).append(entry)
        return True

    # ---------------- check ----------------
    def check_in_last_24h(self, bucket: str, vehicle_number: str, threshold: float = 0.85) -> list[dict]:
        canon = canonical_plate(vehicle_number)
        if not canon:
            return []
        now = time.time()
        cutoff = now - _DAY
        if self.use_redis:
            try:
                matches = []
                for key in self.redis.zrangebyscore(self.bucket_24h_key[bucket], cutoff, now):
                    data = self.redis.hgetall(key)
                    if data and self._is_similar(canon, data.get("canonical", ""), threshold):
                        matches.append(self._row(data, bucket))
                return matches
            except Exception as e:
                logger.error(f"Redis check {bucket} error: {e}")
                return self._check_local(canon, bucket, cutoff, threshold)
        return self._check_local(canon, bucket, cutoff, threshold)

    def _check_local(self, canon, bucket, cutoff, threshold) -> list[dict]:
        out = []
        with self._local_lock:
            for stored_canon, entries in self._local.items():
                if self._is_similar(canon, stored_canon, threshold):
                    for e in entries:
                        if e.get("service") == bucket and int(e.get("detection_time", 0)) >= cutoff:
                            out.append(e)
        return out

    @staticmethod
    def _row(data: dict, bucket: str) -> dict:
        try:
            api = json.loads(data.get("api_response", "{}"))
        except Exception:
            api = data.get("api_response", "")
        return {
            "vehicle_number": data.get("vehicle_number"),
            "canonical": data.get("canonical"),
            "detection_time": int(data.get("detection_time", 0)),
            "api_response": api,
            "anpr_image_path": data.get("anpr_image_path", ""),
            "service": data.get("service", bucket),
        }

    @staticmethod
    def _is_similar(c1: str, c2: str, threshold: float = 0.85) -> bool:
        if not c1 or not c2:
            return False
        if c1 == c2:
            return True
        if len(c1) >= 4 and len(c2) >= 4 and c1[:2] != c2[:2]:
            return False
        if abs(len(c1) - len(c2)) > 2:
            return False
        max_len = max(len(c1), len(c2))
        if max_len == 0:
            return True
        matches = sum(a == b for a, b in zip(c1, c2))
        return (matches / max_len) >= threshold

    # ---------------- auto gate-out ----------------
    def _find_keys(self, bucket: str, canon: str) -> list[str]:
        if not self.use_redis:
            return []
        try:
            now = time.time()
            keys = self.redis.zrangebyscore(self.bucket_24h_key[bucket], now - _DAY, now)
            return [k for k in keys if self._is_similar(canon, self.redis.hgetall(k).get("canonical", ""))]
        except Exception:
            return []

    def get_pending_auto_gateouts(self) -> list[dict]:
        if not self.use_redis:
            return []
        try:
            now = time.time()
            out = []
            for key in self.redis.zrangebyscore(self.pending_auto_gateout_key, 0, now):
                data = self.redis.hgetall(key)
                if data:
                    out.append({"vehicle_number": data.get("vehicle_number"),
                                "canonical": data.get("canonical"),
                                "detection_time": int(data.get("detection_time", 0)),
                                "anpr_image_path": data.get("anpr_image_path", ""),
                                "key": key})
            return out
        except Exception as e:
            logger.error(f"pending auto-gateout error: {e}")
            return []

    def remove_pending_auto_gateout(self, key: str) -> None:
        if self.use_redis:
            try:
                self.redis.zrem(self.pending_auto_gateout_key, key)
            except Exception:
                pass

    def cleanup_old_entries(self) -> None:
        if not self.use_redis:
            return
        try:
            cutoff = time.time() - _DAY - 3600
            for k in self.bucket_24h_key.values():
                self.redis.zremrangebyscore(k, 0, cutoff)
        except Exception as e:
            logger.error(f"tracker cleanup error: {e}")
