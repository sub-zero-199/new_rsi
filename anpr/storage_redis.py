"""Redis-backed work queue for the S3-upload / portal background processor.

Ported from v11_1 redis_manager.py and generalized for the single unified process:
namespaced keys, a dict payload that carries point + camera_id, and a JSON-file
local fallback so the system keeps working when redis-server isn't running (e.g. dev box).
The `redis` package import is optional so the module imports even before `uv sync`.
"""
from __future__ import annotations

import os
import glob
import json
import time
import threading
from datetime import datetime
from typing import Optional

from anpr.log import logger

try:
    import redis as _redis
except Exception:  # package not installed yet
    _redis = None


class RedisQueue:
    MAX_RETRY_COUNT = 3

    def __init__(self, host="localhost", port=6379, db=0, namespace="anpr"):
        self.namespace = namespace
        self.redis_client = None
        self._lock = threading.Lock()
        if _redis is not None:
            try:
                self.redis_client = _redis.Redis(
                    host=host, port=port, db=db, decode_responses=True,
                    socket_connect_timeout=2, socket_timeout=2,
                )
                self.redis_client.ping()
                logger.info(f"Connected to Redis ({host}:{port} db={db})")
            except Exception as e:  # ConnectionError/TimeoutError/OSError
                logger.warning(f"Redis unavailable ({e}); using local-file queue fallback")
                self.redis_client = None
        else:
            logger.warning("redis package not installed; using local-file queue fallback")

        self.detection_queue_key = f"{namespace}:detections"
        self.processed_key = f"{namespace}:processed"
        self._local_dir = f"local_queue_{namespace}"

    # ---------------- store ----------------
    def store_detection(self, detection: dict) -> bool:
        """Store one detection payload for background S3 upload + (optional) portal push.

        Expected keys: vehicle_number, plate_img_path, full_img_path, point, camera_id,
        detection_time, offline_mode. Missing keys are tolerated.
        """
        d = dict(detection)
        d.setdefault("detection_time", int(time.time()))
        d["detection_time"] = int(d["detection_time"])
        for k in ("plate_img_path", "full_img_path"):
            if d.get(k):
                d[k] = os.path.abspath(str(d[k]))
        d.setdefault("processed", "false")
        d["offline_mode"] = "true" if d.get("offline_mode") else "false"
        d.setdefault("retry_count", "0")
        d = {k: ("" if v is None else v) for k, v in d.items()}

        if self.redis_client:
            try:
                key = f"detection:{d.get('vehicle_number','')}:{d['detection_time']}"
                # Redis hashes need str/number values.
                self.redis_client.hset(key, mapping={k: (str(v) if not isinstance(v, (int, float)) else v)
                                                      for k, v in d.items()})
                self.redis_client.lpush(self.detection_queue_key, key)
                logger.info(f"Queued detection {d.get('vehicle_number','')} @ {d['detection_time']}")
                return True
            except Exception as e:
                logger.error(f"Redis store error: {e}")
                return self._store_locally(d)
        return self._store_locally(d)

    def _store_locally(self, d: dict) -> bool:
        try:
            os.makedirs(self._local_dir, exist_ok=True)
            fn = f"{self._local_dir}/detection_{d['detection_time']}_{d.get('vehicle_number','')}.json"
            with open(fn, "w") as f:
                json.dump(d, f)
            logger.info(f"Queued detection locally: {d.get('vehicle_number','')}")
            return True
        except Exception as e:
            logger.error(f"Local queue store error: {e}")
            return False

    # ---------------- read ----------------
    def get_pending_detections(self, batch_size=10) -> list[tuple[str, dict]]:
        if not self.redis_client:
            return self._get_local(batch_size)
        try:
            out = []
            for _ in range(batch_size):
                key = self.redis_client.rpop(self.detection_queue_key)
                if not key:
                    break
                data = self.redis_client.hgetall(key)
                if data:
                    out.append((key, data))
            return out
        except Exception as e:
            logger.error(f"Redis read error: {e}")
            return []

    def _get_local(self, batch_size) -> list[tuple[str, dict]]:
        if not os.path.exists(self._local_dir):
            return []
        files = sorted(glob.glob(f"{self._local_dir}/detection_*.json"))[:batch_size]
        out = []
        for fp in files:
            try:
                with open(fp) as f:
                    out.append((fp, json.load(f)))
            except Exception as e:
                logger.error(f"Local queue read error {fp}: {e}")
        return out

    # ---------------- complete ----------------
    def mark_detection_processed(self, detection_key: str, success=True) -> None:
        if self.redis_client and not str(detection_key).startswith(self._local_dir):
            try:
                if success:
                    data = self.redis_client.hgetall(detection_key)
                    if data:
                        data["processed"] = "true"
                        data["processed_time"] = datetime.now().isoformat()
                        self.redis_client.hset(detection_key, mapping=data)
                        self.redis_client.sadd(self.processed_key, detection_key)
                else:
                    data = self.redis_client.hgetall(detection_key)
                    retries = int(data.get("retry_count", "0"))
                    if retries >= self.MAX_RETRY_COUNT:
                        data.update(processed="true", failed="true",
                                    failure_reason="max retries exceeded")
                        self.redis_client.hset(detection_key, mapping=data)
                        self.redis_client.sadd(self.processed_key, detection_key)
                        logger.warning(f"Detection {detection_key} permanently failed")
                    else:
                        data["retry_count"] = str(retries + 1)
                        self.redis_client.hset(detection_key, mapping=data)
                        self.redis_client.lpush(self.detection_queue_key, detection_key)
            except Exception as e:
                logger.error(f"Redis mark error: {e}")
        else:
            # local file
            if success and os.path.exists(detection_key):
                try:
                    os.remove(detection_key)
                except Exception as e:
                    logger.error(f"Local queue remove error: {e}")

    def get_queue_size(self) -> int:
        if self.redis_client:
            try:
                return int(self.redis_client.llen(self.detection_queue_key))
            except Exception:
                return 0
        return len(glob.glob(f"{self._local_dir}/detection_*.json")) if os.path.exists(self._local_dir) else 0
