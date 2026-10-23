"""Durable SQLite store: known_plates (the local match dictionary), events (audit /
the 3 buckets), journeys (gate-in -> weighbridge -> gate-out continuity), and outbox
(failed portal pushes for retry).

Thread-safe via a single connection + lock with WAL mode. Survives restarts, unlike Redis.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import json
from typing import Optional

from anpr.log import logger

_SCHEMA = """
CREATE TABLE IF NOT EXISTS known_plates (
    plate_norm    TEXT PRIMARY KEY,
    first_seen    INTEGER NOT NULL,
    last_seen     INTEGER NOT NULL,
    source        TEXT NOT NULL,            -- 'anpr_added' | 'active'
    portal_status TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_norm   TEXT NOT NULL,
    plate_raw    TEXT,
    point        TEXT NOT NULL,             -- gate_in | weighbridge | gate_out (== bucket)
    camera_id    TEXT,
    ts           INTEGER NOT NULL,
    confidence   REAL,
    s3_plate_key TEXT,
    portal_status TEXT,
    journey_id   INTEGER,
    continuity_status TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_plate ON events(plate_norm);
CREATE INDEX IF NOT EXISTS idx_events_point_ts ON events(point, ts);

CREATE TABLE IF NOT EXISTS journeys (
    journey_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_norm    TEXT NOT NULL,
    started_point TEXT NOT NULL,
    current_point TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'active',  -- active | completed
    gate_in_ts    INTEGER,
    wb_in_ts      INTEGER,
    wb_out_ts     INTEGER,
    gate_out_ts   INTEGER,
    skipped_flag  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_journeys_plate ON journeys(plate_norm, status);

CREATE TABLE IF NOT EXISTS outbox (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    payload_json TEXT NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending | sent | failed
    created_ts   INTEGER NOT NULL,
    last_try_ts  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox(status);
"""


class SqliteStore:
    def __init__(self, path: str = "anpr.db"):
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        logger.info(f"SQLite store ready at {path}")

    def close(self):
        with self._lock:
            self._conn.close()

    # ---------------- known_plates ----------------
    def upsert_known_plate(self, plate_norm: str, source: str = "anpr_added",
                           portal_status: Optional[str] = None, ts: Optional[int] = None) -> None:
        if not plate_norm:
            return
        ts = int(ts or time.time())
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO known_plates (plate_norm, first_seen, last_seen, source, portal_status)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(plate_norm) DO UPDATE SET
                    last_seen=excluded.last_seen,
                    source=excluded.source,
                    portal_status=COALESCE(excluded.portal_status, known_plates.portal_status)
                """,
                (plate_norm, ts, ts, source, portal_status),
            )
            self._conn.commit()

    def all_known_plates(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT plate_norm FROM known_plates").fetchall()
        return [r["plate_norm"] for r in rows]

    def has_known_plate(self, plate_norm: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM known_plates WHERE plate_norm=?", (plate_norm,)
            ).fetchone()
        return row is not None

    # ---------------- events ----------------
    def insert_event(self, *, plate_norm: str, plate_raw: str, point: str, camera_id: str,
                     ts: Optional[int] = None, confidence: Optional[float] = None,
                     s3_plate_key: Optional[str] = None, portal_status: Optional[str] = None,
                     journey_id: Optional[int] = None, continuity_status: Optional[str] = None) -> int:
        ts = int(ts or time.time())
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO events
                  (plate_norm, plate_raw, point, camera_id, ts, confidence,
                   s3_plate_key, portal_status, journey_id, continuity_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (plate_norm, plate_raw, point, camera_id, ts, confidence,
                 s3_plate_key, portal_status, journey_id, continuity_status),
            )
            self._conn.commit()
            return cur.lastrowid

    def events_for_point(self, point: str, since_ts: Optional[int] = None) -> list[dict]:
        q = "SELECT * FROM events WHERE point=?"
        args: list = [point]
        if since_ts is not None:
            q += " AND ts >= ?"
            args.append(int(since_ts))
        q += " ORDER BY ts DESC"
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    # ---------------- journeys ----------------
    def active_journey(self, plate_norm: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM journeys WHERE plate_norm=? AND status='active' "
                "ORDER BY journey_id DESC LIMIT 1",
                (plate_norm,),
            ).fetchone()
        return dict(row) if row else None

    def create_journey(self, plate_norm: str, started_point: str, ts: int,
                        skipped: bool = False) -> int:
        col = {"gate_in": "gate_in_ts", "weighbridge": "wb_in_ts", "gate_out": "gate_out_ts"}[started_point]
        with self._lock:
            cur = self._conn.execute(
                f"""
                INSERT INTO journeys (plate_norm, started_point, current_point, status, {col}, skipped_flag)
                VALUES (?, ?, ?, 'active', ?, ?)
                """,
                (plate_norm, started_point, started_point, int(ts), 1 if skipped else 0),
            )
            self._conn.commit()
            return cur.lastrowid

    def advance_journey(self, journey_id: int, point: str, ts: int,
                        complete: bool = False) -> None:
        ts = int(ts)
        sets = ["current_point=?"]
        args: list = [point]
        # weighbridge can record both wb_in and wb_out; set whichever is empty.
        if point == "gate_in":
            sets.append("gate_in_ts=COALESCE(gate_in_ts, ?)"); args.append(ts)
        elif point == "weighbridge":
            sets.append("wb_in_ts=COALESCE(wb_in_ts, ?)"); args.append(ts)
            sets.append("wb_out_ts=CASE WHEN wb_in_ts IS NOT NULL AND wb_in_ts<>? THEN ? ELSE wb_out_ts END")
            args.extend([ts, ts])
        elif point == "gate_out":
            sets.append("gate_out_ts=COALESCE(gate_out_ts, ?)"); args.append(ts)
        if complete or point == "gate_out":
            sets.append("status='completed'")
        args.append(journey_id)
        with self._lock:
            self._conn.execute(f"UPDATE journeys SET {', '.join(sets)} WHERE journey_id=?", args)
            self._conn.commit()

    # ---------------- outbox ----------------
    def enqueue_outbox(self, payload: dict) -> int:
        now = int(time.time())
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO outbox (payload_json, attempts, status, created_ts) VALUES (?, 0, 'pending', ?)",
                (json.dumps(payload), now),
            )
            self._conn.commit()
            return cur.lastrowid

    def pending_outbox(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM outbox WHERE status='pending' ORDER BY id LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload_json"])
            out.append(d)
        return out

    def mark_outbox(self, outbox_id: int, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE outbox SET status=?, attempts=attempts+1, last_try_ts=? WHERE id=?",
                (status, int(time.time()), outbox_id),
            )
            self._conn.commit()
