"""Doppelganger corrections: a CSV mapping a detected (misread) plate -> its actual plate.

Ported from v11_1 doppelganger_manager.py. Reloads when the CSV changes on disk so
operators can add corrections without a restart.
"""
from __future__ import annotations

import csv
import os
import threading

from anpr.log import logger


class DoppelgangerManager:
    def __init__(self, csv_path: str = "doppelganger.csv"):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        self.mapping: dict[str, str | None] = {}
        self.last_modified = 0.0
        self._ensure_csv()
        self._load()

    def _ensure_csv(self) -> None:
        if not os.path.exists(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["detected", "actual"])
            logger.info(f"Created {self.csv_path}")

    def _load(self) -> None:
        try:
            mtime = os.path.getmtime(self.csv_path) if os.path.exists(self.csv_path) else 0.0
            if mtime <= self.last_modified:
                return
            with self.lock:
                mapping: dict[str, str | None] = {}
                with open(self.csv_path, encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        det = (row.get("detected") or "").strip()
                        act = (row.get("actual") or "").strip()
                        if det:
                            mapping[det] = act or None
                self.mapping = mapping
                self.last_modified = mtime
        except Exception as e:
            logger.error(f"Error loading doppelganger.csv: {e}")

    def check_and_replace(self, detected: str) -> str:
        self._load()
        actual = self.mapping.get(detected)
        if actual:
            logger.info(f"Doppelganger: {detected} -> {actual}")
            return actual
        return detected

    def add_detected(self, detected: str) -> None:
        if not detected:
            return
        self._load()
        if detected in self.mapping:
            return
        try:
            with self.lock:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([detected, ""])
                self.mapping[detected] = None
                self.last_modified = os.path.getmtime(self.csv_path)
                logger.info(f"Added detected to doppelganger.csv: {detected}")
        except Exception as e:
            logger.error(f"Error adding to doppelganger.csv: {e}")
