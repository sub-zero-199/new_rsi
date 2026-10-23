"""Vehicle whitelist (VAHAN-bypass). Ported from v11_1 whitelist_manager.py.

Whitelisted plates get disableVahanVerification applied and bypass strict validation.
Loaded from a YAML file with exact `vehicles:` and regex `patterns:`.
"""
from __future__ import annotations

import os
import re

import yaml

from anpr.log import logger
from anpr.normalize import normalize_raw


class WhitelistManager:
    def __init__(self, config_path: str = "shared_config/whitelist.yaml"):
        self.config_path = config_path
        self.whitelist: set[str] = set()
        self.patterns: list[re.Pattern] = []
        self.enabled = False
        self._load()

    def _load(self) -> None:
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Whitelist config not found at {self.config_path}")
                return
            with open(self.config_path) as f:
                cfg = yaml.safe_load(f) or {}
            wl = cfg.get("whitelist", {}) or {}
            self.enabled = bool(wl.get("enabled", False))
            if not self.enabled:
                logger.info("Whitelist disabled")
                return
            self.whitelist = {normalize_raw(v) for v in (wl.get("vehicles") or []) if v}
            for pat in (wl.get("patterns") or []):
                if pat:
                    try:
                        self.patterns.append(re.compile(pat, re.IGNORECASE))
                    except re.error as e:
                        logger.error(f"Invalid whitelist pattern {pat!r}: {e}")
            logger.info(f"Whitelist loaded: {len(self.whitelist)} vehicles, {len(self.patterns)} patterns")
        except Exception as e:
            logger.error(f"Error loading whitelist: {e}")

    def is_whitelisted(self, vehicle_number: str) -> bool:
        if not self.enabled or not vehicle_number:
            return False
        norm = normalize_raw(vehicle_number)
        if norm in self.whitelist:
            return True
        return any(p.match(norm) for p in self.patterns)

    def reload(self) -> None:
        self.whitelist, self.patterns, self.enabled = set(), [], False
        self._load()
