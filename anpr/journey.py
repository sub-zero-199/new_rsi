"""Point logic + journey state machine.

`PointProcessor.evaluate()` is the pre-push gate: it applies same-place dedup/cooldown
and the per-point rules (ported from v11_1 gatein/gateout/weighbridge) and returns a
Decision(allow, reason). `record_success()` runs AFTER a successful portal push: it writes
the durable journey/event (SQLite) and the 24h bucket (Redis tracker) and marks the plate
known. Mid-flow joins (a vehicle first seen at weighbridge/gate-out) are allowed and flagged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from anpr.log import logger, point_logger
from anpr.normalize import get_canonical_plate


@dataclass
class Decision:
    allow: bool
    reason: str
    continuity_status: str = "ok"   # ok | mid_flow | out_of_order


class PointProcessor:
    def __init__(self, cfg, store, tracker):
        self.cfg = cfg
        self.store = store
        self.tracker = tracker
        self.dedup_threshold = cfg.matching.dedup_threshold
        self.cd = cfg.cooldown

    # ---------------- pre-push gate ----------------
    def evaluate(self, point: str, plate: str, ts: int) -> Decision:
        # 1) same-place dedup / cooldown (the "20% variation at same place" rule)
        same = self.tracker.check_in_last_24h(point, plate, threshold=self.dedup_threshold)
        for e in same:
            age = ts - int(e["detection_time"])
            if 0 <= age < self.cd.same_place_seconds:
                return Decision(False, f"cooldown: {point} added {age}s ago", "ok")

        # 2) in/out dedup — the portal always creates the trip now, so we skip duplicates here:
        #    a gate-in for a vehicle already inside, or a gate-out for one already out.
        #    A gate-in OR a weighbridge read both mean the vehicle is ON THE PREMISES, so
        #    either re-establishes "inside": e.g. a weighbridge read after a gate-out means
        #    the vehicle is back in, so a later gate-out is allowed (not "already out").
        #    A never-seen gate-out is NOT "already out" — it's a mid-flow join and still created.
        last_inside, last_out = self._in_out_state(plate)
        if point == "gate_in":
            inside = last_inside is not None and (last_out is None or last_inside >= last_out)
            if inside:
                return Decision(False, "already inside; skip duplicate gate-in")
        elif point == "gate_out":
            already_out = last_out is not None and (last_inside is None or last_out >= last_inside)
            if already_out:
                return Decision(False, "already out; skip duplicate gate-out")

        # 3) point-specific rules
        if point == "gate_in":
            return self._eval_gate_in(plate, ts)
        if point == "weighbridge":
            return self._eval_weighbridge(plate, ts)
        if point == "gate_out":
            return self._eval_gate_out(plate, ts)
        return Decision(True, "ok")

    def _in_out_state(self, plate: str) -> tuple[Optional[int], Optional[int]]:
        """Return (last_inside_ts, last_out_ts) for `plate` in the last 24h (or None each).

        last_inside is the most recent "on the premises" signal — a gate-in OR a weighbridge
        read. A vehicle on the weighbridge is inside, so a weighbridge read after a gate-out
        makes the vehicle "in" again (and a subsequent gate-out valid).
        """
        ins = self.tracker.check_in_last_24h("gate_in", plate, self.dedup_threshold)
        wbs = self.tracker.check_in_last_24h("weighbridge", plate, self.dedup_threshold)
        outs = self.tracker.check_in_last_24h("gate_out", plate, self.dedup_threshold)
        inside_ts = [int(e["detection_time"]) for e in ins] + [int(e["detection_time"]) for e in wbs]
        last_inside = max(inside_ts, default=None)
        last_out = max((int(e["detection_time"]) for e in outs), default=None)
        return last_inside, last_out

    def _eval_gate_in(self, plate: str, ts: int) -> Decision:
        for e in self.tracker.check_in_last_24h("gate_out", plate, self.dedup_threshold):
            age = ts - int(e["detection_time"])
            if 0 <= age < self.cd.reentry_gap_seconds:
                return Decision(False, f"re-entry too soon after gate-out ({age}s)")
        return Decision(True, "ok")

    def _eval_weighbridge(self, plate: str, ts: int) -> Decision:
        # Optional minimum gap after a gate-out before a weighbridge read is accepted.
        # Default 0 => disabled (a weighbridge read is allowed immediately after gate-out,
        # which re-establishes the vehicle as "inside"). Set >0 to require a wait.
        gap = self.cd.weighbridge_after_gateout_gap_seconds
        if gap > 0:
            for e in self.tracker.check_in_last_24h("gate_out", plate, self.dedup_threshold):
                age = ts - int(e["detection_time"])
                if 0 <= age < gap:
                    return Decision(False, f"weighbridge too soon after gate-out ({age}s)")
        wbs = self.tracker.check_in_last_24h("weighbridge", plate, self.dedup_threshold)
        if len(wbs) >= self.cd.weighbridge_max_per_24h:
            return Decision(False, f"weighbridge max {self.cd.weighbridge_max_per_24h}/24h reached")
        if len(wbs) >= 1:
            first = min(int(e["detection_time"]) for e in wbs)
            age = ts - first
            if 0 <= age < self.cd.weighbridge_second_gap_seconds:
                return Decision(False, f"too soon for 2nd weighbridge ({age}s)")
        ins = self.tracker.check_in_last_24h("gate_in", plate, self.dedup_threshold)
        continuity = "ok" if ins else "mid_flow"
        return Decision(True, "ok", continuity)

    def _eval_gate_out(self, plate: str, ts: int) -> Decision:
        # Duplicate "already out" exits are handled by the in/out check in evaluate(); here we
        # only enforce the minimum dwell after a gate-in. A vehicle with no gate-in (e.g. seen
        # only at the weighbridge) is a mid-flow join and is allowed out.
        ins = self.tracker.check_in_last_24h("gate_in", plate, self.dedup_threshold)
        if not ins:
            return Decision(True, "ok", "mid_flow")
        recent_in = max(int(e["detection_time"]) for e in ins)
        age = ts - recent_in
        if 0 <= age < self.cd.reentry_gap_seconds:
            return Decision(False, f"gate-out too soon after gate-in ({age}s)")
        return Decision(True, "ok")

    # ---------------- post-success record ----------------
    def record_success(self, *, point: str, plate: str, plate_raw: str, ts: int,
                        camera_id: str, s3_key: Optional[str], api_response=None,
                        continuity_status: str = "ok") -> int:
        """Persist a successful detection: 24h bucket + journey + event + known_plate."""
        # Redis 24h bucket
        self.tracker.store_entry(point, plate, ts, api_response, s3_key or "")

        # SQLite journey state machine
        journey = self.store.active_journey(plate)
        if journey is None:
            journey_id = self.store.create_journey(plate, point, ts, skipped=(point != "gate_in"))
        else:
            journey_id = journey["journey_id"]
            self.store.advance_journey(journey_id, point, ts)

        # durable event + known plate
        self.store.insert_event(
            plate_norm=plate, plate_raw=plate_raw, point=point, camera_id=camera_id,
            ts=ts, s3_plate_key=s3_key, portal_status="added",
            journey_id=journey_id, continuity_status=continuity_status,
        )
        self.store.upsert_known_plate(plate, source="anpr_added", portal_status="added", ts=ts)
        point_logger(point).info(
            f"Recorded {point} success for {plate} (journey {journey_id}, {continuity_status})")
        return journey_id
