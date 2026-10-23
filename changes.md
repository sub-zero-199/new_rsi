# Changes — CWC ANPR v2 (`new_rsi`)

Changes made to `/home/lynkit/new_rsi` on 22–23 Oct 2026, during the switch-over from the
old `CWC-script` services to the unified `cwc-anpr` service on the site workstation.

This folder is not under git. A backup of the code, tests, config, `.env` and `anpr.db` was
taken before the code fixes in section 3:
`/home/lynkit/new_rsi.backup-20261023-115123.tar.gz`
(restore: `cd ~ && tar -xzf new_rsi.backup-20261023-115123.tar.gz`). It already includes
the changes in sections 1–2 and the test/hand-off changes in 3.1–3.2, but not 3.3.

**Status:** all changes below are live — `cwc-anpr` was last restarted at 23 Oct 11:53,
after the last code change. Test suite: **105 passed**.

---

## Summary

| # | Area | Change | Files |
|---|---|---|---|
| 1.1 | Setup | `--no-build` flag to skip the TensorRT build | `setup.sh` |
| 1.2 | Setup | Service file gets the real install path automatically | `setup.sh`, `systemd/anpr.service` |
| 1.3 | Setup | Camera status service file pointed at this folder | `camera_status_service/camera_status.service` |
| 2.1 | Config | Camera stream timeout 1 s → 8 s | `camera_status_service/config.json` |
| 2.2 | Config | Slow health checks every 60 s → 300 s | `camera_status_service/config.json` |
| 2.3 | Docs | README updated for all of the above | `README.md` |
| 3.1 | Tests | Out-of-date tests fixed | `tests/test_service.py`, `tests/test_rsi_flow.py`, `tests/test_matching.py`, `tests/test_journey.py` |
| 3.2 | Code | API calls no longer freeze the cameras | `anpr/service.py`, tests |
| 3.3 | Code | No duplicate trips after an outage | `anpr/service.py`, tests |

---

## 1. Setup and deployment

### 1.1 `--no-build` flag in `setup.sh`

**Problem.** Step 6 of `setup.sh` rebuilds the three TensorRT `.engine` files whenever
`config.yaml` is newer than them. Any config edit (even a camera address) triggered a
several-minute rebuild, and there was no way to skip it.

**Change.** New option:

```bash
./setup.sh --no-build
```

- Skips the build and uses the existing `models/{detect,keypoint,ocr}.engine` as they are.
- If any engine is missing while `inference.use_tensorrt: true`, setup lists the missing
  files and exits **before** starting the service (same guarantee as the normal build).
- Step 6 still retires the old v11_1 services and stops `cwc-anpr` before step 7 restarts it.
- Listed in the opt-out comment block at the top of `setup.sh`.

**Caveat.** Don't use it after changing `detector_imgsz`, `rectifier_imgsz` or `ocr_imgsz`:
the image size is fixed into the engine at build time, so those need a rebuild.

### 1.2 Service file uses the real install path

**Problem.** `systemd/anpr.service` hard-codes `/home/lynkit/CWC-script/new_rsi`, which does
not exist on this workstation (the code is in `/home/lynkit/new_rsi`). `setup.sh` copied it
unchanged, so `cwc-anpr` failed every 10 s with
`status=226/NAMESPACE ... /home/lynkit/CWC-script/new_rsi: No such file or directory` and
never started. A side effect was that `logs/` was never created, so `anpr-log cs` failed.

**Change.** `setup.sh` (step 7) now writes the folder it is run from into the unit while
installing it:

```bash
sed "s|/home/lynkit/CWC-script/new_rsi|$HERE|g" "$HERE/systemd/anpr.service" \
  | sudo tee /etc/systemd/system/cwc-anpr.service >/dev/null
```

`WorkingDirectory`, `EnvironmentFile`, `ExecStart` and `ReadWritePaths` all follow. The
template in `systemd/anpr.service` is unchanged apart from its comment, which now says
setup fills the path in. The folder can be moved without editing paths by hand.

### 1.3 Camera status service file paths

**Problem.** `camera_status_service/camera_status.service` pointed at
`/home/lynkit/CWC_NEW_CODE`, which does not exist here.

**Change.** All paths now point at `/home/lynkit/new_rsi`:

```ini
WorkingDirectory=/home/lynkit/new_rsi/camera_status_service
EnvironmentFile=-/home/lynkit/new_rsi/.env
ExecStart=/home/lynkit/new_rsi/.venv/bin/python -u client.py
```

**Note.** This standalone service should stay **disabled**. `cwc-anpr` already runs the same
status reporter internally (`anpr/service.py` imports `camera_status_service/client.py`), so
running both sent every status packet to the portal twice. See "Open items".

---

## 2. Configuration and documentation

### 2.1 `rtspTimeoutSec`: 1 → 8

File: `camera_status_service/config.json`. How long the status reporter waits for a
camera's video stream before calling it failed. 1 s could mark slow-but-working cameras as
failed on the portal; 8 matches the old system.

### 2.2 `checkIntervalSec`: 60 → 300

Same file. Controls how often the slower checks run:

| Check | Effect of 300 s |
|---|---|
| Internet speed test | at most every 5 min (was ~every 2 min; the code never runs it more often than every 120 s) — far less test traffic |
| Camera video-stream check (resolution / fps / codec) | a broken stream on a reachable camera shows within 5 min |
| System health (internet, VPN, SSH, DNS, gateway, CPU, RAM, disk) — sets the site's STRONG/WEAK state on the portal | an internet/VPN outage shows within 5 min |

Unaffected: status packets are still sent every 30 s (`sendIntervalMs`), and every camera is
pinged on each send, so a powered-off or unplugged camera still shows WEAK within ~30 s.

### 2.3 README

`README.md` updated:

- Setup path corrected (`/home/lynkit/CWC_NEW_CODE` → `/home/lynkit/new_rsi`); notes that
  setup writes the install path into the service file.
- New section **"Skip the model build (`--no-build`)"**, listing the other setup options.
- **Logs:** documents the `anpr-log gi|wb|go|cs|all|sys` shortcuts and what "file does not
  exist" means.
- New section **"Camera status"**: settings table and the trade-off of `checkIntervalSec`.
- **Extras:** manual engine rebuild now points to `./tools/build_engines.sh`.
- New section **"The old system (CWC-script) — retired"**: what was switched off, why the old
  nightly cleanup job was removed, never run the old ANPR services alongside `cwc-anpr`,
  and the Redis keys both systems share.

---

## 3. Code changes

### 3.1 Out-of-date tests fixed

**Problem.** 13 tests failed. They still used an "active vehicles" lookup that was removed
from the service (`svc.active`, `PlateMatcher(..., active, ...)`,
`evaluate(..., in_active_list=...)`, `resolve(det, mac)`). The application code was fine.

**Change.**

- `tests/test_service.py`, `tests/test_rsi_flow.py`: removed
  `svc.active.get_active = lambda mac: []`.
- `tests/test_matching.py`: matcher built as `PlateMatcher(cfg, store, wl, dg)`; `resolve()`
  called without the camera argument; removed the `FakeActive` helper and the test
  `test_active_match_when_no_local`.
- `tests/test_journey.py`: removed `test_gate_in_skipped_when_in_active_list`.

### 3.2 API calls no longer freeze the cameras

**Problem.** The GPU inference worker called `AnprService._on_finalized()` directly, so all
three API calls for a vehicle (RSI cloud → local ANPR Data API → RSI cloud again) ran on the
GPU thread. While they ran, **no camera at any point was read**. Measured on 23 Oct before
the fix: every accepted vehicle paused all cameras for **6–8 s** with both APIs answering
normally. With the cloud slow or down, the worst case was ~1.5 min per vehicle
(3 × 30 s timeout + retry delays + 10 s local API + 30 s step 3).

**Change** (`anpr/service.py`):

- `_on_finalized(det)` — still the worker's callback — now only puts the detection on a
  queue (`queue.Queue(maxsize=500)`) and returns at once. If the queue is ever full the
  detection is dropped with an error log.
- New thread `decisions` (`_decision_loop`) takes detections off the queue **one at a time**
  and calls `_handle_finalized(det)`, which is the old `_on_finalized` body, unchanged
  (matching → duplicate check → three-call RSI flow → record → S3 queue).
- A single consumer is deliberate: vehicles are handled in order, so the duplicate check
  always sees the previous vehicle's record.
- `start()` starts the thread before the worker. `stop()` stops the worker first (which
  flushes its last windows into the queue), then sends a stop marker and waits up to 60 s
  for the queue to finish.

**Tests.** Tests that check results immediately now call `svc._handle_finalized(...)`. New
test `test_slow_api_does_not_block_the_gpu_worker`: each API call takes 1 s, yet two
vehicles are handed off in under 0.2 s and then processed in order with none lost.

**How to confirm on site.** After vehicles pass, the per-point heartbeat
(`processing: ...` every 2 s in `logs/gi.log`) should show no gaps when a vehicle is
accepted. Before the fix there was a 6–8 s gap at each one.

### 3.3 No duplicate trips after an outage

**Problem.** When the internet **and** the local ANPR server are both down, nothing accepts
the vehicle, so it goes to the SQLite `outbox` for retry. It was not recorded anywhere the
duplicate check looks, so a truck waiting at a point was re-read every few seconds and
**added to the outbox again each time**. When connectivity returned, every copy was sent
and became its own trip.

**Change** (`anpr/service.py`). Detections are still processed and logged exactly as
before; only the outbox step changes.

1. **Queue once.** Before adding to the outbox, `_recent_outbox_id(point, plate, ts)` looks
   for a pending entry with the **exact same plate** (canonical form) at the **same point**,
   queued within a window before this read:

   | Point | Window | From config |
   |---|---|---|
   | gate_in, gate_out | 300 s | `cooldown.reentry_gap_seconds` |
   | weighbridge | 600 s | `cooldown.weighbridge_second_gap_seconds` |

   If found, the repeat is not queued, its staged images are deleted, and the log says
   `already waiting in outbox (#id), not queued again`. Past the window a new entry is
   queued, so a genuine second weighing during an outage is kept. Exact-plate matching
   means two different trucks with similar numbers are never merged.

2. **Safety net at send time.** The retry loop body moved into `_drain_outbox()` (same
   behaviour, now testable). Before sending each item, it runs the same rules a live read
   gets (`PointProcessor.evaluate`). If an earlier item for the vehicle was just sent and
   recorded (e.g. it is now "already inside"), this item is marked `duplicate`, its images
   are deleted, and it is not sent. New outbox status value: `duplicate`.

**Tests** (`tests/test_rsi_flow.py`):

| Test | Checks |
|---|---|
| `test_repeat_reads_while_both_down_are_queued_once` | truck re-read 3 times → 1 outbox item; repeats' images removed |
| `test_different_plates_are_never_merged` | `MH12AB1234` and `MH12AB1235` → 2 items |
| `test_a_genuine_second_weighing_while_down_is_kept` | weighbridge read again after 600 s → 2 items |
| `test_outbox_sends_once_and_drops_a_repeat_on_recovery` | 2 items past the window, connection returns → only 1 trip |

Verified that the first and last of these fail when the fix is switched off.

---

## Open items (found in review, not yet fixed)

| # | Issue | Impact |
|---|---|---|
| 1 | Plate similarity is too loose: plates one character apart score ~0.90 against a 0.80 threshold (e.g. `MH12AB1234` vs `MH12AB1235`, `MH12AB1234` vs `MH14AB1234`). Used by plate correction (`matching.py`) and the duplicate check (`journey.py`, `tracker.py`). | A new truck can be recorded under a similar known plate, or discarded as "already inside". Highest-priority remaining fix. |
| 5 | S3 upload retries 3 times immediately, then gives up; items being processed are lost if the service restarts (`storage_redis.py`). | Images can be missing from S3, and the deferred step-3 call for offline detections is never sent. |
| 6 | A refused auto gate-out is retried every 5 min, then silently dropped after ~1 h when its Redis entry expires (`auto_gateout.py`, `tracker.py`). | Vehicle never gets an auto gate-out, with no log message. |
| 7 | Finished `detection:*` queue entries are never deleted from Redis. The old system's nightly cleanup (now removed) used to do this. | Slow growth in Redis. Fix: 7-day expiry when an entry is marked done. |
| — | README "Camera status" section still says to install `camera_status.service`. | Following it would bring back duplicate status packets (see 1.3); the section should say status is sent by `cwc-anpr`. |
| — | `inference.save_rectified: true` writes up to 500 debug images per service start. | Disk use (cleaned after 14 days). |
| — | Every detection is compared against the full `known_plates` table. | Gets slower as the table grows over months. |

---

## Related changes outside `new_rsi`

For context only — these are on the workstation, not in this folder:

- Old `gatein-service`, `gateout-service`, `weighbridge-service` stopped and disabled.
- Old `camera_status` service (from `CWC-script`) replaced; the standalone service is now
  disabled because `cwc-anpr` reports status itself.
- Old nightly cron job `CWC-script/cleanup_old.sh` removed (crontab backup:
  `/home/lynkit/crontab.backup.20261022-201730`). It deleted Redis `detection:*` keys in
  dbs 0–2 after 7 days, which this system also uses.
- `CWC-script/CLAUDE.md` marked as retired.
