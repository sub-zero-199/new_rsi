# CWC ANPR v2

Reads vehicle number plates from cameras at three points — **gate-in**, **weighbridge**,
and **gate-out** — and reports each trip over the **RSI API**. One program on one workstation
handles all cameras and one GPU. (It replaces the old three separate services.)

## How it works (in short)

1. Each camera streams video; the program grabs frames.
2. The GPU reads the number plate from the frame (detect → straighten → read characters).
3. Over a few seconds it picks the **best read** for each vehicle.
4. It fixes obvious misreads by matching against plates it already knows.
5. It checks: is this a duplicate? (skip a gate-in if the vehicle is already inside; skip a
   gate-out if it already left).
6. If it's a real event it runs the **three-call RSI flow** — post the detection to the
   cloud, send it to the local ANPR Data API (that call carries the plate image), then post
   it to the cloud again with the local server's reply attached — saves a record, and
   uploads the plate image.
7. If the internet is down, it keeps working locally and sends everything later.

---

## Setup — one command

On the workstation:

```bash
cd /home/lynkit/new_rsi
./setup.sh
```

`./setup.sh` does everything and is safe to run again anytime. It:
- installs the Python environment (with the GPU libraries),
- makes sure Redis is running,
- checks your cameras and turns the connected ones **on** (and fills in each camera's MAC),
- builds the faster GPU model files (TensorRT `.engine`) when needed,
- **starts the service** and sets it to start on boot.

The folder can live anywhere — setup writes the folder it's run from into the service file,
so you don't need to edit any paths.

### Skip the model build (`--no-build`)
Building the `.engine` files takes several minutes. Setup rebuilds them whenever
`config.yaml` is newer than the engines — so even a small config edit (like a camera
address) triggers a rebuild. If the engines are already built, skip it:

```bash
./setup.sh --no-build
```

It uses the existing `models/*.engine` files as they are. If any is missing, setup stops and
tells you (it won't start the service without them). **Don't use `--no-build` after changing
`detector_imgsz`, `rectifier_imgsz` or `ocr_imgsz`** — the image size is baked into the engine,
so those changes need a rebuild (plain `./setup.sh`).

Other options: `--no-mac` (skip camera detection), `--no-redis`, `--no-service`, `--no-ml`.

Once started, the service keeps running in the background — it does **not** stop when you
close the terminal, and it restarts itself if it crashes or the machine reboots.

Check the GPU is being used:
```bash
uv run python -c "import torch; print('GPU ready:', torch.cuda.is_available())"
```

### First time only: fill in your details
Open `config.yaml` and set:
- each camera's **IP address** (`address:`),
- the camera **password** (`rtsp_default_password:`),
- the **AWS keys** under `s3:` (or set `s3.enabled: false` if you don't upload images).

Then run `./setup.sh` again so it re-checks the cameras. On this machine these are already
filled in, so you don't need to change anything.

---

## Cameras

To re-check cameras any time (for example after moving or replacing one):
```bash
uv run python tools/fetch_camera_macs.py
```
It turns each camera **on** if it's connected (off if not) and fills in its MAC. To just see
what it would do without changing anything, add `--dry-run`.

---

## Start / stop the service

```bash
sudo systemctl status  cwc-anpr.service     # is it running?
sudo systemctl restart cwc-anpr.service     # restart (e.g. after a config or code change)
sudo systemctl stop    cwc-anpr.service     # stop it
```

---

## Logs — how to see what's happening

Setup installs a shortcut that works from any folder:

```bash
anpr-log gi     # gate-in       anpr-log wb    # weighbridge
anpr-log go     # gate-out      anpr-log cs    # cameras
anpr-log all    # all four      anpr-log sys   # full systemd log
```

If `anpr-log` says a file doesn't exist, the service hasn't started yet (the `logs/` folder
is created on first start) — check `systemctl status cwc-anpr.service`.

Or read the files directly. Each point and the cameras have their **own log file** in the
`logs/` folder. Use `tail -F` (capital **F** — it works even before the file exists):

```bash
tail -F logs/gi.log   # gate-in activity
tail -F logs/wb.log   # weighbridge activity
tail -F logs/go.log   # gate-out activity
tail -F logs/cs.log   # all cameras: connected / disconnected
```

Watch everything at once:
```bash
tail -F logs/gi.log logs/wb.log logs/go.log logs/cs.log
journalctl -u cwc-anpr -f       # the full combined log (systemd)
```

**What you'll see in a point log:**
- a heartbeat every 2 seconds, so you know it's working even when idle:
  `processing: 74 frames, 0 plate reads (last 2s)`
- its cameras connecting/disconnecting,
- and, when a vehicle passes, the full story:
  ```
  detected 'MH12AB1234' (conf 0.92, 5 reads)
  matched -> MH12AB1234 via local (0.95)
  accept MH12AB1234 -> RSI step-1
  Recorded gate_in success for MH12AB1234 (journey 3, ok)
  ```

If a point log shows heartbeats but never any "detected ..." lines, the software is fine —
the camera just isn't seeing a readable plate (check it's **in focus and aimed at the lane**).
You can grab one frame from a camera to check its view:
```bash
ffmpeg -rtsp_transport tcp -i "rtsp://admin:PASSWORD@CAMERA_IP:554/rtsp/streaming?channel=1&subtype=0" -frames:v 1 -y view.jpg
```

---

## Camera status (portal health reporting)

A separate small service, `camera_status`, reports this site's health to the portal
(`13.126.219.12:8004`): each camera's state, internet/VPN/SSH status, CPU/RAM/disk and an
internet speed test. Its files are in `camera_status_service/`; settings are in
`camera_status_service/config.json`.

Install / update it (after changing `camera_status.service`):
```bash
sudo cp camera_status_service/camera_status.service /etc/systemd/system/camera_status.service
sudo systemctl daemon-reload
sudo systemctl restart camera_status
```
After changing `config.json`, just restart: `sudo systemctl restart camera_status`.

Check it's sending:
```bash
journalctl -u camera_status -n 20 --no-pager | grep -E "Packet sent|SpeedTest"
```

Key settings in `config.json`:

| Setting | Value | What it does |
|---|---|---|
| `sendIntervalMs` | `30000` | Sends a status packet to the portal every 30 s. Each camera is pinged on every send, so a camera that goes offline shows **WEAK** within ~30 s. |
| `checkIntervalSec` | `300` | How often (5 min) the slower checks run: the **speed test**, the **camera video-stream check** (resolution/fps/codec), and **system health** (internet, VPN, SSH, DNS, CPU, RAM, disk). |
| `rtspTimeoutSec` | `8` | How long to wait for a camera's video stream before calling it failed. Lower values can mark slow-but-working cameras as failed. |

About `checkIntervalSec`: the site's main **STRONG/WEAK** status on the portal comes from the
system-health check, so an internet or VPN outage can take up to this long to show. Lower
means faster alerts but more speed-test data use (the speed test never runs more than once
every 2 minutes); higher saves data but alerts later. 300 s is the chosen balance.

---

## Extras (optional)

Rebuild the faster GPU model files by hand (setup normally does this for you):
```bash
./tools/build_engines.sh     # stops the service, builds, restarts it only if all 3 built
```

Test run without sending anything to either API:
```bash
uv run cwc-anpr -c config.yaml --dry-run  # logs what it would do, but doesn't send
```

Run on a computer with no GPU (for testing): set `inference.engine: mock` in `config.yaml`,
then `uv run pytest` to run the tests.

---

## Notes

- **Cameras are identified by MAC** — every RSI call sends the detecting camera's MAC as
  `cameraId` (that's the `mac:` in `config.yaml`, filled in automatically). Every camera's
  MAC must be registered on the portal, or its trips are refused.
- **The RSI `secret_key` belongs in the environment.** Set `RSI_SECRET_KEY` and leave
  `api.secret_key` blank in `config.yaml`; the value is sent as a header on every RSI call.
  The system this replaced hard-coded it in source — don't reintroduce that.
- **No gate hardware.** This deployment only reads plates, processes them and posts them.
  The boom-barrier and LED code was removed (originals kept in `.removed_hardware/`).
- **Credentials** (camera password, AWS keys) live in `config.yaml`. Setup keeps that file
  private (`chmod 600`). Don't share it.
- **Redis** must be running (setup handles it) — it powers the "already in / already out"
  checks and the automatic gate-out after 24 hours.

---

## The old system (CWC-script) — retired

This replaced the old code in `/home/lynkit/CWC-script`. On this workstation it is fully
switched off:

- `gatein-service`, `gateout-service`, `weighbridge-service` — stopped and **disabled**
  (setup does this; they won't start on reboot).
- `camera_status` — now runs from `new_rsi/camera_status_service` (same service name, so it
  replaced the old one).
- The old nightly cleanup job (`CWC-script/cleanup_old.sh` in crontab) — **removed**. It
  deleted Redis `detection:*` keys older than 7 days in Redis databases 0–2, which would also
  delete this system's queued detections.

**Never run the old ANPR services alongside `cwc-anpr`.** Both would process every vehicle —
double portal entries, double S3 uploads, and both fighting over the GPU and cameras.

Both systems use the same Redis keys for the 24-hour gate-in / gate-out records
(`anpr:shared:gatein:24h`, `anpr:shared:gateout:24h`, `anpr:shared:pending_auto_gateout`),
so a vehicle that gated in under the old system can still gate out under this one.
