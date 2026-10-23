#!/usr/bin/env python3
"""Auto-configure the camera slots in config.yaml from what's actually on the network.

For every camera slot it:
  1. checks whether the camera is CONNECTED (TCP-connects to its RTSP port),
  2. sets `enabled: true` if connected, else `enabled: false`,
  3. if connected, tries to read the camera's MAC and writes it into `mac`
     (the portal cameraId). MAC methods, in order:
       * ONVIF GetNetworkInterfaces over HTTP (works across subnets / routed cameras),
       * ARP neighbor table (only if the camera is on the same L2 subnet).
Only `enabled:` and `mac:` lines are rewritten — all comments/formatting are preserved.

Run it ON THE WORKSTATION (it needs real network access to the cameras):
  uv run python tools/fetch_camera_macs.py             # apply changes to config.yaml
  uv run python tools/fetch_camera_macs.py --dry-run    # just show what it would do
  uv run python tools/fetch_camera_macs.py -c other.yaml
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import re
import socket
import subprocess
import urllib.request
import uuid

from anpr.config import load_config

_MAC_RE = re.compile(r"([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})")
_ONVIF_PORTS = (80, 8000, 8899, 8080)


def normalize_mac(mac: str) -> str:
    """Lowercase, strip separators -> the config/cameraId format (e.g. 90f721000b2f)."""
    return re.sub(r"[^0-9a-f]", "", mac.lower())


def is_connected(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _onvif_envelope(user: str, password: str) -> bytes:
    created = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    nonce = uuid.uuid4().bytes
    digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + password.encode()).digest()).decode()
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">'
        '<s:Header><Security s:mustUnderstand="1" xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">'
        f'<UsernameToken><Username>{user}</Username>'
        f'<Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest}</Password>'
        f'<Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{base64.b64encode(nonce).decode()}</Nonce>'
        f'<Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">{created}</Created>'
        '</UsernameToken></Security></s:Header>'
        '<s:Body><GetNetworkInterfaces xmlns="http://www.onvif.org/ver10/device/wsdl"/></s:Body></s:Envelope>'
    ).encode()


def onvif_mac(ip: str, user: str, password: str) -> str | None:
    body = _onvif_envelope(user, password)
    for port in _ONVIF_PORTS:
        url = f"http://{ip}:{port}/onvif/device_service"
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/soap+xml; charset=utf-8"})
            resp = urllib.request.urlopen(req, timeout=4).read().decode("utf-8", "ignore")
        except Exception:
            continue
        m = re.search(r"HwAddress>\s*(" + _MAC_RE.pattern + r")", resp)
        if m:
            return normalize_mac(m.group(1))
    return None


def arp_mac(ip: str) -> str | None:
    """Same-subnet only: ping then read the neighbor/ARP table."""
    subprocess.run(["ping", "-c", "1", "-W", "1", ip],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for cmd in (["ip", "neigh", "show", ip], ["arp", "-n", ip]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        m = _MAC_RE.search(out)
        if m:
            return normalize_mac(m.group(1))
    return None


def _apply(path: str, updates: dict[str, dict]) -> int:
    """Rewrite `enabled:`/`mac:` lines per camera id, preserving everything else."""
    with open(path) as f:
        lines = f.readlines()
    cur: str | None = None
    changed = 0
    id_re = re.compile(r"^\s*-?\s*id:\s*(\S+)")
    en_re = re.compile(r"^(\s*enabled:\s*)(\S+)(.*)$")
    mac_re = re.compile(r"^(\s*mac:\s*)(\S+)(.*)$")
    for i, line in enumerate(lines):
        if re.match(r"^[^\s#]", line):  # a top-level key ends the cameras block
            cur = None
        mid = id_re.match(line)
        if mid:
            cur = mid.group(1)
            continue
        if cur not in updates:
            continue
        up = updates[cur]
        me = en_re.match(line)
        if me:
            want = "true" if up["enabled"] else "false"
            if me.group(2) != want:
                lines[i] = f"{me.group(1)}{want}{me.group(3)}\n"
                changed += 1
            continue
        mm = mac_re.match(line)
        if mm and up.get("mac"):
            want = f'"{up["mac"]}"'
            if mm.group(2).strip('"') != up["mac"]:
                lines[i] = f"{mm.group(1)}{want}{mm.group(3)}\n"
                changed += 1
    if changed:
        with open(path, "w") as f:
            f.writelines(lines)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-set enabled/mac for camera slots from the LAN")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="show what would change; don't write")
    args = ap.parse_args()

    cfg = load_config(args.config)
    user = cfg.rtsp_default_username or "admin"
    pw = cfg.rtsp_default_password or ""

    updates: dict[str, dict] = {}
    print(f"{'camera id':<18} {'point':<12} {'ip':<16} {'connected':<10} {'mac'}")
    print("-" * 74)
    for cam in cfg.cameras:
        connected = bool(cam.address) and is_connected(cam.address, cam.port)
        mac = None
        if connected:
            # ARP first (fast, same-subnet); ONVIF fallback for routed cameras.
            mac = arp_mac(cam.address) or onvif_mac(cam.address, user, pw)
        updates[cam.id] = {"enabled": connected, "mac": mac}
        mac_note = mac or ("(unreadable — set manually)" if connected else "-")
        print(f"{cam.id:<18} {cam.point:<12} {str(cam.address):<16} "
              f"{'yes' if connected else 'no':<10} {mac_note}")

    if args.dry_run:
        print("\n[dry-run] no changes written.")
        return 0
    n = _apply(args.config, updates)
    print(f"\nUpdated {n} line(s) in {args.config}." if n else "\nNo changes needed.")
    unreadable = [cid for cid, u in updates.items() if u["enabled"] and not u["mac"]]
    if unreadable:
        print("MAC still needs manual entry (couldn't read from camera): " + ", ".join(unreadable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
