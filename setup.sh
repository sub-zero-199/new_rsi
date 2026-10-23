#!/usr/bin/env bash
# CWC ANPR v2 installer for the on-site workstation. Idempotent — safe to re-run.
#
# Just run:   ./setup.sh
# It runs 7 numbered steps ([1/7]..[7/7]) and does EVERYTHING:
#   1 install uv (if missing)
#   2 sync the local .venv with ML deps (torch + ultralytics + tensorrt)
#   3 ensure redis-server is installed + running
#   4 create + secure config.yaml (chmod 600)
#   5 detect cameras: set enabled per slot + auto-fill MACs from the network
#   6 convert models to TensorRT (.pt -> .engine) when use_tensorrt is on
#   7 install + start the systemd service (runs on boot, survives closing the terminal)
#
# Opt OUT of any part (rarely needed):
#   --no-ml        skip torch + ultralytics (mock engine only, no GPU)
#   --no-redis     don't touch redis-server
#   --no-mac       skip camera detection / MAC auto-fill
#   --no-service   don't install the systemd unit
#   --no-build     skip TensorRT conversion; reuse existing models/*.engine as-is
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WITH_ML=1; WITH_REDIS=1; WITH_MAC=1; INSTALL_SERVICE=1; WITH_BUILD=1
for arg in "$@"; do
  case "$arg" in
    --no-ml) WITH_ML=0 ;;
    --no-redis) WITH_REDIS=0 ;;
    --no-mac) WITH_MAC=0 ;;
    --no-service) INSTALL_SERVICE=0 ;;
    --no-build) WITH_BUILD=0 ;;
    # old opt-in flags kept as harmless no-ops (everything is on by default now)
    --with-ml|--with-redis|--install-service) ;;
    *) echo "unknown arg: $arg"; exit 1 ;;
  esac
done

STEP=0; TOTAL=7
step(){ STEP=$((STEP + 1)); echo; echo "==> [$STEP/$TOTAL] $1"; }

step "Checking uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "  installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "  uv present"
fi

step "Syncing Python environment (.venv) — uv shows download progress below"
if [ "$WITH_ML" = "1" ]; then
  uv sync --extra ml --extra dev --project "$HERE"
else
  uv sync --extra dev --project "$HERE"
fi

step "Ensuring Redis"
if [ "$WITH_REDIS" = "1" ]; then
  if command -v redis-cli >/dev/null 2>&1 && redis-cli ping >/dev/null 2>&1; then
    echo "  already running"
  else
    echo "  installing + enabling redis-server"
    sudo apt-get update -y
    sudo apt-get install -y redis-server
    sudo systemctl enable redis-server
    sudo systemctl start redis-server
  fi
else
  echo "  skipped (--no-redis)"
fi

step "Configuring config.yaml"
[ -f "$HERE/config.yaml" ] || { cp "$HERE/config.yaml.example" "$HERE/config.yaml"; echo "  created config.yaml (edit it)"; }
chmod 600 "$HERE/config.yaml"
echo "  secured config.yaml (chmod 600)"
[ -f "$HERE/.env" ] && chmod 600 "$HERE/.env" || true
if [ "$WITH_ML" = "1" ]; then
  missing=""
  for w in detect.pt keypoint.pt ocr.pt; do
    [ -f "$HERE/models/$w" ] || missing="$missing $w"
  done
  [ -n "$missing" ] && echo "  WARNING: missing model weights under models/:$missing" || true
fi

step "Detecting cameras (enable per slot + auto-fill MAC)"
if [ "$WITH_MAC" = "1" ] && [ -f "$HERE/tools/fetch_camera_macs.py" ]; then
  uv run --project "$HERE" python "$HERE/tools/fetch_camera_macs.py" -c "$HERE/config.yaml" \
    || echo "  (detection incomplete — set camera addresses in config.yaml, then re-run ./setup.sh)"
else
  echo "  skipped (--no-mac)"
fi

step "Building TensorRT engines (.pt -> .engine)"
# free the GPU first: retire the old v11_1 services and stop the unified service
if [ "$INSTALL_SERVICE" = "1" ]; then
  echo "  retiring old v11_1 services"
  for old in gatein-service gateout-service weighbridge-service anpr-setup2 cwc-updater; do
    sudo systemctl disable --now "$old.service" 2>/dev/null || true
  done
fi
sudo systemctl stop cwc-anpr.service 2>/dev/null || true
if [ "$WITH_BUILD" = "0" ]; then
  # Reuse existing engines; they must all be present since the service runs on .engine only.
  miss=0
  for w in detect keypoint ocr; do
    if [ ! -f "$HERE/models/$w.engine" ]; then echo "  MISSING models/$w.engine"; miss=1; fi
  done
  if [ "$miss" = "1" ] && grep -qE '^[[:space:]]*use_tensorrt:[[:space:]]*true' "$HERE/config.yaml"; then
    echo "  !! --no-build given but engines are missing and use_tensorrt is true."
    echo "     Re-run ./setup.sh without --no-build to build them. Service NOT started."
    exit 1
  fi
  echo "  skipped (--no-build); using existing engines"
elif [ "$WITH_ML" = "1" ] && grep -qE '^[[:space:]]*use_tensorrt:[[:space:]]*true' "$HERE/config.yaml"; then
  need_build=0
  # Rebuild if the engine is missing, the .pt changed, OR config.yaml changed (imgsz lives
  # in config and is baked into the engine's static input shape at export time).
  for w in detect keypoint ocr; do
    pt="$HERE/models/$w.pt"; en="$HERE/models/$w.engine"
    [ -f "$pt" ] || continue
    if [ ! -f "$en" ] || [ "$pt" -nt "$en" ] || [ "$HERE/config.yaml" -nt "$en" ]; then need_build=1; fi
  done
  if [ "$need_build" = "1" ]; then
    echo "  converting (a few minutes; progress shown below)"
    uv run --project "$HERE" cwc-anpr -c "$HERE/config.yaml" export
    miss=0
    for w in detect keypoint ocr; do
      if [ ! -f "$HERE/models/$w.engine" ]; then echo "  MISSING models/$w.engine"; miss=1; fi
    done
    if [ "$miss" = "1" ]; then
      echo "  !! TensorRT build failed. use_tensorrt is true, so the service needs the engines."
      echo "     Fix the errors above and re-run ./setup.sh. Service NOT started."
      exit 1
    fi
    echo "  engines ready"
  else
    echo "  already up to date; skipping conversion"
  fi
else
  echo "  skipped (use_tensorrt not enabled)"
fi

step "Installing + starting cwc-anpr.service"
if [ "$INSTALL_SERVICE" = "1" ]; then
  # Point the unit at this install location (the template hard-codes a default path).
  sed "s|/home/lynkit/CWC-script/new_rsi|$HERE|g" "$HERE/systemd/anpr.service" \
    | sudo tee /etc/systemd/system/cwc-anpr.service >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable cwc-anpr.service
  sudo systemctl reset-failed cwc-anpr.service 2>/dev/null || true
  sudo systemctl restart cwc-anpr.service
  echo "  started + enabled — runs on boot and keeps running after you close the terminal."
  # Install CLI shortcuts to /usr/local/bin so they work in any terminal immediately.
  chmod +x "$HERE/tools/bin"/anpr-*
  # Update the log path inside anpr-log to match this install location.
  sed -i "s|^LOGS=.*|LOGS=$HERE/logs|" "$HERE/tools/bin/anpr-log"
  for script in "$HERE/tools/bin"/anpr-*; do
    sudo cp "$script" /usr/local/bin/
  done
  echo "  shortcuts installed: anpr-log, anpr-status, anpr-start, anpr-stop, anpr-restart"
else
  echo "  skipped (--no-service)"
fi

echo
echo "==> Setup complete ($STEP/$TOTAL steps done)."
echo "    If any camera above showed 'connected: no', set its address in config.yaml and"
echo "    re-run ./setup.sh (or: uv run python tools/fetch_camera_macs.py)."
echo "    Run 'source ~/.bashrc' then use the shortcuts:"
echo "        anpr-log gi|wb|go|cs|all|sys   # view logs"
echo "        anpr-status / anpr-restart / anpr-stop / anpr-start"
