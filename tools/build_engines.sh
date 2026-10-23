#!/usr/bin/env bash
# Convert the 3 .pt models to TensorRT .engine on THIS GPU, then restart the service so it
# runs on them. The service is configured to use ONLY .engine at runtime (inference.use_tensorrt
# is true), so the .pt files are used only here, as the source for the build.
#
# The service is stopped during the build to free the 4 GB GPU (building + running would
# risk out-of-memory), then restarted at the end — but ONLY if all engines built, so we
# never restart into a broken (engine-only, engines missing) state.
#
#   ./tools/build_engines.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

echo "==> Stopping cwc-anpr to free the GPU"
sudo systemctl stop cwc-anpr.service 2>/dev/null || true

echo "==> Ensuring TensorRT + export deps are installed (.venv)"
uv sync --extra ml --extra dev --project "$HERE"

echo "==> Building TensorRT engines (detect / keypoint / ocr) — a few minutes"
uv run --project "$HERE" cwc-anpr -c "$HERE/config.yaml" export

echo "==> Verifying engines:"
missing=0
for w in detect keypoint ocr; do
  if [ -f "$HERE/models/$w.engine" ]; then
    echo "  ok   models/$w.engine"
  else
    echo "  MISSING models/$w.engine"; missing=1
  fi
done
if [ "$missing" = "1" ]; then
  echo "!! Engine build incomplete — NOT restarting (service left stopped)."
  echo "   Fix the errors above and re-run ./tools/build_engines.sh"
  exit 1
fi

echo "==> Restarting cwc-anpr (runs on .engine only)"
sudo systemctl restart cwc-anpr.service
sleep 2
echo "==> Service: $(systemctl is-active cwc-anpr.service 2>/dev/null || true)"
echo "==> Loaded models (should all say .engine):"
journalctl -u cwc-anpr -n 30 --no-pager 2>/dev/null | grep "Loaded model" || \
  echo "   (check logs: journalctl -u cwc-anpr -n 30)"
