#!/usr/bin/env bash
# Soil2Crop — deploy to a server (systemd + uvicorn). Run from the repo root.
#   DEST=/opt/soil2crop DOMAIN=soil2crop.example.com bash deploy/deploy.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${DEST:-/opt/soil2crop}"

echo ">> syncing app to $DEST"
sudo mkdir -p "$DEST"
sudo rsync -a --delete \
  --exclude '.git' --exclude '.venv' --exclude 'models' --exclude 'data/raw' \
  "$REPO/" "$DEST/"

# venv + deps on the server
if [ ! -x "$DEST/venv/bin/python" ]; then
  sudo python3 -m venv "$DEST/venv"
fi
sudo "$DEST/venv/bin/pip" install -q --upgrade pip
sudo "$DEST/venv/bin/pip" install -q -r "$DEST/requirements.txt"

# TabPFN-3.5 checkpoint
sudo mkdir -p "$DEST/models"
if [ ! -f "$DEST/models/tabpfn-v3.5-20260909.safetensors" ]; then
  echo ">> fetching checkpoint (~876 MB)"
  sudo curl -L --fail -o "$DEST/models/tabpfn-v3.5-20260909.safetensors" \
    https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/main/tabpfn-v3.5-20260909.safetensors
fi
sudo mkdir -p "$DEST/logs"

# systemd
sudo cp "$DEST/deploy/soil2crop.service" /etc/systemd/system/soil2crop.service
sudo systemctl daemon-reload
sudo systemctl enable --now soil2crop
sudo systemctl restart soil2crop

echo ">> done. health check:"
sleep 2
curl -s "http://127.0.0.1:8877/health" || echo "(app still warming — retry in ~30 s)"
echo
echo ">> now point Caddy at it:  see deploy/Caddyfile.example (domain: ${DOMAIN:-<your-domain>})"
