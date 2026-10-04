#!/usr/bin/env bash
# Soil2Crop — one-command local run (Linux / macOS).
# Creates a venv, installs deps, fetches the TabPFN-3.5 checkpoint, serves the app.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

PORT="${PORT:-8877}"
python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

mkdir -p models
CKPT="models/tabpfn-v3.5-20260909.safetensors"
if [ ! -f "$CKPT" ]; then
  echo ">> downloading TabPFN-3.5 checkpoint (~876 MB) ..."
  curl -L --fail -o "$CKPT" \
    https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/main/tabpfn-v3.5-20260909.safetensors
fi

export SOILHACK_CKPT="$REPO/$CKPT"
export SOILHACK_LIBRARY="${SOILHACK_LIBRARY:-$REPO/data/samples/reference_library.csv}"
export PYTHONPATH="$REPO/src"
echo ">> serving on http://127.0.0.1:${PORT}  (first request warms the model, ~30-90 s on CPU)"
exec python -m uvicorn app.server:app --host 127.0.0.1 --port "$PORT"
