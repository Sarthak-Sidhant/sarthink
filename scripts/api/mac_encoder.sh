#!/usr/bin/env bash
# Make this Mac the 8B query encoder for the VPS: runs embed_server.py on the Apple GPU (MPS) and keeps a
# reverse SSH tunnel open so the VPS reaches it at its own 127.0.0.1:8101. Stops the Mac sleeping while it runs.
#   usage: scripts/api/mac_encoder.sh [vps=root@185.2.102.128]
# One-time setup:  python3 -m venv .venv && .venv/bin/pip install torch sentence-transformers fastapi uvicorn
#                  ssh-copy-id root@185.2.102.128
# Close the lid / quit (Ctrl+C) and the VPS falls back to keyword search until this runs again.
set -u
cd "$(dirname "$0")/../.."
VPS=${1:-root@185.2.102.128}
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3

"$PY" scripts/index/embed_server.py --model Qwen/Qwen3-Embedding-8B --port 8100 &
SERVER=$!
caffeinate -i -w "$SERVER" &
trap 'kill $SERVER 2>/dev/null' EXIT

echo "Loading the model (first run downloads ~16 GB)..."
until curl -sf http://127.0.0.1:8100/health | grep -q '"ok":true'; do
  kill -0 "$SERVER" 2>/dev/null || { echo "embed_server exited"; exit 1; }
  sleep 2
done
echo "Encoder ready: $(curl -s http://127.0.0.1:8100/health)"

while kill -0 "$SERVER" 2>/dev/null; do
  ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ConnectTimeout=20 \
      -N -R 127.0.0.1:8101:127.0.0.1:8100 "$VPS"
  echo "[$(date +%H:%M:%S)] tunnel to VPS dropped (exit $?), reconnecting in 3s"
  sleep 3
done
