#!/usr/bin/env bash
# Start Sarthink locally: SSH tunnel to the GPU encoder box + the API/viewer on http://127.0.0.1:8000
#   usage: scripts/api/run.sh <ssh_host> <ssh_port>      e.g. scripts/api/run.sh ssh6.vast.ai 30618
# Without args the API still runs, with keyword-only search.
set -u
cd "$(dirname "$0")/../.."
HOST=${1:-}; PORT=${2:-}
if [ -n "$HOST" ]; then
  ssh -i ~/.ssh/vast_sarthink -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
      -N -L 8100:127.0.0.1:8100 -p "$PORT" "root@$HOST" &
  TUNNEL=$!
  trap 'kill $TUNNEL 2>/dev/null' EXIT
  sleep 3
  export EMBED_URL=http://127.0.0.1:8100
fi
python3 scripts/api/server.py --port 8000
