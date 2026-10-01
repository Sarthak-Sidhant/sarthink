#!/usr/bin/env bash
# Self-healing SSH tunnel to the GPU embed_server: reconnects whenever the vast.ai proxy drops the link.
#   usage: scripts/api/tunnel.sh <ssh_host> <ssh_port> [local_port=8101]
set -u
HOST=$1; PORT=$2; LOCAL=${3:-8101}
while true; do
  ssh -i ~/.ssh/vast_sarthink -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      -o ConnectTimeout=20 -N -L "$LOCAL:127.0.0.1:8100" -p "$PORT" "root@$HOST"
  echo "[$(date +%H:%M:%S)] tunnel dropped (exit $?), reconnecting in 3s"
  sleep 3
done
