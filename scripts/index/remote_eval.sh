#!/usr/bin/env bash
# Needle eval split into one process per model group so each gets the whole GPU.
set -uo pipefail
cd /workspace/sarthink/scripts/index
export HF_HUB_OFFLINE=1
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "eval 8b";     python3 needle_eval.py run --configs 8b/ctx:dense 8b/ctx:fts 8b/ctx:hybrid
log "eval 4b";     python3 needle_eval.py run --configs 4b/ctx:hybrid
log "eval 0.6b";   python3 needle_eval.py run --configs 0.6b/ctx:hybrid
log "eval rerank"; python3 needle_eval.py run --configs 8b/ctx:hybrid+rerank 0.6b/ctx:hybrid+rerank
log "ALL DONE"
