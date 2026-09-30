#!/usr/bin/env bash
# Replaces the tail of remote_embed.sh once the 8B ctx embedding is saved: skip the slow 8B raw pass,
# measure the context-line effect on 0.6B instead.
set -uo pipefail
cd /workspace/sarthink/scripts/index
log() { echo "[$(date +%H:%M:%S)] $*"; }
until [ -f ../../processed_data/index/emb_qwen3-embedding-8b_ctx.ids.json ]; do sleep 10; done
pkill -f remote_embed.sh; sleep 1; pkill -f "variant raw"; sleep 3
export HF_HUB_OFFLINE=1
log "embedding 0.6b ctx"; python3 build_embeddings.py --model Qwen/Qwen3-Embedding-0.6B --variant ctx --batch-size 128
log "embedding 0.6b raw"; python3 build_embeddings.py --model Qwen/Qwen3-Embedding-0.6B --variant raw --batch-size 128
log "embedding 4b ctx";   python3 build_embeddings.py --model Qwen/Qwen3-Embedding-4B   --variant ctx --batch-size 64
log "needle eval"
python3 needle_eval.py run --configs \
  0.6b/raw:dense 0.6b/ctx:dense 4b/ctx:dense 8b/ctx:dense \
  8b/ctx:fts 0.6b/ctx:hybrid 4b/ctx:hybrid 8b/ctx:hybrid \
  0.6b/ctx:hybrid+rerank 8b/ctx:hybrid+rerank
log "ALL DONE"
