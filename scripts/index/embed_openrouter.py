"""Embed chunks with Qwen3-Embedding-8B on OpenRouter: no GPU to rent. Same weights as the GPU-built vectors
(cosine ~0.9998), $0.01 per million tokens (~$0.08 for 26k chunks).

Writes <out>.npy (float16) + <out>.ids.json, the format merge_embeddings.py folds into the main vector file:
    python3 scripts/index/embed_openrouter.py processed_data/index/updates/wa --platform whatsapp
    python3 scripts/index/merge_embeddings.py processed_data/index/updates/wa
--missing embeds only chunks that have no vector yet (after an import). Texts are built exactly as
build_embeddings.py builds them (documents get no instruction prefix; queries do, see search.py).
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
from dotenv import dotenv_values

from build_embeddings import OUT_DIR, chunk_texts
from common import REPO_ROOT, connect_index
from search import OPENROUTER_ROUTE

BATCH_CHARS = 40_000      # per request; chunks are <= ~1,000 tokens
BATCH_MAX = 64


def batches(items):
    cur, chars = [], 0
    for cid, text in items:
        if cur and (chars + len(text) > BATCH_CHARS or len(cur) >= BATCH_MAX):
            yield cur
            cur, chars = [], 0
        cur.append((cid, text))
        chars += len(text)
    if cur:
        yield cur


def main(a):
    key = os.environ.get("OPENROUTER_API_KEY") or dotenv_values(REPO_ROOT / ".env").get("OPENROUTER_API_KEY")
    http = httpx.Client(base_url="https://openrouter.ai/api/v1", headers={"Authorization": f"Bearer {key}"},
                        timeout=httpx.Timeout(120, connect=10))
    conn = connect_index()
    base = OUT_DIR / f"emb_{a.model.split('/')[-1]}_ctx"
    meta = json.load(open(f"{base}.ids.json"))
    ids, texts = chunk_texts(conn, "ctx", meta.get("ctx_model", "deepseek-flash"), meta.get("ctx_version", "v3"))
    if a.platform:
        keep = {r[0] for r in conn.execute("SELECT chunk_id FROM Chunks WHERE platform = ?", (a.platform,))}
    else:
        keep = set(ids)
    if a.missing:
        keep -= set(meta["ids"])
    items = [(c, t) for c, t in zip(ids, texts) if c in keep]
    print(f"{len(items)} chunks to embed with {a.model} via OpenRouter")
    todo = list(batches(items))
    tokens, done, t0 = 0, 0, time.time()

    def embed(batch):
        for attempt in range(6):
            try:
                r = http.post("/embeddings", json={"model": a.model, "input": [t for _, t in batch],
                                                   "provider": OPENROUTER_ROUTE})
                r.raise_for_status()
                data = sorted(r.json()["data"], key=lambda d: d["index"])
                if len(data) != len(batch):
                    raise ValueError(f"got {len(data)} vectors for {len(batch)} texts")
                v = np.asarray([d["embedding"] for d in data], dtype=np.float32)
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                return batch, v, (r.json().get("usage") or {}).get("prompt_tokens", 0)
            except Exception as e:  # rate limit, provider hiccup: back off and retry
                err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"batch failed after retries: {err}")

    out_ids, out_vecs = [], []
    with ThreadPoolExecutor(a.workers) as ex:
        for batch, v, n in ex.map(embed, todo):
            out_ids += [c for c, _ in batch]
            out_vecs.append(v.astype(np.float16))
            tokens += n
            done += len(batch)
            if len(out_vecs) % 20 == 0 or done == len(items):
                print(f"  {done}/{len(items)}  {done / (time.time() - t0):.0f}/s  tokens {tokens:,}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.save(f"{a.out}.npy", np.vstack(out_vecs))
    json.dump(out_ids, open(f"{a.out}.ids.json", "w"))
    print(f"saved {len(out_ids)} vectors to {a.out}.npy in {time.time() - t0:.0f}s; "
          f"{tokens:,} tokens ≈ ${tokens / 1e6 * 0.01:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="output prefix, e.g. processed_data/index/updates/wa")
    ap.add_argument("--platform", help="only chunks of this platform")
    ap.add_argument("--missing", action="store_true", help="only chunks without a vector in the main file")
    ap.add_argument("--model", default="qwen/qwen3-embedding-8b")
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args())
