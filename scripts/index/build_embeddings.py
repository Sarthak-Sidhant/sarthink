"""Embed chunks into a flat vector file (exact search in numpy; 54k vectors needs no vector DB).

Variants (so retrieval experiments can compare them):
  raw  – header + chunk text
  ctx  – header + LLM chunk context (if any) + chunk text

Output: processed_data/index/emb_<model-slug>_<variant>.npy (float16, L2-normalised)
        processed_data/index/emb_<model-slug>_<variant>.ids.json (chunk_ids, same row order)
Resumable per batch-block; runs on cuda / mps / cpu (auto).

Usage: python3 scripts/index/build_embeddings.py --model Qwen/Qwen3-Embedding-0.6B --variant ctx
"""
import argparse
import json
import time

import numpy as np

from common import REPO_ROOT, connect_index

OUT_DIR = REPO_ROOT / "processed_data" / "index"


def slug(model):
    return model.split("/")[-1].lower()


def chunk_texts(conn, variant, ctx_model, ctx_version):
    rows = conn.execute("""
        SELECT c.chunk_id, c.header, c.text, cc.context
        FROM Chunks c
        LEFT JOIN ChunkContexts cc ON cc.chunk_id = c.chunk_id AND cc.model = ? AND cc.prompt_version = ?
        ORDER BY c.chunk_id""", (ctx_model, ctx_version)).fetchall()
    ids, texts = [], []
    for r in rows:
        parts = [r["header"]]
        if variant == "ctx" and r["context"]:
            parts.append(f"Context: {r['context']}")
        parts.append(r["text"])
        ids.append(r["chunk_id"])
        texts.append("\n".join(parts))
    return ids, texts


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(a):
    from sentence_transformers import SentenceTransformer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / f"emb_{slug(a.model)}_{a.variant}"
    conn = connect_index()
    ids, texts = chunk_texts(conn, a.variant, a.ctx_model, a.ctx_version)
    if a.limit:
        ids, texts = ids[:a.limit], texts[:a.limit]

    device = a.device or pick_device()
    kwargs = {}
    if device == "cuda":
        import torch
        kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
    model = SentenceTransformer(a.model, device=device, **kwargs)
    model.max_seq_length = a.max_len
    print(f"Embedding {len(texts)} chunks with {a.model} on {device} -> {base}.npy")

    # sort by length so batches pad little; restore order afterwards
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vecs = None
    if a.reuse:
        # Only texts longer than the old limit change when max_len grows; keep the rest from the old file.
        old = np.load(a.reuse)
        with open(a.reuse.replace(".npy", ".ids.json")) as f:
            old_ids = json.load(f)["ids"]
        assert old_ids == ids, "reuse file rows do not match current chunk ids"
        tok = model.tokenizer
        order = [i for i in order if len(tok(texts[i], add_special_tokens=False)["input_ids"]) > a.reuse_len]
        vecs = old.copy()
        print(f"Reusing {len(texts) - len(order)} vectors; re-embedding {len(order)} texts longer than {a.reuse_len} tokens")
    t0 = time.time()
    block = a.batch_size * 50
    for start in range(0, len(order), block):
        idx = order[start:start + block]
        e = model.encode([texts[i] for i in idx], batch_size=a.batch_size, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
        if vecs is None:
            vecs = np.zeros((len(texts), e.shape[1]), dtype=np.float16)
        vecs[idx] = e.astype(np.float16)
        done = min(start + block, len(order))
        rate = done / (time.time() - t0)
        print(f"  {done}/{len(order)}  {rate:.1f} chunks/s  eta {(len(order) - done) / rate / 60:.1f} min", flush=True)

    np.save(f"{base}.npy", vecs)
    with open(f"{base}.ids.json", "w") as f:
        json.dump({"model": a.model, "variant": a.variant, "ctx_model": a.ctx_model,
                   "ctx_version": a.ctx_version, "ids": ids}, f)
    print(f"Saved {vecs.shape} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--variant", choices=["raw", "ctx"], default="ctx")
    ap.add_argument("--ctx-model", default="deepseek-flash")
    ap.add_argument("--ctx-version", default="v3")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse", default=None, help="old .npy built with a smaller max-len; only longer texts are re-embedded")
    ap.add_argument("--reuse-len", type=int, default=512, help="max-len the --reuse file was built with")
    main(ap.parse_args())
