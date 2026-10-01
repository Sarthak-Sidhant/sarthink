"""Merge freshly embedded chunks into the existing vector file, aligned to the current Chunks table.
Rows for re-embedded chunks come from the update; every other row is reused by chunk id.
Fails loudly if any current chunk has no vector.

Usage: python3 scripts/index/merge_embeddings.py <update_prefix> [--model qwen3-embedding-8b --variant ctx]
"""
import argparse
import json
import shutil

import numpy as np

from build_embeddings import OUT_DIR
from common import connect_index


def main(a):
    base = OUT_DIR / f"emb_{a.model}_{a.variant}"
    old = np.load(f"{base}.npy")
    meta = json.load(open(f"{base}.ids.json"))
    old_row = {cid: i for i, cid in enumerate(meta["ids"])}
    upd = np.load(f"{a.update}.npy")
    upd_row = {cid: i for i, cid in enumerate(json.load(open(f"{a.update}.ids.json")))}

    ids = [r[0] for r in connect_index().execute("SELECT chunk_id FROM Chunks ORDER BY chunk_id")]
    out = np.zeros((len(ids), old.shape[1]), dtype=np.float16)
    missing, reused, fresh = [], 0, 0
    for i, cid in enumerate(ids):
        if cid in upd_row:
            out[i] = upd[upd_row[cid]]
            fresh += 1
        elif cid in old_row:
            out[i] = old[old_row[cid]]
            reused += 1
        else:
            missing.append(cid)
    if missing:
        raise SystemExit(f"{len(missing)} chunks have no vector (e.g. {missing[:3]}); nothing written")
    shutil.copy(f"{base}.npy", f"{base}.prev.npy")
    shutil.copy(f"{base}.ids.json", f"{base}.prev.ids.json")
    np.save(f"{base}.npy", out)
    json.dump({**meta, "ids": ids}, open(f"{base}.ids.json", "w"))
    print(f"wrote {out.shape}: {fresh} re-embedded, {reused} reused (previous file kept as .prev)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("update")
    ap.add_argument("--model", default="qwen3-embedding-8b")
    ap.add_argument("--variant", default="ctx")
    main(ap.parse_args())
