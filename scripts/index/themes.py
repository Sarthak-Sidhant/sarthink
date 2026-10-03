"""Recurring themes of a set of conversations, computed locally: no LLM call, free, instant, repeatable.

Each conversation (session) gets one vector, the mean of its chunks' embedding vectors (already built, see
build_embeddings.py). The sessions are clustered with spherical k-means, and each cluster is named after the
topic phrases its sessions' summaries share and the other clusters don't (the summaries already list ~5 topics
each). The description is the summary sentence of the session closest to the cluster's centre.

Used for person and account profiles (scripts/api/agent.py): same output shape the LLM version had,
[{"theme", "description", "sessions", "session_ids"}].
"""
import json
import math
import os
import re
from collections import Counter

import numpy as np

from common import EMBED_MODEL, INDEX_DIR

MAX_THEMES = 8
MODEL_SLUG = {"8b": "qwen3-embedding-8b", "4b": "qwen3-embedding-4b", "0.6b": "qwen3-embedding-0.6b"}
_vec_cache = {"mtime": None, "vecs": None, "row": {}}


def _vectors():
    """Chunk vectors, memory-mapped (nothing is loaded until rows are read); reloaded if the file changes."""
    base = INDEX_DIR / f"emb_{MODEL_SLUG.get(EMBED_MODEL, EMBED_MODEL)}_ctx"
    path = f"{base}.npy"
    mtime = os.path.getmtime(path) if os.path.exists(path) else None
    if mtime != _vec_cache["mtime"]:
        if mtime is None:
            _vec_cache.update(mtime=None, vecs=None, row={})
        else:
            with open(f"{base}.ids.json") as f:
                ids = json.load(f)["ids"]
            _vec_cache.update(mtime=mtime, vecs=np.load(path, mmap_mode="r"), row={c: i for i, c in enumerate(ids)})
    return _vec_cache["vecs"], _vec_cache["row"]


def _session_vectors(sessions, chunk_rows):
    """One unit vector per session (mean of its chunks); sessions without vectors are left out."""
    vecs, row = _vectors()
    by_session = {}
    for sid, cid in chunk_rows:
        if cid in row:
            by_session.setdefault(sid, []).append(row[cid])
    keep, out = [], []
    for s in sessions:
        rows = by_session.get(s[0])
        if vecs is None or not rows:
            continue
        v = np.asarray(vecs[sorted(rows)], dtype=np.float32).mean(axis=0)
        n = np.linalg.norm(v)
        if n > 0:
            keep.append(s)
            out.append(v / n)
    return keep, (np.vstack(out) if out else np.zeros((0, 1), np.float32))


def _kmeans(X, k, iters=30, seed=0):
    """Spherical k-means with k-means++ seeding; deterministic for a given input."""
    rng = np.random.default_rng(seed)
    centres = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = 1 - np.max(X @ np.vstack(centres).T, axis=1)
        d = np.clip(d, 0, None) ** 2
        centres.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
    C = np.vstack(centres)
    labels = None
    for _ in range(iters):
        new = np.argmax(X @ C.T, axis=1)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = X[labels == j]
            if len(m):
                c = m.sum(axis=0)
                C[j] = c / (np.linalg.norm(c) or 1)
    return labels, C


def _norm_topic(t):
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def _first_sentence(text, cap=180):
    text = re.sub(r"\s+", " ", text or "").strip()
    m = re.match(r"(.{20,}?[.!?])(\s|$)", text)
    s = m.group(1) if m else text
    return s if len(s) <= cap else s[: cap - 1].rsplit(" ", 1)[0] + "…"


def _describe(idx, X, centre, data, topics, names):
    """A summary sentence about the theme: from the most central sessions that carry its topics, the first
    sentence mentioning one of the theme's words (summaries open with a generic "X and Y chat on <date>")."""
    words = {w for t in names for w in re.findall(r"\w{4,}", t)}
    central = sorted(idx, key=lambda i: -float(X[i] @ centre))[:8]
    central.sort(key=lambda i: -sum(_norm_topic(t) in names for t in topics[i]))
    for i in central:
        for sent in re.split(r"(?<=[.!?])\s+", data[i].get("summary") or ""):
            if words & set(re.findall(r"\w{4,}", sent.lower())):
                return _first_sentence(sent)
    return _first_sentence(data[central[0]].get("summary"))


def themes(sessions, chunk_rows):
    """sessions: [(session_id, start_ts, summary json)], chunk_rows: [(session_id, chunk_id)] for those sessions."""
    if len(sessions) < 2:
        return []
    keep, X = _session_vectors(sessions, chunk_rows)
    n = len(keep)
    if n < 2:
        return []
    data = [json.loads(s[2]) for s in keep]
    topics = [[t for t in (d.get("topics") or []) if str(t).strip()] for d in data]
    k = max(1, min(MAX_THEMES, round(math.sqrt(n / 3)), n // 2))
    labels, C = _kmeans(X, k) if k > 1 else (np.zeros(n, dtype=int), X.mean(axis=0, keepdims=True))

    overall = Counter(_norm_topic(t) for ts in topics for t in set(ts))
    shown = {}                                   # most common original casing of each topic
    for ts in topics:
        for t in ts:
            shown.setdefault(_norm_topic(t), Counter())[str(t).strip()] += 1
    out, used = [], set()
    for j in sorted(range(k), key=lambda j: -int((labels == j).sum())):
        idx = np.flatnonzero(labels == j)
        if len(idx) < max(2, round(0.02 * n)):
            continue                             # one-off conversations are noise, as before
        inside = Counter(_norm_topic(t) for i in idx for t in set(topics[i]))
        rest = n - len(idx)
        score = {}
        for t, c in inside.items():
            if c < min(2, len(idx)) or t in used:
                continue
            fin, fout = c / len(idx), ((overall[t] - c) / rest) if rest else 0.0
            score[t] = fin * math.log((fin + 0.01) / (fout + 0.01))   # common here, rare elsewhere
        names = [t for t, v in sorted(score.items(), key=lambda x: (-x[1], x[0])) if v > 0][:3]
        if not names:
            continue
        used.update(names)
        out.append({"theme": " · ".join(shown[t].most_common(1)[0][0] for t in names),
                    "description": _describe(idx, X, C[j], data, topics, names),
                    "sessions": len(idx), "session_ids": [keep[i][0] for i in sorted(idx, key=lambda i: keep[i][1])]})
    out.sort(key=lambda t: -t["sessions"])
    return out
