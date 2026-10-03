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


STOP = {"and", "the", "of", "with", "about", "for", "to", "in", "on", "a", "an", "chat", "chats", "talk", "talks",
        "discussion", "discussions", "conversation", "conversations", "casual", "general", "some", "various"}
# words that say where or how people talk, not what about: two themes sharing only these share no subject
GENERIC = {"twitter", "tweet", "reddit", "whatsapp", "discord", "instagram", "facebook", "telegram", "dm", "group",
           "banter", "exchange", "argument", "thread", "reply", "repli", "joke", "meme", "mem", "link", "messag",
           "friend", "friendly", "teas", "roast", "humor", "humour", "greet", "update", "plan", "sarthak"}


def subject(key):
    """True if a theme topic key (space-separated stemmed words) names a subject, not just a medium or a tone."""
    return any(w not in GENERIC for w in key.split())


MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def _stem(w):
    for suf in ("ing", "ed", "es", "s", "y"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _canon(t):
    """A topic phrase's meaning, for spotting near-duplicates: its content words, stemmed.
    "group chat banter" = "group banter"; "flirty banter" = "flirting and banter"."""
    words = [w for w in re.findall(r"[^\W\d_]{2,}", t.lower()) if w not in STOP]
    return frozenset(_stem(w) for w in words) or frozenset([t.lower()])


def _similar(a, b):
    return a <= b or b <= a


def _chat_title(title):
    """A chat's title, or None when it is only an id ("Tweet Thread 123", "DM 1-2", "Ghost Thread 9")."""
    t = re.sub(r"\s*\[\d+\]", "", title or "").strip()
    if not t or re.match(r"^(tweet|ghost|reddit) (thread|post)\b|^dm \d|^\d+$", t, re.I):
        return None
    return re.sub(r"^Direct Messages - ", "DMs with ", t)


def _span(tss):
    import datetime as dt
    d = [dt.datetime.utcfromtimestamp(t) for t in tss if t]
    if not d:
        return ""
    a, b = min(d), max(d)
    f = lambda x: f"{MONTHS[x.month - 1]} {x.year}"  # noqa: E731
    if (a.year, a.month) == (b.year, b.month):
        return f(a)
    return f"{MONTHS[a.month - 1]} – {f(b)}" if a.year == b.year else f"{f(a)} – {f(b)}"


def _where(idx, titles):
    """Where a theme happens: its main chat, or how many chats it spans."""
    named = Counter(t for t in (_chat_title(titles[i]) for i in idx) if t)
    chats = len({titles[i] for i in idx})
    if named:
        top, n = named.most_common(1)[0]
        if n >= 0.5 * len(idx):
            return f"Mostly in “{top}”"
        return f"In “{top}” and {chats - 1} other chat{'s' if chats > 2 else ''}" if chats > 1 else f"In “{top}”"
    kinds = Counter(_kind(titles[i]) for i in idx)
    noun = kinds.most_common(1)[0][0] if kinds else "chat"
    return f"Across {chats} {noun}{'s' if chats > 1 else ''}"


def _kind(title):
    """What an untitled chat is: a tweet thread, a Reddit thread, a DM."""
    t = (title or "").lower()
    return ("tweet thread" if re.match(r"^(tweet|ghost) thread", t) else "Reddit thread" if t.startswith("reddit")
            else "DM" if re.match(r"^dm\b", t) else "chat")


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
    """sessions: [(session_id, start_ts, summary json[, chat title])], chunk_rows: [(session_id, chunk_id)].
    Each theme: name (its distinctive topics), description (where and when), an example sentence, its main
    topics (to link similar themes across accounts) and its sessions."""
    if len(sessions) < 2:
        return []
    keep, X = _session_vectors(sessions, chunk_rows)
    n = len(keep)
    if n < 2:
        return []
    data = [json.loads(s[2]) for s in keep]
    titles = [s[3] if len(s) > 3 else "" for s in keep]
    topics = [[t for t in (d.get("topics") or []) if str(t).strip()] for d in data]
    keys = [{_canon(_norm_topic(t)) for t in ts} for ts in topics]
    k = max(1, min(MAX_THEMES, round(math.sqrt(n / 3)), n // 2))
    labels, C = _kmeans(X, k) if k > 1 else (np.zeros(n, dtype=int), X.mean(axis=0, keepdims=True))

    overall = Counter(c for ks in keys for c in ks)
    shown = {}                                   # the most common wording of each topic (near-duplicates merged)
    for ts in topics:
        for t in ts:
            shown.setdefault(_canon(_norm_topic(t)), Counter())[str(t).strip()] += 1
    out, used = [], []
    for j in sorted(range(k), key=lambda j: -int((labels == j).sum())):
        idx = np.flatnonzero(labels == j)
        if len(idx) < max(2, round(0.02 * n)):
            continue                             # one-off conversations are noise, as before
        inside = Counter(c for i in idx for c in keys[i])
        rest = n - len(idx)
        score = {}
        for t, c in inside.items():
            if c < min(2, len(idx)) or any(_similar(t, u) for u in used):
                continue
            fin, fout = c / len(idx), ((overall[t] - c) / rest) if rest else 0.0
            score[t] = fin * math.log((fin + 0.01) / (fout + 0.01))   # common here, rare elsewhere
        names = []
        for t, v in sorted(score.items(), key=lambda x: (-x[1], sorted(x[0]))):
            if v <= 0 or len(names) == 3:
                break
            if not any(_similar(t, u) for u in names):
                names.append(t)
        if not names:
            continue
        used.extend(names)
        words = [shown[t].most_common(1)[0][0] for t in names]
        when = _span([keep[i][1] for i in idx])
        out.append({"theme": " · ".join(words),
                    "description": _where(idx, titles) + (f" · {when}" if when else ""),
                    "example": _describe(idx, X, C[j], data, topics, {_norm_topic(w) for w in words}),
                    "topics": [" ".join(sorted(t)) for t, _ in inside.most_common(8)],
                    "sessions": len(idx), "session_ids": [keep[i][0] for i in sorted(idx, key=lambda i: keep[i][1])]})
    out.sort(key=lambda t: -t["sessions"])
    return out
