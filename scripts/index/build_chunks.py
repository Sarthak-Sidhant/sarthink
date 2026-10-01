"""Structure-aware, traceable chunking. Replaces scripts/semantic/chunk_builder.py.

Two levels ("small-to-big"):
  Sessions  – a whole conversation span (<= SESSION_CAP tokens). What the LLM reads after a hit.
  Chunks    – retrieval windows (~TARGET tokens) inside a session. What gets embedded.
Every chunk links to its exact messages (ChunkMessages), so answers can cite msg_ids and the
viewer can map hits to graph nodes (thread_id -> T_<id>, author user_id -> U_<id>).

Threads are classified by structure, not title:
  chat  – messages reply to the previous one, or have no parent (DMs, group chats)
  tree  – replies branch (Reddit comment trees, tweet reply trees); stitched across threads
          via parent links into one conversation per root

Usage: python3 scripts/index/build_chunks.py [--gap-min 45] [--target 300] [--overlap 0.3]
"""
import argparse
import datetime as dt
import hashlib
import json
import re
from collections import defaultdict

from common import connect_index, snowflake_to_unix

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def ntok(s):
        return len(_enc.encode(s, disallowed_special=()))
except ImportError:
    def ntok(s):
        return int(len(s.split()) * 1.4) + 1

SCHEMA = """
DROP TABLE IF EXISTS Sessions;
DROP TABLE IF EXISTS Chunks;
DROP TABLE IF EXISTS ChunkMessages;
DROP TABLE IF EXISTS MessageSkips;
CREATE TABLE Sessions (
    session_id TEXT PRIMARY KEY,
    platform TEXT, structure TEXT,
    thread_ids TEXT,              -- JSON list of src.Threads.id
    person_ids TEXT,              -- JSON list, ordered by message count
    start_ts INTEGER, end_ts INTEGER,
    message_count INTEGER, token_count INTEGER,
    me_involved INTEGER,          -- 1 if I sent any message in the session
    header TEXT, text TEXT
);
CREATE TABLE Chunks (
    chunk_id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES Sessions(session_id),
    platform TEXT, structure TEXT,
    thread_ids TEXT, person_ids TEXT,
    start_ts INTEGER, end_ts INTEGER,
    message_count INTEGER, token_count INTEGER,
    ego_weight REAL,              -- share of tokens written by me
    me_involved INTEGER,          -- session-level: 1 if I took part in the conversation
    index_policy TEXT,            -- embed | fts_only
    header TEXT, text TEXT
);
CREATE TABLE ChunkMessages (
    chunk_id TEXT, msg_id TEXT, position INTEGER,
    PRIMARY KEY (chunk_id, msg_id)
);
CREATE INDEX idx_cm_msg ON ChunkMessages(msg_id);
CREATE INDEX idx_chunks_session ON Chunks(session_id);
CREATE INDEX idx_chunks_ts ON Chunks(start_ts);
CREATE TABLE MessageSkips (msg_id TEXT PRIMARY KEY, reason TEXT);
"""

URL_RE = re.compile(r"https?://(?:www\.)?([^/\s]+)\S*")
MEDIA_RE = re.compile(
    r"^\s*(<media omitted>|sent an attachment\.?|sent a photo\.?|sent a video\.?|"
    r"\[?attachment\]?|image omitted|video omitted|sticker omitted)\s*$", re.I)

PLATFORM_LABEL = {"twitter": "Twitter", "reddit": "Reddit", "discord": "Discord",
                  "instagram": "Instagram", "facebook": "Facebook"}

HEAD_CAP = 80        # tokens of ancestor context prefixed to tree windows
SESSION_CAP = 3000   # max tokens per session (LLM read unit)
MIN_EMBED_TOKENS = 8  # below this a lone chunk is kept for keyword search only
MIN_SESSION_TOK = 120  # a chat session shorter than this absorbs the next burst across a normal gap
HARD_GAP = 24 * 3600   # ...but never across a gap longer than this


def clean(content):
    s = (content or "").strip()
    if not s:
        return None
    if MEDIA_RE.match(s):
        return "[media]"
    s = URL_RE.sub(lambda m: f"[link {m.group(1)}]", s)
    return re.sub(r"\s+", " ", s)


def fmt_date(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%-d %b %Y")


def fmt_time(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


def stable_id(prefix, msg_ids):
    return prefix + hashlib.sha1("|".join(msg_ids).encode()).hexdigest()[:16]


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def load(conn):
    person_of = {r["user_id"]: r["person_id"] for r in conn.execute("SELECT user_id, person_id FROM PersonAliases")}
    per_chat = {(r[0], r[1]): r[2] for r in conn.execute("SELECT user_id, thread_id, person_id FROM PersonThreads")}
    person_name = {r["person_id"]: r["name"] for r in conn.execute("SELECT person_id, name FROM Persons")}
    threads = {r["id"]: dict(r) for r in conn.execute("SELECT id, platform, title FROM src.Threads")}
    msgs, skips, recovered = {}, [], 0
    for r in conn.execute(
            "SELECT msg_id, thread_id, author_id, timestamp_utc, content, parent_msg_id FROM src.Messages"):
        ts = r["timestamp_utc"]
        if ts is None and r["msg_id"].startswith("twitter_"):
            ts = snowflake_to_unix(r["msg_id"][len("twitter_"):])
            recovered += ts is not None
        if ts is None:
            skips.append((r["msg_id"], "no_timestamp"))
            continue
        text = clean(r["content"])
        if text is None:
            skips.append((r["msg_id"], "empty"))
            continue
        pid = per_chat.get((r["author_id"], r["thread_id"]), person_of.get(r["author_id"]))
        msgs[r["msg_id"]] = {
            "id": r["msg_id"], "thread": r["thread_id"], "ts": ts, "text": text,
            "parent": r["parent_msg_id"], "person": pid,
            "name": person_name.get(pid, "unknown"),
        }
    print(f"Loaded {len(msgs)} messages ({recovered} timestamps recovered from tweet IDs, {len(skips)} skipped)")
    return msgs, threads, skips


def group_conversations(msgs, threads):
    """Stitch threads connected by cross-thread parent links (reply trees spread over threads)."""
    uf = UnionFind()
    for m in msgs.values():
        uf.find(m["thread"])
        p = msgs.get(m["parent"])
        if p and p["thread"] != m["thread"]:
            uf.union(m["thread"], p["thread"])
    convs = defaultdict(list)
    for m in msgs.values():
        convs[uf.find(m["thread"])].append(m)
    for c in convs.values():
        c.sort(key=lambda m: (m["ts"], m["id"]))
    return list(convs.values())


def classify(conv):
    """tree if a meaningful share of all messages reply to something other than the previous
    message (comment trees). Chats with occasional reply-to quotes (Discord) stay chats."""
    if len(conv) < 3:
        return "chat" if len({m["thread"] for m in conv}) == 1 and not any(m["parent"] for m in conv) else "tree"
    branching = sum(1 for prev, m in zip(conv, conv[1:]) if m["parent"] and m["parent"] != prev["id"])
    return "tree" if branching / (len(conv) - 1) > 0.3 else "chat"


# ---------- rendering units ----------
# A "unit" is the smallest thing windows are built from: list of msgs + rendered line.

def chat_units(conv):
    """Merge consecutive messages by the same author within 5 minutes into one turn."""
    units = []
    for m in conv:
        u = units[-1] if units else None
        if u and u["msgs"][-1]["person"] == m["person"] and m["ts"] - u["msgs"][-1]["ts"] <= 300:
            u["msgs"].append(m)
        else:
            units.append({"msgs": [m]})
    for u in units:
        first = u["msgs"][0]
        u["line"] = f"[{fmt_time(first['ts'])}] {first['name']}: " + " / ".join(x["text"] for x in u["msgs"])
        u["tok"] = ntok(u["line"])
    return units


def tree_units(conv):
    """Depth-first order so replies follow their parent; each unit carries its ancestor chain."""
    by_id = {m["id"]: m for m in conv}
    children = defaultdict(list)
    roots = []
    for m in conv:
        if m["parent"] in by_id:
            children[m["parent"]].append(m)
        else:
            roots.append(m)
    units, stack = [], [(r, 0) for r in reversed(roots)]
    while stack:
        m, depth = stack.pop()
        line = f"{'  ' * min(depth, 4)}[{fmt_time(m['ts'])}] {m['name']}: {m['text']}"
        units.append({"msgs": [m], "line": line, "tok": ntok(line), "depth": depth})
        for c in reversed(children[m["id"]]):
            stack.append((c, depth + 1))
    return units, by_id


def split_long(unit, cap):
    """Split one oversize unit (a very long message) into word-bounded pieces."""
    words, pieces, cur = unit["line"].split(" "), [], []
    for w in words:
        cur.append(w)
        if ntok(" ".join(cur)) >= cap:
            pieces.append(" ".join(cur))
            cur = []
    if cur:
        pieces.append(" ".join(cur))
    return [{"msgs": unit["msgs"], "line": p, "tok": ntok(p), **({"depth": unit["depth"]} if "depth" in unit else {})}
            for p in pieces]


def sessionize(units, gap_sec):
    """Split chat units on inactivity gaps, then cap session size. A normal gap only closes a
    session that already has MIN_SESSION_TOK of content, so slow back-and-forth DMs are not
    shredded into one-message sessions; a gap over HARD_GAP always closes it."""
    sessions, cur, cur_tok = [], [], 0
    for u in units:
        delta = u["msgs"][0]["ts"] - cur[-1]["msgs"][-1]["ts"] if cur else 0
        gap = cur and (delta > HARD_GAP or (delta > gap_sec and cur_tok >= MIN_SESSION_TOK))
        continuation = cur and u["msgs"] is cur[-1]["msgs"]  # another piece of the same split unit
        if cur and not continuation and (gap or cur_tok + u["tok"] > SESSION_CAP):
            sessions.append(cur)
            cur, cur_tok = [], 0
        cur.append(u)
        cur_tok += u["tok"]
    if cur:
        sessions.append(cur)
    return sessions


def windows(units, target, overlap):
    """Greedy windows of ~target tokens aligned to unit boundaries; token-capped overlap."""
    out, i, n = [], 0, len(units)
    while i < n:
        j, tok = i, 0
        while j < n and (j == i or tok + units[j]["tok"] <= target):
            tok += units[j]["tok"]
            j += 1
        out.append(units[i:j])
        if j >= n:
            break
        # step back so the next window repeats at most overlap*target tokens (never the whole window)
        back, k = 0, j
        while k - 1 > i and back + units[k - 1]["tok"] <= overlap * target:
            k -= 1
            back += units[k]["tok"]
        i = k
    return out


def ancestor_head(unit, by_id, cap):
    chain, m = [], unit["msgs"][0]
    while m["parent"] in by_id and len(chain) < 6:
        m = by_id[m["parent"]]
        chain.append(f"{m['name']}: {m['text']}")
    if not chain:
        return ""
    head = " ← ".join(chain)
    while ntok(head) > cap and len(head) > 40:
        head = head[: int(len(head) * 0.8)]
    return f"(replying to) {head}\n"


# ---------- emit ----------

def header_for(platform, structure, threads_meta, person_ids, names, start, end):
    title = next((t["title"] for t in threads_meta if t["title"]), "")
    title = re.sub(r"^Tweet Thread \d+$|\s*\[\d+\]", "", title).strip()  # drop ID-only noise
    kind = "conversation" if structure == "chat" else "thread"
    who = ", ".join(names[p] for p in person_ids[:6]) + (f" +{len(person_ids) - 6}" if len(person_ids) > 6 else "")
    when = fmt_date(start) if fmt_date(start) == fmt_date(end) else f"{fmt_date(start)} – {fmt_date(end)}"
    parts = [f"{PLATFORM_LABEL.get(platform, platform)} {kind}", who, when]
    if title and title.lower() not in who.lower():
        parts.append(title[:80])
    return " · ".join(parts)


def people_by_volume(msgs):
    count = defaultdict(int)
    for m in msgs:
        count[m["person"]] += 1
    return sorted(count, key=lambda p: -count[p])


def build(gap_min, target, overlap):
    conn = connect_index()
    conn.executescript(SCHEMA)
    msgs, threads, skips = load(conn)
    names = {r["person_id"]: r["name"] for r in conn.execute("SELECT person_id, name FROM Persons")}
    convs = group_conversations(msgs, threads)

    sess_rows, chunk_rows, cm_rows = [], [], []
    stats = defaultdict(int)

    for conv in convs:
        structure = classify(conv)
        platform = threads[conv[0]["thread"]]["platform"]
        if structure == "chat":
            units = []
            for u in chat_units(conv):
                units.extend(split_long(u, target) if u["tok"] > target else [u])
            session_list = [(s, None) for s in sessionize(units, gap_min * 60)]
        else:
            units, by_id = tree_units(conv)
            units = [p for u in units for p in (split_long(u, target) if u["tok"] > target else [u])]
            # trees are sessions by size only; a reply tree is one conversation regardless of gaps
            session_list = [(s, by_id) for s in sessionize(units, float("inf"))]

        for sess_units, by_id in session_list:
            smsgs = list({m["id"]: m for u in sess_units for m in u["msgs"]}.values())
            s_ids = [m["id"] for m in smsgs]
            pids = people_by_volume(smsgs)
            tids = sorted({m["thread"] for m in smsgs})
            start, end = min(m["ts"] for m in smsgs), max(m["ts"] for m in smsgs)
            s_header = header_for(platform, structure, [threads[t] for t in tids], pids, names, start, end)
            s_text = "\n".join(u["line"] for u in sess_units)
            sid = stable_id("s_", s_ids)
            me_involved = int(1 in pids)
            sess_rows.append((sid, platform, structure, json.dumps(tids), json.dumps(pids), start, end,
                              len(smsgs), ntok(s_text), me_involved, s_header, s_text))

            for w_idx, win in enumerate(windows(sess_units, target, 0 if by_id else overlap)):
                wmsgs = list({m["id"]: m for u in win for m in u["msgs"]}.values())
                w_ids = [m["id"] for m in wmsgs]
                body = "\n".join(u["line"] for u in win)
                if by_id:
                    body = ancestor_head(win[0], by_id, HEAD_CAP) + body
                wp = people_by_volume(wmsgs)
                ws, we = min(m["ts"] for m in wmsgs), max(m["ts"] for m in wmsgs)
                w_header = header_for(platform, structure, [threads[t] for t in sorted({m["thread"] for m in wmsgs})],
                                      wp, names, ws, we)
                tok_by_me = sum(u["tok"] for u in win if u["msgs"][0]["person"] == 1)
                total = sum(u["tok"] for u in win) or 1
                wtok = ntok(body)
                content_tok = sum(ntok(m["text"]) for m in wmsgs)
                policy = "embed" if content_tok >= MIN_EMBED_TOKENS else "fts_only"
                stats[policy] += 1
                cid = f"c_{sid[2:]}_{w_idx}"
                chunk_rows.append((cid, sid, platform, structure, json.dumps(sorted({m["thread"] for m in wmsgs})),
                                   json.dumps(wp), ws, we, len(wmsgs), wtok, round(tok_by_me / total, 3),
                                   me_involved, policy, w_header, body))
                cm_rows.extend((cid, mid, i) for i, mid in enumerate(w_ids))
            stats[structure + "_sessions"] += 1

    conn.executemany("INSERT INTO Sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", sess_rows)
    conn.executemany("INSERT INTO Chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", chunk_rows)
    conn.executemany("INSERT OR IGNORE INTO ChunkMessages VALUES (?,?,?)", cm_rows)
    conn.executemany("INSERT INTO MessageSkips VALUES (?,?)", skips)
    conn.commit()
    print(f"Conversations: {len(convs)} | Sessions: {len(sess_rows)} | Chunks: {len(chunk_rows)}")
    print(dict(stats))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-min", type=float, default=45, help="inactivity gap (minutes) that starts a new chat session")
    ap.add_argument("--target", type=int, default=300, help="target tokens per retrieval chunk")
    ap.add_argument("--overlap", type=float, default=0.3, help="max overlap between chat windows, as share of target")
    a = ap.parse_args()
    build(a.gap_min, a.target, a.overlap)
