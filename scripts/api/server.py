"""Local Sarthink API: hybrid search over the memory index + the 3D viewer, served from one origin.

Query vectors come from embed_server.py on a GPU box (via an SSH tunnel, EMBED_URL); everything
else — vectors, keyword index, chats — stays on this machine.

  GET /                          the 3D viewer
  GET /api/health
  GET /api/search?q=...&k=10&platform=&person=&since=YYYY-MM-DD&until=&me_only=false
  GET /api/session/{session_id}  full conversation span + LLM summary + neighbours
  GET /api/chunk/{chunk_id}
  GET /api/person?name=...       name -> people (P_) and accounts (A_) matching it
  GET /api/account/{id}[/dossier] one account: info, its person, suggestions / stats + themes
  GET /api/people[/{id}[/dossier]] people (groups of accounts) / one person / stats + themes over their accounts
  POST /api/people, /api/people/{id}/accounts, /name; DELETE /api/people/{id}[/accounts/{aid}]   (instant)
  GET /api/suggestions, POST /api/suggestions/reject   same-name accounts that may be one human

Accounts vs people: docs/accounts-and-people.md. Every hit carries message ids (for citations) and graph node
ids to highlight: A_<account> for who wrote, P_<person> for the people those accounts belong to, T_<thread>.

Run: EMBED_URL=http://127.0.0.1:8100 python3 scripts/api/server.py [--port 8000]
"""
import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "index"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "api"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "utils"))
from common import DATA_DIR, EMBED_MODEL, GRAPH_DIR, PROFILE_CACHE, USER_DB, connect_index  # noqa: E402
from search import Searcher  # noqa: E402
from agent import Agent  # noqa: E402
from people import ME, PeopleStore  # noqa: E402
from themes import subject as theme_subject  # noqa: E402
from pydantic import BaseModel  # noqa: E402

SUMMARY_MODEL, SUMMARY_VERSION = "deepseek-flash", "v3"

app = FastAPI(title="Sarthink")
S: Searcher = None
NAMES = {}          # account_id -> account name
ACCOUNTS = {}       # account_id -> {key, platform, is_me, placeholder, messages}
KEY2AID = {}        # (platform, raw_id, chat) -> account_id
PEOPLE: PeopleStore = None
LOCK = threading.Lock()   # one shared read-only SQLite connection; requests run on worker threads


def locked(fn):
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        with LOCK:
            return fn(*a, **kw)
    return wrapper


IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
LINE_TS = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]")
HEADER_DATE = re.compile(r"^\d{1,2} \w{3} \d{4}( – \d{1,2} \w{3} \d{4})?$")


def iso(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat() if ts else None


def ist_text(text):
    """Chunk/session text stores message times in UTC; show them in IST (Sarthak's timezone).
    Converted at serve time so stored text and embeddings stay unchanged."""
    def conv(m):
        t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)
        return "[" + t.astimezone(IST).strftime("%Y-%m-%d %H:%M") + "]"
    return LINE_TS.sub(conv, text or "")


def ist_header(header, start_ts, end_ts):
    """Replace the UTC date segment of a header ('... · 12 Mar 2023 · ...') with IST dates."""
    fmt = lambda ts: dt.datetime.fromtimestamp(ts, IST).strftime("%-d %b %Y")  # noqa: E731
    when = fmt(start_ts) if fmt(start_ts) == fmt(end_ts) else f"{fmt(start_ts)} – {fmt(end_ts)}"
    parts = header.split(" · ")
    for i, part in enumerate(parts):
        if HEADER_DATE.match(part.strip()):
            parts[i] = when
            break
    return " · ".join(parts)


def to_ts(date_str, end=False):
    if not date_str:
        return None
    d = dt.datetime.fromisoformat(date_str)
    if d.tzinfo is None:
        d = d.replace(tzinfo=IST)      # dates the user/agent types are IST days
    if end and len(date_str) <= 10:
        d += dt.timedelta(days=1)
    return int(d.timestamp())


def people_index():
    """Current people (user DB, small; read fresh so links show up at once): {pid: name}, {pid: [aid]}, {aid: pid}."""
    names = PEOPLE.people()
    members, person_of = {}, {}
    for key, pid in PEOPLE.membership().items():
        aid = KEY2AID.get(key)
        if aid is not None and pid in names:
            members.setdefault(pid, []).append(aid)
            person_of[aid] = pid
    return names, members, person_of


def account_row(aid, person_of=None, names=None):
    a = ACCOUNTS.get(aid)
    if not a:
        raise HTTPException(404, f"account {aid} not found")
    if person_of is None:
        names, _, person_of = people_index()
    pid = person_of.get(aid)
    return {"account_id": aid, "node": f"A_{aid}", "name": NAMES.get(aid, "?"), "platform": a["platform"],
            "is_me": a["is_me"], "placeholder": a["placeholder"], "messages": a["messages"], "key": list(a["key"]),
            "person_id": pid, "person_node": f"P_{pid}" if pid else None, "person": names.get(pid) if pid else None}


def person_row(pid, names=None, members=None):
    if names is None:
        names, members, _ = people_index()
    if pid not in names:
        raise HTTPException(404, f"person {pid} not found")
    aids = sorted(members.get(pid, []), key=lambda a: -ACCOUNTS[a]["messages"])
    return {"person_id": pid, "node": f"P_{pid}", "name": names[pid], "is_me": pid == ME,
            "messages": sum(ACCOUNTS[a]["messages"] for a in aids),
            "platforms": sorted({ACCOUNTS[a]["platform"] for a in aids}),
            "accounts": [{"account_id": a, "node": f"A_{a}", "name": NAMES.get(a, "?"), "platform": ACCOUNTS[a]["platform"],
                          "messages": ACCOUNTS[a]["messages"], "is_me": ACCOUNTS[a]["is_me"]} for a in aids]}


def resolve(ref):
    """'P_<id>' (a person: all their accounts), 'A_<id>' (one account) or a name (people first, then accounts)
    -> {kind, id, node, name, scope, account_ids}."""
    ref = str(ref).strip()
    names, members, person_of = people_index()
    kind, ident = None, None
    if ref[:2] in ("P_", "A_") and ref[2:].isdigit():
        kind, ident = ("person" if ref[0] == "P" else "account"), int(ref[2:])
    elif ref.isdigit():
        kind, ident = "account", int(ref)
    else:
        hits = candidates(ref, limit=1, names=names, members=members, person_of=person_of)
        if not hits:
            raise HTTPException(404, f"no person or account matching {ref!r}")
        kind, ident = hits[0]["kind"], hits[0]["id"]
    if kind == "person":
        p = person_row(ident, names, members)
        return {"kind": "person", "id": ident, "node": p["node"], "name": p["name"], "scope": f"P:{ident}",
                "account_ids": [a["account_id"] for a in p["accounts"]], "accounts": p["accounts"]}
    a = account_row(ident, person_of, names)
    return {"kind": "account", "id": ident, "node": a["node"], "name": a["name"], "scope": f"A:{ident}",
            "account_ids": [ident], "person": a["person"], "person_node": a["person_node"]}


def resolve_accounts(ref):
    return set(resolve(ref)["account_ids"]) if ref else None


def candidates(name, limit=10, names=None, members=None, person_of=None):
    """People whose name matches (with all their accounts) first, then accounts that match."""
    if names is None:
        names, members, person_of = people_index()
    q = name.lower().strip()
    out = []
    for pid, pname in names.items():
        if q and q in pname.lower():
            p = person_row(pid, names, members)
            out.append({"kind": "person", "id": pid, "node": p["node"], "name": pname, "is_me": pid == ME,
                        "platforms": p["platforms"], "accounts": len(p["accounts"]), "messages": p["messages"],
                        "exact": pname.lower() == q})
    rows = S.conn.execute("""SELECT account_id FROM Accounts WHERE lower(name) LIKE ?
                             ORDER BY message_count DESC LIMIT ?""", (f"%{q}%", limit * 3)).fetchall()
    for (aid,) in rows:
        a = account_row(aid, person_of, names)
        out.append({"kind": "account", "id": aid, "node": a["node"], "name": a["name"], "is_me": a["is_me"],
                    "platforms": [a["platform"]], "accounts": 1, "messages": a["messages"],
                    "person": a["person"], "person_node": a["person_node"], "exact": a["name"].lower() == q})
    out.sort(key=lambda c: (not c["exact"], c["kind"] != "person", -c["messages"]))
    return out[:limit]


def nodes_for(account_ids, thread_ids, person_of=None):
    """Graph nodes to light up: the accounts, the people they belong to, the threads."""
    if person_of is None:
        _, _, person_of = people_index()
    pids = []
    for a in account_ids:
        p = person_of.get(a)
        if p and f"P_{p}" not in pids:
            pids.append(f"P_{p}")
    return [f"A_{a}" for a in account_ids] + pids + [f"T_{t}" for t in thread_ids]


def people_for(account_ids, names=None, person_of=None):
    """Who is in a chunk/session: each account with the person it belongs to. My accounts carry my name, as
    the conversation text shows them."""
    if names is None:
        names, _, person_of = people_index()
    return [{"node": f"A_{a}", "name": names.get(ME) if ACCOUNTS.get(a, {}).get("is_me") else NAMES.get(a, "?"),
             "person": names.get(person_of.get(a)) if person_of.get(a) else None,
             "person_node": f"P_{person_of[a]}" if a in person_of else None} for a in account_ids]


def chunk_payload(cid, score=None):
    r = S.conn.execute("""
        SELECT c.*, cc.context FROM Chunks c
        LEFT JOIN ChunkContexts cc ON cc.chunk_id = c.chunk_id AND cc.model = ? AND cc.prompt_version = ?
        WHERE c.chunk_id = ?""", (SUMMARY_MODEL, SUMMARY_VERSION, cid)).fetchone()
    if not r:
        raise HTTPException(404, f"chunk {cid} not found")
    accts = json.loads(r["account_ids"])
    threads = json.loads(r["thread_ids"])
    names, _, person_of = people_index()
    msgs = [m[0] for m in S.conn.execute(
        "SELECT msg_id FROM ChunkMessages WHERE chunk_id = ? ORDER BY position", (cid,))]
    return {
        "chunk_id": cid, "score": score, "session_id": r["session_id"], "platform": r["platform"],
        "header": ist_header(r["header"], r["start_ts"], r["end_ts"]), "context": r["context"],
        "text": ist_text(r["text"]),
        "start": iso(r["start_ts"]), "end": iso(r["end_ts"]), "me_involved": bool(r["me_involved"]),
        "people": people_for(accts, names, person_of),
        "nodes": nodes_for(accts, threads, person_of),
        "message_ids": msgs,
    }


@app.get("/api/health")
@locked
def health():
    enc = None
    if S.encoder_url == "openrouter":       # hosted encoder: no health endpoint of ours to ask
        enc = {"ok": True, "provider": "openrouter"}
    elif S.encoder_url:
        try:
            import httpx
            enc = httpx.get(f"{S.encoder_url}/health", timeout=5).json()
        except Exception as e:  # tunnel down / box stopped
            enc = {"ok": False, "error": str(e)[:200]}
    return {"chunks": len(S.ids), "dense": S.vecs is not None, "encoder": enc}


@app.get("/api/search")
def search(q: str, k: int = Query(10, le=50), platform: str = None, person: str = None,
           since: str = None, until: str = None, me_only: bool = False, mode: str = "hybrid"):
    if mode != "fts":
        S.prefetch([q])                # the query vector comes over the network: fetch it before taking the lock
    with LOCK:
        filters = dict(platform=platform, accounts=resolve_accounts(person), since=to_ts(since),
                       until=to_ts(until, end=True), me_only=me_only)
        try:
            hits = S.search(q, k=k, mode=mode, **filters)
        except Exception as e:  # e.g. encoder unreachable -> fall back to keyword search, say so
            hits = S.search(q, k=k, mode="fts", **filters)
            return {"query": q, "mode": "fts", "warning": f"dense search unavailable: {str(e)[:150]}",
                    "results": [chunk_payload(c, s) for c, s in hits]}
        results = []
        for c, sc in hits:
            try:
                results.append(chunk_payload(c, sc))
            except HTTPException:          # vector index briefly ahead/behind the chunk table during rebuilds
                continue
        return {"query": q, "mode": mode, "filters": {k2: v for k2, v in filters.items() if v}, "results": results}


@app.get("/api/chunk/{chunk_id}")
@locked
def chunk(chunk_id: str):
    return chunk_payload(chunk_id)


@app.get("/api/session/{session_id}")
@locked
def session(session_id: str):
    r = S.conn.execute("SELECT * FROM Sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not r:
        raise HTTPException(404, f"session {session_id} not found")
    summ = S.conn.execute("SELECT data FROM SessionSummaries WHERE session_id=? AND model=? AND prompt_version=?",
                          (session_id, SUMMARY_MODEL, SUMMARY_VERSION)).fetchone()
    nb = lambda op, order: S.conn.execute(  # noqa: E731
        f"SELECT session_id FROM Sessions WHERE thread_ids=? AND platform=? AND start_ts {op} ? "
        f"ORDER BY start_ts {order} LIMIT 1", (r["thread_ids"], r["platform"], r["start_ts"])).fetchone()
    prev, nxt = nb("<", "DESC"), nb(">", "ASC")
    accts, threads = json.loads(r["account_ids"]), json.loads(r["thread_ids"])
    msgs = [m[0] for m in S.conn.execute("""
        SELECT cm.msg_id FROM ChunkMessages cm JOIN Chunks c USING(chunk_id)
        WHERE c.session_id = ? GROUP BY cm.msg_id ORDER BY MIN(c.start_ts), MIN(cm.position)""", (session_id,))]
    return {
        "session_id": session_id, "platform": r["platform"],
        "header": ist_header(r["header"], r["start_ts"], r["end_ts"]), "text": ist_text(r["text"]),
        "message_ids": msgs,
        "start": iso(r["start_ts"]), "end": iso(r["end_ts"]), "message_count": r["message_count"],
        "summary": json.loads(summ["data"]) if summ else None,
        "people": people_for(accts),
        "nodes": nodes_for(accts, threads),
        "previous_session": prev["session_id"] if prev else None,
        "next_session": nxt["session_id"] if nxt else None,
    }


@app.get("/api/person")
@locked
def person(name: str):
    return {"query": name, "candidates": candidates(name)}


class _Backend:
    """Adapter the agent uses: the same functions as the HTTP endpoints (each takes the DB lock itself)."""

    def prefetch(self, queries):
        S.prefetch(queries)              # one encoder request for all phrasings, outside the DB lock

    def search(self, q, k, person=None, platform=None, since=None, until=None, me_only=False):
        return search(q=q, k=k, platform=platform, person=person, since=since, until=until,
                      me_only=bool(me_only), mode="hybrid")["results"]

    def session(self, session_id):
        return session(session_id)

    def person(self, name):
        return person(name)["candidates"]

    def resolve(self, ref):
        with LOCK:
            return resolve(ref)

    def chunk_for_message(self, msg_id):
        with LOCK:
            r = S.conn.execute("SELECT chunk_id FROM ChunkMessages WHERE msg_id = ? LIMIT 1", (msg_id,)).fetchone()
            return chunk_payload(r[0]) if r else None

    def timeline(self, q, k=120, **filters):
        """Month histogram of the top-k hybrid matches for a topic (IST months) + the best hit per busy month."""
        S.prefetch([q])
        with LOCK:
            accts = resolve_accounts(filters.get("person"))
        f = dict(platform=filters.get("platform"), accounts=accts,
                 since=to_ts(filters.get("since")), until=to_ts(filters.get("until"), end=True),
                 me_only=bool(filters.get("me_only")))
        with LOCK:
            hits = S.search(q, k=k, mode="hybrid", **f)
            months, best = {}, {}
            for rank, (cid, score) in enumerate(hits):
                i = S.row.get(cid)
                if i is None:
                    continue
                m = dt.datetime.fromtimestamp(int(S.start_ts[i]), IST).strftime("%Y-%m")
                months[m] = months.get(m, 0) + 1
                best.setdefault(m, cid)           # hits are ranked, so the first per month is its best
            top = sorted(months, key=lambda m: -months[m])[:4]
            return {"months": dict(sorted(months.items())), "busiest": top,
                    "examples": {m: chunk_payload(best[m]) for m in top}}


AGENT = None


class AskRequest(BaseModel):
    question: str
    history: list[dict] = []
    model: str = "deepseek-v4-pro"
    thinking: bool = True
    verify: bool = True    # fact-check pass: unsupported claims 7.8% -> 4.0% on the eval, ~+9s median
    multi: bool = False    # "deep research": planner splits broad questions across parallel researchers (~2 min, ~10x tokens)


@app.post("/api/ask")
def ask(req: AskRequest):
    global AGENT
    key = (req.model, req.thinking, req.verify)
    if AGENT is None or AGENT.key != key:
        AGENT = Agent(_Backend(), model=req.model, thinking=req.thinking, verify=req.verify)
        AGENT.key = key
    return AGENT.ask(req.question, req.history, multi=req.multi)


def _agent():
    global AGENT
    if AGENT is None:
        AGENT = Agent(_Backend())
        AGENT.key = ("deepseek-v4-pro", True, True)
    return AGENT


DOSSIER_VERSION = "v4"   # v4: a person's themes are their accounts' themes
_DOSSIER_LOCK = threading.Lock()


def _dossier_db():
    c = sqlite3.connect(PROFILE_CACHE)
    c.execute("CREATE TABLE IF NOT EXISTS ScopeDossiers (scope TEXT PRIMARY KEY, version TEXT, stamp TEXT, "
              "data TEXT, built TEXT)")
    return c


def _dossier(sc, refresh=False):
    """Stats, monthly activity, themes with example conversations and the busiest chats of a scope (one account,
    or a person = all their accounts). Cached until the accounts in it, their counts or the summaries change,
    so attaching an account rebuilds that person's dossier on the next view and nobody else's."""
    with LOCK:
        msgs = sum(ACCOUNTS[a]["messages"] for a in sc["account_ids"])
        n = S.conn.execute("SELECT COUNT(*) FROM SessionSummaries").fetchone()[0]
    stamp = json.dumps([sorted(sc["account_ids"]), msgs, n, sc["name"]])
    with _DOSSIER_LOCK:
        c = _dossier_db()
        row = None if refresh else c.execute("SELECT data FROM ScopeDossiers WHERE scope=? AND version=? AND stamp=?",
                                             (sc["scope"], DOSSIER_VERSION, stamp)).fetchone()
        c.close()
    if row:
        return {**json.loads(row[0]), "cached": True}
    d = _build_dossier(sc)
    with _DOSSIER_LOCK:
        c = _dossier_db()
        c.execute("INSERT OR REPLACE INTO ScopeDossiers VALUES (?,?,?,?,?)",
                  (sc["scope"], DOSSIER_VERSION, stamp, json.dumps(d, ensure_ascii=False),
                   dt.datetime.now(IST).isoformat(timespec="minutes")))
        c.commit()
        c.close()
    return {**d, "cached": False}


THEMES_SHOWN = 8


def _theme_examples(theme_lists):
    """Date, header and summary of the last 3 sessions of each theme: one query for all of them."""
    want = [sid for ts in theme_lists for t in ts for sid in t["session_ids"][-3:]]
    brief = {}
    if want:
        with LOCK:
            for sid, start, end, header, data in S.conn.execute(f"""
                    SELECT s.session_id, s.start_ts, s.end_ts, s.header, x.data FROM Sessions s
                    LEFT JOIN SessionSummaries x ON x.session_id = s.session_id AND x.model = ? AND x.prompt_version = ?
                    WHERE s.session_id IN ({",".join("?" * len(want))})""", (SUMMARY_MODEL, SUMMARY_VERSION, *want)):
                brief[sid] = {"session_id": sid, "date": iso(start)[:10], "header": ist_header(header, start, end),
                              "summary": json.loads(data).get("summary") if data else None}
    return [[{"theme": t["theme"], "description": t["description"], "example": t.get("example"), "sessions": t["sessions"],
              "topics": t.get("topics", []), "examples": [brief[sid] for sid in t["session_ids"][-3:] if sid in brief]}
             for t in ts] for ts in theme_lists]


def _shared_themes(account_themes):
    """Themes on different accounts of one person that share a subject (not just 'banter' or 'twitter')."""
    links = []
    for i, a in enumerate(account_themes):
        for b in account_themes[i + 1:]:
            for x, ta in enumerate(a["themes"]):
                for y, tb in enumerate(b["themes"]):
                    shared = sorted(k for k in set(ta["topics"]) & set(tb["topics"]) if theme_subject(k))
                    if shared:
                        links.append({"a": f"{a['node']}:{x}", "b": f"{b['node']}:{y}", "shared": shared[:4]})
    return links


def _build_dossier(sc):
    """One account: its stats and themes. A person: totals, activity and chats over all their accounts, and the
    themes of each account under that account (themes belong to accounts; a person is made of accounts)."""
    ag = _agent()
    person = sc["kind"] == "person"
    prof = ag.profiles.get(sc["scope"], sc["account_ids"], sc["name"], themes=not person)
    q = lambda sql, args=(): ag.sql.conn.execute(sql, args).fetchall()  # noqa: E731
    threads = ag.profiles.threads(sc["account_ids"])     # takes ag.sql.lock itself, so not inside the block below
    tj = json.dumps(threads)            # one parameter: a person can span tens of thousands of chats
    with ag.sql.lock:
        months = q("""SELECT substr(day,1,7) m, COUNT(*), SUM(is_me) FROM msg_stats
                      WHERE thread_id IN (SELECT value FROM json_each(?)) AND day IS NOT NULL GROUP BY m ORDER BY m""", (tj,))
        chats = q("""SELECT thread_id, MAX(thread), MAX(platform), COUNT(*), MIN(day), MAX(day) FROM msg_stats
                     WHERE thread_id IN (SELECT value FROM json_each(?)) GROUP BY thread_id ORDER BY COUNT(*) DESC LIMIT 5""",
                  (tj,))
    out = {"kind": sc["kind"], "id": sc["id"], "node": sc["node"], "name": sc["name"],
           "platforms": prof.get("platforms"), "messages_total": prof.get("messages_total"),
           "messages_by_you": prof.get("messages_by_sarthak"), "messages_by_them": prof.get("messages_by_them"),
           "first_day": prof.get("first_day"), "last_day": prof.get("last_day"),
           "months": [{"month": m, "total": n, "you": y} for m, n, y in months],
           "top_chats": [{"node": f"T_{t}", "title": re.sub(r"\s*\[\d+\]", "", ti or ""), "platform": p,
                          "messages": n, "from": a, "to": b} for t, ti, p, n, a, b in chats]}
    if not person:
        out["themes"] = _theme_examples([prof.get("themes", [])[:THEMES_SHOWN]])[0]
        return out
    accts = []
    for a in sc["accounts"]:
        if not a["messages"]:
            continue
        ap = ag.profiles.get(f"A:{a['account_id']}", [a["account_id"]], a["name"])
        accts.append({"node": a["node"], "account_id": a["account_id"], "name": a["name"], "platform": a["platform"],
                      "messages_total": ap.get("messages_total"), "messages_by_you": ap.get("messages_by_sarthak"),
                      "messages_by_them": ap.get("messages_by_them"), "first_day": ap.get("first_day"),
                      "last_day": ap.get("last_day"), "themes": ap.get("themes", [])[:6]})
    for a, ts in zip(accts, _theme_examples([a["themes"] for a in accts])):
        a["themes"] = ts
    out["themes"] = []                   # a person has no blended themes: see account_themes
    out["account_themes"] = accts
    out["shared_themes"] = _shared_themes(accts)
    return out


# ─── Accounts and people (docs/accounts-and-people.md) ──────────────────────────────────────────────────
def _suggestions_for(aid, names, members, person_of):
    """People/accounts that may be the same human as this account: same normalised name on another platform.
    Pairs the user turned down, and accounts already in the same person, are left out."""
    mine = person_of.get(aid)
    out, seen = [], set()
    for (ids,) in S.conn.execute("SELECT account_ids FROM MergeCandidates"):
        ids = json.loads(ids)
        if aid not in ids:
            continue
        for o in ids:
            if o == aid or o in seen or (mine and person_of.get(o) == mine) or ACCOUNTS.get(o, {}).get("is_me"):
                continue
            seen.add(o)
            if PEOPLE.rejected([ACCOUNTS[aid]["key"]], [ACCOUNTS[o]["key"]]):
                continue
            out.append({**account_row(o, person_of, names), "why": "same name on another platform"})
    return out


@app.get("/api/account/{aid}")
@locked
def account_info(aid: int):
    names, members, person_of = people_index()
    a = account_row(aid, person_of, names)
    return {**a, "suggestions": [] if a["is_me"] else _suggestions_for(aid, names, members, person_of)}


@app.get("/api/account/{aid}/dossier")
def account_dossier(aid: int, refresh: bool = False):
    with LOCK:
        sc = resolve(f"A_{aid}")
    return _dossier(sc, refresh)


@app.get("/api/accounts")
@locked
def accounts_search(q: str = "", limit: int = Query(30, le=500), unlinked: bool = False, platform: str = None):
    """Accounts by name, biggest first; unlinked=true: only accounts that belong to no person."""
    names, _, person_of = people_index()
    rows = S.conn.execute("SELECT account_id FROM Accounts WHERE lower(name) LIKE ? AND (? IS NULL OR platform = ?) "
                          "AND message_count > 0 ORDER BY message_count DESC", (f"%{q.lower()}%", platform, platform)).fetchall()
    out = []
    for (aid,) in rows:
        if unlinked and aid in person_of:
            continue
        out.append(account_row(aid, person_of, names))
        if len(out) >= limit:
            break
    return {"accounts": out}


@app.get("/api/people")
@locked
def people_list():
    """Every person with their accounts (also the graph's people layer: each person is drawn linked to these)."""
    names, members, _ = people_index()
    out = [person_row(pid, names, members) for pid in names]
    out.sort(key=lambda p: (not p["is_me"], -p["messages"]))
    return {"people": out}


@app.get("/api/people/{pid}")
@locked
def people_get(pid: int):
    names, members, person_of = people_index()
    p = person_row(pid, names, members)
    sugg, seen = [], set()
    if pid != ME:
        for a in p["accounts"]:
            for s_ in _suggestions_for(a["account_id"], names, members, person_of):
                if s_["account_id"] not in seen:
                    seen.add(s_["account_id"])
                    sugg.append(s_)
    return {**p, "suggestions": sugg}


@app.get("/api/people/{pid}/dossier")
def people_dossier(pid: int, refresh: bool = False):
    with LOCK:
        sc = resolve(f"P_{pid}")
    if not sc["account_ids"]:
        return {"kind": "person", "id": pid, "node": f"P_{pid}", "name": sc["name"], "messages_total": 0, "themes": []}
    return _dossier(sc, refresh)


class CreatePerson(BaseModel):
    name: str
    account_ids: list[int] = []


class AttachAccount(BaseModel):
    account_id: int


class NameRequest(BaseModel):
    name: str


class RejectRequest(BaseModel):
    a: int
    b: int


def _key(aid):
    if aid not in ACCOUNTS:
        raise HTTPException(404, f"account {aid} not found")
    if ACCOUNTS[aid]["is_me"]:
        raise HTTPException(400, "your own accounts always belong to you (identity_map.json)")
    return ACCOUNTS[aid]["key"]


@app.post("/api/people")
@locked
def people_create(body: CreatePerson):
    """A new person, optionally with accounts (each moves out of any other person)."""
    keys = [_key(a) for a in body.account_ids]
    pid = PEOPLE.create(body.name, keys)
    return person_row(pid)


@app.post("/api/people/{pid}/accounts")
@locked
def people_attach(pid: int, body: AttachAccount):
    if pid == ME:
        raise HTTPException(400, "your own accounts come from identity_map.json")
    names, _, _ = people_index()
    if pid not in names:
        raise HTTPException(404, f"person {pid} not found")
    PEOPLE.attach(pid, _key(body.account_id))
    return person_row(pid)


@app.delete("/api/people/{pid}/accounts/{aid}")
@locked
def people_detach(pid: int, aid: int):
    key = _key(aid)
    if PEOPLE.person_of(key) != pid:
        raise HTTPException(404, "that account is not part of this person")
    PEOPLE.detach(key)
    return person_row(pid)


@app.post("/api/people/{pid}/name")
@locked
def people_rename(pid: int, body: NameRequest):
    if not body.name.strip():
        raise HTTPException(400, "a name can't be empty")
    person_row(pid)
    PEOPLE.rename(pid, body.name)
    return person_row(pid)


@app.delete("/api/people/{pid}")
@locked
def people_delete(pid: int):
    """Remove a person; their accounts stay, just without a person."""
    person_row(pid)
    try:
        PEOPLE.delete(pid)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"deleted": pid}


@app.get("/api/suggestions")
@locked
def suggestions(limit: int = Query(60, le=500)):
    """Same-name accounts on different platforms that are not yet in the same person, biggest first."""
    names, members, person_of = people_index()
    out, seen = [], set()
    for (ids,) in S.conn.execute("SELECT account_ids FROM MergeCandidates"):
        ids = [i for i in json.loads(ids) if i in ACCOUNTS and not ACCOUNTS[i]["is_me"]]
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                pa, pb = person_of.get(a), person_of.get(b)
                if (a, b) in seen or (pa and pa == pb) or PEOPLE.rejected([ACCOUNTS[a]["key"]], [ACCOUNTS[b]["key"]]):
                    continue
                seen.add((a, b))
                out.append({"a": account_row(a, person_of, names), "b": account_row(b, person_of, names)})
    out.sort(key=lambda m: -(m["a"]["messages"] + m["b"]["messages"]))
    return {"suggestions": out[:limit]}


@app.post("/api/suggestions/reject")
@locked
def suggestions_reject(body: RejectRequest):
    """Not the same human: stop suggesting this pair."""
    PEOPLE.reject([_key(body.a)], [_key(body.b)])
    return {"ok": True}


# ─── Second-brain features: commitments inbox, on this day, day review ─────────────────────────────────
def _status_db():
    c = sqlite3.connect(USER_DB)
    c.execute("CREATE TABLE IF NOT EXISTS CommitmentStatus (cid TEXT PRIMARY KEY, status TEXT, updated TEXT)")
    return c


@app.get("/api/commitments")
def commitments(owner: str = "all", status: str = "open", q: str = "", limit: int = Query(40, le=200),
                offset: int = 0):
    st = dict(_status_db().execute("SELECT cid, status FROM CommitmentStatus").fetchall())
    with LOCK:
        rows = S.conn.execute("SELECT * FROM Commitments ORDER BY COALESCE(due, said_on) DESC").fetchall()
    items = []
    for r in rows:
        s = st.get(r["cid"], "open")
        if owner != "all" and r["owner"] != owner:
            continue
        if status != "all" and s != status:
            continue
        if q and q.lower() not in f"{r['title']} {r['what']} {r['people']}".lower():
            continue
        items.append({"cid": r["cid"], "owner": r["owner"], "who": r["who"], "title": r["title"], "what": r["what"],
                      "due": r["due"], "said_on": r["said_on"], "platform": r["platform"],
                      "people": json.loads(r["people"]), "nodes": json.loads(r["nodes"]),
                      "session_id": r["session_id"], "status": s})
    counts = {"total": len(items), "mine": sum(i["owner"] == "me" for i in items),
              "theirs": sum(i["owner"] == "them" for i in items), "with_due": sum(bool(i["due"]) for i in items)}
    return {"counts": counts, "items": items[offset:offset + limit]}


class StatusUpdate(BaseModel):
    status: str   # open | done | dismissed | calendar


@app.post("/api/commitments/{cid}")
def set_commitment(cid: str, body: StatusUpdate):
    if body.status not in ("open", "done", "dismissed", "calendar"):
        raise HTTPException(400, "status must be open, done, dismissed or calendar")
    c = _status_db()
    c.execute("INSERT OR REPLACE INTO CommitmentStatus VALUES (?,?,?)",
              (cid, body.status, dt.datetime.now(IST).isoformat(timespec="minutes")))
    c.commit()
    return {"cid": cid, "status": body.status}


@app.get("/api/commitments.ics")
def commitments_ics():
    """Calendar file of everything added to the calendar that has a due date (all-day events)."""
    st = dict(_status_db().execute("SELECT cid, status FROM CommitmentStatus WHERE status='calendar'").fetchall())
    with LOCK:
        rows = [r for r in S.conn.execute("SELECT * FROM Commitments WHERE due IS NOT NULL") if r["cid"] in st]
    esc_ics = lambda s: str(s).replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;").replace("\n", " ")  # noqa: E731
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Sarthink//commitments//EN"]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for r in rows:
        d = r["due"].replace("-", "")
        nxt = (dt.date.fromisoformat(r["due"]) + dt.timedelta(days=1)).strftime("%Y%m%d")
        who = ", ".join(json.loads(r["people"]))
        lines += ["BEGIN:VEVENT", f"UID:{r['cid']}@sarthink", f"DTSTAMP:{stamp}", f"DTSTART;VALUE=DATE:{d}",
                  f"DTEND;VALUE=DATE:{nxt}", f"SUMMARY:{esc_ics(r['title'])}",
                  f"DESCRIPTION:{esc_ics(r['what'] + (' — with ' + who if who else '') + ' (said ' + r['said_on'] + ')')}",
                  "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return Response("\r\n".join(lines) + "\r\n", media_type="text/calendar",
                    headers={"Content-Disposition": "attachment; filename=sarthink-commitments.ics"})


def _sessions_on(where, args, limit):
    with LOCK:
        rows = S.conn.execute(f"""
            SELECT s.session_id, s.header, s.platform, s.start_ts, s.message_count, s.account_ids, s.thread_ids, x.data
            FROM Sessions s LEFT JOIN SessionSummaries x ON x.session_id = s.session_id
                 AND x.model = ? AND x.prompt_version = ?
            WHERE s.me_involved = 1 AND {where} ORDER BY s.message_count DESC LIMIT ?""",
                              (SUMMARY_MODEL, SUMMARY_VERSION, *args, limit)).fetchall()
    out = []
    _, _, person_of = people_index()
    for r in rows:
        summ = json.loads(r["data"]).get("summary") if r["data"] else None
        out.append({"session_id": r["session_id"], "header": ist_header(r["header"], r["start_ts"], r["start_ts"]),
                    "platform": r["platform"], "time": dt.datetime.fromtimestamp(r["start_ts"], IST).strftime("%Y-%m-%d %H:%M"),
                    "messages": r["message_count"], "summary": summ,
                    "nodes": nodes_for(json.loads(r["account_ids"]), json.loads(r["thread_ids"]), person_of)})
    return out


@app.get("/api/on_this_day")
def on_this_day(date: str = None):
    """Conversations from this calendar day (IST) in earlier years, biggest first."""
    d = dt.date.fromisoformat(date) if date else dt.datetime.now(IST).date()
    md = d.strftime("%m-%d")
    items = _sessions_on("strftime('%m-%d', s.start_ts, 'unixepoch', '+330 minutes') = ? "
                         "AND strftime('%Y', s.start_ts, 'unixepoch', '+330 minutes') < ?", (md, str(d.year)), 40)
    years = {}
    for it in items:
        years.setdefault(it["time"][:4], []).append(it)
    return {"date": d.isoformat(), "years": dict(sorted(years.items(), reverse=True))}


@app.get("/api/day")
def day(date: str):
    items = _sessions_on("date(s.start_ts, 'unixepoch', '+330 minutes') = ?", (date,), 60)
    return {"date": date, "sessions": items}


@app.post("/api/day_review")
def day_review(date: str):
    """A short journal-style recap of one day, written only from that day's conversation summaries."""
    items = [it for it in day(date)["sessions"] if it["summary"]][:25]
    if not items:
        return {"date": date, "review": "No conversations with summaries on this day.", "sources": []}
    ag = _agent()
    if not ag.client:
        return {"date": date, "review": "Day reviews need a DeepSeek API key in `.env`.", "sources": items}
    blocks = "\n".join(f"[{i + 1}] {it['time'][11:]} · {it['header']}\n{it['summary']}" for i, it in enumerate(items))
    r = ag.client.chat.completions.create(
        model="deepseek-flash", max_tokens=2000, extra_body={"thinking": {"type": "disabled"}},
        messages=[{"role": "user", "content":
                   f"Write a short first-person journal entry for Sarthak for {date} (as \"you\", 4-8 sentences), "
                   "using ONLY these summaries of his conversations that day. Cite each fact with [n]. Mention who he "
                   "talked to and what happened; no speculation about feelings unless stated; no addresses or phone "
                   f"numbers.\n\n{blocks}"}])
    from agent import redact
    return {"date": date, "review": redact(r.choices[0].message.content or ""),
            "sources": [{"n": i + 1, **it} for i, it in enumerate(items)]}


@app.post("/api/ask_stream")
def ask_stream(req: AskRequest):
    """Server-sent events: {"type":"step",...} progress lines while the agent works, then {"type":"answer",...}."""
    import queue
    global AGENT
    key = (req.model, req.thinking, req.verify)
    if AGENT is None or AGENT.key != key:
        AGENT = Agent(_Backend(), model=req.model, thinking=req.thinking, verify=req.verify)
        AGENT.key = key
    q = queue.Queue()

    def work():
        try:
            res = AGENT.ask(req.question, req.history, multi=req.multi, emit=q.put)
            q.put({"type": "answer", "data": res})
        except Exception as e:
            q.put({"type": "error", "message": str(e)[:300]})

    threading.Thread(target=work, daemon=True).start()

    def events():
        while True:
            ev = q.get()
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if ev["type"] in ("answer", "error"):
                break

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def viewer():
    return FileResponse(REPO_ROOT / "sarthink_graph.html")


@app.middleware("http")
async def revalidate_graph(request, call_next):
    """Graph files change on rebuilds; make browsers revalidate them (cheap 304 via ETag)."""
    resp = await call_next(request)
    if request.url.path.startswith("/processed_data/graph"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


app.mount("/processed_data/graph", StaticFiles(directory=GRAPH_DIR), name="graph")   # URL kept for the viewer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=EMBED_MODEL, help="embedding model: 8b (GPU encoder) or 0.6b (runs on CPU)")
    a = ap.parse_args()
    S = Searcher(a.model, "ctx", encoder_url=os.environ.get("EMBED_URL"))
    S.conn = connect_index(check_same_thread=False)
    for r in S.conn.execute("SELECT account_id, name, platform, raw_id, chat, is_me, placeholder, message_count "
                            "FROM Accounts"):
        key = (r["platform"], r["raw_id"], r["chat"])
        NAMES[r["account_id"]] = r["name"]
        KEY2AID[key] = r["account_id"]
        ACCOUNTS[r["account_id"]] = {"key": key, "platform": r["platform"], "is_me": bool(r["is_me"]),
                                     "placeholder": bool(r["placeholder"]), "messages": r["message_count"]}
    from common import IDENTITY_MAP
    PEOPLE = PeopleStore(USER_DB, json.load(open(IDENTITY_MAP)).get("master_persona", "Me"))
    PEOPLE.import_merges(DATA_DIR / "person_merges.json")      # a hand-written list, imported once
    print(f"Data: {DATA_DIR} | {len(S.ids)} chunks | encoder: {S.encoder_url or 'local ' + a.model + ' model'}")

    def _warm():          # dossiers take ~30-40s to build (yours spans the whole archive): have them ready, not on
        for pid in [ME] + [p for p in PEOPLE.people() if p != ME]:      # first click (cached after)
            try:
                people_dossier(pid)
            except Exception as e:
                print(f"[warm-up] person {pid}: {type(e).__name__}: {e}")
    threading.Thread(target=_warm, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port)
