"""Local Sarthink API: hybrid search over the memory index + the 3D viewer, served from one origin.

Query vectors come from embed_server.py on a GPU box (via an SSH tunnel, EMBED_URL); everything
else — vectors, keyword index, chats — stays on this machine.

  GET /                          the 3D viewer
  GET /api/health
  GET /api/search?q=...&k=10&platform=&person=&since=YYYY-MM-DD&until=&me_only=false
  GET /api/session/{session_id}  full conversation span + LLM summary + neighbours
  GET /api/chunk/{chunk_id}
  GET /api/person?name=...       name -> person candidates

Every hit carries message ids (for citations) and graph node ids (P_<person>, T_<thread>) to highlight.

Run: EMBED_URL=http://127.0.0.1:8100 python3 scripts/api/server.py [--port 8000]
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "index"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "api"))
from common import connect_index  # noqa: E402
from search import Searcher  # noqa: E402
from agent import Agent  # noqa: E402
from pydantic import BaseModel  # noqa: E402

SUMMARY_MODEL, SUMMARY_VERSION = "deepseek-flash", "v3"

app = FastAPI(title="Sarthink")
S: Searcher = None
NAMES = {}
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


def resolve_person(name_or_id):
    if not name_or_id:
        return None
    if str(name_or_id).lstrip("P_").isdigit():
        return int(str(name_or_id).lstrip("P_"))
    hits = person_candidates(name_or_id, limit=1)
    if not hits:
        raise HTTPException(404, f"no person matching {name_or_id!r}")
    return hits[0]["person_id"]


def person_candidates(name, limit=10):
    rows = S.conn.execute("""
        SELECT p.person_id, p.name, p.is_me, a.platform, COUNT(m.msg_id) AS messages
        FROM Persons p JOIN PersonAliases a USING(person_id)
        LEFT JOIN src.Messages m ON m.author_id = a.user_id
        WHERE lower(p.name) LIKE ? OR lower(a.name) LIKE ?
        GROUP BY p.person_id ORDER BY messages DESC LIMIT ?""",
                          (f"%{name.lower()}%", f"%{name.lower()}%", limit)).fetchall()
    return [{"person_id": r["person_id"], "node": f"P_{r['person_id']}", "name": r["name"], "is_me": bool(r["is_me"]),
             "platform": r["platform"], "messages": r["messages"]} for r in rows]


def chunk_payload(cid, score=None):
    r = S.conn.execute("""
        SELECT c.*, cc.context FROM Chunks c
        LEFT JOIN ChunkContexts cc ON cc.chunk_id = c.chunk_id AND cc.model = ? AND cc.prompt_version = ?
        WHERE c.chunk_id = ?""", (SUMMARY_MODEL, SUMMARY_VERSION, cid)).fetchone()
    if not r:
        raise HTTPException(404, f"chunk {cid} not found")
    persons = json.loads(r["person_ids"])
    threads = json.loads(r["thread_ids"])
    msgs = [m[0] for m in S.conn.execute(
        "SELECT msg_id FROM ChunkMessages WHERE chunk_id = ? ORDER BY position", (cid,))]
    return {
        "chunk_id": cid, "score": score, "session_id": r["session_id"], "platform": r["platform"],
        "header": ist_header(r["header"], r["start_ts"], r["end_ts"]), "context": r["context"],
        "text": ist_text(r["text"]),
        "start": iso(r["start_ts"]), "end": iso(r["end_ts"]), "me_involved": bool(r["me_involved"]),
        "people": [{"node": f"P_{p}", "name": NAMES.get(p, "?")} for p in persons],
        "nodes": [f"P_{p}" for p in persons] + [f"T_{t}" for t in threads],
        "message_ids": msgs,
    }


@app.get("/api/health")
@locked
def health():
    enc = None
    if S.encoder_url:
        try:
            import httpx
            enc = httpx.get(f"{S.encoder_url}/health", timeout=5).json()
        except Exception as e:  # tunnel down / box stopped
            enc = {"ok": False, "error": str(e)[:200]}
    return {"chunks": len(S.ids), "dense": S.vecs is not None, "encoder": enc}


@app.get("/api/search")
@locked
def search(q: str, k: int = Query(10, le=50), platform: str = None, person: str = None,
           since: str = None, until: str = None, me_only: bool = False, mode: str = "hybrid"):
    filters = dict(platform=platform, person_id=resolve_person(person), since=to_ts(since),
                   until=to_ts(until, end=True), me_only=me_only)
    try:
        hits = S.search(q, k=k, mode=mode, **filters)
    except Exception as e:  # e.g. encoder unreachable -> fall back to keyword search, say so
        hits = S.search(q, k=k, mode="fts", **filters)
        return {"query": q, "mode": "fts", "warning": f"dense search unavailable: {str(e)[:150]}",
                "results": [chunk_payload(c, s) for c, s in hits]}
    return {"query": q, "mode": mode, "filters": {k2: v for k2, v in filters.items() if v},
            "results": [chunk_payload(c, s) for c, s in hits]}


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
    persons, threads = json.loads(r["person_ids"]), json.loads(r["thread_ids"])
    msgs = [m[0] for m in S.conn.execute("""
        SELECT cm.msg_id FROM ChunkMessages cm JOIN Chunks c USING(chunk_id)
        WHERE c.session_id = ? GROUP BY cm.msg_id ORDER BY MIN(c.start_ts), MIN(cm.position)""", (session_id,))]
    return {
        "session_id": session_id, "platform": r["platform"],
        "header": ist_header(r["header"], r["start_ts"], r["end_ts"]), "text": ist_text(r["text"]),
        "message_ids": msgs,
        "start": iso(r["start_ts"]), "end": iso(r["end_ts"]), "message_count": r["message_count"],
        "summary": json.loads(summ["data"]) if summ else None,
        "people": [{"node": f"P_{p}", "name": NAMES.get(p, "?")} for p in persons],
        "nodes": [f"P_{p}" for p in persons] + [f"T_{t}" for t in threads],
        "previous_session": prev["session_id"] if prev else None,
        "next_session": nxt["session_id"] if nxt else None,
    }


@app.get("/api/person")
@locked
def person(name: str):
    return {"query": name, "candidates": person_candidates(name)}


class _Backend:
    """Adapter the agent uses: the same functions as the HTTP endpoints (each takes the DB lock itself)."""

    def search(self, q, k, person=None, platform=None, since=None, until=None, me_only=False):
        return search(q=q, k=k, platform=platform, person=person, since=since, until=until,
                      me_only=bool(me_only), mode="hybrid")["results"]

    def session(self, session_id):
        return session(session_id)

    def person(self, name):
        return person(name)["candidates"]

    def chunk_for_message(self, msg_id):
        with LOCK:
            r = S.conn.execute("SELECT chunk_id FROM ChunkMessages WHERE msg_id = ? LIMIT 1", (msg_id,)).fetchone()
            return chunk_payload(r[0]) if r else None

    def timeline(self, q, k=120, **filters):
        """Month histogram of the top-k hybrid matches for a topic (IST months) + the best hit per busy month."""
        f = dict(platform=filters.get("platform"), person_id=resolve_person(filters.get("person")),
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


app.mount("/processed_data/graph", StaticFiles(directory=REPO_ROOT / "processed_data" / "graph"), name="graph")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="8b")
    a = ap.parse_args()
    S = Searcher(a.model, "ctx", encoder_url=os.environ.get("EMBED_URL"))
    S.conn = connect_index(check_same_thread=False)
    NAMES = {r["person_id"]: r["name"] for r in S.conn.execute("SELECT person_id, name FROM Persons")}
    print(f"Loaded {len(S.ids)} chunks; encoder: {S.encoder_url or 'local model'}")
    uvicorn.run(app, host="127.0.0.1", port=a.port)
