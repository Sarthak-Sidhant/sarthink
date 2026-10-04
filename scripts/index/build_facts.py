"""Memory layer: lasting facts about you, your people and the things around you, with time and receipts.

The session summaries already hold ~150k atomic facts ("facts") and who-did-what ("people"). This turns them into
what everything else reads (docs/PLAN-memory.md, Phase 1):

  pass 1   normalise   per session (batched): each summary fact → subject, kind, category, lasting or not, and a
                       self-contained statement dated to the conversation. Only lasting facts are kept (RawFacts).
  link                 subjects → entities: you (self), the accounts/people in that conversation, or a named place,
                       organisation, project or thing (Entities, aliases merged).
  pass 2   consolidate per entity: its raw facts are embedded and clustered by topic; topics with several facts
                       go to the model, which merges repeats and dates what changed → current and past facts with
                       valid_from/valid_to, how often confirmed, and the sessions they come from (Facts).

Everything is derived and labelled (model, prompt version); evidence is the session, whose messages the API can
show. Incremental: pass 1 skips sessions already done, pass 2 only redoes entities with new raw facts.

Usage:
  python3 scripts/index/build_facts.py pass1 [--limit N --sample]      # cost test first
  python3 scripts/index/build_facts.py link
  python3 scripts/index/build_facts.py pass2 [--entity NAME] [--limit N]
  python3 scripts/index/build_facts.py all
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import random
import re

from dotenv import dotenv_values
from openai import OpenAI

from common import REPO_ROOT, connect_index

MODEL = "deepseek-flash"
P1_VERSION, P2_VERSION = "f1", "f1"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SESSIONS_PER_CALL = 6
KINDS = ("self", "person", "place", "org", "project", "thing")
CATEGORIES = ("identity", "relationship", "education", "work", "location", "possession", "health", "money",
              "preference", "belief", "event", "goal", "plan", "skill", "family", "social")

SCHEMA = """
CREATE TABLE IF NOT EXISTS RawFacts (
    rf_id TEXT PRIMARY KEY, session_id TEXT, said_on TEXT, subject TEXT, kind TEXT, category TEXT,
    statement TEXT, entity_id INTEGER, model TEXT, version TEXT);
CREATE INDEX IF NOT EXISTS idx_rawfacts_session ON RawFacts(session_id);
CREATE INDEX IF NOT EXISTS idx_rawfacts_entity ON RawFacts(entity_id);
CREATE TABLE IF NOT EXISTS FactSessions (session_id TEXT PRIMARY KEY, version TEXT, n INTEGER);
CREATE TABLE IF NOT EXISTS Entities (
    entity_id INTEGER PRIMARY KEY, kind TEXT, name TEXT, norm TEXT UNIQUE, aliases TEXT DEFAULT '[]',
    account_id INTEGER, person_id INTEGER);
CREATE TABLE IF NOT EXISTS Facts (
    fact_id TEXT PRIMARY KEY, entity_id INTEGER, category TEXT, statement TEXT, valid_from TEXT, valid_to TEXT,
    status TEXT, confidence INTEGER, sources TEXT, model TEXT, version TEXT, updated TEXT);
CREATE INDEX IF NOT EXISTS idx_facts_entity ON Facts(entity_id);
CREATE TABLE IF NOT EXISTS FactEntityState (entity_id INTEGER PRIMARY KEY, raw_count INTEGER, updated TEXT);
"""

P1 = """Below are conversations of Sarthak's (date, who took part, and facts a summariser pulled out of each).
Keep only facts that belong in a picture of someone's LIFE: would it still be worth knowing a month from now?
KEEP: where someone studies/works/lives and moves; relationships (dating, family, friendships, falling out);
real life events (admissions, exams, jobs, trips, illnesses, purchases of note, launches); lasting possessions
(laptop, phone, bike); health; significant money matters (salary, loans, big payments); skills and what they build;
goals and plans that matter; strong personal preferences (food, music, people) and core values.
DROP: chat moves (asked, replied, joked, agreed, shared a link, sent a sticker), banter, sarcasm, hypotheticals,
one-off opinions on random topics ("xerox costs 2 rupees"), trivia about strangers, technical details of a single
task, logistics already over, and anything about the conversation itself. Never keep phone numbers, emails, IDs,
passwords or addresses. When in doubt, drop it: fewer, solid facts beat many weak ones.
Examples — keep: "Sarthak got admission to BTech CSE (cybersecurity) at Ramaiah." / "Devanshi likes Oreo Dairy
Milk." / "Mihir works at X and hasn't been paid yet." Drop: "Sarthak shared four amazon links." / "Nisarga's
friend has a houseparty." / "Sarthak said GeM is not scraped by them."
For each kept fact give:
- "c": conversation number
- "subject": who/what it is about: "Sarthak" for him; a participant exactly as named in that conversation's
  people list; or the name of a place, organisation, project or thing
- "kind": one of {kinds}
- "category": one of {categories}
- "statement": one self-contained sentence in English, naming the subject, with the date if it matters
  ("As of 2026-06-04, Sarthak plans to study in Bangalore.")
Return JSON: {{"facts": [{{"c": 0, "subject": "...", "kind": "...", "category": "...", "statement": "..."}}]}}

{conversations}"""

def client():
    return OpenAI(api_key=dotenv_values(REPO_ROOT / ".env")["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com",
                  timeout=600)


def ask(cl, prompt, thinking=False, max_tokens=8000):
    for _ in range(3):
        try:
            r = cl.chat.completions.create(
                model=MODEL, response_format={"type": "json_object"}, max_tokens=max_tokens,
                extra_body=None if thinking else {"thinking": {"type": "disabled"}},
                messages=[{"role": "user", "content": prompt}])
            return json.loads(r.choices[0].message.content or ""), r.usage
        except Exception as e:    # refusal, timeout, empty or bad JSON
            err = e
    return {"error": str(err)[:200]}, None


def norm(name):
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


# ─── pass 1 ───────────────────────────────────────────────────────────────────

def pass1(conn, cl, limit=0, sample=False, since=None):
    conn.executescript(SCHEMA)
    names = dict(conn.execute("SELECT account_id, name FROM Accounts"))
    rows = conn.execute("""
        SELECT x.session_id, s.start_ts, s.account_ids, x.data FROM SessionSummaries x JOIN Sessions s USING(session_id)
        WHERE x.model = 'deepseek-flash' AND x.prompt_version = 'v3'
          AND x.session_id NOT IN (SELECT session_id FROM FactSessions WHERE version = ?)
          AND s.start_ts >= strftime('%s', ?)
        ORDER BY s.start_ts""", (P1_VERSION, since or "1970-01-01")).fetchall()
    if sample:
        random.Random(7).shuffle(rows)
    if limit:
        rows = rows[:limit]
    convs = []
    for sid, ts, aids, data in rows:
        d = json.loads(data)
        if not d.get("facts"):
            continue
        convs.append({"sid": sid, "day": dt.datetime.fromtimestamp(ts, IST).strftime("%Y-%m-%d"),
                      "who": [names.get(a, "?") for a in json.loads(aids)],
                      "people": [p.get("name") for p in d.get("people") or [] if isinstance(p, dict)],
                      "facts": d["facts"]})
    batches = [convs[i:i + SESSIONS_PER_CALL] for i in range(0, len(convs), SESSIONS_PER_CALL)]
    print(f"pass 1: {len(convs)} conversations in {len(batches)} calls", flush=True)

    def run(batch):
        text = "\n\n".join(
            f"### Conversation {i} — {c['day']} — participants: {', '.join(c['who'])}\n"
            f"people: {', '.join(n for n in c['people'] if n)}\n" + "\n".join(f"- {f}" for f in c["facts"])
            for i, c in enumerate(batch))
        out, usage = ask(cl, P1.format(kinds=", ".join(KINDS), categories=", ".join(CATEGORIES), conversations=text))
        return batch, out, usage

    tin = tout = kept = 0
    done = 0
    with cf.ThreadPoolExecutor(16) as ex:
        for batch, out, usage in ex.map(run, batches):
            if usage:
                tin, tout = tin + usage.prompt_tokens, tout + usage.completion_tokens
            if "error" in out:
                continue                      # these sessions stay undone and are retried next run
            rows_out, per = [], {c["sid"]: 0 for c in batch}
            for f in out.get("facts", []):
                i = f.get("c")
                if not isinstance(i, int) or not 0 <= i < len(batch) or not f.get("statement"):
                    continue
                c = batch[i]
                kind = f.get("kind") if f.get("kind") in KINDS else "thing"
                cat = f.get("category") if f.get("category") in CATEGORIES else "event"
                rf = hashlib.sha1(f"{c['sid']}|{f['statement']}".encode()).hexdigest()[:16]
                rows_out.append((rf, c["sid"], c["day"], str(f.get("subject") or "")[:120], kind, cat,
                                 f["statement"][:600], None, MODEL, P1_VERSION))
                per[c["sid"]] += 1
            conn.executemany("INSERT OR REPLACE INTO RawFacts VALUES (?,?,?,?,?,?,?,?,?,?)", rows_out)
            conn.executemany("INSERT OR REPLACE INTO FactSessions VALUES (?,?,?)",
                             [(s, P1_VERSION, n) for s, n in per.items()])
            conn.commit()
            kept += len(rows_out)
            done += len(batch)
            if done % 300 < SESSIONS_PER_CALL:
                print(f"  {done}/{len(convs)} conversations, {kept:,} lasting facts, tokens in {tin:,} out {tout:,}",
                      flush=True)
    print(f"pass 1 done: {kept:,} lasting facts from {done} conversations | tokens in {tin:,} out {tout:,}")


# ─── link ─────────────────────────────────────────────────────────────────────

def link(conn):
    """Raw fact subjects → entities. You are one entity; people/accounts come from the conversation's participants
    (matched by name), so 'Mihir Karbelkar GOAT' in one chat and 'Mihir' in another land on the same person when
    their accounts are grouped; anything else is a named place/org/project/thing, merged by normalised name."""
    conn.executescript(SCHEMA)
    import sqlite3
    from common import USER_DB
    with sqlite3.connect(USER_DB) as u:
        pa = {(p, r, c): pid for p, r, c, pid in u.execute("SELECT platform, raw_id, chat, person_id FROM PersonAccounts")}
        pnames = dict(u.execute("SELECT person_id, name FROM People"))
    acc = {r[0]: r for r in conn.execute("SELECT account_id, name, platform, raw_id, chat, is_me FROM Accounts")}
    person_of = {aid: pa.get((r[2], r[3], r[4])) for aid, r in acc.items()}
    sess_accts = {sid: json.loads(a) for sid, a in conn.execute(
        "SELECT session_id, account_ids FROM Sessions WHERE session_id IN (SELECT DISTINCT session_id FROM RawFacts)")}
    cache = {}

    def entity(kind, name, account_id=None, person_id=None):
        key = f"p:{person_id}" if person_id else f"a:{account_id}" if account_id else f"{kind}:{norm(name)}"
        if key in cache:
            return cache[key]
        row = conn.execute("SELECT entity_id, aliases FROM Entities WHERE norm = ?", (key,)).fetchone()
        if row:
            eid, aliases = row[0], json.loads(row[1])
            if name and name not in aliases:
                conn.execute("UPDATE Entities SET aliases = ? WHERE entity_id = ?",
                             (json.dumps((aliases + [name])[-20:], ensure_ascii=False), eid))
        else:
            eid = conn.execute("INSERT INTO Entities(kind, name, norm, aliases, account_id, person_id) VALUES (?,?,?,?,?,?)",
                               (kind, name, key, json.dumps([name], ensure_ascii=False), account_id, person_id)).lastrowid
        cache[key] = eid
        return eid

    me = entity("self", "Sarthak", person_id=1)
    n = 0
    for rf, sid, subject, kind in conn.execute(
            "SELECT rf_id, session_id, subject, kind FROM RawFacts WHERE entity_id IS NULL").fetchall():
        s = norm(subject)
        if kind == "self" or s in ("sarthak", "sarthak sidhant", "sarthak sidhant me", "me"):
            eid = me
        elif kind == "person":
            best = None
            for a in sess_accts.get(sid, []):
                if acc.get(a) and not acc[a][5]:
                    an = norm(acc[a][1])
                    if s and (s == an or s in an or an in s or s.split()[0] == an.split()[0] if an else False):
                        best = a
                        break
            if best is not None:
                pid = person_of.get(best)
                eid = entity("person", pnames.get(pid) if pid else acc[best][1], account_id=None if pid else best,
                             person_id=pid)
            else:
                eid = entity("person", subject)      # someone mentioned, not in the chat
        else:
            eid = entity(kind, subject)
        conn.execute("UPDATE RawFacts SET entity_id = ? WHERE rf_id = ?", (eid, rf))
        n += 1
    conn.commit()
    top = conn.execute("""SELECT e.kind, e.name, COUNT(*) FROM RawFacts r JOIN Entities e USING(entity_id)
                          GROUP BY e.entity_id ORDER BY 3 DESC LIMIT 12""").fetchall()
    print(f"linked {n:,} raw facts; entities: {conn.execute('SELECT COUNT(*) FROM Entities').fetchone()[0]:,}")
    for k, nm, c in top:
        print(f"   {c:>6,}  {k:<8} {nm}")


# ─── pass 2 ───────────────────────────────────────────────────────────────────
# One call per person doesn't scale (you have thousands of facts) and without room to think the model just copies.
# So, like Graphiti: each raw fact is embedded, an entity's facts are clustered by meaning (locally), and only
# clusters with several facts go to the model — in small batches, with thinking — to merge repeats and work out
# what changed and when. A lone fact passes through as it is.

SIM = 0.72               # cosine at which two facts about the same entity are "the same topic" (pilot-tuned)
CLUSTERS_PER_CALL = 12


def vectors(conn, rf_ids):
    """8B vectors for raw facts (OpenRouter, cached in RawFactVectors as float16 blobs)."""
    import numpy as np
    import httpx
    import os
    from search import OPENROUTER_ROUTE
    conn.execute("CREATE TABLE IF NOT EXISTS RawFactVectors (rf_id TEXT PRIMARY KEY, v BLOB)")
    have = {r: np.frombuffer(v, dtype=np.float16).astype(np.float32) for r, v in conn.execute(
        "SELECT rf_id, v FROM RawFactVectors WHERE rf_id IN (SELECT value FROM json_each(?))", (json.dumps(rf_ids),))}
    todo = [r for r in rf_ids if r not in have]
    if todo:
        text = dict(conn.execute("SELECT rf_id, statement FROM RawFacts WHERE rf_id IN (SELECT value FROM json_each(?))",
                                 (json.dumps(todo),)))
        key = os.environ.get("OPENROUTER_API_KEY") or dotenv_values(REPO_ROOT / ".env").get("OPENROUTER_API_KEY")
        http = httpx.Client(base_url="https://openrouter.ai/api/v1", headers={"Authorization": f"Bearer {key}"},
                            timeout=120)

        def emb(batch):
            for attempt in range(5):
                try:
                    r = http.post("/embeddings", json={"model": "qwen/qwen3-embedding-8b", "provider": OPENROUTER_ROUTE,
                                                       "input": [text[i] for i in batch]})
                    r.raise_for_status()
                    return batch, [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
                except Exception:
                    import time
                    time.sleep(2 ** attempt)
            return batch, None
        with cf.ThreadPoolExecutor(4) as ex:
            for batch, vs in ex.map(emb, [todo[i:i + 128] for i in range(0, len(todo), 128)]):
                if not vs:
                    continue
                rows = []
                for rid, v in zip(batch, vs):
                    v = np.asarray(v, dtype=np.float32)
                    v /= np.linalg.norm(v) or 1
                    have[rid] = v
                    rows.append((rid, v.astype(np.float16).tobytes()))
                conn.executemany("INSERT OR REPLACE INTO RawFactVectors VALUES (?,?)", rows)
        conn.commit()
    return have


def clusters(facts, vec):
    """Greedy clustering in date order: a fact joins the most similar cluster (centroid cosine >= SIM)."""
    import numpy as np
    out, cents = [], []
    for f in facts:
        v = vec.get(f[0])
        if v is None:
            out.append([f]); cents.append(None)
            continue
        best, bs = None, SIM
        for i, c in enumerate(cents):
            if c is not None:
                sim = float(c @ v) / (float(np.linalg.norm(c)) or 1)
                if sim >= bs:
                    best, bs = i, sim
        if best is None:
            out.append([f]); cents.append(v.copy())
        else:
            out[best].append(f); cents[best] += v
    return out


P2 = """Today is {today}. Each group below holds facts Sarthak's memory gathered about ONE topic of {name} ({kind}),
oldest first, with the date and the conversation id. For each group, write the consolidated fact(s):
- merge repeats into one; "n" = how many items support it
- when something changed over time, write one fact per state: earlier ones get valid_to and status "past", the
  latest is "current" (e.g. planning to come to Bangalore → studying at Ramaiah in Bangalore)
- a plan whose date has passed with no sign it happened: status "past"
- valid_from / valid_to as YYYY-MM-DD (or YYYY-MM); valid_to null if still true as far as known
- status "disputed" only if items truly contradict and it's unclear which holds
- sources: conversation ids it rests on (at most 5); keep specifics; one sentence each, naming {name}
Return JSON: {{"groups": [{{"g": <group number>, "facts": [{{"category": "...", "statement": "...",
"valid_from": "...", "valid_to": null, "status": "current", "n": 1, "sources": ["s_..."]}}]}}]}}

{groups}"""


def pass2(conn, cl, only=None, limit=0):
    conn.executescript(SCHEMA)
    today = dt.datetime.now(IST).strftime("%Y-%m-%d")
    ents = conn.execute("""
        SELECT e.entity_id, e.kind, e.name, COUNT(r.rf_id) n FROM Entities e JOIN RawFacts r USING(entity_id)
        LEFT JOIN FactEntityState st USING(entity_id)
        WHERE st.raw_count IS NULL OR st.raw_count != (SELECT COUNT(*) FROM RawFacts r2 WHERE r2.entity_id = e.entity_id)
        GROUP BY e.entity_id ORDER BY n DESC""").fetchall()
    if only:
        ents = [e for e in ents if norm(only) in norm(e[2])]
    if limit:
        ents = ents[:limit]
    print(f"pass 2: {len(ents)} entities", flush=True)
    # embed every raw fact still without a vector in one batched pass (per-entity requests were mostly overhead)
    vectors(conn, [r[0] for r in conn.execute(
        "SELECT rf_id FROM RawFacts WHERE rf_id NOT IN (SELECT rf_id FROM RawFactVectors)")])
    jobs, singles, tin, tout = [], [], 0, 0
    for eid, kind, name, n in ents:
        facts = conn.execute("""SELECT rf_id, said_on, session_id, category, statement FROM RawFacts
                                WHERE entity_id = ? ORDER BY said_on""", (eid,)).fetchall()
        vec = vectors(conn, [f[0] for f in facts])
        groups = clusters(facts, vec)
        multi = [g for g in groups if len(g) > 1]
        singles += [(eid, g[0]) for g in groups if len(g) == 1]
        for i in range(0, len(multi), CLUSTERS_PER_CALL):
            jobs.append((eid, kind, name, multi[i:i + CLUSTERS_PER_CALL]))
        conn.execute("DELETE FROM Facts WHERE entity_id = ?", (eid,))
    print(f"   {len(singles):,} lone facts pass through; {sum(len(j[3]) for j in jobs):,} topics in {len(jobs)} calls",
          flush=True)

    def put(eid, f, srcs, cat):
        st = str(f.get("statement") or "").strip()
        if not st:
            return
        fid = hashlib.sha1(f"{eid}|{st}".encode()).hexdigest()[:16]
        conn.execute("INSERT OR REPLACE INTO Facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (fid, eid, f.get("category") if f.get("category") in CATEGORIES else cat, st, f.get("valid_from"),
                      f.get("valid_to"), f.get("status") if f.get("status") in ("current", "past", "disputed") else
                      "current", int(f.get("n") or 1), json.dumps(srcs), MODEL, P2_VERSION, today))

    for eid, (rf, day, sid, cat, st) in singles:
        put(eid, {"statement": st, "valid_from": day, "status": "current", "n": 1, "category": cat}, [sid], cat)

    def run(job):
        eid, kind, name, groups = job
        text = "\n\n".join(f"### Group {gi}\n" + "\n".join(f"[{d}] ({sid}) [{c}] {st}" for _, d, sid, c, st in g)
                           for gi, g in enumerate(groups))
        out, usage = ask(cl, P2.format(today=today, name=name, kind=kind, groups=text), thinking=True, max_tokens=32000)
        return job, out, usage

    with cf.ThreadPoolExecutor(16) as ex:
        for (eid, kind, name, groups), out, usage in ex.map(run, jobs):
            if usage:
                tin, tout = tin + usage.prompt_tokens, tout + usage.completion_tokens
            done = set()
            for g in out.get("groups", []) if "error" not in out else []:
                gi = g.get("g")
                if not isinstance(gi, int) or not 0 <= gi < len(groups):
                    continue
                sids = {f[2] for f in groups[gi]}
                for f in g.get("facts") or []:
                    put(eid, f, [s for s in f.get("sources") or [] if s in sids][:5] or [groups[gi][-1][2]],
                        groups[gi][-1][3])
                done.add(gi)
            for gi, grp in enumerate(groups):     # model failed on a group: keep its latest item as is
                if gi not in done:
                    rf, day, sid, cat, st = grp[-1]
                    put(eid, {"statement": st, "valid_from": day, "status": "current", "n": len(grp)}, [sid], cat)
    for eid, *_ in ents:
        conn.execute("INSERT OR REPLACE INTO FactEntityState VALUES (?,?,?)",
                     (eid, conn.execute("SELECT COUNT(*) FROM RawFacts WHERE entity_id = ?", (eid,)).fetchone()[0], today))
    conn.commit()
    print(f"pass 2 done: {conn.execute('SELECT COUNT(*) FROM Facts').fetchone()[0]:,} facts | "
          f"tokens in {tin:,} out {tout:,}")


# ─── pass 3: support and reconcile ─────────────────────────────────────────────
# The model's own "n" can't be trusted (a lone misreading inside a big topic came out with n=36). So support is
# measured: a consolidated fact's sources are the conversations whose raw facts say something close to it, and
# its confidence is how many such conversations there are. Then near-duplicate or contradicting consolidated
# facts of one entity are reconciled by the model, which sees each one's support.
SUPPORT_SIM, DUP_SIM = 0.78, 0.86

P3 = """Today is {today}. These are facts Sarthak's memory holds about {name}; each group has facts that look like
duplicates or contradictions of each other. Each fact shows how many separate conversations support it and a few
of the supporting statements. For each group return the facts that should remain:
- merge duplicates into one (keep the specifics; earliest valid_from, latest valid_to)
- when facts contradict, the one with clearly more support wins; drop a weakly supported claim (1 conversation)
  that the better supported ones contradict; if support is similar, keep both with status "disputed"
- keep changes over time as separate past/current facts
Return JSON: {{"groups": [{{"g": <n>, "keep": [{{"from": [<fact numbers merged>], "category": "...",
"statement": "...", "valid_from": "...", "valid_to": null, "status": "current|past|disputed"}}]}}]}}

{groups}"""


def fact_vectors(conn, rows):
    """Embed consolidated facts (same model as raw facts); returns {fact_id: vec}."""
    import numpy as np
    conn.execute("CREATE TABLE IF NOT EXISTS FactVectors (fact_id TEXT PRIMARY KEY, v BLOB)")
    have = {f: np.frombuffer(v, dtype=np.float16).astype(np.float32) for f, v in conn.execute(
        "SELECT fact_id, v FROM FactVectors WHERE fact_id IN (SELECT value FROM json_each(?))",
        (json.dumps([r[0] for r in rows]),))}
    todo = [r for r in rows if r[0] not in have]
    if todo:
        vec = _embed_texts([(r[0], r[1]) for r in todo])
        conn.executemany("INSERT OR REPLACE INTO FactVectors VALUES (?,?)",
                         [(k, v.astype(np.float16).tobytes()) for k, v in vec.items()])
        conn.commit()
        have.update(vec)
    return have


def _embed_texts(items):
    import numpy as np
    import httpx
    import os
    import time
    from search import OPENROUTER_ROUTE
    key = os.environ.get("OPENROUTER_API_KEY") or dotenv_values(REPO_ROOT / ".env").get("OPENROUTER_API_KEY")
    http = httpx.Client(base_url="https://openrouter.ai/api/v1", headers={"Authorization": f"Bearer {key}"}, timeout=120)
    out = {}

    def emb(batch):
        for attempt in range(5):
            try:
                r = http.post("/embeddings", json={"model": "qwen/qwen3-embedding-8b", "provider": OPENROUTER_ROUTE,
                                                   "input": [t for _, t in batch]})
                r.raise_for_status()
                return batch, [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
            except Exception:
                time.sleep(2 ** attempt)
        return batch, None
    with cf.ThreadPoolExecutor(4) as ex:
        for batch, vs in ex.map(emb, [items[i:i + 128] for i in range(0, len(items), 128)]):
            for (k, _), v in zip(batch, vs or []):
                v = np.asarray(v, dtype=np.float32)
                out[k] = v / (np.linalg.norm(v) or 1)
    return out


def support(conn, eid):
    """Measured sources and confidence for an entity's facts; returns rows with their vectors."""
    import numpy as np
    facts = conn.execute("SELECT fact_id, statement FROM Facts WHERE entity_id = ?", (eid,)).fetchall()
    raws = conn.execute("SELECT rf_id, session_id, said_on, statement FROM RawFacts WHERE entity_id = ?", (eid,)).fetchall()
    if not facts or not raws:
        return {}, {}, raws
    fv = fact_vectors(conn, facts)
    rv = vectors(conn, [r[0] for r in raws])
    ids = [r[0] for r in raws if r[0] in rv]
    R = np.stack([rv[i] for i in ids]) if ids else np.zeros((0, 4096), dtype=np.float32)
    meta = {r[0]: r for r in raws}
    sup = {}
    for fid, st in facts:
        if fid not in fv or not len(ids):
            continue
        sims = R @ fv[fid]
        hit = [ids[i] for i in np.argsort(-sims) if sims[i] >= SUPPORT_SIM]
        sess = []
        for h in hit:
            if meta[h][1] not in sess:
                sess.append(meta[h][1])
        sup[fid] = (len(sess), sess[:5], [meta[h][3] for h in hit[:3]])
        conn.execute("UPDATE Facts SET confidence = ?, sources = ? WHERE fact_id = ?",
                     (max(1, len(sess)), json.dumps(sess[:5] or json.loads(
                         conn.execute("SELECT sources FROM Facts WHERE fact_id = ?", (fid,)).fetchone()[0])), fid))
    conn.commit()
    return fv, sup, raws


def reconcile(conn, cl, only=None):
    import numpy as np
    today = dt.datetime.now(IST).strftime("%Y-%m-%d")
    ents = conn.execute("""SELECT e.entity_id, e.name, COUNT(*) FROM Entities e JOIN Facts f USING(entity_id)
                           GROUP BY e.entity_id ORDER BY 3 DESC""").fetchall()
    if only:
        ents = [e for e in ents if norm(only) in norm(e[1])]
    jobs, tin, tout = [], 0, 0
    for eid, name, n in ents:
        fv, sup, _ = support(conn, eid)
        if n < 2:
            continue
        rows = conn.execute("SELECT fact_id, category, statement, valid_from, valid_to, status FROM Facts WHERE entity_id = ?",
                            (eid,)).fetchall()
        rows = [r for r in rows if r[0] in fv]
        # groups: connected components of facts with cosine >= DUP_SIM
        V = np.stack([fv[r[0]] for r in rows]) if rows else None
        seen, groups = set(), []
        for i in range(len(rows)):
            if i in seen:
                continue
            comp, stack = [], [i]
            while stack:
                j = stack.pop()
                if j in seen:
                    continue
                seen.add(j)
                comp.append(j)
                stack += [k for k in np.nonzero(V @ V[j] >= DUP_SIM)[0].tolist() if k not in seen]
            if len(comp) > 1:
                groups.append([rows[k] for k in comp])
        for i in range(0, len(groups), 10):
            jobs.append((eid, name, groups[i:i + 10], sup))
    print(f"reconcile: {sum(len(j[2]) for j in jobs)} groups of look-alike facts in {len(jobs)} calls", flush=True)

    def run(job):
        eid, name, groups, sup = job
        text = []
        for gi, g in enumerate(groups):
            lines = []
            for fi, r in enumerate(g):
                n, _, ex = sup.get(r[0], (1, [], []))
                lines.append(f"  {fi}. [{r[5]}; {r[3] or '?'}–{r[4] or 'now'}; {n} conversation(s)] {r[2]}"
                             + "".join(f"\n       · {e[:160]}" for e in ex))
            text.append(f"### Group {gi}\n" + "\n".join(lines))
        out, usage = ask(cl, P3.format(today=today, name=name, groups="\n\n".join(text)), thinking=True,
                         max_tokens=32000)
        return job, out, usage

    with cf.ThreadPoolExecutor(16) as ex:
        for (eid, name, groups, sup), out, usage in ex.map(run, jobs):
            if usage:
                tin, tout = tin + usage.prompt_tokens, tout + usage.completion_tokens
            if "error" in out:
                continue
            for g in out.get("groups", []):
                gi = g.get("g")
                if not isinstance(gi, int) or not 0 <= gi < len(groups):
                    continue
                grp = groups[gi]
                keep = g.get("keep") or []
                if not keep:
                    continue
                conn.executemany("DELETE FROM Facts WHERE fact_id = ?", [(r[0],) for r in grp])
                for k in keep:
                    st = str(k.get("statement") or "").strip()
                    merged = [grp[i] for i in k.get("from") or [] if isinstance(i, int) and 0 <= i < len(grp)]
                    srcs = []
                    for r in merged:
                        for s_ in sup.get(r[0], (0, [], []))[1]:
                            if s_ not in srcs:
                                srcs.append(s_)
                    if not st:
                        continue
                    fid = hashlib.sha1(f"{eid}|{st}".encode()).hexdigest()[:16]
                    conn.execute("INSERT OR REPLACE INTO Facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                 (fid, eid, k.get("category") if k.get("category") in CATEGORIES else merged[0][1]
                                  if merged else "event", st, k.get("valid_from"), k.get("valid_to"),
                                  k.get("status") if k.get("status") in ("current", "past", "disputed") else "current",
                                  max([sup.get(r[0], (1,))[0] for r in merged] or [1]), json.dumps(srcs[:5]),
                                  MODEL, P2_VERSION, today))
            conn.commit()
    for eid, *_ in ents:                 # the merged facts get measured support too
        support(conn, eid)
    print(f"reconcile done: {conn.execute('SELECT COUNT(*) FROM Facts').fetchone()[0]:,} facts | "
          f"tokens in {tin:,} out {tout:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pass1", "link", "pass2", "reconcile", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--entity", help="pass 2: only entities whose name contains this")
    ap.add_argument("--since", help="pass 1: only conversations from this date (YYYY-MM-DD)")
    a = ap.parse_args()
    conn, cl = connect_index(), client()
    if a.step in ("pass1", "all"):
        pass1(conn, cl, a.limit, a.sample, a.since)
    if a.step in ("link", "all"):
        link(conn)
    if a.step in ("pass2", "all"):
        pass2(conn, cl, a.entity, a.limit)
    if a.step in ("reconcile", "all"):
        reconcile(conn, cl, a.entity)
