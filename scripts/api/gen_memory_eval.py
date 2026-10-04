"""Memory eval: ~40 questions with checked answers, shaped like the questions Sarthak asks in a normal week.

Answers never come from the assistant being evaluated. They are drafted from the dated facts in the session
summaries (deepseek-v4-pro, thinking), then each expected fact is checked against the raw messages of the sessions
it cites, plus anything newer about the same subject (deepseek-flash, thinking): unsupported or outdated facts are
dropped, questions left without facts are dropped. "When did I last talk to X" answers come from SQL.
Then a human reads the list (docs/PLAN-memory.md, Phase 0).

Output: processed_data/eval/agent/memory_v1.json (agent_eval.py --set memory)
Usage: python3 scripts/api/gen_memory_eval.py [--as-of 2026-10-04]
"""
import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "index"))
from common import connect_index  # noqa: E402

OUT = REPO / "processed_data" / "eval" / "agent" / "memory_v1.json"

# (pack name, how to pick facts, kinds and counts to write)
SELF_TOPICS = ["college", "admission", "ramaiah", "comedk", "laptop", "macbook", "bangalore", "bengaluru", "ranchi",
               "bokaro", "job", "offer", "intern", "hostel", " pg", "flat", "moved", "girlfriend", "salary", "contract",
               "fellowship", "health", "doctor", "money", "loan", "project"]
PEOPLE = ["Devanshi", "Mihir", "Jitesh", "Manav", "Nilesh", "Aparajita", "Apurva", "Harsh", "datavorous", "Nivedha"]
ENTITIES = ["FOSS", "CBSE", "OSM", "parliament", "Juggernaut", "Cashify", "Ramaiah", "SJU", "Joseph", "hackathon",
            "3D print", "Malpani", "Hindu", "NTA"]

PACKS = [
    ("me", "Sarthak himself: where he studies, lives, works, what he owns and uses",
     {"current": 5, "history": 3, "change": 3}),
    ("people", "the people closest to Sarthak: where they study/work/live, what they're doing, who they are to him",
     {"others": 7, "relationship": 3}),
    ("devanshi", "Sarthak and Devanshi: what's been happening between them recently",
     {"lately": 3}),
    ("entities", "places, organisations, projects and investigations in Sarthak's life, and who is involved",
     {"entity": 4, "who_to_ask": 3, "sent_me": 3}),
    ("followthrough", "what Sarthak said he would do about his health, money and projects, and whether he did",
     {"followthrough": 3}),
]
KINDS = {
    "current": "his current state as of {as_of} (e.g. 'Where do I study now?'); the LATEST facts win",
    "history": "something that was true before and changed (e.g. 'Where did I live before X?')",
    "change": "how his plans for college/work/life changed over the months (e.g. 'How did my college plans change?')",
    "others": "a fact about another person (e.g. 'Where does Jitesh live?', 'What is Mihir studying?')",
    "relationship": "who someone is to Sarthak, or who his closest people are",
    "lately": "what has been going on recently, things to remember or follow up on",
    "entity": "what Sarthak knows about a place/project/organisation and who is involved",
    "who_to_ask": "who in his network knows about or could help with a topic",
    "sent_me": "something specific someone sent or told him about a topic ('What was that thing X sent me about Y?')",
    "followthrough": "a promise about health/money/projects and whether it happened",
}

GEN = """You are writing an evaluation set for Sarthak's personal memory assistant. Today is {as_of}.
Below are dated facts from summaries of his chats (each with a session id). Write questions of these kinds,
about {about}:
{kinds}
Rules:
- Phrase questions the way Sarthak would ask ("Where do I…", "What's going on with…"), in English.
- Every expected fact must be clearly stated in at least one listed fact; cite the session ids it comes from.
- Prefer things stated more than once or by the person themselves; avoid jokes, sarcasm and banter.
- For anything about the present, use the most recent facts; if later facts contradict earlier ones, the later win.
- 1-3 expected facts per question, each short and checkable.
Return JSON: {{"questions": [{{"kind": "...", "question": "...", "facts": ["..."], "sessions": ["s_..."]}}]}}

FACTS
{facts}"""

CHECK = """Check an evaluation question about Sarthak's chats against the actual messages. Today is {as_of}.
QUESTION: {question}
EXPECTED FACTS: {facts}

MESSAGES FROM THE CITED CONVERSATIONS
{evidence}

NEWER FACTS ABOUT THE SAME SUBJECT (from later summaries; the latest wins for anything about the present)
{newer}

For each expected fact: is it supported by the messages (not a joke or a misreading), and still true as of today?
Return JSON: {{"facts": [{{"fact": "...", "supported": true/false, "current": true/false, "fixed": "<corrected
wording if it needs one, else empty>"}}], "question_ok": true/false, "note": "one line"}}"""


def fact_rows(conn, pack):
    """Dated summary facts relevant to a pack: [(day, session_id, fact)], newest first, capped."""
    rows = conn.execute("""
        SELECT date(s.start_ts, 'unixepoch', '+330 minutes'), x.session_id, j.value
        FROM SessionSummaries x JOIN Sessions s USING(session_id), json_each(json_extract(x.data, '$.facts')) j
        WHERE x.model = 'deepseek-flash' AND s.start_ts > strftime('%s', '2025-06-01')
        ORDER BY s.start_ts DESC""").fetchall()
    low = lambda t: t.lower()  # noqa: E731
    if pack == "me":
        keep = [r for r in rows if r[2].startswith("Sarthak") and any(k in low(r[2]) for k in SELF_TOPICS)]
    elif pack == "people":
        keep = [r for r in rows if any(p.lower() in low(r[2]) for p in PEOPLE)]
    elif pack == "devanshi":
        keep = [r for r in rows if "devanshi" in low(r[2]) and r[0] >= "2026-08-01"]
    elif pack == "entities":
        keep = [r for r in rows if any(e.lower() in low(r[2]) for e in ENTITIES)]
    else:
        keep = [r for r in rows if r[2].startswith("Sarthak") and any(
            k in low(r[2]) for k in ("will ", "plans to", "promised", "going to")) and any(
            k in low(r[2]) for k in ("doctor", "health", "gym", "money", "pay", "project", "build", "ship", "website"))]
    return keep[:700]


def session_messages(conn, sid, cap=7000):
    text = "\n".join(t for (t,) in conn.execute(
        "SELECT text FROM Chunks WHERE session_id = ? ORDER BY start_ts, chunk_id", (sid,)))
    return text if len(text) <= cap else text[:cap // 2] + "\n[…]\n" + text[-cap // 2:]


def last_contact(conn, as_of):
    """'When did I last talk to X, and what about?': date from the messages, topic from that session's summary."""
    out = []
    for name in ("Mihir Karbelkar", "Jitesh Kumar Sahoo", "Nilesh Yadav", "Aparajita"):
        last = conn.execute("""
            SELECT f.day, f.ts, f.thread_id FROM MessageFacts f JOIN Accounts a ON a.account_id = f.account_id
            WHERE a.name LIKE ? AND a.is_me = 0 AND f.day <= ? ORDER BY f.ts DESC LIMIT 1""",
                            (f"%{name.split()[0]}%", as_of)).fetchone()
        r = None
        if last:
            sess = conn.execute("""SELECT s.session_id, json_extract(x.data, '$.summary') FROM Sessions s
                LEFT JOIN SessionSummaries x ON x.session_id = s.session_id AND x.model = 'deepseek-flash'
                WHERE json_extract(s.thread_ids, '$[0]') = ? AND s.start_ts <= ? ORDER BY s.start_ts DESC LIMIT 1""",
                                (last[2], last[1])).fetchone()
            r = (last[0], *(sess or (None, None)))
        if r:
            out.append({"kind": "last_contact", "question": f"When did I last talk to {name.split()[0]}, and what about?",
                        "facts": [f"last messages on {r[0]}", f"about: {(r[2] or '')[:160]}"], "sessions": [r[1]]})
    return out


def main(a):
    key = dotenv_values(REPO / ".env")["DEEPSEEK_API_KEY"]
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=600)
    conn = connect_index()

    facts_of = {name: fact_rows(conn, name) for name, _, _ in PACKS}     # reads stay on this thread

    def draft(pack):
        name, about, kinds = pack
        facts = facts_of[name]
        lines = "\n".join(f"[{d}] ({sid}) {f}" for d, sid, f in facts)
        kind_lines = "\n".join(f"- {n} x {k}: {KINDS[k].format(as_of=a.as_of)}" for k, n in kinds.items())
        r = client.chat.completions.create(
            model="deepseek-v4-pro", response_format={"type": "json_object"}, max_tokens=16000,
            messages=[{"role": "user", "content": GEN.format(as_of=a.as_of, about=about, kinds=kind_lines,
                                                             facts=lines)}])
        qs = json.loads(r.choices[0].message.content).get("questions", [])
        print(f"  {name}: {len(facts)} facts -> {len(qs)} questions", flush=True)
        return [dict(q, pack=name) for q in qs]

    drafts_file = OUT.with_name("memory_v1.drafts.json")
    if drafts_file.exists() and not a.redraft:        # drafting is the costly part: reuse it
        drafts = json.loads(drafts_file.read_text())
    else:
        with cf.ThreadPoolExecutor(len(PACKS)) as ex:
            drafts = [q for qs in ex.map(draft, PACKS) for q in qs]
        drafts += last_contact(conn, a.as_of)
        drafts_file.parent.mkdir(parents=True, exist_ok=True)
        drafts_file.write_text(json.dumps(drafts, ensure_ascii=False, indent=1))

    def context(q):
        sids = [s for s in q.get("sessions", []) if isinstance(s, str)][:4]
        evidence = "\n\n".join(f"=== {s}\n{session_messages(conn, s)}" for s in sids)
        words = [w for w in q["question"].replace("?", "").split() if len(w) > 4][:4]
        newer = conn.execute(f"""
            SELECT date(s.start_ts, 'unixepoch', '+330 minutes') || ' ' || j.value
            FROM SessionSummaries x JOIN Sessions s USING(session_id), json_each(json_extract(x.data, '$.facts')) j
            WHERE s.start_ts > strftime('%s', '2026-07-01') AND ({' OR '.join(['j.value LIKE ?'] * len(words)) or '0'})
            ORDER BY s.start_ts DESC LIMIT 40""", [f"%{w}%" for w in words]).fetchall()
        return evidence, newer

    ctx = [None if q["kind"] == "last_contact" else context(q) for q in drafts]

    def check(item):
        q, c = item
        if q["kind"] == "last_contact":      # from SQL: the date is exact; keep the topic loose
            return dict(q, facts=[q["facts"][0], "names a topic from that conversation"], note="from SQL")
        evidence, newer = c
        v = None
        for _ in range(3):           # thinking can use up the reply budget and leave it empty: ask again
            r = client.chat.completions.create(
                model="deepseek-flash", response_format={"type": "json_object"}, max_tokens=16000,
                messages=[{"role": "user", "content": CHECK.format(
                    as_of=a.as_of, question=q["question"], facts=json.dumps(q["facts"], ensure_ascii=False),
                    evidence=evidence or "(none)", newer="\n".join(n for (n,) in newer) or "(none)")}])
            try:
                v = json.loads(r.choices[0].message.content or "")
                break
            except json.JSONDecodeError:
                continue
        if v is None:
            return dict(q, facts=[], note="check failed", question_ok=False)
        facts = [(f.get("fixed") or f.get("fact")) for f in v.get("facts", []) if f.get("supported") and f.get("current")]
        return dict(q, facts=facts, note=v.get("note", ""), question_ok=v.get("question_ok", True))

    with cf.ThreadPoolExecutor(16) as ex:
        checked = list(ex.map(check, zip(drafts, ctx)))
    keep = [q for q in checked if q["facts"] and q.get("question_ok", True)]
    cases = [{"id": 200 + i, "type": q["kind"], "question": q["question"], "facts": q["facts"], "abstain": False,
              "sessions": q.get("sessions", []), "pack": q.get("pack"), "note": q.get("note", ""), "as_of": a.as_of}
             for i, q in enumerate(keep)]
    cases.append({"id": 200 + len(cases), "type": "abstain", "question": "What did Devanshi and I do on our trip to Paris?",
                  "facts": [], "abstain": True, "as_of": a.as_of})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    print(f"{len(drafts)} drafted, {len(keep)} kept after checking -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="2026-10-04")
    ap.add_argument("--redraft", action="store_true", help="draft the questions again instead of reusing them")
    main(ap.parse_args())
