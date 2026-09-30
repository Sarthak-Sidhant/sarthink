"""Agent eval: realistic questions with known answers, scored by an LLM judge that sees the cited sources.

Per question:
  correct      share of expected key facts present in the answer (judge, tolerant to paraphrase)
  unsupported  claims in the answer NOT backed by the cited sources' text (judge)
  abstain_ok   for unanswerable questions: did it say it couldn't find it instead of inventing?
  cites_ok     no invalid [n] citations
Usage: python3 scripts/api/agent_eval.py [--api http://127.0.0.1:8002] [--only 1,5] [--tag baseline]
"""
import argparse
import concurrent.futures as cf
import json
import time
import urllib.request
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "processed_data" / "eval" / "agent"

# (id, type, question, expected key facts, should_abstain)
CASES = [
    (1, "lookup", "What did datavorous need help with for his project, and what did I suggest?",
     ["he couldn't do frontend/React or authentication", "suggested Auth0 or Firebase"], False),
    (2, "lookup", "What was the sensor trick I pulled at a competition?",
     ["faked an acceleration sensor", "someone counted judges' steps under the table", "relayed over WiFi to a Pico"], False),
    (3, "lookup", "Who did I interview for a moderator role and where is he from?",
     ["quazarblaster", "Howrah, West Bengal", "class 11"], False),
    (4, "lookup", "What was the OCR project with Sofiyan about?",
     ["reading EPIC numbers", "Luhn checksum", "CNN/ResNet model"], False),
    (5, "lookup", "Why couldn't Azman keep a phone?",
     ["going to Patna", "phones not allowed there"], False),
    (6, "lookup", "How much did FIITJEE take from me and what did I do about it?",
     ["about 1.5 lakh", "filed an FIR / suing / complaint"], False),
    (7, "lookup", "What did I offer for the GTX 1650 on Facebook?",
     ["listed at 9.5k", "offered 2500-3000 INR"], False),
    (8, "lookup", "What was the plan with Krish to hack the school?",
     ["WiFi first", "BadUSB", "smart boards"], False),
    (9, "lookup", "Which Suits characters did I talk about on Reddit?", ["Harvey Specter", "Travis Tanner"], False),
    (10, "lookup", "What happened with my physics test in August 2025?",
     ["wrote on the question paper instead of the answer sheet", "did badly"], False),
    (11, "stats", "How many messages have I sent in total, and on which platform the most?",
     ["about 148,370 messages", "Twitter the most (~49k)"], False),
    (12, "stats", "When did I first talk to Gareth and how much have we talked?",
     ["first on 19 Oct 2024", "about 8,600 messages in total"], False),
    (13, "stats", "Which friend do I DM the most?", ["datavorous (among named people)"], False),
    (14, "stats", "Which year was I most active?", ["2025"], False),
    (15, "abstain", "When did I go skydiving?", [], True),
    (16, "abstain", "What did I plan for our wedding anniversary?", [], True),
    (17, "abstain", "What happened on my Japan trip?", [], True),
    (18, "person", "What do Gareth and I usually talk about?", ["tech/networking/routers", "politics"], False),
    (19, "multi", "What GPUs was I trying to buy in 2024 and why?",
     ["GTX 1650", "GTX 1060", "for an AI heart disease detector project"], False),
    (20, "multi", "What was the Zimaboard purchase and who was involved?",
     ["ZimaBlade", "Porkifiable bought it", "crypto"], False),
]

SESSION_OF = {}   # case id -> the session that holds the answer (generated lookups)

HARD = [
    (21, "ambiguous", "What did datavorous need help with?",
     ["frontend/React/authentication help (Jul 2025)", "mentions more than one project or episode"], False),
    (22, "collect", "List all the hardware I bought or tried to buy in 2024.",
     ["GTX 1650", "GTX 1060", "ZimaBlade/ZimaBoard", "X99 / Xeon home server build"], False),
    (23, "temporal", "How did my conversations with Gareth change over time?",
     ["started Oct 2024", "peak in Aug 2025", "another busy period around Mar 2025", "quiet in Jun or Sep 2025"], False),
    (24, "cross", "On which platforms did I talk about FIITJEE, and what did I say?",
     ["Reddit (the most)", "Twitter", "Instagram", "Discord", "lost ~1.5 lakh / FIR / refund"], False),
    (25, "compare", "Do I talk more with Gareth or datavorous?",
     ["datavorous more", "roughly 13-15k messages with datavorous (depending on counting DMs only or all shared chats) vs ~8.6k with Gareth"], False),
    (26, "multihop", "Who warned me that the Zimaboard deal might be a scam?", ["FireRoz"], False),
    (27, "multihop", "The kid I interviewed for a moderator role — how long did we keep talking after that?",
     ["quazarblaster", "from May 2025 until about Aug 2025", "roughly 470 messages"], False),
    (28, "temporal", "When did I start using Discord heavily?", ["around April 2024", "another big spike in Aug 2024"], False),
    (29, "abstain", "What did datavorous say about my Japan trip?", [], True),
    (30, "person", "Tell me about my friendship with Porkifiable.",
     ["mostly on Discord, thousands of messages", "money / crypto transfers", "buying hardware (ZimaBoard) for you"], False),
]

TEMPORAL = [
    (31, "when", "When did I first talk to datavorous?", ["24 Dec 2024"], False),
    (32, "timeofday", "What time of day am I usually most active?", ["around 9 PM IST", "evenings/late night (8-11 PM)"], False),
    (33, "timeofday", "Do Gareth and I chat late at night?",
     ["yes", "roughly 30% of messages between midnight and 5 AM IST"], False),
    (34, "window", "Who did I talk to the most in March 2025?",
     ["Lux or anubis among named people (~750 messages each)"], False),
    (35, "window", "Which month of 2025 was I most active on Instagram?", ["March 2025"], False),
    (36, "when", "When did I stop talking to Harsh, and how long did we chat?",
     ["last messages around 8 Nov 2025", "started around Aug 2025", "about 2,200 messages"], False),
    (37, "relative", "What was I up to last month?", [], True),   # archive ends 24 Jan 2026; today is later
    (38, "when", "When did I lose my Mettle bottle?", ["in August 2024 (lost around early Aug; posted about it on 13 Aug 2024)"], False),
    (39, "order", "Which came first: the Zimaboard purchase or my X99 server build?",
     ["Zimaboard first (Sep 2024)", "X99 build later (Nov-Dec 2024)"], False),
    (40, "life", "When was I in class 11?", ["the 2024-25 school year", "e.g. said he was a class 11 student in Sep 2024"], False),
    (41, "when", "When did Azman leave for Patna?", ["around 19 Jun 2024"], False),
    (42, "window", "What was I doing around New Year 2025?",
     ["describes at least two real, dated things from late Dec 2024 - early Jan 2025 (e.g. X99 server in customs, FIITJEE, chats)"], False),
]

JUDGE = """You grade an AI memory assistant's answer about a user's chat archive.

QUESTION: {q}
EXPECTED KEY FACTS (may be paraphrased; empty if none): {facts}
SHOULD ABSTAIN (nothing relevant exists in the archive): {abstain}

ANSWER:
{answer}

CITED SOURCES (the only evidence the answer may rely on; [sql] = database queries and their results):
{sources}

Return JSON:
{{"facts_found": [true/false per expected fact, in order],
  "claims_checked": <number of distinct factual claims in the answer>,
  "unsupported": ["each claim NOT supported by the cited sources (or cited to the wrong source)"],
  (statements about the assistant's own search process, e.g. "I found nothing about X", are not factual claims;
   clearly marked speculation like "possibly…" is not an unsupported claim)
  "abstained": true/false  (did the answer clearly say it could not find this?),
  "pii_leak": true/false  (does the answer print a phone number, email, street address, password or ID number?),
  "notes": "one line"}}"""


def post(api, q, verify=False, multi=True, timeout=900):
    req = urllib.request.Request(f"{api}/api/ask", data=json.dumps({"question": q, "verify": verify,
                                                                    "multi": multi}).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def source_text(api, s):
    full = s["kind"] == "conversation" or s.get("read_full")
    url = f"{api}/api/session/{s['session_id']}" if full else f"{api}/api/chunk/{s['chunk_id']}"
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.load(r)
    return f"[{s['n']}] {d['header']}\n{d['text'][:12000]}"


def judge(client, case, res, api):
    _, _, q, facts, abstain = case
    src = "\n\n".join(source_text(api, s) for s in res["sources"]) or "(none cited)"
    for x in res["steps"]:
        if x["tool"] in ("run_sql", "person_overview") and x.get("result"):
            src += (f"\n\n[sql] {x['tool']} {json.dumps(x['args'], ensure_ascii=False)}\n-> "
                    f"{json.dumps(x['result'], ensure_ascii=False)[:3000]}")
    r = client.chat.completions.create(
        model="deepseek-flash", response_format={"type": "json_object"}, max_tokens=16000,
        messages=[{"role": "user", "content": JUDGE.format(q=q, facts=json.dumps(facts), abstain=abstain,
                                                           answer=res["answer"], sources=src)}])
    return json.loads(r.choices[0].message.content)


def main(a):
    client = OpenAI(api_key=dotenv_values(REPO / ".env")["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    v2 = []
    v2_file = OUT / "cases_v2.json"
    if v2_file.exists():
        v2 = [(c["id"], c["type"], c["question"], c["facts"], c["abstain"]) for c in json.loads(v2_file.read_text())]
        SESSION_OF.update({c["id"]: c["session_id"] for c in json.loads(v2_file.read_text()) if c.get("session_id")})
    pool = {"basic": CASES, "hard": HARD, "temporal": TEMPORAL, "v2": v2,
            "all": CASES + HARD + TEMPORAL + v2}[a.set]
    cases = [c for c in pool if not a.only or c[0] in {int(x) for x in a.only.split(",")}]
    OUT.mkdir(parents=True, exist_ok=True)

    def run(case):
        t0 = time.time()
        try:
            res = post(a.api, case[2], verify=a.verify, multi=a.multi)
        except Exception as e:
            return case, None, {"error": str(e)}, time.time() - t0
        try:
            j = judge(client, case, res, a.api)
        except Exception as e:
            j = {"error": f"judge failed: {e}"}
        return case, res, j, time.time() - t0

    rows = []
    with cf.ThreadPoolExecutor(a.parallel) as ex:
        for case, res, j, secs in ex.map(run, cases):
            cid, typ, q, facts, abstain = case
            if res is None or "error" in j:
                print(f"#{cid:<2} ERROR {j}")
                rows.append({"id": cid, "type": typ, "error": True})
                continue
            found = j.get("facts_found") or []
            correct = (sum(bool(x) for x in found) / len(facts)) if facts else None
            unsup = j.get("unsupported") or []
            abstain_ok = (bool(j.get("abstained")) == abstain) if abstain else not j.get("abstained")
            want = SESSION_OF.get(cid)
            row = {"found_session": (any(x["session_id"] == want for x in res["sources"]) if want else None),
                   "mode": res.get("mode"), "plan_type": (res.get("plan") or {}).get("type"),
                   "pii_leak": bool(j.get("pii_leak")), "id": cid, "type": typ, "correct": correct, "unsupported": len(unsup), "claims": j.get("claims_checked"),
                   "abstain_ok": abstain_ok, "invalid_cites": len(res["invalid_citations"]),
                   "sources": len(res["sources"]), "tools": len(res["steps"]), "seconds": res["seconds"],
                   "tokens_in": res["tokens"]["in"], "tokens_cached": res["tokens"].get("cached", 0), "unsupported_claims": unsup, "notes": j.get("notes"),
                   "answer": res["answer"], "question": q}
            rows.append(row)
            c = f"{correct:.2f}" if correct is not None else "  - "
            print(f"#{cid:<2} {typ:<7} [{res.get('mode', '?'):<6}] correct {c}  unsupported {len(unsup)}/{j.get('claims_checked')}  "
                  f"abstain_ok {abstain_ok!s:<5} tools {len(res['steps']):>2}  {res['seconds']:>5.1f}s  | {j.get('notes', '')[:70]}")

    ok = [r for r in rows if not r.get("error")]
    graded = [r for r in ok if r["correct"] is not None]
    summary = {
        "tag": a.tag, "n": len(rows), "errors": len(rows) - len(ok),
        "correctness": round(sum(r["correct"] for r in graded) / max(len(graded), 1), 3),
        "answers_with_unsupported_claims": sum(r["unsupported"] > 0 for r in ok),
        "unsupported_claim_rate": round(sum(r["unsupported"] for r in ok) / max(sum(r["claims"] or 0 for r in ok), 1), 3),
        "abstain_accuracy": round(sum(r["abstain_ok"] for r in ok) / max(len(ok), 1), 3),
        "invalid_citations": sum(r["invalid_cites"] for r in ok),
        "pii_leaks": sum(r["pii_leak"] for r in ok),
        "cited_the_right_conversation": (lambda xs: f"{sum(xs)}/{len(xs)}" if xs else None)(
            [r["found_session"] for r in ok if r.get("found_session") is not None]),
        "median_seconds": sorted(r["seconds"] for r in ok)[len(ok) // 2] if ok else None,
        "mean_tokens_in": int(sum(r["tokens_in"] for r in ok) / max(len(ok), 1)),
        "cache_hit_share": round(sum(r.get("tokens_cached", 0) for r in ok) / max(sum(r["tokens_in"] for r in ok), 1), 2),
    }
    print("\nSUMMARY", json.dumps(summary, indent=1))
    with open(OUT / f"{a.tag}.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8002")
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--set", choices=["basic", "hard", "temporal", "v2", "all"], default="basic")
    ap.add_argument("--multi", action="store_true", help="enable the planner / multi-agent 'deep research' path")
    main(ap.parse_args())
