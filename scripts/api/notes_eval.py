"""Notes eval: a scripted sequence (order matters) checking that plain-language notes are saved, recalled only
when relevant, cited as [note], respected (pronouns, corrections, aliases), listed and forgotten.
Test notes are removed at the end; pre-existing notes (e.g. FireRoz's pronouns) are left alone.

Usage: python3 scripts/api/notes_eval.py [--api http://127.0.0.1:8000]
"""
import argparse
import json
import re
import sqlite3
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOTES_DB = REPO / "processed_data" / "db" / "user_notes.db"


def ask(api, q, history=None):
    req = urllib.request.Request(f"{api}/api/ask", data=json.dumps({"question": q, "history": history or []}).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def tools(d):
    return [x["tool"] for x in d["steps"]]


def words(text, ws):
    t = f" {re.sub(r'[^a-z]+', ' ', text.lower())} "
    return any(f" {w} " in t for w in ws)


def active_notes():
    c = sqlite3.connect(NOTES_DB)
    return c.execute("SELECT note_id, subject, text FROM Notes WHERE active=1").fetchall()


def main(api):
    before = {r[0] for r in active_notes()}
    results = []

    def check(name, ok, detail):
        results.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  — {detail}")

    # 1. plain-language save: alias for a person
    d = ask(api, "please remember that quazarblaster usually goes by Quaz")
    check("save person note (alias)", "remember" in tools(d), f"tools={tools(d)}")

    # 2. alias recall: the question only says "Quaz"
    d = ask(api, "What did I interview Quaz for, and where is Quaz from?")
    a = d["answer"]
    check("alias note resolves the person", "howrah" in a.lower() or "moderat" in a.lower(),
          a[:160].replace("\n", " "))

    # 3. pre-existing pronoun note (FireRoz she/her) used without naming her in the question
    d = ask(api, "Who refused to do the PayPal-to-crypto transfer for me, and why?")
    a = d["answer"]
    fire = "fireroz" in a.lower()
    check("pronoun note respected", fire and words(a, ["she", "her"]) and not words(a, ["he", "him", "his"]),
          f"mentions FireRoz={fire}, she/her={words(a, ['she', 'her'])}, he/him={words(a, ['he', 'him', 'his'])}")

    # 4. topic note with trigger keywords
    d = ask(api, "remember that darshi is the civic-issues app for Ranchi that anubis prototyped")
    check("save topic note", "remember" in tools(d), f"tools={tools(d)}")
    d = ask(api, "what happened with darshi?")
    check("topic note recalled + cited", "[note]" in d["answer"], d["answer"][:160].replace("\n", " "))

    # 5. correction note that contradicts the archive's framing
    d = ask(api, "remember that Porkifiable was not a stranger: we had been chatting on Discord for weeks before the money stuff")
    check("save correction note", "remember" in tools(d), f"tools={tools(d)}")
    d = ask(api, "Was the guy who bought me the Zimaboard stuff a stranger?")
    a = d["answer"].lower()
    check("correction note used", "[note]" in a and ("not a stranger" in a or "weeks" in a), d["answer"][:200].replace("\n", " "))

    # 6. precision: an unrelated question must not drag notes in
    d = ask(api, "What was the plan with Krish to hack the school?")
    check("no notes on unrelated question", "[note]" not in d["answer"], f"[note] present={'[note]' in d['answer']}")

    # 7. list
    d = ask(api, "what have I asked you to remember?")
    a = d["answer"].lower()
    check("list notes", all(x in a for x in ("quaz", "darshi", "porkifiable")), a[:160].replace("\n", " "))

    # 8. forget + no recall afterwards
    d = ask(api, "forget what I told you about Quaz")
    gone = not any("quaz" in (r[2] or "").lower() for r in active_notes())
    check("forget removes the note", gone, f"tools={tools(d)}")

    # cleanup: deactivate test notes created by this run
    c = sqlite3.connect(NOTES_DB)
    new = [r[0] for r in c.execute("SELECT note_id FROM Notes WHERE active=1") if r[0] not in before]
    c.executemany("UPDATE Notes SET active=0 WHERE note_id=?", [(i,) for i in new])
    c.commit()
    passed = sum(ok for _, ok in results)
    print(f"\nNOTES EVAL: {passed}/{len(results)} passed (cleaned up {len(new)} test notes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    main(ap.parse_args().api)
