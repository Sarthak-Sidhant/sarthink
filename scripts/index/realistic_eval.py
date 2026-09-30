"""Hand-written, realistic questions (vague, Hinglish, typos, person/date filters) against the live API.
A question passes if a known distinctive string from the right conversation appears in the top-k hits.

Usage: python3 scripts/index/realistic_eval.py [--api http://127.0.0.1:8000] [-k 5]
"""
import argparse
import json
import time
import urllib.parse
import urllib.request

# (style, question, extra params, strings that identify the right conversation — any one counts)
QUESTIONS = [
    ("casual",      "what was I buying from zimaboard",                       {}, ["zimaboard"]),
    ("vague",       "that time I got sad because someone called me ghonchu",  {}, ["ghonchu"]),
    ("casual",      "what did people say about my reddit stalking tool",      {}, ["redstalk", "stalk"]),
    ("specific",    "lost my mettle bottle",                                  {}, ["mettle"]),
    ("hinglish",    "wayland ya x11 kaunsa better hai wali baat",             {}, ["wayland"]),
    ("hinglish",    "harsh ne linux ke baare mein kya pucha tha",             {}, ["wayland", "x11", "linux"]),
    ("casual",      "someone told me to sign up on upwork",                   {}, ["upwork"]),
    ("specific",    "argument about beej mantra and asuras",                  {}, ["beej", "asura"]),
    ("vague",       "when did I tell someone I'd go offline to study",        {}, ["not reply the following days", "studying"]),
    ("typo",        "x99 sever bild with tesla p40",                          {}, ["x99", "p40"]),
    ("specific",    "home server xeon build questions",                       {}, ["xeon", "home server"]),
    ("filter",      "minecraft server",                                       {"person": "gareth"}, ["minecraft", "server", "mc"]),
    ("filter",      "internship",                                             {"since": "2025-01-01"}, ["intern"]),
    ("casual",      "cheapest place to buy electronics in ranchi",            {}, ["ranchi"]),
]


def run(api, k):
    passed, t_all = 0, []
    for style, q, extra, needles in QUESTIONS:
        params = {"q": q, "k": k, **extra}
        t0 = time.time()
        with urllib.request.urlopen(f"{api}/api/search?{urllib.parse.urlencode(params)}", timeout=120) as r:
            d = json.load(r)
        t_all.append(time.time() - t0)
        hits = d["results"]
        rank = next((i + 1 for i, h in enumerate(hits)
                     if any(n.lower() in (h["text"] + " " + (h["context"] or "")).lower() for n in needles)), None)
        passed += rank is not None
        mark = f"#{rank}" if rank else "MISS"
        filt = f" {extra}" if extra else ""
        print(f"[{mark:>4}] ({style}) {q}{filt}")
        top = hits[0]["header"][:90] if hits else "(no results)"
        print(f"         top: {top}")
    print(f"\n{passed}/{len(QUESTIONS)} found in top {k}; median latency {sorted(t_all)[len(t_all) // 2]:.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    run(a.api, a.k)
