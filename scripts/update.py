"""Add a newer export to Sarthink, end to end, and report what it brought.

    python3 scripts/update.py ~/Downloads/instagram-*.zip            # Instagram/Facebook export (all parts)
    python3 scripts/update.py archive/whatsapp-msgstore.db --contacts archive/whatsapp-contacts.vcf
    python3 scripts/update.py --from summaries --tag update-2026-10-04   # resume at a stage

Stages (each is safe to rerun; --from starts at one):
  parse       backs up the databases, adds the new messages (already-stored ones are skipped) and reports
              what is new: messages per platform and month, new chats, renamed chats
  index       accounts, sessions/chunks and keyword search, rebuilt; unchanged conversations keep their ids
  summaries   DeepSeek summaries for sessions without one. Paid: a ~100-session sample runs first, its tokens
              are priced at DeepSeek's list price and projected; the rest runs only if the projection is within
              --budget (or you confirm). --offpeak waits for DeepSeek's half-price hours
  embeddings  8B vectors for chunks without one, on OpenRouter (~$0.01 per million tokens)
  inbox       promises/plans from the new summaries, then triage: current items the chat already settled
              (done, passed, trivial) move to "Auto-cleared" (DeepSeek, cents)
  graph       graph files re-exported; existing nodes keep their positions, new ones sit by their chats

Restart the API afterwards (it caches accounts and vectors). The report is saved to
processed_data/logs/<tag>.md.
"""
import argparse
import datetime as dt
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "index"))
from common import DATA_DIR, INDEX_DB, PROFILE_CACHE, SRC_DB, USER_DB  # noqa: E402

STAGES = ["parse", "index", "summaries", "embeddings", "inbox", "graph"]
SAMPLE = 100          # sessions summarized first to measure the real cost
STATE = DATA_DIR / "logs" / "update_state.json"     # what the parse stage saw, for the report after a resume


def say(msg=""):
    print(msg, flush=True)
    REPORT.append(msg)


def run(script, *args):
    say(f"\n$ {script} {' '.join(map(str, args))}")
    subprocess.run([sys.executable, str(REPO / script), *map(str, args)], check=True, cwd=REPO)


def src():
    return sqlite3.connect(f"file:{SRC_DB}?mode=ro", uri=True)


# ─── parse ────────────────────────────────────────────────────────────────────

def kind_of(path):
    name = path.name.lower()
    if name.endswith(".zip") and ("instagram" in name or "facebook" in name):
        return "meta"
    if name.endswith(".db") and "msgstore" in name:
        return "whatsapp"
    return None


def backup():
    dest = DATA_DIR / "backups" / f"{dt.datetime.now():%Y-%m-%d_%H%M}_pre-update"
    dest.mkdir(parents=True, exist_ok=True)
    for db in (SRC_DB, INDEX_DB, USER_DB):
        if db.exists():
            with sqlite3.connect(db) as s, sqlite3.connect(dest / db.name) as d:
                s.backup(d)
    say(f"Backed up the databases to {dest.relative_to(REPO) if dest.is_relative_to(REPO) else dest}")


def stage_parse(a):
    files = [Path(f).expanduser().resolve() for f in a.exports]
    if not files:
        sys.exit("parse: give the export files (or --from a later stage)")
    kinds = {f: kind_of(f) for f in files}
    bad = [str(f) for f, k in kinds.items() if not k]
    if bad:
        sys.exit("Not supported by update.py yet (see the README's pipeline): " + ", ".join(bad))
    with src() as c:
        before = {"rid": c.execute("SELECT MAX(rowid) FROM Messages").fetchone()[0] or 0,
                  "tid": c.execute("SELECT MAX(id) FROM Threads").fetchone()[0] or 0}
        titles = dict(c.execute("SELECT id, title FROM Threads"))
    backup()
    meta = [f for f, k in kinds.items() if k == "meta"]
    if meta:
        run("scripts/parsers/meta_parser.py", *meta)
    for f in [f for f, k in kinds.items() if k == "whatsapp"]:
        run("scripts/parsers/whatsapp_parser.py", "--db", f, *(["--contacts", a.contacts] if a.contacts else []))
    with src() as c:
        changed = [(i, titles[i], t) for i, t in c.execute("SELECT id, title FROM Threads WHERE id <= ?",
                                                            (before["tid"],)) if titles.get(i) != t]
    renamed = [(o, t) for _, o, t in changed]
    save_state(a.tag, **before, renamed=renamed, exports=[str(f) for f in files])
    return report_new(before["rid"], before["tid"], renamed)


def load_state(tag):
    try:
        st = json.load(open(STATE))
        return st if st.get("tag") == tag else {"tag": tag}
    except (OSError, ValueError):
        return {"tag": tag}


def save_state(tag, **kv):
    st = load_state(tag)
    st.update(kv)
    json.dump(st, open(STATE, "w"), ensure_ascii=False)


def report_new(rid, tid, renamed):
    with src() as c:
        q = lambda sql: c.execute(sql, (rid,)).fetchall()  # noqa: E731
        n = q("SELECT COUNT(*) FROM Messages WHERE rowid > ?")[0][0]
        say(f"\n## What the export added\n\n{n:,} new messages")
        if not n:
            return 0
        for plat, k, a_, b_ in q("""SELECT t.platform, COUNT(*), MIN(m.timestamp_utc), MAX(m.timestamp_utc)
                                    FROM Messages m JOIN Threads t ON t.id = m.thread_id WHERE m.rowid > ?
                                    GROUP BY t.platform ORDER BY 2 DESC"""):
            f = lambda t: dt.datetime.fromtimestamp(t).strftime("%d %b %Y") if t else "?"  # noqa: E731
            say(f"- {plat}: {k:,} ({f(a_)} – {f(b_)})")
        months = q("""SELECT strftime('%Y-%m', timestamp_utc, 'unixepoch') m, COUNT(*) FROM Messages
                      WHERE rowid > ? AND timestamp_utc IS NOT NULL GROUP BY m ORDER BY m""")
        top = max(k for _, k in months) if months else 1
        say("\nBy month:")
        for m, k in months[-14:]:
            say(f"  {m}  {'█' * max(1, round(28 * k / top))} {k:,}")
        if len(months) > 14:
            say(f"  (+ {sum(k for _, k in months[:-14]):,} older messages, mostly chats the previous export lacked)")
        chats = c.execute("""SELECT t.title, t.platform, COUNT(*), t.id > ? FROM Messages m JOIN Threads t ON t.id = m.thread_id
                             WHERE m.rowid > ? GROUP BY t.id ORDER BY 3 DESC LIMIT 8""", (tid, rid)).fetchall()
        new_chats = c.execute("SELECT COUNT(*) FROM Threads WHERE id > ?", (tid,)).fetchone()[0]
        say(f"\n{new_chats:,} new chats. Busiest:")
        for title, plat, k, new in chats:
            say(f"  {k:>6,}  {title[:50]} ({plat}{', new' if new else ''})")
    if renamed:
        say(f"\n{len(renamed)} chats renamed by the newer export, e.g.:")
        for old, new in renamed[:6]:
            say(f"  {old[:40]}  →  {new[:40]}")
    return n


# ─── index ────────────────────────────────────────────────────────────────────

def counts():
    with sqlite3.connect(INDEX_DB) as c:
        return {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("Accounts", "Sessions", "Chunks")}


def chunk_texts():
    with sqlite3.connect(INDEX_DB) as c:
        return {cid: hash((h, t)) for cid, h, t in c.execute("SELECT chunk_id, header, text FROM Chunks")}


def stage_index(a):
    old, before = counts(), chunk_texts()
    run("scripts/index/build_identity.py")
    run("scripts/index/build_chunks.py")
    run("scripts/index/check_chunks.py")
    run("scripts/index/build_fts.py")
    new, after = counts(), chunk_texts()
    changed = [cid for cid, h in after.items() if cid in before and before[cid] != h]
    save_state(a.tag, changed_chunks=sorted(set(load_state(a.tag).get("changed_chunks", [])) | set(changed)))
    say("\n## Index\n\n" + ", ".join(f"{k}: {old[k]:,} → {new[k]:,}" for k in new)
        + f"\n{len(changed):,} existing chunks changed text (new messages, renamed chats): re-embedded later")


# ─── summaries ────────────────────────────────────────────────────────────────

# DeepSeek list prices, $ per million tokens at peak (off-peak is half), from api-docs.deepseek.com/quick_start/pricing
# (Oct 2026). Input is priced as a cache miss, so estimates are an upper bound. (The account balance can't measure a
# sample: it trails spending by minutes to hours.)
PRICE = {"in": 0.30, "out": 1.20}


def pending():
    """(sessions, their tokens) still without a summary (same rule as summarize_sessions.py)."""
    from summarize_sessions import PROMPT_VERSION
    with sqlite3.connect(INDEX_DB) as c:
        return c.execute("""SELECT COUNT(*), COALESCE(SUM(token_count), 0) FROM Sessions s WHERE token_count >= 100
                            AND NOT EXISTS (SELECT 1 FROM SessionSummaries x WHERE x.session_id = s.session_id
                                            AND x.model = 'deepseek-flash' AND x.prompt_version = ?)""",
                         (PROMPT_VERSION,)).fetchone()


def spent(tag, offpeak):
    """Tokens and cost of the summaries made under this tag so far, and their conversations' size."""
    with sqlite3.connect(INDEX_DB) as c:
        tin, tout, conv = c.execute("""SELECT COALESCE(SUM(x.input_tokens), 0), COALESCE(SUM(x.output_tokens), 0),
                                              COALESCE(SUM(s.token_count), 0)
                                       FROM SessionSummaries x JOIN Sessions s USING(session_id) WHERE x.tag = ?""",
                                    (tag,)).fetchone()
    k = 0.5 if offpeak else 1.0
    return tin, tout, conv, (tin * PRICE["in"] + tout * PRICE["out"]) * k / 1e6


def stage_summaries(a):
    n, tokens = pending()
    say(f"\n## Summaries\n\n{n:,} sessions need a summary ({tokens:,} tokens of conversation)")
    if not n:
        return True
    off = ["--offpeak-only"] if a.offpeak else []
    run("scripts/index/summarize_sessions.py", "--limit", min(SAMPLE, n), "--sample", "--tag", a.tag, *off)
    tin, tout, conv, cost = spent(a.tag, a.offpeak)
    n2, tokens2 = pending()
    if n2 == n:                     # none of the sample went through: DeepSeek refuses these (content filter)
        say(f"DeepSeek declined all {n:,}; they stay unsummarized (search still finds them)")
        return True
    est = cost / conv * tokens2 if conv else None
    say(f"So far under {a.tag}: {tin:,} tokens in, {tout:,} out ≈ ${cost:.2f}. The other {n2:,} sessions: "
        + (f"about ${est:.2f}" if est is not None else "cost unknown")
        + (" (off-peak price)" if a.offpeak else " (peak price; --offpeak halves it)"))
    if not n2:
        return True
    if not confirm(a, est, n2):
        return False
    run("scripts/index/summarize_sessions.py", "--tag", a.tag, "--concurrency", 32, *off)
    tin, tout, _, cost = spent(a.tag, a.offpeak)
    say(f"Summaries under {a.tag}: {tin:,} tokens in, {tout:,} out ≈ ${cost:.2f}")
    return True


def confirm(a, est, n):
    if a.yes or (est is not None and est <= a.budget):
        return True
    if sys.stdin.isatty():
        ans = input(f"Summarize the remaining {n:,} sessions{f' for about ${est:.2f}' if est else ''}? [y/N] ")
        if ans.strip().lower().startswith("y"):
            return True
    say(f"Stopped before paying for the rest. To continue:\n"
        f"  python3 scripts/update.py --from summaries --tag {a.tag} --yes")
    return False


# ─── embeddings, inbox, graph ─────────────────────────────────────────────────

def stage_embeddings(a):
    out = DATA_DIR / "index" / "updates" / a.tag
    extra, changed = [], load_state(a.tag).get("changed_chunks")
    if changed:
        out.parent.mkdir(parents=True, exist_ok=True)
        f = out.parent / f"{a.tag}.changed.json"
        json.dump(changed, open(f, "w"))
        extra = ["--chunks-file", f]
    run("scripts/index/embed_openrouter.py", out, "--missing", "--tag", a.tag, *extra)
    if Path(f"{out}.npy").exists():
        run("scripts/index/merge_embeddings.py", out)


def stage_inbox(a):
    run("scripts/index/build_commitments.py", "--tag", a.tag)
    run("scripts/index/triage_inbox.py")       # settle current items against the chat (DeepSeek, cents)


def stage_graph(a):
    say("\n$ export the graph (positions kept)")
    sys.path.insert(0, str(REPO / "scripts" / "utils"))
    import export_cosmograph
    export_cosmograph.export(keep_layout=True)
    if PROFILE_CACHE.exists():      # dossiers re-check their fingerprints anyway; this drops the stale ones now
        with sqlite3.connect(PROFILE_CACHE) as c:
            if c.execute("SELECT 1 FROM sqlite_master WHERE name = 'ScopeDossiers'").fetchone():
                c.execute("DELETE FROM ScopeDossiers")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exports", nargs="*", help="export files: Instagram/Facebook zips (all parts), WhatsApp msgstore.db")
    ap.add_argument("--contacts", help="WhatsApp: phone contacts (.vcf) for names")
    ap.add_argument("--from", dest="start", choices=STAGES, default="parse", help="start at this stage")
    ap.add_argument("--only", choices=STAGES, help="run just this stage")
    ap.add_argument("--tag", default=f"update-{dt.date.today()}", help="labels this update's summaries and files")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="go ahead with summaries without asking if the projected cost is at most this many dollars")
    ap.add_argument("--yes", action="store_true", help="don't ask before paid steps")
    ap.add_argument("--offpeak", action="store_true", help="summaries only in DeepSeek's half-price hours")
    a = ap.parse_args()

    global REPORT
    REPORT = [f"# Sarthink update {a.tag}", f"Data: {DATA_DIR}", f"Started {dt.datetime.now():%d %b %Y %H:%M}"]
    stages = [a.only] if a.only else STAGES[STAGES.index(a.start):]
    t0 = time.time()
    for st in stages:
        if st == "parse":
            if not stage_parse(a):
                say("\nNothing new in this export; the rest is skipped.")
                break
        elif st == "index":
            stage_index(a)
        elif st == "summaries":
            if not stage_summaries(a):
                break
        elif st == "embeddings":
            stage_embeddings(a)
        elif st == "inbox":
            stage_inbox(a)
        elif st == "graph":
            stage_graph(a)
    else:
        say(f"\nDone in {(time.time() - t0) / 60:.0f} min."
            + (" Restart the API to load the new data." if set(stages) - {"parse"} else ""))
    out = DATA_DIR / "logs" / f"{a.tag}.md"
    with open(out, "a") as f:
        f.write("\n".join(REPORT) + "\n\n")
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
