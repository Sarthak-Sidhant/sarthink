"""Export the social graph for the 3D viewer.

Nodes:
  P_<person_id>  one node per person (identity-resolved via sarthink_index.db; all my accounts
                 collapse into a single "me" node)
  T_<thread_id>  one node per conversation thread
Edges: person -> thread, weighted by the number of messages that person wrote there.

Thread groups come from title/id patterns per platform (DMs vs public/comment threads).
Requires scripts/index/build_identity.py to have been run.

Usage: python3 scripts/utils/export_cosmograph.py   (then compute_layout.py)
"""
import csv
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "index"))
from common import GRAPH_DIR, connect_index  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(message)s')

GROUP_COLORS = {
    "me":                      "#FFD54A",  # gold — the centre of the galaxy
    "twitter_user":            "#1DA1F2",
    "twitter_public_thread":   "#0a3d6b",
    "twitter_dm_group":        "#4fc3f7",
    "reddit_user":             "#FF4500",
    "reddit_public_thread":    "#c13a00",
    "reddit_dm_group":         "#ff8c42",
    "instagram_user":          "#E1306C",
    "instagram_comment_thread": "#833ab4",
    "instagram_dm_group":      "#fd5c87",
    "discord_user":            "#00E676",  # green (not Discord blurple: too close to Twitter blue)
    "discord_dm_group":        "#1DE9B6",  # teal
    "facebook_user":           "#FFFFFF",  # white/silver (not Facebook blue, same reason)
    "facebook_comment_thread": "#90A4AE",
    "facebook_dm_group":       "#CFD8DC",
}

OUT_DIR = GRAPH_DIR


def thread_group(platform, title, platform_thread_id):
    t = (title or "").strip()
    tl = t.lower()
    if platform == "discord":
        return "discord_dm_group"
    if platform in ("facebook", "instagram"):
        return f"{platform}_comment_thread" if tl.startswith("comment on") else f"{platform}_dm_group"
    if platform == "reddit":
        is_dm = tl.startswith("dm") or ":reddit.com" in (platform_thread_id or "")
        return "reddit_dm_group" if is_dm else "reddit_public_thread"
    if platform == "twitter":
        return "twitter_dm_group" if tl.startswith("dm") else "twitter_public_thread"
    return f"{platform}_public_thread"


def clean(s, n=None):
    s = str(s or "").replace(",", "").replace("\n", " ").replace("\r", "").replace('"', "").strip()
    return (s[:n] + "...") if n and len(s) > n else s


def export():
    conn = connect_index()
    person_of = {r["user_id"]: r["person_id"] for r in conn.execute("SELECT user_id, person_id FROM PersonAliases")}
    per_chat = {(r[0], r[1]): r[2] for r in conn.execute("SELECT user_id, thread_id, person_id FROM PersonThreads")}
    persons = {r["person_id"]: dict(r) for r in conn.execute("SELECT person_id, name, is_me FROM Persons")}
    user_platform = {r["id"]: r["platform"] for r in conn.execute("SELECT id, platform FROM src.Users")}

    weights = defaultdict(int)          # (person, thread) -> messages
    person_platform = {}
    for author_id, thread_id, n in conn.execute(
            "SELECT author_id, thread_id, COUNT(*) FROM src.Messages GROUP BY author_id, thread_id"):
        pid = per_chat.get((author_id, thread_id), person_of.get(author_id))
        if pid is None:
            continue
        weights[(pid, thread_id)] += n
        person_platform.setdefault(pid, user_platform.get(author_id))

    size = defaultdict(int)
    for (pid, tid), n in weights.items():
        size[f"P_{pid}"] += n
        size[f"T_{tid}"] += n

    nodes = []
    for pid in {p for p, _ in weights}:
        p = persons[pid]
        group = "me" if p["is_me"] else f"{person_platform[pid]}_user"
        nodes.append((f"P_{pid}", clean(p["name"]), group, size[f"P_{pid}"], GROUP_COLORS.get(group, "#aaaaaa")))

    # identical labels for different people (e.g. many deleted accounts) get #2, #3… in the graph only
    seen = defaultdict(int)
    for i in sorted(range(len(nodes)), key=lambda i: -nodes[i][3]):     # biggest keeps the plain label
        nid, label, group, weight, color = nodes[i]
        seen[label] += 1
        if seen[label] > 1:
            nodes[i] = (nid, f"{label} #{seen[label]}", group, weight, color)

    thread_ids = {t for _, t in weights}
    for r in conn.execute("SELECT id, platform, title, platform_thread_id FROM src.Threads"):
        if r["id"] not in thread_ids:
            continue
        group = thread_group(r["platform"], r["title"], r["platform_thread_id"])
        nodes.append((f"T_{r['id']}", clean(r["title"], 45) or "Untitled", group, size[f"T_{r['id']}"],
                      GROUP_COLORS.get(group, "#aaaaaa")))

    edges = [(f"P_{pid}", f"T_{tid}", n) for (pid, tid), n in weights.items()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "cosmograph_nodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["id", "label", "group", "size", "color"])
        w.writerows(nodes)
    with open(OUT_DIR / "cosmograph_edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["source", "target", "weight"])
        w.writerows(edges)

    counts = defaultdict(int)
    for n in nodes:
        counts[n[2]] += 1
    logging.info(f"Wrote {len(nodes)} nodes, {len(edges)} edges")
    for g, c in sorted(counts.items(), key=lambda x: -x[1]):
        logging.info(f"  {g:<26} {c}")


if __name__ == "__main__":
    export()
