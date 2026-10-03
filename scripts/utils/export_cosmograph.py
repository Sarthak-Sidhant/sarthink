"""Export the social graph for the 3D viewer.

Nodes:
  P_<person_id>  one node per person (identity-resolved via sarthink_index.db; all my accounts
                 collapse into a single "me" node)
  T_<thread_id>  one node per conversation thread
Edges: person -> thread, weighted by the number of messages that person wrote there.

Thread groups come from title/id patterns per platform (DMs vs public/comment threads).
Requires scripts/index/build_identity.py to have been run.

Usage: python3 scripts/utils/export_cosmograph.py   (then compute_layout.py)
export(keep_layout=True) re-exports after people were linked/unlinked without a new layout: nodes keep their
positions, and a person new to the graph is placed next to its chats.
"""
import csv
import logging
import os
import random
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
    "whatsapp_user":           "#B388FF",  # violet (not WhatsApp green: Discord already uses green)
    "whatsapp_dm_group":       "#7C4DFF",
}

OUT_DIR = GRAPH_DIR


def thread_group(platform, title, platform_thread_id):
    t = (title or "").strip()
    tl = t.lower()
    if platform in ("discord", "whatsapp"):
        return f"{platform}_dm_group"
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


# country calling codes seen in WhatsApp exports (longest match wins); others fall back to "+<digits>"
CALLING_CODES = {"1", "7", "20", "27", "30", "31", "32", "33", "34", "39", "41", "43", "44", "45", "46", "47", "49",
                 "52", "55", "60", "61", "62", "63", "64", "65", "66", "81", "82", "84", "86", "90", "91", "92", "93",
                 "94", "95", "98", "212", "234", "254", "351", "353", "880", "960", "961", "962", "963", "964", "965",
                 "966", "967", "968", "970", "971", "972", "973", "974", "975", "977"}


def format_phone(raw_id):
    """A WhatsApp account id (country code + number, digits only) as a readable phone number: +91 82407 81962.
    Hidden-number contacts ('lid:…') and anything that isn't a plain number give ''."""
    d = str(raw_id or "")
    if not d.isdigit() or not 8 <= len(d) <= 15:
        return ""
    cc = next((d[:k] for k in (3, 2, 1) if d[:k] in CALLING_CODES), "")
    rest = d[len(cc):]
    if cc == "91" and len(rest) == 10:
        rest = f"{rest[:5]} {rest[5:]}"
    elif cc == "1" and len(rest) == 10:
        rest = f"{rest[:3]} {rest[3:6]} {rest[6:]}"
    elif len(rest) == 8:
        rest = f"{rest[:4]} {rest[4:]}"                                             # +65 9123 4567
    else:
        rest = " ".join(x for x in (rest[:-6], rest[-6:-3], rest[-3:]) if x)       # +44 7700 900 123
    return f"+{cc} {rest}" if cc else f"+{d}"


def export(conn=None, keep_layout=False):
    """Account (A_) and thread (T_) nodes, account->thread edges. People are not in these files: the API serves
    them and the viewer draws them linked to their accounts, so linking never rewrites the graph."""
    conn = conn or connect_index()
    account_of = {r[0]: r[1] for r in conn.execute("SELECT user_id, account_id FROM AccountUsers")}
    per_chat = {(r[0], r[1]): r[2] for r in conn.execute("SELECT user_id, thread_id, account_id FROM AccountThreads")}
    accounts = {r["account_id"]: dict(r) for r in conn.execute(
        "SELECT account_id, name, platform, is_me, raw_id FROM Accounts")}

    weights = defaultdict(int)          # (account, thread) -> messages
    for author_id, thread_id, n in conn.execute(
            "SELECT author_id, thread_id, COUNT(*) FROM src.Messages GROUP BY author_id, thread_id"):
        aid = per_chat.get((author_id, thread_id), account_of.get(author_id))
        if aid is not None:
            weights[(aid, thread_id)] += n

    size = defaultdict(int)
    for (aid, tid), n in weights.items():
        size[f"A_{aid}"] += n
        size[f"T_{tid}"] += n

    nodes = []
    for aid in {a for a, _ in weights}:
        a = accounts[aid]
        group = f"{a['platform']}_me" if a["is_me"] else f"{a['platform']}_user"   # my accounts: "<platform>_me"
        label = a["name"]
        if a["is_me"]:          # "Me on Discord (sarthak sidhant)", or just "Me on WhatsApp" when it carries my name
            plat = a["platform"].capitalize().replace("Whatsapp", "WhatsApp")
            label = f"Me on {plat}" + ("" if "(me)" in a["name"] else f" ({a['name']})")
        color = GROUP_COLORS["me"] if a["is_me"] else GROUP_COLORS.get(group, "#aaaaaa")
        phone = format_phone(a["raw_id"]) if a["platform"] == "whatsapp" and not a["is_me"] else ""
        nodes.append((f"A_{aid}", clean(label), group, size[f"A_{aid}"], color, phone))

    # identical labels for different accounts (e.g. many deleted accounts) get #2, #3… in the graph only
    seen = defaultdict(int)
    for i in sorted(range(len(nodes)), key=lambda i: -nodes[i][3]):     # biggest keeps the plain label
        nid, label, group, weight, color, phone = nodes[i]
        seen[label] += 1
        if seen[label] > 1:
            nodes[i] = (nid, f"{label} #{seen[label]}", group, weight, color, phone)

    thread_ids = {t for _, t in weights}
    for r in conn.execute("SELECT id, platform, title, platform_thread_id FROM src.Threads"):
        if r["id"] not in thread_ids:
            continue
        group = thread_group(r["platform"], r["title"], r["platform_thread_id"])
        nodes.append((f"T_{r['id']}", clean(r["title"], 45) or "Untitled", group, size[f"T_{r['id']}"],
                      GROUP_COLORS.get(group, "#aaaaaa"), ""))

    edges = [(f"A_{aid}", f"T_{tid}", n) for (aid, tid), n in weights.items()]

    header = ["id", "label", "group", "size", "color", "phone"]   # phone: WhatsApp contacts only
    if keep_layout:
        nodes = with_old_layout(nodes, edges)
        header += ["layout_x", "layout_y", "layout_z"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "cosmograph_nodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(header)
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


def with_old_layout(nodes, edges):
    with open(OUT_DIR / "cosmograph_nodes.csv", encoding="utf-8") as f:
        pos = {r["id"]: (r["layout_x"], r["layout_y"], r["layout_z"]) for r in csv.DictReader(f)}
    for k in [k for k in pos if k.startswith("P_")]:     # files written before accounts and people were split
        pos.setdefault("A_" + k[2:], pos[k])
    near = defaultdict(list)
    for p, t, _ in edges:
        if t in pos:
            near[p].append([float(v) for v in pos[t]])
    rnd = random.Random(42)
    out = []
    for n in nodes:
        xyz = pos.get(n[0])
        if xyz is None:
            pts = near.get(n[0]) or [[0.0, 0.0, 0.0]]
            xyz = tuple(f"{sum(p[i] for p in pts) / len(pts) + rnd.uniform(-60, 60):.4f}" for i in range(3))
        out.append((*n, *xyz))
    return out


if __name__ == "__main__":
    export()
