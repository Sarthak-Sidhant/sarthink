"""Structural checks for the chunk index. Exits non-zero if a hard check fails.

Usage: python3 scripts/index/check_chunks.py [--target 300]
"""
import argparse
import statistics
import sys

from common import connect_index


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else 0


def main(target):
    conn = connect_index()
    failures = []

    total, = conn.execute("SELECT COUNT(*) FROM src.Messages").fetchone()
    chunked, = conn.execute("SELECT COUNT(DISTINCT msg_id) FROM ChunkMessages").fetchone()
    skipped, = conn.execute("SELECT COUNT(*) FROM MessageSkips").fetchone()
    lost = total - chunked - skipped
    print(f"Coverage: {chunked} chunked + {skipped} skipped (with reason) of {total} messages; unaccounted: {lost}")
    for r in conn.execute("SELECT reason, COUNT(*) FROM MessageSkips GROUP BY 1"):
        print(f"   skipped {r[0]}: {r[1]}")
    if lost:
        failures.append(f"{lost} messages neither chunked nor skipped")

    rows = conn.execute("SELECT platform, structure, token_count, message_count, index_policy FROM Chunks").fetchall()
    toks = [r["token_count"] for r in rows]
    print(f"Chunks: {len(rows)} | tokens p10 {q(toks, .1)} p50 {q(toks, .5)} p90 {q(toks, .9)} max {max(toks)}")
    # body may exceed target by the tree ancestor prefix (<=80) plus one unit at the boundary
    hard_cap = target + 120
    over = sum(t > hard_cap for t in toks)
    print(f"   over hard cap {hard_cap}: {over}")
    if over:
        failures.append(f"{over} chunks over {hard_cap} tokens")

    print("Per platform/structure (chunks, embed, median tokens, median msgs):")
    groups = {}
    for r in rows:
        groups.setdefault((r["platform"], r["structure"]), []).append(r)
    for (p, s), rs in sorted(groups.items()):
        print(f"   {p:<10} {s:<5} {len(rs):>6} {sum(x['index_policy'] == 'embed' for x in rs):>6} "
              f"{statistics.median(x['token_count'] for x in rs):>6} {statistics.median(x['message_count'] for x in rs):>5}")

    refs, = conn.execute("SELECT COUNT(*) FROM ChunkMessages").fetchone()
    dup = refs / max(chunked, 1) - 1
    print(f"Duplication (extra message references from overlap): {dup:.0%}")
    if dup > 0.4:
        failures.append(f"duplication {dup:.0%} > 40%")

    cross, = conn.execute("""
        SELECT COUNT(*) FROM (SELECT cm.msg_id FROM ChunkMessages cm JOIN Chunks c USING(chunk_id)
        GROUP BY cm.msg_id HAVING COUNT(DISTINCT c.session_id) > 1)""").fetchone()
    print(f"Messages appearing in more than one session: {cross}")
    if cross:
        failures.append(f"{cross} messages span sessions")

    me_share, = conn.execute("SELECT AVG(ego_weight) FROM Chunks WHERE index_policy='embed'").fetchone()
    print(f"Mean ego_weight (embed chunks): {me_share:.2f}")

    print("\nFAIL:\n  " + "\n  ".join(failures) if failures else "\nAll hard checks passed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=300)
    main(ap.parse_args().target)
