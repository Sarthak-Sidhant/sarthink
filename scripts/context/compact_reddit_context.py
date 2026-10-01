"""Shrink fetched Reddit post files that repeat comments, keeping every unique comment once.

An earlier version of reddit_fetch_context.py saved each comment again under every level above it, so a
comment at depth d appears 2^d times (one post grew to 4.5 GB for 2,345 comments). This keeps the first copy
of each comment, at the position the parser already uses (it keeps the first copy too: INSERT OR IGNORE),
with all its fields. Replies found only under a later copy are attached to the kept copy, as the parser would.
A file is left untouched if any repeated copy differs from the kept one in a field.

Files are streamed (ijson), so the 4.5 GB file needs little memory. Originals are not modified: the shrunk
files go to OUT, and you swap folders yourself after checking.

Usage: python3 scripts/context/compact_reddit_context.py <posts_dir> <out_dir>
"""
import argparse
import json
import os
import sys

import ijson


def compact_thread(f):
    """Stream conversation_thread items; return (my_post_details, kept top-level list, stats)."""
    seen = {}                     # comment_id -> kept comment dict
    stats = {"entries": 0, "kept": 0, "differing": 0, "other_keys": 0}
    post, top = None, []
    # frames: ["list", attach_list] for a comment list; ["comment", dict, key, attach_list, registered]
    #         ["map", dict, key] / ["array", list] for any other value (my_post_details and its contents)
    stack = []

    def put(value):
        """Attach a finished value to the frame on top of the stack."""
        fr = stack[-1]
        if fr[0] in ("comment", "map"):
            fr[1][fr[2]] = value
        elif fr[0] == "array":
            fr[1].append(value)

    def finish_comment(fr):
        _, d, _, attach, registered = fr
        stats["entries"] += 1
        cid = d.get("comment_id")
        if registered:                       # first copy: registered when its replies began
            attach.append(d)
            return
        if cid in seen:                      # later copy: drop, but check nothing would be lost
            kept = seen[cid]
            if any(kept.get(k) != v for k, v in d.items() if k != "replies"):
                stats["differing"] += 1
            return
        d.setdefault("replies", [])
        seen[cid] = d
        stats["kept"] += 1
        attach.append(d)

    path = []                                # top-level key we are under
    for event, value in ijson.basic_parse(f, use_float=True):
        if not stack:
            if event == "map_key":
                path = [value]
                stats["other_keys"] += value not in ("my_post_details", "conversation_thread")
            elif event == "start_map" and path == ["my_post_details"]:
                post = {}
                stack.append(["map", post, None])
            elif event == "start_array" and path == ["conversation_thread"]:
                stack.append(["list", top])
            continue
        fr = stack[-1]
        if fr[0] == "list":
            if event == "start_map":
                stack.append(["comment", {}, None, fr[1], False])
            elif event == "end_array":
                stack.pop()
            continue
        if event == "map_key":
            fr[2] = value
            if fr[0] == "comment" and value == "replies":
                d = fr[1]
                cid = d.get("comment_id")
                if cid is None:
                    raise ValueError("comment without comment_id before its replies")
                if cid in seen:              # later copy: its replies go to the kept copy
                    kept = seen[cid]
                    if any(kept.get(k) != v for k, v in d.items()):
                        stats["differing"] += 1
                    target = kept["replies"]
                else:
                    d["replies"] = []
                    seen[cid] = d
                    stats["kept"] += 1
                    fr[4] = True
                    target = d["replies"]
                stack.append(["list", target])     # the replies' own start_array is ignored by the list frame
        elif event in ("start_map", "start_array"):
            stack.append(["map", {}, None] if event == "start_map" else ["array", []])
        elif event in ("end_map", "end_array"):
            stack.pop()
            if fr[0] == "comment":
                finish_comment(fr)
            elif stack:
                put(fr[1])
        else:
            put(value)
    return post, top, stats


def main(a):
    os.makedirs(a.out, exist_ok=True)
    total_in = total_out = changed = skipped = 0
    for name in sorted(os.listdir(a.posts)):
        if not name.endswith(".json"):
            continue
        src, dst = os.path.join(a.posts, name), os.path.join(a.out, name)
        size = os.path.getsize(src)
        with open(src, "rb") as f:
            post, top, st = compact_thread(f)
        total_in += size
        if st["differing"] or st["other_keys"] or st["kept"] == st["entries"]:
            # nothing repeated, copies that differ, or fields this script does not know: keep the original
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                fo.write(fi.read())
            skipped += bool(st["differing"] or st["other_keys"])
            if st["differing"] or st["other_keys"]:
                print(f"KEPT AS IS {name}: {st['differing']} repeated copies differ from the first, "
                      f"{st['other_keys']} unknown top-level keys", flush=True)
        else:
            with open(dst, "w", encoding="utf-8") as fo:
                json.dump({"my_post_details": post, "conversation_thread": top}, fo, ensure_ascii=False, indent=4)
            changed += 1
            if size > 10_000_000:
                print(f"{name}: {size / 1e6:,.0f} MB -> {os.path.getsize(dst) / 1e6:,.2f} MB "
                      f"({st['entries']:,} entries, {st['kept']:,} unique comments)", flush=True)
        total_out += os.path.getsize(dst)
    print(f"done: {changed} files shrunk, {skipped} kept as is (differing copies or unknown keys); "
          f"{total_in / 1e9:.2f} GB -> {total_out / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("posts")
    ap.add_argument("out")
    sys.exit(main(ap.parse_args()))
