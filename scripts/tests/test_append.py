"""Appending a newer Meta export must add only the new messages.

Meta message ids are positional (index inside message_N.json), and a newer export splits a chat across files
differently, so the same message gets a different id. The parser must recognise it by content instead.

Usage: python3 scripts/tests/test_append.py   (uses a throwaway DB; touches no real data)
"""
import json
import os
import sys
import tempfile
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "utils"))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "parsers"))
from database import SarthinkMemoryLayer  # noqa: E402
from meta_parser import process_meta_zip  # noqa: E402

T0 = 1_700_000_000_000  # ms
OLD = [("Alice", 0, "hey"), ("Me", 1000, "hi"), ("Alice", 2000, "ok"), ("Alice", 2000, "ok"), ("Me", 5000, "bye")]
NEW = [("Alice", 9000, "you there?"), ("Me", 2000, "ok")]   # a third "ok" in the same second is a new message


def make_zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, msgs in files.items():
            z.writestr(f"your_instagram_activity/messages/inbox/alice_123/{name}", json.dumps({
                "participants": [{"name": "Alice"}, {"name": "Me"}],
                "thread_path": "inbox/alice_123", "title": "Alice",
                "messages": [{"sender_name": s, "timestamp_ms": T0 + t, "content": c} for s, t, c in msgs]}))


def run(db, path):
    process_meta_zip(db, path, "instagram", {"msgs": 0}, {"instagram": {"me"}}, "Me")
    db.commit()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        db = SarthinkMemoryLayer(db_path=os.path.join(tmp, "m.db"), jsonl_dir=tmp)
        old_zip, new_zip = os.path.join(tmp, "instagram-old.zip"), os.path.join(tmp, "instagram-new.zip")
        make_zip(old_zip, {"message_1.json": OLD})
        # newer export: same chat plus two messages, split across two files (shifts every positional id)
        everything = sorted(OLD + NEW, key=lambda m: m[1])
        make_zip(new_zip, {"message_1.json": everything[3:], "message_2.json": everything[:3]})

        run(db, old_zip)
        assert db.inserted == 5, db.inserted
        run(db, old_zip)
        assert db.inserted == 5, f"re-importing the same export added {db.inserted - 5}"
        run(db, new_zip)
        assert db.inserted == 7, f"newer export added {db.inserted - 5} messages, expected 2"
        run(db, new_zip)
        assert db.inserted == 7, "re-importing the newer export added messages"

        rows = db.conn.execute("SELECT msg_id, content, parent_msg_id FROM Messages ORDER BY timestamp_utc, rowid").fetchall()
        assert [r[1] for r in rows].count("ok") == 3
        ids = {r[0] for r in rows}
        dangling = [r for r in rows if r[2] and r[2] not in ids]
        assert not dangling, f"reply links to messages that are not stored: {dangling}"
        print(f"ok: {len(rows)} messages, no duplicates, reply chain intact")


if __name__ == "__main__":
    main()
