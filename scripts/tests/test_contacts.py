"""Contacts: linking accounts (incl. a deleted account's chat) into one person, and unlinking them again.

Runs on a throwaway copy of the demo archive; touches no real data.
Usage: python3 scripts/tests/test_contacts.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECK = r"""
import sys
sys.path.insert(0, "scripts/index")
import build_identity
from common import connect_index
from contacts import ContactStore, accounts_of, refresh_person_lists

conn = connect_index()
persons = lambda: {r[0]: (r[1], r[2], r[3]) for r in conn.execute(
    "SELECT person_id, name, placeholder, message_count FROM Persons")}
build_identity.build(verbose=False)
base = persons()
assert base[2][0] == "Rohan", base[2]                       # person_merges.json was imported as a contact
store = ContactStore()
assert store.links()[("discord", "rohan.dev", "")] == 2

# twitter rohan_dev (54) and the deactivated Instagram DM (10) are Rohan
deleted = accounts_of(conn, 10)[0]["key"]
assert deleted[2], deleted                                   # a deleted account is linked per chat
store.link(2, 54, [a["key"] for a in accounts_of(conn, 2)], [a["key"] for a in accounts_of(conn, 54)])
store.link(2, 10, [a["key"] for a in accounts_of(conn, 2)], [deleted])
build_identity.build(verbose=False)
now = persons()
assert 54 not in now and 10 not in now, "linked people should be folded into the contact"
assert now[2] == ("Rohan", 0, base[2][2] + base[54][2] + base[10][2]), now[2]
assert refresh_person_lists(conn) > 0
lists = [set(eval(r[0])) for r in conn.execute("SELECT person_ids FROM Chunks")]
assert not any({10, 54} & s for s in lists), "chunks still point at folded people"

# unlinking gives each account its own person id back
store.unlink(("twitter", "rohan_dev", ""))
store.unlink(tuple(deleted))
build_identity.build(verbose=False)
assert persons() == base, "unlinking should restore the original people"
refresh_person_lists(conn)
print("ok")
"""


def main():
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "demo")
        shutil.copytree(os.path.join(REPO_ROOT, "demo"), data, ignore=shutil.ignore_patterns("index", "*.db-*"))
        for f in os.listdir(os.path.join(data, "db")):
            if f == "user_notes.db":
                os.remove(os.path.join(data, "db", f))
        env = dict(os.environ, SARTHINK_DATA=data)
        r = subprocess.run([sys.executable, "-c", CHECK], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        print(r.stdout + r.stderr)
        assert r.returncode == 0 and "ok" in r.stdout
    print("contacts: link and unlink round-trip OK")


if __name__ == "__main__":
    main()
