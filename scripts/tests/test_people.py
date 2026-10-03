"""Accounts and people (docs/accounts-and-people.md): people group accounts without ever changing them.

On a throwaway copy of the demo archive (touches no real data):
- person_merges.json becomes people; my accounts belong to person 1
- attach / detach / create / rename / delete only write the people tables: accounts are untouched
- rebuilding keeps every account id and every person exactly as they were

Usage: python3 scripts/tests/test_people.py
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
from common import USER_DB, connect_index
from people import ME, PeopleStore

conn = connect_index()
accounts = lambda: {r[0]: tuple(r[1:]) for r in conn.execute(
    "SELECT account_id, name, platform, is_me, placeholder, raw_id, chat, message_count FROM Accounts")}
build_identity.build(verbose=False)
store = PeopleStore(USER_DB)
base_accounts, base_people, base_members = accounts(), store.people(), store.membership()
names = {v: k for k, v in base_people.items()}
assert {"Rohan", "Kabir", "Meera", "Dev"} <= set(names), base_people          # person_merges.json -> people
mine = {k for k, p in base_members.items() if p == ME}
assert mine and all(base_accounts[a][2] for a in [r[0] for r in conn.execute(
    "SELECT account_id FROM Accounts WHERE is_me = 1")]), "my accounts belong to me"
assert len(mine) == conn.execute("SELECT COUNT(*) FROM Accounts WHERE is_me = 1").fetchone()[0]

# an account in no person -> a new person; attach a second one; move it; detach; rename; delete
free = [(r[0], (r[1], r[2], r[3])) for r in conn.execute(
    "SELECT account_id, platform, raw_id, chat FROM Accounts WHERE is_me = 0 ORDER BY message_count DESC")
    if (r[1], r[2], r[3]) not in base_members][:2]
(a1, k1), (a2, k2) = free
pid = store.create("Test Person", [k1])
assert store.person_of(k1) == pid and store.members(pid) == [k1]
store.attach(pid, k2)
assert sorted(store.members(pid)) == sorted([k1, k2])
store.attach(names["Rohan"], k2)                                  # moving it to another person
assert store.person_of(k2) == names["Rohan"] and store.members(pid) == [k1]
store.detach(k2)
assert store.person_of(k2) is None
store.rename(pid, "Renamed")
assert store.people()[pid] == "Renamed"
store.delete(pid)
assert store.person_of(k1) is None and pid not in store.people()
try:
    store.delete(ME); raise SystemExit("deleting me must fail")
except ValueError:
    pass
assert accounts() == base_accounts, "people changes must never touch accounts"

# a rebuild keeps account ids and people as they were
build_identity.build(verbose=False)
assert accounts() == base_accounts, "rebuild changed accounts"
store2 = PeopleStore(USER_DB)
assert store2.people() == base_people and store2.membership() == base_members, "rebuild changed people"
print("ok")
"""


def main():
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "demo")
        shutil.copytree(os.path.join(REPO_ROOT, "demo"), data, ignore=shutil.ignore_patterns("index", "*.db-*"))
        for f in ("user_notes.db",):
            p = os.path.join(data, "db", f)
            if os.path.exists(p):
                os.remove(p)
        env = dict(os.environ, SARTHINK_DATA=data)
        r = subprocess.run([sys.executable, "-c", CHECK], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        print(r.stdout + r.stderr)
        assert r.returncode == 0 and "ok" in r.stdout
    print("people: group, move, detach, rename, delete; accounts untouched; rebuild keeps everything OK")


if __name__ == "__main__":
    main()
