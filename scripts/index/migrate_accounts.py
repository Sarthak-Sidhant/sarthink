"""One-time migration of a data folder to accounts + people (docs/accounts-and-people.md).

Before: accounts were merged into "persons" (contacts folded accounts together, P_ graph nodes were accounts).
After: Accounts are the base layer (A_ nodes, never merged) and people are user-written groups of accounts (P_).

Steps (each safe to rerun):
  1. build_identity.py  accounts; migrates PersonIds -> AccountIds, contacts -> People/PersonAccounts, account notes
  2. build_chunks.py    Chunks/Sessions.account_ids (chunk and session ids and texts stay the same)
  3. inbox items        node lists P_<old person id> -> A_<account id> (old person ids are today's account
                        ids), except P_1: the old merged "me" is now person 1
  4. caches             old person-keyed profile/dossier caches are dropped (rebuilt on demand)
  5. graph              export (A_/T_ nodes) + a fresh layout

Usage: SARTHINK_DATA=<data dir> python3 scripts/index/migrate_accounts.py
"""
import os
import sqlite3
import subprocess
import sys

from common import PROFILE_CACHE, REPO_ROOT, connect_index


def run(script, *args):
    print(f"\n== {script}", flush=True)
    subprocess.run([sys.executable, str(REPO_ROOT / script), *args], check=True, env=os.environ)


def main():
    run("scripts/index/build_identity.py")
    run("scripts/index/build_chunks.py")
    conn = connect_index()
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'Commitments'").fetchone():
        # old P_<id> were accounts, except P_1 (the merged "me"), which is now the person me
        n = conn.execute("""UPDATE Commitments SET nodes = replace(replace(nodes, '"P_', '"A_'), '"A_1"', '"P_1"')
                            WHERE nodes LIKE '%"P\\_%' ESCAPE '\\'""").rowcount
        conn.commit()
        print(f"\n== inbox: {n} items now point at accounts")
    if os.path.exists(PROFILE_CACHE):
        c = sqlite3.connect(PROFILE_CACHE)
        for t in ("Profiles", "Dossiers", "AccountProfiles"):
            c.execute(f"DROP TABLE IF EXISTS {t}")
        c.commit()
        print("== caches: old person-keyed profiles dropped")
    run("scripts/utils/export_cosmograph.py")
    run("scripts/utils/compute_layout.py")
    print("\nDone. Restart the server.")


if __name__ == "__main__":
    main()
