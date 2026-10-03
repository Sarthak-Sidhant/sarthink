"""WhatsApp (Android) parser: a decrypted msgstore.db -> the shared SQLite memory DB.

Input: archive/whatsapp-msgstore.db (decrypt msgstore.db.crypt15 with wa-crypt-tools: `wadecrypt <key> ...`).
Optional names: a phone-contacts export (.vcf) via --contacts, or archive/whatsapp-contacts*.vcf. WhatsApp's
backups carry no contact names (wa.db's contact table is left out), so without it people are named by their
WhatsApp profile name where the backup has one, else by a masked number. Rerun with contacts later: names and
chat titles are updated in place, nothing is duplicated.

Stable across backups, so a newer msgstore appends cleanly:
  person   raw_id = phone number (LID accounts are mapped to their number via jid_map), else "lid:<id>"
  chat     platform_thread_id = "dm:<phone>" or "group:<group id>"
  message  msg_id = whatsapp_<chat>_<key_id> (WhatsApp's own message id)

Kept: text, captions of photos/videos/documents/GIFs, contact cards, polls. Skipped: system notices, deleted
messages, media without a caption, channels, status updates and broadcast lists. A quoted reply links to the
quoted message when it is in the same chat.

Usage: python3 scripts/parsers/whatsapp_parser.py [--db archive/whatsapp-msgstore.db] [--contacts file.vcf]
"""
import argparse
import glob
import json
import logging
import os
import re
import sqlite3
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "utils"))
from database import SarthinkMemoryLayer  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

PLATFORM = "whatsapp"
JSONL_OUTPUT = "whatsapp_logs.jsonl"
ME = "whatsapp_me"
SKIP_SERVERS = ("newsletter", "broadcast", "status_me", "bot", "temp")
KEPT = """{m}.message_type != 7 AND {m}.text_data IS NOT NULL AND {m}.text_data != ''
          AND NOT EXISTS (SELECT 1 FROM message_revoked r WHERE r.message_row_id = {m}._id)"""
LABEL = {1: "[Photo]", 3: "[Video]", 9: "[Document]", 13: "[GIF]", 4: "[Contact card]", 66: "[Poll]"}


def load_contacts(path):
    """Phone contacts export -> {last 10 digits: name}."""
    names, name = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN:VCARD"):
                name = None
            elif line.startswith("FN:"):   # vCard escapes: \, \; \\ \n
                name = re.sub(r"\\(.)", lambda m: " " if m.group(1) in "nN" else m.group(1), line[3:]).strip()
            elif line.upper().startswith("TEL") and name:
                digits = re.sub(r"\D", "", line.split(":", 1)[-1])
                if len(digits) >= 10:
                    names[digits[-10:]] = name
    return names


def masked(phone):
    """+91 98··· ··210: recognisable to you, not a usable number."""
    if len(phone) < 8:
        return "WhatsApp user"
    cc, rest = phone[:-10] or "", phone[-10:]
    return f"+{cc} {rest[:2]}··· ··{rest[-3:]}".replace("+ ", "+")


class People:
    """Resolve WhatsApp jids to a stable account id and the best available name."""

    def __init__(self, wa, contacts):
        self.contacts = contacts
        self.jid = {r[0]: (r[1] or "", r[2]) for r in wa.execute("SELECT _id, user, server FROM jid")}
        self.phone_of_lid = {lid: self.jid[pj][0] for lid, pj in wa.execute("SELECT lid_row_id, jid_row_id FROM jid_map")
                             if pj in self.jid and self.jid[pj][1] == "s.whatsapp.net"}
        self.lid_of_phone = {pj: lid for lid, pj in wa.execute("SELECT lid_row_id, jid_row_id FROM jid_map")}
        self.pushname = {r[0]: r[1] for r in wa.execute(
            "SELECT lid_row_id, display_name FROM lid_display_name WHERE display_name != ''")}

    def account(self, jid_row):
        """(raw_id, phone or None) for a person jid."""
        user, server = self.jid.get(jid_row, ("", ""))
        if server == "s.whatsapp.net":
            return re.sub(r"\D", "", user), re.sub(r"\D", "", user)
        phone = self.phone_of_lid.get(jid_row)
        if phone:
            return re.sub(r"\D", "", phone), re.sub(r"\D", "", phone)
        return f"lid:{user}", None

    def name(self, jid_row):
        raw_id, phone = self.account(jid_row)
        if phone and phone[-10:] in self.contacts:
            return self.contacts[phone[-10:]]
        lid = jid_row if self.jid.get(jid_row, ("", ""))[1] == "lid" else self.lid_of_phone.get(jid_row)
        if lid in self.pushname:
            return self.pushname[lid]
        return masked(phone) if phone else f"WhatsApp user ··{raw_id[-4:]}"


def upsert_name(db, table, key_col, key, name_col, name):
    """Newer runs may know a better name (e.g. once contacts are added): update it in place."""
    db.cursor.execute(f"UPDATE {table} SET {name_col} = ? WHERE platform = ? AND {key_col} = ? AND "
                      f"COALESCE({name_col}, '') != ?", (name, PLATFORM, key, name))
    return db.cursor.rowcount


def process_whatsapp(db_path, contacts_path=None):
    wa = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    contacts = load_contacts(contacts_path) if contacts_path else {}
    logging.info(f"Reading {db_path}" + (f" with {len(contacts)} contact numbers from {contacts_path}" if contacts else
                                          " (no contacts file: profile names or masked numbers)"))
    people = People(wa, contacts)
    db = SarthinkMemoryLayer()

    me_id = db.get_or_create_user(PLATFORM, ME, master_persona())

    # chats worth keeping: 1:1 and groups that have at least one kept message (WhatsApp also keeps a row
    # for every account seen in a group, most with no messages)
    chats = {}
    for cid, jid_row, subject in wa.execute(f"""SELECT _id, jid_row_id, subject FROM chat c
            WHERE EXISTS (SELECT 1 FROM message m WHERE m.chat_row_id = c._id AND {KEPT.format(m="m")})"""):
        user, server = people.jid.get(jid_row, ("", ""))
        if server in SKIP_SERVERS or not user:
            continue
        if server == "g.us":
            key, title = f"group:{user}", subject or "WhatsApp group"
        else:
            raw_id, _ = people.account(jid_row)
            key, title = f"dm:{raw_id}", people.name(jid_row)
        tid = db.get_or_create_thread(PLATFORM, key, title)
        renamed = upsert_name(db, "Threads", "platform_thread_id", key, "title", title)
        chats[cid] = (tid, key, server == "g.us", jid_row, renamed)
    logging.info(f"{len(chats)} chats ({sum(c[2] for c in chats.values())} groups); "
                 f"{sum(c[4] for c in chats.values())} chat titles updated")

    # (chat, key_id) pairs that are not unique get the sender appended, so ids never collide
    dup_keys = {(r[0], r[1]) for r in wa.execute(
        "SELECT chat_row_id, key_id FROM message GROUP BY chat_row_id, key_id HAVING COUNT(*) > 1")}

    def msg_id(chat_row, key_id, from_me, sender):
        base = f"{PLATFORM}_{chats[chat_row][1]}_{key_id}"
        return f"{base}_{'me' if from_me else sender}" if (chat_row, key_id) in dup_keys else base

    users, renamed_users = {}, 0

    def author(jid_row):
        nonlocal renamed_users
        if jid_row not in users:
            raw_id, _ = people.account(jid_row)
            name = people.name(jid_row)
            users[jid_row] = db.get_or_create_user(PLATFORM, raw_id, name)
            renamed_users += upsert_name(db, "Users", "raw_id", raw_id, "display_name", name)
        return users[jid_row]

    # a quote links to its parent only if the parent is in the same chat and is kept too
    rows = wa.execute(f"""
        SELECT m._id, m.chat_row_id, m.from_me, m.key_id, m.sender_jid_row_id, m.timestamp, m.message_type,
               m.text_data, p.chat_row_id, p.key_id, p.from_me, p.sender_jid_row_id
        FROM message m
        LEFT JOIN message_quoted q ON q.message_row_id = m._id
        LEFT JOIN message p ON p.chat_row_id = q.parent_message_chat_row_id AND p.from_me = q.from_me
                           AND p.key_id = q.key_id AND q.parent_message_chat_row_id = m.chat_row_id
                           AND {KEPT.format(m="p")}
        WHERE {KEPT.format(m="m")}
        ORDER BY m.chat_row_id, m.sort_id""")
    seen = skipped_chat = no_sender = 0
    for (_id, chat_row, from_me, key_id, sender, ts_ms, mtype, text,
         p_chat, p_key, p_from_me, p_sender) in rows:
        if chat_row not in chats:
            skipped_chat += 1
            continue
        tid, key, is_group, chat_jid, _ = chats[chat_row]
        if from_me:
            author_id = me_id
        elif is_group:          # in groups the sender column says who; without it the author is unknown
            if not sender:
                no_sender += 1
                continue
            author_id = author(sender)
        else:                   # in a 1:1 chat the sender is the chat partner
            author_id = author(chat_jid)
        content = f"{LABEL[mtype]} {text}" if mtype in LABEL else text
        mid = msg_id(chat_row, key_id, from_me, sender)
        parent = msg_id(p_chat, p_key, p_from_me, p_sender) if p_key and p_chat in chats else None
        ts = ts_ms // 1000 if ts_ms else None
        db.insert_message(mid, tid, author_id, ts, content, parent,
                          {"log_id": mid, "platform": PLATFORM, "thread_name": key, "timestamp_utc": ts,
                           "content": content, "is_reply_to": parent}, JSONL_OUTPUT, commit_now=False)
        seen += 1
        if seen % 5000 == 0:
            db.commit()
    db.commit()
    db.close()
    logging.info(f"WhatsApp parse complete. {seen} messages read, {db.inserted} new messages added "
                 f"(already-stored ones skipped); {len(users)} people, {renamed_users} renamed; "
                 f"{skipped_chat} messages in channels/status/broadcasts and {no_sender} group messages without a "
                 f"sender skipped.")


def master_persona():
    data_dir = os.environ.get("SARTHINK_DATA") or os.path.join(REPO_ROOT, "processed_data")
    for path in (os.path.join(REPO_ROOT, data_dir, "identity_map.json"), os.path.join(REPO_ROOT, "config", "identity_map.json")):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("master_persona", "Me")
    return "Me"


def default_contacts():
    hits = sorted(glob.glob(os.path.join(REPO_ROOT, "archive", "whatsapp-contacts*.vcf")))
    return hits[-1] if hits else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Add a decrypted WhatsApp msgstore.db to the DB.")
    ap.add_argument("--db", default=os.path.join(REPO_ROOT, "archive", "whatsapp-msgstore.db"))
    ap.add_argument("--contacts", default=None, help="phone contacts export (.vcf) for names")
    a = ap.parse_args()
    process_whatsapp(a.db, a.contacts or default_contacts())
