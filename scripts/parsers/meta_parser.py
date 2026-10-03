import os
import json
import zipfile
import logging
import sys
import re
import unicodedata

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Inject utils path for database import
sys.path.append(os.path.join(REPO_ROOT, "scripts", "utils"))
from database import SarthinkMemoryLayer, same_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ─── Encoding fix ─────────────────────────────────────────────────────────────

def decode_meta_string(s):
    if not isinstance(s, str):
        return s
    try:
        # Meta GDPR exports are double-encoded: bytes were read as latin-1 instead of utf-8
        return s.encode('latin1').decode('utf-8')
    except Exception:
        return s

# ─── Identity loading ─────────────────────────────────────────────────────────

def load_meta_ego_aliases():
    """
    Load the owner's known display names for each Meta platform from identity_map.json.
    Returns ({ platform: set_of_name_strings }, master_persona)
    Used to detect when a message sender is the archive owner, without hardcoding.
    """
    map_path = os.path.join(REPO_ROOT, "config", "identity_map.json")
    aliases = {}
    master_persona = "User"
    if os.path.exists(map_path):
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            master_persona = data.get("master_persona", "User")
            for platform, names in data.get("aliases", {}).items():
                aliases[platform] = {n.lower() for n in names if n}
        except Exception as e:
            logging.warning(f"Could not load identity_map.json: {e}")
    return aliases, master_persona

# ─── Participant ID extraction ────────────────────────────────────────────────

def extract_participant_id(thread_path: str, sender_name: str, platform: str) -> str:
    """
    Extract a stable, unique participant ID from the thread_path field.

    Meta GDPR exports embed a stable numeric suffix in thread_path:
      Instagram:  inbox/amodkumargupta_1400046008060667   → "1400046008060667"
      Facebook:   inbox/2073942549573572                  → "2073942549573572"
                  inbox/abhisheksingh_1705810169715912    → "1705810169715912"

    For 1:1 threads, this number is the OTHER participant's stable Meta ID,
    which remains consistent across archive re-exports (unlike display names
    which can change).

    Fallback: if no numeric suffix is found, use a slug derived from the
    sender's display name (same as old behaviour, scoped to platform).
    """
    if thread_path:
        folder = thread_path.split('/')[-1]  # e.g. "amodkumargupta_1400046008060667"

        # Case 1: pure numeric folder (Facebook group chats / some 1:1)
        if folder.isdigit():
            return f"{platform}_{folder}"

        # Case 2: slug_NUMERICID pattern (most Instagram + some Facebook)
        match = re.search(r'_(\d{10,})$', folder)
        if match:
            return f"{platform}_{match.group(1)}"

    # Standardize Unicode (e.g. 𝓐𝓻𝔂𝓪𝓷 -> Aryan)
    sender_name = unicodedata.normalize('NFKC', decode_meta_string(sender_name))
    
    # Fallback: derive from display name (no stable ID available)
    slug = sender_name.replace(' ', '_').lower()
    slug = re.sub(r'[^\w]', '', slug)  # strip non-word chars (preserves non-Latin letters)
    return f"{platform}_user_{slug}" if slug else f"{platform}_user_unknown"




def build_participant_id_map(thread_path: str, participants: list, platform: str,
                              ego_names: set) -> dict[str, str]:
    """
    Build { sender_name_lower -> stable_raw_id } for all participants in a thread.

    For 1:1 DMs (2 participants), we know the thread_path ID belongs to the
    non-ego participant. For group chats (3+ participants), we can't map the
    single thread_path number to a specific person, so we fall back to
    name-based IDs for non-ego participants.
    """
    id_map = {}
    non_ego = [p for p in participants if p.get('name', '').lower() not in ego_names]

    if len(non_ego) == 1 and thread_path:
        # 1:1 DM — thread_path ID is unambiguously the other person
        thread_id = extract_participant_id(thread_path, non_ego[0].get('name', ''), platform)
        id_map[non_ego[0]['name'].lower()] = thread_id
    else:
        # Group chat — use name-based fallback for everyone (best effort)
        for p in non_ego:
            name = unicodedata.normalize('NFKC', decode_meta_string(p.get('name', '')))
            slug = name.replace(' ', '_').lower()
            slug = re.sub(r'[^\w]', '', slug)
            id_map[name.lower()] = f"{platform}_user_{slug}" if slug else f"{platform}_user_unknown"


    return id_map

# ─── Main parser ─────────────────────────────────────────────────────────────

PLACEHOLDER_TITLE = re.compile(r'^\s*(instagram|facebook) user\s*$|^\s*$', re.I)

# a conversation's messages: message_1.json, message_2.json… (other .json under a chat folder are attachments)
MSG_FILE = re.compile(r'messages/inbox/[^/]+/message_\d+\.json$')


def profile_ego_id(zip_path):
    """Your account id (User ID, else email) from an export's profile file, or None: a large export comes split
    into several zips and only one of them carries the profile."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        for pf in [f for f in z.namelist() if 'personal_information.json' in f or 'profile_information.json' in f]:
            try:
                data = json.loads(z.read(pf))
                users = data.get('profile_user', []) or data.get('profile_info', [])
                if users and isinstance(users, list):
                    info = users[0].get('string_map_data', {})
                    found = info.get('User ID', {}).get('value') or info.get('Email', {}).get('value')
                    if found:
                        return str(found)
            except Exception:
                pass
    return None


def process_meta_zip(db, zip_path, platform, counters, ego_aliases, master_persona="User", ego_hint=None):
    logging.info(f"Scanning {platform} archive: {zip_path}")

    # Ego names for this platform (lowercased set for fast membership test)
    ego_names: set = ego_aliases.get(platform, set())

    db.begin_source()   # repeats are matched against the DB per export (see stored_copy)
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()

        # ── Identify Ego ──────────────────────────────────────────────────────
        # your id from this zip's profile, else from another part of the same export, else a synthetic one
        ego_id   = profile_ego_id(zip_path) or ego_hint or f"{platform}_ego_user"
        ego_name = master_persona
        logging.info(f"{platform} ego ID: {ego_id}")

        # ── Parse DMs ─────────────────────────────────────────────────────────
        msg_files = [f for f in file_list if MSG_FILE.search(f)]
        logging.info(f"Found {len(msg_files)} DM files for {platform}.")

        for mf in msg_files:
            try:
                data        = json.loads(z.read(mf))
                thread_path = data.get('thread_path', '')
                participants = data.get('participants', [])

                # convo_id from thread_path for stable DB key
                convo_id    = thread_path.split('/')[-1] if thread_path else mf.split('messages/inbox/')[1].split('/')[0]
                thread_title = decode_meta_string(data.get('title', f"DM {convo_id}"))
                # a renamed account renames the chat folder (<username>_<number>): same number, same chat.
                # Keep the stored key (ids hang off it) and take the newer title
                num = re.search(r'(\d{6,})$', convo_id)
                old = db.thread_with_suffix(platform, num.group(1)) if num else None
                if old and old[1] != convo_id:
                    convo_id = old[1]
                    # a deactivated account turns a real name into "Instagram user": never trade down
                    if old[2] != thread_title and not PLACEHOLDER_TITLE.match(thread_title):
                        db.cursor.execute("UPDATE Threads SET title = ? WHERE id = ?", (thread_title, old[0]))
                thread_db_id = db.get_or_create_thread(platform, convo_id, thread_title)

                # Build participant → stable_id map for this thread
                participant_id_map = build_participant_id_map(
                    thread_path, participants, platform, ego_names
                )

                messages = data.get('messages', [])
                messages_sorted = sorted(messages, key=lambda x: x.get('timestamp_ms', 0))

                previous_msg_id = None
                for idx, msg in enumerate(messages_sorted):
                    ts_ms     = msg.get('timestamp_ms', 0)
                    utc_epoch = ts_ms // 1000 if ts_ms > 0 else None

                    content = decode_meta_string(msg.get('content', ''))

                    # Shared links / stickers come through 'share' instead of 'content'
                    if not content and 'share' in msg:
                        share = msg['share']
                        link  = share.get('link', '')
                        text  = decode_meta_string(share.get('share_text', ''))
                        content = f"[Attachment Shared] {text} {link}".strip() or "[Attachment Shared]"

                    if not content:
                        continue

                    # Already stored from an earlier export? The positional id below differs between
                    # exports, so match by content and keep the reply chain pointing at the stored row.
                    stored = db.stored_copy(thread_db_id, utc_epoch, content)
                    if stored:
                        previous_msg_id = stored
                        continue

                    sender_raw = decode_meta_string(msg.get('sender_name', ''))
                    is_ego     = sender_raw.lower() in ego_names or sender_raw == ego_name

                    if is_ego:
                        author_id      = ego_id
                        author_display = ego_name
                    else:
                        # Use thread_path-derived stable ID when available,
                        # fall back to name slug for group chats.
                        # For empty sender_name (deactivated accounts), use
                        # the thread_path ID directly so raw_id is never blank.
                        if not sender_raw:
                            author_id      = extract_participant_id(thread_path, '', platform)
                            author_display = "[Deleted Account]"
                        else:
                            author_id      = participant_id_map.get(
                                sender_raw.lower(),
                                extract_participant_id(None, sender_raw, platform)
                            )
                            author_display = sender_raw

                    author_db_id = db.get_or_create_user(platform, author_id, author_display)

                    raw_id    = f"{convo_id}_{ts_ms}_{idx}"
                    global_id = db.free_msg_id(f"{platform}_{raw_id}")

                    flat_entry = {
                        "log_id":        global_id,
                        "platform":      f"{platform}_dm",
                        "thread_name":   thread_title,
                        "author":        author_display,
                        "author_id":     author_id,
                        "timestamp_utc": utc_epoch,
                        "content":       content,
                        "is_reply_to":   previous_msg_id,
                    }

                    db.insert_message(
                        global_id, thread_db_id, author_db_id, utc_epoch,
                        content, previous_msg_id, flat_entry, 'meta_logs.jsonl',
                        commit_now=False
                    )
                    counters['msgs'] += 1
                    if counters['msgs'] % 1000 == 0:
                        db.commit()
                    previous_msg_id = global_id

            except Exception as e:
                logging.error(f"Error parsing DM file {mf}: {e}")

        # ── Parse Comments ────────────────────────────────────────────────────
        comment_files = [
            f for f in file_list
            if ('comments/' in f or 'comments_and_reactions/' in f or 'comments_v2/' in f)
            and f.endswith('.json')
        ]

        # your comments already stored, by (second, text), whichever post they were filed under: newer exports
        # can drop the post owner's name ("unknown_media"), which would file a stored comment under a new post
        stored_comments = set()
        for ts_, c_ in db.cursor.execute(
                "SELECT m.timestamp_utc, m.content FROM Messages m JOIN Threads t ON t.id = m.thread_id "
                "WHERE t.platform = ? AND t.title LIKE 'Comment on %'", (platform,)):
            stored_comments.add((ts_, same_text(c_)))

        for cf in comment_files:
            try:
                data  = json.loads(z.read(cf))
                items = data if isinstance(data, list) else data.get('comments_v2', [])

                for idx, c in enumerate(items):
                    if isinstance(c, dict) and 'string_map_data' in c:
                        # Instagram structure
                        smd     = c.get('string_map_data', {})
                        content = decode_meta_string(smd.get('Comment', {}).get('value', ''))
                        ts      = smd.get('Time', {}).get('timestamp', 0)
                        owner   = decode_meta_string(smd.get('Media Owner', {}).get('value', 'unknown_media'))

                    elif isinstance(c, dict) and 'data' in c:
                        # Facebook structure
                        d_arr   = c.get('data', [])
                        if not d_arr:
                            continue
                        comp    = d_arr[0].get('comment', {})
                        content = decode_meta_string(comp.get('comment', ''))
                        ts      = comp.get('timestamp', 0)
                        owner   = decode_meta_string(c.get('title', 'Facebook Post'))
                    else:
                        continue

                    if not content:
                        continue

                    if (ts or None, same_text(content)) in stored_comments:
                        continue
                    thread_db_id = db.get_or_create_thread(platform, owner, f"Comment on {owner[:40]}")
                    if db.stored_copy(thread_db_id, ts, content):
                        continue
                    raw_id       = f"comment_{owner}_{ts}_{idx}"
                    global_id    = db.free_msg_id(f"{platform}_{raw_id}")

                    author_db_id = db.get_or_create_user(platform, ego_id, ego_name)

                    flat_entry = {
                        "log_id":        global_id,
                        "platform":      f"{platform}_comment",
                        "thread_name":   f"Comment on {owner[:40]}",
                        "author":        ego_name,
                        "author_id":     ego_id,
                        "timestamp_utc": ts,
                        "content":       content,
                        "is_reply_to":   None,
                    }
                    db.insert_message(
                        global_id, thread_db_id, author_db_id, ts,
                        content, None, flat_entry, 'meta_logs.jsonl',
                        commit_now=False
                    )
                    counters['msgs'] += 1

            except Exception as e:
                logging.error(f"Error parsing comment file {cf}: {e}")

        db.commit()

# ─── Entry point ──────────────────────────────────────────────────────────────

def process_all_meta(fresh=False, paths=None):
    """Add Instagram/Facebook zips to the DB: the given paths, else every one in archive/. Messages already
    stored (from this or an older export) are skipped, so a newer export can sit next to the old one: nothing
    is deleted, and messages that only the old export still has are kept."""
    # Load ego aliases from config before touching the DB
    ego_aliases, master_persona = load_meta_ego_aliases()

    db = SarthinkMemoryLayer()
    if fresh:  # wipe and reparse (renumbers ids the index is keyed on; see purge_platform)
        jsonl_path = os.path.join(db.jsonl_dir, 'meta_logs.jsonl')
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
            logging.info("Cleared old meta_logs.jsonl for fresh reparse.")
        for platform in ['instagram', 'facebook']:
            db.purge_platform(platform)

    counters = {'msgs': 0}

    archive_dir = os.path.join(REPO_ROOT, "archive")
    if not os.path.exists(archive_dir):
        logging.info(f"Archive directory {archive_dir} not found. Skipping meta parsing.")
        db.close()
        return

    zips = paths or [os.path.join(archive_dir, a) for a in sorted(os.listdir(archive_dir)) if a.endswith('.zip')]
    by_platform = {}
    for z in zips:
        name = os.path.basename(z).lower()
        plat = 'instagram' if 'instagram' in name else 'facebook' if 'facebook' in name else None
        if plat:
            by_platform.setdefault(plat, []).append(z)
    for plat, group in by_platform.items():
        # your id, from whichever part of the export has the profile (the others are media/other chats)
        ego = next((e for e in map(profile_ego_id, group) if e), None)
        for z in group:
            process_meta_zip(db, z, plat, counters, ego_aliases, master_persona, ego_hint=ego)

    db.commit()
    db.close()
    logging.info(f"Meta parse complete. {db.inserted} new messages added (already-stored ones skipped).")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Add Instagram/Facebook exports from archive/ to the DB.")
    ap.add_argument("--fresh", action="store_true", help="wipe instagram+facebook first (not needed to update)")
    ap.add_argument("zips", nargs="*", help="export zips to add (default: every Instagram/Facebook zip in archive/)")
    a = ap.parse_args()
    process_all_meta(fresh=a.fresh, paths=a.zips or None)
