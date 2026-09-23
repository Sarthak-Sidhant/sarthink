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
from database import SarthinkMemoryLayer

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

def process_meta_zip(db, zip_path, platform, counters, ego_aliases, master_persona="User"):
    logging.info(f"Scanning {platform} archive: {zip_path}")

    # Ego names for this platform (lowercased set for fast membership test)
    ego_names: set = ego_aliases.get(platform, set())

    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()

        # ── Identify Ego ──────────────────────────────────────────────────────
        # Start with a synthetic fallback; try to resolve a real ID from profile files.
        ego_id   = f"{platform}_ego_user"
        ego_name = master_persona

        profile_files = [
            f for f in file_list
            if 'personal_information.json' in f or 'profile_information.json' in f
        ]
        for pf in profile_files:
            try:
                profile_data = json.loads(z.read(pf))
                user_list = (
                    profile_data.get('profile_user', []) or
                    profile_data.get('profile_info', [])
                )
                if user_list and isinstance(user_list, list):
                    info = user_list[0].get('string_map_data', {})
                    potential_id = (
                        info.get('User ID', {}).get('value') or
                        info.get('Email', {}).get('value')
                    )
                    if potential_id:
                        ego_id = str(potential_id)
                        logging.info(f"Resolved {platform} ego ID from profile: {ego_id}")
                        break
            except Exception:
                pass

        # ── Parse DMs ─────────────────────────────────────────────────────────
        msg_files = [
            f for f in file_list
            if 'messages/inbox/' in f and f.endswith('.json')
        ]
        logging.info(f"Found {len(msg_files)} DM files for {platform}.")

        for mf in msg_files:
            try:
                data        = json.loads(z.read(mf))
                thread_path = data.get('thread_path', '')
                participants = data.get('participants', [])

                # convo_id from thread_path for stable DB key
                convo_id    = thread_path.split('/')[-1] if thread_path else mf.split('messages/inbox/')[1].split('/')[0]
                thread_title = decode_meta_string(data.get('title', f"DM {convo_id}"))
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
                    global_id = f"{platform}_{raw_id}"

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

                    thread_db_id = db.get_or_create_thread(platform, owner, f"Comment on {owner[:40]}")
                    raw_id       = f"comment_{owner}_{ts}_{idx}"
                    global_id    = f"{platform}_{raw_id}"

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

def process_all_meta():
    # Load ego aliases from config before touching the DB
    ego_aliases, master_persona = load_meta_ego_aliases()

    # Fresh reparse: clear old meta platform data
    db = SarthinkMemoryLayer()
    for platform in ['instagram', 'facebook']:
        jsonl_path = os.path.join(REPO_ROOT, 'processed_data', 'logs', 'meta_logs.jsonl')
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
            logging.info("Cleared old meta_logs.jsonl for fresh reparse.")
        db.purge_platform(platform)
        break  # only delete JSONL once

    counters = {'msgs': 0}

    archive_dir = os.path.join(REPO_ROOT, "archive")
    if not os.path.exists(archive_dir):
        logging.info(f"Archive directory {archive_dir} not found. Skipping meta parsing.")
        db.close()
        return

    for arc in sorted(os.listdir(archive_dir)):
        if not arc.endswith('.zip'):
            continue
        arc_path = os.path.join(archive_dir, arc)
        if 'instagram' in arc.lower():
            process_meta_zip(db, arc_path, 'instagram', counters, ego_aliases, master_persona)
        elif 'facebook' in arc.lower():
            process_meta_zip(db, arc_path, 'facebook', counters, ego_aliases, master_persona)

    db.commit()
    db.close()
    logging.info(f"Meta parse complete. Ingested {counters['msgs']} messages.")

if __name__ == "__main__":
    process_all_meta()
