import sqlite3
import argparse
import json
import statistics
import datetime
import random
import sys
import os
from pathlib import Path
from collections import defaultdict
import hashlib

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

try:
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")
    def estimate_tokens(text):
        return len(encoder.encode(text))
except ImportError:
    def estimate_tokens(text):
        return int(len(text.split()) * 1.3)

DB_PATH = REPO_ROOT / "processed_data" / "db" / "sarthink_memory.db"
IDENTITY_MAP_PATH = REPO_ROOT / "config" / "identity_map.json"
TWITTER_MAP_PATH = REPO_ROOT / "processed_data" / "metadata" / "twitter_id_map.json"
MAX_TOKENS_PER_CHUNK = 4000
MIN_GAP_THRESHOLD_SEC = 60 * 30      
MAX_GAP_THRESHOLD_SEC = 60 * 60 * 12 

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def load_identity_config():
    """Returns a set of all lowercase aliases and the raw map."""
    try:
        with open(IDENTITY_MAP_PATH, 'r') as f:
            data = json.load(f)
            aliases = []
            for platform in data.get('aliases', {}):
                aliases.extend([str(a).lower() for a in data['aliases'][platform]])
            return set(aliases), data
    except Exception as e:
        print(f"Warning: Could not load identity_map.json: {e}")
        return set(), {}

def load_twitter_map():
    try:
        if TWITTER_MAP_PATH.exists():
            with open(TWITTER_MAP_PATH, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Warning: Could not load twitter_id_map.json: {e}")
        return {}

def save_twitter_map(tw_map):
    try:
        with open(TWITTER_MAP_PATH, 'w') as f:
            json.dump(tw_map, f, indent=4)
    except Exception as e:
        print(f"Error saving twitter_id_map.json: {e}")

ADJECTIVES = ["Vocal", "Bright", "Clever", "Silent", "Brave", "Swift", "Sharp", "Quiet", "Golden", "Iron"]
ANIMALS = ["Lion", "Wolf", "Tiger", "Fox", "Eagle", "Raven", "Owl", "Panda", "Deere", "Hawk"]

def generate_natural_alias(raw_id):
    """Deterministic alias generation based on ID."""
    seed = int(hashlib.md5(str(raw_id).encode()).hexdigest(), 16)
    rng = random.Random(seed)
    adj = rng.choice(ADJECTIVES)
    ani = rng.choice(ANIMALS)
    return f"@@{adj}{ani}"

def generate_chunk_id(platform, channel_id, start_timestamp, end_timestamp, chunk_index):
    raw = f"{platform}_{channel_id}_{start_timestamp}_{end_timestamp}_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]

def compute_chunk_density(window_bursts):
    all_msgs = []
    for burst in window_bursts:
        for m, f, a in burst:
            all_msgs.append((m, f, a))
            
    if not all_msgs:
        return 0.0

    unique_participants = len(set(m['display_name'] or m['raw_id'] or "" for m, f, a in all_msgs))
    avg_message_length = statistics.mean([len(f.split()) for m, f, a in all_msgs])
    has_links = any('http' in f for m, f, a in all_msgs)
    
    score = (
        min(avg_message_length / 20.0, 1.0) * 0.5 + 
        min(unique_participants / 3.0, 1.0) * 0.3 + 
        (0.2 if has_links else 0)
    )
    return round(score, 3)

def compute_ego_weight(window_bursts, identity_set):
    your_tokens = 0
    total_tokens = 0
    
    for burst in window_bursts:
        for m, f, a in burst:
            tokens = estimate_tokens(f)
            total_tokens += tokens
            
            author = str(m['display_name'] or "").lower()
            author_id = str(m['raw_id'] or "").lower()
            
            # Robust identity check
            is_me = (author in identity_set or 
                    author_id in identity_set or 
                    any(h in author for h in identity_set if len(h) > 3))
            
            if is_me:
                your_tokens += tokens
                
    if total_tokens == 0:
        return 0.0
    return round(your_tokens / total_tokens, 3)

def compute_gap_threshold(timestamps):
    if len(timestamps) < 2:
        return MIN_GAP_THRESHOLD_SEC
    
    deltas = []
    for i in range(1, len(timestamps)):
        deltas.append(timestamps[i] - timestamps[i-1])
        
    if not deltas:
        return MIN_GAP_THRESHOLD_SEC
        
    mean_delta = statistics.mean(deltas)
    if len(deltas) > 1:
        std_delta = statistics.stdev(deltas)
    else:
        std_delta = 0
        
    threshold = mean_delta + std_delta
    return max(MIN_GAP_THRESHOLD_SEC, min(threshold, MAX_GAP_THRESHOLD_SEC))

def format_message(msg_row, platform, author_name, is_dm=False):
    timestamp_str = datetime.datetime.fromtimestamp(msg_row['timestamp_utc'], datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    content = msg_row['content'] or ""
    
    if platform in ('twitter', 'reddit') and not is_dm:
        is_root = msg_row['parent_msg_id'] is None
        if platform == 'twitter':
            prefix = "[TWEET]" if is_root else "[REPLY]"
            identifier = f"@{author_name}"
            return f"{prefix} {identifier} — {timestamp_str}\n{content}\n"
        elif platform == 'reddit':
            prefix = "[POST]" if is_root else "[REPLY/COMMENT]"
            identifier = f"u/{author_name}"
            return f"{prefix} {identifier} — {timestamp_str}\n{content}\n"
    else:
        return f"[{timestamp_str}] {author_name}: {content}"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_stitched_sessions(cursor):
    """Groups thread_id's based on cross-thread parent_msg_id links."""
    print("Stitching fragmented threads...")
    
    cursor.execute("""
        SELECT DISTINCT m1.thread_id as t1, m2.thread_id as t2
        FROM Messages m1
        JOIN Messages m2 ON m1.parent_msg_id = m2.msg_id
        WHERE m1.thread_id != m2.thread_id
    """)
    links = cursor.fetchall()

    parent = {}
    def find(i):
        if i not in parent: parent[i] = i
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for row in links:
        union(row['t1'], row['t2'])

    cursor.execute("SELECT id, platform, title, platform_thread_id FROM Threads")
    threads = cursor.fetchall()
    
    sessions = defaultdict(list)
    for t in threads:
        root = find(t['id'])
        sessions[root].append(t)
        
    return sessions

def extract_subreddit(titles):
    for title in titles:
        if title and 'r/' in title:
            parts = title.split('r/')
            if len(parts) > 1:
                return parts[1].split()[0].split('|')[0].strip()
    return None

def build_chunks(dry_run=False):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    identity_set, identity_raw = load_identity_config()
    master_persona = identity_raw.get("master_persona", "Me")
    twitter_aliases = identity_raw.get("aliases", {}).get("twitter", [])
    twitter_owner = twitter_aliases[0] if twitter_aliases else master_persona
    tw_map = load_twitter_map()
    map_changed = False
    
    sessions = get_stitched_sessions(cursor)
    session_chunks = []
    print(f"Processing {len(sessions)} unique conversational sessions...")
    
    for root_id, thread_list in sessions.items():
        thread_ids = [t['id'] for t in thread_list]
        platform = thread_list[0]['platform']
        titles = [t['title'] for t in thread_list if t['title']]
        primary_title = titles[0] if titles else "Untitled Session"
        subreddit = extract_subreddit(titles) if platform == 'reddit' else None

        cursor.execute(f"""
            SELECT m.msg_id, m.author_id, m.timestamp_utc, m.content, m.parent_msg_id, u.display_name, u.raw_id
            FROM Messages m
            LEFT JOIN Users u ON m.author_id = u.id
            WHERE m.thread_id IN ({','.join(['?'] * len(thread_ids))}) AND m.timestamp_utc IS NOT NULL
            ORDER BY m.timestamp_utc ASC
        """, tuple(thread_ids))
        
        messages = cursor.fetchall()
        if not messages: continue
            
        is_dm = False
        for t in thread_list:
            if platform == 'twitter' and t['title'] and t['title'].startswith('DM'):
                is_dm = True; break
            elif platform == 'reddit':
                if (t['title'] and t['title'].startswith('DM:')) or (t['platform_thread_id'] and ':reddit.com' in t['platform_thread_id']):
                    is_dm = True; break
        
        # Twitter Participation Filter & ID resolution
        if platform == 'twitter' and is_dm:
            user_participated = False
            for m in messages:
                raw_id = str(m['raw_id'] or "").lower()
                disp = str(m['display_name'] or "").lower()
                if (raw_id in identity_set or disp in identity_set):
                    user_participated = True; break
            if not user_participated: continue

        is_forum = platform in ('reddit', 'twitter') and not is_dm
        
        # Identity and Alias Resolution
        final_messages = []
        for msg in messages:
            raw_id = str(msg['raw_id'] or "")
            display_name = str(msg['display_name'] or "")
            
            author_final = display_name or raw_id or "Unknown"
            
            # Twitter ID Resolution logic
            if platform == 'twitter':
                if raw_id in identity_set or display_name.lower() in identity_set:
                    author_final = twitter_owner
                elif raw_id in tw_map:
                    author_final = tw_map[raw_id]
                elif raw_id.isnumeric() and len(raw_id) > 5:
                    alias = generate_natural_alias(raw_id)
                    tw_map[raw_id] = alias
                    author_final = alias
                    map_changed = True
                else:
                    author_final = raw_id or display_name or "Unknown"
            elif platform == 'instagram' or platform == 'facebook' or platform == 'discord':
                # Check display name against identity map
                if display_name.lower() in identity_set or raw_id.lower() in identity_set:
                    author_final = master_persona
                    
            formatted = format_message(msg, platform, author_final, is_dm=is_dm)
            final_messages.append((msg, formatted, author_final))

        bursts = []
        if not is_forum:
            timestamps = [m[0]['timestamp_utc'] for m in final_messages]
            gap_threshold = compute_gap_threshold(timestamps)
            current_burst = []
            current_burst_tokens = 0
            
            for msg_obj, formatted, author_resolved in final_messages:
                m_tokens = estimate_tokens(formatted)
                if not current_burst:
                    current_burst.append((msg_obj, formatted, author_resolved))
                    current_burst_tokens = m_tokens
                else:
                    prev_msg, _, _ = current_burst[-1]
                    delta = msg_obj['timestamp_utc'] - prev_msg['timestamp_utc']
                    if delta > gap_threshold or current_burst_tokens + m_tokens > MAX_TOKENS_PER_CHUNK:
                        bursts.append(current_burst)
                        current_burst = [(msg_obj, formatted, author_resolved)]
                        current_burst_tokens = m_tokens
                    else:
                        current_burst.append((msg_obj, formatted, author_resolved))
                        current_burst_tokens += m_tokens
            if current_burst: bursts.append(current_burst)
        else:
            for m_obj, fmt, auth in final_messages:
                bursts.append([(m_obj, fmt, auth)])
            
        current_window_bursts = []
        current_tokens = 0
        chunk_index = 0
        
        # Chunking loop
        def process_window(window, idx):
            chunk_text = "\n\n".join(["\n".join([f for m, f, a in b]) for b in window])
            start_time = window[0][0][0]['timestamp_utc']
            end_time = window[-1][-1][0]['timestamp_utc']
            
            # Accurate participants from the resolved author names
            participants = set()
            for b in window:
                for m, f, author_name in b:
                    participants.add(author_name)
            
            start_iso = datetime.datetime.fromtimestamp(start_time, datetime.timezone.utc).isoformat()
            end_iso = datetime.datetime.fromtimestamp(end_time, datetime.timezone.utc).isoformat()
            
            ctx = f"Platform: {platform.upper()}\n"
            ctx += f"Title: {primary_title}\n"
            if subreddit: ctx += f"Subreddit: r/{subreddit}\n"
            ctx += f"Participants: {', '.join(sorted(list(participants)))}\n"
            ctx += f"Timeframe: {start_iso} to {end_iso}"

            return {
                "chunk_id": generate_chunk_id(platform, root_id, start_iso, end_iso, idx),
                "channel_id": str(root_id),
                "platform": platform,
                "title": primary_title,
                "participants": sorted(list(participants)),
                "start_time": start_iso,
                "end_time": end_iso,
                "message_count": sum(len(b) for b in window),
                "estimated_tokens": estimate_tokens(chunk_text),
                "density_score": compute_chunk_density(window),
                "ego_weight": compute_ego_weight(window, identity_set),
                "text": chunk_text,
                "summary": "",
                "context": ctx
            }

        for idx, burst in enumerate(bursts):
            burst_text = "\n".join([f for m, f, a in burst])
            b_tokens = estimate_tokens(burst_text)
            
            if current_tokens + b_tokens > MAX_TOKENS_PER_CHUNK and current_window_bursts:
                session_chunks.append(process_window(current_window_bursts, chunk_index))
                chunk_index += 1
                
                # Sliding window logic
                overlap_bursts = []; overlap_tok = 0
                for b in reversed(current_window_bursts):
                    o_text = "\n".join([f for m, f, a in b])
                    overlap_tok += estimate_tokens(o_text)
                    overlap_bursts.insert(0, b)
                    if overlap_tok >= 400: break
                current_window_bursts = overlap_bursts
                current_tokens = estimate_tokens("\n\n".join(["\n".join([f for m, f, a in b]) for b in current_window_bursts]))
                
            current_window_bursts.append(burst)
            current_tokens += b_tokens
            
        if current_window_bursts:
            session_chunks.append(process_window(current_window_bursts, chunk_index))

    if map_changed: save_twitter_map(tw_map)

    output_path = REPO_ROOT / "processed_data" / "semantic" / ("dry_run_chunks.json" if dry_run else "session_chunks.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(session_chunks, f, indent=2, ensure_ascii=False)
        
    print(f"Generated {len(session_chunks)} session chunks. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    build_chunks(dry_run=args.dry_run)
