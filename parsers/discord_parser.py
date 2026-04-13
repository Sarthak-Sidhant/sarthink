import os
import json
import glob
import logging
from datetime import datetime, timezone
from database import SarthinkMemoryLayer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- CONFIGURATION ---
PLATFORM = "discord"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
JSONL_OUTPUT = "discord_logs.jsonl"

def parse_discord_timestamp(dt_str):
    """Converts a discord timestamp (e.g. 2025-01-26T04:05:31.195+05:30) to a valid UTC Epoch Integer."""
    try:
        dt = datetime.fromisoformat(dt_str)
        return int(dt.astimezone(timezone.utc).timestamp())
    except Exception as e:
        logging.error(f"Error parsing timestamp {dt_str}: {e}")
        return 0

def process_discord():
    db = SarthinkMemoryLayer()
    
    # 1. Find all exported JSON files recursively in the archive/discord-export folder
    search_path = os.path.join(REPO_ROOT, "archive", "discord-export", "**", "*.json")
    json_files = glob.glob(search_path, recursive=True)
    
    if not json_files:
        logging.error(f"No Discord JSON files found in {search_path}. Check your directory structure.")
        return

    logging.info(f"Identified {len(json_files)} localized Discord files.")
    
    total_messages = 0
    # 2. Iterate and process
    for filepath in json_files:
        filename = os.path.basename(filepath)
        logging.info(f"Parsing: {filename}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Check standard format
        if 'channel' not in data or 'messages' not in data:
            continue
            
        channel = data['channel']
        channel_id = channel.get('id', 'unknown')
        
        # In DMs, the file name often serves as a good title (e.g., "Direct Messages - John [1234].json")
        thread_title = filename.replace(".json", "")
        
        thread_db_id = db.get_or_create_thread(
            platform=PLATFORM, 
            platform_thread_id=channel_id, 
            title=thread_title
        )
        
        for msg in data['messages']:
            # Skip empty contents (like system messages or pure reaction events)
            if not msg.get('content') and not msg.get('attachments') and not msg.get('embeds'):
                continue
                
            raw_msg_id = msg.get('id')
            global_msg_id = f"{PLATFORM}_{raw_msg_id}"
            
            author_data = msg.get('author', {})
            author_raw_id = author_data.get('id', 'unknown')
            # Prefer nickname if set, fallback to name
            author_display = author_data.get('nickname') or author_data.get('name') or 'unknown'
            
            author_db_id = db.get_or_create_user(
                platform=PLATFORM,
                raw_id=author_raw_id,
                display_name=author_display
            )
            
            timestamp_str = msg.get('timestamp')
            utc_epoch = parse_discord_timestamp(timestamp_str) if timestamp_str else 0
            
            # Extract parent ID if it's a specific reply!
            parent_global_id = None
            if msg.get('type') == 'Reply' and 'reference' in msg:
                ref = msg['reference']
                if ref and ref.get('messageId'):
                    parent_global_id = f"{PLATFORM}_{ref['messageId']}"
            
            # Combine content with embedded domains if needed (e.g., reddit links in discord)
            content = msg.get('content', '')
            for embed in msg.get('embeds', []):
                if 'title' in embed or 'url' in embed:
                    content += f"\n[Embed: {embed.get('title', 'Link')} - {embed.get('url', '')}]"
            
            flat_json_entry = {
                "log_id": global_msg_id,
                "platform": PLATFORM,
                "thread_name": thread_title,
                "author": author_display,
                "author_id": author_raw_id,
                "timestamp_utc": timestamp_str,
                "content": content,
                "is_reply_to": parent_global_id
            }
            
            db.insert_message(
                msg_id=global_msg_id,
                thread_id=thread_db_id,
                author_id=author_db_id,
                timestamp_utc=utc_epoch,
                content=content,
                parent_msg_id=parent_global_id,
                json_data=flat_json_entry,
                jsonl_filename=JSONL_OUTPUT
            )
            total_messages += 1

    db.close()
    logging.info(f"Successfully processed {len(json_files)} threads and inserted {total_messages} messages for {PLATFORM}.")

if __name__ == "__main__":
    process_discord()
