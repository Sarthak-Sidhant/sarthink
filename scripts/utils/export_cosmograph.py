import sqlite3
import csv
import logging
import json
import os
from pathlib import Path

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Platform-native color palette. Threads are darker, users are vivid, DMs are lighter tints.
GROUP_COLORS = {
    # Twitter — signature X/Twitter blue family
    "twitter_user":            "#1DA1F2",  # Twitter blue (vivid)
    "twitter_public_thread":   "#0a3d6b",  # deep navy (thread hubs)
    "twitter_dm_group":        "#4fc3f7",  # sky blue (DM rooms)

    # Reddit — orange family
    "reddit_user":             "#FF4500",  # Reddit orange (vivid)
    "reddit_chat":             "#ff8c42",  # warm amber (chat/DM rooms)
    "reddit_public_thread":    "#c13a00",  # deep burnt orange (thread hubs)

    # Instagram — gradient brand family (pink → purple)
    "instagram_user":          "#E1306C",  # Instagram pink (vivid)
    "instagram_comment_thread":"#833ab4",  # Instagram purple (thread hubs)
    "instagram_dm_group":      "#fd5c87",  # light rose (DM rooms)

    # Discord — Blurple family
    "discord_user":            "#5865F2",  # Discord blurple (vivid)
    "discord_dm_group":        "#7289da",  # classic Discord lighter blurple (DM rooms)

    # Facebook — Facebook blue family
    "facebook_user":           "#1877F2",  # Facebook blue (vivid)
    "facebook_comment_thread": "#0a5abf",  # deep Facebook navy (thread hubs)
    "facebook_dm_group":       "#74b3f7",  # light Facebook blue (DM rooms)

    # Fallback
    "group":                   "#aaaaaa",  # neutral grey
}

def get_thread_group(platform, title):
    """Categorize the kind of conversational thread for Cosmograph Coloring."""
    title_lower = title.lower()
    
    if title_lower.startswith('dm ') or 'direct message' in title_lower:
        return f"{platform}_dm_group"
    elif title_lower.startswith('comment on'):
        return f"{platform}_comment_thread"
    elif platform == 'reddit' and not title_lower.startswith('thread'):
        # Reddit DM edge case
        return f"reddit_chat"
    elif platform == 'twitter' and 'twitter_' in title:
        return 'twitter_tweet_thread'
    
    return f"{platform}_public_thread"

def load_identity_map():
    map_path = REPO_ROOT / "config" / "identity_map.json"
    if not map_path.exists():
        return None, {}
    
    with open(map_path, 'r') as f:
        data = json.load(f)
        
    master = data.get("master_persona", "User")
    alias_dict = {}
    for platform, aliases in data.get("aliases", {}).items():
        for alias in aliases:
            alias_dict[alias] = master
    return master, alias_dict

def export_to_cosmograph():
    logging.info("Connecting to Sarthink Database for Cosmograph Export...")
    
    master_persona, identity_aliases = load_identity_map()
    if master_persona:
        logging.info(f"Identity Map Loaded: Resolving configured aliases to -> '{master_persona}'")

    db_path = REPO_ROOT / 'processed_data' / 'db' / 'sarthink_memory.db'
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    nodes = [] # format: (id, label, group, weight)
    edges = [] # format: (source, target, weight)
    
    # Track node weights manually to size the Nodes properly in Cosmograph!
    node_weights = {}

    # 1. EXTRACT BINDING EDGES
    logging.info("Collapsing Messages into relational Vectors...")
    cursor.execute("""
        SELECT author_id, thread_id, COUNT(*) as weight
        FROM Messages
        GROUP BY author_id, thread_id
    """)
    interactions = cursor.fetchall()
    
    for author_id, thread_id, weight in interactions:
        u_node = f"U_{author_id}"
        t_node = f"T_{thread_id}"
        
        edges.append((u_node, t_node, weight))
        
        node_weights[u_node] = node_weights.get(u_node, 0) + weight
        node_weights[t_node] = node_weights.get(t_node, 0) + weight

    # 2. EXTRACT LOGICAL USERS
    logging.info("Formatting Entity Nodes...")
    cursor.execute("SELECT id, platform, display_name, raw_id FROM Users")
    users = cursor.fetchall()
    
    for uid, platform, display_name, raw_id in users:
        node_id = f"U_{uid}"
        
        # Prioritize true Usernames (raw_id) over display names, except for Discord/Meta where raw_id is a giant UUID
        primary_name = str(raw_id) if platform in ['twitter', 'reddit'] else str(display_name)
        if primary_name.isdigit() and display_name:
            primary_name = str(display_name)
            
        # UNIVERSAL IDENTITY RESOLUTION OVERRIDE
        if primary_name in identity_aliases:
            primary_name = identity_aliases[primary_name]
        elif display_name in identity_aliases:
            primary_name = identity_aliases[display_name]
        elif str(raw_id) in identity_aliases:
            primary_name = identity_aliases[str(raw_id)]
            
        # Clean rogue formatting chars
        label = primary_name.replace(',', '').replace('\n', ' ').replace('\r', '').replace('"', '') 
        weight = node_weights.get(node_id, 1)
        group = f"{platform}_user"
        color = GROUP_COLORS.get(group, "#ffffff")
        nodes.append((node_id, label, group, weight, color))
                   
    # 3. EXTRACT LOGICAL THREADS
    logging.info("Formatting Hub Nodes...")
    cursor.execute("""
        SELECT DISTINCT t.id, t.platform, t.title 
        FROM Threads t
        JOIN Messages m ON t.id = m.thread_id
    """)
    threads = cursor.fetchall()
    
    for tid, platform, title in threads:
        node_id = f"T_{tid}"
        weight = node_weights.get(node_id, 1)
        
        group = get_thread_group(platform, title)
        
        # Clean title
        clean_title = str(title).replace(',', '').replace('\n', ' ').replace('\r', '').replace('"', '')
        clean_title = clean_title[:45] + "..." if len(clean_title) > 45 else clean_title
        
        color = GROUP_COLORS.get(group, "#ffffff")
        nodes.append((node_id, clean_title, group, weight, color))
        
    conn.close()
    
    # 4. WRITE THE OUTPUTS
    logging.info(f"Writing {len(nodes)} Nodes & {len(edges)} Edges to Disk...")
    
    nodes_csv_path = REPO_ROOT / "processed_data" / "graph" / "cosmograph_nodes.csv"
    with open(nodes_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['id', 'label', 'group', 'size', 'color'])
        writer.writerows(nodes)

    edges_csv_path = REPO_ROOT / "processed_data" / "graph" / "cosmograph_edges.csv"
    with open(edges_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['source', 'target', 'weight'])
        writer.writerows(edges)
        
    logging.info("Cosmograph rendering structure complete!")

if __name__ == "__main__":
    export_to_cosmograph()
