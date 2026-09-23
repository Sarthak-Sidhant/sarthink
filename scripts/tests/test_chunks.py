import json
import os
from pathlib import Path

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

INPUT_FILE = REPO_ROOT / "processed_data" / "semantic" / "dry_run_chunks.json"

if not INPUT_FILE.exists():
    print(f"Error: {INPUT_FILE} not found!")
    exit(1)

with open(INPUT_FILE, "r") as f:
    chunks = json.load(f)

categories = {
    "Reddit Thread": lambda c: c["platform"] == "reddit" and not c["title"].startswith("DM:"),
    "Reddit DM": lambda c: c["platform"] == "reddit" and c["title"].startswith("DM:"),
    "Twitter Post/Thread": lambda c: c["platform"] == "twitter" and not c["title"].startswith("DM "),
    "Twitter DM": lambda c: c["platform"] == "twitter" and c["title"].startswith("DM "),
    "Discord Message": lambda c: c["platform"] == "discord",
    "Instagram Message": lambda c: c["platform"] == "instagram",
}

found = set()
for c in chunks:
    for name, func in categories.items():
        if name not in found and func(c):
            found.add(name)
            print(f"=== {name} ===")
            print(f"Title: {c['title']}")
            print(f"Platform: {c['platform']} | Messages: {c['message_count']} | Tokens: {c['estimated_tokens']}")
            print(c['text'][:800] + "...")
            print("-------------------------------------------------\n")
            break
    if len(found) == len(categories):
        break
