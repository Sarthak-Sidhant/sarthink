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
    "Twitter Thread": lambda c: c["platform"] == "twitter" and not c["title"].startswith("DM ") and 3 <= c["message_count"] <= 8,
    "Twitter DM": lambda c: c["platform"] == "twitter" and c["title"].startswith("DM "),
    "Reddit Thread": lambda c: c["platform"] == "reddit" and "[POST]" in c["text"],
    "Reddit DM": lambda c: c["platform"] == "reddit" and "[POST]" not in c["text"],
    "Instagram DM": lambda c: c["platform"] == "instagram",
    "Discord DM": lambda c: c["platform"] == "discord",
}

found = set()
md_content = "# Sample Chunks\n\n"

for c in chunks:
    for name, func in categories.items():
        if name not in found and func(c):
            found.add(name)
            md_content += f"## {name}\n\n"
            md_content += "### Prompt Header\n```text\n"
            md_content += c['context'] + "\n```\n\n"
            md_content += "### Log Preview\n```text\n"
            md_content += c['text'] + "\n```\n\n"
            md_content += "---\n\n"
            break
    if len(found) == len(categories):
        break

output_file = REPO_ROOT / "processed_data" / "semantic" / "sample_chunks.md"
with open(output_file, "w") as f:
    f.write(md_content)

print(f"Successfully generated {output_file} with full markdown text!")
