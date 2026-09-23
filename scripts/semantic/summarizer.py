import os
import json
import asyncio
import argparse
import sys
from typing import List, Dict, Optional
from collections import defaultdict
from openai import AsyncOpenAI
from pathlib import Path

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

# ── Configuration ─────────────────────────────────────────────────────────────
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
try:
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(encoder.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        # Worst-case truncation estimation: 2 chars ≈ 1 token
        return len(text) // 2

MODEL_NAME       = "llama3.1-8b"

MAX_RPM          = 5          # requests per minute (Cerebras free tier)
BATCH_SIZE       = MAX_RPM     # process N at a time, then sleep 60s
BATCH_SLEEP      = 62          # seconds between batches (safe margin)
ESTIMATED_OUTPUT = 2000        # max tokens for summary output

# Model context cap
MODEL_CTX_LIMIT  = 8192
SYSTEM_OVERHEAD  = 600
USABLE_TOKENS    = MODEL_CTX_LIMIT - SYSTEM_OVERHEAD - ESTIMATED_OUTPUT  # ~5592
CURRENT_BUDGET   = int(USABLE_TOKENS * 0.75)   # ~4194 tokens for current chunk
PREV_BUDGET      = int(USABLE_TOKENS * 0.25)   # ~1398 tokens for N-1 context

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """SYSTEM PROMPT — Sarthink Cognitive Memory Engine

You are Sarthink's memory engine. Your sole job is to compress a conversation log into a retrieval-optimized memory record from the owner's first-person cognitive perspective.

STRICT GROUNDING RULES:
- Never write from a third-person or neutral perspective.
- Never use filler phrases like "the conversation discusses" or "the users talk about".
- ANTI-HALLUCINATION: Do not invent emotional states, future plans, or background details not explicitly mentioned in the text. If the log is brief/vague, keep the summary brief/vague.
- Do not summarize the N-1 context block — it is background only.
- Ensure the summary captures the totality of the conversation, including specific numbers, links, and technical details.

OUTPUT FORMAT (always follow this exactly):

TOPICS: [comma-separated list of specific topics, concepts, technologies, or themes]

PEOPLE: [Name — their position/stance in one clause. Repeat per person.]

FACTS: [Bullet list of specific facts, decisions, links, numbers, or conclusions mentioned]

MEMORY: [2-3 sentences written as Sarthak's internal monologue — what he argued, learned, felt, or decided. Use "I" voice. Be specific, not generic. Do NOT invent internal feelings (e.g., "I'm left wondering") unless they are stated in the text.]

KEYWORDS: [10-15 keywords optimized for vector search. RULES: Lowercase only. Space-separated only. Strictly NO camelCase or compound words (e.g., use 'ultimate pigeon' instead of 'ultimatePigeon').]"""

PLATFORM_HINTS = {
    "reddit":    "This is a public forum debate. Capture the argumentative stance clearly.",
    "discord":   "This is a casual DM conversation. Capture the relationship dynamic and informal decisions.",
    "twitter":   "This is a public thread. Capture the core claim and any notable responses.",
    "instagram": "This is a social post with comments. or casual DM conversation. Capture the subject and sentiment. relationship dynamic and informal decisions.",
    "facebook":  "This is a social post or DM. Capture the key interaction and context.",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _truncate(text: str, max_tokens: int) -> str:
    """Worst-case truncation: 2 chars ≈ 1 token (safe for Unicode/emoji-heavy content)."""
    char_cap = max_tokens * 2
    if len(text) <= char_cap:
        return text
    return text[:char_cap] + "\n[... truncated ...]"

# ── Core summarizer ───────────────────────────────────────────────────────────
async def summarize_chunk(
    chunk: Dict,
    prev_text: Optional[str],
    client: AsyncOpenAI,
) -> tuple[str, int]:
    text    = chunk.get("text", "")
    context = chunk.get("context", "")
    if not text.strip():
        return "Empty conversation block.", 0

    platform = chunk.get("platform", "unknown").lower()
    hint = PLATFORM_HINTS.get(platform, "Capture the key information and interactions.")

    # Hard-cap inputs to stay within model context window
    text = _truncate(text, CURRENT_BUDGET)
    if prev_text:
        prev_text = _truncate(prev_text, PREV_BUDGET)

    user_content = f"[METADATA CONTEXT]\n{context}\nContext Hint: {hint}\n------------------\n\n"
    if prev_text:
        user_content += f"[PREVIOUS CONTEXT - DO NOT SUMMARIZE]\n{prev_text}\n\n==================\n\n"
    user_content += f"[CURRENT LOG - SUMMARIZE THIS]\n{text}"

    system_tokens = count_tokens(SYSTEM_PROMPT)
    user_tokens   = count_tokens(user_content)
    total_input_tokens = system_tokens + user_tokens

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=ESTIMATED_OUTPUT,
        )
        return response.choices[0].message.content.strip(), total_input_tokens
    except Exception as e:
        print(f"  [ERROR] chunk {chunk.get('chunk_id')}: {e}")
        return f"[FAILED] {e}", total_input_tokens

# ── Batch processor ───────────────────────────────────────────────────────────
async def process_batch(chunks: List[Dict]) -> List[Dict]:
    client = AsyncOpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=CEREBRAS_API_KEY,
    )

    # Build N-1 prev_map
    channel_groups: dict = defaultdict(list)
    for c in chunks:
        channel_groups[c["channel_id"]].append(c)

    prev_map: dict = {}
    for ch_id, ch_chunks in channel_groups.items():
        ch_chunks.sort(key=lambda x: x.get("start_time", ""))
        for i, c in enumerate(ch_chunks):
            prev_map[c["chunk_id"]] = ch_chunks[i - 1]["text"] if i > 0 else None

    total = len(chunks)
    completed = 0
    cumulative_tokens = 0

    # Process in batches of BATCH_SIZE, sleeping between each batch
    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} — chunks {batch_start+1}–{batch_start+len(batch)} of {total}")

        tasks = [
            summarize_chunk(c, prev_map.get(c["chunk_id"]), client)
            for c in batch
        ]
        results = await asyncio.gather(*tasks)

        for chunk, (summary, tokens) in zip(batch, results):
            chunk["summary"] = summary
            completed += 1
            cumulative_tokens += tokens

        if batch_start + BATCH_SIZE < total:
            print(f"  Rate limit pause: sleeping {BATCH_SLEEP}s...")
            await asyncio.sleep(BATCH_SLEEP)

    return chunks, cumulative_tokens

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int, default=0,
                        help="Number of chunks to process. 0 = all.")
    parser.add_argument("--input",  default=str(REPO_ROOT / "processed_data" / "semantic" / "session_chunks.json"))
    parser.add_argument("--output", default=str(REPO_ROOT / "processed_data" / "semantic" / "summarized_chunks.json"))
    args = parser.parse_args()

    if not CEREBRAS_API_KEY:
        print("ERROR: CEREBRAS_API_KEY not set in environment.")
        return

    try:
        with open(args.input, "r") as f:
            session_chunks = json.load(f)
    except FileNotFoundError:
        print(f"{args.input} not found — run chunk_builder.py first.")
        return

    process_list = session_chunks[:args.limit] if args.limit > 0 else session_chunks
    total_batches = (len(process_list) + BATCH_SIZE - 1) // BATCH_SIZE
    est_minutes = total_batches * (BATCH_SLEEP / 60 + 0.2)

    print(f"Processing {len(process_list)} chunks via Cerebras ({MODEL_NAME})")
    print(f"Batch size: {BATCH_SIZE} | Sleep between batches: {BATCH_SLEEP}s")
    print(f"Estimated time: ~{est_minutes:.1f} minutes\n")

    updated_chunks, total_tokens = asyncio.run(process_batch(process_list))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(updated_chunks, f, indent=2, ensure_ascii=False)

    avg_tokens = total_tokens / len(updated_chunks) if updated_chunks else 0
    print(f"\nDone. Processed {len(updated_chunks)} chunks.")
    print(f"Total input tokens sent: {total_tokens:,}")
    print(f"Average tokens per chunk: {avg_tokens:.1f}")
    print(f"Saved summaries → {args.output}")


if __name__ == "__main__":
    main()
