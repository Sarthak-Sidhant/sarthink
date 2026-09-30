# Sarthink Roadmap

Status: DONE / PARTLY / TODO. Compute is not a constraint (local M5 Pro + rented GPUs).

## Principles
- SQLite stays the deterministic source of truth (exact stats, never LLM-counted).
- Indexing is cheap (embeddings, FTS5). LLMs reason at query time via an agent with tools.
- Every LLM-produced record is enrichment, labeled, with provenance (source msg_ids, model, date), deletable.
- Raw exports are never modified. Tag quality/relevance classes at ingest, filter at query time, never delete.
- Data is organised into typed segments: Conversations, Writings (email/docs), Activity (search/browser history), AI chats. Shared thin core + per-segment tables. Joined by entity and time.
- Suggestions (tasks, reminders, events) go through an approval queue.
- Keep an eval set; judge every phase by it.

## Done
- Parsers: Twitter, Reddit, Discord, Meta. Context fetchers (Twitter, Reddit). Twitter ID map.
- SQLite store (22.6k users, 60.9k threads, 302.8k messages) + JSONL logs.
- Identity map for own handles. Graph export, 3D layout, Three.js viewer (82k nodes).
- Chunking: 32,849 chunks (density, ego_weight, tokens).
- Partly: summarizer (5 chunks run), embedder (written, not run, no metadata).

## Index layer (DONE 2026-09-30) — scripts/index/, output processed_data/db/sarthink_index.db
Source DB is attached read-only; delete the index DB to rebuild. Run in order:
1. `build_identity.py` – Persons/PersonAliases. Owner matched by account id / screen name; display-name
   aliases only when unique on that platform (generic "Sarthak" matched 7 other people before). 21 cross-platform
   MergeCandidates left for manual review.
2. `build_chunks.py` – structure-aware chunks (replaces scripts/semantic/chunk_builder.py). Chat vs tree by reply
   structure; chat sessions on 45-min gaps (min 120 tok, 24h hard gap); ~300-token windows, <=30% overlap; trees in
   DFS order with ancestor context. Sessions (LLM read unit) + Chunks (embed unit) + ChunkMessages (citations).
   7,868 null tweet timestamps recovered from Snowflake IDs.
3. `check_chunks.py` – 100% coverage, 0 chunks over cap, 17% overlap duplication, no cross-session messages.
Result: 44,016 sessions, 53,807 chunks (49,928 embed, 3,879 fts_only).
4. `summarize_sessions.py` (DONE) – deepseek-flash (thinking disabled), prompt v3, sessions >=100 tokens:
   9,298/9,300 summarized (2 refused by provider moderation), 19,078 chunk context lines. Neighbor sessions
   passed as read-only context. Tables SessionSummaries / ChunkContexts, keyed by model + prompt_version. ~$14.
5. `build_fts.py` – FTS5 over header/text/context/session keywords.
6. `build_embeddings.py` / `search.py` / `needle_eval.py` – flat numpy vectors (no vector DB), hybrid RRF,
   optional Qwen3-Reranker, me_involved boost. Needle test: 325 LLM-written recall queries, R@k + MRR.
   Embeddings run on vast.ai (`remote_embed.sh`); this laptop's CPU is too slow. Upload via ssh proxy
   (ssh3.vast.ai) — the direct-IP route was ~8 KB/s.
7. Graph (DONE 2026-10-01): `scripts/utils/export_cosmograph.py` now exports person nodes `P_<person_id>`
   (one gold "me" node) + thread nodes `T_<thread_id>`; thread groups fixed (Instagram name-titled threads and
   Reddit "DM:" threads are DMs). `compute_layout.py` uses a me-centred relational layout when a "me" node
   exists: people at log-closeness radius in per-platform sectors, threads at the message-weighted centroid of
   participants, my solo posts as an even cloud. Old CSVs kept as *.pre-persons.bak.
   Known: deactivated-account placeholders ("Instagram User", reddit deleted_user_room) merge many people.
Retrieval result (needle test, 325 queries): Qwen3-Embedding-8B + FTS5 hybrid (RRF), no reranker →
R@1 0.88, R@10 0.98. Vectors in processed_data/index/emb_qwen3-embedding-8b_ctx.npy (1024-token inputs).
Next: search API on a GPU box, agent with citations, viewer chat + fly-to-citation.

## Phases
0. Housekeeping (TODO): pyproject/uv, dotenv, .env key mismatch, stale docs, run order (export before layout).
1. Eval set (TODO, START HERE): 30-50 questions with known answers (stats, factual, temporal, open-ended, cross-source).
2. Core data model (PARTLY): segments schema, MessageFeatures, FTS5, reproduce the person-comparison stats table.
3. Semantic index (PARTLY): embed raw chunks with metadata (author, thread, platform, segment, ts, ego_weight), contextual prefixes, reranker.
4. Tool server + agent (TODO): MCP tools run_sql(ro), semantic_search, keyword_search, read_thread, resolve_person, run_python, describe_schema. Abstain on weak evidence.
5. Viewer integration (TODO): click node -> dossier; semantic hits highlight the map; chat panel with source cards.
6. Bulk LLM enrichment (TODO): summaries, topic/sentiment tags, task/event/reminder extraction (deterministic filter then LLM -> Suggestions table), RAPTOR levels, Graphiti or own Facts table on top relationships.
7. More segments (TODO): AI chats (chatgpt/claude exports), WhatsApp, Gmail (promo/transactional classes), Activity; cross-source entity resolution; idempotent re-import.
8. Second-brain features (TODO): approval queue UI, .ics export, journal/day review, "a year ago today", person profiles.
9. Stage 3 + polish (TODO): LoRA persona fine-tune, incremental ingestion, forget-person/source, demo dataset, one-command setup.
