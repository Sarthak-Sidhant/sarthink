# Plan: archive + researcher + memory (Oct 2026)

Sarthink is three layers over one truth:
1. **Archive**: what was actually said, word for word, never changed. The only ground truth.
2. **Researcher**: reads the archive to answer anything, including questions nobody anticipated; slow, thorough,
   cited, and it **interprets** (patterns, how a relationship changed, what you seem to want), labelled as
   interpretation with its evidence.
3. **Memory** (new): what Sarthink currently believes about you, the people in your life and the things around you:
   compact, current, time-aware facts, each with receipts. Memory is the researcher's notebook, not a replacement:
   it tells the researcher where to look, the researcher writes what it establishes back into it, and the researcher
   can always overrule it by reading the source. mem0 / Graphiti keep the conclusion and drop the conversation;
   Sarthink keeps both.

Decided (4 Oct 2026):
- **One user: you.** Other AI tools (Hermes, Claude, Cursor…) read from it — search, facts, people, the
  researcher — over MCP / HTTP, on this machine. They extract memory; they don't own it.
- **Facts about other people are in**, fully: the researcher needs them. It's your archive.
- **The researcher interprets**; memory stores what was said or done, interpretations are marked as such.
- **Proactivity is out of scope for now** (no briefs, notifications or nudges); to be discussed later.

The raw material already exists: 17,690 session summaries carry **151,213 atomic facts** (`facts`) and 71,990
"who did what" entries (`people`), ~3.4M tokens in all. Nothing below re-reads the 740k messages.

## Principles (unchanged, restated because the facts layer is where they bite)
- A fact is enrichment, never ground truth: model, prompt version, date and **evidence (session + message ids)** on
  every row. Deleting the facts tables loses nothing.
- Time is first-class: a fact has `valid_from` / `valid_to`. New information closes old facts instead of
  overwriting them ("in Bokaro until Jul 2026" → "in Bangalore since Jul 2026").
- You outrank the model: your notes and corrections are facts with the highest authority; a rebuild never undoes
  them. You can hide a person from memory (the archive keeps everything).
- Measure first: every phase has eval questions it must move, written before the phase is built.

## Phase 0 — Measure (first, ~half a day)
1. **Refresh the agent eval**: fix the ~6 questions whose answers changed with WhatsApp/Instagram (293k messages,
   Devanshi as top contact, 2026 as the busiest year).
2. **Add a memory eval** (~40 questions with verified answers), starting from the questions you actually ask in a
   normal week, in four kinds:
   - current state: "Where do I study now?", "Which laptop do I use?", "Who is my girlfriend?"
   - history: "Where did I live before Bangalore?", "What laptop did I have before the MacBook?"
   - about others: "Where does Jitesh live?", "What does Mihir do?", "What's Nivedha studying?"
   - relationships: "Who are my closest friends?", "Which people do I work with?"
   Answers are drafted from the summaries, then checked against the messages by hand-verifiable citations.
3. Baseline both evals on today's system (correctness, unsupported claims, **latency**).

## Phase 1 — Facts layer (core, ~2 days, ~$2–4 of DeepSeek off-peak)
Tables in the index DB (derived, rebuildable):
- `Entities(entity_id, kind, name, aliases, account_ids, person_id)`: you, people (linked to People/Accounts),
  places, organisations (colleges, companies, communities), projects, things (devices, vehicles).
- `RawFacts(rf_id, session_id, said_on, subject, kind, category, statement, durable, evidence)`:
  pass 1 output.
- `Facts(fact_id, entity_id, category, statement, valid_from, valid_to, status current|past|disputed,
  confidence, sources, superseded_by, model, updated)`: pass 2 output, what everything reads.

Pipeline (`scripts/index/build_facts.py`, incremental by session):
1. **Normalise (pass 1, flash, no thinking)**: each session's facts + people + date → structured raw facts:
   subject, kind, category (identity, relationship, education, work, location, possession, health, preference,
   belief, event, goal), and **durable or not** (most chat facts are moment-to-moment: "said the crowd is weird"
   is dropped, "is going to SJU" is kept). Evidence = the session's message ids.
2. **Resolve entities**: subject names → the accounts/people present in that session (deterministic first: the
   session's `people` roles name accounts), else an `Entities` row (string-normalised aliases; ambiguous merges
   go to a small LLM check, never silent).
3. **Consolidate (pass 2, flash with thinking, per entity)**: an entity's raw facts in date order → current and
   past facts with validity ranges, repetition → confidence, contradictions → `disputed` with both sources.
   You (thousands of facts) are consolidated per category.
4. **Incremental**: `update.py` gets a `facts` stage: new sessions → pass 1 → only the touched entities are
   re-consolidated.

Uses:
- **Agent fast path**: a `recall` tool (FTS + vectors over `Facts`) the agent calls first; answers cite the fact
  *and* its source messages. Target: person/self questions in **<10 s** (now 25–60 s) at equal or better accuracy.
- **People pages**: "What I know" — current facts with since-dates, and a timeline of past ones, each linking to
  the conversation.
- **About me page**: your own profile (studies, work, places, devices, projects, goals) with history.
- **Graph**: places, organisations and projects as nodes linked to the people they involve.

## Phase 2 — Relationships (who people are to you, ~half a day, cents)
- From facts + notes + the rhythm of messages: each person's relation to you (partner, family, close friend,
  friend, classmate, colleague, client, online acquaintance) with `since`, as a fact you can correct.
- Feeds Reminders (a person's area comes from who they are, not a per-item guess), people lists (group by
  relation), and the agent ("my college friends", "people I work with").

## Phase 3 — Reminders that run themselves (~1 day, cents per update)
What's there: Now / Auto-cleared / Older, areas, triage on current items. What's missing is the *lifecycle*:
1. **Close loops automatically**: after each update, re-triage pending items whose chat has new messages (done,
   cancelled, moved to a new date → the due date follows).
2. **One reminder per promise**: the same plan said in several chats or sessions ("Cubbon Park date") is merged,
   keeping its history and the latest due date.
3. **Priority**: a score from who it's with (Phase 2 relation), how firm (explicit date, repeated), and urgency;
   "Now" shows the top of it, not 153 flat rows.
4. **Learn from you**: your done/dismiss choices become examples the triage sees, so it stops surfacing what you
   always dismiss.
5. **Backfill**: triage the 2026 "Older" items (~2,400, ~$1.50) so nothing real stays buried.
(Daily briefs, notifications and other proactive behaviour: deferred, to be discussed.)

## Phase 4 — Open it up to your other AI tools (~1 day)
- **MCP server** on this machine (search, recall facts, person, ask the researcher, reminders, notes) so Hermes,
  Claude Desktop / Claude Code, Cursor or any MCP client can pull your memory with citations; plain HTTP for
  anything that doesn't speak MCP. Read-only by default; writing notes/facts needs your approval.
- **Memory from conversation**: what you tell Sarthink in chat ("I switched to Ramaiah") becomes a proposed fact
  you approve, instead of only an explicit note.

## Order and checkpoints
0 → 1 → 2 → 3 → 4. Each phase ends with: eval numbers vs baseline, cost, a browser check, a commit, and a short
note in this file. Phase 1 is the big one; it is only "done" when the memory eval shows the fast path matching or
beating the slow path on correctness, with lower latency.

## Costs (DeepSeek list price, off-peak)
| step | tokens | ~cost |
| --- | --- | --- |
| Phase 1 pass 1 (normalise 151k facts) | ~3.5M in, ~1.5M out | ~$1.50 |
| Phase 1 pass 2 (consolidate, thinking) | ~2M in, ~1.5M out | ~$1.20 |
| Phase 2 relations | small | <$0.20 |
| Phase 3 backfill of Older 2026 | ~3M in, 1M out | ~$1.50 |
| evals (two full runs per phase) | | ~$1–2 per run |
Incremental cost per future import: cents.
