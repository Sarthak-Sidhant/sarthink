# What went wrong (1–2 Oct 2026): appender, Reddit, WhatsApp, contacts, dossiers

Notes for picking this up again. Each item: what happened → cause → status.

## Data / pipeline
1. **Re-imports could wipe or duplicate data.** Twitter and Meta parsers purged the whole platform before importing,
   person ids were renumbered on every rebuild, and Meta message ids were positional (a newer export would store
   every message twice). → **Fixed**: append-only parsers, content dedupe for Meta, stable `PersonIds`. Verified:
   re-importing all four platforms added 0 messages, and all ids were unchanged.
2. **Re-import created an empty `facebook_ego_user`.** The author row was created before the duplicate check. → **Fixed.**
3. **Reddit context files were 5.1 GB for ~10 MB of content.** An older fetcher saved each comment under every
   ancestor (2^depth copies). → **Compacted** to 9.6 MB, with identical parser output (70,062 messages, 0 diffs).
   Originals are in `processed_data/context/reddit/Reddit_Context_Archive/posts.orig-2026-10-01/`.
   *Mistake:* I first claimed I had "fixed" the fetcher based on a guess about asyncpraw. The library source
   proved the guess wrong, and I reverted the change.
4. **WhatsApp first run made 15,824 empty chat threads and 8,960 dangling reply links.** → **Fixed** before the real
   import (only chats with kept messages; quote links only to stored parents).
5. **567 WhatsApp sessions were built as reply trees** (quote-heavy chats), which reordered messages. → **Fixed**:
   WhatsApp is always a chat (`CHAT_PLATFORMS`).
6. **Renaming people moved chunk boundaries.** Names count toward chunk length, so the contact rename changed ids for
   1,068 sessions and 11,301 chunks, which would have orphaned paid summaries and vectors. → **Fixed**: `SummaryLabels`
   freezes the layout name and the current name is displayed. Ids identical afterwards. One contact "name" was
   202 characters; renamed people's names are capped at 40 characters in chunk text (other platforms untouched).

## Cost / time estimates
7. **The DeepSeek summary estimate was low**: I said ~$5, the 50-session test said $6.60, and the actual was ~$7.00.
   I scaled from old prices and token counts instead of measuring. → Always run the sample test first (now in the README).
8. **DeepSeek balance ran out mid-run** (314 sessions failed); resumed after a top-up. Check the balance before big runs.
9. **Time estimates were off**: GPU embedding "10–15 min" took 41 min (long chunks come last); the summarizer looked
   slow at the start because it builds all prompts first.
10. **`build_commitments` doesn't log token usage**, so the ~$0.18 inbox cost is an estimate.

## Privacy (still open)
11. **Phone numbers reached DeepSeek.** Names went as masked labels (country code + 5 of 10 digits), but
    **802 WhatsApp sessions contained full numbers typed in chat**, and **733 summaries repeat a full number**.
    → **Open**: mask phone numbers and emails before sending anything to DeepSeek, and scrub the stored summaries.
    Check DeepSeek's API data-retention policy.
12. `wa.db` from the backup has no contact names (WhatsApp strips them). The Google Takeout contacts were the wrong
    list. Solved with `archive/whatsapp-contacts.vcf` built from the WhatsApp Web export. Both files hold full phone
    numbers: keep them in `archive/` (gitignored).

## Speed
13. **Dossier themes via DeepSeek took 5+ minutes** for a big contact. → **Replaced** with local themes
    (`scripts/index/themes.py`: vectors + k-means + summary topics). The first version repeated "flirty banter" in
    three themes; fixed with distinctiveness scoring.
14. **Clicking a person was slow (18 s uncached).** The `messages` SQL view scanned all 692k messages on every query,
    and the dossier cache was wiped by any DB write. → **Fixed in code, not yet timed live**: inline ts in the
    view (verified identical row by row), per-person cache stamp, indexed chat lookup (`PersonProfiles.threads`).
    Offline: biggest contact 5.6 s, typical 0.8 s, small 0.4 s. `:8000` was restarted with all three fixes.

## Process mistakes (mine)
15. Ran an O(n²) comparison that hung, and twice killed my own shell with `pkill`/`pgrep` patterns that matched
    my own command.
16. Two Claude sessions shared one working tree: stale copies of the other session's files were left in `main`,
    and its server restarted in the middle of my rebuild. Resolved by messaging each other; next time stop servers
    and agree on file ownership first.

## To do tomorrow
- [ ] Time dossiers on the live server (`/api/person/<id>/dossier?refresh=true`).
- [ ] Commit: `scripts/api/agent.py`, `scripts/api/server.py`, `scripts/index/build_chunks.py`,
      `scripts/index/themes.py` (new), `scripts/parsers/whatsapp_parser.py`, and the other session's
      `web/js/people.js`, `web/style.css`. Then decide on pushing `main` (13 commits ahead of GitHub).
- [ ] Privacy: mask before DeepSeek, scrub the 733 summaries, maybe shorter masked labels ("Contact ··962").
- [ ] Inbox flood: 5,935 WhatsApp items, many years old. Add a date cutoff or bulk dismiss.
- [ ] Query encoder (no GPU running, so search is keyword-only).
- [ ] `update.py`: four open decisions (GPU automation with a cap, stopping servers, confirmations, v1 scope).
- [ ] Disk cleanup once happy: `processed_data/backups/*`, `emb_*_ctx.prev.npy`, `processed_data/index/updates/`,
      `posts.orig-2026-10-01/`.
