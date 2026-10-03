# Accounts and people

Sarthink has two layers. Keep them apart.

**Accounts** are what the data contains: one per platform account (`platform`, `raw_id`), and one per chat for
a deleted/deactivated account (`chat` = the platform thread id), because one placeholder id ("Instagram User")
covers many humans. Every account has its own stats, summary and graph node, and none of that changes when
people are linked. Accounts are derived from the exports and rebuilt with the index.

**People** are what the user says: a person is a node that groups accounts ("Riya" = her WhatsApp + Instagram).
Linking attaches an account to a person; it never merges accounts. An account belongs to at most one person, and
an account nobody attached stays an account without a person. People are user-written and survive rebuilds.
Person 1 is **me**: all of the owner's accounts (config `identity_map.json`) belong to it automatically.

```
P_1 me ─── A_ my twitter, A_ my discord, A_ my whatsapp …
P_7 Riya ─┬─ A_ whatsapp +91 98··· ··210 ──── T_ chats
          └─ A_ instagram @riya.s ─────────── T_ chats
A_ someone unlinked ─────────────────────────── T_ chats
```

## Where things live

| | Table | DB | Notes |
|---|---|---|---|
| account ids | `AccountIds (platform, raw_id, chat) -> account_id` | user_notes.db | stable forever (was `PersonIds`) |
| accounts | `Accounts (account_id, name, platform, is_me, placeholder, message_count)` | index | was `Persons`, no more merging |
| account of a src user | `AccountUsers (user_id -> account_id)` | index | was `PersonAliases` |
| per-chat placeholder accounts | `AccountThreads (user_id, thread_id -> account_id)` | index | was `PersonThreads` |
| people | `People (person_id, name, created, updated)` | user_notes.db | person 1 = me (was `Contacts`) |
| membership | `PersonAccounts (platform, raw_id, chat) -> person_id` | user_notes.db | an account in at most one person (was `ContactAccounts`) |
| turned-down suggestions | `PersonRejections` | user_notes.db | was `ContactRejections` |
| notes | `Notes (…, person_id, account_id)` | user_notes.db | about a person or about an account |
| who is in a chunk/session | `Chunks.account_ids`, `Sessions.account_ids` | index | was `person_ids`; never changes on linking |

Ids: `A_<account_id>` and `P_<person_id>` are separate id spaces; graph threads stay `T_<thread_id>`.

## Behaviour

- **Linking is one row.** Attach, detach, create, rename and delete a person write `PersonAccounts`/`People` only.
  No rebuild of identity, chunks or the graph files, so it is instant.
- **Summaries:** an account's summary covers the conversations that account wrote in. A person's view combines
  their accounts: totals across them and themes over the union of their conversations.
- **Graph:** the graph files hold account and thread nodes only (rebuilt on import). People are served by the API
  and drawn as nodes linked to their accounts, so a link shows up without reloading the graph.
- **Search and the assistant:** "chats with Riya" expands Riya to her accounts. An account name works too.
- **Suggestions:** accounts with the same name on different platforms are offered as "same person?"; turning
  one down is remembered.
