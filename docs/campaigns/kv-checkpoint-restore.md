# kv-checkpoint-restore — conversation state kept between requests and across a restart

**Open (backlog).** Lever 7 of `research-reference-audit.md` §4.

## Charter

An agent re-sending a long history continues from saved conversation state
(the KV and the GDN/recurrent state at a block boundary) instead of
prefilling it again: in process alongside the adaptive expert tier, and from
disk after a server restart.

## Reference to follow

**Strata** (`~/src/Strata-ref`):
- **Parked conversations** (`code`: `include/strata/core/conversation_cache.hpp`,
  `src/core/conversation_snapshot.cpp`, `include/strata/core/conversation_snapshot.hpp`).
  The KV of the attention layers (with the QSA indexer state) and the
  recurrent state are saved as an image; reuse needs token equality, image
  identity and the same steering mode; images are validated before restore
  (`conversation_kv_validate`).
- **Tests** (`code`: `tools/conversation_cache_parity.py`,
  `tools/conversation_cache_isolation.py`, `tools/conversation_cache_soak.py`).
- **Effect** (`paper` §7(a)): named as the largest single improvement for
  agents, a state of a few hundred MB of recurrent state plus KV.

## Gate

A conversation of N tokens (the coder's served depth class, tens of
thousands of tokens) restored into the same process and into a fresh one:
the continuation passes the answer-level bar (`CLAUDE.md`) against a cold
prefill; the restore's wall time is below the prefill it replaces, stated per
card and precision; an image written under a different artifact hash,
precision, block size, device or runtime is refused at load.

## Current state

- The prefix cache (`src/core/prefix_cache.h`) is in-process: KV pages by
  reference on the card, GDN checkpoint blobs (~32 MiB a row) in host memory;
  nothing is written to disk (`code`).
- DESIGN §3.4 Amendment 2 (2026-10-01) lifts the refusal of the prefix cache
  with an adaptive expert tier; `tier_prefix_cache_decision`
  (`src/config.cpp`) still refuses the pair and is owed the change.
- To read before building: the paged plugin's KV page layout at u8 and
  u8:i4 (patches 0008–0010), the GDN blob's contents, and the image's
  identity tuple.

## Where it lives

`src/core/prefix_cache.h`/`.cpp`, `src/exec/backend_ov.cpp` (the paged pool,
`alloc_kv_pools`, the GDN rows, the snapshot grid), `tests/equivalence/run.sh`
(the continuation-restore check).

Full history: `git show b0447b8:docs/campaigns/kv-checkpoint-restore.md`.
