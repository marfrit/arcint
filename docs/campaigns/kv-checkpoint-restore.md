# kv-checkpoint-restore — a restarted server continues a long conversation without prefilling it again

## The defect, as measured

Not a defect: a capability that does not exist, opened from an operator
question on 2026-09-05 ("if I restart the server, can I restore a long
agent conversation without going through the whole session's prefill
again?"). Today the answer is no. The prefix cache (`src/core/prefix_
cache.h`, DESIGN §3.4) is in-process memory: a hit maps KV pages that are
still on the card and restores a GDN checkpoint blob held in host RAM;
demoted entries park in host memory; nothing is written to disk. When
the process ends, every cached prefix goes with it, and the next process
prefills the conversation from token one at the artifact's prefill rate
— for the coder, roughly 500–2,800 t/s depending on the card (DESIGN
§7.0.2al's references), so a 100,000-token session costs some 40 s to
3 min once per restart.

What the engine does guarantee across a restart is the *text*: with the
tier on, patch 0018's static partition makes a continuation restored
from the prefix cache byte-identical to the same continuation prefilled
cold in a fresh process (DESIGN §7.0.2ae, §7.0.2ai; the cold-start
window's eight identical outputs, §7.0.2aq). A restart today changes the
time to the first token of the continuation, never the continuation.

## Known against hypothesised

Known: the prefix cache's two halves and where each lives (KV pages by
reference on the card, GDN checkpoint blobs of ~32 MiB per row in host
memory, the host tier for demoted pages); the entry key (a token hash,
verified against the tokens on a hit — OV GenAI's #3489 is the reason);
the KV footprint per token at the served precisions (§7.0.3: 11.3 KiB at
u8 on the coder, 8.8 at u8:i4); the snapshot grid (2,048 tokens, the
load line "prefix-cache snapshot grid"). Hypothesised, nothing measured:
that a page's bytes can be read back from the card into a file and
written back into a fresh process's pool with the block layout intact
(the paged plugin's KV layout is its own, per precision, and the block
size is a server flag); that the GDN blob is position-independent
beyond its row; that a restore is cheaper than the prefill it replaces
(100,000 tokens × 11.3 KiB = 1.1 GB of KV plus the GDN rows — a disk
read of seconds against minutes of prefill, on paper).

## Gate

A conversation of N tokens served by process A, checkpointed; process B
started fresh restores it and serves the continuation: (1) byte-identical
to the continuation process A would have served and to a cold prefill in
a third process (the §3.4 invariant, extended across processes); (2) the
restore's wall time below the prefill it replaces at the same depth,
stated per card and precision; (3) a checkpoint written under one
artifact hash, precision, block size, device or runtime refused by a
process with any of them different, loudly. N at least the coder's
served depth class (tens of thousands of tokens), on both cards.

## Entry criteria

Not met, and deliberately not started: this is a backlog entry. Before
it starts, three things on the record: (a) the paged plugin's KV page
layout at u8 and u8:i4 read from the plugin source, with what a byte-
exact readback and writeback needs (the runtime's page tensors are the
plugin's, not the engine's); (b) the GDN checkpoint blob's contents and
whether it carries anything process-bound; (c) a decision on the
checkpoint's identity — the prefix cache's own key (tokens + a hash) or
the conversation as the client sends it — and on who triggers it (a
request flag, a server-side policy at the snapshot grid, or shutdown).

## Scope — in / out

In, when it starts: an on-disk checkpoint of one prefix (KV pages + GDN
rows + the token list + the identity tuple), its restore into a fresh
process's pool, the refusal rules, the byte-identity test across three
processes, the cost per token measured.

Out: any change to the prefix cache's in-process behaviour or eviction;
speculative drafters' state (the DFlash window, the MTP state — a
restored prefix starts them cold, as a cache hit does today); anything
that trades §3.4 for a faster restore; the cold start (`static-
partition-cold-start`) — a restored checkpoint still pays that
process's warm-up.

## Where it lives

`src/core/prefix_cache.h`/`.cpp` (entries, the state blob, the host
tier); `src/exec/backend_ov.cpp` (the paged pool, `alloc_kv_pools`, the
GDN rows, the snapshot grid, the restore path the equivalence suite's
"restored continuation vs cold" check exercises); the paged plugin's KV
tensors (patches 0008–0010 for the packed-value layout); `tests/
equivalence/run.sh` (the continuation-restore check, the template for
the cross-process one); DESIGN §3.4, §7.0.3.

## Pipeline for this campaign

Recon (a–c above, written down with file:line) → design note (the
checkpoint's format and identity, the refusal table, the cost model) →
red case: the cross-process byte-identity test, failing because the
restore does not exist → the change → one card window: N at the served
depth on both cards and precisions, restore time against prefill time,
the three-process identity → review → commit → DESIGN §7.0.2x, a
CHANGELOG line, README's status table.

## Invariants

DESIGN §3.4 across processes: a restored continuation equals a cold one,
byte for byte, or the checkpoint is refused. §3.8 untouched. No restore
is served from a checkpoint whose identity tuple differs from the
process's own.

## Status

- 2026-09-05 — opened as a backlog entry from the operator's question;
  the answer today is recorded above (no; the text survives a restart,
  the time does not). Not started; ordered after the campaigns already
  open.
