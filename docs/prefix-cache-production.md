# Prefix cache in production: pre-registration

Written before the journals are read (2026-08-30). Every large win in this
project was not computing something; the cache is the largest such lever and
the only one with no production number against it. The order: (1) hit rate
from the journal per endpoint, as tokens served from cache against tokens
prefilled, with the distribution; (2) classify the misses by where the prefix
diverged; (3) size the branch case against the spare pool, and state the
eviction policy; (4) only then the resume-after-mismatch design question.

## What I expect, before looking

**Coder (b5 on the A770, 98304 context, 2211 spare pages ≈ 35k tokens).**
One-shot "code me this function" traffic: short prompts, mostly distinct, so
a **low hit rate by count** — many requests share only the system block —
but the system block is the same every time, so a **non-trivial hit rate by
tokens** if the client's system prompt is stable. Where the operator branches
with the rewind command, hits land at the branch point. Expectation: hits on
under half of requests, cached share of prompt tokens between 20% and 50%,
and capacity **not** the binding constraint on this traffic because the
prompts are short. This is the part where I differ from the work order's own
expectation (high hit rate, capacity-bound); the disagreement is on the
record so the data can settle it.

**Agent (35B on the B60, 262144 context, 7211 spare pages ≈ 115k tokens).**
Multi-turn agentic traffic: every turn re-sends the whole conversation plus
new tool output, so the previous turn's prompt is a prefix of the next one's.
Expectation: **hits on most requests, cached share of prompt tokens above
70%**, misses concentrated in two places — the front (a system block with a
timestamp or a rewritten early message, hit 0 on a long prompt) and a
mid-conversation rewrite. Capacity fine.

**Distribution.** Skewed either way: a few cold long prompts carry most of
the prefilled tokens on the agent; on the coder, most requests are small and
cold. A mean would hide both, hence the distribution.

**Misses, classified from divergence position alone** (the journal has no
prompt text): hit 0 on a prompt longer than one chunk = front miss; hit at a
grid multiple well short of the previous request's length = mid divergence;
hit ≈ previous length rounded down to the grid = the normal append. I expect
front misses to be the largest class by *tokens lost* on the agent, and
"first request of a fresh session" to dominate misses by count on both.

**Eviction policy.** Read from the code before the numbers, stated as found.

---

# Result (2026-08-30, same day)

## (1) Hit rate over real traffic: there is no real traffic

Both unit journals were read end to end — `arcint.service` since its creation
on 2026-08-29 15:19, `arcint-agent.service` since 11:26 today. **Every prefill
line in them is a benchmark.** The coder's journal holds the 08-29 evening
depth sweep (25842 / 103242 / 206393 tokens, twice) and six warm/cold pairs at
53.5k — the six "hits", all 99.4%, all repeats of the same prompt — then one
282-token request at 20:26 and nothing since: **twenty hours without a
request**. The agent's journal holds no prefill line at all. The pi agents
were on `llama-agent` (llama.cpp) until 11:15 today, and the coder endpoint
has had no consumer; this session's own A/B runs went to the binary directly
and never through a unit. So the hit rate, the distribution and the miss
classes cannot be computed from anything that exists, and neither
pre-registered expectation — the work order's or mine — can be scored.

What can be stated now is the mechanism and the capacity, from the code and
the boot lines, so that the first real traffic is read correctly.

## (3) Policy and capacity, as found

**Policy: LRU, byte-budgeted on the GDN rows, pages by refcount.** `lookup`
walks the prompt in 32-token hash blocks and takes the *deepest* stored entry
whose tokens compare equal (a hash hit that disagrees on tokens is counted as
a collision and skipped), then splices that entry to the front of the list.
`insert` pushes to the front. `evict_oldest` pops the back — so the entry that
goes first is the least recently *hit or inserted*, not the shortest or the
longest. The byte budget (`--prefix-cache-mib`) counts only the GDN checkpoint
row of each entry (`estimated_paged_state_bytes`: `la_row_bytes` plus 64 per
state name; the MTP head's state when present). **KV pages are not in the
budget at all**: an entry holds a refcount on its prefix's pages in the pool,
complete pages are shared between entries and with the live lane (copy-on-
write, DESIGN §4.2), and when a lane cannot get pages the backend evicts
entries LRU-first until it can (`ensure_blocks`). One snapshot per request,
taken at the last multiple of the prefill chunk below the prompt length — so
the grid is **2048 tokens on the agent and 128 on the coder**, because the
A770 clamped the chunk.

**Capacity, per endpoint, from the boot lines:**

| | coder (A770) | agent (B60) |
|---|---|---|
| GDN row per entry | 95.6 MiB | 95.6 MiB |
| cache budget | 2048 MiB → **~21 entries** | 8192 MiB → **~85 entries** |
| pool | 8357 pages × 16 tok = 133k tokens | 23593 pages = 377k tokens |
| per lane / spare | 6146 / **2211 (35k tokens)** | 16386 / **7207 (115k tokens)** |
| snapshot grid | 128 | 2048 |

**Branch arithmetic.** A rewind keeps tokens 0..k and diverges after; two
branches share the pages below k and each pins its own tail. The pages, not
the entry budget, bind first: the branches' *unique* tails plus the live
request's growth must fit in the pool. On the coder, a 98k conversation with
two branches diverging at 60k needs ~3750 + 2×2400 = 8550 pages against 8357
— the older branch is evicted; diverging at 50k with 40k tails just fits. On
the agent the same shapes fit with 115k tokens of spare to hold tails. So the
work order's expectation — coder capacity-bound — is **true for branching
near full context on the coder**, and moot if the coder's traffic is one-shot
as described; the agent, where the rewind command is actually used, has room
for two or three long branches and dozens of short ones.

**Under LRU, branching behaves well:** switching to branch B hits B's entry if
it survived and re-fronts it; if B's entry was evicted, the deepest surviving
ancestor below k still hits, and only B's tail is recomputed. Longest-prefix
eviction would do worse here (it would keep the longest branch and evict the
one being returned to). The failure mode is pool pressure evicting the entry
of the branch about to be resumed while a long unrelated request runs.

## (2) and (4): waiting on traffic

Misses cannot be classified without requests, and the resume-after-mismatch
question (llama.cpp's `--cache-reuse`) is a solution to a problem whose size
is unknown; it stays on paper. The productive next step is **replay**: real
conversation records (the pi agent's transcripts, or a proxy's request log)
replayed in order against the endpoints, which yields the hit rate, the
divergence positions and the branch behaviour under the real pool sizes, and
does so in an afternoon rather than after a month of use. The infra question
of where such records live is out to the infrastructure agent.

---

# Replay of real transcripts (same day, tools/cachesim.py)

The operator's pi sessions exist as trees — every event carries a `parentId`,
so rewinds are explicit branches, assistant turns record the prompt length
the backend saw (`usage.input`), and compactions are events. 80 sessions,
8093 assistant turns, prompts to 595k tokens, 28 compactions, 6 branch
points. `tools/cachesim.py` reconstructs each turn's prompt as the tree path,
estimates tokens per message from characters with a per-session least-squares
fit against `usage.input` (19 sessions carry enough of it to fit: **2.63
chars/token, 13.6k tokens of system prompt and tool schema**; the rest take
those medians), and replays arcint's policy exactly as read — grid snapshot,
deepest-prefix lookup, LRU, entry budget, pool as the union of pinned tokens.
The corpus is agent-shaped traffic that ran on other backends; the coder's
one-shot traffic is not in it, so the coder row below is "the coder's
configuration under agent traffic", which is the branch case the work order
asked about, not the coder's real load.

## Hit rate and where the prefill goes

| | coder config (A770, grid 128, 21 entries, 133k pool, 98304) | agent config (B60, grid 2048, 85 entries, 377k pool, 262144) |
|---|---|---|
| turns servable / over n_ctx | 4900 / **3193** | 7133 / **960** |
| hits, by request | 95.5% | **97.3%** |
| prompt tokens served from cache | 94.6% | **96.0%** |
| prefilled per request p50 / p90 / p99 | 386 / 2863 / 67955 | **1583** / 3455 / 71776 |
| total prefilled | 12.6M | 24.0M |
| — normal append turns | 4.1M (33%) | **13.4M (56%)** |
| — front miss, session's entries **evicted by pool pressure** | 6.7M (53%), 110 requests | **8.8M (37%), 82 requests** |
| — first turn of a session | 1.1M | 1.1M |
| — front miss after compaction | 0.55M, 30 | 0.44M, 27 |
| — front changed (rewritten early message) | 0 | 0 |
| — mid (rewind to a branch) | 3 requests, 466 tokens | 2 requests, 1969 tokens |

**Scoring the pre-registrations.** Both were right about the rate — hits on
nearly every request, over 95% of prompt tokens from cache — and **both were
wrong about capacity in the same direction**: the work order expected the
coder capacity-bound and the agent fine; I expected neither bound. In the
replay the agent is bound. Its 82 pool-eviction misses average 107k tokens
each and are 37% of all its prefill work: the operator interleaves sessions,
two or three long ones do not fit a 377k-token pool together, and switching
back to an evicted one re-prefills it whole (~35 s at 3000 t/s). The rewrite
causes the order named — a timestamp in the system block, an early message
edited — do not occur in this corpus at all (front-changed = 0), with one
caveat: the simulation's system block is a constant per session, so a
timestamp that pi inserts at request time would be invisible here and would
show up in the real journal as a hit rate near zero. The six real hits on
record (99.4%, repeated prompts) say the block was stable in those runs.
Rewinds are negligible under LRU, as predicted in (3).

## Two levers, both ours, priced by the replay

1. **The snapshot grid.** A snapshot is taken at the last multiple of the
   prefill chunk, so on the agent every append turn re-prefills up to 2047
   tokens it already had: **1933 tokens per append turn against 974 at a
   128-token grid**. Snapshotting at 128 (splitting the last chunk's forward at
   the grid boundary: one extra sub-chunk forward per request) takes the
   agent's total prefill from **24.0M to 17.3M tokens, −28%**, p50 per request
   1583 → 403. Rung zero, small, and the cost side — one extra small forward
   per request — is a measurement before it ships.
2. **Pool capacity for evicted sessions.** 37% of the agent's prefill is
   re-prefilling sessions the pool could not hold. The pool total is fixed by
   the card (17.38 GiB of weights leave 4.08 GiB of KV on the B60); changing
   the n_ctx/spare split does not change the total, and 960 turns already
   exceed 262144. The lever that fits the numbers is a **host tier**: an
   evicted entry's pages copied to host RAM (11.3 KiB/token; 150k tokens is
   1.7 GB, ~0.12 s back over the link at the measured 14 GB/s) instead of
   dropped, against ~35 s of re-prefill. A design item, not a flag.

**Not levers:** a frontier-only entry policy (drop superseded ancestors)
changes no hit rate — the 85-entry budget never bound — and only shrinks the
GDN-row budget from 8 GiB to ~1.5 GiB of host RAM (15 live entries at most);
worth taking for the RAM, not for speed. And (4), resume-after-mismatch:
no observed miss class is a near-front mismatch, so it is a solution to a
problem this traffic does not have, and it stays on paper.

## Caveats
Token counts are character-calibrated on 19 sessions; prompt sizes above the
endpoints' context reflect the cloud backends these sessions actually ran on;
the coder row is agent traffic on the coder's configuration. The first week
of real journal lines replaces all of it.

---

# Follow-up: both levers built and gated (same day)

**Host tier:** implemented (`--cache-host-mib`), gated with a capped pool —
three prompts, the first evicted to the host and promoted back in 0.02 s for
45 MiB, answer byte-identical to its cold one; with the tier off the same
request prefills cold. DESIGN §4.4 has the table. Next: the 1.7 GB case on
the agent's real pool.

**Snapshot grid:** `--cache-grid 128` is byte-exact across cuts inside a chunk
(the absolute-grid rule was the dense model's, not the paged path's), but a
continuation from a fine-grid hit carries ~0.45 s of fixed cost whose
mechanism is open; the arm lost on wall time. Default stays the chunk.
DESIGN 7.0.2l.
