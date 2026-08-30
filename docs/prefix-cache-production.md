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
