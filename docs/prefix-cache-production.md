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
