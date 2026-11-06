# mtp-head-fit — refit the MTP head against what the served model writes

**Open.** Step one measured (2026-11-06); no fitting tool yet.

## Charter

The coder's draft head (`blk.40` in its GGUF, provenance not recorded) sets how many tokens a verify yields. One lever: raise acceptance by fitting the head on the coder's own output. The served path does not change. The data is generated, never session content, and the corpus and any fitted head stay private: only this method is published (operator, 2026-11-06).

## Reference to follow

HASS (arXiv 2408.15766, `paper`): ranking-distillation loss and context-aligned training on self-generated prefixes, no added inference cost, 2.81-4.05x from acceptance alone, on other models and hardware. EAGLE-3 (arXiv 2503.01840, `paper`). HyperQwen lists MTP head fine-tuning among what it tried and lost (`docs/optimizations.md`, `doc-measured`, no detail read here). Neither has been tried in arcint: no verdict in either direction.

## Gate

Held out on generated text not used in the fit: first-draft acceptance up by at least 5 points on code and thinking, measured with `ARCINT_MTP_TRACE`; then served decode on the deployed flags per the answer-level bar in `CLAUDE.md`, acceptance task included. A verify still checks every draft, so answers cannot get worse through the head; the task runs anyway.

## Current state (`measured-here`, A770, 2000 MHz, the coder's file, arcint 1.5.4 + 0101, 24 generated prompts, one run an arm, q8_0:q4_0)

Kept given reached, by draft position, greedy with 4 drafts a verify:

| Text | rounds | 1 | 2 | 3 | 4 | tokens a verify |
|---|---:|---:|---:|---:|---:|---:|
| code | 293 | 90 % | 81 % | 78 % | 68 % | 3.59 |
| tool calls | 55 | 89 % | 82 % | 82 % | 79 % | 3.69 |
| thinking | 1,144 | 80 % | 74 % | 75 % | 72 % | 3.15 |
| prose | 1,134 | 69 % | 56 % | 53 % | 43 % | 2.36 |

Deployed sampling with the cost policy: first draft 92, 88, 78, 71 % in the same order, so sampling costs the first draft nothing here; tokens a verify 3.52, 3.45, 2.65, 2.03, at 3.0, 3.0, 2.2, 1.5 drafts a round, 29 ms a round against 37.

What a fit could buy (`inferred`, arithmetic on the rows above, a verify's cost held): 10 points at every position takes thinking from 3.15 to 3.83 tokens a verify (+21 %) and prose from 2.36 to 2.81 (+19 %); code has less room. Whether a fit reaches 10 points is unknown.

## Next

1. A corpus: the coder on the battery and public code prompts, tokens and the target's hidden rows; size to be set by a learning curve, not assumed.
2. A fitting tool outside the served path; the head's shape from the GGUF's `blk.40`. Where it trains is open: no Spark needed unless the curve says so.
3. The first fit, on one split, against the gate. Stop there if it misses.

## Where it lives

The measurement's scripts move to `tools/` with the corpus step. `ARCINT_MTP_TRACE`: `src/exec/mtp_trace.h`.
