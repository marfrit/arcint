# expert-foresight — stage the experts the next tokens will need, before they need them

**Status: open (operator, 2026-10-29).** Flash-Next on the B60; the A770 or a CPU core as a predictor's home.

## The lever

The expert cache decides which ~24 % of Flash-Next's experts sit in the B60's slots (5,915 slots at the unit's 11,600
MiB, ~123 of 512 a layer) from **past** routing. Replaying the 2026-10-15 routing trace through the cache's own
mechanism (`research-strata-history.md`, `measured-here` offline):

- every past-based policy (Strata's adapt, LFRU, windows, the best fixed set in hindsight) lands at ~89-90 % of
  decode's routed experts on the card;
- a planner that **knows** the next 24 tokens' routing reaches **96.6 %** inside the same cadence, swap cap and
  landing (next 6: 91.8 %, next 96: 97.6 %; no cap: 99.8 %).

Those ~7 points are foresight. On the card, the expert share and decode move together (today: 71.5 % -> 81.2 % of
routed experts came with 31.08 -> 34.30 t/s, `measured-here`, `research-concept-mining.md`), so a predictor that
recovers part of the oracle's 7 points is worth several percent of decode.

## Where foresight can come from

1. **The MTP window.** The verify pass computes every drafted token's routing at every layer; the drafts are the
   model's own guess of the next tokens. Free, already on the card.
2. **The prompt.** The experts a prompt routes most are likely to recur in its answer. Free at decode's start.
3. **A small learned predictor** ("a matrix, no LLM magic"): per layer, recent routing (+ 1, + 2) -> the expert
   probabilities for the next ~20 tokens; a linear map or a tiny MLP, trained offline on our own traces. Its output
   is a swap plan of a few hundred expert ids, so it is **off the critical path**: late means today's behaviour, not
   a stall. That makes the A770 (or a CPU core) a natural home: the B60 serves, the predictor plans.

Prior art: Strata predicted one layer ahead within a token (recall 27-34 %, precision ~8 %) and dropped its
foresight slots (decode x0.91-1.00, `paper`, 14f9e4c5, ef362cc4); cross-token foresight is untested by Strata and by
us. Published work on learned expert prefetching for MoE offloading exists; survey it (and read the code where there
is code) before stage 3.

## Stages

0. **Trace:** a diagnostic patch that records decode and prefill routing (and the MTP drafts' routing) per token and
   layer from the served unit (the 2026-10-15 trace knob is not in the served series any more).
1. **Replay with predictors** (`tools/expert_cache_replay.py`): past-based (the baseline), MTP-draft routing, prompt
   prior, both, and a trained linear/small model; hit rate inside the real swap budget, against the oracle. Train on
   some requests, score on others.
2. **Build what the replay justifies**: the planner fed with predicted routing; the predictor where its measured cost
   is lowest (B60, A770 or CPU).
3. **Served A/B** under the adoption rule.

Alongside: re-measure the two-card layer split on today's stack (`layer-split-two-cards.md`; CPU-computed misses
arrived after its 2026-10-08 numbers and keep the A770's misses off its x4 link).

## Gate (operator's rule, 2026-10-29)

Adopted if decode (or prefill) rises at all and no other target variable degrades: the other phase within its
run-to-run spread, the answers at `CLAUDE.md`'s answer-level bar, peak VRAM and context unchanged.

## What we learned and what would change it

Every outcome is recorded here, with its evidence class: what a predictor reached against the oracle, what it cost,
and the conditions under which a result would change - more VRAM (a larger card raises the slot share and the
baseline), a faster link for a second card (the A770 sits on PCIe 3.0 x4, 1.83 GB/s), a larger swap budget, or a
better predictor. A ceiling found here is a boundary with its numbers, for whoever comes back with the hardware or the
idea that moves it.
