# research-strata-history — what Strata kept and dropped, and why our LFRU gains nothing (2026-10-26)

Strata's commit history (`origin/main`, 2026-09-27 .. 2026-10-25, ~4,200
commits; the long subjects carry most of the measurements) read for the
concepts it adopted, kept opt-in or dropped, with the question that started
this survey: patch 0055's LFRU (Infernix's policy) measured +1.6 % on the gate
set and -2.1 % on six other prompts, and was dropped in 0.8.1
(`flash-next-llama-engine.md`). Is that a concept issue or an implementation
issue?

Evidence classes: `code` (read in Strata's source), `paper` (Strata's commit
messages, docs and bench READMEs: their measurements, not repeated here),
`measured-here` (an offline replay of our own routing trace, below).

## The expert cache: concept or implementation

**Answer: both, separately.** Every policy that ranks experts by their past
routing hits the same ceiling under our swap budget. A better ranking only
shows once the mechanism moves more experts, and whether that pays depends on
what a swap costs in our engine.

The trace: the 2026-10-15 routing trace (`LLAMA_EXPERT_CACHE_TRACE`, B60,
Flash-Next IQ3_XXS, eight requests, 788 decode steps, 1.22 M routed pairs,
9,724 of 24,576 experts slotted = 39.6 %). It is replayed through
`tools/expert_cache_replay.py`, which tracks the engine within a point (89.8 %
vs 89.0 % measured). That run's routing does not depend on residency, so each
policy replays the same requests. Frames are counted, not bytes, and a swap's
cost is not modelled (`measured-here`, offline):

| policy, inside the engine's mechanism (a plan every 6 tokens, <= 96 swaps, landing 2 steps later) | on the card |
|---|---|
| the load-time profile, never adapted | 63.4 % |
| Strata's adapt (our default) | 89.8 % |
| LFRU (0055), exact form, global pool | 89.8 %, 89.7 %, 90.2 % |
| ranking by the last W steps, W = 6 .. 200 | 82.1 .. 90.0 % |
| the best fixed set per layer, in hindsight | 89.3 % |
| a planner that knows the next 6 / 24 / 96 steps' routing | 91.8 / 96.6 / 97.6 % |
| the same, no swap cap, next 24 steps | 99.8 % |
| Belady's MIN per layer (every miss admitted at once) | 97.3 % |

- **The concept ceiling.** Under the engine's budget, every policy that looks
  back (decayed frequency, LFRU's frequency over recency, plain windows) lands
  within a point of the best fixed set in hindsight. A layer touches ~139
  distinct experts in 24 steps, which fits in its 140-256 slots, so the cache
  could hold the near-future working set. But that set moves faster than past
  routing follows: a planner that knows the next 24 steps reaches 96.6 %
  inside the same cadence, cap and landing. The 7-8 points (misses from ~10 % to
  ~3 %) are foresight, which no past-based policy has.
- **The implementation limit.** With more swap throughput, LFRU's ranking pulls
  ahead of Strata's rule, which saturates:

  | swaps a plan / a plan every / landing | Strata's rule | LFRU |
  |---|---|---|
  | 96 / 6 tokens / 2 steps (the engine) | 89.8 % | 89.8 % |
  | 192 / 6 / 2 | 90.5 % | 91.7 % |
  | 400 / 6 / 2 | 90.6 % | 92.4 % |
  | 400 / 3 / 1 | 91.1 % | 94.3 % |
  | Strata's "busier tier": 160 / every token / decay 0.97, landing 1 / 2 | 92.4 / 90.9 % | |

  So 0055 measured nothing because 96 swaps a plan (about 2 per layer) starve
  every ranking. The swaps a busier plan needs are not free here: LFRU outside
  the gate set ran 1.5-2x the swaps and lost 2.1 % (`measured-here`,
  2026-10-15). What decides it is the price of a swap on the B60: the copy
  path (patch 0042's queue after the graph), its share of the PCIe link, and
  the 2-step landing. The replay does not model that price.
- **Hit rate is not the speed.** Strata says so in several places (`paper`):
  - a run at 0 MiB free had the higher hit rate and was 7x slower;
  - `--pcie-frac` raises the hit rate while decode falls;
  - 92.1 vs 92.3 % hit came with 89.2 vs 74.7 tok/s.

  Ours is confounded too: 82-96 % on the card maps to 29-42 t/s across six
  prompts (`measured-here`, 2026-10-15). Every row above has to be priced in
  t/s on the card before it counts.

## What Strata did with the same question

- **Its policy question was never settled.** `ExpertCache::admit` never evicts,
  "because eviction policy is a measured question". The replay harness has
  fill-only, LRU, LFU-decay and Belady (49c50f49; `tools/bench_eviction.py`,
  `tools/replay_cache_policy.py`), but no commit, doc or bench README records
  a verdict (`paper`). The shipped rule is decayed frequency with a 1.5
  margin: `adapt_every` 4 windows, `adapt_swaps` 96, `adapt_decay` 0.7
  (`src/program/generate.cpp:691-727`, the lambda at `:10328-10354`; `code`).
  Our port runs the same rule every 6 tokens.
- **Prediction was tried and dropped.**
  - Foresight swap (`STRATA_FS_SLOTS`, 14f9e4c5, ef362cc4): a few VRAM slots a
    layer, refilled from the misses and from a next-layer router look-ahead.
    "The smaller cache cost more than the slots returned in every box
    measured", decode x0.91-1.00 in 6 pairs a card (`paper`).
  - The look-ahead's probe: recall 27-34 %, precision ~8 % (`paper`).

  That predictor is one layer ahead within a token. The foresight the replay
  prices is which experts the next ~24 tokens use: another mechanism, not
  tested by either engine.
- **What paid was swapping more or more cheaply**, all `paper`:
  - The busier tier (every window, 160 swaps, decay 0.97): +15.6 % / +14.8 %
    on a DDR3 Xeon with a slow CPU tier; neutral on the B70; it stays a
    per-box calibration.
  - Asynchronous swaps (`--adapt-async`): x1.07-1.11 on a 3090; since 10-24
    the engine measures 64 blocking against 64 async windows and picks.
  - Overlapped swaps (`STRATA_ADAPT_OVERLAP`): +5.5-8.3 % on an A4000, -21 % on
    a 5090 laptop: made default, then opt-in again.
  - Handing the evicted buffer over instead of copying it
    (`STRATA_EXCHANGE_ROTATE`): +5.2 % on a 3060.
  - The RAM copy following routing (`STRATA_RAM_ADAPT`): +16.5 % in the file
    tier.

  The lesson from that list: on a slow-CPU box the tier's throughput is the
  lever, and whether it pays is measured per box.
- **Routing around the misses.** The route tail skip (65268d99; default on
  CUDA since b4e3407b) neither copies nor computes a missed expert that every
  token of the window routes at rank >= 7, without renormalising. Decode
  x1.10-1.24 at KL 0.0058 / 0.0032 against a floor of 0.0006. It is off from
  90 % cached (slower there), and costs the K-quant files more (KL 0.056 ->
  0.086). Cache-conditioned routing (`STRATA_ROUTE_PRIOR`, after arXiv
  2412.00099) and the route-resident swap add speed and KL: "Not recommended"
  (`paper`).

## The rest of the catalogue, filtered for arcint

- **MTP.**
  - `--spec 4 --spec-min-p 0.5` adopted; deeper windows did not pay.
  - Draft-vocabulary subsets are kept per language: a wrong shared subset cost
    Russian decode 68 -> 54 tok/s, and CJK answers ran 15-38 % faster with
    their own subset.
  - DFlash2 is opt-in ("no card has shown DFlash2 winning everywhere yet").
  - A cost-aware window width (`STRATA_SPEC_COST`) is opt-in.
  - Faster start-up adaptation in speculative runs went default, then was
    withdrawn: "the earlier gain did not reproduce" (a4c503d7).
  - All `paper`. Ours: MTP 5 on the coder lost to 4 (`fusion-coder-night.md`).
- **Decode fusions.** Bitwise switches are adopted per architecture behind a
  ~+3 % bar (2725c23f: +9.6-17.8 % from 16-switch sets). Switches that change
  rounding stay opt-in unless a KL gate passes (`paper`). That is the line we
  drew for 0064 and 0071.
- **Prefill.**
  - 8,192-token chunks with MMQ (572 -> 1,290 tok/s on Q2_0).
  - Expert streaming in 1,024-token chunks (+17-28 % on 1-4K prompts).
  - A CPU share of the prompt below 1,024 tokens.
  - All `paper`.
- **Intel.** On Battlemage:
  - fused int8 DPAS prompt experts adopted (B70 +12.5-20 %, KL 2e-4);
  - XMX prompt attention and the Xe GDN kernel discarded (-24 %, -3.5 %);
  - the copy engine for prompt copies adopted.

  On the A770: `--pcie-frac` was inert and `--spec-split` ~7 % slower (`paper`).

## What follows for arcint

1. **Price a swap on the B60 before choosing a policy.** The engine keeps
   Strata's knobs (`LLAMA_EXPERT_CACHE_MAX_SWAPS`, `_ADAPT_EVERY`, `_DECAY`,
   `_MIN_GAIN`). A served Flash-Next A/B of the busier settings with Strata's
   rule (the replay's +0.7 to +2.6 points) prices a swap in t/s without new
   code. Only if that pays does a re-port of LFRU with a larger budget
   (+2.6 to +4.5 points in the replay) have a case.
2. **Landing in one step matters.** Every row with one-step landing beat its
   two-step twin by 1-1.5 points. The 2-step landing is the non-blocking swap's
   two phases (0042); its cost is part of item 1.
3. **Foresight across tokens is untested by anyone.** Seeding the cache from
   the request's own prompt routing at the start of decode would be one such
   mechanism; the drafts of an MTP window are another. It needs a trace with
   prefill routing to price it offline first.
4. **The route tail skip is an answer-changing lever.** It gets judged by the
   answer-level bar; Strata gates it off from 90 % cached, where our gate set
   sits.

## Addendum 2026-10-30: a lossy default taken back

Strata 0.2.7 (1d45da3a, 2026-10-26) turned residency-biased routing on by default
(`STRATA_ROUTE_PRIOR=1.5` on a discrete NVIDIA card of 13 GiB or less): the router
leans toward experts already in VRAM, so it changes which experts compute a token.
Three days later 20a63cf0 (#4402, 2026-10-29) took the default back: user reports
and the project's own agent check (4.5 of 10 tasks with it on, 5.6 with it off) tie
it to repetition loops and weaker results in long agent sessions; with nothing set
the engine routes as 0.2.6 again, and 1.5 stays an opt-in. Users of Flash-Next IQ3_S
on small cards reported misspellings, characters from another script mid-sentence
and loops on the updated builds (`paper`: the commit message and public reports,
not measured here). The route tail skip (rank 7, default on CUDA since 0.1.42, item
4 above) stays on by default.

For arcint this is the quality filter's case in practice (operator, 2026-10-29: only
levers that do not trade quality for speed): an answer-changing lever is opt-in and
passes the answer-level bar and the task battery before it can become a default.
Strata's reference checkout here is at 4b9e05a2 (2026-10-25); upstream was at
870aedd9 (2026-10-29) when this was written.
