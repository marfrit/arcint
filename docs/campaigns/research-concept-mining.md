# research-concept-mining — every commit of three engines, ranked for arcint, audited against our code (2026-10-29)

The full commit histories of Strata (4,690 commits), Infernix (1,788) and FreeToken (190), read for concepts
that could make arcint faster, then checked against arcint's own source. This document is the method, the
filter and the order; the material is in `research-concept-mining/`:

- `candidates.md`: the ranked concept list (top 10, next 10, lower ranks), each with its evidence commits,
  the numbers as the commits state them, the cost and an experiment;
- `audit.md`: the kept candidates against arcint's code (`qfndev` 3f449ff and llama.cpp at the pin with the
  served series 0001-0067): what arcint has, how its build differs, the quality class, the units, the
  pricing measurement, an implementation sketch with red-first tests;
- `design-layer-major.md`: how candidate 1 would be built in the llama engine, and why its prize here is small.

Evidence classes: an engine's numbers are `paper` for us; what was read in a source tree is `code`; nothing
in this survey was run on our cards. Every row has to be priced on the cards before it counts.

## Method

1. **Pass 1, classification.** A Qwen3.8-27B agent read every commit message in chunks of 20 and wrote one
   record a commit: area, kind, status (adopted, opt-in, dropped, reverted, ...), concept name, the measured
   numbers as stated, and an arcint score 0-3. Result: 4,723 / 1,185 / 662 / 98 commits at 0 / 1 / 2 / 3.
2. **Pass 1b and 2, ranking and research.** DeepSeek 4.1 Flash with a 1M-token context read all records at
   2-3 and the per-concept notes in one view, merged the same idea across the three engines, re-read the
   evidence commits of the top 20 (five records corrected, among them two defaults the engine later made
   opt-in and one revert), and researched the top 10 in the engines' source (file:line).
3. **Wave 0, the audit.** The same model read arcint's tree and the patched llama.cpp for every candidate that
   does not trade quality for speed. Its arcint claims were spot-checked against the code
   (the CPU-miss cap `GGML_OPENCL_CPU_MISS_MAX`, QSA in `qwen4exp.cpp`, the stage-5 chunk sweep in
   `flash-next-llama-engine.md`, the patch numbers): all held.

## The quality filter (operator, 2026-10-29)

In scope are the levers that keep the answer: **exact** ones (the same tokens, or bit-identical by design)
and **rounding-only** ones (reordered sums, a quantized activation), the latter judged by `CLAUDE.md`'s
answer-level bar. Out of scope, because they change the answer by design: skipping low-ranked missed experts
and the route prior (candidate 4), the screened vocabulary head (8, its normalizer is 0.15-0.19 nats off),
the late-layer approximate prefill (12) and a lower-precision K cache (17).

## What the audit found

| candidate | arcint (`code`) | class | units | disposition |
|---|---|---|---|---|
| 7 xe doorbell (uncached loads, SVM-scope release/acquire) | has it: patch 0056 | exact | Flash-Next | done |
| 9 tier overlap (async swaps behind the graph) | has it: patch 0042 (+2.4 % decode, `measured-here`) | exact | Flash-Next | done |
| 10 PCIe copy discipline | mostly: 0047, 0042, 0046, 0032; one leftover (the staging cap at depth) | exact | all | price the leftover |
| 6 CPU-miss AVX2 rows | mostly: 0063's multi-token named-accumulator rows; the prefetch lacks | exact | Flash-Next | price the prefetch |
| 18 decode-once expert matvecs | the reference's mechanism was tested here and lost (30.5 -> 29.5 t/s, `measured-here`); 0031 dropped | exact | Flash-Next | out (a new kernel shape would be a new candidate) |
| 1 layer-major prefill | partly: llama.cpp already walks the layers inside each ubatch; the bank stream copies each bank tensor once a ubatch | rounding-only | Flash-Next, coder | demoted: the across-ubatch saving measured +3.1 % at a 8,192 chunk (`measured-here`, stage 5); the remaining prize is expert-tile padding, measured first |
| 2 int8 DPAS prompt expert GEMM | partly: grouped-by-expert DPAS GEMM exists, int8 DPAS only on the dense path | rounding-only | Flash-Next, coder | build after pricing (B70: +12.5-20 %, KL 2e-4, `paper`) |
| 3 PCIe probe + per-layer miss balance | lacks: no probe; the miss split is a fixed 64 pairs a layer | rounding-only | Flash-Next | build after pricing |
| 13 MTP width policy | lacks: fixed width with the min-p stop | rounding-only | all | build after pricing |
| 11 prefill chunk / residency dial | partly: flags, swept by hand | rounding-only | Flash-Next, coder | sweep first |
| 14 per-card expert profile | partly: reads Strata's profile file, no save path | exact | Flash-Next | served A/B of profiles |
| 15 agent prefix reuse | partly: checkpoint restore and longest-prefix reuse; no tokenise-from-prefix | rounding-only | agent, coder, Flash-Next | price the tokenizer's share first |
| 5 QSA select kernels | partly: the score + top-k graph exists; the kernel forms lack | exact | Flash-Next only | price the select's share at 100k+ |
| 16 decode launch fusions | partly: eight fusion patches; qk-norm + rope lacks | exact | coder, Flash-Next | bundle of 2-3, gate at the wall |
| 19 xe device hygiene | partly | exact | all | cheap, correctness class |
| 20 quality instruments (assistant-token mask, noise-twin floor) | partly: the KL tools exist | gate | all | **first**: every rounding-only A/B needs the twin floor |

## Order

1. **Wave 2a, no card:** the quality instruments (20), and two serving fixes found while running this survey
   (a client disconnect is not seen during a prefill or a buffered tool call; a stop signal does not cancel
   a running request).
2. **Wave 1, pricing on the cards with existing knobs** (the commands are in `audit.md`): tile padding at
   2,048 and 8,192 (1, 2), the CPU-miss cap sweep (3), MTP widths (13), profile placement (14); a profiling
   build for the GEMM share (2), the QSA select share (5) and the fusion counts (16); the agent's widths,
   prefix-reuse share and input copies (13, 15, 10).
3. **Wave 2, builds in the priced order**, each its own campaign document with its gate on the record:
   provisionally 2, 3, 13, 11, 14, 15, 5, 6, 10, 16, 1, 19.

## Wave 1, window 1: the CPU-miss cap and the MTP width (2026-10-29)

`measured-here`: the B60, Flash-Next IQ3_S with the unit's flags (11,600 MiB of slots, the CPU misses, the
bank stream, MTP 3 with the min-p 0.5 stop, the draft vocabulary), the dev binary of `c50e5e6`; one fresh
process an arm through `tools/fn_sweep.py` (the capital, Strata's RFC 4180 request at 500 tokens, five
300-token answers, greedy, no thinking); two rounds, the second in reverse order. The capital was right in
every arm. Decode in t/s, the mean of the two rounds, and the spread of the six-answer mean between them:

| arm | strata | hash | compiler | tcp | btree | gc | mean | spread |
|---|---|---|---|---|---|---|---|---|
| miss cap 16 | 34.60 | 26.65 | 34.90 | 33.45 | 30.45 | 31.55 | 31.93 | 0.9 % |
| miss cap 32 | 40.30 | 31.65 | 36.90 | 34.10 | 32.30 | 33.00 | 34.71 | 1.4 % |
| miss cap 64 (the unit; MTP 3) | 40.70 | 32.50 | 36.10 | 31.95 | 32.85 | 32.80 | 34.48 | 1.2 % |
| miss cap 128 | 40.75 | 32.95 | 36.20 | 32.05 | 33.05 | 33.00 | 34.67 | 0.0 % |
| MTP 2 | 35.80 | 34.15 | 34.75 | 34.80 | 33.45 | 31.35 | 34.05 | 1.0 % |
| MTP 4 | 38.65 | 29.80 | 33.45 | 30.65 | 31.70 | 31.25 | 32.58 | 1.5 % |
| MTP 5 | 39.65 | 29.55 | 31.20 | 28.25 | 30.70 | 30.25 | 31.60 | 1.7 % |

The answers' md5s are stable within an arm and differ between most arms (the verify batch and the CPU or
card placement of a miss change the rounding), so the per-answer rates compare different texts; the
six-answer mean is the comparison.

- **Candidate 3 (the miss balance), decode: no prize at this shape.** Caps 32, 64 and 128 lie within the
  window's spread, and 64 and 128 give byte-identical answers: the cap does not bind above 64 here (a verify
  of 4 tokens routes 40 pairs a layer, ~19 % of them missed). Only cap 16 binds, at -7.4 %. Strata's gains
  for this concept came from prefill and from a slow-CPU decode shape; the prefill side here goes through
  the bank stream, not the CPU rows, and was not measured in this window.
- **Candidate 13 (the MTP width): the fixed width 3 is the best fixed arm.** Width 2 is -1.2 %, 4 is
  -5.5 %, 5 is -8.4 % (acceptance 73.5 / 66.6 / 60.3 / 54.8 % for 2 / 3 / 4 / 5). The best width per answer
  is 2 for three prose answers and 3 for the code and two others; an oracle choosing the best width per
  answer would reach 35.33 t/s, +2.5 % over width 3. A real policy collects part of that ceiling, so the
  prize is ~1-2 % (Strata's adaptive width measured +2.0-2.1 % geomean, `paper`).

## Wave 1, window 2: tile padding and the prefill chunk (2026-10-29)

`measured-here`: the B60, Flash-Next IQ3_S, the unit's flags, a 19,196-token corpus prompt, one fresh process
an arm (`tools/fn_sweep.py --prompt-tokens 20000`), peak VRAM from the process's fdinfo. The padding arms ran
a diagnostic build (`w1-probe`, not on qfndev: an env-gated count of the grouped GEMM's per-expert tiles in
`ggml_cl_mul_mat_id_kq`, `GGML_OPENCL_KQ_MM_ID_PAD_DEBUG`); the same binary without the variable printed no
probe line and timed the rate arms.

**Tile padding** (rows the 32-token tiles hold against the rows routed, summed over the prompt's calls):

| chunk | gate/up, bank experts | gate/up, slot experts | down, bank | down, slots |
|---|---|---|---|---|
| 2,048 | +396.5 % | +192.5 % | +154.2 % | +77.7 % |
| 8,192 | +115.8 % | +48.6 % | +45.5 % | +20.1 % |

At 2,048 tokens most experts get a few rows of a tile whose expert slice is dequantized whole; at 8,192 the
same work fills the tiles four times better.

**Prefill rate against the VRAM it needs** (t/s over the 19,196 tokens):

| chunk | slots (MiB) | prefill | peak VRAM (of 24,480) |
|---|---|---|---|
| 2,048 | 11,600 (the unit) | 442.1 | 23,519 MiB |
| 2,048 | 9,600 | 433.0 / 430.6 / 433.3 / 433.4 / 433.3 | 21,577 MiB |
| 4,096 | 9,600 | 532.9 | 22,929 MiB |
| 4,096 | 11,600 | 364.1 | 24,437 MiB |
| 8,192 | 8,000 | 569.9 / 572.1 | 23,632 MiB |
| 8,192 | 9,600 | 499.0 / 432.2 / 393.2 / 417.6 / 417.2 | 24,400 MiB |

A larger chunk pays **+29 %** (8,192 at 8,000 MiB of slots) and +20.5 % (4,096 at 9,600) when the compute
buffer fits; at the card's edge the same chunk scatters by 25 % and loses (the eviction thrash recorded for
Flash-Next). Stage 5's +3.1 % at 8,192 ran at 24.3 GB of VRAM, at that edge, so it did not measure the
mechanism: the audit's demotion of candidate 1 rested on it and is withdrawn.

**What a static setting would cost:** decode on the six answers at 8,000 MiB of slots 31.08 t/s against 34.30
at 11,600 (-9.4 %; the cache's share of routed experts 71.5 against 81.2 %). So the prize is collected only
by giving the slots back for decode: **Strata's lend region** (candidate 11, `code`: the prompt path borrows
the tail slots of the GPU expert cache, `src/core/expert_source.cpp:247`, `:3571`, `:3649`, and refills them
after the prompt with queued copies and one wait, 68072d30, d5733321; Battlemage's borrowing configuration
recommends a 8,192 prefill, 219ae436, `paper`). Refilling ~3.6 GiB of slots over the B60's 14.3 GB/s link is
~0.25 s (inferred) against ~10 s saved on a 20k prompt. Candidate 11's loan moves to the front of Wave 2;
candidate 1's span walk is the same prize without the loan and stays behind it.

## Direction: the same lending for the dense agent and the small MoE (operator, 2026-10-29)

The three served units are the target, and the lever is not specific to the expert cache: a phase's unused memory
lent to the phase that is short of it. The agent (dense, 245,760 cells of KV reserved at load, a 512-token prefill
ubatch) and the coder (a small MoE on the 16 GB A770, 98,304 cells; `--prefill-ubatch 1024` measured +15 %, 2,048
spills 4.1-4.3 GiB to host memory, `measured-here`, `fusion-coder-night.md`) hold KV cells that a shallow
conversation does not use. A KV-tail loan would lend them to a larger prefill ubatch while the depth allows, with
nothing to refill (an unused cell holds nothing). First its price, with static arms and no code: the coder at a
smaller context with ubatch 2,048, the agent with ubatch 1,024 and 2,048 at a smaller context; then a campaign of its
own if it pays.

### Priced: no KV-tail loan for the agent or the coder (2026-10-29)

`measured-here`, the units' dev binary (`c50e5e6`), an 18,646-token corpus prompt, one fresh process an arm
(`tools/prefill_arm.py`), peak VRAM and GTT from the process's fdinfo:

| unit, card | context | ubatch | prefill | peak VRAM / GTT |
|---|---|---|---|---|
| agent, B60 | 245,760 (the unit) | 512 | 870.4 t/s | 24,280 / 56 MiB |
| agent, B60 | 245,760 | 1,024 | 669.0 t/s | 24,438 / 15,940 MiB (spill) |
| agent, B60 | 245,760 | 2,048 | 22.0 t/s | 24,438 / 16,733 MiB (thrash) |
| agent, B60 | 131,072 | 2,048 | 863.9 t/s | 21,927 / 58 MiB |
| coder, A770 | 98,304 (the unit) | 1,024 | 1,134.5 t/s | 16,159 / 32 MiB |
| coder, A770 | 98,304 | 2,048 | 896.3 t/s | 16,250 / 4,251 MiB (spill) |
| coder, A770 | 49,152 | 2,048 | 1,162.1 t/s | 15,911 / 32 MiB |
| coder, A770 | 49,152 | 4,096 | 1,123.4 t/s | 16,273 / 378 MiB |

With the memory to hold it, a larger ubatch gains nothing on the dense agent (863.9 against 870.4) and +2.4 % on the
coder (single legs). The lending lever pays where the work per weight is lumpy (Flash-Next's 512-expert MoE and its
tile padding, +29 %), not here: no KV-tail loan campaign. For these two units the open levers are the MTP width
policy and prefix reuse. Also noted: the agent unit peaks at 24,280 of the B60's 24,480 MiB on this prompt.
