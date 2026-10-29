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
