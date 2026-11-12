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
- `arc-b580/`: the same method on one Arc inference fork (208 commits), ranked and audited against our code.
- `ipex-llm/`: the same method on Intel's IPEX-LLM (4,113 commits, plus its llama.cpp wheel read as a binary, 2026-11-05).

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
  prize is ~1-2 % (Strata's cost-aware width measured +4.7 % on one GB10 in its first run and 0.996x and 1.02x in its own later check, which did not reproduce it; Infernix's adaptive MTP +2.0-2.1 % geomean on two models and a loss on a third; all `paper`, see `research-mtp-width-policy.md`).

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

## Strata pull request #4185: the mixed-IQ3 decode kernels for Battlemage (2026-11-12)

A narrow read of one upstream change (ported into the reference on 2026-10-31). The occasion was the reference's
own two-card Arc figures below, which sit well above what arcint serves on one card. The reference checkout is now at
`5f0ae5b5` (2026-11-11). The question was whether the change holds a kernel form arcint lacks. It holds one, the
IQ4 codebook as a select tree in the decode matvec, and that one is slower here than what arcint ships.

**What the reference states** (`paper`: `docs/INTEL.md`, "Mixed-IQ3 decode kernels for Battlemage"). Two Arc Pro
B65, Flash-Next GSQ-RCO IQ3_S, all experts resident, a layer split, `--spec 2 --pipeline-windows 2`, int8 KV, an
ahead-of-time build for the card: decode +9.7 % on a 4K prompt and +11.1 % on a free-form answer, the same tokens,
prompt speed unchanged. The build option alone gives +5.9 %. Fourteen runtime controls come with it; six are on by
default on a Battlemage card, and the other eight "measured neutral or slightly slower on that configuration (an
internal measurement; the single-control table is not published)".

**What it is at source** (`code`):

- The build option `STRATA_SYCL_IQ_DECODE_FAST` (`sycl/CMakeLists.txt`, default on) sets six definitions. They
  select three integer forms in `sycl/src/kernels/cuda/iq_kernels.dp.cpp` and
  `sycl/src/kernels/cuda/native_mmvq.dp.cpp`:
  - **IQ3_S signs** (`iq3s_sign_mask`, `iq3s_apply_sign`): four sign bits to four byte masks by one multiply,
    `(nibble * 0x00204081) & 0x01010101`, then `(grid ^ sign) + (sign & 0x01010101)`.
  - **The packed Q2 word** (`q2_decode8`): four 2-bit fields to four signed bytes with shifts, masks and one
    add/xor, in place of four byte permutes a 16-bit word.
  - **The IQ4 codebook** (`sycl/include/strata/kernels/iq4_lut.hpp`, `iq4_lut4_swar`): the 16 values as a
    select tree over the code's four bit planes, four bytes a word, 15 `bitselect`s and 4 masks, no memory
    read. For IQ4_NL (`iq4nl_lut4`) and the wide dense matvec (`iq4_lut4`) it replaces four register words with
    a compare-select and a shift per nibble; for the IQ4_XS dot product it replaces the byte-permute lookup of
    llama.cpp's CUDA path (`get_int_from_table_16`).
- The six default-on controls select kernel shapes, not arithmetic: two weight rows that share each
  activation read (Q4_K / Q5_K dense projections of fixed shapes, IQ3_S / IQ2_S / IQ3_XXS gate/up), the reuse of
  a low-rank output in the hyper-connection, and a merge of only the active chunks in batched QSA decode.
- The reference has no IQ4 codebook in local memory. Its local-memory options stage other things: the IQ2_S /
  IQ3_S / IQ3_XXS grids (`STRATA_GRID_SLM`, a compile-time switch, off) and the q8_1 activations of the IQ4_NL
  down projection (`STRATA_IQ4NL_DOWN_SLM`, one of the eight opt-in controls, off).

**arcint against it:**

| part of #4185 | arcint | evidence | disposition |
|---|---|---|---|
| IQ3_S sign arithmetic | has it: the spread of four sign bits by the same multiply constant is patch 0053 (`lb_sgnb4`, in the series since 2026-10-15, served +0.9 % on the B60 together with the Q2 spread); the apply form `(g ^ m) + (m & 0x01010101)` is `lb_sgn4` | `code`, `measured-here` | done |
| packed Q2 word | has a word form for the same step: patch 0053 spreads the fields with two multiplies (`* 0x1001`, `* 0x40040`). The arithmetic differs from the reference's two-step shift spread by a few integer operations a word. 0053 timed its multiplies against arcint's own earlier shift form (`-DQ2_SPREAD_SHIFT`), not against the reference's | `code` | present in another form; the reference's form not timed |
| IQ4 codebook as a select tree | not in the decode matvec; tested, below: correct, slower than the local-memory table arcint ships (patch 0035). The tree itself is in arcint's prompt path (`id_kv4`, patch 0103, default; `fq_kv4`, patch 0115, opt-in) | `measured-here`, `code` | out on the B60 |
| several rows a sub-group with shared activation reads (`*_GU_ROW_REUSE`, `MMVQ_SHAPE_RPW`) | has the mechanism at 4 rows a shared read, against the reference's 2: a sub-group of 16 takes `N_DST` 4 rows and loads each activation block once for all of them, up to 4 columns a pass, in `mul_mv_iq_q8_1.cl` (patch 0002; 0022 for the low-bit types) and in `mul_mv_kq_q8_1.cl` (patch 0001; Q4_K, Q5_K, Q6_K). 8 rows in the low-bit MUL_MAT_ID were faster in isolation and ~1 % slower served (0053's record) | `code`, `measured-here` | done; the reference's 2-row shape was not built |
| reuse of the low-rank output in the hyper-connection (`GR_LO_REUSE`) | the reference's up kernel recomputed the low-rank SiLU (sum of the down partials, the division, the activation) in every work-group; the control computes it once in the down reduce. arcint's graph computes it once already: the scale and the activation are one fused kernel (`kernel_scale_act_f32`) whose output tensor the up projection reads | `code` | nothing to take |
| merge of the active chunks only (`ATTN_MERGE_ACTIVE`) | the reference's decode attention writes partial results into a chunk grid sized for the step's id-list capacity, and its merge used to walk the whole grid; the control bounds the walk to the chunks the step filled and issues eight partial reads together. arcint's decode attention derives its split count from the step's KV rows at dispatch, so below its 256-split clamp its combine has no empty split to skip | `code` | nothing to take |

**The IQ4 lookup, measured** (`measured-here`, 2026-11-12, the B60, `test-backend-ops`, a tool build of the pin
with the served series through 0117 plus a measurement switch in `mul_mv_iq_q8_1.cl`, kernels compiled with
`GGML_OPENCL_IQ_OPTS`). The switch `IQ4_KV_MUX` replaces `kv4`:

```c
#define KVM_K(v) ((uint)(v) * 0x01010101u)
#define KVM_SEL(x, y, m) bitselect((uint)(x), (uint)(y), (uint)(m))   // -DIQ4_MUX_XOR: x ^ ((x ^ y) & m)
static inline uint kvm_mask(uint n, int b) { const uint m = (n >> b) & 0x01010101u; return (m << 8) - m; }
static inline uint kv4_mux(uint n) {
    const uint m0 = kvm_mask(n, 0), m1 = kvm_mask(n, 1), m2 = kvm_mask(n, 2), m3 = kvm_mask(n, 3);
    const uint a0 = KVM_SEL(KVM_K(0x81u), KVM_K(0x98u), m0), a1 = KVM_SEL(KVM_K(0xADu), KVM_K(0xBFu), m0);
    const uint a2 = KVM_SEL(KVM_K(0xCFu), KVM_K(0xDDu), m0), a3 = KVM_SEL(KVM_K(0xEAu), KVM_K(0xF6u), m0);
    const uint a4 = KVM_SEL(KVM_K(0x01u), KVM_K(0x0Du), m0), a5 = KVM_SEL(KVM_K(0x19u), KVM_K(0x26u), m0);
    const uint a6 = KVM_SEL(KVM_K(0x35u), KVM_K(0x45u), m0), a7 = KVM_SEL(KVM_K(0x59u), KVM_K(0x71u), m0);
    const uint b0 = KVM_SEL(a0, a1, m1), b1 = KVM_SEL(a2, a3, m1), b2 = KVM_SEL(a4, a5, m1), b3 = KVM_SEL(a6, a7, m1);
    return KVM_SEL(KVM_SEL(b0, b1, m2), KVM_SEL(b2, b3, m2), m3);
}
```

The identity against the table holds for all 65,536 two-byte code words (checked off the card).

- **Correctness:** MUL_MAT 201/201 in the shipped form, the select tree and its xor form; MUL_MAT_ID 51/51 with
  the select tree. The red case (codes 0 and 1 swapped) fails 32 of 201 MUL_MAT and 3 of 51 MUL_MAT_ID cases, so
  the suite sees the lookup. The compare-select form had no correctness leg in this window, and the xor form none
  on MUL_MAT_ID; both are timing arms only.
- **Timing**, median us/run of the perf mode, arms interleaved, one process a leg. Repeats: MUL_MAT 3 for the
  shipped form and the select tree, 2 for the other two; MUL_MAT_ID 2 each. The shipped form's spread is under
  1.7 %.

  | case | shipped (local table) | select tree | select tree, xor form | compare-select (`IQ4_KV_SELECT`) |
  |---|---|---|---|---|
  | MUL_MAT iq4_nl 2560 x 6144, 1 column | 20.2 us | +23.5 % | +61.3 % | +104.6 % |
  | MUL_MAT iq4_nl 4096 x 14336, 1 column | 78.0 | +20.9 % | +42.0 % | +79.4 % |
  | MUL_MAT iq4_nl 6144 x 2560, 1 column | 21.8 | +41.8 % | +62.7 % | +105.2 % |
  | MUL_MAT iq4_xs 2560 x 6144, 1 column | 22.0 | +16.1 % | +46.0 % | +89.5 % |
  | MUL_MAT iq4_xs 4096 x 14336, 1 column | 92.1 | +6.6 % | +22.7 % | +56.6 % |
  | MUL_MAT iq4_xs 6144 x 2560, 1 column | 24.9 | +27.6 % | +46.2 % | +83.6 % |
  | MUL_MAT_ID iq4_xs, 128 experts top 8, 768 rows, 1 / 4 / 8 tokens | 21.9 / 75.5 / 160.1 | +26.4 / +12.3 / +6.4 % | +49.8 / +34.2 / +25.4 % | +76.6 / +59.9 / +48.9 % |
  | MUL_MAT_ID iq4_xs, 32 experts top 4, 1792 rows, 1 / 4 / 8 tokens | 24.8 / 83.2 / 175.8 | +22.4 / +20.6 / +11.5 % | +46.4 / +42.8 / +32.7 % | +72.1 / +70.2 / +57.7 % |

  The six 3-column MUL_MAT cases stay within 0.6 % in every arm: they take the few-column int8 product (patch
  0115), which has its own table and runs no `kv4` form.

**Reading.** The select tree beats the compare-select form by a wide margin here too, which is the direction the
reference reports. Both lose to the local-memory table in every decode-shaped case, by 6 to 42 % for the select
tree. The reference's gain is measured against a compare-select baseline (IQ4_NL, the wide dense matvec) and a
byte-permute baseline (the IQ4_XS dot product), and it publishes no number per form; arcint's baseline has been the local-memory table since
patch 0035. No served arm was run: no kernel got faster.

**How this build differs from the reference's**, so that the result is read as what it is:

- OpenCL C through the compute runtime's online compiler, against SYCL with an ahead-of-time build for the card;
- arcint's kernel loop and work-group shape around the lookup, not Strata's lane kernels;
- the baseline is a codebook in local memory, a form the reference does not have and so did not compare (`code`);
- an operator suite on synthetic shapes on one B60, not a served IQ3_S Flash-Next on two B65;
- only the IQ4 lookup of the build option. The option's +5.9 % is the three forms together and is not split by
  form in the reference's record.

It is a verdict on this lookup form in arcint's decode matvec on the B60. It is not a verdict on #4185, and the
A770 was not timed.

The same select tree is the default of arcint's int8 prompt path on Alchemist (`id_kv4`, patch 0103), where a
table read would cost a gather per lane. There it was timed against the table form (`-DID_KV_TABLE`, the
reference's byte-pair table, in constant memory): with 0103 alone 1,636 t/s against 1,680 with the table; with
0104, 2,151 against 2,149 at two row sets a sub-group and 2,042 against 2,092 at three (`measured-here`, the
coder's file on the A770, the series records of 0103 and 0104). That tree is the xor form; the `bitselect` form
and a local-memory table were not timed there. Patch 0115's few-column product keeps a table and has the tree as
an opt-in (`-DFEWQ_IQ4_MUX`, not timed in its record). So the tree has lost or tied wherever it was timed here.

**Also in the newer checkout** (`code`, `paper` for its numbers): the SYCL port carries the layer split and runs
one conversation's draft windows overlapped across two cards (`--pipeline-windows 2`); see
`layer-split-two-cards.md`.
