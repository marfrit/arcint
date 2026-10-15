# flash-next-llama-engine — Qwen3.8-Flash-Next served by the libllama engine on one Arc card + RAM

**Open.** Stage 2b (expert cache: slots on the card, a USM bank) passed
2026-10-06: prefill 235 t/s at 20k, decode 12.4-13.0 (UD-Q3_K_XL). Stage 3
(2026-10-06): the GSQ-RCO IQ2_XS file runs (patch 0022), 326 t/s prefill;
decode 21.3 t/s plain and 27.0 with MTP 2 after stage 3b, answers right. Strata on the same card
reads 620.5 / 37.2-37.8 (`strata-sycl-b60.md`).

## Charter

Serve Flash-Next (`qwen4exp`, 48 layers, 512 experts of which 10 a token,
the UD-Q3_K_XL GGUF: 52 GiB of experts in IQ3_XXS / IQ4_NL / IQ4_XS / Q8_0,
a 26.8 GiB per-layer embedding table, ~4 GiB of Q8_0 rest) through
`--engine llama` on the B60, the experts that do not fit VRAM in RAM and on
NVMe, with the mechanisms of the roadmap's P1-P6 rebuilt in this engine.

Stage 1: whole layers' experts on the card as far as they fit, the rest
computed by llama.cpp's CPU backend from RAM; the IQ types' matrix kernels
on Intel (they do not exist in ggml-opencl at the pin). Stage 2: Strata's
per-expert cache inside the OpenCL backend's `MUL_MAT_ID` (slots sized per
layer, decayed-usage swaps, non-blocking admission, the miss split over the
link, the doorbell), replacing the per-layer split.

## Reference to follow

- Strata (`~/src/Strata-ref`, `code`; source map and effects in
  `expert-hot-set-lru.md` and `tier-handoff-doorbell.md`, which built the
  same mechanisms on the OpenVINO path): the expert cache
  (`src/core/expert_cache.cpp`, `src/program/generate.cpp:4413-4482`), the
  miss split (`src/core/expert_source.cpp:1614-1696`), the doorbell
  (`src/core/layer.cpp:995`, `src/kernels/cuda/elementwise.cu:186-311`).
- llama.cpp at the pin (`code`): `--n-cpu-moe` / `-ot` place a layer's
  experts per tensor (`src/llama-model-loader.cpp:1235-1260`); the CPU
  backend's `MUL_MAT_ID` touches only the selected experts
  (`ggml/src/ggml-cpu/ggml-cpu.c:1663-1667`); the OpenCL backend declares no
  mmap support, which turns mmap off for the whole model under
  `--load-mode auto` (`src/llama-model.cpp:1512-1519`), and no `offload_op`.
  The IQ dequantizers to follow: `ggml/src/ggml-quants.c`
  (`dequantize_row_iq3_xxs`, `_iq4_nl`, `_iq4_xs`) and the CUDA MMVQ dots
  (`ggml/src/ggml-cuda/vecdotq.cuh`).

## Gate

On the B60, the Flash-Next GGUF from NVMe, against the stage's baseline arm
(stage 1: every expert on the CPU, `-ot exps=CPU --load-mode mmap`) in the
same window:
- the answer-level bar (`CLAUDE.md`): the capital check and the 20,085-token
  needle answered; mean KL against the baseline arm's reference within 0.03
  nats of the baseline, top-1 agreement down by at most 1 point;
- per-phase speed: decode faster at 4k and 20k, prefill within the spread or
  faster;
- reported beside the rate: VRAM and RAM in use, the experts' split, and
  for stage 2 the GPU hit share and the share of misses over the link.

For scale (not a gate): the OpenVINO path's Flash-Next on the B60 (`d48q8`,
patches 0076/0077, `measured-here` 2026-10-02) decodes 12.7 t/s after the
20,085-token needle and 16.0 t/s on the 500-token long answer.

## Current state

- Stage 1 built (`contrib/llama.cpp/patches/0002`: IQ3_XXS / IQ4_XS / IQ4_NL
  matvec and GEMM on Intel; arcint `--llama-cpu-moe N --llama-threads N`,
  `qwen4exp` admitted). `measured-here` 2026-10-03, B60, the GGUF from NVMe:
  - llama-bench (512-token prefill / 64 decode tokens, page cache warm):
    every expert on the CPU 0.2 t/s decode (A770, from ZFS); 14 expert
    layers on the card 44.4 / 9.0 t/s; 17 layers, CPU repacking on, 8 CPU
    threads 54.0 / 12.6 t/s (more layers overcommit the B60's 22.7 GB, and
    xe then pages VRAM over the link: 0.05 t/s on the A770 at 15.9 GB);
  - served through arcint (16 expert layers on the card, 32,768 context):
    the capital check "Paris"; the 20,045-token needle answered
    ("ORANGE-FALCON-77") at 49.2 t/s prefill; the 500-token long answer at
    10.9 t/s decode. The OpenVINO path's arms after 0076/0077: needle decode
    12.7 t/s, long answer 16.0 (`expert-hot-set-lru.md`,
    `tier-handoff-doorbell.md`).
- KL leg (`measured-here` 2026-10-03, B60, `llama-perplexity` 8 x 512
  tokens, reference: the whole model on the CPU backend): baseline arm
  (every expert layer on the CPU, `-ncmoe 48`) mean KL 0.0289 +- 0.0018,
  top-1 95.54 %; stage 1 (16 expert layers on the card, `-ncmoe 32`) 0.0294
  +- 0.0019, 95.59 %: +0.0006 nats, top-1 +0.05 points. Passes.
- Owed for the stage gate: the baseline arm's decode at 4k and 20k on the B60
  from NVMe (the 0.2 t/s above is the A770 from ZFS).
- Under `--load-mode auto` the load reads the whole model into anonymous
  memory and is OOM-killed; arcint sets mmap whenever experts stay on the CPU.

## Stage 2 — the per-expert cache (opened 2026-10-05, operator)

**Mechanism** (Strata's, `expert-hot-set-lru.md` has its source map; FreeToken
for the concepts it adds):
- Every expert tensor stays in host memory (memory-mapped from the GGUF on
  NVMe).
- Each layer gets GPU slot tensors of the same type, sized in bytes per layer,
  plus one all-zero slot.
- A CPU-side op on the router's top-k ids writes two id sets: slot ids for the
  hits (misses go to the zero slot) for a GPU `MUL_MAT_ID`, and expert ids for
  the misses (hits become -1, skipped) for the CPU's. The two branches are
  summed before the router weights.
- Usage is counted per layer and expert; counts decay x0.7 per adapt. Every 12
  decode tokens the adapt step swaps hot misses in (usage >= 2.0, gain > 1.5,
  at most 96 swaps). The newcomer's bytes come from the GGUF file between
  steps.

**Baseline arm**: stage 1's best static placement (16 expert layers on the
card), same card, same window, same GGUF on NVMe.

**Prediction** (written before the measurement): about 31 % of the experts fit
the slots. A GPU hit share of 0.5-0.7 cuts the CPU's expert evaluations per
token from 320 (32 layers x 10) to about 150-240. Decode 10.9 -> 14-17 t/s at
20k; prefill within the spread (the CPU still computes most prefill misses;
prefill on the GPU is P5).

**Gate**: the stage gate above (needle and capital; KL within 0.03 nats of
the baseline arm's, top-1 down at most 1 point; decode faster at 4k and 20k,
prefill within the spread). Copied bytes exact: a slot's bytes equal the
file's, red first on a mutated copy. Reported beside the rate: the GPU hit
share, swaps per token, VRAM and RAM.

| arm (B60, B60 window, GGUF on NVMe) | decode 4k | decode 20k | prefill 20k | hit share | KL / top-1 | needle |
|---|---|---|---|---|---|---|
| stage 1, 16 layers on the card | | | | n/a | | |
| stage 2, per-expert cache | | | | | | |

### Stage 2a, hits on the card and misses on the CPU (built and measured 2026-10-05/06)

Built as above (`contrib/llama.cpp` work tree, not shipped). `measured-here`,
B60, the GGUF on NVMe, `--n-ctx 32768`, 8 CPU threads, tonight's binary for
every arm:

| arm | prefill 20,045 | decode 500 tokens | decode experts on the card | answers |
|---|---|---|---|---|
| stage 1, 16 expert layers on the card | 81.9 t/s | 12.4 t/s (80.6 ms) | 1/3 by layer | Paris, needle |
| every expert layer on the CPU | | 9.4 t/s (106.9 ms) | 0 | Paris |
| 2a, 16 GiB of slots, profile only | | 10.7 t/s (93.3 ms) | 66.1 % | Paris |
| 2a, 16 GiB of slots, adaptive | 75.0 t/s | 10.5-10.9 t/s (95.4 ms) | 81.5-83.2 % (3,767 swaps in 42 adapts, 2.0 s writing) | Paris, needle |

Slot bytes exact against the file at load (0 of 597 slots of layer 0) and
after the first swap (0 of 3 tensors).

**Decode is bound by the GPU-CPU hand-off, not by the CPU's arithmetic.** From
the arms: a CPU expert evaluation costs ~0.043 ms (480 of them ~21 ms); each
layer whose experts touch the CPU costs ~1.2 ms of hand-off (the OpenCL
backend has no async copies or events: the scheduler drains the queue, copies
ids and activations down, computes, copies the rows up); the rest of a token
is ~28 ms. 2a moves misses to the CPU in all 48 layers, so it pays 48
hand-offs to save CPU work that was never the cost; stage 1 pays 32. The
cache itself works: 83 % of decode's routed experts on the card, against
Strata's ~0.72 (`paper`).

### Stage 2b, every expert on the GPU: slots on the card, a bank in USM host memory (built and measured 2026-10-06)

`contrib/llama.cpp` patch 0021, arcint `--llama-expert-cache MIB
--llama-expert-profile FILE` with `--llama-cpu-moe 48`:
- **slots**: per layer, VRAM tensors of the experts' own types, filled from
  Strata's decode profile (`data/expert-profile.bin`, `code`) up to the byte
  budget;
- **bank**: every other expert of the layer in USM host memory
  (`clHostMemAllocINTEL`), 36.5 GiB here. Kernels reach it through pointer
  arguments. A `cl_mem` over host memory (`ALLOC_HOST_PTR`, `USE_HOST_PTR`,
  even over a USM pointer) is migrated to the card by the driver: kernel reads
  at 1.1 TB/s against 14.3 GB/s for USM host, the link's rate
  (`measured-here`, B60);
- **tables**: per layer, expert -> slot and expert -> bank entry on the
  card. The router's ids go through them with `get_rows` (I32, added to
  ggml-opencl), and both branches run on the GPU and are summed. A pair whose
  entry is -1 is skipped by the kq kernels (the matvec writes its zeros; the
  GEMM's tile router drops it after a zero fill). The five Q8_0 down
  projections take the general MoE path, which cannot skip: their tables keep
  a zero slot and entry, and their bank stays on the card (3.2 GB of slots);
- **gather**: before a bank tensor's kernels, the routed experts are copied
  from USM into a card-side mirror with 16-byte loads (FreeToken's
  `copy_missing`, `code`). The MoE kernels' own loads reached ~1.5 GB/s over
  the link; the gather alone reaches 12-13.8 GB/s for 1-4 experts
  (`measured-here`);
- **swaps**: Strata's adapt every 12 decode tokens from decode usage (>= 2.0,
  gain > 1.5, <= 96, decay 0.7), each an exchange on the card's queue
  (newcomer from the bank into the victim's slot, the victim into that bank
  entry: Strata's RAM exchange, `code`) ahead of the next graph. Swapping from
  the file through host buffers cost 6-19 ms a token; the exchange 0.4 ms.

Measured (`measured-here`, B60, the GGUF on NVMe, 12,800 MiB of slots:
5,865 slots, 93-153 a layer):

| arm | prefill 20,045 | decode 500 tokens | decode experts on the card | answers |
|---|---|---|---|---|
| stage 1, 16 expert layers on the card | 81.9 t/s | 12.4 t/s after the needle; 10.5 and 10.7 cold | 1/3 by layer | Paris, needle |
| 2b | **149.6, 149.0 t/s** (two runs) | 12.1-12.3 t/s (four runs) | 74.7-75.1 % | Paris, needle, the long answer |

- KL against the CPU reference, 8 x 512 tokens, same window: 2b 0.029634 /
  95.784 % same top-1, stage 1 0.029206 / 95.931 %. +0.0004 nats, -0.15
  points.
- Copies exact: at load 0 of 405 slots and bank entries of layer 0 differ
  from the file; after the first swap 0 of 6 (both runs).
- Memory: 12,800 MiB of slots, plus the Q8_0 banks and the gather mirror:
  16.06 GiB on the card; 36.5 GiB of bank, pinned, in host memory. A review
  (2026-10-06) moved the mirror's allocation to load, limited the cache to
  IQ3_XXS / IQ4_XS / IQ4_NL experts (Q8_0 down), let one context drive it
  (an MTP draft context would count its drafts) and dropped the file pages
  read for the fill. The red case (one swapped
  slot corrupted, `LLAMA_EXPERT_CACHE_MUTATE=1`) reports 1 of 6.
- The gate (per phase): prefill +83 %, decode at or above stage 1's range, the
  answer-level bar met. **Passed.**
- **One ubatch per prefill chunk** (arcint sets `n_ubatch = n_batch` with the
  cache): a ubatch gathers each layer's routed bank experts once, so 2,048
  tokens a ubatch read the link a quarter as often as 512. The needle
  prefills at **235.0 t/s** (2,048, the default chunk) and 258.4 t/s
  (`--prefill-chunk 4096`), decode 12.2 / 12.0 t/s, answered both times
  (`measured-here`, B60, 32,768 context).
- **One branch in decode**: a decode ubatch (up to 8 tokens) runs the slot
  tensors alone. One table maps an expert to its slot, or to n_slots + its
  bank entry. The backend gathers the bank experts into the mirror, and the
  IQ matvec reads an id at or above n_slots there. The two-branch graph
  stays for prefill (the GEMM path) and for the Q8_0 down projections. Decode
  12.2 -> 12.4-12.6 t/s, prefill 233.9 t/s, answered (`measured-here`).
  What decode spends now is mostly the ~25 % of routed experts read over the
  link (~120 a token, ~264 MB): more slots (the Q8_0 banks off the card: 3.2
  GB) or a better hit rate are the next levers.
- **Adapt cadence**: every 6 decode tokens instead of Strata's ~13 (every 4
  verify windows): 78 % of decode's experts on the card and 13.0 t/s against
  75 % / 12.4-12.6 at 12, 12.8 at 4, 10.0 at 3 (planning cost), 12.4 at 24
  (short runs; the gate run after the needle 12.5 t/s, prefill 233.8,
  answered; `measured-here`). A deviation from Strata's cadence, measured.
- **Tried, no gain**: an integer-dot Q8_0 matvec for the dense Q8_0 weights
  (activations quantized to q8_1 as for the kq path, the IQ4_NL loop with the
  codes read directly; correct, 22/22 `test-backend-ops` cases). Served decode
  12.6 t/s without it, 12.5 with it; on the 4096 x 14336 test shape 162 us
  against ggml's 154 (~400 GB/s already). Not shipped (`measured-here`, B60).

### Stage 3 — the IQ2_XS file and MTP (2026-10-06)

Strata's own engine on this card (`strata-sycl-b60.md`, `measured-here`)
serves Flash-Next IQ2_XS (ISTA-DASLab GSQ-RCO) with its MTP draft layer at
620.5 t/s prefill (20,045 tokens) and 37.2-37.8 t/s decode. This stage puts
the same two inputs into arcint: the IQ2_XS file (contrib/llama.cpp patch
0022: IQ2_S / IQ2_XXS / IQ1_M / Q2_0 / IQ3_S kernels on Intel, and F16 x 2-8
columns) and the MTP head from an MTP-only GGUF (`--llama-mtp-gguf`, the
pin's converter `--mtp` over the 31 `mtp.*` tensors plus the checkpoint's
embedding and LM head; experts Q4_K, the rest Q8_0, embedding and head
Q6_K: 2.54 GiB on the card).

`measured-here`, B60, one window per row group, the gate's requests (capital,
the 20,045-token needle, the 500-token long answer), greedy:

| arm | slots | prefill 20k | decode 500 | drafts accepted | verify / round | answers |
|---|---|---|---|---|---|---|
| UD-Q3_K_XL, no MTP (a0) | 10,200 MiB | 230.9 t/s | 11.7 t/s (71.7 % on the card) | | | right |
| UD-Q3_K_XL, MTP 3 (b3) | 10,200 MiB | 201.1 | 10.7 | 64.8 % | 257 ms (3.0 steps) | right |
| UD-Q3_K_XL, MTP 2 (b2) | 10,200 MiB | 200.7 | 11.2 | 76.0 % | 211 ms (2.5 steps) | right |
| IQ2_XS, no MTP (c0) | 12,500 MiB | **337.0** | **16.0** (88.2 %) | | | right |
| IQ2_XS, MTP 3, before the F16 fix (c3) | 12,500 MiB | 275.0 | 5.1 | 62.5 % | 545 ms | right |
| IQ2_XS, MTP 3 (d3) | 12,500 MiB | 278.5 | 13.8 | 62.5 % | 191 ms (3.1 steps) | right |
| IQ2_XS, MTP 2 (d2) | 12,500 MiB | 277.7 | 13.8 | 71.7 % | 163 ms (2.6 steps) | right |

`--llama-mtp-vocab` does not apply to Flash-Next: its draft head reads
`output.weight` at `n_embd_out` (the four hyper-connection streams, 10,240),
the LM head takes the mixed 2,560 (refused at load, arms a3/a2).

Where an IQ2_XS decode token goes (`measured-here`, a profiling build of
llama-bench, tg32 from a cold cache, so ~55 % hits, not served decode's 88 %):
4,938 kernels a token, 62.0 ms of kernel time in an 87.3 ms span (5.1 us of
gap a kernel); the gathers of bank experts 18.0 ms (they scale with the
misses: ~5 ms at 88 %), the expert matvecs 11 ms at 40-77 GB/s of weights
(IQ2_XXS 111 us, IQ2_S 68 us, Q2_0 74 us a call), the hyper-connection ops
~6 ms.

Why MTP does not pay here, against Strata (`code`, `src/core/verify.cpp:854-905`):
Strata's verify groups a window's (token, expert) pairs per expert, copies a
missed expert over the link once (`fetch_blobs`), and computes each expert's
tokens in one kernel (`native_expert_grouped`). This engine's decode path
gathers per pair (a repeated expert copied again) and computes per pair (an
expert's weights read once per token): a 4-row verify costs ~3 plain steps,
Strata's ~2. Next, in that order: the grouped verify (tile the window's pairs
by expert with `kernel_moe_route_tiles`, gather unique experts, a matvec over
an expert's columns); the low-bit kernels' rate (local-memory grids, wide
aligned loads, as Strata's port did for IQ4_XS / Q6_K); fewer launches.

### Stage 3b — closing the gap to Strata on the same card (2026-10-06, operator: speed first, quality later)

Every arm `measured-here`, B60 (the residents' coder on the A770), the gate's
requests, greedy, answers right in every row; contrib/llama.cpp patch 0022 at
its state of each window. Between loads the same binary varies by about
+-0.5-1 t/s on Flash-Next (one 500-token sample a load).

| change (cumulative) | IQ2_XS plain | IQ2_XS MTP 2 | agent (B60, MTP 5) | coder (A770, MTP 4) |
|---|---|---|---|---|
| 0022 kernels only | 16.0 | 13.8 | 35.9 | 40.1 |
| + F16 x 2-8 columns, K-split for few rows, flat ADD, 32-bit low-bit expanders | 17.5 | 17.8 | 38.7 | 43.1 |
| + fused hyper-connection ops (DSV4_HC_PRE / POST) | 17.8 | | | |
| + a verify's bank experts gathered once each, 13,500-14,500 MiB of slots | 17.9 | 22.0 | 38.7 | |
| + the IQ4 codebook per card | 17.8 | 20.9-22.0 | 38.7 | 43.2 |
| + the MoE router fused (topk_moe) | 18.4 | 22.5 | 38.7 | **43.3** |
| + the low-bit types in planes on the card (prefill 371 t/s at 20k) | **21.3** | **27.0** | | |

Strata's SYCL engine on the same card and file: 37.2-37.8 t/s
(`strata-sycl-b60.md`).

Tried and measured slower, not kept as defaults (`measured-here`, B60 unless
named): the matvec grouped by expert for a verify (the coder's verify 7.85 ->
9.18 s on the A770; Flash-Next 125 -> 131 ms a verify), local-memory grids
for IQ2_S / IQ3_S (146 -> 245, 255 -> 335 us), u16-vector loads for IQ2_S /
IQ3_S / Q2_0 (146 -> 229, 255 -> 353, 184 -> 195 us), the IQ4 codebook by
vector shuffle from registers (129 -> 504 us), by register selects on the B60
(130 -> 146 us; faster on the A770, where it is kept), 2 or 8 rows a
sub-group instead of 4 for the kq / IQ matvecs (mixed: IQ4_XS MV_ID 36.6 ->
27.7 us at 8, Q4_K 77 -> 129 us).

Where a Flash-Next token goes (profiles, `llama-bench` with OpenCL
profiling): ~4,000 kernels a token at ~5 us of gap each (Strata measured the
same gap on the B70 and cut nodes); dense IQ4_XS ~8 ms, BF16 ~6 ms, the
expert matvecs ~11 ms at 60-110 GB/s of weights (the K-quant kernels reach
377-410 GB/s on their struct-of-arrays planes; the IQ and low-bit kernels
read blocks as stored), the gathers of missed experts ~5 ms.

Deviations from the references:
- No share of the misses on the CPU (Strata computes the rest concurrently):
  on this backend a CPU hand-off costs ~1.2 ms a layer (stage 2a).
- Adaptation by Strata's decayed usage, not FreeToken's LRU.
- The Q8_0 banks on the card (above): the general MoE path has no pointer
  arguments yet.

Where decode goes now (estimates from the counters, the profiler's
timestamps for USM-pointer kernels are not usable): ~22 ms of gather for
~120 missed experts a token, ~28 ms of the model's other work, the rest the
two branches' extra kernels (two MUL_MAT_ID per projection, the table
lookups). Next: one MUL_MAT_ID per projection reading slots and the gathered
mirror by two base pointers; the Q8_0 path with pointer arguments (3.2 GB more
slots); prefill gathers overlapped with the previous layer's compute.

### Stage 3c — the misses read where they lie, the draft layer on the card (2026-10-06)

`measured-here`, B60, IQ2_XS, MTP 2 from the MTP-only file, 13,500 MiB of
slots, the gate's requests, greedy; answers right in every arm (Paris,
ORANGE-FALCON-77, the long answer), and the long text identical between the
two window-22 arms.

| change (cumulative) | MTP 2 decode | propose / verify (500 tokens) | accepted | needle prefill |
|---|---|---|---|---|
| stage 3b, gather (window 22, same binary as the next row) | 26.0 | 2.28 / 16.87 s | 73.5 % | 296.6 |
| the expert matvec reads a missed expert from the USM bank itself | 28.2 | 2.29 / 15.40 s | 73.5 % | 295.8 |
| + a 2D weight times columns lying back to back (an MTP step's [K, 4, tokens]) as one few-column product; the F32 GEMV up to 1,024 rows | 29.9 | 1.56 / 15.11 s | 73.5 % | 324.7 |
| + the MTP file's routed down experts IQ4_NL instead of Q5_0 | **31.0** | 1.34 / 14.73 s | 74.8 % | 339.5 |

Plain decode with the direct read: 21.9 t/s (21.4 with the gather in
window 20).

- **The direct read** is Strata's mechanism (the expert kernels read the
  pinned host mirror, `strata-sycl-b60.md`; `code`): ids at or above the
  split index the bank in USM host memory by pointer argument instead of the
  gathered card-side mirror (`GGML_OPENCL_BANK_DIRECT=1`, the decode branch
  only; prefill's GEMMs still gather). The earlier record of "the MoE
  kernels' own loads reach ~1.5 GB/s over the link" was taken on the blocks
  as stored, before the planes gave each lane aligned 32-bit words; it was
  not a verdict on the mechanism. The verify spends 7.3 ms less a round.
- **The MTP file's down experts ran on the CPU.** llama-quantize's Q4_K
  override cannot take K = 640 and fell back to Q5_0, which ggml-opencl has
  no MUL_MAT_ID for: every draft pass split GPU -> CPU -> GPU
  (`ggml_vec_dot_q5_0_q8_0` was 7 % of the host's samples during decode,
  `perf` on the served process). Re-quantized with
  `--tensor-type ffn_down_exps=iq4_nl` (450 MiB):
  `mtp-flash-next-q8e4n.gguf`.
- **The draft pass's batched products** ([K, 4 streams, tokens] against a 2D
  weight) went to the tiled GEMM through the F16 dequant (Q8_0 `eh_proj`
  0.96 ms at 2 tokens against 0.06 at one), the F32 router (512 rows) to the
  tiled GEMM (0.37 ms). Profiled at window 21.

- **The Q2_0 down projections, 4 lanes a row**: a 2560 x 640 row is 20
  sub-blocks, so the matvec's 16 lanes a row left 12 idle in their second
  pass. 4 lanes a row with the sub-group's 4 rows side by side (two xor
  shuffles to reduce): 135 -> 89.4 us at 3 tokens, 46.9 -> 32.0 us at one
  (test-backend-ops perf, VRAM); served 29.9 -> 30.6 t/s, verify 15.31 ->
  14.93 s, one binary (`-DLB_NO_LB4` for the off arm). The test's red case
  (one reduction step dropped) fails 47 of 87 cases; green 193/193.

Kernel microbenchmarks (test-backend-ops perf, B60, weights in VRAM):

| kernel, shape | as built | variant | |
|---|---|---|---|
| IQ4_XS dense, 6144 x 2560, 1 / 3 columns | 44.2 / 49.6 us | codebook by byte pairs (256 x ushort, constant): 61.1 / 64.7 | slower |
| the same | | codebook in local memory, a 16-byte copy per lane: 59.5 / 58.7 | slower |
| IQ2_S MUL_MAT_ID, 640 x 2560, 30 pairs (random ids) | 120 us | grouped by expert (`idg`): 219 | slower |
| BF16 320 x 10240, 3 columns | 17.9 us | 4 rows a K-split work-group (`GGML_OPENCL_F16_KSPLIT4=1`): 12.7 | faster in isolation (weights in L2); off by default |

The constant-memory gathers stay the fastest IQ4 decode found on the B60
(register selects were slower too, stage 3b). (Superseded on 2026-10-11 by
patch 0035, stage 6: one table per work-group in local memory, with each
value already in its byte. The local-memory row above copied 16 bytes per
lane, a different build.) Layout is not the bound at a
verify's width either: IQ4_NL in flat planes against IQ4_XS as stored, the
same codebook, 6144 x 2560: 36.2 vs 44.3 us at one column, 47.8 vs 49.2 at
three. A planar IQ4_XS was not built.

The combined settings, one window each (`measured-here`, f16 KV, answers
right): MTP 3 with the 0.5 stop at 14,500 MiB 30.5 t/s; with the MTP file's
head in IQ4_XS (1,212 -> ~330 MiB on the card, the same acceptance) at
14,700 MiB (10,667 slots) 31.1 t/s, prefill 386.5. The single settings move
decode by less than the run-to-run spread (30.5-31.6 t/s across today's
arms of the same configuration).

- **The draft head over Strata's 106,299-token subset** (its
  `data/draft_vocab.bin`, `--llama-mtp-vocab`, now also for hyper-connection
  models: DraftHead mixes the four streams with the MTP file's
  `nextn.hc_head_*` weights as `build_hc_mix` does, `code`): propose 1.33 ->
  1.07 s, but 74.8 -> 68.8 % of drafts accepted: 30.6 -> 29.7 t/s on one
  binary. Strata drafts from the same subset (`code`, `mtp.cpp:389-403`,
  "+143 MiB for the draft head" in its log here).

- **Strata's draft stop** (`--llama-mtp-min-p P`, `code`: `mtp.cpp:804`,
  `generate.cpp:5297`: drafting goes on while a draft's probability under
  the MTP head is at least P; a draft below P is not verified). Strata ran
  `--spec 4 --spec-min-p 0.5` here. One binary, 0.5 (`measured-here`):

  | drafts | decode | accepted | propose / verify |
  |---|---|---|---|
  | 2, no stop | 30.3-30.6 | 73.5-74.1 % | 1.33 / 14.9-15.1 s |
  | 2, stop at 0.5 | 29.1 | 84.3 % of 343 | 1.67 / 15.48 s |
  | 3, stop at 0.5 | 30.3 | 79.4 % of 403 | 1.95 / 14.49 s |
  | 4, stop at 0.5 | 24.3 | 73.6 % of 454 | 2.19 / 18.33 s |

  The probability was a double-precision softmax over all 248,320 logits on
  the host, ~0.8 ms a draft; now summed in float over the logits within 16
  of the top.

  **Re-measured on 0001-0051 (2026-10-14, `measured-here`, B60, the unit's
  flags otherwise, one binary).** The verify's rows got cheaper since; the
  stop pays now at 3 drafts. Three 300-token answers (the gate's), total
  time: 2 drafts 25.21-25.25 s (two arms), 3 drafts 25.25, 3 drafts with
  the stop at 0.5 25.01 (+0.9 %), 4 with the stop 32.51 (-29 %), 2 with
  the stop 24.99 (+1.0 %). Strata's own request (the RFC 4180 long answer,
  500 tokens, `~/wp/smoke-0076-long.json` on the dev host; second run, the
  first is 2-7 % slower on each arm):

  | drafts | decode | rounds | tokens a round | ms a round |
  |---|---|---|---|---|
  | 2 | 35.4 t/s | 209 | 2.39 | 67.6 |
  | 2, stop 0.5 | 36.2 | 219 | 2.28 | 63.1 |
  | 3, stop 0.5 | **37.9** | 181 | 2.76 | 72.9 |
  | 4, stop 0.5 | 29.6 | 169 | 2.96 | ~100 |
  | Strata (`--spec 4 --spec-min-p 0.5`, its run log, 2026-10-06) | 37.8 | 212 | 2.36 | 63.4 |

  Strata's log shows it captured only 2- and 3-token verify windows: 433
  drafts in 212 rounds, at most 2 a round. At that setting the two engines'
  rounds cost the same (63.1 against 63.4 ms); Strata's head offers more
  drafts past the same stop (2.04 a round against 1.56). Strata normalizes
  the draft probability over its 106,299-token subset (`code`, `mtp.cpp`),
  which raises it: a likely cause, not measured. With a third draft arcint
  matches Strata's decode on Strata's text. A 5-row verify costs ~100 ms
  against ~73 for 4: a cliff between 4 and 5 rows, not located (the K-quant
  dense matmuls take the few-column kernel to 16 columns on the B60, and
  the MoE matvec runs to 15 tokens). Every arm: Paris, the needle; the
  texts differ between draft counts (the verify's shape changes the last
  bits; the answer-level bar allows it).

  **The context is the price of the slots** (`measured-here`, the same
  request, 3 drafts with the stop, the 20k needle first; decode after the
  needle unchanged in each, so nothing evicted; VRAM residency from fdinfo
  every 2 s). Strata's run held 12,332 slots at 32K with int8 KV; the unit
  holds 8,191 at 131,072 (the KV 2.0 GiB at q8_0, the compute buffer 3.0):

  | context / slots | decode, first / second run | experts on the card | peak VRAM |
  |---|---|---|---|
  | 131,072 / 11,300 MiB (8,191) | 35.0 / 38.0 t/s | 85.2 / 88.1 % | (the unit's 24.2-24.3 GB) |
  | 65,536 / 12,500 MiB (9,069) | 36.7 / 41.0 | 87.3 / 91.1 % | 22.12 GiB |
  | 32,768 / 13,700 MiB (9,939) | 39.0 / 43.9 | 89.5 / 93.6 % | 21.95 GiB |

  Every arm: Paris, the needle, the same text. At Strata's context arcint
  decodes 5-16 % faster than Strata on Strata's request. The operator keeps 131,072 (2026-10-14).

  At 131,072 the prefill chunk is the other VRAM lever: with the expert
  cache the ubatch is the whole chunk (`backend_llama.cpp`), and the
  compute buffer scales with it. `--prefill-chunk 1024` (`measured-here`,
  same request, the needle first):

  | chunk / slots | compute buffer | 20k needle prefill | decode, first / second | peak VRAM |
  |---|---|---|---|---|
  | 2,048 / 11,300 MiB (8,191) | 3,048 MiB | 657 t/s (30.5 s) | 34.8 / 38.1 t/s | 23.83 GiB |
  | 1,024 / 12,500 MiB (9,069) | 1,543 MiB | 538 t/s (37.3 s) | 36.8 / 41.1 | 23.40 GiB |

  Decode +6-8 % and 0.4 GiB more margin for prefill -18 %: each chunk
  streams the bank once, so half the chunk streams it twice as often. The operator keeps 2,048 (2026-10-14).

  **Next: the prompt buffers borrowed from the expert cache** (Strata's
  default, `code`: `generate.cpp:333` "the prompt path borrows the top
  expert-cache slots for its buffers and refills them after the prompt",
  `prefill.cpp:513-721`; `--prefill auto` takes the largest chunk, up to
  8,192, whose buffers the slots can lend). arcint reserves the prefill's
  3,048 MiB compute buffer for the whole session; decode needs a fraction of
  it. Lent to the slots between prompts, it is worth about what the 1,024
  chunk measured (+6-8 % decode for 1.5 GiB) without the prefill's loss, and
  a larger chunk could take the borrowed memory during a prompt (fewer bank
  passes). Not built; the price is the table above.

  What decode needs of it (`measured-here`, the unit's flags loaded with
  `--prefill-chunk N`, the scheduler's reservation from the load log; the
  ubatch is the chunk with the expert cache, the draft context's capped at
  512):

  | chunk | main context | draft context |
  |---|---|---|
  | 4 (a 3-draft verify) | 93 MiB | 84 MiB |
  | 16 | 99 | 90 |
  | 64 | 122 | 112 |
  | 256 | 406 | 394 |
  | 2,048 (the unit) | 3,048 | 770 |

  About 3.5 GiB of the reservation serves prompts of 256 tokens and more
  only; as slots it is ~2,700 more (8,191 -> ~10,900), about the 32K arm's
  hit rate (39.0 / 43.9 t/s there). The design facts: the slots are three
  tensors a layer in one buffer (`llama-expert-cache.cpp`), so a contiguous
  loan is whole layers' slots (~240 MB a layer); a prompt ubatch of 256
  tokens and more streams each bank tensor whole (0043/0044), so a lent
  layer costs a long prefill little; the refill after a prompt is ~3.5 GiB
  over the link, ~0.25 s. The hazard: a graph that writes into the lent
  range while it holds experts corrupts weights without changing the
  answer's look; the loan needs a per-graph bound check.

  **The reservation is a worst case** (`measured-here`, the load logs at
  chunk 2,048): the compute buffer grows with the context, 988 MiB at
  32,768, 1,631 at 65,536, 3,048 at 131,072 (the draft context 229 / 410 /
  770). Over all eight points it is ~90 MiB + chunk x (0.146 MiB + 10.7 bytes
  x cache cells): the QSA selection mask is cells x chunk tokens in f16,
  built as a repeat, a `set_rows` into it and an add with the attention
  mask (`qwen4exp.cpp` `build_qsa_sel`, its own TODO: "reduce the large
  compute buffer"). The scheduler reserves for a full 131,072-cell cache,
  which only the last chunks of a 131K prompt reach; a 20k prompt's chunks
  see at most ~20k cells. The loan's alternative, without moving any
  weights: reserve for a budget (a 512-token chunk at the full context,
  ~790 MiB) and cut each prompt chunk to what fits the budget at the
  cache's current fill: 2,048 tokens to ~18-20k cells, ~1,000 at 65K, 512
  at 131K. Built as patch 0052 (README 0052, `measured-here`): the
  dominant tensor was the indexer's head scores (pools x 4 heads x tokens,
  f32, 1 GiB at 131K) and its ReLU copy, not the mask. The ReLU in place
  alone frees ~1 GiB (decode +2-3 % spent on slots, everything else
  unchanged); with `--llama-reserve-cells 20480` and 13,400 MiB of slots
  decode is 38.4 / 43.0 t/s against 34.8 / 38.0 at 131,072 tokens, the 20k
  prefill unchanged, a 72k prompt 128.7 -> 173.0 s. Released in 0.7.9; the
  Flash-Next unit carries it (`--llama-reserve-cells 20480`, 13,400 MiB).

- **What the misses cost after 0052** (`measured-here`, 2026-10-15, B60, the
  unit's 0.7.9 flags). A `GGML_OPENCL_PROFILING` build of 7014a61 on Strata's
  request and two gate answers: decode-side kernel time 26.7 s; the expert
  matvecs 10.0 s, of which 5.05 s (18.9 % of all decode-side kernel time)
  is each call's excess over the fastest call of its kernel and token
  count: the missed experts read over the link inside the same launch (a
  three-token IQ2_S projection 73 us with no miss, up to ~231 us). In
  isolation (test-backend-ops perf, all experts on the card) the IQ2_S
  expert matvec runs 31 us at one token and 71 us at three (165-215 GB/s),
  Q2_0 22 / 52 us. The ceiling, `LLAMA_EXPERT_CACHE_CEILING=1` (bank pairs
  read the slots' zero matrix: wrong answers, the time of a hit): Strata's
  request 38.5 / 43.1 -> 49.9 / 50.6 t/s, a verify 63.3 / 55.7 -> 45.3 /
  46.7 ms a round. With no miss the first run is no slower than the second:
  the first-run gap is the misses. The CPU backend computes ten IQ2_S
  experts for one token in 174 us (16 threads, ~29 GB/s).

- **The misses on the CPU, Strata's doorbell tier, on the libllama engine**
  (`measured-here`, 2026-10-15; llama tree `p0053-cpu-misses`, not in the
  series). At a decode layer's first expert projection a kernel publishes
  the activation and marks the first N missed pairs -1 in the combined ids
  (the card's three projections then skip them) and publishes those pairs
  and their bank rows to USM host memory with a system-scope release; a
  coordinator thread with a spinning pool computes their gate, up, SwiGLU
  and down from the bank in host RAM with ggml-cpu's dot products (the
  planar bank rows gathered per row); after the down projection a kernel
  polls the done flag with L1/L3-uncached loads (0077's form) and copies the
  rows in. Strata's request, the 0.7.9 flags, one binary:

  | CPU takes a layer | verify a round, first / second run | decode |
  |---|---|---|
  | none | 63.0 / 55.6 ms | 38.6 / 43.2 t/s |
  | up to 1 | 61.0 / 55.4 | 39.7 / 41.5 |
  | up to 2 | 60.4 / 55.3 | 39.0 / 43.1 |
  | up to 3 | 61.7 / 57.0 | 40.4 / 42.0 |
  | up to 4 | 64.4 / 57.8 | 38.5 / 40.3 |
  | all (6 / 8 threads) | 75.7 / 65.2, 73.2 / 62.7 | 31.5 / 38.1, 32.4 / 39.4 |

  Answers right, the texts differ (the CPU's single q8 term). The hand-off
  itself is nearly free: with the CPU computing nothing (a table bug in the
  first build) the doorbell's publish and poll in all 48 layers ran at the
  ceiling's speed (49.2 / 50.0 against 49.8 / 50.6 t/s). The cost is the
  CPU's: ggml-cpu's IQ2_S dot product decodes ~1.9 GB/s a thread on this
  host's 8-core Zen 3 with DDR4, so a missed expert is no faster on the CPU
  than over the link. How this differs from Strata's: its own AVX kernels
  that decode a block once for the whole window (`code`,
  `src/kernels/cpu/iq_avx2.cpp`), a Zen 4 with AVX-512 and DDR5 (`paper`),
  and a recorded pass. Not a verdict on the mechanism: on this host it
  needs a CPU expert kernel at least twice as fast as ggml-cpu's.

- **Not taken, 2026-10-15** (`measured-here`, B60, the 0.7.9 flags): the
  hyper-connection gate (`hc_gate`, f16 10,240 x 320, ~32 us a call, 4.3 %
  of decode kernel time) one row a work-item instead of four rows a
  sub-group with a cross-lane reduction (llama tree `p0053-f16-rowlane`):
  MUL_MAT f16 283/283, but served decode unchanged (on the answer whose text
  stayed the same, verify 6.85 against 6.80-6.81 s); the gate's time is not
  on the round's critical path, or the kernel is no faster in place. The
  kernel lever that remains is the IQ2_S expert matvec at 165-215 GB/s with
  every expert on the card (Q2_0 reaches 234-295); its hit-only time is
  ~19 % of decode kernel time, and the local-memory grid was already
  measured slower (stage 3b); the next step there is an ISA reading. Rows a sub-group (`GGML_OPENCL_KQ_Q8_1_NDST`,
  global), test-backend-ops perf in isolation: 8 against 4 is faster for
  gate/up (K 2,560, 640 rows: IQ2_S 31.2 -> 23.1 us at one token, 71.3 ->
  67.5 at three; Q2_0 52.0 -> 49.0 at three) and slower for down (K 640,
  2,560 rows: 63.6 -> 81.1 us at three); 1 and 2 are slower everywhere. A
  per-shape choice would gain ~5 % of gate/up at a verify's width.
  Down lost because `MV_LB4` (4 lanes a row where 16 lanes do not divide
  K = 640) is compiled only at 4 rows (`code`, `mul_mv_iq_q8_1.cl`).
- **IQ2_S's grid packed, not taken** (`measured-here`, 2026-10-15, B60,
  test-backend-ops perf, MUL_MAT_ID 512 experts, top 10, 640 x 2560, two
  repeats): the grid's bytes take three values (8, 25, 43), so its 1,024
  entries fit 16-bit codes (2 KiB against 8). Expanded with integer
  operations from constant memory 43.6 / 101.4 us at one / three tokens,
  from local memory 46.8 / 105.5, through a local byte table (four codes to
  four bytes) 39.3 / 82.1, against the 64-bit constant grid's 31.1 / 70.8.
  The constant gather is not the bound; the kernel's integer issue is (the
  ISA, offline, bmg: 487 instructions with the 64-bit grid, 594-737
  packed). Its red case (43 read as 42) failed 1 of 13 MUL_MAT cases and no
  MUL_MAT_ID case: a one-level grid error is near the suite's tolerance.
- **Fewer operations in the decode: patch 0053** (`measured-here`,
  2026-10-15, B60; README 0053). IQ2_S / IQ3_S sign bits and Q2_0's 2-bit
  fields spread to bytes by multiplies instead of shifts and masks, exact
  for every input byte. IQ2_S 31.1 / 70.9 -> 29.1 / 63.6 us at one / three
  tokens (487 -> 425 instructions), Q2_0 gate/up 21.8 / 52.0 -> 20.9 / 49.7.
  Served (the 0.7.9 unit's flags, one binary, two processes an arm):
  decode +0.9 % on the mean (38.6 / 43.0 / 38.3 / 41.4 -> 38.7-39.2 /
  43.3-43.4 / 38.6-38.8 / 41.8-42.0 t/s), the same texts, the needle right.
  The per-shape 8 rows, measured with it, lost ~1 % served and is not in
  the patch.
  The same multiply for IQ2_XXS's signs (the eighth sign the parity of
  seven, by `popcount`, instead of the local `ksigns` table; llama tree
  `p0056-iq2xxs-sign`, not in the series): 333 -> 394 instructions and
  24.5-25.0 / 55.3-55.5 -> 25.6 / 56.6-56.8 us at one / three tokens
  (`measured-here`, B60, two repeats; suites pass, its red case fails 8/18
  + 37/75). The local table stays, as stage 5 found for its register form.

- **The draft head's LM head quantized** (`measured-here`, 2026-10-15,
  B60, the 0.7.9 package and unit flags; Strata's request twice and two
  gate answers): Q6_K (deployed, 521 MB) propose 1.44 s a 500-token
  answer, acceptance 79.4 %, 38.6 / 43.1 t/s; IQ4_XS (322 MiB) propose 1.35
  s, acceptance 75.6 %, 37.2 / 41.3 (the IQ4_XS matvec reads ~240 GB/s, and
  the head loses drafts); Q4_K (341 MiB, `mtp-flash-next-q8e4n-hk.gguf`)
  propose 1.27 s, acceptance 79.8 %, 38.6 / 43.0, and on the answer whose
  text stayed the same 38.8 against 38.4 t/s. Q4_K is worth ~1 % and frees
  ~180 MB for slots: a unit change for the operator.
- **The gap between kernels** (a chain of 4096-wide ADDs, test-backend-ops
  perf): 3.03 us a kernel. compute-runtime debug keys: skipping the walker's
  post-sync 2.91 us, optimized in-order barriers, L3 prefetch, relaxed
  ordering, queue drain mode, no cache flush after the walker, no
  PIPE_CONTROL before the post-sync: 3.00-3.10 us; direct submission off
  4.38 us. The ~3,900 kernels of a round pay ~11 ms of it: only fewer
  kernels move it (Strata measured the same on the B70 and cut nodes).

- **The resident models** (`measured-here`, the 500-token long answer twice,
  each unit's own arguments, the same text across builds): the agent (dense
  27B, B60, MTP 5) 35.5 / 35.6 t/s on the installed 0.5.12 (patch 0021),
  38.2 / 38.3 on 0022 as of stage 3b, 38.2 / 38.3 with all of stage 3c; the
  coder (A770, MTP 4) 40.4 / 40.5, 43.5 / 43.5, 43.5 / 43.4. 0022's general
  kernels pay for both (+7.6 %, +7.4 %); the Flash-Next-specific work since
  leaves them unchanged.
- **A hyper-connection read fused** (Strata's `fused_gr.cu` `gr_down` /
  `gr_up`, `code`): RMS_NORM, the w_norm MUL, the down product, SCALE, SILU,
  the up product and the gated DSV4_HC_PRE in two kernels (a group per 16
  down rows and stream with the rms scale applied afterwards; a group per 32
  output columns), 1-4 tokens, F16/BF16 weights. test-backend-ops HC_MIX
  10/10 against the CPU at NMSE 1e-5; the red case (the mix 5 % off) fails
  9 of 10 (the tenth is the 5-token case it does not take). Served it fires
  (the target's 96 sites; the MTP layer's Q8_0 weights are not taken) and
  pays nothing: 31.2 t/s off, 30.8 on, one binary. Profiled: the down kernel
  50 us and the up kernel 35 us a site against ~75 us for the five kernels
  (and 288 fewer launches a verify, 3,679 -> 3,391): the down kernel's 80
  work-groups, each staging its stream's activations, are slower than the
  K-split product's 320. A second version after Strata's v3 split
  (`fused_gr.cu:300`, `code`: the down product per 8-row block x stream x K
  half, 320 groups; a one-group kernel for the rms scales and `lo`; the up
  product 16 columns a group with 8 lanes a row) is correct (HC_MIX 10/10,
  red 9 of 10 failing) and no faster: down 52.7 us, lo 4.7, up 29.8, ~87 us
  a site against ~75 unfused; served 30.4 on, 30.8 off, one binary. The
  staged activations and the weights cost the fused down kernel about what
  the K-split product's cached reads cost. Opt-in
  (`GGML_OPENCL_FUSE_HC_MIX=1`), the second version kept.
- **The BF16 tensors as Q8_0** (1,457 MiB: the hyper-connection products
  1,212, routers 120, the rest small) were sized and not tried: the MTP
  layer's own Q8_0 products on this backend take 42.6 us for the 4-row
  inject (9.3 us on the F16 K-split) and 33.7 us for the down product at two
  tokens (37 at three in F16); they would need few-row Q8_0 kernels first.

- **The order of a verify's expert pairs**, from a permutation computed once
  per layer (`kernel_moe_pair_perm`, one work-group; the three projections
  share the ids). One binary, the long text identical across orders
  (`measured-here`): bank pairs first 30.7 t/s (verify 14.92 s), token order
  29.8 (15.40), bank pairs last 29.3 (15.68). The link reads that start first
  overlap more of the slot pairs' work; bank-first is the default
  (`GGML_OPENCL_PAIR_ORDER`). The earlier bank-first attempt that lost (below)
  scanned the ids in every work-group.
- **Where the expert matvecs spend it** (window 25, a profiling build): 456 to
  953 us a layer averaged over 207 verifies (the gate's, up's and down's
  together), ~330 us with every expert on the card: the misses cost ~12 ms
  of an ~80 ms round, concentrated in some layers and some verifies (layer 3
  once took 1.7 ms).
- **More slots**: the MTP context's buffer and the direct read left room for
  14,500 MiB of slots with MTP 2 (10,522 slots, 14.34 GiB): 31.6 t/s, verify
  14.45 s, answers right. VRAM (`-v`): the target's weights 3,264 MiB, KV
  768 + 192, recurrent state 338, compute 1,024; the MTP model 1,944 (521 of
  it its own Q6_K head), KV 80, compute 197.

- **The next projection's misses copied by the current one's matvec**
  (`GGML_OPENCL_MOE_PREFETCH=1`: gate's work-groups copy up's missed experts
  into a card-side mirror, up's copy down's; the long text identical): 31.4
  -> 31.6 t/s, verify 14.54 -> 14.46 s, one binary. The link is the bound,
  not where its reads sit: a layer's misses are ~3 experts x 3 projections x
  ~0.5 MB, ~14 ms a verify at 14.3 GB/s, and a layer has no other work for
  them to overlap after its router. Off by default. Fewer misses (more slots)
  or fewer bytes per miss are what is left.

- **Prefill: the F16 / BF16 products on XMX.** A 2,048-token chunk spent
  ~0.9 s of its ~5.1 s of kernels in the hyper-connection products, routers
  and injects through ggml's generic tiled GEMM (no XMX; the 4-row inject
  1.9 ms for 84 MFLOP). They now take the K-quant GEMM's XMX tiles with an
  F16 dequantizer (16+ columns): 320 x 10240 at 512 tokens 1,907 -> 468 us,
  the inject 1,769 -> 416, 10240 x 320 786 -> 161 (test-backend-ops perf);
  the needle 342.8 -> 383.4 t/s on one binary, answers right. Strata
  prefills it at 620.5. What the chunk spends besides (window 25): the
  expert GEMMs of both branches ~1.5 s, the bank gathers 0.81 s (the link),
  attention 0.37 s.

- **KV q8_0** (the agent unit's setting) to free ~450 MiB for slots: 23.6
  t/s (MTP 3 with the 0.5 stop, 14,500 MiB; verify 18.52 s, propose 2.57 s,
  prefill 364 t/s) against 31.3 on f16 KV. Flash-Next's attention heads are
  256 wide; the quantized-KV attention kernels were searched at 128. Not used.

- **The adaptation's constants** (`LLAMA_EXPERT_CACHE_ADAPT_EVERY`,
  `_MAX_SWAPS`, `_MIN_GAIN`, `_MIN_USAGE`, `_DECAY`), one binary, MTP 2,
  13,500 MiB: every 6 tokens (Strata's constants) 90.4 % of decode's routed
  experts on the card, 30.7 t/s; every 3: 90.1 %, 31.1; every 12: 89.9 %,
  31.0; 192 swaps with a gain bar of 1.0: 90.5 %, 29.9; decay 0.85: 90.7 %,
  31.1. The hit rate is set by the slot count, not the policy.
- **What a verify routes** (counted by the cache, 209 MTP 2 steps): 1,440
  pairs over the 48 layers go to 1,041 distinct experts (72 %); ~128
  distinct experts are in the bank against ~135 bank pairs. Grouping a
  verify's pairs by expert can save ~28 % of the decode work on the card's
  experts and ~5 % of the link's bytes. The misses are ~128 experts x 3
  projections x ~0.5 MB, ~190 MB a verify, ~13 ms at 14.3 GB/s: the floor of
  reading misses over the link at this slot count. The host's RAM reads at
  roughly three times the link; Strata computes part of its misses on the
  CPU concurrently, which stage 2a found too costly through llama.cpp's
  graph splits (~1.2 ms a layer for the hand-off).

- **A verify's pairs grouped by expert, decoded once** (Strata's
  `row_dot_multi` with up to 4 entries; groups from a per-layer kernel, the
  bank's experts first; `GGML_OPENCL_KQ_GROUPED=1`): correct (MUL_MAT_ID
  193/193, the long text identical) and slower: 30.5 -> 29.5 t/s, verify
  15.01 -> 15.56 s (test-backend-ops on random routing: IQ2_S 120.6 ->
  146.9 us). The 28 % overlap does not repay the per-entry cost of this
  kernel shape. Opt-in.

### Stage 3d — the activation as one int8 term (2026-10-07)

ggml-opencl's K-quant and IQ matvecs quantize the activation to two int8
terms (codes plus the codes of what they leave, `kernel_quantize_q8_1`;
added because q8_1 alone moved the dense 27B's decode KL by 0.0026 nats) and
do every integer dot product twice. llama.cpp's MMVQ and Strata's kernels
use a single q8_1 (`code`). `GGML_OPENCL_KQ_ONE_TERM=1` builds the kernels
without the residual's dot products (its loads are then dead). One binary
(src-fn31), B60, `measured-here`:

| kernel (test-backend-ops perf) | two terms | one term |
|---|---|---|
| IQ4_XS 6144 x 2560, 1 / 3 columns | 44.0 / 49.6 us | 41.4 / 43.9 |
| IQ4_XS 4096 x 14336, 2 columns | 178.7 | 174.0 |
| MUL_MAT_ID 30 pairs, 3 tokens: IQ2_S / IQ2_XXS / Q2_0 640 x 2560 / Q2_0 2560 x 640 | 119.8 / 96.2 / 83.2 / 90.0 | 117.7 / 92.0 / 79.8 / 83.8 |
| Q4_K, Q6_K (already at 400-430 GB/s) | | unchanged |

MUL_MAT and MUL_MAT_ID for the IQ, Q2_0 and K types: 294/294 and 286/286
against the CPU in both builds.

| served, 500-token long answer | one term | two terms |
|---|---|---|
| Flash-Next IQ2_XS, MTP 2, 14,500 MiB (on, off, on) | 31.8, 31.6 | 31.0 |
| the agent (dense 27B, B60) | 38.3 / 38.3 | 38.2 / 38.3 (text identical: its kernels do not take this path) |
| the coder (A770, MTP 4) | 40.7 / 40.8, drafts accepted 38.2 % | 43.3 / 43.3, 42.7 % |

Flash-Next answers right in every arm (Paris, ORANGE-FALCON-77, the long
answer; its text differs between the arms). On the coder the coarser target
logits agree less with the MTP head's drafts, and decode loses 6 %. The
switch stays opt-in: +2 % on Flash-Next does not repay a default that costs
the coder. Its KL against the reference was not measured.

The kernels stay well under the card's bandwidth with either activation:
the expert matvecs read ~131 GB/s (IQ2_S) to ~173 GB/s (Q2_0, whose decode
is shifts only) at 3 tokens, against 400-430 GB/s for the dense Q4_K/Q6_K
products, so the low-bit decode is not bounded by the codebook alone.

### Stage 3e — where a round's time went: the host, and the indexer on the CPU (2026-10-07)

**Strata's kernels on this card** (`measured-here`: its CLI greedy run of the
long-answer prompt, 500 tokens, the served flags, under `unitrace -d`; 34.1
t/s traced, 211 rounds of 2.38 tokens): about 64 ms of device time and
~3,140 kernels a round against arcint's 64.5 ms (60.1 target, 4.4 drafts)
and ~3,735. Its IQ2_S/IQ2_XXS/IQ1_M gate-up and Q2_0 down kernels take ~27
ms a round (arcint ~26.6), its GR read ~8 ms (arcint ~8.4). The kernels are
not the gap; a round is (69.4 ms traced against ~79).

**The gap between kernels is the host's.** A chain of 2,000 tiny dependent
kernels on the B60 (`measured-here`, a standalone probe): 2.64 us a kernel
on an OpenCL in-order queue, which is exactly the host's enqueue time (2.62
us); 0.72 us a kernel, the kernel included, when a 440 ms kernel lets the
host queue all 2,000 first. SYCL: 2.36 us direct, 1.07 us replaying a
recorded graph (Strata's "window graphs captured at load"). The driver
offers no `cl_khr_command_buffer` (`clGetExtensionFunctionAddressForPlatform`
returns null). Stage 3c's "3.03 us a kernel" and its driver knobs measured
the host, not the GPU.

**Where the GPU waited** (the profiling build's `cl_trace.json`, host
enqueue and device start/end per kernel; that build runs at 26.5 t/s): the
last draft's head to the next verify 8.8 ms a round, the target's head to
the first draft 1.6 ms, between the drafts 0.6 ms, all with the host late.
`perf` on the served process: 70 % of its samples wait in
`ggml_backend_opencl_synchronize`, and the scheduler copies split inputs
(`ggml_backend_sched_compute_splits`). `GGML_SCHED_DEBUG=2`: a verify graph
has 98 splits, a draft graph 10. Per attention layer, the QSA indexer's TOP_K,
its F32<->I32 casts and the F16 FILL / REPEAT / SET_ROWS that build its mask
had no OpenCL kernel, so the scheduler ran them on the CPU: four round trips
a layer, each draining the queue.

**Those ops on the card** (0022): TOP_K (a radix select a row, the indices
unordered as ggml-cpu leaves them), CPY F32->I32 and I32->F32, FILL, REPEAT
and SET_ROWS for F16. test-backend-ops against the CPU (B60): CPY 144/144,
FILL 6/6 (two F16 cases added), REPEAT 12/12, SET_ROWS 91/91, TOP_K 525/525;
the red kernels (each new path broken) fail 2, 2, 2, 16 and abort. Served,
one session, the long text identical across the arms (`measured-here`):

| build | decode | needle prefill |
|---|---|---|
| with the ops (run 1) | 34.2 | 422.4 |
| without (src-fn31) | 32.0 | 381.4 |
| with the ops (run 2) | 33.0 | 425.0 |

Answers right in every arm. The residents do not take the indexer: the
agent 38.2 / 38.3, the coder 43.4 / 43.4, unchanged.

A second long answer right after the first, the cache adapted to that very
text, decoded at 37.0 t/s (verify 14.42 -> 12.16 s, the same drafts): the
misses cost ~11 ms a round on this text. Strata's profile placement does not
adapt, so that run is no comparison; it prices the slot count.

**What a round's host does now** (`LLAMA_DECODE_TIMING=1`, per decode,
steady state, the normal build, `measured-here`): a verify reuses its graph
every time and takes 66.6-68.2 ms, of which the host enqueues for 20.0 ms
(~5.9 us a kernel, overlapping the GPU's ~60 ms) and counts the previous
step's routing (the expert cache's step) for 0.72 ms; a draft decode takes
2.9 ms (0.7 enqueue, the rest its GPU work), its graph rebuilt in 86 of 100
calls at ~0.08 ms. The cache's step read the 48 layers' routed ids with a
wait at the start of the next verify (1.17 ms); the ids are now read behind
the graph that routes them and counted after the host's sync
(`LLAMA_EXPERT_CACHE_ASYNC_IDS=0` restores the wait): 32.8 -> 33.4 / 32.9
t/s, the long text identical, within the spread. Per round ~75 ms against
~64.5 ms of kernels: ~10 ms of host time remains, ~6.5 of it inside the
verify. `GGML_SCHED_TIMING=1` times the scheduler's splits per context.

### Stage 4 (closed 2026-10-07, not built) — Strata's CPU share of the misses

**Retracted premise.** The reference measured here does not use a CPU share
(`code`, Strata 7ba023e): it ran with `STRATA_VERIFY_DEVICE_PLAN=1
STRATA_VERIFY_NO_HOST=1` (`sycl/serve/strata-sycl.sh:24`, the port's own
serving script). With those the host does no per-layer work at all: the
per-layer wait-and-pool loop is skipped (`sycl/src/core/verify.cpp:1356-1364`,
"on this platform a kernel's writes to host-mapped memory are not reliably
visible while the graph runs"), the GPU plans every layer itself
(`resident_plan`), and an expert missing from VRAM counts as resident when
the pinned mirror holds it (`resident_plan_set_mirror`,
`sycl/src/kernels/cuda/verify_kernels.dp.cpp:1115-1126`; the mirror table
from `sycl/src/program/generate.cpp:3815-3827`), so the GPU's expert kernels
read every miss over the link and the CPU rows are zeros
(`copy_or_zero_from_mapped`, `verify.cpp:985`). The "pcie_frac 0.38" its log
prints is computed and unused in that mode. 37.2-37.8 t/s is therefore
Strata with all misses over the link, as arcint's direct read does; the
per-round gap (~80 ms here against its 63 ms at about the same tokens per
round) is not a CPU share. The text below is the plan as it was written;
nothing of it was built.

What Strata does on this card (`measured-here`, its log here: "PCIe probe:
13.7 GB/s host->device -> pcie_frac 0.38"; `code`: `generate.cpp:1704-1723`,
`expert_source.hpp:185-360`): per layer the host receives the activations
and routed ids through mapped memory, publishes the GPU's plan (the VRAM
hits and `pcie_num`/256 of the layer's distinct missed experts, read over
the link), and its CPU pool computes the other missed experts from host RAM
at the same time, each distinct expert once for its tokens; both land in
`parts` at the router's index before the combine. On the B60 38 % of the
misses cross the link and 62 % never do. arcint reads all of them over the
link (~13 ms a verify, above). Stage 2a's "a CPU hand-off costs ~1.2 ms a
layer" measured llama.cpp's synchronous graph split, not this mechanism.

Plan, for the libllama engine (no DESIGN invariant against it: §3.4 names "a
timing-probed miss split" as acceptable):
1. A split both sides compute from the ids alone (a hash of the bank entry
   against `pcie_num`), so nothing about the split crosses the link.
2. At a layer's gate product the GPU writes the activations and ids to a
   host mailbox and raises the layer's flag; the down product's kernel
   leaves the CPU's pairs at zero, and a kernel after it waits for the CPU's
   flag (bounded spin, LSC uncached loads: memory
   `project-b60-host-flag-needs-lsc-uncached`, ~5 us) and copies the CPU's
   rows in before the combine.
3. A CPU worker pool (the host's 16 threads less the submitting one) polls
   the flags, computes gate, up, SwiGLU and down for its experts from the
   USM bank (planar on the card's side: converted back per expert, or read
   by its own dot products), and writes the rows.
4. First a standalone handshake microbenchmark on the B60 (GPU -> CPU flag,
   CPU -> GPU flag, bounded spins), because a GPU spin that never sees its
   flag can wedge the card, and a wedged xe card has needed a host reboot
   (the operator's call).

Expected (estimate, not measured): the CPU's share at ~20 GB/s (Strata
measured its pool at ~19 GB/s in the engine) is ~125 us a layer against
~330 us of GPU expert work, so it hides behind it; the link's share drops to
~5 ms: up to ~7 ms of an ~80 ms round.

Tried and measured slower (`measured-here`, B60, MTP 2, one binary, the long
text identical): the bank's pairs first in the expert matvec (each
work-group maps its index so the pairs read over the link start first, to
overlap them with the slot pairs' work): 30.3 -> 26.5 t/s, verify 15.11 ->
17.49 s. Kept behind `-DLB_BANK_FIRST`.

With the direct read the expert matvecs take 204-207 us a call (IQ2_S,
Q2_0; 135 and 122 us reading the card-side mirror), 28.3 of the target
verify's 60.1 ms of kernels (window 25, a profiling build): the misses' link
time adds to the slot pairs' instead of overlapping it.

Where the host spends a verify: spinning in `clFinish` (60 % of samples),
so the host is ahead and the GPU-side gaps between ~3,800 kernels are what
the wall time adds to the kernel time (82.5 ms a verify against 65.4 ms of
kernels at window 21).

## Where it lives

`contrib/llama.cpp/patches/` (ggml-opencl kernels, later the expert tier);
`src/exec/backend_llama.cpp` (placement flags).

### Stage 5 — the B70 projects' tricks on the B60 (2026-10-10)

The levers from `research-b70-field.md`, tried on the B60. All
`measured-here`: Flash-Next IQ2_XS on the ext4 NVMe, 13,000 MiB of slots,
32,768 ctx, MTP 2, fresh processes. Answers right in every arm.

- **Larger prefill chunks** (sybil-solutions' recipe stages experts once per
  8k chunk). arcint 0.7.2's `--prefill-chunk` (= the ubatch with the expert
  cache), the 20,045-token needle:

  | chunk | slots | prefill | peak VRAM | decode, 300 tokens |
  |---|---|---|---|---|
  | 2,048 | 13,000 | 422.1 | 21.2 GB | 28.2 |
  | 4,096 | 13,000 | 443.9 | 22.2 GB | 29.8 |
  | 8,192 | 13,000 | 435.0 | 24.3 GB | 30.5 |
  | 8,192 | 11,500 | 432.4 | 22.8 GB | 27.5 |

  GTT stayed at the bank: nothing was evicted. +3-5 % at most.
  - arcint's prefill is not bound by expert ingestion or by small
    per-expert matrices at 2,048 tokens, which is what the recipe's large
    chunks fix.
  - How this differs from the reference: the recipe stages experts from
    NVMe; arcint's bank is already in pinned host memory, read in place by
    the kernels.
- **Sign masks in registers** for IQ2_XXS / IQ2_XS / IQ3_XXS (valarauca's
  SYCL rework, `vecdotq.hpp:634-642`: `ksigns_iq2xs[i]` = the 7 bits plus
  their parity, expanded to byte masks with ALU ops), against the table.
  One build, `GGML_OPENCL_IQ_OPTS=-DLB_SIGNS_TABLE` selecting the table.
  test-backend-ops passes both ways on both cards.
  - **Kernels** (MUL_MAT_ID, Flash-Next's 512-expert / top-10 /
    2560 -> 640 shape):

    | kernel | B60, computed | B60, table | A770, computed | A770, table |
    |---|---|---|---|---|
    | IQ2_XXS, 1 token | 47.5 us | 40.5 | 27.5 | 26.7 |
    | IQ2_XXS, 3 tokens | 114.9 us | 96.1 | 64.2 | 69.2 |
    | IQ2_XS, 1 token | 48.5 us | 43.8 | 31.8 | 31.9 |

  - **Served:** Flash-Next 26.3-28.0 / 34.3-34.9 t/s either way. The coder
    (IQ3_XXS experts, A770) 33.5 / 42.1 t/s both ways, the text identical.
  - **Not adopted.** How this differs from the reference: valarauca's table
    sat in global memory, while arcint's is in local memory (0022), which
    the computed masks do not beat. The table stays; the patch is kept on
    the llama.cpp tree's branch `p0032-regsigns`, not in the series.
- **What the kernel rates show instead:** the IQ2_XXS expert matvec reads
  ~4.2 MB in 40.5 us at one token, ~104 GB/s, about a quarter of the card's
  bandwidth. The A770 runs the same call in 26.7 us. The next decode step
  is a stall profile of that kernel (unitrace `VectorEngineStalls`), not
  the sign source.
- **Is the GPU fed during prefill?** (`measured-here`, 2026-10-10; arcint
  0.7.2 and llama-bench pp2048, 2,048-token ubatches)
  - **The prefill graph** runs in two splits: the input and n-gram embedding
    lookups on the CPU, then one OpenCL split. There is no CPU island
    inside it (`GGML_SCHED_DEBUG=2`).
  - **The host** (unitrace `-h`) enqueues a chunk's ~4,900 kernels in about
    0.1 s (9 us a launch), then waits: the longest single
    `clWaitForEvents` was 4.0 s. `perf` puts 71 % of the host's prefill
    time in the driver's spin wait (`clock_gettime` and syscall entry), and
    under 5 % in real work.
  - **The card's own engine counters** (xe fdinfo `drm-cycles-ccs` against
    `drm-total-cycles-ccs`, per second), served needle: the compute engine
    97 % busy through the 47 s prefill (min 61 %), the copy engine 2 %.
    During the 300-token decode: compute 91 %, copy 81 %.
  - **Prefill is kernel-bound.** Host-side batching (Strata's recorded
    graphs, the B70 recipe's async step) has nothing to win here.
  - **A trap:** unitrace's device timing summed to about half a chunk's
    wall time, which read like a GPU idle half the time. The engine counters
    say otherwise. Kernel timestamps from unitrace leave about half of the
    prefill's GPU time unattributed; use the backend's own event profiling
    or the engine counters for totals.
- **Where a prefill chunk goes, counted fully** (`measured-here`, a
  `GGML_OPENCL_PROFILING` build; per-kernel event times sum to ~4.0 s a
  2,048-token graph, ~85 % of its wall time; unitrace covered about half):

  | part | per chunk |
  |---|---|
  | the expert GEMMs (IQ2_S, Q2_0, IQ2_XXS, IQ1_M) | ~1.67 s (42 %) |
  | the bank gathers | ~0.83 s (21 %) |
  | the dense IQ4_XS / F16 products and conversions | ~0.63 s (16 %) |
  | GDN, hyper-connections, norms | ~0.4 s |
  | attention at depth 0 | ~0.06 s |

  The expert GEMMs run ~9.7 TFLOP a chunk at ~5.8 TFLOPS, about a tenth of
  the card's matrix throughput.
- **The bank stream** (patch 0032, opt-in): the gathers move to the copy
  engine, ahead of the GEMMs (Strata's prefill ring). Served needle prefill
  416.1 -> 497.6 t/s, KL unchanged, the red case aborts.
  - **Open:** eviction with the deployed slot budgets (+751 MiB at a ring
    of 4); a ring of 2 or 3.
  - **Next lever:** the expert GEMMs' efficiency. Strata's notes put its
    Intel prompt path's bound in the dequant that feeds XMX; the B70
    recipe's MoE kernel decodes in registers into DPAS operands, in 32-row
    blocks.
- **The expert GEMM's tile** (`measured-here`, 2026-10-11). The B60's
  MUL_MAT_ID tile `2,2,1,1` is one sub-group of 16 rows x 32 tokens; it
  dequantises its weights once per 32 tokens. At ~40 tokens an expert per
  2,048-token chunk, each expert paid one full and one nearly empty tile.
  This is also why larger chunks did not help: more tokens meant more
  tiles, each dequantising again.
  - llama-bench pp2048 at depth 4,096, stream on: 424 (default) -> 617 t/s
    at `4,4,2,2`.
  - Served needle: 494.7 (default) -> **634.1 t/s at `4,4,2,4`**, answers
    right, KL unchanged (0.294050).
  - With the stream: 0.7.2's 416-422 -> 634 t/s, above Strata's own engine
    on this card (620.5).
  - Before it becomes the B60's default: the coder's prefill on the B60
    with it, and decode in the deployed configuration outside the backup
    window.
- **Adopted** (`measured-here`, 2026-10-11):
  - 0033 makes `4,4,2,4` the B60's MUL_MAT_ID tile.
  - In the deployed configuration (9,500 MiB of slots, 131,072 ctx,
    MTP 2): prefill 408 -> 640-645 t/s with 0032's stream on. Decode equal,
    nothing evicted, answers right.
  - The stream stays opt-in (`GGML_OPENCL_BANK_STREAM=1`). Its ring leaves
    ~220 MiB of the card free in that configuration (24.26 of 24.48 GB),
    and the xe eviction lesson says thin margins bite. Turning it on for
    the unit is the operator's call.
- **In service** (2026-10-11): `arcint-flashnext` runs arcint 0.7.2 with
  llama.cpp 0001-0033 and `GGML_OPENCL_BANK_STREAM=1` (operator).
  - Measured through the unit on :8089: needle prefill 645.1 t/s, decode
    27.3 / 28.4 t/s, answers right, nothing evicted (VRAM peak 24,267 MiB).
  - Neither change touches the dense agent (no experts) or the coder (its
    experts are resident: no bank; on the B60 they take the tokens-as-a
    kernel, which does not read the tile).

### Stage 6 — the low-bit expert matvecs (decode), 2026-10-11

**Where they stall** (`measured-here`, B60, unitrace hardware metrics over
`test-backend-ops perf` at Flash-Next's 512-expert shapes; time-weighted per
kernel):

| kernel | XVE active | stalled | ALU1 (int) busy | main stall |
|---|---|---|---|---|
| IQ2_XXS MUL_MAT_ID matvec | 88.3 % | 10.7 % | 83.5 % | ALU write-back 12.7 % |
| Q2_0 MUL_MAT_ID matvec (the down projections) | 93.5 % | 5.9 % | 89.7 % | ALU write-back 9.5 % |
| Q4_K matvec (reference) | 51.6 % | 48.0 % | 41.4 % | SBID (memory) 53.3 %, 431 GB/s |

- The low-bit expert kernels are bound by integer instruction issue, not
  by memory: unpacking the weights before each `dp4a` fills the pipe.
  Strata's Intel notes record the same for its expert kernels (77 % XVE
  active, ALU-bound). This is also why computing the sign masks in
  registers (stage 5) lost: it moved local-memory loads onto the full
  pipe.
- **Levers, in order of cost:**
  1. **One-term activations.** The matvec dots each weight block with two
     int8 activation terms (the main one and a residual, for accuracy).
     `-DKQ_ONE_TERM` halves the dots; the bar decides.
  2. **Q2_0 unpacked by plane.** Today ~8-9 ALU operations per 4 weights;
     `(w >> 2k) & 0x03030303` on a 32-bit word gives 4 weights in 2
     operations. It needs activations in the matching transposed order (the
     quantize pass, once per token) and the -1 offset as one block-sum
     correction.
  3. The grid types (IQ2_XXS, IQ2_S, IQ1_M): fewer operations around each
     lookup.
- **One-term activations, tried** (`measured-here`, 2026-10-11, B60,
  `GGML_OPENCL_IQ_OPTS=-DKQ_ONE_TERM`):
  - Kernels only 3-11 % faster: IQ2_XXS 40.7 -> 38.5 us at one token,
    IQ2_S 50.6 -> 48.4, Q2_0 down 32.0 -> 28.5.
  - Served decode slightly slower: 29.9-31.9 against 31.3-32.6 t/s, two
    processes each, answers right.
  - Not adopted. The residual dot is not where the time goes; the
    weights' unpacking is.
  - Its KL arm (0.294050) does not count: llama-perplexity's 512-token
    batches take the GEMM, which the matvec's one-term mode does not touch.
- **Q2_0 unpacked by plane, tried** (`measured-here`, 2026-10-11; llama.cpp
  tree branch `p0034-q2-planes`):
  - The idea: `(w >> 2k) & 0x03030303` gives four weights in two
    operations (about nine before), with the activations' 16 bytes
    transposed to match.
  - test-backend-ops passes on both cards; the red switch (activations not
    transposed) fails 49 of 87 Q2_0 cases.
  - **B60:** the 16-lane path is unchanged (33.0 against 33.4 us at one
    token). The 4-lane down path is slower: 52.1 against 32.1 us at one
    token, 152.7 against 90.0 at three, because each lane transposes the
    activations for its single row.
  - **A770:** 7-14 % faster.
  - **Served decode (B60):** 29.8-31.2 against 30.3-31.8 t/s. Not adopted.
  - **What it shows:** cutting the unpacking's instructions by about two
    thirds did not move the 16-lane path. The integer pipe's load is not
    mainly the unpacking. Next suspect: the per-sub-block index arithmetic
    in 64-bit integers (`ulong` block indices, emulated on Intel GPUs).
- **The ISA, read** (`measured-here`, 2026-10-11, `ocloc compile -device bmg`
  plus `ocloc disasm`):
  - IGC lowers `char4` arithmetic one byte at a time: a move into a byte
    register, a widen, the operation, a pack back.
  - Moves are 526 of the IQ2_XXS matvec's 1,405 instructions, the largest
    class. The 64-bit index arithmetic (stage 6's suspect) was not counted
    separately.
- **Patch 0034, adopted:** the per-byte arithmetic on 32-bit words (SWAR),
  bit-exact. The details are in `contrib/llama.cpp/README.md`.
  - Movs fall by about half (IQ2_XXS 526 -> 206).
  - B60 kernels are 22-39 % faster: IQ2_XXS 40.7 -> 25.0 us, Q2_0 down
    32.1 -> 24.9.
  - Served Flash-Next decode is ~5 % faster: 31.7-33.5 against 30.7-31.9
    t/s, answers right, KL unchanged.
  - The coder on the A770: 42.1 t/s both ways.
  - A 22-39 % kernel gain gives ~5 % served: the expert matvecs are only a
    part of the decode round. Their share after 0034 is not measured yet.
  - **In the unit** (`measured-here`, 2026-10-11): `arcint-flashnext` runs
    0.7.2 + 0001-0034 since 0034's adoption, in the deployed configuration
    (9,500 MiB, 131,072 ctx, MTP 2, 0032's stream). The live check through
    Roundhouse:
    - answers right (capital, needle);
    - 20k prefill 648.0 t/s;
    - decode 27.5 / 29.3 t/s, against 27.3 / 28.4 on the 0033 binary. That
      is inside the spread: with 9,500 MiB the misses take a larger share of
      the round than in the 13,000 MiB A/B;
    - VRAM peak 24.24 GB, nothing evicted.
- **The decode round after 0034** (`measured-here`, 2026-10-11, B60, the
  unit's configuration):
  - profile: a profiling build of the same tree, served, the second long
    answer's decode only (300 tokens, MTP 2, ~2.3 tokens a round);
  - 35.6 ms a token of wall time, 26.9 ms of it kernels (1,810 launches a
    token).

  | share of a token | time |
  |---|---|
  | expert matvecs | 10.8 ms |
  | dense matvecs | 11.1 ms (IQ4_XS 4.5, F16 4.2, the draft head's Q6_K 1.05) |
  | the rest | 4.9 ms |

  - **The expert matvecs are bound by the misses' link time,** not by
    their ALU work: an IQ2_S call takes 30-50 us when all its experts hit
    and ~50 us more for each expert read over the link (a llama-bench
    profile's duration histogram). That fits 0034's kernel gain showing
    less at 9,500 MiB than at 13,000; it was not measured further.
  - **The dense IQ4_XS matvec was ALU-bound** (XVE active 68 %, ALU1 62 %;
    Q4_K 52 / 41 % with a 53 % memory stall) at half of Q4_K's bandwidth:
    patch 0035.
  - **The F16 products** (the hyper-connection pair 2.9 ms, the router 0.44,
    the injects 0.39) were measured in stage 3c: `KSPLIT4` was faster in
    isolation and not served, and the fused hyper-connection read was no
    faster. Not repeated.
- **Patch 0035, adopted:** the IQ4 codebook as four pre-shifted tables in
  local memory, on both cards. The details are in `contrib/llama.cpp/README.md`.
  - Kernels 22-43 % faster on the B60, 21-27 % on the A770 at one column.
  - Served Flash-Next: device time 26.89 -> 26.09 ms a token; wall inside
    the spread.
  - The coder on the A770: 41.0-42.1 -> 41.9-42.9 t/s.
  - Bit-identical outputs.
- **Release 0.7.3 in service** (`measured-here`, 2026-10-11, the package's
  `/usr/bin/arcint`, `87259ef`):
  - `arcint-flashnext` (B60, 9,500 MiB of slots, 131,072 context, MTP 2,
    `GGML_OPENCL_BANK_STREAM=1`), live through Roundhouse: the capital, the
    needle and both long answers right; 20k prefill 648.5 t/s; decode 28.2
    / 29.9 t/s; VRAM peak 24.24 GB.
  - The coder (A770, 98,304 context): decode 42.9 / 41.8 t/s, answers right.
  - The dense agent (B60, 131,072 context): decode 37.2 / 44.6 t/s, answers
    right.
- **OpenCL on Level Zero ("LEO") and command buffers** (2026-10-11).
  - **The driver's own plan** (`code`, intel/compute-runtime
    `documentation/LEO.md`, `LEO_COMMAND_BUFFER.md`): LEO re-implements
    OpenCL on Level Zero. There `cl_khr_command_buffer` is a regular Level
    Zero command list, the primitive SYCL graphs replay. It is partial and
    off by default (`NEOReadDebugKeys=1 EnableLEO=1
    EnableClKhrCommandBuffer=1`); mutable dispatch is not implemented.
  - **Which release has what** (`measured-here`): 26.27 (installed) has LEO
    but no command buffer. 26.35, in the test container, exposes the
    extension, but recording a kernel (`clCommandNDRangeKernelKHR`)
    resolves to null; that entry point landed in the source after the
    release.
  - **LEO's enqueue cost** (`measured-here`, B60, the chain of 2,000
    dependent tiny kernels): 1.6-1.9 us a kernel against native 2.63 (26.27);
    2.06 against 2.66 (26.35).
  - **LEO and our kernels** (`measured-here`): LEO reports no non-uniform
    work-groups (native: yes). test-backend-ops MUL_MAT and FLASH_ATTN_EXT
    abort on `CL_INVALID_WORK_GROUP_SIZE`, e.g. upstream's Q1_0 conversion
    (64-wide work-groups over a global size that is not a multiple of 64).
    MUL_MAT_ID passes (468/542 B60, the same as native). ggml-opencl falls
    back for devices without non-uniform work-groups in three places only.
  - **Served under LEO** (`measured-here`, B60, the units' configurations,
    two processes per arm, answers right in every arm):

    | | native | LEO |
    |---|---|---|
    | dense agent decode | 36.8 / 40.2, 36.8 / 40.2 | 37.1 / 40.5, 37.1 / 40.5 |
    | Flash-Next decode | 28.7 / 28.6, 28.2 / 30.3 | 28.6 / 30.2, 27.9 / 30.1 |
    | Flash-Next needle prefill | 632.1, 646.4 | 632.5, 631.7 |

  - **What it shows:** a 25-40 % cheaper enqueue does not move Flash-Next's
    decode, so the GPU is not waiting on enqueues. Stage 3e's idle stretches
    (8.8 ms a round between the last draft and the next verify) are sync
    waits and host work (graph building, reading routing ids). A command
    buffer removes enqueue time as well, so on this evidence it would not
    buy decode either. Its replay was not measured: no release records
    kernels yet. The dense agent's +0.8 % is within one run's spread but
    repeated.

### Stage 7 — the VRAM budget against Strata's (2026-10-11)

Strata keeps 12,332 experts on the B60 (16.53 GiB) at 32K, arcint 9,432
(12.88 GiB) at the same context. Each expert a token does not find on the
card costs ~50 us over the link (stage 6), so the budget is a decode lever.

**Where each engine's VRAM goes at 32,768 tokens** (`measured-here`, B60;
Strata from its own load log, arcint 0.7.3 from llama.cpp's buffer lines,
its expert-cache line and fdinfo; 13,000 MiB expert budget, MTP 2, the bank
stream on):

| item | arcint | Strata |
|---|---|---|
| expert slots | 9,432, 12.88 GiB | 12,332, 16.53 GiB (sized per expert, 1.37 MiB mean) |
| dense weights | 3,264 MiB | ~2,940 MiB (1,466 from its pack, 1,475 read natively) |
| MTP draft model | 1,944 MiB (Q4_K / IQ4_NL experts, Q6_K head) | 839 MiB (Q2_0 experts) + 138 MiB draft head |
| KV cache | 1,040 MiB (f16) | int8 |
| recurrent state | 338 MiB (MTP 2: three copies) | |
| compute buffers | 1,217 MiB, reserved at load for 2,048-token batches | allocated per request by its prompt path |
| bank-stream ring | 751 MiB (4 x 187.7 MiB, at the first prefill) | |
| gather mirror | 187.7 MiB | |
| VRAM | 19,695 MiB after load, 21,128 after the first answer, 21,932 peak in the 20k needle | 1,559 MiB free with everything loaded |

- Slots are sized per layer from the layer's own tensors (`code`,
  `llama-expert-cache.cpp:310-333`): no padding. The GGUF's experts are
  1.123-1.440 MiB (mean 1.376).
- The growth on the first answer is not scratch: the lazily grown
  `ggml_cl_buffer`s add 1.4 MiB on it (an env-gated allocation log in the dev
  tree, `measured-here`), and 751 MiB of ring plus ~50 MiB on the needle. xe
  makes pages resident on first touch, so the compute buffers reserved at
  load show in fdinfo only after the first answer.
- Strata's draft experts are Q2_0, "the same format as the main model's
  experts", picked by measured draft acceptance (`code`, Strata
  `tools/mtp_pack.py:1-15`). Ours are Q4_K gate/up and IQ4_NL down. A
  Q2_0-expert draft model (`mtp-flash-next-q8e2n`, 1,767 MiB file against
  2,452) had been built on 2026-10-06 and never measured.

**The levers** (VRAM freed for slots): Q2_0 draft experts ~0.68 GiB; q8_0 KV
~0.5 GiB at 32K and ~1.9 GiB at 131K (an earlier q8_0 arm, 0.6.0, was
slower: 23.6 against 31.3 t/s); prefill buffers per request instead of
resident, up to ~2 GiB (Strata's prompt path); the dense weights ~0.3 GiB.

**The levers measured** (`measured-here`, B60, the unit's configuration:
131,072 tokens, MTP 2, the bank stream; each arm's freed VRAM given to slots;
one process per arm, the needle and two different long answers; answers
right in every arm; the card has 24,480 MiB, xe debugfs `vram_mm`):

| arm | slots | hit rate | drafts accepted | prefill | decode | peak VRAM |
|---|---|---|---|---|---|---|
| A: today (Q4_K/IQ4_NL draft experts, f16 KV, 9,500 MiB) | 6,877 | 89.3 % | 339/518 | 646.9 | 27.4 / 29.4 | 24,259 MiB |
| B: Q2_0 draft experts, 10,150 MiB | 7,351 | 90.5 % | 320/556 | 642.6 | 26.7 / 29.4 | 24,208 |
| C: q8_0 KV, 11,300 MiB | 8,191 | 91.5 % | 334/531 | 521.5 | 13.3 / 13.4 | 24,102 |
| D: both, 11,950 MiB | 8,669 | 92.2 % | 315/566 | 519.4 | 12.5 / 12.6 | 24,033 |

- **Q2_0 draft experts** (Strata's choice) lose about as many accepted
  drafts (65 -> 58 %) as the 474 extra slots gain in hits: no decode gain.
  Not adopted on this evidence; Strata's choice was measured on its engine.
- **q8_0 KV** halves Flash-Next's decode and costs 19 % of prefill, while
  the dense agent runs q8_0 KV at f16's speed (agent-context). The quantized
  cache takes a slow path in Flash-Next's attention (QSA, 256-wide heads):
  a kernel gap, not a property of q8_0 KV. Not adopted until it is found; it
  would free ~1.9 GiB at 131K.
- The KL of q8_0 against f16 KV was not measured: the perplexity run
  thrashed on the first shard's memory-mapped reads from the ZFS pool with
  the pinned bank in host memory (126 GB read in 31 min), and was stopped.

**Open after stage 7** (2026-10-12):
- **Why q8_0 KV halves Flash-Next's decode.** The slowdown shows on 300-token
  answers over an almost empty cache, so it is a fixed per-token cost, not
  attention length. `--llama-kv` also types the QSA indexer's key cache,
  whose scoring is plain ops; the first check is the scheduler's split dump
  (`GGML_SCHED_DEBUG=2`), f16 against q8_0. A window for it was aborted:
  the load stalled (below).
- **The served GGUF's first shard is on the ZFS pool.** The model directory
  on the NVMe holds shard 2 (the 28.8 GB n-gram table) and a symlink for
  shard 1 (39 GB: dense weights and experts) into the pool
  (`measured-here`). After other models' runs had pushed it out of the page
  cache, a Flash-Next load sat 20 minutes in ZFS reads (50 % I/O pressure),
  and a KL run read 126 GB in 31 minutes. Moving shard 1 to the NVMe needs
  ~32 GB freed there first (7.2 GB free).
- **Prefill buffers per request** (Strata's prompt path): up to ~2 GiB more
  for expert slots; needs code.

**Stage 7, the q8_0 KV slowdown found** (2026-10-13): OpenCL GET_ROWS had
no q8_0 source, so the QSA indexer's gathers from a q8_0 key cache ran on
the CPU. The scheduler copied each layer's 8 MB cache across, in the target
graph and in every MTP draft (`measured-here`: 222 of 444 splits on the CPU
against 52 of 104 with f16 KV). Patch 0036 adds the kernel. At the unit's
131K with the freed VRAM given to slots: decode 30.9 / 31.0 t/s against f16's
30.0 / 31.3 (13.3 before), hit rate 91.4 against 89.3 %, needle prefill 617
against 650, answers right, KL 0.294037 against 0.294050. The decode gain is
inside the spread: at this budget the extra 2 points of hits do not show.
What q8_0 KV buys the unit is room: the q8_0 arm ran with 1,800 MiB more
expert slots than the f16 arm (11,300 against 9,500 MiB) at 131K, room the
deployed unit (peak 24,259 of 24,480 MiB) could keep as margin or spend on
slots. A lane's checkpoint restore does not touch the q8_0 caches (it
restores the f32 recurrent state only; 0036's README section).
- **Adopted with 0.7.4** (2026-10-13): the unit runs q8_0 KV with 11,300
  MiB of slots (`measured-here`: the 0.7.4 package, the unit's flags, one
  window; the needle, the capital and a long answer right in both arms):

  | arm | prefill (20k) | decode | peak VRAM |
  |---|---|---|---|
  | q8_0 KV, 11,300 MiB | 662.7 t/s | 31.7 t/s | 24,241 MiB |
  | f16 KV, 9,500 MiB | 660.9 t/s | 28.9 t/s | 24,278 MiB |

  An earlier f16 run in the same session decoded at 31.2 t/s, so decode is
  equal within that spread. 0.7.4's attention kernels against 0.7.3's on
  f16: 664.6 / 31.2 against 653.0 / 31.1, answers the same.
- **The served GGUF's first shard** is a real file on the NVMe since
  2026-10-13 (sha256 = Hugging Face's); loads no longer read the ZFS pool.

### Stage 8 — the verify's prologue: what the GPU waits for before each verify (2026-10-13)

Stage 3e left ~10 ms a round outside the kernels; LEO (stage 6) ruled out
the enqueue cost. Where it goes (`measured-here`, B60, the 0.7.4 package,
the unit's flags: q8_0 KV, 11,300 MiB of slots, MTP 2, the bank stream;
`LLAMA_DECODE_TIMING=1`, `GGML_SCHED_TIMING=1`; 300-token answers):

- A verify graph runs in two splits: on the CPU the token embedding and the
  n-gram table's rows (`GET_ROWS` from `per_layer_token_embd`, 27,465 MiB),
  then one OpenCL split with 34 inputs. The drafts have been synchronized,
  so the GPU idles through everything before the OpenCL split's first
  kernel:
  - the CPU split: 0.9-8.9 ms a verify over the exploration runs and the
    cadence sweep below, depending on the page cache;
  - the OpenCL split's input copies: 0.94 ms a verify without swaps (the
    sweep's "off" arm; its first block, 1.85 ms, holds prefill graphs),
    plus 0.4-5 ms waiting for the expert cache's swaps;
  - the expert cache's step (counting the previous verify's routing):
    0.68 ms.
- **The n-gram rows were synchronous faults.** The CPU split costs 36 us
  when every row is in the page cache (a repeated prompt, 0 major faults)
  and 1.2-5.4 ms otherwise (the second exploration run: ~20 major faults a
  verify, 2,162-2,370 an answer).
  - The cause (`code`): llama.cpp marks the table for lazy reading
    (`TENSOR_READ_LAZY`: `MADV_RANDOM` on its range, each graph's rows
    prefetched with `MADV_WILLNEED`, `llama_prefetch_rows`), but its AUTO
    lazy mode resolves to OFF for any device without mmap support
    (`src/llama-model.cpp:1525-1533`), the OpenCL device among them. The
    table was then an ordinary mapped weight: no prefetch, and each fault
    read the file's readahead around the row (the loader's
    `POSIX_FADV_SEQUENTIAL` makes the kernel double it), which also evicted the table's
    own pages (54-73 % resident with the bank pinned).
  - A file probe (`measured-here`, 48 random rows a gather, the table 70 %
    resident): plain faults through an `MADV_RANDOM` mapping 1,433 us
    (median), `MADV_WILLNEED` a row then the reads 258 us, `O_DIRECT` on 8
    threads 874 us. The last is not Strata's reader: that runs 16 I/O
    threads (`src/platform/direct_file.cpp:313`), a submit worker and an
    8-way row cache (`src/ngram/ple_reader.cpp:34-97`), and the probe's
    `O_DIRECT` reads skip the page cache that served ~70 % of the other two
    arms' rows. Not a verdict on it.
  - **Fix** (arcint `2e3b964`; the measured binary was a build of the same change before its comment was extended): `--llama-cpu-moe` (the memory-mapped load)
    sets `LLAMA_LAZY_MODE_ON`; it applies only to tensors an arch marks
    lazy (`qwen4exp`, `gemma4`). A/B, A B A B, one process per arm, the
    capital, the 20k needle, three 300-token answers (`measured-here`):

    | | 0.7.4 package (two runs) | lazy table (dev build of `2e3b964` + 0001-0041) |
    |---|---|---|
    | CPU split a verify | 2.0-7.7 ms | 0.30-0.59 ms |
    | decode: hash table / compiler / TCP | 30.5, 31.8 / 33.8, 35.7 / 32.6, 34.0 t/s | 32.2, 32.3 / 36.2, 36.2 / 34.4, 34.5 |
    | 20k needle prefill | 661.2, 662.8 t/s | 671.6, 673.2 |
    | load to listening | 137, 126 s | 92, 86 s |

    The dev build also carries 0041, a prompt-attention kernel: it can
    account for part of the prefill row, not for the decode rows. The load
    is shorter because the shard holding the table is no longer mapped with
    `MAP_POPULATE` (`code`, `src/llama-mmap.cpp:492-493`): its other weights
    are faulted on demand.

    Decode +3.7 % on the mean and steady across runs; the answers right in
    every arm, the draft acceptance counts identical (170/259, 169/262,
    163/252): the same tokens.
- **The swaps wait on the compute queue.** Each adapt exchanges experts
  between slots and the bank on the card's in-order queue ahead of the next
  graph (stage 2b), so the copies' link time stalls the GPU. The cadence
  priced (`measured-here`, one process per arm, three answers):

  | adapt every | decode t/s | hit rate | swaps an answer | input copies a verify |
  |---|---|---|---|---|
  | 6 tokens (today) | 31.5 / 34.9 / 33.3 | 90.0-93.9 % | 1,789-3,144 | 1.3-3.0 ms |
  | 12 | 31.7 / 35.8 / 33.7 | 88.7-93.8 % | 1,798-2,768 | 2.4-5.8 ms |
  | 24 | 28.6 / 32.9 / 32.1 | 84.8-92.2 % | 1,536-1,604 | 2.4-3.3 ms |
  | off | 23.3 / 24.5 / 21.1 | 65.3-72.3 % | 0 | 0.95 ms |

  Adapting is worth +45 % of decode (the means, 33.2 against 23.0 t/s; +35
  to +58 % an answer); its copies cost 0.4-5 ms a verify on
  top, and with the n-gram rows fixed nothing on the host overlaps them any
  more. Strata (`code`, `src/program/generate.cpp:4396-4480`,
  `resident_stage_swaps` at 122-158) runs them on a separate non-blocking
  stream: in its resident RAM mode (the analogue of arcint's bank) the
  victim's bytes are copied back into a spare RAM buffer, waited for on the
  adapt thread, and it is evicted (served from RAM), the newcomer is copied into the freed slot,
  and its table entry changes only once an event says the copy landed,
  polled each round. arcint's blocking exchange is a deviation from that
  mechanism, not from its tuning.
- **The swaps off the compute queue: patch 0042** (`contrib/llama.cpp/README.md`
  has the mechanism and the table).
  - The discriminator (`measured-here`): adapting with
    `LLAMA_EXPERT_CACHE_MAX_SWAPS=0` (no copies, no table uploads) leaves the
    verify's input wait at 0.93-1.0 ms, against 2.5-6 ms with the exchange:
    the swaps are the wait.
  - The first build moved the copies to a second queue but enqueued them
    just before the next verify: the wait stayed (2.4-5.9 ms) and decode
    lost ~1.5 %. A probe showed the second queue does overlap a kernel (a 48
    MiB copy in 3.5 ms beside a 37.5 ms kernel, the pair in 37.5 ms), so what the verify waited for
    was its own input writes on the copy engine, queued behind the swaps.
    Enqueued right after the graph instead, the copies run while it
    computes: the wait 0.94-0.95 ms, decode +2.4 % (33.1 / 37.0 / 35.3
    against 32.3 / 36.2 / 34.5 t/s, means of two runs), prefill equal,
    answers and draft counts identical.
  - An adapt's table uploads are separate small writes (Flash-Next: 3 a
    touched layer): 240 of 2 KiB take 1.64 ms on the B60 against 54 us as
    one write (`measured-here`, a probe). Built as patch 0046, after Strata's
    one residency table and one upload (`code`, `generate.cpp:3589-3606`,
    `4406-4413`): every layer's tables are rows of one tensor, written in one
    call per adapt step. Decode +0.1-0.6 % on each of the three answers (two
    A/B pairs against the 0.7.7 package), the time writing 0.06-0.09 -> 0.03-0.05 s
    an answer, the same text; the record is README 0046.
- **A round's kernels** (`measured-here`, a `GGML_OPENCL_PROFILING` build of
  the 0.7.5 tree, the unit's flags, two 300-token answers; the build decoded
  31.5 / 34.6 t/s against 33.1 / 36.9 for the plain one): 17.1 s of kernel
  time in the whole run (three short prefills and the capital's step
  included) over the answers' 261 rounds: at most ~65.5 ms of kernels in a
  round of ~69.6 ms (the plain build), so at least ~4.1 ms of it is not
  kernel time. Stage 3e measured
  Strata's kernels at ~64 ms a round on this card; what remains between the
  engines is mostly tokens per round on these texts, not idle time.
- **Short prompts: patch 0043** (`measured-here`). Every request on Flash-Next paid ~1.9 s to
  prefill a 24-30-token prompt: a GEMM ubatch with the bank stream copies
  each bank tensor whole (~22 GiB over the link) whatever its tokens. An
  expert cache's tensors now take the GEMM from 64 tokens: 24 tokens 1.87 ->
  0.64 s, 28 tokens 1.82 -> 0.57; 64 tokens and up unchanged (a 166-token
  figure of 1.51 was the stream's learning graph, gathered), KL inside the
  bar (`contrib/llama.cpp/README.md`, 0043, has the tables). Open: the
  streamed GEMM costs ~1.84 s a graph up to ~1,000 tokens; gathered is
  faster only below ~250 tokens (173: 1.56 against 1.84; 317: 1.96 against
  1.84), at most ~0.3 s: priced; built as patch 0044 (the stream from 256
  tokens a ubatch: 77 tokens 1.84 -> 1.11 s, 173 1.83 -> 1.54, 269 and up and
  the 20k needle unchanged, the same tokens; `measured-here`, README 0044).
- **Where decode stands** (`measured-here`, B60, the unit's flags): the
  0.7.4 package 30.5-31.8 / 33.8-35.7 / 32.6-34.0 t/s on the three answers;
  with the lazy table and 0042 33.1 / 36.9-37.0 / 35.2-35.4. Strata on this
  card: 37.2-37.8 (stage 3, its long-answer run).
- **The kernel count (2026-10-14, patch 0049, README 0049):** the 0.7.5
  profile's verify graph: ~4,150 launches, 58.4 ms of kernels; the small ones
  dominate the count (quantize 284, rms_norm_mul 202, state copies 274,
  hc_pre/post 202, scale_act 210 a graph). Built: the rollback snapshots
  without a cont, and llama.cpp CUDA's rms_norm+scale, ssm_conv+silu and
  sigmoid+mul fusions, bit-identical (an opaque zero keeps the fused norm's
  two roundings; regrouped, it passed KL but cut draft acceptance). The
  verify graph's scheduler time 12.35-12.41 -> 11.67-11.68 ms; decode +0.5 to
  +1.1 %, the same text. What remains of the count: the activation
  quantizations (one before each quantized matvec), the f16 hyper-connection
  matvecs with their scale_act, and the norms; each fusion is worth ~0.4 % of
  decode a hundred launches.
- **The GDN stage after vllm-xpu-kernels (patch 0050 and on).** Intel's
  Xe2 GDN for speculative decode is two fused stages a layer (`code`,
  `csrc/xpu/gdn_attn/gdn_attn_interface.cpp`): the conv stage (conv1d,
  SiLU, the q/k l2-norm, the conv cache advanced by the accepted count) and
  the delta stage (the a/b gating inside, the state updated in place). Built
  so far: 0050, the snapshots written into the cache by the kernel (README
  0050: Flash-Next +1.2-1.9 %, the dense agent +5.1-5.3 %, the coder +1.8-2.0
  %, bit-identical). 0051, the gating into the GDN kernel (README 0051, 4
  launches a layer): bit-identical; Flash-Next neutral (32.7-32.8 / 38.3 /
  36.6-36.7 against 32.7-32.8 / 38.2-38.3 / 36.6-36.7), the dense agent +0.3
  %, the coder +0.9-1.2 % (`measured-here`). The launches leave the host
  (Flash-Next's verify graph 11.5-11.6 -> 11.1-11.4 ms of enqueue time), but
  Flash-Next's round (~63 ms) is not set by the host: a launch fusion pays
  where the host is the bound (the coder on the A770). Two traps on the
  way, both in the README: the first build read raw alpha after the
  allocator had reused its memory (answers still right, md5s moved), and a
  map over the whole graph each compute cost what the launches saved.
  Next: the conv state's concat and snapshots into the conv kernel; the
  rollback by an accepted-count offset instead of K snapshots is the larger
  change, priced separately.
- **The conv stage, priced** (`measured-here`, 2026-10-15, one binary with
  0053; llama tree `p0054-conv-ceiling`, not in the series).
  `LLAMA_CONV_SNAP_CEILING=1` drops the conv state's K-1 non-final rollback
  snapshots (one strided copy each, a GDN layer): rollbacks restore stale
  state, acceptance falls, so the comparison is a verify round at the same
  draft rows, rounds = tokens - accepted.

  | service | rounds, drafts a round | verify a round, with / without the copies |
  |---|---|---|
  | dense agent, B60, MTP 5 | 108 / 45, 4.97 / 5.1 | 61.1 / 60.4 ms (-1.1 %) |
  | coder, A770, MTP 4 | 115 / 197, 3.96 / 3.95 | 37.4 / 36.8 ms (-1.6 %) |

  Two processes an arm, identical numbers. A conv kernel that writes its
  snapshots and takes the concat too saves about K / (K - 1) of that:
  ~1.3 % and ~2.0 % of a verify, ~1.1 % and ~1.4 % of decode. The
  reference's form (`code`, vllm-xpu-kernels `gated_delta_rule.hpp`: every
  token's state to its own slot, the next step reading slot
  `num_accepted - 1`, no copy back; its conv kernel keeps one window and
  stages a multi-token step through a temporary) is a change to the memory
  module's rollback, not a fusion. Below the charter's 2 %; not built.
- **The activation quantize, priced** (`measured-here`, 2026-10-15, llama
  tree `p0054-conv-ceiling`, not in the series): the OpenVINO-style post-op
  fold would write the q8_1 activation in its producer instead of a launch
  of its own (one a quantized matvec input; q/k/v and gate/up already share
  one). `GGML_OPENCL_KQ_QUANT_CEILING=1` skips the launch (the matvec reads a
  stale buffer):
  - the dense agent (B60, MTP 5): every draft rejected but the answers right
    (its 6-row verify does not take this path); at 4.95 drafts a round the
    propose step 8.9 -> 8.3 ms and the verify 61.2 -> 60.8 ms, ~1.4 % of a
    ~70 ms round as the whole ceiling;
  - the coder and Flash-Next: empty answers (their verify depends on it), no
    round to compare (Flash-Next's arm also faulted the B60, `xe` "Fault
  response: Unsuccessful -ENOENT" with a coredump: a stale activation
  buffer read as indices; the card served normally in every run after,
  no reboot). Flash-Next's verify graph has 284 quantize launches
    (0.7.5 profile); at 0049's rate (~0.4 % of decode a hundred launches)
    that is ~1.1 %.
  A fold keeps the quantization work and removes the launch: below the
  charter's 2 % on each service; not built.
- **Infernix's LFRU against Strata's adapt: patch 0055** (`measured-here`,
  2026-10-15, B60, the 0.7.9 unit's flags with 0053/0054, one binary an
  arm pair; README 0055). Infernix scores an expert uses / (now - last + 1)
  with uses never reset, protects the round's routed experts and admits a
  miss only over a lower-scoring victim (`code`, `expert_cache.cpp`
  `LfruPolicy`). Here per layer, inside the existing slots and swap queue
  (`LLAMA_EXPERT_CACHE_POLICY=lfru`, opt-in), up to 4 admissions a layer
  an adapt:

  | arm | decode t/s, Strata's request 500 / 300 / 300 tokens | on the card |
  |---|---|---|
  | Strata's adapt, 5 processes | 38.7-39.4 / 38.6-39.0 / 41.7-42.0 | 89.0 / 93.3 / 95.4 % |
  | LFRU, budget 1, 2 | 38.2-38.9 / 39.6 / 42.1 | 88.3 / 94.4 / 95.8 % |
  | LFRU, budget 4, 4 | 39.0-39.5 / 39.9-40.2 / 42.1-42.4 | 89.6 / 95.1 / 96.2 % |
  | LFRU, budget 8, 1 | 39.1 / 40.0 / 42.0 | 89.6 / 95.1 / 96.2 % |
  | LFRU 4, rejected drafts not credited, 1 | 38.8 / 39.8 / 42.0 | 89.5 / 94.9 / 96.0 % |
  | LFRU 4, Infernix's exact clock and candidates, 2 | 39.0 / 39.8 / 42.2-42.4 | 89.3 / 94.6 / 96.0 % |

  Every arm: Paris, the 20k needle in 30.8-31.2 s, the same texts (md5).
  Budget 4: +1.6 % on the means (40.4 against 39.8 t/s; first written +1.3 %,
  an arithmetic slip corrected in the 0.7.10 release review), hit rate +0.6 to
  +1.8 points; the short answer after the 20k prompt sits lower (53 against
  59 % of its 24 tokens' experts, the needle time unchanged). Not crediting
  rejected drafts (Infernix's `live` columns, the next ubatch's position
  minus the last) did not help and is not in the patch. The review noted
  the port's two departures from the reference (a clock tick a step, not a
  layer call; every used bank expert a candidate, not only the step's
  misses); the exact form (`LLAMA_EXPERT_CACHE_LFRU_EXACT=1`) measured
  between the port and Strata's rule, so the reference's mechanism pays
  here too. The cadence (Infernix admits every round; here at each adapt):
  every 6 tokens at budget 4 39.1 / 39.9 / 42.2 t/s, every 3 at budget 2 or
  4 38.8-38.9 / 39.7 / 42.1-42.2, every 12 at budget 8 38.3 / 39.6 / 42.3
  (one process each, the same texts): the defaults stay.
- **Infernix's warm start, tried** (`measured-here`, 2026-10-15, B60, the
  LFRU policy; llama tree `p0056-warm`, not in the series): the live
  placement written at exit as an STRP profile (`LLAMA_EXPERT_CACHE_SAVE`:
  the slotted experts first, then by use) and loaded as
  `--llama-expert-profile`. Warmed on three 300-token answers, measured on
  Strata's request and the needle: 89.5 % on the card, 38.4 t/s, against
  the static calibration profile's 90.1 % and 38.9 (twice); the needle and
  the texts unchanged. The calibration profile is the better seed here.
  How this differs from Infernix (PI:2353-2357): it saves the policy's
  counts and fills half the frames from them; this saves a ranking and
  fills all slots. Not a verdict on a warm start seeded from the same
  workload.
- **The series head against 0.7.9** (`measured-here`, 2026-10-15, B60, the
  unit's flags, the installed package against 0001-0055 with
  `LLAMA_EXPERT_CACHE_POLICY=lfru`, alternated): the package 38.5-39.0 /
  38.4-38.8 / 41.4-41.7 t/s, the head 39.4 / 40.2 / 42.5 (+2.7 % on the
  means; a second head process ran its first answer at 25.5 t/s with the
  same hit rate, all of it in verify, nothing in the kernel log: not
  explained, not counted). The head with Strata's rule 38.8 / 38.7 / 41.7.
  Paris, the 20k needle and the texts (md5) the same in every arm. Infernix's global
  pool (one capacity over all layers) is not built: the slots are
  per-layer tensors.
- **The input copies.** The OpenCL backend has no events and no async copy,
  so the scheduler synchronizes the queue and writes blocking for each of a
  verify's 34 inputs (positions, masks, indices): 17 us a pair on the B60
  (`measured-here`, a probe: a sync 10.6 us, a blocking 1 KiB write 10.4,
  USM-staged non-blocking copies 4.1 us an input). 0.94 ms a verify
  measured, against 34 x 17.3 us = 0.59 ms from the probe: the larger
  inputs (masks, the rotations) take longer than its 1 KiB. A draft graph
  (17 inputs) spends ~0.4 ms (the first exploration run).
  Built as patch 0047 (README 0047): the scheduler waits once a split
  before its user inputs, and the OpenCL backend stages small writes into
  non-weight buffers in a USM host ring, enqueued without a wait. A verify's
  inputs 926-949 -> 582-593 us, a draft's 395-404 -> 134-141; decode +1.1 to
  +1.4 % on each of the three answers (31.6 / 36.8 / 35.3 -> 32.0 / 37.3 /
  35.7-35.8 t/s, two A/B pairs), prefill and the text unchanged
  (`measured-here`). The same path on the other units: the dense agent's
  decode +0.8-1.3 %, the coder's on the A770 +14-15 % (its inputs ~2.0 ->
  0.13-0.25 ms a graph behind the chipset link), the same text.
