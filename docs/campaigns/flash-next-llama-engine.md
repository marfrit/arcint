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
(register selects were slower too, stage 3b). Layout is not the bound at a
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
