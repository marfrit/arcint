# flash-next-llama-engine — Qwen3.8-Flash-Next served by the libllama engine on one Arc card + RAM

**Open.** Stage 1 (static placement, IQ kernels) built and served. Stage 2b
(per-expert slots on the card, a bank in USM host memory, Strata's adaptive
swaps) passed its gate 2026-10-06: prefill 81.9 -> 149.6 t/s at 20k, decode
at parity, needle answered.

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

## Where it lives

`contrib/llama.cpp/patches/` (ggml-opencl kernels, later the expert tier);
`src/exec/backend_llama.cpp` (placement flags).
