# Research: Infernix (Flash-Next on one RTX 5090, at source)

Infernix (`github.com/wallawalla47/infernix`, read at `fd2aa93c`, checked out
as `~/src/Infernix-ref`) is a C++/CUDA engine for Qwen3.8 that grew from
NInfer. It serves Flash-Next with offloaded experts on one RTX 5090 (PCIe
Gen5 x8, 27.5 GB/s measured by its author).

It claims, against Strata on the same machine:
- 4.6-6.4x sooner first token and 1.6-1.8x decode, from 8K to 250K;
- in a replayed agentic workload: 5.9 s against 32.9 s TTFT, 84.6 % against
  53.8 % of prompt tokens from cache, 125 against 86 tok/s, 7.8 against
  22.3 min of wall time.

Every throughput number below is Infernix's own (`paper` for arcint). Code
claims carry file:line (`code`, read 2026-10-09). Paths are relative to the
checkout, with these abbreviations:

- **PI:** `src/models/qwen4_exp/program/program_impl.h`
- **ML:** `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`
- **ER:** `src/models/qwen4_exp/program/expert_residency.{h,cpp}`
- **EC:** `.../program/expert_cache/expert_cache.{h,cpp}`
- **D:** `docs/maintainer/qwen3_8-flash-next-design.md`

## What the Strata comparison mixes

- **Expert bytes:** Infernix runs NVFP4 experts (2,764,800 B each); Strata
  runs UD-Q4_K_XL (~13 % more bytes per expert). `docs/maintainer/model-quality.md`
  §4.2.
- **Dense weights:** Infernix's are 8-bit, 5.1 against 8.6 GB read a token.
- **Lanes:** Strata ran one request at a time, Infernix two lanes
  (`bench/agentic_ab/README.md`).
- **No ablation against Strata.** The speedup is not split by mechanism.
  The replay's launch flags are not in the repository.

## Expert offload in decode

- **Tiers:**
  - VRAM is one global pool of frames, one expert record each, about 9,300
    frames (23.95 GiB) in a VMM arena that grows and shrinks at round
    boundaries (ER.h:28-31, ER.cpp:101-192);
  - pinned host RAM holds all 24,576 records (64.5 GiB);
  - an optional SSD tier sits behind a decayed-LFU RAM tier
    (`expert_cache/host_tier.h:17-40`).
- **The residency table lives on the device.** The kernels look experts up
  themselves, so an all-hit layer never involves the host (ML:116-123).
- **Policy: a global LFRU,** scored as count / (now - last + 1). The clock
  advances per routed layer call, and counts survive eviction
  (EC.h:39-43, EC.cpp:36-38, 60-100).
  - **Admission:** budgeted, on demand: the best-scoring misses, each only
    if it outranks its victim (EC.cpp:147-165).
  - **Promotions:** one per layer a token until the frames fill, then one
    every N tokens. N = 4 x 27.5 GB/s / link, clamped to 1-8, which gives 8
    at the B60's link (PI:2043-2078).
  - **Credit:** prefill promotes nothing (PI:1560-1566). Experts routed only
    by rejected draft columns are not credited (ER.cpp:452-453).
  - **Landing:** up to 16 free frames a layer are reserved before each round
    and adopted as resident after it (ER.cpp:339-389, 465-486).
  - **Warm start:** the saved LFRU state fills half the frames at load
    (PI:2353-2357, ER.cpp:533-560).
- **Policy against Strata's, in Infernix's own engine** (`paper`, D:2661):
  LFRU 32.05 against 29.57 tok/s and 83.4 % against 77.9 % hits over 512
  tokens. The Strata-style policy won only on a replayed identical
  sequence.
- **A miss, inside the captured graph** (ML:858-1135):
  - a GPU kernel picks the misses the CPU takes:
    `min(16, M - M/divisor)`, divisor = max(2, round(1 + 2 x 27.5/link)),
    about 5 at 14 GB/s (PI:3891-3894);
  - it writes x and a request to mapped host memory and bumps a sequence
    word. A host thread spins on it, and six AVX2 / VNNI / AVX-512 workers
    compute the experts (`cpu/miss_service.cpp:198-283`);
  - the PCIe share is copied by a 16-CTA kernel in 16 KiB chunks into
    staging slots or landing frames, not by DMA (ML:307). It is computed on
    a side stream while the main stream runs the resident experts and the
    shared expert (ML:1353-1380);
  - a wait kernel spins on the CPU's done word.
  - Measured (`paper`, D:8469): `stage` 34 % and `cpu_wait` 14 % of the main
    stream at 8K; a staged miss costs ~105-120 us.
- **Lookahead prefetch:** studied, not built (D:2749: recall@10 65 % from
  the previous layer).

## Prefill and TTFT

- **Layer walk:** consecutive chunks (up to 65,536 tokens) run
  layer-major, so a layer's experts cross the link once per span. The chunk
  grid and the output bits are unchanged (`program/prefill_walk.cpp:1-75`,
  PI:4157). Measured (`paper`, D:7489-7516): x1.85 at 128K, x2.57 at
  pp16384, x4.64 at chunk 1024.
- **Blind DMA stream:** for chunks of 256+ columns, every non-resident
  expert of a layer goes by the copy engine, two layers ahead, into ring
  halves lent by the cache (`execution/expert_stream.cpp:53-164`). Measured
  (`paper`, D:7279): +38.8 % / +45.8 %; sizing the ring to the worst layer
  +9.2 %.
- **Wide route:** experts with more than 8 columns go to a tensor-core FP4
  grouped GEMM (`cuda/wide_expert.h`). Measured (`paper`, D:7220): pp4096
  +45.8 %, pp16384 +67.3 %.
- **No promotions in or after a prompt.** Measured (`paper`, D:8601):
  decode after 128K 66.3 -> 89.5 tok/s.
- **Short prompts:** the CPU takes the thinnest misses (PI:4174-4223).
  Measured (`paper`, D:8039): pp512 x1.57.

## Prefix cache (the 84.6 %)

- **Structure:** a radix tree of content-addressed 64-token blocks; equal
  blocks are one page across requests (`runtime/prefix_cache/prefix_index.h`,
  `program/prefix/*`).
- **Recurrent snapshots,** 110.3 MiB each: GDN, conv, n-gram history, QSA
  tails, the MTP residual (`state_image.h`). Taken at client breakpoints,
  the generation opener, the end of the system / tools block, the prompt
  tail, and a geometric ladder n - 4096 x 2^k; at most 8 a request, plus an
  endpoint snapshot.
- **Tiers:** device blocks in idle KV pages (LRU) and a pinned host slab
  pool of 4 GiB (`runtime/engine/model_instance.cpp:200`). Host eviction:
  dead KV, then superseded snapshots, then GDSF (prefill seconds saved minus
  restore cost, per byte).
- **Restore:** layer by layer on its own stream, overlapping the first call.
- **arcint today:** a lane resumes only its own last prompt (context
  checkpoints). There is no sharing across requests or lanes.

## Decode path

- **MTP:** up to 7 drafts.
  - Per row, K maximises E[tokens(K)] / (1 + 0.38 K) from an EWMA of each
    position's acceptance (PI:3504-3519).
  - A single row verifies only up to the first draft with p < 0.5
    (PI:1751-1755).
  - Two rows share K <= 2; three or more rows decode plain (PI:1692-1694).
  - An n-gram copy proposal (match 12, up to 15 tokens) replaces the drafts
    when it is longer (PI:1721-1728).
  - Measured (`paper`): MTP K=3 +37 %; the confidence cut 138.8 -> 147.2.
- **Formats:** NVFP4 experts, `q8_g32` dense weights, BF16 router / GDN a,b /
  QSA / MTP dense. The `.infernix` container is not GGUF.
- **Launch overhead:** one CUDA graph per batch shape (PI:3039-3076).
- **KV:** INT8-G64 by default, with Hadamard-rotated keys.
- **Elastic KV:** the KV pool takes frames from the expert pool in 4,096-token
  steps (PI:3908-4016). Measured (`paper`): +5.9 % tg512 at a 262K maximum.

## Quality evidence

- **Reference:** Strata's UD-Q4_K_XL, not BF16 (D:2432-2475,
  `model-quality.md` §4). Teacher-forced on 3 texts, 2,557 positions:
  ΔNLL -0.040 ± 0.011 against Strata; KL(Strata ‖ Infernix) 0.023 / 0.052
  / 0.106 by text.
- **Not covered:** no Flash-Next task scores (`eval/` covers Qwen3.6 and the
  27B); no long-context quality.
- **Not comparable** to arcint's KL against unsloth's Q8_0.

## What applies to arcint on the B60 (ranked; all gains `paper`)

**Prefill / TTFT, portable:**
1. **The layer walk.** First measure whether the 422 t/s at 20k is
   link-bound. At half Infernix's link the effect would be larger.
2. **The blind stream**, two layers ahead into slots lent by the cache: a
   second in-order queue, USM copies, per-layer events.
3. **A shared prefix cache** with recurrent snapshots and a host tier. The
   largest effort; it pays for agentic traffic (repeated preambles,
   subagents).
4. **No promotions in or after prefill.** A small policy change.

**Decode, portable:**
1. **The global LFRU** with budgeted admission, landing frames and no credit
   for rejected drafts. It replaces Strata's constants, which arcint follows
   by rule: a reference choice for the operator to price, not to drop.
2. **A GPU-planned CPU miss service** over a mapped doorbell, with the shared
   and resident experts overlapping the CPU wait. A B60 spin needs the LSC
   uncached loads (memory `project-b60-host-flag-needs-lsc-uncached`).
3. **Warm start** from the saved live state. Cheap.
4. **The MTP length policy** and the p < 0.5 cut: per-step draft
   probabilities.

**CUDA-only (the idea at most):** CUDA graphs and PDL, `cp.async` / TMA,
NVFP4 tensor-core GEMMs (the analogue is an XMX kernel for arcint's
quants), VMM arenas.

Related: `research-reference-audit.md` (the rule that a reference's mechanism
is priced before it is dropped), `strata-sycl-b60.md`,
`flash-next-llama-engine.md`.
