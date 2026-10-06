# Patches carried against the pinned OpenVINO

Per `DESIGN.md` §1.1 in the arcint repository, rung 2 of "smallest sufficient
divergence": a numbered patch set **applied at build time here**, not a
divergent checkout. Each patch stays PR-shaped so it can be re-offered
upstream, and this file says what each one does and what it measurably
changes. Every patch's own header carries its full derivation; the dated
measurement record behind each number is on the arcint development branch
(`qfndev`), under the DESIGN section named.

The pin is `2026.4.0-22849-71640275d29` — upstream commit `71640275`. A patch
that does not apply cleanly to that commit is a bug in this directory, not a
reason to move the pin. `build-openvino.sh` resets the checkout hard and
applies `patches/*.patch` in numeric order; the whole series 0003–0067 is
packaged as **`+p20`**. Upstream status: none of these is filed as a PR yet
except where named (0032 is upstream's own fix, backported).

"The 16 GiB card" is the Arc A770 (Xe-HPG), "the 24 GB card" the Arc Pro B60
(Xe2). Unless a line says otherwise a patch is byte-neutral on every path it
does not target, and its correctness cells are in its own header.

## MoE expert offload and the host compute tier (0003–0019, 0037, 0042)

- **0003 — MoE expert-mask subbuffer churn.** At `token_num > 1` the MoE
  implementation rebuilt its per-expert mask subbuffers every inference (20,480
  `create_subbuffer` calls per two-token forward) for a prefill fallback the
  batched-GEMV path never reads. Skipped below the GEMV threshold, created
  lazily for the fallback. Byte-identical; the 35B's two-token verify forward
  27.3 → 18.1 ms on the 24 GB card, which is what lets MoE speculation pay.
- **0004 — OTD perf counters.** `[OTD_PERF]` counters (evictions,
  acquisitions, slot tiers, staging bytes). An instrument; no served change.
- **0005 — device-resident slot pool.** Expert slot buffers charged against a
  per-compile device budget (`MOE_OTD_DEVICE_POOL_BYTES`), so slots that fit
  live in VRAM instead of host memory.
- **0006 — async batched slot uploads.** One call's misses uploaded as one
  batch through a staging ring, waited on once.
- **0007 — drop the redundant per-layer `stream.finish()`** before the
  top-k read on an in-order queue.
  Together, 0005–0007: the 35B on the 16 GiB card from 0.4 t/s (ratio 25,
  unpatched) to 9.1–10.4 t/s at ratio 50 with an 8 GiB pool.
- **0011 — the host CPU compute-tier kernel.** AVX2 and scalar kernels over
  the plugin's grouped-int4 layout, a thread pool, the `MOE_CPU_TIER`
  property. Built at per-source `-O3`: the library builds at `-Os`, under which the
  in-situ task took 2,542 against 270 µs per expert.
- **0012 — the tier's decode split.** A `(token, expert)` pair that would evict
  a slot is computed on the host instead of uploaded; the OpenCL kernels skip
  it by a sentinel; the host result joins before `mlp_reduce`. With 0011: the
  35B at ratio 50 / 8 GiB 15.0–15.5 against 10.4–10.6 t/s, byte-identical on
  the measured prompt, 10/10.
- **0013 — routing histogram** (`MOE_OTD_ROUTING_HIST`), counted before the
  hit/miss split. A diagnostic.
- **0017 — the tier's readback decomposed.** Named readback counters,
  `usm_host` destinations for the hidden state and routing weights (the
  former 283 µs "readback" → 53 µs), the reads hoisted into one wait
  (`MOE_OTD_READBACK_NOHOIST=1` restores the old order). What remains per
  layer is the host waiting for the GPU to reach that layer's router.
- **0018 — the static residency partition.** The tier's LRU residency chose
  device-f16 or host-f32 arithmetic per expert by request history, so a
  continuation restored from the prefix cache could fork from a cold one
  (a DESIGN §3.4 violation). Each layer's resident set is now fixed at
  `bind()` by a `splitmix64(seed, layer_key, expert)` rank, independent of
  history; `MOE_CPU_TIER_STATIC_PARTITION` (on the compiled model, true only
  with the tier on) lets arcint admit the prefix cache with the tier. The
  equivalence suite passes with the tier and the cache, continuation-restore
  included.
- **0019 — the prefill fallback's weight answer is three-way** (no offload
  tier / device slot / host tier), closing a dormant misrouting 0018 left in a
  path arcint does not drive.
- **0037 — hybrid prefill split.** Under the static partition every prefill
  batch holds a non-resident expert, so the grouped GEMM refused every layer.
  Now resident experts run through the grouped GEMM and the rest on the host
  tier afterwards (`hybrid_prefill_layers` counter). The grouped fallback is
  gone (400 → 0) and tier decode improves (18.2 against 12.5 t/s OFF on the
  16 GiB card); the tier's prefill stays at a third of tier OFF's, owned by the
  serial host dispatch.
- **0042 — 0037's gather ran past its tables.** Only resident pairs fill the
  grouped tables, but the gather launched over every pair and a `-1` table
  entry wrapped its `uint` offset ~8 GiB past the buffer — a page fault on the
  24 GB card. The gather runs over the filled count, the tables are
  zero-initialised, a batch with no resident expert takes the per-expert path.
  The 35B with the tier at ratio 99 serves on the 24 GB card at 23.6 t/s.
  Known, not fixed: the filled count is also the oneDNN grouped-primitive
  cache key, so a new prompt can rebuild primitives.

## Paged attention and KV precision (0008–0010, 0014–0016, 0020, 0032–0036)

- **0008 — `VALUE_CACHE_PRECISION`**, an independent value-cache precision
  beside `KV_CACHE_PRECISION`, plus the signed-read fix for the value cache
  (the value read's signedness had followed the declared port type).
- **0009 — the asymmetric-KV kernel plan**: a genuine packing mismatch refused
  at config time instead of miscompiling.
- **0010 — u8-key / i4-value decode and write kernels** (the write path split
  per operand, mirroring the read). With 0008–0009: `u8:i4` KV at 8.8 against
  11.3 KiB/token on the coder, +28% auto-fit context, 10/10.
- **0014 — GPU Assign adopts a same-type, same-rank output layout** instead of
  asserting — the DFlash2 head's state window at exactly its row count. The
  drafter now drafts past 2,048 prompt tokens.
- **0015 — bounded attention partials.** `tmp_out` sized at the output type's
  width (it had been allocated at twice what the kernel addresses);
  `PAGED_ATTENTION_MAX_PARTITIONS` (0 = unbounded, the default, bit-identical to
  the unpatched plugin) bounds the mixed stage's partial buffers with an online
  merge; and a forced argument rebind when an intermediate's identity changes.
  **Upgrade note, still true:** this inserts an option into the GPU
  model-cache blob's positional property list — clear the GPU model cache
  (`--cache-dir`) when upgrading from a level below it.
- **0016 — intermediates sized from the current call**, not the previous
  call's partition count.
- **0020 — `u8:i4` prefill on micro-SDPA.** The value operand's type and
  layout follow the value precision, and the selector admits eight-bit keys
  with four-bit values; the values stay four-bit in VRAM and are unpacked in
  registers. `u8:i4` prefill at `u8`'s rate at a held chunk (459 against 457
  t/s at 37.7k, 401 against 398 at 71.7k, 16 GiB card), the generic path's
  depth-scaled scratch no longer allocated, 10/10.
- **0032 — micro-SDPA's next-K-tile prefetch bound** (upstream PR #37878,
  backported). The pinned nightly prefetched the next K tile with a transposed
  geometry and read up to 256 rows past the buffer for prefill chunks of
  129–255 keys; whether the pages behind were mapped decided between a served
  prompt and an engine reset. Plus a regression test at the served geometry.
- **0033 — micro-SDPA value alignment under `u8:i4`.** The V·S micro-gemm took
  the f16 row's alignment for a 132-byte packed row, so what it read depended
  on the physical pages a request got — the agent's text alternated by request
  parity. One condition; MTP on becomes byte-equal to MTP off. The test harness
  now permutes page tables and uses a fill that tells pages apart.
- **0034 — a micro-SDPA tail test** (output must not depend on what lies past
  the sequence length) and the by-token reproducer (disabled; see 0035).
- **0035 — the by-token test's key fill.** The NaN 0034 recorded for four-bit
  values under by-token keys was the harness's own constant fill overflowing
  f16 in its zero point; with a ramp fill every case passes. Test-only.
- **0036 — Flash-Next's attention geometry** (24 query heads, 2 KV heads, head
  256) through the `u8:i4` micro-SDPA harness. Test-only; 7/7 pass.

## The GGUF K-quant kernel (0021–0031)

- **0021 — `FullyConnectedKQuant`.** GGUF K-quant rows (Q4_K, Q5_K, Q6_K,
  Q8_0) served as stored: an op arcint builds over a u8 constant holding the
  file's rows, decoded in the kernel's inner loop — no unpack at load, no
  second copy. A decode variant (M = 1) and a tiled prefill variant on the
  matrix unit. The first GGUF-opened model scored 10/10.
- **0022 — the decode variant split by architecture**: the one-row matrix
  multiply on Xe-HPG (3× faster there), fused multiply-add on Xe2, the
  work-group sized by the projection's width.
- **0023 — the decode variant in llama.cpp's shape**: lanes along K, one
  super-block per subgroup iteration, sub-group block reads; exact f32
  accumulation. Served decode on the native form 9.9 → 12.1 t/s at 856
  tokens; the timing test now streams and warms (earlier launch figures were
  L2-assisted).
- **0024 — Q6_K tail as one block read** (six messages to three): a gain on
  the 16 GiB card (510 → 385 µs on the down projection), none on the 24 GB
  card.
- **0025 — Q6_K without variable-index shuffles**: the down projection 400 →
  259 µs on the 24 GB card; served decode on the mixed form 13.4 → 15.3 t/s.
- **0026 — Q6_K in 224-byte dword-aligned blocks** (type 114, laid out by
  arcint at load, `--gguf-q6k aligned`, +6.7% bytes on that set): the down
  projection 262 → 204 µs; prefill up a quarter.
- **0027 — the runtime fusion check accepts the K-quant kernel.** Every fused
  residual add had run through the unfused-subgraph fallback, whose output
  read drains the queue (79 `clFinish` per step). Decode step 59.8 → 54.7 ms
  at 856 tokens, byte-identical.
- **0028 — the tiled variant's activation tile in the matrix unit's layout**
  (one block read per operand instead of eight gathers).
- **0029 — 2D block loads on Xe2** for both operands of the tiled variant, and
  a 64-row tile at 256 registers there (Xe-HPG keeps the staged path at 32
  rows). Mixed-form prefill 672 → 907 t/s at 856 tokens. The timing test's
  operands moved to device memory (earlier tiled figures timed the bus).
- **0030 — tall activation reads**: 32 rows per 2D message on Xe2, 16
  subgroups per work-group. Warm 856-token prefill 940 → 1,001 t/s; 71.7k
  451 → 464 t/s, the first GGUF form over the 460 t/s depth bar.
- **0031 — oneDNN's deterministic attribute on the f16-activation compressed
  FC.** A split-K strategy with atomic accumulation made a 235-token prompt
  give five texts in one process; with the attribute, one. No rate change.
  (Correction on the record: the attribute scores the *global* k-parallel
  strategies out; a local split-K keeps its work-group count — the patch
  comment's wording to the contrary is wrong, its effect stands.)

## Per-expert dispatch and the native expert formats (0038–0041, 0043–0074)

- **0038 — the per-expert dispatch framework** (`MOE_PER_EXPERT_DISPATCH`):
  the fused GEMV path is bypassed and only the routed experts are computed.
- **0039 — the per-expert SwiGLU GEMV kernel** with in-kernel u4 dequant.
- **0040 — the per-expert kernel wired into the live path**: resident experts
  on the card, misses on the host tier, the shared expert still fused.
- **0041 — skip expert-constant processing at compile** under per-expert
  dispatch (no mmap faulting of every expert at compile).
- **0043 — the native expert formats through the tier.** The checkpoint's own
  IQ3_XXS / IQ4_XS / IQ4_NL / Q8_0 expert blocks carried in the fused op's
  rank-4 group-32 layout, lowered straight to `MOECompressed` with a
  `weight_format` per projection, with CPU-tier row decoders (llama.cpp's
  tables verbatim). Removed the u4 repack's error on Flash-Next (KLD at depth
  48 0.54 → 0.42 nats on the first native serve).
- **0044 — routing trace** (`MOE_OTD_ROUTING_TRACE`), a per-call trail beside
  0013's aggregate. An instrument.
- **0045 — the native formats' OpenCL decode** inside the per-expert kernels
  (no dequantised row ever written); also the helper-guard fix that let the
  per-expert `.cl` build at all (see the hazards below).
- **0046 — census-seeded static partition** (`MOE_CPU_TIER_SEED`): the
  resident set from a recorded routing census instead of 0018's
  frequency-free rank (measured at chance); malformed or budget-mismatched
  seeds refuse the load. History-independent by construction.
- **0047 — the per-expert slot pool is resident-sized**, not 0041's one-expert
  placeholder (whose second slot was written out of bounds).
- **0048 — the load-time pinned NVMe fill's schedule** (`MOE_OTD_PINNED_NVME_FILL`,
  depth 4, a pinned expert not landed by the barrier is a load failure).
- **0049 — the arcwell transport and the OpenCL slot import**: the pinned
  expert set DMA'd from NVMe into xe VRAM BOs and imported as the slot pool
  (24 GB card only). Cold TTFT 92.5 against 99.7 s host-fed at depth 4.
- **0050 — IQ2_S as a native format** (Qwen3.6-35B's gate/up); IQ4_XS down
  rides the IQ4_NL layout.
- **0051 — the all-resident native pool**: `OFFLOAD_RATIO` 0 with a native
  format enables the offload provider with every expert resident.
- **0052 — IQ2_S-packed**: the checkpoint's own 82-byte block verbatim
  (expert fill −48.6% against the re-laid form at depth 4).
- **0053 — OpenCL load diagnostics** (batch and program build prints,
  `rethrow` backtraces). Prints only.
- **0054 — the packed block's fused-op scale anchor is the f16 `d` Constant.**
  Without it no packed block fused, and the compile constant-folded every
  decode chain (233.5 GiB allocated for a depth-4 compile).
- **0055 — IQ2_S-packed decode walks all eight sub-blocks** (it had decoded
  32 of every 256 values).
- **0056 — per-expert dispatch: aliased zero points and down strides.** For
  IQ4_NL / Q8_0 / IQ2_S-packed the zero point aliases the scale and its raw
  upload overwrote the transposed scale; the Q8_0 down kernel had no slot
  offset. Every per-expert-dispatch reading on these formats taken before this
  patch is void.
- **0057 — native blocks accept compressed value Constants**
  (`--dense-fp16` artifacts fused 0 of 40 layers before, 40 of 40 after).
- **0058 — the all-resident pool is filled at bind** (it had re-read each
  expert from disk on first routing): depth-1 decode 3.0 → 7.3 t/s, load
  245 → 155 s.
- **0059 — per-expert dispatch batched**: one launch per stage for all of a
  call's pairs instead of two per pair. Full-depth 35B on the 16 GiB card:
  prefill 12.5 → 143.9 t/s at 4,096, decode 7.9 → 15.2.
- **0060 — grouped by expert**: a tile of up to eight pairs of one expert
  decodes each weight once (`MOE_DISPATCH_MODE`, auto = grouped at 64+ pairs).
  Prefill → 222.9 t/s, decode → 18.0.
- **0061 — several rows per load**: gate and up at 2 rows over a 4-pair tile,
  down at 4 rows, tile-uniform indices through `sub_group_broadcast`, every
  native kernel spill-free. Prefill → 625.7 t/s.
- **0062 — no speculative hidden-state readback on the all-resident pool**
  (no expert can miss). Prefill → 653.7 t/s.
- **0063 — weight-rounding emulation arm** (`MOE_NATIVE_W_ROUND=f16|bf16`, off
  by default): the measurement that ruled out an f16-rounded matrix-unit form.
- **0064 — IQ2_S-packed gate/up on the matrix unit, exact.** The B operand
  `(2s + 1) · grid · sign` is an integer ≤ 1,333, exact in f16; `d/8` scales
  each 256-value chain once. One route for every call size (tiles of 16 pairs
  for large calls, a K-split one-pair kernel for small ones), so a token's
  bytes do not depend on the call (DESIGN §3.4). Xe-HPG only; down stays
  scalar; `MOE_NATIVE_GU=scalar` restores 0061. Prefill → 952 t/s, 10/10,
  the full equivalence suite passing.
- **0065 — the CPU tier decodes a native expert row once per call** and dots it
  with every job, each job keeping its own order and rounding. Flash-Next
  prefill on the 16 GiB card 1.04 → 3.13 t/s at 512 tokens, same digests.
- **0066 — the CPU tier's native dots one job per AVX2 lane**, compiled
  `fp-contract=off` so each lane is the scalar multiply-then-add (0 `vfmadd` in
  the built routine). Flash-Next prefill → 6.4 t/s at 512 tokens, same digests.
- **0067 — decode routes on the device on the all-resident pool.** A kernel
  writes the decode pair table from the router's ids (calls under 64 pairs),
  so no layer waits on a host readback. The full-depth 35B decodes 19.3 →
  28.1 t/s after 4,096 tokens on the 16 GiB card, same digests, 10/10;
  `MOE_DEVICE_ROUTE=0` restores the host route.
- **0068 — the CPU tier decodes a native expert row once and runs one row per
  AVX2 lane** for a single-job (decode) call.
- **0069 — native IQ3_XXS gate/up and IQ4_NL/IQ4_XS down decode whole blocks**,
  not row by row.
- **0070 — the resident slot pool moves into device memory.**
- **0071 — a decode step's experts are balanced over the pool** with a
  two-phase split (gate/up row chunks, then down column chunks), same bytes.
- **0072 — the tier's expert bytes come from a host RAM bank** filled at load
  by sequential O_DIRECT reads instead of page-faulting on the workers. Flash-Next
  on the B60: first answer 48.55 → 38.18 s, byte-identical.
- **0073 — QSA's own selection reaches PagedAttention** through the optional
  input 28, so Flash-Next's 12 full-attention layers serve the model's sparse
  selection. Non-default, `n_ctx ≤ 32768`; decode 0.87× dense at 20k, prefill
  1.27×. The route gate keeps the below-2,051 mask byte-identical to dense.
- **0074 — the tier's native dot runs in the quantised domain** (ggml's
  approach), routed by call shape: a decode-shaped call (≤ 8 jobs) takes the
  int8 quantised dot, a prefill-shaped call keeps the f32 path (DESIGN §3.4).
  B60 dense `d48q8`, 20,085 tokens: prefill 63.1 vs base 64.2 t/s, decode 6.5
  vs base 5.9.

- **0076 — adaptive expert cache and Strata's RAM budget** (`+p26`): the
  card's expert slots follow the conversation (decayed usage counts,
  non-blocking admission), the host bank's RAM exchange, a fixed bank, and
  the router lookahead. Opt-in: `MOE_CPU_TIER_ADAPTIVE=1`,
  `MOE_CPU_BANK_FIXED=1`, `MOE_CPU_TIER_LOOKAHEAD`. B60 dense `d48q8`,
  20,085 tokens: needle decode 7.5 -> 10.3 t/s, 500-token decode 10.8 ->
  12.4, needle right.
- **0077 — the tier hand-off doorbell** (`+p27`): the decode MoE layers
  route on the device, publish the CPU tier's work to host memory and poll
  for its rows. The enqueueing thread submits the whole decode step ahead.
  Opt-in: `MOE_DOORBELL=1`. Over 0076: needle decode 10.7 -> 12.7 t/s,
  500-token decode 14.0 -> 16.0. On the B60 a kernel sees a host-written
  flag only through an L1/L3-uncached load.

(0075 is not in the series.) The shipped runtime floor stays `+p25`; 0076
and 0077 serve opt-in switches only.

Standing configuration these add up to: the full-depth Qwen3.6-35B native
artifact all-resident on the 16 GiB card at ~960 t/s prefill (4,096 tokens)
and 28.1 t/s decode, max context 112,288 at u8 KV.

## Deliberately NOT applied

These live at the top of the arcint repository's `patches/` and not in this
directory, so the recipe's glob never applies them. Listed so nobody
re-derives the decision by trying them.

- **0001-null-implementation-control.patch** — an instrument, not a fix: it
  forces a null implementation so a node's cost can be measured by removal.
  Shipping it would disable real work.
- **0002-fc-horizontal-fusion-bound.patch** — raises the horizontal FC fusion
  bound. Fusing the MoE block's FC quartet produces wrong output (its fourth
  member is the width-1 `shared_expert_gate`); restricted to the GDN sets the
  gain is inside the noise.

The per-stage timing accumulators a measurement tree may carry
(`network.cpp`, `primitive_inst.cpp`, `stage_acc.hpp`) are part of no patch;
the recipe's hard reset keeps them out of a package.

## Hazards, still true

- **One program per primitive, one copy of the `.cl` per kernel.** The plugin
  concatenates a primitive's kernel sources into one program, so file-scope
  helpers must sit behind a persistent guard, and the guard must be the bare
  `#ifndef`/`#define` pair: the kernel-db generator exempts only that pattern
  from the `#undef` list it appends to each copy. Per-kernel JIT constants that
  helpers read need distinct names in every generator of the program (0045,
  0061, 0067).
- **The executing impl is a clone.** `clone()` copies an explicit field list;
  a new member left out of it is default-constructed in the impl that actually
  runs (0043's native-format members, 0064's `_gu_dpas` — both caught, the
  first after it wedged a card).
- **Equality across kernel forms is by measurement, not by construction**: the
  plugin builds with `-cl-mad-enable`, so contraction is the compiler's choice
  per kernel. Every byte-equality claim above is a measured cell.
- **The version stamp does not identify every build.** Measurement builds of
  0044–0067 before the `+p20` package kept the `marfrit-p19` stamp (which is
  also 0003–0043's); such a build is identified by its symbols (e.g.
  `routing_trace`, `expert_gate_up_native`, `MOE_OTD_PINNED_NVME_FILL`), not by
  the version string. A release names its package level.
- **A measurement tree with the series applied but uncommitted** silently
  loses a patch to a file-level `git checkout` (it once dropped 0005's
  device-slot lock guard). Run `git diff --stat <file>` before and after any
  file-level revert there, and re-apply what is lost.
- **The GPU model cache keys on neither the patch level nor the plugin's
  environment switches.** Clear it when changing levels (0015 changed the
  blob's schema); arcint's paged load compiles its language model with the
  cache off.
- **A new source file must be listed** in the transformations library's
  `sources.cmake` (no glob there): an unlisted source is silently not built
  (0043).
