# 0.4.1 — the GGUF path at the card's speed

Recorded 2026-09-06, the day 0.4.0 stage 1 served (DESIGN §7.0.2ay). The
charter is one sentence: **a GGUF-opened model runs at the rate its bytes
allow on the card**, measured against the same model's Intel int4 IR on
the same card, which is the reference this milestone has to close on.
Nothing here changes what the path serves; 0.4.0's 10/10 through the
GGUF-opened model is the floor every step must keep.

## Where it starts

The benchmark that closed 0.4.0 stage 1 (24 GB card, `u8` KV, one lane,
MTP off, prefix cache off, one fresh process per cell, the chunk the
fit's choice per arm):

| arm | prompt | chunk | prefill | decode (64 tok) | resident |
|---|---|---|---|---|---|
| GGUF Q4_K_M, K-quant kernel (v8) | 856 | 256 | 213.0 t/s | 9.9 t/s | 14.94 GiB |
| GGUF Q4_K_M, K-quant kernel (v8) | 71,727 | 256 | 173.8 t/s | 8.5 t/s | 14.94 GiB |
| Intel int4 IR | 856 | 2048 | 1,609.4 t/s | 23.1 t/s | 13.06 GiB |
| Intel int4 IR | 71,727 | 2048 | 551.7 t/s | 16.5 t/s | 13.06 GiB |

Prefill 3.2× to 7.6× slower, decode about 2×. The kernel ladder that got
there (eight versions on one shape, the gate projection N 17,408 × K
5,120 Q4_K, 50 MB of rows, one launch on the 24 GB card) ends at 225 µs
for one row and 39.5 ms for 2,048 rows. The bytes allow about 110 µs for
one row (50 MB at the card's ~450 GB/s); 2,048 rows on the matrix unit
allow single-digit milliseconds.

## The levers, each named by a measurement

1. **Decode at bandwidth.** The one-row kernel reads rows at 222 GB/s of
   ~450. The compiled kernel has no private memory and no spills (the
   compiler dump's `.zeinfo`); its per-super-block body is ~1,900
   instructions per lane, of which the word-based decoders are most. The
   bound is instruction issue, not the memory system. Candidates, in the
   order to try: decode straight to f16 and multiply in f16 with an f32
   accumulate (halves the converts); the scale/min extraction hoisted per
   super-block instead of per sub-block (it is recomputed eight times);
   two columns per lane so the activation broadcast is shared. Each is one
   build and one run of the plugin's timing test
   (`--gtest_filter=*kquant_perf*`, both cards) before it touches the
   served path.
2. **The prefill tile's A operand.** The 2,048-row launch went from 298
   ms to 39.5 ms when the activation tile was staged in local memory once
   per work-group of eight subgroups; dropping the activation loads
   altogether gives 11 ms, so 28 ms of the 39.5 are still the staging and
   the local-memory reads (64 per sub-block per lane). Xe2 has 2-D block
   loads (`cl_intel_subgroup_2d_block_io`: one instruction reads an 8- to
   32-row × 16-column tile of 16-bit elements straight into the matrix
   unit's A layout, with prefetch variants); the 16 GiB card does not, and
   keeps the staged path. Second candidate on both cards: wider work-groups
   (16 subgroups sharing the stage) and a larger row tile.
3. **The activation reservation that pins the GGUF arm at chunk 256.**
   The fit charges 2.65 GiB of activations at chunk 256 where the IR's
   path charges 0.03 (§7.0.2ay): the kernel's f32 outputs and the fused
   ops around them, sized by the plugin's own memory estimate. Measure
   which node carries it (the reservation's census per node), then either
   an f16 output from the kernel (the reference kernel's own choice for
   f16 activations) or a corrected estimate. At chunk 2048 the prefill
   arm gets the IR's chunk, and the 71.7k prefill rate is the number to
   watch.
4. **The `+p7` package.** The served plugin is a dev-tree build staged on
   the dev host; the recipe at `+p7` (patch 0021) has to be built and
   deployed before any rate in this milestone is a deployment's rate.

Deferred from 0.4.0 stage 1 and carried here as items, not as gates: the
embedding gather from the file (the template's i8 embedding serves; the
file's Q4_K needs a gather kernel, a later plugin patch) and the MTP layer
from the file (the template's serves; `--mtp on` on a GGUF-opened model is
untested). Neither moves a rate in the table above.

## Gate

- **Rates:** on the 24 GB card, same protocol as the table, the GGUF arm
  within 1.5× of the Intel IR at prefill and within 1.2× at decode at
  both depths, one fresh process per cell, no fault line. The 16 GiB card
  measured and recorded at both depths, gated only on "serves, 10/10".
- **Quality:** Prüfstand 10/10 through the GGUF-opened model after every
  kernel change that reaches the served path; the plugin's eleven
  correctness cases on both cards before it.
- **Not a regression:** the IR path's unit set and acceptance cells
  unchanged; the equivalence gates (warm/cold, one/two lanes) on the GGUF
  path measured once at the end.

## Entry criteria

0.4.0 tagged; `docs/design-gguf-native.md` §3.3 and DESIGN §7.0.2ay's
ladder re-read so no rung is re-climbed; the plugin dev tree at patch 0021
with the timing test in place; the Intel int4 IR allowlisted for the
comparison, as it is.

## Pipeline

The 0.3.0 pipeline: recon (a short read, the ladder is the recon), a
design paragraph per lever in the design note, red-first implementation
(the timing test is the red case: it prints the rate before and after),
one card window per lever, the outside review, the DESIGN record. One
lever per window; a lever that does not move the number is recorded and
not carried.

## Size

Three windows and a package build. Lever 1 and lever 3 are host-side or
kernel-local and short; lever 2 is a kernel rewrite on Xe2 with a fallback
to keep on the other card.

## Status

- 2026-09-06 — recorded at the close of 0.4.0 stage 1; nothing started.
- 2026-09-06 — lever 1 taken (DESIGN §7.0.2az, plugin patch 0022, `+p8`
  recipe): the packed matrix-unit decode is 3× on the 16 GiB card and a
  loss on the 24 GB card, where a one-row matrix multiply costs like an
  eight-row one; the split by width lands on both. The 24 GB card's
  decode launch stays at 200–230 µs against a 148 µs skeleton, and the
  served decode rate did not move (10.0 t/s). No lever tried hides the
  arithmetic behind the stream; the gate is not in reach on the native
  path with what is measured. Open decision, the operator's: a repack at
  load into the plugin's own compressed layout (u4 per group of 32 with
  an f16 scale and zero point for Q4_K, u8 for Q5_K/Q6_K/Q8_0), which
  runs Intel's own kernels at the IR's rate but is the unpack at load the
  0.4.0 rule excluded, at 2^-11 relative rounding of the block scales and
  ~10 % more bytes for the 6-bit tensors.
- 2026-09-06 — **lever 2, the repack, taken and served** (DESIGN
  §7.0.2ba, design note §3.6): the K-quant rows repacked at load into
  the runtime's compressed form with no zero point, the mins as columns;
  equivalence a measured bound per weight (0.029 steps at most over the
  served file); 1,005 / 16.1 t/s at 856 tokens (native 213 / 9.9, the
  IR 1,609 / 23.1; 420 / 13.4 at 71.7k at `u8:i4`, the KV that fits),
  Prüfstand 10/10, f16 activations byte-identical to the native output,
  chunk 2048 — lever 3 closed with
  it (the reservation was the f16 zero point's kernels, not the K-quant
  kernel). The gate's decode target (within 1.2× of the IR) is not met:
  the repacked model is 18.73 GiB resident against the IR's 13.06 and
  decodes at the same effective bandwidth. Open: the resident size
  (Q6_K/Q5_K at u8; a mixed open is a flag away), the load time, the
  deferred stage-1 items.
- 2026-09-06 — **the native decode kernel on the integer dot, measured
  and not carried** (DESIGN §7.0.2bb): llama.cpp's q8_1 recipe on Xe,
  correct and 14 % faster in isolation, scores 8/10 on the Prüfstand and
  serves at the same rate; the timing instrument was L2-assisted and now
  streams (patch 0023). Finding that re-aims the milestone: the kernels
  stream at ~300 GB/s, the served decode step is at 154 GB/s effective,
  and the difference is the fixed cost of 401 launches and the narrow
  projections — on both paths. Next lever: that fixed cost, measured per
  node with the profile.
- 2026-09-06 — **the native decode kernel in llama.cpp's shape, served**
  (DESIGN §7.0.2bc, patch 0023 now the kernel): block reads along K,
  the activation block shared by a work-group's rows, Q6_K as dwords
  with a shuffle; exact. Faster than 0022 on every tensor shape of the
  served model on the 24 GB card (gate 156 µs against 170, Q6_K down
  508 against 646); served 12.0 t/s decode at 856 tokens against 9.9,
  8.6 at 71.7k against 8.5, Prüfstand 10/10, byte-identical outputs.
  The §7.0.2bb sentence "the kernel was never what it was waiting on"
  is retracted for the 1k step (a fifth of it was the kernel) and
  stands for the deep step, which did not move: the profile at 71.7k is
  the next measurement. Also open: the 16 GiB card's Q6_K form, the
  first-process stall of a freshly compiled kernel, `+p8` not built.
- 2026-09-06 — **root causes and the survey** (DESIGN §7.0.2bd): the
  instrument was in a second-execution regime (steady state: gate
  355 GB/s, narrow shapes 20–33 µs); the first-process stall is a
  driver-side cold-cache effect, not a kernel property; the profiler at
  depth prefilled in one forward (fixed) and shows the forward 10–11 ms
  faster at both depths, while the served deep step hides it behind
  per-step work outside the graph. Four research agents surveyed other
  projects' Intel kernels; one item measured as a win (a next-super-
  block prefetch for Q6_K on Xe2: down projection 500 → 397 µs) and is
  in patch 0023. Served 12.1 / 10.1 t/s at 856 / 71.7k tokens, 10/10,
  byte-identical. Open: the served loop's per-step host work at depth
  (a host profile), an exact dword-aligned reorder of native Q6_K, the
  LSC cache hints, `+p8` not built.
- 2026-09-06 — **the table worked through** (DESIGN §7.0.2be): the
  served step split on the host (graph 70–92 ms of a 71–96 ms step; the
  template's embedding 1–7 ms, gone; the first step after a prefill 2.3×;
  the first emitted piece at depth a second); the mixed open as the
  default (16.26 GiB, 71.7k at `u8`, 302 / 12.7 and 258 / 10.2 t/s,
  Prüfstand 10/10); lever 3's carrier measured by the mode census (the
  native op's outputs kept, not pooled); the load 285 → 88 s with the
  verdicts kept; `+p8` built and deployed, the IR path 9/9 on the 16 GiB
  card; the 16 GiB card cannot hold the GGUF-opened model; the embedding
  from the file; `--mtp on` served; the equivalence suite's stateful
  sections cannot run on a GGUF-opened model (paged only) and the suite
  needs a variant for it. The decode bar stays unmet: next the
  fused post-op on the K-quant kernel (160 launches and the
  intermediates), the tiled variant's 2D block loads (the mixed form's
  prefill), the first-piece second at depth.
- 2026-09-07 — **the other stacks measured on the same card and bytes**
  (DESIGN §7.0.2bf): llama.cpp Vulkan 126 / 108 t/s prefill and 7.8 / 7.0
  decode at 1k / 10k, SYCL 249 / 206 and 14.2 / 12.4, arcint mixed 309 /
  358 and 13.4 / 11.7, repack 937 / 1,051 and 16.3 / 16.1, the IR 1,598 /
  1,434 and 23.4 / 23.4. Vulkan is the slowest on this card; SYCL's
  decode equals the mixed form's from the same bytes. The operator's
  goal (within 20 % of the IR at the mixed form's size) is not met by
  either GGUF form.
- 2026-09-07 — **the decode step on the device timeline** (DESIGN
  §7.0.2bg): the OpenCL intercept layer on one served process per form
  (`tools/cl_timeline_steps.py`), 24 GB card, 856 tokens, `u8`. No form
  is launch-bound: the mixed default keeps the card busy 62.7 of its
  71.1 ms (IR 37.9 of 43.3, repack 55.3 of 57.6, native 60.3 of 74.5).
  Every fully-connected kernel runs at 86–89 % of the 453 GB/s ceiling
  except the K-quant kernel on Q6_K (188 GB/s, 23.7 ms per step) and
  Q5_K (316). The decode gap against the IR is bytes (17.4 against
  14.0 GB per token, 1.3 GB of it the repack's augmentation), the Q6_K
  kernel (12 ms), and host time per K-quant node (6 ms); the bar sits at
  the floor of the file's bytes. Retracted: §7.0.2be's "30 ms launch
  sequence" and "160 unfused eltwise around gate/up" (fused since patch
  0021). `--dyn-quant on` on the mixed form: prefill unchanged, Prüfstand
  2/10 — dead. The prefill trace found the logits slice not applied to
  a K-quant lm_head (the walk knew only a MatMul): every chunk ran the
  head over all rows and copied 850 MB of logits; fixed red-first, the
  mixed form's prefill at 856 tokens 288 → 413 t/s, byte-identical,
  the activation fit 3,438 → 2,129 KiB per chunk token.
  At 71.7k: prefill 258 → 291 t/s, byte-identical, the served ceiling at
  `u8` 86k → 109k tokens; Prüfstand 10/10. Next: the Q6_K decode rate (the dword-aligned
  reorder), the host cost per K-quant node (plugin-side), the tiled
  variant's register spill and its 2D block loads.
- 2026-09-07 — **the Q6_K decode rate, one 30-minute window** (DESIGN
  §7.0.2bh, patch 0024): the tail of the super-block as one block read
  and broadcasts (six messages → three): exact, byte-identical served,
  10/10, the 16 GiB card's down projection 510 → 385 µs, the 24 GB
  card's unchanged at 400. Measured there: the arithmetic and the
  shuffles ~105 µs each, the three reads alone 330–358, alignment and
  plane layouts null, the prefetch null, 16 × 4 the best dispatch of
  eight; a probe puts the current read shape at 289 GB/s against
  383–418 for one or two wide messages per block, and byte-wise block
  reads need a dword address. The lever is a load-time reorder into
  224-byte blocks read in Q4_K's shape (predicted 183–210 µs, ~12 ms
  per step), designed on the record, not started. 28 of the 30
  minutes used across four card slots.
- 2026-09-07 — **`+p9` built and installed** (patches 0003–0024, 14
  minutes); **the Q6_K row's ISA read and the shuffles removed**
  (DESIGN §7.0.2bi, patch 0025): 118 of 296 instructions per row and
  super-block were word fetches from other lanes; 16-bit block reads at
  the dword below the block put the words in their lanes. Q6_K down
  projection 400 → 259 µs (24 GB card), 385 → 262 (16 GiB card),
  14/14, byte-identical served, Prüfstand 10/10; the mixed form's
  decode 13.4 → 15.3 t/s at 856 tokens and 9.6 → 11.9 at 71.7k (the
  step 70.6 → 61.0 and 88.2 → 79.9 ms). The decode bar (51.8 ms) is
  9 ms away; the 224-byte layout is next.
- 2026-09-07 — **the native Q6_K rows in 224-byte blocks** (DESIGN
  §7.0.2bj, patch 0026, `--gguf-q6k`): the long-K decode 262 → 204 µs
  as the probe predicted, 16/16, byte-identical, 10/10; served, the
  prefill 418 → 531 t/s at 1k and 291 → 335 at 71.7k (the tiled
  variant's loads were paying for the alignment too), the decode step
  60.8 → 59.8 and 79.9 → 77.7 ms — the 33 Q6_K tensors with K = 5,120
  (the lm_head among them) looked unmoved in the timing test (its
  N 1,024 shape is at the launch floor); the served timeline shows them
  at 123 µs against 229 and the lm_head at 2.56 ms against 5.35 —
  the Q6_K set 23.7 → 13.3 ms per step, the device 62.7 → 52.3 of a
  59.8 ms step; the 7.5 ms of host idle around the K-quant nodes is
  the largest item left against the 51.8 ms bar.
- 2026-09-07 — **the host idle named and removed** (DESIGN §7.0.2bk,
  patch 0027): a call log found 81 `clFinish` per served step against
  the IR's 2; a thread-local and a gdb stack traced them to the
  plugin's unfused-subgraph fallback, taken because its runtime fusion
  check did not know the K-quant kernel — the kernel's fused ops had
  never run. Accepted by name, the fused residual add runs in the
  kernel bit-for-bit as before: 21/21 plugin cases on both cards, the
  mixed form's decode step 59.8 → 54.7 ms at 1k (17.2 t/s, Prüfstand
  10/10 at 18.4) and 77.7 → 73.3 at 71.7k, byte-identical. The decode
  bar (51.8 ms) is 3 ms away at 1k.
- 2026-09-07 — **the mins' packing as an option** (DESIGN §7.0.2bl,
  `--gguf-mins`): exact stays the default; `shared` gives 0.66 GiB and
  1 ms back at 10/10 with the 1k output byte-identical; `nibble` gives
  0.94 GiB and 2.2 ms (a 52.5 ms step, 0.7 ms from the bar) at 10/10
  with a different text. The runtime's int4 kernel faulted on an odd
  augmented group count; padded to even.
- 2026-09-07 — **the tiled variant's tile staged in the matrix unit's
  layout** (DESIGN §7.0.2bm, patch 0028): 128 one-element local-memory
  gathers per loop body became 16 block reads; exact, 21/21 both cards,
  byte-identical served; the 2,048-row gate launch 39.5 → 34.6 ms, the
  prefill 551 → 672 t/s at 1k and 341 → 385 at 71.7k (50 % and 84 % of
  the bars). The gathers were a fifth of the launch; the per-tile decode
  and the weights re-read per 32-row tile are the rest — the row tile
  and the GRF mode are the next sweep.
- 2026-09-07 — **the tiled variant's operands by 2D block loads**
  (DESIGN §7.0.2bn, patch 0029): the activation block from global
  memory in the matrix unit's layout, the sixteen weight rows by
  transposed block reads, then 64-row tiles in the 256-register mode
  on Xe2 (Xe-HPG: 32 rows, the mode); exact, byte-identical served,
  10/10; prefill 672 → 907 t/s at 1k (68 % of the bar) and 385 → 451
  at 71.7k (98 %), the Q6_K down launch 11.1 → 4.4 ms at 856 rows on
  the fixed instrument. The packed B operand (2.7× slower: a
  `dpas` chain broken at every sub-block) and hoisted loads (50 %
  slower served) measured and dropped. The timing test's activations
  had been in host memory: every tiled figure before this entry timed
  the bus, and the §7.0.2bm tile sweep is retracted and redone. Reviewed
  (the 2D gate on every input axis); `+p11` (0003–0029) built and
  installed on the dev host. The native form on the same patch meets
  the decode bars (51.3 / 70.3 ms) at 15.22 GiB and 662 / 395 t/s
  prefill; mixed stays the default.
- 2026-09-07 — **tagged 0.4.1** on `+p11`. The gate: prefill 907 t/s at 1k
  (1.77× the IR — missed) and 451 at 71.7k (1.22× — met); decode 54.9 ms
  at 1k (1.27× — missed by the mixed form, met by the native form at
  51.3) and 73.3 at 71.7k (1.21×; native 70.3). Prüfstand 10/10, the
  forms byte-identical. Carried as point releases: 0.4.2 the repacked
  set's gemm, 0.4.3 the tiled kernel's next 30 %, 0.4.4 the Q5_K decode
  rate (the handoff has the measurements each starts from).
