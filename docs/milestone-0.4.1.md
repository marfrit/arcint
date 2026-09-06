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
