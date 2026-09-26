# The native per-expert kernels on the matrix unit (2026-09-26)

Campaign: `docs/campaigns/sub4bit-vram-kernel.md` (the OpenCL decode is its
rate lever; its gate includes the Prüfstand) and `docs/window-054.md` (LYON
row 3c, prefill at depth). Predecessors: plugin patches 0059–0062 (DESIGN
§7.0.2cj–§7.0.2cn). Evidence classes: `code`, `paper`, `measured-here`. Every
card number names the card: the A770 is `GPU.1`, PCI `8086:56a0`, Xe-HPG.

## 1. Where the prefill is

`measured-here`, A770, the full-depth packed u8 Qwen3.6-35B-A3B, all-resident,
u8 KV, chunk 1024, plugin 0062: prefill **653.7 t/s** at 4096. An OpenCL
timeline of that request (window 6.48 s):

| term | share |
|---|---|
| grouped gate/up (IQ2_S-packed, 40 layers) | 48.8 % |
| grouped down (IQ3_XXS 37 layers, IQ4_NL 3) | 28.7 % |
| GDN core | 6.0 % |
| dense GEMMs | 2.8 % |
| attention (sdpa_micro) | 1.6 % |

Gate/up does 34.4 GFLOP per 1,024-token launch (arithmetic from the
geometry) in 19.8 ms: about 1.7 TFLOPS, on the vector units (`code`:
0059–0061). The matrix unit on the same card runs the K-quant tiled kernel's
gate projection (17,408 x 5,120 at 856 rows, 152.6 GFLOP) in 8.0 ms, about
19 TFLOPS effective (DESIGN §7.0.2bn, v37, `measured-here`). That is a
lower bound on what the unit gives a decode-in-the-loop kernel on this card,
not its roof.

## 2. Prior art in this repository

`fully_connected_gpu_kquant.cl` (patches 0021–0028; DESIGN §7.0.2az,
§7.0.2bm, §7.0.2bn, §7.0.2bo) decodes GGUF K-quants in the loop and feeds
`intel_sub_group_f16_f16_matrix_mad_k16`
(`cl_intel_subgroup_matrix_multiply_accumulate`; both cards list it —
`clinfo`, `measured-here`). Its record constrains this design:

1. **Decode to f16 and chain the multiplies** into the same accumulators.
   The form with a per-32 scale **and offset** applied after each multiply
   (every `dpas` from zero, Σx staged) was built and withdrawn (§7.0.2bn,
   `measured-here`).
2. **Stage the activation tile in local memory on the A770**, in the matrix
   unit's A layout (0028). The reason is that Xe-HPG has no 2D block loads
   (§7.0.2bn). On Xe2 the global/2D path won once the instrument was fixed.
3. **Compile-time loop bounds and constant private indices.** A run-time
   bound sent the accumulators to scratch, a 10x loss (`measured-here`).
   Rows and columns past the end are clamped and dropped.
4. **The A770: SIMD 8** for the matrix path (`code`: patch 0021's host,
   `(kernel_number == 0 || arch >= xe2) ? kSimd : 8`). dpas is M8 x N8 x K16:
   A is `int8` (one sub-group block read of local memory in 0028's layout),
   B is `int8` (the lane's column, 16 k packed in pairs), the accumulator is
   `float8`, one register per row group row.
5. **32 rows at 256 registers** on the A770 (v37: the 256-GRF mode took the
   gate 10.3 -> 8.0 ms; 64 rows won two of three shapes and lost the third,
   §7.0.2bn). That mode is 4 threads per EU. It cost the batched decode kernel
   15–45 % (§7.0.2bn), so the tiled K-quant kernel is built in its own batch.
6. **Measured losses, not to repeat**: prefetching the next super-block
   (§7.0.2az); hoisting the A loads ahead of the decode (v33, served prefill
   672 -> 437 t/s, §7.0.2bn).
7. **The one-row matrix multiply was a win on Xe-HPG** (decode, M = 1: 509 ->
   170 µs, §7.0.2az) and a loss on Xe2.

## 3. The route

**One route for every call size on the A770.** Today `auto` switches between
the batched kernel (under 64 pairs) and the grouped one. That was harmless
while both gave the same bytes (0060–0062, measured). A matrix-unit grouped
kernel and a vector-unit batched kernel would not. A token's bytes would then
depend on the size of the call it landed in: a 1,030-token cold run does
tokens 1024–1029 in a 6-token chunk, and a warm hit at a block boundary
refills them in a longer suffix. That breaks the invariant that any cache
state is byte-identical to a cold run (DESIGN §3, "tested in CI"). So every
native per-expert call goes through the matrix kernel, decode included, by
§2.7. The scalar kernels stay selectable as the comparand
(`MOE_DISPATCH_MODE=pair|batched|grouped`); the new route is a new value
(`dpas`) and becomes `auto`'s only answer once the gates pass.

## 4. The mapping

- **Tile**: up to TM pairs of one expert slot, with its own constant
  (`NATIVE_DPAS_TM`) and its own table (stride 2 + 2·TM). `kNativeTileM`
  sizes the scalar kernels' private arrays and must not grow (0061: 8 x 2
  spilled, 313 ms). At chunk 1024 the mean is 1024·8/256 = 32 pairs per
  expert, equal to TM = 32. Under uniform routing a 32-tile fills 69 %, a
  16-tile 82 % (arithmetic; the served skew is unmeasured). So empty row
  groups are the common case: skip them uniformly (`g*8 < cnt`, the indices
  stay constants). TM is swept 16/32 on the block bench.
- **Work-group**: one tile and a range of columns. SIMD 8, `reqd_sub_group_size(8)`,
  its own entry and local size. None of the scalar decoders carry over: they
  split K across the lanes of one row and reduce. The matrix decoders are
  lane-per-column.
- **A operand**: the tile's activation rows, gathered by pair — hidden row
  `flat / top_k` for gate/up, intermediate row `flat` for down. They are
  staged per 256-wide K super-block, as 0021 does with `stage_src[p]` (only
  the row index changes, and each row's 512 B per super-block stays
  contiguous). A token routes to distinct experts, so no row repeats in a
  tile. Local memory: TM x 256 x 2 B = 16 KiB at TM = 32, the configuration
  0021 ran on this card (§7.0.2bn).
- **B operand**: the lane's column, 16 consecutive k per `dpas`. IQ2_S-packed
  rows are 8 x 80 B per 2,048 k (640 B). The plan is five 16-B loads per
  super-block, and the grid read as one 8-byte constant load per entry
  instead of 0061's per-byte `NATIVE_IQ2S_GRID[gi*8+m]`. IQ3_XXS: four
  4-value grid entries, the sign bytes, one f16 scale per 32. IQ4_NL: 16
  table lookups, one f16 scale per 32.
- **gate/up with SwiGLU**: both accumulators for the same column in one
  subgroup. Register budget at TM = 32 (`paper`, one GRF per f32 row at
  SIMD 8): accumulators 64 + B 32 + A 16 = 112 before decode temporaries.
  That spills at 128 registers. The options are measured on the block bench:
  the 256-GRF mode in its own batch (§2.5); TM = 16; or gate on lanes 0–3 and
  up on lanes 4–7 with a shuffle in the epilogue.
- **down**: K = 512, N = 2048; the routing weight in the epilogue; 0059's
  scatter.
- **The tile table** sits in `usm_host` (0060) and every work-group reads it
  over the A770's x4 link. Its cost is measured on the block bench; the table
  moves to the device if it shows.

## 5. What changes in the numbers

Two B-operand forms for IQ2_S-packed, both built as arms:

- **(a) Rounded**: the decoded weight `d · (2s+1)/8 · grid · sign`, computed
  in f32 and rounded to f16. `code`: d carries 11 bits, (2s+1) up to 5 and the
  grid up to 6 (43). The product rounds by at most 2^-11 relative, for
  **normal d only**. A subnormal d breaks the bound, and the record has met
  one: a scale in f16's subnormal range, design-fit-levers §4 (the dense u8 lever). The
  emulation arm as built (patch 0063) does not count subnormal weights; form
  (b), chosen in §6.1a, carries d outside the operand and has no such case.
- **(b) Exact**: B = f16((2s+1)·grid·sign), with |v| ≤ 31·43 = 1,333 < 2,048,
  so it is exact in f16. d/8 is applied once per 256-k super-block, after a
  16-deep dpas chain, as one fma per accumulator register. This is not
  §2.1's withdrawn form: there is no offset, no Σx, and one drain per 16
  `dpas`. It costs a second accumulator set, which the register budget of §4
  may not afford at TM = 32.

IQ3_XXS and IQ4_NL already store a per-32 f16 scale (`code`: the 0045 fill
precomputes d·(0.5+s)·0.5). Their products s·grid (≤ 16 bits) and s·kvalue
(≤ 18 bits) round to f16 at 2^-11 relative for normal s. There is no exact
fold without a fill-format change.

The matrix unit's internal accumulation is `paper`-class only: it is not
known to be f32 round-to-nearest per step. Whether a row's result is
independent of its tile-mates is therefore measured, not assumed (gate 5).
In form (a) the output bytes will not equal 0062's; in form (b) not either
(different summation). This is the first route change since 0058 that a
byte cell cannot gate.

## 6. Gates

1. **Block numerics, failable.** The DESIGN §7.0.2ay precedent: the tiled
   K-quant variant's tolerance was widened by 2^-11 of Σ|x·w| for its f16
   weight copies.
   - Read the projection sums in f32 (a debug tap or an f32 build of the
     block).
   - On the host, build two f64 models from the decoded weights: exact W, and
     W16 = RNE-f16(W). Let S = Σ|x·w| per element.
   - Assert per element |y_new − y_W16| ≤ 2 · r · S, where r = max over
     elements of |y_scalar − y_W| / S, pre-registered from the scalar run.
     Form (b) is checked against y_W.
   - Reds:
     - form (a) against y_W at the same bound must fail, so the instrument
       sees the rounding;
     - mutants that truncate to 10 bits or round to bf16 must fail;
     - a wrong grid row, a swapped dlo/dhi, and an A gather off by one pair
       must fail.
   - Geometry: the cell's T 1/6/17 never fill a 32-tile (4 experts), so add
     T ≈ 80, with a slot at exactly 32 and 33 pairs.
2. **Served logits** (`ARCINT_LOGITS_DUMP` + `tools/logits_dump_diff.py`,
   `--no-logits-slice`).
   - Add an emulation arm first: 0062's scalar decoders with each weight
     rounded to f16 (one `convert_half`, env-switched). KL_emu = KL(0062 ‖ emu)
     is the served cost of the rounding alone.
   - Gate: KL(new ‖ emu) ≤ KL_emu and KL(new ‖ 0062) ≤ 2·KL_emu, at depth 4 and
     full depth. A bf16-rounding emulation arm must separate from the f16 one
     (the instrument's red).
   - There is no copied bar: the dense-u8 form's 1.95e-4 and 983/1000
     (design-fit-levers §4.3) were outcomes, not a gate.
3. **Prüfstand 10/10**, the campaign's own gate. §7.0.2bb is the precedent:
   a digest-identical numerics change scored 8/10.
4. **Equivalence.** The equivalence suite: warm against cold, a warm suffix
   under 8 tokens, two lanes against one. Plus two cold runs byte-identical
   (the A770 is bit-readable).
5. **Position independence.** A cell asserting that a pair's bytes do not
   change with its row position or its tile-mates: permuted pair order, and
   tiles padded to 1 / 31 / 32 / 33 pairs.
6. **Rate.** Full-depth prefill at 4096 above 0062's 653.7 t/s. Full-depth
   decode not below 0062's 21.1 / 19.9 (the one-row matrix path, §2.7).

### 6.2a Gate 2, measured before any kernel (2026-09-26)

`measured-here`, A770. Patch 0063's emulation arm, the packed u8
Qwen3.6-35B-A3B, all-resident, `--no-logits-slice` (chunk 512 at full depth,
1024 at depth 4), a 4096-token prompt. The instrument is
`tools/logits_dump_diff.py`. The rows compared are the request's prefill
records; the load probe's and the post-divergence decode records are excluded.

| arms | depth | KV | per-record mean KL (request) | argmax agree (request) |
|---|---|---|---|---|
| unrounded, repeated | 4 | u8 | 0 (dumps byte-identical) | all |
| unrounded vs f16-rounded | 4 | u8 | 0.94–1.24e-4 | 1015–1023 / 1024 |
| unrounded vs bf16-rounded | 4 | u8 | 0.71–1.00e-4 | 1018–1020 / 1024 |
| unrounded vs f16-rounded | 4 | f16 | 0.81–1.27e-4 | 1014–1021 / 1024 |
| unrounded vs bf16-rounded | 4 | f16 | 0.51–1.04e-4 | 1015–1020 / 1024 |
| unrounded vs f16-rounded | 40 | u8 | 0.042–0.241 | 465–495 / 512 |
| unrounded vs bf16-rounded | 40 | u8 | 0.014–0.180 | 467–496 / 512 |

Greedy digest at 4096 tokens: depth 4 `8cccdbac48ed` in all three arms;
depth 40 `b1a16fbc9d4c` unrounded, `29e9267e1d2d` for both f16 and bf16.

What this settles:
- **The instrument's red fails on prefill rows.** bf16 is 8× coarser than
  f16 but reads no farther on the request's records, at either depth. The
  tool's all-record summary (bf16 worst 0.43 at depth 40) includes load-probe
  and post-divergence records, and is not used. Only the single-token decode
  records order (depth 4, u8 KV: f16 3.6–9.1e-6, bf16 8.9–23.7e-6). Two
  perturbation sizes were tried, no smaller one.
- **The floor is deterministic** (the repeat is byte-identical).
- **The floor is not the u8 KV cache**: f16 KV reads the same.
- **The floor is present inside one forward** (the request's first chunk,
  past = 0, reads 1.2e-4).
- The reading consistent with all of it, NOT measured as a mechanism: once a
  perturbation exceeds the f16 activation rounding, the downstream f16
  roundings decorrelate. The served distance is then the path's own f16
  activation noise, about 1e-4 per prefill row at depth 4 and 0.01–0.24 at
  depth 40 (both arms), and not the size of the cause.

So gate 2 cannot measure the rounding. Revised:

- The served A/B is a **gross-error bound**, at twice the emulation arms'
  own spread (both arms, per request record):
  - depth 4: mean KL ≤ 2.6e-4 and argmax ≥ 1004/1024 per record;
  - depth 40: mean KL ≤ 0.49 and argmax ≥ 418/512 per record.
- **Its red is a gross mutant**, not bf16: a wrong grid row in the new
  kernel must exceed the bound. That red is OWED: it runs with the kernel,
  and until then the gate's failability is asserted, not shown.
- The numerics of the rounding itself are **gate 1's** (block-level, against
  the f64 host models) and the **Prüfstand's** (gate 3).
- That depth 40 moves the greedy digest under any ulp-level change is now
  known. A digest change is expected and is not, by itself, a failure.

### 6.1a Gate 1's instrument, and the forms, measured (2026-09-26)

`tools/native_kernel_harness.py` runs the plugin's own captured program (via
`tools/cldump.c`) outside the plugin. It calls the served IQ2_S-packed row
decoder with f32 sums out, next to a matrix-unit candidate. Both are checked
against the f64 host models of §6.1, built from gguf-py's IQ2_S
dequantisation of the same bytes. `measured-here`, A770, 512 rows, gate and
up; the errors are max over elements of |y − y_ref| / S:

| kernel (pairs) | ms (median of 30) | vs exact W | vs W16 | spill |
|---|---|---|---|---|
| served scalar, 0063 unset (32) | — | 4.96e-8 | 4.37e-5 | — |
| served scalar, `NATIVE_W_ROUND` f16 (32) | — | 4.37e-5 | 4.6e-8 | — |
| served scalar, bf16 (32) | — | 3.43e-4 | 3.49e-4 | — |
| served scalar decoder, harness geometry (256) | 0.715–0.72 | 4.96e-8 | 4.37e-5 | — |
| matrix, form (a), TM 16 (256) | 0.236 | 4.37e-5 | 3.96e-7 | 0 |
| matrix, form (a), TM 32 (256) | 0.282 | 4.37e-5 | 3.96e-7 | 416 B |
| matrix, form (a), TM 64 (256) | 0.516 | 4.37e-5 | 3.96e-7 | 4,160 B |
| matrix, form (b), half2 grid table, TM 16 (256) | 0.293 | **4.48e-8** | 4.37e-5 | 0 |
| matrix, form (b), half2 grid table, TM 32 (256) | 0.291 | 4.48e-8 | 4.37e-5 | 2,496 B |

What it settles:
- **The instrument sees the rounding** by three orders of magnitude. Block
  level can gate what the served logits could not (§6.2a).
- **Form (b) meets the pre-registered bound** (≤ 2 · 4.96e-8 against exact
  W). Form (a) fails it, and against W16 it reads 3.96e-7, 8× the scalar's
  4.96e-8 against W (the cause is not separated: the prototype's
  accumulation or its rounding).
- **Form (a)'s error against W16 is the same at every TM** (form (a)'s
  error against W is the weight rounding and carries no tile-mate signal).
  That points to rows being independent of their tile-mates; gate 5
  measures it in the plugin.
- **Plan change.** Gate/up (IQ2_S-packed, 48.8 % of the window) moves to the
  matrix unit in form (b), TM 16, for every call size.
- **Down (IQ3_XXS/IQ4_NL)** has no exact fold. Its per-32 scale would bring
  back the per-scale drain §2.1 withdrew. So down stays on 0061's scalar row
  kernel, also for every call size. Each projection's kernel is then
  independent of the call size, which is what §3 requires.
- The block's pairs here share one expert: its weights stay in cache, and
  the served tile reads a different expert each time. `code`: that is 656 KiB
  of gate/up bytes per expert, 168 MB per 256-expert layer-chunk; at a
  bandwidth of about 400 GB/s (`paper`) that is about 0.4 ms. The speed-ups
  in this table are harness against harness. The served scalar decoder runs
  here in the harness's own launch geometry, and its rate (1.5 TFLOPS)
  matches the served kernel's 1.74.

### 6.3 Outcome (2026-09-26)

`measured-here`, A770, patch 0064, full depth unless named:

| gate | result |
|---|---|
| 1 block numerics | form (b): 4.48e-8 of S (bound ≤ 9.9e-8); form (a) red at 4.37e-5 |
| 2 served logits | depth 4: KL 4.0–7.8e-5, argmax ≥ 1015/1024; depth 40: KL 0.016–0.206, argmax ≥ 474/512; the gather mutant 0.47–0.72, 233–294/1024 (red) |
| 3 Prüfstand | 10/10 (0062 in the same window: 10/10; the greedy answers differ, 663 and 529 tokens) |
| 4 equivalence | all checks passed at full depth: warm = cold (hit 81.7 %), a restored continuation = cold, greedy repeat identical, speculative decoding deterministic; chunked vs unchunked differs (reported, not gated); stateful and MTP sections skipped |
| 5 position independence | token 0 identical at T = 1 / 17 / 40, batched and grouped, at hidden 512 and 2048; the tiled gather mutant red at T = 40; a one-pair block-order mutant red at hidden 2048 (green at 512) |
| 6 rate | prefill 952.1 t/s @4096 (0062: 653.7); decode within the spread of four interleaved runs each, the means 2.8 % / 0.9 % below |

Changes against §4, all measured:
- **TM 16**: at 32 the gate/up accumulators spill.
- **8 subgroups** per work-group for prefill.
- **The one-pair decode kernel with K split across subgroups.** Served
  decode read 17.5 / 15.0 t/s with the tiled kernel on single-pair tiles,
  and 19.5 / 18.2 with a one-pair kernel without the split, against 0062's
  21.1 / 19.9 in the same session. With the split it reads within the spread
  (the table).
- **Byte identity.** The prototype's three forms (tiled, one-pair, one-pair
  K-split) are bitwise identical at f32 in the harness. The served two
  (tiled, K-split one-pair) are byte-identical at their f16 output in the
  cell at hidden 512 and 2048. A block-order mutant in the served one-pair
  kernel is red at 2048.
- The first build did not run the kernel: the impl's `clone()` copies only
  listed fields (0012's lesson, again).
- Provenance: the harness figures (4.48e-8, the byte identity of the
  one-pair forms) are the prototype's (`--dpas` file), whose kernels are the
  patch's by construction. The served kernel is exercised by the lowering and
  call-independence cells.

## 7. Pipeline

Recon (this note) → review (done, applied) → the emulation arm and gate 1's
instrument, red-first → the IQ2_S-packed gate/up kernel, forms (a) and (b),
TM 16/32, with the register options, on the block bench
(`tools/native_moe_block_ab.cpp`, `ARCINT_BLOCK_AB_REPEAT`) → IQ3_XXS/IQ4_NL
down → decode through the same route → one card window with gates 1–6 →
review → DESIGN record, CHANGELOG, patch README.

## 8. Not in scope

- An int8 matrix path: the native codes are not affine ints, and the int8
  roof needs activation quantisation, a second numerics change.
- The GDN core (6 %).
- The B60: Xe2's 16-lane dpas, its 2D block loads and its 256-register mode
  are a second variant, after the A770 gates.
