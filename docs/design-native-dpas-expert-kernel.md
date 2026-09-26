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
  one: a scale in f16's subnormal range, design-fit-levers §4 (the dense u8 lever). The arm
  asserts or counts subnormal d at fill.
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
