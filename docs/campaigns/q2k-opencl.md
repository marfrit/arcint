# q2k-opencl — Q2_K weights on the OpenCL backend, decode to prompt

**Open (2026-11-09).** Patch `0117` of `contrib/llama.cpp` (on top of 0116) is written, built and
unit-tested on the recipe host (no device there). Its card runs are the 1.6.2 release gate's (2026-11-12, `measured-here`, RC 9c0c730e): the op suite 193 of 193 on both cards, the red build 0 of 193 on both (the B60 in the gate window, the A770 the evening before), the break build declined on the card with its `not-built` keys, mean KL on Llama 2 7B Q2_K 0.18746-0.18749 against the CPU's 0.18828 on the same file, llama-bench 1952.29 / 77.95 (A770) and 3491.61 / 98.54 t/s (B60, pp512 / tg128). The 1-16 column curve against Q4_K is not measured and no Q2_K model was run through the server. The operator's decision of
2026-11-09: Q2_K too, in 0116's structure and with its gates.

## Charter

Dense `Q2_K` weights on the card at every column count, as `q3k-opencl.md` did for Q3_K: a matvec for
decode, the flat few-column product for an MTP verify, the Xe2 2D-block GEMM for prompts with the tile
GEMM as fallback, `GET_ROWS`, DOWNGRADE keys, the view rule. `MUL_MAT_ID` stays on the CPU with its
key. Before (`code`): `supports_op` declined Q2_K `MUL_MAT` (no type branch took it), so a Q2_K tensor ran
on the CPU.

## Reference to follow

Strata has no Q2_K kernel (`code`, `include/strata/kernels/native_mmvq.hpp`: its MMVQ list has Q2_0, Q3_K,
Q4_K, Q5_K, Q6_K, Q4_0, IQ4_XS and others, no Q2_K). So the forms are arcint's own, each the one this
backend already uses for a K-quant with a min (Q4_K) or with per-16 scales (Q6_K, Q3_K). The block math is
ggml's `dequantize_row_q2_K` (`ggml-quants.c`): 84 bytes = `scales[16] | qs[64] | d | dmin`, weight
`d (s & 15) q - dmin (s >> 4)` with the 2-bit code `q` and the scale byte `s` of its 16 weights.

## Design (0117)

- Planes per slice (`ggml-opencl-planes.h`): `dm[nblk]` (d and dmin, one half pair) | `scales[16 nblk]` |
  `qs[64 nblk]`; an even block count a slice, as for every planar type.
- 1 to 4 columns: `mv_q2_K` (`mul_mv_iq_q8_1.cl`), Q3_K's lane map, unsigned x signed `dp4a` on the codes,
  the min term on the sum of the lane's activation codes (both int8 terms), as ggml's `vec_dot_q2_K_q8_K`
  uses its q8_K block sums.
- 2 to 16 columns: `kernel_mul_mm_few_q2_K_f16` (`mul_mm_kq_few.cl`), 0009's fp16 DPAS product: codes as
  halves `1024 + q`, the chain from `-1024 S`, then Q4_K's `xsc (D sc t - M m S)` per k16 step.
- Prompts: `kernel_mul_mm_kq2d_q2_K` (`mul_mm_kq_2d.cl`) on Xe2, Q4_K's `(p - 1024) dl + nml` decode on
  the qs plane's block reads (its start 64-byte aligned: a slice's block count a multiple of 16); else
  `kernel_mul_mm_q2_K_f16`, the tile GEMM (`q2k_v32` in `lb_v32`), keyed `path=tile reason=2d-shape`.
- `kernel_get_rows_q2_K`; the view rule, the odd-block key; `GGML_OPENCL_Q2K_OPTS` (`-DQ2K_RED`: the scale
  byte's nibbles swapped; `-DQ2K_BREAK_BUILD`); 0116's build fallback widened to drop Q2_K's text alone.

## Gate

As `q3k-opencl.md`: every Q2_K `MUL_MAT` and `GET_ROWS` case green on both cards (or declined by design),
the full `MUL_MAT`, `MUL_MAT_ID` and `GET_ROWS` suites to completion; `-DQ2K_RED` red; the
`-DQ2K_BREAK_BUILD` fallback costing Q2_K alone; the 1-16 column curve and the AREX-2 roles' shapes at 1,
2, 6, 8, 16 and 512 columns next to Q4_K (1 column at least Q4_K's speed per byte: 2.625 against 4.5 bits
a weight, 45 us at Q4_K's 427 GB/s on 4096 x 14336, `inferred`).

## Current state

- Built and unit-tested on the recipe host (the RECIPE line in the patch README's section).
- `tools/q2k_layout_emul.py` (`measured-here`, an emulation of the kernels' arithmetic in Python): the
  tile GEMM's and `GET_ROWS`' decode and the 2D GEMM's operands equal `dequantize_row_q2_K` on every
  weight of 24 random blocks; the matvec's lane map with its min term and the few-column product give the
  reference dot to rounding (1e-15, 3e-13); `--red` breaks all four.
- `tests/test_planes.cpp`: the Q2_K planes against `ggml-common.h`'s `block_q2_K`, a range across slice
  boundaries; the red R5 (Q2_K's dm and scales fields swapped) fails it.
- The card window: the 1.6.2 release gate (above; the record is `contrib/llama.cpp/README.md`, patch 0117). Not measured: the per-column curve, a served Q2_K model.

## Where it lives

`contrib/llama.cpp/patches/0117-opencl-q2_K.patch`, its section in `contrib/llama.cpp/README.md`; the
switches in `docs/switches.md`.
