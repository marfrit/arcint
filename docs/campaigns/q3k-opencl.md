# q3k-opencl — Q3_K weights on the OpenCL backend, decode to prompt

**Open (2026-11-09).** Patch `0116` of `contrib/llama.cpp` is written, its
tree built and unit-tested on the recipe host (no device there). The card
windows 186 and 189 have run (op suites on both cards green, the red case red,
the 1-16 column curve against Q4_K on the B60; see Current state); owed: the
prompt curve of the Xe2 2D GEMM (window 190 on a05b036e: only its dry runs are
recorded) and the served check.

## Charter

Give dense `Q3_K` weights a device path at every column count, so that a
Q3_K tensor no longer leaves the card: a matvec for decode, a flat
few-column product for an MTP verify (like Q4_K / Q5_K / Q6_K), a GEMM for
prompts, `GET_ROWS` for a Q3_K embedding, and a DOWNGRADE key wherever a
Q3_K product still falls to a slow path. Target: an AREX-2 27B mix the GA may
put in Q3_K (its FFN is 64 % of the weight bytes: gate / up 17,408 x 5,120,
down 5,120 x 17,408; then linear attention's qkv and gate, ssm_out, full
attention's q, k, v and output, the head 248,320 x 5,120). Out of scope:
`MUL_MAT_ID` (MoE experts), which stays on the CPU with its key; a follow-up.

State before (`measured-here`, B60, `test-backend-ops perf -o MUL_MAT`,
4096 x 14336): Q3_K "not supported" at every width (1-512 columns), so a
Q3_K tensor runs on the CPU in a served graph. Upstream's OpenCL backend has
no Q3_K kernel either (`code`, the tree of 2026-10-20). For comparison, us a
product: Q4_K 77.4 at 1 column, 83.7-85.6 at 2-8, 133.7-136.0 at 9-16; Q5_K
95.5 at 1, ~110 at 2-8; Q6_K 124.3 at 1, ~171 at 2-8.

## Reference to follow

Strata (`~/src/Strata-ref`, `4b9e05a2`), `code`:
- `include/strata/kernels/native_mmvq.hpp:63-66`, `native_q3_k_mmvq` and
  `native_q3_k_f32`: the activation quantized to q8_1 once, then an
  integer-dot MMVQ; `sycl/src/kernels/cuda/native_mmvq.dp.cpp:436-534`
  (`load_int_b2`, `q3_q8_dot_impl`, `q3_q8_dot`, `native_q3_k_mmvq_kernel`):
  llama.cpp CUDA's `vec_dot_q3_K_q8_1` form, 16 threads a block, a thread one
  qs word and one hmask word against 4 q8_1 blocks, `dp4a` after a per-byte
  saturating subtract of the hmask's 4; AoS blocks read as 16-bit pairs (the
  110-byte stride leaves every other block 2-byte aligned); one q8_1 term
  with an f16 scale; `native_mmvq.dp.cpp:3570-3628` takes 1 column there and
  `launch_multi` (`:2245`) 2 to 8 columns, `NCOLS` a template parameter.
- `sycl/src/kernels/cuda/dequant_bf16.dp.cpp:109-131`: Q3_K dequantized to
  bf16 for the wide products, then a GEMM.
- The block math: ggml's `dequantize_row_q3_K` (`ggml-quants.c`): 110 bytes =
  `hmask[32] | qs[64] | scales[12] | d`, weight `d (sc - 32) (q2 - 4 [hmask
  bit clear])`, the 16 6-bit scales packed in 12 bytes.

How arcint's build differs (`code`), per the reference rules:

| | Strata | arcint 0116 |
|---|---|---|
| storage | GGUF blocks, 16-bit loads | planes per weight slice (d, hmask, qs, scales; 0022's `ggml_cl_lb_fields`), every field an aligned 32-bit word; an odd block count stays in blocks and goes to the CPU with a key |
| 1 column | dp4a MMVQ, one q8_1 term, 16 lanes a block | dp4a matvec, two int8 terms per 32 (`kernel_quantize_q8_1`, arcint's activation since 0001: a single-term q8_1 matvec moved the dense 27B's KL by 0.0026-0.0028 nats, `llama-engine-kquant-kernels.md`), Q6_K's lane map: 8 lanes a block, a lane 8 qs and 8 hmask bytes for 32 weights; `-DQ3K_ONE_TERM` is Strata's single term, an arm |
| 2-8 columns | the same MMVQ, `NCOLS` 2..8 | the few-column fp16 DPAS product of 0009 (Q6_K's form: a lane is a weight row, its codes the b operand as halves 1024 + u, the tokens the a operand), 2-16 on Xe2, 4-16 on Xe-HPG (the matvec keeps 2-3 there, as for the K-quants); `GGML_OPENCL_KQ_MV_NCOLS=5..8` routes 2 to that many columns to the matvec built up to it instead (Strata's form, the arm) |
| 9-16 columns | the bf16 GEMM | the same few-column product, two token groups |
| prompts | bf16 dequant, then a GEMM | Xe2: the 2D-block GEMM of Q4_K-Q6_K (`mul_mm_kq_2d.cl`) on Q3_K's planes, fp16 DPAS; Xe-HPG and unaligned planes: the dequantizing tile GEMM (`mul_mm_kq_f16.cl`, `deq_lb`). Not the int8 GEMMs of 0013 / 0102: Q3_K's scales are per 16, the int8 DPAS sums 32 |

The reason for the DPAS form at 2-16 is this backend's K-quant evidence, not
an invariant: 0009's product held 111 us at 2, 4 and 8 columns where the q8_1
matvec went 123 / 169 us at 2 / 4 (Q4_K, B60, `measured-here`), and the
matvec needs 256 GRF to keep 2-4 columns in registers (0001, A770). The arm
is there to measure Strata's form on these cards for Q3_K itself.

## Gate

- Correctness (`test-backend-ops`, both cards, `-b GPUOpenCL`): every Q3_K
  `MUL_MAT` and `GET_ROWS` case passes or is declined by design (an odd block
  count); the full `MUL_MAT`, `MUL_MAT_ID` and `GET_ROWS` suites stay green
  (0116 touched the K-quant, IQ and tile-GEMM builds).
- The red case: `GGML_OPENCL_Q3K_OPTS=-DQ3K_RED` (the scales' top two bits
  dropped) fails the Q3_K cases each kernel takes, with the build line
  `Q3_K kernels (...) built with GGML_OPENCL_Q3K_OPTS=-DQ3K_RED` in the log;
  the same binary without it passes.
- Speed: the 1-16 column curve at 4096 x 14336 next to Q4_K, and the AREX-2
  roles' shapes at 1, 2, 6, 8, 16 and 512 columns next to Q4_K; at 1 column
  Q3_K at least as fast per byte as Q4_K (Q3_K is 3.4375 bits a weight
  against 4.5: 59 us at Q4_K's 427 GB/s on the B60, `inferred` from the
  bytes and the 77.4 us above).
- Served (an AREX-2 mix with Q3_K roles, once the GA's file exists): the
  answer-level bar against the same file on the CPU, no `type=q3_K`
  DOWNGRADE key on the dense roles.

## Current state

- Built and unit-tested on the recipe host, tree `d3444fc8`
  (`RECIPE d3444fc: 100% tests passed, 0 tests failed out of 36 EXIT 0`);
  `test-backend-ops` with the new cases compiles. No device there: the
  OpenCL C kernels are first compiled by the card's driver.
- Device-free check of the index math (`tools/q3k_layout_emul.py`,
  `measured-here`, an emulation of the kernels' arithmetic in Python, not
  the kernels): the planes round-trip to the GGUF blocks; the tile GEMM's and
  `GET_ROWS`' decode equals `dequantize_row_q3_K` on every weight; the
  matvec's lane map and the few-column product's half codes against swapped
  activations give the reference dot product to rounding; with `--red` (the
  mutation of `-DQ3K_RED`) all three differ.
- The first card window (window 186, b6d0e532, `measured-here`): the kernels built on both
  cards; the Q3_K `MUL_MAT` cases crashed (SIGSEGV) at the permuted `[K, 1, n]` case, in
  the tensor setup: a planar view written through its own shape (0022's conversion; the
  patch README's section has the measurement). Fixed in the follow-up (planar IO through
  the base, `ggml-opencl-planes.h`, `tests/test_planes.cpp`), with the review's items: the
  per-kernel work-group fit, the Q3_K-only build fallback, the view rule, K 768 cases.
  The card window on the fix: window 189 (next entry).
- Window 189 (aa0720c3, `measured-here`): green on both cards (Q3_K `MUL_MAT` 178 / 0,
  the full `MUL_MAT` 2,725 / 0 to completion, `GET_ROWS`), the red 179 FAIL, the
  build-break fallback costing Q3_K alone. Window 190 (a05b036e, the 2D GEMM): only its dry runs are recorded (clean: the 2D case
  without a fallback key, the crash case declined `reason=view`); its results were not recorded. The 1.6.2 release gate (2026-11-12, `measured-here`, RC 9c0c730e) ran the 2D GEMM's build on the B60: the op suite 186 of 186 on both cards, the red build 0 of 186, mean KL on Llama 2 7B Q3_K_S 0.08300-0.08304 against the CPU's 0.08416 on the same file, llama-bench 3389.67 / 79.64 (B60) and 1928.99 / 74.86 t/s (A770).
- Window 186's numbers (b6d0e532, B60, us, Q3_K / Q4_K, 4096 x 14336): 78 / 78 at 1
  column (the 1-column gate holds: equal time at 3.4375 against 4.5 bits a weight),
  78-79 / 85-87 at 2-8, 97-99 / 136-140 at 9-16; ffn_down at 6 columns 129 / 158; the
  tile GEMM at 512 columns 1,816 / 826, ffn_down 2,442 / 1,191 (about 2x: the prompt
  path missed). Strata's NCOLS matvec arm 145-475 us at 2-8 (rejected for the default),
  one term about equal. The B60's prompt path is now the 2D-block GEMM (Q6_K's form on
  the planes), target Q4_K's 826 us; its per-shape time is not measured (the 1.6.2 gate has llama-bench only, above). The A770 keeps the tile
  (no 2D block reads; int8 does not fit Q3_K's per-16 scales).

## Where it lives

`contrib/llama.cpp/patches/0116-opencl-q3_K.patch`, its section in
`contrib/llama.cpp/README.md`; the switch in `docs/switches.md`.
