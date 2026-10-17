# imggen-q8_0-xmx-gemm — a q8_0 GEMM on the XMX units for the DiT

**Open** (2026-10-17). Named by the image profile: `kernel_mul_mm_q8_0_f32_l4_lm`,
upstream's generic q8_0 GEMM, was **68.17 %** of the first profile's device time
and, with the fused RoPE (0057, fixed), is **80.9 %** of the 512² q8_0 run and
**68.84 %** of the 1024² run (`measured-here`, `docs/imggen.md`). It is the
durable service form at 1024² (the f16 route does not fit there).

## Charter

The DiT and the Qwen3 text encoder run their large GEMMs in q8_0 (the
Z-Image-Turbo and Qwen3-4B Q8_0 files). On the B60 that lands on upstream's
Adreno-shaped f16 dot `kernel_mul_mm_q8_0_f32_l4_lm` — no XMX path for q8_0
(the milestone `§0.3` gap). Build the q8_0 GEMM on the XMX/DPAS units so the
largest kernel of the image step is not the generic one, and hold the
answer-level bar (the image gate) while the step time falls.

One lever, one campaign: a q8_0 GEMM. The rest of the profile (repeat, mul,
im2col, group_norm, ~5 % together) is other campaigns if priced.

## Reference to follow

- **arcint's own K-quant XMX GEMM**, `kernel_mul_mm_kq_f16` and its few-column
  and 2D-block forms (`contrib/llama.cpp/patches` 0001, 0009, 0011-0014; the
  weights decoded per 32-bit lane inside the kernel, then DPAS). The q8_0
  kernel's shape: decode the `q8_0` int8 code + f16 block scale to f16 and run
  the XMX f16-weights path. **Reference's measured effect**: patch 0001 took
  the dense 27B decode 10.3 -> 18.0 t/s on the B60 (`measured-here`).
- **arcint's f16-weights XMX GEMM**, `kernel_mul_mm_f16w_f16` (patch 0022): the
  DPAS operand form to reuse once the q8_0 weights are decoded.
- **upstream's own int8 dp4a q8_0 dense GEMM**, `gemm_noshuffle_q8_0_q8_1_dp4a`;
  switch `GGML_OPENCL_Q8_DENSE_DP4A` (default ON for Adreno X2E, OFF on Intel;
  conditions `has_integer_dot && N > 8 && K % 32 == 0 && M % 64 == 0`). One env
  var, no code: **measure it first**. Expectation recorded: the int8 Q4_K GEMM
  measured 46-59 % slower on this card (campaign `llama-engine-kernel-gap`,
  lever 7: the per-(token, column) scale binds), so the dp4a route is expected
  to lose — but it must be measured on the same mechanism before the XMX build,
  not assumed (`CLAUDE.md`, "a negative that was never tried is not verified").
- **Vulkan's q8_0 path** (the B3 yardstick: sd.cpp Vulkan on the same B60 runs
  1.78 s a step against OpenCL's 3.69, `measured-here`): read what its q8_0
  dequant + mul-mat does on Xe2 (`code`, to name in this document before
  building).

## Gate

The per-phase speed gate and the comparative image gate (`tools/imggen_gate.py`,
`docs/milestone-1.1.0.md` `§3`): the candidate at most the tolerance worse per
step than the **baseline arm** (the same file, seed, steps and card with the
series OFF in the same window); the final PSNR/SSIM against the baseline; the
step time (seconds a step) against the last phase's. The red case first: a
deliberately broken tile in the new GEMM turns the step curve red.

**The bar (operator, 2026-10-17): >= 70 % of the B60's XMX peak.** The step is
compute-bound (intensity ~1,900 FLOP/byte against the ridge 215; 456 GB/s is
not the bind): the XMX FP16 peak is ~98.3 TFLOP/s, INT8 ~196.6. At 512² a DiT
step is 12.3 TFLOP, at 1024² 49.2, so the bar is ~**0.18 s a step at 512²**,
~0.72 at 1024². The generic q8_0 kernel runs at ~4 TFLOP effective; Vulkan
1.78 s a step (7 %); this campaign's work is 10-20x over the generic kernel.

## Current state (2026-10-17)

`measured-here`, B60, Z-Image Turbo Q8_0 + Qwen3-4B Q8_0 + `ae.safetensors`,
512²/8 steps, seed 42, upstream mode + the series, `GGML_OPENCL_PROFILING`:

| kernel | device ms | share |
|---|---|---|
| `kernel_mul_mm_q8_0_f32_l4_lm` | 23,195.9 | 68.17 % |
| `kernel_repeat_f32` | 3,052.6 | 8.97 % |
| `kernel_mul` | 2,187.9 | 6.43 % |
| `kernel_mul_mat_f16_f16` | 1,991.3 | 5.85 % |
| `kernel_im2col_f16` | 1,094.1 | 3.22 % |
| `flash_attn_f32_f16` | 7.1 | 0.02 % |

Steady step 3.69 s; Vulkan 1.78 s a step on the same card (B3). The B60's XMX
FP16 peak is ~98.3 TFLOP/s (INT8 ~196.6); 512² a step is 12.3 TFLOP, so the
3.69 s is **3.4 %** of the FP16 peak and the bar is 0.18 s.

**Measured (2026-10-17):** the f16 route works -- `--type f16` puts arcint's
existing `kernel_mul_mm_f16w_f16` (0022) on the DiT and gives **1.32 s a step**
(2.8x, 51.2 s end to end), the GEMM at **31 % of the FP16 peak** (was ~4
TFLOP). But it makes the DiT 11.75 GB and the peak 21.1 GB, so 1024² does not
fit as f16. `GGML_OPENCL_Q8_DENSE_DP4A=1` changed nothing (the generic kernel
stayed at 68 %): a dead end, as the recorded expectation said.

**With the fixed fused RoPE (0057), the GEMM is the run** (`measured-here`, 512²
and 1024²): the 512² q8_0 run's device total is 28,678 ms with the generic q8_0
GEMM 23,196 ms (**80.9 %**); the 1024² run's is 120,314 ms with the GEMM
82,822 ms (**68.84 %**), the VAE conv 13,513 ms (11.2 %) and attention
(`soft_max_4`) 4.78 %. **The durable form fits 1024²**: q8_0 peak
`drm-resident-vram0` **17,137 MiB** of 24,480, step **13.14 s** (pre-0057
15.78), no CPU island (`measured-here`). The f16 route's 512² peak is already
21.1 GB, so q8_0 carries the memory f16 cannot. Build the q8_0 XMX GEMM for the
memory and the remaining GEMM time; the other campaigns carry the rest.

## Recon (2026-10-17, `code`)

- **The kernel to model on**: `mul_mm_kq_f16.cl` (arcint's XMX file) defines
  `kernel_mul_mm_f16w_f16` and the K-quant kernels through `MM_BODY(DEQ)`: a
  work-group stages a tile of weight rows in local memory, dequantizing one
  256-weight super-block at a time to f16, then runs fp16 DPAS (k16). The
  activation operand is f16 columns (`xh`) with a per-column f32 scale
  (`xscale`); the result is `xscale[c] * acc` accumulated f32. `PLAIN_ARGS`
  fixes the arg order `(xh, xscale, dst, offd, nb1, ne00, ne01, ne11)`; a DEQ
  takes `(wt, <weight args>, wrow0, row0, ne01, <nblocks>, ib)`. The dispatch
  branch is `ggml_cl_mul_mat` at `ggml-opencl.cpp:27763` (guards: src0 F16,
  src1 F32, dst F32, `ne11 >= 16`, `ne00 % 16 == 0`, 2D-weights/columns,
  contiguous).
- **What q8_0 needs**: a `deq_q8_0` reading the layout the tree already holds:
  the generic `mul_mm_q8_0_f32_l4_lm.cl` (154 lines, BM/BN/BK 64/64/32) shows
  q8_0 weights split into two device buffers -- `global char4 * src0_q` (int8
  codes) and `global half * src0_d` (the per-32 scale). The XMX form is
  `kernel_mul_mm_q8_0_f16(global const uchar * q, ulong offq, global const half * dp, PLAIN_ARGS)`
  with `MM_BODY(deq_q8_0(...))`, registered in `ggml-opencl/CMakeLists.txt`,
  and one dispatch branch beside the f16w one (`src0t == GGML_TYPE_Q8_0`,
  src1 F32, dst F32, the same tile guards). The IM2COL shape needs nothing new
  for the DiT.
- **Reference to follow**: Vulkan's q8_0 path on Xe2 (`code`,
  `ggml-vulkan/vulkan-shaders/mul_mm.comp` binding `BUF_Q8_0_P16` and
  `dequant_funcs.glsl` `DATA_A_Q8_0`): the blocks are dequantized to f16 **in
  the shader** (packed `block_q8_0_packed16`) and fed to the coopmat/DPAS
  path -- dequantize-then-GEMM, the same shape as `mul_mm_kq_f16.cl`. arcint's
  K-quant XMX (0001/0009/0011-0014) and `kernel_mul_mm_f16w_f16` (0022) are
  the reuse forms. **Reference's measured effect**: patch 0001 took the dense
  27B decode 10.3 -> 18.0 t/s on the B60 (`measured-here`). The generic kernel
  runs at ~4 TFLOP effective; Vulkan's 1.78 s a step on the same B60 is the
  B3 yardstick (`measured-here`).
- **Durable-form fit** (`measured-here`, fixed 0057): q8_0 at 1024² peak
  17,137 MiB of 24,480, no CPU island; the DiT as f16 is 11.75 GB and its 512²
  sampler peak 21.1 GB, so 1024² cannot.

## Where it lives

- The campaign's patch will be `contrib/llama.cpp/patches` (the next free
  number after 0057), with its README section and its red case.
- `docs/switches.md` gains any env the leave-in form adds, in its class.
- The engine record: `docs/imggen.md`; the charter: `docs/milestone-1.1.0.md`.
