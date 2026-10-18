# imggen-wan-vae-conv3d — the Wan VAE's 3D convolutions on the OpenCL card

**Open** (1.1.1). 0059 is built and its op suite is measured; the Qwen-Image
image gate and the speed figure are owed.

## Charter

Qwen-Image's VAE is the Wan VAE: its decoder convolves in 3D (a
`[KW, KH, KD, IC*OC]` kernel over `[IW, IH, ID, IC*N]`), and the pin's
ggml OpenCL carries neither `CONV_3D` nor `IM2COL_3D`. So sd.cpp's
`ggml_ext_conv_3d` falls to the CPU for every 3D layer, which is why
Qwen-Image is not usable on the card yet. The lever is a device `CONV_3D`
(or `IM2COL_3D`) so the Wan decoder runs on VRAM, like Z-Image's 2D VAE.

## Reference to follow

- **CUDA `im2col_3d`** (`ggml-cuda/im2col.cu:123-218`, `code`): the winner
  folds the depth into the reduction axis of an implicit GEMM; CUDA has no
  direct-conv 3D at all.
- **Vulkan `conv3d_mm.comp`** (`code`): the cooperative-matrix implicit
  GEMM for `GGML_OP_CONV_3D`; Vulkan's `IM2COL_3D` is a separate pass.
- **Metal `kernel_conv_3d`** (`ggml-metal/kernels/conv.metal:636`, `code`):
  a direct gather-and-sum, the fallback Metal uses because it lacks
  `IM2COL_3D` (sd.cpp's `#850` comment).
- **arcint's own `conv2d_f16_f32.cl`** (patch 0001's series, `code`): the
  implicit-GEMM tiling this patch copies, with `K = OC`,
  `CRS = IC*KD*KH*KW`, `NPQ = N*OD*OH*OW`.

The chosen form is the implicit GEMM: it never materialises the im2col
matrix (no `IC*KD*KH*KW` expansion in global memory), stages 1,024 weight
and 1,024 input cells in local memory a CRS block, and reuses the tile
throughout. The CPU's own `conv_3d` (`ggml-cpu/ops.cpp:7562`) also does
im2col + `mul_mat`, so the kernel is compared against the same arithmetic.

## Gate

- **Op suite first** (`test-backend-ops -o CONV_3D`): the backend against
  the CPU reference, the existing 276 3D cases (strides, padding, dilation,
  `KD/KH/KW` 1 and 3, kernel offsets, the degenerate zero-size shapes).
- **Red case**: `GGML_OPENCL_CONV3D_RED=1` makes the kernel ignore the
  weights, so every output is 0 — a gross tripwire, like 0022's f16 red.
- **Baseline**: `GGML_OPENCL_CONV_3D=0` returns `supports_op` false, the
  arm this patch does not build.
- **The image gate**: Qwen-Image against a CPU reference with the same
  files, prompt, size, steps and seed, every arm through
  `tools/imggen_gate.py`; 512² while 1024²'s CPU reference takes hours
  (stated when it is).
- **Speed**: seconds a step and a VAE decode at the measured resolution,
  the first step separate, against the CPU fallback in the same window.

## Current state

`0059-opencl-intel-conv-3d.patch`: `conv3d.cl` (f32 weights) and
`conv3d_f16_f32.cl` (f16 weights), both on the conv2d implicit-GEMM tiling
with the depth folded into CRS; `supports_op` for
(f16 weights x f32 input -> f32) and (f32 x f32 -> f32); the dispatch and
the two switches.

**Measured-here** (2026-10-18, B60 `8086:e211`, OpenCL 0:0):

| arm | `test-backend-ops -o CONV_3D` |
|---|---|
| 0059 on | **276/276 pass** vs the CPU reference |
| 0059 red (`CONV3D_RED=1`) | **270 FAIL**, 6 pass (the `CRS = 0` degenerate shapes, 0 in both) |
| 0059 off (`CONV_3D=0`) | every case unsupported/skipped |

The image gate and the VAE speed are open; the CPU reference for
Qwen-Image is the next leg.

## Where it lives

- `contrib/llama.cpp/patches/0059-opencl-intel-conv-3d.patch`, its README
  section, and `contrib/sdcpp/build-sdcpp.sh`'s kernel assertion.
- The model set and memory plan: `docs/milestone-1.1.1.md`.
