# imggen-leaky-relu-esrgan — LEAKY_RELU on OpenCL, the print upscaler

**Closed 2026-10-18** (1.1.1). 0060 is built, its op suite is green and red,
and the ESRGAN upscale runs on the card.

## Charter

An A3 poster from a 1024² image wants a ~4x upscale (`arithmetic`). sd.cpp's
ESRGAN upscaler (`src/model/upscaler/esrgan.hpp`, the `RRDBNet` 23-block 4x
model behind `--upscale-model` / `--mode upscale`) is a 2D-conv CNN whose
`ggml_leaky_relu` the pin's OpenCL did not support, so a mixed CPU/OpenCL
graph sent `GGML_OP_LEAKY_RELU` to the CPU and segfaulted (`measured-here`,
2026-10-18). The likelier mechanism, from the scheduler's code (not
measured): RRDBNet's `ggml_leaky_relu` is in place, so the node moved to the
CPU writes through a view of an OpenCL buffer; the CPU forward is not at
fault. The lever is the missing unary on the device, which removes the
cross-backend in-place view.

## Reference to follow

- **`ggml_leaky_relu`** (`ggml.c:2774-2788`, `code`): the op is top-level
  `GGML_OP_LEAKY_RELU`, not a `GGML_UNARY_OP` enum; the negative slope is
  `dst->op_params`'s first float. The CPU reference is
  `ggml-cpu/ops.cpp:2794` (`measured-here` for the match).
- **`unary_ext.cl`** (`code`): the six-variant shape (f32, f32_4, f16,
  f16_4, f32_nc, f16_nc) and the dispatch pattern this patch copies.
- **Comfy-Org's `RealESRGAN_x4plus.safetensors`** (BSD-3-Clause, 60k
  downloads; `paper`): the classic 4x photo model.

## Gate

- **Op suite**: `test-backend-ops -o LEAKY_RELU` against the CPU reference,
  serial (`-j 1`); the red case negating the slope
  (`GGML_OPENCL_LEAKY_RELU_RED=1`).
- **The upscale**: `--mode upscale` on the card, a real PNG, seconds and an
  output size; the image judged by eye, not a metric.
- **Speed**: seconds for 512² -> 2048², the same command on the CPU for the
  fallback's price.

## Current state

`0060-opencl-intel-leaky-relu.patch`: `leaky_relu.cl`, the load block, the
`GGML_OP_LEAKY_RELU` admission (f32/f16, contiguous or strided), the
dispatch, the red switch.

**Measured-here** (2026-10-18, B60 `8086:e211`):

| arm | result |
|---|---|
| `test-backend-ops -o LEAKY_RELU -j 1` | **7/7 pass** vs the CPU reference |
| the red (`LEAKY_RELU_RED=1`) | **7/7 FAIL** |
| `-o ADD -j 4` (control) | 105/105 pass -- the card is healthy |
| `-o LEAKY_RELU -j 4` | a Neo driver abort before any case (recorded, unexplained) |
| ESRGAN 4x, 512² -> 2048², OpenCL | **27 s**, rc=0 |

Before the patch the same upscale command segfaulted in the CPU
`ggml_compute_forward_leaky_relu`; the pure-CPU `--mode upscale` completed
(`code`/`measured-here`), so the model and the graph were fine and the
missing unary was the cause.

## Where it lives

- `contrib/llama.cpp/patches/0060-opencl-intel-leaky-relu.patch` and its
  README section; `contrib/sdcpp/build-sdcpp.sh`'s kernel assertion.
- The upscaler's model and licence: `docs/milestone-1.1.1.md`.
