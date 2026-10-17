# Image generation on the Arc cards (1.1.0)

The workstream of `docs/milestone-1.1.0.md` (draft charter, 2026-10-17): the
same ggml OpenCL base and arcint kernels that serve the language models, built
for diffusion transformers (DiT/MMDiT) and their VAEs on the B60 and the A770.
This document is the running record; the charter, its criteria A/B/C and its
gate `§3` stay in the milestone document.

## Reference

**stable-diffusion.cpp** (`f0aac52`, 2026-10-17, MIT): the engine, its
`sd-server` surface and its models. The code facts checked at that commit are
in the milestone's `§0`; they are not repeated here.

## Route

D2(a) per the charter's recommendation: **sd.cpp's `sd-server`, built against
arcint's patched ggml tree**, as its own package and unit. arcint supplies the
OpenCL kernels (the series on `contrib/llama.cpp/patches`) and the gates; the
integration code is sd.cpp's own. Consequence: `sd-server` and arcint's
libllama are **separate binaries** and never share one process.

`GGML_MAX_NAME` is the reason that separation is not optional: sd.cpp sets
`-DGGML_MAX_NAME=160` for its whole build (before the ggml subproject), which
changes `ggml_tensor`'s layout, while arcint's libllama build stays at the
pin's 64. libllama and sd.cpp in one process would need one value (charter D5);
out of one process, each keeps its own (`measured-here`, the `§0` review).

## Build — A1

`contrib/sdcpp/build-sdcpp.sh` is the recipe. It requires the sd.cpp tree at
`f0aac52` and a llama.cpp tree at the pin `3e4dfa8` with the series applied, and
asserts both plus the arcint kernels before it configures:

    -DSD_USE_UPSTREAM_GGML=ON -DSD_GGML_SOURCE_DIR=<llama>/ggml
    -DSD_OPENCL=ON -DSD_VULKAN=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF

Upstream mode turns INT8/SageAttention off and FP8 to F16 at load, and takes
RoPE's generic path (`code`, `cmake/ggml.cmake`, `rope.hpp`). `SD_GGML_SOURCE_DIR`
is the ggml **subtree**, so the llama.cpp root is not built.

**Met** (`measured-here`, 2026-10-17): sd.cpp at `f0aac52` in upstream mode
builds against `3e4dfa8` + the 54-patch series (0031, 0055 dropped) on the dev
host, both OpenCL and Vulkan, in 1m45s. The binary embeds arcint's kernels
(`kernel_mul_mm_kq2d_q4_K_i8`, `flash_attn_f32_f16_dpas{,2}`, `strings`), and
278 compile units carry `-DGGML_MAX_NAME=160`, so the 160 width is the whole
build's and not only sd.cpp's translation units.

**Open**: whether the backend actually selects those kernels at run time, and
every A2/Phase-0 criterion below.

## Phase 0 (inventory, open)

One card window, results `measured-here`: Z-Image Turbo on the CPU (F32/BF16)
as the gate's reference and on the card; the CPU islands counted from
`GGML_SCHED_DEBUG`; a profile per op and step (B1); the same run on Vulkan on
the same card (B3). Outcome: kernels ranked by share of the step's time, which
is what Phase 2 works from. The sampler and the card-window rules are
`docs/sop-card-window.md`, first.

Planned cells and flags (operator fills the numbers):

| cell | card | configuration |
|---|---|---|
| reference | none (CPU) | Z-Image Turbo, BF16/F32, seed fixed, same sampler |
| candidate | B60 (GPU.0, `8086:e211`) | sd.cpp OpenCL, arcint series, `--offload-to-cpu` OFF |
| comparison | B60 | sd.cpp Vulkan, same file, seed, steps |
| first op profile | B60 | the candidate run under `cl_timeline_steps.py`/IGA, first step separate |

Models (staged to the node that runs the legs, not into the repository): the
Z-Image-Turbo diffusion GGUF, the FLUX `ae.sft` AutoEncoderKL (Z-Image uses
it), and the Qwen3-4B-Instruct-2507 text encoder. No 3D-convolution VAE yet;
that is Qwen-Image's, Phase 4.

## Phases

Per the charter `§4`: 0 inventory, 1 the image gate (`§3`), 2 the gaps the
profile names (q8_0 XMX GEMM, mask-free attention tile search, VAE conv2d and
GroupNorm, CONV_3D only with Qwen-Image), 3 the service (package, unit,
`/v1/images/generations`), 4 more models, 5 release 1.1.0. The gaps become
`docs/campaigns/` documents with their own gates when the profile names them.

## Where it lives

- The recipe: `contrib/sdcpp/build-sdcpp.sh`.
- The engine: stable-diffusion.cpp `f0aac52` (reference source, not vendored).
- The kernels: `contrib/llama.cpp/patches` (0001-0056 without 0031, 0055), and
  `docs/campaigns/dense-prefill-attention.md` for the attention family.
