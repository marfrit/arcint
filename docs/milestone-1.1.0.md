# 1.1.0 — image generation on the Arc cards (draft)

**Draft** (2026-10-17, branch `imggen`). Nothing here is decided. The
operator amends and decides this charter; an agent does not. Every claim
carries its evidence class: `paper` (a published text, README, model card),
`code` (read in source, file and commit named), `measured-here` (run on this
project's cards), `arithmetic` (derived, not measured) or `decision` (the
operator's, dated).

Goal: text-to-image with diffusion transformers (DiT/MMDiT) on the B60 and the
A770, on the same ggml OpenCL base and with the same Intel kernels as the
language services. The reference engine is **stable-diffusion.cpp**
(<https://github.com/leejet/stable-diffusion.cpp>, commit `f0aac52`, 2026-10-17, MIT).

## 0. Starting point

### 0.1 What the plan assumes, and what of it holds

The starting thesis (external, 2026-10-17): image generation is a prefill
computationally, not a decode; the DiT's linear layers and attention are the
same kernel family as arcint's prompt path; what is new is the VAE and the
gate. Checked against the source:

| thesis | finding | class |
|---|---|---|
| The DiT is GEMM + attention + RMSNorm/SiLU/GELU/RoPE | holds for Z-Image (`src/model/diffusion/z_image.hpp`), Qwen-Image (`qwen_image.hpp`), FLUX.2 (`flux.hpp`, shared with FLUX.1) | `code` |
| Head size 128 | Z-Image: 30 heads x 128, hidden 3840 (`z_image.hpp:32-38`); Qwen-Image and FLUX: 128 | `code` (Z-Image), `paper` (the others) |
| Attention without a causal mask, without a KV cache | yes; sd.cpp calls `ggml_flash_attn_ext` through `ggml_ext_attention_ext` (`src/core/ggml_extend.cpp:615, :713`), K/V cast to F16, F32 accumulation | `code` |
| arcint's DPAS attention fits | yes: `flash_attn_dpas` / `dpas2` take head size 128 or 256, more than 8 query rows, the mask optional (`mask == nullptr` accepted), K/V f16 or quantized (`ggml-opencl.cpp` with the series, the selection before `flash_attn_dpas2`) | `code` |
| `conv_2d`, `group_norm`, `upscale`, `concat`, `pad` are missing from ggml OpenCL | **wrong** at the pin `3e4dfa8`: `supports_op` takes CONV_2D (f16/f32), IM2COL, GROUP_NORM (contiguous), UPSCALE (nearest/bilinear, no antialias), PAD (not circular), CONCAT and even TIMESTEP_EMBEDDING. Whether they are fast on Arc is open: they are upstream's kernels, written for Adreno | `code` |
| What is really missing | CONV_3D and IM2COL_3D (Qwen-Image's Wan VAE, `wan_vae.hpp`), CONV_2D_DW (not on the three models' main path). `custom_4d` is only in the INT8 path, which upstream mode turns off; sd.cpp already precomputes `arange` on the CPU (`flux.hpp:1038, :1539`) | `code` |
| "pp512 +43.6 %" of the prompt attention | not in this project's record; not used | — |
| stable-diffusion.cpp has an OpenCL backend | yes, as a pass-through only: `SD_OPENCL` sets `GGML_OPENCL` (`CMakeLists.txt:123-126`), sd.cpp itself has no OpenCL code; its docs call OpenCL "Adreno only, Q4_0" (`docs/build.md:157-161`, `paper`); no report of Arc + OpenCL in its issues | `code` / `paper` |

### 0.2 What sd.cpp brings (`code`, commit `f0aac52`)

- **ggml** as a submodule on a fork (`leejet/ggml`, `dc99877`, 13 commits on
  ggml-org `a05c992` of 2026-10-15). The fork's commits do not touch
  `ggml-opencl`.
- **Upstream mode**: `-DSD_USE_UPSTREAM_GGML=ON -DSD_GGML_SOURCE_DIR=<tree>`
  (`CMakeLists.txt:98-99`, `cmake/ggml.cmake:24-34`). INT8 and SageAttention
  are then off, FP8 is converted to F16 at load, RoPE takes the generic path
  (`src/model/common/rope.hpp:1075`). Every other `ggml_*` call exists
  upstream (checked against the ggml-org headers, not built).
- **`GGML_MAX_NAME=160`** (`CMakeLists.txt:326-329`); the pin has 64
  (`ggml/include/ggml.h:229`). The value changes `ggml_tensor`'s layout:
  libllama and sd.cpp in one process need the same build.
- **Models** (`src/model.h:14-71`): among them Z_IMAGE, QWEN_IMAGE (2.1),
  FLUX, FLUX2, FLUX2_KLEIN. The text encoders are sd.cpp's own LLM
  implementation (`src/model/te/llm.hpp`): Qwen3 (Z-Image, FLUX.2 klein),
  Qwen2.5-VL (Qwen-Image), Mistral Small 3.2 (FLUX.2)
  (`conditioner.hpp:2002-2024`). The encoder loads a GGUF with llama.cpp's
  tensor names (`name_conversion.cpp:136-154`).
- **VAE**: AutoEncoderKL (Z-Image, FLUX.2: 2D convolutions, GroupNorm with 32
  groups, nearest upscale); the Wan VAE (Qwen-Image, 3D convolutions).
  `--vae-tiling` (256x256 tiles, overlap 0.5); `--conv-direct auto` picks
  between `ggml_conv_2d_direct` and im2col + mul_mat by a probe.
- **Weights**: GGUF and safetensors; F16, BF16, Q4_0 to Q8_0, the K-quants,
  the IQ types (`include/stable-diffusion.h:101-146`).
- **Service**: `sd-server` with `/v1/images/generations` and `/v1/images/edits`
  (`examples/server/routes_openai.cpp:253-334`), the A1111 API `/sdapi/v1/*`
  and an asynchronous job API. Library: `new_sd_ctx`, `generate_image`
  (`stable-diffusion.h:518, :533`); the server serialises one context with a
  mutex.
- **Randomness**: the noise is made on the CPU (`--rng cuda`, the default, is
  a CPU port of Philox; `cpu` = MT19937), so for a fixed seed it is the same
  on every backend. The gate in §3 rests on that (`code`, not measured).

### 0.3 What arcint's kernels already do for a DiT (`code`, series 0001-0056)

- **GEMM**: Intel XMX paths for Q4_K, Q5_K, Q6_K, the IQ types
  (`ggml_cl_kq_type_index`) and F16/BF16 weights (`kernel_mul_mm_f16w_f16`,
  patch 0022). **Q8_0 has none**: it runs upstream's generic kernel. Q8_0 is
  the usual format of FLUX GGUFs.
- **Attention**: as above; built for head size 256 (the Qwens) and 128
  (Mistral Small 24B); measured causal, with a mask, on the agent (256).
- The shapes differ: M ≈ 4,096 tokens a step, no mask, other hidden sizes.
  The searched tile shapes (`kq_tune`, the GA runs) were chosen for the
  language models.

### 0.4 Orders of magnitude (`arithmetic`, not measured)

- 1024 x 1024 pixels, VAE factor 8, 2 x 2 patches: 64 x 64 = **4,096 image
  tokens** a step, plus the text tokens.
- Z-Image (6B, `paper`): ~2 x 6e9 x 4,096 ≈ 49 TFLOP a step without
  attention. The dense 27B on the B60 prefills 787-837 t/s at 4k
  (`measured-here`), ≈ 43-45 TFLOP/s effective. At the same utilisation
  ≈ 1.1 s a step, 8 steps (Turbo) ≈ 9 s, without the encoder and the VAE.
- Memory (weights, rough): Z-Image BF16 ≈ 12 GB + Qwen3-4B Q8_0 ≈ 4.3 GB;
  Qwen-Image 20B Q4_K ≈ 11-12 GB + Qwen2.5-VL-7B Q4_K ≈ 4.7 GB; FLUX.2 dev 32B
  Q4_K ≈ 18 GB + Mistral Small 3.2 Q4_K_M ≈ 14 GB (they do not fit on one card
  together; see D4).

## 1. The operator's decisions (open)

| # | question | options, with their price where known |
|---|---|---|
| D1 | Scope. README and DESIGN §8.8 name multimodal as a non-goal (`decision`). Image generation is output, not input, but it widens the scope from "LLM server" to "inference server" | (a) 1.1.0 takes in text-to-image; (b) a sister project on arcint's ggml; (c) kernel contributions only, the service stays sd.cpp |
| D2 | The integration route | (a) sd.cpp's **sd-server**, built against arcint's patched ggml tree, as its own package and unit: the least code, arcint supplies kernels and gates; (b) an **arcint endpoint** `/v1/images/generations` over libstable-diffusion in the arcint process: one surface, but a second runtime in the process, and `GGML_MAX_NAME` has to change for libllama too; (c) a DiT path of arcint's own: dropped, sd.cpp is the reference and already has it |
| D3 | Card and unit | the B60 on demand (`Conflicts=` with the agent, the writer, Flash-Next, like the writer) or the A770 (16 GiB, only without the coder); either way one card per process (DESIGN §3, README) |
| D4 | First model | **Z-Image Turbo** (6B, 8 steps, Qwen3 encoder, AutoEncoderKL without 3D convolution): the smallest gap. Then Qwen-Image (needs CONV_3D for the Wan VAE) and FLUX.2 (32B + a 24B encoder: load the encoder and the DiT one after the other, or the encoder on the second card, sd.cpp `--backend te=...`) |
| D5 | `GGML_MAX_NAME` | 160 for the whole build (libllama included; an A/B per DESIGN §5 must show the language services do not suffer), or 64 and sd.cpp built at 64 (sd.cpp's tensor names are longer: check for truncation) |
| D6 | The gate (§3): tolerances | proposal in §3; the operator sets the numbers |

## 2. Criteria for 1.1.0

### A. It runs

| # | criterion | state |
|---|---|---|
| A1 | sd.cpp in upstream mode builds against the pin `3e4dfa8` with the series (0001-0056 without 0031, 0055) and `GGML_OPENCL_USE_ADRENO_KERNELS=OFF` | open |
| A2 | The first model (D4) makes a 1024x1024 image on the chosen card (D3) at a fixed seed, with no op silently falling back to the CPU: `GGML_SCHED_DEBUG` counts the splits, every CPU island is named and either built or justified by its share of the time | open |
| A3 | The language services stay as they are: the release A/B (`tests/acceptance/ab.py`) against 1.0.0 passes, the acceptance task ≥ 9/10 | open |
| A4 | A service: unit, package, `/v1/images/generations` (OpenAI shape), `/health`, `/props` with model, card, steps, sampler | open |

### B. It is fast

| # | criterion | state |
|---|---|---|
| B1 | A profile per op and step on the card (device time, the first step separate, since it builds kernels) | open |
| B2 | Every kernel above 10 % of a step's time is set against its measured ceiling (as B4 of the 1.0 charter) or open as a campaign | open |
| B3 | A comparison with a reference on Arc: sd.cpp on Vulkan on the same card, same file, same seed, same steps | open |

### C. It is right

| # | criterion | state |
|---|---|---|
| C1 | The image gate (§3) passes for every model that ships | open |
| C2 | Every new or changed kernel has its red case, which turns the gate red | open |

## 3. The gate

There are no logits, so there is no KL. The answer-level equivalent is the
image; because denoising amplifies small errors over the steps, comparing the
final image alone does not attribute a failure to a kernel. Proposal (the
operator sets the numbers, D6):

- **Reference**: sd.cpp on the CPU in F32 (or BF16), the same file, seed and
  sampler; the noise does not depend on the backend (§0.2).
- **Per step**: cosine similarity and MSE of the latents against the
  reference, logged for each step. Comparative, as the KL bar: the candidate
  is at most a tolerance worse per step than the baseline arm (unchanged
  ggml OpenCL on the same card in the same window).
- **At the end**: PSNR and SSIM (LPIPS optional) of the decoded image against
  the reference, compared with the baseline arm.
- **Content**: a fixed set of prompts with checkable content (an object, a
  count, a colour, a short lettering). How the content is checked
  automatically is open: arcint refuses image input (DESIGN §8.8).
- **Red case first**: a deliberately broken kernel (for example a swapped
  tile in the attention) must visibly push the step curve away.
- **Speed**: seconds a step and an image at 1024x1024, a fresh process per
  cell, the first step separate; against the last release in the same
  window, as for the language services.

## 4. Phases

1. **Phase 0, inventory** (one card window, results `measured-here`): build
   sd.cpp in upstream mode against arcint's ggml tree (A1); Z-Image Turbo on
   the CPU as the reference and on the card; count the CPU islands with
   `GGML_SCHED_DEBUG`; a profile per op (B1); the same run with sd.cpp on
   Vulkan for comparison (B3). Outcome: the list of kernels by their share of
   the time.
2. **Phase 1, the gate**: the reference captures, the step curves, the red
   case (§3). Before any kernel work.
3. **Phase 2, closing gaps**, by the profile, not by guess. Candidates from
   the source:
   - a Q8_0 GEMM on the XMX units (generic today, §0.3), modelled on the
     K-quant GEMMs;
   - the attention without a mask at M ≈ 4,096 (no mask reads; search the
     tile shape for non-causal again, the GA's genome stays, fitness =
     seconds a step);
   - CONV_2D and GROUP_NORM for the VAE, if the profile shows them
     (`--vae-tiling` bounds the memory);
   - CONV_3D / IM2COL_3D only with Qwen-Image.

   Each change as a numbered patch of the series with its README section,
   measurement and red case (DESIGN §1.1).
4. **Phase 3, the service** (after D1-D3): package, unit, endpoint, gates in
   `tests/acceptance/cells.json`.
5. **Phase 4, more models** (Qwen-Image, FLUX.2), each with its own gate.
6. **Release 1.1.0** by the existing procedure (review, pre-tag check, fresh
   container, A/B against 1.0.0).

## 5. Risks

- Upstream's OpenCL kernels for convolution, GroupNorm and upscale are
  written for Adreno and unmeasured on Arc. A slow VAE can eat what the steps
  save (`arithmetic`; phase 0 settles it).
- The text encoders run in sd.cpp's own LLM implementation, not in libllama:
  arcint's MTP, checkpoints and lanes do not apply there; the kernels do, as
  far as they are the same ggml ops.
- Quantization: errors integrate over the steps; Q4_K for the DiT is riskier
  than for an LLM (`paper`, community experience). The gate (§3) decides per
  file.
- Memory at high resolution: attention grows quadratically in the tokens
  (2048x2048 ≈ 16,384 tokens, `arithmetic`); the VAE decoder needs a lot of
  activation memory without tiling.
- Two runtimes on one ggml: the pin has to suit both; every pin move then
  checks sd.cpp too.

## 6. References

- stable-diffusion.cpp (`f0aac52`, <https://github.com/leejet/stable-diffusion.cpp>):
  `src/model/diffusion/{z_image,qwen_image,flux}.hpp`,
  `src/model/vae/{auto_encoder_kl,wan_vae}.hpp`, `src/model/te/llm.hpp`,
  `src/core/ggml_extend.cpp`, `examples/server/`.
- arcint's kernels: `contrib/llama.cpp/patches` (0008, 0017, 0037, 0038 for
  the attention; 0001, 0009, 0013, 0022 for the GEMMs) and
  `contrib/llama.cpp/README.md`.
- The agent's prompt attention: `docs/campaigns/dense-prefill-attention.md`.
