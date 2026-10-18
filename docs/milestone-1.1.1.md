# 1.1.1 — more image models, editing, and upscaling

The 1.1.1 charter, drafted 2026-10-18 after the 1.1.0 release (Z-Image Turbo
through sd.cpp's `sd-server` on arcint's ggml, patches 0057/0058). §1 is the
operator's mandate, §2 the model set it implies, with licences and the
staging list; §3 the memory plan; §4 the kernel gap; §5 the editing surface;
§6 the upscaler; §7 the service form; §8 the phases. Every claim carries its
evidence class: `paper`, `code`, `measured-here`, `arithmetic` or `decision`.

Goal: make image generation **useful for the operator's marketing work**
(German lettering on posters and flyers), then carry on to 1.2.0 (`decision`,
operator, via the session's mandate, 2026-10-18). 1.1.0 made a service that
serves one English-prompted 1024² image; 1.1.1 adds the models the work
needs, editing, and print-resolution upscaling.

## 1. The mandate

- **German lettering is the reason** for Qwen-Image: it renders text in the
  image, including German, which Z-Image Turbo does weakly (`decision`,
  operator, 2026-10-18).
- **Editing is wanted**: background swap, object removal, a variant of an
  uploaded photo, through `/v1/images/edits` (`decision`, operator). The
  model is FLUX.2 klein or Qwen-Image-Edit.
- **Print resolution**: an A3 poster from a 1024² image needs ~4x, so an
  ESRGAN-style upscaler (`decision`, operator).
- The 1.1.x line stays usable for a non-technical user: each model its own
  on-demand unit behind the same front, the same OpenAI request shape.

## 2. The model set

Rule (`decision`, operator, 2026-10-18): official or high-download
(>10k downloads) weights only, safetensors or GGUF from known publishers;
anything low-reputation only inside a container. Licences checked and
recorded for commercial marketing use.

| role | repo / file | licence | downloads | size | sha256 (HF) |
|---|---|---|---|---|---|
| Qwen-Image DiT | `QuantStack/Qwen-Image-gguf`, `Qwen_Image-Q4_K_M.gguf` | Apache-2.0 | 26k | 13.07 GB | `645473886d7dbb01…` |
| Qwen-Image TE | `mradermacher/Qwen2.5-VL-7B-Instruct-GGUF`, `Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf` | Apache-2.0 | 41k | 4.68 GB | `0f00a930ba3108b6…` |
| Qwen-Image VAE | `Comfy-Org/Qwen-Image_ComfyUI`, `split_files/vae/qwen_image_vae.safetensors` | (Comfy-Org mirror of Wan2.1 VAE) | ~ | 0.25 GB | `a70580f0213e6796…` |
| Qwen-Image-Edit-2509 (optional) | `QuantStack/Qwen-Image-Edit-2509-GGUF` + `mradermacher` TE + `Qwen2.5-VL` `mmproj` | Apache-2.0 | 108k | ~ | — |
| FLUX.2 klein 4B DiT | `leejet/FLUX.2-klein-4B-GGUF`, `flux-2-klein-4b-Q8_0.gguf` | Apache-2.0 | 70k | 4.30 GB | `0bba6951258ec8f9…` |
| FLUX.2 klein TE | `unsloth/Qwen3-4B-GGUF`, `Qwen3-4B-Q8_0.gguf` | Apache-2.0 | (Qwen3-4B line) | 4.28 GB | `eed555233267a33c…` |
| FLUX.2 VAE | `black-forest-labs/FLUX.2-klein-4B`, `vae/diffusion_pytorch_model.safetensors` | Apache-2.0 | 438k (repo) | 0.17 GB | `ca70d2202afe6415…` |
| ESRGAN | `Comfy-Org/Real-ESRGAN_repackaged`, `RealESRGAN_x4plus.safetensors` | BSD-3-Clause | 60k | 0.067 GB | `37f9a931c215f040…` |

Sources checked on HF's API on 2026-10-18 (`paper` for the model cards and
licence tags, `measured-here` for the sizes and hashes). Notes:

- Qwen-Image is the 20B MMDiT with **Qwen2.5-VL-7B** as the text encoder and
  the **Wan VAE** (3D convolutions) (`code`, `docs/milestone-1.1.0.md` §0.2).
  sd.cpp's own docs recommend exactly these three files
  (`docs/qwen_image.md`, `code`). city96's `qwen-image-Q4_K_M.gguf` (46k) is
  the same file size and a second publisher; QuantStack additionally ships
  the VAE.
- **FLUX.2 klein 4B** is the distilled, Apache-2.0, edit-capable model of the
  family (438k downloads; `black-forest-labs/FLUX.2-klein-4B`, `paper`). Its
  transformer is 5 double + 20 single layers, hidden 3072, 128 head
  (`code`, `transformer/config.json`); its TE is structurally `Qwen3-4B`
  (hidden 2560, 36 layers, 32/8 heads, vocab 151936, `code`, `config.json`),
  so a `Qwen3-4B` GGUF loads. Its VAE is `AutoencoderKLFlux2` (32 latent
  channels, 2D, `code`), not FLUX.1's `ae.safetensors`. `leejet`'s GGUF is
  converted **with sd.cpp itself**, the safe publisher for this engine.
  Editing passes the reference image to the DiT (`ref_latents`,
  `pass_to_vlm=false`, `code`, `flux.hpp:1648` + `model.hpp:35`), so no
  vision `mmproj` is needed.
- **Qwen-Image-Edit** is the alternative if FLUX.2 klein's editing is not
  good enough: `Qwen/Qwen-Image-Edit-2509` (470k, Apache-2.0) with a
  `--llm_vision` `mmproj` (`code`, `docs/qwen_image_edit.md`). It costs a
  second 20B DiT on disk and the same Wan VAE; it is not staged first.
- The ESRGAN safetensors is the Comfy-Org repack of `RealESRGAN_x4plus`
  (BSD-3-Clause), the classic 4x photo upscaler; `--upscale-model` takes it
  (`code`, `examples/common/common.cpp:543`).

Staging list (operator-local, `/models/imggen/`): the seven files above;
`docs/milestone-1.1.0.md`'s Z-Image set stays for the 1.1.0 unit. Nothing
goes into `/flash-model` between legs (SOP).

## 3. Memory on the 24 GB card

The B60 reports 23,256 MB global / 24,480 MiB when idle (`measured-here`,
OpenCL device query, 2026-10-18). The Qwen-Image **Q4_K_M** form: weights
16.8 GiB (DiT 12.17 + TE 4.36 + VAE 0.24) plus activations; the Z-Image
service's f16 form peaked at 18.8 GiB with a 5.6 GiB margin
(`measured-here`, `docs/imggen.md`), so the fit is thin but likely; the
**Q4_K_S** DiT (11.3 GiB) is the fallback if it does not (D-quant note:
charter §5's "Q4_K for the DiT is riskier" — the image gate decides).
Choices, in the order to try (`decision` pending measurement):
1. both on the B60, `--auto-fit off` like the Z-Image unit;
2. the **text encoder on the A770** (GPU.1, 16 GB) via
   `--backend te=<A770 device>` (`code`, `common.cpp:558`; device names are
   the `ggml_backend_dev_name`s), the DiT and VAE on the B60;
3. **one after the other**: `--offload-to-cpu` keeps weights in RAM and
   loads on demand (`code`), so the TE and DiT are never resident together;
   slower, but it is what `docs/qwen_image.md`'s own example uses.
The FLUX.2 klein form is small (DiT 4.0 + TE 4.28 + VAE 0.16 GiB) and needs
no plan. The ESRGAN model is 0.067 GB.

## 4. The kernel gap — `CONV_3D` / `IM2COL_3D`

The pin's ggml OpenCL takes `CONV_2D`, `IM2COL`, `GROUP_NORM`, `UPSCALE` and
`CONCAT`, but **neither `CONV_3D` nor `IM2COL_3D`** (`code`,
`ggml-opencl.cpp`'s `supports_op`). sd.cpp's `ggml_ext_conv_3d`
(`src/core/ggml_extend.cpp:451`) probes `IM2COL_3D` and falls to
`ggml_conv_3d_direct`, so a backend with neither computes the Wan VAE's 3D
layers on the CPU. **0059** adds `GGML_OP_CONV_3D` as an implicit GEMM on the
conv2d tiling, with the depth folded into `CRS` — no im2col expansion in
global memory. Campaign: `docs/campaigns/imggen-wan-vae-conv3d.md`.
`IM2COL_3D` stays out until the VAE's device time is known (`arithmetic`).

## 5. Editing

`sd-server` already has `POST /v1/images/edits` (`code`,
`examples/server/routes_openai.cpp:334`): multipart, `prompt` required,
`image[]` (and optional `mask`), `n`, `size`, `output_format`, the same
`<sd_cpp_extra_args>` seed suffix on the generation path. FLUX.2 klein
passes the reference latent to the DiT, so background swap, object removal
and photo variants are its native jobs (`code`, `paper`). Gate: the same
image gate, a reference image arm added to it; the CPU reference runs the
same edit.

## 6. Upscaling — ESRGAN

`--upscale-model <esrgan model>` runs sd.cpp's `RRDBNet`
(`src/model/upscaler/esrgan.hpp`, 23 blocks, 4x) on the same ggml backends;
the DiT path can also upscale in-graph (`code`). Gate: image quality is
judged by the answer bar only as "the upscale runs on the card and the
image is not worse"; the content check is the operator's eye for print.
Print arithmetic: 4x from 1024² gives 4096², ~350 dpi on A3's short side
(`arithmetic`).

## 7. The service

Each model its own **on-demand unit next to `arcint-imggen`**, its own
port, the same `Conflicts=` (the B60's language units), the same contract:
OpenAI shape, `data[0].b64_json`, the sd.cpp seed suffix, `/health` and
`/props` (A4 of 1.1.0). The front and the client side are the operator's;
the unit names, ports, model names and request formats are documented at
the end of the work. Planned:

| unit | port | model | pipes |
|---|---|---|---|
| `arcint-imggen` | 8085 | Z-Image Turbo | 1.1.0, stays |
| `arcint-imggen-qwen` | 8086? | Qwen-Image | :8086 is the front; the next free 8090-8098 |
| `arcint-imggen-flux2klein` | next | FLUX.2 klein | `/v1/images/generations` + `/v1/images/edits` |
| upscaler | next | (in-graph / one-shot) | an option on a unit or its own |

Ports 8090-8098 are the card-window range; an on-demand service unit needs a
free one (an operator call). The exact wiring is fixed after the gates.

## 8. Phases

1. **Model staging and the kernel** (done for the kernel): 0059 built, op
   suite green and red measured.
2. **Qwen-Image end to end**: run it on the card, the CPU reference and the
   image gate (512² if 1024²'s CPU reference takes hours, stated), the
   memory arm chosen, the German-lettering content check by eye.
3. **FLUX.2 klein end to end**: generation and edit gates.
4. **ESRGAN**: the upscale on the card.
5. **The service**: the units and the request formats.
6. **Release 1.1.1** by the existing procedure (the release session owns
   push/tag/package).

## 9. References

- stable-diffusion.cpp `f0aac52`: `src/core/ggml_extend.cpp:451`,
  `src/model/vae/wan_vae.hpp`, `src/model/diffusion/{qwen_image,flux}.hpp`,
  `src/model/upscaler/esrgan.hpp`, `examples/server/routes_openai.cpp`.
- arcint's kernels: `contrib/llama.cpp/patches` 0001 (the conv2d XMX tile
  ancestry), 0057, 0058, 0059, and `contrib/sdcpp/build-sdcpp.sh`.
- The models' own configs and cards, and HF's API for downloads/licences.
