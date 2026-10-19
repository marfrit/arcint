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

Staged as (the names the units use): `Qwen_Image-Q4_K_M.gguf`,
`Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`, `qwen-image-vae.safetensors`,
`flux2-klein-4b-Q8_0.gguf`, `Qwen3-4B-Q8_0.gguf`,
`flux2-klein-4b-vae.safetensors`, and `RealESRGAN_x4plus.safetensors` in a
directory of its own (the upscale route loads every file there).
Staging list (operator-local, `/models/imggen/`): the seven files above;
`docs/milestone-1.1.0.md`'s Z-Image set stays for the 1.1.0 unit. Nothing
goes into `/flash-model` between legs (SOP).

## 3. Memory on the 24 GB card

The B60 reports 23,256 MB global / 24,480 MiB when idle (`measured-here`,
OpenCL device query, 2026-10-18). The Qwen-Image **Q4_K_M** form: the files
are 13.07 + 4.68 + 0.25 GB (DiT, TE, VAE; 16.8 GiB as planned from the
tensor sizes), sd.cpp's load reports 11.9 + 3.9 + 0.24 GiB of params on the
card (§9); plus activations; the Z-Image
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
`IM2COL_3D` stays out: the VAE decodes in 2.66 s at 512² with the direct op
(`measured-here`, §9).
**0060** adds `GGML_OP_LEAKY_RELU` (f32/f16, contiguous or strided), which
the ESRGAN upscaler needs; without it the mixed CPU/OpenCL upscaler graph
segfaults (`measured-here`; the likelier mechanism an in-place op moved to
the CPU over an OpenCL buffer's view, `code`, not measured). Campaign:
`docs/campaigns/imggen-leaky-relu-esrgan.md`.

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
port, `Conflicts=` with the B60's language units and with the other image
units (one model on the card at a time), the same contract: OpenAI shape,
`data[0].b64_json`, the sd.cpp seed suffix, `/health` and `/props` (A4 of
1.1.0). As released (the contract in `docs/imggen-service.md`):

| unit | port | model | routes |
|---|---|---|---|
| `arcint-imggen` | 8085 | Z-Image Turbo | `/v1/images/generations` (1.1.0, unchanged) |
| `arcint-imggen-qwen` | 8081 | Qwen-Image | `/v1/images/generations`, `/sdcpp/v1/upscale` |
| `arcint-imggen-flux` | 8082 | FLUX.2 klein 4B | `/v1/images/generations`, `/v1/images/edits`, `/sdcpp/v1/upscale` |

The upscaler is not a unit of its own: sd-server's upscale route runs
ESRGAN from `--hires-upscalers-dir` on the loaded unit's backend. The front
and the client side are the operator's.

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

## 9. State as measured (2026-10-18)

All on the B60 `8086:e211`, OpenCL 0:0, a card window per leg, the owning
language unit stopped and restored; the files sha256-identical to their HF
LFS pointers; every arm through `tools/imggen_gate.py` with its `arm.json`.

**Qwen-Image** (512², 8 steps, seed 42, cfg 2.5, euler, `--flow-shift 3`):
all **16.7 GB of params on VRAM** (DiT Q4_K_M 11.9, TE Q4_K_M 3.9, VAE
0.24 GiB), no RAM fallback; the first request loads the TE in 17.1 s and the
DiT in 47.9 s (from the ZFS mirror). 8 steps **55.7 s** (7.0 s a step); the
Wan VAE decodes in **2.66 s** on the card -- with 0059 off the same graph
crashes (`GGML_ASSERT(dst->extra)` in the OpenCL compute; the likelier
mechanism, from the scheduler's code and not measured, is the conv's bias
`add_inplace` pinned to OpenCL by its weight while its destination is a view
of the CPU buffer the moved conv wrote), so pre-0059 Qwen-Image does not run
at all. The vision model reads the image's text as **"WILLKOMMEN"**, in
black on a white sign against a green forest: the German-lettering mandate
is met. Against the CPU reference (same binary, same files/prompt/steps,
384 s): per-step cosine worst **0.998676**, PSNR 26.26 dB, SSIM 0.9912 --
**outside the placeholder `TOL_COS 0.001`**; the DiT or the text encoder,
not the VAE (the previews are the latent projected to RGB, so the 0059 red
arm's are bit-identical to the green arm's: cos 1.000000). 0059's own red shows in the final image: PSNR **8.70 dB**, SSIM
0.0105 vs the green arm.

**FLUX.2 klein 4B** (512², 4 steps, seed 42, cfg 1.0, euler): DiT Q8_0 +
Qwen3-4B Q8_0 TE + the FLUX2 VAE (32 latent channels), ~8.5 GiB, fits
comfortably. 4 steps **18.8 s**, VAE 2.53 s; the vision model confirms the
prompt (a cat on a windowsill). Against the CPU reference (143.7 s): cosine
worst **0.998115**, PSNR 25.31 dB -- outside the placeholder like Qwen-Image.

**FLUX.2 klein edit** (`/v1/images/edits` shape, `--ref-image`, background
swap): 4 steps **19.5 s**, VAE 2.13 s; the vision model confirms the swap (beach
behind the cat). Against the CPU reference (227.1 s): cosine worst
**0.999539**, PSNR 32.43 dB, SSIM 0.9966 -- **PASS**. The edit path matches
the reference more closely than text-to-image does.

**ESRGAN** (`--mode upscale`, `RealESRGAN_x4plus.safetensors`): 512² ->
2048² in **27 s** on OpenCL after 0060; before 0060 the same command
segfaulted in the CPU fallback. 4x from 1024² gives 4096² (~350 dpi on A3's
short side, `arithmetic`).

**The gate tolerance, D6** (`decision`, operator, 2026-10-18): per-model
`TOL_COS`, about 0.003 for the 1.1.1 models, whose text-to-image curves sat
0.0013-0.0019 below 1.0 (the edit curve inside 0.001). Encoded in
`tools/imggen_gate.py`'s `MODEL_TOL`, keyed by the diffusion model's file:
Qwen-Image and FLUX.2 klein 0.003, Z-Image Turbo 0.001 (its gap measured
0.00007). The release review found the per-step gate blind to the VAE (the
previews are the latent projected to RGB, so 0059's red passed it) and to an
arm that crashed after sampling; without a baseline arm, which the unchanged
build cannot give for Qwen-Image, the final image is now held to a floor
against the CPU reference (Qwen-Image and klein 20 dB / SSIM 0.95, Z-Image
30 dB / 0.99), and a missing final image or a one-sided preview step fails.
The kernel equivalence itself is carried by the op suites.

### 9.1 The release check (`measured-here`, 2026-10-18)

The release build (sd.cpp `f0aac52` on llama.cpp `3e4dfa8` + 0001-0060), the
B60, one window, the 1.1.1 session's arms replayed with their own
`arm.json` flags (identical inputs on both arms), the 1.1.1 session's CPU
references reused, the gate with the per-model tolerances above:

| arm | step | VAE | peak VRAM | gate vs the CPU reference |
|---|---|---|---|---|
| Qwen-Image 512², 8 steps, the German sign | 0.95 s (1.05 it/s) | 2.21 s | 16,995 MiB | PASS: cos worst 0.998676, 26.26 dB, SSIM 0.991232 |
| Qwen-Image, 0059 red | | | | **FAIL**: the final floor (8.75 dB, SSIM 0.0108); the per-step cosine alone passed it |
| FLUX.2 klein edit 512², 4 steps (reference only, the CLI arm) | 0.71 s | 2.13 s | 8,296 MiB | PASS: cos worst 0.999539, 32.43 dB, SSIM 0.996623 |
| klein edit, 0058 red (the last 16 of K zeroed) | | | | passes at TOL_COS 0.003 (cos worst 0.999177, 26.41 dB) |
| FLUX.2 klein **served** edit (`/v1/images/edits`: init image + reference, strength 0.75), B60 sd-server against a CPU sd-server given the same request | | | | **below the proposed floor**: final 17.21 dB, SSIM 0.905 (no previews: sd-server writes none, so no per-step figure) |
| Qwen-Image 1024², 20 steps (the unit's default) through sd-server | 4.44 s | 8.24 s | **18,827 MiB** | 158.9 s a request; no CPU reference at 1024² |

The release build reproduces the 1.1.1 session's B60 images pixel for
pixel (Qwen-Image, the klein edit). §9's "8 steps 55.7 s (7.0 s a step)"
for Qwen-Image at 512² is not a step time: the window's 8 steps ran at
1.05 it/s, consistent with 4.44 s a step at 1024². Two limits of D6's
0.003, recorded:
klein's text-to-image curve sits at 0.0019, so a defect as small as 0058's
red (−6 dB on the final image) passes the klein gate; it fails 52 of 151
Q8_0 op cases, the op suite is the kernel's gate. Op suites on the release
tree: `CONV_3D` 284/284 (the 8 new Wan VAE shapes included), red 6/284;
LEAKY_RELU 7/7 with the red 0/7, after 0060 was narrowed to contiguous
input (its strided kernels measured wrong on a row-padded view; the ESRGAN
path is contiguous).

**The rope-off attribution** (`measured-here`): the 1.1.1 references run
0057's fused CPU RoPE. The same arms on a build without the RoPE
integration patch (sd.cpp's generic RoPE) differ from them by a per-step
cosine of 0.999809 (Qwen-Image; final 32.06 dB) and 0.999986 (the klein
edit; 44.05 dB), and the OpenCL arm sits closer to the fused reference
(0.998676) than to the rope-off one (0.998194). So 0057's CPU forward is
~0.0002 of Qwen-Image's 0.0013 gap; the rest lies between the OpenCL and
the CPU paths of the other ops, not attributed further.

**Upscale through the units**: `POST /sdcpp/v1/upscale` with
`--hires-upscalers-dir`, a 512² PNG to 2048² in 27.9 s on the flux unit
(the 1.1.1 session's 27 s was the CLI). **Edits through the units**: the
HTTP edit uses the init image as well (`docs/imggen-service.md`), so it is
not the CLI arm above (10.96 dB apart, both valid edits by eye); the served
form is gated separately (the row above) and falls below the floor: the
two images agree within 1-7 levels over most of the frame and differ by
~56 in one region (the lower left), so the img2img trajectory diverges
locally. Not the VAE: with `--vae-on-cpu` the B60's edit is 43.72 dB from
the all-OpenCL one and still 17.25 dB from the CPU reference. `sd-cli
--init-img` does not reproduce the server's edit (it returned the
reference-only image unchanged), so there is no per-step arm for it.
Released with this gate open (`decision`, operator, 2026-10-18); the
campaign is `docs/campaigns/imggen-served-edit-divergence.md`. **Unchanged paths**: Z-Image
Turbo 512², seed 42, through the release build's sd-server is pixel-identical
to the 1.1.0 service image; the 1.1.1 language binary (the pre-tag
package) answers "Paris" on the B60 and on the A770, so 0059's and 0060's
OpenCL programs, built at every OpenCL start, compile on both cards.

## 10. References

- stable-diffusion.cpp `f0aac52`: `src/core/ggml_extend.cpp:451`,
  `src/model/vae/wan_vae.hpp`, `src/model/diffusion/{qwen_image,flux}.hpp`,
  `src/model/upscaler/esrgan.hpp`, `examples/server/routes_openai.cpp`.
- arcint's kernels: `contrib/llama.cpp/patches` 0001 (the conv2d XMX tile
  ancestry), 0057, 0058, 0059, and `contrib/sdcpp/build-sdcpp.sh`.
- The models' own configs and cards, and HF's API for downloads/licences.
