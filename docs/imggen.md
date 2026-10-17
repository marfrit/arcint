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
that is Qwen-Image's, Phase 4. The staged files and their hashes are below.

## Phase 0 — first card cell (`measured-here`, 2026-10-17)

One window on the B60 (GPU.0, `8086:e211`, OpenCL 0:0): the owning unit
`arcint-agent.service` stopped and restored, the SOP sampler on the physical
host, one fresh process. sd.cpp upstream mode + the series, Z-Image Turbo Q8_0
+ Qwen3-4B Q8_0 + `ae.safetensors`, 512x512, 8 steps, seed 42,
`--cfg-scale 1.0`, `--sampling-method euler`, `--backend GPUOpenCL
--auto-fit off`, `GGML_SCHED_DEBUG=1`.

| module | params | placement | segments |
|---|---|---|---|
| text encoder (Qwen3) | 3,974 MB, 386 tensors | VRAM, OpenCL | 1 |
| diffusion (Z-Image) | 6,273 MB, 452 tensors | VRAM, OpenCL | 1 |
| VAE | 95 MB, 138 tensors | VRAM, OpenCL | 1 |

**No CPU island for this cell**: every module ran as one segment on VRAM (the
model-manager and compute-buffer lines read VRAM/OpenCL, "peak across 1
segment"); nothing landed on RAM or disk, and no "not supported" fallback was
logged. A2's "no op silently falls back to the CPU" holds here; other cells
still need the same check.

| figure | value |
|---|---|
| first step (kernel build + warm-up) | 24.03 s |
| steady step, 512² | 3.69 s |
| sampling, 8 steps | 49.85 s |
| VAE decode, 512² | 4.50 s |
| loads (te 10.67 + dit 20.22 + vae 1.02) | ~32 s |
| `generate_image` total | 65.13 s |
| peak `drm-resident-vram0` (sampler) | 12,185 MiB |
| peak `drm-resident-gtt` | 66 MiB |

3.69 s/step at 512² (1,024 image tokens) sits inside the sd.cpp-Vulkan web
band (RX 470 7 s/it at 512², RTX 2070 1.19 s/it) but is `measured-here`, not
comparable across engines. At 1024² the GEMMs quadruple and attention grows
quadratically. **B1's profile** is served by a `GGML_OPENCL_PROFILING` build
(there is no OpenCL Intercept layer on the host): the backend dumps
`cl_profiling.csv` (op, kernel, device ms) and `cl_trace.json` in its cwd at
free, read by `tools/imggen_profile.py`. The profiling leg is the next card
cell. The 12,185
MiB peak leaves a 24 GB card ~11 GiB for the 1024² activations.

Next card cells: 1024²/8 steps (A2), a `GGML_OPENCL_PROFILING` per-op profile
(B1), the same file/seed/steps on sd.cpp Vulkan (B3), and the red-first gate
(Phase 1).

### Phase 0 — second window (`measured-here`, 2026-10-17)

One B60 window, the agent unit stopped and restored; the SOP sampler peaked at
`drm-resident-vram0` **12,765 MiB**.

**A2, 1024²/8 steps (GPUOpenCL).** 10,509 MB of params all on VRAM (text
encoder 4,076, diffusion 6,273, VAE 160); the DiT ran one segment with a
2,380 MB compute buffer, and the VAE the tiled decode (49 tiles). First step
36.48 s, then **15.78 s a step**, `generate_image` **194.77 s**. No CPU island;
the 24 GB card held it, so **A2 met**.

**B1, the per-op profile (the `GGML_OPENCL_PROFILING` build, 512²).** Device
total 34,025 ms over the 8-step run; a 1-step run's 7,395 ms is the first step
separate. Ranked by kernel (the csv's first column is the output tensor, so the
kernel is the key):

| kernel | device ms | share | launches |
|---|---|---|---|
| `kernel_mul_mm_q8_0_f32_l4_lm` | 23,195.9 | **68.17 %** | 1,360 |
| `kernel_repeat_f32` | 3,052.6 | 8.97 % | 1,096 |
| `kernel_mul` | 2,187.9 | 6.43 % | 1,153 |
| `kernel_mul_mat_f16_f16` | 1,991.3 | 5.85 % | 39 |
| `kernel_im2col_f16` | 1,094.1 | 3.22 % | 39 |
| `kernel_mul_mm_f32_f32_l4_lm` | 901.0 | 2.65 % | 629 |
| `kernel_group_norm` | 310.5 | 0.91 % | 30 |
| `flash_attn_f32_f16` | 7.1 | **0.02 %** | 6 |

**The Phase-2 lever this names:** 68 % of device time is
`kernel_mul_mm_q8_0_f32_l4_lm`, upstream's **generic** q8_0 GEMM -- the §0.3 gap
("Q8_0 has no XMX path"). The DiT attention is 0.02 % and ran upstream's
`flash_attn_f32_f16`, **not** arcint's `flash_attn_f32_f16_dpas` (the DPAS path
was not selected for these shapes). The VAE's conv (im2col + f16 mul_mat) and
GroupNorm are ~5 %. So Phase 2 starts at a **q8_0 XMX GEMM**, and the
non-causal-attention tile search is not the first lever.

**B3, sd.cpp Vulkan on the same B60 (`Vulkan1`, BMG G21), same file, seed and
steps, 512²/8 steps.** First step 22.87 s, then **1.78 s a step** against
OpenCL's 3.69; `generate_image` **49.52 s** against 65.13; all VRAM, one
segment. The gate against the CPU reference: cosine worst 0.999984, PSNR
45.08 dB, SSIM 0.999781 -- correct, and **~2.1x faster per step than the OpenCL
path**. Closing that gap is B3's work, and the q8_0 GEMM is its likely lever
(upstream's Vulkan q8_0 dequant GEMM runs faster than the OpenCL Adreno one).

## Staged models (Phase 0/1)

On the node that runs the legs, not in the repository (measured-here,
2026-10-17). One file set for both arms, so only the backend differs:

| file | source | sha256 | size |
|---|---|---|---|
| Z-Image-Turbo diffusion | `leejet/Z-Image-Turbo-GGUF`, `Q8_0` | `df1c5baa…9a1` | 6.58 GB |
| text encoder | `unsloth/Qwen3-4B-Instruct-2507-GGUF`, `Q8_0` | `391c1e41…068` | 4.28 GB |
| VAE | `Comfy-Org/z_image_turbo`, `split_files/vae/ae.safetensors` | `afc8e282…e38` | 0.34 GB |

`Q8_0` on both arms isolates the backend, not the quantisation. The gated
`black-forest-labs/FLUX.1-schnell` `ae.safetensors` is the same VAE and needs
a token; the Comfy-Org mirror is ungated.

## Gate instrument (Phase 1, first form)

`tools/imggen_gate.py`, stdlib only (the card host has no numpy, PIL or
ffmpeg). Two subcommands: `run` drives sd-cli with the fixed flags and
writes `final.png`, `prev_%03d.png` and `arm.json`; `compare` reads two arms
and prints the per-step cosine and MSE and the final PSNR and a global SSIM,
then PASS/FAIL against the tolerances.

The per-step figure is a **latent proxy**: sd.cpp's `PREVIEW_PROJ` is the
latent projected to RGB, the closest the CLI comes to the latent itself
(`code`, `diffusion_engine.cpp:2145`). Raw F32 latents need a driver that
hooks `sd_set_backend_eval_callback` or an sd.cpp change; the sampler's
latents live in C++ between steps, so the graph-eval tensors are the denoiser
output, not the state. The proxy is enough to move red-first; the raw hook is
the next step if the operator wants the logits-equivalent figure.

**Measured-here** (2026-10-17): the CPU reference arm at 256x256, 4 steps,
seed 42, `--cfg-scale 1.0`, euler, runs in 70 s (first step 28-30 s, 8.1 s
per step after; 2 x 6.0e9 x 256 MACs a step) and the tool's self-check
(reference against itself) reads cosine 1.000000 and PSNR inf every step.
The tolerances are placeholders: `TOL_COS=0.001`, `TOL_MSE=0.05`,
`PSNR_TOL=1.0 dB`, `SSIM_TOL=0.01`, printed with every verdict, the
operator's to set (charter D6).

**First CPU-vs-B60 comparison** (`measured-here`, 2026-10-17; the same file,
seed 42, 8 steps, euler): the CPU reference at 512² took a 51.33 s first step
and ~31 s a step; the B60 OpenCL candidate's previews matched to cosine worst
**0.999984** every step, MSE per step at most **1.61** (pixel-squared), and the
decoded images read **PSNR 45.11 dB, SSIM 0.999783**. So the arcint OpenCL path
reproduces the CPU reference to the eye. The MSE/PSNR/SSIM rows are **not
gated** here: the rule is comparative, and the baseline arm (the unchanged
OpenCL on the same card) is a Phase-2 cell, not yet run. The tool's first
version gated the candidate's worst MSE against its own mean -- fixed: the
comparative figures need the baseline, as charter §3 says.

## Speed reference (web, 2026-10-17)

No own numbers exist yet (Phase 0 is the first card leg), so this is the external
yardstick. Evidence class `web`: community or vendor reports read online, named
by URL. **Only the sd.cpp rows are the same engine as ours**; the ComfyUI /
PyTorch-XPU rows are context, not the B3 comparison, and the closest Intel Arc
data point uses PyTorch, not sd.cpp.

| card, engine | resolution / steps | figure | source |
|---|---|---|---|
| **sd.cpp Vulkan, RX 470 4 GB** | 512², Q3_K | 7 s/it (7.3 with `RADV_PERFTEST=nogttspill`) | discussion 1026 |
| **sd.cpp Vulkan, RX 470** | 1024², Q4_K | 35 s/it | discussion 1026 |
| **sd.cpp Vulkan, RX 580 4 GB** | 512², Q3_K | 20 s/it; Q4_K 17 s/it | discussion 1026 |
| **sd.cpp Vulkan, RTX 2070** | 512², Q3_K_M | 1.19 s/it | discussion 1026 |
| **sd.cpp Vulkan, RTX 3050 laptop** | 512² | 2.17 s/it | discussion 1026 |
| **sd.cpp, RTX 2060 6 GB** | 1024² | 4 s/it (ComfyUI 2.5 after its fix) | reddit 1pchpjb |
| **sd.cpp, RTX 2060S** | 1024² | 2.23 s/it | reddit 1pchpjb |
| **sd.cpp, GTX 1080, Q6_K + Qwen3-4B Q6_K** | 1024² | ~30 s/it → ~220-240 s/image | reddit 1r1s3w1 |
| **PyTorch XPU (SDNQ), Arc A770, Xe driver** | 1024², BF16, batch 4, 9 steps | 7.03 s/it; OOM past batch 4 (π.16 GB) | reddit 1p9xzhf |
| **PyTorch XPU (SDNQ), Arc A770** | 1024², quantized, batch 1 | ~1.6 s/it; INT8 ~0.88 s/it (1.14 it/s) | reddit 1p9xzhf |
| **ComfyUI / PyTorch, RTX 4090** | 1024², 8 steps | ~2.3 s | saascrmreview, runaihome |
| **ComfyUI, RTX 5090 / Pro 6000 Blackwell** | 512² batch 16 (=4×1024²) | ~1.1 s / ~0.8 s a 512² image (÷4 ≈ 0.2-0.3 s a 1024²) | reddit 1pfyk0y |
| **Measured 43-card board, ComfyUI** | Z-Image Turbo | B300 39.67, 5090 10.72, 4090 7.24, 16 GB 1.56 images/min | gpubattle |
| **ComfyUI, 3080 laptop / 3090 Ti / 4070 Ti / M4 Pro** | 1024², 7 steps | interactive board, per-device load and gen times | miroleon |

**A source gap:** no sd.cpp-on-Intel-Arc report was found for Z-Image Turbo.
The `B580`/`B60` Vulkan discussion in llama.cpp is text models only, and the Arc
A770 image row is PyTorch. Our B60 OpenCL number and the sd.cpp-Vulkan-on-Arc
comparison (B3) fill that gap.

**CPU (context):** sd.cpp advertises a CPU-only build, Z-Image included
(`web`); a third-party CPU Z-Image build (3.51 GB) is published. Our own CPU
reference is above (`measured-here`): 8.1 s/step at 256² means ~4× at 1024²
plus the encoder and VAE -- about 5-6 min/image, which is why the reference
arm in the gate is a slow but card-free cell.

Sources: `https://github.com/leejet/stable-diffusion.cpp/discussions/1026`,
`https://www.reddit.com/r/StableDiffusion/comments/1pchpjb/`,
`1r1s3w1`, `1p9xzhf`, `1pfyk0y`,
`https://gpubattle.com/guides/z-image-turbo-gpu-benchmarks`,
`https://miroleon.github.io/z-image-turbo-benchmark/`,
`https://saascrmreview.com/z-image-turbo-review/`,
`https://runaihome.com/blog/z-image-turbo-16gb-vram-home-lab-image-gen-2026/`.

## Performance target (`web` + `arithmetic`, 2026-10-17)

The operator set the bar (2026-10-17): **image-step performance at least 70 % of
the limiting factor**, using the B60's compute and its memory bandwidth.

**The B60 Pro (`web`, Intel datasheet / reports):** 20 Xe cores, 160 XMX
units, 2400 MHz boost, 24 GB GDDR6 on a 192-bit bus at 19 Gbps -> **456 GB/s**;
12.28 TFLOPS FP32 (the vector engines) and **197 TOPS INT8 dense (XMX)**. Xe2's
XMX does 4096 INT8 OPS/clock per Xe core (2048 FP16), so the XMX FP16 peak is
~**98.3 TFLOP/s** and INT8 ~196.6 (the advertised 197, `web`).

**The step (`arithmetic`):** Z-Image Turbo's DiT is 6 B params; a step's MACs =
params x tokens. 512² = 1,024 tokens -> **12.3 TFLOP a step**; 1024² = 4,096 ->
**49.2 TFLOP**. Against the weights (1 byte a weight + a f16 scale per 32) the
intensity is ~1,900 FLOP/byte, far above the ridge 215 FLOP/byte (98.3 TFLOP /
456 GB/s), so the step is **compute-bound**: the limiting factor is the XMX
peak, not the bandwidth (456 GB/s binds only below ~256 tokens, and the whole
6.58 GB of weights is 14 ms a read).

**The bar:** >= 70 % of that peak -- <= ~**0.18 s a step at 512²**, ~**0.72 s at
1024²** for the DiT. Measured today: OpenCL 3.69 s a step (**3.4 %** of the
98.3 TFLOP FP16 peak), Vulkan 1.78 s (**7 %**); the q8_0 GEMM that is 68 % of
the step runs at ~4 TFLOP effective. The implementation target is a q8_0 XMX
GEMM at 70 % of the XMX rate (int8 196 or fp16 98), 10-20x over the generic
kernel.

**Cheapest first implementation step** (no new kernel): the DiT weights as f16
(`--type f16` or a `--tensor-type-rules` pattern) so the GEMM takes **arcint's
existing XMX f16-weights kernel** (`kernel_mul_mm_f16w_f16`, patch 0022) instead
of the generic q8_0 one. If that nears the bar, the q8_0 XMX form is only for
the memory fit; if it does not, the q8_0 XMX GEMM is the work.

### Implementation findings (`measured-here`, 2026-10-17)

**f16 DiT weights + the XMX f16-weights GEMM (0022)**, 512²/8 steps: steady
step **1.32 s** against q8_0's 3.69, total 51.2 s against 65.1 -- **2.8x**, and
faster than Vulkan (1.78). All weights on VRAM (te 7.67 + DiT 11.75 GB as f16 =
19.6, sampler peak 21.1 GB), so 1024² would change the fit. The XMX GEMM
(`kernel_mul_mm_f16w_f16`) is 22.8 % of device time: **31 % of the 98.3 TFLOP
FP16 peak** (the generic q8_0 kernel was ~4 TFLOP).

**The fused RoPE (`0057`), f16, same window, 512²/8 steps:** a steady step of
**0.62 s** (1.61 it/s) against the generic **1.32** (2.1x), sampling total
45.03 s against 51.2 -- taken with a kernel that turned out to have a defect
(below): the fused op removed the generic RoPE and saved the time, but the
image was **all-white** and the gate is what caught it. The op
(`GGML_OP_ROPE_APPLY`) is ported to arcint's tree with an original OpenCL
kernel as
`contrib/llama.cpp/patches/0057-opencl-intel-rope-apply-fused.patch` (9 files),
the sd.cpp upstream-mode guard is lifted by
`contrib/sdcpp/upstream-rope-apply.patch` (applied by `build-sdcpp.sh`), and
both builds rebuilt green.

**The defect and the fix (`measured-here`, 2026-10-17).** The ported OpenCL
kernel offset its output rows by `row*ne0` and its elements by `i0`/`i1` added
to a `global char * dst`: on a char pointer those are **byte** offsets, so
every write landed at a quarter of the intended address and most `dst` cells
were never written. The fork's CUDA kernel offsets a `float * out` by
`row*half_dim*2`, element arithmetic (`code`, `ggml-cuda/rope.cu:12-70`); the
port lost the `*sizeof(float)`. Two arms localised it: the fused **CPU** path
was right (256²/4 CPU A/B against the generic CPU reference, cosine worst
0.999979, PSNR 45.18 dB) and a standalone CPU-vs-OpenCL op harness differed by
up to 4.7e34 on 1,023 of 1,024 cells. Fixed by adding `*sizeof(float)`; then,
same window: the **q8_0** arm PASSes (step **3.14 s**, cos worst **0.999986**,
MSE 1.434, PSNR 45.78 dB, SSIM 0.999814), the **f16** arm PASSes (step
**0.649 s**, 1.54 it/s, cos worst **0.999987**, MSE 1.276, PSNR 46.87 dB,
SSIM 0.999855). The deliberate **red case** (cos/sin swapped) FAILs the q8_0
gate curve: cos worst **0.890391**, PSNR 9.70 dB, SSIM 0.072. `measured-here`;
the timing is **CONFIRMED** in the campaign
`docs/campaigns/imggen-rope-apply-opencl.md`, which now also carries the
defect and the red case.

**Where the remaining device time is** (`measured-here`, profiling build,
the **fixed** 0057, 512²; the full 8-step run and the DiT-only per-step mix
from subtracting a 1-step run):

f16, full 8-step run, device total **8,714 ms** (was 14,053 before 0057):

| kernel | ms | share | what it is |
|---|---|---|---|
| `kernel_mul_mm_f16w_f16` | 3,213 | 36.9 % | the XMX f16 GEMM (58.9 % of the DiT-only per-step mix) |
| `mul_mat_f16_f16` + `im2col_f16` | 1,988 + 1,095 | **35.4 %** | the **VAE decode conv** via im2col + mul-mat |
| `kernel_mul_mm_f32_f32_l4_lm` | 895 | 10.3 % | upstream's generic f32 GEMM |
| group_norm / scale / soft_max / rms / rope_apply | 311 / 227 / 178 / 151 / 70 | 3.6 / 2.6 / 2.0 / 1.7 / 0.8 % | |

The old `repeat_f32`+`mul` RoPE (5,242 ms, 37 % of 14,053) is **gone**; the
fused `rope_apply_f32` is **70 ms a run (9 ms a DiT step)**.

q8_0, full 8-step run, device total **28,678 ms** (step 3.14 s):

| kernel | ms | share |
|---|---|---|
| `kernel_mul_mm_q8_0_f32_l4_lm` | 23,196 | **80.9 %** |
| `mul_mat_f16_f16` + `im2col_f16` | 3,082 | 10.7 % |
| `kernel_mul_mm_f32_f32_l4_lm` | 901 | 3.1 % |
| rest | ~1,500 | ~5 % |

**Root cause of the 37 %.** Upstream mode defines `SD_USE_UPSTREAM_GGML`, so
sd.cpp's `apply_rope` takes the generic path (`rope.hpp:1075`, lines
1103-1122): per head it repeats `x_0` and `x_1` into a `[.,2]` copy and muls
by `pe_0`/`pe_1` -- **2 repeats + 4 muls a layer a step** over 30 layers and 8
steps = 1,056 of each. The reference's fused `GGML_OP_ROPE_APPLY` (fork commit
`e05d18a1`, `ggml.c:4381`) does the same in one pass, but the fork implements
it for **CPU, CUDA and Vulkan only -- no OpenCL** (`listing the fork's
rope.cl`), so the upstream-mode OpenCL build cannot call it.

**dp4a q8_0** (`GGML_OPENCL_Q8_DENSE_DP4A=1`): no change -- the generic kernel
stayed at 68 % and the step at 3.81 s. A dead end, as the recorded expectation
said.

**Next implementation by measured share** (fixed 0057): with the RoPE gone the
shares are (1) the q8_0 generic GEMM at **80.9 % of the q8_0 run** -- the
campaign `docs/campaigns/imggen-q8_0-xmx-gemm.md` (the durable service form);
(2) the VAE decode conv at **35.4 % of the f16 run** (1,095 im2col + 1,988
mul_mat a run) off im2col + mul-mat -- a third campaign, the biggest single
non-GEMM share; (3) the f16 XMX GEMM (36.9 % of the f16 run, 58.9 % of the
DiT-only per-step mix) from 31 % toward 70 % of the peak. The elementwise
dispatch flatten (the recon's second lever) is untouched. The bar
`§Performance target` holds.

## Phases

Per the charter `§4`: 0 inventory, 1 the image gate (`§3`), 2 the gaps the
profile names (q8_0 XMX GEMM, mask-free attention tile search, VAE conv2d and
GroupNorm, CONV_3D only with Qwen-Image), 3 the service (package, unit,
`/v1/images/generations`), 4 more models, 5 release 1.1.0. The gaps become
`docs/campaigns/` documents with their own gates when the profile names them.

## Where it lives

- The recipe: `contrib/sdcpp/build-sdcpp.sh`.
- The engine: stable-diffusion.cpp `f0aac52` (reference source, not vendored).
- The kernels: `contrib/llama.cpp/patches` (0001-0057 without 0031, 0055), the
  sd.cpp integration patch `contrib/sdcpp/upstream-rope-apply.patch`, and
  `docs/campaigns/dense-prefill-attention.md` for the attention family.
