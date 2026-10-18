# imggen-q8_0-xmx-gemm — a q8_0 GEMM on the XMX units for the DiT

**Shipped in 1.1.0 (2026-10-18); open for its second lever.** Patch `0058`:
the 1024² image step 10.95 -> **2.14 s** (5.1x), 512² 3.03 -> 0.476 s
(6.4x), the gate PASSes against the baseline arm, the red case FAILs
(`measured-here`, the release window, `contrib/llama.cpp/README.md`). The
service runs the DiT as q8_0 on it. **Next lever (B2):** the kernel is 69 %
of a 1024² step at ≈ 33.5 TFLOP/s, 34 % of the XMX f16 peak against the
operator's 70 % bar; the tile shape was searched for the language models'
prompts, not for M ≈ 4,096 image tokens times K = 3,840/10,240: a tile
search with the GA's genome and seconds a step as the fitness
(`docs/campaigns/kernel-autotune-ga.md`).

## Charter

The DiT and the Qwen3 text encoder run their large GEMMs in q8_0 (the
Z-Image-Turbo and Qwen3-4B Q8_0 files). On the B60 that landed on upstream's
Adreno-shaped f16 dot `kernel_mul_mm_q8_0_f32_l4_lm` — no XMX path for q8_0
(the milestone `§0.3` gap). Build the q8_0 GEMM on the XMX/DPAS units so the
largest kernel of the image step is not the generic one, and hold the
answer-level bar (the image gate) while the step time falls.

One lever, one campaign: a q8_0 GEMM. The rest of the profile (repeat, mul,
im2col, group_norm, ~5 % together) is other campaigns if priced.

## Reference to follow

- **arcint's own K-quant XMX GEMM**, `kernel_mul_mm_kq_f16` (`contrib/llama.cpp/patches`
  0001, 0009, 0011-0014): a work-group stages a tile of weight rows in local
  memory, dequantizes one 256-weight super-block at a time to f16, then runs
  fp16 DPAS (k16); the activation operand is f16 columns with a per-column f32
  scale. `kernel_mul_mm_f16w_f16` (0022) is the operand form to reuse. The
  q8_0 shape: decode the int8 code + f16 per-32 scale to f16, then that path.
  **Measured effect**: 0001 took the dense 27B decode 10.3 -> 18.0 t/s on the
  B60 (`measured-here`).
- **Vulkan's q8_0 path on Xe2** (`code`): `mul_mm.comp` binds `BUF_Q8_0_P16`
  and `dequant_funcs.glsl`'s `DATA_A_Q8_0` -- dequantize-then-GEMM, the same
  shape. The B3 yardstick: sd.cpp Vulkan on the same B60, 1.78 s a step against
  OpenCL's 3.69 (`measured-here`).
- **upstream's dp4a q8_0 dense GEMM** (`GGML_OPENCL_Q8_DENSE_DP4A`): measured,
  changed nothing; a dead end as the recorded expectation said
  (`measured-here`, 2026-10-17).

## Gate

The comparative image gate (`tools/imggen_gate.py`, `docs/milestone-1.1.0.md`
`§3`): candidate at most the tolerance worse per step than the **baseline arm**
(the same file/seed/steps/card with the mechanism OFF, same window), final
PSNR/SSIM against it; step time against the baseline's. **Red case first**: a
deliberately broken tile turns the step curve red. **The operator's bar**:
>= 70 % of the B60's XMX peak (~98.3 TFLOP FP16 / 196.6 INT8; ~0.18 s a step
at 512², ~0.72 at 1024²).

## What exists (`measured-here`, 2026-10-18, before the release review)

The release review corrected this section (`docs/imggen.md`, "1.1.0 as
released"): the 512² XMX step is 0.476 s (the log's 2.11 it/s), the red case
now zeroes the last 16 weights of a row (the one below garbled every row's
stride), the branch reads only SoA tensors through `view_src`, and the
17,137 MiB peak is the im2col VAE's.

Patch `0058-opencl-intel-q8_0-xmx-gemm.patch`: `deq_q8_0` + `kernel_mul_mm_q8_0_f16`
on the f16w tile/`PLAIN_ARGS`, the two tree q8_0 buffers read directly, one
dispatch branch beside the f16w one; `GGML_OPENCL_Q8_XMX` (off switch) and
`GGML_OPENCL_Q8_XMX_RED` (red). **The first build was wrong**: the deq wrote
`put16(wt, rr, ib*QK_K + c*16, ...)` (a global K) instead of `c*16` into the
per-super-block tile, so the normal arm FAILed the 512² gate (cos worst 0.83);
fixed, then red FAILs and normal PASSes.

| arm | resolution | steady step | gate vs the CPU reference |
|---|---|---|---|
| q8_0 XMX | 512² | 2.92 s | PASS cos worst 0.999989 |
| generic q8_0 (`Q8_XMX=0`) | 512² | 3.02 s | PASS (same window) |
| q8_0 XMX | 1024² | **2.13 s** | PASS cos worst 0.999902, PSNR 43.03 dB, SSIM 0.999333 |
| generic q8_0 (`Q8_XMX=0`) | 1024² | **10.95 s** | PASS cos worst 0.999898, PSNR 42.75 dB |
| q8_0 XMX red | 512² | — | FAIL cos worst 0.829442, PSNR 7.21 dB, SSIM 0.0435 |

1024² peak `drm-resident-vram0` 17,137 MiB of 24,480 (the q8 footprint). The
2.13 s is against the f16 XMX route's 2.21 s, so the durable form carries q8
memory at f16 step speed (`measured-here`, `docs/imggen.md`).

## Where it lives

- `contrib/llama.cpp/patches/0058-opencl-intel-q8_0-xmx-gemm.patch`, README
  section and red case; the campaign's full recon/current-state history is in
  git before 2026-10-18.
- Switches in `docs/switches.md`; the engine record `docs/imggen.md`; the
  charter `docs/milestone-1.1.0.md`.
