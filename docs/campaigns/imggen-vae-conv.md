# imggen-vae-conv — the VAE decode conv off im2col + mul_mat

**Lever adopted in 1.1.0 (2026-10-18), the auto probe not fixed.** The
service pins `--conv-direct auto,vae=on`: at 1024² the VAE decodes in 6.33 s
against 15.41 s for im2col + mul_mat (2.43x), 1.73 against 3.69 s at 512²,
with the same image (finals 69.4 dB apart, both PASS the gate) and a peak of
13,373 against 17,211 MiB (`measured-here`, the release window, both arms in
the same, controlled trajectory: the caveat below is void, see
`imggen-attention-path.md`). Timing both forms on OpenCL as sd.cpp does on
Vulkan does not make `auto` right: the probe's small shapes favour im2col
(0.40 against 1.49 ms). Open: a probe at the VAE's own shapes, and the
direct kernel itself (with it the decode is a quarter of a 1024² image's
time; GroupNorm 1.27 s of it).

Before (2026-10-17): After the fused RoPE, the largest non-GEMM share of the
f16 run is the VAE decode conv: `mul_mat_f16_f16` 1,988 + `im2col_f16` 1,095 =
3,083 ms, **35.4 %** of the 8,714 ms run (`measured-here`).

## Charter

sd.cpp's VAE builds `ggml_conv_2d` = im2col + mul-mat. `--conv-direct` can
force `ggml_conv_2d_direct` (one `GGML_OP_CONV_2D`): OpenCL supports that op
(f16 x f16 x f16 / f16 x f32 x f32, `code`, `ggml-opencl.cpp:10832`, `:34907`)
and its upstream `kernel_conv_2d` is **2.7x faster on the device** and **1.83x
on the wall** than im2col + mul-mat on the B60, with the same final image
(`measured-here`, below). The auto probe never picks it:
`sd_probe_conv_direct` returns `sd_backend_supports_conv_implicit_gemm`, which
is CUDA/HIP/Vulkan-only (`code`, `ggml_extend_backend.cpp:572`), so OpenCL gets
"off (no implicit GEMM path)" even though the direct kernel is faster and
equivalent. Make the conv direct on the Intel tiles (pin the flag, or a probe
that times both forms as it already does for Vulkan), and hold the image gate
while the VAE time falls. One lever: the VAE decode conv.

## Reference to follow

- sd.cpp's own `ggml_conv_2d_direct` and `sd_probe_conv_direct`
  (`src/core/probe/conv_direct.cpp`, `f0aac52`): its **Vulkan** path times
  normal vs direct over two shapes and picks the faster (`code`); the
  non-Vulkan path is the wrong test above.
- ggml's Vulkan `conv2d_mm.comp` (implicit GEMM) and the kernel OpenCL actually
  runs, `conv2d.cl` / `conv2d_f16_f32.cl` (`code`).
- arcint's XMX tiles are the reuse form if the upstream conv kernel is not
  enough; the conv GEMM geometry (K = 9*C_in, M = H_out*W_out) differs from the
  LM prompt shapes.

## Gate

The comparative image gate (`tools/imggen_gate.py`, milestone §3) and the
per-run speed gate (VAE decode seconds a run at 512² / 1024², fresh process a
cell; the red case first: a wrong tile in a new conv kernel turns the decoded
image red). Because the VAE conv runs once an image, not per step, its share is
per-image.

## Current state (2026-10-17, `measured-here`)

B60, 512²/8, f16, upstream mode + fixed 0057, one window, `--conv-direct` on vs
off (the same auto/flash flags, so the two arms were in the same attention
trajectory):

| arm | VAE decode wall | conv device | final vs the CPU reference |
|---|---|---|---|
| `vae=on` (direct) | **2.54 s** | `kernel_conv_2d` **1,140 ms** | 21.4285 dB / 0.948224 |
| `vae=off` (im2col) | **4.64 s** | `mul_mat_f16_f16`+`im2col_f16` **3,081 ms** | 21.4285 dB / 0.948224 |

The conv is **2.7x** on the device and the VAE decode **1.83x** on the wall; the
two forms' final images are the same (their PSNR/SSIM against the reference are
identical to six decimals, and the finals against each other 69.4 dB). So the
direct form computes the same image here. **Caveat**: both arms ran in the
failing trajectory of the open defect `imggen-attention-path.md`, so the on-vs-
off **comparison** is valid but the run's own gate is red for that reason; the
adopted config must be re-gated once the trajectory is controlled.

## Where it lives

- The config pin goes in `contrib/sdcpp/build-sdcpp.sh`'s recipe and the unit
  (or, if the probe is changed, a small sd.cpp integration edit).
- The engine record: `docs/imggen.md`; the charter: `docs/milestone-1.1.0.md`.
- The trajectory caveat: `docs/campaigns/imggen-attention-path.md`.
