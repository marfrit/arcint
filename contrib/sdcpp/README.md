# stable-diffusion.cpp on arcint's ggml

The image service of 1.1.0 (`docs/milestone-1.1.0.md`, `docs/imggen.md`) is
stable-diffusion.cpp's own `sd-server`, built in upstream mode against
llama.cpp at arcint's pin with `contrib/llama.cpp/patches` applied, so the
DiT and its text encoder run on the same Intel kernels as the language
services. This directory holds the recipe and the two sd.cpp patches it
applies; the package is `contrib/packaging/arcint-imggen`.

    build-sdcpp.sh <sdcpp-src> <llama-tree> <build-dir> [--server]

- `<sdcpp-src>`: stable-diffusion.cpp at `f0aac52`. A git checkout (a
  worktree too) is checked against that commit and built from a pristine
  `git archive` export of it in `<build-dir>/sdcpp-src`, so the caller's tree
  is never patched and a dirty tree cannot pass for `f0aac52`. A tree without
  git (the release tarball) is patched in place.
- `<llama-tree>`: llama.cpp at the pin `3e4dfa8` with the series applied; the
  recipe refuses a tree without arcint's kernels, without 0057 or 0058, or
  with a `GGML_MAX_NAME` other than the pin's 64.
- `--server` builds `sd-server` too. `SD_VULKAN=OFF` leaves the Vulkan
  backend out (the package is OpenCL only); `ARCINT_BUILD` is the build
  string `/props` reports.
- After the build it asserts arcint's kernels in `sd-cli` (the K-quant XMX
  GEMM, the DPAS attention, 0057's `kernel_rope_apply_f32`, 0058's
  `kernel_mul_mm_q8_0_f16`), that `GGML_MAX_NAME=160` reached the compile
  lines, and the `/props` build string in `sd-server`.

sd.cpp compiles its whole ggml with `GGML_MAX_NAME=160`, which changes
`ggml_tensor`'s layout; arcint's libllama build keeps 64. The two are
separate binaries and never share a process (charter D5).

## The patches, in the order they are applied

Each is applied unless its reverse already applies, so a re-run on a
patched tarball tree is a no-op.

**`upstream-rope-apply.patch`** (`src/model/common/rope.hpp`). sd.cpp
attempts its fused `ggml_rope_apply` only outside `SD_USE_UPSTREAM_GGML`;
upstream ggml has no such op, so upstream mode always took the generic path
(two repeats and four muls a layer a step, 37 % of the 512² f16 run's device
time, `measured-here`). Since 0057 the op exists in arcint's tree, and the
patch lifts the guard. sd.cpp still returns the fused node only when the
backend's `supports_op` takes it (`rope.hpp:1076-1083`, `code`): OpenCL with
0057 and the CPU do, Vulkan does not and keeps the generic path.

**`server-health-props.patch`** (`examples/server/`). The two probes the
language units answer, which the 1.1.0 charter asks of the image service too
(A4): `GET /health` (`{"status":"ok"}` once the server listens; sd-server
loads the weights at the first request unless `--eager-load` is given, so a
first request carries the load, ~30 s from the ZFS mirror on the dev host)
and `GET /props`: the
diffusion model, the text encoder, the VAE (file names), the tensor-type
rules, the backend assignment and its devices (ggml's device name and
description, i.e. the card), the flash-attention and direct-conv modes, the
default width, height, seed and sample parameters (steps, sampler,
scheduler, guidance) and the build string (sd.cpp's commit, the llama.cpp pin
and the patch count; sd.cpp's own version fields read "unknown" in a
tarball build, so they are not reported).
`/v1/models`, `/v1/images/generations` and the rest of sd-server are
unchanged.

## `--conv-direct auto` on OpenCL: why the unit pins `vae=on`

`--conv-direct auto` decides per backend whether the VAE uses
`ggml_conv_2d_direct` (one `CONV_2D`) or im2col + mul_mat. sd.cpp times both
forms on Vulkan and picks the faster; every other backend is asked whether
it has an implicit-GEMM conv (CUDA, HIP, Vulkan, MUSA) and otherwise gets
"off" (`code`, `src/core/probe/conv_direct.cpp`,
`ggml_extend_backend.cpp:572`). OpenCL gets "off".

That is the wrong answer on the B60 (`measured-here`, 1.1.0 pre-tag,
Z-Image Turbo, 8 steps, seed 42): ggml-opencl's direct `kernel_conv_2d`
decodes the 1024² VAE in **6.33 s against 15.41 s** for im2col + mul_mat
(512²: 1.73 against 3.69 s), with the same image (the two finals 69.4 dB
apart, both passing the gate) and 13,373 against 17,211 MiB of peak VRAM.

Timing the forms on OpenCL as sd.cpp does on Vulkan was tried and does not
help: the probe's shapes (32x32x64 and 16x16x128 channels) favour im2col,
0.40 against 1.49 ms, so the timed probe also chose "off" (`measured-here`,
the same window). Making `auto` right would need a probe at the VAE's own
shapes (up to 512 channels at 128x128); that is not built. The unit
therefore pins `--conv-direct auto,vae=on`: auto for the DiT, the direct
conv for the VAE.
