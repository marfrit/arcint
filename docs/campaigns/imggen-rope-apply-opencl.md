# imggen-rope-apply-opencl — the fused RoPE for the DiT (the generic path's 37 %)

**Open** (2026-10-17). The f16 profile of Z-Image Turbo on the B60 puts the
generic RoPE (`repeat_f32` + `mul`) at **37 %** of device time; the fused
`GGML_OP_ROPE_APPLY` that sd.cpp uses in non-upstream mode has no OpenCL form
in the reference fork.

## Charter

sd.cpp built in **upstream mode** (`SD_USE_UPSTREAM_GGML`, the 1.1.0 route)
takes the generic RoPE path: `apply_rope` (`rope.hpp:1075`) is guarded out, and
the fallback at `rope.hpp:1103-1122` repeats `x_0`/`x_1` into a `[.,2]` copy
per head and muls by `pe_0`/`pe_1` — **2 repeats + 4 muls a layer a step**, over
Z-Image's 30 layers and 8 steps = 1,056 each. Give arcint's OpenCL backend the
fused op so the subgraph is one pass, and hold the answer-level bar while the
step time falls. One lever: the RoPE.

## Reference to follow

- **the fork's `GGML_OP_ROPE_APPLY`** (leejet/ggml, commit `e05d18a1` "add
  precomputed RoPE apply op for CPU, CUDA and Vulkan"): `ggml.h:1877`,
  `ggml.c:4381`, the CPU forward, the CUDA and Vulkan kernels. It takes the
  precomputed `pe` (`[L, d_head/2, 2, 2]`) and computes the rotation in one
  pass without materialising the repeat. **Read at source before building**
  (`leejet/ggml` (sd.cpp's ggml submodule), the fork; the fork's own
  `ggml-opencl/kernels/rope.cl` is the classic RoPE, **not** the apply op —
  `code`).
- **sd.cpp's `rope.hpp:1075` guard**: the op is called only when
  `SD_USE_UPSTREAM_GGML` is **not** defined. Since upstream ggml-org lacks the
  op, the upstream-mode build must provide it (arcint patches the tree) and
  sd.cpp's guard needs to allow the op under upstream mode when the backend
  supports it — a small integration edit to the reference source we build.
- **Alternative, cheaper first measurement**: the generic path's `repeat_f32`
  (21.7 %) and `mul` (15.6 %) as device-bandwidth ops — the work is ~7-8 MB a
  call at ~6-8 GB/s vs the 456 GB/s the card does. If tuning the upstream
  repeat/mul kernels closes most of it, the fused op is only for the launch
  count.

**Reference's measured effect here** (`measured-here`, f16 profile, 512²,
device total 14,053 ms): `repeat_f32` 3,054 ms (21.7 %) + `mul` 2,188 ms
(15.6 %) = 5,242 ms (37 %); 1,056 launches each over the run.

## Gate

The comparative image gate (`tools/imggen_gate.py`, milestone `§3`): the
candidate at most the tolerance worse per step than the **baseline arm** (the
same file, seed, steps, card, series OFF in the same window); the final
PSNR/SSIM against the baseline; the step time against the prior phase. The bar
is the milestone's performance target; the red case first (a wrong sign or a
swapped half in the rotation kernel must turn the step curve red).

## Current state (2026-10-17)

Not started: reading the fork op, the repeat/mul bandwidth probe, the OpenCL
kernel, the sd.cpp guard edit, the op in arcint's ggml core. `measured-here`:
the generic path is 37 % of the B60 f16 step; the reference is `code` and the
OpenCL form is ours (the fork has none).

## Where it lives

- The op: an `contrib/llama.cpp/patches` patch (the next free number after
  0056) touching `ggml/include/ggml.h`, `ggml/src/ggml.c`, the CPU forward and
  `ggml/src/ggml-opencl`; `test-backend-ops` cases and a red case.
- The engine record: `docs/imggen.md`; the charter: `docs/milestone-1.1.0.md`
  `§Performance target`.
