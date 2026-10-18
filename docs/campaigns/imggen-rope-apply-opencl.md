# imggen-rope-apply-opencl — the fused RoPE for the DiT (the generic path's 37 %)

**Shipped in 1.1.0 (2026-10-18).** In the release window the fused op takes
a 1024² step to 2.14 s against 4.77 s with sd.cpp's generic RoPE (2.2x, the
same window, both arms gated PASS); `test-backend-ops` carries
`test_rope_apply` (26 cases, B60 pass, a cross-check against sd.cpp's own
decomposition); the RPC op count and an `INT_MAX` bound were added
(`contrib/llama.cpp/README.md`). The "1.32 -> 0.62 s" below was the
defective kernel, and "timing CONFIRMED" a cross-window comparison; the
second lever (flattening the elementwise dispatch) is not started: after
0058 every elementwise kernel is under 4 % of a step.

**Open** (2026-10-17). The lever fired and its timing is CONFIRMED; the image
gate then caught, and the fix removed, a port defect (the OpenCL kernel's `dst`
byte offsets); the q8_0 and f16 gates now PASS and the red case FAILs. The fused
`GGML_OP_ROPE_APPLY` that sd.cpp uses in non-upstream mode had no OpenCL form
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

## Recon (2026-10-17, `code`)

- **The op to port** (the fork, `leejet/ggml` (sd.cpp's ggml submodule)):
  `ggml_rope_apply` (API `ggml.h:1877`): `a` F32 `[head_dim, heads, tokens,
  batch]`, `pe` F32 `[2, 2, head_dim/2, tokens]` (stores
  `[[cos,-sin],[sin,cos]]`), returns contiguous F32 `[head_dim, tokens,
  heads*batch]`; `interleaved` selects pairs `(2i, 2i+1)` against
  `(i, i+head_dim/2)`. The CPU forward is `ops.cpp:6289` (read; the pair
  arithmetic: non-interleaved `out[i0]=a*c - b*s`, `out[i1]=a*s + b*c`).
  The fork has CUDA (`ggml-cuda/rope.cu`) and Vulkan (`rop_apply_f32.comp`) but
  **no OpenCL**; upstream ggml-org at the pin has neither (`code`).
- **sd.cpp gate** (`rope.hpp:1075`): the op is attempted only when
  `SD_USE_UPSTREAM_GGML` is **not** defined. Upstream mode must provide the op
  and the guard has to allow it when the backend supports it — a two-line
  integration edit to the reference source we build.
- **The OpenCL kernel** (ours): per row (head, token, batch) read the strided
  F32 `x` and the `pe` pair, write `dst` contiguous; the CPU forward is the
  exact model. `supports_op` + the dispatch + `test-backend-ops` cases.
- **Alternative / second lever found while reading**: the generic path's
  `repeat_f32` and `mul` dispatches use `global = {ne1*nth, ne2, ne3}` with
  `nth=64` and a kernel that loops over `ne0`. For the RoPE operands
  (`ne0=2`, `ne1=64`, `ne2=1024`, `ne3=30`) that is **125.8M workgroups a
  launch**, ~62/64 threads idle — pure oversubscription, and it explains the
  2.8 ms a call at ~8 MB. The same geometry hits every elementwise op on
  small-`ne0` shapes (ADD, scale, cpy, ...). Flattening the elementwise
  dispatch is a second, general patch; the fused op removes the launches
  themselves, which is why it is the first lever.

## Plan (a patch of the series)

1. Port `GGML_OP_ROPE_APPLY` into arcint's tree: `ggml.h`/`ggml.c` (the fork's
   op definition), the CPU forward from `ops.cpp:6289`, the OpenCL kernel +
   `supports_op` + dispatch, and `test-backend-ops` cases.
2. Allow sd.cpp's `rope.hpp` op attempt under upstream mode when
   `ggml_backend_supports_op` holds.
3. Red case first: a swapped half or a wrong sign in the kernel turns the
   gate's step curve red; then the card A/B against the generic path.

## Current state (2026-10-17, `measured-here`)

The lever fired and its **timing is CONFIRMED**; the campaign's numeric gate
then caught a defect in the port, now fixed. The op is
**`0057-opencl-intel-rope-apply-fused.patch`** (9 files: `ggml.h`, `ggml.c`, the
CPU forward, `ggml-opencl.cpp`, the new `kernels/rope_apply_f32.cl`, the OpenCL
CMake list), with the sd.cpp integration patch
**`contrib/sdcpp/upstream-rope-apply.patch`** (removes the
`SD_USE_UPSTREAM_GGML` guard around `rope.hpp:1075`; `build-sdcpp.sh` applies it
idempotently). Applied in the dev host's sd.cpp workbench (the patched llama.cpp
tree at the pin with the series, and the sd.cpp tree); both rebuilds green.

**B60, f16, 512², 8 steps, same prompt/seed (the A/B in one window, agent
off, the production units restored after):**

| arm | steady step | sampling total | first step | total |
|---|---|---|---|---|
| generic (`0056`, f16) | 1.32 s/step (0.758 it/s) | — | 22.42 s | 51.2 s |
| fused (`0057`, f16) | **0.62 s/step (1.61 it/s)** | — | 22.17 s | **45.03 s** |

That **0.62 s** reading was taken with the defective kernel (below): the timing
saving is real, the image was not. The ~5.2 s the generic RoPE spent a run is
gone (`repeat_f32`+`mul` were 37 % of device, `code`-predicted). The OpenCL
kernel compiled at first use — `#pragma OPENCL FP_CONTRACT OFF` was accepted by
the IGC. Peak VRAM f16 19895 MiB, unchanged memory profile.

**The port's defect, found by the gate and fixed (`measured-here`,
2026-10-17).** The OpenCL kernel offset its output rows by `row*ne0` and its
elements by `i0`/`i1` added to a `global char * dst`: on a char pointer those
are **byte** offsets, so every write landed at a quarter of the intended
address and most `dst` cells were never written. The fork's CUDA kernel offsets
a `float * out` by `row*half_dim*2`, element arithmetic (`code`,
`ggml-cuda/rope.cu:12-70`); the port lost the `*sizeof(float)`. The first q8_0
gate arm and the committed f16 leg produced an **all-white image** (cosine
0.000000, PSNR ~9 dB) — the "visually correct" line above was never measured
and was wrong. Two arms localised it: the fused **CPU** path was right (256²/4
CPU A/B against the generic CPU reference, cosine worst 0.999979, PSNR
45.18 dB), and a standalone CPU-vs-OpenCL op harness differed by up to 4.7e34
on 1,023 of 1,024 cells. Fixed by adding `*sizeof(float)`; the patch
regenerated from the worktree base..HEAD and byte-matched the stored file.

**Green, fixed kernel (`measured-here`, one window, same prompt/seed):**

| arm | steady step | first step | gate vs the 512² CPU reference |
|---|---|---|---|
| q8_0, 512²/8 | **3.14 s** | 23.45 s | PASS: cos worst **0.999986**, MSE 1.434, PSNR 45.78 dB, SSIM 0.999814 |
| f16, 512²/8 | **0.649 s (1.54 it/s)** | 20.66 s | PASS: cos worst **0.999987**, MSE 1.276, PSNR 46.87 dB, SSIM 0.999855 |

The fixed f16 step is 0.649 s (from the generic 1.32, ~2.0x), a different
window from the 0.62 reading.

**Red case (`measured-here`).** A deliberately swapped cos/sin in the
interleaved branch turns the q8_0 gate curve red: cos worst **0.890391**, MSE
10,553, PSNR **9.70 dB**, SSIM 0.072 -> FAIL. Restoring the green kernel
rebuilds the same binary sha and the arm PASSes again. The port's own defect
above is red-first evidence of the same kind.

**Device time (`measured-here`, profiling build, fixed kernel, 512²).** Full
8-step run: f16 **8,714 ms** (14,053 before 0057); the fused `rope_apply_f32`
**70 ms a run (9 ms a DiT step)** against the generic `repeat_f32`+`mul`'s
5,242 ms (37 % of 14,053). q8_0 **28,678 ms**: the generic q8_0 GEMM 23,196 ms
(**80.9 %**) and the VAE conv (`mul_mat_f16_f16`+`im2col_f16`) 3,082 ms
(10.7 %); on the f16 run the VAE conv is 35.4 % and the XMX f16 GEMM 36.9 %.

**Verdict: the timing lever CONFIRMED** (the generic RoPE's 5,172 ms a run is
gone; step 1.32 -> 0.649, ~2.0x; q8_0 3.69 -> 3.14); **correctness now gated and
green** (both arms PASS, the red case FAILs). Not separately measured: how much
of the gain is fewer launches vs less bandwidth (the op removes 2,112 launches
a run).

**Remaining:** the `test-backend-ops` cases (they need a llama.cpp full test
build besides sd.cpp's ggml-only build), and the second lever in the recon —
flattening the elementwise dispatch (`global={ne1*64, ne2, ne3}` for `ne0=2`,
125.8M workgroups).

## Where it lives

- The op: an `contrib/llama.cpp/patches` patch (the next free number after
  0056) touching `ggml/include/ggml.h`, `ggml/src/ggml.c`, the CPU forward and
  `ggml/src/ggml-opencl`; `test-backend-ops` cases and a red case.
- The engine record: `docs/imggen.md`; the charter: `docs/milestone-1.1.0.md`
  `§Performance target`.
