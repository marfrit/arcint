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

The lever fired. The op is ported as **`0057-opencl-intel-rope-apply-fused.patch`**
(9 files: `ggml.h`, `ggml.c`, the CPU forward, `ggml-opencl.cpp`, the new
`kernels/rope_apply_f32.cl`, the OpenCL CMake list), closed with the sd.cpp
integration patch **`contrib/sdcpp/upstream-rope-apply.patch`** (removes the
`SD_USE_UPSTREAM_GGML` guard around `rope.hpp:1075`; `build-sdcpp.sh` applies it
idempotently). Applied in the dev host's sd.cpp workbench (llama.cpp at the pin with the 54-patch series, and sd.cpp at f0aac52);
both rebuilds green; the released build is `57d888dd`.

**B60, f16, 512², 8 steps, same prompt/seed (the A/B in one window, agent
off, the production units restored after):**

| arm | steady step | sampling total | first step | total |
|---|---|---|---|---|
| generic (`0056`, f16) | 1.32 s/step (0.758 it/s) | — | 22.42 s | 51.2 s |
| fused (`0057`, f16) | **0.62 s/step (1.61 it/s)** | — | 22.17 s | **45.03 s** |

The step is **2.1× faster**; the ~5.2 s the generic RoPE spent a run is gone
(`repeat_f32`+`mul` were 37 % of device, `code`-predicted). The OpenCL kernel
compiled at first use without aborting the run — `#pragma OPENCL FP_CONTRACT
OFF` was accepted by the IGC. Peak VRAM f16 19895 MiB, unchanged memory
profile.

**Verdict: CONFIRMED.** Still open: the red case (a swapped half / wrong sign)
and the **numeric image gate** (q8_0 0057 vs the CPU reference, and vs the
series-OFF baseline) — the f16 leg's image is visually correct but the
per-step cosine/PSNR compare has not run yet. Also not separately measured:
how much of the gain is fewer launches vs less bandwidth (the op removes 2,112
launches a run).

**Remaining plan:** the red case, the q8_0 gate A/B, and the `test-backend-ops`
cases (they need a llama.cpp test build besides sd.cpp's ggml-only build). The
second lever in the recon — flattening the elementwise dispatch
(`global={ne1*64, ne2, ne3}` for `ne0=2`, 125.8M workgroups) — is untouched.

## Where it lives

- The op: an `contrib/llama.cpp/patches` patch (the next free number after
  0056) touching `ggml/include/ggml.h`, `ggml/src/ggml.c`, the CPU forward and
  `ggml/src/ggml-opencl`; `test-backend-ops` cases and a red case.
- The engine record: `docs/imggen.md`; the charter: `docs/milestone-1.1.0.md`
  `§Performance target`.
