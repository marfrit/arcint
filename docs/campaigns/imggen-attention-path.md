# imggen-attention-path — the DiT's numeric path is not reproduced across processes

**Open** (2026-10-17, defect). One window, the same sd.cpp flags: two different
images on the B60, one passing the CPU gate and one failing it, at the same
wall step (`measured-here`).

## Charter

The Z-Image DiT computes differently across fresh processes of the same build
and the same flags. Pin it: name the trigger, force the path (or fix the
variant that diverges), then re-gate the image where the gate was measured.
Until the trajectory is controlled, a fresh process may pass or fail on its
own and every image-gate comparison carries the condition.

## Evidence (`measured-here`, 2026-10-17)

B60, 512²/8, seed 42, upstream mode + fixed 0057, the same flags (`--type f16`,
auto flash, `--backend GPUOpenCL --auto-fit off`, `build-prof` and `build-up`
both):

- four processes took one trajectory: cos worst **0.991745**, MSE 836.2, PSNR
  **21.43 dB** — FAIL;
- four processes took another: cos worst **0.999987**, PSNR **46.87 dB** —
  PASS.

The numbers are identical to six decimals **within** a trajectory, and the
**wall step is 0.649 s in both** — so it is not a speed or graph-size
difference but a numeric one the denoiser amplifies. Two of the failing and
three of the passing arms were `build-prof`-vs-`build-up` and `--log-level
warn`-vs-`info` pairs with otherwise identical flags, so neither the build nor
the log level separates them.

The profiling csvs also differ — the failing arms' DiT kernels show ~2.4x the
launch counts (`mul_mm_f16w_f16` 3665 against 1613, `soft_max_4` 693/285,
`scale_f32_4` 2108/872, `rms_norm_mul` 4240/1780) while the VAE kernels are
unchanged (n=39) — yet the wall is equal. That profile difference is either
real-but-cheaper-per-launch or a **profiling artifact**; it is not settled.

`flash attention auto: OpenCL -> off (probe: normal … flash …)` is printed in
**both** kinds of arm, so the printed decision does not track the applied path.

## Gate

A pair of auto runs against the CPU reference in one window in a **fresh**
`GGML_OPENCL_KERNEL_CACHE_DIR`, the first and the second of the pair, until the
trajectory is stable; then run the image gate with the path pinned (force flash
or fix the divergent variant).

## Current state

Opened; no verdict. The suspects to test first: the `flash_attn` auto probe's
decision actually feeding the DiT graph (`src/core/probe/flash_attn.cpp`,
`src/pipeline/diffusion_engine.cpp:1204` `flash_attention_for`, sd.cpp
`f0aac52`), a lazy-compiled flash prepass read by `ggml_backend_supports_op`
before it exists, and the on-disk `cl-program-cache`.

## Where it lives

- The engine record: `docs/imggen.md`; the affected gate `tools/imggen_gate.py`.
- `docs/campaigns/imggen-vae-conv.md` carries the same caveat for its A/B.
