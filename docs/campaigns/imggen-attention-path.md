# imggen-attention-path — the DiT's numeric path is not reproduced across processes

**Open** (2026-10-17, defect; rechecked 2026-10-18). One window, the same
sd.cpp flags: two different images on the B60, one passing the CPU gate and one
failing it, at the same wall step (`measured-here`). The fresh-process variant
did **not** reproduce in eight fresh-cache/global-cache/VAE arms; the
suspected 1024² binary split was a **measurement fault of our own** (the CLI
arms ran without a prompt) and is closed below (`measured-here`).

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

**Ruled out 2026-10-17/18 (`measured-here`, B60, 512² and 1024²/8, seed 42):**

- **Disk program cache / build / VAE conv.** Eight fresh processes at 512² -- a
pair in a fresh `GGML_OPENCL_KERNEL_CACHE_DIR` (`cache1-a`/`cache1-b`), a
fresh-cache-disabled run (`cache2-off`), two global-cache runs
(`global-a`/`global-b`), `bu-auto-g512` (`build-up`), `prof-auto-g512`
(`build-prof`) and `prof-conv-on512` -- all PASS `refcpu512` with identical
figures (cos worst 0.999987, PSNR 46.87 dB) and identical
`mul_mm_f16w_f16` launches (**1613**). The single FAIL, `prof-plain512`, ran
**3665** launches; the doubled path was not reproduced.
- **`--flash-attn` is not the separator.** At 1024², `--flash-attn off` (global)
and `--flash-attn on` (global) give **agreeing** images (cos 0.999826, PSNR
38.55) at 4.41 vs 2.21 s a step, and both sit in the same CLI cluster that
diverges from the CPU reference. `fa-on1024` logged the lazy prepass,
`fa-off1024` did not, yet their images agree. The printed `flash_attn_dpas2`
line tracked neither assignment (it also prints with `--flash-attn off`).
- **The RoPE path is not the separator.** A variant build with the sd.cpp
integration patch reverted (generic repeat+mul RoPE, `build-nopatch`) at 1024²
clusters with the fused-0057 arms (vs the f16-DiT arm cos 0.999934, vs q8_0
0.999989) and runs 4.85 vs 2.21 s a step; the fused and generic paths are
numerically the same here. (Both arms still lacked a prompt; that is the same
fault below.)
- **Sampler/request-parity.** `sd-server --log-level verbose` and `sd-cli
--log-level verbose` build the same effective params (txt_cfg 1.00, euler,
8 steps, seed 42, discrete scheduler, eta/flow_shift default); the server at
512² matches both `refcpu512` and the CLI f16 arm to cos >0.99997.

**The 1024² "split" was our own missing prompt (`measured-here`, 2026-10-18).**
Every `sd-cli` OpenCL arm and the 1024² CPU reference that night were launched
**without `-p`**, so they rendered the empty prompt, while the HTTP request
carried the prompt "a red cube on a white table, centered". The CLI and server
images accordingly differed (cos 0.913631, PSNR 9.26) -- not a numeric defect.
Re-run with the same prompt, the `sd-cli` arm is **bit-identical to the server**
(cos 1.000000, PSNR inf, SSIM 1.000000), and three separate server processes
(`served1`, `served3`, `srvcache0`/`condstatsrv3`) are identical too. This
campaign therefore has **one** open question left: the 512² fresh-process
variant, unreproduced in eight arms. The 1024² gate now needs a CPU reference
with the prompt (run 2026-10-18); the RoPE/generic arms cluster, the flash
assignments cluster, and the conditioning census (`ENCSTAT`/`CONDSTAT`) shows
the empty-prompt and prompted runs as different inputs, not different binaries.

## Where it lives

- The engine record: `docs/imggen.md`; the affected gate `tools/imggen_gate.py`.
- `docs/campaigns/imggen-vae-conv.md` carries the same caveat for its A/B.
