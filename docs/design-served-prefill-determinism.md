# design-served-prefill-determinism — the B60 run-to-run floor of the GDN path

Campaign: `docs/campaigns/served-prefill-determinism.md`. Upstream:
openvinotoolkit/openvino#38099 (sibling report `issuecomment-5751935449`).

## Current state

- **The floor is per card** (`measured-here`). On the B60 (Xe2, `bmg-g21`,
  `GPU.0`) two identical forwards of the served Flash-Next path differ:
  `KL(A‖B)` mean ~0.07 nats for consecutive same-content forwards at chunk 512.
  On the A770 (ACM-G10, `GPU.1`) the depth-48 served path is bit-identical
  across repeats (floor 0).
- **Location** (`measured-here`): `layer0/mixer_out`, the GDN state update,
  with all nine input ports bit-identical across 8 repeats. Fingerprint:
  `gated_delta_state_table` row 0, the same 14 of 48 heads
  `[3,5,6,7,10,13,17,22,31,39,41,42,43,47]`, one f16 ulp, a varying flip count
  (2423..3924); the conv state is stable; the first forward of a cold process
  is reproducible.
- **Excluded** (`measured-here` / `code`): warm-up, the GPU/host expert mix,
  chunking, launch geometry (static dispatch), JIT output (byte-identical),
  inter-kernel ordering (a `clFinish` after each enqueue), the reduction
  lowering (a fixed tree at both subgroup widths), the dense GEMM.
- **Open:** the within-kernel mechanism in the GDN arithmetic on Xe2. Routes:
  a standalone one-block reproducer on the B60 with the A770 as control, or
  the `ref` GDN kernel through a debug-caps plugin build.

## How it is judged now

The operator's rule of 2026-10-01 (`CLAUDE.md`) judges correctness at the
answer, comparatively on the same card and window. The B60 floor is therefore
a property of that card that both arms of a comparison carry; it blocks only a
bit-level claim on the B60. Bit-level comparisons run on the A770.

Instruments: `tools/kld_served.py` (dump replay) and `tools/kld_bar.py`
(`floor_pair` on a dump's own replays).

Full history: `git show b0447b8:docs/design-served-prefill-determinism.md`.
