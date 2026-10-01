# served-prefill-determinism — the B60 serves Flash-Next with run-to-run variance at long context

**Open (low priority, not gating).** Gates compare a candidate against a
baseline arm on the same card and window (`CLAUDE.md`, the answer-level bar),
so the B60's floor no longer blocks a reading; the A770 is bit-reproducible.

## Charter

Find the mechanism of the B60's run-to-run variance, or carry it to an
upstream fix.

## Current state

- **The floor** (`measured-here`, B60, d48, ratio 99 + tier, u8 KV, chunk
  512): two forwards of the same 2,735-token window differ by mean KL 0.136 /
  0.151 nats, argmax agreement 0.85 / 0.90. On the A770 the same served path
  is bit-identical across eight forwards.
- **Located** (`measured-here`): at `layer0/mixer_out` with bit-identical
  inputs, in the GDN state; one f16 ulp on row 0 of 14 fixed heads; the first
  forward of a process reproduces across processes. The JIT output is
  byte-identical, serialising every enqueue changes nothing, launch geometry
  is static, and the reduction is a fixed tree at both subgroup widths; `xe2`
  requires subgroup width 16. Chunking, warm-up and the GPU/host residency
  mix move nothing.
- **Upstream:** openvinotoolkit/openvino#38099, comment
  `issuecomment-5751935449` (2026-09-20); its "acm-g12" label is to be
  corrected to ACM-G10 (`docs/upstream-38099-comment.md`).
- Reproducer: `docs/handoff-served-prefill-determinism.md`.

## Gate

The mechanism named with a measurement, or an upstream fix measured: two
forwards of the same window on the B60 bit-identical, or their difference
attributed to a named kernel stage.

Full history: `git show b0447b8:docs/campaigns/served-prefill-determinism.md`.
