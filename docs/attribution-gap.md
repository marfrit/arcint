# The prefill attribution gap (2026-08-30)

A device timeline of one served coder prefill (B60, 12,916 tokens, f16 KV,
chunk 2048) settled where prefill time goes: the card is busy 95.1 % of the
prefill span, and PERF_COUNT reports about 55 % of each kernel's device time
and none of the transfers or composite sub-kernels (`measured-here`). The
per-chunk logits copy it exposed is fixed (the logits slice cuts the paged
graph's token axis, DESIGN §7.0.2e), and decode is launch-bound in the
plugin's per-primitive host path (DESIGN §7.0.2f).

Full history: `git show b0447b8:docs/attribution-gap.md`.
