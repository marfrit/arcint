# 0.5.0 — Qwen Flash Next support

Closed. Qwen3.8-Flash-Next (`qwen4_exp`, 512 experts, 10 routed + 1 shared per
token, 48 layers, n-gram table at layer 2) is allowlisted and served from the
serving-shape IR (`tools/q4e/serving_shape.py`,
`tools/export_serving_artifact.py`); the current full-depth artifact is
`qwen3.8-flash-next-d48q8` (dense projections in the checkpoint's Q8_0/Q6_K
form, experts native IQ3_XXS/IQ4_NL/IQ4_XS/Q8_0), served on the B60 with the
CPU expert tier: 6.6 t/s decode, ~61–65 t/s prefill at 20–27k tokens
(`measured-here`, `docs/campaigns/research-reference-audit.md`). Open levers
are campaigns (`docs/campaigns/README.md`), ranked against the references in
`docs/campaigns/research-reference-audit.md` §4.

Full history: `git show b0447b8:docs/milestone-0.5.0.md`.
