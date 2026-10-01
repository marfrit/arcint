# serving-shape-logits — the served Flash-Next logits are the model's

**Closed 2026-09-19.** The fill fixes (folded norm gammas and `ssm_a` in
`gguf_feed`, the sigmoid output gate, the tiled GDN key-head pairing; DESIGN
§7.0.2bz) and the native expert formats (patch 0043) put the served d48
artifact at mean KL 0.37 / median 0.18 / argmax 0.83 against the model's own
f32 forward on window 0 (`measured-here`). The f32 reference is
`tools/ref_forward_stream.py`; the served replay is `tools/kld_served.py
--replay` with `tools/kld_vs_capture.py`. Rows at or above position 2,051 of
the existing captures are void until the re-capture with the fixed BF16
indexer feed (qsa T8).

Full history: `git show b0447b8:docs/campaigns/serving-shape-logits.md`.
