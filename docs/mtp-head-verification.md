# Qwen3.8-27B MTP head: verified against unsloth's GGUF and llama.cpp

The reconstructed head (`tools/export_mtp.py`) matches the MTP block of
`unsloth/Qwen3.8-27B-GGUF` (weights at cosine 0.9998–0.9999, norms exactly
`1 + raw`) and llama.cpp's `qwen35` `graph_mtp` semantics: sigmoid attention
gate, per-head q/gate interleave, embedding-then-hidden concat (`code` +
`measured-here`, 2026-09-01). Half-split and interleaved rope give draft
acceptance within single-prompt noise of each other (75.4 / 80.6 % and
76.7 / 77.4 %, code / prose, B60); the shipped head is interleaved.

Full history: `git show b0447b8:docs/mtp-head-verification.md`.
