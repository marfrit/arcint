# 0.4.0 — open and run GGUF models (stage 1 tagged 2026-09-06)

`--gguf FILE --model DIR` opens a GGUF of an allowlisted architecture in
process on the served IR of the same architecture as its topology template;
the file's Q4_K / Q5_K / Q6_K / Q8_0 projections replace the template's
(`src/core/gguf.*`, `src/core/gguf_dequant.*`, plugin patch 0021). The dense
Qwen3.8-27B Q4_K_M scores 10/10 on the Prüfstand through it (B60,
`measured-here`, DESIGN §7.0.2ay). The design and the current form (repack at
load, 0.4.1) are in `docs/design-gguf-native.md`. Owed: `--gguf` for MoE files
(stage 2) and for the sub-4-bit types (stage 3).

Full history: `git show b0447b8:docs/milestone-0.4.0.md`.
