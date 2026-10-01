# 0.4.1–0.4.3 — the GGUF path at the card's speed (tagged 2026-09-07/08)

A GGUF-opened model's projections are repacked at load into the runtime's own
compressed form by default (`--gguf-mode mixed`; `src/core/gguf_repack.*`,
DESIGN §7.0.2ba), with the native K-quant rows decoded by the tiled kernel of
plugin patches 0028–0030 (`--gguf-native`). Dense Qwen3.8-27B Q4_K_M on the
B60, u8 KV, one lane (`measured-here`, DESIGN §7.0.2bp): warm prefill 1,001 t/s
at 856 tokens and 464 t/s at 71.7k, against Intel's int4 IR at 1,609 / 552;
10/10 on the Prüfstand; the forms byte-identical to each other.

Full history: `git show b0447b8:docs/milestone-0.4.1.md`.
