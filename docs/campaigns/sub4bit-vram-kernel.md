# sub4bit-vram-kernel — a per-expert GPU kernel with in-kernel decode of the checkpoint's own blocks

**Closed 2026-09-26.** `--moe-per-expert-dispatch` computes only the routed
experts: patches 0038–0040 (dispatch, `moe_expert_swiglu.cl`), 0043 (the
checkpoint's native IQ3_XXS / IQ4_NL / IQ4_XS / Q8_0 blocks through the
matcher and the CPU tier), 0045/0047 (their OpenCL decode in the per-expert
kernel), 0050/0052 (IQ2_S, packed at the GGUF's 82 B/256), 0051 (the fully
resident native route, `--offload-ratio 0 --moe-per-expert-dispatch`),
0054–0058 (decode fixes; 0058 fills the resident pool at bind), 0067 (decode's
expert pair table written on the device from `topk_id`). Qwen3.6-35B-A3B at
full depth serves fully resident on the A770: decode 19.3 → 28.1 t/s after
4,096 tokens with 0067, Prüfstand 10/10 (`measured-here`, DESIGN §7.0.2ci,
§7.0.2cs). Flash-Next serves on this route with a CPU tier. The GPU expert
cache that follows the conversation is `expert-hot-set-lru`. Open defect:
compressed q, k and v fused horizontally serve garbage, so k/v and the shared
expert stay plain (DESIGN §7.0.2ci). Design notes
`docs/design-routing-aware-expert-execution.md`, `docs/design-fit-levers.md`.
Prior art: `research-sub4bit-weights.md`.

Full history: `git show b0447b8:docs/campaigns/sub4bit-vram-kernel.md`.
