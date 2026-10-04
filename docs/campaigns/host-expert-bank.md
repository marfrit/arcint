# host-expert-bank — the CPU tier reads its experts from RAM, filled at load at bulk disk speed

**Closed 2026-10-01.** Patch 0072 (`moe/host_expert_bank.hpp`; knobs
`MOE_CPU_BANK_BYTES`, `MOE_CPU_BANK_SEED`, `MOE_CPU_BANK_FILL_PER_LAYER`,
`MOE_CPU_BANK_IO_THREADS`; the `[MOE_BANK]` exit line) fills a RAM bank by
sequential reads at load; B60 first answer 48.55 → 38.18 s at 46 GiB
(`measured-here`, DESIGN §7.0.2da, `+p23`). At decode the bank serves 297 of
the 308 CPU-tier experts a token (`measured-here`). Patch 0074 runs the tier's
decode-shaped calls (≤ 8 jobs) as an AVX2 dot in the quantised domain
(`MOE_CPU_TIER_Q8_DOT`; cell
`moe_cpu_expert_native.shape_routing_sends_large_calls_to_the_f32_path`):
B60, d48q8, 20,085 tokens, decode 5.9 → 6.5 t/s, prefill unchanged, window-0
KL +0.0183 nats (`measured-here`, shipped in `+p25`). Follow-on levers:
`tier-handoff-doorbell` (the per-layer GPU/tier hand-off),
`kquant-host-storage` (multi-token dot), `expert-hot-set-lru` (fewer
experts on the CPU: the adaptive cache, a link-probed share of the misses,
and this bank pinned).

Full history: `git show b0447b8:docs/campaigns/host-expert-bank.md`.
