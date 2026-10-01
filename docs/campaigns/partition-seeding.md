# partition-seeding — the static partition seeded from a routing census

**Closed 2026-09-22.** Patch 0046 reads a census seed
(`MOE_CPU_TIER_SEED=<path>`, "hot-set seed v2", keyed by the layer's
`weight_0` `.bin` offset) and pins it at `bind()`; `tools/hot_set_census.py
select` emits it from a patch-0044 routing trace (`code`). Served Flash-Next
runs with census seeds (census112, census128). Placement that follows the
conversation is `expert-hot-set-lru`.

Full history: `git show b0447b8:docs/campaigns/partition-seeding.md`.
