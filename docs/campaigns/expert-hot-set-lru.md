# expert-hot-set-lru — the GPU expert cache follows the conversation

**Open.** Lever 1 of `research-reference-audit.md` §4.

## Charter

Flash-Next serves on the per-expert dispatch route: experts resident in the
card's slot pool run on the GPU, the rest on the CPU tier. Make the resident
set follow the conversation, as both reference engines do, so most routed
experts run on the GPU.

## Reference to follow

**Strata** (`~/src/Strata-ref`, written for this model on one GPU + RAM):
- **Swap policy** (`code`: `src/program/generate.cpp:4395-4478`). A
  per-(layer, expert) usage count, decayed ×0.7 per round. Every
  `adapt_every = 4` rounds (`:358`), up to `adapt_swaps = 96` swaps (`:380`):
  candidates with usage ≥ 2.0, victims by least usage, a swap only when the
  candidate leads by > 1.5, all layers sorted together by gain.
- **Non-blocking admission** (same lines). The victim leaves the residency
  table at once (`host_res[out] = kNotResident`; the CPU computes it
  meanwhile), the newcomer's copy runs on its own stream, and it is admitted
  when the copy's event has completed (`pending`, `apply_pending`). Nobody
  waits on a copy.
- **Slot storage** (`code`: `src/core/expert_cache.cpp`): `admit`, `slot_of`,
  `fill_slot*`, and `verify_slot` (slot bytes against the host blob).
- **Start set** (`code`: `tools/make_profile.py`): every (layer, expert) pair
  ranked from routing traces (`--dump-routing`), interleaved across layers;
  `--expert-cache auto` fills the slots the card affords.
- **Effect** (`paper` §3.4, §6 Findings 4 and 10): hit rate ≈ 0.50 from the
  profile alone, ≈ 0.72 with adaptive swaps at 4K on a 12 GB card;
  non-blocking admission 91.7 → 94.4 t/s. The cache flips the top-1 token at
  2–5 % of positions at equal perplexity (the author's measurement,
  `bench/results/2026-09-27-cache-parity/README.md`).

**FreeToken** (`~/src/FreeToken-ref/python/freetoken/moe/`): one slot pool
for all layers, `id = layer * num_experts + expert`, victims by least recent
use, the slot table rewritten on the GPU (`code`: `offload_cache.py:169-184`,
`offload_kernels.py` `lru_ensure`).

## Gate

On the served Flash-Next arm (B60, `d48q8`, ratio 75, host bank, the
20,085-token needle prompt), against the static census128 arm in the same
window:
- decode faster; prefill within the run-to-run spread;
- the answer-level bar (`CLAUDE.md`): the needle answered; window-0 mean KL
  against the f32 reference no more than 0.03 nats above the static arm's;
  argmax agreement down by at most 1 point (`tools/kld_served.py --replay`;
  rows below 2,051 until the qsa T8 re-capture);
- copied-data integrity stays exact: an admitted slot's bytes equal the host
  bank's, red-first on a mutated copy;
- the per-token GPU hit share reported beside the rate.

## Current state

- **Today** (`measured-here`, B60, `d48q8`, ratio 75 + census128, 0074, 20,085
  tokens): decode 6.5 t/s; per decode token 308 CPU-tier experts and 170 GPU
  hits, a 36 % GPU share.
- **Placement is static.** Patch 0018 ranks by `splitmix64`, or patch 0046
  pins a census seed (`MOE_CPU_TIER_SEED`). Seeds are keyed on the layer's
  `weight_0` `.bin` offset, so a re-export moves them (`code`).
- **Instruments exist** (`code`): the routing trace (patch 0044,
  `MOE_OTD_ROUTING_TRACE`), `tools/hot_set_census.py`,
  `tools/expert_policy_compare.py`, `tools/expert_lru_replay.py`.
- **Offline replay** on a served A770 trace (`measured-here`): a per-layer
  LRU covers 55.6 % / 69.2 % of decode accesses at 32 / 64 slots a layer,
  where the census seed covers 21.2 % / 38.5 %. A seed calibrated on decode
  covers 14.3 % of decode at 5 slots a layer, one calibrated on prefill
  5.4 %: a start set is calibrated on the regime it will serve.
- **Preconditions met.** DESIGN §3.4 Amendments 1–2 (2026-10-01): placement
  may follow the conversation, judged at the answer-level bar. The dev
  host's TTM pinned cap is 40 GiB, so the host bank can be the pinned source
  of slot copies. `tier_prefix_cache_decision` (`src/config.cpp`) still
  refuses an adaptive tier with the prefix cache and is owed the change
  Amendment 2 allows.

## Build design (pinned 2026-10-01, before the build)

- **Pool:** one expert pool shared across all layers (FreeToken), with slots
  sized in bytes per layer (Strata finding 8, +13 % decode).
- **Swaps:** batched, from decayed usage counts (Strata
  `generate.cpp:4414-4478`: decay 0.7, margin 1.5, up to 96 swaps every 4
  rounds).
- **Admission:** evict at once and admit when the copy lands; a step never
  waits on a copy. Fetched experts stay cached.
- **Bank:** pinned USM-host, 30 GiB under the dev host's 40 GiB TTM cap, so
  swaps and fetches are DMA from it.
- **Start set:** a decode-built census with per-layer budgets.
- **Miss split (reference):** each layer's misses go partly to the CPU tier
  and partly to a DMA into the pool, concurrently, so both finish together.
  - Balance: (1 − f) · t_cpu = f · t_link, so the **fetched share is
    f = t_cpu / (t_cpu + t_link)**.
  - With the served tier rate (t_cpu ≈ 153 µs per expert in service, not the
    81 µs microbench) and the link (t_link ≈ 172 µs per 2.34 MiB expert at
    ~13 GB/s pinned): **f ≈ 0.47 fetched, 0.53 on the CPU**.
  - Re-measure at load (Strata's probe) or per step (FreeToken).
  - Anchors: FreeToken `moe/offload_kernels.py:290-410`,
    `moe/benchbw.py:538-600`; Strata `src/core/expert_source.cpp:1614-1696`,
    `src/program/generate.cpp:1709-1724`.
- **Cells:**
  - no step blocks on an in-flight copy;
  - evicted slots are never read stale;
  - the split follows the measured rates;
  - a fetched expert is a hit on its next use;
  - replay determinism, kept only if it is free.
- **State:** design pinned; build not started.

## Where it lives

Plugin: slot pool and upload ring (patches 0005–0007), the partition
(0018), routing trace (0044), census seed (0046), host bank (0072);
`moe_3gemm_swiglu_opt.cpp`, `expert_weight_providers.{hpp,cpp}`. Engine:
`src/exec/fit.h` (`expert_slot_bytes`), `src/exec/flash_next_offload.h`,
`src/config.cpp`. Acceptance record `docs/window-052.md`; design note
`docs/design-expert-hot-set-lru.md`. Related: `hybrid-expert-fetch` shares
the pool.

Full history: `git show b0447b8:docs/campaigns/expert-hot-set-lru.md`.
