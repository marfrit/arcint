# expert-hot-set-lru — the GPU expert cache follows the conversation, and a share of the misses crosses the link

**Open.** Lever 1 of `research-reference-audit.md` §4, together with lever 6
(the miss split) and the bank pinning: in Strata they are one mechanism, so
this one campaign owns all three (`decision`, operator's architect,
2026-10-01). DESIGN §8.1, §8.4, §8.6. First in the build order: the cache,
then the doorbell, then multi-draft MTP, then prefill on the GPU.

## Charter

Flash-Next serves on the per-expert dispatch route: experts resident in the
card's slot pool run on the GPU, the rest on the CPU tier. Make the resident
set follow the conversation, serve a share of each layer's misses over the
link from a pinned host bank while the CPU computes the rest, and never let
a step wait on a copy.

## Reference to follow

**Strata alone** (`~/src/Strata-ref` at `c499bd1`, written for this model on
one GPU + RAM; `decision`, operator's architect, 2026-10-01). Paths are
`code` unless marked.
- **Budgets.** Per-layer slot budgets fixed at start; slots sized in bytes
  per layer (`include/strata/core/expert_cache.hpp:72` `open_sized`); the
  budgets drawn from one global usage profile built from **decode** routing
  (`tools/make_profile.py`; the profile fill in
  `src/program/generate.cpp:2826-2845`).
- **Swaps** (`generate.cpp:4413-4482`, `adapt`). Batched within each layer
  from decayed usage counts: candidates with usage ≥ 2.0, victims by least
  usage, a swap only when the gain exceeds 1.5; the top 96 swaps by gain
  across layers (`adapt_swaps`, `:380`); every count decayed ×0.7 per adapt
  call. Strata adapts every 4 verify windows (`adapt_every`, `:358`) of
  2.4–3.2 tokens each, so every ~10–13 tokens.
- **Non-blocking admission** (same lines, `pending`, `apply_pending`). The
  victim is evicted from the residency table at once
  (`host_res[out] = kNotResident`; the CPU computes it meanwhile), the
  newcomer's copy runs on its own stream, and it is admitted when the copy
  has landed. No step waits.
- **Miss split** (`src/core/expert_source.cpp:1614-1696`). Of each layer's
  distinct missed experts, a share is read by the GPU directly from the
  pinned arena (not cached: the cache changes only by swaps), and the CPU
  computes the rest, concurrently. The share for native/i-quant packs is set
  from a link probe at load (`generate.cpp:1709-1724`):
  `pcie_frac = min(0.55, max(0.05, 0.55 × link_GBps / 26))` (0.55 from
  20 GB/s up, 0 below 4 GB/s). How the share reaches the GPU:
  `pcie_mode` (`generate.cpp:355`; `include/strata/core/verify.hpp:150`).
- **Source memory** (`include/strata/core/pinned.hpp` `PinnedArena`,
  `src/core/pinned.cu`): every expert in one pinned RAM arena, the source of
  both the swap copies and the miss reads.
- **Slot storage** (`src/core/expert_cache.cpp`): `admit`, `slot_of`,
  `fill_slot*`, and `verify_slot` (slot bytes against the host blob).
- **Effect** (`paper` §3.4, §6 findings 4, 8, 9, 10): hit rate ≈ 0.50 from
  the profile alone, ≈ 0.72 with adaptive swaps at 4K on a 12 GB card; slots
  sized in bytes per layer +13 % decode; 55 % of misses over the link for
  i-quants on its x16 card, IQ3_XXS 56 → 65 t/s; non-blocking admission
  91.7 → 94.4 t/s. The cache flips the top-1 token at 2–5 % of positions at
  equal perplexity (the author's measurement,
  `bench/results/2026-09-27-cache-parity/README.md`).

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
- the per-token GPU hit share, the share of misses read over the link and
  the probed link rate reported beside the rate.

## Current state

- **Today** (`measured-here`, B60, `d48q8`, ratio 75 + census128, 0074, 20,085
  tokens): decode 6.5 t/s; per decode token 308 CPU-tier experts and 170 GPU
  hits, a 36 % GPU share. The CPU tier computes every miss.
- **Placement is static.** Patch 0018 ranks by `splitmix64`, or patch 0046
  pins a census seed (`MOE_CPU_TIER_SEED`). Seeds are keyed on the layer's
  `weight_0` `.bin` offset, so a re-export moves them (`code`).
- **Inputs** (`measured-here`, B60): the link moves a 2.46 MB expert at
  13.9 GB/s from pinned memory (DESIGN §2), so Strata's formula gives a
  share of about **0.29**; the tier serves ~153 µs per expert in service
  (47.2 ms of tier wait per decode token over 308 tier experts, shape-routed
  0074), against 81 µs in the 0074 microbench (651 µs per 8-expert call).
- **Instruments exist** (`code`): the routing trace (patch 0044,
  `MOE_OTD_ROUTING_TRACE`), `tools/hot_set_census.py`,
  `tools/expert_policy_compare.py`, `tools/expert_lru_replay.py`.
- **Offline replay** on a served A770 trace (`measured-here`): a per-layer
  LRU covers 55.6 % / 69.2 % of decode accesses at 32 / 64 slots a layer,
  where the census seed covers 21.2 % / 38.5 %. A seed calibrated on decode
  covers 14.3 % of decode at 5 slots a layer, one calibrated on prefill
  5.4 %: a start profile is built from the regime it will serve.
- **Admissible.** DESIGN §3.4 (amended 2026-10-01): placement may follow the
  conversation and admission may depend on timing, judged at the
  answer-level bar. The dev host's TTM pinned cap is 40 GiB.
- **Owed in the same build:**
  - the host bank pinned (patch 0072's bank is anonymous pageable memory);
  - the `tier_prefix_cache_decision` change (`src/config.cpp`), which still
    refuses an adaptive tier with the prefix cache.
  The partition layer key (DESIGN §8.10) matters only for static seeds.

## Build design (pinned 2026-10-01, before the build)

- **Budgets:** per-layer slot budgets fixed at start, slots sized in bytes
  per layer, budgets from one global decode-built usage profile.
- **Swaps:** batched within each layer from decayed usage counts (usage ≥
  2.0, gain > 1.5, the top 96 by gain across layers, decay ×0.7 per adapt
  call). arcint has no Flash-Next MTP yet, so the interval is in decode
  tokens: adapt every **12 decode tokens** until MTP lands.
- **Admission:** evict at once, admit when the copy lands; no step waits.
- **Miss split:** a share of each layer's misses read by the GPU directly
  from the pinned bank (not cached), the rest computed by the CPU tier,
  concurrently; the share is Strata's probe formula, the link probed at
  load (≈ 0.29 at 13.9 GB/s).
- **Bank:** pinned USM-host, 30 GiB under the dev host's 40 GiB TTM cap; the
  source of swaps and miss reads.
- **Replaces** the `MOE_CPU_TIER_PARTITION=lru` mode; keeps the slot pool
  infrastructure (patches 0005–0007, 0070).
- **Cells:**
  - no step blocks on an in-flight copy;
  - evicted slots are never read stale;
  - the miss share follows the probed link;
  - an admitted slot's bytes equal the bank's (red-first on a mutated copy);
  - replay determinism, kept only if it is free.
- **State:** design pinned; build not started. Lands as plugin patch
  **0076** (0075 is staged, not in the series).

## Where it lives

The plugin source is OpenVINO's GPU plugin at the pinned nightly; arcint's
changes to it are the numbered patches in
`contrib/packaging/marfrit-openvino/patches/` (documented in that
directory's `README.md`), applied to a checkout of the pinned tree when the
runtime is built. The files this campaign touches, under
`src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/` of that tree:
`moe_3gemm_swiglu_opt.cpp`, `expert_weight_providers.{hpp,cpp}`,
`lru_cache.{hpp,cpp}`, `static_partition.hpp`, `census_seed.hpp`,
`host_expert_bank.hpp`; and `src/plugins/intel_gpu/src/plugin/ops/moe_offload_constant.cpp`.
The patches that put them there:
- `0005-moe-otd-device-resident-slot-pool.patch`,
  `0006-moe-otd-async-batched-slot-upload.patch`,
  `0007-moe-otd-drop-redundant-stream-finish.patch` (slot pool, upload ring);
- `0018-moe-cpu-tier-static-partition.patch` (the partition and its LRU mode);
- `0044-moe-otd-routing-trace.patch`, `0046-moe-cpu-tier-census-seed.patch`;
- `0070-per-expert-slot-pool-in-device-memory.patch`;
- `0072-cpu-tier-host-expert-bank.patch` (the bank to pin).

Engine: `src/exec/fit.h` (`expert_slot_bytes`), `src/exec/flash_next_offload.h`,
`src/config.cpp` (`tier_prefix_cache_decision`). Acceptance record
`docs/window-052.md`; design note `docs/design-expert-hot-set-lru.md`.
`hybrid-expert-fetch` is a stub pointing here.

Full history: `git show b0447b8:docs/campaigns/expert-hot-set-lru.md`.
