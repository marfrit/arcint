# expert-hot-set-lru — a GPU expert cache that follows the conversation

Campaign: `docs/campaigns/expert-hot-set-lru.md` (it also owns the miss split
and the bank pinning). Lever 1 of `docs/campaigns/research-reference-audit.md`.
Open, in progress. The design follows Strata alone, the reference written
for this model (`decision`, operator's architect, 2026-10-01).

## Mechanism to build

1. **Per-layer budgets, sized in bytes.** Each layer gets a slot budget fixed
   at start; a slot is sized to its layer's expert bytes, not to the largest
   expert. The budgets are drawn from one global usage profile recorded on
   **decode** routing; the total is what the card has left after weights, KV
   and activations.
2. **Start from the profile, then adapt.** At load, fill each layer's slots
   with its highest-ranked experts of the profile. While serving, keep a
   decayed usage count per `(layer, expert)`; every adapt call, within each
   layer pair the most-used missing experts (usage ≥ 2.0) with the
   least-used residents, keep the swaps whose gain exceeds 1.5, take the top
   96 by gain across layers, then decay every count ×0.7. Strata adapts every
   4 verify windows (~10–13 tokens); arcint has no Flash-Next MTP yet, so it
   adapts every 12 decode tokens until MTP lands.
3. **Never wait on a swap.** The evicted expert is marked non-resident at once
   (the CPU tier computes it meanwhile); the new one is copied from the
   pinned host bank on its own queue and becomes resident when its copy has
   landed.
4. **Misses.** A share of each layer's misses is read by the GPU directly
   from the pinned bank (not cached: the cache changes only by swaps); the
   CPU tier computes the rest, concurrently. The share is Strata's link probe
   at load, `min(0.55, max(0.05, 0.55 × link_GBps / 26))`, about 0.29 at the
   B60's 13.9 GB/s pinned link (DESIGN §2, §8.6).
5. **The bank is pinned USM-host**, 30 GiB under the dev host's 40 GiB TTM
   cap, owed in the same build. The build replaces the
   `MOE_CPU_TIER_PARTITION=lru` mode and keeps the slot pool (patches
   0005–0007, 0070); it lands as plugin patch 0076.

## Reference implementation

- **Strata** (`~/src/Strata-ref` at `c499bd1`). `src/program/generate.cpp`
  lines 4413-4482: `apply_pending` admits a newcomer when its copy's event
  has completed; `adapt` keeps `drive.d.usage` per `(layer, expert)`, picks
  candidates and victims within each layer (candidate usage at least 2.0 and
  1.5 above the victim), keeps the top `adapt_swaps` by gain across layers,
  issues `cudaMemcpyAsync` from the host blob into the victim's slot on
  `adapt_stream`, marks the victim `kNotResident` at once, and decays every
  count by 0.7. Defaults: `adapt_every = 4` windows, `adapt_swaps = 96`
  (lines 356-380). The profile fill: lines 2826-2845, from
  `tools/make_profile.py`. Slot store: `src/core/expert_cache.cpp`
  (`ExpertCache::open_sized` takes per-layer slot bytes,
  `include/strata/core/expert_cache.hpp:72`). The miss share:
  `src/core/expert_source.cpp` lines 1614-1696 and the probe at
  `generate.cpp` lines 1709-1724. The pinned arena:
  `include/strata/core/pinned.hpp` (`PinnedArena`). `code`
- **Strata's measured effect** (its paper, `docs/paper/Strata-Paper.pdf`,
  §3.4 and Lessons 4, 8, 9, 10): a profile-filled cache serves 50 % of routed
  experts from 4,500 VRAM slots; adaptive swapping raises that to about 72 %
  (0.72 at 4K context on the 12 GB card); sizing slots per layer in bytes fits
  3,556 experts instead of 2,673 and raised decode by 13 % at 1K; 55 % of
  misses over the link for i-quants; non-blocking refill 91.7 → 94.4 t/s.
  `paper`

## What arcint already has

- **The census instrument.** Patch 0044 (`MOE_OTD_ROUTING_TRACE=<path>`)
  writes one line per routed call; `tools/hot_set_census.py` converts it to the
  v1 trace (`token layer id...`, ids ascending, provenance header required) and
  to the `layer,expert,count` census, and `select` emits a v2 seed keyed by
  `layer_key`. Patch 0046 (`MOE_CPU_TIER_SEED=<path>`) loads such a seed into
  the static partition and refuses a malformed or mismatched file. `code`
- **Replay and policy tools.** `tools/expert_lru_replay.py` (per-layer and
  global LRU) and `tools/expert_policy_compare.py` replay a trace at a budget.
  Measured on the served Flash-Next census (A770, window 003, 512 decode tokens
  scored after calibrating on prefill): demand-warm per-layer LRU 55.6 % at 32
  slots per layer and 69.2 % at 64, a static census seed 21.2 % and 38.5 %. A
  seed calibrated on prefill under-predicts decode hotness about 4x.
  `measured-here`
- **Served today**: B60, Flash-Next `d48q8`, static partition: a 36 % GPU hit
  rate, 6.6 t/s decode at 20–27k tokens. `measured-here`
- **Device slots under a byte budget** for the per-expert route (patch 0070,
  `ARCINT_MOE_DEVICE_POOL_BYTES`) and a host RAM bank for the CPU tier, pageable today (patch
  0072). `code`

## Gate

The answer-level bar in `CLAUDE.md` against the static partition on the same
card and window (facts, needle, task battery; mean KL at most 0.03 nats worse;
argmax agreement down by at most 1 point), and a decode rate above the static
partition's at the same budget, with the GPU hit rate, the share of misses
read over the link and the probed link rate printed. Copied expert
bytes stay exact: a host-versus-slot digest per resident expert, with a
red-first mutation on the swap path.

Full history: `git show b0447b8:docs/design-expert-hot-set-lru.md`.
