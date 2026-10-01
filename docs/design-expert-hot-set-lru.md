# expert-hot-set-lru — a GPU expert cache that follows the conversation

Campaign: `docs/campaigns/expert-hot-set-lru.md`. Lever 1 of
`docs/campaigns/research-reference-audit.md`. Open, in progress.

## Mechanism to build

1. **One pool for all layers, sized in bytes.** The card's expert slots form a
   single pool keyed by the flat id `layer * num_experts + expert`; any slot
   may hold any layer's expert. Each slot is sized to its layer's expert bytes,
   not to the largest expert. The budget is what the card has left after
   weights, KV and activations.
2. **Start from a profile, then adapt.** At load, fill the pool with the
   highest-ranked `(layer, expert)` pairs of a routing profile recorded in the
   regime that will be served (decode-heavy traffic: a decode census). While
   serving, keep a decayed per-expert usage count; every few rounds rank the
   swap candidates across all layers by gain (a missed expert's usage minus the
   least-used resident one's) and swap a bounded batch.
3. **Never wait on a swap.** The evicted expert is marked non-resident at once
   (the CPU tier computes it meanwhile); the new one is copied from pinned host
   memory on its own queue and becomes resident when its copy has landed.
4. **Misses** go to the CPU tier or are fetched to the card by the split in
   `docs/design-routing-aware-expert-execution.md`.

## Reference implementations

- **Strata** (`~/src/Strata-ref`, written for Qwen3.8-Flash-Next on one GPU
  plus RAM). `src/program/generate.cpp` lines 4414-4484: the `adapt` step (from line 4430)
  keeps `drive.d.usage` per `(layer, expert)`, ranks swaps across all layers by
  gain (candidate usage at least 2.0 and 1.5 above the victim), keeps the top
  `adapt_swaps`, issues `cudaMemcpyAsync` from the host blob into the victim's
  slot on `adapt_stream`, marks the victim `kNotResident` at once, admits the
  new expert in `apply_pending` when the event has completed (lines 4414-4428),
  and decays every count by 0.7. Defaults: `adapt_every = 4` rounds,
  `adapt_swaps = 96` (lines 356-380). The slot store and the profile reader are
  `src/core/expert_cache.cpp` (`ExpertCache::open_sized` takes per-layer slot
  bytes); the profile is built by `tools/make_profile.py`. `code`
- **Strata's measured effect** (its paper, `docs/paper/Strata-Paper.pdf`,
  §3.4 and Lessons 4 and 8): a profile-filled cache serves 50 % of routed
  experts from 4,500 VRAM slots; adaptive swapping raises that to about 72 %
  (0.72 at 4K context on the 12 GB card); sizing slots per layer in bytes fits
  3,556 experts instead of 2,673 and raised decode by 13 % at 1K. `paper`
- **FreeToken** (`~/src/FreeToken-ref`): `python/freetoken/moe/offload_cache.py`
  keeps one LRU slot pool shared by all layers (`slot_for_id`, `id_of_slot` over
  the flat id, lines 169-184), makes the routed experts resident and rewrites
  their ids to slot ids on the GPU (`ensure_experts`, line 843;
  `moe/offload_kernels.py` lines 19-41), and copies only the missing rows
  (`copy_missing`, line 1011). Details in `docs/research-freetoken.md` §1.
  `code`

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
  slots per layer and 69.2 % at 64, a static census seed 21.2 % and 38.5 %. On
  the WP6b trace a single global LRU pool reads 93.8 % at a 16 GiB resident
  budget against 88.1 % per layer. A seed calibrated on prefill under-predicts
  decode hotness about 4x. `measured-here`
- **Served today**: B60, Flash-Next `d48q8`, static partition: a 36 % GPU hit
  rate, 6.6 t/s decode at 20–27k tokens. `measured-here`
- **Device slots under a byte budget** for the per-expert route (patch 0070,
  `ARCINT_MOE_DEVICE_POOL_BYTES`) and a host RAM bank for the CPU tier (patch
  0072). `code`

## Gate

The answer-level bar in `CLAUDE.md` against the static partition on the same
card and window (facts, needle, task battery; mean KL at most 0.03 nats worse;
argmax agreement down by at most 1 point), and a decode rate above the static
partition's at the same budget, with the GPU hit rate printed. Copied expert
bytes stay exact: a host-versus-slot digest per resident expert, with a
red-first mutation on the swap path.

Full history: `git show b0447b8:docs/design-expert-hot-set-lru.md`.
