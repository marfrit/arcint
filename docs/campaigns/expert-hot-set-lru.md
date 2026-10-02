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
- **State:** gate passed 2026-10-02 (history below): swaps with
  non-blocking admission, the RAM exchange, the fixed bank and the router
  lookahead, as plugin patch **0076**. Not yet built: the miss share, the
  pinned bank as the swap source with copies on their own queue (Strata's
  adapt stream), a decode-built start profile.

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

- 2026-10-01. **First build (patch 0076, not adopted): the gate FAILED.** The
  adaptive cache, Strata's swaps every 12 decode tokens and the link-share
  miss split, loaded and served but is a large loss on the B60 (`d48q8`,
  ratio 75, census128, 20,085-token needle, one fresh process per arm):

  | arm | prefill t/s | decode t/s | answer | gpu hit % | cpu-tier experts | evictions | tensor_loads | disk IO s |
  |---|---|---|---|---|---|---|---|---|
  | base (static census128) | 64.1 | 6.0 | `ORANGE-FALCON-77` | 32.2 | 131,770 | 0 | 46,816 | 43 |
  | adaptive (`MOE_CPU_TIER_ADAPTIVE=1`) | **42.9** | **0.9** | **empty** | 22.9 | 91,937 | 70,916 | 658,244 | 422 |

  - The answer-level bar fails outright (no answer at all), and prefill and
    decode both regress hard; the hit rate **drops**, not rises.
  - Mechanism, from the counters: the split's fetched share is uploaded from
    the pageable weight reader, and the swaps evict on the same pool, so the
    arm issues **14x the tensor loads** and **70,916 evictions** (10x the
    disk IO). The static arm issues none of these: it computes every miss on
    the CPU. The transient read (evicted at the next call's start) adds more
    churn.
  - **Not adopted**; patch 0076 stays out of the packaging series. The load
    itself first crashed (the static partition pins the whole pool, leaving
    the LRU nothing to evict); admitting the census unpinned and routing the
    no-destination probe calls through the LRU fixed the load but not the cost.
  - Differences from Strata's mechanism (why this is not a verdict on it):
    the census was admitted unpinned, so misses went to the LRU demand path
    and evicted; the miss share was uploaded from the pageable reader and
    cached; planning ran per layer, with rounds counted per call.

- 2026-10-02. **Second build (patch 0076, in progress): the swaps work and
  pay at decode.** Strata's adapt over the static partition, applied as
  in-place swaps of pinned slots (`LRUCache::replace_pinned`), never as
  evictions. Non-blocking admission: a background reader reads and
  transposes the newcomer, the victim keeps serving until the bytes are
  ready, then one copy is enqueued on the in-order compute queue with no
  wait (it lands after the victim's last readers and before the newcomer's
  first). Cells: `replace_pinned_*`, `pinned_residents_*`,
  `adaptive_split_upload_matches_fill_weights_memory` (red on a shifted
  slot offset). B60, `d48q8`, ratio 75 + census128, 15.4e9 device pool,
  30 GiB host bank, fresh process per arm, a 2,076-token needle then a
  99-token prompt with 500 decode tokens (`measured-here`, three windows
  agree within 0.3 t/s):

  | arm | prefill 2k t/s | decode 78 tok | decode 500 tok | decode GPU hit share | swaps applied | evictions | needle |
  |---|---|---|---|---|---|---|---|
  | static census128 | 54.9–56.1 | 8.2 | 10.7–11.0 | 35.9 % | 0 | 0 | `ORANGE-FALCON-77` |
  | adaptive (`MOE_CPU_TIER_ADAPTIVE=1`) | 53.9–56.2 | 7.3–7.6 | 11.4–11.8 | 65.9 % | ~4,550 | 0 | `ORANGE-FALCON-77` |

  - The 78-token decode is slower adaptive: the first swaps land inside it.
  - **Run without the device pool, it loses** (9.4 → 7.4 t/s): the slot pool
    sits in host memory (`host_slot_buffers=382`), so each added GPU "hit"
    is a 2.46 MB read over the link (+0.54 ms GPU wait a layer). Every
    window of this campaign sets `ARCINT_MOE_DEVICE_POOL_BYTES`.
  - **Where a decode token goes** (decode-only tier timers): static 91 ms =
    44 ms tier join + 47 ms GPU and hand-offs; adaptive 88 ms = 42 ms tier
    join + 46 ms. Inside the join, layers whose CPU expert is not in the
    host bank wait on an O_DIRECT read: 3.8 ms each against 0.5 ms (static)
    and 0.3 ms (adaptive) without. Those stalls are ~23 ms of the static
    token and ~30 ms of the adaptive one: the adaptive cache cut the tier's
    compute by 40 % and its bank misses ate most of it (each demoted expert
    was resident, so the bank fill skipped it).
  - **Bank misses are capacity-bound.** The non-resident experts are 45.5 GiB
    (384 a layer × 48 × 2.53 MiB) against the 30 GiB bank; Strata keeps
    every expert in one RAM arena. Tried and not adopted: prefetching
    demoted experts into the full bank (it evicts others: 4,054 pending
    layers against 4,151, decode 11.2 against 11.7); filling the bank evenly
    across layers (`MOE_CPU_BANK_FILL_PER_LAYER=248`: unchanged). A miss is
    disk-bound: one 2.5 MiB O_DIRECT read is 2.0 ms on the model NVMe, an
    expert's nine scattered spans 4.7 ms (synthetic). Strata stores each
    expert as one contiguous blob; an expert-major sidecar (~62 GiB) does
    not fit the NVMe's 32 GB free.
  - **Pinned bank as USM-host chunks: slower** (`MOE_CPU_BANK_PINNED=1`,
    opt-in): tier join 49.2 s against 21.9 s, decode 7.7 against 11.8. CPU
    read bandwidth of a USM-host buffer equals the heap's (30.7 vs 28.2 GB/s,
    sequential), so not an uncached mapping; cause unmeasured. It matters
    only once swaps or the miss share copy from the bank.
  - **Strata's answer for experts that exceed RAM** (`code`: `docs/DETAILS.md`
    "A RAM budget", engine 0.1.31; `src/core/expert_source.cpp:380-390`,
    `:814-930`; `src/program/generate.cpp:3001-3030`): a fixed RAM copy of
    the profile-hottest experts, changed only by exchanges; every other
    expert read through the OS page cache from a plain mapping ("retention
    is the whole game": a random-access hint cost 14×); and a router
    lookahead thread that applies layer L+1's router to layer L's MoE input
    and `madvise(MADV_WILLNEED)`s the predicted experts' pages (their effect,
    `paper`: 72 GiB of experts on a 64 GB PC, ~3 to 7–8.5 t/s). arcint's
    bank instead admits every miss by an O_DIRECT read into LRU slots. The
    two negatives above (prefetching demoted experts into the full bank,
    buffered miss reads) did not test this mechanism and are not verdicts
    on it.
  - **Built after the reference reading, and the gate PASSED** (patch 0076,
    `ov-0076d` plugin `3669c8e0`): the RAM exchange (each applied swap
    re-keys the newcomer's bank slot to the demoted expert and reads it in
    the background; the deviation: Strata copies the victim back from its
    GPU slot, this reads the file, because the slot's scales are in device
    layout), the fixed bank (`MOE_CPU_BANK_FIXED=1`: a miss reads the
    mapping and takes no slot), and the router lookahead
    (`MOE_CPU_TIER_LOOKAHEAD=<routers>`, `tools/export_router_lookahead.py`:
    layer L+1's F32 router from the GGUF applied to layer L's MoE input,
    top-10 per token, `MADV_WILLNEED` on the predicted pages the card and the
    bank do not hold). Plus the review fixes: Strata's no-plan-while-pending
    rule, a reader exception no longer ends the process, an empty plan never
    marks a slot filled. B60, `d48q8`, ratio 75 + census128, 15.4e9 device
    pool, 30 GiB bank, fresh process per arm (`measured-here`):

    | arm | prefill 20,085 | needle decode (82 tok) | 500-tok decode | needle | window-0 KL / argmax |
    |---|---|---|---|---|---|
    | static census128 | 64.0 t/s | 7.5 t/s | 10.8 t/s | `ORANGE-FALCON-77` | 0.577 / 73.2 % |
    | adaptive + fixed bank + lookahead | 65.5 t/s | **10.3 t/s** | **12.4 t/s** | `ORANGE-FALCON-77` | 0.386 / 81.5 % |

    At 2k context the same arms read 8.2 / 10.7 and 8.6 / 12.9 t/s; the
    adaptive arm with the LRU bank 7.1 / 11.6 (the bank misses and the
    early swaps cost a short answer). Zero bank-pending layers in decode;
    the tier's decode join 16.1 s against 26.6 s per 578 tokens. The static
    arm's KL moved 0.474 to 0.577 between two windows on this card (its
    run-to-run floor), so the KL difference is no finding; the bar (not
    worse by more than 0.03, argmax not down by more than 1 point) is met.
  - Open in this campaign, each against Strata's code: the exchange's
    victim copied back from its GPU slot instead of read from the file
    (`include/strata/core/expert_source.hpp:440-455`; needs the scales
    transposed back to file layout); newcomers copied from the pinned bank on
    their own queue instead of read from the file and copied on the compute
    queue (`generate.cpp:4452-4482`, `adapt_stream`); a decode-built start
    profile (`tools/make_profile.py`); the miss share (link-probed ~0.29,
    ~3 ms a token at the measured split).
