# hybrid-expert-fetch — the GPU takes a measured share of each decode step's CPU-tier experts over PCIe

**Open** (reopened 2026-10-01 against the references). Lever 6 of
`research-reference-audit.md` §4.

## Charter

At each decode step, a share of a layer's missed experts is copied from
pinned host memory into GPU slots and computed there, while the CPU tier
computes the rest; the share is set from the measured CPU and link speeds,
and fetched experts stay cached for later steps.

## Reference to follow

**FreeToken** (`~/src/FreeToken-ref/python/freetoken/`):
- **The hybrid decode** (`code`: `moe/offload_cache.py:121-143`,
  `decode_target = "hybrid"`). Each layer fetches a capped subset of its
  misses over PCIe into the shared slot cache; the GPU computes those plus
  the hits; the CPU computes the overflow; the partials merge.
- **Which misses** (`code`: `moe/offload_kernels.py`
  `_ensure_experts_hybrid_kernel`, `:291`). The most recently active misses
  are fetched first, victims are the least recently used slots, and the
  fetched experts stay in the pool.
- **How many** (`code`: `moe/bench_profile.py:156-191`
  `load_hybrid_fetch_fraction`). The fraction is `pcie / (pcie + cpu)` from
  bandwidths measured with both running at once, so the copy and the CPU
  finish together; wired in `engine/engine.py:623-661`.
- **Which mode** (`code`: `moe/benchbw.py:598-600` `recommend`). Hybrid when
  the CPU MoE bandwidth exceeds 2× the PCIe gather bandwidth, otherwise every
  miss goes to the GPU ("offload").
- **Source memory** (`code`: `moe/host_banks.py`). Only pinned banks feed the
  GPU movement paths, pinned after fill.

**Strata** (`~/src/Strata-ref`):
- **The split** (`code`: `src/core/expert_source.cpp:1614-1631`). Distinct
  experts in routing order; the resident ones and the last `pcie_num/256` of
  the missed ones go to the GPU.
- **The copy** (`code`: `include/strata/core/verify.hpp:150`,
  `src/core/verify.cpp:1240`). DMA into staging slots on the copy engine,
  issued when the CPU plans the layer, from the pinned arena
  (`src/core/pinned.cu`).
- **Effect** (`paper` §6 Finding 9): 55 % of misses over PCIe for i-quant
  experts (CPU arithmetic-bound), 20 % for Q2_0 (RAM-bound).

## Gate

On the served Flash-Next arm (B60, `d48q8`, ratio 75 + census128, host bank,
the 20,085-token needle prompt), against the bank-only arm in the same
window: decode faster, prefill within the run-to-run spread, and the
answer-level bar (`CLAUDE.md`: needle answered; window-0 KL no more than
0.03 nats above the baseline arm's; argmax down at most 1 point). Report the
fetched share and the per-token GPU hit share.

## Current state

- **Inputs** (`measured-here`, B60): the link moves a 2.46 MB expert at
  13.9 GB/s from pinned memory (~0.19 ms); the CPU tier computes a
  decode-shaped 8-expert layer call in ~651 µs with 0074 (microbench) and
  serves 308 experts a decode token in the served arm.
- **Preconditions met.** The dev host's TTM pinned cap is 40 GiB, so the
  served bank (30 GiB) can be pinned whole. DESIGN §3.4 Amendments 1–2
  (2026-10-01) allow a history- and timing-dependent choice of device,
  judged at the answer-level bar.
- **To build:** the bank as a pinned (usm_host) source; the cap-limited
  fetch into the shared slot pool of `expert-hot-set-lru`; the copy issued
  when the layer is planned, overlapped with the CPU tier; the bandwidth
  bench that sets the fraction and picks the mode.

## Where it lives

`moe_3gemm_swiglu_opt.cpp` (`dispatch_cpu_tier`, `cpu_tier_join`), the slot
buffers in `ops/moe_offload_constant.cpp`, the host bank
`moe/host_expert_bank.hpp` (patch 0072).

Full history: `git show b0447b8:docs/campaigns/hybrid-expert-fetch.md`.
