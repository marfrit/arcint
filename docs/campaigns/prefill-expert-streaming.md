# prefill-expert-streaming — Flash-Next prefill computes every expert on the card, streaming the non-resident ones

**Open** (reopened 2026-10-01 against the references). Lever 2 of
`research-reference-audit.md` §4.

## Charter

At prefill every expert of a layer is computed on the GPU with batched
kernels. The experts not resident in VRAM stream from pinned host memory
into slots borrowed from the expert cache, on their own copy queue, the next
layer's arriving while the current layer computes; chunks are large enough
that every expert serves many tokens.

## Reference to follow

**Strata** (`~/src/Strata-ref`):
- **The ring** (`code`: `src/prefill/prefill.cpp:71-104`). From a chunk of
  `stream_all_min()` = 1,024 tokens on, every non-resident expert of every
  layer streams in a fixed order through a ring borrowed from the cache slots
  (384 slots when ≥ 90 % of experts sit in the pinned arena, else 96), so the
  next layer's experts arrive during the current layer's attention. Experts
  the arena could not pin are copied to pinned buffers by helper threads.
- **The kernels** (`code`: `:458-486`). llama.cpp's MMQ kernels on the
  quantised experts; the weights stay quantised.
- **The chunk** (`bench/results/2026-09-28-prefill-speed/README.md`):
  `--prefill auto` takes the largest of
  8,192 / 6,144 / 4,096 whose buffers fit in the slots it may borrow.
- **Effect** (the author's measurement,
  `bench/results/2026-09-28-prefill-speed/README.md`; RTX 5070 12 GB, 32K
  prompt): Q2_0 572 → 1,290 t/s, IQ3_S 383 → 1,208 t/s, step by step.

**FreeToken** (`~/src/FreeToken-ref/python/freetoken/`):
- **The overlap** (`code`: `layers/moe.py:388-390`; `moe/offload_cache.py:606-841`).
  A prefill MoE layer prefetches layers L and L+1 and waits on L. Two
  whole-layer buffers are borrowed from the slot cache
  (`cache_size >= 2 * num_experts`); the next layer is copied on a copy
  stream with ready and release events while the current one computes.
- **Hits stay on the device** (`code`: `_prefetch_split`, `:746`). With
  `prefill_hit_d2d`, resident experts are gathered device to device and only
  the misses cross PCIe.
- **The chunk** (`code`: `scheduler/config.py:16`): `max_extend_tokens = 8192`.

## Gate

On the served Flash-Next arm (B60, `d48q8`, ratio 75 + census128, host bank,
the 20,085-token needle prompt), against today's arm in the same window:
prefill at least 1.5× faster at the chunk that serves best, decode within the
run-to-run spread, and the answer-level bar (`CLAUDE.md`: needle answered;
window-0 KL no more than 0.03 nats above the baseline arm's; argmax down at
most 1 point).

## Current state

- **Today** (`measured-here`, B60, the arm above): prefill 63–68 t/s.
- **Where prefill time goes** (`measured-here`, B60, 4,096 tokens, chunk
  512): the CPU tier takes ~152 ms a MoE layer call (322 host experts, 3,227
  token-expert pairs), ~51 s of the ~80 s the prompt adds; the grouped native
  kernels on the card take 1.98 ms a prompt token.
- **Preconditions met.** The dev host's TTM pinned cap is 40 GiB, so the
  served bank (30 GiB) can be the pinned source. DESIGN §3.4 Amendments 1–2
  allow shape- and timing-dependent placement at the answer-level bar; gates
  are per phase (`CLAUDE.md`). The grouped native kernels and the hybrid
  prefill split exist (patches 0037/0042, 0043–0058).
- **Untested on the B60:** prefill chunks above 2,048 (`research-reference-audit.md` §2).

## Where it lives

The plugin's prefill route in `moe_3gemm_swiglu_opt.cpp` (grouped native
kernels, the hybrid prefill of 0037), the slot buffers of
`ops/moe_offload_constant.cpp`, the host bank `moe/host_expert_bank.hpp`
(0072) as the byte source.

Full history: `git show b0447b8:docs/campaigns/prefill-expert-streaming.md`.
