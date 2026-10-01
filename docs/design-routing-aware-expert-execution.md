# Routing-aware expert execution — design note

What exists was built under `docs/campaigns/sub4bit-vram-kernel.md` (closed).
The open item below, the miss split, belongs to
`docs/campaigns/expert-hot-set-lru.md`, together with the cache policy
(`docs/design-expert-hot-set-lru.md`).

## What exists

Only the routed experts are computed. Under `--offload-ratio` with
`--moe-cpu-tier --moe-per-expert-dispatch`, each MoE layer reads the router's
top-k, runs resident experts through per-expert OpenCL kernels that decode the
checkpoint's own blocks in registers (IQ3_XXS, IQ4_NL, IQ4_XS, Q8_0, IQ2_S;
patches 0038-0041, 0043, 0045, 0050, 0059-0061, 0069) and sends the rest to the
host CPU tier (patches 0011-0012, 0043, 0065-0068, 0071, 0074). The resident
slots sit in device memory under `ARCINT_MOE_DEVICE_POOL_BYTES` (0070); the
tier reads a host RAM bank filled at load (0072). With every expert resident
the route serves without the tier (0051, 0058) and decode routes on the device
without a host readback (0067). `code`

Current numbers (`measured-here`): Flash-Next `d48q8` on the B60 (static
partition, CPU tier, dispatch), 6.6 t/s decode and about 61–65 t/s prefill at
20–27k tokens, a 36 % GPU hit rate; the full-depth packed Qwen3.6-35B-A3B
all-resident on the A770, about 28 t/s decode and about 960 t/s prefill at
4,096 tokens.

## Open: how a decode step's misses are served

Mechanism (Strata alone; `decision`, operator's architect, 2026-10-01): of
each layer's distinct missed experts, a share is read by the GPU directly
from the pinned host bank while the CPU tier computes the rest,
concurrently. Those experts are not cached; the cache changes only by the
swaps of `docs/design-expert-hot-set-lru.md`. The share is set from a link
probe at load.

Reference implementation:

- **Strata** (`~/src/Strata-ref`): `src/core/expert_source.cpp` lines
  1614-1696 take the distinct experts in routing order; the resident ones and
  the last `pcie_num / 256` of the missed ones go to the GPU, the rest to the
  CPU pool. `--pcie-frac` is set at startup from a probed host-to-device
  bandwidth (`src/program/generate.cpp` lines 1709-1724): for native/i-quant
  packs `min(0.55, max(0.05, 0.55 × link_GBps / 26))`, 0.55 from 20 GB/s up,
  0 below 4 GB/s. How the share reaches the GPU is `pcie_mode`
  (`generate.cpp:355`; `include/strata/core/verify.hpp:150`). `code`
- **Effect** (its paper, Lesson 9): 55 % of misses over the link for
  i-quants on its x16 card, IQ3_XXS 56 → 65 t/s. `paper`

Here: at the B60's 13.9 GB/s pinned link (DESIGN §2) the formula gives about
0.29. Where it lands: the per-expert dispatch's miss path in the plugin,
beside the CPU tier's job submission, fed from the bank once it is pinned
USM-host (30 GiB under the dev host's 40 GiB TTM cap), in plugin patch 0076.

Gate: the answer-level bar in `CLAUDE.md` against the CPU-only miss path on the
same card and window, and a higher decode rate at the same budget, with the
link share, the probed link rate and the hit rate printed. Bytes read for a
miss stay exact (a digest per expert, red-first on a corrupted source).

Full history: `git show b0447b8:docs/design-routing-aware-expert-execution.md`.
