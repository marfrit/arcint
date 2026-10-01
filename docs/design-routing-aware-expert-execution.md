# Routing-aware expert execution — design note

Campaign: `docs/campaigns/sub4bit-vram-kernel.md`. The cache policy is its own
note: `docs/design-expert-hot-set-lru.md`.

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

Mechanism: of each layer's missed experts, the GPU fetches a share from pinned
host memory over PCIe while the CPU tier computes the rest, so both finish
together; the share comes from a measured bandwidth comparison at startup,
fetched experts stay in the cache, and the copies are issued as DMA from the
host the moment the layer is planned, so the GPU computes its cached experts
first and the fetched ones when they land.

Reference implementations:

- **FreeToken**: `ensure_experts_hybrid` in
  `python/freetoken/moe/offload_cache.py` (line 855) fetches
  `hybrid_fetch_fraction * misses` and rewrites the overflow to slot `-1` for
  the CPU, choosing the most recently active misses first; the fraction is
  `pcie / (pcie + cpu)` from the overlapped `ft bench bw` measurement
  (`python/freetoken/moe/bench_profile.py`, `load_hybrid_fetch_fraction`), and
  `recommend` (`python/freetoken/moe/benchbw.py` lines 598-600) picks hybrid
  only when the CPU is more than 2x the link, otherwise a GPU fetch of every
  miss. The CPU executor's `decode_submit` / `decode_sync` let the GPU work
  overlap the CPU's (`python/freetoken/moe/cpu_executor.py`). `code`
- **Strata**: `--pcie-frac` is the share of each layer's distinct missed
  experts read over PCIe from the pinned arena, set at startup from a probed
  host-to-device bandwidth (`src/program/generate.cpp` lines 1709-1722; 0.55
  for native packs at 20 GB/s and above, scaled down below). Its paper (Lesson
  9) reports that issuing those reads as DMA copies from the CPU thread at plan
  time, rather than as a copy kernel, is what made them pay. `code` / `paper`

Where it lands: the per-expert dispatch's miss path in the plugin, beside the
CPU tier's job submission, fed from the pinned bank. The pinned bank's size is
bounded by the host's TTM page limit (`ttm.pages_limit`); raising it is lever 4
of `docs/campaigns/research-reference-audit.md`.

Gate: the answer-level bar in `CLAUDE.md` against the CPU-only miss path on the
same card and window, and a higher decode rate at the same budget, with the
fetched share and the hit rate printed. Fetched expert bytes stay exact (a
digest per fetched expert, red-first on a corrupted copy).

Full history: `git show b0447b8:docs/design-routing-aware-expert-execution.md`.
