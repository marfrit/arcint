# Research: FreeToken, what it does

FreeToken is an edge-MoE serving engine: CPU-resident experts, a GPU expert
cache, and a bandwidth-adaptive split of the misses between the CPU and a PCIe
fetch. Paper: Yang et al., "FreeToken: Efficient Edge-Native MoE Serving with
Bandwidth-Adaptive Execution", arXiv:2608.16157 (2026-08-17). Source:
`github.com/FlashML-org/FreeToken` (Apache-2.0), read at `505477a` in the
checkout `~/src/FreeToken-ref`. Every `file:line` below is under
`~/src/FreeToken-ref/python/freetoken/` at that commit. Where the paper and
the code disagree, the code is what FreeToken does.

Evidence class per item: `code` (read in the tree) or `paper` (the paper
only). Nothing here is measured on arcint's hardware; FreeToken's numbers are
its own.

---

## 1. The unit of residency is one expert, and only routed experts move

- **Flat expert ids, one shared pool.** `OffloadMoeCache`
  (`moe/offload_cache.py:104`) keeps one GPU slot pool for all MoE layers.
  `slot_for_id` is `[num_layers, num_experts]` and `id_of_slot` maps a slot
  back to the flat id `layer * num_experts + expert` (`moe/offload_cache.py:169-184`),
  so a slot can hold any layer's expert. Banks are `[L*E, ...]` per bank kind
  (`_BANK_SCHEMAS`, `moe/offload_cache.py:35-60`). `code`
- **LRU, on the device.** `cache_policy = "lru"` is the only policy
  (`moe/offload_cache.py:108`, `:149-151`). `ensure_experts(layer_id, expert_ids)`
  (`moe/offload_cache.py:843`) takes the ids the router chose this step, makes
  those resident and rewrites the ids to slot ids in place
  (`moe/offload_kernels.py:19-41`, `lru_ensure` from the external `flashlib`
  package, a `pyproject.toml` dependency not in the checkout); `usage` holds the
  step a slot was last used and the victim is `argmin(usage)`. `copy_missing`
  (`moe/offload_cache.py:1011`) moves only the missing rows, one fused launch
  over all banks. `code`
- **Decode stays graph-capturable.** The ensure and copy kernels are
  device-side and fixed-shape; miss statistics accumulate on the device
  (`moe/offload_cache.py:219-243`). `code`
- **Cache size is a byte budget split with the KV pool, MoE first.**
  `expert_bytes_per_slot` and `plan_cache_budget`
  (`engine/cache_budget.py:17`, `:48`) size the slot count from free memory
  after weights; `rebuild(cache_size)` (`moe/offload_cache.py:465`) resizes
  at a scheduler safe point without reloading the host banks. `code`
- **The host copy is the source of truth**; GPU memory affects speed, never
  correctness (paper §3.3). `paper`

## 2. Dequantisation happens inside the expert GEMM

`_BANK_SCHEMAS` comments (`moe/offload_cache.py:39-49`): `fp8_block`
experts are read by the grouped GEMM and dequantised in the K-loop "(no bf16
materialization)"; `q4_0` blocks are "dequantized inside the borrowed ggml MoE
kernels"; `nvfp4` rows go to "Triton inline-dequant kernels". The packed bytes
reach the kernel and are widened in registers. `code`

## 3. Misses: offload, CPU, or hybrid, chosen from a measured benchmark

- **Three decode targets** (`moe/offload_cache.py:124-134`): `gpu` streams
  misses over PCIe into the slot cache; `cpu` computes experts on a CPU
  executor; `hybrid` fetches a capped share of each step's misses over PCIe
  and the CPU computes the rest, then the partials merge. `code`
- **The hybrid split.** `ensure_experts_hybrid` (`moe/offload_cache.py:855`)
  fetches at most `hybrid_max_fetch`, or `hybrid_fetch_fraction * misses`,
  of this step's misses and rewrites the overflow to slot `-1`, which the CPU
  kernel computes (`moe/offload_cache.py:862-866`). Which misses to fetch is
  the most recently active first (`expert_recency`,
  `moe/offload_kernels.py:9-16`). `code`
- **The fraction is measured.** `ft bench bw` (`moe/benchbw.py`) measures CPU
  MoE bandwidth and PCIe gather bandwidth, alone and overlapped.
  `load_hybrid_fetch_fraction` (`moe/bench_profile.py:156`) sets the fetched
  share to `pcie_ov / (pcie_ov + cpu_ov)` from the overlapped pair, else
  `pcie / cpu`; the engine applies it for `--moe-hybrid-max-fetch -1`, the
  default (`engine/engine.py:636-663`). `code`
- **The mode is measured too.** `recommend` (`moe/benchbw.py:598-600`) picks
  `hybrid` only when CPU bandwidth exceeds 2x the PCIe gather bandwidth,
  otherwise `offload` (GPU fetch of every miss); the engine reads that
  recommendation at boot (`engine/engine.py:1455-1489`). `code`
- **The paper's crossover** (§3.2, Equation 4): with `B_P` the pinned PCIe
  bandwidth and `B_H` the host expert bandwidth, fetch `q* ≈ m · B_P / B_H`
  of `m` misses so both branches finish together, keeping at least one fill
  per step so the cache keeps warming; both bandwidths profiled on the
  deployed machine. `paper`

## 4. The CPU executor

- **One worker per physical core, pinned** (`moe/cpu_executor.py:92-142`):
  SMT siblings are skipped because the GEMV is bandwidth-bound. `code`
- **Asynchronous submit.** `decode_submit` (`moe/cpu_executor.py:543`) copies
  activations and routing to pinned host memory and rings the CPU pool, then
  returns so GPU work (the cached experts, the PCIe fetch) overlaps;
  `decode_sync` (`:589`) waits and copies the result back. With flag-sync the
  submit and the wait are stream memory operations on a doorbell in pinned
  memory (`memop_submit` / `memop_sync`), not host callbacks. `code`

## 5. Host banks: pinned, filled by O_DIRECT, pinned after the fill

`moe/host_banks.py:1-15`: a bank is a lazy anonymous `mmap`, filled by
chunked multi-threaded `O_DIRECT` `preadv` straight from disk, and only then
`cudaHostRegister`ed ("pin-after-fill"), which avoids a zero-fill pass. Only
pinned banks feed the GPU movement paths; mlocked or pageable layers decode on
the CPU executor (`moe/host_banks.py:43-50`). The FTW checkpoint format
(`checkpoint/ftw.py:1-25`) lays every tensor at a 4096-aligned offset so any
slice is one aligned direct read. `code`

## 6. Prefill: whole layers through a double buffer, copies on their own stream

- With `prefill_overlap`, two full-layer buffers are borrowed from the slot
  pool's first `2 * num_experts` slots (`moe/offload_cache.py:606-633`). The
  MoE layer prefetches layer `l` and `l + 1`, then waits on `l`
  (`layers/moe.py:384-390`); copies run on `prefill_copy_stream` ordered by a
  ready event and a release event per buffer
  (`moe/offload_cache.py:668-701`, `:818-841`). `code`
- **Hits are not re-copied.** With `prefill_hit_d2d`, experts already in the
  slot cache are gathered device-to-device into the buffer and only the
  misses cross PCIe, as one batched copy of coalesced runs
  (`moe/offload_cache.py:746-816`). `code`
- `materialize_layer` (`moe/offload_cache.py:876`) is the non-overlapped
  whole-layer path. `code`
- Prefill chunks up to `max_extend_tokens = 8192` (`scheduler/config.py:16`).
  `code`

## 7. Recurrent-state checkpoints for hybrid models

`kvcache/hybrid_radix_cache.py:1-12`: a radix prefix cache whose nodes
carry, besides KV pages, an optional GDN state snapshot at chunk-aligned
boundaries, with its own LRU eviction. A tool-call anchor freezes the state
just after the first tool-call opener token so a client-side rewrite of the
echoed call can still resume there (`core.py:55-61`). The paper calls these
"semantic anchors" (thinking blocks, tool calls, tool outputs, turns; §3.1).
`code` / `paper`

## 8. The n-gram (PLE) table of Qwen3.8-Flash-Next is disk-backed by default

- **The math.** `models/qwen4_exp/ple.py:1` implements Flash-Next's
  per-layer embedding: hashed n-gram features injected at layer 1 (HF's
  one-indexed id; `models/qwen4_exp/config.py:149-150` converts to
  zero-based). `code`
- **Row indexing.** `NGramEmbedding.row_ids` (`models/qwen4_exp/ple.py:465-478`):
  the rolling context is the last `ngram_size - 1` tokens, an eos is a hash
  boundary (`_shift_ignore_eos`, `:448-463`); per order `n`,
  `mixed = XOR_p(token[p] * layer_multipliers[p])` in wrapping int64 and
  `row[h] = mixed % head_vocab_sizes[h] + head_offsets[h]`. The three
  buffers are derived constants (`derive_ngram_hash_constants`,
  `models/qwen4_exp/ple.py:239-268`). `num_ngram_heads = (ngram_size - 1) *
  heads_per_ngram` (`models/qwen4_exp/config.py:49-51`): 16 for Flash-Next. `code`
- **The residual combination.** `PLELayer` (`models/qwen4_exp/ple.py:524`):
  `E = lookup(row_ids)`; keys and values projected from `E`, queries from the
  residual; a signed-sqrt sigmoid gate; a depthwise conv (kernel 4, dilation
  `ngram_size`); `R += D` before the attention hyper-connection mix. `code`
- **Table backends.** `PLETableBackend` (`models/qwen4_exp/ple.py:47`) is
  the lookup contract (row ids in, dequantised rows out). Implementers:
  `GpuResidentTable` (`:71`, the small-table oracle), `ZeroTable` (`:98`),
  `PinnedUVATable` (`:119`, pinned host plus a UVA gather) and
  `DiskRowTable` (`models/qwen4_exp/ple_disk.py:101`). The model attaches
  one in `models/qwen4_exp/model.py:164-203`. `code`
- **Disk is the default.** `engine/config.py:32` sets
  `ple_backend: str = "disk"`; `server/args.py:533` exposes the flag.
  `DiskRowTable` hashes the n-gram windows, batch-reads only the named rows
  from the checkpoint's shards into bounded pinned staging
  (`max_graph_rows = 256` decode rows, `max_extend_tokens = 8192` prefill
  tokens, `models/qwen4_exp/ple_disk.py:109-150`), with io_uring on by default
  (`FREETOKEN_PLE_IO_URING`, `:28`, `:139`); the captured lookup is a
  fixed-shape H2D copy plus dequant (`ple_disk.py:1`). `code`
- **Overlap.** `start_prefetch` (`models/qwen4_exp/ple.py:575-581`) hashes on
  the main stream and starts the table gather on a side stream before layer
  0; the forward joins it. `code`
- The active-expert count is read from the config
  (`num_experts_per_tok`, `models/qwen4_exp/config.py:232`). `code`

## 9. Reported results (FreeToken's own, `paper`)

- RTX 5090 (32 GB, PCIe 5.0 x16), BF16 Qwen3.6-35B-A3B: 77–83 tok/s
  (Abstract; §5.2). An RTX 4060 laptop (8 GB, PCIe 4.0 x8), NVFP4: 39.3 tok/s.
- All bandwidths are measured on the deployed tensor shapes, not taken from
  specifications; rented servers are capped to 6 threads and pinned to the
  GPU's NUMA node to match edge machines (53.8 GB/s on a 16-core desktop,
  47.5 GB/s on a 14-core laptop; §5.1).
- Its headline elsewhere is a 753B GLM-5.2 served from one workstation GPU.

---

Full history: `git show b0447b8:docs/research-freetoken.md` and
`git show b0447b8:docs/research-freetoken-code.md`.
