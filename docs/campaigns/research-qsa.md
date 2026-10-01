# research-qsa — prior art for serving Qwen Sparse Attention (surveyed 2026-09-30)

Scope: how other runtimes serve QSA (Qwen3.8-Flash-Next's indexer plus
micro-block sparse attention), what they cache, what they measured, and the
pitfalls they report. Evidence classes: `paper` (the Qwen report), `code`
(merged or open source read through its PR or docs), and PR-reported
measurements, which are someone else's `measured` (not measured here).

## The model's own claims (`paper`: "On the Design of Qwen3.8-Next Architecture", arXiv 2608.30320)

- Indexer: MQA with 4 query heads and 1 shared key head, partial RoPE on 64 of
  128 dims, block ratio r = 4, budget K = 2048 (512 complete blocks plus the
  tail). The compression lowers the indexer cost from O(n²) to O(n²/r).
- **Where QSA pays:** kernel-level attention speedups begin at **64K
  context** and grow with length: 7.6× prefill and 4.9× decode at **1M**
  (FlashInfer baseline; prefill in 16K chunks, decode at batch 4 with 3 MTP
  steps). Below 64K the paper claims no speedup.
- Quality: QSA matches or beats full attention on short-context benchmarks.
  RULER beyond 512K goes from 90.08 to 93.00. The model was trained with QSA
  on (final CPT stage).
- The MTP module reuses QSA indices across speculative steps (relevant to
  ROMA R1).

## Implementations

| runtime | what is cached | decode attention | reported effect |
|---|---|---|---|
| vLLM (`models/qwen4_exp/nvidia/indexer_qsa`, `code`) | **two paged caches** by the same block table: raw keys (bf16) and **compressed keys** (fp8 e4m3 or bf16, RMSNormed before quantisation) | `qsa_select_paged_decode` → block indices → `expand_qsa_block_indices`; a packed `[tokens, width+1]` buffer (−1-padded token indices plus a count column) read by the sparse kernel as its loop bound | an SM90 native sparse prefill kernel (#59010); top-k routed through a shared `SparseIndexerTopk` dispatcher (#57548) |
| llama.cpp upstream (#27742 / #27739, `code`) | raw indexer keys | masked dense ("TODO: enable sparse attention") | the reference arcint's KLD captures used |
| llama.cpp #28213 (open) | — | **gather** the ~2,051 selected K/V into a compact buffer, then dense FA; decode only, gated at `n_kv ≥ 4·width` (~9k) | 2× A6000: **+6 % at 31k, +19 % at 62k, +50 % at 130k**; byte-identical on retrieval answers |
| llama.cpp #28699 (draft) | an **incremental pooled-key cache**: one f32 row per complete block per QSA layer, written via `set_rows` only for newly completed blocks | unchanged | pre-PR, block summaries were "regathered and recomputed over the whole cached context every token, in every QSA layer — **the dominant remaining decode cost at depth**"; +9.3 % at 63k, +9.4 % at 114k; greedy bit-identical |
| llama.cpp fork 2x4ever #8 | an incremental indexer cache (pool, norm and rope once per completed block; rebuilt in 8,192-block chunks on invalidation) | native indexed FA reading selected cells directly (tiles ≤ 32 queries) | +53 % decode after 65k (30.5 → 46.7 t/s), +14 % prefill at 65k, **−5 % prefill at 8k** |
| llama.cpp-lab #17 (Vulkan) | a pooled-key cache; pooling via `pool_2d` (the r strided copies were the costliest op, 14 → 3.5 ms/step) | an opt-in per-stream FA bound | 145 → 134 ms/step at 30k |
| llama.cpp fork #11 (Vulkan, prefill) | — | grouped-union sparse prefill (64 rows' selections unioned, gathered, dense FA); admission gate, crossover ~43k cells | +16.7 % at d32k, +70.9 % at d131k; decode unchanged |

Also active: sglang (context parallelism for QSA and the indexer, #39721),
sglang-jax (#1675), Megatron-LM (trainable QSA with TileLang, #7060).

## What they agree on

1. **Cache the compressed block keys; don't recompute them per token.** A
   complete block never changes, so pool, norm and rope happen once, when the
   block completes. vLLM keeps the compressed keys as their own paged cache;
   three llama.cpp lines add the same thing as a fix after measuring the
   per-token recompute as the decode bottleneck at depth.
2. **Decode attention reads only the selection** (gather or indexed kernel).
   A mask over the whole cache keeps the full-context read.
3. **Short contexts don't gain.** Gains start in the tens of thousands of
   tokens (paper: 64K at kernel level; #28213 gates below ~9k and reports
   +6 % at 31k). Some paths lose a little at 8k.
4. **Deterministic tie-breaking by (score, block id)**, because relu scores
   tie at zero. arcint's opset11 stable TopK (lower block index wins) is the
   same rule.
5. **Reported pitfalls:** cross-sequence corruption in parallel mode, M-RoPE
   ranking mismatches (#28699), subgroup-count assumptions in a Vulkan scan
   kernel (#11: waves 8–15 silently shrank the top-k to ~927), and
   rollback/state-restore coverage for the cache (#28699, 2x4ever).

## Against arcint's served QSA (measured here, 2026-09-30; `docs/campaigns/qsa.md`)

- arcint's indexer (`q4e.attention._qsa_mask_dynamic`) pools, norms and
  ropes the **whole raw-key history on every call**, the exact per-token cost
  #28699 names as dominant. It also builds a `[T, N]` mask over the full
  context. Option A's plain Variable grows by one row per token, so every
  indexer tensor changes shape every token.
- On the B60 at 20,085 tokens, QSA decode was 1.7 t/s against dense's 6.0.
  Device nodes are cheaper with QSA (attention −36 %), and the ~0.42 s/token
  excess is off the device nodes. That is consistent with the per-token
  recompute and reshape the prior art removed, but not yet attributed.
- By the paper's own curve, no decode gain is expected at 20k. The gate at
  this depth is "no loss"; gains belong at ≥ 64K.

## Implications for the next code step

- Replace the raw-key history with a **compressed-key cache**: one pooled,
  normed, roped row per completed block, plus the ≤ 3 raw keys of the
  incomplete tail. Per decode step: at most one new block row, scores over
  N/4 cached rows, top-k. This is vLLM's `compressed_key_cache` and
  llama.cpp #28699 in arcint's option-A form.
- Keep the state's shapes stable (bucketed capacity), so the plugin doesn't
  redo shape inference and allocation per token.
- Hold selection bit-identical to the step-1/2 cells (#28699 and #28213 both
  report bit- or byte-identical greedy output against the recompute path).
