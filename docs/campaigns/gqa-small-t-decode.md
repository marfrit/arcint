# gqa-small-t-decode — decode and verify attention in one pass over K/V per KV head

**Open.** Started 2026-10-04.

## Charter

The libllama engine's decode attention (0007, quantized KV in 0015) gives
every (query head, query row, KV split) its own 16-lane sub-group. Each one
reads its KV slice:
- the 6 query heads of a KV head read the same K/V. L2 catches part of
  that;
- every verify row of an MTP step re-reads the whole KV. One row's pass is
  ~285 MB per attention layer at 131k (q8_0), far beyond L2.

Measured on the B60, one attention layer of the dense 27B (24 query heads on
4, head size 256), f16 K/V (`measured-here`, test-backend-ops perf,
2026-10-04):

| keys | 1 row | 6 rows |
|---|---|---|
| 8,192 | 0.17 ms | 0.67 ms |
| 32,768 | 0.63 ms | 2.75 ms |
| 131,072 | 1.91 ms | 10.3 ms |

Times the model's 16 attention layers, a 6-row verify at 131k spends ~165 ms
in attention. The served agent (131,072 tokens, q8_0 KV, MTP 5) measured
~350 ms a verify step and 7.5 t/s at 128k depth.

The lever: one pass over K/V per KV head and split, covering all of that
head's query heads and all verify rows.

## Reference to follow

NInfer (`~/src/ninfer`, `code`, read 2026-10-04):
`src/ops/attention/causal_softmax/small_t_bf16.cuh`, `small_t.cu`.
- One CTA per (KV head, split, batch): `dim3 grid(Geometry::KVHeads,
  splits, batch)` (`small_t.cu:99`).
- The CTA covers the KV head's whole GQA group (6 query heads on the 27B)
  and up to `TokenTile` = 6 query rows (`small_t_bf16.cuh:18-26`), the MTP
  verify window.
- Tensor-core MMA for QKᵀ and PV even at decode (`small_t_bf16.cuh:196-273`).
- Split-K partials merged by a log-sum-exp reducer (`small_t.cuh:156-270`).

Measured there: 54.8 t/s decode with 260,096-token prompts, dense 27B, RTX
5090 (`paper`: its model card's table, not reproduced here).

OpenVINO's `paged_attention_opt.cl` shares K/V reads across query heads in
registers (HEADS_PER_WI, `code`, cited in `llama-engine-kernel-gap.md`
lever 2).

The record has one failed form (`llama-engine-kernel-gap.md` lever 2): a KV
head's query heads in one work-group, 16-key K/V tiles staged in local
memory, a barrier every 16 keys, six sub-groups, no XMX. It measured
slower.

## Design (Xe2, sub-group 16), as built in 0016

NInfer's form was built first (v1: no local memory, no barriers, K and V
read by every sub-group). The rows then shared nothing: 8.86 ms for 6 rows
at 131k against 0015's 10.3 (`measured-here`). The cause was the per-lane V
gathers and the repeated K reads.

The shipped form (v3) stages, like the failed one above, but differs in
three ways:
- both products on XMX;
- 32-key tiles;
- only V (and a quantized K) staged.

So the earlier verdict was on a build without XMX and with 16-key barriers,
as that record had said.
- One work-group per (KV head, split). Two sub-groups (`GQA_SIDES`) per
  8-row query tile, each owning 128 of the 256 output columns. Row m is
  (query head m mod G, token m div G), G·T rows padded to 8.
- Each sub-group holds:
  - its query tile as the DPAS A operand (8 rows × 256, half);
  - S for 32 keys (C layout: lane = key);
  - O as 8 rows × 128 floats;
  - its own running max and sum.
- Per 32 keys the work-group stages V in local memory as f16, transposed
  with the key fastest, with two barriers.
- K enters QKᵀ as the B operand:
  - f16: read straight from the cache by its key's lane (16 consecutive
    values are the packed pair layout);
  - quantized: staged too, converted once per work-group. Direct quantized K
    made every sub-group convert the same blocks: served decode 10.8 against
    15.6 t/s.
- P leaves the softmax already in A layout (lane = key).
- Partials go to the existing merge kernel, same record layout.
- Routed from 4 rows. 1-3 rows stay on 0015's split kernel, which is faster
  there.

## Gate

- **Correctness:** FLASH_ATTN_EXT in full on the B60, including 0015's 90
  cases (batch 1, 3, 8; GQA 6 and 8; f16, q8_0, q4_0). A fault injected in
  the new kernel must fail them.
- **Speed (the attention op, B60, dense geometry):**
  - a 6-row call at 131k costs at most 2x a 1-row call (now 5.4x);
  - a 1-row call is no slower than 0015's, at 8k and at 131k.
- **Speed (served, the agent's configuration):**
  - decode with MTP at 128k depth above 7.5 t/s;
  - decode on the acceptance prompt within the run-to-run spread.
- **Answers:**
  - KL for 8:8 within the answer-level bar against 0015;
  - the sampled task within the noise of 0015's (mean 7.4, n=30).

## Current state (`measured-here`, B60, 2026-10-04)

One attention layer at the dense geometry, f16, test-backend-ops perf:

| keys | rows | 0015 | v1 | v2 |
|---|---|---|---|---|
| 8k | 1 | 0.165 ms | 0.24 ms | 0.57 ms |
| 8k | 6 | 0.67 ms | 0.72 ms | 0.52 ms |
| 32k | 1 | 0.63 ms | 0.87 ms | 1.69 ms |
| 32k | 6 | 2.75 ms | 2.65 ms | 1.92 ms |
| 131k | 1 | 1.91 ms | 2.56 ms | 5.39 ms |
| 131k | 6 | 10.3 ms | 8.86 ms | 5.27 ms |

- **v1** (no local memory; K and V read per sub-group, V per lane): the rows
  shared nothing. Ablation at 131k, 1 / 6 rows:
  - PV's loads cost 1.6 / 6.9 ms;
  - QKᵀ costs 1.3 / 4.4 ms.
- **v2** (K and V staged per work-group, 32 keys a tile, as 0008 does): 6
  rows now cost what 1 row costs. Ablation at 131k:
  - staging alone 2.0 / 2.2 ms;
  - the products alone 1.3 / 3.0 ms;
  - the two add up, so they don't overlap;
  - BK 16: 3.8 / 5.6 ms; BK 64: 12.0 / 6.3 ms (local memory limits the
    work-groups per core).

- **v3 (0016 as built):** K straight from the cache when f16, staged when
  quantized; V staged; two sides; BK 32. At 131k:
  - f16, 6 rows: 4.83 ms (0015: 10.3);
  - other forms: K staged for f16 5.27; one side 10.6; four sides 6.96; BK 16
    4.91; BK 64 9.67.

  q8_0, llama-bench at 32k:
  - a 6-row forward 47.7 -> 52.6 t/s with K staged, 30.7 with K direct;
  - a 3-row forward 29.2 -> 27.7.

  Routed from 4 rows (`GGML_OPENCL_FA_GQA_MIN_ROWS`).

Served (the agent's flags: 131,072 tokens, q8_0, MTP 5), a 62,597-token
prompt:
- decode at that depth 11.0 -> 15.6 t/s (+42 %);
- verify time 9.78 -> 6.65 s;
- identical draft statistics;
- prefill unchanged;
- greedy task 10/10;
- sampled 10 runs mean 7.6 (0.5.6: 7.2-7.4).

- FLASH_ATTN_EXT 2,757 of 2,758 at threshold 4, with 24 cases at 4-7 rows.
- Red (BK=24) fails all 42 cases at 4-8 rows.
- KL through the kernel (q8_0, a 6-token ubatch): 0.003589 against 0015's
  0.003593.

Gate:
- the 6-row cost is 2.5x the 1-row call at 131k (bar 2x): not met;
- 1 row is unchanged (it stays on 0015);
- served decode at depth +42 %, within the bar;
- the answers hold: greedy task 10/10, sampled within the noise, KL equal.

Open: overlap of staging and products (prefetch into registers), S computed
once per tile, an SG8 form for the A770. These are the first targets of the
code-level GA (`kernel-autotune-ga.md`).

## Where it lives

`contrib/llama.cpp` patch 0016:
- `kernels/flash_attn_f32_f16.cl` or a kernel file of its own;
- the decode route in `ggml-opencl.cpp`;
- the depth perf cases in test-backend-ops.
