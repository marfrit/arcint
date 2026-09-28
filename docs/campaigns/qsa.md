# qsa — Qwen Sparse Attention served: the indexer's per-query key selection in Flash-Next's 12 full-attention layers

Charter: 0.5.4 LYON L2. The operator decided on 2026-09-27 that QSA is
required: it is one of the Qwen architecture's key points, and arcint has
served its 12 full-attention layers dense so far.

## The defect, as measured

- **What arcint drops.** The exporter leaves the indexer out
  (`tools/q4e/attention.py`, `code`), so every full-attention layer attends
  to every key. That is exact up to 2,051 tokens and not the model beyond:
  the pin's indexer prunes `max(0, T - 2051)` rows (`measured-here`,
  `test_attention_piece.py` cell 3: 1/2052, 29/2080 rows).
- **What it costs.** None of QSA's long-context saving: at 32k context a
  decode token reads all 32k keys per attention layer instead of 2,051.
- **A correction to the record.** The "QSA price" 2.3856e-02 (window-050,
  `attention.py`) is a **max-abs difference of one attention block's
  output** at T=2080. windows 051 and 054 add it to a KL bar in nats
  (3.0905e-03 + 2.3856e-02). Those units do not mix, so the "above 2,051"
  bar built that way is not a KL quantity (`code`, the cited definitions).
  The bar is withdrawn here; the served KL above 2,051 stays owed as a
  measurement.

## Known against hypothesised

**Known** (`code`: the transformers pin `modeling_qwen4_exp.py` 673-779,
sha256 `ca9f00bb…`; llama.cpp `qwen4exp.cpp` 525-750 at the KLD reference's
commit):
- **Geometry.** The indexer: 4 query heads x 128 and 1 raw key per token,
  from one fused projection (GGUF `indexer.q_proj` BF16 [512, 2560],
  `indexer.k_proj` BF16 [128, 2560], two RMSNorm gammas). The attention
  layers are blk 3, 7, …, 47 (`compress_ratios`).
- **Query and raw key.** The query is normed and roped at its own position
  (the main attention's partial rope, 64 of the dims). The key is cached
  **raw**.
- **Blocks.** The raw keys are mean-pooled over aligned 4-token groups (f32),
  normed, and roped at the group's first position.
- **Scores.** `relu(q_h · k_b)` summed over the 4 heads, divided by √128.
- **Selection.** Per query token and per layer, shared by all 24 attention
  heads: the top `min(512, complete blocks)` blocks (2,048 tokens) plus the
  0–3-token incomplete tail. No sink, no local window.
- **Consumption.** The selection is an additive mask on the causal one, and
  attention is otherwise ordinary. llama.cpp does the same: masked dense,
  "TODO: enable sparse attention", so no speed win there.
- **The served path cannot take a mask today.** `SDPAToPagedAttention`
  matches the SDPA mask as `any_input()` and paged attention derives
  visibility from `past_lens`. A mask emitted into the stateful SDPA would
  be dropped silently (`code`, `state_management_pattern.cpp`). No existing
  plugin op or kernel takes a per-query selection over past keys: `qq_bias`
  covers only new tokens, sliding windows are contiguous, and XAttention is
  128/256-token blocks, prefill-only and xe2-only.

**Measured here (step 1, below):**
- **Ties.** relu makes exactly-zero block scores common. When the zeros
  straddle the top-k cut, the pin keeps some of them in an order torch.topk
  does not specify. arcint's rule is fixed instead: among equal scores the
  lower block index is kept (OpenVINO TopK `stable`).

## Gate

The campaign's gate, from ROADMAP LYON L2:
- On the A770, the served KL (window 0 and a long-context window) above
  2,051 is at or below the below-2,051 regime.
- 32k prefill t/s is measured against row 3c's bar.
- Decode reads 2,051 keys per attention layer at long context, shown by the
  kernel's own counters or its device time.

## Entry criteria

Met: the reference read from source, the GGUF's indexer tensors located
(48 = 12 layers x 4), the plugin's attention paths read.

## Scope — in / out

In:
- the indexer in the exported graph;
- its compressed-key state across decode;
- the selection carried into the served attention;
- a plugin input that restricts paged attention to the selected keys
  (decode and prefill).

Out: MTP's own indexer (the checkpoint's MTP layer carries one; no MTP head
is exported yet).

## Where it lives

`tools/q4e/attention.py` (`_qsa_additive_mask`,
`build_qsa_attention_model`), `tools/q4e/serving_shape.py`
(`emit_stateful_attention`), `tools/q4e/gguf_feed.py` (the indexer
tensors), the plugin's `SDPAToPagedAttention` and `paged_attention_opt`,
and `src/core/artifact.cpp` (counts the QSA layers as served dense today).

## Pipeline for this campaign

1. Static-T equality against the pin, device-free (done, below).
2. Stateful, un-paged: the indexer's compressed-key state across prefill
   chunks and decode; parity against the pin with its cache across the
   2,051 boundary, depth 1, CPU.
3. Served: a new paged-attention input carrying each layer's selected
   positions, honoured by the decode and prefill kernels (a plugin patch);
   byte-exact against the un-paged stateful graph at the A770 floor.
4. The gate's window.

## Invariants

DESIGN §3.4: the selection is a pure function of the tokens (ties included,
by the fixed rule). Below 2,051 tokens every row stays dense and the served
answer must not move.

## Status

- 2026-09-28. **Step 1 done** (`measured-here`, CPU plugin, real blk.3
  tensors, `test_attention_piece.py`
  `test_qsa_attention_piece_selects_what_the_pin_indexer_selects`).
  `build_qsa_attention_model` emits the indexer statically: one pooled key
  per block for all rows, TopK with k = min(512, blocks) over scores masked
  to each row's complete blocks, the tail kept, the result replacing the
  causal mask. Against the pin's real indexer:

  | T | rows the pin prunes | rows differing | of them exact ties at the cut | output on the other rows |
  |---|---|---|---|---|
  | 2052 | 1 | 1 | 1 | 5.7e-8 |
  | 2080 | 29 | 19 | 19 | 5.6e-8 |
  | 4096 | 2,045 | 23 | 23 | 6.1e-8 |

  Every row also keeps exactly min(512, complete blocks) blocks. A row
  counts as a tie only if it differs in blocks, all of them at the cut
  score, and its tail is identical (review: an empty difference must not
  pass as a tie). Five mutants fail the cell by assertion, none by a crash:
  the dense mask, no key norm, the block rope at the group's last token, the
  tail dropped, the tail missing the diagonal key. Before the stable rule,
  every difference was also an exact tie (1/20/24 rows, measured).
