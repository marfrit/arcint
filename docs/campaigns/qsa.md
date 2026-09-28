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
`build_qsa_attention_model`; stateful: `_qsa_mask_dynamic`,
`build_qsa_stateful_attention_model`), `tools/q4e/serving_shape.py`
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

## Step 3 design — the served path (2026-09-28; operator decision owed)

The served graph needs two things the paged path does not carry today.

**1. The indexer's raw-key history per attention layer** (128 values a
token; ~16 MB a layer at 32k tokens in f32).
- **A. A plain state Variable in the paged model.** `SDPAToPagedAttention`
  removes only the Assigns it matched and keeps the others (`code`,
  `sdpa_to_paged_attention.cpp`: `var_ids_to_remove`), provided the state is
  not gathered by `beam_idx`, whose parameter the pass deletes. It is the
  cheapest route. But a Variable is one per infer request, not one per
  sequence, so QSA would serve one lane only, and the prefix cache and
  KV checkpoints would not cover the history without extra runtime code.
- **B. A paged indexer cache** beside the key and value caches, keyed by the
  same block tables: every serving feature kept. It is a new cache kind
  through the pass, the plugin and arcint's cache ledger (the GDN states'
  paged ops are the precedent).

**2. The selection into paged attention.**
- `PagedAttentionExtension` has no input for it, and no kernel reads one:
  `qq_bias` covers only new tokens, sliding windows are contiguous, and
  XAttention is prefill-only and xe2-only (`code`).
- Both routes need a core-op input (an optional last input), the pass to
  wire the indexer's output into it, and the decode (`pa_sdpa_opt`) and
  prefill kernels to honour it.
- **As a mask** it is correct but reads every key.
- **As a list of the selected positions** (2,048 + the tail, a count known
  host-side) the decode reads 2,051 keys instead of N. That is QSA's
  long-context saving.

**Recommendation:** A + the mask first, for correctness on the served path
at one lane (Flash-Next serves one lane today), then the position list for
the speed, then B if multi-lane QSA is wanted. Each step is a card window
against the step-2 graph (byte-exact on the A770).

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
  | 2080 | 29 | 18 | 18 | 5.6e-8 |
  | 4096 | 2,045 | 23 | 23 | 6.1e-8 |

  Every row also keeps exactly min(512, complete blocks) blocks. A row
  counts as a tie only if it differs in blocks, all of them at the cut
  score, and its tail is identical (review: an empty difference must not
  pass as a tie). Five mutants fail the cell by assertion, none by a crash:
  the dense mask, no key norm, the block rope at the group's last token, the
  tail dropped, the tail missing the diagonal key. Before the stable rule,
  every difference was also an exact tie (1/20/24 rows, measured, with the
  (1 + w) fold applied twice; see the correction below).
- 2026-09-28. **Step 2 done** (`measured-here`, CPU plugin, real blk.3
  tensors, `test_qsa_stateful_piece_matches_the_pin_with_its_cache`).
  `build_qsa_stateful_attention_model` keeps K, V and the indexer's raw keys
  (128 floats a token) in Variables, dynamic in T. Every call pools the
  blocks from the whole history and applies each query's visibility at its
  absolute position (`_qsa_mask_dynamic`), as the pin does per query.
  - **Against the pin with its own `DynamicCache`** (indexed layer), fed a
    2,048-token prefill, a 40-token chunk and 12 decode steps (2,100 tokens,
    across the 2,051 boundary). The prefill prunes no row (a query below
    position 2,051 keeps every complete block), so the selection is tested
    on the 49 rows past the boundary. None differs for real, 24 differ only
    by exact ties at the cut, and the outputs sit 2–6e-8 from the pin on
    every row without a tie (2–7e-8 after the gamma correction below).
  - **Mutants:** the indexer without its history fails (at run time, the mask
    no longer matches the key length); visibility by the row's relative
    position fails by assertion.
  - **What step 3 has to carry into the served graph:** a raw-key state per
    attention layer (~16 MB at 32k tokens in f32, less at f16/u8), and the
    selection into paged attention. Today `SDPAToPagedAttention` drops any
    SDPA mask (`code`: `state_management_pattern.cpp` matches it as
    `any_input()`).
- 2026-09-28. **Two corrections to the indexer's weights: the norm gammas
  in the step-1/2 cells, and the projections in the full-depth reference.**
  - **The gammas (the cells).** The converter folds both indexer gammas,
    stored = 1 + w (llama.cpp `conversion/qwen4exp.py`: `data_torch + 1` on
    `.indexer.{q,k}_layernorm.weight`, `code`). llama.cpp applies them as a
    plain RMSNorm (`qwen4exp.cpp` `build_norm`, `code`). blk.3's stored
    values have mean 0.96 (`measured-here`). The pin adds the 1 itself
    (pin 171), and `gguf_feed` already undoes the fold (kind `gamma1`). But
    the cells' `indexer_state` fixture read the stored values raw, so the
    pin applied the fold twice: a scale of ~1.96. Parity held because both
    sides read the same state.
    - The fixture now takes all three indexer tensors from the feed and
      asserts the gammas' mean is near 0. That check is red when the stored
      values are fed (`measured-here`).
    - Re-run on the CPU plugin, 17/17 cells (`measured-here`; the
      projections through `gguf.quants`, bit-identical to the fixed feed):
      - Step 1: 1/18/23 differing rows at T 2052/2080/4096, all exact ties
        at the cut (the table above; 2080 was 19). The other rows are
        unchanged at 5.7/5.6/6.1e-8.
      - Step 2: 24 tie rows, none real, 2–7e-8 on the other rows.
      - Of the mutants, only no-key-norm was re-run; it still fails by
        assertion.
  - **The projections (the reference).** `indexer.q_proj` and
    `indexer.k_proj` are the checkpoint's only BF16 tensors (24,
    `measured-here`). gguf-py hands BF16 over as raw bytes (uint8, twice the
    row width), and `gguf_feed` cast those bytes to f32 from its first
    commit (36e0129) until 2026-09-28. Nothing fed a BF16 tensor until the
    indexer was mapped and the reference tool landed (1271dc3, af465dc, both
    2026-09-19). `pin_tensor` gave [640, 5120] with
    values 0..255; `fitted` cropped it to [640, 2560] (`measured-here`, on
    the dev host and on the reference host).
    - The step-1/2 cells were not affected: they read the projections through
      `gguf.quants`, which converts BF16 bit-exactly (`measured-here`).
    - `tools/ref_forward_stream.py` feeds every parameter through `fitted`,
      so the f32 reference captures ran every indexer on byte garbage. Below
      position 2,051 the selection keeps every complete block whatever the
      scores, so those rows are unaffected (`code`, the selection rule). At
      or above it, the reference's selection is not the model's.
    - **Consequence:** a capture scores rows 1,368–2,734, 684 of them at or
      above 2,051. So every reading against the reference at or above 2,051
      is void, **and so is every whole-window figure**:
      - the served artifacts' 0.369 / 0.181 / 0.827 and 0.380 / 0.191 / 0.792;
      - llama.cpp's 0.339 / 0.065 / 0.802 (the medians 0.0649 / 0.0283 that
        `kld_bar.py` prints as the acceptance candidate);
      - the "above 0.455 — the dense-for-sparse price" of
        `sub4bit-vram-kernel.md`, the number and the attribution alike (a
        garbage selection in the reference explains the jump equally well).
    - The below-2,051 split stands, and can be re-read from the existing
      captures now (`kld_vs_capture.py` / `capture_vs_capture.py` print it).
      The rows at or above 2,051 need a re-capture with the fixed feed before
      the gate's above-2,051 row is read. Dated notes sit at each figure.
    - Fixed in `gguf_feed._dequant`, which now also refuses any tensor whose
      dequantised shape differs from its header's (the check whose absence
      let `fitted` crop), and in the same path in `expert_store` (latent: no
      expert tensor is BF16). Two red-first cells: every real BF16 tensor
      against an independent bit conversion, and a BF16 expert through the
      store. Both were red on the old cast (`measured-here`).
  - The pin's dense-vs-QSA max-abs at T 2080 is 1.064551e-03 with both folds
    undone (`measured-here`). The 2.385560e-02 on record was drawn at
    692c0a6 with all four q/k gammas folded twice, so the two figures do not
    isolate either fix. The bar built on it is withdrawn anyway (above).
