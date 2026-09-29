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
- 2026-09-28. **Step-3 pre-flight (T0) and the served indexer (T1a).**
  - **Operator decision owed; default A in force.** How the indexer's
    raw-key history lives in the paged model -- A, a plain state Variable
    (one lane, the prefix cache and KV checkpoints not covered without
    runtime code), or B, a paged indexer cache beside K/V -- was asked and
    is not yet answered, so the decision-independent work proceeded on the
    recorded recommendation: A + the mask first. See the step-3 design note
    above.
  - **Upstream check** (`code`): the pinned `paged_attention.hpp`
    (the pinned tree) and upstream OpenVINO `master` are byte-identical
    (sha256 `ee721fc5…`, both 28 inputs); the file's last change is
    2026-05-13 (`18742d7213`), after which nothing was added. No sparse /
    XAttention / "selected blocks" input exists upstream beyond the pin's
    own 17-19 (`xattention_*`) and 25 (`token_type_ids`). The selection thus
    takes a NEW last input -- index 28 -- which is the append a later
    upstream input would collide with; stated as such, not hidden.
  - **T1a done.** The served emitter (`emit_stateful_attention(qsa=True)`)
    carries the indexer through the same `q4e.attention._qsa_mask_dynamic`
    the step-2 graph runs -- one function, not a copy -- over the whole
    raw-key history in a PLAIN state Variable
    (`cache_params.past.indexer_key.N`, deliberately NOT gathered by
    `beam_idx`, whose Parameter the pass deletes), and replaces the baked
    causal mask with the token-major selection `[T,1,1,N]`. The indexer is a
    standalone emitter (`_qsa_indexer_mask_served`) so it can be RUN without
    that function's token-major KV Concat, which is well-formed only after
    `SDPAToPagedAttention` (the CPU plugin refuses the pre-pass graph; the
    served attention core's own numerics are T2/T6's).
    - `test_qsa_served_indexer_selects_what_the_step2_stateful_graph_selects`
      (`measured-here`, CPU, real blk.3 tensors): 2,048 + 40 + 12x1 tokens,
      the served mask BIT-IDENTICAL to
      `build_qsa_stateful_attention_model`'s at every chunk (max|d| 0.0).
      Mutant (pooled from the current chunk only, history dropped) fails by
      assertion: shape [1,1,40,40] vs [1,1,40,2088].
    - `test_qsa_served_mask_is_exactly_causal_below_the_2051_boundary`
      (`measured-here`): at T=2051 the served mask equals the dense causal
      mask exactly; at T=2052 the first pruned row appears. The boundary is
      derived (`block_topk*ratio + ratio - 1 = 2051`), not tuned.
    - `test_qsa_off_leaves_the_serving_shape_graph_unchanged`: `qsa=False`
      adds no indexer Variable and no `qsa_selection` marker and matches the
      default build's node count and constant bytes; `test_serving_shape.py`
      stays green (37 passed, 1 skipped).
    - C++ `artifact_counts_qsa_layers_only_when_the_manifest_declares_them`:
      `serving-shape.json` `qsa:true` makes `n_qsa_layer == n_attn_layer`,
      without it 0. Red first, shown by dropping the manifest read
      (`arcint-test`: 600 cases, 1 failed).
  - **T1a review refinements** (same day). Added
    `test_qsa_served_attention_consumes_the_marked_mask`: the marker must
    reach the SDPA's attention-mask input (input 3), not merely exist --
    red-first, shown by a mutant handing the SDPA the causal mask (failed,
    measured). The served indexer now asserts `norm_plus_one` (a
    plus_one=False family would diverge silently from
    `_qsa_mask_dynamic`), and `artifact.cpp` counts EVERY attention layer of
    a QSA manifest, not only those spelled `qwen_sparse_attention`. A parser
    cell asserts `--qsa` defaults off and is refused for qwen35moe.
    **Caveat on the record:** option A's plain Variable is not covered by the
    prefix cache, KV checkpoints or MTP rejection yet, so a `--qsa` artifact
    is safe only in a cold, one-lane run until T4 lands the refusals -- do
    not register one (T5) before the gate (T6).
  - 2026-09-28. **T1b done, option A** (`measured-here`, device-free, the
    pinned OpenVINO pass).
    `test_qsa_indexer_state_survives_the_paged_attention_pass`:
    `SDPAToPagedAttention` consumes the KV pair into `key_cache.0` /
    `value_cache.0` and deletes `beam_idx`, while the indexer's ReadValue and
    its Assign (`cache_params.past.indexer_key.3`) stay in place -- the
    plain-Variable route is well-formed after the pass, as
    `sdpa_to_paged_attention.cpp`'s `var_ids_to_remove` implies. Red first: a
    mutant gathering that Variable by `beam_idx` makes the pass refuse the
    model (`Model references undeclared parameters: beam_idx`).
  - 2026-09-28. **Operator decision: option A.** The indexer's raw-key
    history rides a plain state Variable per QSA layer (one lane; the prefix
    cache and KV checkpoints stay uncovered until T4's runtime work), as
    T1a/T1b implemented. The position list (T7) follows for the speed; a
    paged indexer cache (B) remains the route to multi-lane QSA if wanted.
  - 2026-09-28. **T2 landed as source + patch; the OV unit/plugin ladders
    are OWED** (`measured-here`, device-free, at the graph level; plugin
    patch `0073`).
    - **core.** `PagedAttentionExtension` accepts 28 inputs or 29; input 28
      is `[T_new, past + T_new]` u8 (1 = keep the key, 0 = drop it; rank-1
      `[0]` = none), and absent keeps the node at 28 -- today's form byte for
      byte, so no existing artifact or arch hash moves.
      `validate_and_infer_types` and `paged_attention_shape_inference.hpp`
      accept both; type tests prove the accepted form and refuse a wrong
      type.
    - **pass.** `StateManagementPattern` inspects the SDPA's attention-mask
      node; when it carries `rt_info arcint = "qsa_selection"` (the
      exporter's marker, T1a) the mask is squeezed to `[T, N]`, compared
      with -0.5 and converted to the u8 visibility mask appended at 28. An
      untagged mask -- every causal mask -- is dropped exactly as before.
      ONE LANE is implicit (the mask rows are the flattened new tokens);
      lanes > 1 is T4's load-time refusal until option B.
    - **red first, runtime-level.** On the pre-0073 runtime the tagged mask
      is dropped: a `--qsa` depth-4 serving-shape graph reaches
      PagedAttention with 28 inputs (`measured-here`). On the rebuilt core
      lib the same graph reaches 29 inputs, input 28 a u8 `Convert` named
      `qsa_selection`; `qsa=False` stays at 28 (`measured-here`).
    - **patch.** `0073-qsa-selection-paged-attention-input.patch` is written
      to `contrib/packaging/marfrit-openvino/patches/` and to the `patches/`
      mirror; `git apply --check -R` on the series tree is clean. The GPU
      plugin's input-count check accepts 28 or 29 and refuses 29 by name
      until T3's kernel path lands (never a silent fallback); the GPU plugin
      target linked clean (`measured-here`).
    - **OWED.** The OpenVINO unit ladder (two type_prop cells and two
      transformation tests are already in the patch) needs a tests-enabled
      OpenVINO build, which does not exist here; the dev build's CPU plugin
      link is broken by a stale snippets archive, unrelated to this change,
      so only the `openvino` and `openvino_intel_gpu_plugin` targets were
      rebuilt. The plugin unit ladder for the PA tests is therefore NOT run
      in this leg.
  - 2026-09-28. **T3 done** (`measured-here`, `ov_gpu_unit_tests` on both
    cards; plugin patch 0073 now carries T2+T3).
    - **Build.** `build-prod` reconfigured with `-DENABLE_TESTS=ON`
      (objects reused); the CPU-plugin link failure was a stale
      `libopenvino_snippets.a`, fixed by rebuilding that target
      (11.9 -> 12.7 MB).
    - **Kernel read.** `paged_attention_opt.cl`'s single-token, GQA
      single-token and multi-token stages read input 28 and set a dropped
      key's score to `SOFTMAX_ACCUMULATOR_VAL_MIN`, beside the existing
      `token_idx >= seq_len` and `qq_bias` sites (query row = `seq_idx -
      subsequence_begin` for multi-token, 0 for decode; width = past + new
      / `seq_len`). `supports_micro_sdpa` returns false under QSA -- the
      micro stages do not read the mask, so they are taken away loudly --
      and a QSA prefill is routed through MIXED so one kernel file carries
      the read. A one-lane refusal guards the flattened rows.
    - **The red/green.** The cells failed before the read (dense output vs
      the pruned reference: 0.0147 / 0.0021) and now pass:
      `regression_paged_attention_qsa/paged_attention_qsa_test` is 8/8 on
      the A770 and 8/8 on the B60, over decode `{{1, 35}}`, MIXED
      `{{128, 2048}}`, past-0 prefill `{{2100, 0}}` and the GQA decode at
      5,001 tokens (many fully masked partitions). `*paged_attention*`
      reads 276 passed on each card, unchanged from the 28-input path.
    - **Finding on the record (T6).** The causal-equal control is
      byte-identical on the GENERATE params but only within a <= 1-ulp f16
      floor on the MIXED ones (2.0e-6 at 128/2048, 3.0e-5 at 2100/0): QSA
      changes that route (micro off, past-0 prefill -> MIXED) and the added
      kernel block shifts f16 codegen. The DESIGN §3.4 byte-identity clause
      below 2,051 therefore needs the served dense and QSA graphs to use the
      SAME route; T6 must either accept the measured floor or the
      PREFILL/micro routes must learn the mask too. Measured, not narrated.
    - **Staged.** The runtime is installed at the dev prefix `ov-0073`
      (version `2026.4.0-22849-71640275d29-marfrit-p24`); the GPU plugin's
      sha256 prefix is `3456feeb18edaaca`. The tests-enabled build is the
      dev tree's `build-prod` with `ENABLE_TESTS=ON`; its `ov_gpu_unit_tests`
      runs with `--device_suffix=1` (the deterministic card) and `=0`.
      The host paths live in the operator-local notes.
  - 2026-09-28. **T3b landed: the route gate makes the below-boundary invariant
    hold by construction** (`measured-here`, `ov_gpu_unit_tests`, both cards;
    patch 0073 now carries T2+T3+T3b).
    - **Why.** T3 left a route-parity risk: below 2,051 the mask is exactly
      causal, but QSA disabled micro and routed a past-0 prefill through
      MIXED, so the 29-input graph ran different kernels than the dense
      28-input one -- a <=1-ulp f16 shift that compounds with depth (the B60
      record: 0.14 nats and 10-15% argmax flips at depth 48). Accepting that
      floor would change the campaign's own below-2,051 invariant, so it is
      fixed by construction instead.
    - **Mechanism.** The exporter writes the boundary
      `block_topk * ratio + ratio - 1` (2051) as `rt_info:qsa_boundary` on the
      marked mask; the pass copies it onto the PagedAttention node; the
      primitive carries it. `qsa_above_boundary` reads the call's own
      `max_context_len` from the kernel's memory deps. At or below the
      boundary the impl keeps today's stage (PREFILL stays PREFILL) and
      `can_use_micro_sdpa_for` leaves micro available -- the mask is a no-op
      and the route equals the dense graph's. Above the boundary micro is
      taken away and a QSA prefill is routed through MIXED, so the opt kernel
      reads the mask. An unknown boundary (0) or an absent `max_context_len`
      is treated as above (read the mask), never silently ignored.
    - **Red first.** The prior control cell asserted a 2e-4 floor on the MIXED
      params; T3b's cell asserts byte-identity at or below the boundary. Red
      before the gate (measured 2.0e-6 at 128/2048, 3.0e-5 at 2100/0), green
      now.
    - **Measured.** `regression_paged_attention_qsa` reports 8 passed / 2
      skipped on the A770 and the B60. `selection_is_honoured` now skips at or
      below the boundary (a pruned mask is not a valid case there) and passes
      above it; the causal-equal control asserts `max_delta == 0` on every
      below-boundary param -- including the new MIXED `{{128, 1900}}` (total
      2,028) -- and the f16 floor only on the above-boundary MIXED params,
      where the route legitimately differs because QSA prunes.
      `*paged_attention*` has no failures on either card. The Python cell
      `test_qsa_route_gate_boundary_is_on_the_marked_mask` pins the rt_info
      boundary at 2051, red when the exporter omits it.
    - **For T6.** DESIGN §3.4's byte-identity below 2,051 now holds by
      construction: the dense and QSA graphs take the same route there. Above
      the boundary the comparison is KL, not bytes, which is where the gate
      reads.
  - 2026-09-28. **T4 landed: the runtime accepts the indexer state and refuses
    what option A cannot honour** (`measured-here`: the C++ unit ladder, no
    card).
    - **Accept.** `load_paged` reads the indexer geometry off the served
      graph's own Variables (`cache_params.past.indexer_key.<layer>`,
      `[1, past, 128]` f32), not a config key, so the charge cannot drift:
      12 x 128 x 4 B = 6 KiB/token (192 MiB at 32k). The ReadValue scan and
      the `conv_proto`/`gdn_proto` classification already skip it (dynamic
      seq dim, id not `.key.`), so the state tables keep their own shapes.
    - **Refuse, loudly, before any compile.** `qsa_runtime.h`'s
      `runtime_refusal` returns the message load_paged throws: lanes > 1
      (`--parallel 1`); the prefix cache (its blob carries the KV pages and
      the GDN rows but not the indexer Variable, so a hit would build the
      selection from an empty/stale history); and paged speculative decoding
      (--mtp/--dflash: a rejected draft has already appended raw keys, and
      the paged rollback moves the committed GDN row back without trimming
      that history).
    - **Reset.** A paged lane's request lives across requests; the la_* port
      tensors are zeroed per request but nothing cleared the graph Variable.
      `generate_paged` now calls `lane.req.reset_state()` at past == 0 when
      the model carries indexer state -- graph Variables only, so the KV
      (ports) and the GDN/conv state (la_state_names_ tensors) are untouched.
    - **Ledger.** `qsa_state_bytes_token_` is folded into the fit's
      `kv_bytes_token` (never the true KV rate) and printed as
      `+ QSA state X GiB (Y KiB/token)` beside the KV term.
    - **Ledger read (2026-09-28): the multiplier is the fit's budget CEILING,
      not `n_ctx`; there is no device over-reservation to free.** The two
      boots' figures (d4qsa 1.78 GiB = 0.5 KiB x 3,739,776; d48q8qsa 4.47 GiB
      = 6.0 KiB x 781,872) are `lanes * max_ctx * rate`, where `max_ctx` is
      the fit's computed maximum, not the served depth. The state is
      **device** memory, not host: the plugin's `VariableState` allocates
      `usm_device` when USM is in use, else `cl_mem`, and an `Assign` sets
      that variable's layout to its own output layout (`code`:
      `src/plugin/variable_state.cpp` `update_device_buffer`,
      `src/graph/primitive_inst.cpp` the `assign` branch), i.e. the actual
      `past`, allocated lazily per forward. So the charge is a correct worst
      case at the ceiling, and exact under auto-fit (`n_ctx = max_ctx`).
      Under an explicit `--n-ctx` the device holds `n_ctx x rate` -- 1.5 GiB
      at 262,144, depth 48 -- not the printed 4.47 GiB; the fit's `max_ctx`
      is never preallocated. The expert slot pool is config/probe-sized, not
      derived from `max_ctx`, so the QSA term takes nothing from it. The line
      now prints the multiplier (`x 781872 tok`) so the ceiling is not read as
      the served allocation.
    - **/props.** `qsa` (on/off) and `n_qsa_layer`, resolved at load.
    - **Red first.** `tests/test_qsa_runtime.cpp`: `state_bytes_per_token`
      (6144), a synthetic graph with two indexer Variables and one KV
      Variable (counts the indexer only), and one cell per refusal. Removing
      the guard makes its cell fail (measured: 3/3 refusal cells red when
      `runtime_refusal` returns nullopt); restored, 7/7 green. The whole OV
      unit ladder reads 624 run / 1 failed, and that one failure
      (`gguf_pass_neutralises_awq_multipliers_and_compares_norms`) reproduces
      on a clean OV build at the same HEAD with the same runtime -- a
      pre-existing environment mismatch, not this change.
  - 2026-09-28. **T5 landed: both artifacts are exported and registered.** The
    Paris-cell boots are the remaining step, owed to the T6 window (the served
    arcint must be built against the ov-0073 runtime so the T4 accept path and
    the T3b route gate are in the binary that loads them).
    - **Exports.** `qwen38-flash-next-d4qsa-ov` on rpool (lm_xml_sha
      `ec98641c20277204`, .bin 8.64 GiB, peak host 27.59 GiB) and
      `qwen38-flash-next-d48q8qsa-ov` on the NVMe (`--layers 48 --qsa
      --dense-q8 --dense-u8`; lm_xml_sha `e248c2e11761b40e`, .bin 63.88 GiB,
      peak host 51.41 GiB). Both carry `serving-shape.json` `qsa: true`, 1 and
      12 QSA layers. The d48 export first ran out of space writing straight to
      the NVMe (the 58 GiB arena plus the 64 GiB .bin against 94 GiB free); the
      retry put the arena on rpool (`--arena`) and the artifact on the NVMe.
    - **Registration.** `models/allowlist-raw.json` and
      `src/core/model_registry.cpp` gain `qwen3.8-flash-next-d4qsa` and
      `qwen3.8-flash-next-d48q8qsa`; `tests/test_registry.cpp` reads 27 ids and
      pins both hashes. The registry ladder is green (21 cases).
    - **Served (d4qsa).** Built arcint from this tree against the staged
      `ov-0073` runtime and booted the depth-4 artifact on the A770
      (`--mtp off --ngram-gguf /flash-model/ngram/…`). `/props` reports
      `qsa: true, n_qsa_layer: 1`; the reservation line prints `+ QSA state
      1.78 GiB (0.5 KiB/token)` beside the KV term, i.e. the T4 ledger works
      on the real artifact. The chat cell returns finite text (garbage, as
      depth 4 of 48 must). The d48q8qsa Paris cell is the T6 leg.
    - **Served (d48q8qsa).** 36 GDN + 12 attn layers, `QSA served: 12 indexer
      state layer(s), 6.0 KiB/token; one lane, no prefix cache, no paged
      speculation`; the reservation line prints `+ QSA state 4.47 GiB (6.0
      KiB/token)`. On the B60 with `--offload-ratio 78 --moe-cpu-tier
      --moe-per-expert-dispatch`, the Paris cell answers **Paris**. Both T5
      artifacts therefore boot and answer.
    - **T6 as trimmed (operator, 2026-09-28).** The gate is now the
      smoke+repeat: a needle question placed past 2,051 tokens in a 4-8k
      prompt, answered by `d48q8qsa`, sent twice with identical text -- the
      first served exercise where the selection actually prunes (both Paris
      cells were short, below the boundary). The full byte-exact sweep and the
      KL above 2,051 move to T8.
  - 2026-09-28. **T6 smoke + repeat: PASS on the A770; the B60 diverges (its
    known per-card defect).** `d48q8qsa` served a 3,832-token needle prompt
    ("the vault passphrase is ORANGE-FALCON-77" placed past 2,051) and
    answered `ORANGE-FALCON-77` twice with byte-identical text on the A770
    (`GPU.1`, ratio 99 + tier + dispatch; prefill 3,832 tok in 517.16 s then
    491.17 s, decode 62 tok in 8.90 s then 7.08 s). On the B60 the same
    request diverged between the two runs (run A reasoned 80 tokens and
    truncated the answer at `ORANGE-FALCON-`, run B answered
    `ORANGE-FALCON-77`) -- the `served-prefill-determinism` Xe2 GDN
    nondeterminism, not a QSA fault. Both runs kept the needle, so the
    selection actually pruned past the boundary and the answer survived it.
    The full byte-exact sweep and the KL above 2,051 stay deferred to T8.
  - 2026-09-28. **T7 landed: the decode path reads only the chunks that hold
    a selected key.** (operator ruling: chunk-uniform skip)
    - **Cause of the reverted per-lane skip** (`code`): the key load
      (`BLOCK_READN`) and the query `sub_group_broadcast` are subgroup
      collectives. A per-token skip diverges lanes around them -- undefined
      behaviour. The B60 compiles the `XE2_QK_MULTIPLICATION` branch and the
      A770 the other one, which is why the corruption showed only on the B60.
    - **Fix:** decode-only chunk-uniform skip. Per chunk of `SUBGROUP_SIZE`
      keys, `any = sub_group_any(selected(token))`; if no lane's key is
      selected, skip the chunk's key reads and dot products and set
      `qk_acc = SOFTMAX_ACCUMULATOR_VAL_MIN` for every lane. The same uniform
      predicate skips the value reads (the main blocks and the partial block).
      When at least one key is selected the chunk runs exactly as today and
      the per-token mask drops the rest. Prefill/MIXED stay masked dense
      (FreeToken and llama.cpp both do, `code`).
    - **Measured:** `regression_paged_attention_qsa` 8 passed / 2 skipped on
      the A770 and the B60 (including the GQA decode at 5,001 on the B60);
      whole `*paged_attention*` filter 276 passed on each card. Patch 0073
      regenerated (T2+T3+T3b+T7) and mirrored; the staged runtime's plugin
      sha256 prefix is `ffc34950d4658cf3`.
    - **Timing (32k decode vs dense d48q8):** pending, recorded below when run.
  - The export flag `--qsa` (default off, so existing artifacts and the arch
    hash do not move) records `qsa` in the manifest and feeds the indexer
    tensors through `gguf_feed` (`self_attn.indexer.*`; the two norm gammas
    are kind `gamma1`, the stored (1 + w) undone, never a raw GGUFReader).
