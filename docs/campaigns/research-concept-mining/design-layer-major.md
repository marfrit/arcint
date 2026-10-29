# 01 — layer-major prefill in arcint's llama.cpp engine

Candidate 1 of `state/candidates.md`, audited in `state/audit.md` §1. The audit changed the brief:
llama.cpp **already** walks the layers over a whole ubatch, and arcint's bank stream already copies each
bank tensor once per ubatch. "Layer-major" is not a missing walk. What is missing is that the ubatch
cannot be large, because the prompt's compute buffers scale with it. This document builds the dial
that decouples the two, behind a flag, and prices it first.

Evidence classes: `code` = read in `llama-src/` or `arcint-src/`; `paper` = a document or an arcint
campaign record states it; `(inferred)` = my reasoning.

---

## 1. What the engine does today

- The batch is split into ubatches of `n_ubatch` tokens: `llama_context::decode` computes
  `n_ubatch_eff` (cut further to fit a compute reservation) at
  `llama-src/src/llama-context.cpp:1986-2007`, `init_batch` gets it at `:2007`, and every ubatch runs
  `process_ubatch` (`:1548`) and builds its own graph (`:2094`), `code`.
- Each graph walks **all layers over the ubatch's whole token set**:
  `llama-src/src/models/qwen4exp.cpp:449-490` (`for il … build_layer_attn … build_layer_ffn`) — the
  residual is `[n_embd, n_tokens]` and the MoE's `MUL_MAT_ID` has `ne2 = n_tokens`, `code`. So a
  prefill ubatch of T tokens is already layer-major over T; "chunk-major" is only the sequence of
  ubatches, each re-walking the 48 layers.
- With the expert cache, `n_ubatch = n_batch = --llama-prefill-chunk` (`code`:
  `arcint-src/src/exec/backend_llama.cpp:577-588`), so the served Flash-Next chunks its prompts at
  2,048 tokens and pays 10 layer walks for a 20k prompt.
- The bank stream (0032) copies **each bank tensor whole**, once per ubatch, into a VRAM ring ahead
  of its GEMM (`code`: `llama-src/ggml/src/ggml-opencl/ggml-opencl.cpp:27696-27814`; ring
  `GGML_OPENCL_BANK_STREAM_RING` default 4, `min_tok` 256 at `:27691`). That is the per-layer
  read-once property at ubatch granularity; the duplication is only across ubatches, `code`.
- The prefill expert GEMM groups `(token, slot)` pairs by expert into 32-token tiles on the device
  (`kernel_moe_route_tiles` + one fp16-DPAS tile per work-group, `code`:
  `ggml-opencl.cpp:30870-31080`; README 0006 at `arcint-src/contrib/llama.cpp/README.md:350-401`).
  The tile's token width is the padding lever; a probe prints it (`GGML_OPENCL_MOE_TILES_DEBUG`,
  `code`: `ggml-opencl.cpp:30780-30797`).
- The compute buffers are reserved for the cache's cells × the ubatch's tokens: `--llama-reserve-cells`
  (0052) reserves for N cells and cuts deep chunks to fit (`code`: `:1986-2007`'s budget and
  `:1994-2000`'s floor). At Flash-Next's 131,072 the served reservation is ~3,048 MiB with chunk
  2,048, the draft context 770; decode needs ~93-770 MiB (`paper`, stage 3c's table).

## 2. Why the naive arm is not the prize

The measured arm that already does the span (one ubatch of 8,192 tokens is a layer walk over 8,192,
with one bank copy per layer) is arcint's own chunk sweep, IQ2_XS on the B60 at 13,000 MiB of slots,
the 20,045-token needle (`paper`):

| chunk | prefill t/s | peak VRAM | decode, 300 tokens |
|---|---|---|---|
| 2,048 | 422.1 | 21.2 GB | 28.2 |
| 4,096 | 443.9 (+5.2 %) | 22.2 | 29.8 |
| 8,192 | 435.0 (+3.1 %) | 24.3 | 30.5 |

Three readings: the bank's 4× fewer copies did not show (the copy engine overlaps them, 0032's own
gain was 416.1→497.6 t/s, `paper`); the peak VRAM rose to 24.3 GB, which at the deepest prompt is
the eviction margin (a thin margin bites under xe, DESIGN §3.3, `paper`); and the tile padding at
2,048 is the one effect the sweep cannot separate from the copy. The reference's 4.8-5.7× at 20K
(`paper`, Strata 37e4832a on 2×RTX A4000) is against a chunk-major engine that streams only the
chunk's routed experts — it does not transfer to an engine that already copies the whole bank once
per ubatch. So the ceiling on this card, for the *span* the reference measured, is +3.1 % minus the
VRAM price; what this design can add is to keep 2,048's buffers while taking 8,192's MoE tiles.

## 3. The span, and what each part costs

"The span" (S) = the ubatch's token count: 4,096 or 8,192. "The tile" (T) = the token width of the
buffers that cannot scale: 2,048 today (the reserve-cells width). The design's one sentence: **keep
S large for the residual, the MoE GEMM and the bank copies, run the indexer and attention per T.**

| part at S = 8,192, Flash-Next, 131,072 ctx | can it run over S? | why |
|---|---|---|
| residual stream + hyper-connection mix/inject | yes | `res_hc [n_embd·hc, S]` f32 = 10,240 × 8,192 × 4 B = 320 MiB; the per-layer mix intermediate ~80 MiB (`code`: `qwen4exp.cpp:441-490`) |
| expert cache's two-branch MoE GEMM (slot + bank) | yes, and this is the point | `MUL_MAT_ID` at `ne2 = S`; the pairs are routed and tiled on the device, so a 32-token tile is full at 8,192 (~160 pairs an expert) and nearly empty at 2,048 (~40) (`code`: the route-tiles path; `inferred` for the pair counts) |
| bank copies | yes, once per layer per span | the stream's order is the layer order; a fatter ubatch reads each bank tensor once per 8,192 instead of 4× (`code`: `27696-27814`) |
| GDN layers (36 of 48) | yes | the prompt kernel walks `n_tokens` in `TC` chunks with one work-group per (head, seq, rows) (`code`: `kernels/gated_delta_net_intel.cl:59-106`); the work-group count does not grow, so the per-token cost is the same by construction, occupancy unchanged (`code`, `inferred` for the rate) |
| KV writes | yes, one `cpy_k`/`cpy_v` per layer for S cells | the memory module's ubatch is the span (`code`: `qwen4exp.cpp:762`, `build_inp_kpool`'s `mctx_idx->cpy_k`; attention K/V through the hybrid memory's `set_k`/`set_v`) |
| QSA indexer: `pool_mask` input | **no** | the input is `[n_pool, n_tokens]` f32 (`code`: `qwen4exp.cpp:708`); at 131K, n_pool ~ n_kv/compress_ratio, ~1 GiB at T=2,048 and ~4 GiB at S=8,192 |
| QSA indexer: head scores | **no** | f32 `[n_pool, n_heads, n_tokens]`, "1 GiB at 131K and 2,048 tokens" (`paper`, 0052's find, with the ReLU copy) — ×4 at S |
| QSA selection mask (`mask_all`, `[n_kv+n_sel, n_tokens]`) | **no** | the same scaling (`code`: `qwen4exp.cpp:850-853`, the TODO at `:844`) |
| flash attention / `FLASH_ATTN_EXT` | **no** (by choice) | its V and mask buffers scale with the query rows; tiling query rows is cheaper than growing them (`code`: the mask's shape) |

So the build is one mechanism: the indexer and attention run over T tiles inside a span's graph, and
everything else over S. That is not a new prefill walk; it is a loop in two builders plus a
memory-module argument. This is the correction the audit forces, and it makes the candidate far
smaller than `candidates.md`'s "~1,000 lines plus the workspace lending".

## 4. Where the span walk lives (files, functions)

1. **The ubatch is the span.** `llama-context.cpp:1986-2007` already sets `n_ubatch_eff`; the span
   needs no new split. Raise `n_ubatch` to S (`--llama-prefill-chunk`, `backend_llama.cpp:577`)
   and keep the reservation cut *inside* the graph instead of cutting the ubatch. The flag that
   separates them: `--llama-prefill-span S --llama-prefill-tile T`, T default = the reserve-cells
   width.
2. **The graph's layer loop** (`qwen4exp.cpp:449-490`) stays one walk over S for the residual and
   `build_layer_ffn`. For a sparse-attention layer (`hparams.is_recr(il)==false`), the attention
   block (`build_layer_attn`, `:949-1035`) becomes a loop over `ceil(S/T)` tiles:
   - Q/K/V projections and the `cpy_k`/`cpy_v` run over S once per layer (dense GEMMs, cheap at
     S — `code`: the projections are `build_lora_mm` over `n_tokens`);
   - per tile, `build_qsa_sel` (`:739-844`) takes a `(t0, T)` window and builds its score, top-k,
     selection and `mask_all` at T rows; then `build_attn_mha` over the tile's Q rows against the
     keys up to that tile (`code`: the causal mask is per tile, so the order of tiles within the
     layer is the causal order and nothing is reordered);
   - the KV for the span is written before the first tile reads it; the tile's mask carries the
     causal restriction, so no tile sees a future cell (`code`: `pool_mask` from the memory module
     already masks by position).
3. **The memory module's indexer inputs.** `llama_memory_hybrid_idx`'s pool inputs
   (`build_input_k_idxs`, `new_pool_idxs`, `pool_mask`, `tail_idxs`, `code`:
   `qwen4exp.cpp:691-732`) keep the span's `k_idxs`/`new_pool_idxs` (host-side, small) and gain a
   `pool_mask` **per tile**: the API gets a token window (`llama-memory-hybrid.cpp`'s idx context),
   so the mask is `[n_pool, T]` and the graph views it per tile. Files:
   `src/llama-memory-hybrid-idx.cpp`, `src/models/qwen4exp.cpp`.
4. **The dense agent and the coder** (`qwen35`, `qwen35moe`) have no QSA; their attention masks are
   `kq_mask` over the ubatch (`code`), whose buffer scales with S as well — the same tiling applies
   to `build_attn` in `qwen35.cpp` (T rows per `FLASH_ATTN_EXT` call). They gain the MoE tile
   padding (coder) and nothing else; they are worth sweeping after Flash-Next.
5. **The bank stream** needs nothing new: its order is the layer order and its ring is
   `4 × the largest bank tensor`. With S there are fewer, larger GEMMs between copies; the K-1
   look-ahead it issues (`code`: `27780-27783`) still covers the next tensor behind the previous
   GEMM. One addition is worth taking: with S the copy engine is idle in stretches, so raise the
   ring to `max(4, 2 × S/2048)` and issue the *next layer's* bank tensor as the current layer's
   last GEMM is enqueued (Infernix's double buffer, `paper`) — measured in step 3 below.

## 5. The expert cache and the CPU-miss path at a span

- **Prefill is cache-neutral by construction** (`code`): `advance` counts routing only for
  `last_n <= decode_max` (8) and adapts only for `n_tokens <= decode_max`
  (`llama-expert-cache.cpp:910-918`), and `after_graph` returns before the ids read when
  `n_tokens > decode_max` (`:963`). So a span's two-branch GEMM neither counts usage nor causes a
  read-back. Keep those guards; the span's `ids_out` stays marked output (unread) and costs nothing.
- **The CPU misses are decode-only**: the publish is called only in the per-pair matvec,
  `!gemm && cache_src` (`code`: `ggml-opencl.cpp:30864-30867`). At S the GEMM path runs (the 64-token
  cache threshold, `code`: `:30837-30844`), so the CPU never computes a prefill miss — Strata computes
  prefill misses on its pool, and that is a different candidate (audit §3), not this one.
- **The zero fill** for the two-branch GEMM (`code`: `ggml-opencl.cpp:30887`) fills
  `ggml_nbytes(dst)` per projection, so its bytes scale with S (Flash-Next: 640 × 10 × S × 4 B per
  gate/up call, ~210 MB at S=8,192, 30 GB a span over 48 layers — ~67 ms at ~450 GB/s,
  `inferred`). Not free, but the same per-token cost as today, and it stays.
- **The swaps** run on their own queue behind a graph (`code`: `llama-expert-cache.cpp:711-760`);
  during a span they are only ever enqueued right after a decode graph — no change.

## 6. VRAM budget at 24 GB (24,480 MiB)

Served facts, `paper`: Flash-Next IQ3_S on 1.2.1 at 131,072 peaks at **23,478 MiB** with 11,600 MiB
of slots, a 36.06 GiB host bank, `GGML_OPENCL_BANK_STREAM=1`, `--llama-reserve-cells 20480`, MTP 3,
q8_0 KV. Margin today: ~1,000 MiB.

| item | today | with a span S = 8,192, T = 2,048 |
|---|---|---|
| expert slots | 11,600 MiB | 11,200 MiB (the span's buffers come from here; the slot count and the hit rate are then measured, `inferred`) |
| dense weights (target) | ~3,264 MiB | same (`paper`, stage 7's table for the IQ3_XXS file; IQ3_S's card-side weights are 0.37 GiB larger, `paper`) |
| MTP model | ~1,944 MiB (Q6_K head, `paper`, stage 3c) | same |
| KV, q8_0, 131,072 | ~2.3 GiB (`inferred` from the f16 1,040 MiB at 32K, `paper`, stage 7, and the q8_0 1.9 GiB saving at 131K, `paper`) | same |
| recurrent state (MTP 3: three copies) | 338 MiB (`paper`, stage 7) | same |
| bank-stream ring | 751 MiB (4 × 187.7) (`paper`, stage 7) | `max(4, 2 × S/T)` slots, ~1.5 GiB if 8 — **+750 MiB** |
| residual `res_hc` | one ubatch's 2,048 rows | 320 MiB at S (f32), **+240 MiB** over T |
| hyper-connection mix intermediates | per ubatch | ~80 MiB more |
| compute reservation | ~3,048 MiB at chunk 2,048 (of which decode needs 93-770) | ~the same at T = 2,048 (the indexer's score and mask at T), not ×4 |
| **peak** | 23,478 MiB | ≈ 24,100 MiB (`inferred`, all above counted) — inside 24,480 with the ring at 4 and 11,200 MiB of slots; the arm is measured, not assumed |

The three numbers that decide it: the tile's padding probe (step 0), the peak at the deepest prompt
(fdinfo), and decode after the needle at the reduced slot budget. If the peak passes 24,300 MiB at the
20k needle, the arm is over budget at 131K and the span is 4,096.

## 7. What must stay exact

- **Copied data**: the bank copies are byte-checked against the file (`code`:
  `llama-expert-cache.cpp:570-589`) and the span does not change which bytes are copied — the ring
  holds whole bank tensors, not slots.
- **The expert GEMM's per-row sums**: the tile kernel's work-group owns a row's whole K loop
  (one DPAS chain per sub-block), so a bigger tile adds work-groups, not a K split — the MoE output
  is the same bytes up to the kernel's own rounding order (`code`: `kernels/mul_mm_kq_f16.cl`).
- **The attention and indexer tiles** change an f32 accumulation's order (a tile's rows against the
  same keys), so they are **rounding-only**: judged at the answer-level bar, mean KL ≤ 0.03 nats
  worse than the baseline arm and argmax within 1 point, the needle and the task battery unchanged.
- **Not approximate, anywhere**: no row is skipped, no layer approximated, no KV form changed. The
  selection's ids stay the same scores and the same k; ties are the only risk, and the ids are a
  tripwire.

## 8. Staged plan

Every step has its own gate; a step that fails it stops the series.

**Step 0 — price it, no code.** One card window, the served binary, knob-only arms:
- `GGML_OPENCL_MOE_TILES_DEBUG=1`, the 20,045-token needle at chunk 2,048 and at 8,192
  (`--prefill-chunk 8192 --llama-reserve-cells 20480`), read `[MOE_TILES] … pad=`. **Bar to build the
  span: pad ≥ 30 % at 2,048 and < 10 % at 8,192.** If both are small, the MoE gain is not there and
  the series ends here.
- The same two arms with the slot budget as served (11,600) and with the peak read from fdinfo, to
  see whether 8,192 fits at depth: the peak is the second gate.
- One profiling build, `tools/cl_timeline_steps.py` over a 20k prefill, and the copy-engine counters
  (`drm-cycles-ccs` against `drm-total-cycles-ccs`) for the bank copies' share — the third gate:
  copies ≥10 % of the prefill wall at 2,048, or there is no copy prize.
- Commands: `tests/acceptance/cells/served_check.py flashnext <arcint> <IQ3_S.gguf> GPU.0 <assets>`
  and `tests/acceptance/ab.py flashnext --old <arcint> --new <arcint> --gguf <IQ3_S.gguf> --device
  GPU.0 --assets <assets> --rounds 2`.

**Step 1 — tile the attention, keep the ubatch.** No span yet: the attention and QSA builders take
S = the ubatch and loop T = the reserve-cells width; the mask, `mask_all` and the score are per
tile; `pool_mask` gains a token-window argument in the memory module's idx context. Flag
`--llama-prefill-tile T`. Measured on its own: prefill at 20k with chunk 4,096 at the served slot
budget against chunk 2,048, peak VRAM, decode after the needle; then chunk 8,192. **Gate: the
4,096 arm ≥ +5 % prefill and peak ≤ the 2,048 arm's +500 MiB; the 8,192 arm ≥ +3 % and inside the
24,480 MiB.** This step alone reproduces the chunk sweep's gain without its VRAM cost — that is the
first measurable increment, and it is also the prerequisite for step 2.
- Red-first: (a) `test-backend-ops FLASH_ATTN_EXT` with a tile boundary — mutation: the mask offset
  built for the whole ubatch instead of the tile, cases must fail; (b) a served two-prompt check
  (a 20k needle at chunk 8,192 against 2,048) — mutation: `t0` of the second tile set to 0, the
  needle must be lost; (c) the QSA id tripwire — mutation: `pool_mask` window off by one token,
  the selection ids must differ on 16 prompts; (d) KL as served (`llama-perplexity -ub 4`, 8 x 512,
  IQ3_S) against the chunk-2,048 arm — mutation: the score's ReLU moved into the head sum, KL must
  pass the bar.

**Step 2 — the span's MoE and residual.** Raise the ubatch to S (`--llama-prefill-span`) with T
fixed by step 1: the residual and `build_layer_ffn` over S, the bank stream once per layer per S.
No new walk — `n_ubatch` and the graph's own token dimension already carry S (`code`:
`qwen4exp.cpp:449-490`). Gate: prefill ≥ step 1's +2 %, peak inside budget, decode after the needle
within the spread at the reduced slot budget (the arm that pays the slots for the buffers).
- Red-first: (a) `test-backend-ops MUL_MAT_ID` at S (512 experts, top-10) — mutation: the route
  tile's `n_tok` clamped to 2,048, cases must fail; (b) the bank stream's red case
  (`GGML_OPENCL_BANK_STREAM_RED=1`, each slot filled from the next tensor's bytes) must abort —
  mutation: the ring's K raised without the look-ahead bound, the red must fire; (c) the cache's
  copy check reports `0 of N` after the span (the load-time verify, `code`: `:570-589`) — mutation:
  one swapped slot corrupted via `LLAMA_EXPERT_CACHE_MUTATE=1`, it must report 1.

**Step 3 — the copy schedule and the ring.** With S the copy engine has idle stretches: raise the
ring to `max(4, 2 × S/T)` and issue the next layer's bank tensor behind the current layer's last
GEMM (Infernix's double buffer, `paper`; FreeToken's `_prefetch_split` gathers resident experts
device-to-device and copies only misses, `paper`). Gate: the bank copies' wall share halves and
prefill ≥ +1 %; peak inside budget. Red: the stream's red case must still abort (a ring slot must
never be reused before its GEMM's marker, `code`: `27815-27833`).

**Step 4 — the other two units.** The coder (A770, resident experts: only the tile padding and the
smaller buffer; chunk 1,024 is its measured best, so the span is 2,048 and the gate is its own
spread) and the dense agent (no MoE: the attention tiling only). Gate: each unit's prefill ≥ +1 %
and decode within its spread; the answer gates as everywhere.

## 9. What would make it pay less or more

- **Less:** if the tile padding at 2,048 is small (step 0), the MoE gain is nil and the span's only
  effect is fewer bank copies, which 0032 overlaps — the series stops at step 0. If the ring eats
  slots (750 MiB at S = 8,192), decode at the reduced budget loses what prefill gains; the arm is
  A/B'd, not assumed.
- **More:** if the 2,048-token tiles waste ≥30 % (the `[MOE_TILES] pad` line), the grouped GEMM's
  dequantization is paid twice per expert, and the span takes it to once — this is the same lever
  audit §2 (int8 DPAS) attacks from the kernel side, so the two should be sequenced (2 first, then
  the span on the new kernel), not run in parallel.
- **Not measured anywhere:** the GDN layer's per-token rate at S (the kernel accepts S; the
  work-group count is head-bound, so a longer walk per group is the risk, `inferred`), and the
  indexer's tiling's own cost (a tile boundary adds a launch per attention layer per tile — 12
  attention layers × 4 tiles × ~6 kernels). The step-1 arm prices both.

## 10. Handover for the card window (Claude runs it)

Step 0 is the whole decision and needs no new code. One window, the served binary and model, the
units' flags otherwise, one fresh process an arm:

     GGML_OPENCL_MOE_TILES_DEBUG=1 \
       tests/acceptance/cells/served_check.py flashnext <arcint-1.2.6> <IQ3_S.gguf> GPU.0 <assets>

and the same with `--prefill-chunk 8192 --llama-reserve-cells 20480` added; then
`tests/acceptance/ab.py flashnext --old <arcint-1.2.6> --new <arcint-1.2.6> --gguf <IQ3_S.gguf>
--device GPU.0 --assets <assets> --rounds 2` with the two arms as the "old"/"new" binaries' flags
(the A/B needs the span flags in the new arm's unit file, not a new binary). Read: the
`[MOE_TILES] pad=` lines, the needle prefill, peak VRAM from the log's fdinfo sampling, and decode
after the needle. That is step 0's gate.
