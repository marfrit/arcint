# dflash2-agent — DFlash2 block drafting for the dense agent on the libllama engine

**Open** (charter B2, operator 2026-10-16: to be built before 1.0). Recon and design done; nothing built.

## Charter

The dense agent (Qwen3.8-27B Q4_K_M, the B60, q8_0:q4_0 KV) drafts with
the public DFlash2 block-diffusion head instead of its own MTP head: one
drafter forward proposes a whole block (up to 7 tokens), conditioned on the
target's hidden states from five layers, and the target verifies the block in
one decode through arcint's verify walk and recurrent rollback. The lever is
tokens per verify at a drafter cost that does not grow with depth (the
drafter attends a 2,048-position window).

## Reference to follow

**llama.cpp at the pin** (`3e4dfa8` + `contrib/llama.cpp` 0001-0056; none of
the patches touch the DFlash files). All `code`.

- **The architecture** `dflash` (`src/models/dflash.cpp`).
  - Hparams `:7-119`: `dflash.block_size`, `dflash.conv_kernel_size` /
    `conv_group_size` / `selector_rank` / `selector_top_k` (DFlash2),
    `dflash.target_layers` (required, `:50-52`), optional sliding window
    (`:111-116`), M-RoPE degenerate sections for M-RoPE targets (`:29`).
    `n_embd_inp_enc` = n_target_layers x n_embd.
  - Tensors `:121-334`: `fc` (fuses the concatenated target features),
    `enc.output_norm` (the reference's `hidden_norm`), per layer Q/K/V/O with
    q/k norms, the SwiGLU FFN, and for DFlash2 `blk.N.attn_conv_base/proj`,
    `blk.N.ffn_conv_base/proj`, `selector_predecessor`, `selector_successor`,
    `selector_hidden`. `token_embd` and `output` are optional: a draft without
    them borrows the target's through `ctx_other` (`:918-926`, `:1052-1061`;
    `src/llama-context.cpp:246-253` keeps `ctx_other` for `dflash`).
  - The decoder graph is dual-mode by batch type (`:803-1106`):
    - an **embedding batch** of target features is projected by `fc` + norm
      and written as K/V into the drafter's own cache at the target
      positions (`:844-916`, "KV injection");
    - a **token batch** `[id_last, mask x n]` (the noise block) runs the
      layers with non-causal attention over the injected context plus the
      block (`:953-1045`), the target's LM head (`:1063`), then for DFlash2
      the selector (`:1101-1102`).
  - DFlash2 adds two-tap grouped dynamic convolutions around attention and
    FFN (`build_dflash2_conv`, `:483-546`) and the selector
    (`build_dflash2_selector`, `:550-639`): top-16 candidates a position from
    the logits, pairwise transition scores `successor . (predecessor * gate)`
    plus the candidate's logit, packed into the `nextn` output row.
- **The drafter loop** (`common/speculative.cpp`,
  `common_speculative_impl_draft_dflash`, `:981-1429`).
  - Setup `:1024-1139`: reads `dflash.block_size` (clamps n_max to
    block_size - 1, `:1089-1097`), `dflash.attention.causal`, the mask token
    (`llama_vocab_mask`), DFlash2 when `selector_top_k > 0`; enables the
    target's layer inputs (`llama_set_embeddings_layer_inp`, `:1131-1134`),
    the drafter's unmasked `nextn` rows (`:1137`) and non-causal attention
    (`:1138`).
  - `process()` `:1173-1274`: after **every** target decode (prompt chunks
    and verifies alike), gathers the target's layer-input rows of the batch
    into `[n, 5 x n_embd]` f32 and decodes them into the drafter as an
    embedding batch at the same positions (4 positions a token for an M-RoPE
    draft).
  - `draft()` `:1276-1424`: one drafter decode for all drafting sequences, a
    block of `n_max + 1` tokens each at `pos0..pos0+n_max`; DFlash2 walks
    the lattice greedily, row i scored from the previous pick's
    transition row, and stops below `p_min` on the softmax over the 16
    scores (`:1335-1366`).
  - The draft context (`:2655-2700`): `n_ctx` = the target's (`:2663`),
    `ctx_other` = the target (`:2668`), its own KV cache types; outputs
    `n_max + 1` a sequence (`:2616-2630`).
- **The server's cycle** (`tools/server/server-context.cpp`): draft, then
  drop the drafter's block cells from `pos0` (`:3444`); the target decodes
  `[id_last, drafts]`; `process()` injects all of its rows (`:4256`);
  after the walk both contexts are cut at the accepted end (`:4529`);
  prompt checkpoints stash the drafter's partial (SWA) state with the
  target's (`:2597`, `:2722`).
- **The target side.** `qwen35` exposes every layer's input
  (`src/models/qwen35.cpp:160`); `llama_context::extract_layer_inputs`
  (`src/llama-context.cpp:2488-2518`) copies the enabled layers' rows to host
  per ubatch; the staging API is `src/llama-ext.h:133-150`.
- **The converter** (`conversion/qwen.py:746-948`, `DFlash2DraftModel`):
  needs `--target-model-dir` for the vocabulary and M-RoPE detection; writes
  `target_layers` as the checkpoint's ids + 1 (`:825`; layer input i = the
  output of layer i - 1, the reference's `hidden_states[id + 1]`); maps the
  conv and selector tensors (`gguf-py/gguf/tensor_mapping.py:1574-1599`).

**The model** `incoai/Qwen3.8-27B-DFlash2` (mirror `z-lab/Qwen3.8-27B-DFlash2`;
reference forward `z-lab/dflash`, `dflash/model.py`). Config (read from the
checkpoint on the dev host, `code`): 5 layers, hidden 5,120, 32 heads / 8 KV
heads of 128, FFN 17,408, rope theta 1e7, all layers sliding window 2,048,
non-causal, block 8 (7 drafts), mask token 248,070, selector rank 256 /
top-16, conv kernel 2 / group 16, target layers 5, 19, 33, 47, 61 of 64,
vocabulary 248,320; no embeddings or LM head of its own; 1.924 B parameters.

**Measured effect.**
- `paper` (the model card; SGLang, one H200, 7 drafts, sampled): acceptance
  length DFlash2 4.10-5.46 against the target's MTP chained to 7 drafts
  3.74-5.02 (HumanEval 4.39 / 3.91, MBPP 4.79 / 3.99); concurrency 1:
  3.11-3.43x autoregressive against MTP's 2.20-2.59x.
- `measured-here` (the retired OpenVINO engine, 2026-09, B60, int4 drafter
  RTN g64, f16-rescaled; tag `openvino-history`, README and DESIGN §7.0.2r,
  §7.0.2z, §7.0.2ag): 44.8 t/s at 3.13 tokens per verify against 24.0 plain
  and 33.0 with the MTP head drafting **one** token; offline acceptance 3.39
  code / 3.76 prose against a shuffled-feature null of 1.23 / 1.10; at 77k
  18.8 t/s against plain 15.3; the drafter on the A770 39.8 t/s; blocks of
  12/16 and a Viterbi selector lost; greedy selector the default.
- The libllama MTP the gate compares against (`measured-here`,
  `mtp-cycle-wall.md`): agent, 5 drafts, 40,960-row draft vocabulary, 52.6 t/s
  at 61.0 % accepted (~4.05 tokens a verify) on the acceptance task's
  prompt.

## Gate

On the record before the work; the comparison is on the B60, one window,
the same binary, prompts and sampler for every arm.

- **Answers (the answer-level bar, `CLAUDE.md`).** DFlash2 against the
  served MTP 5 arm: the capital, the needle (40 % depth of the deepest
  prompt), the acceptance task **10/10 target, 9/10 floor** greedy (10 runs
  sampled at the agent's settings reported); KL of the verify rows against
  plain decode at most 0.03 nats above MTP 5's against the same reference,
  argmax agreement down at most 1 point. Byte cells stay tripwires.
- **Decode (the target phase).** DFlash2's served decode beats MTP 5's
  outside the run-to-run spread **at short context (~1k and ~4k prompts) and
  at depth (~130k, and the deepest prompt the chosen context holds)**.
- **Prefill (the other phase).** At ~55k and ~138k within the run-to-run
  spread of MTP 5's (0.8.0: 695.3 / 491.9 t/s).
- **Memory.** Peak VRAM below the card's idle free 24,433 MiB with GTT flat
  through the deepest prefill and decode (fdinfo every second); a verify at
  depth in milliseconds.
- **Context.** Whatever served context the operator picks (open question 1)
  is the context of both arms' deep cells.

## Current state

- **Engine (code):** arcint drafts only with MTP (`--llama-mtp`,
  `src/exec/llama_spec.cpp`); no DFlash code. The `LlamaSpec` interface
  (`src/exec/llama_spec.h`) is general: decode/decode_multi,
  draft/draft_multi, accept, seq_rm on both contexts, a per-lane carried
  state for checkpoints, KV windows. MTP-specific: the carried hidden row
  and its pending-row bookkeeping, `DraftHead`, the `n_rs_seq` and
  `n_outputs_max_per_seq` sizing from `--llama-mtp`.
- **Drafter GGUF: none usable exists.** The dev host's two DFlash GGUFs are
  other models' drafters (`code`, GGUF metadata read 2026-10-16):
  `dflash-kquant.gguf` is a MuseGlimmer drafter (hidden 6,656, vocabulary
  202,048, target layers 2/14/26/38/50, SWA 2,048, Q4_K/Q6_K, 1.62 GB);
  `laguna-s-2.1-DFlash-BF16.gguf` is Poolside Laguna's (hidden 3,072,
  vocabulary 100,352, six layers). The **HF checkpoint** of
  `incoai/Qwen3.8-27B-DFlash2` (bf16 `model.safetensors` 3.85 GB,
  `config.json`, model card) is on the dev host from the OpenVINO export,
  and so is the target's HF directory (config, tokenizer). Its 81 tensors
  match the pin's `dflash` loader one for one (conv base `[2,2,5120]` =
  `{n_embd, kernel, 2}`, conv projection `[1280,5120]` = `2 x kernel x
  groups`, codebooks `[248320,256]`, hidden projection `[256,5120]`). The
  conversion needs torch on CPU (none of the dev host's venvs has it).
- **Memory (arithmetic, not measured):** see Design, "Memory". It does not
  fit at 245,760 tokens.

## Design

### Flags

- `--llama-dflash-gguf FILE`: the converted drafter. Loads it as a second
  model (as `--llama-mtp-gguf`), every tensor on the target's card.
  `--dflash`, the OpenVINO engine's flag, stays refused by name (0.8.0).
- `--llama-dflash N`: drafts a verify, 1 to block_size - 1 (7); the drafter
  decodes a block of N + 1 every cycle (llama.cpp's `n_max + 1`, one graph
  shape) and the step's budget truncates the result, as
  `common_speculative_draft` does. Default 7.
- `--llama-dflash-min-p P`: the selector's stop (llama.cpp's `p_min` on the
  16-way softmax, `:1349-1357`); 0 drafts N always.
- Refusals: with `--llama-mtp` (one drafter a context; the MTP layer is then
  not loaded: `load_mtp` false), with `--llama-mtp-vocab`, under
  `--llama-layer-split` (open question 4), and when the GGUF's
  `general.architecture` is not `dflash`, its `n_embd` x targets does not
  match the target's, or its tokenizer differs in size.

### Loading (`backend_llama.cpp`)

1. Target context as today, with `n_rs_seq = N` (the rollback depth: a
   verify of N + 1 rows rolls back up to N),
   `n_outputs_max_per_seq = 1 + N`.
2. `llama_model_load_from_file(FILE)`, `n_gpu_layers = 999`.
3. `make_llama_dflash(...)` (new, `src/exec/llama_dflash.cpp`) creates the
   draft context as llama.cpp does: `n_ctx` = the target's (the SWA cache
   then holds `pad(2048 x n_seq + n_ubatch, 256)` cells,
   `src/llama-kv-cache-iswa.cpp:73`), `ctx_other = ctx_tgt` (embeddings and
   LM head shared), `kv_unified` as the target's, f16 K/V (llama.cpp's draft
   default), `n_ubatch` min(target's, 512), `n_seq_max` = lanes,
   `n_outputs_max` = lanes x (N + 1); `llama_set_causal_attn(ctx_dft,
   dflash.attention.causal)`; `llama_set_embeddings_nextn(ctx_dft, true,
   false)`; `llama_set_embeddings_layer_inp(ctx_tgt, id, true)` for each
   `target_layers` id. Reads `dflash.block_size`, the mask token and
   `llama_model_dflash_selector_top_k`.

### How the target's hidden states reach the drafter

Through host memory, as llama.cpp does: after a target decode,
`llama_get_embeddings_layer_inp(ctx_tgt, id)` for the five ids (each `[n x
5120]` f32), interleaved per token into `[n x 25600]`, added with
`llama_batch_ext_add_embd` at the token's position (`{p, p, p, 0}`: the
draft carries M-RoPE sections because the target does), decoded into
`ctx_dft` in chunks of its ubatch. Cost: 100 KiB a token each way
(~0.06 ms over the B60's 14.3 GB/s link per 8-row verify; ~1 % of a
2,048-token prefill batch, arithmetic). A device-side handoff would need a
llama.cpp patch and is not in the reference.

### The cycle (`LlamaDflash : LlamaSpec`)

arcint's step order (`step_combined`: `draft_multi`, then `decode_multi`,
then the walk, `accept`, `seq_rm`) already matches the server's.

- `decode_multi(parts)`: one target decode of all parts (as today), then
  `process()` for each part: inject the part's rows at their positions.
  Prompt chunks inject every row (llama.cpp's mechanism; open lever below).
- `draft_multi(parts)`: one drafter decode of `[id_last, mask x N]` at
  `pos0..pos0+N` per lane, no logits requested; per lane the greedy lattice
  walk over `llama_get_embeddings_nextn` rows 1..N (top-k candidate ids then
  the k x k transition scores, the `p_min` stop), truncated to the step's
  `n_max`; then `llama_memory_seq_rm(ctx_dft, seq, pos0, -1)` so the block's
  K/V leave before the verify's features land there. The walk is a pure
  function (`src/exec/dflash_walk.h`, unit-tested on synthetic lattices,
  like `verify_walk.h`).
- `accept()`: nothing (llama.cpp's is a no-op).
- `seq_rm(seq, from)`: both contexts, as `LlamaMtp::seq_rm` minus the
  carried-row bookkeeping. The verify walk and the target's rollback through
  its per-token GDN snapshots (`n_rs_seq`) are unchanged: they are not
  MTP-specific.
- `reset()`: nothing to reset (the drafter carries no row).
- First draft after a prompt: arcint prefills all but the last prompt token;
  the drafter then holds features through `pos0 - 1` and the anchor is the
  last prompt token at `pos0` — the server's state after a prompt
  (`pos0 = pos_next`, `id_last = sampled`).

### Checkpoints

llama.cpp stashes the drafter's partial state (its SWA window: the K/V of up
to ~2,560 cells, `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` writes only SWA caches,
`src/llama-kv-cache.cpp:2280`) with each prompt checkpoint and loads it at a
restore. arcint does the same: the `carried_row` / `set_carried_row` pair of
`LlamaSpec` generalizes to opaque state bytes (`llama_state_seq_get_data_ext`
/ `set_data_ext` on `ctx_dft`); MTP keeps its row through the same calls.
Price: ~50 MiB of host memory a checkpoint at f16 (2,560 cells x 5 layers x
2 x 1,024 x 2 B), up to 32 a lane (open question 3). Restored drafter state
is copied data: exact (DESIGN §3.4 rule 2).

### Lanes, KV types, MTP, the context policy

- **Lanes:** one drafter decode drafts every lane's block (`n_blocks =
  n_seqs_unq` in the graph); `set_kv_windows` passes the lanes' windows to
  the drafter where its memory takes them (it may not: an iSWA cache; then
  logged as for MTP).
- **KV:** the target keeps `--llama-kv`; the drafter's cache is f16 (52 MiB).
- **MTP:** mutually exclusive in one process (flags above).
- **DESIGN §3.8:** the drafter's 2,048 window is its trained attention, not
  a context policy: the target sees every token and the verify decides every
  emitted token. No invariant is touched.

### The GGUF

`convert_hf_to_gguf.py <DFlash2 dir> --target-model-dir <Qwen3.8-27B HF dir>
--outtype bf16` at the pin, then `llama-quantize` to Q4_K (Q8_0 as the
second arm) **keeping the two selector codebooks F16 or Q8_0**: they are read
by `GET_ROWS`, which the OpenCL backend runs for F32/F16/I32/Q8_0 only
(Q4_0 not under the SoA layout); a K-quant codebook would run on the CPU
with a 127 MB copy a draft. Every matmul width divides 256 (25,600; 5,120;
4,096; 17,408). The recipe goes into `docs/model_requirements.md`.

### Memory on the B60 at 245,760 tokens (arithmetic from file sizes and the record)

Baseline: the agent's peak 24,195 MiB at a 236,938-token prompt with MTP 5
(`agent-context.md`, 2026-10-12; 24,293 in the current brief) against
24,433 MiB free: a 140-238 MiB margin.

| item | MiB | basis |
|---|---|---|
| **freed:** MTP layer weights (`blk.64.*`) | -276 | the GGUF, `code` |
| draft vocabulary head (40,960 Q6_K rows) | -164 | arithmetic |
| MTP context KV (1 layer, 8:4, 1,664 B/token) | -390 | arithmetic; 338 at 212,992 matches the record's 208-338 |
| MTP context compute buffer | -(50-100) | estimate |
| **added:** drafter weights, Q4_K (Q8_0) | +963 (+1,820) | 1,796 M matmul parameters |
| selector codebooks F16 (Q8_0) | +242 (+129) | 2 x 248,320 x 256 |
| drafter KV, SWA, f16 | +52 | 2,560 cells |
| drafter compute buffer | +(100-250) | estimate: a 512-row injection ubatch, 8 x 248,320 logits |
| target: 2 more GDN snapshots (N = 7 vs 5) | +299 | 149.6 MiB each, `agent-context.md` |
| target: five layer inputs kept live per ubatch | +50 | 5 x 5,120 x 512 x 4 B |
| **net, Q4_K drafter, F16 codebooks, N = 7** | **+0.8 to +1.0 GiB** | |
| net, same, N = 5 | +0.5 to +0.7 GiB | |
| net, Q8_0 drafter | +1.7 to +1.9 GiB | |

**Verdict: it does not fit at 245,760.** At the record's slope (+442 MiB of
peak per 16,384 tokens, `measured-here`) the agent gives up two steps
(212,992) for a Q4_K drafter, N = 5 or 7 — three if the compute buffers
land high — and four to five (180,224-163,840) for a Q8_0 drafter. The
load probes in phase 1 replace these numbers; the context is the
operator's decision (question 1).

### Integration size

| file | change | LOC (rough) |
|---|---|---|
| `src/exec/llama_dflash.cpp` (new) | the `LlamaSpec`: context, injection, block draft, walk, seq_rm, state | 450-550 |
| `src/exec/dflash_walk.h` (new) | the lattice walk, pure | 50 |
| `src/exec/llama_spec.h` | `make_llama_dflash`; carried state as bytes | 30 |
| `src/exec/llama_spec.cpp` | the MTP's carried row through the byte state | 20 |
| `src/exec/backend_llama.cpp` | load, sizing, refusals, checkpoint state, status | 100-150 |
| `src/config.h`, `src/config.cpp` | three flags, refusals, help | 60 |
| tests (`test_config`, `test_dflash_walk`, equivalence and acceptance cells) | | 200-250 |
| tools: drafter parity against the HF reference forward | | 150 |
| docs (`docs/llama-engine.md`, `docs/switches.md`, `docs/model_requirements.md`, DESIGN §3.5/§7/§8.5, CHANGELOG, the agent unit) | | — |

About 1,100-1,300 lines; no llama.cpp patch expected. A patch follows only
if a drafter op runs on the CPU on OpenCL (count `GGML_SCHED_DEBUG` splits
first): the selector's 4D f32 batched matmul, `TOP_K` over 248,320,
`PAD` with a left pad, `CONCAT`, the I32-to-F32 cast and `FILL` are each
listed as supported by `ggml_opencl_supports_op`, unmeasured.

### Red cases (each must be able to fail; run the red arm first)

1. **Walk:** a synthetic lattice whose best path differs from the per-row
   argmax and from a walk indexed by the wrong predecessor; the unit test
   fails on both mutants.
2. **Features matter:** inject the layer inputs of the wrong layers (the
   checkpoint's ids without the converter's + 1, then shuffled rows):
   acceptance must fall toward the OpenVINO record's null (1.1-1.2 tokens a
   verify).
3. **Rollback:** skip the drafter's `seq_rm` of the block, then of the
   rejected rows: acceptance must drop; a debug check that the drafter's
   `pos_max` equals the target's after every cycle must fire.
4. **Lossless:** DFlash2 greedy against plain greedy and against MTP 5 on
   the acceptance task and the KL window (the gate); the mutant "accept the
   draft token without sampling" must fail the task or the KL.
5. **Depth:** acceptance at 70k and at the deepest prompt is not zero (the
   OpenVINO drafters lost every draft past 65,504 positions to an f16
   rotary; here positions are f32 in `ggml_rope_multi`, `code`).
6. **The window edge:** prompts of 2,047, 2,049 and 5,000 tokens draft (the
   OpenVINO head stopped at its 2,048-row state).
7. **Parity:** the converted drafter on llama.cpp's CPU backend against the
   reference forward (`z-lab/dflash`, f32) on captured target features:
   the top-16 candidate sets and the walked path agree on a fixed set; the
   OpenCL drafter against the CPU drafter likewise (and `test-backend-ops`
   for each op it uses, with the right `-b` name: a wrong one skips all and
   prints OK).
8. **Checkpoint restore:** an edited message resumes from a checkpoint with
   the drafter state restored; the mutant without the drafter state shows
   lower acceptance on the first 2k tokens, the answers stay right in both.
9. **Config:** the refusals above, each with a test.

### Measurement plan

All on the B60 in one card window per phase (agent unit stopped and
restored by name; Flash-Next and writer not running). Every cell names card,
depth, KV types, binary sha and drafter quant.

- **Phase 0, the reference itself** (before any arcint code): the pin's
  `llama-server` (this tree's llama.cpp with its patches, Adreno kernels
  off) with `-md` the converted drafter, `--spec-type draft-dflash`,
  `--spec-draft-n-max 7`, against the same server with `draft-mtp` n-max 5:
  acceptance length and decode on the acceptance task's prompt and a 4k
  prompt, greedy, Q4_K and Q8_0 drafters; the drafter op splits counted.
  ~1 h. This measures the reference's mechanism here before arcint's build
  and answers the acceptance-length question below.
- **Phase 1, load probes**: VRAM after load and peak after one answer for
  the arms below at 212,992 and 229,376 (and 245,760 to see it refused or
  thin), fdinfo GTT.
- **Phase 2, served** (arcint, same window): arms plain, MTP 5 (+ the
  40,960-row vocabulary, the unit's flags), DFlash2 N = 5 and 7, `p_min` 0
  and the best of {0.3, 0.5} on N = 7; prompts: the acceptance task (~1k),
  4k, 55k, 130k, the deepest the chosen context holds; greedy (and the
  agent's sampled settings on the task). Per cell: decode t/s, tokens a
  verify, draft ms and verify ms a cycle, prefill t/s, peak VRAM / GTT. The
  acceptance task 10 runs on the winning arm, KL window (dense-ref, 16 x
  512) for MTP 5 and DFlash2.
- **Budget:** a deep cell is ~25 min of prefill (157 t/s at 237k); two arms
  at depth plus the short cells fit the 2 h budget if the deepest cell runs
  once per arm; a row above 2 h is TBD.

### What decides it (arithmetic, to be replaced by phase 0)

At short depth an 8-row verify costs ~1.21 plain steps on the B60
(`measured-here`, 0009) — ~62 ms on a ~51 ms step — and a drafter pass reads
~2 GB (1 GB drafter at Q4_K, the 1.04 GB Q6_K LM head for 8 rows): ~5-7 ms.
At ~69 ms a cycle DFlash2 needs **~3.6 tokens a verify** to match MTP 5's
52.6 t/s. The OpenVINO record's 3.13 (int4 RTN, f16-rescaled drafter) would
lose (~45 t/s); the model card's 4.1-5.5 would win (~60-80 t/s). At depth
the drafter's cost stays flat (2,048 window) while MTP's five head steps
attend the whole context, so the margin grows with depth.

## Open questions for the operator

1. **Context.** DFlash2 does not fit beside a 245,760-token agent
   (arithmetic: +0.5 to +1.0 GiB against a 140-238 MiB margin). Options:
   (a) the agent at 212,992 (Q4_K drafter, N = 5 or 7; -32,768 tokens);
   (b) 229,376 with N = 5 if the probes land at the low end; (c) the
   drafter on the A770 (the OpenVINO record: 39.8 against 44.8 t/s on the
   same card), which costs the coder unit ~2.3 GB there (the drafter plus its
   own LM head copy) and 100 KiB a token over the 1.83 GB/s chipset link
   (~13 s on a 237k prefill, ~1 %); (d) MTP 5 at 245,760 kept as the deep
   configuration, DFlash2 at a smaller context as the default. Price of
   each: phase 1's probes and phase 2's deep cells.
2. **Drafter quantization.** Q4_K (~1.2 GiB with codebooks) or Q8_0
   (~2.0 GiB, four to five more context steps). The upstream drafter
   serves in bf16; the OpenVINO record's int4 RTN reached 3.13 tokens a
   verify. Phase 0 prices both.
3. **Drafter state with checkpoints** (llama.cpp's mechanism): up to
   ~50 MiB of host memory a checkpoint, ~1.6 GiB at 32 a lane. Building it is
   the default; scoping it out costs acceptance on the first ~2k tokens after
   a resumed edit (red case 8 measures it).
4. **Layer split.** `--llama-dflash-gguf` under `--llama-layer-split` is
   refused in this campaign: the agent runs on one card, and no DFlash
   drafter for Flash-Next is known. llama.cpp supports a draft device
   (`-devd`); building it is a separate campaign if wanted.
5. **Prefill injection of the window only** (a lever beyond the
   reference): the drafter keeps 2,048 positions, so injecting only a
   prompt's last ~2,560 rows gives the same drafter state with ~1 % less
   prefill work; llama.cpp injects every row. Build only with your yes and a
   red case proving the identical state.

## Where it lives

- Planned: `src/exec/llama_dflash.cpp`, `src/exec/dflash_walk.h`,
  `src/exec/llama_spec.h`, `src/exec/backend_llama.cpp`, `src/config.*`,
  `tests/test_dflash_walk.cpp`, `tests/equivalence/run.sh` (a DFlash2
  section), `tests/acceptance/cells.json` (`agent-dense` with DFlash2),
  `contrib/systemd/arcint-agent.service`.
- The reference: llama.cpp at the pin, `src/models/dflash.cpp`,
  `common/speculative.cpp` (`draft-dflash`), `tools/server/server-context.cpp`,
  `conversion/qwen.py`.
- The OpenVINO engine's drafter, exporter and probes: tag `openvino-history`
  (`tools/export_dflash.py`, `src/core/dflash_select.*`,
  `docs/dflash-pairing-probe.md`).
