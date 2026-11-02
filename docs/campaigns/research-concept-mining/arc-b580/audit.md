# Pass 4 — the kept candidates against arcint's own code (2026-11-02)

Scope: the top-10 candidates except those that trade answer quality by design. #6 changes the
text only under sampling (audited for the greedy path); #10 changes the draft's proposal, not the
answer (the target verifies every draft). Nothing was built or run. `arcint` = the workspace at
`qfndev` 0fb156e; `llama` = llama.cpp at the pin `3e4dfa8` with the served series 0001-0079
applied (gaps 0031/0055/0061/0073/0076/0078 intended), extracted at `~/research/arc-b580/llama-served`.
Line numbers are that patched tree's. The fork is at head `e04b0cd4`.

| item | arcint has | class | units | prize (`paper`) | cost |
|---|---|---|---|---|---|
| 1 int8-DPAS Q-K + 1-3 row staged DPAS | **partly**: GQA-DPAS from quantized cache exists (0015/0016/0017), f16 DPAS, 4+ rows; int8 Q-K lacks | rounding-only | agent, Flash-Next, coder | 16K 1 tok 266->123 us; 48K 1 tok 433->244; MTP +4-7 % at 115K | med-high |
| 2 GDN rollback by replay | **lacks**: per-token snapshots in the cache (0050) | exact | all three | 432 MiB at n_rs_seq 4; bit-identical state | medium |
| 3 MTP head prompt pass K/V-only | **lacks**: full MTP graph over every prompt row | exact | all three | long prompts ~5 % | low-med |
| 4 round-trip discipline (sync trim, rotations on device, one id read) | **lacks** all three | exact | agent, coder, Flash-Next | 2 syncs + ~272 KB per graph; 0.2 ms per small D2H read; ~2 % | low |
| 5 GDN state gather folded | **lacks**: `GET_ROWS` gather into a temp | exact | all three | +3.8 % TG; dirty temp slows the next GEMV ~15 % | low-med |
| 6 draft-depth cutoff | **lacks** | answer unchanged at greedy; under sampling by design | all three | +85 % at 0 depth -> -27 % at 64k; 101/48/37 vs 101/45/27 | low |
| 7 two-pass top-k + direct selection | **lacks** | exact | agent, Flash-Next, coder | 550 -> 9.6 us per draft token; +1.9 % decode with MTP | low |
| 8 hybrid fusions: joint q/k l2_norm, bf16 48-row gate matvec | **partly**: raw gates (0051), conv/silu (0049), same-input? | exact | all three | 2B +7.7 %, 27B +2.6 % (Metal bundle); bf16 48-row 10.5 -> 4.4-7.3 us | low |
| 10 trimmed draft head | **partly**: a compact 40,960-row draft vocabulary exists | exact at the answer | agent, coder | 7.6x fewer head bytes, 90.3 -> 94.8-95.8 t/s | low |
| 11 q4_0 small-batch XMX GEMM | **partly**: 0009 few-column XMX up to 8 columns | rounding-only | coder/agent if n-gram | 16 columns 292 -> 89 us | not priced |
| 12 MMVQ chunk max | **partly**: 0009's row range | rounding-only | >32-column verify only | rename 79 -> 186 t/s | not priced |

Quality filter (operator, 2026-10-29): exact and rounding-only in scope; answer-changing levers
listed, not audited. #6 and #10 are listed with the reason.

---

## 1. int8-DPAS Q-K straight from the quantized cache, plus a staged DPAS kernel for 1-3 rows

- **What arcint already has** (`code`): a GQA decode/verify attention on the XMX units reading a
  quantized cache, one work-group per KV head and split, every query head of the group and every
  row from one read: `llama/ggml/src/ggml-opencl/kernels/flash_attn_gqa_dpas.cl:10-22` (the grid
  comment) and `:119` (the kernel), built from patch 0016 and extended to head size 128 by 0017;
  the quantized loader `:95-107` (`load8`/`load16`: q8_0 and q4_0 blocks to half8).
  **Yes, a GQA group shares one K/V read** (`:10-22`, row `m = hk * G + m % G`), and the dispatch
  starts it at 4 rows (`llama/ggml/src/ggml-opencl/ggml-opencl.cpp:23186-23200`,
  `GGML_OPENCL_FA_GQA_MIN_ROWS` default 4). 1-3 rows take 0015's split ALU kernel
  (`ggml-opencl.cpp:23173-23177` and `:23197-23206`). The program is built with
  `-cl-intel-256-GRF-per-thread` unconditionally (`:4876`).
- **How the build differs from the fork's** (`code`): the fork's Q-K is an int8 x int4 DPAS on the
  raw q4_0 bytes (`arc-b580/ggml/src/ggml-sycl/fattn-dec.cpp:688-696`): Q int8 with one RNE scale
  per row per 32-dim block, the key's 16-byte block XOR 0x88 as the B operand, one key per lane,
  int accumulator scaled by `dq[row] * dk[key]`. Ours dequantizes the K blocks to f16 and runs
  `intel_sub_group_f16_f16_matrix_mad_k16` (`flash_attn_gqa_dpas.cl:95-101`). Our own measurement
  of the alternative is the fork's premise: direct quantized K made **every sub-group convert the
  same blocks** (patch 0018's record: 6-row q8_0 at 32k 47.7 -> 52.6 t/s with K staged; direct
  30.7); the fork removes exactly that conversion by making the raw bytes the DPAS operand.
  Ours has no `prefetch` of the next phase's rows; a `GQA_PF` (next V tile into registers) was tried
  and left off (0018: "8 alone gains 7 %, but with GQA_S_SPLIT or at 4, IGC falls back to a rolled
  loop"). The one-wave split count is ours as a host constant (`kv_per_split` 512, cap 256 splits).
- **Quality class: rounding-only.** The int8 Q rows add one quantization rounding to Q.K; the fork's
  KLD is unchanged within error and its outputs identical; the accumulator is integer, scaled per
  block. Judged at the answer-level bar.
- **Units**: agent (B60, 6-row verify and 1-row decode, head 256, q8_0:q4_0), Flash-Next (B60,
  4-row verify, head-size 256 layers + QSA), coder (A770, 5-row verify; sub-group 8 on the A770 —
  the int8 DPAS form must be built for sub-group 8/16 both, 0016/0017 already build both).
- **Measurement that prices it**: B60, agent flags, q8_0:q4_0, one attention layer in
  test-backend-ops perf at 16K/64K/138K × 1/6 rows, then served decode. Bar: the 6-row call's per-
  layer time at 131K below 0016's 4.83 ms and the 1-row call below 0015's 1.91 ms, answers right,
  KL within the bar.
- **Implementation sketch** (new patch 0080): a `KT`-selected Q-K branch in
  `flash_attn_gqa_dpas.cl` — quantize the 8-row tile's Q to int8 into SLM with one RNE scale per
  32-block; read the key's block as `uchar16`/`uint4` XOR, feed `intel_sub_group_i8_i4_matrix_mad_k32`
  (i8 x i8 for a q8_0 cache, i8 x i4 for q4_0); scale the integer S by `dq*dk` after the DPAS; keep
  P.V on f16 DPAS. A 1-row instantiation with K/V staged (`GQA_ST`), host dispatch at
  `ggml-opencl.cpp:23186` for `n_q >= 1`. **Red-first tests**: (a) `-DBK=24`-style nibble fault
  fails every new case at 1-6 rows; (b) an int8-QK selftest comparing the DPAS S against the f16
  DPAS S on the same 16-key tile, max abs error below 1e-3; (c) the 0016 geometry cases (4-7 rows)
  plus 1/2/3 rows at KV 1,013 and 4,096 for q8_0, q8_0:q4_0.

## 2. GDN rollback by replay

- **What arcint already has** (`code`): per-token snapshots written into the recurrent cache by
  patch 0050 (`llama/ggml/src/ggml-opencl/ggml-opencl.cpp:10862` selects the GDN nodes; the
  snapshot copy fusion is `fuse_0050 & 1`), and the rollback index in the memory module
  (`llama/src/llama-memory-recurrent.cpp:200-215`: a per-token `rs_idx` bounded by `n_rs_seq`).
  `llama/src/models/delta-net-base.cpp:521-545` writes K = n_rs_seq + 1 conv snapshots a layer a
  verify; `:565` is the `keep` branch for the state.
- **How the build differs** (`code`): the fork appends `GGML_OP_GATED_DELTA_NET_REPLAY`
  (`arc-b580/ggml/include/ggml.h:599`, `:2649`), decides replay before allocating the tensors and
  sizes S to 2 planes (`arc-b580/src/llama-memory-recurrent.cpp:75-106`), keeps the base in plane 0
  with pending records in plane 1 (`:811-861`), and leaves records for the last m tokens (`:1424`).
  The SYCL kernel has a replay prologue re-applying stored records with the step's own update and
  an epilogue storing the base plus one record per later token
  (`arc-b580/ggml/src/ggml-sycl/gated_delta_net.cpp:857-864`, `:973`). The CPU reference is
  `arc-b580/ggml/src/ggml-cpu/ops.cpp:10998-11134`: "a replayed record repeats exactly its state
  update (scale by g, then mad k * delta)". Ours writes whole states; the fork writes three small
  vectors and replays. The conv state keeps its snapshot planes in both.
- **Quality class: exact** (bit-identical state claimed and tested by the replay == snapshot chain
  test in test-backend-ops).
- **Units**: agent (MTP 5), Flash-Next (MTP 3), coder (MTP 4). The VRAM saving is per verify and
  per layer: 432 MiB on the fork's 48 GDN layers at n_rs_seq 4; our agent's MTP 5 has K = 6.
- **Measurement**: B60 Flash-Next IQ3_S MTP 3 at 4K/32K and agent MTP 5 at 138K, one fresh process
  an arm; peak VRAM from fdinfo, decode t/s, verify ms, KL on assistant tokens + the task battery.
- **Implementation sketch** (new patch 0081 + a llama.cpp patch): the op and CPU reference as the
  fork has them; an `emit` template flag in `gated_delta_net_intel.cl` (the blocked kernel emits
  `delta,k,g` per token); a replay prologue at the top of the kernel; the memory module's S
  allocation to 2 planes behind the replay decision. **Red-first tests**: feed a record with one
  wrong `g` and require the state to differ (the test must fail on the injected fault); the
  replay-vs-snapshot chain equality over 2/3/6 tokens; a save/restore round trip with pending
  records.

## 3. The MTP head's prompt pass computes only its K/V

- **What arcint already has** (`code`): the MTP graph computes the whole block for every prompt
  row: `llama/src/models/qwen35.cpp:498-665` builds embedding, hnorm/enorm, eh_proj, Q/K/V,
  attention, gated out, residual, FFN and the shared head over `n_tokens` rows; only the
  output-row gather is cut when masked (`:655-665`). MTP layers of the other archs inherit the
  same shape. Patches 0010 and 0067 cut the *outputs* rows, not the graph.
- **How the build differs**: the fork gates the graph after the cache store when `n_outputs == 0`
  and masked extraction (`arc-b580/src/models/qwen35.cpp:756-793`: `kv_only`), keeping the K/V ops
  identical on the same shapes (bit-identical cache rows; the unused Q projection is kept for <=8
  rows so the same-input tile matches).
- **Quality class: exact** — the retained rows are the full graph's.
- **Units**: all three, prompt side (the agent's long prompts, Flash-Next's prefill, the coder's
  18K prompts).
- **Measurement**: B60 agent and coder A770, an 18K corpus prompt, MTP on; TTFT and prefill t/s,
  then decode; the arm must show identical accept counters and KL 0 on the draft's K/V rows.
- **Implementation sketch**: in our MTP builder, after `cpy_k`/`cpy_v` (the builder currently
  keeps `build_attn` at `:611`), return early when `cparams.embeddings_nextn_masked &&
  n_outputs == 0`; keep the Q projection for `n_tokens <= 8` (our same-input fusion's tile depends
  on the largest member). **Red-first tests**: compare the draft's retained K/V rows against the
  full graph's (byte equality), and a full-vs-KV-only draft-acceptance equality over 3 prompts.

## 4. Round-trip discipline

- **What arcint already has** (`code`): none of the three. `llama/src/llama-context.cpp:1757-1762`
  (`synchronize`) and every getter calls it (`:5288`, `:5350`, `:5366`); the attention rotation
  matrices are host inputs (`llama/src/llama-kv-cache.cpp:1665-1687` `ggml_set_input`, no device
  allocation). The MTP path reads draft ids a step at a time (the draft trace record flushes after
  one synchronize, `llama-context.cpp:1764-1766`).
- **How the build differs**: the fork skips the wait when the context queued nothing since its last
  synchronize (`arc-b580/src/llama-context.cpp:825-848`), allocates the rotations once in a device
  buffer (`arc-b580/src/llama-kv-cache.cpp`, the cache's own buffer) and packs the per-step chain
  inputs into two tensors, one async copy for all draft ids (`arc-b580/src/llama-graph.cpp`,
  `src/models/qwen35.cpp`). Its measured facts: 2 split inputs + 2 synchronizes + 2 blocking copies
  (~272 KB) per attention graph; each small D2H read ~0.2 ms.
- **Quality class: exact** — the results were complete already; the copies carry the same bytes.
- **Units**: agent and coder (MTP with per-row sampling), Flash-Next (MTP 3).
- **Measurement**: B60 agent MTP 5 at 32K/138K; a host counter of synchronizes and blocking copies
  per verify round (the fork's sync-trace form); decode t/s; answers identical.
- **Implementation sketch**: a `queued_since_sync` flag in `llama_context`; allocate
  `attn_inp_k_rot`/`v_rot` in the cache's device buffer and return those tensors; concatenate the
  MTP draft ids into one I32 and read once. **Red-first tests**: a counter test that a getter after
  a completed decode issues no backend synchronize; the rotation tensors' buffer residency; draft
  ids equal to the per-step read.
- **Disposition (2026-11-02, package B)** (`code`): patch 0083 carries bits 1 and 2 as the fork
  has them, and bit 4 as the form of "one id read where there are several" that applies here
  (backend-sampled rows; not on arcint's served path today, `src/core/sampler.cpp:1107` samples
  host logits). The fork's draft-chain form does not apply: arcint's drafter runs one graph a
  step (`src/exec/llama_spec.cpp:615-635`) and the route trace's ids are already one batch
  (0069). `LLAMA_SYNC_TRIM` defaults 0 here (byte-identical off). Counts and the red cases are in
  `contrib/llama.cpp/README.md` 0083; the B60 served A/B is owed to a card window.

## 5. The GDN recurrent-state gather folded into the kernel

- **What arcint already has** (`code`): a gather. `llama/src/llama-graph.cpp:3959`
  (`ggml_get_rows(ctx0, states, state_copy)`) builds a temp of the live state rows; the kernel then
  reads that temp (`gated_delta_net.cl:82-89` takes `state_buf`), and patch 0050 folds only the
  snapshot *writes* into the cache. So ours pays the gather kernel and the 3 MB-per-layer temp
  exactly as the fork did.
- **How the build differs**: the fork's graph evaluator skips the `GET_ROWS` when its only
  consumer (optionally through a reshape) is the GDN state input, and records (cache base, ids, row
  stride) for that node; the kernel indexes `s_ids[seq]` directly (single sequence only, because
  multi-seq rows may alias). Ours passes a gathered tensor.
- **Quality class: exact** (the same bytes; 400 greedy tokens identical on the fork).
- **Units**: all three; the agent's 138K decode and verify.
- **Measurement**: B60 agent, MTP 5 at 138K; the GDN layer's gate/up GEMV time and decode t/s;
  arm A today, arm B fused; bar >=+2.5 % with identical answers.
- **Implementation sketch** (new patch 0082): pass the cache base pointer, the state row index and
  stride as kernel arguments; skip the gather node in the graph's evaluation (a host-side matcher
  in `ggml-opencl.cpp`, beside patch 0050's prepass). **Red-first test**: a kernel selftest that the
  indexed row equals the gathered row byte for byte; a multi-sequence case must decline the fold.

## 6. Stop drafting past a depth (greedy path audited)

- **What arcint already has** (`code`): no cutoff. `llama/common/arg.cpp` has the draft width but
  no depth; the server's `spec_process` is unconditional. The units run fixed K with the min-p
  stop.
- **How the build differs**: the fork gates on the batch's positions
  (`arc-b580/tools/server/server-context.cpp:3654-3657`) and on the prompt length (`:447`) with
  `--spec-draft-depth-max` (`common/arg.cpp:4123-4128`), default 0.
- **Quality class**: at greedy the target argmax is the same token either way, so the answer is
  unchanged; under sampling the text can diverge, by design. Audited only for the greedy path.
- **Operator decision (2026-11-02):** in scope. "If MTP becomes a burden at some point, it is ok to let go after that
  point. If Paris stays Paris but with a French accent, that's okay with me": a sampled text that diverges because
  drafting stops is acceptable; the right facts are the bar. Greedy stays identical. Still owed (`code`): confirm that
  arcint's verify keeps the target distribution under sampling.
- **Units**: all three at depth; the agent's 245k context is the extreme.
- **Measurement**: B60 agent MTP 5, Flash-Next MTP 3 at 8K/64K/138K greedy; arms off/24K/32K/64K;
  decode t/s, answers.
- **Implementation sketch**: a per-slot cutoff in the server's speculative hook and a sweep to set
  it; no kernel work. **Red-first test**: the greedy token stream before and after the cutoff is
  identical on a prompt shorter than the cutoff, and the verify hook is skipped past it.
- **Disposition (2026-11-02, package C)** (`code`): **the verify keeps the target's distribution exactly**, so
  the cutoff cannot change quality. `src/exec/verify_walk.h:36` samples each verified row with the
  sampler's state as of that token (history fed at `:49`) and keeps a draft only when the drawn token
  equals it (`:72`); call site `src/exec/backend_llama.cpp:2166`. Every emitted token is the sample the
  plain loop would draw at that point, and a mismatch ends the walk, so the sampler's stream advances
  once per emitted token, never over a dropped draft. That is sample-and-compare, not llama.cpp's
  rejection sampler; it needs no importance correction, and it is not a rejection-sampling
  approximation. Built as the arcint flag `--llama-mtp-depth-max N` (0 off = today, byte-identical),
  `/props mtp.depth_max`, the round's width from `src/exec/spec_depth.h` at the loop's live depth
  `pos0 + 1` (`backend_llama.cpp:2104`), so a shorter later request on the slot drafts again; the MTP
  context stays fed. Cells `tests/test_mtp_depth.cpp` (7, plus `/props` in `tests/test_props.cpp`),
  red cases a-e in the commit. **Not owed any more**: the distribution check. **Owed to a card window**:
  the B60 agent MTP 5 and Flash-Next MTP 3 at 8K/64K/138K greedy, arms off/24K/32K/64K, decode t/s
  and the answers; the coder (A770, MTP 4, 98K) at 8K/48K/90K -- the driver
  `tests/acceptance/cells/mtp_depth_sweep.py` (one server an arm, a needle a depth, the verify count a
  depth as the mechanism check) and the exact commands in the handover.

## 7. Sampler top-k on a long vocabulary

- **What arcint already has** (`code`): `llama/common/sampling.cpp:134-160` (`set_logits` fills
  `cur` with all `n_vocab` rows from `llama_get_logits_ith`), and the top-k sampler partial-sorts
  over it (`llama/src/llama-sampler.cpp:334`, `:1572`). Our MTP verify samples each row through
  this chain.
- **How the build differs**: the fork selects the k largest logits directly when every sampler
  before top-k is a no-op, repeating `std::partial_sort`'s heap steps so the entries and the order
  of equal logits are the same (`common/sampling.cpp:112-282`, `:742-753`); its GPU top-k is a
  two-pass merge over 256 groups (`ggml-sycl/ggml-sycl.cpp:3075-3228`).
- **Quality class: exact** (120,000 random arrays incl. ties/NaN/inf and server outputs agree).
- **Units**: agent (248,320-row head, 6 sampled rows a verify step), Flash-Next, coder.
- **Measurement**: B60 agent greedy + Hermes settings; decode t/s and ids.
- **Implementation sketch**: a direct-selection branch in `common_sampler_sample` before
  `set_logits`; if our drafter's top-k runs on the GPU, the two-pass merge beside it.
  **Red-first test**: the direct selection against `std::partial_sort` on random arrays with ties,
  NaN and inf, plus server-output id equality.
- **Disposition (2026-11-02, package B)** (`code`): the audit's "our MTP verify samples through
  this chain" is corrected by the code — the served path is arcint's own `Sampler`
  (`src/core/sampler.cpp:1107`), which builds the full 248320-entry array and partial-sorts it;
  llama's common library is not built by arcint (`LLAMA_BUILD_COMMON OFF`). Patch 0084 adds
  `llama_top_k_from_logits` (the fork's heap steps, exact on ties/NaN/inf), and the sampler calls
  it under `LLAMA_SAMPLER_FAST_TOPK=1` (default 0, old path byte-identical). The GPU two-pass
  merge is not taken: our top-k runs on the host, not on the OpenCL TOP_K kernel (the backend
  sampler's, not served). Exactness cell `tests/test_fast_topk.cpp`; red case in README 0084; the
  B60 A/B is owed to a card window.

## 8. Hybrid decode fusions

- **What arcint already has** (`code`): raw GDN gates folded into the kernel (patch 0051,
  `llama/ggml/src/ggml-opencl/ggml-opencl.cpp:9942-10070`: alpha/beta read raw, softplus/sigmoid
  inside), conv + silu and sigmoid+mul (0049), GDN snapshots (0050), GDN verify (0066). **Missing:**
  the fork's one `l2_norm` over the adjacent q/k head groups — ours runs two
  (`llama/src/models/qwen35.cpp:441-442`), the fork one over the joint view
  (`arc-b580/src/models/qwen35.cpp:548-554`). The gate-block fusion (9 nodes/7 launches -> 1) is
  partly ours in a different form.
- **bf16 48-row gate mat-vec**: our GDN gate projections are bf16 [5120 x 48] (the shapes the fork
  measured at 96 calls a token); whether our 2-6-column path takes a matvec kernel is not verified
  (`inferred` from the dispatch tables, not read). The fork's `mmvf.cu` fix takes the mat-vec for
  bf16 under 64 rows at 2-8 columns: 10.5 -> 4.4-7.3 us.
- **Quality class: exact** (the fork claims byte-identical outputs for its fusions; one l2_norm
  over a joint view computes the same normalization per row).
- **Units**: all three.
- **Measurement**: B60 agent and coder A770, decode and 6-row verify; encoded nodes per token and
  launches per step; A/B 2-3 fusions; bar >=+2 % bundled with identical answers.
- **Implementation sketch**: one `l2_norm` over a joint q/k view in our `qwen35.cpp`/`qwen35moe.cpp`
  GDN builder (both heads are adjacent and equally shaped in the conv output); a dispatch check for
  the bf16 48-row matvec. **Red-first tests**: the fused norm's q/k rows equal the two separate
  norms bitwise; the 0051 fusion tests unchanged; a launch count per step before/after.

## 10. The trimmed draft head (from the full vocabulary)

- **What arcint already has** (`code`/`paper`): a compact 40,960-row draft vocabulary for the
  coder and the agent (the earlier survey's record, `quality-instruments` and the compact head);
  the MTP head then projects onto the compact rows. What is missing vs the fork is (a) a per-model
  K chosen from a frequency corpus and (b) the scatter into a full-vocab row with `set_rows`, so
  downstream code sees the full row. If our head is already compact, the remaining lever is K and
  the scatter, not the pruning mechanism.
- **How the build differs**: the fork stores `blk.<n>.nextn.draft_head` (rows of output.weight for
  K frequent tokens plus special tokens) and `draft_head_ids`, computes those K rows and scatters
  with `set_rows` into a -inf full row; K = 32768 covers 99.6 % of a 4M-token corpus and reads 7.6x
  fewer bytes.
- **Quality class**: exact at the answer (the target verifies every draft); the proposal's support
  narrows by design.
- **Units**: agent and coder.
- **Measurement**: B60 agent MTP 5 on code prompts, K 16K/32K/64K; accept length, decode t/s, and
  teacher-forced KL of the target unchanged.
- **Implementation sketch**: a K sweep against our existing compact head and, if absent, the
  scatter. **Red-first test**: the draft logits at the K rows equal the full head's rows.

---

## What the audit changes in the ranking

- #3 (KV-only prompt pass) stays: our MTP graph runs the full block and the full head over every
  prompt row, which is larger than the fork's target because the fork's public config has a
  trimmed head.
- #4's rotation half is the cheapest exact item here: our rotations are host inputs uploaded per
  graph.
- #5 is a full match: ours gathers (`llama-graph.cpp:3959`) exactly where the fork removed it.
- #8 loses the raw-gates half (we have it, 0051) and keeps the joint q/k norm; the bf16 matvec
  needs a dispatch read before it counts.
- #9 (instrumentation) is the pricing tool for #1, #4, #5 and folds into their experiments.
- #10 is "partly": the compact head exists, so it is a K sweep, not a new mechanism.
