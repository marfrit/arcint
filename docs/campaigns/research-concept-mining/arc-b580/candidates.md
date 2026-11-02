# Pass 2 — ranked concepts from the `arc-b580` fork, for arcint (2026-11-02)

Merged from the records at 2-3 (43 commits into 20 concepts; fixes and reverts folded in). The
fork's numbers are `paper` for us; source readings are `code`; nothing was measured here. Scores
1-10 are this session's judgement against the three served units, with the pass-1 score in
parentheses. Status is the fork's final state.

## Corrections the top-10 re-read forced

- **`b0376aec39` (one read-back for all draft ids, +~2 %)** is a bit of the MTP *chain*
  (`LLAMA_MTP_CHAIN_TIGHT` bit 16), and the chain is **off by default with the public mtp-lean
  model** (`4798af11c8` measured it at -0.2..-0.8 % before the graph was reused; `e04b0cd4` says
  leave it unset unless the GGUF has a trimmed head). The 0.2 ms per small D2H read is the
  transferable fact; the +2 % figure only lands if a chain is adopted.
- **`e5a4d8a8f7` (DPAS decode attention) is opt-in for the whole run** (`GGML_SYCL_FA_DEC_DPAS=1`),
  not default-on. Its sub-levers are default within it: staging (`9cfffcb06d`), the 256 cap
  (`064461a0b4`), 5-token chunks (`081d86870f`), prefetch (`dc575f4ff5`), 256 register file
  (`1eb2307223`, 3+ tokens). `fb75ee9e56` (2D block loads) is a **measured negative** (128K 5
  tokens 1069 -> 1220 us) and stays off.
- **`befcc5c5f9` (no F16 scratch for the KV) is not a gap here**: arcint's 0015 reads q8_0/q4_0
  blocks into f16 SLM tiles directly and never materializes a full-cache f16 copy, on either the
  decode or the prompt path (`code`, below). The fork's cap `acd86a2b23` exists only because its
  oneDNN/MKL path does materialize one.
- **`36761482b0` (hybrid speculation fell back to whole-KV checkpoints) is a fix for
  `draft-simple`**, not MTP: `need_n_rs_seq()` omitted `DRAFT_SIMPLE`, and the draft context zeroed
  `n_rs_seq`. Our units use MTP; whether arcint's MTP verify ever takes the checkpoint path is an
  audit item, not a port.
- **`afe4c82d51`** was landed with `GGML_SYCL_TOPK_OLD=1` as the A/B switch, default on.
  **`a3586a10d8`** removed its own A/B switch in the same series, so the direct top-k selection is
  unconditional when it applies.

---

## Top 10

### 1. int8-DPAS Q-K straight from the quantized cache, plus a staged DPAS kernel for 1-3 rows — score 9 (p1 3)
- **Evidence**: `e5a4d8a8f7` (concept, opt-in), `9cfffcb06d` (SLM staging for 1 token, one-wave
  splits), `dc575f4ff5` (prefetch), `081d86870f` (5-token chunks), `1eb2307223` (256-register file),
  `064461a0b4` (cap 32 -> 256), `152`-series order:
  `e5a4d8a8f7`<`9cfffcb06d`<`dc575f4ff5`<`081d86870f`<`1eb2307223`. Also `fb75ee9e56` (the
  measured no for 2D block loads). `paper`.
- **Measured**: per layer at the 27B shape on a B580, `e5a4d8a8f7`: 16K 1 token 266 -> 123 us,
  4 tokens 457 -> 244, 16 tokens 2217 -> 776; 48K 1 token 768 -> 433, 4 tokens 1295 -> 671,
  16 tokens 6629 -> 1915. `9cfffcb06d`: 48K 1 token 433 -> 244 us, whole model plain decode +12 % at
  48K, +21 % at 115K. `dc575f4ff5` per layer 1 token: 48K 263 -> 226, 115K 554 -> 496; whole model
  plain +2.7 %/+4.4 %, MTP +4.2 %/+7.5 %. `081d86870f`: 5-token verify 637 -> 502 us at 48K;
  MTP +0.7-1.4 t/s. `1eb2307223`: 128K 5 tokens 1073 -> 1004 us; whole model 48K 106.1 -> 108.2
  t/s (+2.0 %), 115K 86.8 -> 90.6 (+4.4 %), output md5 identical. `064461a0b4`: 48K 64 tokens
  28.8 -> 4.9 ms, 115K 256 tokens 253 -> 44 ms (KLD 0.012699 vs 0.012800).
- **What the fork's kernel does that arcint's does not** (`code`):
  - Q-K is an **int8 x int4 DPAS on the raw q4_0 bytes** (`fattn-dec.cpp:688-696`): Q rows are
    quantized to int8 once per call with one RNE scale per 32-dim block; the key's 16-byte q4_0
    block XOR 0x88 is the B operand, one key per lane; the int result is scaled by `dq[row] *
    dk[key]`. arcint instead dequantizes K blocks to f16 in SLM and runs an f16 DPAS
    (`flash_attn_gqa_dpas.cl:95-101`, `:101-107`); its measured reason is that direct quantized K
    made every sub-group convert the same blocks (10.8 vs 15.6 t/s at 62,597 keys), which is
    exactly what the int8 B operand removes. With our q8_0 K, the same form is s8 x s8 DPAS
    (native on Xe2), and for the 8:4 pair the Q-K would be s8 x s4 as theirs.
  - **GQA-shared reads: arcint already has them** (`code`: `flash_attn_gqa_dpas.cl:10-22`, one
    work-group per (KV head, KV split) takes every query head of that head and every row;
    `ggml-opencl.cpp:23186-23200` dispatches it from 4 rows). The fork's grid is the same shape,
    so this is not the gap; the gap is the operand form.
  - **1-3 rows**: arcint's GQA kernel starts at 4 rows (`GGML_OPENCL_FA_GQA_MIN_ROWS`, default 4,
    `ggml-opencl.cpp:23186-23200`); 1-3 rows go to the split ALU kernel. The fork's DPAS kernel
    serves 1 token with SLM staging at 48K 244 us and measured faster than its own split path
    (1 row at 131K: 1.91 ms split against 2.24 GQA in arcint's record; the fork's staged 1-token
    path is a third form). Plain decode is entirely 1-row, so this is the biggest single opening.
  - **256-register file**: arcint already builds the GQA program with
    `-cl-intel-256-GRF-per-thread` unconditionally (`ggml-opencl.cpp:4876`); the fork only takes it
    for 3+ rows. Partly ours.
  - **Prefetch**: the fork prefetches the next phase's key rows (`fattn-dec.cpp:751-753`, `:857-869`,
    `:880-881`, `:1002-1003`); arcint has no next-tile prefetch in the GQA kernel. In OpenCL the
    form is an early load into registers/SLM before the barrier, not a `prefetch` builtin.
  - **One-wave split count**: the fork computes splits from Xe cores x work-groups per core / KV
    heads (`fattn-dec.cpp:1180-1196`: 48K 1 token 432 -> 302 us, staged 247). arcint's split is a
    host constant (`kv_per_split`, `GQA_SIDES` from a structural search); a runtime one-wave count
    is the lever.
  - **5-token chunks** are not missing: arcint passes `n_q` rows directly, so a 5-row verify makes
    30 rows, not the fork's 8-token/48-row path.
- **Cost to build**: medium-high — a new kernel body in `flash_attn_gqa_dpas.cl` for the int8 Q-K
  (Q quantize in-kernel, B operand from the block bytes, two accumulator scales), the V side stays
  f16 DPAS with V staged (ours) or dequantized in registers (theirs); a 1-3-row instantiation with
  SLM staging; a runtime split count. Red-first tests are the 0015 FLASH_ATTN_EXT geometry cases
  plus a kernel selftest over the int8 Q-K vs the f16 DPAS score.
- **Experiment**: B60, agent 27B q8_0:q4_0, 16K/64K/138K, MTP 5 (6 rows) and plain 1 row; arm A
  today (f16 DPAS, 4+ rows and split ALU for 1), arm B int8 Q-K + staged 1-row; metric per-layer
  attention us under `GGML_OPENCL_PROFILING` and served decode t/s; bar: >=+20 % of the attention
  layer with the answers and KL inside the bar.

### 2. GDN rollback by replay (and S planes 0/1 only) — score 9 (p1 3)
- **Evidence**: `e37b71a33b` (op + SYCL kernel + memory module + CPU reference, default on for a
  single sequence on qwen35/SYCL), `bda090edcc` (planes 2..n_rs_seq dropped), `a9fea0c521` and
  `abb95093c6` (merges). `code`.
- **Measured**: `e37b71a33b` states the state is "bit-identical to the snapshot path" and that the
  snapshot path wrote K = n_rs_seq + 1 full states per layer per verify. `bda090edcc`: **432 MiB
  less VRAM at n_rs_seq = 4 on the 27B** (48 GDN layers x 3 planes x 3 MiB, independent of -c).
  No decode number in either message.
- **Source** (`code`): `ggml/src/ggml.h:599` appends `GGML_OP_GATED_DELTA_NET_REPLAY`;
  `src/llama-memory-recurrent.cpp:75-106` decides replay before the tensors are created and sizes S
  to 2 planes; `:811-861` keeps the base state in plane 0 with pending records in plane 1;
  `:1424` leaves a base after token n-1-m plus records for the last m tokens;
  `ggml/src/ggml-sycl/gated_delta_net.cpp:857-864` is a replay prologue re-applying n_rec stored
  records with the step's own update and an epilogue storing the base plus one record per later
  token; `ggml/src/ggml-cpu/ops.cpp:10998-11134` is the reference ("a replayed record repeats
  exactly its state update (scale by g, then mad k * delta)").
- **arcint differences**: our 0050 writes per-token snapshots into the recurrent cache
  (`llama-served/src/llama-memory-recurrent.cpp:200-215` partial rollback via a per-token snapshot
  index bounded by `n_rs_seq`; patch 0050's GDN stage writes the snapshots straight into the cache).
  So ours pays the state writes and keeps 1 + n_rs_seq planes; the replay would keep 2 and write
  records of three small vectors.
- **Quality class**: exact by design (bit-identical state claimed; the records are the step's own
  update, re-applied in order).
- **Units**: agent (B60, MTP 5), Flash-Next (B60, MTP 3), coder (A770, MTP 4) — every hybrid and
  every verify.
- **Cost**: medium — the op and CPU reference are portable; the OpenCL kernel needs an
  emit-records variant and a replay prologue on the blocked GDN kernel; the memory module's plane
  layout is a llama.cpp change (one patch). The measured mechanism is on SYCL; as an OpenCL kernel
  it is the same three-vector update the blocked kernel already does.
- **Experiment**: B60, Flash-Next IQ3_S, MTP 3, 4K and 32K depth; agent MTP 5 at 138K; arm A
  snapshots (today), arm B replay; metric decode t/s, ms/verify round, peak VRAM, and
  teacher-forced KL + the task battery; bar: decode unchanged or better, KL within 0.03 nats,
  >=300 MiB less VRAM on the agent at MTP 5.

### 3. MTP head's prompt pass computes only its K/V — score 8 (p1 3)
- **Evidence**: `595d554c90` (default on, `LLAMA_MTP_KV_ONLY=0` off). `code`.
- **Measured**: the guide states, not the commit body: "long prompts read ~5 % faster with MTP on".
  The commit's own claim is stronger and checkable: the K/V ops are the full graph's on the same
  shapes and in the same expand order, so cache rows are bit-identical; only masked nextn
  extraction qualifies (unmasked readers copy h for every row); <=8 rows keep the unused Q
  projection in the graph so the same-input merge's tile matches the full graph.
- **Source** (`code`): `src/models/qwen35.cpp:756-793` (`kv_only = kv_only_on && n_outputs == 0 &&
  embeddings_nextn_masked`), the graph then stops after the cache store; the full path continues to
  attention, FFN and the head.
- **arcint today**: the MTP graph computes the whole block for every prompt row: `code`
  `llama-served/src/models/qwen35.cpp:498-665` builds Q/K/V, attention, gated attn out, residual,
  FFN and the head over `n_tokens` rows; the only cut is the output-row gather when masked
  (`:655-665`). Our drafter runs every prompt row through the MTP context (also recorded in
  `research-arc-community.md`, lead 1: the A770 project's `LLAMA_MTP_PREFILL_TAIL`).
- **Quality class**: exact — the retained K/V rows are the full graph's, bit-identical by design.
- **Units**: all three (agent MTP 5, Flash-Next MTP 3, coder MTP 4), prompt side.
- **Cost**: low-medium — a branch in our MTP graph builder after the K/V projections, plus keeping
  the Q projection when the batch is <=8 rows (our same-input fusion depends on the largest
  member). No kernel change.
- **Experiment**: B60 agent + coder A770, an 18K corpus prompt with MTP on; arm A full MTP graph,
  arm B K/V-only; metric prefill t/s and TTFT, then decode on the first answer; bar: >=+3 % prefill,
  identical accept counters and answers, KL 0 on the draft's K/V rows.

### 4. Round-trip discipline: one read-back for the draft's ids, sync trim, the KV rotations on the device — score 8 (p1 3)
- **Evidence**: `60ef575f3b` (skip a context synchronize when nothing was queued, `LLAMA_SYNC_TRIM`
  bit 1, default on), `4ec75c3416` (keep the attention rotation matrices on the device, bit 2),
  `196`/`f3aa69d6a2` (V index rides with K indices; the output-row index rides on the position
  input), `200`/`b0376aec39` (all draft ids in one tensor, one async read), `189`+`9d3313b24d` (the
  chain's per-step inputs made once, the mask row read in place). `code`.
- **Measured**: `60ef575f3b`: every result getter synchronizes and an MTP cycle calls several per
  row (server samples each verified row, the drafter reads the target's h_nextn row by row), each a
  full device wait on the one in-order queue every context shares. `4ec75c3416`: **2 split inputs,
  2 device synchronizes + 2 blocking copies (~272 KB) per graph**, in the verify graph and in the
  MTP draft graph alike. `b0376aec39`: **each small D2H read costs ~0.2 ms on this stack**
  (measured: reading only step 0's id cut the draft call from 6.07 to 5.16 ms), and the guide
  attributes ~2 % with a draft head to the single read. `196`: -1..2 split inputs per chain decode.
- **Source** (`code`): fork `src/llama-context.cpp:825-848` (the trim predicate);
  `src/llama-kv-cache.cpp` allocates `attn_inp_k_rot`/`attn_inp_v_rot` once in a device buffer;
  `src/llama-graph.cpp` skipped V index input; `src/models/qwen35.cpp` packs the output-row index
  at the end of the position tensor. Our `llama-served` has none of the three (`grep`: no
  `LLAMA_SYNC_TRIM`, no device rotation tensors).
- **Quality class**: exact — results were complete already; same values, same kernels.
- **Units**: agent and coder (MTP cycles with per-row sampling), Flash-Next (MTP 3).
- **Cost**: low — one llama.cpp patch each: the trim predicate, the rotation allocation, and the
  id read-back batching. Our MTP path reads draft ids per step; verify how many `tensor_get_async`
  calls a step makes.
- **Experiment**: B60 agent, MTP 5, 32K and 138K; arm A today, arm B trim + device rotations; a
  host-side counter of synchronizes and blocking copies per verify round under the sync trace
  (see #9's instrumentation); metric served decode t/s; bar: >=+2 % and identical answers.

### 5. The GDN recurrent-state gather folded into the kernel — score 8 (p1 3)
- **Evidence**: `9b4be76a1c` (default on, `GGML_CUDA_GDN_GATHER_FUSION=0` off). `code`.
- **Measured**: `RTX 4070, Bonsai 2 27B PTQ1_0: tg128 56.4 -> 58.5 t/s (+3.8 %), pp512 unchanged,
  400 greedy tokens identical`. The gather was ~8 us/layer and its dirty 3 MB/layer temp sat in L2
  until the FFN stream evicted it, making the GDN layers' gate/up GEMVs ~15 % slower than the
  identical GEMVs in attention layers.
- **Source** (`code`): the graph evaluator skips the `GET_ROWS` when its only consumer is the GDN
  state input and records (cache base, ids, row stride) for that node; the kernel indexes the cache
  row `s_ids[seq]` directly; single-sequence only (multi-seq rows may alias).
- **arcint today**: our GDN recurrence reads the recurrent state out of the cache through a graph
  tensors/gather (`code`: `llama-served/src/models/delta-net-base.cpp:498-565` builds the state
  input from the cache; patch 0050 fuses snapshot writes but not the read). Whether our OpenCL GDN
  kernel takes a gathered temp or an index is the audit question.
- **Quality class**: exact (no temp, same bytes; the 400-token identity).
- **Units**: all three hybrid units, decode and verify.
- **Cost**: low-medium — the graph skip plus an index argument in the OpenCL GDN kernel; also pays
  when a temp row's writeback pollutes the next mat-vec's cache.
- **Experiment**: B60 Agent (138K) and coder A770; arm A today, arm B fused gather; metric decode
  t/s and the GDN layer's gate/up GEMV time; bar: >=+2.5 % decode with identical answers.

### 6. Stop drafting past a depth — score 7 (p1 3)
- **Evidence**: `9f280958dc` (`--spec-draft-depth-max`, default 0 = off). `code`.
- **Measured**: `Bonsai 2 27B on an RTX 4070: the MTP draft is +85 % at zero depth, even near
  24-32k, and -27 % at 64k; 0 / 32k / 64k tokens: 101 / 48 / 37 tok/s with a 24k cutoff against
  101 / 45 / 27 without`. Past the cutoff the slot decodes one token per step and the hook is
  skipped for batches that sit entirely beyond it (by position, so early prompt ubatches still
  reach the draft context and stay reusable as a prefix).
- **Source** (`code`): `common/arg.cpp:4123-4128`; `tools/server/server-context.cpp:208`,
  `:3654-3657` (`spec_process = batch_view.pos[i] <= n_depth_max`).
- **arcint today**: no depth cutoff exists (`grep`: nothing in `llama-served`); the units' MTP is
  fixed width with the min-p stop. At 138K on the agent and deep on the coder, a verify's read is
  added without shortening it.
- **Quality class**: exact for a greedy-decode comparison *per step*? No: it stops drafting by
  depth, so the sampled text can differ (the drafted token is accepted when the target would have
  picked it; a step without drafts samples the same distribution but the text can diverge through
  sampling). At temperature 0 the target argmax is the same token either way, so the answer is
  unchanged. Class: exact at greedy, changes the answer by design under sampling. Listed, audited
  for the greedy path only.
- **Units**: all three at depth; the agent's 245k context is the extreme.
- **Cost**: low — a server-side predicate on the batch position and a per-slot cutoff derived from
  a sweep; no kernel work.
- **Experiment**: B60, agent MTP 5 and Flash-Next MTP 3, decodes at 8K/64K/138K; arms off / cutoff
  24K / 32K / 64K; metric decode t/s and the answers at greedy; bar: >=+5 % at 138K, answers right.

### 7. Sampler top-k on a long vocabulary: two-pass GPU top-k and direct selection from the logits — score 7 (p1 3)
- **Evidence**: `afe4c82d51` (two-pass top-k, `GGML_SYCL_TOPK_OLD=1`), `a3586a10d8` (top-k straight
  from the logits, its own switch removed in the same series). `code`.
- **Measured**: `afe4c82d51`: the one-work-group top-k took **~550 us per MTP draft token on the
  B580 (248K-entry vocabulary)**; now up to 256 groups per row with a pairwise local-memory merge
  and one group merging the winners, ties keep the lower index: 200K entries, k=1 241 -> 9.6 us,
  k=10 647 -> 71 us; TOP_K 325/325; live fresh +1-2 %, rename/edit within noise.
  `a3586a10d8`: `+1.9 % decode with MTP n-max 2 (greedy), +1.85 % with Hermes sampling (thinking,
  budget 16384, top_k 20)`; identical entries and order of equal logits, checked against
  `std::partial_sort` on 120,000 random arrays (ties, NaN, inf) and on server outputs.
- **Source** (`code`): fork `ggml/src/ggml-sycl/ggml-sycl.cpp:3075-3228` (two-pass with
  `tree_merge`); `common/sampling.cpp:112-282`, `:742-753` (the direct selection when every
  earlier sampler is a no-op).
- **arcint today**: the pin's sampler builds the whole token array with `set_logits` and
  partial-sorts it; `grep` finds no fast path in `llama-served/common/sampling.cpp`. Our MTP
  verify samples every verified row through that chain.
- **Quality class**: exact — the same entries and the same order of equal logits.
- **Units**: agent (248,320-row head, MTP 5 samples 6 rows a step), Flash-Next, coder.
- **Cost**: low — host code only: a `partial_sort` replacement in the sampler and, if arcint runs
  a GPU top-k for the drafter, the two-pass merge. No kernel for the sampling side.
- **Experiment**: B60 agent, greedy and Hermes settings, 256-token answers; arm A today, arm B
  direct selection; metric decode t/s and the token ids over 120,000 random arrays; bar: >=+1.5 %
  with identical ids.

### 8. Hybrid decode fusions: raw GDN gates, one l2_norm over q and k, the bf16 48-row gate mat-vec — score 7 (p1 2-3)
- **Evidence**: `00e8187618` (six fusions, -25 % encoded ops/token), `b56951ee0b` (conv chain),
  `38c6be55fd` (gate block), `31ae86e098` (same-input mat-vecs), `cf4b155cac` (SwiGLU/pre-norm),
  `b2ee0db22f` (bf16 <64-row matrices at 2-8 columns take the mat-vec kernel), `bfb2d7654a` (the
  ARCLAB fusion tests). `paper`/`code`.
- **Measured**: `M5 2B +7.7 %, 27B +2.6 % decode; CUDA 9B +2.5 %; greedy byte-identical`;
  conv chain `+1.3-1.8 %`; gate block `+0.5-1.4 %`; same-input `+1.8-2.0 %`; bf16 48-row at 2-8
  columns `10.5 us -> 4.4-7.3 us, 96 calls per token on the 27B`.
- **Source** (`code`): fork `src/models/qwen35.cpp` raw-gate flag and `ggml_gated_delta_net_set_raw_gates`;
  `ggml/src/ggml.c:2649` region; `delta-net-base.cpp` one l2_norm over the joint q/k view; the
  ssm_conv store applies silu; `ggml/src/ggml-sycl/ggml-sycl.cpp` fusion matchers.
- **arcint today**: 0003/0049/0050/0051/0066 fuse conv chain, silu, sigmoid+mul, GDN verify and
  snapshots; whether the alpha/beta gate projections for 2-8 columns take a mat-vec rather than a
  GEMM on bf16, and whether q+k share one l2_norm, is the audit question.
- **Quality class**: exact (byte-identical outputs claimed; same arithmetic, same order).
- **Units**: agent, Flash-Next, coder.
- **Cost**: low each, bundle 2-3, gate at the wall (the launch-count lesson in the earlier survey).
- **Experiment**: B60 agent and coder A770, decode and 6-row verify; count encoded nodes per token
  and launches per step with the graph's own counters, then A/B 2-3 fusions; bar: >=+2 % decode
  bundled, answers identical.

### 9. Instrumentation that prices the wall: sync trace, op timing, host timing, a runtime knob file — score 6 (p1 2)
- **Evidence**: `b13c1b921e` (`GGML_SYCL_SYNC_TRACE`, attributed causes), `ec71714bcb`
  (`GGML_SYCL_HOST_TIMING`, between/enqueue/wait), `60edb38e7d` + `24424cd538` +
  `505f7b8cd1` (`GGML_SYCL_OP_TIMING` per op type, draft-size graphs, per-weight-type mat-muls,
  chain reuse and `process_ubatch` time), `9db940800e` (`GGML_SYCL_KNOBS_FILE`, re-read per graph),
  `bfb2d7654a` (fusion tests build the exact node pattern and compare with the CPU backend:
  75/75, each fusion confirmed firing), `1fa26f1561` (perf cases at the verify shapes). `code`.
- **Measured**: `60edb38e7d`: "~6.5 ms per draft call, chain or loop, against a ~3.4 ms weight-read
  floor"; `505f7b8cd1`: the chain's `llama_decode` at 1.9 ms short / 3.3-3.5 ms at 48K/115K;
  `9db940800e`: sizes are not stated (a method).
- **Quality class**: n/a (no math), but the fusion tests are the exactness gate.
- **Units**: all three; this is the pricing tool for every candidate above.
- **Cost**: low — one trace hook on arcint's synchronous points, per-op timers behind a build
  flag, and a fusion selftest modelled on ARCLAB_*.
- **Experiment**: on the B60, one verify round and one prefill, attribute every synchronize and
  blocking copy; then re-price #1, #4, #5 with the numbers it yields.

### 10. The trimmed draft head (top-K frequent rows, FR-Spec style) — score 6 (p1 3)
- **Evidence**: `e75a9943ca` (default on, `LLAMA_MTP_DRAFT_HEAD_OFF=1` off), `7c577265b0` (the
  exporter and the self-distilled head). `code`.
- **Measured**: `Bonsai 27B, K = 32768 (99.6 % token coverage of a 4M-token code/prose corpus): the
  draft head reads 7.6x fewer bytes; fresh code temp 0 90.3 -> 94.8-95.8 t/s with identical
  output, temp 0.6 neutral within noise; K = 16384 is worse (92.1, 76 vs 81)`. `7c577265b0`: the
  self-distilled B580 head is `+4.8 % with MTP, outputs identical`.
- **Source** (`code`): `src/models/qwen35.cpp` computes logits for the K `draft_head` rows and
  scatters them into a full-vocab -inf row with `set_rows`; the head is listed under the target's
  Hadamard names.
- **arcint today**: the coder/agent have a frequency-ordered compact **draft vocabulary** of
  40,960 rows (the earlier survey's record), so the draft head is already pruned once. The
  remaining form is per-model K and a scatter into the full row; whether our MTP head already
  computes only the compact rows is the audit.
- **Quality class**: exact for the *draft* path (the target verifies every draft); the draft's
  proposal distribution narrows by design, acceptance is where it shows.
- **Units**: agent and coder (both MTP), Flash-Next (MTP 3).
- **Cost**: low if a compact head exists (a K sweep); medium if the scatter is missing.
- **Experiment**: B60 agent, MTP 5, K 16K/32K/64K; metric accept length and decode t/s on the
  code prompts, teacher-forced KL of the target unchanged; bar: >=+3 % decode at unchanged
  answers.

---

## Next 10 (ranked 11-20, verified)

11. **q4_0 small-batch GEMM on XMX, weights streamed once** — 6 (`03594d53c9`, opt-in
    `GGML_SYCL_Q4_0_DPAS`; 15360x3840: 8 columns 150 -> 81 us, 16 columns 292 -> 89 us; Gemma
    rename 188 -> 275 t/s). Ours: 0009 does few-column XMX from K-quants up to 8 columns. The
    mechanism beyond ours is one weight read for up to 64 columns (n-gram verify) and int8
    activations. Our units: MTP 5/4 = 6/5 rows, already 0009's range; price only if n-gram drafts
    are adopted. `code`: `mmvq.cpp` in the fork.
12. **`GGML_SYCL_MMVQ_CHUNK_MAX`** — 5 (`e8bd53439d`: Gemma 4 12B q4_0, n-gram 33-64 columns,
    94 % of GPU time in `dequantize_row_q4_0_sycl_reorder`; rename 79 -> 186 t/s). Same condition
    as 11: only a >32-column verify. `code`.
13. **KV mean-centering for a Q4_0 K cache** — 5 (`7fb3d5bf7c`, `36d1deef27`, `5d4126e7de`; a
    calibrated per-channel bias, a corpus helper, hybrid support). Our served KV is q8_0 K/q4_0 V,
    so it would apply to V's quant error; class rounding-only (a calibrated bias, added back at
    attention). Needs a calibration corpus and a quality gate; price after the free levers.
14. **Per-shape bit-exact tile table from a sweep** — 5 (`a0d82ad1ea`, `4bd894b1b7`,
    `0729c9d366`, `a70fe64480`, `6dd4481f07`: five rows at 0.775-0.974x the default tile, B580
    MTP-only +2.9/+3.1 %; a per-batch-size tile `{8,128,2,4}` 7-11 % faster at 12-32 rows). Ours
    has 0014's searched tiles for attention; a GEMV tile table for the MTP/draft shapes is the
    same method. `code`.
15. **The fusion/op test harness** — 5 (`bfb2d7654a`: ARCLAB_* build the exact node pattern,
    75/75, each fusion confirmed firing). Folded into #9; it is the red-first instrument for #8.
16. **Checkpoint buffer reuse + `seq_rm` used range** — 4 (`8a6d2a172f`: ~150 MiB checkpoint
    zero-filled and page-faulted again each turn, 18 ms saved per turn, 645 -> 627 ms;
    `5449810769`: `seq_rm` walked all cells on every call, visible in the host profile with
    speculative decoding). Our kv-checkpoint campaign is open; cheap host fixes.
17. **Catch-up rows decoded with the first draft row** — 4 (`d8a070b5e3`: 23.0 -> 22.4 ms/step,
    101.0 -> 103.3 t/s, draft-side launches 4 -> 3). Our MTP runs the draft over the verify rows;
    the saving is one launch and one round trip per step.
18. **Adaptive n-gram draft length** — 4 (`e576a2879f`: 48K short copy 38.6 -> 57.3 t/s). Only
    with n-gram drafts, which our units do not run; a record for a future drafter family.
19. **Draft-chain graph reuse** — 3 (`d6435c6f00`, `4798af11c8`: off by default, -0.2..-0.8 %
    before reuse; the graph cached per batch shape). Fold into a future chain only.
20. **Serving policy from the guide** — 3 (`e04b0cd48f`: one slot per card, MTP acceptance 31-74 %
    at -np 1 against 21-31 % at --parallel 4 on two B70s; keep n-gram drafts on; leave the chain
    unset with mtp-lean, forcing it byte-identical but 2-3 % slower). A slot policy for the agent,
    testable with existing knobs.

## Lower ranks and out (kept for the record)

- **The T2/XMX ternary machinery** (`125`, `133`, `134`, `138`, `149`, `160`, `161`, `168`, `169`,
  `174-176`) and every PTQ1_0/PQ2_0/Q2_0 codec: 0 for our formats (q4_0, q8_0, Q4_K, IQ3_S). What
  carries from the bundle is read above as separate mechanisms.
- **SYCL-under-xe**: the branch runs SYCL on Linux on a B580 with the legacy Level Zero adapter
  (`5336e12df3`). arcint's record says SYCL aborts under xe; by the reference-audit rule that
  negative is not settled until the legacy adapter is tried on the B60. `paper`/`code`; a record
  to re-test, not a mechanism.
- **Measured negatives worth keeping**: `fb75ee9e56` (2D block loads, 128K 5 tokens 1069 -> 1220
  us); `4798af11c8` (draft chain -0.2..-0.8 %); `a68382d688` (int8 prompt GEMM pp512 901 -> 818 at
  a 1024 threshold, pays only from 1024 columns); `36761482b0`'s checkpoint path (74-77 % of a
  round, ~0.5 s per drafted token).
- **Drafter families not served** (dflash/dspark/DFly, 28-36, 62-69, 71-90): 0-1, a second drafter
  family with its own model format; `research-arc-community` and the earlier survey already hold
  the block-drafter candidate.
