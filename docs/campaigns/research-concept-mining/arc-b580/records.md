# Pass 1 — one record a commit, the `arc-b580` fork of llama.cpp

208 commits since upstream `5ea87ddad2`, oldest first, read from the fork's own
`b580-commits.txt` (messages, bodies and stats). Head `e04b0cd4`. Evidence class: every number
below is the commit's own statement, `paper` for arcint; nothing here was measured on our cards.
Score 0-3: 3 = a mechanism our three units would use on our cards and not already in our served
series (0001-0079); 0 = does not carry (ternary/PTQ1_0/PQ2_0 formats, other backends, CI);
1 = carries only as background/record; 2 = plausible, needs pricing.

Area codes: attn, gemm, deltanet, moe, mtp, kv, quant, mem, sync, build, docs, other.
Counts at 0/1/2/3: 113 / 42 / 32 / 21 (208 commits).
Kind: kernel, fusion, sched, knob, instr, fix, revert, docs.

The fork's target differs from ours: one B580 12 GB, a ternary 27B (PTQ1_0, 1.75 bpw) and
Gemma 4 12B, q4_0 K/V, 128K context. Its SYCL backend runs under xe with the legacy Level Zero
adapter; our backend is OpenCL, so a SYCL kernel is read for what it would be as an OpenCL kernel.

---

## Commits 1-50 (Prism format work, kv-mean-center, first Arc port)

- `67c1af28b6` **quant** · kernel · superseded by `8bbb28b7` — adds ggml type Q2_0_g128 (142), a
  group-128 Q2_0 beside upstream g64; reference quant/dequant, quantize entry, gguf sizes.
  No numbers. `score 0`
- `64d1ed1b6b` **quant** · kernel · superseded by `8bbb28b7` — CPU scalar q8_0 vec-dot for the
  group-128 type, traits entry, test thresholds. `test-quantize-fns q2_0_g128 passes`. `score 0`
- `c09e6defb1` **quant** · kernel · superseded by `8bbb28b7` — CUDA MMQ/MMVQ/dequant/getrows for
  type 142. `MUL_MAT q2_0_g128 = q2_0 parity, 0 FAIL, 1231/1231`. `score 0`
- `34af866afc` **quant** · kernel · superseded by `8bbb28b7` — llama ftype, quantize dispatch,
  row validation for the type. `1.7B F16 -> Q2_0_G128 on CUDA, 415 t/s`. `score 0`
- `df01263d18` **quant** · fix · superseded by `8bbb28b7` — get_rows dispatch and ftype name for
  the type. `--pure g128: 5.6 % smaller, +1.8 % tg, KL ~equal to g64`. `score 0`
- `7fb3d5bf7c` **kv** · knob · default off (`--kv-mean-center`) — per-channel K-cache
  mean-centering for Q4_0: a calibrated bias subtracted from K channels before quantization and
  added back at attention; tool, tests, doc. No numbers in this commit. `score 2`
- `74bbd18b15` **other** · tool · kept — make-calib-corpus.sh: self-generated calibration corpus
  through a temporary server, gzip-ratio degenerate-output guard. Bias "agrees with one calibrated
  on a multi-domain set within sampling noise". `score 1`
- `c01d41ed0b` **other** · fix · kept — the corpus helper's option parsing, gzip preflight,
  `-ngl` no longer forced. `score 0`
- `36d1deef27` **kv** · fix · default off — mean-centering extended to hybrid models plus a
  calibration-basis guard. `score 2`
- `fcd13b78a8` **build** · docs · kept — Windows Vulkan/HIP release bundles + README notes.
  `score 0`
- `b73833108c` **quant** · kernel · superseded — AVX-512-VNNI dot for group-128 on x86; g64 stays
  scalar. `test-quantize-fns PASS, CPU MUL_MAT 45 OK / 0 FAIL (VNNI == reference)`. `score 0`
- `911888f7bc` **quant** · fix · superseded — ARM link and Metal template fixes for the type.
  `M5 Pro: 45/45 g128 cases; --pure greedy byte-identical to g64; tg128 +0.7 % at 5.6 % smaller`.
  `score 0`
- `ed4119b4bb` **build** · fix · superseded — switch coverage for the type under
  `-Werror=switch`. `score 0`
- `8bbb28b761` **quant** · fix · superseded by nothing (the rename itself) — Q2_0_G128 renamed
  PQ2_0, type id 142 kept. `score 0`
- `5d4126e7de` **kv** · fix · kept — draft model contexts do not inherit the K mean-centering
  bias. `score 1`
- `f2cf28855c` **quant** · fix · kept — GGUF loader detects legacy group-128 Q2_0 and points at
  the PQ2_0/group-64 downloads. `score 0`
- `1313d0a4a2` **quant** · kernel · superseded — PQ2_0 fast path enabled on AVX-VNNI CPUs.
  `score 0`
- `bef5fd7ea7` **quant** · kernel · kept (unverified on ARM) — NEON dot for PQ2_0, self-declared
  `[UNVERIFIED - needs ARM HW]`. `score 0`
- `c572efdcdc` **gemm** · kernel · kept — AVX-512-VNNI 4x8 repack GEMV/GEMM for Q1_0 and Q2_0
  with shared per-block activation sums. No end-to-end numbers stated. `score 0`
- `084596253b` **gemm** · kernel · opt-in, later default (`6ac5eb08`), self-declared
  `[UNVERIFIED - needs Hopper HW]` — Hopper wgmma Q1_0/PQ2_0 prefill, sm_90a. `score 0`
- `73f0fed042` **gemm** · kernel · kept, self-declared `[UNVERIFIED - needs Apple HW]` — Metal
  PQ2_0 backend. `score 0`
- `4ad963fa4f` **quant** · fix · kept — aarch64 repack fallback aliases for Q1_0/PQ2_0.
  `score 0`
- `cd3f8ed4f7` **gemm** · kernel · kept — ARM NEON+DP repack GEMV/GEMM for Q1_0. `score 0`
- `5511458a21` **quant** · fix · kept — PQ2_0 ftype id aligned to 141 with published files.
  `score 0`
- `269d191599` **docs** · docs · kept — prism-v7 Q2_0 format-change notice. `score 0`
- `bed6e9a6bf` **deltanet** · kernel · kept — GDN rows-indexed state read plus a snapshot
  write-fold (a ring decode path): the state read is indexed by rows instead of copied, the
  snapshot write folds into the kernel. New op in ggml, backend entries, 373 lines. No numbers.
  `score 2`
- `08d8c9035c` **deltanet** · sched · kept — qwen35 decode path over `gated_delta_net_rows`; the
  graph keeps the recurrent state in a ring and addresses it by row index. No numbers. `score 2`
- `b8e0bf0b63` **mtp** · fix · kept — dflash drafter: log-SNR conditioning tensors loaded and
  applied; without them the loader refused the file. `Metal, block-4 drafter: drafts at 65.891 %
  acceptance, mean accepted length 3.58` (it could not load before). `score 0`
- `e0c828b90b` **gemm** · kernel · kept — Hadamard weight-fold runtime for prism-v7: per-width FWHT
  (register and shared-memory forms to 8192), sign flip fused into the load, inverse transform
  for latent embedding tables, graph-walk coverage check; CUDA inverse 27.7 -> 80.9 tok/s when
  device-side. A model-format lever (folded GGUF), not a kernel we can apply to our GGUFs.
  `score 0`
- `a4f0714766` **mtp** · sched · kept — the log-SNR input lets the decoder graph be reused per
  ubatch instead of rebuilt. `acceptance unchanged 65.891 %, mean length 3.58`. `score 0`
- `adc79e8afa` **mtp** · fix · kept — the draft block layout is taken from the draft model, not
  the CLI type; a mismatched layout reads every drafted token one row late. `RTX 4090:
  dflash layout 0.272 % acceptance, mean len 1.01; dspark layout 51.385 %, mean len 3.04`.
  `score 1` (correctness class for a drafter family we do not serve)
- `e8b6452b1f` **other** · fix · kept — Metal im2col dispatch and a wide-row set_rows; two
  rollback harnesses that fail when no rollback happened. `M5: SET_ROWS 175/175, IM2COL 93/93`.
  `score 0`
- `a7cc29645c` **gemm** · kernel · kept — HIP byte-permute unpack for Q1_0/Q2_0 and the CDNA
  split-K geometry. `MI300X decode: 4B Q1_0 112 -> 322 t/s, 8B 81 -> 269, 27B 31.5 -> 97, 8B
  PQ2_0 113 -> 259, 27B 48.6 -> 91.7; KL 0 mean, 99.99+ % top agreement`. `score 0`
- `aeeacb3e8c` **build** · fix · kept — Metal unused-parameter error without residency sets.
  `score 0`
- `18d5846d31` **mtp** · fix · kept — legacy dspark -> dflash re-export; a vocab-less drafter
  fails at load instead of accepting nothing. `31 tensors byte-for-byte; 41.243 % acceptance,
  mean 2.62; 14 legacy drafters converted; three families 65.891/63.158/41.243 %`. `score 1`
- `0c448991e2` **mtp** · fix · kept — the confidence-head metadata key is read and cross-checked
  against the tensor. No behaviour change for valid files. `score 0`
- `4725def0ac` **docs** · docs · kept — README framing for the pointer switch. `score 0`
- `151ec3119f` **docs** · docs · kept — AGENTS.md trimmed. `score 0`
- `c06b847984` **docs** · docs · kept — AGENTS.md trimmed. `score 0`
- `f40a7bf69b` **docs** · docs · kept — CONTRIBUTING.md trimmed. `score 0`
- `b10f61bc52` **quant** · knob · kept — the pre-rename PQ2_0 ftype 142 is named instead of
  reported "unknown". Naming only. `score 0`
- `633168fb66` **gemm** · fusion · kept — the Hadamard activation transform is memoized on
  (input, rotation) and shared by folded weights that consume the same activation: 144 of 401
  folded weights were doing a duplicate transform (36 % of FWHT calls per token). `M5, three
  interleaved pairs: pp512 358.29 -> 370.74 (+3.5 %), tg128 23.20 -> 23.91 (+3.1 %), greedy
  byte-identical`. `score 0` (folded-format only; the same-input sharing idea is in `31ae86e0`)
- `01fd9521c9` **quant** · kernel · kept, partially superseded by its own later commits — PTQ1_0,
  ternary at group 128: base-3 trits, 28 bytes/128 weights = 1.75 bpw against PQ2_0's 2.125,
  lossless vs PQ2_0. `PQ2_0 357.47 MiB -> PTQ1_0 294.47 MiB (17.62 % smaller); perplexity
  12.6944 identical; Metal PTQ1_0 0.24-0.89x PQ2_0 decode at first`. `score 0`
- `e7938d5210` **build** · fix · kept — registers the two new PTQ1_0 tests. `score 0`
- `acdccf9c12` **gemm** · fix · kept — restores the src1 guard on the F16 Hadamard matmul path
  where a quantized src0 would write 4 bytes per element off the end of a 1-byte-row buffer.
  Not reachable through `supports_op` today. `score 0`
- `6f1c6908d8` **gemm** · kernel · kept — FWHT kernels for block widths 4096/8192 via
  shared-memory butterflies (registers spill at N=4096). `L40S backend-op cases: 24/26 -> 26/26,
  the two flipping cases are the block-4096 f32 ones`. `score 0`
- `ca02f9b833` **gemm** · kernel · kept — Metal threadgroup and Vulkan shared-memory FWHT to 8192;
  Vulkan cutover keyed on the register block, not width. `score 0`
- `d0ce941ec4` **gemm** · kernel · kept, opt-in parts dropped in the same series — Metal
  multi-column Q1_0 mat-vec for verify batches: one block read once, dotted against 2-4 columns.
  `M5, 27B Q1_0: pp2 29.6 -> 54.2, pp4 44.7 -> 69.2, pp8 48.6 -> 77.4, pp16 65.3 -> 80.4;
  tg128 unchanged`. A 1-bit quant path; the mechanism is our 0009's. `score 0`
- `7113f6462a` **gemm** · kernel · kept — PTQ1_0 trits extracted in exact float (a base-3
  fraction identity) with activation coefficients staged once, replacing integer recurrence.
  `M5: small decode 221 -> 287 (+29.5 %), large 59 -> 82 (+37.8 %)`. Not bit-identical
  (reassociation). `score 0`
- `ecd0addc3e` **gemm** · kernel · kept — PTQ1_0 mat-mul dequantize in exact float with the digit
  index hoisted per call. `M5: small prefill 5741 -> 6153 (+7.2 %), large 1186 -> 1274 (+7.4 %)`.
  `score 0`
- `b984d5f2f6` **gemm** · kernel · kept — PQ2_0 and Q2_0 mat-vec dot in exact float ("three
  floors and four fmas per byte replace the select chain"). `M5: PQ2_0 small decode 285.8 ->
  305.1 (+6.8 %), large 80.2 -> 83.1 (+3.6 %); Q2_0 283.6 -> 297.3 (+4.8 %); Q1_0 tried and
  rejected, 5-7 % worse`. `score 0`

## Commits 51-100 (hybrid-graph fusions, drafter families, format reach)

- `8c0170d198` **gemm** · kernel · kept — every folded Hadamard width runs on the threadgroup FWHT
  kernel instead of one simdgroup. `M5: 2048-wide decode 147.4 -> 200.6 (+36 %), 512 -> +3.6 %,
  1024 -> +8.2 %`. `score 0`
- `7d2da03143` **gemm** · fix · kept — PTQ1_0 CPY crash: supports_op claimed the op with no kernel;
  instantiated, F32 -> PTQ1_0 declined. `M5: CPY 203/203 (was exit 139)`. `score 0`
- `3dd767559a` **sync** · fix · kept — CUDA device type queried with
  `cudaDeviceGetAttribute` (20 ns) instead of `cudaDeviceGetProperties` (660 us); the hybrid
  graph builder asked per recurrent layer per graph build. `H100: 409 property calls / 337 ms
  per llama-bench run -> 25 / 22 ms, about 0.3 ms per decoded token`. `score 1`
- `9294043b25` **gemm** · kernel · kept — native PTQ1_0 MMQ tile loader and packed dequantize;
  prefill had fallen back to dequantize + cuBLAS. `2.26x slower than PQ2_0 at n=512 before`. Not
  compiled by the committer (`code`: later built on a 5090 with `6f1c...`-era fixes). `score 0`
- `7f292be192` **gemm** · fix · kept — the PTQ1_0 MMQ batch cap is lifted: the cuBLAS fallback was
  where PTQ1_0's extra error came from. `L40S largest band mean KLD 9.2e-4 -> 5.6e-4, top-1
  97.78 % -> 98.39 %; 9B 7.7e-4 -> 5.0e-4; cost about 7 % pp512 on Ada; MMQ wins below ubatch
  ~400, 1.6-2.5x at 64-256; decode unchanged`. `score 0`
- `1bc46a8a70` **gemm** · kernel · kept — FWHT block-per-row for widths 512+ (one 256-thread block
  per row; stages below the warp width shuffle, three above through shared memory). `H100: the
  transform was 25.8 % of GPU time at block 4096 (13.8 us/row), 22.5 % at 2048 (7.7 us/row);
  +17 % / +11 % decode on folded bands`. `score 0`
- `00e8187618` **deltanet** · fusion · kept — six fusions for the hybrid-attention decode graph,
  -25 % encoded ops/token: Hadamard sign flip into the FWHT load; raw GDN gates (sigmoid beta,
  A*softplus(g+dt_bias)) applied inside `gated_delta_net`; gated norm as one swiglu_split;
  one l2_norm over the adjacent q and k head groups; silu fused into the ssm_conv store.
  `M5 2B +7.7 %, 27B +2.6 % decode; CUDA 9B +2.5 %; greedy byte-identical`. `score 2`
- `d8d96cf051` **gemm** · kernel · kept — Bonsai low-bit kernels for GB10 (MMQ Y-tile prefetch,
  concat, GDN, norm, quantize). No end-to-end numbers in the body. `score 0`
- `d8f26eec76` **build** · fix · kept — the PTX L2 prefetch hints guarded off HIP/MUSA. `score 0`
- `6ac5eb0870` **gemm** · kernel · default on in a Hopper build — wgmma Q1_0/PQ2_0 prefill path
  inverted to opt-out; dispatch bounded to Hopper (Blackwell is 1000/1100/1200, the old bound let
  sm_100 reach sm_90a). "results are not bit-identical; per-128-K activation scale is coarser
  than q8_1's per-32". `score 0`
- `2863b1c7d7` **mtp** · kernel · kept — AngelSpec DFly drafter: per-draft-layer context fusion,
  a TreeFlash predecessor correction chained across the block. "Not yet measured" here. `score 1`
- `857e5f205e` **mtp** · fix · kept — DFly against the real checkpoint: conversion map, bare
  `layer_fusion` tensor, draft context anchor for ordinary batches. `M5 Qwen3-8B + DFly: coherent
  output at 12.9 tok/s with draft_n=637; acceptance 0, the chain is not enabled yet`. `score 1`
- `b505570057` **mtp** · fix · kept — draft context round-trip sized by `n_embd_out`, not
  `n_embd` (DFly's fused context is n_layer times wider; the batch read past its end). `score 1`
- `26db2347a8` **mtp** · fix · default on — DFly decoder embd input widened (`n_embd_inp_impl`),
  chain default-on; root cause of the zero acceptance. `M5, Qwen3-8B bf16 + DFly block-8, greedy,
  64 tokens, interleaved: baseline 18.35-18.52 tok/s; DFly 38.00-38.65, 52.7 % acceptance
  (49/93); chain off 30.89, 21.0 % (37/176)`. `score 1`
- `7e73eebd80` **docs** · docs · kept — DFly documented; the draft-depth optimum is interior
  (`42.0 tok/s at n_max 6 against 37.5 at 7`). `score 1`
- `b995a032e0` **mtp** · fix · kept — a correction-without-fusion drafter is named instead of
  failing on a tensor count. `score 0`
- `63c9d09237` **mtp** · fix · kept — DFly `target_*` validated against the target model, disabled
  correction weights dropped. The reference checkpoint declares depth 80/vocab 120832 against a
  real 36/151936. `score 0`
- `844b8ce746` **mtp** · fix · kept — DFly hidden size checked against the target, not itself.
  `score 0`
- `36761482b0` **mtp** · fix · kept — **plain drafter+target speculation on a hybrid model fell
  back to whole-KV checkpoints every round** because `need_n_rs_seq()` omitted `DRAFT_SIMPLE` and
  the draft context zeroed `n_rs_seq`. `27B qwen35: checkpointing was 74-77 % of each round,
  ~0.5 s per drafted token, ~15x the drafter's compute; speculation about 2x slower than plain
  decoding`. `Arc B390, 2B-PTQ1_0 drafting 27B-PTQ1_0, greedy, 4.43 t/s decode-only baseline:
  k=1 1.137x / 1.356x; k=2 0.482x -> 1.085x; k=4 0.535x -> 0.727x; text byte-identical to a k=1
  reference over the common prefix; acceptance (now counting partial rounds) 14-87 %`.
  `score 3`
- `6067dbea98` **mtp** · kernel · kept — DSpark corrected draft runtime + shared packed target
  head: Markov head in CUDA/Metal, 781 lines in `common/speculative.cpp`, a whole dspark model.
  6,790 lines. `score 1`
- `fa8a883858` **mtp** · docs · kept — terminology. `score 0`
- `4ddfa1e4f0` **mtp** · fix · kept — target taps restored, parity diagnostics enabled. `score 0`
- `aadbebf589` **mtp** · docs · kept — exporter name dropped from comments. `score 0`
- `1ecc8d65d9` **mtp** · fix · kept — unsupported DFly anchor convention rejected. `score 0`
- `f0a2b5dc9e` **mtp** · fix · kept — merge of the DFly PR. `score 0`
- `86a6fb1516` **other** · fix · kept — merge. `score 0`
- `08ac1fd17e` **mtp** · fix · kept — DSpark mask selection and graph-reuse contracts validated.
  `score 0`
- `b31adbd2a5` **other** · fix · kept — capture-buffer byte counts widened before the size_t
  conversion (uint32 product wrapped). Correctness. `score 1`
- `888fb70b81` **mtp** · fix · kept — three DSpark context mirror fields initialised. `score 0`
- `288859a96f` **mtp** · fix · kept — a dflash drafter that borrows a folded target's embedding
  and head must apply the target's Hadamard transforms; silently wrong draft logits otherwise.
  `DSpark drafts against Ternary-Bonsai-2-27B-PQ2_0 accepted 1-2 % instead of 43-57 %`.
  `score 0`
- `65ac430ab7` **build** · fix · kept — Metal language version 4.0 for the tensor API. `pp4096
  132 -> 370 tok/s on a stamped old SDK`. `score 0`
- `01041db99d` **build** · fix · kept — release compilation. `score 0`
- `9a9394a895` **build** · fix · kept — ARM64/Windows Vulkan release jobs. `score 0`
- `5476d46e95` **gemm** · kernel · kept — HIP native PQ2_0 unpack in the MMQ tile loader.
  `score 0`
- `518ad108f0` **mtp** · fix · kept — the MTP token-embedding lookup applies the Hadamard inverse;
  without it the coverage check refused in-file MTP on a folded model ("failed to initialize the
  context"). `score 0`
- `164c33700e` **gemm** · kernel · kept — AVX2/AVX-VNNI PQ2_0 kernels with Q8_K activations and
  tiled GEMV/GEMM for x86 without AVX-512. `i7-13620H: decode N=1 1.3 -> 3.9-4.2 tok/s; ms/token
  N=4 123 -> 78, N=8 99 -> 68, N=16 89 -> 54, N=64 89 -> 69; prompt 6.7 -> 16 tok/s; perplexity
  unchanged within 0.004 (7.0542 -> 7.0583); batched and single-token bit-identical`. `score 0`
- `10df298811` **quant** · fix · kept — non-x86 link alias for PQ2_0's Q8_K dot; 128-column rows
  must fall back to q4_0. `128/384-column rows sized the Q8_K activation buffer to zero bytes`.
  `score 0`
- `49cc6774d7` **other** · fix · kept — comment. `score 0`
- `ede9f73442` **other** · fix · kept — merge. `score 0`
- `422590f5d4` **other** · fix · kept — merge. `score 0`
- `922be44aa6` **other** · fix · kept — merge. `score 0`
- `d2c9ddd053` **gemm** · fix · kept — PQ2_0 rows not a multiple of 256 keep the Q8_0 activation
  path; shape-aware vec_dot selection at all six sites. Before: `K=128 rel_rms 1.0000, K=384
  0.4793, K=640 up to 3.3e36` silently wrong; after `15/15 OK, rel_rms <= 0.0068`; K=256/512
  unchanged to all digits. `score 0`
- `76e7487ce2` **build** · fix · kept — Metal tensor API disabled when the library is not
  embedded (wrong MUL_MAT output, not a load error). `score 0`
- `01ae597e3f` **other** · fix · kept — merge. `score 0`
- `3ae4f51087` **gemm** · kernel · kept — Vulkan integer-dot mat-vec for PTQ1_0 and full PQ2_0
  support; multi-column passes of 3. `RX 6750 XT: PTQ1_0 decode 4.7 -> 40.8 tok/s; PQ2_0 CPU-only
  -> 38.5; prefill 200 t/s; batch 8 seq 66.3 -> 86.4 aggregate; Arc B390 tg128 1.56 -> 13.02`.
  Ternary/2-bit types; our OpenCL serves q4_0/q8_0/Q4_K/IQ3_S. `score 0`
- `a419d438a1` **deltanet** · fix · kept — Vulkan declined raw-gate `GATED_DELTA_NET` instead of
  computing it wrong (`ERR ~1.0`). The pattern: a backend that does not implement the raw-gate
  flag must decline the op. `score 1`
- `ea50aba8cd` **gemm** · kernel · kept — AVX2/AVX-VNNI Q1_0 4x8 GEMV/GEMM with shared-scale Q8
  activations (one scale per 128). `kernel microbench GEMV 1.75-2.6x, GEMM 16 rows 2.0-2.9x;
  end-to-end pp512 8.22 -> 14.28 (1.74x) at 6 threads, tg128 5.34 -> 6.47 (1.21x); greedy outputs
  unchanged in practice`. `score 0`
- `bdc23b56b4` **gemm** · kernel · kept — branch-free PTQ1_0 MMQ tile loader (all lanes one
  5-iteration loop) and the full Ampere tile table (PTQ1_0 was the only type capped at mmq_x 64).
  `RTX 4070: pp2048 630 -> 914 -> 1304 t/s; decode 66.9 unchanged; perplexity 7.6740 vs 7.6742`.
  `score 0`
- `f8e39b6987` **gemm** · kernel · kept — planar-transposed activation layout for the PTQ1_0
  mat-vec: 9 aligned 16-byte loads instead of 32 scattered 4-byte ones. `RTX 3060, 4096x14336,
  us: 1/2/3/4 columns 47/104/154/201 -> 46/55/65/75; nwarps pinned so the K partition does not
  depend on the batch`. `score 0`
- `cdb1fcdbdd` **gemm** · kernel · kept — dedicated PTQ1_0 mat-vec with full lane utilization:
  (row group, K block) flattened into one index space, rows_per_cta chosen on the host, partials
  in shared memory summed in a fixed order; a column's result bit-identical for every column
  count 1-8. `RTX 3060: attn_qkv 72.4 -> 41.8 us, ffn_up 120.0 -> 66.7, attn_gate 44.8 -> 27.7;
  tg32 26.1 -> 39.8 tok/s, pp2 34.1 -> 61.1, pp3 33.3 -> 72.0`. `score 0`
- `0536a07d40` **other** · instr · kept — Bonsai projection shapes added to the mat-mul perf
  cases. `score 0`
- `7dbe57dd3b` **attn** · knob · opt-in — `GGML_CUDA_BATCH_INVARIANT`: kernel choices so a token's
  per-column arithmetic does not depend on the column count (mul_mat_vec_f 1-8 columns, vector
  attention 1-8 queries, KV split sized as for one query tile). `Greedy output with draft-mtp
  equals greedy without it at n-max 1/2/3; batches 5-8 agree with each other but not with 1-4;
  costs throughput at depth (vector attention at 3 queries over 42K keys)`. `score 1`
- `b2ee0db22f` **gemm** · fix · kept — bf16 matrices under 64 rows at 2-8 columns use the mat-vec
  kernel: the qwen35 GDN gate projections are bf16 [5120 x 48], 96 calls per token. `RTX 3060
  perf: 10.5 us -> 4.4-7.3 us at 2-8 columns; 3.3 us at one column`. `score 3`
- `9b4be76a1c` **deltanet** · fusion · kept, `GGML_CUDA_GDN_GATHER_FUSION=0` opt-out — the
  recurrent-state gather (`GET_ROWS` into a 3 MB per-layer temp) is folded into the GDN kernel:
  the kernel indexes the cache row directly, no temp, no kernel, no writeback. `RTX 4070, Bonsai
  2 27B PTQ1_0: tg128 56.4 -> 58.5 t/s (+3.8 %), pp512 unchanged; 400 greedy tokens identical`.
  A dirty 3 MB/layer temp also sat in L2 and made the GDN layers' gate/up GEMVs ~15 % slower than
  identical GEMVs in attention layers. `score 3`
- `f5ac2cb627` **deltanet** · kernel · kept — the 4-column-per-warp GDN layout on all Ampere+
  (was gated to GB10). `RTX 4070 pp2048 1220 -> 1297 t/s; paired pp512 +3.7 % RTX 3090, +1.3 %
  4090, +0.6 % H100, +0.3 % 5090; decode within noise`. `score 2`
- `4c74b7f489` **gemm** · kernel · kept — PTQ1_0 one-column decode on consumer Ampere/Ada: SoA
  q8 activations with exact int16 sums and warp-per-row small-K GEMV. `+14 % TG`. `score 0`
- `afced8f9ee` **gemm** · sched · kept — the SoA/planar choice as a template parameter; MMQ from
  5 columns (mat-vec wins at 2-4). `RTX 4070: at n=5 163 vs 137 t/s, n=8 258 vs 148`. `score 0`
- `befcc5c5f9` **attn** · kernel · kept — CUDA flash attention MMA reads q4_0/q8_0 K/V in place
  from the cache instead of converting the whole KV to an F16 scratch buffer every micro-batch.
  `1 GB per context at 262k with q4_0 caches, 2 GB with a draft context, full re-conversion per
  ubatch during prefill; Bonsai 27B + MTP on a 4070 12 GB spilled to system RAM (41 tok/s);
  resident, 222 MiB shared, 86 tok/s mean over 3 prompts; 48 new FLASH_ATTN_EXT cases pass,
  suite 2984/2984; dispatch limited to D=128/256 and matching K/V types`. `score 3`
- `9e95bd97fe` **gemm** · kernel · kept — PTQ1_0 multi-column mat-vec: raw digits with a mixed-sign
  dp4a and the exact activation sum subtracted once per sub-block (same integer, bit-identical);
  the epilogue one thread per (row, column) pair with four interleaved accumulators over a padded
  row stride. Profiled: 2135 -> 1493 instructions per warp, 56.3 -> 43.5 us, DRAM 62 % -> 83 % of
  peak. `RTX 4070 +1500 MHz memory: pp3 150 -> 169 tok/s, pp4 172 -> 212; pp1 unchanged`.
  The epilogue shape (division + shuffle butterfly = 50 % of warp stalls) is backend-generic.
  `score 2`
- `963c95f009` **mtp** · fix · kept — the MTP graph publishes only the output rows of `h_nextn`
  when masked; a batch with more tokens than outputs returned row 0's hidden state for the draft
  row. Correctness. `score 1`
- `d8a070b5e3` **mtp** · sched · kept — MTP decodes the catch-up rows together with the first
  draft row instead of eagerly over the whole verify batch. `RTX 4070, Bonsai-2-27B PTQ1_0 + MTP,
  262k, q4_0 K/V: 23.0 -> 22.4 ms/step, three-prompt mean 101.0 -> 103.3 tok/s, identical draft
  acceptance (0.86/0.46/0.65), byte-identical output; draft-side launches per step 4 -> 3`.
  `score 2`
- `9786654397` **gemm** · fusion · kept, `GGML_CUDA_FWHT_FUSION=0` opt-out — the Hadamard
  transform quantizes its own output to q8_1 when every consumer is a PTQ1_0 mat-vec; removes
  ~390 launches per decode step. Safety: every reader must be such a mat-vec; registry keyed by
  tensor pointer; aliased output held in a pool block (writing into the aliased buffer raced the
  kernel's own reads, poisoned the recurrent state, ~once per 170 requests). `340 consecutive
  requests, zero errors, output byte-identical`. `score 0`
- `0d48d36b0d` **sync** · fix · kept — an out-of-vocab sampled id (INT_MAX pad from an all-NaN
  logits row) is rejected and falls back to the CPU chain instead of killing the request.
  Correctness. `score 1`
- `84620e5a97` **mtp** · fix · kept — stale deferred MTP catch-up rows are dropped when a new
  task starts on the slot (decoding them writes stale positions into the recurrent draft state).
  `score 1`
- `bb02cb4119` **mem** · fix · kept — the FWHT-q8 held pool blocks released LIFO and before the
  pools are destroyed (`GGML_ASSERT(ptr == pool_addr + pool_used)` at llama-bench exit). `score 1`
- `9f280958dc` **mtp** · knob · default 0 (off) — `--spec-draft-depth-max`: past a depth the slot
  decodes one token per step and the draft hook is skipped for batches entirely beyond it (by
  position, so early prompt ubatches still reach the draft context). `Bonsai 2 27B RTX 4070: the
  MTP draft is +85 % at zero depth, even near 24-32k and -27 % at 64k; 0/32k/64k tokens: 101/48/37
  tok/s with a 24k cutoff against 101/45/27 without`. `score 3`
- `d8bf1fc6f3` **gemm** · fix · kept — PTQ1_0 kernel params restricted; Ampere takes the planar
  1-column kernel (3060: +5.9 % tg128 vs SoA). `score 0`
- `bf4d86343b` **build** · fix · kept — drafting disabled when deferred catch-up decode fails;
  in-place quantized K/V FA gated to NVIDIA with alignment checks; BOMs stripped. `score 1`
- `4364582b89` **gemm** · fix · kept — the PTQ1_0 mat-vec smem budget shared between guard and
  launch (one float pad per partials row). `score 0`
- `a76c8aa777` **mtp** · fix · kept — MTP catch-up flushed when it plus anchors would exceed
  n_batch. `score 1`
- `814354fe88` **gemm** · fix · kept — scalar PTQ1_0 vec-dot restored on MUSA. `score 0`
- `1b053d3dbd` **mtp** · fix · kept — MTP catch-up survives a failed draft decode (`catchup_failed`
  was never cleared, disabling that slot for every later task). `score 1`
- `18f113d8dd` **build** · fix · kept — PTQ1_0 layout helper defined after `ggml_cuda_info`.
  `score 0`
- `753323e602` **attn** · kernel · kept — the Arc B580 port itself, a 43-file bundle: SYCL PTQ1_0
  weights repacked to 2-bit and multiplied on XMX with TernSYCL int8 x int2 DPAS; q4_0 decode
  attention serving 1-4 query tokens per launch; blocked gated delta-net; PTQ1_0 mat-vec kernels
  (single column, decode-once multi column, same-input merge, fused gate/up); narrow concat;
  Vulkan siblings; draft KV cache type honoured by the MTP draft context;
  `LLAMA_ARG_SPEC_DRAFT_UBATCH`. Guide numbers: 128K on a B580, Bonsai 2 27B. `score 2`
- `6706e28843` **docs** · docs · kept — README points at the guide. `score 0`
- `cd65308030` **docs** · docs · kept — Vulkan build instructions. `score 0`
- `332d2c1c83` **docs** · docs · kept — feedback section, funding file dropped. `score 0`
- `877ed690fa` **docs** · docs · kept — temperature caveat, chat-UI numbers, before/after against
  the first port (`fresh code 55.8 -> 85.5 t/s, rename 217.5 -> 368.8, edit 143.0 -> 255.1,
  plain 31.4 -> 40.7`). `score 0`
- `d1aababd97` **docs** · docs · kept — model links and public-model numbers (`mtp-lean 128K:
  fresh 77, rename 306, edit 205, plain 38`). `score 0`
- `479017e7e0` **docs** · docs · kept — demo video. `score 0`
- `acd86a2b23` **docs** · knob · kept — `GGML_SYCL_FA_ONEDNN_MAX_KV` cap for the oneDNN prompt
  attention path; the fast path converts the whole KV to f16 (~4 KB per token of context), and
  at ~0.5 GB of spare VRAM a ~119K-token prompt ran out of memory and took the server down.
  Above the cap (98304) a chunked path reads at ~330 t/s at 120K; shorter prompts unaffected.
  Needle 9/9 at 32K/64K/120K. `score 3`
- `5056bde56e` **quant** · kernel · kept — PQ2_0 remapped in place onto the same 2-bit XMX
  layout; code 3 (+2) has no 2-bit form and aborts with a message. `MUL_MAT pq2_0 84/84`.
  `score 0`
- `731c1a8006` **quant** · fix · kept — the XMX path only on Xe2+, warning fallback elsewhere
  (Arc 140T "No kernel named ... was found"). `score 1`
- `69fe3fd3d8` **docs** · docs · kept — Ternary Bonsai 8B rows and tool-calling results; XMX is
  Xe2+ only. `score 0`
- `c098c02c38` **build** · docs · kept — one Windows SYCL CI workflow; upstream workflows dropped
  (they expect the upstream repo's runners and secrets). `score 0`
- `ebf104ca22` **docs** · fix · kept — a 2022 driver's system `ze_loader.dll` (v1.8.1) crashes
  the 2025.3 runtime's Level Zero adapter; Level Zero publishes no Windows loader to bundle.
  `score 0`
- `244c4d44c5` **gemm** · knob · lab, default stays best — `GGML_SYCL_PTQ1_T2_TILE` for the
  TernSYCL large-M tile. `B580 pp1024 (ub 1024, all on XMX): 8x128/8x2 939 t/s (default),
  8x128/16x1 949, others 480-857`. `score 0`
- `afe4c82d51` **moe** · kernel · default on, `GGML_SYCL_TOPK_OLD=1` opt-out — two-pass top-k for
  long rows: up to 256 groups per row with a pairwise local-memory merge, then one group merges
  the winners; ties keep the lower index. `The one-work-group top-k took ~550 us per MTP draft
  token on the B580 (248K-entry vocabulary); 200K entries: k=1 241 -> 9.6 us, k=10 647 -> 71 us;
  TOP_K 325/325; live fresh +1-2 %, rename/edit within noise`. `score 3`
- `e5a4d8a8f7` **attn** · kernel · opt-in `GGML_SYCL_FA_DEC_DPAS` — q4_0 decode attention on XMX
  via s8 x s4 DPAS: the raw 16-byte q4_0 block XOR 0x88888888 is the B operand, one key per lane,
  against int8 Q with one RNE scale per row and block; P.V on f16 DPAS so each lane dequantizes
  its own key's V block from 5 dword loads; token chunks of 8 serve 1-32 or =N query tokens.
  `Per layer (Bonsai 27B shape, real cache layout), us old -> new: 16K 1 tok 266 -> 123, 4 tok
  457 -> 244, 16 tok 2217 -> 776; 48K 1 tok 768 -> 433, 4 tok 1295 -> 671, 16 tok 6629 -> 1915;
  FLASH_ATTN_EXT decode incl. 12/16/17/32 tokens and the permuted layout 57/57`. `score 3`
- `a122b2b8c8` **attn** · kernel · kept — MKL flash-attention softmax one work-group per query
  row: one work-item per row walked the whole KV chunk twice, neighbours reading chunk_size floats
  apart. `Above the oneDNN KV cap this kernel was 72 % of generation time (Bonsai 27B, 115K:
  verify batches > 32 tokens and every new prompt); 114.6K code context: new code 17.6 -> 44.0
  t/s, edit 10.8 -> 27.5, identical output; short prompts unchanged`. `score 3`
- `e75a9943ca` **mtp** · kernel · default on, `LLAMA_MTP_DRAFT_HEAD_OFF=1` opt-out — trimmed
  draft head: `blk.<mtp>.nextn.draft_head` (rows of output.weight for the K most frequent tokens
  plus all special tokens) + `draft_head_ids`; the draft computes logits for those K rows only
  and scatters them into a full-vocab -inf row with set_rows. `Bonsai 27B, K = 32768 (99.6 %
  token coverage of a 4M-token code/prose corpus): the draft head reads 7.6x fewer bytes; fresh
  code temp 0 90.3 -> 94.8-95.8 t/s with identical output, temp 0.6 neutral; K = 16384 worse
  (92.1, 76 vs 81)`. `score 3`
- `e8bd53439d` **gemm** · knob · default `4 x MMVQ_MAX_BATCH_SIZE` (32→128 knob) — quantized
  mul_mat with 33+ columns fell through to dequantize-all-of-src0 + GEMM per call.
  `Gemma 4 12B q4_0 with n-gram drafts of 33-64 tokens spent ~94 % of GPU time in
  dequantize_row_q4_0_sycl_reorder; with GGML_SYCL_MMVQ_CHUNK_MAX=128: rename 79 -> 186 t/s,
  edit 57 -> 136; Bonsai 27B unchanged within noise`. `score 2`
- `226166a466` **attn** · fix · kept — oneDNN/MKL flash attention only for packed interleaved
  K/V; the permuted layout returned `ERR ~1.2` for every KV length from 1024 up (the test builds
  the cache view with a gap between tokens; the SDPA/MKL dequant take the token stride from the
  head count). Real caches are packed, so models were unaffected. `FLASH_ATTN_EXT 4137/4137 (was
  2 failing); perplexity 8.7273 unchanged`. `score 1`
- `ff03e32552` **docs** · docs · kept — guide: long-context attention, 4 drafts, Gemma. `score 0`
- `d4410763a5` **attn** · fix · kept — the XMX decode attention only on Xe2+, warning fallback
  elsewhere (Alchemist would fail at the first launch). `B580 unchanged: 4-token verify at 48K
  671 us (1295 without)`. `score 1`
- `c759ddb2b7` **attn** · kernel · kept — the s8 x s4 DPAS decode kernel takes head size as a
  template parameter: Gemma 4's global layers (512, 1 KV x 16 Q heads, token chunks of 2) and
  sliding layers (256, 2 per KV head, chunks of 8); splits target ~256 work-groups (a single KV
  head used to leave the GPU idle); combine templated on D. `Global-layer attention at 48K, us:
  1 token 1114 (f16 tile) -> 236, 4 tokens 2011 -> 635; Gemma 4 12B QAT + assistant, q4_0 KV,
  128K: short 92 -> 100 t/s new code, at 48K 47 -> 65 / 44 -> 57; Bonsai shape unchanged;
  FLASH_ATTN_EXT Gemma global 20/20, sliding 10/10, Bonsai 36/36`. `score 2` (our heads are 128)
- `03594d53c9` **gemm** · kernel · opt-in `GGML_SYCL_Q4_0_DPAS` — reordered q4_0 small-batch GEMM
  on XMX via s8 x s4 DPAS: each weight row streamed once for up to 64 columns (chunks of 64
  beyond), one lane per weight row, the row's 16 q4_0 bytes XOR 0x88888888 as B, activations
  quantized once per call to int8 (elements l / l+16 paired). `15360x3840 q4_0: 8 columns 150 ->
  81 us, 16 columns 292 -> 89 us; Gemma 4 12B: rename 188 -> 275 t/s, edit 138 -> 168, new code
  unchanged, outputs byte-identical; MUL_MAT Gemma shapes 1..256 columns 20/20`. `score 2`
- `a68382d688` **gemm** · kernel · opt-in (default 0) — large T2 prompt batches through oneDNN's
  int8 GEMM: converts the repacked 2-bit layout to int8 (per-row scale, group ratios). The
  conversion is a fixed cost per call, so it pays from ~1024 columns. `Bonsai 27B pp1024 941 ->
  1062, pp512 901 -> 818; 48.6K prompt read 67.7 -> 61.6 s (719 -> 789 t/s); PPL 8.7273 ->
  8.7359 (+0.1 %, int8 rounding)`. `score 0` (our 0071 is the same mechanism for MoE experts)
- `dd8589df8b` **docs** · docs · kept — Gemma at ~100 t/s, int8 prompt batches. `score 0`
- `8160e3b418` **docs** · docs · kept — comparison with PrismML's SYCL path (`prompt 207 vs 895
  t/s, plain 20.6 vs 42.3, 32K depth 8.1 vs 35.8, chat with MTP 37.6 vs 88.4`); the Claude Code
  section; a chat template whose mid-conversation system message becomes a user-turn note. This
  last one is a serving detail our units may want. `score 1`
- `9cfffcb06d` **attn** · kernel · default on, opt-outs `GGML_SYCL_FA_DEC_STAGE=0`,
  `GGML_SYCL_FA_DEC_SPLITS=64` — SLM-staged K/V tiles for 1 query token: each 128-key K and V tile
  copied into SLM with overlapped 16-byte loads, lane = key read from there; the split count
  fills one wave of work-groups (Xe cores x work-groups per core / KV heads) instead of a fixed
  64 per head. `B580: 48K 1 token 433 -> 244 us, 4 tokens 640 -> 456; whole model plain +12 % at
  48K, +21 % at 115K, same KLD and output`. `score 3`
- `dd980d3922` **docs** · docs · kept — when to turn thinking on. `score 0`
- `f31a4f611d` **build** · fix · kept — the s8 x s4 DPAS as inline vISA instead of the
  `intel_sub_group_i8_i4_matrix_mad_k32` builtin (not in the public extension; a Lunar Lake
  driver failed to load the device image). `B580: FA 4167/4167, MUL_MAT q4_0 59/59, output
  byte-identical; single-token attention ~6 % slower (48K 244 -> 259 us), whole model 1-2 %`.
  `score 1` (for OpenCL the DPAS builtin is the standard path, so no vISA needed)
- `2e8a0e6754` **build** · fix · kept — no IGC builtins left; vendored TernSYCL at main; f16
  DPAS via vISA; `-fsycl-device-code-split=per_kernel` so the Xe2 XMX kernels are JIT-compiled
  only when launched (GPUs without DPAS never compile them). `first request 4 s, startup
  unchanged; MTP 48K 73.6, 115K 59.6`. `score 1`
- `064461a0b4` **attn** · knob · default on, `GGML_SYCL_FA_DEC_DPAS=32` opt-out — the DPAS decode
  cap 32 -> 256 query tokens so long n-gram verify batches stay on the q4_0 kernel instead of
  oneDNN's path (which converts the whole KV cache to f16 each call). `B580 per layer: 48K 64
  tokens 28.8 -> 4.9 ms, 115K 256 tokens 253 -> 44 ms; KLD (ub 64-256) 0.012699 vs 0.012800,
  same top p 92.60 vs 92.89 %; MTP 48K 73.2 -> 76.1 t/s, 115K 60.3 -> 65.1`. `score 3`
- `e576a2879f` **mtp** · sched · default on, `LLAMA_ARG_SPEC_NGRAM_ADAPT=0` opt-out — adaptive
  n-gram draft length: starts at 64, doubles after a fully accepted draft up to n_max, halves
  after a rejection. A copy ends where the model stops but the context goes on, so with n_max 256
  a short copy drafted ~200 tokens past its end and verifying them cost ~0.3 s each at 48K.
  `B580 Bonsai 27B MTP + ngram: 48K short copy 38.6 -> 57.3 t/s, 1262-token copy 161.6 -> 172.9,
  48K new code 78.5 -> 79.4, short rename 428 -> 411, edit 305 -> 299`. `score 2`
- `081d86870f` **attn** · kernel · default on, `GGML_SYCL_FA_DEC_NQ5=0` opt-out — 5-token query
  chunks for MTP verifies: MTP with 4 drafts verifies 5 tokens, which ran the 8-token DPAS
  kernel with 3 empty tokens (48 rows of Q.K/softmax for 30). `128-key tiles; the kernel 637 ->
  502 us per layer at 48K, 215 -> 175 us at 16K; whole model MTP 48K 77.4 -> 78.1 t/s, 115K
  65.7 -> 67.1, output identical; =2 SLM-staged and =3 256-key tiles were slower`. `score 3`
- `9db940800e` **sync** · knob · lab — runtime-switchable SYCL tuning knobs through
  `GGML_SYCL_KNOBS_FILE` (re-read at the start of a graph compute when the file changes, so one
  server can A/B variants); `LLAMA_ARG_SPEC_NGRAM_FRESH=1` forgets earlier requests at the start
  of each one so repeated benchmark prompts don't replay earlier answers as drafts. `score 2`
- `254dfb2ca1` **gemm** · fusion · default on — T2 quantized-activation cache (back-to-back T2
  mat-vecs on one tensor reuse its int8 activations) + fused Hadamard rotate/quantize.
  `B580 Bonsai 27B plain decode +3.6-4.9 %, KLD unchanged`. `score 1`
- `4d50f848c9` **gemm** · fusion · default on — residual add + RMS norm fused into the T2
  rotate/quantize kernel (ADD + RMS_NORM + MUL(weight) + MUL(signs) + RESHAPE + Hadamard MUL_MAT
  as one work-group per token; outputs only used inside the chain are not written).
  `B580 plain decode +1.9-2.6 %, KLD unchanged`. `score 1`
- `546f94c81c` **build** · fix · kept — vendored TernSYCL -> 8b0ae42, the `-fp-model=precise`
  override dropped (upstream fixed a GemmMT row-scale broadcast returning NaN under fast math).
  `MUL_MAT ptq1_0/pq2_0 178/178, KLD unchanged, decode speed unchanged`. `score 0`
- `2dbd83934f` **instr** · instr · lab — `GGML_SYCL_T2_ZERO_WEIGHTS` zeroes T2 weights to profile
  non-weight cost. `score 1`
- `ec71714bcb` **instr** · instr · lab — `GGML_SYCL_HOST_TIMING=1` splits decode graphs into
  between / enqueue / wait host time plus per-op enqueue cost; W8A8 threshold as a runtime knob;
  test shapes. `score 2`
- `b56951ee0b` **deltanet** · fusion · default on, `GGML_SYCL_CONV_CHAIN` off switch — CONCAT +
  conv-cache CPY + SSM_CONV + SILU + L2_NORM in one kernel per layer (one work-group per
  128-channel head); outputs written at the CONCAT's position so nodes in between must not touch
  them; an L2 output on top of the old conv state falls back. Also `GGML_SYCL_FAST_SET_TENSOR`
  (opt-in pinned input uploads). `B580 Bonsai 27B plain decode +1.3-1.8 %, KLD ub1 unchanged`.
  `score 2`
- `38c6be55fd` **deltanet** · fusion · default on, `GGML_SYCL_DN_GATES` — bf16 alpha + beta
  mat-vecs (2 x 48 rows, same input, up to 8 tokens) with the dt-bias add, softplus, ssm_a
  multiply and sigmoid in one kernel: 9 graph nodes / 7 launches per layer -> 1 (2 when an output
  aliases the input); `GGML_SYCL_DN_GATES_DEBUG=1` prints why a block is not fused.
  `B580 plain decode +0.5-1.4 %, KLD unchanged`. `score 2`
- `31ae86e098` **gemm** · fusion · default on, `GGML_SYCL_T2_MULTI` — same-input mat-vecs in one
  launch: q+k+v, GDN qkv+z, FFN gate+up (<= 8 tokens) share one quantization and one GEMV
  launch; the Gemv body takes its work-group indices, a wrapper maps each group to its matrix
  through one call site (three inlined copies spilled: 34.6 vs 46.3 t/s). `B580 plain decode
  +1.8-2.0 %, KLD unchanged`. `score 2`
- `cf4b155cac` **deltanet** · fusion · default on — SwiGLU and the DeltaNet pre-norm folded into
  the T2 rotation: SWIGLU -> MUL(signs) -> RESHAPE -> Hadamard in the rotate/quantize kernel;
  ADD + RMS_NORM + MUL whose rotation comes ~9 nodes later computed with the norm when no node
  in between touches its output. `B580 plain +0.3-0.8 %, MTP rename +2 %, edit +1.1 %;
  byte-identical`. `score 1`
- `a70fe64480` **gemm** · knob · default on, `GGML_SYCL_PTQ1_T2_TILE_SMALL_MAX` (default 32) —
  per-batch-size GemmMT tile for T2: `{8,128,2,4}` is 7-11 % faster per pass for 12-32 rows,
  the default `{8,128,8,2}` from 48 rows up; `GEMV_MAX` (default 8) makes the GEMV/GemmMT switch
  a knob. `MTP rename/edit +0.5 %, 48K copy +1.3 %, outputs and KLD identical`. `score 1`
- `dc575f4ff5` **attn** · kernel · default on, `GGML_SYCL_FA_DEC_PF` — the decode attention
  prefetches the next phase's key rows: each tile prefetches its V rows before Q.K and the next
  tile's K rows before P.V (`sycl_ext_oneapi_prefetch`, L1). Phase probes showed Q.K and P.V
  serialised behind the K/V loads. `Per layer at 1 token: 48K 263 -> 226 us, 115K 554 -> 496;
  whole model plain 48K +2.7 %, 115K +4.4 %; MTP 48K +4.2 %, 115K +7.5 %; outputs identical`.
  `score 3`
- `bfb2d7654a` **other** · instr · kept — `test-backend-ops ARCLAB_GLU_ROT / ADD_NORM_ROT /
  CONV_CHAIN / DN_GATES / T2_MULTI` build the exact node pattern each fusion matches, at Bonsai
  shapes and 1-16 tokens, 3 seeds each, and compare every fused output with the CPU backend:
  `75/75 pass, each fusion confirmed firing`. Found and fixed: the far norm->rotation fusion
  required the signs to be a weight; the conv chain refused a later reader of the cache slot;
  the same-input merge repacks weights on first use. `score 2`
- `7c577265b0` **mtp** · tool · kept — `llama-mtp-export` writes, per position, the trunk state
  the NextN head consumes (`h_nextn`) and the trunk's greedy pick, for training/validating the
  draft head offline; `LLAMA_MTP_DRAFT_LOG` (top-3 per draft step); LAB_SPEC knobs. Used for the
  B580 self-distilled MTP head (`+4.8 % with MTP, outputs identical`). `score 2`
- `fc18b1a60b` **docs** · docs · kept — v0.8 notes (decode fusions: plain short ~41 -> 46 t/s,
  48K ~36 -> 40, 115K ~31.5 -> 35; MTP attention prefetch alone +4 % at 48K, +7.5 % at 115K). `score 0`
- `a0d82ad1ea` **gemm** · knob · lab, default 0 — `GGML_SYCL_PTQ1_T2_TUNED` per-shape GEMV tile
  table plumbing (empty table), a GRF mode and work-group order through GemvTuned; `=2` runs the
  default tiles through the tuned launcher as a self-test (outputs must stay identical). `score 1`
- `4bd894b1b7` **gemm** · knob · lab, default 0 — the table from a B580 GEMV sweep, five rows,
  "each with the same LS as the default tile, so the outputs are bit-identical":
  `ffn_down n=5 0.775, gate+up n=5 0.896, output head n=5 0.882, q+k+v n=5 0.923, q+k+v n=1
  0.974 (time vs the default tile)`. `score 1`
- `0729c9d366` **gemm** · knob · default on for a B580 (`GGML_SYCL_PTQ1_T2_TUNED=0` off) — tuned
  GEMV tiles. `B580 A/B: MTP-only +2.9/+3.1 %; live MTP + n-gram short +4.4 %, 48K +3.1 %,
  115K flat; outputs identical`. `score 1`
- `5336e12df3` **sync** · knob · default on, `SYCL_UR_USE_LEVEL_ZERO_V2` opt-in — default to the
  legacy Level Zero adapter. No number. `score 1`
- `595d554c90` **mtp** · kernel · default on, `LLAMA_MTP_KV_ONLY=0` off — the MTP head's prompt
  pass computes only its K/V: a draft-context decode without output rows (the NextN head's
  prompt catch-up) stops the graph after the cache store (embedding, hnorm/enorm, eh_proj,
  attn_norm, K/V projection, K norm, RoPE, cpy_k/cpy_v; no query, attention, output projection,
  FFN or head). "The K/V ops are the full graph's on the same shapes and in the same expand
  order, so the cache rows are bit-identical." Only masked nextn extraction qualifies. Guide:
  long prompts read ~5 % faster with MTP on. `score 3`
- `bcbe5a0ebc` **mtp** · sched · default on only with a trimmed draft head, `LLAMA_MTP_CHAIN`
  — one-graph MTP draft chain: all draft steps in one llama_decode, each step takes the previous
  step's in-graph argmax and head output row, ids read back once; a 36 MiB device copy of the
  draft-vocabulary embedding rows plus SYCL GET_ROWS keeps lookups on the GPU. No numbers here.
  `score 2`
- `e37b71a33b` **deltanet** · kernel · default on single sequence/qwen35/SYCL,
  `LLAMA_GDN_REPLAY=0` off — rollback by replay instead of state snapshots: a new op
  `GGML_OP_GATED_DELTA_NET_REPLAY` keeps one base state (the oldest a rollback can need) plus a
  small update record (delta, k, g) per later token; the next pass re-applies the kept records,
  then runs the new tokens. "The replay repeats the step's own state update, so the state is
  bit-identical to the snapshot path." `score 3`
- `a9fea0c521` **other** · fix · kept — merge cr-replay. `score 0`
- `4798af11c8` **mtp** · knob · default **off** — the draft chain turned off by default until its
  graph is reused; `measured -0.2..-0.8 % on the B580`. `score 1` (a measured negative for the
  chain; its later parts still pay, see `d6435c6f`, `b0376aec`)
- `d6435c6f00` **mtp** · sched · kept, still default off — reuse the draft-chain graph across
  drafts (cached per batch shape): the chain graph was built, split and allocated on every draft,
  costing more host time than the chain saved. The context keeps a scheduler + graph result per
  chain batch shape and swaps them around `process_ubatch`; per-step position and output-row
  inputs packed into two tensors (two uploads per draft instead of 2 x steps). "Same graph, same
  values: outputs unchanged." `score 2`
- `60edb38e7d` **instr** · instr · off by default — `LLAMA_MTP_DRAFT_TIMING=1` host phases of
  every draft call (averaged, printed every 64); `GGML_SYCL_OP_TIMING=1` GPU time per op type for
  draft-sized graphs. "Both measured runs of the draft chain spend ~6.5 ms per draft call, chain
  or loop, against a ~3.4 ms weight-read floor." `score 2`
- `bda090edcc` **mem** · fix · with replay — replay allocates only S planes 0 and 1: planes
  2..n_rs_seq were allocated but never used. `432 MiB on Bonsai at n_rs_seq = 4 (48 GDN layers x
  3 planes x 3 MiB), independent of -c`. `score 2`
- `24424cd538` **instr** · fix · kept — OP_TIMING matches draft graphs; mat-muls by weight type.
  `score 1`
- `6dd4481f07` **gemm** · kernel · knobs, default off — tuned Q6_K mat-vec (`GGML_SYCL_Q6K_GEMV`)
  and fused Q6_K gate+up+SWIGLU (`GGML_SYCL_Q6K_GLU`) for the MTP draft head: the same lane ->
  quant-int mapping, blocks in order, same vec_dot expression and sub-group reduction, but U
  blocks' operands loaded before use, aligned 32-bit ql/qh loads, sub-groups per work-group;
  `GGML_SYCL_Q6K_CHECK=1` reruns the original kernel per launch and counts rows that differ in
  any bit. "The draft head's Q6_K mat-vecs are ~60 % of a draft step and read at ~70 % of the
  B580's bandwidth (op timing)". `score 2`
- `abb95093c6` **other** · fix · kept — merge cr-replay-vram. `score 0`
- `0e1ba9c6ce` **mtp** · sched · `LLAMA_MTP_CHAIN_TIGHT` bit 1, default on — per-step input
  slices made once up front from device copies: every view of a host input was its own split
  input, so ~20 blocking H2D copies per draft (5 per step), each behind a backend sync, landing
  between compute nodes; now one whole-tensor copy per packed input, views slice it, everything
  added before the first compute node; the device split keeps ~9 inputs instead of ~24.
  "Same values in the same kernels: drafts unchanged." `score 2`
- `c9c5053ac8` **mtp** · sched · bit 2, default 3 — copy a step's mask rows only when the slice
  is not contiguous: a 1-row step's mask row is read in place (the decode kernels take the mask
  row stride from nb[1], never used with one row); only step 0 with catch-up rows copies. Saves
  n-1 copy kernels per draft (n = 4: 3). `score 1`
- `9d3313b24d` **mtp** · sched · bit 4, default 7 — gather the output row only for a multi-row
  step; without the gather, the head's input chain ADD -> RMS_NORM -> MUL(norm w) -> MUL(signs)
  -> RESHAPE -> MUL_MAT(Hadamard) folds into one kernel (new for drafts, first suspect if drafts
  differ). `score 1`
- `505f7b8cd1` **instr** · instr · off by default — per chain decode: graph reused/rebuilt with
  a reason, reuse check or build+alloc, set_inputs and compute submit host time.
  `the chain's llama_decode took 1.9 ms at short context and 3.3-3.5 ms at 48K/115K`. `score 2`
- `b13c1b921e` **sync** · instr · off by default — `GGML_SYCL_SYNC_TRACE=1` attributes every
  synchronize and blocking upload to its cause: the scheduler tags its copies
  ("sched: split input <name>", "sched: backend switch"), anything else labelled with the first
  two named callers outside libggml (e.g. "llama_get_logits_ith <- common_sampler_sample"),
  printed per 64 graphs of >= 2000 nodes with totals per big graph. "Costs a backtrace per sync
  when on (host timing is then not representative)". `score 2`
- `60ef575f3b` **sync** · fix · `LLAMA_SYNC_TRIM` bit 1, default on — skip a context synchronize
  when nothing was queued since the last one: every result getter calls `synchronize` and an MTP
  cycle calls several per row (the server samples each verified row, then the drafter reads the
  target's h_nextn row by row), each a full device wait on the one in-order queue every context
  shares. "Only the first after a decode waits for this context's work; a later one can only
  wait for the other context's work queued in between, which this context does not need."
  Results unchanged (they were complete already). `score 3`
- `4ec75c3416` **kv** · fix · `LLAMA_SYNC_TRIM` bit 2, default on — keep the attention rotation
  matrices on the device: with a quantized KV cache every attention graph reads
  `attn_inp_k_rot` (256 x 256 f32) and `attn_inp_v_rot` (64 x 64) as host graph inputs, so each
  graph uploads them again: 2 split inputs, 2 device synchronizes + 2 blocking copies (~272 KB)
  per graph, in the target's verify graph and in the MTP draft graph alike (upstream has a TODO).
  Same values, outputs identical. `score 3`
- `f3aa69d6a2` **mtp** · sched · bit 8, default 15 — one index upload fewer twice over: the
  output-row index rides at the end of the position input (one upload, sliced on the device);
  V stored with the K indices for a non-transposed V cache, so the V index input is left out.
  Per chain decode -1..2 split inputs (sync + blocking copy each). `score 1`
- `1a0e209ecb` **docs** · docs · kept — v0.10 notes: tuned verify tiles (FFN down 0.78x the
  time, gate+up 0.90x, output head 0.88x, q+k+v 0.92x at 5-token verify, bit-identical), replay
  (432 MiB less VRAM at 128K), one-graph drafts, K/V-only MTP prompt pass, draft-head tool.
  B580 against v0.8: `MTP on mtp-lean 71.9 -> 78.3 t/s (+9 %); trimmed 32K head 73.8 -> 80.4;
  live MTP+n-gram at 128K: short 69.9 -> 75.9, 48K 73.7 -> 78.7, 115K 78.8 -> 83.5`. `score 0`
- `135a7667b6` **build** · fix · kept — Windows build: the sync trace's caller lookup is
  Linux-only. `score 0`
- `b5a7db2682` **build** · fix · kept — sync trace counters renamed (the Windows SDK defines
  `small` as a macro). `score 0`
- `b0376aec39` **mtp** · sched · bit 16, default 31 — all draft ids in one tensor, read back with
  one copy: the chain read its n per-step argmax ids with n separate 4-byte
  `tensor_get_async` calls; on this stack each small D2H read costs ~0.2 ms. `reading only step
  0's id cut the draft call from 6.07 to 5.16 ms`. Now the ids are concatenated into one I32 [n]
  tensor at the end of the graph and read with one async copy. Guide: this alone is worth about
  2 % with a draft head. `score 3`
- `1fa26f1561` **other** · instr · kept — FLASH_ATTN_EXT perf cases at the Bonsai verify shapes
  (kv 16K/64K/128K x 1/5/16 tokens, real cache layout). `score 2`
- `fb75ee9e56` **attn** · kernel · default **off**, `GGML_SYCL_FA_DEC_B2D` — transposed 2D block
  loads for the DPAS kernel's lane = key reads: one `lsc_load_block2d d32.8x16tn` message gives
  each lane 8 dwords of its own key row, replacing per-lane dword gathers in Q.K (5 messages per
  16 keys instead of 40) and P.V (1 instead of 5). Same bytes and math, 89/89 cases. **Measured
  slower on its own at 128K** (register spills): 5 tokens 1069 -> 1220 us, 16 tokens 2945 ->
  3610 us. `score 1`
- `1eb2307223` **attn** · kernel · default 256, `GGML_SYCL_FA_DEC_GRF=0` off — 256-register file
  for the DPAS decode kernel at 3+ query tokens: the 3+ token launch is one wave of 2 work-groups
  x 16 subgroups per Xe core (32 threads), which is what the 256-register mode allows, so it
  costs no occupancy and removes the spills; 1-2 tokens keep the default file (their wave puts 4
  work-groups on a core). `B580, us per layer at 128K (real cache layout) 5/8/12/16 tokens:
  1073/1434/2920/2939 -> 1004/1240/2465/2445; 1 token unchanged; whole model 48K 106.1 -> 108.2
  t/s (+2.0 %), 115K 86.8 -> 90.6 (+4.4 %), output md5 identical`. `score 3`
- `5449810769` **kv** · fix · kept — `seq_rm` limited to the used cell range: it walked all cells
  on every call; with a large context and speculative decoding (seq_rm after rejected drafts)
  this showed up in the host profile. `score 2`
- `8a6d2a172f` **mem** · fix · default on — reuse the buffers of evicted prompt checkpoints:
  `create_checkpoint()` evicts old ones and emplaces a new one whose data vectors are resized
  from empty, so every checkpoint (~150 MiB for a hybrid model) is zero-filled and page-faulted
  again. `Growing multi-turn conversation: about 18 ms per turn saved (645 -> 627 ms); decode
  speed and outputs unchanged`. `score 2`
- `a3586a10d8` **other** · sched · default on — top-k straight from the logits when top-k starts
  the sampler chain: `common_sampler_sample()` built a token array for the whole vocabulary and
  partial-sorted it; now the k largest logits are selected directly when every sampler before
  top-k does nothing (penalties/dry/top-n-sigma off, no logit bias, no mirostat, no backend
  sampling), no grammar first, no reasoning budget forcing. Repeats libstdc++
  `std::partial_sort`'s heap steps, so entries and the order of equal logits are the same as
  before; checked against it on 120,000 random arrays (ties, NaN, inf) and on server outputs
  (sampling, tool calls, penalties, top_k 1/200, logprobs). `Decode with MTP n-max 2: +1.9 %
  (greedy), +1.85 % with Hermes sampling (thinking, budget 16384, top_k 20)`. `score 3`
- `7adb2dd207` **docs** · docs · kept — v0.10.1 notes. `score 0`
- `e04b0cd48f` **docs** · docs · kept — serving tips: one slot per card (MTP drafts land far more
  often: a tester with two B70s measured 31-74 % draft acceptance at -np 1 against 21-31 % at
  --parallel 4), keep n-gram drafts on (83.8 t/s combined on BetterBench), leave
  `LLAMA_MTP_CHAIN` unset with mtp-lean (forcing it gives byte-identical output but is about
  2-3 % slower). `score 2`

