# Candidates for arcint (pass 1b, 2026-10-29)

Ranked for arcint's mission (ARCINT.md): **1st** Flash-Next decode t/s on the B60 with right answers
(fewer/cheaper expert misses, cheaper swaps, cheaper CPU miss compute on 8 AVX2 cores, better MTP
acceptance, fewer launches), **2nd** prefill at 20k–240k on both cards, **3rd** the coder's/A770 and the
agent's decode. PCIe transfer and host CPU tuning score as first-class (ARCINT.md, "The host").

Evidence class: `paper` = what an engine's commit message or doc states it measured; `code` = what this
session read in the source (only where marked). Every number below is quoted as its commit states it.
Porting judgements arcint has not measured are marked "(inferred)". Scores 1–10 are this session's
judgement; pass 1's 0–3 score is in the evidence column as `p1:3` where useful.

Status language is the engine's final state — where an engine later reverted or made a concept opt-in,
the candidate says so (see `state/corrections.md` for the four records corrected).

---

## Top 10

### 1. prefill-layer-major — score 9
- **Evidence**: St ffe40c16 (concept), St 37e4832a (default), St a6df0023 / df7068b6 / 42094da9 /
  d1932476 / 5602253e / a732a34c / 18c53576 (kernels, FP16 path, declines, even split), In 58110e22
  (same idea in Infernix: layer walk per span, elastic KV). `paper`.
- **Measured**: Strata 37e4832a on z440, 2× RTX A4000 16 GB, Qwen3.8-Flash-Next IQ3_XXS, experts
  streamed, 5 interleaved pairs, TTFT medians: 8K 2.8×, 20K 4.8–5.7×, 32K 5.1–5.5× faster
  (141 s → 26–28 s), decode 38.6 vs 38.4, needles 12/12 both arms. In 58110e22: 40K prompt 1.39k →
  2.90k tok/s at chunk 4096, every output bit equal to chunk-major. ffe40c16: expert traffic drops by
  about S/512 (a 12.9K prompt ~509 GB → ~79 GB).
- **Status trail**: ffe40c16 opt-in → 37e4832a DEFAULT **only for a CUDA layer split with 2+ stages**;
  one GPU stays opt-in, HIP/SYCL/Metal and 3+ stages not covered; a card bigger than the pack (experts
  VRAM-resident) has little to re-read.
- **arcint has it?** No. arcint's prefill is chunk-major (ubatch 1024 on the coder, 2048 spills VRAM);
  its Flash-Next prefill is ~620 t/s at 20k with ~40 % residency, so the expert stream is the bulk.
  The idea transfers to OpenCL: a super-chunk read layer by layer, each layer's routed experts staged
  once for the whole super-chunk, residual rows in VRAM or pinned RAM with copies on the copy stream.
- **Why it ranks first**: the largest measured prefill lever on the exact model (Flash-Next: GDN +
  attention + 512-expert top-10), and it is the only concept that attacks the *re-read* of
  non-resident experts per chunk rather than the per-chunk kernel speed.
- **Cost to try**: high — a new prefill walk (Strata's is ~1,000 lines in `src/prefill/prefill.cpp`),
  plus the workspace lending and the KV-borrows-expert-frames trade (In 58110e22: workspace 1,774 →
  284 MiB, +565 frames; KV 8,197 → 9,299 frames, tg512 103.5 → 109.6). The declines are the QA
  checklist: MTP taps, KVA, stop_at_chunk, a pending force_chunk, a peer GPU, APR/NVFP4/WMMA layers.
- **Experiment**: B60, Flash-Next IQ3_S, 20k prompt; arm A chunk-major 1024 (today), arm B layer-major
  super-chunk 8192; metric TTFT + prefill t/s + decode after the prompt; bar: +1.5× prefill with
  answers and mean KL inside 0.03 nats (Strata's bit-identical claim suggests KL 0).

### 2. xe int8-DPAS grouped expert GEMM for the prompt — score 9
(merged: Strata `xe-xmx-grouped-expert-gemm` + `prefill-fused-int8-moe`; the "fused int8 prompt
experts" and the "XMX grouped GEMM" are one lever: run a prompt chunk's routed experts as a grouped
int8/tensor-core GEMM instead of dequantize + vendor GEMM / per-expert MMQ.)
- **Evidence**: St ebdfaacd (+10/16/20 % with identical tokens), 9a01b6e3 (work-group kernels, measured
  per-GPU crossover, 4K +12.5 %, 1K +20 %), c765c5f8 (default auto on Battlemage), 983fa18a (B70
  follow-ups), 12fd8cff (the A770 sibling: PF_XMX=2 default, A750 4K 79.3 → 110.4 tok/s), 2856a804
  (first correct version), 70f9c417/72a67c8e (sub-group 16 / 8×16×16 tile for bmg), ce60cc37,
  0e583707. `paper`.
- **Measured**: B70 (Battlemage, B60's silicon family), IQ3_S pack, 32K config, --prefill 4096:
  1,024-token prompt 645.8 → 775.4 (=1) / 783.5 (=auto) tok/s (+20/+21 %), 4,095-token 1,257.0 →
  1,413.8/1,407.7 (+12.5/+12.0 %), output tokens identical in all runs; quality teacher-forced KL
  against the FP16 path over 6,002 positions of 28 prompts mean 2e-4 nats, p99 6e-3, top-1 99.7 %,
  ppl ratio 1.0004. The measured crossover: int8 wins below ~130–250 rows per expert, the FP16 chain
  above (auto times both per GPU/driver/format and caches the decision). A770 sibling: A750 prefill
  +30–39 % at 4K with PF_XMX=2.
- **arcint has it?** Partly — arcint has XMX GEMM patches and K-quant/IQ matvecs, but not the grouped
  int8 expert GEMM with the in-kernel IQ decode and the measured auto crossover. The A770 (Alchemist)
  is arcint's second card and the A750 numbers are its sibling; the B60 (bmg/Xe2) is where 9a01b6e3's
  +12.5–21 % was measured. "The sub-group 8 tile does not build for Xe2" (72a67c8e) is resolved by
  70f9c417 (sub-group 16, 8×16×16), so a B60 arm exists.
- **Why it ranks second**: measured on the B60's own silicon family with identical tokens, and it
  hits the same prefill wall as #1 but at the kernel level (the prefill expert GEMM is where Strata's
  atomics/dedup say the prompt time sits).
- **Cost to try**: medium-high — ESIMD/DPAS kernels (or a joint_matrix form) plus the per-format decode
  coverage (code at HEAD: gate/up IQ3_S/IQ3_XXS/IQ2_S x down IQ4_NL/Q2_0, six type pairs, parity
  tested; IQ4_XS gate/up - the pack's fourth type - and the two 2-bit gate/ups stay on the FP16
  dequantize+oneMKL chain, see state/corrections.md), the measured crossover probe, a per-group
  fallback.
- **Experiment**: B60, IQ3_S Flash-Next, 1K / 2K / 4K / 20K prompt; arm A dequant+vendor GEMM (today),
  arm B int8 DPAS grouped (auto crossover); metric prefill t/s; bar: ≥+10 % at 1–4K, ≥+5 % at 20K,
  tokens identical + mean KL ≤ 0.03 nats.

### 3. pcie-bandwidth-probe-share — score 8
(the link probe + the every bandwidth-derived constant + the per-layer online balance that minimizes
the longer side of CPU-pool time vs GPU-PCIe time; merged with Infernix's `link_probe` family)
- **Evidence**: St f4820f50 (probe scales pcie_frac), 9abc623f (best-of-four bursts, continuous
  share), 7dbe2bf8 (below 4 GB/s → share 0), 8d56d417 (gfx12 start 0.15, +9.9 % at 8K / +25.6 % at
  32K), aa7b0cb1 (refined from the CPU pool's measured speed in the first windows), 7475df93, ab97f174,
  04be241a (per-layer balance, opt-in), e3b3ba1c (empty-group drop) → 83426781 (REVERT, no gain),
  In bf0d46db (warm the GPU during the probe), In a1444658 (every constant derived from the link),
  In cb3eda21 (quiet-retry CPU rate probe), In c349ab7d (divisor from the measured CPU team),
  In 4ee28eb1 (miss divisor by context d=1 <1,536 tokens, d=2 from there). `paper`.
- **Measured**: 04be241a (opt-in STRATA_PCIE_BALANCE=1, IQ2_XS, RTX 5090 + 9950X3D, one AVX2 pool
  worker emulated): 16.77/17.17 → 12.98/12.82 ms a round; fit says CPU 42–60 µs/expert vs GPU 28 µs
  per PCIe expert, 0.8 of the misses move. On the 16-worker 9950X3D the pool is DRAM-bound and
  nothing moves (12.19 → 12.17, 3 pairs) — the measured guard "on a DRAM-bound pool moving experts to
  PCIe saves nothing". In bf0d46db: ~1 IQ3_S startup in 7 misread the link at 3.0 GB/s (once 12.0)
  instead of ~27.5, then pp4096 −37 %; the busy-SM probe went 2/20 misreads → 0/60. 8d56d417: the
  decode miss split starts at 0.15 on gfx12 native packs, +9.9 % at 8K, +25.6 % at 32K, first-token
  KL 0.0013, top-1 12/12.
- **arcint has it?** Partly: ARCINT.md says the B60 link is measured (14.3 GB/s) and the split is a
  *fixed* pcie_frac. The per-layer split that minimizes the longer side, fitted online per window, is
  the missing piece, and the probe-misread guard matters wherever the link rate sets the session's
  split. Infernix's "derive every bandwidth constant from the probe" is the fuller form.
- **Why it ranks third**: it is mission item #1 on the exact mechanism (CPU-computed misses over
  PCIe 4.0 x8 on a slow CPU): the −22 % at the slow-CPU shape is the largest measured decode number
  in this batch that arcint does not have, and it changes only host policy, not kernels.
- **Cost to try**: low-medium — a per-layer balance pass (least squares with forgetting), a stamped
  flag wait, and the startup probe with the busy-SM rule; no kernel rewrite.
- **Experiment**: B60 + 5700X, Flash-Next IQ3_S, 8k–32k context decode; arm A fixed pcie_frac (today),
  arm B per-layer balance minimizing the longer side; metric decode t/s and ms/window; bar: ≥+10 %
  at the slow-CPU shape with tokens and KL unchanged.

### 4. route-resident-bias — score 8
(the residency-aware routing family: tail-skip at a rank, and the logit-bonus route-prior; merged
across Strata and Infernix)
- **Evidence**: St 65268d99 (tail-skip opt-in), b4e3407b (default rank 7 on CUDA serve), 6ab541f6
  (off at ≥90 % cached), 46d22108/60abf63e (split 70 %/60 % lines), ad2aef76 (HIP default), 55e2072c
  (NOT a default with NVFP4 experts), 5df35dcb (residency-biased routing), 2de9ae06 (logit bonus λ),
  33fd01b4 (one warp per token, measured margin table), c30656aa (route-prior default 2.0 ≤13 GiB,
  +37.6 %/+35.2 %), 0afaaa39 (owner drops 2.0 → 1.5 after real defects at 2.0), 7bbfab81 (numbers),
  In 722d5f4b (rank-dependent size on the exact quant). `paper`.
- **Measured**: 33fd01b4: the swap kernel one warp a token instead of one thread: decode median
  +8.8 %, ms/window −7.8 %, 4/4 pairs both orders, bit-identical to the serial form; teacher-forced
  top-64 over 7,677 positions, off KL 3.2033; margin 0.25 keeps 84.9 % argmax at KL 0.235, +0.6 %
  loss; margin 1.00 is 63.5 % argmax at KL 1.001, +49 % loss. In 722d5f4b (RTX 5090, INT8 KV,
  pp2048+tg512, MTP auto): rank 7 Dense8 155 → 207 tok/s (+33 %), FP8 47 → 61 (+28 %), IQ3_S
  210 → 234 (+12 %); rank 9 on FP8 +0.4 %. c30656aa: route-prior bonus 2.0 on an RTX 3060 12 GB
  story 44.2 → 60.9 (+37.6 %), code 47.3 → 64.0 (+35.2 %); on bigger cards −1.9/−0.5 %. 0afaaa39:
  at 2.0 the copy/code quality check found REAL DEFECTS; at 1.5 none in 19 tasks, story +15 %,
  code +37 % vs off.
- **Status trail**: tail-skip default-on on CUDA serve (rank 7), off at ≥90 % cached, per-topology
  lines on splits (70 % → 60 %); NOT a default with NVFP4 experts (KL 0.035, +11.5 %); route-prior
  default 1.5 on a ≤13 GiB discrete card only. Both change the output.
- **arcint has it?** No (known family from Strata's verdicts, not built): bottom experts per token get a read-back and the residency of the next layer is known. No.
  The rank-7+ missed experts are exactly the PCIe + CPU work arcint pays ~40 % residency; the
  route-prior turns misses into VRAM hits at a measurable routing deviation.
- **Why it ranks fourth**: the largest decode gain per line of code, on the exact quant, with the
  price measured. It ranks below the first three because it changes the answer — the 0.03-nats bar
  and the task battery must pass, and at bonus 2.0 real task defects appeared on a 12 GB card.
- **Cost to try**: low for tail-skip (a predicate in the plan + a skip in staging/compute); medium
  for the route-prior (a second top-10 pick with the bonus, the route_prior kernel). Quality work is
  the real cost.
- **Experiment**: B60, Flash-Next IQ3_S, ~40 % residency, 256-token decodes; arms off / tail-skip 7 /
  tail-skip 8 / route-prior 1.0 / 1.5; metric decode t/s + teacher-forced KL on assistant tokens +
  the task battery; bar: ≥+15 % decode at KL ≤ 0.03 nats mean and 0 new task failures.

### 5. qsa-select-gemm-topk — score 8
(merged with Infernix `qsa-radix-block-select`: the long-context QSA block-score + top-k selection on
the prompt and decode paths)
- **Evidence**: St 2f9c38f0 (GEMM block scores + register top-k, 128K prompt +14.6 %), 586367cb
  (wide top-k past 135K cells, 9–12×, 243K prompt +26 %), 33237ff3 / 6e76ae23 (thread-per-query-head,
  4-lanes-per-query scorers, 4.6–5.2×), 050229ec (decode halving scorer + split top-k, bitwise),
  82922f55 (wide kernel on HIP), 6ef23b89 (cooperative-scan top-k), d5b80b2e (one-read sample top-k),
  c298fa55 (gfx12 default), 3f691eb3, 4f21be87, In cf6a4e09 (multi-CTA scoring + histogram-radix
  select), 1817329c, 2a605de7, a5b4aea8-adjacent. `paper`.
- **Measured**: 2f9c38f0: select was 18 % of a 128K prompt; 5070, 256 queries: 32K 1.18 → 0.36 ms,
  128K 4.68 → 1.33 ms; prompt 32K 1,742 → 1,797 tok/s, 128K 1,608 → 1,843 (select 14.6 → 4.2 s),
  TTFT 82.7 → 72.3 s; needles 15/15, 16K continuation 32/32 identical KL 0.005, 32K top-1 same
  KL 0.03. 586367cb: RTX 3060 12 GB, 256 queries: 262K 16.09 → 1.71 ms, 524K 37.53 → 3.24 ms;
  243K-token prompt +26 %. 33237ff3: 141 M scores bitwise identical; per call 4.6–5.2× faster
  (1,024 queries at 200K 52.4 → 10.8 ms); block_scores was the top prompt kernel at 1,045 ms/card
  (17 %) at 140K; agent read 476 → 632 tok/s (+33 %), TTFT −32 %. 050229ec: decode select ~6 ms of a
  ~52 ms window at 420K cells; 8-query scorer 774 → 255 µs, top-k 0.21 → 0.033–0.064 ms.
- **arcint has it?** No on the prompt path (ARCINT.md records the QSA select as a mission lever; the
  agent runs 245,760 context, the 524K-cell regime where the select took the slow kernel). The
  GEMM-form block scores and the wide/register top-k are an OpenCL port idea (arcint's OpenCL has no
  TF32 MMA; the score form is FP32 tiled, as 3f691eb3/4f21be87 do).
- **Why it ranks fifth**: it is mission items 2 and 3 on the agent (long-context prefill and decode at
  depth) with the largest measured select numbers and a bitwise id guard; it ranks below the decode
  miss levers because it is the agent's path, not the Flash-Next mission path.
- **Cost to try**: medium — two kernels (scores, top-k) plus a dispatch by live block count (not by
  --max-context); the bitwise/selftest discipline is the guard.
- **Experiment**: B60, 27B agent, 100K / 240K prompt and a 240K-context decode round; metric TTFT,
  prefill t/s, ms/window; bar: ≥+20 % prefill at 240K, decode ms/window down, ids identical on the
  QSA selection.

### 6. CPU miss compute on AVX2: named accumulators, prefetch, multi-token rows — score 8
(merged: St `cpu-avx2-accum-in-regs`, `iq-row-prefetch-avx2` incl. the page-fault arm, `cpu-iq-multitoken-rows`,
`cpu-expert-worker-team` (In), `cpu-miss-divisor-by-context` / `cpu-rate-probe-quiet-retry` (In),
`iq-avx2-grid-gather` as the measured no, `cpu-q2-two-block-unpack` as the measured no)
- **Evidence**: St 2a669f3c (router dot 2.1×), 79376a27 (q5_1/q8_0 down rows 1.24/1.30×), 95883f46
  (−4.0 % gate/up on a 5700X3D), 9e3fe046 (cold pages 78 % of the CPU lane; parallel page prefetch),
  89c62e53 (multi-token i-quant rows), d4b8b6ad/7d8fc8fd/31ad9627 (measured price and the 1.4
  tokens/group wash), 0736f05d → a7336175 (gather wins on Intel P-cores, loses on Zen), f1216678 →
  a88904e5 (two-block Q2 unpack reverted), In b36f35ae (GGUF team without barriers +0.75 %), In
  c349ab7d / cb3eda21 / 4ee28eb1 (measured divisor + quiet probe). `paper`.
- **Measured**: 2a669f3c (Ryzen 7 5700X3D, Zen 3 = arcint's CPU class): one layer's router [512×2560]
  at 4 tokens 0.41 → 0.19 ms (nt=2 0.36 → 0.16), bit-identical; 79376a27 (Zen 5, 4-token window):
  Q5_1 down [2560×640] 190–197 → 155–156 µs (1.24×), Q8_0 135 → 104 µs (1.30×), 0/36 token-sets
  differ in any bit; q4k_dot left alone (+5 % at 4 tokens, −4..−7 % at 6–8). 95883f46 (5700X3D):
  gate/up 13.571 → 13.024 ms/round (−4.0 %), whole round 45.54 → 44.94 (−1.3 %); prefetch 2048 B T0.
  9e3fe046: cold reads 78 % of the CPU lane. In b36f35ae (i9-13900K): GGUF team without barriers
  +0.75 % IQ3_S decode; the tiled AVX2 GGUF rows +0.4–0.6 %.
- **arcint has it?** No for the named-accumulator idiom, the prefetch and the multi-token rows;
  arcint's miss path is per-token AVX2 rows on the 5700X's 8 cores (ARCINT.md: 8 threads 11.4 t/s vs
  16 threads 9.0). The iq-avx2-grid-gather measured NO on Zen 3 is the negative that saves an
  experiment.
- **Why it ranks sixth**: mission item #1 is "better CPU miss compute on 8 AVX2 cores"; these are
  bit-identical wins on arcint's exact CPU class — but each is modest (a few % of decode), so it sits
  below the miss-count levers.
- **Cost to try**: low — one kernel edit each (the named accumulator idiom 4 tokens/pass, the prefetch
  hint, the page prefetch), plus per-core parity tests; the worker-team/divisor parts are host policy.
- **Experiment**: 5700X, arcint's CPU-miss decode path at 8 threads; arms: today / named accumulators /
  named accumulators + prefetch / + multi-token rows; metric CPU miss-compute ms/round and served
  decode t/s; bar: ≥+3 % decode with bit-identical expert rows (parity harness).

### 7. xe doorbell / host handshake (uncached hint, stepped window, payload checksum) — score 7
(merged: `sycl-doorbell-uncached-hint` + `sycl-stepped-verify-window` + `doorbell-payload-checksum`)
- **Evidence**: St 557c2b67 (load side, 263 ms → 2.4 µs, 8.7–10.5 → 19.3 t/s on a B60), 9c790e3d
  (store side, host sees the store at 230 ms vs 2038 ms), a79e7225 (per-device spin bound), b70bc033
  (driver gates the hint: xe uncached, i915 plain), c2d008bc (stepped window, A770 +40 %), 88a1c23f
  (two launches fewer per layer), 46a77907 (lean NO_HOST graph, B70 −2 %), fde1d795 (batch windows
  keep the doorbell), 0d4e82c6 (allocation-level uncached), 18ee03e5 (0.5 % stale payloads on A770),
  3157c789 (checksum wait costs 17 % decode on A750 → opt-in). `paper`.
- **Measured**: 557c2b67: a system-scope atomic load of host USM on an Arc Pro B60 is still
  cache-served, so a device spin never sees the host store; the uncached L1+L3 read hint bounds the
  ping-pong at 263 ms → 2.4 µs and per-layer waits 8.7–10.5 → 19.3 tok/s decode. c2d008bc (A770):
  GPU wait/window ~70 → ~43 ms, decode 22.8–25.0 → 31.8–32.5. 18ee03e5: 14 of 2,832 rings visible
  before the last activation store; 3157c789: the checksum costs 17 % of decode on an A750.
- **arcint has it?** Unknown — arcint's B60 handshake works at ~38–40 t/s today, so it already avoids
  the cache-served trap somehow; the audits are (a) does any host-written flag/result row a B60
  kernel polls come from an uncached allocation/read hint, (b) is the window one graph (host serves
  only after it) or stepped segments (host serves between publishes), (c) is there a payload
  checksum, at the measured 0.5 % stale rate and its 17 % price.
- **Why it ranks seventh**: it is correctness and a measured 2× class on the exact card, but arcint's
  current handshake evidently works — the value is in the stepped window (+40 % on the A770 sibling)
  and the driver-gated store/load form, which are audits, not a rewrite.
- **Cost to try**: medium for the stepped window (the window captured as segments + a serving helper
  thread pinned to the SMT sibling); low for the hint on any polling load/store.
- **Experiment**: B60, Flash-Next IQ3_S, CPU miss pool; arms today / stepped window; metric GPU
  wait per window and decode t/s; bar: wait down ≥20 %, tokens identical.

### 8. screened-vocab-head-2bit — score 7
- **Evidence**: In 257789da (op + conversion, 178.8 MB lm_head screen, 94.4 MB proposal head),
  abc0a064 (measured verbs). `paper`.
- **Measured**: Flash-Next FP4-Local (MXFP8 dense, 16-bit experts), INT8 KV, RTX 5090, ABBA:
  screened-lm-head plain decode +7.5 %, MTP 3 +3.7 % (4 of 4), MTP drafter's screened draft head
  +1.6 % at MTP 3 (greedy unchanged). Op test: normalizer within 0.15–0.19 nats of the exact
  log-normalizer, same argmax as the full head, argmax equal at every position over 1,024 positions.
  The 248,320-row head reads ~1/8 the bytes per step.
- **arcint has it?** No. arcint's decode step ends in a matvec over the full 248,320-row lm_head and
  the MTP step over the 40,960-row draft head; the screen (a 2-bit copy scored for every row, exact
  rescore only near each column's best, the tail folded into the normalizer with a v/2 correction) is
  a direct bytes lever. The screen fit is `grouped_mse_norm`; the persistent int8-MMA screen kernel
  has no straight Xe port (inferred: an OpenCL int8 dot screen at 2-bit widths is the form).
- **Why it ranks eighth**: +7.5 % on the mission model's decode with a real quality price
  (normalizer 0.15–0.19 nats), so it is behind the free levers; it is the largest head-bytes lever
  in the batch.
- **Cost to try**: medium-high — conversion (2-bit screen copies), a screen kernel + exact rescore
  path, the normalizer correction, and a sampled-output quality gate.
- **Experiment**: B60, Flash-Next IQ3_S, 256-token decodes plain and MTP 4; arms full head / screened
  head; metric decode t/s + mean KL on assistant tokens + argmax-equal rate over 1,024 positions;
  bar: ≥+5 % decode at KL ≤ 0.03 nats.

### 9. adapt-tier-overlap (async swaps / copy overlap / plan placement) — score 7
(merged: St `adapt-tier-async` incl. `adapt-tier-overlap-boundary`, plus In `expert-cache-plan-overlap`
and `post-prefill-promotion-negative`)
- **Evidence**: St b437b8ea (+10 %, +32 % with tuned draft), 9ef44842, f95f2eab → 661ff293 (opt-in
  everywhere, 5090 Laptop −21 %), 0d761d99 (resident mode +7.0 % A4000 / −2.6 % RAM-bound 3060),
  e989c671/495dc436 (batch windows), 25e0a2ef (host-function signal → 4-byte D2H poll),
  0e20d9ae (callback thread inherits the host's CPU mask → 2–4× slower), f95f2eab, 661ff293, d69424c9
  (lag 2), 63b4bf0c (wait determinism, 0.26 top-logit shift), 1aa0abf6 (arena mode), cb22a53e (WDDM
  trap), In f1864300 (plan inside the commit, +2.4–5.1 %), In a6a6b4f7/b454cd83 (post-prefill
  promotions −3.3…−8.9 %). `paper`.
- **Measured**: b437b8ea: blocking tier ~90 swaps = ~24 ms every 4th window; --adapt-async 1 decode
  79.7/78.8 → 87.6/86.2 (+10 %), with the tuned draft layer + --spec-min-p 0.65 81.0/81.8 →
  108.1/107.0 (+32 %), RTX 5080 --resident-experts 26.7/26.6 → 29.5/28.5; greedy byte-identical
  without the flag. 0d761d99: +7.0 % on an A4000, −2.6 % on a RAM-bound 3060 box. 661ff293: RTX 5090
  Laptop with --expert-cache-per-layer 84.6 vs 106.6 tok/s with the overlap (21 % slower). In
  f1864300: host plan/routes/promotions moved into the commit: tg512 plain 88.70 → 89.96, MTP
  133.54 → 138.70. In a6a6b4f7: post-prompt promotions −3.3 %/−7.7 %/−8.9 % at 32K/131K.
- **arcint has it?** No for the overlap forms: ARCINT.md's tier is "a plan every 6 tokens, ≤96 swaps,
  landing 2 steps later" — blocking in the same way. The measured boundary (per-layer-cache box
  −21 %, RAM-bound box −2.6 %) is the A/B the B60 needs; the determinism cost (not bit-exact run to
  run) is inside the 0.03-nats bar but outside a determinism audit.
- **Why it ranks ninth**: +10–32 % decode is large, but the boundary measurements say it can lose on
  a per-layer slot box (the B60's shape is exactly arcint's per-layer slots), and the async tier
  makes runs non-bit-exact. It must be A/B'd, not adopted.
- **Cost to try**: medium — the four-step round on a helper thread, the copy-stream landing signal
  (a 4-byte D2H poll, not the host function), the per-layer wait; the plan-in-the-commit move is
  trivial and free.
- **Experiment**: B60, Flash-Next IQ3_S 40 % residency; arms blocking / async / async+overlap /
  overlap alone; metric decode t/s + ms/window + teacher-forced KL; bar: ≥+5 % decode, KL within bar,
  and the run must stay non-degenerate over 20 varied requests.

### 10. PCIe copy discipline (small per-round uploads off the copy engine; batched expert copies; the
copy-only queue) — score 7
(merged: `pcie-copy-kernel-no-host-calls`, `pcie-copy-batching`, `xe-copy-engine`, `expert-upload-batch`)
- **Evidence**: In ecd4de5d (upload_pinned: tg512 85.46 → 88.81, MTP 121.20 → 131.78), 5a5a4a9e
  (70.29 → 83.30), b5bacdb1 (batched promotions, first token −8.6 %/−12.2 %), In a6934c2b (KV page
  tables + grammar masks off the copy engine), St 9253c92a (host stuck in cudaMemcpyAsync), 01fe68dd
  (copy engine ~13 % of a B70 prompt), cd566059 (a copy-only queue runs no kernel — the flag must be
  a copy), a3642eb4 (STRATA_DMA_BATCH), In a8a042ae (scattered 16-byte reads of pinned host memory
  ~10–12 GB/s vs 24 GB/s in a compact window). `paper`.
- **Measured**: ecd4de5d: nsys before — 4.7 ms per MTP round between verification and commit, 3.4 ms
  of it in the stalled copy (~40 × 2.76 MB per MTP round); after: same ids, plain +3.35 tok/s
  (85.46 → 88.81), MTP +10.6 (121.20 → 131.78). b5bacdb1: 12,300 records as single memcpyAsync
  blocked the host 316–416 ms vs 1.3 ms batched; pp32k first token −8.6 %, pp128k −12.2 %. 01fe68dd:
  the Level Zero v2 adapter ran the prompt's expert copies on the compute engine, ~13 % of a B70
  prompt. St 9253c92a: IQ3_S decode 45.3 → 44.8 with the copy kernel (no loss), --pcie-mode dma
  keeps the old path.
- **arcint has it?** Unknown/partly: ARCINT.md says the B60 copy engine is shared with the input
  uploads and "background copies must go right after a graph". The audits: which per-round inputs
  still go through the copy engine (round inputs, sampling configs, acceptance inputs, commit counts,
  drafter ids, residency table, KV page tables, grammar masks); do a plan's ≤96 swaps issue as one
  copy each; are prompt expert copies on a copy-only queue (and any flag a copy, not a kernel).
- **Why it ranks tenth**: measured gains are +3–10 % decode and first-token −9…−12 %, and the B60
  shape matches — but it is a set of small mechanical fixes whose size depends on how many of them
  arcint still pays; the audit is the first step.
- **Cost to try**: low — a timeline (nsys/unitrace) of one decode round and one MTP round, then
  per-round uploads as one kernel copy over USM-mapped pinned memory, swaps batched, a copy-only
  queue for fills.
- **Experiment**: B60, Flash-Next IQ3_S, MTP 4; timeline one round; then arm per-round uploads via
  kernel copy vs copy engine; metric decode t/s and the round's inter-phase gap; bar: ≥+3 % aged
  over the current path, tokens identical.

---

## Next 10 (verified, ranked 11–20)

11. **prefill chunk / residency dial + ring byte budget — 7.** St 8fef4f1f (ring slot 0.53 tok/s vs
    chunk token 0.05, +5.9 %), 56501a38 (short prompts keep 0.1.39's ring), e8de49eb (group gathers,
    −8.6…−12.6 % from per-expert waits), 6fd12038 (B70 4,095-token prompt 4096 chunks 784 → 980),
    219ae436 (B70 borrowing config 16K 1,222 → 1,347, 32K 1,187 → 1,361; KL 1.1e-2 nats vs 4096),
    1fac68d6 (4096 chunk under 16 GiB: decode +2 % agent / +0.4–9 % code, cold 8K prompt 9 % slower),
    In 4bf28507 (16,384 chunk pp32768 9,639 → 11,211, pp65536 9,887 → 11,268, exact route identical).
    all `paper`. arcint has neither the byte-budget bisection nor the borrowing 8192 chunk; its
    coder's ubatch 1024 (+15 %) / 2048 (spills) is an ad-hoc version of the dial. Cost low:
    sweep chunk × ring on the B60. Bar: ≥+10 % prefill at 20–32K with decode unchanged.
12. **late-layer-approximate-prefill (KVA) — 7.** St f4012d63 (P100: 5,642 tok 203.3 → 265.7 ppts
    1.31×/54 %; 14,581 202.5 → 311.2 1.54×/85 %; 29,997 198.7 → 326.5 1.64×/92 %; decode unchanged,
    needles identical at 10/50/90 %), 011e5bec (QSA layers 0.153 residual at 1.08× vs GDN layers
    0.339 at 1.18×), c5cf218e (state-only GDN), 750ca8c3 (default-on) → **e3450e45 ships OFF in 0.2.7**
    after "lost records and invented rows" in long extraction and agent-loop defects with a ~10K
    system prompt, bda30050 (final opt-in: layers 36–47, 2048 tail; RTX 5070 agents 6/10 vs exact
    4 and 7, extraction 585/606 vs 585; +11 % at 14K / +18 % at 30K). `paper`. arcint lacks it; the
    prize is 1.3–1.6× prefill but the answer-level damage is real — rank limited by risk. Cost
    high; experiment must run the task battery, not the speed arm alone.
13. **MTP width / acceptance policy family — 7.** In 3d1931c7 (auto width: 27B +17 % on code at one
    request, geomean +1.6 %; 35B +6.1 %; 64K +3.5 %/−0.6 %; copy-heavy rewrites auto n-gram 1,298.5
    vs 612.0 tok/s), 070b91f8 (two gated drafts +14.0 % geomean on an offloaded 122B), 140faefc
    (device IF gate: 34 % of draft steps past the cut, +2.2 %/+3.5 %), 3d299bce/3d1931c7 (adaptive
    +2.0–2.1 % geomean over best fixed), St e83b82af (cumulative confidence cut: 71.7 → 76.9 tok/s
    on 4× R9700, same ids), 5e0c43ba (chain policy: +6.4 % tokens/step for +7 % window time).
    `paper`. arcint has fixed K (5 general, 4 coder) and the min-p 0.5 stop; the auto/cost-aware
    width and the cumulative cut are the gaps. Cost low-medium (host policy + a few measurements).
14. **expert profile built from own routing traces — 7.** St 8cb26e60 (per-card xe profile: B70
    +5..64 % decode, mean +21 %, 12 unseen prompts, same tokens), 3d6f1718 (--expert-profile-save
    from device counts: code +13..19 %, prose −18 % vs the xe default), 017a672a (bench must not
    overwrite the user's profile), 90d32517 (the shipped base made --reorder a no-op), In df5b88ac
    (credit 256 × prompt-route share: −1.8 % time, hits 82.1 → 82.5 %). `paper`. arcint has a
    calibration profile but ARCINT.md does not say it is per-card/per-workload; the +21 % mean is the
    prior. Cost low (traces + profile tool). Measure per-card profile vs shipped on the B60.
15. **agent prefix reuse (branch tap, tokenise-from-prefix, checkpoint spread, tool-call hold,
    slot-turn affinity) — 6.** In e8d09e27 (40K transcript re-sent: 40,128 vs 32,768 tokens reused,
    TTFT 0.6–1.1 s vs 2.1–2.2 s), St f13a870d (tokenise only after the longest shared boundary),
    a7d4b754 (spread checkpoints: 210,528-token prompt resumed at 9,525 and read 201,003 again vs
    8K with the flag), In 4d97316e (tool-call holds: TTFT 12–17 % below every base run), dd08a7d0
    (slot by held-turn checkpoint), 624070c5 (min-tokens so side requests don't evict the long one).
    `paper`. arcint's kv-checkpoint-restore campaign is open; these are the missing pieces. Cost
    low-medium; arcint already has per-token snapshots, not whole-turn checkpoints.
16. **decode launch fusions (GDN/HC/qk-rope/commit/verify window) — 6.** In 3fa751eb (+1.8 % plain /
    +1.1 % MTP), 46141825 (+0.8 % plain; gate-into-recurrence), 21dbcad5 (5 → 2 launches, T ≤ 16),
    4881576c (HC collapse in epilogue: 512 columns −16 %), 1a78f383/441ca140 (qk norm+rope fused:
    +22–52 % through 256 tokens, wins at every width to 8192), St b08cf3e1 (−4 launches/MoE layer,
    −(n−1)/attention layer), 352cad8f (one-token self-commit), 6f3e3810 (commit 2.35 → 0.65 ms).
    `paper`. arcint has some fusions (router, conv snapshots, GDN verify, norm→q8_1) but not these;
    the sum is a few % of decode, and `launch-count-beats-kernel-sum` (In 0d76fe11: +96 launches/step
    made decode 4.86 % slower while kernel sum fell 6.58 %; In e935cce4: op-bench-faster E2E −0.46 %)
    says judge at the wall, not the kernel. Cost low each; bundle 3–4.
17. **KV bytes for the 245k agent — 6.** St 0616dfb7/21ccac1b (B70: 128K int8 8.5 → 38.7 tok/s;
    256K 3.7 → 31.3), 3597c47f (k8v4 816 B/cell vs int8 1,056; needles identical), 40542e8b (q4_0 KV:
    ppl +8–12 %, needles 5/5, +3.7 % decode at 128K), In 083c4a65 (vq2 132 B, k4v2 196 B per token
    per KV head; PPL 64K: vq2 4.1859, k4v2 4.1709 vs BF16 4.1661), In 7a4a13b4 (every KV format on
    Flash-Next; 27 → 18 µs per decode call), In 10143504 (page space host plane; 128K prefill
    attention 248 → ~52 ms per late chunk). `paper`. arcint runs q8_0:q4_0 fully resident; the
    streaming/host-plane residency trade is the open one; the quality price of q4_0 K is known.
18. **decode-once expert matvecs in the verify window — 6.** St e65858f8 (KqSplit, B70, 2-/4-token
    windows: Q4_K/Q5_1 362 → 402 / 369 → 425, Q5_K/Q8_0 395 → 440 / 404 → 464 GB/s, bitwise),
    983fa18a (KQ split 4-token +13 %, Q6_K +20 %, Q8_0 +15 %, 1 token −1.8 %, bitwise), ffac97f7
    (interleaved q8_1 copy, 2–4 token dense projections), 2715a752, 75fabbd9 (5–8 column kernels,
    T=4 → T=5 cost cliff +2.0 ms), 546d7675 (NVFP4 per-entry kernels 1.14×→2.48× by tokens/expert),
    In 99458126 (small-T routes C4 −12.9 %/C8 −34.1 %). **Measured negative**: St 2a0d2f55 (opt-in
    wide kernels reach 2× at kernel level but "NONE reaches +3 % end to end" on the B70, 54.31 →
    54.00 decode); f7610d2c/668072aa (interleaved matvec default reverted: 15 % slower on a two-GPU
    Windows split). `paper`. arcint's verify window is where most of this lands; the negative says
    price at the engine, not the kernel.
19. **xe device hygiene (aliased pages, JIT >4 GiB pointers, pageable host, arena fill, USM DMA
    deadlock) — 5, correctness-heavy.** St 78f61744 (a B70's 22 GiB arena aliased two 2 MiB pages →
    NaN at layer 4, expert 249; tag-check + re-allocate behind a spacer), defadd6e (IGC JIT 32-bit
    pointer offsets: experts in the last 0.46 GiB of a 4.46 GiB cache dequantized to NaN; pass
    -ze-opt-greater-than-4GB-buffer-required), 7a07f281 (the staging is NOT needed on bmg: first-token
    logits identical at 4K/49K; it cost ~10 % of a prompt), 73e01fa5 (a pageable host read on xe
    faults and loses the device), 6d1cb3a6 (one 25 GiB memset times out on the blitter), fca3c595
    (pageable-arena DMA deadlock on A750 i915; B70 no-stream prefill 680–740 vs 77–456 tok/s),
    01fe68dd. `paper`. Cheap audits; silent corruption class on arcint's exact arena.
20. **quality instruments (assistant-token KL mask, precise reference, noise-twin floor, FP32
    router/head logits) — 5, cheap and gating.** In 3ed21427 (99 % of every arm's KL sat in
    positions the model never writes; on assistant tokens 1.4e-3 to 3.9e-4; noise twin 4.8e-5),
    bf216716 (precise reference: block error 2.4e-3 → 4.2e-4; floors 5.7e-3 BF16/INT8, 2.0e-2 VQ2;
    the floor-0 trap), 4d4beab3/34a0def1 (noise twin + group/strength attribution), 75029865 (the
    model rounds router logits to BF16 → ties), 0f2945f3 (FP32 head logits: top-1 flips 1.53× →
    0.74× the noise twin), 32731f06 (BF16 head logits flip greedy top-1 at 1.3 % of positions),
    b02043eb (two valid FP64 references disagree — do not chase bit-exactness). `paper`. arcint's
    0.03-nats bar has no floor and is measured over whole windows (inferred from ARCINT.md); the
    mask + a noise floor is a cheap correctness gate for every candidate above.

---

## Lower-ranked concepts (kept for completeness, 1–4)

- **prefill: idle-wide-prefill** (In 0c69497b/d64894bf: 27B pp1024 +19.3 %, pp2048 +11.5 %) — 4,
  single-request first step at the idle width; cheap sweep, 27B agent not Flash-Next.
- **prefill: prefill-cpu-share-small-chunk / prefill-cpu-split** (St 978d3558 512-token prompts
  −49 %, ed6a02cd table; In 44b5a108/e8eb9ee3 0.3 of measured rates, 3 workers −11.1…−18.9 %) — 4,
  agent tool-result chunks; on the 5700X over-planning the CPU share is the risk.
- **prefill: prefill-mmq-ring group gathers** (St e8de49eb −8.6…−12.6 %, 653f0113) — 3.
- **prefill: hyper-connection-fusion prefill** (St 2a6bbc73/3945aac6 +4–5 % prompt at 32K/128K;
  In 4881576c) — 3.
- **prefill: prefill-matrix-core-attention** (St f05ccd3b R9700 −33 % attention stage; 1f079f9d/
  fd81df69 default on gfx12) and **prefill-tc-flash-attention** (In 34bf2816 848 → 2,730 tok/s
  pp4096) — 4: arcint already runs XMX attention patches; the AMD kernels do not port.
- **decode: mtp-head-multiblock-argmax / mtp-drafter-head-read** (St 3e4aa1d6: 128 blocks per row;
  e4328457: head walked once 4.55 → 2.83 ms; a266bdfe Q8 head 66 → 33 MB) — 4, cheap.
- **decode: fused-multitoken-router** (St 822251be/868f2ef0: one warp per token, 8/CTA; bitwise) — 3.
- **decode: gdn-tree-closed-form** (In 0f5707ce: 0.65–0.93× the walk at 1–4 rows; +1.19–1.6 %
  decode registered) — 4, arcint's MTP verify is K=4–5.
- **decode: small-t-streaming-gemm** (In 848cd126/98458126/80171ee1: C4 −12.9 %, C8 −34.1 %;
  Flash-Next dense projections 5.5 ms vs ~138 ms at T≤8) — 4, per-shape route table.
- **decode: split-kv-decode-attention** (In b3caabb4/d0564202/82830da6: split keys across CTAs,
  live-context split count, W=8 512 keys 37.6 → 27.4 µs) — 4, agent at depth.
- **decode: lm-head-dramp-bound-route** (In a702ed09: BF16 TC head 701–710 µs at a ~709 µs DRAM
  floor; Q8 head 9–16 columns 1,209–1,494 → 849–1,123 µs) — 4.
- **decode: moe-narrow-pass-column-bound** (In 9c47608f/df3d789c: bound 4 → 8 columns, +2.1 % at
  C=1/2, +12.1 % plain at C=8; per-width bench W4A4 1.26–2.29×) — 4.
- **decode: qsa-decode-scores-rows / qsa-decode-attention-tile-stage** (St 5b6c9398: 47/57/78/101 →
  36/38/49/55 µs at T 1–4; In cbcde52a: exposed attention 45 → 18 µs) — 4, folds into #5.
- **decode: gdn-small-t-layer-fusion** (In 3b6c32cb/ad17bc62: 12 → 8 graph nodes; T6 73.5 → 58.5 µs)
  — 4, folds into #16.
- **decode: draft-cumulative-confidence-cut / spec-window-cost-aware-policy / policy-gated-suffix-draft**
  — folds into #13.
- **decode: dflash2-drafter-experiment / dflash-block-drafter** (In 7dbe99a3: IQ3_S 258/210/183 vs
  MTP auto 233/192/167; ce717f08: block drafter +9.9 % on long reasoning, −11.4 % code, −39.8 %
  story) — 5, a second drafter family; big port (776 MB drafter, taps, cost model), mission item 2.
- **decode: mtp-verify-rows-per-req / mtp-draft-step-confidence-gate** — folds into #13/#16.
- **decode: kv-int8-groupwise / int8-decode-attention-pipeline** (In 35cff1e7 design; e408bd3d
  measured negative 2.4–4.8× slower than bf16; 8bb2ce10 positive: 512 keys 37.6 → 29.3 µs) — 4,
  integer KV decode in-register is the working form, dequant-in-the-loop is the trap.
- **decode: cross-kernel-weight-l2-prefetch / l2-warm-next-layer-cpu-wait** (In ce954918 −6.67 % op;
  7f14d963 +1.7–1.9 % decode; 1e3cf453 +1.6 %) — 4: warm the next layer's weights in the CPU-miss
  wait; OpenCL has no per-load hint (inferred: a hint kernel).
- **decode: decode-kernel-load-issuance-order** (In e2638a8c: +11.0 %/ +10.8 % MTP) — 4.
- **decode: no-p2p-peer-sums / layer-split-handoff-pipe / layer-split-overlap / mtp-draft-stage0**
  (St: 3.4K 332 → 805, 8K 762 → 1,284; hand-off 3.8–4.0 → 1.85–1.89 ms; split overlap 53 → 70.1
  tok/s; draft on stage 0 +21 %) — 4: the B60+A770 split's cards; arcint single-card today.
- **expert cache: eviction-policy-replay-harness / expert-cache-lfru / per-layer-cache-quota-negative**
  (St 49c50f49 GPU-free replay; In db71d3f6 halved LFRU 9–12 % fewer misses; In 8c30c097 hard
  per-layer quotas lost 3–34 %) — 4: the ARCINT.md verdict is LFRU no-gain at 96 swaps; the
  harness is the tool.
- **expert cache: foresight-swap-space / expert-cache-elastic-resize / resident-ram-follow-decode-routing
  / post-prefill-promotion-negative** — 3–4, folds into #9 (plan/promotion placement).
- **expert cache: expert-ram-exclusive-refill** (In d8652998: Dense8 49.1 vs 64.4 GiB, decode 0.98×;
  IQ3_S 41.1 vs 48.0, 0.88×; GGUF adaptive 1.07× median, LFRU 0.97×) — 4: the RAM-footprint dial.
- **host: huge pages / pinned DMA** (St 8ca4e160 ~750 TLB misses/expert; bb19becc THP=always
  madvise trap 25 s → 432 s; 6b395578 compaction stall; 2dde837d/3c8a4529 2 MiB pages + pin:
  11.9 vs 30.4 s DMA for the same 140.2 GB stream) — 4, cheap startup check.
- **host: startup-unbuffered-fill / linux-read-ahead / expert-profile-fill-pipeline** (In 8480d610
  90.9 → 25–27 s; St cb33c0ba 920 → 70 s; cbe442bf 76.2 → 18.8 s) — 4, start time not decode.
- **host: host-governor-power-throttle** (In 5c127ef9: Windows throttling cut decode 2.6×; pinning
  added nothing) — 3, Linux governor check.
- **host: cgroup-aware-memory-guard / host-ram-ledger-headroom / arena-hugetlb-guard** — 3.
- **serving: conversation-cache family** (17 commits; in #15) — 6.
- **serving: serve-reply-room / serve-fit-max-tokens** (St 17ba2d71, f31b6efd: Claude Code asks
  32,000 max_tokens every request; clamp instead of 400) — 3, agent serve.
- **serving: prompt-tokenise-from-prefix** — in #15.
- **multi-gpu: pcie-link-aware-split** (St 0391e478 ~350 → ~1,410 tok/s on an x1 link; 9d41e18b
  relative 4× gate; b8844516 measured negative: at a 4.2× gap decode −4…−8 %) — 4: the B60+A770
  split's placement; the negative is why it is not higher.
- **multi-gpu: sycl-split-headroom-pick / gpu-fit-rule / pipeline-windows** (St 9c5a45f8 ~5× gap;
  ac38793c largest-card+RAM-first rule) — 3.
- **multi-gpu: primary-device-helper-cache / prefill-helper-source** (St 10552345 37.5 → 44.2 tok/s
  "NOT COMPILED YET, not run"; 268647d2 91.5 → 65.7 s prefill, −28.3 %) — 4, A770 as helper.
- **quant: q8-packed-planes / plane-split-nibble-quant-layout / full-group-decode-gemv /
  smem-staged-tile-gemv** — 3: decode GEMV layouts; the none-E2E negatives (2a0d2f55) apply.
- **quant: ple-fp8-table / ple-raw-upload-device-dequant / embed-table-vram** — 3.
- **quality: q8_1-finite-quantizer / nonfinite-logits-self-heal / router-logsit-precision-guard** — 4,
  correctness (NaN → one token repeated; BF16 ties flip experts).
- **quality: calibrated-order expert seed** (In 2337586b: first request MTP decode +9 %/+15 %/+56 %)
  — folds into #14.
- **wrong-tool warnings: usm-dma-pageable-staging-deadlock, sycl-pinned-miss-mirror, sycl-* and
  Strata SYCL port commits** — 3–4: the `strata-sycl-port` cluster (32 commits) is arcint's
  strata-sycl-b60 campaign reference (fp32 divide flags, doorbell atomics, per-kernel module, AOT
  build 245 → 1 ms first launch); not a new lever.
- **measured negatives worth recording**: St 5865998a (polling the final wait cost IQ3_S ~3 %),
  a74acb6d (GPU-computed PCIe misses no faster than CPU), 83426781 (empty-PCIe-group drop no gain),
  0aa72eab (device plan within noise), 6b5761d3 (selection union too sparse to share), f7610d2c
  (interleaved matvec −15 % on a two-GPU split), 39841946 (mmvq staging 10–20 % slower),
  In 8c30c097 (per-layer quotas), In 21d6d7ff (T=1 long-context attention 9× slower than T=2 on the
  same KV bytes), Fr 95ac86f (hybrid −33 % at 25 % offload / +15.9 % at 10 %).
