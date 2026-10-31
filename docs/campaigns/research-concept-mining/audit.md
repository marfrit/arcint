# Wave 0 audit — the kept candidates against arcint's actual code

## The table

Scope: the candidates in `state/candidates.md` that do not trade quality for speed (operator,
2026-10-29), audited against source; nothing was built or run. `prize` is what a reference measured,
marked `paper`; `cost` is for arcint (S = days of card-window work, not calendar); `order` is my
proposed build order after the audit — see "Order" at the end.

| item | arcint status | quality class | units | prize | cost | order |
|---|---|---|---|---|---|---|
| 1 prefill-layer-major | **partly** — llama.cpp already walks layers inside every ubatch (`code`); the bank stream reads each bank tensor whole once per ubatch (`code`); the across-ubatch part was only ever priced as a chunk sweep | rounding-only | Flash-Next (B60), coder (A770) | Strata 4.8-5.7× at 20K on 2×RTX A4000 (`paper`); **arcint's own chunk arm: +3.1 % at 8192 against 2048** (`paper`, arcint `measured-here`) | high | 11 |
| 2 int8-DPAS prompt expert GEMM | **partly** — grouped-by-expert DPAS GEMM exists (`code`), int8 DPAS exists only on the dense 2D path (`code`) | rounding-only | Flash-Next (B60), coder (A770) | IQ3_S B70 1,024 tok 645.8→775.4, 4,095 1,257.0→1,413.8 (`paper`); KL 2e-4 (`paper`) | med-high | **1** |
| 3 PCIe probe + miss balance | **lacks** — no probe anywhere (`code`); the miss split is a fixed 64 pairs a layer (`code`) | rounding-only | Flash-Next (B60) | slow-CPU shape 16.77/17.17→12.98/12.82 ms a round (`paper`); +9.9 % at 8K, +25.6 % at 32K (`paper`) | low-med | **2** |
| 5 QSA select | **partly** — the block-score + top-k graph exists (`code`); the kernel forms are the missing part | exact (ids) | **Flash-Next only** (QSA is its 12 attention layers) | 128K prompt 1,608→1,843 t/s (+14.6 %), 243K +26 % (`paper`) | medium | 7 |
| 6 CPU miss AVX2 | **has most** — multi-token rows, named accumulators, decode-once chunks (`code`, 0063); prefetch and cold-page prefetch lack | exact | Flash-Next (B60) | gate/up −4.0 % on a 5700X3D (`paper`) | low | 8 |
| 7 xe doorbell | **has** — the uncached load hint and SVM-scope release/acquire are in 0056 (`code`) | exact | Flash-Next | B60 263 ms→2.4 µs, 8.7-10.5→19.3 t/s (`paper`) | none | — |
| 9 tier overlap | **has** — patch 0042: own queue, behind the graph, landed later (`code`) | exact | Flash-Next | +2.4 % decode, input wait 2.6-6.0→0.94-0.95 ms (arcint `measured-here`) | none (one sub-piece left) | — |
| 10 PCIe copy discipline | **has most** — 0047 inputs, 0042 swaps, 0046 tables, 0032 copy queue (`code`) | exact | all three | coder inputs 1.93-2.02 ms→133-255 µs, +14-15 % (arcint `measured-here`) | low (one leftover) | 9 |
| 11 prefill chunk/residency dial | **partly** — chunk and slot budget are flags, swept by hand; ring fixed at 4 (`code`) | rounding-only | Flash-Next, coder | +5.9 % (`paper`); B70 16K 1,222→1,347, 32K 1,187→1,361 (`paper`) | low | 4 |
| 13 MTP width policy | **lacks** — fixed K with a min-p stop only (`code`) | rounding-only | all three | auto width +17 % code / +1.6 % geomean (`paper`); adaptive +2.0-2.1 % geomean (`paper`) | low-med | **3** |
| 14 per-card expert profile | **built, opt-in, off by default** — reads and writes Strata's STRP file, one file for all layers (`code`); the save is 0070, the continuous trace 0069, and the replay/profile tools are device-free (`code`) | rounding-only | Flash-Next (B60) | B70 mean +21 % over 12 unseen prompts, same tokens (`paper`) | low | 5 |
| 15 agent prefix reuse | **partly** — checkpoint restore + longest-prefix reuse exist (`code`); no tokenise-from-prefix, no branch taps, no tool-call hold | rounding-only | agent, coder, Flash-Next | 40K transcript TTFT 2.1-2.2→0.6-1.1 s (`paper`) | low-med | 6 |
| 16 decode launch fusions | **partly** — 0004/0027/0049/0050/0051/0064/0065/0066 are built (`code`); qk-norm+rope and the commit form lack | exact | coder (A770), Flash-Next | 0064 router 34.1→13.5 µs; coder −1.4…−2.5 %/step (arcint `measured-here`) | low | 10 |
| 18 decode-once expert matvecs | **measured negative** on the reference's mechanism (`paper`, arcint); the dense few-column part is built (`code`, 0009/0001/0022) | exact | Flash-Next (verify), denser units | St KqSplit 362→402 GB/s bitwise (`paper`); **arcint 30.5→29.5 t/s, verify 15.01→15.56 s** (arcint `measured-here`) | — | out |
| 19 xe device hygiene | **partly** — no JIT 4-GiB option and no alias pre-check (`code`); the load-time copy verify is a tripwire (`code`) | exact/tripwire | all (Flash-Next's arena most) | NaN at layer 4, expert 249; experts in the last 0.46 GiB of 4.46 GiB to NaN (`paper`) | low | 12 |
| 20 quality instruments | **partly** — KL capture/compare tools exist with the QSA boundary (`code`); no assistant-token mask, no noise twin | n/a (gate) | all | 99 % of an arm's KL in positions the model never writes (`paper`) | low | 13 |

Two corrections to `state/candidates.md` the audit forces: **#5's unit is Flash-Next, not the agent**
(QSA is `qwen4exp`'s 12 attention layers, `code`: `llama-src/src/models/qwen4exp.cpp:739-844`; the
agent's arch has no indexer, `code`: nothing in `qwen4exp`'s path is used by `qwen35`), and **#1's
arcint prize is not 4.8-5.7×** — the mechanism is already inside a ubatch, and the measured
across-ubatch arm is +3.1 % (`paper`, arcint stage 5). See item 1 below for why.

Evidence classes on every claim: `code` = read in `arcint-src/` (arcint at `qfndev`, 3f449ff) or
`llama-src/` (llama.cpp at the pin `3e4dfa8` with the served series, `0001-0072`, without 0031,
0055, 0061); line numbers below are that patched tree's, which is not the pin's -- the series
shifts them (the router's f32 accumulator: `:2171` here, `:2155` at the pin); `paper` = what a commit message, a document or arcint's own campaign record states
(arcint's campaign numbers are its own `measured-here` class, quoted here as `paper`); "(inferred)" =
my reasoning from the code and the record. Out of scope and not audited: 4 route-resident bias, 8
screened vocab head, 12 late-layer approximate prefill, 17 lower-precision KV — all four change the
answer by design.

---

## 1. prefill-layer-major

- **arcint status: partly.** Two separate facts decide this.
  - *Inside* a ubatch, llama.cpp already walks the layers over the ubatch's whole token set:
    `llama-src/src/llama-context.cpp:1658-1699` (`process_ubatch`), the graph is built per ubatch at
    `:2094`, and the layer loop runs at `llama-src/src/models/qwen4exp.cpp:449-490` with `n_tokens` of
    the whole ubatch. So a 2,048-token ubatch is a layer-major walk of 2,048 tokens; "chunk-major" is
    only the *sequence* of ubatches, each of which re-walks all 48 layers (`code`).
  - *Across* ubatches, arcint's bank stream (0032) does not read routed experts per chunk as Strata
    does: it copies **each bank tensor whole** once per ubatch into a VRAM ring, whatever the routing
    (`llama-src/ggml/src/ggml-opencl/ggml-opencl.cpp:27696-27814`, `GGML_OPENCL_BANK_STREAM`, ring
    default 4, `min_tok` 256 at `:27691`). At 2,048 tokens a ubatch a layer's bank is already read
    once, so the per-layer read-once effect Strata's layer-major buys is already there at ubatch
    granularity (`code`).
  - What is left is the duplication across ubatches: a 20k prompt is 10 ubatches, so the bank is read
    ~10 times; a span-major walk of 8,192 tokens would read it ~3 times (`code`, `inferred`).
  - **The measured arm does not support a large prize.** arcint's own chunk sweep (stage 5, IQ2_XS,
    B60, 13,000 MiB of slots, the 20,045-token needle): chunk 2,048 **422.1** t/s, 4,096 443.9
    (+5.2 %), 8,192 **435.0** (+3.1 %) with peak VRAM 21.2 / 22.2 / 24.3 GB, and at 11,500 MiB 8,192
    432.4 / 27.5 decode (`paper`, arcint `measured-here`). An 8,192-token ubatch *is* the span-major
    bank saving for that prompt, and it bought +3.1 %: the bank copies overlap on the copy engine
    (0032's own gain was 416.1→497.6 t/s, `paper`) and are not on the critical path at this size.
  - The remaining headroom is not bank bytes but **expert-tile padding**: 0006's tile is 32 tokens wide
    per expert (`paper`, README 0006) and a 2,048-token chunk routes ~40 pairs an expert, so many
    experts get a second, nearly empty tile that dequantizes the expert's whole slice again. There is a
    probe for exactly this: `GGML_OPENCL_MOE_TILES_DEBUG` prints `[MOE_TILES] … pad=%.1f%%`
    (`code`, `ggml-opencl.cpp:30780-30797`). Nobody has read it at Flash-Next's routing. That number
    decides whether a span-major build is worth more than +3 %.
  - Strata's campaign numbers (8K 2.8×, 20K 4.8-5.7×, 32K 5.1-5.5× on 2×RTX A4000, `paper`) were
    taken against a chunk-major engine that streams only the chunk's routed experts; they do not
    transfer to an engine that already copies the whole bank once per ubatch. Infernix's 40K prompt
    1.39k→2.90k tok/s at chunk 4,096 (`paper`) is the sibling shape, likewise a streaming engine.
- **How arcint's build differs from the reference's mechanism.** Strata stages *routed* experts per
  layer for the span in a ring borrowed from cache slots, on a copy queue with a next-layer prefetch;
  arcint copies *all* bank experts per layer per ubatch on a copy queue in a ring of 4, and keeps the
  expert cache's slot/bank split (a decode policy) untouched by prefill. Both are OpenCL here
  (the reference figures are CUDA/SYCL); the GPU reads the bank by pointer in decode and from the ring
  in prefill (`code`). A negative arcint recorded — chunk 8,192 is slower in decode because the
  compute reservation eats slot budget (`paper`) — did test the reference's *span* (one ubatch at a
  span) but not its per-layer staging, so it is not a verdict on the reference's mechanism.
- **Quality class: rounding-only.** The MoE GEMM per-row sums do not change with the tile count (each
  row's K is walked inside its own work-group's DPAS chain, `code`: `kernels/mul_mm_kq_f16.cl`), so
  the expert arithmetic is the same bytes; attention tiled at different row blocks reorders an f32
  accumulation, and the bank copies stay byte-checked (`llama-expert-cache.cpp:570-589`). Strata
  measured bit-equal output at its arm (`paper`). Judged at the answer-level bar, never approximate.
- **Units.** Flash-Next (B60) is the only unit with a host bank: the coder's 184 experts are resident
  on the A770 (`code`: no `.bank` tensors) and the agent is dense. The coder's prefill chunk 1,024 is
  already the dial's smallest point; span-major would apply to its padding only, with no bank saving.
- **Pricing measurement (existing knobs, no new code).**
  1. Tile padding at Flash-Next's routing, the cheapest decisive number:
     `GGML_OPENCL_MOE_TILES_DEBUG=1 tests/acceptance/cells/served_check.py flashnext <arcint> <IQ3_S.gguf> GPU.0 <assets>`
     and read the `[MOE_TILES] … pad=` lines at chunk 2,048 and at chunk 8,192 (a second process with
     `--prefill-chunk 8192 --llama-reserve-cells <the arm that fits>`). A pad of ~50 % at 2,048 and
     ~0 % at 8,192 is the prize; near 0 % at both, the build is dead.
  2. The bank copies' share of the wall: a `GGML_OPENCL_PROFILING` build of the pin, one 20k prefill,
     `tools/cl_timeline_steps.py` on the OpenCL intercept trace (`code`, the tool), then the copy
     engine counters from the card's fdinfo (`drm-cycles-ccs` against `drm-total-cycles-ccs`) — the
     method stage 5 used (`paper`). Bar: if copies are <5 % of the wall at 2,048, only padding counts.
  3. A chunk sweep with the reserve-cells flag to see if 8,192 can keep 2,048's VRAM:
     `--prefill-chunk 8192 --llama-reserve-cells 20480` against `--prefill-chunk 2048`, the needle
     prefill and decode after it, one fresh process an arm, `tests/acceptance/ab.py flashnext …`.
- **Implementation sketch** (see `design/01-layer-major.md` for the full design): a new
  `llama_context::decode` span path that splits the batch into spans first, then ubatches inside the
  span; the graph gains a span-token dimension in `build_moe_ffn`'s GEMM arms
  (`llama-graph.cpp:2487-2520`) and the attention tile loop stays at the current ubatch width
  (`qwen4exp.cpp:467-469`). Where: `src/llama-context.cpp`, `src/llama-graph.cpp`,
  `src/models/qwen4exp.cpp`, a new patch 0068. Size: ~400-600 lines, plus the memory module's
  span batch handling. Cost: high.

- **Re-priced by the 2026-10-31 profiles (`measured-here`, the B60, Flash-Next IQ3_S, the intercept layer's chrome
  trace; 2,048 / 8,192 tokens without the loan, 19,196 / 103,638 with it at chunk 8,192):** the main queue is the
  critical path (busy 96.3 % of the 100K span); the bank stream's `clEnqueueWriteBuffer` runs on its own queue (36.8 %
  of the span, overlapped, as 0032 says). The largest main-queue cost is `kernel_moe_gather_experts` (0021's copy of
  the routed experts' bytes from the host bank into a card-side mirror): 38.5 % of the span at 2K, 31.8 % at 8K,
  20.3 % at 20K, 25.7 % at 100K; 1,998 calls at 4.08 ms at 20K and 8,892 at 8.69 ms at 100K - the calls scale with the
  ubatch count (4.5x for 4.3x the ubatches) and the per-call cost doubles under the loan (fewer slots, more bank
  experts). It is paid per ubatch and per layer on the critical path, which the chunk sweep above (where the bank's
  4x fewer copies 'did not show' because the copy engine overlaps them) did not separate. Prize if each layer's
  gather were paid once per prefill: up to ~24 % of the 100K prefill (`inferred`: 25.7 % x 12/13). The open question is
  why the bank stream and the gather both move expert bytes, and whether the gather can move to the copy queue,
  dedupe across ubatches or shrink to the misses (FreeToken's `_prefetch_split`, `paper`) - research first.

## 2. int8-DPAS grouped expert GEMM for the prompt

- **arcint status: partly.** The *grouped-by-expert* half is built: the prefill expert GEMM sorts
  `(token, slot)` pairs by expert into 32-token tiles on the device (`kernel_moe_route_tiles`, a route
  cache shared by gate/up/down) and runs one fp16-DPAS tile per work-group with the tokens as the
  DPAS a operand (`code`: `ggml-opencl.cpp:30870-31080`; README 0006 at
  `arcint-src/contrib/llama.cpp/README.md:350-401`). The *int8* half is not: int8 DPAS exists for the
  dense 2D GEMM only, Q4_K (`mul_mm_kq_2d.cl`'s `kernel_mul_mm_kq2d_q4_K_i8`, README 0013 at
  `:735-800`), and the expert path's K-quants build the DPAS b operand as halves 1024+q instead
  (`code`). So this candidate is one kernel form away from the reference, not a new walk.
- **How arcint's build differs from the reference's mechanism.** Strata decodes the quant *inside* the
  GEMM into int8 DPAS operands with an in-kernel IQ decode and a measured row-count crossover
  (auto, per GPU/driver/format); arcint groups by expert too, but dequantizes K-quants to a f16 DPAS
  operand (integer-built codes, one chain per sub-block) and sends the IQ types through a per-k16 f16
  dequantize. It gates the GEMM by token count (16, or 64 for a cache tensor,
  `code`: `ggml-opencl.cpp:30820-30843`) — the same idea as Strata's crossover, a fixed value. Its
  activation is f16; the reference's int8 path quantizes activations, and arcint's one int8
  activation arm was per 256 values and measured as a quality price on the *dense* path
  (`paper`, `docs/campaigns/llama-engine-kquant-kernels.md`: "moved the dense model's KL by
  0.0026-0.0028 nats and top-1 by 0.9 points").
- **Quality class: rounding-only.** An int8 activation is a quantized activation; the reference's own
  teacher-forced KL against the f16 path was mean 2e-4 nats, p99 6e-3, top-1 99.7 % (`paper`), and
  arcint's int8-dense arm stayed inside the bar. The bar is the answer-level one (mean KL ≤ 0.03 nats,
  argmax within 1 point), not bit equality.
- **Units.** Flash-Next (B60) is the target: its prefill chunk spends ~1.67 s of ~4.0 s kernel time
  in the expert GEMMs at ~5.8 TFLOPS, about a tenth of the card's matrix throughput (`paper`, arcint
  stage 5). The coder (A770) takes the same grouped GEMM with resident experts and would gain from
  int8, but the B60 is where the reference measured (+12.5-21 %) and where the doubling of the
  multiply rate pays.
- **Pricing measurement.** Size the prize before a kernel: Flash-Next prefill is GEMM-bound at
  ~1/10 of peak, so an int8 form's ceiling is the GEMM share of the wall (1.67 of ~5.1 s a chunk,
  `paper`) at 2× the multiply rate → at most ~+20 %. Confirm the share on the current build rather
  than the 2026-10-10 record, with a profiling build and `tools/cl_timeline_steps.py`, one 2,048-token
  prefill graph; the `[MOE_TILES]` padding line bounds how much of that share is already wasted.
  Kernel microbench before any new code: `test-backend-ops perf` MUL_MAT_ID at Flash-Next's shapes
  (512 experts, top-10, 640 x 2560 gate/up, Q2_0 down) with `GGML_OPENCL_KQ_MM_ID2=1` and `=0` on
  the B60 — the two shipped shapes' rates are the f16-DPAS floor the int8 form must beat.
- **Implementation sketch.** A new `mul_mm_kq_f16.cl` sibling (or a variant of
  `kernel_mul_mm_kq2d_q4_K_i8`'s operand build) for the MUL_MAT_ID tile: weight codes as s8, the I-quant
  grids decoded in-kernel to s8 pieces with the block correction folded into a per-super-block f16
  DPAS, activations quantized per 256; dispatch under the existing route cache
  (`ggml-opencl.cpp:30914-30980`), patch 0068+, ~300-500 lines of kernel plus a dispatch and a
  per-format fallback. Red-first: each test below fails on its stated one-line mutation.
  - `test-backend-ops MUL_MAT_ID` (Flash-Next's types, both cards) — mutation: the sub-block's
    `8 D sc - DM m S` constant dropped, the suite must fail.
  - A KL arm (`llama-perplexity -ub 4`, 8 x 512, IQ3_S, the cache on) against the f16 GEMM arm —
    mutation: activation scale read from the wrong sub-block, KL must move past the bar.
  - The served check's `[MOE_TILES] pad` and prefill rate at 1K/4K/20K — mutation: the dispatch
    threshold forced to 16, the rate must regress the arm (this test is the crossover's red).
  - A byte test on the dequantized operand: the s8 codes rebuilt from the file's blocks must equal
    the f16 chain's codes exactly (an integer identity) — mutation: the grid index read unshifted,
    one value differs.

## 3. PCIe link probe + miss balance

- **arcint status: lacks the probe, partly the balance.** There is no link probe in `arcint-src/src`
  or `llama-src/src` (`code`: no hit for `pcie`, `link probe`, a bandwidth probe). The miss split is a
  fixed number: the publish kernel hands the **first** `GGML_OPENCL_CPU_MISS_MAX` (64) missed pairs of
  a layer to the CPU and leaves the rest to the card's direct read (`code`:
  `ggml-opencl.cpp:15744-15790`, `:15915-15920`; the pair selection is index order, not a rate
  comparison). ARCINT.md's "fixed pcie_frac" is not an arcint knob at all — the decode path has two
  ways for a miss, CPU rows or a link read, and which pairs take which is decided by a constant.
- **How arcint's build differs from the reference's mechanism.** Strata probes the link at startup,
  derives every bandwidth constant from it, and per layer splits the distinct misses to minimize the
  longer side of CPU-pool time against GPU-over-PCIe time, with a busy-SM probe retry; arcint's CPU
  side is Strata's own AVX2 rows on the 8 AVX2 cores (`code`, 0063) behind a release/acquire handshake,
  and the link read is the GPU's in-place USM pointer read (`code`: `GGML_OPENCL_BANK_DIRECT`). The
  capacity is there; the allocation policy is not. arcint's measured negative on the *CPU share as a
  whole* is attributed: the handshake is nearly free, ggml-cpu's single-token dots were the cost, and
  porting Strata's rows turned +15.2 % (`paper`, 0056). Nothing tested a per-layer balance, so the
  reference's mechanism is untested here.
- **Quality class: rounding-only.** Moving pairs from the card to the CPU changes the summation for
  those rows (q8 terms, `paper`: +0.0028 nats as served, top-1 +0.05 points). Inside the bar.
- **Units.** Flash-Next only: the CPU miss path exists only where an expert cache does
  (`LLAMA_EXPERT_CACHE_CPU_MISSES=1`, a cache with kq-typed projections, `code`:
  `llama-expert-cache.cpp:519`).
- **Pricing measurement.** The fit's cost side is already measurable without code:
  - CPU rate per expert: one `llama-bench`-shaped measure of the miss rows is not exposed, so use the
    served counters — the cache prints hits, misses and swaps (`llama-expert-cache.cpp:stats()`), and
    the decode time per verify round; run the six answers at `--llama-threads 8` with
    `GGML_OPENCL_CPU_MISS_MAX` at 16 / 32 / 64 / 128 on one binary (the arm 0063 already swept to 128,
    `paper`: 16.68 t/s at 64 and 128, 16.75 with 12 miss threads — on UD-Q3_K_XL). The slope of
    decode against the cap *is* the balance's upper bound.
  - Link rate: the B60's H2D is measured at 13.9-14.3 GB/s from pinned memory (`paper`, DESIGN §2). A
    probe is only worth building if a window shows the link rate differing from that on the day (the
    reference's reason: a misread link cost −37 % on pp4096, `paper`, In bf0d46db).
  - Bar for a balance: the same decode on the six answers with the cap fixed at 64 (today) against a
    per-layer balance at the same total CPU share; ≥+10 % decode, KL and answers inside the bar.
- **Implementation sketch.** `llama-expert-cache.cpp` gains a per-layer split: a least-squares fit of
  CPU-µs-per-expert and link-µs-per-expert from the counters it already keeps, a per-layer
  `n_cpu` computed at adapt time instead of `n_cpu_max`, and the publish kernel takes it as an
  argument (`ggml-opencl.cpp:15915`, kernel `kernel_miss_publish`'s `n_cpu_max`). A startup probe is
  a separate, small addition in the OpenCL backend (one 48 MiB USM→card copy, timed, before the
  first graph). Patch 0068+, ~120-200 lines host-side, no kernel change beyond an argument.
  Red-first:
  - A unit test on the split arithmetic (the fit's constants from two measured points) — mutation:
    the divisor's sign, the test must fail.
  - The served check with the CPU misses off and on at the same cap — mutation: the split forced to
    all-CPU, decode must regress the arm (the discriminator stage 8 used).
  - A KL arm against the fixed split — mutation: the balance's pairs taken in reverse order, KL must
    move past the bar only if the rows are wrong (a bitwise tripwire, not a veto).

- **Priced (`measured-here`, 2026-10-31, the B60, Flash-Next IQ3_S with the unit's flags, the six answers, two rounds,
  the second reversed; the capital right in all 12 legs):** decode against `GGML_OPENCL_CPU_MISS_MAX` - 16: 31.57 /
  32.15 t/s, 32: 34.48 / 34.77, 64: 34.40 / 34.58, 96: 34.57 / 34.67, 128: 34.57 / 34.62, 256: 30.05 (a one-off; its
  reversed twin 34.55) / 34.55. Flat from 32 to 256; only 16 costs (~-7 %). The cap slope is this item's upper bound
  (the pricing above), so a per-layer balance has no prize on this workload: **not built**. What this does not test:
  depth (the reference's +9.9 % / +25.6 % are at 8K / 32K on a slower-CPU shape, `paper`) - a decode-at-depth arm
  (fn_sweep with --prompt-tokens 8192 / 32768 then the answers) is the remaining check before the item is closed.
- **At depth (`measured-here`, 2026-10-31, the same card and build, fn_sweep --prompt-tokens N --prompt-max 300, the
  long prompt's own answer, two rounds reversed):** 8K - cap 16: 23.0 / 23.9 t/s, 64: 27.1 / 27.5, 256: 25.7 / 26.1;
  32K - 16: 18.2 / 15.4, 64: 21.7 / 14.2, 256: 21.3 / 14.8 (the two rounds differ by ~30 %, more than any cap effect;
  the answers stopped at 35-72 tokens, so each figure is a short sample). No cap above 64 is faster at either depth:
  the balance's prize is not visible here either. **Closed: not built** (`measured-here` on the cap, the reference's
  per-layer rate split itself untested; reopen only with a long-answer depth method that resolves a few per cent).

## 5. QSA select (wide block scores, register/one-read top-k)

- **arcint status: partly, and on the wrong unit in `candidates.md`.** The graph is there:
  `build_qsa_sel` pools the indexer keys of blocks of `compress_ratio` cells, scores them (one f32
  product over all heads, a ReLU in place, the heads summed as slices, a scale), and selects the top
  blocks with `ggml_top_k` (`code`: `llama-src/src/models/qwen4exp.cpp:739-844`, the select at
  `:830-838`). The top-k lands in the OpenCL backend's radix select (`code`:
  `ggml-opencl.cpp:34542-34580`, `kernels/argsort.cl:98-145`): one 256-lane work-group per row, four
  histogram passes plus one write pass — five reads of the score row per token. The OpenCL backend
  has no `LIGHTNING_INDEXER`, so `cparams.fused_lid` is false and the plain-ops path runs (`code`,
  patch 0052's resolve; the product is a `mul_mat` f32 of pooled [idx_dim, n_pool] against q
  [idx_dim, n_heads x n_tokens]).
  - What lacks against the reference: the block-score *GEMM form* with a register top-k, and the
    one-read sample/cooperative-scan select; the five-pass radix is the shape the references
    replaced. The biggest known cost is not the select algorithm but the shared f32 accumulate
    buffer `[n_pool, n_heads, n_tokens]` and its ReLU copy — arcint named it
    (`code`, the TODO at `qwen4exp.cpp:851`) and 0052's ReLU-in-place freed ~1 GiB.
  - **Unit correction:** QSA is Flash-Next's 12 sparse-attention layers. The dense 27B agent
    (`qwen35`) has no indexer at all (`code`: `qwen35.cpp` carries no indexer tensor or op), so
    "agent at 245k" in `candidates.md` #5 is wrong; the depth regime is Flash-Next at 131,072.
- **How arcint's build differs from the reference's mechanism.** Strata scores the block matrix with
  a tiled GEMM and takes a wide/register top-k at 4 lanes a query head; arcint scores with one f32
  `mul_mat` (no XMX, head size 32/64 does not fit the DPAS forms, `code`: `qwen4exp.cpp:432`'s
  comment) and selects with a 5-pass radix, one work-group per token. Both run per layer over the live
  block count (arcint's `n_pool` is the memory module's, not `--max-context`, `code`). The reference's
  measured negatives (the "wide top-k past 135K cells" wins) have no arcint counterpart.
- **Quality class: exact** for the selection: the same scores, the same k, ties are the only risk
  (the reference kept ids bitwise, `paper`), and arcint's select already produces ggml-cpu's unordered
  indices (`code`: the kernel's comment).
- **Units.** Flash-Next (B60) at 131K, prefill at depth and decode at depth.
- **Pricing measurement.** Existing knobs cannot split the select from the layer, so use the tools:
  a `GGML_OPENCL_PROFILING` build, a 20k and a 100k+ prefill of Flash-Next, `tools/cl_timeline_steps.py`
  per-kernel device time (`code`, the tool) to read `indexer_score`, the head-sum adds, `top_k` and
  the mask ops at each depth; and, for decode, `LLAMA_DECODE_TIMING=1` (`code`, stage 3c's per-step
  host view). Compare the select's device time at 20k and at 131k against the record's "select 18 % of
  a 128K prompt" (`paper`). Bar for a build: the select + score share ≥10 % of the prompt's device
  time at 100k+, otherwise not worth a kernel.
- **Implementation sketch.** Two kernels in `kernels/`: a block-score form that writes `[n_pool,
  n_tokens]` straight from the pooled keys and q (the reference's GEMM form; fp32 tiled, since arcint
  has no TF32), and a one-read select (a sample on a strided stride, then a local refine, or a
  cooperative scan) whose output is the same unordered ids. Dispatch by live block count in
  `qwen4exp.cpp:830`. Patch 0068+, ~200-350 lines. Red-first:
  - `test-backend-ops TOP_K` with the new kernel — mutation: the histogram shift set to 0, the suite
    must fail on 525/525-style counts (`paper`: TOP_K 525/525 at the pin).
  - A served id-equality tripwire: the same prompt through old and new select must give the same
    `sel_idx` rows on 16 prompts — mutation: the tail blocks not appended, ids must differ.
  - A KL arm at 100k+ against the old select — mutation: the ReLU moved after the head sum, KL must
    move past the bar.

- **Priced (`measured-here`, 2026-10-31, the B60, Flash-Next IQ3_S with the unit's flags and the slot loan, the OpenCL
  intercept layer's chrome trace, device time of the prefill segment against its span):** at 19,196 tokens
  `kernel_top_k_f32` 0.45 %, `kernel_relu` 0.17 %, `flash_attn_mask_cls` 0.20 %; at 103,638 tokens (346.4 t/s, the
  loan arm's untraced rate) 0.98 % / 0.49 % / 0.59 %, plus `kernel_mul_mm_f32_f32_l4_lm` 1.6 % if all of it is the
  indexer's f32 score product (`inferred`: the score is the only f32 x f32 mul_mat named in this section) - at most
  ~3.7 % at 100K against the bar of 10 %: **not built**. The record's "select 18 % of a 128K prompt" (`paper`) does not
  hold on this build. The same 100K trace: the bank stream's `clEnqueueWriteBuffer` 38.2 %, `kernel_moe_gather_experts`
  25.7 %, `flash_attn_f32_f16_dpas2` 21.4 % - the prefill's levers at depth are the expert weights' movement and the
  attention, not the select.

## 6. CPU miss compute on AVX2, the remaining pieces

- **arcint status: has most of it.** 0063's rows are Strata's: multi-token (`NT` up to 8) decode-once
  chunks with **named register accumulators** (`code`: `llama-src/ggml/src/ggml-opencl/ggml-opencl-miss-cpu.cpp:200-228`,
  two accumulators for IQ2_XS at `:270-297`, IQ3_S's single gather at `:181-196`), the bank's planar
  fields instead of GGUF blocks, the per-format fast path and the 64-pair cap
  (`code`: `ggml-opencl.cpp:15816-15823`, `:15915`). Measured on the served service: IQ3_XXS
  33.85/33.87 → 36.13/36.08 t/s (+6.6 %), cap 64 37.93 (+12.0 %), KL 0.2191→0.2178 (`paper`).
  - What lacks: any prefetch hint (`code`: no `_mm_prefetch` in the file), the cold-page/first-touch
    prefetch (Strata: cold reads were 78 % of the CPU lane, `paper`, St 9e3fe046), and the
    named-accumulator idiom for a router dot — not applicable here because the router runs on the
    card (0064, `code`), so that half of the candidate has no arcint target.
  - The remaining prize is small: the rows' decode-once shape is already taken; what a prefetch can
    buy is the bank's page-cache misses on first use, which the campaign's own note already
    measured as a host-memory fact rather than a kernel one (`paper`, stage 8's n-gram rows).
- **How arcint's build differs from the reference's mechanism.** Strata's rows read a pinned arena
  with its own planar layout; arcint reads the bank's planes (`ggml_cl_lb_fields`, 0022) and the
  IQ4_NL split form, with ggml-cpu's quantizer rules, and falls back to ggml-cpu's single-token dot
  per type it does not cover (`code`: `ggml_cl_miss_cpu_type`). Both are AVX2 on the same Zen 3
  class; the reference figure was read on a Zen 4 with AVX-512 and DDR5 (`paper`), so its ratios are
  optimistic here.
- **Quality class: exact.** The rows match ggml-cpu's `vec_dot` exactly on quantized random weights
  (nt 1-8, both layouts, an odd-block slice, `paper`), and the integer sums are exact; the float
  steps are the same count and order as ggml's (`code`).
- **Units.** Flash-Next (B60) only (the cache owns the CPU path).
- **Pricing measurement.** A prefetch hint is priced by the arm that isolates page faults:
  the six answers with the bank's pages cold (drop caches, or a fresh model load and a first request)
  against warm, at `GGML_OPENCL_CPU_MISS_MAX=64`; the gap is the ceiling. Then a probe outside the
  engine: `perf stat -e cache-misses,minor-faults,major-faults` on one served decode, the campaign's
  `perf` method (`paper`). Bar: a prefetch only if major faults > ~5 % of the CPU lane's time.
- **Implementation sketch.** One `_mm_prefetch` ahead of the next expert's first field in each row
  function, and a page-touch pass over the layer's candidate slices at publish time — patch 0068+,
  ~30-60 lines in `ggml-opencl-miss-cpu.cpp` and the pool's job loop. Red-first:
  - The row parity test (nt 1..8, both layouts) — mutation: the prefetch address advanced by one
    field, the rows must stay identical (a prefetch cannot change bits; this test proves it).
  - The served check with and without the hint at the same cap — mutation: the hint moved to the
    wrong expert, decode must regress the arm.

## 7. xe doorbell (uncached hint, stepped window, payload checksum)

- **arcint status: has the hint and the fence; no checksum; the stepped window is not a concept here.**
  Patch 0056's handshake is the reference's form: the card publishes a sequence number with a
  system-scope release, the CPU spins on it, the card's wait kernel polls the host's done word with
  `__builtin_IB_lsc_load_global_uint(..., MISS_L1UC_L3UC)` (`code`:
  `llama-src/ggml/src/ggml-opencl/ggml-opencl.cpp:15744-15790`), with
  `atomic_work_item_fence(…, memory_scope_all_svm_devices)` on both sides and a bounded poll that
  writes NaN rather than a wrong token on a lost call. That is exactly Strata's `557c2b67` finding
  (a device spin is cache-served on a B60: 263 ms→2.4 µs, 8.7-10.5→19.3 t/s, `paper`) applied to the
  same need. There is a second word (`back + 17`) the wait compares against the sequence — a
  lost-publish detector, not a payload checksum.
  - Lacks by choice: a payload checksum. Strata measured it at 0.5 % stale payloads on an A770 but
    17 % of decode on an A750, opt-in (`paper`, St 18ee03e5, 3157c789). arcint's NaN-on-lost-call is
    cheaper and the red case is the same class.
  - The stepped window: arcint's verify is one graph; the publish is per layer inside it, so the host
    already serves between publishes (layer granularity). The B60's handshake was measured at the
    ceiling: 49.2/50.0 t/s with the CPU computing nothing against 49.8/50.6 with the ceiling switch
    (`paper`, arcint stage 3d) — the wait costs nothing.
- **How arcint's build differs.** Strata's doorbell is SYCL/CUDA in its native engine with its own
  ring form; arcint's is OpenCL USM with the same cache hints, and it is the CPU-miss path only
  (decode, not prefill, `code`: `:30864-30867`'s `!gemm` condition).
- **Quality class: exact.** Handshake only; the rows' arithmetic is item 3/6's.
- **Units.** Flash-Next (only where the CPU misses are on).
- **Pricing measurement: none needed.** The mechanism exists and was measured at the ceiling. If a
  future arm changes the handshake (`LLAMA_EXPERT_CACHE_CPU_MISSES` off/on), the discriminator is the
  ceiling switch the campaign used (`paper`).
- **Implementation sketch: none.** Do not build.

## 9. adapt-tier overlap (async swaps / copy overlap / plan placement)

- **arcint status: has it — `candidates.md` and ARCINT.md are stale here.** Patch 0042 is Strata's
  adapt stream: each layer's bank has spare entries, a swap runs in two steps on a **second in-order
  queue**, the victim's bytes are copied out, then the newcomer in, the tables change only between
  graphs behind a fence, and the copies are enqueued **right after** the graph so they run while it
  computes (`code`: `llama-expert-cache.cpp:711-760`, `ggml-opencl.cpp:1144-1160`'s
  `adapt_stream_state`, the fence at `:16590`). Measured: the verify's input wait 2.6-6.0 → 0.94-0.95
  ms, decode +2.4 % on the mean, the same texts (`paper`, README 0042 at
  `arcint-src/contrib/llama.cpp/README.md:2601-2671`). The reference figure for the same mechanism is
  +10 % (`paper`, St b437b8ea); arcint's smaller gain is its slot shape (the newcomer lands two
  verifies later, ~1 point of hits lost, `paper`).
  - One sub-piece lacks: In f1864300 put the *plan* (host route/promotion work) inside the commit,
    +2.4-5.1 % (`paper`). arcint plans on the host between ubatches — 0.72 ms counted per step
    (`paper`, stage 3e) — and `LLAMA_EXPERT_CACHE_ASYNC_IDS` already reads the previous step's ids
    behind the graph (`code`: `llama-expert-cache.cpp:315`), which is half of that move.
- **How arcint's build differs.** Strata's copies are non-blocking stream ops with an event polled
  each round; arcint's are non-blocking `clEnqueueWriteBuffer`s on a second queue with a ticket and
  `adapt_done` polling between ubatches (`code`). Blocking is already gone; the reference's
  "no plan while a swap is in flight" is in arcint (`code`: the `pend`/`pend_phase` state).
- **Quality class: exact.** A hit and a bank read run the same kernel on the same bytes, and the
  copies are byte-checked against the file (`code`: `llama-expert-cache.cpp:570-589`); measured draft
  counts identical, same text (`paper`).
- **Units.** Flash-Next (B60).
- **Pricing measurement.** The leftover (plan placement) is priced by `LLAMA_DECODE_TIMING=1` on the
  served service: the cache's step time per verify (`paper`: 0.72 ms) is its whole ceiling; if <1 %
  of a 70 ms round, drop it.
- **Implementation sketch: none** for the mechanism; the plan-in-commit move is ~30 lines in
  `llama-expert-cache.cpp::step` and only if the timing shows it pays.

## 10. PCIe copy discipline

- **arcint status: has most, one leftover that matters at depth.** Built: 0047 stages small graph
  inputs into a 32 MiB USM ring and enqueues without a wait, the scheduler waits once a split
  (`code`: `ggml-opencl.cpp:12168-12200`, the cap `size > (4u << 20)`); 0042 keeps the swaps off the
  compute queue; 0046 uploads every layer's tables in one write (`code`:
  `llama-expert-cache.cpp:167-176`); 0032 copies bank tensors on a copy-only queue (`code`:
  `ggml-opencl.cpp:27700-27706`) — a copy-only queue doing only copies, which is the reference's
  lesson. Measured: a verify's inputs 926-949 → 582-593 µs, a draft's 395-404 → 134-141, decode +1.1
  to +1.4 %; the coder (A770 behind the chipset) 1.93-2.02 ms → 133-255 µs an input, decode +14-15 %
  (`paper`, README 0047).
  - **The leftover:** the staging cap is 4 MiB and the ring is 32 MiB, so any graph input above 4 MiB
    is still a blocking write, and a ring wrap forces a `clFinish` (`code`: `:12166-12199`). At
    Flash-Next's depth the QSA inputs are large: the indexer's `pool_mask` is f32 `[n_pool, n_tokens]`
    (`code`: `qwen4exp.cpp:707`) and the `mask_all` path was sized at ~1 GiB for the scores at 131K
    (`paper`, 0052's finding) — the masks and rotations are the ones the artifact's own note says
    "take longer than a 1 KiB input" (`paper`). So per prompt chunk at depth there are still a few
    big blocking writes. That is the audit's remaining piece.
  - The counters: the card's copy engine was 81 % busy during decode and 2 % during prefill
    (`paper`, stage 5's fdinfo read), so the copy engine is not the prefill bound and the leftover's
    ceiling is decode-side at depth.
- **How arcint's build differs.** Strata runs the prompt's expert copies on a copy-only queue after a
  Level Zero adapter put them on the compute engine; arcint's bank stream does the same on its own
  queue, and its graph inputs go on the compute queue as non-blocking writes from a USM ring. Both
  OpenCL/SYCL; arcint has no async-copy/event primitive in the backend's device objects (`paper`:
  the backend has no events, stage 8), so a big write cannot be split and awaited at the split's
  first kernel — that is what the leftover needs.
- **Quality class: exact.** Copy discipline only; the staged bytes are the same bytes (the red case's
  stale ring answers garbage, which is the proof the path is exercised, `paper`).
- **Units.** coder (A770, its link is slow: the biggest measured gain), agent, Flash-Next.
- **Pricing measurement.** Count what still blocks: `GGML_SCHED_TIMING=1` plus the backend's own
  input-copy time on a 100k+ agent prefill and Flash-Next's 20k needle; the diff between "inputs a
  graph" with `GGML_OPENCL_STAGE_INPUTS=1` and `=0` per graph is the ceiling (the campaign's method,
  `paper`). Then the default 4 MiB arm against `GGML_OPENCL_STAGE_INPUTS=1` with a bigger ring only
  if the count is nonzero. Bar: ≥+0.5 % decode or ≥+1 % prefill at depth.
- **Implementation sketch.** Raise the ring to what a split's inputs need (a flag, 32 → 64/128 MiB)
  and let a large write go non-blocking in chunks with one wait at the split's first kernel; the
  write is already in `ggml_cl_stage_write` (`ggml-opencl.cpp:12166`), patch 0068+, ~60-100 lines.
  Red-first: the served staging red (`GGML_OPENCL_STAGE_RED=1`) — mutation: the wait moved after the
  first kernel, garbage answers (the existing red case must fire).

## 11. prefill chunk / residency dial + ring byte budget

- **arcint status: partly.** The chunk is `--llama-prefill-chunk` (= the ubatch with a cache,
  `code`: `arcint-src/src/exec/backend_llama.cpp:577-588`), the slot budget is
  `--llama-expert-cache MIB`, the ring is a fixed 4 (`GGML_OPENCL_BANK_STREAM_RING`, ≥2, `code`:
  `ggml-opencl.cpp:27699-27704`), and the stream threshold is 256 tokens (`code`: `:27691`). The
  VRAM-margin dial exists as 0052's `--llama-reserve-cells` (`code`, patch 0052). Nobody bid against
  the byte budget: the sweeps are by hand (2048/4096/8192 at a fixed slot count, `paper`, stage 5;
  1024/12500 vs 2048/11300 at 131K, `paper`, stage 3c: prefill 657 vs 538, decode 34.8 vs 36.8).
  - Lacks: the byte-budget bisection and the borrowing of prompt buffers from slots (Strata's
    `--prefill auto`, `paper`). arcint reserves the chunk's compute buffer for the session
    (`paper`: 3,048 MiB at 131K with chunk 2,048, 1,543 at 1,024) and decode needs a fraction
    (`paper`: ~93-770 MiB at chunk 4-16).
- **How arcint's build differs.** Strata borrows slots for the prompt's buffers and picks the
  largest chunk that fits the lent bytes; arcint reserves the buffers, so a larger chunk costs slots
  and a smaller chunk streams the bank more often. Same mechanism as the reference's two-sided
  trade, without the lending.
- **Quality class: rounding-only.** Chunk size changes the batch shape and the draft numbers
  (`paper`: KL 1.1e-2 nats for Strata's 16K chunk against 4,096; arcint's chunk 8,192 texts differ
  only in the decode arm). Answer-level bar.
- **Units.** Flash-Next (B60), coder (A770: 1,024 is its own measured best, +15 %, and 2,048 spills
  4.1-4.3 GiB to GTT, `paper`).
- **Pricing measurement.** All knobs exist; one fresh process an arm:
  `--prefill-chunk {1024, 2048, 4096}` × `--llama-expert-cache {slots}` × `--llama-reserve-cells
  {9216, 20480}` at the 20k needle, decode after it, peak VRAM from fdinfo (`code`, served_check's
  `resident()`). The extension to Infernix's 16,384-chunk regime is only reachable after the
  loan; priced first by the compute-buffer reservation over the chunk sweep (which is a straight
  line of ~10.7 bytes × cells × chunk, `paper`).
- **Implementation sketch (the loan).** §3.3's lines reserve the compute buffer; a loan is a
  contiguous whole-layer range of slots (`paper`: the slots are three tensors a layer in one
  buffer), returned after the prompt with a refill (~0.25 s) and a per-graph bound check that a
  graph never writes a lent range while it holds experts (`paper`: the hazard). Files:
  `llama-expert-cache.cpp` (range, refill), `ggml-backend-sched`'s reservation (arcint's
  `src/exec/backend_llama.cpp`), patch 0068+, ~200-300 lines. Red-first:
  - The served loan red: a graph that writes into a lent range must be refused — mutation: the
    bound check removed, the red case must answer garbage or the check must trip.
  - The refill byte test: every lent slot's bytes equal the file's after the prompt (the cache's
    `verify` at `llama-expert-cache.cpp:132-140`) — mutation: one slot not refilled, the test fails.
  - The A/B: the loan arm's decode at 131K ≥ the 1,024-chunk arm's and its prefill ≥ the 2,048
    arm's — mutation: the loan size halved, prefill must regress.

## 13. MTP width / acceptance policy

- **arcint status: lacks.** `n_draft_` is a fixed flag (`--llama-mtp N`, 3/5/4 per unit) and the only
  policy is Strata's min-p stop on the last draft's probability (`code`:
  `arcint-src/src/exec/llama_spec.cpp:562-580`). No cumulative-confidence cut, no cost-aware width,
  no per-request width. The draft vocabulary subset also exists (`--llama-mtp-vocab`, `code`: the
  draft head's row subset), and its measured effect is large on the coder (+10.1-12.6 %, `paper`,
  fusion-coder-night).
- **How arcint's build differs.** Strata's chain is the same stop rule; Infernix's auto width picks
  the width from a cost model and a gate, and Strata's cumulative cut stops the chain on the product
  of confidences. arcint drafts a fixed chain with the single stop, and its verify window cost has a
  known cliff between 4 and 5 rows (`paper`: ~100 ms against ~73, stage 3c) that a width policy
  should respect.
- **Quality class: rounding-only.** The target's own picks change with the number of rows a verify
  carries — arcint measured that the text differs while the answers stay right (`paper`, DESIGN
  3.5); Strata's cumulative cut was bit-identical on its arm (`paper`). Judged at the answer bar and
  the task battery.
- **Units.** All three: Flash-Next 3 with the 0.5 stop (matching Strata's 37.8 on its own request,
  `paper`), agent 5, coder 4 (5 lost 3.6-9.2 % and dropped acceptance 37.9→33.6 %, `paper`).
- **Pricing measurement.** The width's payoff is per-request and the round cost is measurable
  without code: `LLAMA_DECODE_TIMING=1` gives the propose/verify split per round, and the served
  log prints rounds, tokens a round and accepted drafts (`paper`). Run the six answers at `--llama-mtp
  2/3/4/5` on Flash-Next and 4/5/6 on the agent, one fresh process an arm, and read tokens/round,
  ms/round and the accepted share — the adaptive width's ceiling is the best fixed arm's mean plus
  the spread. Bar: the policy's geomean over the six ≥ the best fixed width's geomean by ≥1.5 %.
- **Implementation sketch.** A width policy in `llama_spec.cpp`'s `draft()`: a per-request budget
  from a small cost model (per-row verify cost from the last rounds' `LLAMA_DECODE_TIMING`, the
  acceptance from the last window), a cumulative-confidence product replacing the last-token min-p,
  and the flag as an upper bound. Patch: none in llama.cpp; arcint-only `src/exec/llama_spec.cpp`.
  ~150-250 lines. Red-first:
  - The verify walk's unit test (`tests/test_verify_walk.cpp`) with a width change — mutation: the
    walk takes a draft past the target's disagreement, the test must fail.
  - The equivalence suite's MTP on/off arm — mutation: the policy drafts with the target's logits,
    answers must change and the arm must fail the answer gate.
  - The served A/B against the fixed width — mutation: the policy forced to width 1, decode must
    regress.

## 14. per-card expert profile from own routing traces

- **arcint status: partly.** The cache fills from Strata's STRP decode profile, read-only, one file
  for every layer, per-device caches (0024) but no per-device profile (`code`:
  `llama-src/src/llama-expert-cache.cpp:45-61`, `:345-372`). No save, warm start, or trace in the
  series; the replay harness `tools/expert_cache_replay.py` exists but its trace switch
  (`LLAMA_EXPERT_CACHE_TRACE`) is on a dev branch, not in `contrib/llama.cpp/patches` (`code`: the
  tool's docstring). Infernix's warm start was tried and lost to the calibration profile
  (38.4 against 38.9 t/s, `paper`, arcint 2026-10-15).
- **How arcint's build differs.** Strata's profile is a routing-rank file from its own traces;
  Infernix saves the live policy's placement at exit, seeded from the same workload. arcint reads
  one static ranking and adapts from decode usage (`code`). The B70's per-card profile (+5..64 %,
  mean +21 %, 12 unseen prompts, same tokens, `paper`) is per *silicon*, which arcint does not have:
  the same file is used on the B60 and the A770.
- **Quality class: exact.** A hit and a miss run the same kernel on the same bytes (`code`: the
  decode branch's `tab_comb`, `llama-graph.cpp:2513`), and measured A/Bs kept the text and the draft
  counts identical (`paper`, 0042/0055 arms). Placement is free in quality.
- **Units.** Flash-Next (B60) only; the coder's experts are resident, the agent is dense.
- **Pricing measurement.** No traces needed for the prior: three profiles' *placement* can be
  compared by their hit rate and decode on the six answers, all knobs existing (`--llama-expert-
  profile` at a fixed slot budget, `LLAMA_EXPERT_CACHE_ADAPT_EVERY=0` to hold the profile). The
  save path is priced by replaying a captured decode routing with `tools/expert_cache_replay.py` if a
  trace is captured on a branch — otherwise the first measurement is "the calibration profile's hit
  rate against a per-workload ranking built from the served log's hits" (the log carries per-request
  hit shares, `paper`).
- **Implementation sketch.** A `--llama-expert-cache-save FILE` on `llama_expert_cache::init`'s
  teardown that writes STRP in the load order (the slots first, then by use), and a per-card profile
  argument so both cards' caches take their own file; the replay tool already parses the format.
  Patch 0068+, ~80-120 lines in `llama-expert-cache.cpp`. Red-first:
  - The STRP round-trip test: a saved file re-read as `--llama-expert-profile` gives the same
    placement and the same hit count — mutation: the expert/layer order swapped, the test fails.
  - The served warm-start arm: a second process on the saved profile ≥ the calibration profile's
    decode within the spread — mutation: the file truncated by one record, the read must refuse
    (`code`: `read_profile`'s check) and the fill fall back to round robin.

## 15. agent prefix reuse (tokenise-from-prefix, branch tap, tool-call hold, min-tokens)

- **arcint status: partly.** Checkpoint restore and the longest-prefix reuse are built:
  `arcint-src/src/exec/backend_llama.cpp:988-1010` compares the lane's held token array against the
  new prompt and either trims the KV or restores the newest checkpoint inside the prefix;
  checkpoints are taken at llama.cpp server's points (`:1684-1808`, up to `--llama-checkpoints`,
  8,192 apart), and measured 38.75 → 0.61 s on a 24.3k conversation (`paper`, DESIGN 3.4).
  - Lacks: **tokenisation from the prefix** — the request's whole prompt is tokenised before the
    comparison (`code`: `prompt` is a token vector by then), so a 40k transcript still pays the
    tokenizer pass the reference skips; per-message branch taps (a checkpoint at each user message
    start beyond the step rule, which arcint does have via the step/user rule, `code`: `:1687`), the
    tool-call hold (a checkpoint kept across a held tool turn), and the min-tokens admission that
    keeps a side request from evicting the long lane's pool.
  - The measured pool: In e8d09e27 reused 40,128 of 40,128 against 32,768 with the old cut, TTFT
    0.6-1.1 s versus 2.1-2.2 s (`paper`); St a7d4b754 spread checkpoints resumed a 210,528-token
    prompt at 9,525 instead of re-reading 201,003 (`paper`).
- **How arcint's build differs.** Strata's conversation cache keeps whole turn state and matches on
  token equality before tokenising the new suffix; llama.cpp server's mechanism (arcint's) keeps
  partial recurrent-state checkpoints and compares full token arrays. Same idea, different cut: the
  tokenizer is the part still paid per request.
- **Quality class: rounding-only.** A resumed turn against a re-prefill with the same batch breaks is
  byte-equal (193/193, 209/209 characters); other breaks change the text and the answers stay right
  (`paper`, DESIGN 3.4) — judged at the answer bar.
- **Units.** agent (B60, 245k transcripts), coder (A770), Flash-Next (its indexed memory refuses
  windows, but the checkpoint restore is per lane and applies: `code`: checkpoints only where
  `llama_model_is_hybrid`/recurrent, which qwen4exp is).
- **Pricing measurement.** Existing: the served log's `stats.cache_hit_tokens` and the prefill time
  per request (`code`: `backend_llama.cpp:1006`), plus a two-request transcript pair; the tokenizer's
  share is `perf -e` on the served process or the wall difference between a prefix-hit request and a
  same-length fresh one at a fixed decode. Bar: a tokenise-from-prefix build only if the tokenizer is
  ≥20 % of a prefix-hit request's TTFT (a 40k transcript's tokens at ~1 MB/s order).
- **Implementation sketch.** Tokenise incrementally against the lane's held ids: a `tokenize_prefix`
  in `backend_llama.cpp`'s request path that feeds the tokenizer the longest shared text boundary
  first and only the differing suffix, with the same final id-equality check against the full
  tokenisation (the server's `n_past` logic already has the comparison). Patch: arcint-only,
  ~120-200 lines. Red-first:
  - A unit test: a prompt that shares a 90 % prefix gives the same ids as a full tokenisation on 500
    transcript-shaped strings — mutation: the tokenizer restarted mid-UTF-8, ids must differ.
  - The equivalence suite's prefix-reuse arm (answers right, byte tripwire reported) — mutation: the
    suffix tokenised with a different BOS, answers must change.

## 16. decode launch fusions (the remaining ones)

- **arcint status: partly.** Built and measured: 0004's quantize sub-group form and split GLU/mul
  rows, 0027, 0049 (rollback snapshots without a cont, llama.cpp CUDA's `rms_norm+scale`,
  `ssm_conv+silu`, `sigmoid+mul`, bit-identical, scheduler time 12.35-12.41 → 11.67-11.68 ms,
  decode +0.5 to +1.1 %), 0050/0051 (GDN snapshots in the cache, gating in the kernel), 0064 (top-k
  router fused), 0065 (one launch for a run of narrow strided copies: snapshots 150 launches/0.90 ms
  → 30/0.175 ms, coder +1.8-1.9 %), 0066 (GDN generation kernel for verify batches: 1.044 → 0.674 ms,
  coder +1.1-1.2 %) (`code`: the patch series; `paper`: their README sections and
  `fusion-coder-night.md`).
  - Lacks: the qk norm+rope fusion (Infernix: +22-52 % through 256 tokens and wins at every width to
    8,192, `paper`), the one-token self-commit / commit kernel reductions (St 352cad8f, 6f3e3810
  commit 2.35 → 0.65 ms, `paper`), and the HC collapse (already tried on arcint and no gain, `paper`,
  `GGML_OPENCL_FUSE_HC_MIX`).
  - The discipline: judge at the wall, not the kernel (In 0d76fe11: +96 launches/step made decode
    4.86 % slower while the kernel sum fell 6.58 %, `paper`), and arcint's own ceiling is ~4.1 % of
    Flash-Next's round outside kernels (`paper`, stage 8: at least 4.1 ms of ~69.6), of which ~6.5
    ms inside the verify is host time.
- **How arcint's build differs.** Infernix fuses with CUDA templates and T ≤ 16; arcint fuses with
  OpenCL sub-group kernels and its opaque-zero trick to keep two roundings (`code`: 0049's README).
  The shapes' costs differ: arcint's F16 hyper-connection products on the B60 were 2.9 ms a token
  (`paper`), its norms 0.87 ms on the coder (`paper`).
- **Quality class: exact** where the fusion keeps the same f32 roundings (0049/0065/0066's same
  answers); the qk-norm+rope fusion would be rounding-only (a norm's division folded into the rope
  scale) — measured at the answer bar.
- **Units.** coder (A770: its decode is host-bound, 5.9 µs a kernel enqueue; 1,810 launches a token
  after 0034 and 4,150 in a verify graph at 0.7.5, `paper`), Flash-Next.
- **Pricing measurement.** `tools/cl_timeline_steps.py` on a profiling build gives the per-kernel
  count and time; the fusion's ceiling is the sum of the kernels it removes plus their launch time
  at the wall (`paper`: ~0.4 % of decode a hundred launches at 0049's rate). The qk norm+rope
  fusion's ceiling is the 206 rms-norm/mul launches a Flash-Next verify (`paper`, 0049's count).
  Bar: ≥+0.5 % decode for a bundle of 2-3 fusions, else no build.
- **Implementation sketch.** A qk norm+rope OpenCL kernel next to `rms_norm.cl`/`rope.cl`, dispatched
  in the attention builders (`qwen4exp.cpp:949-1035`, `qwen35.cpp`'s attention); the commit
  reductions next to `cpy.cl` (0065's form). Patch 0068+, ~150-250 lines. Red-first:
  - `test-backend-ops RMS_NORM` and the RoPE cases against the fused form — mutation: the norm's
    epsilon added after the rope scale, cases must fail.
  - The served A/B on the coder's three answers — mutation: the fusion applied at T > 16 where it
    was not measured, decode must regress or the answers must move.

## 18. decode-once expert matvecs in the verify window — **out**

- **arcint status: the reference's mechanism was tested and lost.** The campaign built Strata's
  grouped-by-expert verify (a per-layer kernel groups the window's pairs by expert, the bank's
  experts first, the weight decoded once for the group; `GGML_OPENCL_KQ_GROUPED=1`) and measured it
  correct and slower: 30.5 → 29.5 t/s, verify 15.01 → 15.56 s; in isolation IQ2_S 120.6 → 146.9 µs
  (`paper`, arcint stage 3c), and patch 0031 (`idg` args) was reviewed and dropped in 0.8.1 —
  the switch is not in the served series (`code`: no `KQ_GROUPED` in `llama-src/ggml/src/ggml-opencl/`).
  That is the reference's mechanism (weights read once per expert for the window, bitwise in the
  reference, `paper`, St e65858f8), so the negative counts.
- **What arcint has instead:** the per-pair matvec with the bank-first permutation (decode's single
  branch, `code`: `llama-graph.cpp:2506-2520`, `ggml-opencl.cpp:30890-30910`), and the *dense*
  few-column kernels — 0009 (fp16 DPAS, 2-16 columns, one weight read for all of them; an 8-row
  dense step 1.21× a 1-row step on the B60, `paper`), 0001's integer-dot matvec to 4 columns, 0022's
  F16 2-8 columns, 0013's int8 GEMM. The dense half of this candidate is built; the expert half is
  the measured negative.
- **Why it leaves the scope:** its only remaining lever (a grouped expert matvec) was priced on the
  exact engine, model and card and lost; re-opening it would need a different kernel shape, which is
  a new candidate, not this one (the reference's own note says the grouped form pays only where an
  expert has 4+ entries and the per-entry cost is low, `paper`, St 546d7675).

## 19. xe device hygiene

- **arcint status: partly.** The audits:
  - **JIT >4 GiB pointers:** no `-ze-opt-greater-than-4GB-buffer-required` in the compile options
    (`code`: no hit in `ggml-opencl.cpp`'s opts, only `-cl-intel-256-GRF-per-thread` and defines).
    The slots are a >4 GiB buffer at Flash-Next's budgets (11,600 MiB) and 0035's codebook tables,
    0013's inline vISA and 0022's grids are all in that program — the Infernix/Strata finding (experts
    in the last 0.46 GiB of a 4.46 GiB cache dequantized to NaN, `paper`, St defadd6e) is the class
    that would hit here.
  - **Aliased pages:** no allocation check or spacer (`code`: no allocator check). The cache's
    load-time `verify()` compares every slot's and bank entry's bytes against the file
    (`code`: `llama-expert-cache.cpp:570-589`), which is a tripwire that catches the corruption after
    the fact (Strata's 22 GiB B70 arena aliased two 2 MiB pages → NaN at layer 4, expert 249,
    `paper`, 78f61744), not a pre-check.
  - **Pageable host:** none — the bank and the staging ring are USM host allocations (`code`:
    `usm_host_alloc`), and no pageable staging exists, so the pageable-fault/deadlock class
    (`paper`, 73e01fa5, fca3c595) is avoided by construction.
  - **Blitter/timeout:** no large memsets in the engine's path (`code`); the 25 GiB memset timeout
    (`paper`, 6d1cb3a6) has no analogue.
  - **The bmg staging that was not needed** (`paper`, 7a07f281: identical logits, ~10 % of a prompt)
    — arcint has no such staging.
- **How arcint's build differs.** Strata's is SYCL with its own arena and a tag-check; arcint's is
  OpenCL over the driver's allocator with a byte check at load and the debug-only `GGML_OPENCL_
  CPU_MISS_RED`/`MUTATE` reds. The JIT option is the one concrete gap.
- **Quality class: exact/tripwire.** A hygiene change must not alter numbers; the tests are
  corruption detectors.
- **Units.** all three (the JIT hazard is any >4 GiB buffer: Flash-Next's slots, the image service's
  16.7 GB VRAM, the agent's KV+weights); the arena tripwire is Flash-Next's.
- **Pricing measurement.** None on a card: one build with the JIT option added and the suites
  (`test-backend-ops`, the three services' checks) green is the whole cost. The alias check costs a
  `clGetMemAllocInfoINTEL` round trip at load. If a window is spent, the cheapest is: a fresh load of
  Flash-Next at the unit's slot budget with the copy check's "0 of N" line and a served check.
- **Implementation sketch.** Add the JIT option to the OpenCL program's compile options (a one-line
  change in `ggml_opencl_make_compile_opts`, `ggml-opencl.cpp:1779-1795`) — patch 0068+, ~1 line plus
  its README section; and, optional, an alias check on the slots/ring allocations behind a warning.
  Red-first: a probe kernel that writes a pointer offset past 4 GiB — mutation: the option removed,
  the probe must return the NaN pattern; the served copy check — mutation: one slot byte flipped,
  `1 of N` (the existing `LLAMA_EXPERT_CACHE_MUTATE=1` red must fire).

## 20. quality instruments

- **arcint status: partly.** Built: a reference-logits capture and a per-row KL reader with the QSA
  boundary (2,051 tokens, the row split) — `arcint-src/tools/kld_capture.py`,
  `tools/kld_served.py`, `tools/capture_vs_capture.py`, tests under `tests/python/` (`code`), and KL
  as served via `llama-perplexity -lm mmap --no-repack -ub 4` (DESIGN §5).
  - Lacks: an **assistant-token mask** — the served KL arms are whole-window perplexity (8 x 512,
    16 x 512) and never mask the positions the model does not write (`code`: no assistant/role mask
    in the tools); a **noise-twin floor** — no double-run same-configuration arm (`code`: no "noise"
    in the tools); and the FP32 router/head logit instruments (Infernix's "the model rounds router
    logits to BF16 → ties" and "BF16 head logits flip 1.3 % of positions", `paper`, In 75029865,
    In 32731f06) — arcint has no such probe.
  - The reference's measured case for the mask: 99 % of every arm's KL sat in positions the model
    never writes, and on assistant tokens 1.4e-3 to 3.9e-4 nats, noise twin 4.8e-5 (`paper`, In
    3ed21427). arcint's 0.03-nats bar has no floor, so an arm can pass or fail on text the model never
    emits.
- **How arcint's build differs.** Infernix runs a precise FP32 reference and a noise twin per arm;
  arcint runs one reference (the CPU backend or a higher quant) and reports mean KL and top-1. The
  difference is analytical, not a mechanism.
- **Quality class: n/a** (instruments; they gate the others).
- **Units.** all three.
- **Pricing measurement.** The instrument's own validation, no card: a noise twin = two runs of the
  same binary and configuration on the same prompts; its mean KL is the floor every arm must beat.
  Cost is one extra process per KL arm (the campaign's arms are 8-16 x 512, `paper`).
- **Implementation sketch.** `tools/kld_served.py` gains a position mask from the request's
  assistant spans (the server's rendered template already knows the roles), a `--noise-twin` mode in
  `tools/capture_vs_capture.py`, and an optional `LLAMA_LOGITS_F32=1` diagnostic in the LM-head path.
  Files: `tools/*.py`, the backend's logits read (`src/exec/backend_llama.cpp`). ~150-250 lines.
  Red-first: the mask's unit test (a planted high-KL position in a non-assistant span must not move
  the mean) — mutation: the mask off by one token, the test fails.

### Built (2026-10-31, `code`, device-free tests only)

The mask and the noise floor are in the tools, before the first quality-moving arm:

- **the assistant mask** (`code`). `tools/kld_served.py`'s `assistant_spans` reads a chat request's
  turns and the assistant turns' token ids — the ids the capture's window holds (the pin's
  `llama-tokenize` at the capture's own parse-special setting, since perplexity does not parse
  specials) — and locates each turn's span in the window forward from the previous turn's end, so a
  multi-turn transcript masks the earlier assistant turns and the reply. Because the located
  sequence is the turn's text WITH its closing marker, a user turn quoting the reply's text cannot
  be taken for the reply. A turn whose ids are not in the window is refused by name, never masked
  silently. `tools/capture_vs_capture.py --spans FILE` prints the masked mean and argmax beside the
  unmasked ones.
- **a row scores the next token** (`code`). A `--kl-divergence-base` capture is written by
  `kl_divergence()` (`llama-src/tools/perplexity/perplexity.cpp:1675`): its worker scores row `i`
  with `log_softmax(..., tokens[i+1], local_kld)` (`:287`; the plain-text writer does the same at
  `:157-160`), `first = n_ctx/2` at `:1772`, and the row's token base is `tokens.data() + start +
  seq*n_ctx + first` (`:1887`), so row position `p` holds the
  prediction of token `p + 1` and a token range `[start, end)` is masked as the row range
  `[start - 1, end - 1)` (`token_rows`). The reply's first token's prediction — the one after the
  assistant header, usually the reply's highest-entropy token — is inside the mask, the token after
  the reply is outside. The convention is stated in the tools' docstrings and in the single mode's
  output line.
- **a reply's ids are tokenised with its closing marker** (`code`). Standalone, a reply ending in
  punctuation gets a different last id: the Qwen2 pretokenizer folds the trailing punctuation and
  the marker's leading characters into one token (its regex alternative
  `?[^\s\p{L}\p{N}]+[\r\n]*`). The transcript therefore carries the turn's ids as text PLUS
  marker tokenised together, and the marker's own ids; `reply_end` drops the marker's ids from the
  tail and keeps a merged last token, and refuses a tail that matches neither.
- **a corpus arm has no roles** (`code`). `llama-perplexity --kl-divergence-base` over plain text
  carries no roles; the tool states that every recorded position is scored there and invents no
  mask. The claim is in the tool's output and in its docstring.
- **the noise floor, two numbers** (`code`). The twin mode reports mean KL and argmax agreement as
  separate floor values, each with the two baseline runs' own spread, and flags a candidate "inside
  the floor" when its difference lies within that spread, "outside" otherwise. `CLAUDE.md`'s
  answer-level bar (0.03 nats worse than the baseline arm, argmax at most 1 point below) is printed
  beside it, unchanged, as its own verdict — a card's floor is no bar.
- **the FP32 logit diagnostic is not built** (`code`). The graph DECLARES f32: the head's
  `ggml_mul_mat` result is `GGML_TYPE_F32` (`llama-src/ggml/src/ggml.c:3370`), qwen4exp takes that
  tensor as `res->t_logits` (`llama-src/src/models/qwen4exp.cpp:519-521`, MTP head `:613-615`), and
  the MoE router asks for f32 accumulation CONDITIONALLY — `ggml_prec_set_acc(logits,
  GGML_PREC_F32)` inside `if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS)`
  (`llama-src/src/llama-graph.cpp:2170-2171`; `:2155` at the pin), so not for other gating
  functions. A declaration does not settle the KERNEL's arithmetic — which is what the reference's
  BF16-logit finding is about — so the accumulator forms of kernels whose result this series
  deposits as f32 were read: `float sumf[N_R0_MXFP4]`
  (`llama-src/ggml/src/ggml-opencl/kernels/mul_mv_mxfp4_f32_flat.cl:127`), `float8 sumf`
  (`kernels/mul_mv_iq4_nl_f32_flat.cl:114`), `float4 sumf` in the expert-routed GEMV
  (`kernels/mul_mv_id_q8_0_f32_flat.cl:136`) and the router kernel this tree adds (`float wsum`,
  `contrib/llama.cpp/patches/0064-opencl-intel-topk-moe-subgroup.patch`). That is not a claim about
  the LM head: it is a dense matmul, neither expert-routed nor one of those three weight types on
  the served GGUFs. NOT examined: the dispatch of a given shape to a given kernel for every model,
  which Adreno/Intel path wins, and the f16 GEMM accumulators.

The reference's mask case remains `paper` (99 % of an arm's KL in positions the model never writes);
the arcint counterpart is owed: one served KL arm with `--spans`, masked against unmasked, on the
agent and on Flash-Next, with a twin. No card run has been made here.

The transcript the arm needs is written by `tools/tokenize_turn.py`, which pins the capture's
tokenisation: perplexity's `common_tokenize(ctx, prompt, true)` omits `parse_special`, so it takes the
default `false` (`llama-src/common/common.h:1159-1163`; the calls at
`llama-src/tools/perplexity/perplexity.cpp:310`, KL mode `:473`), while the pin's `llama-tokenize` defaults `parse_special = true` for its own example
(`llama-src/common/arg.cpp:1408`) and its affirmative `--parse-special` is imatrix-only (`:3227-3233`);
`--no-parse-special` is the tokenize example's switch (`:3255-3261`, passed to common_tokenize at
`llama-src/tools/tokenize/tokenize.cpp:185`). The helper always passes it, so the closing marker is
TEXT and matches the window. Per assistant turn it tokenises the turn's text followed by the
template's closing marker — built from parts, never typed literally — as ONE stream, and the marker
alone; the two id lists are the turn's `tokens` and `marker_tokens`. The merged-last-token rule is a
heuristic over the pretokenizer's regex, not a measured case. Then one `llama-perplexity
-f <chat>.txt -c <C> -b <C> -ub 2048 --chunks 1 --save-all-logits` per arm (the reference, the twin's
two runs, the candidate) and

    python3 tools/capture_vs_capture.py REF.kld --baseline A.kld B.kld \
      --candidate C.kld --window 0 --spans request.json

`<C>` must put the reply in the recorded half: rows are `n_ctx/2 .. n_ctx-2`, so the reply's last
token has to sit below `n_ctx - 1`, and a mask selecting no row is refused by name.

---

## Order, and what I would move out

Order (by the prize arcint can actually collect and the cost):
1. **3** PCIe probe + miss balance — the largest decode lever on the mission path that arcint does
   not have, host-only, and its cap sweep is already on record.
2. **2** int8-DPAS prompt expert GEMM — prefill on Flash-Next and the coder, a kernel form on an
   existing route, the reference's gain measured on the B60's silicon family.
3. **13** MTP width policy — cheap, all three units, and the width cliff is already measured.
4. **11** chunk/residency dial — knobs exist; the first step is a sweep, the loan only if the sweep
   says the reservation costs.
5. **14** per-card profile — cheap, exact, and the B70's +21 % mean is the prior; the B60 arm is a
   served A/B.
6. **15** prefix reuse — the tokenizer pass is the residue on a long agent transcript.
7. **5** QSA select — Flash-Next at depth, after 1/2/3/4 (its cost is a kernel pair).
8. **6** CPU rows' prefetch — small, one flag and one hint; the rows themselves are built.
9. **10** copy-discipline leftover — the 4 MiB staging cap at depth, measured first.
10. **16** launch fusions — the qk-norm+rope one; bundle with two more and gate at the wall.
11. **1** layer-major (span-major) — **demoted**: the measured across-ubatch ceiling is +3.1 % and
    the build is the largest here; the tile-padding probe decides whether it is even that. Build
    only after 2, and only if the probe shows padding.
12. **19** xe hygiene — the JIT option and the alias check; cheap, correctness class.
13. **20** quality instruments — the assistant mask and the noise floor; they gate everything, so
    they are worth doing before the first quality-moving arm (3's re-tune, 13, 5) — if that is the
    sequence, move this to order 3.

Move out of the build queue (keep the audit rows):
- **7** and **9**: arcint has both mechanisms, with measurements; no work remains.
- **18**: the reference's mechanism was tested on the exact engine, model and card and lost; the
  dense half is built. Re-opening needs a new kernel shape → a new candidate.
- **1** stays in scope but at order 11, with the first step a measurement rather than a build.

### 14a. the build and its offline pricing (2026-10-30, `code` + `measured-here` offline replay)

- **Built:** `tools/expert_profile_from_trace.py` writes the STRP file arcint's `read_profile`
  reads and Strata's `make_profile.py` / `write_expert_profile` write: header version 1, the
  ranked pairs, then the per-layer rank table (`code` in both writers; the table is written and
  arcint ignores it). The ranking is Strata's `rank_profile` (`code`,
  `tools/make_profile.py` at Strata's upstream, in the mirror at
  `~/src/strata-profile-ref/src/tools/make_profile.py`): the traces' pairs by score, ties by
  `(layer, expert)`, then a base's own order (never reordered), then the fill `expert` outer,
  `layer` inner. **Deviation, named:** Strata's `make_profile_mix` normalizes per trace FILE,
  because a Strata trace file is one prompt's run; an ERT1 file is one process's many requests,
  so the tool's unit is a REQUEST (each request's counts / its total, then its file's weight;
  `--raw` keeps `make_profile`'s additive counts, `--per-file` restores Strata's unit). Prompt
  and decode positions count at the SAME weight: Strata's `read_trace` counts every record's k
  ids alike and does not separate the two (`code`); a DRAFT record never counts, it is another
  layer's ids.
- **Replay seeding:** `--profile FILE` seeds the cache's slots from a ranking, `--profiles` the
  three-arm comparison; the per-layer slot count stays the trace's (the run's budget, the trace
  carries no sizes), so the arms share one budget and differ only in placement. `--every 0` is
  now adapt-off (the engine's `LLAMA_EXPERT_CACHE_ADAPT_EVERY=0`); before it adapted every step
  (`code`, the replay's own `since < 0`).
- **The trace is the served unit's own run.** The replay of the trace's resident set reproduces
  `expert-profile-unit.bin`'s placement to the slot: 5,915 of 5,915, the same 83.88 % hits with
  adapt and 55.19 % placement-only (`measured-here`). So arm (a) is the file the unit loads.
- **Held out, the odd requests, 399 decode steps, 5,915 slots, delay 2, 96 swaps every 6 tokens:**

  | placement | adapt on | adapt off | swaps (on) |
  |---|---|---|---|
  | the shipped unit profile (`expert-profile-unit.bin`) | 83.88 % | 55.19 % | 12,819 |
  | Strata's `data/expert-profile-sycl.bin` (`paper`: the B70's card profile) | 84.21 % | 64.02 % | 12,812 |
  | built from the trace's EVEN requests (5 of 9, 24,161 pairs) | 84.21 % | 66.02 % | 12,772 |
  | built from ALL 9 requests (in-sample, not a result) | 84.81 % | 78.92 % | 12,540 |

  All four rows `measured-here`, offline replay on the served public-corpus trace. Reading: the
  trace-built profile moves placement-only hits +10.8 points over the shipped file and +2.0 over
  the xe profile; with the served adapt on it is +0.33 over the shipped (83.88 -> 84.21), the
  same as the xe file. Adaptation recovers most of placement's gap, which is why the on column
  moves so little; the placement column is the profile's own effect. This is one 9-request trace
  of public corpus text, so it prices the mechanism, not the shipped file's workload; the B70's
  +21 % mean over 12 unseen prompts is `paper` and this trace does not reproduce it.
- **Save path (round 2, `code`; regenerated for 0068, `measured-here` recipe).**
  `--llama-expert-cache-save FILE` (patch **0070**, after the trace's 0069; **off by default**,
  one getenv and nothing else). The patch number moved because the prefill slot loan takes 0068
  when it integrates; the trace keeps 0069. Both 0069 and 0070 touch
  `llama-expert-cache.{h,cpp}`, the destructor and the layer struct the loan already changes,
  so both are regenerated **on top of 0068** and exported in series order; the whole
  0001-0070 series applies on the pin (recipe line below). The writer and reader
  live in the dependency-free `llama-expert-profile.{h,cpp}`; the cache's static `read_profile`
  delegates to it, and the destructor saves at teardown after `adapt_finish` has landed the
  pending copies. The header and the writer now agree: the file is Strata's whole STRP file,
  header, pairs, then the trailing layer x expert rank table (expert -> rank, -1), which the
  writer emits and the reader stops before (`code`; a cell asserts the C++ bytes equal
  `make_profile.py`'s byte for byte).
- **The save's default rule is the xe one, not the cu one (round 2, `code`).** Strata has two
  save paths. Its SYCL/xe `save_profile` (`sycl/src/program/generate.cpp`, commit 3d6f1718)
  mixes the device's cumulative counts with the start's ranking as pseudo-counts: lambda 2,
  `STRATA_PROFILE_PRIOR_ENTRIES` 800,000 routed entries by default, each prior pair worth
  `prior_entries * lam * exp(-2 * rank / n) / (1 - exp(-2)) / n`; "a short conversation must not
  wipe the prior". The default `xe` rule now builds that ranking (no resident head): the
  counts plus the prior mass, then Strata's `rank_learned_profile` comparator. `--llama-expert-
  cache-save-rule cu` keeps Strata's other path (`src/core/expert_cache.cpp`
  `rank_learned_profile`, `code`): the resident pairs by heat, the counted non-residents by
  heat, then the rest, ties by the prior's rank then the index. Which arm uses which: the save
  default is **xe**; the `cu` arm is named only by that option, and the offline profile arms
  below are built by `make_profile.rank_profile` (a third writer), not by either save rule.
  The earlier round 1 build implemented only cu and labelled it as the cited mechanism; that was
  wrong, and the xe path is what the B70 measurement (`paper`: code +13..19 %, prose -18 %)
  measured.
- **Heat, Strata's xe unit (round 3, `code`).** The save ranks from per-pair `heat` in raw
  routing counts, one a routed pair, counted in the cache's `count()` beside `usage += 1`
  (Strata's per-routing-event device counter, `sycl/src/program/generate.cpp:8145-8151`), in
  `double` (a month at ~33 t/s passes float's 2^24 on the hottest pairs). Round 2's
  `heat += usage` before every adapt decay was 1/(1-decay) larger (3.33 at decay 0.7), so the
  800,000-entry prior weighed like ~240k raw entries instead of 800k; and with
  `LLAMA_EXPERT_CACHE_ADAPT_EVERY=0` the heat never accumulated at all, because it was fed
  inside `adapt()`. Both findings are `code` against Strata's source; the fix is `code`, the
  red cells `measured-here` (mutations h/i/j in the round-3 list). The save's xe line prints
  the routed entries counted next to the prior, as Strata's stderr line does (`code`, same
  source).
- **Pending swaps at save (round 2, `code`).** The teardown waits for the copies but does not run
  `progress()`, so the host tables stay at the last phase. The save now marks every pending
  incoming expert resident (a copy in flight counts as done, Strata does), and keeps the
  outgoing expert resident only in phase 1: at phase 2 the host table has already freed its
  slot and the save clears it.
- **Config guard (round 2, `code`).** `--llama-expert-cache-save` equal to `--llama-expert-
  profile` is refused at config, because the save would overwrite the file the run reads. An
  unknown `--llama-expert-cache-save-rule` is refused rather than silently xe. A save without
  `--llama-expert-cache` stays refused.
- **Replay (round 2, `code`).** `profile_slots` now models the engine's global walk: one pass
  over the whole ranking, a pair into its layer's next slot until the trace's total slot
  budget is spent, then a zero expert for a layer with no slot. The budget is counted in
  frames because the trace carries no expert bytes; a Q8_0 down and a gate/up pair differ in
  bytes and the engine walks bytes, so the table's per-layer slot counts are the walk's
  distribution, not the trace's 93-155. `read_profile` refuses a file shorter than the 24-byte
  header and a pair count larger than the geometry, as well as a truncated pair.
- **Device-free cells (round 2, `measured-here` via the recipe):**
  `tools/expert_profile_save_selftest.cpp` and `tests/python/test_expert_profile_save.py` now
  carry round trip and same placement, truncation refusal with no partial ranking, version 2
  refused before the geometry, the prior-rank tie-break, the pending incoming/outgoing rule in
  both phases, the xe prior-mass rule (a 9-count pair does not outrank 800,000 prior entries;
  with 0 prior entries it does), the heat unit (one a routed pair, raw, out-of-range ids
  skipped), the heat width (a count past float's 2^24 stays exact), and the C++ / Python byte
  equality. The Python cell is stdlib `unittest` and CTest runs it with `python3` alone
  (`expert-profile-save`, LABELS unit, `EXPERT_PROFILE_SRC` pointing at the tree the build
  compiled against): pytest is not a build dependency (round 2's `-m pytest` broke the recipe
  on a host with python3 only, `measured-here`), and the cell skips with a message when the
  profile sources are absent. A source cell asserts the heat call sits in `count()` and not in
  `adapt()`, so adapt-off still counts.
- **Red-first, one-line mutations, each run and seen failing before restored (round 2,
  `measured-here`):** (a) heat comparator `>` -> `<`: "the ranking is Strata's rule"; (b) drop
  the version check: "a version 2 profile is refused"; (c) accept a short pair read (`!= 4` ->
  `> 4`): "a truncated profile is refused"; (d) drop the prior-rank tie-break (`rank_of(a) <
  rank_of(b)` -> `false`): "ties take the prior's rank before the index"; (e) drop the pending
  incoming mark: "a pending incoming expert is resident"; (f) clear the outgoing in phase 1 too
  (`>= 2` -> `>= 1`): "an outgoing expert stays resident in phase 1"; (g) drop the xe prior mass
  (`prior_entries > 0 ...` -> `false`): "xe mixes the prior in as pseudo-counts". The replay's
  length guard's mutation (drop `len(b) < 24`) was also seen failing: the cell expected the
  short-header refusal, got the magic refusal.
- **Red-first, round 3 (`measured-here`):** (h) heat counts a decayed weight (`+= 1.0` ->
  `+= 1.0 / (1.0 - 0.7)`): "heat counts one a routed pair, raw"; (i) narrow the heat to float
  (`+= 1.0` -> `= (double)(float)(...+1.0)`): "heat keeps a count past float's 2^24"; (j) move
  the heat call from `count()` to `adapt()`: the source cell's count() assert fails; (k)
  truncate `profile_slots` back to the trace's n_slots: "the walk's distribution, not the
  trace's" fails (a 5-frame budget lands 4). Each was run and seen failing before restored,
  with `PYTHONDONTWRITEBYTECODE=1`.
- **Red-first, a caveat worth keeping:** run mutations with `PYTHONDONTWRITEBYTECODE=1` (or after
  clearing `tools/__pycache__`). A stale `.pyc` from an edit earlier in the same second with the
  same file size made one mutation read green until the bytecode cache was cleared; the
  adapt-off cell was also strengthened (a six-step stream that does meet the engine's own
  `min_usage` 2 / gain 1.5 thresholds, with adapt ON as the positive control) because the
  two-step stream could never swap. `measured-here` is only a claim once the fresh interpreter
  reproduces it.
- **Recipe:** `RECIPE 64908b5: 100% tests passed, 0 tests failed out of 7 EXIT 0 deb
  arcint_1.2.6-1_amd64.deb`, patch 0070 applied in sequence on the pinned tarball
  (`measured-here`); the 7th test is `expert-profile-save`, `7/7 Test #7: expert-profile-save
  ... Passed 1.23 sec`. Pytest is not a build dependency: the cell's own command under a
  site-packages-free interpreter is green and pytest is absent there (`measured-here`):
  `python3 -S -c 'import pytest'` -> `ModuleNotFoundError: No module named 'pytest'`, and
  `PYTHONDONTWRITEBYTECODE=1 EXPERT_PROFILE_SRC=<tree> python3 -S tests/python/
  test_expert_profile_save.py` -> `Ran 2 tests ... OK (skipped=1)`, the skip being the byte
  equality's numpy import. The served warm-start arm and the out-of-sample gate stayed owed in
  that round: no card run was made there. Claude's 2026-10-30 card runs carry them below.
- **Out-of-sample gate (round 2, `code` for the tool, owed for the measurement).** Every trace
  so far came from `fn_sweep.py`'s own six prompts, so a trace-built profile was in-sample for
  them. `tools/fn_sweep.py --prompts FILE` now takes a JSON prompt list;
  `tools/fn_sweep_prompts.json` is the training capture set (36 prompts, English and German:
  public-domain prose summaries, code questions and explanatory prose), written for this
  campaign and nothing private. The gate: capture a trace with the training prompts only,
  build the profile from that trace only, then run the served A/B with the six standard
  prompts as the unseen side — unit profile against trace-built, alternated twice, adapt on,
  plus the adapt-off pair. The six standard prompts are never in the training capture, and
  neither is the capital question: `fn_sweep.py --no-capital` leaves it out of the capture
  (`code`), so no training prompt routes the gate's own question. The served arms carry it,
  which is what makes the capital check shared across arms (`code`). The training set's topics
  were checked against the gate's six and against each other in round 3: five overlapping
  topics (hash tables, compilers, GC, CSV, WAL/B-trees) were replaced by bit counting, blocked
  matrix multiply, a backtracking Sudoku solver, an hour-bucket time series and mRNA vaccine
  prose, and the recursive-descent parser (compiler-adjacent) by Dijkstra — none is the gate's
  topic nor another set member's (`code`, the JSON's own names). The exact card commands are
  in the round-3 handover (`docs/campaigns/research-concept-mining/handover-round3.md`).
- **The card results (2026-10-30, `measured-here`, B60, Flash-Next IQ3_S, Claude's runs; the
  served arms, the trace and the save are off by default and were switched on for these arms):**
  - served item 14: the trace-built profile against the unit profile gains **+0.35 %** decode
    with adapt on and **+5.7 %** with adapt off;
  - the warm start on a saved profile: **+2.5 %** in-sample, **-1.9 %** out-of-sample;
  - profiles change answers: the CPU miss's rounding, a **rounding-only** class;
  - the MTP draft routing is the best predictor: **76.2 %** replacing, **+0.26 blended**;
  - the blend rule is the limit: the oracle with the rule off **+6.52**, blended **+1.28**.
- **Recipe on the integrated branch (2026-10-30, `measured-here`):** `RECIPE 23a4295: 100% tests passed, 0 tests failed out of 7 EXIT 0 deb arcint_1.2.7-1_amd64.deb`, the whole 0001-0068, 0069, 0070 series applied on the pinned tarball in order; the 7th test is `expert-profile-save`. The commit after it records this line; the code is unchanged.
