# gqa-small-t-decode — decode and verify attention in one pass over K/V per KV head

**Open.** Started 2026-10-04.

## Charter

The libllama engine's decode attention (0007, quantized KV in 0015) gives
every (query head, query row, KV split) its own 16-lane sub-group. Each one
reads its KV slice:
- the 6 query heads of a KV head read the same K/V. L2 catches part of
  that;
- every verify row of an MTP step re-reads the whole KV. One row's pass is
  ~285 MB per attention layer at 131k (q8_0), far beyond L2.

Measured on the B60, one attention layer of the dense 27B (24 query heads on
4, head size 256), f16 K/V (`measured-here`, test-backend-ops perf,
2026-10-04):

| keys | 1 row | 6 rows |
|---|---|---|
| 8,192 | 0.17 ms | 0.67 ms |
| 32,768 | 0.63 ms | 2.75 ms |
| 131,072 | 1.91 ms | 10.3 ms |

Times the model's 16 attention layers, a 6-row verify at 131k spends ~165 ms
in attention. The served agent (131,072 tokens, q8_0 KV, MTP 5) measured
~350 ms a verify step and 7.5 t/s at 128k depth.

The lever: one pass over K/V per KV head and split, covering all of that
head's query heads and all verify rows.

## Reference to follow

NInfer (`~/src/ninfer`, `code`, read 2026-10-04):
`src/ops/attention/causal_softmax/small_t_bf16.cuh`, `small_t.cu`.
- One CTA per (KV head, split, batch): `dim3 grid(Geometry::KVHeads,
  splits, batch)` (`small_t.cu:99`).
- The CTA covers the KV head's whole GQA group (6 query heads on the 27B)
  and up to `TokenTile` = 6 query rows (`small_t_bf16.cuh:18-26`), the MTP
  verify window.
- Tensor-core MMA for QKᵀ and PV even at decode (`small_t_bf16.cuh:196-273`).
- Split-K partials merged by a log-sum-exp reducer (`small_t.cuh:156-270`).

Measured there: 54.8 t/s decode with 260,096-token prompts, dense 27B, RTX
5090 (`paper`: its model card's table, not reproduced here).

OpenVINO's `paged_attention_opt.cl` shares K/V reads across query heads in
registers (HEADS_PER_WI, `code`, cited in `llama-engine-kernel-gap.md`
lever 2).

The record has one failed form (`llama-engine-kernel-gap.md` lever 2): a KV
head's query heads in one work-group, 16-key K/V tiles staged in local
memory, a barrier every 16 keys, six sub-groups, no XMX. It measured
slower.

## Design (Xe2, sub-group 16), as built in 0016

NInfer's form was built first (v1: no local memory, no barriers, K and V
read by every sub-group). The rows then shared nothing: 8.86 ms for 6 rows
at 131k against 0015's 10.3 (`measured-here`). The cause was the per-lane V
gathers and the repeated K reads.

The shipped form (v3) stages, like the failed one above, but differs in
three ways:
- both products on XMX;
- 32-key tiles;
- only V (and a quantized K) staged.

So the earlier verdict was on a build without XMX and with 16-key barriers,
as that record had said.
- One work-group per (KV head, split). Two sub-groups (`GQA_SIDES`) per
  8-row query tile, each owning 128 of the 256 output columns. Row m is
  (query head m mod G, token m div G), G·T rows padded to 8.
- Each sub-group holds:
  - its query tile as the DPAS A operand (8 rows × 256, half);
  - S for 32 keys (C layout: lane = key);
  - O as 8 rows × 128 floats;
  - its own running max and sum.
- Per 32 keys the work-group stages V in local memory as f16, transposed
  with the key fastest, with two barriers.
- K enters QKᵀ as the B operand:
  - f16: read straight from the cache by its key's lane (16 consecutive
    values are the packed pair layout);
  - quantized: staged too, converted once per work-group. Direct quantized K
    made every sub-group convert the same blocks: served decode 10.8 against
    15.6 t/s.
- P leaves the softmax already in A layout (lane = key).
- Partials go to the existing merge kernel, same record layout.
- Routed from 4 rows. 1-3 rows stay on 0015's split kernel, which is faster
  there.

## Gate

- **Correctness:** FLASH_ATTN_EXT in full on the B60, including 0015's 90
  cases (batch 1, 3, 8; GQA 6 and 8; f16, q8_0, q4_0). A fault injected in
  the new kernel must fail them.
- **Speed (the attention op, B60, dense geometry):**
  - a 6-row call at 131k costs at most 2x a 1-row call (now 5.4x);
  - a 1-row call is no slower than 0015's, at 8k and at 131k.
- **Speed (served, the agent's configuration):**
  - decode with MTP at 128k depth above 7.5 t/s;
  - decode on the acceptance prompt within the run-to-run spread.
- **Answers:**
  - KL for 8:8 within the answer-level bar against 0015;
  - the sampled task within the noise of 0015's (mean 7.4, n=30).

## Current state (`measured-here`, B60, 2026-10-04)

One attention layer at the dense geometry, f16, test-backend-ops perf:

| keys | rows | 0015 | v1 | v2 |
|---|---|---|---|---|
| 8k | 1 | 0.165 ms | 0.24 ms | 0.57 ms |
| 8k | 6 | 0.67 ms | 0.72 ms | 0.52 ms |
| 32k | 1 | 0.63 ms | 0.87 ms | 1.69 ms |
| 32k | 6 | 2.75 ms | 2.65 ms | 1.92 ms |
| 131k | 1 | 1.91 ms | 2.56 ms | 5.39 ms |
| 131k | 6 | 10.3 ms | 8.86 ms | 5.27 ms |

- **v1** (no local memory; K and V read per sub-group, V per lane): the rows
  shared nothing. Ablation at 131k, 1 / 6 rows:
  - PV's loads cost 1.6 / 6.9 ms;
  - QKᵀ costs 1.3 / 4.4 ms.
- **v2** (K and V staged per work-group, 32 keys a tile, as 0008 does): 6
  rows now cost what 1 row costs. Ablation at 131k:
  - staging alone 2.0 / 2.2 ms;
  - the products alone 1.3 / 3.0 ms;
  - the two add up, so they don't overlap;
  - BK 16: 3.8 / 5.6 ms; BK 64: 12.0 / 6.3 ms (local memory limits the
    work-groups per core).

- **v3 (0016 as built):** K straight from the cache when f16, staged when
  quantized; V staged; two sides; BK 32. At 131k:
  - f16, 6 rows: 4.83 ms (0015: 10.3);
  - other forms: K staged for f16 5.27; one side 10.6; four sides 6.96; BK 16
    4.91; BK 64 9.67.

  q8_0, llama-bench at 32k:
  - a 6-row forward 47.7 -> 52.6 t/s with K staged, 30.7 with K direct;
  - a 3-row forward 29.2 -> 27.7.

  Routed from 4 rows (`GGML_OPENCL_FA_GQA_MIN_ROWS`).

Served (the agent's flags: 131,072 tokens, q8_0, MTP 5), a 62,597-token
prompt:
- decode at that depth 11.0 -> 15.6 t/s (+42 %);
- verify time 9.78 -> 6.65 s;
- identical draft statistics;
- prefill unchanged;
- greedy task 10/10;
- sampled 10 runs mean 7.6 (0.5.6: 7.2-7.4).

- FLASH_ATTN_EXT 2,757 of 2,758 with the 4-row route, with 24 cases at 4-7
  rows.
- Red (BK=24) fails all 42 cases at 4-8 rows.
- KL through the kernel (q8_0, a 6-token ubatch): 0.003589 against 0015's
  0.003593.

Gate:
- the 6-row cost is 2.5x the 1-row call at 131k (bar 2x): not met;
- 1 row is unchanged (it stays on 0015);
- served decode with MTP: +42 % at 62,597 tokens of depth (11.0 -> 15.6
  t/s); the gate's 128k row is not re-measured (0.5.6: 7.5 t/s);
- the answers hold: greedy task 10/10, sampled within the noise, KL equal.

Open: overlap of staging and products (prefetch into registers), S computed
once per tile, an SG8 form for the A770. These are the first targets of the
code-level GA (`kernel-autotune-ga.md`).

## Where it lives

`contrib/llama.cpp` patch 0016:
- `kernels/flash_attn_gqa_dpas.cl`;
- the decode route in `ggml-opencl.cpp`;
- the verify-row cases and the depth perf cases in test-backend-ops.

## Round 14 -- the non-paged number at 240k (`measured-here`, window 291, 2026-11-16)

The production agent's 240k cached 32-token decode, unitrace on the B60: device span 2,713 ms
over ~13 graphs, attention 1,841 ms (68 %). `flash_attn_gqa_dpas` (4-7 rows) 116 calls x
8.34 ms; `flash_attn_q8q4_q1_split_intel` (1-3 rows) 100 calls x 8.74 ms; merge 0.22 ms.
~938 keys x 416 B a call = ~400 MB, ~48 GB/s. Window 289b's paged call at 261k: 9.15 ms,
~47 GB/s. Paging costs ~1 %: the kernel is the bound, non-paged and paged alike, and the
1-row path costs what the 6-row path does because both read the KV once per call.

### A. Analysis (`code`, read 2026-11-16)

Per work-group, per call, q8_0 K / q4_0 V, head 256: a K row is 256/32 blocks x 34 B = 272 B,
a V row 8 blocks x 18 B = 144 B, so 416 B a key. 0097's split (128-256 keys, cap 256) gives
each work-group ~938 keys = ~390 KB. Work-groups a call = 4 KV heads x splits x row tiles;
at 240k and cap 256 that is 4 x 256 x (rows/8-per-side) = ~1024-2048 work-groups. The B60
has **20 Xe cores** (160 vector engines), so with 0045's one work-group a core and 20 a wave
that is **~51-102 waves** a call (corrected 2026-11-16; the first draft said 6-13): the grid
is full many times over, so occupancy is not the problem.
- Both kernels: `#define BK 32` (`flash_attn_gqa_dpas.cl:66`), staged path
  (GQA_K_STAGED, :85) copies BK keys to local memory, then TWO `barrier(CLK_LOCAL_MEM_FENCE)`
  per tile (:305 before the stage, :318 after it, :336 for the q8 staging) plus the V and S
  exchange barriers (:23 says one covers V and S). So staging and the products are strictly
  serialised: while the tile loads, the DPAS units idle; while they compute, the memory
  system idles. Ablations 2026-10-04 measured exactly that (2.0 + 1.3 ms at 131k, 1 row).
- Loads in flight per sub-group: with BK 32 and staged K, one sub-group's stage is
  BK x DK/64 x 4 vector loads = 32 x 4 x 4 = 128 `load16` calls issued by the whole
  work-group's threads (nthr threads stride them, :307), so ~1-4 16-byte loads in flight per
  lane; the barrier drains them all before any product. Little's law, latency assumed
  ~700 ns (**assumed, no latency probe here**: the B60's memory is GDDR6, and the 14.3 GB/s
  copy measurement of 0032/0044 is per-slot streaming, not a latency figure): at
  700 ns and 16 B a load, 300 GB/s needs 300e9 x 700e-9 / 16 = 13,125 loads in flight device-
  wide, ~82 a core. One work-group a core with ~4 loads a lane x 16 lanes is ~64 issued but
  drained at a barrier, so the average in-flight count is far below 82 -- consistent with the
  measured 48 GB/s. This is the bound: not the arithmetic (DPAS is idle during staging), not
  the split, not the merge.
- BK 64 costs local memory and lowers the work-groups a core (the record's ablation: BK 32
  vs 16/64), so growing the tile is not the way out. Making K direct from VRAM (0016's f16
  form, :86: staged only when KT != 0) removes the barrier pair but leaves one sub-group's
  8-row x 16-lane loads as the in-flight count.

### References for long-context decode on this class of GPU (`code`, read 2026-11-16)

- llama.cpp CUDA: `flash-decoding` splits the KV and merges with the log-sum-exp rescale --
  `launch_fattn` (`ggml/src/ggml-cuda/fattn-common.cuh:981`), `parallel_blocks` (:928, the
  wave-fill split that 0045 ports), `flash_attn_combine_results` (:924, the merge). arcint
  already has that shape (0097 + the merge); the reference's advantage is not the split.
- Strata: its decode attention is split-K with an explicit merge too --
  `decode_attn_i8_split_kernel` and the merge block (`src/kernels/cuda/qsa_prompt_attn.cu:805`,
  :1018-1030, one block per (query, head), a thread per dimension). **Correction
  (2026-11-16): the Intel port is in the checkout under `sycl/`** -- my first grep covered
  `src/` only. `docs/INTEL.md`, `docs/INTEL_ARC.md`;
  `sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp` (the decode attention: scores in local
  memory, three `barrier()` a chunk at :186, :305, :326, cells read straight from global --
  no staged tile, no double buffer); `qsa_prompt_attn_esimd.dp.cpp` (:1-16: the FP32 prompt
  kernel is compute bound at ~60 % of the card's FP32 peak; the ESIMD form runs both products
  on the s8 DPAS, scores to local memory, one barrier later every thread has all 128 cells,
  load-block reads); `qsa_prompt_attn_xmx.dp.cpp`; `native_flash_attn.dp.cpp` (:253, :260,
  :297, group barriers a tile, loads by block). Strata's decode attention is **sparse (QSA,
  over selected cells)** -- a different mechanism from our dense GQA decode; read only for
  how its Intel mainloop issues loads: block loads straight into the DPAS a operand, one
  barrier a tile, nothing double-buffered. None of that reaches past our form.
- FreeToken: paged decode goes to `sgl_kernel.flash_attn_with_kvcache` with `num_splits`
  and `pack_gqa` (`python/freetoken/attention/fa.py:169,194`), i.e. it delegates to
  FlashAttention's split + GQA packing; its own Triton path is FA-shape, not a novel
  mainloop.
- Bandwidth fraction reached by any of them: none of the three states a fraction in code or
  comments (`paper`/`code` gap): their splits are about wave fill and occupancy, which this
  kernel already satisfies. So the mechanism to build is the one none of them ship on a
  memory-latency-bound IG: double-buffered staging with the products, i.e. more loads in
  flight per sub-group without more local memory.

### B. The build (next: patch 0130a)

Switch `GGML_OPENCL_FA_DECODE_V2=1`, default off, 4-7 row path first:
- double-buffer BK tiles in local memory (two BK-32 K stages), issue tile t+1's stage at
  `lid` before the S products of tile t, one barrier pair a tile instead of two, and K as
  `int8`/`half` vectors read by the DPAS a operand from the buffer being consumed -- so the
  stage of t+1 is in flight while t's DPAS runs;
- the same for the q8_0 1-3 row form (`ks8` doubled) if the first carries over.
Red first: op cases in `tests/test-backend-ops.cpp` at 4-7 rows, q8_0/q4_0, head 256, GQA 6,
4k/64k/240k keys (0016's depth cases exist; extend them to quantized KV), plus a
broken variant (single buffer, barrier before the product) that must fail.
Bar: CLAUDE.md's answer-level bar (KL within 0.03 nats of the baseline arm), op cases green.

### Round 15 -- built 2026-11-16 (patch 0130b, head 2ca72476)

The mechanism the analysis picked, behind `GGML_OPENCL_FA_DECODE_V2=1` (default off), 4-8 row
form: **two local V stages** (16 KB each at head 256, BK 32), tile t+1's stage at the loop's
top before tile t's Q K^T so its global loads run during the products, P V reading this tile's
buffer, the consumed buffer toggling at the tile's end. `local` a work-group goes 42 KB ->
58 KB (q8s 16 + q8d 2 + vt 32 + sx 8), below the 64 KB device limit and still one work-group a
core per 0045 -- so 0016's BK-64 local-memory loss does not apply. Barrier count unchanged:
one WAR fence at the tile's top, one before P V (the second previously also covered the V
tile, which now comes from the earlier stage). Red: `GGML_OPENCL_FA_DECODE_V2_BROKEN=1` makes
the stage overwrite the buffer the products read, and 0130a adds the quantized depths (4-7
rows, head 256, G 6, q8_0 K / q4_0 V at 4,096 / 65,536 / 245,760 keys).
Expectation at 240k (`inferred`, arithmetic): a call reads 4 KV heads x 240k x 416 B = ~400 MB;
today 8.34 ms = 48 GB/s with the V stage on the critical path. If the overlap lifts the
achieved rate to 75-90 GB/s (the scatter limit of 32-44 B rows on this class of GPU, not a
measured figure here), the same call is 400 MB / 80 GB/s = **~5 ms, 1.7x**; a 1.2x gain is the
conservative floor if only the latency, not the throughput, was hidden.

### Round 16 -- measured negative, window 293b (B60, build 7f4029f4)

**V2 does not pay.** Evidence: `measured-here` for the numbers, `code` for the mechanism.
- Op red gate: V2 BROKEN fails 0/18 on both I8 and f16 arms (8 of 8 kernels replaced; local
  59,392 B I8, 57,344 B f16), V2 green 18/18, base 18/18. So the stage is wired and the
  broken variant is red.
- Served, production Q4_K_M agent, 240k, no paging, 256-token decode: base 222.7 ms/step
  (17.89 t/s) against V2 255.9 ms/step (15.57 t/s) -- **V2 +14.9 % slower**. Needle found
  in both, so this is not a correctness regression, it is a throughput one.
- Served Q5 (UD-Q5_K_M), KV stream pool 1024, 261k, OVERLAP on: 262.2 ms/step, V2 283.3
  ms/step (+8.0 %).
- The perf arms measured nothing: `make_test_cases_perf()` builds its own list and 0130a
  added the depth cases to the test list only. 0130e fixes that (`code`).

**Why the double buffer hides nothing, from the source.** In OpenCL C a stage is a global
load into private registers followed by a store to local memory; there is no asynchronous
copy and no memory fence that lets the store be deferred past a shared-memory barrier.
`FA_GQA_STAGE_V` (`flash_attn_gqa_dpas.cl:258-275`) is exactly `load16` x4 per half16 into
registers, then 64 stores into `local`. The call for tile t+1 sits at `:355`, tile t's
QK^T at `:381` and P V at `:480-507` (line numbers of the 0130b tree, patch 0130a-c+d
applied). Every `load16` result is consumed by a store in the same
thread before any product instruction of tile t is issued: the hardware must have the data
before it can retire the store, the store precedes the barrier before P V, and the thread
stalls on the load's return at the first store. Only IGC could sink the stores below the
DPAS (an ISA question -- `ocloc` below); in the source the sequence is load, store, then
products. The second local buffer therefore buys no latency hiding inside a sub-group: it
adds 16 KB of local memory and one set of stores per tile, which is what the measured
numbers show.

**Occupancy, per Xe core (`code` for the kernel's own attributes, `paper`/`web` for the
hardware figures, marked below).**
- Work-group: `GQA_SIDES x n_tiles x 16` = 2 x 3..6 x 16 = **96-192 work-items** for 4-7 rows
  (G 6, head 256): 6-12 sub-groups per work-group, `intel_reqd_sub_group_size(16)`.
- GRF mode: the backend passes `-cl-intel-256-GRF-per-thread` for every FA GQA build, base
  and V2 alike (`ggml-opencl.cpp` ~5479, ~5548); a 16-lane sub-group at 256 GRF per thread
  needs the 256-register mode (the register file is 64 KB per thread at that mode).
- Local memory: base 42,224 B, V2 58,224 B per work-group (q8s 16,384 + q8d 2,048 + vt
  32,768 + sx 8,192). Both above half of a 64 KB local memory: **one work-group a core
  either arm**; V2's +16 KB does not cost a work-group.
- Hardware threads: B60 is Xe2, 20 Xe cores, 8 vector engines a core (`web`: the datasheet
  table, `docs/imggen.md:326`); 8 threads per engine at 128 GRF, 4 at 256 GRF (`inferred`,
  from `research-inferzero.md:67-68`, a patch 0013 record, not a vendor table). At 256 GRF
  that is 32 thread slots a core; a work-group of 6-12 sub-groups occupies 6-12 of the 32
  slots, and local memory caps it at one work-group, so **6-12 of 32 slots resident, 19-38
  %, identical for base and V2**. The grid's 51-102 waves say nothing about this: waves are
  dispatch rounds, resident threads are the latency-hiding resource, and both arms sit at
  one work-group a core. Reaching more would need a smaller local footprint (<= 32 KB a
  work-group, unreachable at head 256 with the q8s and sx stages) or a second work-group a
  core by shrinking the stages -- not by a second V buffer, which only raises them.

**Offline ISA comparison (`code`, commands for the card host, no GPU here).** `ocloc
compile -device bmg` with the backend's own option set:
```
ocloc compile -device bmg -file flash_attn_gqa_dpas.cl \
  -options "-cl-intel-256-GRF-per-thread -DDK=256 -DGQA_SIDES=2 -DKT=8 -DVT=4 -DGQA_I8=1" \
  -output base.isa
ocloc compile -device bmg -file flash_attn_gqa_dpas.cl \
  -options "-cl-intel-256-GRF-per-thread -DDK=256 -DGQA_SIDES=2 -DKT=8 -DVT=4 -DGQA_I8=1 -DGQA_V2=1" \
  -output v2.isa
ocloc compile -device bmg -file flash_attn_gqa_dpas.cl \
  -options "-cl-intel-256-GRF-per-thread -DDK=256 -DGQA_SIDES=2 -DKT=8 -DVT=4 -DGQA_I8=1 -DGQA_PRE=1" \
  -output pre.isa
ocloc compile -device bmg -file flash_attn_gqa_dpas.cl \
  -options "-cl-intel-256-GRF-per-thread -DDK=256 -DGQA_SIDES=2 -DKT=8 -DVT=4 -DGQA_I8=1 -DGQA_PRE=1 -DGQA_PRE_BROKEN=1" \
  -output prebroken.isa
```
KT/DT are the `gkv` codes (KT 8 = q8_0, VT 4 = q4_0); add `-DGQA_ST=1` for the 1-3 row
form. Compare: **spill bytes** (the "Spill Count" line, or grep `spill` in the `-stats` /
`IGC_ShaderDumpEnable` output), GRF mode line, the **`simd32`/`simd16` send count in the
loop body** (`send` instructions per tile near the DPAS block), and the IGC retry dump with
rolled loops (`IGC_EnableDPASLowering` / `EnableRetry` mentions). A store-count drop in PRE
with the loads hoisted above the first DPAS in the loop confirms or refutes the source
reading above.

### Round 17 -- 0130f, the register prefetch (`code`)

The mechanism 0130b should have been: tile t+1's V halves go into the thread's private
registers (`half16 pre[2][4]`, up to two units a thread at 96-192 threads) at the loop's
top, tile t's QK^T, softmax and P V run, and the registers are stored to the other stage
only after P V (`FA_GQA_PRE_V` / `FA_GQA_STORE_V`) -- the load's value is consumed after
the products, not before. No extra local memory over V2 (same two 16 KB stages), one extra
barrier avoided compared with a single-buffer form by writing the buffer this tile does not
read. Switch `GGML_OPENCL_FA_DECODE_PRE=1`, default off; the build is a separate programme
that replaces a pair's kernel only after it built and fits; `..._PRE_BROKEN=1` stores into
the buffer P V reads before the products, the red op cases. Expected at 240k (`inferred`
arithmetic): the V stage's stall moves off the tile's critical path; if the 400 MB call was
at 48 GB/s bound by that stall, the floor is the 1.2x that V2 promised, but now with no
additional local memory to pay for it. Gate: `tests/python/test_fa_decode_v2.py`'s
`FaDecodePre`, placements and the broken store, three mutations run and seen failing.

### Round 18 -- the private memory, and 0130k (`code`; ISA measured on the card host)

**The ISA (`measured-here`, ocloc 26.27.39122.11, `-device bmg`, source f11639bf, options
`-cl-intel-256-GRF-per-thread -DDK=256 -DGQA_SIDES=2 -DKT=1 -DVT=2`):** base, V2, PRE and
PRE_BROKEN all report `//.private memory size 5888`, 48 dpas, no spill, no retry dump;
instCount 6,448 / 7,418 / 8,395. Inside base's dpas span: 61 private loads, 8 private stores
(`ugm`, address space private) against 27 global loads, 46 SLM loads, 66 SLM stores, 4
barriers; whole kernel 101 private loads, 54 private stores. PRE's ~1,950 extra instructions
leave that private traffic unchanged.

**Every private array in `flash_attn_gqa_dpas` (line numbers at 0130h's tree), declared bytes,
and why IGC keeps it addressable.** Acquired private memory is per work-item; a work-group of
96-192 work-items therefore multiplies these.
| array | line | bytes | why it is materialized |
|---|---|---|---|
| `float8 o[NDH]` | :270 | 256 (NDH 8 = DV/2/16 at head 256) | indexed by `i` in the init, rescale, PV update and record loops; all four are loops with a register index |
| `short8 qa[KC]` | :518 | 256 (KC 16) | written in a `c < KC` loop, read in the DPAS `c < KC` loop, same |
| `float8 s[NKB]` | :599 | 64 | written per `g`, read in the S-split exchange, the mask add, the max reduction, the P build |
| `short8 pa[NKB]` | :699 | 32 | written per `g`, read in the PV `g` loop |
| `float m[8]`, `mnew[8]`, `corr[8]`, `mk[8]`, `mv[8]`, `larr[8]`, `ov[8]` | :266, :662, :689, :669, :685, :746, :756 | 32 each, 224 | every one is indexed by `r` in a `for r < 8` loop; `mk` also by the key group |
| `rh[8]`, `rt[8]` | :256 | 64 | index `r` loops, and `rt` feeds address arithmetic |
| `rv[8]` | :257 | 8 | `rv[r]` in the row loops |
| `mrow[8]` | :538 | 64 | `mrow[r]` in the mask add |
| `const half vv[64]` | stage macro | 128 an instance; base has two call sites | a 64-element store loop `for j < 64` the unroller never takes, so the components round-trip through private |
| `float x[32]`, `int q[32]` | :495, :506 | 256, I8 only, not this build | quantizer staging |
| `half16 pre[2][4]` | :557, PRE only | 256 | PRE's register prefetch, indexed by `it` |
Sum of the base build's declared arrays: ~1,096 B. The dump's 5,888 B is IGC's count after its
own materialization (per-unrolled-instance copies, addressable temporaries, per-work-item
stack); the per-array split above is arithmetic from the source (`inferred`), only an ocloc
A/B attributes exactly.

**Q1 split kernel (1-3 rows, `flash_attn_f32_f16.cl`'s `fdi_split_body`, the quantized K/V codes
KT/VT):** `o[FDI_E]`, `q[FDI_E]`, `s[FDI_KB]`, `kr[FDI_E]`, `vr[FDI_E]` (`FDI_E` = DK/16 = 16,
`FDI_KB` = 4) -- declared 64 floats of `o`, `q` and 4-key `s`, 16-float `kr`/`vr` -- and every
loop over them already carries `#pragma unroll`, so that body is not expected to acquire
private memory beyond the dump's own count; it is not what the production 4-7 row decode
runs. `code`; no separate ISA dump for it in this round.

**0130k (`code`).** `GGML_OPENCL_FA_GQA_NOPRIV=1`, default off, own programme: `FA_UNROLL`
(`_Pragma("unroll")`) on every loop that indexes one of the arrays above (all trip counts are
compile-time: 8, NDH, KC, NKB, 2 for PRE's `it`), and the V stage's `vv[64]` replaced by 64
component stores straight to local memory in both stage macros. The reduction order is
unchanged: every loop keeps its ascending order, `mx8` still starts at `s[0]`, `acc = DPAS(pa[g], b, acc)`
still runs g ascending, and no `mad`/`fma` reassociation is introduced. `GGML_OPENCL_FA_GQA_NOPRIV_BROKEN=1`
is a separate build whose NOPRIV path writes value 15 of every staged V row as zero: the red
op cases fail while the register shape stays. Expected ISA (`inferred`, the second red the
card host checks): `//.private memory size 0`, no `address space: private` messages in the
dpas span, instCount above base's 6,448 for the stores.

0130l (`code`): 0130k's directives sat inside the continued macro bodies and failed `cpp`
(`'#' is not followed by a macro parameter`, rc 1) and ocloc (rc 245, no ISA) for the base
kernel too -- the production decode. The alternatives are whole macro definitions under
file-level `#ifdef` and the broken zero is a value macro; `cpp -P` of the NOPRIV-off base is
identical to 25f3b7a5 (diff 0 lines). `tests/python/test_kernel_preproc.py` runs the
preprocessor over the kernel under every host define set (base, I8, ST, V2, PRE, PRE+BROKEN,
NOPRIV, NOPRIV+BROKEN, NOPRIV+PRE; K/V q8q4 and f16), red on b69f5bee and green on the fix,
and `build-deb.sh` runs the same matrix, so the recipe fails a kernel that does not
preprocess. PRE stays a `measured-here` negative (window 295, build f11639bf: 240k no-paging
223.3 against 279.8 ms/step, +25.3 %; 261k pool 3584 OVERLAP 262.1 against 308.4, +17.7 %;
unset 261k 453.2, so OVERLAP -42 %; op perf 245760 base 94.0-97.6, PRE 97.3-99.1, V2
95.7-97.4 ms/run); NOPRIV+PRE stays one optional arm, dropped if the ISA shows PRE's ~1,950
extra instructions remain with private 0. If the base programme fails to build, the decode
is not silent: `flash_attn_gqa_dpas.cl not built; flash_attn_f32_f16.cl takes decode` with
the downgrade record (`code`, 0130n).
