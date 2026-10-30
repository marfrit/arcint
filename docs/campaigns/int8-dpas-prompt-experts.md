# int8-dpas-prompt-experts — int8 DPAS for the prompt's grouped expert GEMM (MUL_MAT_ID)

**Open (2026-10-30).** Patch `0071` of `contrib/llama.cpp` is written and
device-free tested; the card window (compile gate, pricing, A/B, KL) is owed.
Started from `research-concept-mining/audit.md` §2.

## Charter

Give the prompt's routed-expert GEMM (`MUL_MAT_ID`) the reference's int8 DPAS
form instead of the f16 DPAS operand the K-quants are dequantized into today.
Target: Flash-Next (IQ3_S) on the B60 — gate/up IQ3_S 2,560 -> 640 (512
experts, top-10), down IQ4_NL 640 -> 2,560 — and the same grouped GEMM on the
coder's resident experts where it applies. The prize is bounded by the expert
GEMM's share of the prefill wall at 2x the multiply rate (audit §2: 1.67 s of
~5.1 s a chunk, `paper`, at ~1/10 of the card's matrix throughput).

## Reference to follow

Strata's fused int8 DPAS prompt experts for Battlemage, at source:
- `sycl/src/prefill/esimd_moe.dp.cpp` (`gate_up`, `down`, `quantize_rows`):
  weights decoded in registers to the s8 DPAS B operand, activations q8_1 per
  32 (scale `amax / 127`, round half away from zero), each 32-block's int32
  DPAS converted to float and scaled by (weight scale x activation scale),
  decode once a block applied to every row tile (`code`)
- `sycl/include/strata/prefill/esimd_moe.hpp`: `choose_mode` — unset is auto on
  a Battlemage device by architecture id (`bmg-g21` / `bmg-g31`), per group of
  experts, int8 while the group's mean rows per expert are under the measured
  crossover; `=0` fp16, `=1` forced (`code`)
- `docs/INTEL.md` (6aa8e25f, c765c5f8): a B70 IQ3_S pack, prompts 1,024 /
  4,095 tokens: 645.8 -> 783.5 t/s (+21 %) and 1,257.0 -> 1,407.7 (+12 %)
  (`paper`); teacher-forced KL against the FP16 path mean 2e-4 nats, p99
  6e-3, top-1 99.7 % (`paper`); the crossover 130-250 rows per expert, above
  it every row block decodes the weights again (`paper`).

arcint has the other half already (`code`, README 0006): the route sorts
(token, slot) pairs by expert into tiles on the device and one fp16-DPAS tile
per work-group runs per expert (`ggml-opencl.cpp`, `mul_mm_kq_f16.cl`
`kernel_mul_mm_id2_*` / `kernel_mul_mm_id_*`, the route cache). The int8 DPAS
primitive is also already used, on the dense Q4_K path (`code`, README 0013,
`mul_mm_kq_2d.cl` `kernel_mul_mm_kq2d_q4_K_i8`).

## Gate

- Answer-level (`CLAUDE.md`): mean KL against the fp16 arm on the same card
  no more than 0.03 nats worse, argmax agreement down by at most 1 point; the
  facts/needle/task battery right. Strata's own price is 2e-4 nats (`paper`),
  so the bar is expected to pass; a fail is the STOP.
- Speed, per phase: prefill improves at the measured token counts (2k, 8k,
  19k on fn_sweep against the fp16 arm and against the loan build), decode
  unchanged within the run-to-run spread. The crossover's red: forcing the
  dispatch threshold to 16 at a large prompt must regress the arm.
- Pricing before trusting the kernel: the expert GEMM's share of the prefill
  wall on today's build (a profiling build, 2,048- and 8,192-token graph
  timelines, `tools/cl_timeline_steps.py`, the `[MOE_TILES]` padding line);
  `test-backend-ops perf` MUL_MAT_ID at Flash-Next's shapes and types, fp16
  vs int8; correctness MUL_MAT_ID for those types against the CPU, red by the
  sub-block constant mutation.

## Current state

- Card round 8 (`measured-here`, a B60, windows 27 and 28, `cecdea0`'s build; no
  kernel change in this round): clang-18 on the assembled program at SG16 and
  SG8, 0 errors, and `tests/test-opencl-kq-i8` run with `CLANG` all OK; Intel's
  offline compiler for the B60 (`-device bmg`, the backend's options,
  `-DSG=16`) 0 errors. `test-backend-ops` `MUL_MAT_ID` at the served shapes:
  the INFO line names the formats (`int8 DPAS formats: iq3_s,iq4_nl` and
  `... iq3_s`); forced int8 12/12, the red (both 32-block weight scales
  `1.f`) 2/12, int8 IQ3_S only 12/12; with the eval list widened to n = 256,
  512, 2048 (a diagnostic edit of the test tree, not in the patch), fp16
  12/12 and int8 12/12, no multi-tile defect inside the NMSE bar of 5e-4.
- KL (`measured-here`, the B60, `llama-perplexity`: one 8,192-token window,
  `-ub 2048`, `--n-cpu-moe 48` with the expert cache, the device pinned; this
  window's fp16 twins were bit-identical to the reference: KL 0, argmax
  1.000): the red arm PPL 429,684, KL +11.49; int8 IQ3_S gate/up only +0.0059
  / +0.0039 nats, argmax -2.95 / -2.47 points; IQ4_NL down only +0.0117,
  -4.18 points; both formats +0.0098, -4.27 points. The KL bar (0.03) passes
  every arm; the argmax bar (1 point) fails every arm. Window 24's -1.07
  points was measured against twins at 0.9692, a noisy floor that day.
- Served Flash-Next (`measured-here`, the B60, `tools/fn_sweep.py`, fp16 vs
  int8 IQ3_S only, two rounds reversed): 8k with the loan 463.8/478.7 vs
  475.6/469.7 t/s, 19k 513.0/483.6 vs 511.2/512.1, 8k without the loan
  444.1/444.8 vs 444.1/437.2, decode 34.13/34.43 vs 34.03/34.17 — no gain
  beyond the spread; the capital right in every arm.
- Status (`measured-here`): the arm is opt-in and off by default, not adopted
  on any unit; the argmax bar fails every arm, and the served gain is +1.2 to
  +3.2 % with both formats and none with gate/up only. Owed: the expert GEMM's
  share of the prefill (the profiling step, never run), which bounds what any
  kernel here can give.
- Round 8 test fixes (`code`): `opencl_part`'s inverted `#else` ternary
  (`st.empty()` must read `skip`, not an empty stack) and a unit case for its
  nesting plus a top-level `#else` (red: the ternary inverted -> segfault;
  restored all OK).
- Strata's own quality figure (`paper`, Strata-ref
  `docs/INTEL_INT8_EXPERTS.md:83-88`): KL 0.0002, top-1 99.7 % against its own
  FP16 path (dequantize + oneMKL), the same per-32 q8_1 activations,
  teacher-forced on 6,002 positions of its FP16 path's greedy continuations
  over 28 prompts up to 3.2K tokens. Ours is one 8K corpus window through the
  expert cache, a different method; the comparison is owed (the same method on
  both sides).
- Round 7 review (`measured-here`, clang-18 `-target spir64 -x cl -cl-std=CL3.0
  -DSG=16` on the backend's assembly `.inc` + `mul_mm_kq_f16.cl` +
  `mul_mm_kq_i8.cl`): 4 errors "use of undeclared identifier 'kq_i8_col'".
  `kq_i8_col` is device code (called twice by `I8_STAGE`), but round 6 put it
  inside the host-only `#ifdef __cplusplus` block; `-DSG=8` compiled only
  because the int8 kernels are under `#if SG == 16`. Fixed (`code`; the compile was owed
  then, now `measured-here` in round 8: clang-18 SG16/SG8 and the
  B60 ocloc both 0 errors): `kq_i8_col` and the index helpers sit above the
  guard, and the
  device-free text check requires every `kq_i8_*` identifier the `.cl` kernels
  call to be defined in the `.inc`'s OpenCL part (red: the function moved back
  into the host block -> `FAIL opencl: the kernel calls kq_i8_col, not defined
  in the .inc's OpenCL part`). Also (`code`): `opencl_part` counts nesting, so
  a conditional inside the host block cannot end the skip early; the parser
  treats `\n` and `\r` as whitespace, compares case-insensitively, and a value
  with no token (`,`, ` , , `) is both; one INFO line at the dispatch's first
  decision names the parsed formats; the duplicate `const` on `qq` in the
  shared `mul_mm_kq_f16.cl` text is dropped.
- Round 6 review (`measured-here`, two compilers): the per-format parser's
  string literals sat in the `.inc` that is prepended to the OpenCL program, so
  a literal is `__constant char *` and the int8 program no longer built:
  Intel's offline compiler for the B60 on the exact assembly (`.inc` +
  `mul_mm_kq_f16.cl` + `mul_mm_kq_i8.cl`, `-DSG=16 -cl-std=CL3.0`) reported
  "changes address space of pointer" and failed with -11, clang-18 failed the
  same at CL1.2/2.0/3.0, and the test with a clang present printed the
  assembled-program FAIL at SG16 and SG8 (exit 1). The backend then fell back
  silently (int8 kernels null), so an int8 arm measured the fp16 tile --
  round 3's `half` shape. Fixed (`code`): the `.inc`'s host-only block (the
  dispatch rule, the format list, the launch sizing) is behind
  `#ifdef __cplusplus`, which the device never sees and the test's
  `opencl_part` strips; the device-free text check now rejects a string
  literal in the `.inc`'s OpenCL part (red: a literal added there -> `FAIL
  opencl: a string literal in the .inc's OpenCL part`). The fix is `code`;
  its clang and card compiler halves are `measured-here` in round 8 (0
  errors).
- Card round 4 (`measured-here`, a B60, one announced window, the commit
  `60cf5a6`): `test-backend-ops` on the committed text, forced int8 12/12 at
  the served shapes, the red (both 32-block weight scales `ws = 1.f`) 2/12; the
  int8 perf, us/run: IQ3_S m=640 k=2560 n=16 1,125.75, IQ4_NL m=2560 k=640
  n=16 912.03, IQ3_S n=512 4,156.28, IQ4_NL n=512 3,826.86 (the fp16 floor is
  round 3's column). Served Flash-Next IQ3_S (`tools/fn_sweep.py`, the unit's
  flags, the switch 0 vs 1, two rounds with the second in reversed order),
  prompt t/s fp16 -> int8: with the slot loan 2k 229.6/250.6 -> 252.7/245.4
  (inside the spread), 8k 470.4/476.6 -> 474.0/484.5 (+1.2 %), 19k
  509.4/514.5 -> 524.1/521.0 (+2.1 %); without the loan 8k 442.6/442.3 ->
  456.7/456.2 (+3.2 %); decode 32.60/34.18 vs 33.35/34.08 (equal); the
  capital right in every arm; 5 of 6 answer md5s differ between the arms (a
  byte tripwire, not the gate).
- KL (`measured-here`, the B60, `llama-perplexity`'s form: one 8,192-token
  window, `-ub 2048`, `--n-cpu-moe 48` with the expert cache, the device
  pinned): the red arm PPL 435,809, mean KL +11.5 (the int8 path is exercised);
  int8 twice, mean KL +0.0059 / +0.0054 nats over the fp16 twins' floor 0.0064
  (the 0.03 bar passes); argmax agreement 0.9585 both against the twins'
  0.9692, -1.07 points, the 1-point bar fails by 0.07. Strata's activation
  quantization is the same form (per 32 values, `amax/127`, SwiGLU output to
  int8 for the down; `code`, `sycl/src/prefill/moe_fused_iq.dp.cpp`); its own
  quality figure (top-1 99.7 %) is `paper`.
- The round-4 separation lever (`code`, round 5): the per-format switch
  `GGML_OPENCL_KQ_MM_ID_I8_FMT` (a comma or whitespace separated list — `\n`
  and `\r` too — of `iq3_s` and `iq4_nl`, case-insensitive; unset, empty,
  whitespace-only or a value with no token = both) so the next
  window can price IQ3_S-only and IQ4_NL-only against the same reference and
  fp16 twins, and a served IQ3_S-only arm; the layer whose gate/up take int8
  while its down takes fp16 is a per-call route (the cache key carries
  `rc.i8`), so a reroute is correct. The reroute costs one extra
  `moe_route_tiles` a layer although both tiles have the same geometry; not
  measured (`code`).
- Card round 3 (`measured-here`, a B60, one announced window, the tree at the
  pin with the branch's patches, `GGML_OPENCL_DEVICES=0:0`, `test-backend-ops`
  built from that tree): Intel's offline compiler (`ocloc 26.27 -device bmg`,
  the backend's options, `-DSG=16`) compiles the assembled int8 program with
  0 errors; with a `half` variable restored it reports 11 errors, the round-3
  blocked build. Forced int8 (`GGML_OPENCL_KQ_MM_ID_I8=1`) on the commit as it
  stood: 2 of 12 `MUL_MAT_ID` cases pass (the two matvec cases, n=1 and n=5),
  every GEMM case fails (IQ3_S NaN, IQ4_NL NMSE 1.90); the fp16 arm (the
  switch `=0`) passes 12/12.
- The defect, named (`code`, then `measured-here`): `mul_mm_kq_i8.cl`'s block
  loop counts `kb` in ELEMENTS of K (`k0 += 256`, `kb += 32`), but the IQ3_S
  kernel read `sb = kb / 8, p = kb & 7` and the IQ4_NL kernel
  `blk = (wrow0 + rr) * n32 + kb` as if `kb` were a 32-block index. The
  weight reads landed far beyond the block, and on the last experts out of
  bounds (a read that can fault the card's engines). The fix is the fp16 tile
  kernel's own index (`LB_BI` / `LB_P`: sub-block `j32 = kb / 32`): IQ3_S
  `sb = kb / 256, p = (kb / 32) & 7`, IQ4_NL `blk = row * n32 + kb / 32`, now
  in `kq_i8_ops.inc` helpers the kernels call. With it, forced int8 passes
  12/12; the red (both 32-block weight scales forced to `ws = 1.f` on top of
  the fix) fails 2/12, IQ3_S NMSE 0.86 and IQ4_NL NMSE 1.00, and no engine
  reset occurs.
- Perf (`measured-here`, `test-backend-ops perf`, us/run, fp16 arm -> int8
  arm): IQ3_S m=640 k=2560 n=16 2,259.39 -> 1,142.74; IQ4_NL m=2560 k=640
  n=16 1,199.10 -> 929.77; IQ3_S n=512 8,482.40 -> 4,154.80; IQ4_NL n=512
  5,035.29 -> 3,825.53. The served A/B, the expert GEMM's share and the KL arm
  remain owed.
- Built (`code`) for the arm on (`GGML_OPENCL_KQ_MM_ID_I8` set non-empty and
  not 0, `kq_i8_program_wanted`): `0071` adds the int8 MUL_MAT_ID tile (IQ3_S
  gate/up, IQ4_NL down) and its activation quantizer as their own program
  (`kernels/mul_mm_kq_i8.cl`, `build_program_from_source_optional` after the
  XMX program), so a failure there costs the int8 arm only; dispatched from
  `ggml_cl_mul_mat_id_kq` under the existing route cache with `tile = 32`, a
  per-format fp16 fallback, the `GGML_OPENCL_KQ_MM_ID_I8` switch, and `auto`
  by mean routed pairs per expert with the crossover
  `GGML_OPENCL_KQ_MM_ID_I8_CROSS` (default 128 from Strata's 130-250 on a B70,
  `paper`; the B60 value is owed), and the per-format list
  `GGML_OPENCL_KQ_MM_ID_I8_FMT` (unset = both, a format not named takes the
  fp16 tile). Unset is `0` (off) in this patch; `1` and `auto` are the opt-in,
  and a later commit flips the default after recording the B60's KL/task/speed. With the switch off (the default) the int8 sources
  are not read and the program is not compiled (`code`, `measured-here` by the
  gate test), so a unit start pays no second program compile.
- The int8 activation has its own buffers (`prealloc_kq_id8_{q,d}`) with its
  own dedup entry, so the q8_1 matvec's `prealloc_kq_{xq,xd}` and `kq_q8` key
  are untouched. They are [column][K] codes and [column][K / 32] scales, the
  fp16 path's activation size (`n_cols = ne11 * ne12`): at an 8k chunk and
  K 2560, 21 MB of codes + 2.6 MB of scales against 210 MB + 26 MB for
  [pair][K] (`n_used` 10x; `measured-here`, byte arithmetic). The quantizer
  reads columns, so the route needs no -1 fill and a recomputed route does not
  clear the activation. A layer's gate and up share one quantization of src1.
  The activation is staged a 256-weight super-block a barrier pair (the last
  group takes the remaining blocks), each staged row the pair's activation
  column `kq_i8_col`; the quantizer runs 64-wide groups; IQ3_S's qs and signs
  are read as aligned global `uint` words (`code`).
- Device-free (`measured-here`): `tests/test-opencl-kq-i8.cpp` — the s8 decode
  of IQ3_S and IQ4_NL against ggml's `dequantize_row_*`, value for value; the
  32-value quantizer (scale, round half away from zero, all-zero block); the
  dispatch rule (unset = off in this patch, env 0/1/auto, SG 16 with the
  kernels and SG 8 -> fp16, crossover boundary); the per-format list (unset =
  both, whitespace-only/`\n`/`\r`/separators-only = both, case-insensitive,
  `iq3_s` only -> IQ4_NL fp16 and the reverse, commas and whitespace as
  separators); no string literal in the `.inc`'s OpenCL part (the host-only
  block is stripped), every `kq_i8_*` the `.cl` kernels call defined in that
  part; the
  block indices of an element offset `kb` against the fp16 path's `j32 = kb / 32`, `LB_BI` /
  `LB_P` for every `kb` of K 640 and 2560; the program gate (unset/empty/0/word
  off, 1/auto built); the activation's column `kq_i8_col` and the served 8k
  sizes (21 / 210 / 42 MB); and the ASSEMBLED int8 program as
  `load_cl_kernels` builds it (.inc, then `mul_mm_kq_f16.cl`, then
  `mul_mm_kq_i8.cl`, `-DSG=16`): no C++-only token, no reserved OpenCL type
  name used as an identifier, every overloaded prelude declaration carrying
  `__attribute__((overloadable))`, and, when clang/clang-18/clang-19/`$CLANG`
  is found, a compile at CL3.0, `-target spir64`, SG 16 and 8 with a prelude
  for the Intel builtins the clang header lacks; when none is found the
  compile half is skipped and the text checks are the pre-run. Each has a
  one-line mutation seen failing (README 0071 lists them); the `half` variable
  in `I8_STAGE` (round 3's blocker, caught by the reserved-name check ->
  `FAIL ... reserved type name half`), the `nullptr` red, the per-pair column
  (-> `FAIL col: slot 9 ...`), `return 1` for the program gate, the pre-fix
  element-offset index (-> `FAIL index: IQ3_S super-block of kb`), the
  sub-group guard removed (-> `FAIL dispatch: switch on, SG 8 -> fp16`), the format list's
  unlisted fall-through (-> `FAIL fmt: iq3_s only -> IQ4_NL fp16`) and the
  reversed lookup (-> `FAIL fmt: iq4_nl only -> IQ3_S fp16`), the
  whitespace-only value (-> `FAIL fmt: whitespace-only -> both`), whitespace
  as a separator (-> `FAIL fmt: whitespace is a separator too`), `\n`/`\r`
  dropped (-> `FAIL fmt: a newline-only value -> both`), the case fold dropped
  (-> `FAIL fmt: names are case-insensitive`), a no-token value no longer
  both (-> `FAIL fmt: a separators-only value -> both`), a literal back in the
  `.inc`'s OpenCL part (-> `FAIL opencl: a string literal in the .inc's
  OpenCL part`), `kq_i8_col` back in the host block (-> `FAIL opencl: the
  kernel calls kq_i8_col, not defined`), and the missing
  `overloadable` on one overload (-> `FAIL opencl: overloaded prelude
  declaration ...`).
- The assembled int8 program compiles with Intel's offline compiler for the
  B60 (`measured-here`, round 3: 0 errors); the device-free assembled-text
  check is its pre-run, and a clang compile runs when one is found, otherwise
  that half is skipped.
- Not measured: the expert GEMM's share and the B60 crossover. The compiler
  halves, the `test-backend-ops` correctness (including the widened diagnostic
  sizes), the KL arms per format and the IQ3_S-only served A/B are measured
  (above).

## Where it lives

- `contrib/llama.cpp/patches/0071-opencl-intel-moe-int8-dpas.patch`
  (independent of 0069/0070), section `0071` in `contrib/llama.cpp/README.md`.
- `ggml/src/ggml-opencl/kernels/mul_mm_kq_i8.cl` (the two i8 kernels and
  `kernel_kq_mm_id_quant_i8`, its own program), `kernels/kq_i8_ops.inc`
  (decode, quantize, dispatch, the OpenCL text check — one text for kernel
  and test), `ggml-opencl.cpp` (dispatch, args, launch, the separate program
  build), `ggml-opencl/CMakeLists.txt` (the embed step and the non-embedded
  copy), `tests/test-opencl-kq-i8.cpp` and `tests/CMakeLists.txt`.
- Design: no DESIGN section changes until the card gate decides; the switch
  and the crossover are in the README section.

## Card window

One announced window, one non-service process at a time, sampler and PCI-id
identity first (`docs/sop-card-window.md`). The tree is the pin with
`contrib/llama.cpp/patches/*.patch` (0001-0071); `ARCINT_LLAMA_DIR` points at
it for an arcint build, and `test-backend-ops` / `llama-perplexity` are built
from the same tree. Placeholders: `<arcint>` the served binary, `<gguf>` the
Flash-Next IQ3_S GGUF, `<assets>` the unit's assets; `<tree>` is the pin with
the branch's patches, and `$PERPLEXITY` / `$LOAN_KLD` are its
`build-tools/bin` tools (`contrib/llama-loan-kld/build-tools.sh`).

1. **Compile gate, before any timing.** First the assembled int8 program as the
   backend builds it, with Intel ocloc on the card host (no prelude: Intel's
   compiler has the subgroup builtins): `cat kernels/kq_i8_ops.inc
   kernels/mul_mm_kq_f16.cl kernels/mul_mm_kq_i8.cl > <tree>/kq_i8_prog.cl`,
   then `ocloc compile -device bmg -file <tree>/kq_i8_prog.cl -o
   <tree>/kq_i8_prog.bin -options "-cl-std=CL3.0 -cl-mad-enable
   -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math
   -cl-intel-256-GRF-per-thread -DSG=16"`; a nonzero exit or an error line is
   the STOP. Then a fresh build of `test-backend-ops`
   from the tree (`-DLLAMA_BUILD_TESTS=ON -DGGML_OPENCL=ON
   -DGGML_OPENCL_EMBED_KERNELS=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF
   -DGGML_NATIVE=ON`), run with the int8 arm forced on
   (`GGML_OPENCL_KQ_MM_ID_I8=1`, the device pinned by
   `GGML_OPENCL_DEVICES=0:0`): the log must show the XMX program built
   (MUL_MAT and MUL_MAT_ID) and the int8 DPAS expert GEMM built, and no
   "not built" line. Then the red: break one decode constant (the 32-block
   weight scale `ws = 1.f;` for the IQ3_S `d * (1 + 2 sc)` and the IQ4_NL
   `dp[blk]`), rebuild, and the forced-int8 correctness run must FAIL;
   revert, rebuild.
2. **Pricing (a): the expert GEMM's share.** A profiling run with the OpenCL
   intercept layer's chrome trace (`CLI_ChromePerformanceTiming=1`,
   `CLI_ChromePerformanceTimingOutput=<trace>`), one 2,048-token and one
   8,192-token prefill (`--prompt-tokens`, no answers), `[MOE_TILES]` on
   (`GGML_OPENCL_MOE_TILES_DEBUG=1`); `tools/cl_timeline_steps.py <trace>
   --prefill --top 25` names the expert GEMMs' device time against the
   prefill wall, and the `[MOE_TILES]` line bounds the tile padding waste.
3. **Pricing (b): the floor and the lemma.** `test-backend-ops perf -o
   MUL_MAT_ID -p 'type_a=iq3_s.*n_mats=512'` (and `type_a=iq4_nl.*n_mats=512`)
   with `GGML_OPENCL_KQ_MM_ID_I8=0` and `=1`, the device pinned
   (`GGML_OPENCL_DEVICES=0:0`). The fp16 rates are the floor the int8 form
   must beat; the same shapes appear with `n=16` and `n=512`.
4. **Pricing (c): correctness.** `test-backend-ops -o MUL_MAT_ID -p
   'type_a=iq3_s.*n_mats=512'` against the CPU, fp16 and int8 arms, with the
   red from step 1 seen failing.
5. **Pricing (d): served A/B and KL.** `tools/fn_sweep.py` per arm at
   `--prompt-tokens 2048 / 8192 / 19000`, fp16 and int8, without and with
   `--llama-slot-loan 6400 --llama-loan-chunk 8192`; prefill t/s and the
   `[MOE_TILES]` lines from the log, decode unchanged per arm. KL: the
   loan-kld driver's proven form (`--n-cpu-moe 48`, `--device GPU.0`,
   `GGML_OPENCL_DEVICES=0:0`, one 8,192-token window, `--rows-per-decode
   2048`), fp16 vs int8 against one reference capture, compared with
   `tools/capture_vs_capture.py`; the bar is the answer-level one.
6. **The crossover.** The same perf arms at n=16, 64, 128, 256, 512 and a
   served prefill at 4,096 / 8,192; the mean routed pairs per expert is
   `min(n_expert, pairs)` at each size. Set `GGML_OPENCL_KQ_MM_ID_I8_CROSS` to
   the measured crossover and record it here with its class.
7. **Round 5: per format.** The reference and the fp16 twins are the round-4
   captures (`off-ref.kld`, `off-a.kld`, `off-b.kld`, switch `=0`, no format
   list); reuse them if the corpus and window are the same. The int8 arms are
   `llama-perplexity`'s form: one 8,192-token window, `-ub 2048`,
   `--n-cpu-moe 48`, the device pinned. Built from the same tree,
   `contrib/llama-loan-kld/build-tools.sh <tree>` gives
   `$PERPLEXITY` = `<tree>/build-tools/bin/llama-perplexity`. The IQ3_S-only
   arm twice (a run-to-run pair) and the IQ4_NL-only arm once:

       KQ8="GGML_OPENCL_KQ_MM_ID_I8=1 GGML_OPENCL_KQ_MM_ID_I8_FMT=iq3_s"
       KQ4="GGML_OPENCL_KQ_MM_ID_I8=1 GGML_OPENCL_KQ_MM_ID_I8_FMT=iq4_nl"
       CACHE="GGML_OPENCL_BANK_STREAM=1 LLAMA_EXPERT_CACHE_CPU_MISSES=1 \
         LLAMA_EXPERT_CACHE_MIB=11600 LLAMA_EXPERT_CACHE_PROFILE=<assets>/expert-profile.bin \
         LLAMA_SLOT_LOAN_MIB=0 GGML_OPENCL_DEVICES=0:0"
       for t in a b; do
         env $CACHE $KQ8 $PERPLEXITY -m <flash-next-iq3_s-00001-of-00002>.gguf \
           --device GPU.0 --n-cpu-moe 48 -f <assets>/corpus.txt \
           -c 8192 -b 4096 -ub 2048 --chunks 1 -t 8 -ngl 99 -lm mmap -nr \
           --save-all-logits i8-iq3s-$t.kld
       done
       env $CACHE $KQ4 $PERPLEXITY -m <flash-next-iq3_s-00001-of-00002>.gguf \
         --device GPU.0 --n-cpu-moe 48 -f <assets>/corpus.txt \
         -c 8192 -b 4096 -ub 2048 --chunks 1 -t 8 -ngl 99 -lm mmap -nr \
         --save-all-logits i8-iq4nl.kld

   The comparisons, one line each, against the same twins:

       python3 tools/capture_vs_capture.py off-ref.kld --baseline off-a.kld off-b.kld \
         --candidate i8-iq3s-a.kld --window 0
       python3 tools/capture_vs_capture.py off-ref.kld --baseline off-a.kld off-b.kld \
         --candidate i8-iq3s-b.kld --window 0
       python3 tools/capture_vs_capture.py off-ref.kld --baseline off-a.kld off-b.kld \
         --candidate i8-iq4nl.kld --window 0

   Served A/B at 8k and 19k with the loan and 8k without, fp16 against
   IQ3_S-only, two rounds reversed. `tools/fn_sweep.py`'s defaults are the
   round-4 form; `$LOAN` is the loan's flags:

       FN="python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets>"
       LOAN="--set --prefill-chunk 8192 --set --llama-loan-chunk 8192 --set --llama-slot-loan 6400"
       R16="--env GGML_OPENCL_KQ_MM_ID_I8=0"
       R8S="--env GGML_OPENCL_KQ_MM_ID_I8=1 --env GGML_OPENCL_KQ_MM_ID_I8_FMT=iq3_s"
       NOLOAN=""

       GGML_OPENCL_DEVICES=0:0 $FN r1-8k-loan-fp16  --prompt-tokens 8192  $R16   $LOAN
       GGML_OPENCL_DEVICES=0:0 $FN r1-8k-loan-iq3s  --prompt-tokens 8192  $R8S   $LOAN
       GGML_OPENCL_DEVICES=0:0 $FN r2-8k-loan-iq3s  --prompt-tokens 8192  $R8S   $LOAN
       GGML_OPENCL_DEVICES=0:0 $FN r2-8k-loan-fp16  --prompt-tokens 8192  $R16   $LOAN
       GGML_OPENCL_DEVICES=0:0 $FN r1-19k-loan-fp16 --prompt-tokens 19000 $R16   $LOAN
       GGML_OPENCL_DEVICES=0:0 $FN r1-19k-loan-iq3s --prompt-tokens 19000 $R8S   $LOAN
       GGML_OPENCL_DEVICES=0:0 $FN r2-19k-loan-iq3s --prompt-tokens 19000 $R8S   $LOAN
       GGML_OPENCL_DEVICES=0:0 $FN r2-19k-loan-fp16 --prompt-tokens 19000 $R16   $LOAN
       GGML_OPENCL_DEVICES=0:0 $FN r1-8k-fp16      --prompt-tokens 8192  $R16   $NOLOAN
       GGML_OPENCL_DEVICES=0:0 $FN r1-8k-iq3s      --prompt-tokens 8192  $R8S   $NOLOAN
       GGML_OPENCL_DEVICES=0:0 $FN r2-8k-iq3s      --prompt-tokens 8192  $R8S   $NOLOAN
       GGML_OPENCL_DEVICES=0:0 $FN r2-8k-fp16      --prompt-tokens 8192  $R16   $NOLOAN

   Record the prompt t/s, decode t/s, the capital and the answer md5s per tag,
   and the KL/argmax per capture against the twins; the bar is the
   answer-level one.
