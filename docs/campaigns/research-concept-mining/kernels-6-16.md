# concept-mining 6 and 16, kernel side — built, card arms owed

State: audit items 16 (decode launch fusions: the q/k norm + rope fusion) and 6
(CPU miss compute: the row prefetch and the cold-page prefetch) are built as
patches **0074** and **0075**, both behind a switch that defaults **off**, with
device-free red cells. No card run was made here: the pricing measurements the
audit asks for, the served A/B and the test-backend-ops numerics are owed and
are Claude's in an announced window. Every number in this document is either a
`measured-here` device-free result or a `paper` figure quoted from the audit; no
card figure is claimed as new.

Precedent read first (`code`): `audit.md` §§6 and 16, their cited patch series
(0004/0027/0049/0050/0051/0064/0065/0066 for 16, 0056/0063 for 6), the two
models' attention builders (`src/models/qwen35.cpp:309-321`,
`qwen4exp.cpp:994-1008`) and the miss path's rows
(`ggml-opencl-miss-cpu.cpp`). Both models build `rms_norm -> mul(1-D
[head_dim] weight) -> rope_multi`, and their rope mode is
`GGML_ROPE_TYPE_IMROPE` (40) in `llama_model_rope_type` (`code`).

## 16 — the q/k norm + rope fusion (patch 0074)

**Mechanism** (`code`). `kernel_rms_norm_mul_rope_multi_f32` in
`kernels/rms_norm.cl`: one work-group a row, the norm's float4 sum with the
epsilon, the weight applied on the rounded normed value (the 0049 opaque guard,
so the compiler cannot regroup `x * (scale * w)`), then `rope_yarn`'s cos/sin and
the interleaved M-RoPE position layout — `kernel_rms_norm_mul`'s and
`kernel_rope_multi_f32`'s own expressions. The normed row is no longer stored and
reloaded. Dispatch: the 3-node pattern in `ggml_opencl_can_fuse`, checked before
the 2-node norm+mul prefix, mode IMROPE, a 1-D weight, a 4-aligned contiguous
input row and a contiguous f32 output; two hardenings beyond the first cut
(`code`): `rope->src[0] == mul` is required explicitly (`ggml_can_fuse` also
accepts the mul as rope's `src[1]`, not this kernel's shape), and the sections'
sum must be positive — the per-op path asserts it, and without the guard the
kernel's `ic % sect_dims` divides by zero on the GPU, silently. Switch
`GGML_OPENCL_FUSE_QK_NORM_ROPE=1`, default off.

**Deviation, named** (`code`): the audit's sketch names "the qk norm+rope OpenCL
kernel next to `rms_norm.cl`/`rope.cl`, dispatched in the attention builders".
The dispatch is not in `qwen35.cpp`/`qwen4exp.cpp`; it is the backend's pattern
fuse, which is how this series already fuses (`code`: 0049, 0064, 0065). The
builders stay untouched, so a model other than the two served ones changes
nothing.

**Quality class** (`code`): the audit's `rounding-only`. The fused kernel
rounds the normed value in two steps — `(x * scale)` then `* w`, kept apart by
the 0049 `opaque` guard — but `kernel_rms_norm_mul` has no such guard (its plain
`(x * scale) * f[i00]`, `rms_norm.cl`), so it may fold `x * (scale * w)` itself
and the two paths' bits are **not guaranteed to agree either way**. The gate is
the answer-level bar (`CLAUDE.md`) with an md5/argmax tripwire per fusion on the
card; `test-backend-ops`' NMSE cannot see the rounding-order mutation, because
NMSE is a tolerance on the composed output, not the intermediate rounding.

**Pricing measurement — owed, the audit's own:**
- per-step share of the q/k norm + rope launches on the coder (A770) and
  Flash-Next (B60), from a profiling trace or `LLAMA_DECODE_TIMING` (`code`, the
  audit's wording: `tools/cl_timeline_steps.py` on the intercept trace gives the
  per-kernel count and time; the ceiling is the kernels removed plus their
  launch time at the wall, the A770's 5.9 µs an enqueue, `paper`).
- bar (`paper`, the audit): ≥ +0.5 % decode for a bundle of 2-3 fusions, else
  no build — and the switch stays off until the A/B clears it.

**Device-free cells** (`measured-here`): `tests/python/test_fuse_qk_norm_rope.py`
reads the patched tree — the epsilon in the scale, the weight at both rotated
positions, the norm-then-weight rounding order, the rotation's formula, the
M-RoPE position layout, the switch's default and the 3-before-2 dispatch order,
the kernel's creation, the operand-order guard and the sections guard, and the
new `test-backend-ops` cases. Seven mutations were run and seen failing before
restore: epsilon dropped, weight read at `w[ic]`, weight folded before the
rounding, default flipped on, dispatch moved below the prefix, sections guard
dropped, operand-order guard dropped; the kernel's absence fails five cells.
**These cells are source-text assertions** — they read the patched tree, they do
not run the kernel; the numerics are the card's. `code`: the patch also adds the
served shapes to `test-backend-ops`'s `RMS_NORM_MUL_ROPE` (the strided Q layout
`[128,16,1]` and `[128,16,5]`, contiguous K `[128,8,6]`), which ran green on the
CPU backend here (`measured-here`: 227 cases OK, the two strided ones included)
— a shapes/graph check, not the kernel's numerics.

## 6 — the CPU miss rows' prefetch and cold-page touch (patch 0075)

**The premise, checked against the code — and it is not Strata's** (`code`). The
first cut wrote the audit's cold-read premise ("so the pages may be absent").
The code says otherwise: a bank here is USM host memory (`clHostMemAllocINTEL`,
the `usm_host_alloc` entry points in `ggml-opencl.cpp`, and the buffer
context's comment), and every byte is written at load by
`ggml_cl_usm_set_tensor` — the planar loop, the IQ4_NL split copy — so its pages
are resident before the first miss; `drop_caches` evicts page cache, not this
memory. Strata's 78 % cold-read figure (`paper`, St 9e3fe046) comes from
file-backed rows: a materially different build. **Prediction** (`inferred`):
major faults near 0, and the level-2 pass buys nothing by construction. The
window's fault count decides; no number is claimed here.

**Mechanism** (`code`). One block-ahead `__builtin_prefetch` in each of the six
row functions, and a page touch over a missed expert's three bank slices at
publish (`ggml_cl_miss_cpu_touch`, a byte a 4 KiB page through a volatile
pointer), run as the pool's tasks. Switch `GGML_OPENCL_CPU_MISS_PREFETCH`: 0 off
(default), 1 the hints, 2 also the touch.

**The hints are narrow, and said so** (`code`): each names only the next block's
field 1 (the payload plane), two blocks ahead — 64-128 bytes, inside the
hardware prefetcher's reach; fields 0 (the scale) and 2 (the signs/qh plane) are
unhinted. A hint's gain, if any, is expected in the run's spread, not a step.

**The touch is a synchronous barrier on the lane's critical path** (`code`):
`ggml_cl_miss_pool::run` claims tasks and spins until `done == n`, so the
calling (coordinator) thread waits; "parallel" is across experts and the pool's
workers only, not overlapped with the card's round. The slow path calls it once
a pair, so a multiply-routed expert's slice is touched once a pair — redundant,
kept for the one pass's simplicity (`code`).

**Deviation, named** (`code`): the sketch's hint is "ahead of the next expert's
first field"; the build hints the next block within the same expert — a rows'
task sees one expert only. The cold-page half is built as sketched.

**Quality class** (`code`): exact. A prefetch cannot change a row's bits (the C
cell asserts level-1 parity bit-identical to level 0), and the touch reads bytes
it does not pass on.

**Pricing measurement — owed, the audit's own, on the window's terms:**
- the miss rows' time per verify round: the cap switch's ceiling (`paper`,
  arcint stage 3d: 49.2/50.0 t/s with the CPU computing nothing against
  49.8/50.6 with the ceiling switch — the wait costs nothing) and
  `LLAMA_DECODE_TIMING`'s per-decode split for the round; the prefetch's
  possible saving is the cold/warm gap of the same six answers at
  `GGML_OPENCL_CPU_MISS_MAX=64`.
- the faults: `/proc/<pid>/stat`'s min_flt/maj_flt for the served process around
  one decode (`code`: the leg host has no `perf`; the campaign's `perf` method
  is `paper` and is replaced here by the counter file).
- bar (`paper`, the audit): a prefetch only if major faults > ~5 % of the CPU
  lane's time. On the prediction above, that is ~0 and level 2 stays off.

**Device-free cells** (`measured-here`): `test-opencl-miss-cpu` gains the
level-1 parity arm ("0 of 2 rows differ"), the ahead-distance arm and the
mincore residency arm ("0 of 16 pages resident before, 16 after") — the last one
tests the touch's mechanism on a synthetic mapping, not this premise; mutations
seen red: the touch stride 8192 (8 of 16 resident), the ahead distance 0.
`tests/python/test_cpu_miss_prefetch.py` reads the tree for the default, one
hint in every row function, the volatile touch and the `>= 2` guard in both
compute paths; six mutations (a hint dropped from `row_iq3s_1`, distance 0,
stride, non-volatile pointer, default on, guard removed) each seen failing.
**Both cells are source-text assertions**, not numerics.

## The card commands (Claude runs these in the announced window)

One lane, `fn_sweep.py`'s unit flags; the cards by PCI id per
`docs/sop-card-window.md` (the sampler before the leg, the zombie sweep by pid).
`$ARCINT`, `$MODEL`, `$ASSETS` are the unit's own; `$WP_TMP` is the leg's output
directory (the house rule: under the work partition's wp-tmp, not `/tmp`).

```bash
WP_TMP=${WP_TMP:?the leg's wp-tmp output directory}

# 16 (1): the per-step share of the q/k norm + rope launches, one answer a unit
CLI_ChromePerformanceTiming=1 CLI_ChromePerformanceTimingOutput="$WP_TMP/qk-coder.json" \
  python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" qk-coder --unit coder --device GPU.N
CLI_ChromePerformanceTiming=1 CLI_ChromePerformanceTimingOutput="$WP_TMP/qk-fn.json" \
  python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" qk-fn --unit flashnext --device GPU.0
python3 tools/cl_timeline_steps.py "$WP_TMP/qk-coder.json" --steps 64 --tail 40 --top 25
python3 tools/cl_timeline_steps.py "$WP_TMP/qk-fn.json"    --steps 64 --tail 40 --top 25
#   read per steady step: kernel_rope_multi_f32's count and device time, plus the
#   same count of kernel_rms_norm_mul (each rope launch has one); the ceiling is
#   their sum + those launches at the A770's 5.9 us an enqueue. Host side:
LLAMA_DECODE_TIMING=1 python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" qk-time --unit coder --device GPU.N

# 16 (2): the numerics and the red, then the served A/B at the same flags
GGML_OPENCL_DEVICES=1:0 GGML_OPENCL_FUSE_QK_NORM_ROPE=0 ./bin/test-backend-ops -b GPUOpenCL -o RMS_NORM_MUL_ROPE
GGML_OPENCL_DEVICES=1:0 GGML_OPENCL_FUSE_QK_NORM_ROPE=1 ./bin/test-backend-ops -b GPUOpenCL -o RMS_NORM_MUL_ROPE
#   the red case: in ggml/src/ggml-opencl/kernels/rms_norm.cl drop the epsilon
#   (`sqrt(sumf / ne00 + eps)` -> `sqrt(sumf / ne00)`), rebuild, the on arm must fail
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" qk-off --unit coder --device GPU.N \
  --env GGML_OPENCL_FUSE_QK_NORM_ROPE=0
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" qk-on  --unit coder --device GPU.N \
  --env GGML_OPENCL_FUSE_QK_NORM_ROPE=1
#   the same pair for --unit flashnext on GPU.0. The tripwire the rounding gate
#   needs (test-backend-ops' NMSE cannot see it): the reply lines' md5 in the two
#   arms' summaries, and argmax agreement from three same-window captures —
#   the off arm twice (the twin pair) and the on arm — built on the coder with
#   $PERPLEXITY = contrib/llama-loan-kld/build-tools.sh <tree>/build-tools/bin/llama-perplexity:
#     env GGML_OPENCL_FUSE_QK_NORM_ROPE=0 $PERPLEXITY -m "$MODEL" -c 8192 -b 4096 -ub 2048 \
#       --chunks 1 -t 8 -ngl 99 -lm mmap -nr --save-all-logits "$WP_TMP/qk-off-a.kld"   # twice, -b
#     env GGML_OPENCL_FUSE_QK_NORM_ROPE=1 $PERPLEXITY ... --save-all-logits "$WP_TMP/qk-on.kld"
python3 tools/capture_vs_capture.py "$WP_TMP/qk-off-a.kld" \
  --baseline "$WP_TMP/qk-off-a.kld" "$WP_TMP/qk-off-b.kld" --candidate "$WP_TMP/qk-on.kld" --window 0
#   bar: the answers stay right, mean KL at most 0.03 nats worse than the off arm,
#   argmax agreement at most 1 point down (CLAUDE.md).

# 6 (1): the lane's ceiling, the cold/warm pair, and the faults from /proc
GGML_OPENCL_CPU_MISS_MAX=0  python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" miss-ceiling --unit flashnext --device GPU.0
GGML_OPENCL_CPU_MISS_MAX=64 python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" miss-warm    --unit flashnext --device GPU.0
#   the cold arm is a fresh model load and its first request: drop_caches evicts
#   page cache, not this USM memory (`code`)
GGML_OPENCL_CPU_MISS_MAX=64 python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" miss-cold    --unit flashnext --device GPU.0
#   faults, no perf on the leg host: /proc/<pid>/stat fields 10 (min_flt) and 12
#   (maj_flt), sampled around one decode
GGML_OPENCL_CPU_MISS_MAX=64 python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" miss-faults \
  --unit flashnext --device GPU.0 --prompt-tokens 64 &
sleep 5; pid=$(pgrep -f -- "$ARCINT" | head -1); \
  awk '{print "min_flt", $10, "maj_flt", $12}' "/proc/$pid/stat"; wait

# 6 (2): the A/B at the same cap
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" pf-off --unit flashnext --device GPU.0 \
  --env GGML_OPENCL_CPU_MISS_MAX=64 --env GGML_OPENCL_CPU_MISS_PREFETCH=0
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" pf-on  --unit flashnext --device GPU.0 \
  --env GGML_OPENCL_CPU_MISS_MAX=64 --env GGML_OPENCL_CPU_MISS_PREFETCH=2
```

Reading: 16's bar is ≥ +0.5 % decode at the wall; below it the switch stays off
and the patch stays as a mechanism, and the answer-level tripwire above is its
quality gate. 6's bar is major faults > ~5 % of the CPU lane's time; the
prediction is ~0, so level 2 is expected to stay off. A default follows only
from those numbers.
