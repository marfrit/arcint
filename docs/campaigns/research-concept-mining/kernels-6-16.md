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
input row and a contiguous f32 output. Switch `GGML_OPENCL_FUSE_QK_NORM_ROPE=1`,
default off.

**Deviation, named** (`code`): the audit's sketch names "the qk norm+rope OpenCL
kernel next to `rms_norm.cl`/`rope.cl`, dispatched in the attention builders".
The dispatch is not in `qwen35.cpp`/`qwen4exp.cpp`; it is the backend's pattern
fuse, which is how this series already fuses (`code`: 0049, 0064, 0065). The
builders stay untouched, so a model other than the two served ones changes
nothing.

**Quality class** (`code`): the audit's `rounding-only`; the kernel keeps the
two kernels' rounding order (norm store then weight) and the same cos/sin, so
the fused bits are the two-kernel path's. The bar is the answer-level one
(`CLAUDE.md`), judged on the card.

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
the kernel's creation, and the new `test-backend-ops` cases. Five mutations were
run and seen failing before restore: epsilon dropped, weight read at `w[ic]`,
weight folded before the rounding, default flipped on, dispatch moved below the
prefix; and the kernel's absence fails five cells. `code`: the patch also adds
the served shapes to `test-backend-ops`'s `RMS_NORM_MUL_ROPE` (the strided Q
layout `[128,16,1]` and `[128,16,5]`, contiguous K `[128,8,6]`), which ran green
on the CPU backend here (`measured-here`: 227 cases OK, the two strided ones
included) — that is a shapes/graph check, not the kernel's numerics.

## 6 — the CPU miss rows' prefetch and cold-page touch (patch 0075)

**Mechanism** (`code`). One block-ahead `__builtin_prefetch` of the next block's
first payload field in each of the six row functions, and a page-touch pass over
a missed expert's three bank slices at publish (`ggml_cl_miss_cpu_touch`, a byte
a 4 KiB page through a volatile pointer), run as the pool's tasks so the faults
flow in parallel before the rows read them. Switch
`GGML_OPENCL_CPU_MISS_PREFETCH`: 0 off (default), 1 the hints, 2 also the touch.

**Deviation, named** (`code`): the sketch's hint is "ahead of the next expert's
first field"; the build hints the block two ahead *within the same expert* — a
rows' task sees one expert only, so the next expert is not in scope, and the
page-touch pass covers the slice boundary. The audit's cold-page half is built
as sketched (page touch over the layer's candidate slices at publish time).

**Quality class** (`code`): exact. A prefetch cannot change a row's bits; the
touch reads bytes it does not pass on. The C cell asserts the level-1 parity
bit-identical to level 0.

**Pricing measurement — owed, the audit's own:**
- the miss rows' time per verify round: the cap switch's ceiling (`paper`,
  arcint stage 3d: 49.2/50.0 t/s with the CPU computing nothing against
  49.8/50.6 with the ceiling switch — the wait costs nothing) and
  `LLAMA_DECODE_TIMING`'s per-decode split for the round; the prefetch's
  possible saving is the cold/warm gap of the same six answers at
  `GGML_OPENCL_CPU_MISS_MAX=64`.
- `perf stat -e cache-misses,minor-faults,major-faults` on one served decode,
  the campaign's perf method (`paper`).
- bar (`paper`, the audit): a prefetch only if major faults > ~5 % of the CPU
  lane's time.

**Device-free cells** (`measured-here`): `test-opencl-miss-cpu` gains the
level-1 parity arm ("0 of 2 rows differ"), the ahead-distance arm, and the
mincore residency arm ("0 of 16 pages resident before, 16 after"); mutations
seen red: the touch stride 8192 (8 of 16 resident), the ahead distance 0.
`tests/python/test_cpu_miss_prefetch.py` reads the tree for the default, one
hint in every row function, the volatile touch and the `>= 2` guard in both
compute paths; five mutations (a hint dropped from `row_iq3s_1`, distance 0,
stride, non-volatile pointer, default on, guard removed) each seen failing.

## The card commands (Claude runs these in the announced window)

One lane, `fn_sweep.py`'s unit flags; the cards by PCI id per
`docs/sop-card-window.md` (the sampler before the leg, the zombie sweep by pid).
`$ARCINT`, `$MODEL`, `$ASSETS` are the unit's own.

```bash
# 16 (1): the per-step share of the q/k norm + rope launches, one answer a unit
CLI_ChromePerformanceTiming=1 CLI_ChromePerformanceTimingOutput=/tmp/qk-coder.json \
  python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" qk-coder --unit coder --device GPU.N
CLI_ChromePerformanceTiming=1 CLI_ChromePerformanceTimingOutput=/tmp/qk-fn.json \
  python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" qk-fn --unit flashnext --device GPU.0
python3 tools/cl_timeline_steps.py /tmp/qk-coder.json --steps 64 --tail 40 --top 25
python3 tools/cl_timeline_steps.py /tmp/qk-fn.json    --steps 64 --tail 40 --top 25
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
#   the same pair for --unit flashnext on GPU.0; the answers and the answer-level bar

# 6 (1): the lane's ceiling and the cold/warm gap at the same cap
GGML_OPENCL_CPU_MISS_MAX=0  python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" miss-ceiling --unit flashnext --device GPU.0
GGML_OPENCL_CPU_MISS_MAX=64 python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" miss-warm    --unit flashnext --device GPU.0
sync; echo 1 | sudo tee /proc/sys/vm/drop_caches   # the cold arm, or a fresh load and a first request
GGML_OPENCL_CPU_MISS_MAX=64 python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" miss-cold    --unit flashnext --device GPU.0
perf stat -e cache-misses,minor-faults,major-faults \
  python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" miss-perf --unit flashnext --device GPU.0 --prompt-tokens 64

# 6 (2): the A/B at the same cap
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" pf-off --unit flashnext --device GPU.0 \
  --env GGML_OPENCL_CPU_MISS_MAX=64 --env GGML_OPENCL_CPU_MISS_PREFETCH=0
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" pf-on  --unit flashnext --device GPU.0 \
  --env GGML_OPENCL_CPU_MISS_MAX=64 --env GGML_OPENCL_CPU_MISS_PREFETCH=2
```

Reading: 16's bar is ≥ +0.5 % decode at the wall; below it the switch stays off
and the patch stays as a mechanism. 6's bar is major faults > ~5 % of the CPU
lane's time; below it level 2 stays off. A default follows only from those
numbers, with the answer-level bar green on the same arms.
