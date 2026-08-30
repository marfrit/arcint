# Why these nodes land on reference kernels

Phase A of the plugin work: read the selector before writing anything. The method
has now worked three times — `can_use_micro_sdpa_for`, the head-size and
`token_type_ids` gates, and the three rows below. A node on a reference kernel
usually does not mean no fast kernel exists; it means a fast kernel bailed on a
condition, and the condition is readable.

Source read at the pinned runtime's exact commit: build number
`2026.4.0-22849-71640275d29`, full sha `71640275d29692354223572e77ef717e92b891d9`,
dated 2026-08-21. Nothing here is inferred from a rate.

| ref node | is there another impl? | what rejected it | what it would take |
|---|---|---|---|
| `permute_ref__f16` | **Yes, five** | `GetDivisor(Feature) == 1` | recognise a degenerate case — but first check the node still exists |
| `paged_causal_conv1d_ref` | **No** | nothing bailed; there is nothing else | a kernel, OpenVINO-side |
| `ocl:ref:any__i8` ×40 (`shared_expert_gate`) | **Yes, and the choice is not OpenVINO's** | oneDNN's own dispatch | a fix in **oneDNN**, a different upstream |

## permute_ref — the fast kernel bails on a sequence length of 1

Five optimised permute kernels exist beside the reference one:
`f_y_axes`, `tile_8x8_4x4`, `tile_8x8_4x4_fsv`, `xy_swap`, `bfzyx_to_bfyxz`.

Our transpose is the head-major swap, IE order `(0,2,1,3)` on a rank-4 tensor
`[b, seq, 32, 128]`. In `bfyx` that is `b=1, f=seq, y=32, x=128` — an **f↔y
swap**, which is exactly what `PermuteKernel_f_y_axes` is for. It matches the
order test (`is_swapping_f_with_y`, cldnn order `0 3 2 1`). Then:

```cpp
const auto feature_div = GetDivisor(in.Feature().v);
const auto y_div       = GetDivisor(in.Y().v);
if (feature_div == 1 || y_div == 1) DO_NOT_USE_THIS_KERNEL(p.layerID);
```

with

```cpp
size_t GetDivisor(const size_t input_size) {
    for (size_t d : {16, 8, 4, 2}) if (input_size % d == 0) return d;
    return 1;
}
```

`Feature` is the **sequence length**. At decode it is 1, nothing in {16,8,4,2}
divides 1, the divisor is 1, and the kernel refuses. Every decode step therefore
falls to `permute_ref`. (`tile_8x8_4x4` is not an alternative: it requires
`is_rotating_except_batch`, and `(0,2,1,3)` does not rotate the feature dim to
the back.)

**What it would take is not a faster transpose.** At `F == 1` the swap is a
*no-op*: in `bfyx` the input index is `y*X + x` and the output index is `f*X + x`
with `f == y` — the same bytes in the same order. The honest fix is recognising
the degenerate case, which is a small and upstreamable change to a bail
condition rather than a kernel.

**Settled, and it voids the item: the node does not exist on the served path.**
With the row cap removed, a capture prints all **29** `(op, kernel)` pairs of a
2048-token prefill chunk, and a case-insensitive search of the whole capture —
prefill and decode tables both — returns **zero** occurrences of `permute` or
`transpose`. The paged transformation removed the transposes when the served path
moved to it, and `permute_ref__f16`, the 18%-of-decode finding of §5 and the
target of `kernels/permute.cl`, is simply not in the graph any more.

So the analysis above is correct and inapplicable. It is kept because the
mechanism is real and will apply the moment an f↔y permute reappears — and
because "the fast kernel bails on a sequence length of 1" is a shape this
architecture will meet again.

## paged_causal_conv1d_ref — there is nothing else

`OV_GPU_PRIMITIVE_IMPL("ocl::paged_causal_conv1d::ref")` is the only registration,
and `paged_causal_conv1d_ref` the only kernel string, in both the source tree and
the shipped binary. Nothing bailed. This row's answer is "a kernel to write", and
it is OpenVINO-side, so §1.1's first rung is an OpenVINO PR.

For contrast the GDN pair registers both a `ref` and an `opt`, and our profile
shows the `opt` being used — so the paged transformation gave GDN a fast kernel
and left the causal convolution behind.

## shared_expert_gate — a catalog gap in oneDNN, not a policy rejection

This row changed twice under reading. The second change was a measurement
falsifying the first.

`jit:gemm:any__i8` and `ocl:ref:any__i8` are **not** OpenVINO implementation
names. OpenVINO's are of the form `ocl::sdpa::ref`, `onednn::fc`,
`ocl::moe::moe_3gemm_swiglu_opt`. The strings we see are **oneDNN's own**
primitive descriptors, with the plugin appending the `__i8` dtype tag. And
OpenVINO's own gate does not reject anything here:
`FullyConnectedImplementationManager::validate_impl` admits the compressed case
explicitly and **contains no shape or batch condition at all**.

`ONEDNN_VERBOSE=dispatch` reads the decision from a normal run — no debug build.
It logs rejected implementations with a reason and a source line. For the gate,
whose problem is `M x 2048 : 2048 x 1`:

| impl | reason | site |
|---|---|---|
| `jit:gemm:any` | **matching kernel not found in catalog** | `gemm/jit.hpp:394` |
| `jit:xe_hp:gemm:any` | skipping or dispatching to another impl | `jit_xe_hp_systolic.cpp:86` |
| `ocl:with_po:any` | unsupported attribute | `gemm/with_post_ops.cpp:44` |
| `ocl:ref:any` | unsupported attribute | `gemm/ref.hpp:147` |

The attribute list names the operation exactly: `src_a:s8` x `src_b:u4` with
group-wise scales and zero-points (`64x1`), `attr-precomputed-reductions`, and
**`attr-post-ops:eltwise_logistic`** — the sigmoid is fused into the GEMM. That
had been guessed structurally and is now read, not inferred.

**The boundary is real and the log locates it.** `jit:gemm:any` is rejected at
M = 128, 256, 416, 512, 1024, 2048 — and **not** at M = 1. Decode keeps its fast
kernel; only prefill falls off. This matches the profile, where no `ocl:ref:any`
FC row appears in a decode capture at all.

### The mechanism I read out of the source was wrong

`jit.hpp:394` is the *end* of a loop, not the decision. Above it, each candidate
is filtered:

```cpp
// Global k-parallel kernels don't support post-ops or non-f32/s32
//   accumulation unless fusion is enabled.
if (kernel_desc_.driver_info()->kParallel() && !kernel_desc_.driver_info()->fusedPostOps()) {
    bool po_valid = !non_scale_po_ && !(with_sum_ && with_c_scales())
                    && utils::one_of(d->c_type(), f32, s32);
    valid &= po_valid;
}
```

N=1 over K=2048 is a GEMV, whose viable entries are k-parallel; our problem has a
non-scale post-op (the sigmoid) *and* an f16 destination, and `po_valid` needs
both to go the other way. A tidy explanation, and false.

The same loop prints its skip reasons at `debuginfo >= 5`. Re-run with
`ONEDNN_VERBOSE=dispatch,debuginfo=5`: **no skip line is emitted for any of the
six shapes**, and counting `consider` records between each problem's start and
its miss gives **zero candidates for all six**. `select_kernel()` returned an
empty list; the post-op filter never ran.

The null is trustworthy because the instrument carries a positive control at the
*same* verbosity gate: `"info,gpu,gemm,consider:%s,score:%f"` is guarded by the
same `debuginfo >= 5` in the same selection path, and 2198 of those records are
in the log (2661 lines against 192 without the flag). The channel was live and
loud; it simply had nothing to say about these six problems.

So this is a **shape gap in the kernel catalog**, not an attribute or policy
rejection. No entry exists for (s8 x u4-grouped, N=1, f16 dst) at M >= 128.

### What that means for the fix — and it moves the row down §1.1's ladder

Steerability, asked before committing to the long loop:

* **Not by an environment knob.** The only `ONEDNN_*` string in the shipped
  plugin that touches implementation choice is `ONEDNN_GROUPED_GEMM_USED`, which
  is unrelated. The catalog is compiled in.
* **Not by un-fusing the sigmoid, and not by changing the destination dtype.**
  Both feed a filter that never executes. This is worth stating because it was
  the obvious first idea and it is dead.
* **Possibly by changing N**, because the catalog is matched on *shape*. N=1 is
  the degenerate case. Padding the gate's output width to a catalogued N is a
  pre-compile graph rewrite we own entirely — no OpenVINO change, no oneDNN
  change, **rung zero of §1.1**. The extra columns are discarded. Whether a
  catalogued N=16 kernel doing sixteen times the arithmetic beats a reference
  GEMV is an open question with an obvious experiment, and the graph-surgery
  plumbing is the same plumbing D1 needs.

The earlier conclusion — "the fix is in oneDNN, a different upstream, a longer
loop" — is therefore **premature rather than wrong**: it remains the fallback if
padding does not pay, but it is no longer the first thing to try.

## The complete kernel inventory of a served prefill chunk

With the cap removed, all 29 pairs, 2048-token chunk at past 0, f16 KV. Reference
implementations marked:

| share | nodes | op | kernel | ref? |
|---|---|---|---|---|
| 30.2% | 40 | FullyConnectedCompressed | `ocl:ref:any__i8` | **ref** (oneDNN's) |
| 19.4% | 331 | FullyConnectedCompressed | `jit:gemm:any__i8` | |
| 15.9% | 30 | PagedGatedDeltaNet | `paged_gated_delta_net::opt` | |
| 8.3% | 40 | MOECompressed | `moe_3gemm_swiglu_opt` | |
| 6.5% | 30 | PagedCausalConv1D | `paged_causal_conv1d::ref` | **ref** |
| 4.8% | 104 | Add | `generic_eltwise_ref` | **ref** |
| 3.2% | 60 | Swish | `activation_ref` | **ref** |
| 2.7% | 161 | DynamicQuantize | `dynamic_quantize_gpu_opt` | |
| 2.2% | 131 | RMS | `rms_gpu_bfyx_opt` | |
| 1.8% | 10 | PagedAttentionExtension | `paged_attention::opt` | |
| 1.3% | 41 | StridedSlice | `strided_slice_ref` | **ref** |
| 0.9% | 40 | Multiply | `generic_eltwise_ref` | **ref** |
| 0.7% | 10 | Crop | `generic_eltwise_ref` | **ref** |
| 0.7% | 10 | VariadicSplit | `generic_eltwise_ref` | **ref** |
| 0.6% | 40 | MoERouterFused | `moe_router_fused_opt` | |
| 0.4% | 20 | Concat | `concatenation_gpu_simple_ref` | **ref** |
| 0.1% | 20 | RoPE | `rope::opt` | |
| rest | 17 | Gemm, ScatterNDUpdate, SoftPlus, Reshape, Convert, Gather, Cos, Sin | mixed | mostly ref |

**No `Permute`. No `Transpose`.** Reference implementations account for roughly
**49%** of the chunk's node time, of which the single largest is the oneDNN gate
fallback at 30.2%.

## What those percentages are percentages *of* — the sixth past-0 correction

The table above is one 2048-token chunk **at past 0**, and that denominator is
not the served one. Both figures below are correct; they answer different
questions, and the gap between them is the whole reason to write this down.

| quantity | value |
|---|---|
| node total, 2048-token chunk at past 0 | 179 ms |
| ref-FC (40 nodes) in that chunk | 54 ms = **30.2%** |
| chunks in a 14450-token prefill | 7.06 |
| measured prefill wall, 14450 tokens | 6420 ms |
| **mean chunk wall at that depth** | **910 ms** |
| ratio, mean chunk / past-0 chunk | **5.1x** |

ref-FC costs ~54 ms per chunk regardless of depth, so across that prefill it is
7.06 x 54 = 382 ms of 6420 ms = **6.0% of prefill wall**, falling to roughly
**2.2%** at 115k tokens and to **zero in decode** (all 371 FC nodes dispatch on
`jit:gemm` at M=1). The 30.2% is the share of the *cheapest possible* chunk.

This was **not** a truncation artefact — node totals were always summed over all
pairs, only the printing was capped. It is the past-0 denominator again, for the
sixth time. Every share in the inventory table carries this caveat.

### The corollary is larger than any row in the table

910 ms mean chunk minus 179 ms at past 0 leaves **731 ms, or 80% of a mean
chunk, in the depth-dependent term** — and every entry in the inventory above is
a share of the remaining 20%. All reference implementations together (49% of a
past-0 chunk) are therefore about **10% of prefill wall** at 14450 tokens, and
less at greater depth.

That inverts the board: the kernel rows are single-digit percentages of served
prefill, and whatever grows with depth is most of it. The only quadratic term in
the graph is attention, which sits at 1.8% at past 0 — but "the growth is
attention" is an inference, and the two-depth capture is exactly the measurement
that settles it. It should run before any kernel work is chosen.

## A fourth row, started: `Add` on `generic_eltwise_ref` (104 nodes, 4.8%)

Fast eltwise kernels exist — `vload8`, `blocked_opt`, `fs_b_yx_fsv32`,
`mixed_byxf_and_fs_b_yx_fsv32`. `EltwiseKernel_vload8::Validate` requires
`(count % 8) == 0` and, in `bCheckSizes`, that **every input be identical in
shape to the others and to the output**, unless it is a true scalar
(`PhysicalSize() == 1`):

```cpp
if (ewParams.inputs[i].PitchesDifferFromLogicalDims() ||
    ((ewParams.inputs[0] != ewParams.inputs[i] || ewParams.inputs[i] != ewParams.outputs[0]) &&
     ewParams.inputs[i].PhysicalSize() != 1))
    bCheckSizes = false;
```

So a **broadcast** operand that is not a scalar disqualifies the vectorised path.
That is the shape of a bias or per-channel add against `[1, seq, hidden]`, and it
is the likely reason 104 `Add` nodes are on the reference kernel. Not yet
confirmed against the actual operand shapes — that is the next selector read, and
`ONEDNN_VERBOSE` does not help here because this one is OpenVINO's own decision.

Unlike the permute row, this is not a one-line bail relaxation: supporting a
broadcast operand in a vectorised kernel is a kernel change.
