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

**Caveat that has to be settled first: the node may no longer exist.** The paged
transformation removed 90 `Transpose` nodes when the served path moved to it, and
no `Permute` row appears in any stored paged profile. But the profiler printed
only the top 20 of 28–29 `(op, kernel)` pairs, so up to nine rows were never
shown and a permute could be among them. The cap is now removed — every pair
prints — and the next capture answers it. **Until then, the premise of the
permute work is unverified**, and it is a premise about existence, not about
mechanism.

## paged_causal_conv1d_ref — there is nothing else

`OV_GPU_PRIMITIVE_IMPL("ocl::paged_causal_conv1d::ref")` is the only registration,
and `paged_causal_conv1d_ref` the only kernel string, in both the source tree and
the shipped binary. Nothing bailed. This row's answer is "a kernel to write", and
it is OpenVINO-side, so §1.1's first rung is an OpenVINO PR.

For contrast the GDN pair registers both a `ref` and an `opt`, and our profile
shows the `opt` being used — so the paged transformation gave GDN a fast kernel
and left the causal convolution behind.

## shared_expert_gate — the fallback is inside oneDNN

This row changed the most under reading.

`jit:gemm:any__i8` and `ocl:ref:any__i8` are **not** OpenVINO implementation
names. OpenVINO's are of the form `ocl::sdpa::ref`, `onednn::fc`,
`ocl::moe::moe_3gemm_swiglu_opt`. The strings we see are **oneDNN's own**
primitive descriptors — `"ocl:ref:any"` occurs 30 times in oneDNN's GPU sources —
with the plugin appending the `__i8` dtype tag.

And OpenVINO's own gate does not reject anything here.
`FullyConnectedImplementationManager::validate_impl` checks `supports_immad`, the
architecture, formats, padding, and a long list of dtype combinations — the
compressed case (`u4`/`i4`/`u8`/`i8` weights) is explicitly admitted. **There is
no shape or batch condition in it at all.** The plugin selects `onednn::fc` for
these nodes at every M; oneDNN then chooses its own JIT GEMM below M=128 and its
own reference kernel from M=128.

So the fix is not an OpenVINO selector change. It is in **oneDNN**, which is
vendored as a submodule of the GPU plugin — a different upstream project, and a
longer loop: a oneDNN PR reaches us only when OpenVINO bumps the submodule. That
is worth knowing before choosing this item, and it is the kind of thing §1.1's
ladder exists to make explicit.

**Confirming it needs no debug build.** `ONEDNN_VERBOSE=1` makes oneDNN print the
primitive it created, with shapes, for every dispatch — so the exact rejected
shape and the chosen implementation can be read from a normal run. That is the
next measurement for this row, and it is cheaper than the one already running.
