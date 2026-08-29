# Hand-written kernels

## The question this answers

Does writing our own compute kernels yield a more efficient solution — in RAM
or in speed? Measured on a B60 with the 27B coder q4, in the real model, not a
microbenchmark.

**The kernel: yes, by 8.6×. The mechanism for getting it into the graph: no.**

## What was measured

`permute.cl` implements the (0,2,1,3) head-major swap the GDN layers do three
times per layer — `[B, S, 32, 128] -> [B, 32, S, 128]`, 90 executions per decode
step, the second-largest single line in the profile after the GEMMs.

`--custom-kernels kernels/custom_layers.xml` routes exactly that shape class to
it. Output is **byte-identical** to the built-in path (same sha256 over a
144-token greedy generation), so the kernel and the work-size mapping are right.

Per-node, in the model, on the same op:

| kernel | per node | nodes | total |
|---|---|---|---|
| OpenVINO `permute_ref__f16` | 13.6 µs | 90 | 1226 µs |
| ours, `lgc_permute__f16` | **1.59 µs** | 90 | **143 µs** |

That is a real 1.08 ms saved out of an 11.16 ms decode step — **9.7%**, which
would be ~50 → ~55 t/s.

## Why it still loses

A `CustomLayer` node is opaque to OpenVINO's own graph optimiser, so it is a
**fusion barrier**. Replacing 90 transposes changes the rest of the graph:

| | baseline | with LgcPermute |
|---|---|---|
| the permute itself | 1226 µs / 90 | 143 µs / 90 |
| other `permute_ref` | — | 1555 µs / **180 nodes** |
| `Multiply generic_eltwise_ref__f32` | — | 753 µs / 120 (note: **f32**) |
| `ReduceSum reduce_ref__f32` | — | 274 µs / 60 |
| `Gather gather_ref__f16` | — | 194 µs / 60 |
| **decode step** | **11.16 ms** | **12.92 ms** |

`RoPE ocl::rope::opt` disappears from the profile entirely and comes back as
f32 primitives, and 180 transposes that had been optimised away return. The
collateral is about +1.6 ms against a 1.08 ms win: **50.7 → 44.4 t/s**.

Routing *all* 160 matching transposes instead of the hot 90 is worse again
(36.2 t/s), which is how the barrier was identified: the damage scales with how
much of the graph is made opaque, not with how much work is offloaded.

## What that means

The ceiling is real and the kernel clears it. What does not work is OpenVINO's
only *supported* injection path. To capture the 9.7% the kernel would have to
live inside OpenVINO's kernel selection rather than beside it — either by
improving `permute_ref`, or by relaxing `permute_f_y_axes`, which brings us to:

## The disqualifier, found

`permute_f_y_axes__f16` is the specialised kernel for exactly our permutation
(on BFYX, (0,2,1,3) *is* the f↔y swap). The model never gets it. The condition
is **dynamic shape**, and nothing else — probed directly:

| input shape | kernel chosen |
|---|---|
| static `[1,512,32,128]` | `permute_f_y_axes__f16` |
| static `[1,1,32,128]` | *the transpose is deleted outright* |
| `[1,?,32,128]` | `permute_ref__f16` |
| `[?,?,32,128]` | `permute_ref__f16` |
| `[?,?,?,?]` | `permute_ref__f16` |

Two things follow. First, one static dimension is the difference between a
specialised kernel and the generic fallback, so anything that makes the sequence
dimension static at decode is worth more than a hand-written kernel. Second —
at static `S = 1` OpenVINO **removes the transpose entirely**, because the
permutation is then a layout no-op. The 1.23 ms we are paying at decode is 90
copies that reorder nothing and exist only because the shape is dynamic.

## Intel silicon the kernel could use (B60 only)

Queried on the cards, 2026-08-28. Battlemage exposes, and Alchemist does not:

- `cl_intel_subgroup_2d_block_io` — hardware 2D block load/store, whose Xe2 ISA
  form has **transpose** and VNNI-transform variants: the transposition comes
  free as part of the load. Nvidia's nearest is Hopper TMA, which does async
  bulk 2D copy but still leaves the transpose to SMEM/registers.
- `cl_intel_subgroup_extended_block_read`, `cl_intel_subgroup_buffer_prefetch`
- `cl_intel_subgroup_matrix_multiply_accumulate_tf32`

Both cards have `cl_intel_required_subgroup_size` (SIMD8/16/32 chosen per
kernel, where an Nvidia warp is fixed at 32), `cl_intel_split_work_group_barrier`
and `cl_intel_bfloat16_conversions`. The A770 has `cl_intel_media_block_io` and
`cl_intel_subgroup_split_matrix_multiply_accumulate` instead of the 2D block I/O.

The free hardware transpose does **not** help the case measured here: at decode
`S = 1` there is nothing to transpose. It would pay in prefill, where `S` is
large and the permute genuinely moves megabytes. That is a separate, real
opportunity.

## Status

`--custom-kernels` ships **off by default**. It is a measurement switch, kept
because the number it produces is the argument: our kernel is 8.6× faster than
the one being used, and the reason we cannot have that speed is OpenVINO's
graph optimiser, not the hardware and not the kernel.
