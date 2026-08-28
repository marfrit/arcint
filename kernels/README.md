# Hand-written kernels

## What this proves

Injecting our own OpenCL kernel into OpenVINO works on this hardware, on
**dynamic** shapes, with correct numerics. `permute.cl` is a real kernel — a
(0,2,1,3) permute on BFYX — not a toy, and it was verified against
`numpy.transpose` on both `[2,3,5,8]` and `[?,?,?,?]`.

That matters because it is the cheap end of kernel ownership: one op replaced,
OpenVINO's graph, memory and scheduling kept.

## What it also proves — measured, B60, 2026-08-28

It is **much slower than the built-in**, and the reason is instructive.

| | 30 chained permutes, `[1,512,32,128]` fp16 | per permute |
|---|---|---|
| OpenVINO `Transpose` | 0.97 ms | 32 µs |
| this kernel | 1067 ms | 35.6 ms |

470× slower on the isolated benchmark, at ~0.26 GB/s effective. The `Reorder`
around the custom op costs 0.000 ms, so this is the kernel itself, not layout
conversion. The likely cause is that `INPUT0_PITCHES` / `INPUT0_DIMS` are not
compile-time constants here, so every work-item pays extra indirect loads;
that is worth confirming before a second attempt.

**Do not read this as "hand-written kernels lose."** It says this particular
first attempt loses, through a mechanism that is fixable. What it does settle
is that a custom kernel is not free money — OpenVINO's kernels are tuned, and
beating them requires the same care they had.

## The more valuable finding

In the isolated benchmark OpenVINO chose **`permute_f_y_axes__f16`**, a
*specialised* kernel. In the real model, the same permutation on the same
tensor shape runs **`permute_ref__f16`**, the generic fallback — and that is
18–27% of decode time.

So the model is missing a specialised kernel it is otherwise eligible for.
Finding out which condition disqualifies it is worth more than writing a
replacement, because the fast kernel already exists and would cost nothing.
That is the next thing to chase.
