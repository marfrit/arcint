# Why the 35B's two-token verify runs the prefill path: pre-registration

The claim under test (work order, 2026-08-30): the MoE's M=2 forward costs
1.8x a plain step where the dense 3.8's costs 1.1x, and at decode the MoE
lands on the single-token kernels (`moe_3gemm_swiglu_mlp_gate_up`,
`_mlp_down`) while at M=2 it lands on the prefill path (`grouped_micro_gemm`,
`moe_gather_ref_prefill_gather`, `moe_scatter_reduction`). Either fix alone
(cheaper head, or a decode-path verify) lands on break-even; only both win.

## Readings, written before the source is opened

1. **A gate on `M == 1`** (or a "single token" flag the shape trips): the
   fix is a bounds change, upstream-shaped; the B row is reachable. Name the
   condition and what it would have to become.
2. **The single-token kernels genuinely cannot take M > 1** (their work
   distribution assumes one token per expert set): B is a kernel to write,
   +55% is out of reach for now, and the deliverable is one honest sentence in
   the 35B card.
3. **The selector cannot be read from the pinned build**: say so and stop.

My expectation: (1) with a caveat — the single-token path is likely written
as a per-expert-of-one-token kernel whose launch geometry is derived from
`num_tokens == 1`, so the gate is real but lifting it needs the kernel to
loop over tokens, which is between a bounds change and a new kernel.

## Second item, only after the first: attribute the draft's 11.2 ms

Weight traffic accounts for ~4.8 ms of it (1.69 GB f16 experts + 509 MB
lm_head at ~456 GB/s). The rest is dispatch or the head taking a bad path.
An int4 top-k head removes a share of the 4.8 ms, not of the 11.2, unless the
remainder is attributed first.

## Reading (same day) — and it is none of the three

The pinned plugin already has the path. `moe_3gemm_swiglu_opt.cpp:2565`:

```cpp
// Batched GEMV: for small token counts (including single token, MTP/speculative decoding),
// use optimized GEMV kernels with batch dimension. Avoids gather/scatter overhead.
if (token_num <= batched_gemv_threshold) return exec_batched_gemv(...);
```

with `batched_gemv_threshold = 32` by default, exposed as the release-internal
option `GPU_MOE_BATCHED_GEMV_THRESHOLD` (`options.inl:84`), and the kernel
itself (`moe_3gemm_swiglu_mlp.cl`) written as a batched GEMV over (token,
expert) pairs — "for single token, flat_id == expert_slot and token_idx == 0".
So a two-token verify takes the decode path, not the prefill path.

**Which means the sentence in 7.0.2o — "the MoE's M=2 forward runs the
prefill path" — was an inference from the 1.8x and is retracted.** The 1.8x is
real and unexplained; the trace of an actual two-token verify is the
instrument, and it is running. Two candidates that survive the read: the
batched GEMV at token_num=2 launching 2x the (token, expert) workgroups and
reading 2x the expert weights (the step may be memory-bound on the MoE where
the dense 3.8 is not), and something outside the MoE — the GDN verify pass
with per-token checkpoint rows, or attention at M=2 leaving the single-token
paged-attention kernel.
