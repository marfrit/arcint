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
