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

## Measured (same evening): the loss is not in the plugin's kernels

Same instrument for all three, the OpenCL device timeline, device time only
(spans under the tracer are inflated and not quoted; wall is from the
untraced runs of 7.0.2o):

| | launches | device busy | wall | host + sync |
|---|---|---|---|---|
| plain step, M=1 | 1145 | 8.7 ms | 14.0 ms | 5.3 ms |
| verify forward, M=2 | 1334 | **10.0 ms** | 25.3 ms | **15.3 ms** |
| head draft | 118 | **6.7 ms** | 11.2 ms | 4.5 ms |

The two-token verify costs the device **1.15x** a plain step — the batched
GEMV path doing its job: MoE gate_up 2.58 against 1.29 ms, down 1.38 against
0.83, and the FC `gemm_kernel` *cheaper* at M=2 (3.20 against 4.37 ms), which
is a oneDNN kernel choice worth its own note. The 1.8x is almost entirely
**host and synchronisation**: 15.3 ms of a 25.3 ms verify against 5.3 of a
14.0 ms step. Per accepted pair: 36.5 ms of wall for 16.7 ms of device. Were
the pair device-bound it would be 8.8 ms a token against 14.0 plain — the
+55% of the work order, reached by a different route, and the levers are
ours rather than upstream's:

1. **arcint's verify loop.** ~10 ms per pass beyond a plain step's host time,
   for +189 primitives walked (~2 ms) and the rest in the loop itself —
   synchronous readbacks (two logits rows, the hidden row for the head), the
   acceptance check, the checkpoint-row bookkeeping, and whatever serialises
   the device behind them. Unattributed at the millisecond; the plugin's
   per-infer host-time profile (7.0.2f's instrument) splits the LM's enqueue
   time from the head's and leaves the remainder to arcint.
2. **The head's host side**, 4.5 ms for 118 launches plus host-side input
   copies and the logits readback for the draft argmax.
3. **The head's device side**, 6.7 ms of f16 weight traffic (an int4 head:
   ~2 ms) — third, because it only shows once the pair is device-bound.

So the "B row" is void — the decode path at M=2 exists and is in use — and
the honest sentence for the 35B card is not "when the plugin has a small-M
MoE path" but: *the speculation loop is host-bound on Arc today, 36.5 ms of
wall for 16.7 ms of device per accepted pair; the device budget alone would
be +59%, and the work is in the serving loop, not the kernels.*
