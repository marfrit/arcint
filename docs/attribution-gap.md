# The attribution gap: pre-registration

A profile sums parts and cannot see what happens between them. The 70% of
prefill graph time that no node accounts for is invisible to the instrument that
found it, so this is settled with a **timeline** — kernel enqueue / start / end
on the device — which counts occupancy rather than node totals. Two instruments
that do not share a counting principle; if they disagree about the same run,
the disagreement is the finding.

This file is committed **before the measurement runs.**

## The binary

One served prefill, 13410 tokens, the same length as the discrepancy. Compute
total device-busy time and total gap time.

* **Outcome 1 — device busy ~1.88 s.** The card is idle for ~70% of a prefill.
  The work is host-side: queue submission, synchronisation, per-chunk
  orchestration, transfers attributed to no node. The lever is arcint's own
  scheduling, not anybody's kernel.
* **Outcome 2 — device busy ~6.20 s.** The card is busy and the profiler
  under-reports what it is busy with. The sizing corpus then needs a conversion
  factor rather than a suspension, and the question becomes what PERF_COUNT
  excludes.

## What I expect, written before running

**Outcome 1**, device busy in the region of **1.3–1.9 s** of a 5.76 s prefill,
i.e. the card idle roughly 65–75%. Stated as a range because the reasoning below
gives a bound, not a point.

Three things push me there.

1. **PERF_COUNT carries a large fixed per-node cost, so the node sums are an
   upper bound on kernel time — and a loose one.** At M=1 the profiler reports a
   node total of **128.57 ms** for a single-token forward, while a served decode
   step at 68 t/s is **~14.7 ms**. The profiler over-reports by 8.7x there. A
   chunk executes ~1135 nodes, so that is ~100 us of measurement overhead per
   node execution. Independent support: a 13410-token prefill is ~7434 node
   executions, predicting ~0.74 s of overhead, and the measured cost of turning
   profiling on was **+0.44 s** (6.20 s of graph against 5.76 s) — same order.
   Subtracting ~113 ms of overhead from the 361 ms chunk leaves ~248 ms of true
   kernel time against ~941 ms of chunk wall: ~26% busy.
2. **The effective arithmetic rate is too low for a busy card.** ~3B active
   parameters over 13410 tokens is ~80 TFLOP; in 5.76 s that is ~14 TFLOPS
   effective. If the card were busy the whole time that would be a poor fraction
   of this part's f16 peak for a compute-bound prefill. At ~30% busy the kernels
   would be running at a rate that is high but credible.
3. **The paged graph is dynamic-shape.** Per-execution host-side shape
   inference and impl selection across ~1135 nodes is a known way to become
   host-bound, and it is the mechanism that would produce exactly this shape.

**What would falsify it:** device busy at or above ~4 s. I would then be wrong
about the per-node overhead being large, and the question becomes what PERF_COUNT
excludes rather than what the host is doing between kernels.

**Prior worth stating against myself.** A two-thirds-shaped hole from this same
apparatus was retracted two days ago as a denominator error. This one has depth
accounted for and two counting methods agreeing, which is why it is being
checked rather than believed.

## Controls, before any derived number is trusted

1. **Wall.** Total wall time for the traced prefill must match the served
   figure.
2. **A named kernel.** At least one kernel's total duration in the timeline must
   match the profiler's figure for that same kernel in the same run.
3. **Tracer overhead**, measured the way the profiler's +7.6% was: traced
   against untraced wall for the identical prefill. A tracer that intercepts
   every launch is not free, and if it costs 40% the split it reports is not the
   split of an untraced run.

If the timeline cannot reproduce a quantity already in hand, it is not measuring
what it is thought to be measuring.

## Scope

One measurement, then a decision. If the answer is host-side, naming which
host-side phase is a separate order. D1, the gate row's padding-N lever and
`Add`/vload8 stay parked and unpriced until this prices them.

---

# Result

**The prediction was wrong.** I pre-registered Outcome 1 for prefill — the card
idle 65–75% — and the timeline says the opposite. The answer also splits by
phase, which the binary did not anticipate.

## Controls first

| control | required | result |
|---|---|---|
| 1. wall | traced span matches the served figure | **pass** — cluster span 5.235 s against 5.56 s of reported graph time, 94%. The missing 6% is host time inside the graph call before the first enqueue and after the last completion. |
| 2. a named kernel | timeline duration matches the profiler's, same run | **fail — and this is the finding.** |
| 3. tracer overhead | measured, not assumed | **prefill +0.9%** (5.41 s of graph traced against 5.36 s). **Decode +42%** (0.27 s against 0.19 s). |

Control 3 is why the two phases are reported differently. On prefill the tracer
is free enough that its split is the split of an untraced run. On decode it is
not — decode is host-bound, and a tracer that intercepts every call lands on
exactly the path that is limiting. The decode figure below is corrected for it
and labelled as an estimate.

## Control 2, the disagreement

Same run, same chunk (M=2048, past 0), profiler dump against device timeline for
the two ops with a clean 1:1 between nodes and kernel launches:

| kernel | PERF_COUNT | device timeline | ratio |
|---|---|---|---|
| `paged_causal_conv1d_ref` x30 | 14.97 ms | 27.23 ms | **1.82x** |
| `paged_gated_delta_net_opt` x30 | 36.78 ms | 67.19 ms | **1.83x** |

Two unrelated kernels, the same factor to two digits. That is systematic, not
noise. **PERF_COUNT reports roughly 55% of the device time a kernel actually
occupies**, and on top of that it cannot see memory transfers at all and does
not enumerate the sub-kernels of composite primitives — a prefill issues 13342
device commands against roughly 7150 node executions.

So the answer to "what does the profiler exclude" is three things: about 45% of
each kernel's own duration, every transfer, and every sub-kernel.

## The binary, answered

**Prefill: Outcome 2.** Device busy **4.980 s of a 5.235 s span — 95.1%**. The
card is not idle. Two independent prefill clusters in the same trace agree to
0.05% (4.980 s and 4.978 s busy; 921 and 922 ms of transfers), so this is
reproducible rather than a single reading.

**Decode: closer to Outcome 1.** Device busy 0.082 s. Against the *untraced*
decode graph time of 0.19 s that is **~43% busy, so roughly half of decode wall
is host-side**. Stated as an estimate: it assumes device busy is unchanged by
tracing, which is reasonable but not measured.

## What a prefill is actually made of

Device time, one served prefill, 12916 tokens, f16 KV, chunk 2048. These are
shares of **wall time for that prefill** (span 5.235 s), which is the
denominator that was missing all along:

| item | device time | % of prefill wall | note |
|---|---|---|---|
| `clEnqueueMemcpyINTEL` (all) | 921 ms | **17.6%** | not a node; invisible to the profiler |
| `grouped_micro_gemm` | 912 ms | 17.4% | MoE expert GEMM |
| `sdpa_micro__generate` | 568 ms | 10.9% | attention |
| `ref_matmul` | 528 ms | **10.1%** | 280 = 40/chunk x 7: **the shared_expert_gate** |
| `gemm_kernel` | 468 ms | 8.9% | |
| `paged_gated_delta_net_opt` | 421 ms | 8.0% | |
| `moe_scatter_reduction` | 353 ms | 6.7% | |
| `generic_eltwise_ref` | 181 ms | 3.5% | the `Add` row |
| `paged_causal_conv1d_ref` | 171 ms | **3.3%** | D1's target |
| `moe_gather_ref_prefill` | 130 ms | 2.5% | |
| idle | 255 ms | 4.9% | |

**The transfers are one copy per chunk, and they are astonishingly regular:** six
device-to-host copies of **142.74, 142.73, 142.74, 142.74, 142.74, 142.74 ms**,
one per prefill chunk, 856 ms of the 921 ms total. A fixed per-chunk cost that
no node accounts for and that nothing in the profiler could ever have shown.

## Repricing the parked items

The measurement's purpose was to price them, and it does — mostly downward.

* **D1 (`PagedCausalConv1D` kernel) is dead as a decode lever, which is the only
  lever it had.** Its case was "6.3% of a decode step, about 4 t/s of 68". On
  the device it is 0.9 ms of decode, against 190 ms of untraced decode wall:
  **0.47%**. A perfect conv kernel buys well under 1 t/s. In prefill it is 3.3%
  of wall, which is real but is now the eighth-largest item on a board where
  three things above it are larger and cheaper. **Recommend dropping it.**
* **The gate row is bigger than I converted it to, not smaller.** `ref_matmul`
  is **10.1% of prefill wall** — 528 ms, 40 executions per chunk. My suspended
  "6.0%" understated it. The padding-N lever targets exactly this and is rung
  zero of §1.1. **This is the strongest kernel candidate on the board.**
* **`Add`/vload8 is 3.5% of prefill wall**, roughly as sized, and stays a
  hypothesis until operand shapes confirm it.
* **Two new items outrank all three.** The per-chunk 142.74 ms DtoH copy
  (17.6% of prefill wall) and the host-bound decode (~57% of decode wall spent
  with the card idle). Both are arcint's own code rather than anyone's kernel.

Naming which host-side phase, and what the per-chunk copy is, are separate
orders with their own budgets, per the scope of this one.

## Consequence for the record

Every node share in this repository is a share of a quantity that is ~55% of
device time and blind to transfers and sub-kernels. They are not convertible to
wall shares by any single factor, because the missing mass is not distributed
proportionally — transfers and sub-kernels attach to particular ops. The
profiler tables now say so at the point of printing, alongside the chunk
denominator.
