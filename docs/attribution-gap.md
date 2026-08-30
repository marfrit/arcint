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
