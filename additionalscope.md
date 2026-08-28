# Additional scope — bookkeeping and lever inventory (from the outside reviews)

Queued while the speculative-paged work runs. Standing rule for everything in
this file, non-negotiable: **a claimed defect or explanation is accepted only
with a measurement of its root cause** — wall-time decomposition, per-kernel
profile, byte counts × link rate, or an A/B with one variable. A mechanism that
is narrated but not measured gets the §7.0.1 treatment: retracted, on the
record. (Precedent: the bus-speed explanation survived zero minutes of
measurement.)

## A. Close a §8 question that is already answered

The open question "whether OpenVINO can share one set of device weights
between two compiled models" was answered in thirdopinion §3.3 from plugin
source: **no** — constant dedup is per-compile (`ops/constant.cpp:103-172`),
two compiled models mean two 12.8 GiB allocations. Fold the answer in and
redirect the static-shape win to the paged path, where sequence identity is
input data. If you doubt the source read, the measurement is one compile:
instantiate the second model and read `GPU_MEMORY_STATISTICS`.

## B. Xe2 XMX engagement check (five minutes, decides an ISA question)

Grep one B60 `LIGENCE_PROFILE` decode+prefill profile for `dpas` / systolic
markers in the GEMM kernels (`jit:gemm:*`), against the same profile on the
A770 (where the fleet record says XMX never engaged). Outcome either way is a
DESIGN fact:

- engaged on B60 → prefill/lm_head have an ISA lever the A770 lacks; per-card
  kernel expectations diverge and §7.0.2 gains its explanation;
- not engaged → the cards' advantage stays bandwidth+capacity, no ISA chase,
  written down so nobody proposes it again.

Measurement, not datasheet: the profile line is the evidence.

## C. State the MoE roofline bar next to the paged work

Measured 51 t/s (coder, B60) against a ~200 t/s bandwidth ceiling
(active ~1.5–1.9 GiB/token q4 at 456 GB/s) — the MoE runs at ~25% of roofline
while the dense runs at ~60%. Write both numbers into §7.0 so "done" has a
definition, and recompute the ceiling from the *measured* per-token byte
traffic, not the parameter arithmetic — if the measured traffic disagrees with
1.5–1.9 GiB, that delta is itself a finding (router/shared-expert/KV overhead)
and wants a root-cause profile before the ceiling is quoted anywhere.

## D. i4/u4 KV on the paged path — measure, do not adopt

The plugin accepts i4/u4 for `kv_cache_precision` alongside u8. This is a
bandwidth-at-depth lever with a known quality risk. Protocol before any
default changes: full 10-point harness + the depth probe (32k-class prompt) at
u8 vs i4, greedy, both cards; decode t/s at depth 512 and 4096 alongside. A
quality drop is a defect report with the failing case attached, not a reason
to silently stay on u8 — the *measured* delta is the deliverable either way.

## E. Per-slot InferRequest scheduler — promote to a milestone row

Prose in §3 already admits the mutex serializes everything. Make it a
milestone with the existing planned test as its exit criterion: N slots = N
InferRequests (embeddings + MTP twins included), admission bounded by the
measured reservation curve (§7.0.2a terms, per slot), **the 200-request
24-way concurrency stress passes, and single-stream latency regresses < 5%**
with the suite green. Any slowdown beyond that gets a profile naming the
contended resource before any tuning.

## F. Note-only deferrals (no work now)

- NInfer's Hadamard int8-group64 KV codec with fused rotation: the
  custom-kernel endgame for KV bandwidth, with published quality deltas.
  Record it in §8 as the known upgrade path beyond u8/i4 — after the
  fusion-barrier measurement, any custom-kernel proposal must include the
  fusion-impact profile, not just the kernel micro-benchmark.
- MoE speculation revisit: blocked on dense spec-paged numbers; the 1.37×
  verify amortisation must be re-measured on the paged kernels before any
  verdict is reused.
