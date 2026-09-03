# Patches carried against the pinned OpenVINO

Per `DESIGN.md` §1 in the arcint repository, rung 2 of "smallest sufficient
divergence": a numbered patch set **applied at build time here**, not a
divergent checkout. Each patch stays PR-shaped so it can be re-offered
upstream, and this file names what each one is for.

The pin is `2026.4.0-22849-71640275d29` — upstream commit `71640275`. A patch
that does not apply cleanly to that commit is a bug in this directory, not a
reason to move the pin.

## Applied

### 0003-moe-batched-gemv-expert-mask-subbuffer-churn.patch

`prepare_internal_buffers` in the GPU plugin's MoE implementation rebuilds its
per-expert mask subbuffers on **every** inference when `token_num > 1`:
`create_subbuffer` twice per expert, 512 calls per layer, **20,480 per
two-token forward**. Those masks are read only by the per-expert prefill
fallback; the batched-GEMV path that a small-token forward actually takes never
looks at them. At `token_num == 1` the whole block is skipped, which is why
plain decoding never showed the cost.

The patch skips the mask creation below the batched-GEMV threshold and caches
it against `(token_num, buffer)` for the prefill path proper.

Measured on an Arc Pro B60, u8 KV, 300 tokens, temperature 0, output
**byte-identical** to the unpatched plugin in both arms:

| | unpatched | patched |
|---|---|---|
| verify forward wall | 27.3 ms | 18.1 ms |
| MoE host execute per verify | 8.91 ms | 0.74 ms |
| `--mtp on`, prose | 44–46 t/s | 60.5–61.2 t/s |
| `--mtp off` | 62.3 t/s | 61.7–62.3 t/s |

Full derivation: `docs/moe-m2-path.md` in the arcint repository.

Upstream: not yet filed. This is the patch that motivated giving this recipe a
patch path at all; it should be offered upstream, and this line should then
name the PR.

### 0004-moe-otd-perf-counters.patch

Extends the runtime's `[OTD_PERF]` counters (evictions, acquisitions, slot
tiers, staging bytes) so the offload-dial work in 0005-0007 and the host-tier
work in 0011-0012 can be measured rather than guessed at. No served-path
behaviour changes and no throughput number belongs to this patch on its own;
it is the instrument the numbers on 0005-0007 and 0011-0012 below were taken
with.

Upstream: not yet filed.

### 0005-moe-otd-device-resident-slot-pool.patch

Charges each MoE expert slot buffer against a new per-compile device-memory
budget in program order, so slots that fit land in VRAM instead of the
previous host-only upload path. New plugin property:
`MOE_OTD_DEVICE_POOL_BYTES`.

### 0006-moe-otd-async-batched-slot-upload.patch

Batches one `try_acquire_simultaneous` call's misses into one upload and
replaces the per-tensor blocking copy with a non-blocking one out of a
staging ring, waiting on the whole batch once instead of once per tensor.

### 0007-moe-otd-drop-redundant-stream-finish.patch

Drops the unconditional per-MoE-layer `stream.finish()` ahead of the
batched-GEMV top-k read on an in-order queue, where the read's own blocking
copy already waits for everything enqueued before it.

Measured together (0005-0007, on top of 0004's counters): the 35B on the
16 GiB card goes from 0.4 t/s (ratio 25, unpatched) to 9.1 at ratio 50 / 8 GiB device pool (16-token probe) and 10.4 (64-token probe; same patches, longer probe) —
0005 supplying the device-resident pool, 0006 and 0007 the async upload path
and the redundant-finish removal.

Upstream: not yet filed for any of 0005-0007.

### 0008-paged-kv-value-cache-precision.patch

Adds an independent `VALUE_CACHE_PRECISION` config knob alongside the
existing `KV_CACHE_PRECISION`, so the key and value cache can be asked to
compress to different precisions at the config level. Paged-attention
Parameter ports stay 8-bit-typed regardless; the real, possibly sub-8-bit
packing lives entirely at the config level that this patch extends.

### 0009-paged-kv-asymmetric-kernel-plan.patch

The kernel-side asymmetric-KV plan: PARTIAL. The decode fast path's kernel
hard-codes one KV quant type across both operands, so a genuine
sub-8-bit-vs-not mismatch is now refused earlier, at the config level 0008
adds, rather than reaching the kernel and miscompiling.

### 0010-paged-kv-asymmetric-decode-kernel.patch

The per-side decode-read kernel for u8 keys / i4 values, plus the matching
write-kernel split, on the decode fast path. Prefill still declines to the
OCL fallback per 0009's guard.

Measured together (0008-0010): u8:i4 KV costs 8.8 KiB/token against u8:u8's
11.3, auto-fitting 171,312 tokens against 133,456 on the coder on the
16 GiB card and 199,424 against 155,376 on the agent on the 24 GB card — a
+28% context gain. The acceptance task scores 10/10 at u8:i4. Owed: the
prefill price, and prefix byte-exactness.

Upstream: not yet filed for any of 0008-0010.

### 0011-moe-cpu-expert-kernel.patch

The host CPU compute-tier kernel: AVX2 and scalar implementations for the
plugin's grouped-int4 layout, a thread pool, an mmap weight accessor, the
M14 perf counters, and the `MOE_CPU_TIER` property. Compute-only — nothing
in it changes served behaviour until 0012 wires the decode path to it.
Per-source `-O3` is deliberate: the graph library this file lives in builds
at `-Os`, and the kernel needs the higher setting.

### 0012-moe-cpu-tier-decode-split.patch

The wiring: which `(token, expert)` pairs get redirected to the host tier on
a device-slot capacity miss (LRU probe), the OpenCL kernel sentinel skip for
those pairs, and how the host excursion overlaps the GPU work and joins
before `mlp_reduce`.

Measured together (0011-0012): the 35B on the 16 GiB card at ratio 50 /
8 GiB reaches 15.0/15.5 t/s against 10.4/10.6 without the host tier; at
ratio 75 / 5 GiB, 14.1/14.8 against 7.4/7.5. Text output byte-identical
either way, acceptance task 10/10.

Upstream: not yet filed for either of 0011-0012.

### 0013-moe-otd-routing-histogram.patch

Adds a per-expert routing histogram to `OffloadExpertWeightProvider`,
counted before any hit/miss or capacity-dedup logic runs, behind
`MOE_OTD_ROUTING_HIST`. Companion instrument to 0004's counters and to the
0011/0012 host-tier counters, none of which can be used to reconstruct plain
routing after the fact. No throughput number belongs to this patch; it is a
diagnostic.

Upstream: not yet filed.

## Deliberately NOT applied

These live in the arcint repository's `patches/` as records of measurements.
They are listed here so that nobody re-derives the decision by trying them.

- **0001-null-implementation-control.patch** — an instrument, not a fix: it
  forces a null implementation so a node's cost can be measured by removal.
  Shipping it would disable real work.
- **0002-fc-horizontal-fusion-bound.patch** — raises the horizontal FC fusion
  bound. Measured and rejected: fusing the MLP quartet produces wrong output
  (its fourth member is the width-1 `shared_expert_gate`), and with the bound
  restricted to the GDN sets the gain is 66.6 against 66.4 t/s — inside the
  noise. DESIGN records it as "not carried".

## Not carried either: the measurement instrument

The arcint session's working tree also carries per-stage timing accumulators
(`network.cpp`, `primitive_inst.cpp`, `stage_acc.hpp`). Those are how the
20,480 calls were found. They are **not** part of any patch here, and the
build script resets to the pinned commit and applies only this directory, so a
dirty measurement tree cannot leak into a package.
