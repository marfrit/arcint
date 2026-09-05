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

### 0014-gpu-assign-adopts-output-layout.patch

`assign_impl::execute_impl` asserted when a non-`kv_cache` stateful
primitive's variable layout diverged from the assign's output layout — hit
by the DFlash2 draft head's own K/V state chain (`ReadValue` → `Concat` →
`Slice` → `Assign`) at prompts whose first draft concatenates exactly the
window's row count. When the data type and rank still agree, the patch
adopts the output layout (`variable.set_layout`) instead of asserting — the
update a skipped runtime path would otherwise have made — and copies; a
genuine type or rank mismatch still asserts with the same message. A
variable updated normally takes the same path as before.

Measured on the served int4 draft head: the unpatched plugin disables the
drafter at 2,155 / 2,230 prompt tokens; patched, it drafts at those depths
and further out (1,966 / 2,155 / 2,266 / 2,481 / 3,251), byte-identical to
the unpatched plugin everywhere the unpatched plugin does not disable.

Upstream: not yet filed.

### 0015-paged-attention-bounded-partials.patch

Three changes to the GPU plugin's paged-attention implementation:

- **Host-side sizing fix**: `get_internal_buffer_descs` sized `tmp_out` at
  4 bytes/element unconditionally; the kernel already declares it
  `OUTPUT_TYPE` (f16 on every model this repository serves), so the mixed
  stage was allocating exactly twice the bytes its kernels address. Sized
  from the real output dtype instead — no kernel change, same addresses
  read and written, half the term.
- **A bound on the online-merge partition count**: a new read-write plugin
  property, `PAGED_ATTENTION_MAX_PARTITIONS` (0 = unbounded, today's
  behaviour, the default), the way 0008 added `VALUE_CACHE_PRECISION`.
  arcint's engine side is `--paged-attention-max-partitions N`, which reads
  the key back after compile to detect a plugin that carries it, and never
  refuses a load against a plugin that does not. Bounded, the mixed-stage
  scratch term stops scaling with `n_ctx` past a small fixed partition
  count instead of growing forever.
- **The argument rebind fix**: `realloc_intermediates` can replace an
  intermediate buffer's identity without raising the flags that make
  `execute_stage` rebind kernel arguments — harmless while `tmp_out` grew
  without bound on every chunk, live once the bound above makes its size
  plateau and a later genuine reallocation can occur with nothing else on
  that call touching outputs. `PagedAttentionOptImpl` now records the
  identity of every intermediate its kernel arguments were last bound to
  and forces a rebind on any change, closing a use-after-free that only
  appears once the buffer stops growing every call (this patch's own
  header, section "FIX A"/"the argument rebind"; DESIGN.md §7.0.2ac).

Bit-exact at `PAGED_ATTENTION_MAX_PARTITIONS == 0` by construction. The
plugin's 220 paged-attention unit tests pass. At the unbounded setting the
patched plugin is byte-identical to the unpatched one on both cards. On
the 24 GB card, bound 0 against 32: the 64-token greedy output and the
acceptance answer are byte-identical between the two arms and the task
scores 10/10 on both. On the 16 GiB card, the patched plugin at the bound
serves a 119,074-token prompt from a 131,072-token pool that the
unpatched plugin cannot; with a deeper, 165,680-token pool the same
prompt still crashed at the time this patch's header was written (a
separate, pre-existing defect in the asymmetric packed-value path,
patches 0008–0010, tracked in DESIGN.md §7.0.2ac/§7.0.2ad, not this
patch's). **Upgrade note**: this inserts an option into the GPU
model-cache blob's positionally-serialised property list — clear the GPU
model cache (`--cache-dir`, if set) when upgrading to a plugin level
carrying this patch; 0008's own schema guard (`execution_config.cpp`)
rejects a stale-schema blob outright rather than misreading it.

Upstream: not yet filed.

### 0016-paged-attention-intermediate-sizing.patch

`get_internal_buffer_descs` reused the previous call's `num_of_partitions`
whenever `m_rt_params` was already non-null, because nothing resets that
field between calls — the "already computed" branch fires on every call
after the first and sizes the current call's intermediates
(`exp_sums`/`max_logits`/`tmp_out`) from the *last* call's partition count,
not the one about to execute. Harmless while depth only grows a few rows a
step, real once depth grows a full page and the buffers are undersized for
the call about to run. Fixed by sizing from the current call. Regenerated
on top of the corrected 0015 (same three files, same post-image hashes
0015 leaves them at).

Evidence: four unit tests — two red-first review tests
(`paged_attention_review_swa_mirror_test`, the sliding-window shrink
mirrored into the fresh sizing estimate, and
`paged_attention_review_bound_governing_stage_test`, the bound gated on
the stale stage that computed it) plus the two regression tests this
patch was written for (`paged_attention_asymmetric_kv_deep_pool_test`,
asymmetric deep-pool, and `paged_attention_growth_test`,
repeated-execute growth) — plus a byte-identity ladder at 8,418 tokens
(u8:i4, chunk 128, unbounded, 16 GiB card) — 0015+0016 together 6/6
byte-equal to the untouched plugin's output, against the untouched
plugin's own roughly 1-in-14 run-to-run noise. All 19 tests in the
combined filter pass on the 24 GB card. Does not touch, and is not
evidence about, the deep-prompt crash diagnosed in `DESIGN.md` §7.0.2ad as
a driver/runtime fault outside this plugin (a page-fault storm at the
OpenCL runtime's own direct-submission semaphore buffer) — this patch
neither causes nor closes it.

Upstream: not yet filed.

### 0017-moe-cpu-tier-readback-decomposition.patch

Decomposes the MoE host CPU tier's per-layer readback (patches/0011-0012)
into named counters — `cpu_topk_id_ns`/`cpu_x_enq_ns`/`cpu_x_wait_ns`/
`cpu_x_drain_ns`, a warm/steady split, and an env-gated queue-drain probe
(`MOE_OTD_READBACK_PROBE`) — instead of one folded `avg_cpu_x_us`. Moves
the x (hidden_states) and routing-weight readback destinations from
pageable `std::vector` buffers to `usm_host`, and hoists that readback
into the existing topk_id round trip so a tier-on layer pays one combined
wait instead of two (`MOE_OTD_READBACK_NOHOIST=1` restores the old,
separate order for isolating the hoist's own effect). No numeric
behaviour change on any path: tier OFF is byte-for-byte the pre-patch
code, tier ON runs the exact same `moe_cpu_expert` kernel over the same
bytes, sourced from `usm_host` instead of `std::vector`. MEASURED on the
16 GiB card at the M14 tier cell: the 283 µs readback attributed in an
earlier record was mostly the x read landing on pageable memory after the
queue had already drained on the topk_id read — `usm_host` cuts it to
53 µs; the hoist alone is a null. Full derivation and the retraction of
the cell's own decode-rate record (a device-pool env var was silently
unset for that window) are in the arcint repository's `CHANGELOG.md`
under "Unreleased" and `DESIGN.md` §7.0.2af.

Upstream: not yet filed.

### 0018-moe-cpu-tier-static-partition.patch

Replaces the MoE host CPU tier's process-global LRU expert residency (F0,
patches/0011-0013) with a static partition (F2): for each MoE layer, the
resident set is the `slots` experts with the smallest
`splitmix64(seed, layer_key, expert)` rank, fixed once at `bind()` and
independent of every subsequent request, history, or arrival order. This
closes a real DESIGN §3.4 violation — the LRU tier picked device-f16 vs.
host-f32 arithmetic by residency, so greedy output depended on the
process's request history, not just the request itself; a continuation
restored from the prefix cache could fork from the same continuation
served cold. An earlier draft (F1, a bit-equal host kernel matching
device arithmetic exactly) was retired by review as impractical — F2 does
not make host and device arithmetic agree, it makes the *set* of experts
each one runs on independent of history, so the mismatch has no
opportunity to depend on it.

Five load-time bugs surfaced getting the allow branch to actually pass,
not just compile — three fixed on the arcint side (the device-pool
plateau probe and the prefill fallback both assumed an
evictable/acquirable slot always exists, false under a 100%-pinned pool;
`ARCINT_FIT_SLOT_BYTES` could bypass the `--prefix-cache-mib` refusal
entirely) and two fixed here in the plugin: the refusal gate queried the
static-partition property on the wrong (pre-compile) object, and the
property's own default never checked whether the tier was even on,
reporting the partition active on every load regardless. Full account,
file:line, in this patch's own header and in the arcint repository's
`DESIGN.md` §7.0.2ae. MEASURED, `tests/equivalence/run.sh` against
`--moe-cpu-tier --prefix-cache-mib 4096 --kv-block-size 32` on the 24 GB
card, twice (once per plugin rebuild fixing the two plugin-side bugs):
**all checks pass**, continuation-restore included — the check this
patch exists to fix.

Not fixed, investigated and reported instead of guessed at: a false
return from `on_load_expert_weights` is genuinely overloaded (OTD off
vs. not resident under this partition) and its one caller does not
distinguish them, but the ambiguous path is reachable only when
`MOE_USE_GROUPED_GEMM_PREFILL` is forced off, which arcint never does —
dormant in every configuration this repository drives today. *(Fixed by
0019, below.)*

Upstream: not yet filed.

### 0019-moe-prefill-fallback-tristate.patch

Closes the item 0018's header reported. `on_load_expert_weights` now
answers three ways — no offload tier (every expert's weights are on the
device as initialised), a device slot acquired or pinned, or the host
tier (not resident under the static partition) — and the per-expert
prefill loop takes the device path for both device answers. Under 0018 a
resident-only load reaching that loop, which needs both fast prefill
paths forced off through internal properties arcint never sets, took the
host branch for weights that were on the device through a downcast of the
wrong provider type. An assertion now guards the downcast.

Red first: a new plugin unit test, `moe_3gemm_prefill_fallback.resident_
load_with_grouped_prefill_off_matches_reference` (40 tokens, both fast
paths off, no offload, checked against the suite's own reference), failed
on the 0018 tree — the misread provider tried to map a weight file that
does not exist — and passes with the patch; the four
`moe_3gemm_static_partition.*` and sixteen smoke accuracy cases pass
alongside it, 21 of 21 on each card, 2026-09-05. The three acceptance
cells the campaign named as the no-change proof were not run for this
patch (a quick functional test, on the operator's word); the branch is on
none of their paths. Full account in the patch header and arcint's
`DESIGN.md` §7.0.2ap.

Upstream: not yet filed.

### 0020-paged-kv-asymmetric-micro-sdpa.patch

The served asymmetric pairing — u8 keys by channel, i4 values by token —
ran its prefill on the generic paged-attention kernel because 0009
declined micro-SDPA for any key/value pair of differing packing classes,
and paid +55 % / +90 % prefill time at 37.7k / 71.7k tokens against u8
at the same chunk (DESIGN §7.0.2ar). The generator now sets the value
operand's type and layout from the value precision, the kernel source
gates each side's four-bit layout on its own macro, the value pointer's
per-chunk advance derives its packing from the value cache rather than
the new-token input port, and the selector admits eight-bit keys with
four-bit values (four-bit keys with eight-bit values, and four-bit
values under BY_TOKEN keys, still decline — the latter measured as NaN
past 128 keys, not diagnosed, and not the plugin's default).

Red first: a new mixed-stage unit test with u8 keys and u4 values failed
on the 0019 tree with the dump naming the generic kernel, and passes
with the patch (3/3 shapes, the float reference at 1e-2); 0015's
asymmetric prefill regressions at a 2,048-token past now run on
micro-SDPA and match; 276/277 of the paged-attention set pass.
MEASURED on the recipe-built plugin, 16 GiB card, coder, chunk 128:
u8:i4 459 against u8 457 t/s at 37,707 tokens and 401 against 398 at
71,727 — parity — with the u8:i4 outputs byte-identical to the generic
path's and the Prüfstand 10/10 through the u8:i4 server. The values
stay four-bit in VRAM; the microkernel unpacks them in registers.
DESIGN §7.0.2as.

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
