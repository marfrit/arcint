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

### 0021-fully-connected-kquant.patch

GGUF K-quant weights (Q4_K, Q5_K, Q6_K, Q8_0) served as stored through
the fully-connected path (arcint 0.4.0 stage 1, `docs/design-gguf-
native.md`): arcint builds an op the plugin recognises by type name
("FullyConnectedKQuant", "arcint_opset") over a u8 constant holding the
file's rows, and a new kernel decodes the super-blocks in its inner
loop — no unpack at load, no reorder at compile, no second copy of the
weights. One lane per output column, two variants from one source
chosen by the row count: decode with a broadcast activation read and K
split over four subgroups; prefill on the subgroup matrix multiply
(XMX) with the super-block's activation tile staged in local memory
per work-group of eight subgroups. Eleven correctness cases against a
host reference on both cards, a malformed request refused at shape
inference, the fully-connected suite otherwise unchanged, a timing test
(disabled by name) at the dense model's gate projection. MEASURED: the
dense Qwen3.8-27B Q4_K_M file opens on the dense IR template and scores
10/10 on the Prüfstand through the served endpoint; against Intel's own
int4 IR export on the 24 GB card it prefills 3.2× to 7.6× slower and
decodes at about half the rate (213 / 9.9 t/s at 856 tokens, 174 / 8.5
at 71.7k, the IR 1,609 / 23.1 and 552 / 16.5). The header records the
eight-version ladder that got the kernel here from 28.8 / 3.5. DESIGN
§7.0.2ay.

Upstream: not yet filed.

### 0022-kquant-decode-split.patch

The K-quant kernel's decode variant (arcint 0.4.1 lever 1, `docs/
milestone-0.4.1.md`): the super-blocks packed for the subgroup matrix
multiply on Xe-HPG — the quantised integers become f16 bit patterns with
a shift, a mask and an or, the scale and the offset applied to the
multiply's sums — which makes the 16 GiB card's decode launch 3× faster
(509 → 179 µs at the gate projection); the fused multiply-add path kept
on Xe2, where a one-row matrix multiply occupies the systolic array like
an eight-row one (measured, 46 cycles); and the decode work-group sized
by the projection's width on both, so a narrow projection fills the
card. The served decode rate on the 24 GB card did not move (10.0
against 9.9 t/s). Three readings refuted on the way are in the header.
DESIGN §7.0.2az.

Upstream: not yet filed.

### 0023-kquant-decode-rows.patch

The K-quant kernel's decode variant in llama.cpp's shape (DESIGN
§7.0.2bc): a work-group per group of output rows (four, sixteen on the
long-K down projection), four subgroups with their lanes along K, a
subgroup taking one super-block per iteration. A Q4_K/Q5_K
super-block's 128 quant bytes are one sub-group block read (lane l holds
positions l and l + 16 of every sub-block) and its 256 activations
another, read once and shared by the group's rows; the eight scale/min
pairs are decoded by lanes 0–7 and broadcast. Q6_K's 2-aligned blocks
are block-read as dwords from the dword below and redistributed with one
shuffle (a 16-bit block read two bytes off a dword returns the wrong
word on every lane but the first — measured); on Xe2 the subgroup
prefetches its next Q6_K super-block, one cache line per lane (the
Q4_K/Q5_K shapes and the 16 GiB card lose with the same prefetch and do
not get it). Exact: f32 accumulation over f16 activations. The tiled
prefill variant is unchanged.

Measured streamed at steady state on the 24 GB card against 0022 in the
same instrument: the gate projection 141 µs (355 GB/s, 79 % of the
card's measured random-read ceiling) against 170, the Q4_K down
projection 146 against 219, Q5_K 5,120² 60 against 137, the Q6_K down
projection 397 against 646, the N 1,024 projections 20–33 µs. Served,
dense Qwen3.8-27B Q4_K_M native, `u8` KV: 12.1 t/s decode at 856 tokens
against 0.4.0's 9.9, 10.1 at 71.7k against 8.5, Prüfstand 10/10,
outputs byte-identical at both depths (DESIGN §7.0.2bc–bd).

Also carried: the timing test streams (eight weight buffers in rotation
on one queue; ten launches of one buffer had let a third of it hit the
18 MB L2 and overstated every launch figure of 0021 and 0022), warms
every network to steady state before the clock (a network's second
execution costs twice its third) and runs over the served model's own
tensor types and shapes.

On the 16 GiB card (Xe-HPG), same instrument, against 0022: the gate
projection 154 µs against 159, Q5_K 64 against 98, the N 1,024 shapes
ahead — and the long-K down projections behind, Q4_K 204 against 172
and Q6_K 510 against 467, with the host's rule per architecture (eight
rows × eight subgroups there; sixteen × four, the 24 GB card's, was 224
and 842). That card does not serve a GGUF-opened model of this size.

Built into `+p8` on 2026-09-06 (13 minutes, incremental) and deployed on
the dev host; the IR path's equivalence suite on the 16 GiB card is 9/9
under it (DESIGN §7.0.2be). The per-architecture rule postdates the
package: `+p8` as installed carries the 24 GB card's rule on both; `+p9`
carries this rule.

Upstream: not yet filed.

### 0024-kquant-q6k-tail-block-read.patch

The Q6_K decode row's last twenty bytes (qh's last word, the sixteen
scales, d) as one 16-bit sub-group block read at the dword below them
and broadcasts, in place of four per-lane gathers: six load messages
per super-block per row become three (DESIGN §7.0.2bh). Exact, 14/14 on
both cards, served outputs byte-identical. Measured streamed at steady
state: the 16 GiB card's Q6_K down projection 510 → 385 µs and its
N 1,024 shape 44 → 34; the 24 GB card unchanged (397 → 400, 34 → 34),
where the same window measured the row's arithmetic and shuffles at
~105 µs each and its three reads at 353 of the 400, insensitive to
alignment, layout and dispatch — the next Q6_K form is a load-time
reorder into 224-byte blocks read as two byte-wise block reads, on the
record, not in this patch.

Package: `+p9` (2026-09-07).

### 0025-kquant-q6k-no-shuffles.patch

The Q6_K decode row without variable-index shuffles (DESIGN §7.0.2bi):
16-bit sub-group block reads at the dword below the 2-aligned block put
each lane's own words in place for even blocks, and one fixed-delta
shuffle-down per register does it for odd ones; the scales and `d` are
constant-index broadcasts. The ISA count had put 118 of the row's 296
instructions per row and super-block on word fetches from other lanes.
Exact, 14/14 on both cards, served outputs byte-identical, Prüfstand
10/10. Measured streamed at steady state: the Q6_K down projection
400 → 259 µs on the 24 GB card and 385 → 262 on the 16 GiB card; served,
the mixed form's decode 13.4 → 15.3 t/s at 856 tokens and 9.6 → 11.9 at
71,727 (eight milliseconds off the step at both depths).

Package: `+p10` (2026-09-07).

### 0026-kquant-q6k-224-byte-blocks.patch

A second Q6_K type id (114) whose super-blocks are 224 bytes — the
file's 210 then 14 zero bytes, laid out by arcint at load
(`--gguf-q6k aligned`, the default) so every block is dword-aligned and
0025's decode row takes its shuffle-free path unconditionally; the
same decoders at the wider stride (DESIGN §7.0.2bj). Exact, 16/16 on
both cards, served outputs byte-identical, Prüfstand 10/10. Measured
streamed at steady state: the Q6_K down projection 262 → 204 µs on the
24 GB card (the probe's 183–210 prediction) and 260 → 247 on the 16 GiB
card; the K = 5,120 shapes unchanged. Served, the mixed form's prefill
418 → 531 t/s at 856 tokens and 291 → 335 at 71,727 (the tiled
variant's loads were paying for the alignment too), the decode step
1–2 ms shorter; 6.7 % more bytes on the Q6_K set.

Package: `+p10` (2026-09-07).

### 0027-kquant-fused-ops-run.patch

The plugin's runtime fusion check accepts the K-quant kernel (DESIGN
§7.0.2bk). Until now a fused eltwise on a dynamic fully-connected node
was accepted only on the bf_tiled and reference kernels, so every
K-quant node with a fused residual add was executed through the
unfused-subgraph fallback, whose output read drains the queue: 79
`clFinish` per served decode step, the card idle at each — the "host
time per K-quant node" of §7.0.2bg, named by a call log, a thread-local
and a gdb stack. The kernel's fused ops take the value the unfused path
stored (the sum rounded to the output type), so nothing changes bit-wise;
five correctness cases with a fused residual added. Served on the 24 GB
card: the mixed form's decode step 59.8 → 54.7 ms at 856 tokens (17.2
t/s, Prüfstand 10/10 at 18.4) and 77.7 → 73.3 at 71,727, byte-identical.

Package: `+p10` (2026-09-07).

### 0028-kquant-tiled-a-layout.patch

The tiled (prefill) variant stages its activation tile in the matrix
unit's own layout, so each A operand is one block read of local memory
instead of eight per-lane gathers (128 one-element local-memory
gathers per loop body against sixteen `dpas` in the ISA; DESIGN
§7.0.2bm). Same stores, same arithmetic: exact, 21/21 on both cards,
served outputs byte-identical. The 2,048-row gate launch 39.5 → 34.6 ms
on the 24 GB card; served, the mixed form's prefill 551 → 672 t/s at
856 tokens and 341 → 385 at 71,727.

Package: `+p11` (built 2026-09-07 18:24, installed on the dev host, both units on it).

### 0029-kquant-tiled-2d-block-loads.patch

The tiled (prefill) variant reads both operands by Xe2's 2D block
loads (`cl_intel_subgroup_2d_block_io`): the activation block straight
from global memory in the matrix unit's layout, with no tile staged,
no barrier and no local memory, and the subgroup's sixteen weight rows
by transposed 32-bit block reads, one message per 32 bytes of sixteen
rows instead of a dword gather per lane per dword; the decode runs on
registers through the same arithmetic (DESIGN §7.0.2bn). Xe2 only, for
the dword-aligned layouts (Q4_K, Q5_K, the 224-byte Q6_K); everything
else keeps 0028's staged path. With the loads in place the row tile is
64 in the 256-register mode on Xe2 (the tiled kernel is compiled in
its own batch with `-cl-intel-256-GRF-per-thread`; the decode kernel
stays at 128, where it measured better); Xe-HPG keeps 32 rows and
gains the mode. Exact: 21/21 on both cards, served outputs
byte-identical, Prüfstand 10/10. Served on the 24 GB card, the mixed
form's prefill 672 → 907 t/s at 856 tokens and 385 → 451 at 71,727;
the Q6_K down projection's launch at 856 rows 11.1 → 4.4 ms. The
timing test's operands move to device memory (they were in the
lockable host allocation, which timed the bus, not the kernel: every
tiled figure it gave before this patch is retracted as an absolute),
its rows get valid scales, and it gains the served row count.

Package: `+p11` (built 2026-09-07 18:24, installed on the dev host, both units on it).

### 0030-kquant-tiled-tall-a-reads.patch

The tiled (prefill) variant reads its activation block 32 rows per 2D
message on Xe2 (`intel_sub_group_2d_block_read_16b_32r16x1c`; the
destination holds the rows in order, so each 8-row slice is one row
group's operand unchanged), which takes the activation messages per
super-block per subgroup from 128 to 32 on a 64-row tile; the work-group
is 16 subgroups there (8 on Xe-HPG's staged path); Q5_K takes a 128-row
tile where the 2D loads run and keeps the 8-row read on it (DESIGN
§7.0.2bp). Exact: 22/22 on both cards (a 141-row Q5_K case added),
served outputs byte-identical, Prüfstand 10/10; the decode kernel is
untouched. The timing test in device memory on the 24 GB card, 856 /
2,048 rows: the gate Q4_K 3.37 / 7.18 → 2.86 / 6.16 ms, Q5_K 1.04 /
2.22 → 0.99 / 1.89, the 224-byte Q6_K down projection 4.40 / 9.97 →
3.59 / 8.23, the small Q6_K 1,024 × 5,120 0.295 → 0.250 at 856. Measured
and not shipped: the next super-block's weight prefetch (loses on every
type before the tall read, Q5_K 2.4×; 2–3 % on the gate after it at 4 %
on Q5_K), the next sub-block's activation prefetch (loses everywhere),
16-row reads (54 % worse on Q5_K), a split of the two matrix-unit calls
over the row groups (inert under the tall read). Served on the 24 GB card (arcint 0.4.2, the exact mixed
form): the warm 856-token prefill 940 → 1,001 t/s (first request 903 →
962), 71,727 tokens 451 → 464; the decode steps unchanged (54.8 / 73.7
ms); outputs byte-identical at both depths.

Package: `+p12` (built 2026-09-08 01:00 local on the dev host, installed there at 01:05, both units on it; the served figures above were taken with patch 0030 staged into the +p11 runtime before the package existed).

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
