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

### 0031-fc-deterministic-gemm.patch

The f16-activation compressed fully-connected — the form a GGUF-opened
model's repacked projections take — sets oneDNN's deterministic
attribute before its primitive descriptor is built. oneDNN's gemm
selector scores a k-parallel strategy (split-K across work-groups with
atomic accumulation) best whenever the plain M × N tiling underfills the
device, and that reduction's order varies run to run: the same 235-token
prompt to one process gave five different greedy texts over twelve
requests, on 0029 and 0030 alike, with MTP and the logits slice on or
off, while 64 and 856 rows were byte-stable and the IR at 235 was too
(DESIGN §7.0.2br). With the attribute the 235-token prompt is one text
4/4, and the served rates are unchanged (856 tokens warm 1,008 t/s and a
54.1 ms step; 71,727 tokens 464 t/s and 73.0 ms; the outputs the same;
at 85 and 145 tokens the text changes with the kernel, chosen knowingly).
Found by the equivalence suite, which runs on a GGUF-opened model since
its stateful section became skippable. Open after it: a two-text
alternation at 85 tokens on the GGUF path (the native form too, so not
this gemm), and a process fault at about 190–215 tokens in the mixed form
at the default prefill chunk (an engine memory CAT error at 16.5 GiB
resident; not at `--prefill-chunk 64`, not in the native form, not in the
IR) — both in the patch header, neither fixed.

Package: `+p13` (built 2026-09-08 05:14 local on the dev host, installed there at 05:17, both units on it; the served figures above were taken with patch 0031 staged into the +p12 runtime before the package existed).

### 0032-sdpa-micro-k-prefetch-bounds.patch

The micro-SDPA prefill's two cooperative K-tile prefetches in upstream
master's form (openvinotoolkit/openvino PR #37878, merged 2026-09-08,
eighteen days after the pin). The pinned nightly prefetched the *next* K
tile with its geometry in oneDNN's transposed-K order -- the remaining
keys as the row length, d = 256 as the row count, at a stride of one
element -- so its pointer landed inside row 0 of K and it walked 256
rows of up to 256 B from there whatever the tile, unclamped (the
helper's clamp comes from the same swapped geometry): a prefill chunk
of N keys with a next tile, 129 to 255 keys, read 256 − N rows past the
end of K; 256 keys and beyond are in bounds. Whether the pages behind
the K buffer were mapped decided between a served prompt and an engine
memory CAT error with a compute-engine reset -- the "190–215-token
fault" of 0031's header (190–214 served faults; the reproducer faults
at 193–217), which was never the gemm. The patch fixes the
stride (ldk unless TRANSPOSE_K), the row length (d) and the row count
(the remaining keys) for both calls, and adds a regression test: the
paged-attention primitive alone, the served geometry (24 heads, 4 KV
heads, head 256, u8 KV by channel), one subsequence of 193–217 new
tokens on exact-size buffers -- red on the pinned nightly (faults or
hangs), 11/11 green with the patch (DESIGN §7.0.2bs). Found with the
intercept layer (the launch pinned, every argument and its buffer
dumped, all correct), the disassembled micro-gemm blobs (their loads
are surface-bounded), and one environment switch per access class in
the generator (the host prefetches off: 5/5 pass). Served on the 24 GB
card: 190/205/211 tokens six requests in one process, one text, where
a fresh process died on its first request before; the plugin's
paged-attention and SDPA suites 264/264. Rates unchanged: 856 tokens
warm 1,008–1,009 t/s against 1,010 on +p13, the decode step 54.5–54.7
ms against 54.7, the same greedy text, four requests each.

Package: `+p14` (built 2026-09-08 09:14 local on the dev host, installed
there at 09:15, both units restarted onto it and serving; the served
figures above were taken with patch 0032 staged into the +p13 runtime
before the package existed).

### 0033-sdpa-micro-value-alignment.patch

The micro-SDPA generator gives the V*S micro-gemm the packed row's
alignment whenever the *value* precision is four-bit. Under u8 keys with
i4 values -- the served pairing -- it used the f16 row's, 128 bytes for a
132-byte row (68 at head 128; `alignment_for_ld` returns the lowest set
bit of `head * 2`, capped at 128): patch 0020 keyed the operand's type on the value
precision and left the alignment on the key's. A gemm strategy told its
rows are 128-byte aligned addresses them accordingly, and what it read
depended on the physical pages a request happened to get: the agent
configuration's greedy text alternated by request parity (the page pool
is a stack -- odd requests get an ascending run of pages, even ones a
descending run) and was wrong from the first request, MTP on differing
from MTP off (DESIGN §7.0.2bu). One condition fixes it.

The plugin test of that exact shape was green for two reasons, both
corrected here: the harness built only ascending contiguous block tables
and addressed cache pages as `start + j` in ten places (now through the
table, with `page_order` per test: reversed, or a gap after page 0 -- the
served pool's third request), and its fill made every page look alike
(all past keys −1, past values 0, a query of 8: a one-hot softmax on each
page's last token). The fill now gives every token, head and page its own
key (within 4e-4 of the u8 by-channel grid) and value (all sixteen four-bit
levels in every token-and-head row, stored exactly), with a query of 1/64; the served geometry (24 heads, 4 KV heads, head 256) is in the
cases under the three page orders. The harness also packed past tokens'
four-bit values as (dim, dim + 16) pairs where the production writer and
the readers use adjacent pairs -- invisible while every dim of a token
carried the same value; corrected.

Measured 2026-09-08 on the 24 GB card. Plugin test, full statistics over
every element against the float reference: before the line the u8:i4
cases err by 2.0–2.3 on 98–99 % of elements (value range 15) and the
served three-token case hangs the test binary; with it exact: at most 0.001 on every case, none over 1e-2. The same
fill through f16 KV and through u8 KV is exact to the tolerance. The tests'
tolerance stays at 1e-2. Retracted on the record (DESIGN §7.0.2bu): a first
fill with eight levels per row measured its own quantisation as a 0.11
"floor" of the four-bit value path; the review's arithmetic reproduced the
element map from the fill, and sixteen levels per row removed it.
Served (arcint 0.4.3, the IR agent model, u8:i4, MTP on, one process): 130
tokens six times, 825b1747 ×6 = the MTP-off text (before: two texts
alternating, neither the MTP-off one); 8,005 tokens twice, one text;
prefill 865/870 t/s against 869/874, decode at 8k within the noise. The
equivalence suite on the fixed plugin passes every gate including the new
one (MTP at u8:i4, three requests of one process byte-identical); on the
unfixed plugin that gate fails and the rest pass. The whole paged-attention
suite passes under the default and a reversed table, the SDPA suite too.

Package: `+p15`.

### 0034-sdpa-micro-tail-test-and-by-token-reproducer.patch

Two tests, no kernel change; arcint's runtime floor stays at +p15
(DESIGN §7.0.2bv).

The **single-query micro-SDPA tail test** (5 cases) asks whether the output
depends on what lies *past* the sequence length in the K/V allocation: K and V
as a 129-row view of a 151-row allocation whose tail holds NaN (or 65504),
against the same rows in an exact allocation. It is the non-paged neighbour of
what 0032 fixed on the paged side, where the prefill's next-K-tile prefetch
walked 256 rows from inside row 0 of K. A served drafter head (24 heads, head
256, 129 keys) reuses a grown buffer on its second request, so a kernel reading
past the sequence length would change its draft with no input changing. Green
5/5 on the 24 GB card; it had been carried in no patch.

The **by-token reproducer** is DISABLED, and it is why patch 0020's decline
(four-bit values under BY_TOKEN keys) stays in place. Re-measured 2026-09-08
with 0033's discriminating fill under the three page orders, on the 24 GB card,
routing read from the dispatched kernel list rather than assumed: the by-token
cases fail as *total NaN* on the generic kernel -- and so do **eight-bit**
values under by-token keys, a pairing that decline does not govern and which
runs on micro SDPA. Two kernels, one fill, the same all-NaN output — something the two runs share. The fill is
not the explanation: at the test's default geometry (32 heads, 2 KV heads, head
128) the by-token cases are exact at 36 keys (max 0.002, no NaN, both value
precisions) and all-NaN at 132; at the served head 256 the NaN is a quarter of
the elements already at 36 keys, three quarters at 102, all from 126. At 132 keys and beyond the failure is total at both value
precisions and on both kernels; the partial gradient below 132 was measured on
four-bit values and the generic kernel only. **No mechanism is claimed** -- the common element
(the harness's own by-token page writing, its cache sizing, or the key
dequantisation both kernels share) was not measured. The re-test therefore
cannot reach a verdict on this instrument, since the instrument fails in a
configuration the decline does not govern. The six cases that carry the finding
ship disabled so the next reader starts from the measurement
(`--gtest_also_run_disabled_tests`).

Nothing served is affected: arcint never selects BY_TOKEN keys (the plugin
defaults to BY_CHANNEL and forces BY_TOKEN only for a graph with cache-block
rotation, which arcint's do not carry, DESIGN §7.0.2br), and every by-channel
case is exact. The dispatch assertion is now conditional for that reason: the
by-channel cases still assert micro SDPA, the by-token ones print the kernel
they actually got.

Not retracted, not reproduced: 0020's own note recorded this pairing as NaN
"for every query whose causal context passes 128 keys, and only those",
measured 2026-09-05 on the staged tree with that patch's own by-token test and
the fill of the time. A quarter of the elements NaN at 36 keys is not that
pattern; the two were taken on different fills and geometries, and which
difference accounts for it is unmeasured.

Measured on the patch's exact content, a plugin test build reconfigured with
`ENABLE_DEBUG_CAPS=OFF`: the whole paged_attention and sdpa suites 320 ran,
280 passed, 40 skipped (pre-existing vlsdpa), 0 failed; the disabled
instantiation contributes 0 runs. Served unchanged: 825b1747 on three requests
of the agent configuration's 130-token prompt, and the equivalence suite passes
on both units' configurations.

### 0035-sdpa-by-token-test-key-fill.patch

Test-only: fixes the BY_TOKEN test harness's key fill (constant per-dimension →
per-dimension linear ramp, the same correction FIX 2 applied to the value fill).
No kernel change, no served behaviour change. Part of 0034's measurement pass.

### 0036-sdpa-micro-flash-next-kv2-geometry.patch

RED-C-03: Flash-Next's full-attention geometry (24 query heads, 2 KV heads,
head_dim 256) through the u8:i4 paged-attention/micro-SDPA regression harness.
Every prior cell uses `(24, 4, 256)` (dense/agent) or `(16, 2, ~128)`
(MoE/coder); Flash-Next is a third combination — GQA group size 12 — that no
test exercises. One factory function (`u8i4_mixed_micro_flash_next`, identical
to 0033's `u8i4_mixed_micro_served` with `num_kv_heads` changed from 4 to 2),
seven instantiations under ascending, reversed and gapped page tables, the same
patterns 0033 exercises for the served shape. No kernel change; 0033's
infrastructure (discriminating fill, permuted page tables, float reference,
1e-2 tolerance) carries this shape as-is.

**MEASURED 2026-09-09: 7/7 cells PASS.** RED-C-03 CLOSED GREEN. The "fits,
unverified" verdict for 0010/0020/0032/0033 at Flash-Next's shape becomes
"fits, verified by measurement." Three pre-existing BY_TOKEN failures in
the 0035 suite (original geometry, "reverse" page order) are the known
block-size gate from 0034, not flash-next related.

### 0037-moe-hybrid-prefill-split.patch

Hybrid grouped-GEMM/host prefill for the MoE static partition. Under a
static half-partition every MoE layer's routed batch contains at least one
non-resident expert, so the grouped-GEMM prefill path refused every batch and
fell back to the serial per-expert loop (`grouped_fallbacks=40×layers`,
§7.0.2ai). The fix wires `cpu_tier_misses` into both grouped-GEMM callers
(`on_before_prefill` for the micro-GEMM path, `build_grouped_mask_otd` for
the grouped oneDNN path). Non-resident experts receive `kCpuTierSentinelSlot`
in the lease; the grouped-GEMM proceeds over the resident subset, and the
non-resident subset is dispatched to `moe_cpu_expert` on the host after the
GEMM completes. `get_expert_mask_from_gpu` skips the exact sentinel instead
of throwing; any other out-of-range value still throws (the shape-predictor
overflow guard). The scatter-reduce kernel's pre-existing UINT_MAX sentinel
handling covers the new slot; no GPU kernel change.

New OTD perf counter: `hybrid_prefill_layers` — the number of grouped-GEMM
invocations that took the hybrid path instead of the full per-expert fallback.

Design note: `docs/design-static-partition-prefill.md`. Campaign:
`docs/campaigns/static-partition-prefill.md`.

**MEASURED:** see DESIGN §7.0.2bx. The coding defect is eliminated
(`grouped_fallbacks` 400→0), §3.4 identity and E2 pass, decode improves
(18.2 t/s, above gate of 14.8). The campaign's prefill gate (within 25% of
OFF) is **not met** — the host dispatch for non-resident experts serialises
through ~128 experts per layer and dominates prefill time. The campaign
remains open.

### 0042-moe-hybrid-prefill-gather-filled-count.patch

The fix for patch 0037's page fault on the Arc Pro B60 (Xe2). Under the
hybrid prefill split the grouped-GEMM tables hold only the resident
(token, k) pairs and the rest of `tokens_per_expert_cpu` stays -1, but the
gather kernel was still launched over `token_num * max_topk` work-groups;
every work-group past the fill computed `token_index = -1 * HIDDEN_SIZE`
and added it to the kernel's `uint` offset, which wraps to ~2^32 elements,
~8 GiB PAST the buffer — the B60's faulted address (0x1f0f5e000, ~8.2 GiB)
is consistent with that; why the A770 never faulted on the same read is
NOT explained on the record. The gather
(and the stages it sizes) now runs over the filled count, the token tables
are zero-initialised on both prefill paths (the micro-GEMM path carries the
same over-sized launch, latent for arcint: grouped on), the GPU mask-gen's
device table is zeroed before the kernel, and a batch with no resident
expert takes the per-expert path as before 0037, counted as a grouped
fallback. Known, not fixed: the filled count is also the oneDNN grouped
primitive cache key, routing-dependent on the hybrid path (rebuilds per
prompt; bucketing owed).

Bisected on the served binary (2026-09-17): +p13 serves, +p16 faults,
+p16 without 0037 serves, the LRU partition serves, and within 0037 a
`stream.finish()` after every grouped-path stage showed the first
synchronisation after the gather already throwing. Campaigns:
`docs/campaigns/static-partition-prefill.md`,
`docs/campaigns/sub4bit-vram-kernel.md` (status 2026-09-17).

**MEASURED:** the 35B at `--offload-ratio 99 --moe-cpu-tier` on the B60
serves Paris, warm repeat identical, decode 23.6 t/s (B60, KV u8, f16 inference, prefill chunk 512, one lane), the
hybrid path active; no fault. Owed: the unit-ladder cell (tables with sentinel entries,
filled count against launch size).

Package: `+p18` (built 2026-09-17 20:31–20:43 local on the dev host from tree 83701d6; the packaged plugin's own cell on the B60 — the 35B at ratio 99 with the tier, KV u8, f16 — serves Paris at 23.3 t/s; not installed on any host by the seat that built it).

### 0043-native-expert-formats-through-the-tier.patch

The checkpoint's own expert blocks computed as they are, instead of the u4
grouped-affine repack that costs 0.10–0.13 relative RMS per expert tensor
and 0.73 nats at depth 48 against the model's own llama.cpp logits
(DESIGN §7.0.2bz; `docs/design-routing-aware-expert-execution.md`
§2.3a–d). The serving-shape emitter carries each expert weight in the
fused op's rank-4 group-32 layout — IQ4_NL / IQ4_XS as u4 codes plus an
f16 per-32 scale (a 16-entry table decode), IQ3_XXS as u8 grid indices
plus u8 sign indices in the zero-point slot plus the f16 scale, Q8_0 as i8
codes plus the f16 scale — and decodes them in standard ops. This patch:
(1) three pattern blocks (`pattern_blocks/native_expert_block.*`) and a
pass `ConvertTiledMoeBlockNativeToMoeCompressed` beside the stock tiled
matcher, lowering the three chains straight to `MOECompressed` with a
`weight_format` per projection in the config (visited, so it serialises;
the shape checks relax at the weight's last dimension and the zero-point's
under a native format); (2) the tier executes every routed expert under a
native format (`_native_tier_only`: the batched-GEMV path forced, every
expert a sentinel, the fused kernels never launched) with three row
decoders in `moe_cpu_expert.cpp` (the tables verbatim from llama.cpp's
ggml-common.h, pinned in arcint's `src/core/gguf_dequant.cpp` against
gguf-py on the real shards); (3) the new source is listed in the
transformations library's `sources.cmake` (no glob there — a source not
listed is silently not built); (4) three host-only cells in
`tests/unit/test_cases/moe_cpu_expert_test.cpp` (`moe_cpu_expert_native.*`)
pin the three row decoders to hand-built blocks through
`compute_stage_f32` (declared for them, outside the anonymous namespace).
Reviewed before packaging (2026-09-18): the first form had the three
native-format members missing from `clone()`'s field list (the executing
impl would have run affine on native bytes — patch 0038's defect one
patch earlier) and read `_native_tier_only` before assigning it; both
fixed, the format is read off the primitive's config at the top of the
constructor. The cells run without `ENABLE_TESTS`: compile the test file
with `moe_cpu_expert*.cpp`, gtest from `thirdparty/gtest`,
`-DOV_MOE_CPU_TIER_HAVE_AVX2 -mavx2 -mfma -mf16c`, the source tree's
`src/inference/dev_api` on the include path, linked against the built
`libopenvino` (8 cells, all green on the dev host). The OpenCL decode in
the fused and per-expert kernels is the next patch.

Two more relaxations found on the card: the impl's static
`validate_impl` refused an f16 zero-point slot (IQ4_NL / Q8_0 gate-up: no
impl, "No layout format available"), and the offload runtime's payload
transpose asserted one byte per group in the zero-point slot (IQ3_XXS: four
sign indices per group) — under a native format that slot is neither
type-checked nor transposed; only the tier reads it, from the file.

**MEASURED (2026-09-18, `tests/python/test_native_lowering_gpu.py`, tree
dffd272, plugin 5a6968ec):** on BOTH cards — Arc A770 (GPU.1) and Arc Pro
B60 (GPU.0) — the stock-affine control fuses (`moe_router_fused` +
`moe_3gemm_fused_compressed`, corr 0.999999 against the CPU plugin) and
the two native pairs (IQ3_XXS/IQ4_NL, IQ4_XS/Q8_0) lower to
`MOECompressedNative`, run every routed expert through the tier and match
the CPU plugin at corr 1.000000, max diff at 0.19 / 0.15 of the band the
control's f16 noise calibrates; peaks vram0 94 MiB. The first form of this
patch (before the review's clone-list fix) had wedged the B60 at the
process's first job — with that form the executing impl ran the fused GEMV
over native-layout bytes; the wedge has not recurred since the fix on
either card (three legs), which is consistent with, not proof of, that
mechanism. Owed: the served depth-4 and depth-48 native artifacts through
the tier, the KLD gate's native reading.

### 0044-moe-otd-routing-trace.patch

A per-call routing TRAIL beside patch 0013's aggregate histogram, so the
0.5.2 VENICE census can be taken from the SERVED path (campaign
`docs/campaigns/expert-hot-set-lru.md`, design
`docs/design-expert-hot-set-lru.md` §4b). Patch 0013 answers "which experts
route" but not "in what order", so it cannot feed the per-layer LRU replay;
an aggregate is not a trace. This patch adds an opt-in env
`MOE_OTD_ROUTING_TRACE=<path>`: each `OffloadExpertWeightProvider` reads it
once at construction (same per-provider, construction-time discipline as
`MOE_OTD_ROUTING_HIST`), and `try_acquire_simultaneous` appends one line

    <call_seq> <layer_key> <top_k> <expert id...>

at the SAME point patch 0013 counts (before the dedup/hit-miss split, so it
records what the router picked, not what the pool served). `<call_seq>` is a
process-wide atomic counter under a mutex, and each line is flushed
immediately so a `SIGKILL`'d window keeps every record already written (the
aggregate dump only runs at exit). `layer_key` is the structural weight-file
offset (patch 0018's key), so the trace is independent of the
construction-order `layer_seq_id`.

The offline half is `tools/hot_set_census.py`: `parse_call_trace`,
`split_topk_chunks`, `layer_key_index_map`, `call_trace_to_v1` and the
`from-call-trace` subcommand. Both silent-if-wrong assumptions have
red-first cells in `tools/test_hot_set_census.py`: a call carrying two
tokens' ids splits into the right `top_k` chunks and a mis-sized call is
REFUSED, not truncated; and the `layer_key` -> decoder index map is the
ascending export order by default, while an exported map with a duplicate
index or a missing key is refused. A call carrying more than one token's
ids is refused by the decode converter (per-token `token_idx` is undefined
for a batched/prefill call in this trace), which is the stated caveat;
`from-call-trace --skip-batched` skips and COUNTS the opening prefill call
instead (an all-batched trace is still refused, never an empty census), and
the CLI requires a `--provenance` file with a non-empty `artifact_sha256=`
(or `artifact=`) and `card=` -- the artifact/card part of §2's header is
enforced, the rest is not machine-checked here.

MEASURED (2026-09-21, dev build host): applied on top of the 41 patches
against pin `71640275` and built (`ninja openvino_intel_gpu_plugin`); the
third-prefix install reports plugin version
`2026.4.0-22849-71640275d29-marfrit-p19` and carries the `routing_trace`
string. NOTE: that stamp is deliberately left at `p19`, which the packaging
record already uses for patches 0003-0043, so the stamp alone cannot tell a
0044 build from a 0043 one; the trace build is identified by its
`routing_trace` symbol, and a future window must cite the symbol, not only
the version string. The offline cells are 66 green (`tools/test_hot_set_census.py`,
including the corpus-split census and the raw-`layer_key` join).
OWED: the served card window (the census's own authority) and its
stability statement; the measurement plugin and the debug-caps install are
untouched, the new plugin lives in its own prefix.

### 0045-native-expert-ocl-decode.patch

The OpenCL decode of the checkpoint's own expert blocks, inside the per-expert
kernel (campaign `docs/campaigns/sub4bit-vram-kernel.md`; design
`docs/design-routing-aware-expert-execution.md` §2.3a–d, step 3). Patch 0043
carried IQ3_XXS / IQ4_NL / Q8_0 experts into the fused op and ran every routed
expert on the scalar CPU tier, with an in-code assert that did so "until the
OpenCL decode exists"; this is that decode.

`moe_expert_swiglu.cl` gains the three decode tables (the ones of the
repository's `src/core/gguf_dequant.cpp` / `tools/q4e/native_blocks.py`,
llama.cpp `ggml-common.h` pinned clone 56b9eb28) as `__constant` arrays and two
entry points, `expert_gate_up_native` and `expert_down_native`, with the same
dispatch geometry and argument list as patch 0040's `expert_gate_up` /
`expert_down`. The weight bytes are decoded per element inside the K-loop — no
dequantised row is ever written to memory. The per-tensor slot strides are the
tensor bytes divided by the expert count (`expert_tensor_span`), i.e.
`INTERMEDIATE_SIZE*HIDDEN_SIZE/{2,4,1,8}` by role and format; the scales arrive
transposed to `[groups, oc]` by `maybe_transpose_scale_zp` exactly as the
affine path's do, while the IQ3_XXS sign indices are copied row-major (0043
skips the native zero-point transpose).

`moe_3gemm_swiglu_opt.cpp` compiles those stages for a native config, lifts
0043's "native + per-expert dispatch not combined" refusal, and dispatches the
native stages for the resident experts while the misses keep going to the CPU
tier (patches 0011/0012) exactly as before — the routing-aware split patch 0040
built. A gate/up projection is refused at stage compile unless its format is
IQ4_NL (1) or IQ3_XXS (2), rather than silently running the IQ4_NL decode over
Q8_0 bytes.

The same arithmetic is pinned device-free in the arcint repository before any
card: `tools/q4e/native_expert.py` is the CPU reference (decode fused with the
dot) and `tests/python/test_native_expert_gemv.py` its ladder — block-scale
application per format, the fused-vs-materialised equality, a K that is not a
multiple of 32 REFUSED, an unknown format REFUSED, and a deliberately wrong
(affine) reading of IQ4_NL's bytes that must be caught rather than silently
absorbed.

MEASURED (2026-09-21, dev build host): applied on top of the 44 patches
against pin `71640275` and built (`ninja openvino_intel_gpu_plugin`, clean);
the plugin carries the `expert_gate_up_native` / `expert_down_native` symbols
and the native tables. The version stamp stays deliberate at `marfrit-p19`
(0003–0043's stamp), so the 0045 build is identified by its
`expert_gate_up_native` symbol, not only the version string. The device-free
cells are **16 green** (`tests/python/test_native_expert_gemv.py`).

The per-expert `.cl` also had a **pre-existing** build blocker, found on the
card: the plugin compiles a primitive's kernels into ONE program, so the
`.cl` body appears once per kernel and its file-scope helpers
(`expert_gate_up_gemv_u4`, `expert_down_gemv_u4`, `load_x_interleaved`)
were defined twice -> `clBuildProgram` `CL_BUILD_PROGRAM_FAILURE` on xe2.
That is why patch 0039/0040's per-expert kernel had never built on a card
(the 2026-09-17 record). 0045 wraps every file-scope helper in a persistent
`#ifndef` guard so the concatenated copies define them once; the native
kernel then compiles (`lgc load: language model ready`, device-resident
8.06 GiB).

MEASURED (2026-09-21, 24 GB card GPU.0 = PCI 8086:e211, native d48n, ratio
99 and 80, per-expert dispatch): the model **loads and compiles**, but the
served per-expert path then **faults before the HTTP server comes up** --
`xe ... Faulted Address 0x0000d556aa740000, Fault response: Unsuccessful
-ENOENT` on the blit engine (`EngineClass: 3 bcs`), engine reset, and an
`arcint` `segfault ... in libc.so.6` (memcpy) at the same instant; no
`per_expert_gpu_invocations` was measured. The fault is a finding to
localise (the upload/gather/sentinel path, not the decode arithmetic,
which the device-free cells pin), recorded in the campaign status and the
handoff.

OWED: the served native per-expert reading -- a fix for the card fault,
then the GPU-dispatch counter, rate and correctness.

## Deliberately NOT applied

> [CORRECTED 2026-09-23: only **0001** and **0002** below are genuinely not
> applied — they live at the repository top level and not in this directory,
> so `build-openvino.sh`'s `patches/*.patch` glob skips them. **0046**, **0047**
> and **0048** are in this directory and ARE applied by that glob, in numeric
> order. The heading above is stale for those three entries; they are kept in
> place with this date rather than moved, so the correction stays visible.]

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

### 0046-moe-cpu-tier-census-seed.patch

The census-seeded static partition for the MoE host compute tier (campaign
`docs/campaigns/expert-hot-set-lru.md`, design
`docs/design-expert-hot-set-lru.md` §5.1). Patch 0018 chooses a layer's
resident expert set with a frequency-FREE `splitmix64(seed, layer_key,
expert)` rank, measured (`tools/expert_policy_compare.py`) to sit at CHANCE at
every budget (0.92–1.10× `slots/512`). This patch replaces that ranking with a
measured-frequency rank from a served-routing census, so the pinned half is
the hot half, while keeping DESIGN §3.4: the seed is a pure function of the
RECORDED census, not of run history.

New file `census_seed.hpp` (deliberately OpenVINO-free): a parser for the
"hot-set seed v2" format, one data line per layer

    <layer_key> <expert> <expert> ...

with a MANDATORY `# space=layer_key` header. Malformed lines, duplicate
`layer_key`s, duplicate expert ids, a missing or wrong `space=` header, and an
empty file are REFUSED (`std::runtime_error`), not defaulted. The consumer
half, `census_seed_resident_experts()`, validates one layer against the model:
the `layer_key` must be present, the expert count must equal the pool
capacity (a budget that moved since the census is a mismatch), and every
expert id must be `< num_expert`.

`expert_weight_providers.{hpp,cpp}` gains `set_census_seed()` /
`census_seed_active()`; `bind()` pins the census set when active, else patch
0018's splitmix64 rank. `moe_3gemm_swiglu_opt.cpp` reads the per-run env
`MOE_CPU_TIER_SEED=<path>` once per process (cached across the 48 layers),
validates THIS layer's entry at construction -- so a mismatched seed REFUSES
THE LOAD before any request is served -- and logs
`seed_source=census|census_seed_fp=0x...` beside the existing
seed/`resident_checksum` fields. With the env var unset, patch 0018's
behaviour is unchanged.

`tools/hot_set_census.py`'s `select` now emits this format: `--census
<layer,expert,count CSV>` (the CORPUS census a hot set is seeded from, not the
decode-only v1 rows), `--layer-keys <JSON decoder-index -> layer_key>` to key
the seed by the structural `layer_key`, and a `# space=layer_key` header. A
seed with no layer-key map declares `# space=layer` and is refused by the
plugin parser -- a decoder-index seed can no longer be silently consumed.

MEASURED (2026-09-22, dev build host): applied on top of the 43 carried
patches (0003-0045) against pin `71640275` and built (`ninja
openvino_intel_gpu_plugin`, clean); the plugin carries the
`MOE_CPU_TIER_SEED` / `seed_source=` / `census seed: layer_key` strings. The
version stamp stays at `marfrit-p19` (disclosed: `p19` is also the 0003-0043
stamp, so the 0046 build is identified by its env/parse symbols, not the
stamp). Device-free cells: **14 green** in `tools/test_census_seed.py`
(compile `census_seed.hpp` with g++ and exercise every malformed/mismatched
case; each refusal goes RED when its check is removed) plus **76 green** in
`tools/test_hot_set_census.py`.

MEASURED (2026-09-22, A770 GPU.1 / PCI 8086:56a0): the served quality row is
PASS, no V4. Native d48n artifact, `--offload-ratio 99 --moe-cpu-tier`, KV u8,
chunk 512; incumbent `splitmix64` seed and the corpus census seed, same
256-token prompt and greedy 32 tokens, both produced greedy sha256
`2169836b33e8bc74d7965fff867b13c1d3637388a4b52f11f639f381ce7cc36f` --
byte-identical, because under the native artifact every routed expert runs on
the host tier (patch 0043), so residency moves bytes, not arithmetic. The
corpus S = 6 seed (one slot over the pool) was run first and REFUSED the load
(`census seed: layer_key ... lists 6 experts but the pool has 5 slots
(mismatched budget)`). Finding: the plugin's pool at ratio 99 is 5 slots/layer
(integer division) while the fit ledger prices 6 (`ceil`); the served seed is
the corpus top-5. The speed row stays EMPTY and G UNPINNED.

### 0047-moe-per-expert-slot-pool-size.patch

The per-expert dispatch path's slot pool must be resident-sized, not a
1-expert placeholder (campaign `docs/campaigns/sub4bit-vram-kernel.md`,
DESIGN §7.0.2cd). [code] Patch 0041, when `MOE_PER_EXPERT_DISPATCH` is on,
gives
every routed-expert Constant a 1-expert placeholder (`moe_offload_constant.cpp`:
`upload_shape[0] = 1`, reinterpreted to the full constant layout) so that the
data primitive exists and no mmap page is faulted. But the per-expert
dispatch path uses that same buffer as its weight storage: the provider builds
an LRU pool of `lru_expert_num` slots in it and `fill_weights_memory()` copies
each resident expert to `dst_offset = slot × (tensor bytes / num_expert)`
(`moe_otd_runtime.cpp`), while patches 0043/0045's per-expert OpenCL kernels
index it by `slot_index` (`set_otd_weight_pointers` → `exec_batched_gemv`).
The first slot with index ≥ 1 therefore ran past the 1-expert allocation.

MEASURED (2026-09-22, 24 GB card / PCI 8086:e211): the served native `d48n`
with `--moe-per-expert-dispatch` faulted before the HTTP server started —
`arcint … segfault … error 6 in libc.so.6` (the memcpy vector) and, at the
same instant, `xe 0000:0f:00.0 … Faulted Address 0x0000d556aa740000, Fault
response: Unsuccessful -ENOENT` on the blit engine (`engine_class=bcs`, engine
reset). A gdb attach localises the host crash to `paged_forward → load_paged`
→ `libopenvino_intel_gpu_plugin.so` → `libigdrcl.so` →
`__memcpy_avx_unaligned_erms`, i.e. the slot upload.

**The engine/plugin slot-count off-by-one is disproven.** The fit ledger
(`src/exec/fit.h expert_slot_bytes`) prices `ceil(512*(100-r)/100)` = 6 at
ratio 99 while the plugin (`ops/moe.cpp`) integers to 5. The discriminating
run is the ratio where both agree: ratio 75, both 128. The same segfault and
the same fault address recurred there, so the divergence is not the mechanism;
the fixed 1-expert placeholder is, and it is ratio-independent. The engine's
ceiling only ever sizes the reservation/ledger, never a plugin buffer
(`MOE_OTD_DEVICE_POOL_BYTES` is an env-set byte budget).

Fix: allocate the resident slot pool exactly as the ordinary OTD path does
(`upload_shape[0] = min(num_expert, resident_expert_num)`), while keeping the
compile-time behaviour: `upload_bytes = 0` (deferred to runtime), `skip_evict`
true (no mmap page faulting / VMA churn) and no device pool budget charged.
A defensive `OPENVINO_ASSERT` states the (min()-guaranteed) pool ≤ full-layout
invariant; it is not a runtime guard.

MEASURED (2026-09-22, 24 GB card): the fault is gone, the plateau probe settles
at `0.37 GiB` (`probe-static`), and the served path answers. On the native
`d48n` at ratio 75 + tier, KV u8, chunk 128, one lane:
`[OTD_PERF] … per_expert_dispatches=24676, per_expert_gpu_invocations=135874,
gpu_hits=3623, gpu_misses=21053, gpu_hit_rate=14.6823%, cpu_tier_pairs=187903,
created_onednn_kernels=0`; prefill 5 tokens 14.72 s, decode 16 tokens 28.21 s
(0.6 t/s), answer ` Paris. Paris is the most populous city in France and one
of the most visited`. The 16 GiB card serves the same cell too: load 675 s,
16 tokens 36.5 s (0.44 t/s on the request wall, prefill + decode; the B60's
0.6 t/s is decode-only), `per_expert_gpu_invocations=135634`,
`per_expert_dispatches=24697`, hit 14.89%, `cpu_tier_pairs=188023`. The load
takes 845 s on the 24 GB card — the residual stall is the CPU
tier's scalar native decode during the load-time probe (seven
`moe_cpu_expert` threads at ~90% CPU), not a JIT (no `ocloc`/`llvm-spirv`
child) and not a deadlock; it terminates.

### 0048-moe-otd-pinned-nvme-fill.patch

The load-time pinned NVMe fill's schedule, wired into the static partition
(campaign `docs/campaigns/nvme-direct-expert-tier.md`, design note
`docs/design-nvme-direct-expert-tier.md` D2/D3). Membership is the pinned set
patch 0018/0046 already fixes at `bind()`; the fetch is arcwell's batch
surface (`AW_IOC_SUBMIT_BATCH` / `AW_IOC_BATCH_WAIT`), one batch per MoE
layer, four batches in flight, collected and marked filled before the first
routed call. There is **no fetch on the decode path**.

New file `pinned_nvme_fill.hpp` (deliberately OpenVINO-free): the schedule —
an injected `Transport` with setup/submit/collect and no synchronous read
primitive, a `Scheduler` at depth 4 that fills the window before collecting
the oldest, retries a short batch once, and REFUSES the load (never a silent
demotion to the host tier) if a pinned expert is still not landed. It is the
byte-identical twin of arcint's tracked `src/exec/pinned_nvme_fill.h`, checked
by `tests/test_pinned_nvme_fill.cpp`, which is the one the device-free ladder
tests.

`expert_weight_providers.{hpp,cpp}`: the env opt-in `MOE_OTD_PINNED_NVME_FILL`
(read once at construction, inert when unset); `reserve_static_partition()`
factorised out of `bind()` so the coordinator can reserve every layer's slots
before it marks them filled; `pinned_nvme_fill_batch()` (this layer's pinned
membership as one batch); and `apply_pinned_nvme_fill_slot()` (the cache
`set_filled(slot)` the note's §3.4 requires on collect). A translation-unit
global coordinator starts at the first layer's `bind()`, enumerates the live
providers in structural `layer_key` order, and runs the barrier. A failure
anywhere in setup or the barrier throws — a load failure, per D3.

`require_no_sync_read()` is the campaign's own red-first guard: a would-be
synchronous `AW_IOC_READ_BLOCKS` is refused once serving has begun, so the
losing configuration cannot be reached.

MEASURED (2026-09-23, device-free): the patch applies cleanly to a pristine
checkout of the pin **through the full sequential series 0003–0047**
(`git apply --check` + apply, 45/45 then 0048), and compiles clean against the
0047 tree (`ninja openvino_intel_gpu_plugin`, rc 0; only
`expert_weight_providers.cpp` and `moe_3gemm_swiglu_opt.cpp` rebuilt). The
schedule's ladder is 8 cells green in `tests/test_pinned_nvme_fill.cpp`; three
mutants (guard removed, refusal replaced by a silent fill, depth ignored) each
fail their named cell — raw output in the campaign's evidence packet. The
version stamp stays at `marfrit-p19` (disclosed, the same as 0046/0047): the
0048 plugin is identified by its `MOE_OTD_PINNED_NVME_FILL` / `pinned NVMe fill`
symbols, not the stamp.

**OWED, stated not faked.** The `Transport` has no production implementation
in this patch: under the static partition the expert slot pool is host-mapped
(`MOE_OTD_PERF_LOG` reports `device_slot_buffers=0`), and arcwell requires a
dma-buf from an xe VRAM BO, so there is no destination a byte-transparent fill
can land in yet. The per-expert dma-buf BO the artifact-format step named as
the D2/D3 contract, the arcwell ioctl transport, and the card validation are
OWED. Until they exist, an ENABLED `MOE_OTD_PINNED_NVME_FILL` refuses the load
with that reason — exactly the note's "arcwell cannot be set up at all is a
load failure" rule. With the env unset, patch 0018/0046/0047 behaviour is
unchanged.

## 0049 — the arcwell transport and the OpenCL slot import

`0049-moe-otd-pinned-nvme-transport.patch` supplies the production `Transport`
0048 injected empty, and the OpenCL import that makes the BO-backed slot the
destination the resident expert is read from. Two new files:

- `moe/pinned_nvme_transport.hpp` — `lgc::nvme_fill::ArcwellTransport`, the
  `Transport` implementation. It owns the `/dev/arcwell` fd and the Arc render
  node fd, creates one 64 KiB-rounded xe VRAM BO per (layer, tensor)
  (`DRM_IOCTL_XE_GEM_CREATE` with VRAM placement + `NEEDS_VISIBLE_VRAM` +
  `CPU_CACHING_WC`), exports the dma-buf (`DRM_IOCTL_PRIME_HANDLE_TO_FD`),
  registers it peer-to-peer (`AW_IOC_MAP_BUFFER`, asserting
  `AW_MAP_F_REQUIRE_P2P`), and drives `AW_IOC_SUBMIT_BATCH` /
  `AW_IOC_BATCH_WAIT`, checking `out_submitted`/`out_err` (the ioctl return
  alone is not enough; an unaligned geometry returns 0 with submitted=0,
  err=-22). It expands one expert into THREE page-aligned requests — the store
  record is `gate|up|down` concatenated and the plugin's device layout is
  three per-tensor regions — with the store ordinal `dense_layer * capacity +
  slot` and `expert_%04u.bin` (the ordering was verified layer-major, 0
  mismatches against the manifest).
- `moe/aw_uapi.h` — arcwell's uAPI header (BSD-2-Clause), vendored because the
  plugin build cannot see `~/src/arcwell`.

`expert_weight_providers.*` gains `create_pinned_nvme_pool()` (register the
three BOs before the barrier), `pinned_nvme_geometry()`, and
`bind_pinned_nvme_pool()` — which **imports the dma-bufs with
`engine.import_buffer()`** and **replaces the host-mapped `gate_w`/`up_w`/
`down_w`** with `reinterpret_buffer()`s of the imported pool, so the fused GEMV
kernel reads the controller-DMA'd bytes directly. `moe_otd_runtime.*`'
`fill_weights_memory()` gains `include_weights=false`, used to host-upload only
the six scale/zp tensors (which the DMA slice excludes: adding them is 623.4375
pages, not page-aligned, and they need the `[oc][group]`→`[group][oc]`
transpose), completing each pinned slot at load on the engine's service stream.
Customisation is opt-in and operator-set: `MOE_OTD_PINNED_NVME_FILL`,
`MOE_OTD_PINNED_NVME_DRM`, `MOE_OTD_PINNED_NVME_STORE`,
`MOE_OTD_PINNED_NVME_PART_START`.

MEASURED (2026-09-24, device-free + one B60 leg): the patch, sha256
`d6d3498d20fddf22b2972ba128c7b719dd8b759e4378a4b16f838296b54a630f`, reverse-
applies and re-applies cleanly on the 0048 tree and compiles clean
(`ninja openvino_intel_gpu_plugin`, `ninja_rc=0`); the apply transcript
(`apply-check-0049.txt`) and the complete build log (`build-0049.log`) are in
the packet. The mechanism was proven on the B60 end-to-end by the tracked
non-arcint client `tools/arcwell_cl_slot_proof.c`: three per-tensor VRAM BOs,
two real store experts DMA'd as six requests, imported into OpenCL, read back
through the OpenCL queue **byte-identical** (sha256 `d463d1d5…`),
`via_host_bounce` delta 0, `max_inflight` 6. All five red legs fail as required
(`rc=1`, named failure, one transcript each in `mut-*.txt`): unaligned
geometry, dropped partition offset, system-memory BO, corrupted readback,
OpenCL corruption.

**OWED, stated not faked.** The integrated served number — the fill running
inside the serving loop and the depth-4 gate rows — is NOT measured here; the
plugin was built and the mechanism proven, but the acceptance gate's three
rows (`docs/window-053.md`) stay OPEN. The store ordering used by the
transport is the store's own layer-major ordinal; a run against an artifact
whose layer keys differ from the store's would need the store re-pointed.

## 0050 — a fourth native expert format: IQ2_S (Qwen3.6-35B-A3B)

`0050-native-expert-iq2s-format.patch` adds `MOECompressed::kWeightFormatIq2S
= 4` and carries it through the pattern block, the op validation, the CPU
tier's row decoder and the per-expert OpenCL decode — the shape of 0043/0045
for one more format. `Qwen3.6-35B-A3B` (`qwen35moe`) ships IQ2_S (ggml type
22) gate/up on all 40 layers over IQ3_XXS (37) / IQ4_XS (3) down, and IQ2_S
is not one of 0043's three: one 10-bit grid index per EIGHT values (four per
32) packed as little-endian u16 in a u8 `[E, out, K/32, 8]` weight slot, a
RAW sign byte per 8 values in the zero-point slot, and TWO 4-bit sub-block
scales per 32 (low nibble for values 0..15, high for 16..31) as f16
`[E, out, K/32, 2]`. `moe_otd_runtime.cpp` skips the device scale transpose
for IQ2_S (the two sub-scales are read row-major).

**IQ4_XS down needs no new format**: `iq4_xs_split` folds its 6-bit sub-block
scales into the per-32 f32 scale and lands on the IQ4_NL layout 0043 already
carries, so the three IQ4_XS down tensors ride `kWeightFormatIq4Nl`
unchanged (measured 2026-09-24 on `blk.34/38/39.ffn_down_exps`, split ->
decode vs gguf-py `max|diff| 0.0`).

MEASURED (2026-09-24, device-free): the patch applies, reverse-applies and
re-applies on the 0049 tree, and `ninja -j6 openvino_intel_gpu_plugin` in
`build-prod` is clean (rc 0, 47 targets, the plugin links). The plugin unit
cell `moe_cpu_expert_native.iq2_s_row_decodes_...` compiles to an object with
the plugin's own flags and the vendored gtest headers (that build dir has no
unit-test target configured).

**OWED.** The GPU compile of a 256-expert/IQ2_S block — the plugin's own
`ConvertTiledMoeBlockNativeToMoeCompressed` firing and the native per-expert
kernel launching — is NOT measured here; only the library build and the
serialised pattern are. That is the A770 window. The emitter
(`tools/q4e/serving_shape.py`'s IQ2_S branch and
`build_qwen35moe_serving_shape_ir`) is in the arcint tree, not in this patch.

## 0051 — the all-resident native pool: ratio 0 is a configuration, not an absence

`0051-native-fully-resident.patch` fixes a three-link dead end that made the
all-resident **native** configuration unreachable (and, with it, the fastest
native route: every expert on the GPU, only the routed ones computed). The
links, each on its own tree:

- `src/config.cpp` refused `--moe-cpu-tier` when the ratio was 0;
- the arcint backend set the plugin's `OFFLOAD_RATIO` (and `ov::weights_path`)
only when the ratio was `> 0`, so an explicit `0` was swallowed;
- the plugin's `prepare_moe_otd_params` (`ops/moe.cpp`) computed
`lru_expert_num = 0` when `otd_ratio == 0` in the stock form, so
`moe_3gemm_swiglu_opt.cpp` selected the **Resident** provider — which has no
slot pool and **no native reader** — while patch 0043's assert demanded
`_cpu_tier && is_offloaded()`, a combination unsatisfiable at 0.

This patch:

- `ops/moe.cpp`: for a native format at `otd_ratio == 0`, enable OTD when the
caller supplied `ov::weights_path` (an explicit 0 is thereby distinguishable
from unset) and size the pool at `num_expert` — every expert resident, read
through the offload provider's native reader. `ratio == 100` stays disabled
(all on disk cannot run).
- `moe_3gemm_swiglu_opt.cpp`: the assert requires `_weight_provider->is_offloaded()`
only. The tier was needed because (0043) no OpenCL decode existed yet; 0045
added it, so the tier is unnecessary when there are no misses.

MEASURED (2026-09-25, device-free): the patch reverse-applies cleanly on the
0050 tree, and `ninja -j8 openvino_intel_gpu_plugin` in the debug-caps-OFF
build dir is clean (rc 0, 6 targets, the plugin links). The arcint-side
companion (`src/config.cpp` guard, a `offload_ratio_set` flag, and
`src/exec/backend_ov.cpp` setting the property for an explicit 0) is in the
arcint tree, not in this patch. The version stamp stays at `marfrit-p19`,
disclosed the same as 0046-0050 (the built plugin reports
`2026.4.0-22849-71640275d29-marfrit-p19`; its sha256 is
`7a10444e7ab088232f8404ffb43c4ebbf51268a0e7d8f6983c16f313b25f0324`).

**OWED.** The served gate — the all-resident native arm actually loading on
the A770 and its rate against the tiered arm — is the card leg's, not this
patch's.

## 0052 — IQ2_S-packed: the checkpoint's own block, verbatim

The checkpoint (`Qwen3.6-35B-A3B-UD-IQ3_XXS`) serves 80 IQ2_S expert tensors.
The 0050 route re-laid each into a split form -- a u16 index, a sign byte and
an f16 scale per 8 values -- at 128 B per 256. This patch carries the
checkpoint's own block instead, with the f16 `d` lifted into the scale slot:
**80 self-contained bytes** (32 B qs low-2-bit indices | 32 B RAW sign masks |
8 B qh high-2-bit | 8 B 4-bit sub-block scales) plus one f16 per 256-value
block -- **82 B/256**, the GGUF's own size (design note 12.4: 10.35 GiB of
experts against 14.47). Added additively: `kWeightFormatIq2S` (4) and every
other path are untouched.

- `ov_ops/moe_compressed.{hpp,cpp}`: `kWeightFormatIq2SPacked == 5`, the
`is_native_format` membership, the weight-shape assert
`[E, out, K/256, 80]` (K = dim 2 × 256), and the affine `group_size`
cross-check skipped -- this format's scale is per 256, not per group.
- `native_expert_block.{hpp,cpp}`: `NativeIq2sPackedWeightsBlock`. It matches
the emitter's chain (`tools/q4e/serving_shape.py _native_packed_expert`): four
`Slice`s carve qs / signs / qh / sub-block scales out of the last axis; the
2-bit high index bits and the 4-bit sub-block scales come out in f32
arithmetic (a per-l divisor broadcasts over a new last axis, `floor` + mod --
no shifts, no `Concat`); the magnitudes are `Gather(iq2s_grid[1024,8])` and
the signs the RAW byte (bit j flips value j). Anchors: `weight`, `scale`,
`reshape`.
- `convert_tiled_moe_block_to_gather_matmuls.cpp`: the block joins the native
`Or` (tried first -- its first op after the Constant is a `Slice`, not the
IQ2_S block's `Reshape`); `resolve` sets format 5 with `zp` aliasing `scale`;
`hidden_size` takes 256 values per group row for it.
- `moe_cpu_expert.{hpp,cpp}`: `kQuantFormatIq2SPacked == 5` and the CPU-tier
row decoder (80-byte block, `d` from `m.s[n*nblk + ib]`).
- `moe_expert_swiglu.cl`: `native_dot_iq2s_packed` -- the OCL per-expert decode
-- plus the gate/up/down weight and scale slot strides for format 5.
- `moe_otd_runtime.cpp`: the scale transpose is skipped for format 5 (the
scale slot is a plain per-256 f16 vector, not an `[oc, groups]` payload).
- `moe_3gemm_swiglu_opt.cpp`: the native gate/up and down asserts admit 5.

MEASURED (2026-09-25/26, device-free): the 0050 build dir (`build-meas-0050`,
debug caps OFF) relinks clean with the patch applied -- the GPU plugin links,
rc 0. Emitter side: the packed chain decodes random blocks to `max diff/bound
1.28e-07` against `native_blocks.iq2_s_packed_decode` (bit-exact modulo the
f32 dot's summation order), and all four sampled real IQ2_S expert tensors
(`blk.0` gate/up, `blk.2` gate, and the IQ3_XXS down that stays on its own
route) are byte-identical to the checkpoint's own blocks -- 80 B weight plus
the f16 `d`, `w80exact=True dexact=True`.

**OWED.** The card leg: a packed artifact compiling through the GPU plugin
(the matcher firing) and serving under all-resident. The OTD CPU-tier decode
path is untested against the emitter's chain.

## Not carried either: the measurement instrument

The arcint session's working tree also carries per-stage timing accumulators
(`network.cpp`, `primitive_inst.cpp`, `stage_acc.hpp`). Those are how the
20,480 calls were found. They are **not** part of any patch here, and the
build script resets to the pinned commit and applies only this directory, so a
dirty measurement tree cannot leak into a package.
