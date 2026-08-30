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

## Graph read (next morning): the router is there; the dense read is ours

The reply to the table above asked the right first question: does the
exported head still contain its top-k router, or did `tools/export_mtp.py`
flatten the MoE into a dense read? Read from the XML, no build:

- `openvino_mtp_layer.xml` (3.6-35B head): `TopK` 1, `ScatterElementsUpdate`
  1, `Gather` 6, `MatMul` 12. The router is present — softmax, top-8,
  renormalise, scatter the eight weights into a zero row over 256 experts.
- The base IR's own MoE layer (`layers.0.mlp`): `SoftMax`/`TopK`/`ReduceSum`/
  `Divide`/`ScatterElementsUpdate` plus `Tile`/`Broadcast`/`Transpose` and
  per-expert `MatMul`s over experts stored as `u4 [256, out, groups, 64]` with
  `f16` scales and `u4` zero points — HuggingFace's per-expert loop, which
  the plugin's `fuse_moe_experts` pass matches (NonZero/Split/Gather/Slice/
  Broadcast/ScatterElementsUpdate around each expert) and lowers to the
  `moe_3gemm_fused_compressed` primitive behind the batched-GEMV kernels of
  the table above.

So nothing was lost on export. The 1.69 GB is read because the exporter's
`moe_block` *chose* a dense lowering — every expert computed for every
token, the non-selected ones weighted by exactly zero — to keep a
data-dependent loop out of the graph. That formulation matches no plugin
pattern, so it runs as it is written: two batched f16 MatMuls over all 256
experts, 6.7 ms of device per draft. The plugin's decode kernels for this
exact MoE exist and are in use 40 times per step in the body; the head does
not reach them because its graph is not the shape the fusion pass looks for.

Options, in the order of least new mechanism:

1. **Emit the base's pattern.** Export the head's MoE as the same per-expert
   subgraph the base IR carries, with the experts in the base's `u4
   [E, out, groups, 64]` layout, so `fuse_moe_experts` lowers it to the same
   primitive. At M=1 that is a top-8 gather at int4: ~13 MB of expert traffic
   against 1.69 GB, and the priming forward at M=128 gets the grouped-GEMM
   prefill path for free. The cleanest way to get the pattern exactly right is
   to clone the base IR's `layers.0.mlp` subgraph and swap its constants for
   the head's.
2. **A gather lowering of our own** (`Gather` the eight experts' weights, then
   two MatMuls): correct at M=1, and still an unfused graph with u4
   decompression in front of every MatMul.
3. **The int4 dense head** as it is: the same 256-expert read at a quarter of
   the bytes, ~2 ms of device — the smallest change and the smallest win.

(1) is the one that makes the head sparse like the body; (3) is being
measured meanwhile because it costs nothing but a compression run.

## Measured (same morning): the excess is inside the plugin's enqueue, not in waits

Instrument: the GPU plugin's own host-time profile at level 2
(`OV_GPU_HOST_TIME_PROFILING=2`, debug-enabled plugin build), which splits
every `infer()` into input processing, enqueue, wait and output processing,
averaged per compiled graph. The average includes the load-time probes and
the prefill, so each arm was run at two lengths (300 and 1400 generated
tokens, same prompt, EOS not reached) and the per-forward figures below are
the **differences**, in which everything but the decode forwards cancels
exactly. Card: B60, u8 KV, 8192 context, the 35B with the reconstructed MoE
head; 1100 plain steps against 622 verify passes.

| per forward | inputs | **enqueue** | wait | outputs | host total | untraced wall |
|---|---|---|---|---|---|---|
| plain step, M=1 (`--mtp off`) | 0.37 | **12.6** | 1.15 | 0.12 | 14.3 ms | 14.1 ms |
| verify forward, M=2 (`--mtp on`) | 0.47 | **27.4** | 1.13 | 0.17 | 29.2 ms | 29.8 ms |
| head layer draft (1400 infers) | 0.08 | **3.8** | 1.17 | 0.02 | 5.05 ms | — |
| head lm_head (787 infers) | 0.01 | 0.05 | 1.14 | 0.09 | 1.29 ms | — |

Three things this settles:

1. **The ~15 ms the verify forward costs over a plain step is enqueue time
   in the plugin** — 27.4 ms against 12.6 for a graph with 16% more launches
   (1334 against 1145) — and not synchronous waits: the wait is 1.1 ms in
   every graph, the fixed tail between the last enqueue and the last
   kernel's completion. The reply's expectation ("in waits, not in enqueue")
   is wrong, and so is the per-launch rate model that both sides used: host
   cost per walked primitive is 5.4 µs at M=1 and 10.9 µs at M=2 on the same
   graph. Something the plugin does per primitive per infer costs twice as
   much at two rows as at one. Which thing is the next measurement, not a
   narrative.
2. **arcint's own loop is not where the time is.** Per generated token at
   1400 tokens: 23.4 ms of wall = verify 16.8 (29.8 × 788/1400) + head layer
   5.05 + lm_head 0.73 + embeddings 0.23 + emit 0.39 + ~0.2 of sampling and
   acceptance. Everything outside the plugin's `infer()` calls is under a
   millisecond per token. The "~10 ms unattributed in the serving loop" of
   the section above is retracted: it was the plugin's enqueue at M=2.
3. **The head's device time is smaller than the traced 6.7 ms.** Its
   `infer()` returns 5.05 ms after it starts, 1.17 of that waiting; the
   device cannot have been busy longer than ~5 ms per draft in this run. The
   tracer's per-kernel profiling inflates short kernels; the 6.7 ms stands
   only as an upper bound. The head's host side is 3.8 ms of enqueue for 118
   launches (32 µs per launch, six times the body's rate at M=1).

Per generated token the arithmetic is now closed: plain 14.1 ms = 12.6
enqueue + 1.15 tail + 0.4 arcint; speculative 23.4 ms = the numbers above.
The levers, re-ranked with the plugin's enqueue as the cost:

- the M=2 enqueue at 27.4 ms is the lever (a verify at the M=1 enqueue rate
  would be ~13.5 ms, and the pair 8.0 ms per token: +76% over plain);
- the head layer's 5.05 ms of host per draft is the second (for 118 launches
  it should be under 1 ms at the body's rate);
- the int4 head buys device time, which is not what is being paid for.

### Where in the enqueue (same afternoon): the PagedAttention MIXED stage, by elimination

The plugin's verbose trace of one plain step against one verify forward
(`OV_VERBOSE=4`, one infer block each) walks the same primitives with the same
kernels — 371 `jit:gemm:any`, 40 MoE, 284 eltwise, 44 strided slices in both —
with one difference: the ten PagedAttention layers. At M=1 they run the
single-token stage; at M=2 the plugin classifies the forward as MIXED
(`past_lens != 0` with a multi-token query), selects the `sdpa_micro` stage,
and **rebuilds the primitive's implementation on every infer** (`update_impl:
… impl update: was: ocl::paged_attention::opt now: …`, ten per verify, none
per plain step).

The MIXED stage also reads `past_lens`, `subsequence_begins` and
`max_context_len` on the host in every stage of every layer; on a device
buffer each read is a blocking `clEnqueueMapBuffer` that waits for the
in-order queue. The OpenCL API table counted 105 of them per verify forward
against 3 per plain step. That mechanism was tested and is **not the cost**:
with the index inputs in USM host memory (`ARCINT_PA_HOST_INPUTS=1`, the
plugin dereferences instead of mapping) the maps vanish from the API table
entirely, the output stays byte-identical, and the verify enqueue moves from
27.4 to 26.3 ms. The plain step does not benefit either (its enqueue read
14.2 against 12.6, within the run-to-run spread but not better), so the flag
stays an experiment, default off.

What remains is ~14 ms per two-token forward inside the plugin's enqueue,
localised to the PA MIXED path by elimination and not yet measured to the
function; the per-infer impl rebuild is the suspect. No `perf` in the
container and the plugin's per-stage dump aborts with the head loaded when
profiling is enabled; the next instrument is that dump without profiling, or
a timer around `update_impl` in the debug plugin build.

## Found (same evening): 20,480 subbuffer creations per verify forward

The per-stage accumulators (a timer in the debug plugin around each pipeline
stage of `primitive_inst`, summed per infer, printed per network at
host-time profiling level 3) split the verify's 27.4 ms of enqueue by stage
and then by implementation:

| per LM forward, median | plain M=1 | verify M=2 |
|---|---|---|
| shape inference | 1.78 | 1.87 |
| impl update | 0.01 | 0.02 |
| realloc | 0.19 | 0.58 |
| prepare (total) | 5.25 | 6.92 |
| execute | 7.60 | **17.49** |
| — of which `ocl::moe::moe_3gemm_swiglu_opt_` | 0.69 | **9.04** |

The MoE implementation's host-side execute costs 226 µs per layer at two
tokens against 18 µs at one. The mechanism is in
`prepare_internal_buffers`: at `token_num > 1` it rebuilds the per-expert
mask subbuffers — `create_subbuffer` twice per expert, 512 calls per layer,
**20,480 per two-token forward** — on every infer. The batched-GEMV path a
two-token verify actually takes never reads those masks; they exist for the
per-expert prefill fallback. At one token the block is skipped, which is why
the plain step never showed it.

The fix (patches/0003, applied to the local plugin build): skip the mask
creation at `token_num <= batched_gemv_threshold`, create the masks lazily
in the prefill fallback, and cache them against (token_num, buffer) for the
prefill path proper. Measured on the B60, u8 KV, 300 tokens, temperature 0,
outputs **byte-identical** to the unpatched plugin in both arms:

| | unpatched | patched |
|---|---|---|
| `--mtp off` | 62.3 t/s | 61.7-62.3 t/s |
| `--mtp on`, prose (80.7% accept) | 44-46 t/s | **60.5-61.2 t/s** |
| verify forward wall | 27.3 ms | **18.1 ms** |
| MoE host execute per verify | 8.91 ms | 0.74 ms |

The speculative path is now a wash with plain decode on prose. What remains
of the verify's extra host time is ~3 ms (PagedAttention MIXED stage 0.85,
realloc +0.4, prepare +1.2) — and the head itself, whose 5.05 ms of host per
draft (2.6 of it shape inference over 123 primitives) is now the largest
speculation-only cost. The USM-host index-input experiment and the two
falsified mechanisms (blocking maps, impl rebuild) are recorded above; the
serving loop's "~10 ms unattributed" of 7.0.2o stays retracted.

### The int4 head, measured after the fix (item 3 of the order)

NNCF INT4_ASYM, group 64, all layers, on the exported f16 head: layer
1.69 GB → 455 MB, lm_head int8 kept. Same card, same prompts, 300 tokens,
temperature 0, patched plugin:

| head | prose accept | prose t/s | code accept | code t/s |
|---|---|---|---|---|
| f16 | 80.7% (134/166) | 60.8-61.2 | 78.1% (132/169) | 63.1 |
| int4 | 71.4% (125/175) | 68.4 | **84.0% (137/163)** | **72.9** |

Acceptance moves both ways with quantisation — down 9 points on the prose
prompt, up 6 on the code prompt; single prompts, near-tie tokens flip, and
the draft stays greedy-verified either way, so this is a speed knob, not a
correctness one. The int4 head's smaller read makes `--mtp on` a clear win
on the code prompt: 72.9 t/s against ~62 plain (+17%) — the first
configuration in which the 35B's speculation beats plain decode. Both
numbers want the Prüfstand harness (10/10 acceptance task) before the model
card changes; and none of this reaches production until the plugin fix
ships (the packaged plugin still pays the 9 ms per verify).
