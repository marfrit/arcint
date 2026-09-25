# sub4bit-vram-kernel — a per-expert GEMM kernel with in-kernel dequant, bypassing the MoE fusion for routing-aware expert execution

## The defect, as measured

Not a defect: a milestone re-scope, recorded 2026-09-05 (DESIGN §7.0.2ah,
`docs/milestone-0.3.0.md` M10 row), and re-scoped again 2026-09-15 after
the 0.5.1 port-route measurements (window-051 B.3) and the FreeToken code
reading (`docs/research-freetoken-code.md`).

The pinned runtime's MoE fusion (`keep_moe_3gemm_const_precision.cpp`)
computes every expert for every token. Flash-Next activates 10 of 512
experts per token (`num_experts_per_tok: 10`), so the fusion does 51.2×
the work the architecture requires (`measured-here`: window-051 B.3
CORRECTION). FreeToken's reference implementation (`code`: `~/src/
FreeToken-ref`) never computes an unrouted expert — `ensure_experts
(layer_id, expert_ids)` takes the router's own ids — and never
materialises a dequantised weight (dequant inside the GEMM K-loop,
inside the ggml kernels, or by Triton inline-dequant).

The kernel this campaign builds is the mechanism that makes routing-aware
expert execution possible on the plugin: a per-expert GEMV/GEMM that
takes packed weights and dequantises in-kernel, bypassing the MoE fusion
entirely. With it, arcint computes only the routed experts, keeps a GPU
LRU cache of hot expert weights in their packed form, and feeds misses
from the host pool — the FreeToken architecture, on Intel's kernel
library.

Sub-4-bit precision (Q3_K 3.44 bpw, IQ3_XXS 3.06, Q2_K 2.63) is one
lever for more cache headroom — int4→int3 shrinks the expert pool 25 %
(`docs/design-qwen-flash-next.md` WP6b: ~95 % hit → ~97 % at the same
resident capacity). The kernel works at u4 first, and the price of that
is now measured (2026-09-18, status below): the u4 grouped-affine repack
of the checkpoint's IQ3_XXS / IQ4_NL experts carries 0.10–0.13 relative
RMS per expert tensor (0.07–0.08 even at 16-element groups) and shows as
0.73 nats of KL at depth 48 on a served model that is otherwise the
model's (`serving-shape-logits.md`). The KLD gate therefore does require
the native format: in-kernel decode of the checkpoint's own blocks
(IQ4_NL's 16-entry table per 32-block, IQ3_XXS's 256-entry 8-element
grid), not a re-quantisation of them.

## Known against hypothesised

Known (`measured-here`): the MoE fusion computes all experts per token
(window-051 B.3; the port route's 7.06× device residency and 623× warm
forward are the cost of that fusion); the streaming fit study projects
~18 t/s on a single A770 with a 24 GiB expert LRU at 95 % hit, ~30–40
t/s with MTP amortisation (`docs/design-qwen-flash-next.md` WP6b);
FreeToken serves 35B MoE at 77–83 t/s on a 32 GB card (`paper`: arXiv
2608.16157). Known (`code`): the FreeToken reference computes only routed
experts, with in-kernel dequant, GPU LRU cache, and a bandwidth-adaptive
CPU/GPU split (`~/src/FreeToken-ref`). Known (`measured-here`): the
matcher's `u4`-only requirement and the oneDNN type table (§7.0.2ah);
K-quant byte counts; NNCF 3.3.0 has `INT3_SYM`/`INT2_SYM` in its mode
enum.

Hypothesised: the kernel's size, "800–1,500 lines" (§7.0.2y); the
decode-regression sign for a per-expert kernel with in-kernel dequant on
Xe2 — unmeasured, with a same-shaped precedent: symmetric u4 KV's
in-kernel dequant costs +63 % on the fused `PagedAttentionExtension`
(§7.0.3), a different kernel and tensor class.

Prior art, surveyed 2026-09-05 and recorded with URLs, licenses and Arc
applicability: `research-sub4bit-weights.md` (same directory). Prior art
after the survey date exists and is not yet on the record.

## Gate

A model with ≥ 512 experts serves on one card with routing-aware expert
execution: only the routed experts computed per token, a GPU LRU cache
holding the hot set, misses fed from the host pool. Prüfstand 10/10;
greedy output byte-identical across two cold starts (§3.4); decode t/s
and hit-rate reported at the reference cell; and the KLD against the
model's own capture within the bar re-derived from this model's own
reference round-trip — which, measured 2026-09-18, the u4 repack cannot
reach (0.73 nats): the experts must be computed from the checkpoint's own
blocks. Sub-4-bit cache headroom is then a measured row (the native
format's bytes against the u4 repack's at the same resident capacity),
win or lose.

## Entry criteria

Partially met. (1) The recon is on the record (§7.0.2ah) — met. (2) The
FreeToken code reading (`docs/research-freetoken-code.md`) — met
(2026-09-15, on the `qfndev` branch). (3) The streaming fit study
(`docs/design-qwen-flash-next.md` WP6b) — met (2026-09-10). (4) A
routing histogram (patch 0013, `MOE_OTD_ROUTING_HIST`) over a corpus long
enough to characterise cache-hit distributions — unrun. (5) The resident
format (`u3` group-quant vs K-quant blocks), picked by measurement on
one expert layer against the int4 baseline — unrun.

## Scope — in / out

In: the per-expert GEMV/GEMM kernel with in-kernel dequant at u4 as the
first form; the GPU LRU expert cache (the slot pool of patches 0005–0007
extended to hold packed weights, evict by routing frequency, and feed a
per-expert kernel instead of the fused path); the host miss-tier
integration (patch 0011's CPU kernel as the miss handler, FreeToken's
`q*` split as the reference); the routing histogram for cache-hit
characterisation; the sub-4-bit resident format as a measured
cache-headroom lever; Prüfstand and equivalence measurements.

Out: the MoE fusion matcher itself (bypassed, not patched — the fusion
still exists for the dense-resident case); NVMe direct expert fetch
(`nvme-direct-expert-tier`, a separate campaign); host-side K-quant
native compute (`kquant-host-storage`, a throughput lever for the host
miss tier).

## Where it lives

DESIGN §7.0.2ah (re-scope and recon), §7.0.2y (NNCF/K-quant/runtime
findings), §7.0.3 (u4-KV precedent); `docs/milestone-0.3.0.md` M10 row;
`docs/design-qwen-flash-next.md` WP6b (the streaming fit study);
`docs/research-freetoken-code.md` (the FreeToken code reading);
window-051 B.3 (the port-route measurements that killed the fusion
path); `~/src/FreeToken-ref` (the reference implementation); the plugin's
MoE fusion matcher (`keep_moe_3gemm_const_precision.cpp`); the slot pool
(patches 0005–0007); the host CPU tier (patches 0011–0012).

## Pipeline for this campaign

Design note (`docs/design-routing-aware-expert-execution.md`): the
per-expert kernel's contract, the GPU LRU cache's eviction policy, the
host miss-tier integration, the FreeToken `q*` split adapted to this
hardware's measured bandwidths — with the fit study's numbers as input,
not re-derived → the routing histogram over a real corpus (patch 0013)
→ red-first: a cell proving the fused path is refused when the
per-expert kernel is available, and a cell proving an unrouted expert
is never computed → **the kernel work**: the per-expert GEMV/GEMM with
in-kernel dequant at u4, the cache manager, the host split → one card
window: Prüfstand 10/10, decode t/s and hit-rate at the reference cell,
byte-identical cold starts → the sub-4-bit format measurement (int4 vs
int3 on one expert layer, same source): cache-headroom gain reported,
win or lose → review before commit → DESIGN record, CHANGELOG line.

## Invariants

DESIGN §3.4 (history-independent greedy output): the GPU LRU cache
must not make the served answer depend on which experts happen to be
resident — the same invariant patch 0018's static partition enforces
for the existing tier, and the same invariant FreeToken's own
deterministic cache-fill order preserves. Ground rule 2 (a
fusion-impact profile, not a kernel micro-benchmark) applies.

## Status

- 2026-09-05 — opened from the 0.3.1 backlog; nothing started.
- 2026-09-15 — re-scoped. The "VRAM-resident sub-4-bit" framing retired:
  the port-route measurements (window-051 B.3: 7.06× device residency,
  623× warm forward, the MoE fusion computing all 512 experts for every
  token) and the FreeToken code reading showed that the mechanism is
  routing-aware expert execution with a GPU LRU cache, not smaller
  resident weights. The kernel with in-kernel dequant is the same work;
  its purpose is to bypass the fusion and compute only the routed
  experts. Sub-4-bit is one cache-headroom lever, not the gate. Prior
  art after the 2026-09-05 survey exists and is not yet on the record.
- 2026-09-16 — design note committed (169d350); per-expert dispatch framework
  committed (patch 0038, 7dbcadb). Pipeline steps 1 (design note) and 3
  (red-first cells + dispatch) done. The dispatch routes all routed experts
  through the existing CPU tier (patch 0011) with the fused GEMV bypassed
  entirely — proves the dispatch mechanism before the per-expert OCL kernel
  exists. Fable-reviewed: 3 findings fixed (clone field list, offload guard,
  entry assert). Next: the per-expert OCL kernel (pipeline step 4).
- 2026-09-16 — per-expert kernel dispatch integration committed (patch 0040).
  Wires 0039's moe_expert_swiglu.cl into the live dispatch: GPU-resident
  experts launch per-expert kernels (expert_gate_up, expert_down) with slot
  pool weight pointers; non-residents go to CPU tier (patch 0011); fused GEMV
  bypassed via sentinels for all routed experts. Removes 0038's blanket
  sentinel (the proof-of-concept all-CPU-tier redirect). Fable-reviewed:
  clean (0 findings; prior round's 6 findings C1-C4/M1-M2 all addressed).
  Pipeline step 4 (the kernel work) done. Next: one-card window measurement.
- 2026-09-16 — one-card window blocked: host OOM during compile_model,
  all three attempts. (1) --offload-ratio 100 disables partial upload
  entirely (moe_offload_constant.cpp:62, `otd_ratio < 100` boundary),
  every expert constant gets full allocate_memory + memcpy → 152 GiB
  virtual, OOM-killed. (2-3) --offload-ratio 99 enables partial upload
  (~602 MiB expert slot buffers instead of 64.6 GiB), but read_model
  (backend_ov.cpp:2570) still mmaps the full 77 GiB .bin; graph
  construction walks all 17,354 nodes, paging in mmap regions; the
  host (62 GiB RAM, 20 GiB swap) exhausts both → global OOM at 23:17
  (dmesg: pid 637454, total-vm 36 GiB, 610k swap entries, the unit manager
  killed first). The model cannot be compiled on this host at full
  depth without a plugin change to avoid mmapping expert weight regions
  (`measured-here`, three attempts; the next entry retracts the premise
  that the offload path was engaged at all).
  arcint CLI flag (--moe-per-expert-dispatch) in working tree, not
  tagged. Services restored.
- 2026-09-17 — **the per-expert series is inert on the Flash-Next
  artifact family: the plugin's MoE fusion never matches the
  serving-shape emitter's MoE subgraph** (`measured-here`, B60 = GPU.0,
  22.71 GiB, stock core 2026.4.0-22849 + the p17 plugin series with
  patch 0041 hand-applied, plugin sha 4f881fff…, tools tree = e50148f).
  Instrument: `compile_model` then `get_runtime_model()`, primitive
  types counted off `layerType`, residency off `GPU_MEMORY_STATISTICS`.
  - `qwen38-flash-next-d12-ov` (12 layers, 4,565 IR nodes, expert bodies
    as 36 u4 `Const`), props `OFFLOAD_RATIO=99 MOE_CPU_TIER=YES`:
    1,201 exec nodes, **0 MoE-typed primitives**, 230 `FullyConnected`,
    `usm_device` **17.73 GiB** — the 09-13 stock-plugin figure (c05) to
    the digit. Adding `MOE_PER_EXPERT_DISPATCH=YES`: identical residency
    (17.73 GiB; fdinfo vram0 18.2 GiB, GTT plateau 20.9 GiB), compile
    133 s → 9.6 s (no kernel cache on the host; the speed-up is real and
    unexplained). Partial upload, slot pool, CPU tier, per-expert
    dispatch and 0041 are all keyed on a `MOECompressed` consumer
    (`get_moe_constant_role`, `moe_offload_constant.cpp`) that this IR
    never produces; the 09-16 line "~602 MiB expert slot buffers instead
    of 64.6 GiB" was arithmetic, not a measurement, and is false for this
    family.
  - Control, `qwen36-35b-a3b-int4-ov` (HF export, 40 MoE layers), same
    props without per-expert: `moe_3gemm_fused_compressed` ×40,
    `moe_router_fused` ×40, `usm_device` **1.2 GiB** (from ~17) — the
    fusion and the offload path work where the pattern matches.
  - Control with `MOE_PER_EXPERT_DISPATCH=YES`: `compile_model` fails,
    `clBuildProgram CL_BUILD_PROGRAM_FAILURE` (program_builder.cpp:168) —
    the per-expert OpenCL kernel (0039/0040) has never built on a card;
    "pipeline step 4 done" rested on review, not on a compile.
  - Earlier the same day, two launches of the segmented port-route
    artifact (`…-seg12-ov`, dead since window-051 B.3) took the physical
    host down twice (host thrash, plug pulled); the 115 GiB compile
    footprint was on the record two days before. Host fence changed by
    the operator afterwards: ARC 16 GiB persistent, container 44 GiB.
  - Process slip on the record: the host sampler's watchdog arms only
    on a matched driver pid; the three scratch-script compiles (runtime
    graph dumps) ran without it. MemAvailable never fell below 23 GiB in
    any cell.
  Consequence: before any kernel or residency work continues, the
  artifact has to carry a MoE pattern the plugin fuses — either the
  emitter writes `ov::op::internal::MOE` (or the HF pattern) so the
  whole series applies, or the dispatch hook moves to the
  `FullyConnected` path. That is a design decision, not a window.
  Segmented port-route runtime (306 lines in backend_ov.cpp,
  `load_paged_segmented`) stays uncommitted: its route is dead.
- 2026-09-17, later — **the non-match has a cause, and it is the emitter,
  not the plugin; the (a)/(b) fork above is dissolved.** Dispositions:
  - `code` (plugin source, `convert_tiled_moe_block_to_gather_matmuls.cpp`
    `build_3gemm_pattern`, pinned build 2026.4.0-22849): the tiled matcher
    anchors on `end_reshape` = Reshape(down MatMul) and on `router_reshape`
    = Reshape(Transpose(ScatterElementsUpdate)) → optional Unsqueeze, both
    feeding the router-weight Multiply before the ReduceSum root. The pass
    is registered only under `supports_immad && use_onednn &&
    !moe_disable_fusion` (`transformations_pipeline.cpp`); both cards
    qualify, the 35B control proved it the same day.
  - `code` (this repo): `export_mtp.py:401 moe_block_tiled` carries both
    Reshapes and records (lines 515–531) that 2026.4.0 folds a rank-4
    Reshape whose target dims are all known, so B comes from ShapeOf and S
    is a runtime −1. `serving_shape.py:795 emit_moe_tiled` named that
    function as its source and emitted neither Reshape: down MatMul →
    Multiply, Transpose → Unsqueeze. "Measured to fuse" in its docstring
    was inherited, never re-measured on this emitter.
  - `measured-here` (dev host, CPU only, no card): the constraint walker
    `tools/check_tiled_pattern.py` on the depth-12 artifact's IR — 43
    ReduceSum candidates, 0 matched, all 12 MoE candidates
    `R4.router_reshape.type: observed Transpose, expected Reshape`
    (0.2 s, read_model only). The same walk on the reduced 4-layer
    geometry: 0/4 before the fix, 4/4 after, live and after
    save → read_model.
  - Correction to this document's opening line: "the MoE fusion computes
    every expert for every token" described the UNFUSED graph — the Tile
    over E makes the batched MatMul compute every expert — never the fused
    op. `GatherMatmul` takes the router's `active_indices` (`code`, the
    pass's callback), and the fused kernel's weight provider, slot pool and
    CPU tier all work on the routed set (patches 0005–0012, 0038). Every
    Flash-Next residency and forward figure to date (window-051 B.3's
    7.06×/623×, c05's 17.73 GiB) is the unfused path. The design note's
    §1 already says this; the charter line here did not.
  - Fix: `emit_moe_tiled` now emits both Reshapes exactly as
    `export_mtp.py:532–537` (B from ShapeOf, S = −1). Red-first cell
    `test_every_moe_layer_walks_the_plugins_tiled_3gemm_pattern`
    (tests/python/test_serving_shape.py). A walker PASS is not a compile;
    its docstring lists the blind spots. What proves it is one compile with
    the runtime-graph dump: `moe_3gemm_fused_compressed` ×12 on a
    re-exported depth-12 artifact, residency read off
    `GPU_MEMORY_STATISTICS` — the campaign's next card window, after the
    artifact is re-exported. Until then patch 0041's question and the
    per-expert kernel's build failure stay open behind it.
  - Re-export blocked the same day: the GGUF shards the exporter reads
    are no longer on the dev host (two of three gone with a volume
    re-purposed on 2026-09-15). Route around it for the census:
    `tools/moe_tiled_rewrite.py` inserts the two Reshapes into a pre-fix
    IR in memory before `compile_model`. `measured-here` (dev host, CPU
    only): on the depth-12 IR 12 blocks rewritten, walker 0 → 12, 0.2 s,
    0.08 GiB RSS; on an old-style block the CPU-plugin forward before and
    after is bit-identical. The census window can therefore run on the
    measured artifact; a servable on-disk artifact still needs the
    shards (or a full `save_model` of the rewritten graph).
- 2026-09-17, evening — **the Flash-Next serving-shape artifact fuses, and
  the offload series applies to it.** Three anchors were missing, each
  invisible to the previous check: (1) the two Reshapes (above);
  (2) a ONE-input Swish — the Python binding's `op.swish(x)` appends a
  beta Constant, the pattern declares `Swish({gate_matmul})` with one
  input and the C++ Matcher rejects an argument-count mismatch (`code`:
  `Matcher::match_arguments`; `measured-here`: census 2 with the
  Reshapes alone still 0 MoE primitives; the fusing 35B control carries
  `Swish/opset4 in=1`); (3) for the offload series, the dequant chain in
  f16 with a trailing Convert → f32 — the control's shape — because under
  f16 inference the plugin puts a Convert on an f32 scale Constant
  feeding the fused op and the OTD resolver demands a direct,
  FILE-BACKED Constant (`moe.cpp`: mmap source, weight-sharing buffer or
  an `otd_bin_offset`); an in-memory rewrite therefore fuses on the
  stock plugin (census 2/3) and not with `OFFLOAD_RATIO` (census 3), the
  rewritten graph saved to disk does both (census 4).
  Census ladder (`measured-here`, B60 = GPU.0, depth-12 artifact, KV u8,
  f16, primitive types off `get_runtime_model()`):
  | cell | plugin | props | MoE-typed | FullyConnected | usm_device |
  |---|---|---|---|---|---|
  | unfused (control) | p17+0041 | ratio 99, CPU tier | 0 | 230 | 17.73 GiB |
  | rewrite (Reshapes only) | both | — | 0 | 230 | 17.73 GiB |
  | rewrite (+Swish) | stock | — | 12 + 12 router | 194 | 17.62 GiB |
  | rewrite (+f16 chain), in memory | p17+0041 | ratio 99, CPU tier | compile refused (bin offset) | | |
  | rewritten artifact ON DISK | p17+0041 | ratio 99, CPU tier | **12 + 12 router** | 194 | **3.00 GiB** |
  Host peak ≤ 4.6 GiB in every cell; no watchdog. Tools: `tools/
  moe_tiled_rewrite.py` (pre-fix artifacts), `tools/check_tiled_pattern.py`
  (input counts now checked, matches the 35B control 40/40), the boot
  driver's `--rewrite-tiled-moe` / `--census`, and a device-free oracle:
  the CPU plugin runs the same tiled pass and compiles a matched block
  to three `GatherMatmul` primitives (in the suite). Commits b5946e1,
  481387b, 09daece, 763f044, 0b66c43. Open, in order: a forward on the
  fused offload path (values will differ from the unfused record: other
  kernels, f16 scales), decode at ratio 99, the ratio sweep, then 0041's
  compile-time question at 48 layers and the per-expert kernel's build
  log. A re-export from the GGUF shards replaces the rewritten artifact
  once the shards are back on the dev host.
- 2026-09-17, late — **the fused path served, on both cards, and the
  offload tier faults on the 24 GiB card.** Served binary at 0b66c43
  (registry entry for the rewritten depth-12 artifact, e384c05), the
  +p17 plugin, n-gram shard bound, KV u8, f16 inference, prefill chunk
  512, one lane, `measured-here`:
  | card | artifact | offload | first forward | warm decode 64 tok |
  |---|---|---|---|---|
  | B60 | d12r fused | none (17.62 GiB) | OK, deterministic | **80.5 t/s** (unfused rung 09-13: 18.3) |
  | B60 | d12r fused | ratio 99 or 50 + CPU tier | **xe page fault** at the slot-pool probe | — |
  | B60 | d12r fused | ratio 99, no tier | probe OK; requests: "allocated output memory is necessary to set kernel arguments" | — |
  | B60 | 35B control | ratio 99 + tier, +p17 AND +p16 | the same page fault | — |
  | A770 | 35B control | ratio 99 + tier, +p17 | OK, Paris | 16.1 t/s |
  | A770 | d12r fused | ratio 99 + tier | OK, deterministic | **26.6 t/s** (7.7 cold) |
  Dispositions: (1) the fused MoE kernel runs this family and is 4.4× the
  unfused decode at full residency (`measured-here`); (2) the CPU tier's
  first forward faults on the B60 with everything else equal — plugin
  (+p16 without the per-expert series faults too, so 0038–0040 are not
  the cause), binary, artifact, flags — and serves on the A770: a
  card/driver-side fault, `xe … Faulted Address 0x1f0f5e000, Fault
  response: Unsuccessful -ENOENT`, device coredump; the tier had never
  served on the B60 on the record (the 08-30 B60 figures are full
  residency); 30-second reproducer: the 35B at ratio 99 + tier on GPU.0;
  (3) routing-aware execution with 99 % of the experts on the host runs
  the fused depth-12 rung at 26.6 t/s warm on the A770 behind its 1.8
  GB/s link — the campaign's first offload number on a card, and a
  lower bound for the B60 once its tier fault is fixed; (4) the no-tier
  request failure is a runtime binding defect on the served request
  path (the probe path allocates the output the request path does not)
  — open. Values at depth 12 are not the model's; the served France
  prefix matches the unfused record's first four tokens. Next: the B60
  tier fault (driver-side, needs the coredump and a plugin-level
  reproducer), the no-tier binding defect, then the ratio sweep and the
  gate's Prüfstand at full depth once a 48-layer fused artifact exists.
- 2026-09-17, later still — **compile-time staging on the fused offload
  route, and 0041's question.** `measured-here` (B60, the rewritten
  depth-12 artifact, ratio 99 + CPU tier, compile only, driver-side gtt
  off the host sampler at 2 s): the genuine +p17 package plugin stages
  6.4 GiB through the driver for 12 layers (~0.53 GiB/layer) against
  20.9 GiB unfused (~1.7 GiB/layer, the 09-13 linear probe's ~1.4); with
  patch 0041 on top, the same 6.4 GiB. The fused offload route already
  avoids uploading the offloaded expert bodies at compile; 0041 adds
  nothing measurable there and its question closes for this route. The
  48-layer fused compile at ratio 99 is therefore forecast at ~26 GiB of
  driver memory (`measured-here` extrapolation, linear in layers as the
  09-13 probe found), inside this host's physical budget — the 48-layer
  fused export was started on that forecast. Not forecast to fit: the CPU
  tier's host pool at 48 layers (~58 GiB of expert bodies), the residency
  stream's problem, unchanged. Process note: the "+p17 package plugin" of
  this day's served legs was a hand build (p17 + 0041) copied over the
  package file — found by `dpkg -V` before this A/B; the served results
  stand (0041 is inert without its flag, and this cell shows it inert at
  compile), the labels were wrong.
- 2026-09-17, evening — **the full-depth model compiles on one card.** The
  48-layer artifact exported with the fixed emitter (tree da52858,
  74.6 GiB `.bin`, 144 expert bodies, 78 min, host peak 41 GiB under a
  fence) walks 48/48 and compiles on the 24 GiB card with the genuine
  +p17 package plugin at `OFFLOAD_RATIO=99` with the CPU tier:
  `moe_3gemm_fused_compressed` ×48, `moe_router_fused` ×48, **8.06 GiB
  device-resident**, 119 s, host RSS 2.5 GiB, driver-side peak 15.6 GiB
  (forecast 26), no watchdog (`measured-here`, B60, KV u8, f16, compile
  only). The residency refusal of window-050 §4.10 and the 66 GiB
  compile-time staging that killed every full-depth attempt are gone on
  the fused route. What a full-depth forward still needs: the CPU tier
  on the B60 (its first forward page-faults there, works on the A770,
  card/driver-side), and a host pool that does not hold all 58 GiB of
  offloaded bodies in RSS — the residency stream, 0.5.1's other half.
  Artifact pinned: `qwen3.8-flash-next-d48f`.
- 2026-09-17, night — **the B60 tier fault is patch 0037**, bisected in
  one window (record in `static-partition-prefill.md`, the patch's own
  campaign): +p13 serves, +p16 faults, +p16 without 0037 serves, the LRU
  partition avoids it, the A770 never showed it. Not the driver, not the
  geometry, not 0038–0040. With a plugin without 0037 the 35B serves on
  the B60 with the tier at ratio 99 at 23.5 t/s (`measured-here`). The
  full-depth fused artifact's forward on the B60 is therefore gated on
  patch 0042 plus the residency stream, no longer on an unknown.
- 2026-09-17, night, later — patch 0042 fixes the B60 tier fault (record
  in `static-partition-prefill.md`): the grouped prefill's gather ran
  over every token-expert pair while 0037's tables held only the
  resident ones, reading ~8 GiB past the buffer through a wrapped offset. `measured-here` (B60): the 35B serves with the tier at
  ratio 99, 23.6 t/s (KV u8, f16). Ships as `+p18`. The full-depth fused artifact's
  forward on the B60 is now gated on the residency stream alone.
- 2026-09-17, night, last — `+p18` built and accepted on the B60 with the
  35B (record in `static-partition-prefill.md`). What the full-depth fused
  artifact's forward on the B60 now needs is the residency stream alone;
  the tier serves on that card.
- 2026-09-17, midnight — **the first full-depth forward.** With `+p18`
  the 48-layer fused artifact serves on the 24 GiB card at ratio 99 with
  the CPU tier: compile 82.8 s at 8.06 GiB device, the n-gram table
  bound, the slot-pool probe through 48 layers in 44 min, then France
  raw, greedy, 8 tokens: prefill 5 tokens 113.6 s, decode 8 tokens
  171 s (21 s/token) — the tier reading routed experts from the
  artifact file on demand (`measured-here`, B60, KV u8, f16, one lane;
  anonymous memory ≤ 1.5 GiB, file-backed pages up to 30.7 GiB: the
  tier's host pool is page cache, not a copy — patch 0011's mapped
  accessor, `code` — so the 58 GiB "host pool" of the residency
  arithmetic never existed). The token is not the model's answer
  (`REDPalette convudir…`): the Paris line at depth 48 is NOT obtained,
  and the cause is not localised — no serving-shape artifact has produced
  a meaningful token at any depth on the record, so the emitter's dense
  parts and the table binding are suspects alongside the MoE route,
  which itself answers Paris on the HF-exported 35B. Next: the KLD
  replay against the reference capture at full depth (needs the page
  cache warm or the residency stream; 21 s/token is the disk), and the
  A770 for the same cell as a second witness.
- 2026-09-18, after midnight — **the artifact on the NVMe, and the
  depth-12 ladder.** (1) The 48-layer fused artifact copied to the ext4
  NVMe volume (83 GB, 4.4 min off ZFS): the tier's expert reads are
  then NVMe-bound — the slot-pool probe 9.5 min instead of 44, decode
  64 tokens at 1.5–1.9 t/s cold instead of 0.05, 9.8 t/s with the routed
  experts in the page cache; the France token is byte-identical across
  two cold starts and both storage paths (`measured-here`, B60, ratio
  99 + tier, KV u8, f16, one lane). The residency stream's first half is
  a copy, not a mechanism. (2) The correctness ladder at depth 12, the
  same France prompt, logits dumped and diffed on the request's own
  prefill (`tools/logits_dump_diff.py --from-n 5`; the load ladder's
  records come first and must be skipped): unfused d12 against fused
  d12r at full residency — argmax 3/5, mean KL 0.147, max |logit diff|
  4.6 — far beyond f16 scale rounding; fused with the tier against
  fused without — argmax 4/5, KL 0.020, decode steps on the same history
  within 0.4 logits and KL ≤ 3e-3. The tier is exonerated at depth 12;
  the fused route (its own f16 GEMV) and the unfused route (dynamic int8
  activations into FullyConnected) disagree, and neither is the
  reference. The unfused d12 token reproduces the 09-13 record byte for
  byte. (3) The reference is the llama.cpp logits capture at full depth,
  replayed through the served d48f from the NVMe — running as this entry
  is written.
- 2026-09-18, 00:50 — **the reference speaks: the served full-depth
  logits are noise.** The llama.cpp capture replayed through the served
  d48f from the NVMe (2 × 2,735 tokens, 1,046 s and 1,230 s of prefill):
  mean KL 12.42/12.16 and 12.40/12.34 nats, argmax agreement 0.0007 and
  0.0000 (`measured-here`, B60, ratio 99 + tier, KV u8, f16). For scale,
  ln(248320) = 12.42. The depth-12 and depth-4 rungs read the same on
  09-13; a truncated prefix scores like that too, so they could not decide
  between "missing layers" and "broken" — full depth does: the artifact
  family is broken, and no rung says at which depth. This campaign's work stands — the
  fused route, the tier on both cards, full depth at 8.06 GiB, the NVMe
  — and its gate now waits on `serving-shape-logits.md`: the emitter or
  the fill loses the model before any expert is computed.
- 2026-09-18, 03:53 — **the Paris line at depth 48, served** (`measured-here`,
  Arc Pro B60, the full-depth artifact re-exported through the corrected
  fill — `serving-shape-logits.md`, DESIGN §7.0.2bz — from the NVMe at
  `--offload-ratio 99 --moe-cpu-tier`, KV u8, f16, one lane): "The capital
  of France is" → ` Paris. Paris is a city in France`, cold and warm
  byte-identical; the chat form reasons in the model's own voice; a 64-token
  greedy continuation stays coherent. This campaign's route (the fused MoE,
  the CPU tier, the residency at 8 GiB device) computes the model. The KLD
  replay against the model's own capture runs next as the gate number.
- 2026-09-18, 04:50 — **the KLD residual is this campaign's premise,
  measured end to end** (`serving-shape-logits.md`, last entry): the served
  full-depth model reads 0.73 nats against its own capture, and the whole of
  that residual traces to the u4 grouped-affine repack (group 128) of the
  IQ3_XXS / IQ4_NL experts — 0.13 / 0.13 / 0.11 relative RMS on blk.0's
  expert tensors — not to the card, the precision, the KV cache or the
  route. The native sub-4-bit expert kernel is what the gate waits on.
- 2026-09-18, 13:58 — **a finer repack is not the lever** (`measured-here`,
  CPU, real expert tensors of blk.0 and blk.24): the u4 grouped-affine
  repack's relative RMS error at groups 16 / 32 / 64 / 128 is 0.077 / 0.099
  / 0.117 / 0.130 on IQ3_XXS (gate) and 0.069 / 0.085 / 0.095 / 0.106 on
  IQ4_NL (down). Even 16-element groups leave 7–8% per tensor — a third of
  the KL at 8× the scale bytes, not the order of magnitude the gate needs.
  The fused MoE takes any group size mechanically ({experts, ofm,
  num_groups, group_size}), so this is a numerics limit of the affine grid
  against the I-quant codebooks, not a plumbing one. The native sub-4-bit
  expert kernel — IQ4_NL's 16-entry table per 32-block, IQ3_XXS's 256-entry
  8-element grid — is what the gate waits on.
- 2026-09-18 (afternoon) — **the native format, end to end short of the
  card.** The residual of the corrected depth-48 artifact (0.73 nats) is
  the u4 repack (`measured-here`, per-tensor relative RMS 0.10–0.13; G16
  0.077, so finer groups are not the lever). The checkpoint's per-layer
  formats read off the shards: 43 layers IQ3_XXS/IQ4_NL, layer 2
  IQ4_XS/Q8_0, four layers IQ3_XXS/Q8_0. `q4e.native_blocks` (four splits,
  bit-exact vs gguf-py), `serving_shape._native_expert` (standard-op decode
  in the fused op's group-32 layout, f16 block scale), `NativeExpertFiller`,
  `--expert-format native`; plugin patch 0043 (three pattern blocks, the
  native pass to `MOECompressed`, `weight_format` per projection, tier-only
  execution with three row decoders; +p19). Depth-4 native artifact
  exported. On the B60: attempt 1 — the pass fires, op translation refuses
  the f32 scale (fixed: f16); attempt 2 — the card wedges at the process's
  first job (GuC "not started" cascade, NULL deref in
  `xe_sched_job_set_error`, DKMS xe-ringorder/7.0.14+p1), before the
  pass's output ran. Not attributed: the control (stock affine, same
  harness) is the first leg when a card is back. Host reboot is the
  operator's call. Next: control cell → native cells (GPU.1 first, it is
  the smaller card and the A770 is bit-stable across forwards) → cut4n on
  the d4n artifact vs llama.cpp whole tensors (layer 3 ≥ 0.9999 is the
  target) → the full-depth native export → the KLD gate through the tier.
- 2026-09-18 (evening) — patch 0043 reviewed before packaging (two
  confirmed defects: the native-format members missing from the impl's
  clone field list — the executing impl would have run affine on native
  bytes, patch 0038's defect one patch earlier — and the flag read before
  its assignment; both fixed) and pinned by three host-only decoder cells
  in the plugin's tier tests (8/8 standalone). Full-depth native artifact
  exported: `qwen38-flash-next-d48n`, 77.5 GB, 144 bodies native, built
  in 468 s (the quantising fill took 3,583 s), peak host 39.4 GiB. Ladders
  on the tip: C++ 592/0/0 with the real shard; python 334 passed after the
  suite guard caught (and the fix removed) a new count gate. Everything
  on-card — the control cell, the native cells, the depth-4 cut ladder,
  the served native reading and the KLD gate — waits for the dev host's
  reboot (the operator's call).
- 2026-09-18 (evening, after the host reboot) — **the native path on the
  cards, measured.** The lowering cell passes on both cards (A770 and
  B60): the stock-affine control fuses at corr 0.999999 against the CPU
  plugin, both native pairs lower to `MOECompressedNative` and match at
  corr 1.000000 (`measured-here`, tree dffd272, plugin 5a6968ec). The
  depth-4 native cut ladder against llama.cpp's whole tensors: layer 0/1
  out corr **0.99991 / 0.99989** — where the exact f32 reference itself
  sits against llama.cpp (0.99991 at layer 0), so the u4 repack's error is
  gone from the routed experts; layer 2/3 at 0.99970 / 0.99953 (the u4
  artifact: 0.99924 / 0.99918 / – / 0.99873), a steady per-layer growth
  that a KV-f16 control did not move (bit-identical), consistent with f16
  accumulation plus llama.cpp's own Q8 activation noise; the exact
  reference at depth 3 (the three-way at layer 2) still owed (its first
  run OOM'd at a 36 GiB fence). The full-depth native artifact
  (`qwen3.8-flash-next-d48n`, registered) compiles on the B60 through the
  boot driver (48 fused native ops, 21.7 s, 6.86 GiB device) and through
  the served binary (26.7 s, 8.06 GiB) — and the tier then CRAWLS: every
  (token, expert) pair decodes its rows on the scalar reference path,
  ~2 min per 512-token chunk, so the served KLD gate needs an hour-scale
  window (running) and the designed step 3, the OpenCL decode of the
  native formats in the fused kernels, is now the rate lever, not a
  quality one. Three served attempts before that were my harness's
  fault (the offload flags live in an `EXTRA` variable the driver did not
  set — full residency, 60 GB of USM host, the watchdog), recorded as
  such.
- 2026-09-18 (night) — **the gate's first native reading, and the premise
  re-read.** Served d48n on the B60 (ratio 99 + tier, KV u8, f16, chunk
  512): Paris, coherent 64-token continuation, 0.5–0.8 t/s on the scalar
  tier. KLD against the model's own capture, window 0 (1,367 rows):
  mean 0.42 / median 0.20 nats, argmax 0.79 — the u4 artifact on the same
  window: 0.54 / 0.24, argmax 0.73 (`measured-here`, `tools/kld_position.py`).
  The exact expert formats remove ~20% of the divergence at every position;
  the remainder is flat in position (no rise 1280 → 2815, no chunk-seam
  step, below = above the QSA boundary) — a per-token residual that the
  depth-4 ladder's drift cannot explain at depth 48, so it lives deeper
  than layer 3 and is NOT the experts. The campaign's gate premise ("the
  residual = the u4 repack") was therefore only a fifth of the story; the
  deeper cut ladder (layers 7/15/23/31/47 and the logits vs llama.cpp) is
  the next localisation, before the OpenCL decode work is worth its rate.
- 2026-09-18 (late night) — **the residual's mechanism, measured.** Block
  cuts inside layers 1–3 of the native depth-4 artifact against llama.cpp's
  block taps: the GDN and hyper-connection paths sit at the ~1% floor at
  every token; the routed-expert sums are 15–28% off on specific tokens
  with exact weights. The router is llama's to 1e-8 on llama's own input;
  its top-10 margins are 1e-4…1e-6 and the ten hold 6–27% of the mass. An
  exact recompute of layer 1's routed sum from the GGUF matches llama at
  0.8–1.1% (llama's Q8-activation floor); a 2% input perturbation moves it
  1–2% on four tokens and 15% mean / 29% max on the token the artifact gets
  18.5% wrong. The residual against the model's own llama.cpp capture is
  routing instability under the ~1% activation difference any independent
  forward has — history-dependent, flat in position past ~1,300 tokens
  (median 0.20 nats), the same with an f16 KV cache (0.21) — not a defect
  of the artifact, and not closable without replicating llama's activation
  quantisation. The u4 repack was the removable term (0.54 → 0.42 mean on
  window 0); the gate's bar has to be re-derived from a reference that
  shares this model's routing noise. The rate lever (the OpenCL decode of
  the native formats, design 2.3c step 3) is now the campaign's next work,
  with the quality question closed at this artifact's own floor.
  Confirmed from the other side: the pin's own exact f32 forward differs
  from llama's layer-0 router input by 1.7–2.2% and already routes 2 of
  the 5 tokens to a different expert at layer 0 (at the two smallest
  margins, 3.9e-5 and 5.2e-5). No implementation short of llama's own
  arithmetic matches this capture's routing.
- 2026-09-19 (morning) — **the exact reference at full depth, on the
  Sparks, and the gate re-read against the model's own arithmetic.**
  `tools/ref_forward_stream.py`: the pin's own model, every weight from the
  GGUF, the experts streamed per layer, the n-gram table gathered lazily,
  the two 7.0.2bz corrections from the config, the sparse-attention
  indexer mapped in the feed for it; 48 layers in f32 on a GB10 in 741 s
  (the numpy dequant of the experts is the whole cost). Against it
  (`measured-here`): the native artifact is exact to 0.2% at every token
  through 24 layers (corr 1.00000) where llama.cpp is at 3.9–6.8%; its
  last-token logits sit at KL 0.017 nats where llama.cpp's sit at 0.053;
  from layer 24–27 on two of the five tokens drift (13–16% by layer 43),
  the near-tie flip at the artifact's own 0.2% (f16) difference. The KLD
  gate re-read on window 0 against the f32 reference capture: native
  **mean 0.369 / median 0.181 / argmax 0.827** (below the QSA boundary
  0.283, above 0.455 — the dense-for-sparse price, visible for the first
  time), f16 KV the same, the u4 artifact 0.603 / 0.264 / 0.718; and
  llama.cpp's own capture against the same reference **0.339 / 0.065 /
  0.802**. So the artifact and llama.cpp are equally far from the model in
  the mean and the artifact agrees on the argmax more often, but their
  errors differ in shape: llama's is heavy-tailed (most tokens at
  0.02–0.06), the artifact's is a broad ~0.18-nat floor at every long-
  context token — its own term, absent at 5 tokens, saturated by position
  1,367, not the KV precision, not the experts. Candidates: the f16
  recurrent GDN state (llama keeps it f32), the 512-token chunks' state
  carry (the single-chunk control is running), the f16 attention over
  thousands of keys. The 0.06-nat bar is llama.cpp's own per-token floor
  against the model; reaching it means fixing that long-context term, and
  the quality question of this campaign now has a yardstick that is the
  model, not another implementation.
  The single-chunk control (the window's 2,735 ids in one boot-driver
  forward, to separate the prefill chunks' state carry from the f16 state
  and the long-context attention) did NOT run: the B60 wedged at the
  compile's first job, the third such event, each at the card's first
  submission after a long idle since boot while back-to-back legs never
  wedge — a runtime-PM resume suspect, unmeasured, recorded for the next
  boot. The control stays owed with the term it would split.
- 2026-09-19 (late morning) — **the chunks are not the term.** The
  capture's 2,735 ids in ONE forward through the native artifact (B60,
  ratio 99 + tier, KV u8, f16) read mean 0.380 / median 0.191 / argmax
  0.792 against the f32 reference; the 512-token chunked serve read
  0.369 / 0.181 / 0.827 (`measured-here`). The artifact's long-context
  floor is therefore in the f16 recurrent GDN state or the f16 attention
  over thousands of keys (with the dense-for-sparse price above 2,051);
  the split — reference taps at long context against boot-driver cuts at
  layers 3 and 23 on the same ids, or an f32-state emitter option — is
  the next leg. Beside it: the B60's first job after an idle hour
  survived with runtime suspend disabled (one event; three wedges before
  it at the default).

- 2026-09-19 (evening) — **the f16-state and f16-attention candidates are FALSIFIED at the IR level; the quality lever is re-priced.** `openvino.Core().read_model` on `d48n`, `d48g`, `d48f` and `d4n` (CPU, no card, no compile): every `ReadValue`/`Assign` state variable is **float32** — the 36 GDN recurrent states `[1,48,128,128]`, the 36 GDN conv states `[?,10240,4]`, the 24 KV states `[?,2,?,256]`; **no op in any of the four models emits f16**; the 144 f16 constants are the expert block-scales (`layerN/moe/experts_{gate,up,down}/block_scale`), each immediately `Convert`ed to f32. So neither candidate named in the entry above is a mechanism this artifact HAS, and `tools/q4e/serving_shape.py::stateful_gdn_core` already declares the state `Type.f32` — **an "f32-state emitter option" is a NO-OP**; the exports already carry f32. f16 enters only as the plugin's **compile-time execution precision** (the main model sets no `ov::hint::inference_precision`; only the DFlash drafter has `ARCINT_DRAFT_F32`, backend_ov.cpp:1256/3095). The quality lever is therefore a **main-model f32-execution A/B** — and on the A770 that is BLOCKED: window-051.md A.2 records three f32 attempts at depth 4, three refusals (`primitive_onednn_base.h:559`, `paged_attention.cpp:72` BY_CHANNEL block size, `CL_BUILD_PROGRAM_FAILURE`), so "the f32 row of the round-trip pair is NOT AVAILABLE from the card". Any such A/B must go to the B60 or a different plugin configuration; the L3/L23 tap cuts would chase a mechanism that does not exist. The u4 artifact's worse reading (median 0.264 vs the native 0.181) is priced as **u4 expert quantisation** (`uint4_t` ×288 vs the native's u8/u4/i8 mix), not state dtype.
  **The bar line above is also corrected:** the bar is not llama.cpp's floor — it was DECIDED at BERLIN‑001 (`5d4dd59`) in `docs/window-051.md` clause (d) and MEASURED at A.2: `bar_0.5.1 = 100 × F_ref` = **3.0905e-03 nats below row 2051 / 2.6946e-02 at or above it** (`F_ref` = the capture's uint16 reconstruction error = 3.0905e-05 nats mean), the inherited 0.0599 SUPERSEDED BY LINEAGE (AMENDED 2026‑09‑14). `F_served` at depth 48 stays **EMPTY**, and no verdict is readable without it. [CORRECTED 2026-09-20: **the row is no longer EMPTY** — `F_served` at depth 48 is MEASURED: **0.1361 (w0) / 0.1512 (w1)** on the B60, where the decided bar sits below it and the row is UNREADABLE; and **0 on the A770** (r0↔r1 bit-identical, 0/1367 rows moved), where the bar sits above it and clause (d) **READS**. The floor is a per-card defect, not a property of the served path — see DESIGN §7.0.2cb and `campaigns/served-prefill-determinism.md`.]


- 2026-09-21 (night) — **the native OpenCL decode is authored, built, and its
  device-free ladder is green; the card leg is owed.** [code + measured-here]
  Patch 0045 (`0045-native-expert-ocl-decode.patch`) adds IQ4_NL / IQ3_XXS /
  Q8_0 in-kernel decode to the per-expert kernel (`expert_gate_up_native` /
  `expert_down_native`, the same dispatch geometry and argument list as 0040's
  affine kernels) and lifts patch 0043's refusal of native-format +
  per-expert dispatch, so the resident routed experts compute on the card
  while the misses keep the CPU tier — the mechanism this campaign's charter
  names, and step 3 of design note §2.3b/§2.3c. A gate/up format other than
  IQ4_NL/IQ3_XXS is refused at stage compile rather than decoded with the
  wrong kernel. `measured-here` (dev build host): the patch applies on top of
  0003–0044 and `ninja openvino_intel_gpu_plugin` is clean; the plugin carries
  the two native symbols and the native tables; the version stamp stays
  deliberately `marfrit-p19`, so the 0045 build is identified by its
  `expert_gate_up_native` symbol, not the stamp. The CPU reference
  `tools/q4e/native_expert.py` and its ladder
  `tests/python/test_native_expert_gemv.py` are **16 green, device-free**: block
  scale applied per format, fused-vs-materialised equality, a K that is not a
  multiple of 32 refused, an unknown format refused, and a deliberately wrong
  affine reading of IQ4_NL bytes caught rather than absorbed. **Not yet
  measured** [owed]: the served native artifact through the native per-expert
  kernels — one card window with the GPU-dispatch counter, the rate and the
  correctness reading. No card was taken; the running served census window
  holds the host CPU, so the measurement waits for it to clear.

- 2026-09-21 (late) — **the native decode compiles on the card, and the served
  per-expert path faults before serving; the root cause of the historical build
  failure is found and fixed.** [measured-here + code] Patch 0045 was corrected
  during the card window: the plugin compiles a primitive's kernels into ONE
  program, so the per-expert `.cl` body appears once per kernel and its
  file-scope helpers (`expert_gate_up_gemv_u4`, `expert_down_gemv_u4`,
  `load_x_interleaved`) were defined twice — the `clBuildProgram`
  `CL_BUILD_PROGRAM_FAILURE` on xe2 that the 2026-09-17 record saw and never
  localised. 0045 now wraps every file-scope helper (and the native tables)
  in a persistent `#ifndef` guard, so the concatenated copies define them
  once. With that, the native d48n artifact **loads and compiles** on the
  24 GB card (GPU.0, PCI 8086:e211): `lgc load: language model ready in 23.4 s
  (paged); device-resident 8.06 GiB`, at both ratio 99 and ratio 80 with
  `--moe-per-expert-dispatch`. Then the served path **faults before the HTTP
  server starts**: `xe … Faulted Address 0x0000d556aa740000, Fault response:
  Unsuccessful -ENOENT` on the blit engine (`EngineClass: 3 bcs`, engine
  reset) with an `arcint` `segfault … in libc.so.6` (memcpy) at the same
  instant, three independent launches (two ratio 99, one ratio 80). No
  `per_expert_gpu_invocations` was read. The decode arithmetic itself is
  pinned device-free (16 cells) and the fault is DOWNSTREAM of it — the
  slot upload (`fill_weights_memory`), the per-expert sentinel/gather path, or
  the dispatch bookkeeping; that is the next localisation. The card was left
  clean (sampler, SIGTERM, no leftover process); a fresh wake lock would be
  needed for a leg past the coordinator's.

- 2026-09-22 (early) — **the fault is card-specific: the A770 takes the same
  native per-expert load and survives it.** [measured-here] The identical
  config (native d48n, `--moe-per-expert-dispatch`, ratio 90, KV u8, chunk
  512) on the A770 (GPU.1, PCI 8086:56a0) LOADS and compiles: `language model
  ready in 38.5 s (paged); device-resident 8.06 GiB`, and the plateau probe
  settles at 0.37 GiB (`source: probe-static`, against the 5.71 GiB config
  ceiling — the driver keeps the pool host-mapped, the §7.0.2t two-ledger
  shape). No GPU fault, no segfault. But the served HTTP server did **not**
  come up within ~30 min of the process running at ~475 % CPU after the
  probe (no listener on the port); the leg was killed and the card released
  clean. So the B60's blit-engine page fault is a per-card defect (as the
  B60 faults in this campaign have been), while on the A770 the native
  per-expert execution path reaches the probe and then stalls before serving
  — a JIT/dispatch stall to localise, not a numeric fault. Neither card has
  a served reading yet; the decode arithmetic stays pinned device-free (16
  cells).
- 2026-09-22 (late morning) — **the fault is patch 0041's 1-expert slot-pool
  placeholder; the off-by-one is disproven; the served native per-expert
  reading is taken.** [measured-here + code]
  - The fault, raw: B60 GPU.0 (`xe 0000:0f:00.0`), served native `d48n` with
    `--moe-per-expert-dispatch` at ratio 99 and 80 (the 0045 leg; the ratio-75
    discriminating run is below) — a host memcpy past a buffer
    (`arcint … segfault at … error 6 in libc.so.6`, the memcpy vector) and, at
    the same instant, `Faulted Address 0x0000d556aa740000, Fault response:
    Unsuccessful -ENOENT`, `EngineClass: 3 bcs`, engine reset. A gdb attach
    localises the host crash to `paged_forward → load_paged` → the plugin →
    `libigdrcl.so` → `__memcpy_avx_unaligned_erms`: the slot upload copying an
    expert into the per-tensor slot buffer.
  - Mechanism (`code`): patch 0041's per-expert-dispatch branch gives the
    routed-expert Constant a **1-expert placeholder** (`moe_offload_constant.cpp`,
    `upload_shape[0] = 1`), but that same buffer IS the slot pool —
    `fill_weights_memory` copies each resident expert to `dst_offset =
    slot*(tensor bytes/num_expert)` and the 0043/0045 per-expert kernels index
    it by `slot_index`. The first slot ≥ 1 writes past the allocation.
  - **Off-by-one test (first hypothesis): KILLED.** The engine ledger
    (`src/exec/fit.h`) prices `ceil(512*(100-r)/100)` = 6 at ratio 99, the
    plugin integers = 5. The discriminating run is the ratio where both agree:
    ratio 75, both 128 — the same segfault and the same
    `Faulted Address 0x0000d556aa740000` recurred. The divergence is not the
    mechanism; the fixed placeholder is, and it is ratio-independent. The
    engine's ceiling only ever sizes the reservation/ledger, never a plugin
    buffer (`MOE_OTD_DEVICE_POOL_BYTES` is an env-set byte budget).
  - Fix: **patch 0047** (`0047-moe-per-expert-slot-pool-size.patch`,
    mirror + contrib) allocates the resident slot pool in the per-expert
    branch exactly as the ordinary OTD path does, keeping 0041's
    `upload_bytes = 0`, `skip_evict`, no device-pool charge. Built clean as
    the plugin at prefix `ov-0047` (plugin `f021de51b5812ee2`).
  - **Served native reading** (B60 GPU.0 PCI 8086:e211, `d48n`,
    `--offload-ratio 75 --moe-cpu-tier --moe-per-expert-dispatch --paged-kv
    u8 --prefill-chunk 128`, one lane): `per_expert_gpu_invocations=135874`
    (the owed counter), `per_expert_dispatches=24676`, `gpu_hits=3623`,
    `gpu_misses=21053`, hit 14.68%, `cpu_tier_pairs=187903`,
    `created_onednn_kernels=0`. Prefill 5 tok 14.72 s; decode 16 tok 28.21 s
    (**0.6 t/s**); answer ` Paris. Paris is the most populous city in France
    and one of the most visited`. Load 845 s.
  - **Second card.** The A770 (GPU.1, PCI 8086:56a0) serves the same cell:
    load 675 s, 16 tokens in 36.5 s — **0.44 t/s on the request wall (prefill +
    decode; the B60's 0.6 t/s is decode-only)** — answer ` Paris. The capital of
    Germany is Berlin. The capital of Italy is Rome.`,
    `per_expert_gpu_invocations=135634`, `per_expert_dispatches=24697`, hit
    14.89%, `cpu_tier_pairs=188023`. Both cards' native per-expert route is
    unblocked.
  - The stall is localised: with the fault gone, the load-time activation /
    plateau probe runs `paged_forward` while the **seven `moe_cpu_expert` pool
    threads** burn ~90% CPU each on the scalar native row decoder
    (`movzbl → cvtsi2ss → mulss → movss`); no `ocloc`/`llvm-spirv` child and
    `created_onednn_kernels=0`, so it is compute, not JIT and not a deadlock.
    It terminates; the 0.6 t/s is the tier's price at 25% resident.
  - Open: (1) a served window that skips the load probe (`--fit-ledger-dir`)
    or a cold-start fix, for rate measurement without the 14-minute load;
    (2) the rate win needs the HELD hot-set/LRU campaign; (3) the affine
    per-expert path (0040, u4 artifact) is owed under 0047. The decode
    arithmetic stays pinned device-free (16 cells).

- 2026-09-22 (rate leg) — **the native per-expert route's rate comparison:
the mechanism serves and pays at high residency; the ratio-99 VENICE budget
does NOT meet the pinned gate (V1), and the dispatch route shows a §3.4/V4
answer-divergence.** [measured-here] One fresh process per arm, A770 (GPU.1,
PCI 8086:56a0) and B60 (GPU.0, PCI 8086:e211), native `d48n`, plugin
`ov-0047` (`f021de51b5812ee2`), binary tree `wt-b6dbca5` (`f6abb4027a976ad6`),
`--moe-cpu-tier --moe-per-expert-dispatch`, KV u8, one lane, n_ctx 8192,
greedy 64 tokens, temperature 0, `ignore_eos`, the pinned capture window-0's
first 256 token ids. `--fit-ledger-dir` skips the load-time plateau +
activation probes on the second matching run (probe-skip below).

  **Rate table (decode 64 tokens):**
  | run | card | ratio | seed | decode t/s | prefill t/s | gpu_hit_rate | per_expert_gpu_invocations | cpu_tier_pairs | card-pair share | answer sha256 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | splitmix64 seed | A770 | 99 | splitmix64 | **0.547** | 0.736 | 0.580 % | 5,126 | 212,477 | 1.19 % | `55dff6f2…` |
  | census seed | A770 | 99 | S5 | **0.556** | 0.789 | 3.285 % | 16,230 | 206,925 | 3.77 % | `2e7c508f…` |
  | census seed | A770 | 75 | S128 | **0.842** | 1.328 | 33.22 % | 207,970 | 111,055 | 48.36 % | `5cd2e195…` |
  | host control (no dispatch) | A770 | 75 | — | **0.465** | 0.676 | — | 0 | 159,264 | n/a | `5437683b…` |
  | census seed | B60 | 99 | S5 | **0.555** | 0.746 | 3.098 % | 30,778 | 383,971 | 3.85 % (mixture) | `7e0adcd6…` |

  Card-pair share = `(invocations/2) / (invocations/2 + cpu_tier_pairs)`,
with the two per-expert kernels (gate-up and down) per card pair. It is
defined for the dispatch arms only: they total 215,040 pairs = 448 tokens ×
48 layers × 10 experts exactly (128 warm-up + 320 served tokens). The host
control runs the fused/tier path, whose `cpu_tier_pairs` counts differently
(159,264) and has no card pairs, so it is the rate baseline and not a share
point. The A770 splitmix64/census ratio-99 runs and the A770 ratio-75 rows
are ledger hits (served-only counters); the B60 ratio-99 run is the first at
its key, so its counters include the load probe (*mixture*).

  **Raw evidence (excerpt per arm; OTD_PERF lines elided at `…`, answer
sha256 prefixes):**

      # A770 ratio-99 splitmix64 (ledger hit)
      lgc  slot 0: decode     64 tok in 116.91 s (  0.5 t/s)
      lgc  slot 0: prefill   256 tok in 347.80 s (  0.7 t/s)
      [OTD_PERF] gpu_hits=288, gpu_misses=49398, gpu_hit_rate=0.57964%, …
                 per_expert_dispatches=49686, per_expert_gpu_invocations=5126, …
                 cpu_tier_pairs=212477, created_onednn_kernels=0
      # A770 ratio-99 census S5 (ledger hit)
      lgc  slot 0: decode     64 tok in 115.17 s (  0.6 t/s)
      lgc  slot 0: prefill   256 tok in 324.37 s (  0.8 t/s)
      [OTD_PERF] gpu_hits=1632, gpu_misses=48051, gpu_hit_rate=3.28483%, …
                 per_expert_dispatches=49683, per_expert_gpu_invocations=16230, …
                 cpu_tier_pairs=206925, created_onednn_kernels=0
      # A770 ratio-75 census S128 (ledger hit)
      lgc  slot 0: decode     64 tok in 75.98 s (  0.8 t/s)
      lgc  slot 0: prefill   256 tok in 192.75 s (  1.3 t/s)
      [OTD_PERF] gpu_hits=18008, gpu_misses=36208, gpu_hit_rate=33.2153%, …
                 per_expert_dispatches=54216, per_expert_gpu_invocations=207970, …
                 cpu_tier_pairs=111055, created_onednn_kernels=0
      # A770 ratio-75 host control (no dispatch, ledger hit)
      lgc  slot 0: decode     64 tok in 137.69 s (  0.5 t/s)
      lgc  slot 0: prefill   256 tok in 378.75 s (  0.7 t/s)
      [OTD_PERF] gpu_hits=10185, gpu_misses=47803, gpu_hit_rate=17.564%, …
                 per_expert_dispatches=0, per_expert_gpu_invocations=0, …
                 cpu_tier_pairs=159264, created_onednn_kernels=0
      # B60 ratio-99 census S5 (first run, probe mixture)
      lgc  slot 0: decode     64 tok in 115.31 s (  0.6 t/s)
      lgc  slot 0: prefill   256 tok in 343.19 s (  0.7 t/s)
      [OTD_PERF] gpu_hits=2154, gpu_misses=67369, gpu_hit_rate=3.09826%, …
                 per_expert_dispatches=69523, per_expert_gpu_invocations=30778, …
                 cpu_tier_pairs=383971, created_onednn_kernels=0
      # Affine d48g ratio-75 (first run)
      lgc  slot 0: decode     64 tok in 758.25 s (  0.1 t/s)
      lgc  slot 0: prefill   256 tok in 1172.74 s (  0.2 t/s)
      [OTD_PERF] gpu_hits=22539, gpu_misses=86539, gpu_hit_rate=20.6632%, …
                 per_expert_dispatches=69899, per_expert_gpu_invocations=14930, …
                 cpu_tier_pairs=575473, created_onednn_kernels=0

  **§3.4 / V4 finding — V4 FIRES (RED) on the dispatch route.**
[measured-here, `code`] The A770 ratio-99 arms differ ONLY in the resident
seed and produce different greedy answers: splitmix64 `55dff6f2…` vs census
`2e7c508f…` (the splitmix64 ledger-hit repeat reproduced `55dff6f2…`, and
the A770 served path is bit-identical across forwards, so this is not
run-to-run noise). VENICE clause V4 fires literally: the census-seeded policy
changes a served greedy digest. The cause is structural: under
`--moe-per-expert-dispatch` a resident expert is computed by the GPU
per-expert kernel while a miss is computed by the host tier, and the two
paths are not bit-identical by construction (`code`: the GPU kernels run at
the plugin's execution precision, the host tier decodes rows to f32 scratch);
so the served output depends on which experts happen to be resident — what
DESIGN §3.4 forbids. The VENICE quality row's **PASS / no V4** was measured
WITHOUT `--moe-per-expert-dispatch` (every routed expert on the host tier,
residency moves bytes, not arithmetic) and therefore does NOT cover this
route. This is recorded as a NEW open item, not smoothed: the speed route
must either make GPU and host expert arithmetic bit-identical or be admitted
only where residency cannot move arithmetic.

  **Ratio-99 VENICE budget: V1.** The pinned gate (`docs/window-052.md`,
G = 1.10, 2026-09-22) reads **0.556 < 1.10 × 0.526 = 0.579 t/s** on the
A770 (same-day host-tier comparand) and **0.555 < 1.10 × 0.8 = 0.88 t/s** on
the B60 (recorded band's upper edge). The speed row stays EMPTY.

  **The sweep says where the win is.** At ratio 75 (128 slots/layer, census
top-128 = 67.66 % corpus coverage) the same-config host control is 0.465 t/s
and the census-seeded resident route is **0.842 t/s = 1.81×**. Fitting
`R(h) = H/(1 − h(1−ρ))` to that pair with the unrounded rates
(`H = 0.464810`, `R = 0.842327`, `h = 0.483557`) gives **ρ = 0.073**
`[derived]`, i.e. the card computes a resident expert pair ~13× faster than
the host; the ratio-99 point is consistent within ~2 % (predicted 0.542 vs
measured 0.556). The ratio-99 budget's 5 slots/layer hold only 3.77 % of the
served request's pairs, whose free-card ceiling (`ρ = 0`) is
`1/(1 − 0.0377) = 1.039`, so no 1.10× win is reachable there. This is the
campaign's "the rate win needs the resident fraction" made numeric; it is NOT
a defect. The host control uses the fused/tier path (no per-expert dispatch),
so it prices the host tier without the per-expert dispatch overhead; the
1.81× is therefore a lower bound on the per-expert route's advantage.

  **G-pin correction (dated in place, `docs/window-052.md`).** The prediction
commit's `ρ ≈ 1.55` was drawn from the ratio-75 counters BEFORE their phase
composition was attributed; the served-only measurement gives
`ρ = 0.073 [derived]`, so the card is fast and the ratio-99 shortfall is a
hit-fraction limit, not a per-pair limit. The pinned G stays 1.10 and the
predicted verdict (V1) holds; its reasoning was wrong and is corrected in
place.

  **Ratio 50 refused / unusable.** A770: the plateau probe throws
(`clEnqueueWriteBuffer, error code: -5 CL_OUT_OF_RESOURCES`) and the analytic
pinned-pool fallback prices 28.12 GiB against the 16 GiB card — the static
partition refuses the 16 GiB card at ratio 50 (DESIGN §7.0.2ai). B60: ratio
50 LOADS (probe 0.37 GiB device) but the request did not return in 62 min
(the connection dropped, no decode line); the process was killed and no rate
was read.

  **Probe-skip, shown same-numbers.** `--fit-ledger-dir` skipped the
14–27-minute load probe on the second matching run; the A770 ratio-99
splitmix64 original and its ledger-hit repeat produced the **same greedy
answer** (`55dff6f2…`) at 0.535 / 0.547 t/s, and the ratio-75 census original
and repeat produced the same `5cd2e195…` at 0.836 / 0.842 t/s. The plugin,
artifact and prompt are unchanged across the pair; only the probe skip
differs.

  **Affine per-expert path (patch 0040, u4 artifact `d48g`) under 0047:
UNBLOCKED.** [measured-here] A770 ratio 75, same harness:
`per_expert_dispatches=69899, per_expert_gpu_invocations=14930`, hit
20.66 %, no fault, answer `ed01eb71…`. The rate (decode 64 tok in 758 s) is
NOT comparable: the run was cold on ZFS (`avg_disk_io_us=11901`, 481 s of
disk I/O in the counters). The owed cell is therefore closed as
*serves under 0047 with a non-zero card counter*; a warm rate is not claimed.

  Open after this leg: (1) the dispatch route's §3.4/V4 answer-dependence
(above); (2) the ratio-99 budget's realized hit (3.77 %) is below the corpus
top-5 coverage (9.96 %) because a 256+64-token request does not sample the
whole corpus — whether a longer served request at ratio 99 reaches the
9.96 % coverage is the hot-set/LRU campaign's to measure; (3) the 5-slot
pool's ledger/engine off-by-one (`docs/window-052.md`). [DATED IN PLACE
2026-09-22 (V4 leg): item (3) is CLOSED as an intentional divergence, not an
open defect — see the V4-quantification entry below.]

- 2026-09-22 (V4 quantification leg) — **the card-vs-host arithmetic
divergence on the native dispatch route is measured: the native per-expert
kernels are NOT bit-identical to the host tier, and the affine per-expert
kernels ARE.** [measured-here + code] This leg does not re-derive the V4
finding; it quantifies it. Three results.

  **(1) The text-level difference is a single early branch, not a tail
drift.** The two A770 ratio-99 answers differ from token index 3 (0-based)
of the 64-token greedy continuation: they share 17 characters
(`           (*pos == `) and diverge at byte 18 (`'/'` vs `'>'`), after
which **61 of 64 re-encoded token positions differ** (`tools`-free
re-encode with the artifact's own `tokenizer.json`; incumbent 185 chars /
64 tokens, census 210 chars / 60 re-encoded tokens, both 64 generated
tokens). Command and raw output in the session's `v4-diff.txt`;
`sha256` of the texts `55dff6f2…54550` and `2e7c508f…c10e`. This is the
worst shape for admissibility: a one-token branch at position 3 that never
re-converges.

  **(2) The difference is CONFIG-deterministic, not run-to-run noise.**
`measured-here`, A770 (GPU.1, PCI 8086:56a0), one fresh process per arm,
native `d48n`, plugin `ov-0047` (`f021de51b5812ee2`), ratio 99, KV u8, one
lane, the capture window-0's first 256 ids, greedy 64, temperature 0: the
incumbent `splitmix64` seed produced `55dff6f2…` in its original run and its
ledger-hit repeat; the census S5 seed produced `2e7c508f…` in its original
run and in the repeat taken this leg (`logs/sub4bit-r99-cen-rep/`, sha256
`2e7c508f…` again). Each seed reproduces its own digest; the two seeds keep
differing. No arm failed to reproduce.

  **(3) The numeric divergence, measured on a one-layer native MoE block
(E = 64 experts, top-2, hidden 512, inter 256), one fresh process per arm.**
[measured-here] Two arms differ only in `--moe-per-expert-dispatch`: the
host-tier arm computes every routed expert through the CPU tier, the
dispatch arm routes the resident experts through the GPU per-expert kernel.
At the same ratio the graph, weights, router and shared expert are
identical, so the difference is exactly the per-expert kernel's arithmetic.
The affine control (u4 artifact) is the discriminator. Raw arms:

      ARM mode=affine  E=64 ratio=50 dev=GPU.1 cap=32
        max_abs=0.000000e+00 mean_abs=0.000000e+00 mean_abs/rms=0.000000e+00
        frac_moved=0.0000 bit_identical=True repeat_bit_identical=True
        [OTD_PERF] per_expert_gpu_invocations=40 ...
      ARM mode=native  E=64 ratio=99 dev=GPU.1 cap=1
        max_abs=8.270264e-03 mean_abs=1.753572e-04 mean_abs/rms=4.937727e-02
        max_abs/rms=2.328750e+00 frac_moved=0.1250 bit_identical=False
        repeat_bit_identical=True rms=3.551375e-03
        [OTD_PERF] per_expert_gpu_invocations=4 ...

      PROVENANCE NOTE (2026-09-23, coordinator): these arm lines were printed to
      stdout by the leg and NOT saved by it, so the raw output was recovered
      from the leg's own transcript and persisted on the operator-local evidence
      path for this campaign (not in this PUBLIC repository); the recovered block
      matches every figure quoted here, byte for byte. Recorded because a
      measured claim whose raw output exists only in a transcript is one
      session away from being unverifiable.
      ARM mode=native  E=64 ratio=75 dev=GPU.1 cap=16
        max_abs=1.093864e-02 mean_abs=6.558856e-04 mean_abs/rms=1.846850e-01
        max_abs/rms=3.080115e+00 frac_moved=0.3748 bit_identical=False
        repeat_bit_identical=True rms=3.551375e-03
        [OTD_PERF] per_expert_gpu_invocations=20 ...
      ARM mode=native  E=64 ratio=50 dev=GPU.1 cap=32
        max_abs=1.093864e-02 mean_abs=1.058253e-03 mean_abs/rms=2.979840e-01
        max_abs/rms=3.080115e+00 frac_moved=0.7495 bit_identical=False
        repeat_bit_identical=True rms=3.551375e-03
        [OTD_PERF] per_expert_gpu_invocations=32 ...
      ARM mode=native  E=64 ratio=50 dev=GPU.0 cap=32
        max_abs=1.093864e-02 mean_abs=1.058254e-03 mean_abs/rms=2.979843e-01
        frac_moved=0.7495 bit_identical=False repeat_bit_identical=True
        [OTD_PERF] per_expert_gpu_invocations=32 ...

  Reading: the **affine per-expert route is bit-identical to the host tier**
while dispatching (`per_expert_gpu_invocations=40`), so the dispatch
mechanism, the slot indexing, the gather/reduce and the f16 output buffer
are exonerated. The **native route is not**: 12.5 / 37.5 / 75.0 % of output
elements move as the resident fraction grows (cap 1 / 16 / 32), max |diff|
8.3e-3…1.1e-2 and mean |diff| 1.8e-4…1.1e-3 against a reference rms of
3.55e-3 — a spread far beyond f16 ulp noise, deterministic across two
compiles in-process (`repeat_bit_identical=True`), and **the same numbers on
the 24 GB card as on the 16 GiB card** (card-independent, so it is the
kernel, not the card).

  **Named suspect, and what is exonerated.** [code + measured-here] The two
per-expert kernel families differ in exactly one arithmetic place: the affine
`expert_gate_up` writes `up(x)` into the f16 output first and then multiplies
by `MOE_GATE_ACT(gate)` in place (a two-stage f16 rounding), matching the host
tier's `h = f16(f16(up) * act(gate))` (`moe_cpu_expert.cpp` patch 0043); the
native `expert_gate_up_native` casts the product once,
`yrow[n] = (MOE_DTYPE)(su * MOE_GATE_ACT(sg))`, with `su` kept f32 (a
one-stage rounding), and accumulates in f32 per element rather than the
affine path's half FMA chains. The suspect is that coupling — the native
gate_up's stage structure and its subgroup f32 reduction — against the host
tier's staged f16 arithmetic. **Exonerated by measurement**: (a) the IQ4_NL
decode and indexing, by a delta-input down GEMV on the standalone kernel
(`mid = e_7`, so each output row isolates one decoded weight): max |diff|
0.095 on values of rms 68.9 (~1.4e-3 relative, the f16 output ulp), i.e. the
block decode is exact; (b) the IQ3_XXS decode, by a serial device-side dump
of rows 0–2 against `native_expert`'s decode: `max=0.0`; (c) the slot
addressing and the fill, by the affine route's bit-identity at the same
slot stride. Not exonerated, and **not asserted**: the standalone subgroup
GEMV harness did not reproduce the plugin's exact dispatch geometry, so the
arithmetic attribution above is a suspect whose magnitude is bounded by (3),
not a confirmed single-line cause. What would confirm it: a standalone
per-expert GEMV harness built with the plugin's own `{1, SUBGROUP_SIZE,
SUBGROUP_NUM}` geometry and `N_BLOCK`, compared against references for both
rounding rules (two-stage vs one-stage), or a diagnostic plugin variant that
rounds `up` to f16 before the gate multiply — a measurement, not the retired
F1 bit-equalisation work.

  **What it means for the policy (stated, not decided).** [measured-here]
The divergence is not confined to the resident experts' own rows: at the
ratio-99 VENICE budget only **3.77 %** of served expert pairs are resident
(5 slots/layer), yet that fraction is enough to branch the greedy answer at
token 3 of 64. So the native route's residency moves arithmetic, and a
tiny resident fraction already changes the served text. The evidence favours
limiting the native dispatch route to configurations where residency cannot
move arithmetic — which, for this route, is no residency at all — unless the
native kernel is made bit-equal to the host tier; the affine per-expert
route (bit-identical, measured) is the one route on the record that
satisfies §3.4 while dispatching. The choice between making the native
kernel bit-equal, admitting the route only where arithmetic cannot move, or
falling back to the affine route is the operator's; this leg decides none of
them and does not start F1 work.

  **Process.** One A770 window for the numeric arms (plugin `ov-0047`,
staged runtime), one A770 served window for the census repeat; cards left
clean, no `arcint` process, units inactive as found. Raw arms:
`logs/v4-final/arms.txt`; the repeat: `logs/sub4bit-r99-cen-rep/`.

- 2026-09-22 (slot off-by-one, CLOSED as intentional) — **the plugin's
integer division is the SERVED TRUTH; the engine's `ceil` is a fit-side
ledger ceiling only.** [measured-here + code] Operator decision. The plugin
sizes the resident slot pool as `const_shape[0] * (100 - ratio) / 100`
integer division (`moe_offload_constant.cpp`, `prepare_moe_otd_params`,
both the ordinary OTD path and patch 0047's per-expert branch): at ratio 99
with 512 experts that is **5** slots/layer. The engine's own ledger
(`src/exec/fit.h`, `expert_slot_bytes_static`) prices `ceil(512*1/100) = 6`;
that six was only ever a reservation/ledger figure, **never a plugin buffer
size** — the ratio-75 discriminating run already showed the divergence is
not the mechanism (both formulae give 128 there and the pre-0047 fault
reproduced unchanged), and patch 0047's real fix (a resident-sized pool)
removed it. Every served reading in this campaign and in
`expert-hot-set-lru` is against the **5-slot** pool, and the served census
seed is the corpus top-5 accordingly. The item is **CLOSED as an
intentional divergence**, not an open defect: no plugin change to `ceil`, no
ledger change to integer division, no code movement. Future sessions must
not "fix" it.

- 2026-09-22 (V4 leg, review) — **carry-forward, recorded not patched.**
The V4 numeric harness is operator-local (it needs the staged `ov-0047`
runtime and the dev host's card), so it is not committed as a `tools/`
script; the reproducible command, the raw arm output and both served digests
are on the persistent paths named above. The one-layer MoE block's
`rms = 3.55e-3` makes `mean_abs/rms` a cancellation-sensitive ratio; the
absolute `max_abs`/`mean_abs` are the stable reading. No acceptance row
moves on this leg: `window-052.md`'s speed row is untouched, V1 stands, and
the ratio-75 point stays a sweep point, not a gate.

- 2026-09-24 (dated append — operator backlog: the fully-resident NATIVE route
does not exist) — prompted by the operator's question whether a fully-on-GPU
MoE option still exists. Verified in the dev branch:

  * The percentage range is **[0, 100]** (`code`: `src/config.cpp:849`,
    `--offload-ratio must be a percentage in [0, 100]`). `--offload-ratio 0` is
    therefore accepted, and omitting the flag leaves the default 0; both mean
    **every expert resident on the card**. The fully-on-GPU path EXISTS for the
    fused path: it is the default when the flag is omitted (`code`:
    `src/config.h:149`, `offload_ratio = 0`), and it is what the deployed MoE
    coder unit runs — the other deployed unit is dense and has no MoE at all;
    both omit the flag.
  * `--moe-cpu-tier` with ratio 0 is refused by design (`code`:
    `src/config.cpp:862`): the host tier would have nothing to compute.
  * **The gap** (`code`): for the NATIVE formats there is no fully-resident
    configuration at all. Patch 0043 asserts
    (`contrib/packaging/marfrit-openvino/patches/0043-*:726-729`)
    `OPENVINO_ASSERT(!native || (_cpu_tier && _weight_provider->is_offloaded()),
    "native expert formats (IQ4_NL / IQ3_XXS) need OFFLOAD_RATIO in (0, 100)
    and MOE_CPU_TIER=YES …")`. The **ratio-0 half is verifiable in-tree**: at
    ratio 0 arcint never sets the property, so the provider is not offloaded and
    the native tensors have no reader on the fused path. The **ratio-100 half
    rests on the assert's own message string**, not on a condition evaluable in
    this tree — recorded as such rather than asserted.
  * Root cause is plumbing, not the assert: arcint sets the plugin's
    `OFFLOAD_RATIO` property only when the ratio is `> 0` (`code`:
    `src/exec/backend_ov.cpp:1183`, `:1224`), so at ratio 0 the provider is
    never marked offloaded and the native tensors have no reader on the fused
    path. The assert is the symptom.

  **Lever (backlog).** Give the slot pool an all-resident configuration:
ratio-0 semantics = `n` resident device slots, provider offloaded, NO host
tier. Gate: serve a native-format MoE with every expert resident on the GPU and
measure decode rate plus repeat determinism — byte-identity is against its OWN
configuration, because the native route is NOT bit-identical to the host tier
(DESIGN §7.0.2cf). Owner: this campaign's step 3, the resident-compute path the
VENICE speed row is held on. Evidence class: `code` for both guards;
`measured-here` for the cost of the tier route itself
(`docs/benchmark-served-services.md`, 2026-09-24: the coder's `--moe-cpu-tier`
arms at R = 99 and R = 75 lose 66–99 % of decode/prefill and move the answer
digests, with the resident share deciding whether they diverge).

  **Graduation rule:** recorded as a lever here because the owner is this
campaign's native route; if it grows a gate of its own (a served acceptance),
the campaign rules say it becomes its own document.

- 2026-09-24 (IQ3_XXS leg, dated append — the 35B-A3B conversion is BLOCKED
  before any card window) — Operator decision: *"IQ3_XXS first with conversion
  and benchmark, then continue the roadmap"*, aimed at a fully-resident A770
  route for Qwen3.6-35B-A3B. The fetch is DONE and verified; the conversion is
  BLOCKED on the emitter's model family and on the source file's expert format,
  both measured, so no benchmark arm runs. No conversion, fit or rate is
  fabricated. Evidence class is `measured-here` unless stated.

  * **Fetch (card host, ZFS pool, not the ext4 store):** HF repo
    `unsloth/Qwen3.6-35B-A3B-GGUF`, file `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`,
    **13,211,155,424 B**, magic `GGUF`, sha256
    `9c964e657212fea1f24905dd7b0a89b82fd807d19fab0b41da14251b07b88fbe` — the
    exact HF LFS oid of that path (`measured-here`: `sha256sum` of the landed
    file; size and oid are the API's own). Size alone proves completeness only;
    the hash is the check.
  * **The shard is `qwen35moe`, not the emitter's `qwen4_exp`.**
    `general.architecture = qwen35moe`, `qwen35moe.block_count = 40`,
    `expert_count = 256`, `expert_used_count = 8` (`measured-here`: gguf-py on
    the fetched file). The native serving-shape route emits ONE graph family —
    the Flash-Next `qwen4_exp` backbone: `export_serving_artifact.py` keys the
    shard admission on `qwen4exp.ple.eos_token_id` (`code`:
    `tools/export_serving_artifact.py:78`) and `build_serving_shape_ir`
    hardwires the hyper-connection width and the PLE (`code`:
    `tools/q4e/serving_shape.py:1267`, `:1290`, `:1385`, `:1401`), while
    `q4e/piecewise_export.py:129`'s `REAL_GEOMETRY` is 48 layers / 512 experts /
    `hc_count 4` / `ple_layer_ids [2]`. This GGUF carries none of it
    (`measured-here`: its 54 metadata keys hold no `qwen4exp.*`,
    `hyper_connection.*` or PLE key). Driving the named exporter returns the
    refusal verbatim:

        EXPORT [shards] REFUSED: the first shard lacks one of
        qwen4exp.ple.eos_token_id=None tokenizer.ggml.eos_token_id=248046
        tokenizer.chat_template=present

  * **The experts are not the native formats the route carries.**
    `ffn_gate_exps` and `ffn_up_exps` are **IQ2_S** (ggml type 22) on all 40
    layers; `ffn_down_exps` is IQ3_XXS on 37 layers and IQ4_XS on 3
    (`measured-here`: gguf-py type histogram `Q6_K 252, F32 361, IQ3_XXS 37,
    IQ2_S 80, IQ4_XS 3`). `q4e.native_blocks` splits IQ4_NL / IQ3_XXS / IQ4_XS
    / Q8_0 only (`code`: `tools/q4e/native_blocks.py:220`) and the plugin's
    native `weight_format`s are IQ4_NL=1, IQ3_XXS=2, Q8_0=3 (`code`: patch
    0043, `kWeightFormat*`); neither carries an IQ2_S decode (`code`:
    `src/core/gguf_dequant.cpp`'s `dequantize_row` switch has no IQ2_S case).
    The named fill refuses verbatim:

        blk.0.ffn_gate_exps.weight: IQ2_S is not a native expert format this
        fill carries (['IQ3_XXS', 'IQ4_NL', 'IQ4_XS', 'Q8_0'])

  * **Consequence.** No arcint artifact carrying this checkpoint's experts
    exists or can be driven from the named tooling, so the A770 arms (fully
    resident, or resident + tier) cannot run for the requested model. The
    fully-resident NATIVE gap recorded immediately above is real but belongs to
    the `qwen4_exp` native artifact; it does not transfer to a `qwen35moe` one.
    **OWED: an operator decision** — (a) port the serving-shape emitter to
    `qwen3_5_moe` and add an IQ2_S native decode across emitter, plugin and
    tier (a campaign-sized change, not a leg), or (b) name a different target;
    the Flash-Next native artifact already carries IQ3_XXS/IQ4_NL experts and
    its open lever is the fully-resident configuration.
- 2026-09-24 (IQ3_XXS leg, dated append — the operator decision and the IQ2_S
  decoder) — Operator decision on the blocked conversion: **port the
  serving-shape emitter to `qwen3_5_moe` and add an IQ2_S native decode**.
  Design note: `docs/design-qwen35moe-serving-shape.md`. Landed this session:
  `q4e.native_blocks.iq2_s_split` / `iq2_s_decode` + the 1024-entry
  `iq2s_grid`, red-first cells (`tests/python/test_native_blocks.py`),
  bit-exact against gguf-py on the real shard — `blk.0.ffn_gate_exps.weight`
  and `blk.0.ffn_up_exps.weight`, two experts, every row, **max|diff| 0.0**.
  The hand-built cell is mutation-sensitive: moving the sign bytes into the
  low-index region flips its sign assertion red. Evidence class `code`
  (llama.cpp `dequantize_row_iq2_s`) + `measured-here` (the gguf-py oracle).
  OWED: the plugin `kWeightFormatIq2S` + pattern/tier/OCL (design note §4),
  the `qwen3_5_moe` emitter (design note §5), the full-depth export, and the
  A770 window (gate, design note §6).
- 2026-09-24 (qwen3_5_moe port leg, dated append — the plugin format, the
  emitter, a depth-4 export; the A770 window stays OWED) — Operator decision
  executed: the serving-shape emitter is ported to `qwen3_5_moe` and IQ2_S is
  a native plugin format. One card-leg-free session. Evidence class per
  disposition (`code` / `measured-here`).

  * **The four conventions, measured BEFORE the code** (design note §5; no
    card). (1) GDN output gate = **silu** (`code`: llama.cpp
    `src/models/qwen35moe.cpp` `build_norm_gated` -> `ggml_silu`;
    `measured-here`: the served int4 IR's `linear_attn.norm` chain carries
    `aten::silu/Swish`, the only `Sigmoid` in `linear_attn` is on beta, and
    GGUF `ssm_norm` == the IR Constant to `max|diff| 0.0`). (2) value/key head
    map = **tiled** (`measured-here`: every value-head-indexed GDN tensor in
    the GGUF is the HF interleave order re-laid into llama's tiled order —
    sigma-map vs identity max|diff|: `attn_qkv` v `0.012` vs `0.395`,
    `attn_gate` `0.012` vs `0.297`, `ssm_out` `0.036` vs `0.413`, `ssm_alpha`
    `0.0069` vs `0.171`, `ssm_beta` `0.0043` vs `0.085`; `code`: llama.cpp
    `ggml_repeat_4d`). (3) norms = **plain RMSNorm, no `(1+w)`**,
    pre-norm residual (`measured-here`: GGUF `attn_norm`,
    `post_attention_norm`, `ssm_norm`, `attn_q_norm`, `attn_k_norm` all equal
    the served IR's Constants to `max|diff| 0.0`, no `+1` in the chain; so
    `qattn._rmsnorm_hd` takes `norm_plus_one=False` for this family).
    (4) the tiled MoE lowering is **E-agnostic** (`measured-here`, device-free:
    a 256-expert top-8 block compiles to **3 GatherMatmul** primitives on the
    CPU plugin exactly as 512/top-10; `code`: the matcher has no E bound).
  * **Plugin patch 0050** (`0050-native-expert-iq2s-format.patch`, both patch
    dirs): `MOECompressed::kWeightFormatIq2S = 4`, `NativeIq2sWeightsBlock`,
    the `[E, ofm, K/32, 8]` weight branch, the CPU tier row decoder +
    `kIq2sGrid[8192]`, the OpenCL `native_dot_iq2s` for gate/up, and the IQ2_S
    scale-transpose skip in `moe_otd_runtime.cpp`. **IQ4_XS down (3 tensors)
    needs NO new format**: `iq4_xs_split` folds its 6-bit sub-block scales onto
    the IQ4_NL layout 0043 carries, so those tensors ride
    `kWeightFormatIq4Nl` (`measured-here`: `blk.34/38/39.ffn_down_exps`,
    split -> decode vs gguf-py `max|diff| 0.0`). Built clean on the pinned tree
    2026-09-24 (`ninja -j6 openvino_intel_gpu_plugin` in `build-prod`, **rc
    0**, 47 targets); the plugin unit cell compiles to an object with the
    plugin's flags and the vendored gtest headers. A walker PASS is not a
    compile and no card compile is claimed.
  * **Emitter + depth-4 export** (`measured-here`, device-free): the
    `build_qwen35moe_serving_shape_ir` port (plain pre-norm residual; reuses
    `q4e.gdn` tiled, `emit_stateful_attention` with `norm_plus_one=False`,
    `emit_moe_tiled` with the native filler) wired into
    `tools/export_serving_artifact.py` (`--family qwen35moe`, auto-detected).
    The depth-4 artifact `qwen36-35b-a3b-d4n-ov` (operator-local path):
    **4 layers = 3 GDN + 1 attention, 1476 nodes, 12 native expert bodies
    (2,415,919,104 B), LM `.bin` 4,284,499,713 B**, peak host 10.88 GiB. It
    **loads** (`ov.Core().read_model`, 1476 nodes, CPU compile in the ad-hoc
    leg; the artifact `read_model` alone in the commit-time check) and **all 12
    gate/up IQ2_S and all 4 down IQ3_XXS blocks are byte-exact** — the emitted
    codes/signs equal `iq2_s_split`/`iq3_xxs_split` and the decode reads
    `max|diff| 0.0` against the GGUF. Red-first cells in
    `tests/python/test_qwen35moe_serving_shape.py` (a low-byte-only index
    mutant is caught; the 256-expert cell asserts 3 GatherMatmuls) plus the
    mutation-sensitive `test_native_blocks.py` / `test_native_expert_gemv.py`
    cells.
  * **Suite**: `tests/python/test_native_blocks.py`,
    `test_native_expert_gemv.py`, `test_qwen35moe_serving_shape.py` = **28
    passed, 5 skipped**; `test_serving_shape.py` = **36 passed, 1 skipped**
    (regression on the `_rmsnorm_hd` default). The patches mirror cell = **1
    case run, 0 failed**.
  * **OWED**: the A770 window (the gate: GPU compile of the native IQ2_S
    pattern/kernel + served rate/digests), the full-depth 40-layer export, the
    served binary's `qwen3_5_moe` load path (the artifact config carries the
    geometry; `artifact.cpp`'s admission and the served loop for a non-PLE
    family are unverified), and cell 4's card half. No card was touched
    (`pgrep -x arcint` = 0 before and after); no rate, fit or export beyond
    depth 4 is claimed.
- 2026-09-25 (served-side admission leg, device-free — the A770 window still
  OWED) — the OWED "served binary admits/serves the `qwen3_5_moe` family" item
  is closed to the edge of the card. One card-leg-free session. Evidence class
  per disposition: `code`, `measured-here` (device-free), or a build result.

  * **Registry entry** `qwen3.6-35b-a3b-native-d4` for
    `qwen36-35b-a3b-d4n-ov`: `model_type qwen3_5_moe`, `n_layer 4`, 256
    experts, `arch_hash 391bd21db6368d57`, template `55d4931433fe502b`,
    `weights_bytes 4,284,499,713` — read off the artifact's own manifest with
    `arcint --model <dir> --inspect-artifact`, never guessed. The entry says
    plainly it is a measurement artifact: depth 4 of 40, not the model's
    answers. `models/allowlist-raw.json` carries the matching row
    (`tests/test_provenance.cpp` holds the two together).
  * **`weights_bytes` is a contract now**: the allowlist pinned the byte count
    and `validate_artifact` never read it — a re-exported `.bin` passed on its
    xml hash. `ArtifactInfo` carries it (set in `Artifact::to_info`) and
    `check_u64` refuses a mismatch or a missing report.
  * **No PLE / n-gram table, first-class** (`code` + unit cells): `ngram::
    check_declared_table` returns `""` when the config declares no table and
    the IR declares no `ngram_table.K` port (the `qwen3_5_moe` case: the
    binding is INERT, `--ngram-gguf` is not needed) and a named refusal when
    the config DOES declare a table the graph cannot carry. `bind_ngram_ports`
    calls it in the empty-plan branch; `feed_ngram_ports` feeds the GDN
    `conv_mask` BEFORE the table-plan early return — the first form returned on
    `ngram_ports_.empty()` and left a required input unwritten on exactly this
    family.
  * **Red-first cells, mutation-tested** (`measured-here`): 2026-09-25,
    `check_u64`->no-op fails
    `registry_the_native_qwen35moe_rung_is_admitted_without_a_ple`;
    `check_declared_table`->always-`""` fails
    `ngram_ports_a_declared_table_with_no_port_is_refused_by_name`. Both
    restored green. `tests/test_registry.cpp` 19 cases, `test_ngram_ports.cpp`
    13, `test_provenance.cpp` 4; the whole device-free C++ suite is **609
    cases, 0 failed, 2 skipped**.
  * **Build-verified**: the served binary (`cmake --build` with
    `ARCINT_OPENVINO=ON`) and the test binary compile clean from the tip, rc 0
    on both the OpenVINO and the no-OpenVINO builds. `--inspect-artifact` on
    the depth-4 artifact now reports `admitted as qwen3.6-35b-a3b-native-d4`.
  * **OWED, stated not faked**: the GPU load of the native IQ2_S graph and the
    served arm — rate + digests against the int4 comparand (the recorded
    9.1 t/s at ratio 50), the fit verdict on 15.1 GiB, and the
    V4/determinism reading. No card touched (`pgrep -x arcint` = 0); the
    served load cannot run device-free, so the no-PLE path is `code` + unit
    cells, not a served reading.
- 2026-09-25 (the A770 window — the qwen3_5_moe/IQ2_S gate). The card was
  touched only inside this window (sampler on the physical host, `MemAvailable
  < 4 GiB` watchdog, one `arcint` leg at a time, 0 watchdog trips). Evidence
  class per disposition: `code`, `measured-here`, or `previously-measured`.

  * **The native IQ2_S graph compiles and serves on the A770** (item 1,
    PASS). A measurement build of the pinned OpenVINO with patches 0003–0050
    (debug caps OFF; GPU-plugin sha256 `582c3230…`, version
    `2026.4.0-22849-71640275d29-marfrit-p19`) compiled the depth-4 artifact's
    paged graph on `GPU.1`: "language model ready in 22.0 s (paged);
    device-resident 1.28 GiB". The plugin's per-expert path ran (counters
    below). The pre-existing plugin prefixes were not touched.
  * **The served sweep** (item 2), A770 `GPU.1`, u8 KV, chunk 2048, 8 GiB
    pool, `--moe-cpu-tier --moe-per-expert-dispatch --no-logits-slice`,
    `--n-ctx 32768`, extension = `(prompt − hit)/prefill_s`:

    | arm | R | slots | `T_boot` s | depth | hit | ext t/s | TTFT s | decode t/s | digest |
    |---|---|---|---|---|---|---|---|---|---|
    | n50 | 50 | 128 | 380.377 | 1 | 0 | 2.0 | 0.806 | 4.2 | `3f6d0ab1f8d9` |
    | n50 | | | | 4096 | 0 | 28.6 | 143.006 | 14.2 | `8cccdbac48ed` |
    | n50 | | | | 16384 | 2048 | 28.8 | 497.132 | 14.0 | `e2a836c80c1f` |
    | n75 | 75 | 64 | 351.375 | 1 | 0 | 2.9 | 0.566 | 8.3 | `3f6d0ab1f8d9` |
    | n75 | | | | 4096 | 0 | 19.1 | 214.407 | 13.4 | `8cccdbac48ed` |
    | n75 | | | | 16384 | 2048 | 19.3 | 744.455 | 12.4 | `e2a836c80c1f` |
    | n25 | 25 | 192 | 251.168 | 4096 | 0 | 51.0 | 80.270 | 15.1 | `8cccdbac48ed` |
    | n99 | 99 | 2 | 395.360 | 4096 | 0 | 14.5 | 281.871 | 8.6 | `8cccdbac48ed` |

  * **The int4 comparand** (item 3), the existing full-depth 40-layer int4
    artifact at the same flags: `T_boot` 568.341 s, ext prefill 16.0 t/s
    @4096 / 26.1 t/s @16384, decode 5.4 / 5.3 t/s, digests `012397a89576` /
    `f3e9eb2ffa08` / `2d0ff8b1f891`; coherent text. `previously-measured`
    §7.0.2v's 9.1 t/s (ratio 50, 8 GiB, tier OFF) and §7.0.2x's 15.0/15.5
    (tier ON, no dispatch) are the fused-path references; this window's
    comparand adds the native route's own dispatch, which the int4 arm reads
    at 5.4 t/s. Depth differs (4 vs 40) — the supported delta is per-layer:
    the native IQ2_S layer costs ≈3.8× the int4 affine per-expert layer.
  * **The fit verdict** (item 4): the depth-4 artifact fits with ≈9 GiB
    headroom — reservation `weights+graph 1.28 GiB + drafters 0.95 + expert
    slots 0.15 + activations 3.40 + margin 0.25 + 1 x (GDN rows 9.6 MiB + KV
    1.1 KiB/token) of 15.11 GiB -> max ctx 8397168 per lane`. The
    **fully-resident NATIVE arm is BLOCKED** (`measured-here`, verbatim):
    `Check '!native || (_cpu_tier && _weight_provider->is_offloaded())' failed
    at .../moe_3gemm_swiglu_opt.cpp:1886: native expert formats (IQ4_NL /
    IQ3_XXS) need OFFLOAD_RATIO in (0, 100) and MOE_CPU_TIER=YES`. `code`:
    arcint sets the plugin's `OFFLOAD_RATIO` only when the ratio is `> 0`, and
    ratio 0 IS the fully-resident configuration. Lever unchanged.
  * **V4 does NOT fire here** (item 5) — a negative against the §7.0.2cf
    expectation. The served digest is byte-identical across ratios 25/50/75/99
    at depths 1 and 4096 (`3f6d0ab1f8d9` / `8cccdbac48ed`) and across 50/75 at
    16384 (`e2a836c80c1f`), while the counters show the per-expert route
    exercised hard (`measured-here`, `MOE_OTD_PERF_LOG=1`):
    R=25 slots 192, gpu hit 65.03 %, 407,480 GPU invocations, 1,629 CPU-tier
    experts; R=50 slots 128, 43.15 %, 273,416, 3,382; R=99 slots 2, 0.34 %,
    1,590, 6,818. Two explanations remain open: (a) patch 0050's IQ2_S kernel
    is bit-identical to the CPU tier, or (b) the 4-layer artifact's greedy
    output collapses to a repeated-token attractor robust to the perturbation
    (`measured-here`: its text is repetitive at every depth, unlike the
    coherent int4 comparand). A logits-level A/B is OWED; the digest reading
    does not prove (a).
  * **262144 reachability** (item 6): the run's own reservation supports
    `max ctx 8397168 per lane`, and an arm with `--n-ctx 262144` loaded and
    served. Reachable for this depth-4 artifact.
  * **The logits-slice finding**: without `--no-logits-slice` the load refuses
    (`measured-here`): *"logits slice did not take: 128 row(s) … shape
    [1,128,248320]"*. The `qwen3_5_moe` serving-shape export has the token
    axis 1 (`[1, tokens, vocab]`), the served-path slice assumes axis 0, and
    the same convention (`--no-logits-slice`) covers the `qwen4_exp`
    serving-shape runs (`code`: `kld-d48n.sh:35`).
  * **OWED**: the full-depth 40-layer export and its served reading; the
    logits-level V4 A/B; a depth-4 int4 comparand; the depth-4 artifact's
    answer quality (degenerate greedy text — truncation or an emitter
    weight-mapping defect, unresolved; device-free verification covered expert
    bodies and norms only). The 0.5.4 LYON roadmap item is untouched.
- 2026-09-25 (redirect leg — the speed defect, not another measurement). The
  operator redirected the leg off the benchmark sweep: make the native route
  faster. Evidence class per disposition: `code`, `measured-here`,
  `previously-measured`.

  * **The defect**: the all-resident native configuration — every expert
    resident, only the routed experts computed on the GPU per-expert kernel —
    was unreachable by a three-link dead end. `code`: `src/config.cpp` refused
    `--moe-cpu-tier` at ratio 0; `src/exec/backend_ov.cpp` set the plugin's
    `OFFLOAD_RATIO`/`ov::weights_path` only when the ratio was `> 0`, so an
    explicit `0` was swallowed; the plugin's `prepare_moe_otd_params`
    (`ops/moe.cpp`) set `lru_expert_num = 0` at `otd_ratio == 0`, so
    `moe_3gemm_swiglu_opt.cpp` selected the **Resident** provider (no slot
    pool, no native reader) while patch 0043's assert demanded
    `_cpu_tier && is_offloaded()` — unsatisfiable at 0. The assert's rationale
    ("until the OpenCL decode exists") was obsolete since patch 0045.
  * **The fix**: plugin patch **`0051-native-fully-resident.patch`** — a native
    format at ratio 0 enables the offload provider when `ov::weights_path` is
    supplied (explicit 0 distinguishable from unset), pool sized at
    `num_expert`; the assert requires only `is_offloaded()`. Arcint side: an
    `offload_ratio_set` flag, the pure `moe_offload_active()` decision,
    `Artifact` reading `expert_fill.format`, and the two config guards relaxed.
  * **Red-first**: `tests/test_config.cpp` gained four cells;
    `measured-here` mutation run reverting `moe_offload_active` to
    `ratio > 0` fails exactly
    `config_offload_active_covers_the_all_resident_native_case`; restored,
    `config` = **74 cases, 0 failed** (was 69). Both builds clean.
  * **The gate** (`measured-here`, A770 `GPU.1`, depth-4 artifact, u8 KV, chunk
    2048, `--no-logits-slice`, plugin `ov-0051`, `--offload-ratio 0
    --moe-cpu-tier --moe-per-expert-dispatch`): load **4.4 s**, device-resident
    2.71 GiB; decode **49.9 / 56.8 t/s** at 1 / 4096; ext prefill **155.9 t/s**
    @4096; digests `3f6d0ab1f8d9` / `8cccdbac48ed` — **byte-identical** to the
    tiered ratio-50 arm (`previously-measured`, same day: 4.2 / 14.2 t/s
    decode, 28.6 t/s prefill). **4.0× decode, 5.4× prefill.** Counters:
    `cpu_tier_pairs=0`, `cpu_tier_experts=0`, `per_expert_dispatches=6847`,
    `per_expert_gpu_invocations=544,832`, gpu hit 85.0 % — no expert touched
    the CPU tier. Caveats: the first fit pass failed on **fragmentation, not
    arithmetic** (pass 2 loaded by capping prefix-cache spare); the
    dispatch-without-tier variant refuses at `moe_3gemm_swiglu_opt.cpp:1141`.
  * **The sizing suspicion resolved** (`measured-here`): the artifact carries
    **14.469 GiB** of expert bodies against the GGUF's 10.346 GiB — **1.399×**
    (gate/up IQ2_S 1.561×, IQ3_XXS down 1.143×, IQ4_XS→IQ4_NL 1.059×). No
    mis-map: the same bytes, re-laid (sampled byte-exactness §10.3 of the
    design note). The "24 GB" premise came from the manifest's
    `expert_fill.filled_bytes` (24,662,507,520 B), which sums the **f32 split
    parts** and overstates the artifact by ≈1.55×; the lm `.bin` (21.82 GiB)
    confirms 14.47 GiB experts + ~7.35 GiB dense. The inflation is the
    plugin's native layout (u16 indices, two f16 scales per 32, vs the GGUF's
    packed qs+qh and 4-bit nibbles); a packed layout is the size lever, OWED.
  * **Full-depth coherence, confirmed** (`measured-here`): the interrupted
    d40n sweep (stopped when the priority changed) loaded and served; its
    depth-1 and depth-4096 text is coherent ("# Tools … execute_command …",
    "system: You are a function calling AI model …"), so the depth-4 rung's
    degenerate text was **depth-4 truncation**, not an emitter defect. Rate
    0.5 / 1.5 t/s decode, 2.6 t/s ext prefill @4096; 16384 not run; reservation
    max ctx 581,600/lane.
  * **OWED**: the full-depth all-resident arm does not fit (14.47 GiB experts
    vs 15.11 GiB VRAM) — the speed win is a small-depth one until a packed
    native layout; a logits-level V4 A/B; the 0.5.4 LYON roadmap is the
    operator's.
