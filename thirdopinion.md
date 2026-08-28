# Third opinion — ligence review against design, the four reference engines, and the two cards

Reviewed: `noether:/home/mfritsche/src/ligence` at HEAD `f96d49e` **plus the dirty
worktree** (86 insertions: the session's response to secondopinion findings 2 and 5 —
verified by reading the diff, not assumed). Nothing was touched.

Reference material cloned and read for this review:

- **NInfer** (`Neroued/ninfer`, full clone @ `6e8b2e2`) — same model family
  (Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.8-27B) on RTX 5090. The closest analogue in
  existence; read docs + hot source.
- **OpenVINO** (`openvinotoolkit/openvino` @ `a8354dc`, GPU plugin) and
  **OpenVINO GenAI** (`openvinotoolkit/openvino.genai` @ `468c261`, continuous
  batching) — the stack ligence runs on. GenAI's scheduler/model-runner is the oracle
  for the paged-hybrid interface ligence could not reverse-engineer (§3.1 below).
- **vLLM** (`vllm-project/vllm` v1 core) — hybrid KV cache manager, mamba/GDN paging,
  spec-decode bookkeeping.
- **llama.cpp** (`ggml-org/llama.cpp` @ `f5e85d4`) — qwen3_next support, MoE CPU
  offload, q8 KV, the fleet's own reference engine.

Hardware sources: Intel Arc Pro B60 datasheet (download.intel.com), Intel ARK A770
SKU 229151, Xe2 architecture analyses, and the OV GPU plugin's own Xe1/Xe2 code paths.

## Verdict

The implementation is honest about the design it actually ships — rarer than it
should be. The dirty-tree fixes close two of the second opinion's three remaining
findings with real code (MTP state now travels in the cache blob; snapshot cost is
surfaced). The one structural fact that governs everything else, confirmed from
three independent directions this time: **the paged path is not an optimization, it
is the unlock for four of the five remaining problems at once**, and — this is new —
its decode-side convention is now fully reconstructable from GenAI's code. §3.1
writes it down so it does not have to be reverse-engineered again.

Two corrections to the previous review matter (§5): "fp8 KV on Xe2" is not a thing
the OV GPU plugin offers, and the static-decode-graph-as-second-compiled-model plan
does not survive the VRAM arithmetic.

## 1. Design vs implementation — conformance audit

Checked the shipped claims against `src/`. Conformance is high; the deltas worth a
sentence each:

**Conforms, verified in code:**

- Absolute-grid chunked prefill + checkpoints restricted to the grid + the
  config-time coupling (`--prefill-chunk` must be a non-zero multiple of
  `--kv-block-size` when the cache is on) — `backend_ov.cpp` `prefill()`,
  `config.cpp:246-259`. The second opinion's addendum §2 ("canonical boundary
  grid") is implemented as described.
- q8 KV refused with the exact reason (cast without scales) — `config.cpp:231-239`.
- Blob cache prove-at-load-and-discard — `backend_ov.cpp` ctor; poisoned blob
  cannot reach a request; the custom-kernel switch correctly forces a cold compile.
- Verification = sampler decision (penalties applied), drafted tokens clear
  EOS/max_tokens/n_ctx gates in order — `backend_ov.cpp` decode loop.
- Prefix cache: SHA-256 hash chain keyed over (key, prev-hash, block tokens),
  token re-verification on every hit, collisions counted, LRU, byte budget,
  one-snapshot-larger-than-budget refused — `prefix_cache.cpp`. Matches §3.4's
  "hash chain, keyed 128-bit, token identity on hit".
- Sampling: repetition penalty llama.cpp-convention, prompt scoped to
  repetition-only, unseeded requests get a logged fresh seed — `sampler.cpp`,
  `backend_ov.cpp`.
- Stop/cancel mid-draft now rolls the state back explicitly (secondopinion finding 4)
  — committed at HEAD; the early-exit path carries `restore_tensors(rollback_)` with
  a comment explaining why "safe by structure" was not accepted.
- Dirty tree (uncommitted): prefix-cache snapshots now include the MTP head's own
  KV, its cursor (len/pos/pending), and the pending hidden row, and restore()
  size-checks the blob — a warm run can no longer draft from an unprimed head.
  `snapshot_seconds` surfaces in stats and on the console line. Both are the direct
  closure of secondopinion findings 2 and 5, and the combination gate
  (MTP × prefix cache, hit required, byte-equality required) is added to
  `tests/equivalence/run.sh`. **Commit this; it is a real correctness fix, not
  paperwork** — without it, MTP + prefix cache could silently change answers.

**Deltas (doc > code or code > doc):**

1. **README's M4 status row lies.** The status table says "M4 MTP | done —
   *byte-identical* with `--mtp on`", while the README's own feature bullet,
   DESIGN §3.5.2, and llm.txt all say it is **not** byte-identical to plain greedy
   (one near-tie flip in 64 measured). One-line fix; the wrong direction to drift.
2. **DESIGN §3.5 is stale about the allowlist.** It says "the allowlist now pins
   `has_mtp_head = false` for all three"; the registry pins `true` for qwen3.8-27b
   (`model_registry.cpp:99`), which is correct at HEAD (the head ships beside the
   artifact after `tools/export_mtp.py`). Update the paragraph.
3. **qwen3.8-27b still carries `status = "provisional, 7/10"`** in the registry
   though §7 measured 10/10 greedy through ligence. DESIGN itself flags this as
   deserving re-measurement; `/props` reports the stale number until then.
4. **The §3 architecture diagram promises a scheduler with continuous batching.**
   What exists is a SlotPool that does admission accounting and a backend mutex that
   serializes everything (`backend_ov.cpp` `generate()` takes one lock; there is one
   `lm_req_`). `--parallel N` buys queueing, not concurrency. That is consistent
   with the milestone wording (M1 = single sequence) but not with the diagram, and
   the `/health` slot figures imply concurrency that does not exist. §6 has the
   cheap fix.
5. `estimated_state_bytes()` hard-codes `128 + 4*5120` for the MTP cursor/pending
   row — fine today, but it is an n_embd baked into an arithmetic expression; make
   it derive from the pending tensor's real size.

**Not verifiable from here:** the measurement claims (they were taken on dirac;
pica has neither the cards nor cmake). The suite's *structure* — red-before-green
discipline, anti-inertness gates (non-zero acceptance, cache-hit presence),
reported-vs-gated honesty for chunking and speculation — reads as described and is
the strongest part of the tree.

## 2. The two cards (hardware-assisted inference facts)

From Intel ARK (A770 SKU 229151) and the Intel Arc Pro B60 datasheet:

| | **Arc A770** | **Arc Pro B60** |
|---|---|---|
| silicon | ACM-G10 (Xe-HPG), TSMC N6 | BMG-G21 (Xe2-HPG), TSMC N5, 272 mm² |
| Xe cores / vector engines | 32 / 512 (SIMD8) | 20 / 160 (SIMD16) |
| XMX engines | **512** | **160** (8 per core, 2048-bit per core) |
| peak INT8 (XMX, dense) | **262 TOPS** | **197 TOPS** |
| peak FP32 | not published (ARK) | 12.28 TFLOPS (⇒ ~2.4 GHz boost, derived) |
| memory | 16 GB GDDR6, 256-bit, **560 GB/s** | 24 GB GDDR6, **456 GB/s** (~22.7 usable) |
| PCIe | 4.0 x16 (fleet: **x4 Gen3, ~3.1 GB/s**) | 5.0-capable (fleet: **x8 Gen4, ~15.7 GB/s**) |
| TBP | 225 W | 120–200 W |

Derived: per-XMX-engine INT8 throughput is ~244 ops/cycle on A770 vs ~513 on B60 —
Xe2 roughly doubles the matrix engine per clock (Xe2 XMX handles FP16/BF16/INT8/
INT4/INT2; **no FP8 matrix datatype on Xe2** — the FP8 story starts later). Xe2
vector engines are SIMD16-only; the OV plugin literally encodes that
(`permute_kernel_f_y_axes.cpp:67`: "Xe2+ lacks SIMD8"), so kernel selection differs
between the two cards even for identical graphs.

What the numbers say about ligence's workload:

- **Decode is bandwidth-bound, not TOPS-bound.** The A770 has *more* INT8 TOPS
  (262 vs 197) yet decodes the coder at 29.4 t/s against the B60's 51.4. Roofline
  for the dense 3.8 (13.4 GiB q4): 40.7 t/s on A770 (560 GB/s), 33.2 on B60 — the
  measured 19.9 t/s on B60 is ~60% of bandwidth ceiling, i.e. already a
  bandwidth-shaped problem. For the A3B MoE the per-token weight traffic is only
  the active ~3B params (≈1.5–1.9 GiB q4), so the B60 roofline sits near 200+ t/s
  and the measured 51.4 is ~20–25% of it: **the MoE decode headroom is kernel and
  scheduling overhead, exactly the reference-kernel third §5 of DESIGN profiles.**
- **KV quantization is a bandwidth lever on both cards**, not a capacity lever
  only: at 262k context the fp16 KV reads (5 GiB circulating per token at depth)
  compete with weights for the same 456/560 GB/s.
- **PCIe asymmetry governs offload strategies.** The A770's x4 Gen3 (~3.1 GB/s) is
  brutal for weight streaming but adequate for activation-sized transfers — which is
  why `-ncmoe` works under llama.cpp there (activations cross, weights don't). The
  B60's x8 Gen4 (~15.7 GB/s) could even stream cold experts at decode rates if it
  had to.
- The fleet record that XMX never engaged on A770 (secondopinion's "locked out")
  is consistent with INT8 TOPS not showing up in decode: nothing about these
  decode profiles is matrix-engine-limited. Don't chase TOPS; chase bytes.

## 3. The Kniffe — what the four engines say about ligence's open problems

Ranked by payoff. Code pointers are to the local clones listed above.

### 3.1 THE paged path: the `la.*` decode convention is reconstructed — stop reverse-engineering it

GenAI's continuous-batching pipeline drives exactly the 91-input interface ligence
mapped (`ov::pass::paged_attention_transformation`). Its scheduler is the oracle,
and it is small:

**Non-speculative mode (plain prefill and decode — what ligence needs first)**
(`openvino-genai/src/cpp/src/continuous_batching/scheduler.hpp:1126-1135`,
filled in `model_runner.hpp:1809-1896`):

- One **physical row per live sequence**, shared across **all** state tables —
  the same row index addresses `conv_state_table.N` *and*
  `gated_delta_state_table.N` for every N
  (`cache/linear_attention_cache_manager.hpp:190`: block size **1**, "each block
  holds the full state for one sequence"). Fresh rows are **zeroed**
  (`cache_orchestrator.hpp:987-1005`).
- Per step, per sequence, all dtypes i32:

  ```
  la.block_indices        = [r, r]            # EXACTLY 2 entries, same row
  la.block_indices_begins = [0, 2, 4, ...]    # +2 per sequence
  la.past_lens            = tokens processed before this step
  la.cache_interval       = 0                 # in-place, no checkpoints
  ```

- Why your decode degenerated: the conv kernel **emits zeros when a sequence's
  block span is ≤ 1** (`ov/.../ocl_v2/paged_causal_conv1d_ref.cl:48`), and the GDN
  kernel reads the first entry and writes the second
  (`paged_gated_delta_net_ref.cl:76,170`). One entry per sequence = zeroed  conv output and an out-of-range GDN write — the exact "correct prefill,
  degenerate decode" symptom. There is also a doc trap: the plugin's
  `paged-causal-conv1d.rst` pseudocode says load state from the *last* assigned
  block; the kernel reads the *first* (they coincide only in the 2-slot in-place
  case). Trust the kernel and GenAI, not the rst.
- State tables keep the model's state dtype (f16/f32); the kernels support no
  fp8/u8 there. KV side: GPU block size 16, layouts
  `k [blocks, kv_heads, head_size, block]`, `v [blocks, kv_heads, block, head_size]`
  (`plugin/transformations_pipeline.cpp:838-845`).

**Speculative mode — rollback by block-table arithmetic, native to this stack**
(`scheduler.hpp:1093-1156`, reservation `scheduler.hpp:80-107`,
commit/promote `pipeline_impl.cpp:447-476`, `block_manager.hpp:1058-1104`):

- `la.cache_interval = 1` (checkpoint after *every* token of the verify window),
  `block_indices = [committed] + scratch_rows`, `1 + k` entries.
- Scratch rows are reserved up front; after sampling, the accepted checkpoint row
  is **promoted** to committed and the losers are freed. No copy of state ever
  happens on rejection — this is precisely the "rollback becomes block-table
  arithmetic" property DESIGN §3.5 speculated about, implemented and shipping
  upstream.

**Prefix mode — what upstream does with `cache_interval`, and the vindication of
ligence's §3.3 stance** (`scheduler.hpp:1158-1201`,
`cache_orchestrator.hpp:759-828`): `cache_interval = kv_block_size × multiplier`
with default multiplier **8** and an adaptive `ceil(la_block_bytes/kv_block_bytes)`
clamped to [8, 256] — i.e. upstream really does size GDN checkpoints for memory,
exactly the design choice DESIGN §3.3 argues against. ligence's block-aligned
(multiplier 1) position remains the correctness-first one; now it is also clear
*how* to express it through this interface.

**What adopting this path buys, all at once:**

1. `PagedGatedDeltaNetOptImpl` + `PagedCausalConv1D` — the plugin registry has an
   **optimized GDN kernel only for the paged path**
   (`registry/gated_delta_net_impls.cpp` = Ref only;
   `registry/paged_gated_delta_net_impls.cpp` = Opt + Ref). 30 of 40 layers are
   GDN; today they run on `ocl::gated_delta_net::ref___`.
2. `PagedAttention` replaces `IndirectSDPA` on the 10 attention layers — the
   L^1.46 growth (§5 profile, +13.7 ms at 4k) is the one IndirectSDPA cannot
   express.
3. `KV_CACHE_PRECISION` becomes real: i8/u8/i4/u4 accepted on the paged path
   (`execution_config.cpp:286-302`), scales handled by the plugin — **this is the
   only route to genuine q8 KV with per-block scales**, the reason DESIGN §3.3
   refuses the plain cast.
4. Speculative rollback (§3.5's 66%-of-decode tax) becomes the promote/scratch
   dance above — worth the measured ~2× on the dense model.
5. Prefix caching stops paying the serialization tax: a hit restores page-table
   rows + one LA row index instead of 75–171 MiB of `get_state()` copies.

**Recommendation:** adopt the paged path as the M-next centerpiece, with GenAI's
non-speculative fill as day one (it is ~20 lines of index arithmetic), speculative
mode as the MTP follow-up. The equivalence suite already has the right shape to
gate it: cold-vs-warm under paged restore must stay byte-identical.

### 3.2 MoE expert placement for the 35B on the A770 — there is a native OV knob nobody tried

The M5 gap is 17.4 GiB against 15.1; only ~2.5 GiB must live elsewhere. Three
options now, in order of effort:

1. **`ov::intel_gpu::offload_ratio` (0–100): "Percentage of model weights to
   offload to disk. Currently supported for MoE experts only"**
   (`ov/include/intel_gpu/runtime/options.inl:57`). The GPU plugin already ships an
   `OffloadExpertWeightProvider` with **LRU resident slots** and on-demand loads
   from the weightless `.bin` (`impls/ocl_v2/moe/expert_weight_providers.hpp`;
   partial upload at compile time `plugin/ops/moe_offload_constant.cpp:45-82`).
   This is HETERO-by-memory-pressure done natively, inside the fused MoE op that
   DESIGN §7 says exposes nothing. It wants `ov::weights_path` (which ligence
   already uses for the blob-cache workaround). **Measure this before anything
   else** — on the A770's x4 Gen3 the question is whether the LRU working set of
   hot experts stays resident; the fleet's imatrix statistics (which experts are
   cold) predict the answer.
2. **llama.cpp's actual `-ncmoe` semantics, corrected:** it pins the **first N**
   MoE layers' expert tensors to CPU via regex→buffer-type overrides
   (`common/common.h:1130-1147` — `blk.<i>.ffn_(up|down|gate|gate_up)_(ch|)exps`,
   i = 0..N-1), NOT the coldest/last N. The router stays on GPU; activations cross
   PCIe, weights don't; decode computes experts on CPU via quantized vec-dot,
   prefill (batch ≥ 32) pulls them back to GPU over host-visible buffers
   (`ggml-vulkan.cpp:18930-18949`). The general form is `-ot 'regex=CPU'`, so
   "spill exactly 2.5 GiB by chosen layers" is expressible. Replicating this in OV
   means subgraph-splitting the fused MoE op — which it resists; hence option 1
   first.
3. The fleet's proven fallback stays as the floor: llama.cpp `-ncmoe 30` on the
   A770 serves the 35B today (262k context, q8 KV). Whatever ligence builds must
   beat or match that, not merely exist.

Note for honesty's sake: option 1's provider lives in the very file carrying the
`#37607` assert (`moe_3gemm_swiglu_opt.cpp:2548`) — and at OV HEAD, `load()`
**recreates the weight provider on blob-cache import** (`:1278-1284`). The poisoned
blob class appears fixed upstream; retest on a current 2026.x build before
carrying the prove-and-discard guard forever. Keep the guard until then — it costs
45 s of startup and is strictly correct.

### 3.3 Static shapes — right diagnosis, wrong proposed mechanism

DESIGN §8 and the kernel excursion are correct that static `S=1` deletes the
GDN transposes outright (kernels/README.md measured it). The second opinion's
proposed mechanism — a second compiled model with static decode shapes "sharing
weights via `ov::CompiledModel` from the same `ov::Model`" — **does not exist**:
constant dedup is per-compile only (`ov/src/plugin/ops/constant.cpp:103-172`,
`program_builder.hpp:100-101`); two compiled models get two device allocations of
12.8 GiB. That does not fit the B60 (25.6 > 22.7), let alone the A770.

What does work: on the **paged path**, decode's shapes stop depending on context
entirely (past length is an *input*, state is an *input table*, tokens = 1..k is
the only sequence dim). A single compiled paged model therefore gets the static-ish
decode specialization *within one weight allocation* — the permute death by
dynamic shape dissolves because the paged GDN/conv kernels replace the transposed
subgraph rather than inheriting it. NInfer does the same thing one level up:
exact-batch captured graphs per (batch × frontier bucket), sequence identity as
input data, never as recompilation (`ninfer/src/core/decode_graph.*`,
`targets/qwen3_6/impl/runtime/program_impl.h:9697-9795`). If multi-sequence
serving ever lands (§6), that frontier-bucket scheme is the template; note NInfer
proves C=8 lanes with 2.2–5.7× aggregate decode on exactly these models.

### 3.4 KV quantization — the pattern, and what is NOT available

- llama.cpp's q8 KV (`-ctk/-ctv q8_0`) never dequantizes K: Q is quantized *to*
  K's block format and KQ uses integer dot products; V dequantizes row-wise into
  f32 accumulators (`ggml-cpu/ops.cpp:8553-8560`; Vulkan int8-dot FA variant
  `ggml-vulkan.cpp:4533-4541`). On Arc that is the DPAS/integer-dot path.
- OV's paged plugin does the storage side natively (u8 default with 4-bit weights,
  padded/packed layouts `transformations_pipeline.cpp:878-916`) — ligence does not
  need to write this codec, only to reach the paged path.
- NInfer goes further: INT8 **group-64** scales with a fused 256-wide Hadamard
  pre-rotation, or FP8 E4M3 row-scaled, encode fused into append, decode fused
  into attention (`ninfer/src/ops/kv_cache/*codec*`) — with published AIME/GPQA
  deltas. That is the custom-kernel endgame for KV bandwidth, not the starting
  point.
- **Correction to the second opinion: fp8 KV on Xe2 is not available.**
  `kv_cache_precision` accepts only i8/u8/i4/u4 on this plugin
  (`kv_cache_compression.cpp:142`, `execution_config.cpp:361-367`); the f8e4m3 work
  at OV HEAD serves an ONNX windowed-GQA op, not this path; and Xe2 XMX has no fp8
  matrix dtype. fp8 here would be a storage format with a dequant pass — nothing
  accelerates it. Use u8/i8.

### 3.5 Recurrent-state rollback — three engines, one convergent answer

All three reference engines refuse to copy full recurrent state on rejection:

- **GenAI**: per-token checkpoint rows + promote (3.1 above).
- **vLLM**: pages recurrent state like KV (`1+k` columns per request); rejection
  is a counter decrement (`scheduler.py:1870-1882`) and stale bytes are
  overwritten later; its one remaining full copy/step is an artifact of
  append-only worker block tables (`single_type_kv_cache_manager.py:1650`) — an
  engine that owns its tables can rename entries instead (zero-byte rollback).
- **NInfer (ReplaySSM)**: verify never writes state; it records per-token raw
  transition inputs (~1.7 MiB/token vs the 144 MiB image on 27B, ~86× smaller)
  and a `gdn_replay_fold` replays only the accepted prefix — m=0 is a strict
  no-op (`ninfer/docs/maintainer/replayssm-gdn.md`, `ops/linear_attention/
  gated_delta_net/recurrent.cuh:419-439`). The doctrine that comes with it is
  worth stealing regardless of mechanism: **the fold/replay must execute the same
  finite-precision transition as verify, and committed state must be validated
  bitwise against a no-spec baseline** — output plausibility cannot catch state
  drift.

For ligence this is not a menu: on the stateful graph all three are unreachable
(the state is inside `VariableState`), and on the paged path GenAI's scheme is
already built for this exact interface. Adopt 3.1's speculative mode; keep
ReplaySSM's validation doctrine as the gate.

### 3.6 Prefix caching — ligence is right, two refinements available

- vLLM's structural answer to "restore KV and recurrent state at the same
  boundary" is one shared block-hash namespace across layer groups with a
  fixed-point hit intersection (`kv_cache_coordinator.py:757-880`) — ligence's
  "both or neither" blob is the same invariant expressed more bluntly, and on the
  stateful graph it is the *only* honest expression. On the paged path it would
  relax into per-group pages + shared hashes (vLLM's form).
- vLLM keeps state snapshots only at **proven reuse points** (prompt end, shared
  prefix junctions — `reachable_block_mask`,
  `single_type_kv_cache_manager.py:1378-1433`); ligence's single
  snapshot-at-last-grid-edge is already that policy. No change needed; the
  refinement comes free with paging.
- NInfer's tier is the depth upgrade for later: immutable checkpoints at stable
  frontiers (turn closure / response opener), device-slot residency with priced
  host demotion, completeness rule (full state + all KV planes + exact identity,
  or no hit) (`ninfer/docs/maintainer/resource-scheduling-and-context-cache.md`).
  ligence's completeness rule already matches; the retention classes and the
  machine-cost model are the steal if the cache budget ever grows.
- One concrete addition, cheap: put the **allowlist architecture hash into the
  block-hash key** (vLLM's "extra keys" — LoRA/model revision ride there,
  `kv_cache_utils.py:582-617`). Today ligence's process-wide key is constant and
  single-model, so it is fine; the day a process reloads a different artifact or
  MTP export changes, a stale hit would be silent. Key it to the artifact hash
  now while the change is one line.

### 3.7 Concurrency — the cheap version exists, and llama.cpp states the one rule

The stateful graph already isolates state per `InferRequest` (that is the whole
point of OV's stateful API). One compiled model + **N infer requests** gives N
simultaneous sequences with weights loaded once; per-sequence cost is the GDN
state slab (~70 MiB coder / ~171 MiB dense) plus that sequence's KV. That is the
entire missing scheduler: replace the backend mutex with a per-slot InferRequest,
keep the SlotPool as admission, refuse admission on memory rather than on a lock.
Caveats: the MTP/embedding requests are currently shared too and would need
per-slot twins, and at deep context the per-slot KV is the binding budget
(2 × 5 GiB fp16 KV at 262k does not fit — cap slots by reservation, NInfer's
backfill-proof-over-preemption stance is the right model for a 1–2 lane engine).

llama.cpp's one structural rule for when batching arrives: **never mix sequences
inside one recurrent graph execution** — it asserts equal-sequence ubatches for
qwen3_next (`src/models/qwen3next.cpp:404`) and splits batches accordingly
(`llama-batch.cpp:510`, note `n_keep_tail` so the GDN snapshot tail lands in one
ubatch). Chunk ends are where state gets written, so chunks end at cacheable
boundaries (`_mamba_block_aligned_split`, vLLM `scheduler.py:385-470` — the same
shape as ligence's absolute grid, generalized to "boundaries worth checkpointing").

### 3.8 Audit methodology — steal NInfer's honesty package wholesale

ligence already does revision-pinned measurement better than anyone cited here
(the DESIGN tables *are* the method). What NInfer adds (`ninfer/docs/
performance.md`):

- **Stored-response audits at scale** (their 225-response campaign): sha256 every
  answer, repetition/termination scanner, throughput explicitly separated from
  correctness ("decode throughput is a transport measurement, not a correctness
  score").
- **Pathology disclosed, not hidden**: their best tok/s number is flagged † as a
  degenerate repetition loop and excluded. This is the fleet's anti-theatre rule
  written down as process.
- **Bitwise committed-state validation for speculative paths against a no-spec
  baseline over mixed-length accept chains** — the test that catches state drift
  outputs never reveal (§3.5).
- Saturation metrics computed only from complete intervals.

The 200-prompt nightly sweep the second opinion recommended is exactly item one;
pure scripting, do it.

## 4. What ligence should NOT do (confirmed by the reading)

- **No custom kernels via the CustomLayer path.** The fusion-barrier measurement
  in kernels/README.md is now independently corroborated: the plugin's own
  specialized permute exists (`permute_f_y_axes`) and is gated only by dynamic
  shape — the fix is shapes, not injection. Keep `--custom-kernels` as the
  measurement switch it is.
- **No context shift, ever** — llama.cpp does allow shift on hybrid models
  (`llama-memory-hybrid.cpp:133-136`: shift the attention KV, leave recurrent
  state alone), and even documents the semantic scar: attention sees a hole, GDN
  sees everything. ligence's §3.8 refusal is the defensible position; llama.cpp
  confirms the alternative is an approximation, not a feature.
- **No fp8 chase** (§3.4).
- **No second compiled model for static decode** (§3.3).
- **No plain-cast q8 KV** — the refusal is right, and the paged path is the
  honest route to the real thing.

## 5. Corrections of record

1. **secondopinion Kniff 2's "fp8 KV on Xe2"**: unavailable (plugin accepts
   i8/u8/i4/u4 only; Xe2 XMX has no fp8). Use u8.
2. **secondopinion Kniff 1's weight-sharing assumption**: two compiled models
   duplicate device weights; 12.8 GiB × 2 does not fit. The static-shape win must
   come from within one compiled model — the paged path provides it (§3.3).
3. **secondopinion Kniff 3 / DESIGN §8 "coldest ~15% of experts"**: llama.cpp's
   `-ncmoe` is first-N-layers by index, not coldest. The coldest-expert variant
   needs the fleet's imatrix statistics and is unproven under any engine; first-N
   is the measured-working pattern (and it is what currently serves the 35B on the
   fleet's A770).
4. **`#37607`**: the provider is recreated on blob import at OV HEAD — likely
   fixed upstream; retest before the workaround becomes permanent doctrine.
5. **README M4 row, DESIGN §3.5 allowlist text, qwen3.8 "provisional 7/10"**:
   three doc drifts listed in §1; the M4 row is the one actively misinforming.
6. **NInfer acceptance-rate comparison**: NInfer publishes 67–69% MTP acceptance
   on the Qwen3.6 pair and ~46–49% on Qwen3.8 (nvfp4, counting all positions);
   ligence's 93.3% is primed-over-prompt, generated-positions-only. Not the same
   metric — do not let the two meet in a slide without the footnote.

## 6. Priority recommendation

One sentence: **commit the dirty-tree fixes, then spend the project on the paged
path using GenAI's reconstructed `la.*` convention (§3.1) — it is simultaneously
the optimized GDN kernels, the real q8 KV, the depth-collapse fix, the MTP
rollback fix, and the prefix-cache tax cut — with `offload_ratio` for the 35B on
the A770 (§3.2) and the multi-InferRequest scheduler (§3.7) as the two cheap
parallel wins.**

Ordered:

1. Commit the worktree (MTP-in-cache-blob, snapshot stats, combination gate).
2. Fix the three doc drifts (§1.2–1.4, §5.5) — ten minutes, stops the bleeding.
3. Paged path, non-speculative fill first (recipe in §3.1), gated by the existing
   cold-vs-warm equivalence; then speculative mode for MTP.
4. Retest #37607 on a current OV build; keep prove-and-discard until green.
5. `offload_ratio` measurement for the 35B/A770 (fleet imatrix in hand first).
6. Per-slot InferRequests behind the existing SlotPool; admission bounded by a
   startup reservation curve (NInfer §12 discipline), never by a lock.
7. Nightly 200-prompt sha256 sweep + bitwise spec-state validation (§3.8).
8. Artifact-hash-keyed prefix cache key (one line, §3.6).

The engine's founding bet — OpenVINO owns the math, ligence owns the state —
survives this review intact. Every open problem that remains is one the paged
interface was built to answer, and for the first time the interface has a written
oracle. The measurement culture here is already better than any of the four
reference projects'; the next milestone is mostly reading what upstream wrote.

---

*Generated 2026-08-28 by pica. Repo clones: /tmp/ref/{ninfer,ov,openvino-genai,
vllm,lcpp} on pica; per-repo technique extractions in /tmp/ref/reports/. Hardware
facts: Intel ARK SKU 229151 (A770), Intel Arc Pro B60 datasheet (2026-03), Xe2
architecture analyses. Roofline figures derived from those plus DESIGN's measured
weights sizes.*
