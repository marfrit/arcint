# design-lyon-stateful-prefill — the compile-once half is already landed; the lever is the multi-block stateful core

Recon and design for 0.5.4 LYON. Acceptance commit: `docs/window-054.md`
(LYON-001, all rows EMPTY). This note records what the tree **actually does**
today, separates the solved half from the open one, and fixes the target the
`LYON-001` row-3 cells measure. Evidence classes are `code` (read out of this
tree) / `measured-here` (a device-free number this note produced) / `paper`.

## 1. Recon — where the 2.2M nodes come from, and what is already stateful

**The 2.2M-node static cost is the `perchunk` emission** (`code`:
`tools/q4e/gdn.py`:168–219; `docs/window-050.md` §4.2). `perchunk` hoists the
chunk axis out of EVERYTHING — no emitted op sees a live chunk axis — so the
~1,900-op forward-substitution unroll is emitted **once per chunk**, linear in
C:

| T | C | batched | perchunk | ×36 GDN blocks (48-layer) |
|---|---|---|---|---|
| 64 | 1 | 2,054 | 2,037 | — |
| 256 | 4 | 2,224 | 7,750 | 279,000 |
| 2048 | 32 | 3,680 | 61,062 | **2,198,232** |

**The served serving-shape graph is ALREADY the stateful form** (`code`):
`tools/q4e/serving_shape.py`:1401 sets `T = -1` ("dynamic: every reshape uses
−1"), and the backbone (`:1547`) calls `qgdn.emit_gdn` with
`conv_emitter=stateful_short_conv` (`:1745`) and
`core_emitter=stateful_gdn_core` (`:1870`), plus `emit_stateful_attention`
(`:1967`) for the attention layers. `emit_gdn` (`gdn.py`:566) **asserts** the
hooks are present when `seq_len is None`. The state lives in `ov::Variable`s
(`ReadValue`/`Assign`), the GDN recurrence is a token-sequential `v5::Loop`
whose trip count comes from `ShapeOf` (dynamic in T), and the attention KV state
is a `ReadValue → Concat → SDPA` chain. So the served artifact's node count is
**T-independent by construction**: one compile, replayed across every prompt
length.

**Measured device-free** (new red-first cell,
`tests/python/test_gdn_block.py::test_the_served_prefill_graph_does_not_grow_with_the_prompt_length`):

```
[lyon] served(dynamic T) nodes=164
       static perchunk nodes={64: 2037, 256: 7750, 512: 15366}
       per-chunk growth=13329 over 448 tokens
```

A single served GDN block is **164 nodes** against **2,037** for one static
`perchunk` chunk, and the static form adds ~1,900 nodes per 64-token chunk. The
cell is red-first: replacing the served path's `core_emitter` with the unrolled
core fails it (`dynamic T needs the stateful conv and core hooks`).

**The served ARTIFACT carries it** (`measured-here`, device-free `read_model`
on the two packed artifacts of 2026-09-25/26):

| artifact | nodes | `Loop` | `ReadValue`/`Assign` | Variables | T-dynamic inputs |
|---|---|---|---|---|---|
| `qwen36-35b-a3b-d4packed-ov` | **1,884** | 3 | 8 / 8 | 8 | `[?]`, `[1,?]` |
| `qwen36-35b-a3b-d40packed-ov` | **20,658** | 30 | 80 / 80 | 80 | `[?]`, `[1,?]` |

The **full-depth 40-layer served graph is 20,658 nodes**, T-independent — against
the 2.2M the static `perchunk` form would need at T=2048 over the same stack.
That is the "configuration where a long prompt stops paying the compile": it is
the served serving-shape artifact as exported today, not a new flag.

**Verdict on the premise, stated plainly.** LYON's *"stateful prefill retires
the 2.2M-node static cost"* is **already landed in the served path**. The
2.2M-node figure is a property of the **static test/unit emitter** (`perchunk`,
the flip's incumbent), **not** of what the served binary compiles. No new work
is owed to retire it in serving; it is retired. What this note does **not**
claim is that the served path is fast — see §2.

## 2. The open half — the stateful core is token-sequential

`stateful_gdn_core` (`code`: `serving_shape.py`:1870–1965) is a `v5::Loop`
whose body advances **one token** per iteration (`set_sliced_input(param,
src.output(0), 0, 1, 1, -1, 2)` — axis 2, one element). The roadmap's
**"multi-block"** is exactly what is missing: a 32k prefill pays 32k loop
iterations before any attention output, and the `LYON-001` row-3c rate bar
(≥ 460 t/s, DESIGN §7.0.2's arithmetic) cannot be met one token at a time.

Two structural facts fix the shape of the work:

1. **`FuseGDNLoop` matches the token-sequential Loop** (`code`:
   `stateful_gdn_core`'s own docstring — "`FuseGDNLoop` rewrites the Loop into
   one `ov::op::internal::GatedDeltaNet` node; `PagedGatedDeltaNetFusion` then
   matches THAT node over a ReadValue"). A **chunked** Loop body would not be
   matched by that pass unless a chunked variant is added, or the chunked core
   stays in-graph unfused.
2. **The chunked core exists and is fast, but is T-dependent** (`code`:
   `emit_gdn`'s default `core_emitter`; `gdn.py`:325–560, the `perchunk`
   unroll). It is the algebra `stateful_gdn_core` reproduces token-by-token —
   the existing parity cell
   (`test_the_sequential_serving_core_is_the_same_gdn_as_the_chunked_one`)
   holds the two to the f32 floor, so the chunked form is the **correct
   target**.

So the multi-block increment is: a **chunked stateful core** — a Loop over
**blocks** of `CHUNK` tokens whose merged state is the same rank-4
`cache_params.past.ssm.<layer>` Variable, keeping the T-dynamic trip count.
The chunk algebra (cumsum, the pairwise decay, both contractions, the unroll,
both inverse matmuls) runs **inside** the body at a fixed `CHUNK`, so the graph
stays T-independent (§1's property holds), and the state merge is the existing
`Assign`.

## 3. Where the nodes go, and which become dynamic or reused

The static `perchunk` growth is exactly the per-chunk unroll (§1 table). For
the **served** path the census is short: 164 nodes per GDN block, of which the
token-sequential core's body is a handful of ops, and the rest is the shared
backbone (projections, conv, norms, gated RMSNorm, out_proj). Nothing in the
served graph scales with T; the **only** T-scaling object is the runtime KV
(GDN state `[1, HV, Dk, Dv]` and the attention KV `ReadValue → Concat`), which
is charged by the fit, not by the graph.

**Reuse candidates named, not decided** (`code`):
- the **static `perchunk` default** (`Q4E_GDN_UT_MODE`) is a **unit-test**
  default; the served export never takes it. Its 2.2M-node cost is a **test
  cost**, so the roadmap sentence is best read as retiring a test/emitter cost
  — if the reviewer meant the served path, it is already retired;
- the **unrolled core** itself: the chunked-stateful body can **reuse** the
  existing `perchunk` body ops (they are the same algebra) with the chunk axis
  bound to `CHUNK`, so the increment is a **body**, not a new algebra;
- **compile-once/replay**: already the mechanism — the plugin compiles the
  T-dynamic graph once and the kernel cache absorbs the per-shape JIT
  (`code`: window-051 row 12's "generation rule (the compute runtime's kernel
  cache) is UNTESTED as its cause" is the one open question there).

## 4. The LYON-001 row-3 cells, restated against this recon

- **3a compile-once** — now **device-free and green today**: the served GDN
  block is 164 nodes, T-independent (the new cell). Pinned generation: the node
  count is generated from the emitted graph.
- **3b compile time** — `compile_s ≤ nodes × 2.38 ms` (`code`: the 86k-node
  law, `window-050` §6). The served graph's node count is what the artifact
  reports; the row stays EMPTY until a card compile.
- **3c served rate** — prefill ≥ 460 t/s at 32k on the A770. **This is the
  increment's number** (§2); it cannot be met token-sequentially and cannot be
  met statically (the 2.2M-node compile).

## 4b. What landed, and the card finding (2026-09-26, later)

**Landed device-free** (`serving_shape.py`, `tests/python/test_gdn_block.py`):

- `stateful_gdn_core_chunked` — a Loop body advancing **CHUNK tokens per
  iteration**, reusing the chunked algebra (`gdn.py`'s `perchunk` core),
  T-dynamic (`ceil(T/CHUNK)` off `ShapeOf`), zero-padding the inputs to a whole
  chunk and slicing the buffer back to T, so a length that is not a multiple of
  CHUNK is exact.
- **Parity, byte-exact**: `|chunked-stateful − chunked| = 0.0000e+00` at
  T = 128, 192, 224, 256 (`measured-here`, CPU). Against the token-sequential
  core it is f32-bounded and **NOT** byte-exact (3.1e-07 … 5.7e-07 — different
  summation order). **That is the finding the reviewer's clause anticipated:
  byte-exactness against the token-sequential core is impossible**; the
  byte-exact oracle is the chunked algebra, and it holds.
- **Growth cell extended**: served sequential 164 nodes, served chunked
  **206 nodes**, both T-invariant; the `perchunk` control still grows 13,329
  nodes over 448 tokens. Neither served core is token-count shaped.
- Config `Q4E_GDN_CORE=sequential|chunked` + `Q4E_GDN_CHUNK`, a typo refused.

**THE MATCHER QUESTION — DECISION MEASURED, AND IT BITES.** Keeping the chunked
body **in-graph** was chosen (stated in the emitter's docstring). The A770
load check (`qwen36-35b-a3b-d4chunked-ov`, `Q4E_GDN_CORE=chunked`, GPU.1,
`measured-here`) **refuses the artifact**:

```
Check 'unregistered_parameters.str().empty()' failed at src/core/src/model.cpp:264:
Model references undeclared parameters: opset1::Parameter beam_idx () -> (i32[?])
```

The chunked Loop is **not** rewritten into `GatedDeltaNet` (the fusion reads a
seq-1 body), so its `ReadValue → Gather(beam_idx)` chain **survives** into
`SDPAToPagedAttention`'s rewrite (`backend_ov.cpp`:2637), which drops the
`beam_idx` declaration. The token-sequential path never hits this because the
fusion **consumes** that chain. So the cost of the in-graph decision is **not
just the lost fused kernel — it is a load failure** in the current pipeline.
The two ways out, both NOT done tonight: (a) extend the fusion to a chunked
Loop, or (b) make the chunked path beam-free (the served path is one lane, so a
constant beam instead of the `beam_idx` parameter). Option (b) is small and is
the cheaper next step.

## 4c. The beam-free fix, and where the card stops (2026-09-26)

**Landed**: the chunked core is **beam-free** — it gathers the recurrent state
with a constant row 0 (`op.constant([0])`) instead of the `beam_idx` PARAMETER,
because the served path is one lane and an unfused chunked Loop must not carry a
`ReadValue → Gather(beam_idx)` chain into `SDPAToPagedAttention`'s rewrite
(`backend_ov.cpp`:2637`, which drops the declaration). Red-first cell
`test_the_chunked_served_core_leaves_no_dangling_beam_idx`
(`measured-here`): the chunked core's `beam_idx` consumers are `[]`, the
sequential control's are `['Gather']` — mutation-verified (restoring the
parameter reference fails the cell). `tests/python/test_gdn_block.py` **14
passed**.

**The parameter error is GONE** on the A770 (`qwen36-35b-a3b-d4chunked-ov`,
re-exported after the fix, `measured-here`): no `undeclared parameters`. But the
full artifact now fails **later**, at the GPU program build:

```
Check 'false' failed at program_builder.cpp:168: [GPU] ProgramBuilder build failed!
Exception from .../ocl_memory.cpp:606:
[GPU] clWaitForEvents, error code: -14 CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST
```

The **isolated** chunked GDN block compiles on GPU.1 (`measured-here`: a
128-token `emit_gdn` with the chunked core → `GPU.1 compile OK`; the sequential
control too), so the Loop itself is fine on the plugin. The failure is in the
**full** graph, cause **not localized**. So the in-graph chunked route is
**not viable as emitted** tonight; the alternative the reviewer named — a
**chunked fusion matcher** — is the larger next change, and per the brief this
is the **finding**: the beam-free fix landed, the load check advanced past the
parameter error, and the full-graph build stops with
`CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST`. LYON-001 rows stay EMPTY.

## 4d. The bisect (2026-09-26, bounded leg) — the interaction is NOT the Loop itself

Four small cases, each one GPU.1 compile, sampler + watchdog on the card legs;
`measured-here`:

| case | graph | GPU.1 |
|---|---|---|
| (a) | chunked GDN only, no MoE (pin geometry, T=64) | **compile OK** |
| (a-control) | sequential GDN only, no MoE | compile OK |
| (b) | MoE + sequential (`qwen36-35b-a3b-d4packed-ov`) | compile + serve OK (the standing control) |
| (c) | chunked + MoE (`qwen36-35b-a3b-d4chunked-ov`, depth 4) | **FAIL** |

So **(a) is clean and (c) fails**: the failure is an **interaction in the full
graph**, not the chunked Loop itself. The failing site (raw line):

```
Check 'false' failed at src/plugins/intel_gpu/src/plugin/program_builder.cpp:168:
[GPU] ProgramBuilder build failed!
Exception from src/plugins/intel_gpu/src/runtime/ocl/ocl_memory.cpp:606:
[GPU] clWaitForEvents, error code: -14 CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST
```

The plugin logs **no last-primitive** line before the throw, so the failing op
is not named by the log.

**Not pinned to MoE-vs-attention.** A depth-1 artifact (GDN + MoE, no
attention) cannot discriminate: arcint refuses it **earlier and for a different
reason** — `Check 'ov::op::util::has_op_with_type<ov::scaled_dot_product_attention>(model)'
failed at sdpa_to_paged_attention.cpp:81` (a depth-1 `qwen3_5_moe` rung has 0
attention layers, so the paged-attention pass has nothing to rewrite). A
device-free Python bisect that would have built chunked+MoE at a small geometry
segfaults **inside the MoE emitter** (`emit_moe_tiled`, both a native and a u4
filler), so it yields no GPU answer. The interaction is therefore localised to
**chunked-Loop × the full graph's build**, not further.

**No fix in this leg**, per the brief. The next leg starts with this table; the
likely fix remains the **chunked fusion matcher** (or a plugin-side Loop build
fix), a daylight change.

## 4e. THE A770 FULL-MODEL FIT — VERDICT WITH ARITHMETIC (2026-09-26, device-free)

All inputs are my own measurements (`measured-here`) or the card property; the
arithmetic is `code`.

| term | bytes | GiB | source |
|---|---|---|---|
| d40packed lm `.bin` | 15,623,664,495 | **14.551** | export log |
| d40packed expert fill | 12,918,456,320 | **12.031** | export census |
| **dense + lm_head + norms** = lm `.bin` − experts | — | **2.519** | derived |
| d40f16 (re-laid) lm `.bin` | 19,482,424,091 | 18.144 | export log |
| embeddings `.bin` | 2,034,237,448 | 1.895 | export log |
| A770 VRAM (`GPU.1`, `8086:56a0`) | — | **15.111** | `GPU_DEVICE_TOTAL_MEM_SIZE` |
| drafters (MTP) | — | 0.95 | served load line |
| activations, chunk 1024 / 512 / 256 | — | 1.700 / 0.850 / 0.425 | the reviewer's 1.7 at ≤1024, scaled with the chunk |
| margin | — | 0.25 | the fit's own |

### The flags-only fit does NOT close

```
chunk 1024, MTP on : 12.031 + 2.519 + 1.700 + 0.95 + 0.25 = 17.451  OVER by 2.340
chunk 1024, MTP off: 12.031 + 2.519 + 1.700 + 0.00 + 0.25 = 16.501  OVER by 1.390
chunk  512, MTP off: 12.031 + 2.519 + 0.850 + 0.00 + 0.25 = 15.651  OVER by 0.540
chunk  256, MTP off: 12.031 + 2.519 + 0.425 + 0.00 + 0.25 = 15.226  OVER by 0.115
```

**Every flag-only combination is over.** The best (`chunk 256` + `--mtp off`)
misses by **0.115 GiB**, and chunk 256 also spends the prefill rate this
milestone exists for. So the packed + dense-f16 + MTP-off route does **not**
reach 15.111 GiB.

### The levers and their kind

- **packed experts (`--native-packed`)** — *configuration, exists*: 12.031 GiB
  (the downs, ~5.6 GiB of the fill, stay on their IQ3_XXS/IQ4_XS route).
- **dense-f16 (`--dense-fp16`)** — *configuration, exists*: the 2.519 GiB term
  is already f16 in the packed artifact.
- **chunk (`--prefill-chunk N`)** — *configuration, exists*: the activation
  term, but it costs prefill rate linearly.
- **MTP (`--mtp off`)** — *configuration, exists*: 0.95 GiB.
- **margin** — the fit's own 0.25; not a runtime switch.
- **dense to u8/i4** — ***code*** (does not exist today). Halving the 2.519 GiB
  term gives **14.391 GiB** (chunk 512) → **FITS by 0.720 GiB**. A quarter
  (i4) gives 13.966 → fits by 1.145. Neither form exists in the tree: the
  served int4 artifact is a **different family** (qwen3.5-2b / qwen36-coder
  b5), and its quantisation is the **u4 group-affine repack** the native work
  deliberately left (`_compressed_expert`), not a dense-weights u8 form. It is
  emitter work (a dense u8/i4 Constant form + the plugin's matcher/precision
  path), not a flag.

### VERDICT

**No.** With today's code, **no configuration fits the full-depth (40-layer)
model on the A770's 15.111 GiB**, and the binding constraint is **not** the
card:

- **VRAM ceiling ≈ 38.2 layers** (`(15.111 − 2.519 − 0.850 − 0.25) / 0.3008`),
  where 0.3008 GiB/layer is the packed expert term (`12.031/40`).
- **Host-compile ceiling ≈ 37.4 layers**: the compile materialises
  **2.903×** the artifact's bytes (`measured-here`: 44,292,600 kB RSS for a
  14.551 GiB artifact), and the usable host is ~40 GiB (44 GiB container minus
  the 4 GiB watchdog floor) → artifact ≤ **13.78 GiB** → experts ≤ 11.26 GiB →
  **37.4 layers**.

So the **host compile is the tighter wall**, and a dense-u8 form (the only
lever that would clear the VRAM side) does **not** clear it: 12.031 + 1.260 +
0.850 + 0.25 = 14.391 GiB → 41.8 GB of staging → still over ~40 GiB.

**The A770 full-depth answer today is a ceiling, not a fit: ~36–37 of 40
layers.** Two levers would be needed together — a dense-u8 form (strategy:
emitter + plugin) **and** the compile-materialisation factor (the ~3× that
§14.3 named as its own defect).

**Caveat, stated not assumed**: whether the 1.895 GiB embeddings model is
VRAM-resident is **not established** here (the reviewer's arithmetic omits it;
the served loads I have do not print it). If it is resident, every ceiling
above drops by ~6 layers and the verdict gets **worse**, not better.

### Item 5 — the packed route's `CL_OUT_OF_RESOURCES`, bounded

The premise needs one correction: the **d40 packed** artifact did **not** reach
`CL_OUT_OF_RESOURCES` — it died at **host RAM** during the compile (watchdog
SIGKILL, §13.2). Only the **d4 packed** (rank-5 chain) reached the program
build and failed there:

```
program_builder.cpp:168 -> ocl_memory.cpp:606
clWaitForEvents, error code: -14 CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST
```

**Bound:** the **packed d4 artifact is SMALLER than the re-laid d4 artifact
that compiles** (lm `.bin` 3.63 vs 4.28 GiB; expert fill 1.24 vs 2.42 GiB), and
both routes carry the **same** tiled-MoE graph shape, the same attention/GDN
structure and the same activations. What differs is (a) the **Constants** (one
u8 `[E,out,K/256,80]` block + a f16 `d` `[E,out,K/256,1]` against the re-laid
form's three tensors) and (b) the **OCL kernel**: patch 0052 adds
`native_dot_iq2s_packed`, which the re-laid route never builds. A kernel whose
**build** fails is surfaced by the driver as `CL_OUT_OF_RESOURCES`-class errors.
So the bound is: **not raw size, not the graph; the prime suspect is the packed
OCL kernel's build (or the constant-reorder it needs), i.e. a kernel-build
allocation, not a geometry interaction.** Localizing it needs a plugin-side
print of the failing kernel id in `kernels_cache::build_all` — a daylight
instrument, not attempted here. Not fixed, per the brief.

## 4f. The operator's chunked-matcher leg — BLOCKED, and the viability question ANSWERED (2026-09-26)

### Localization first: the chunked+packed failure is the PACKED route, not the GDN

Instrumenting `gpu_usm::fill` (`ocl_memory.cpp`:595, plugin rebuild + one short
A770 run) shows the throw is a **fill whose kernel failed**, and the last fills
carry **rank-5 f16 weight layouts**:

```
ARCINT_FILL bytes=536870912 pattern=00 blocking=1 layout=f16:bfzyx:256x2048x16x4x8
ARCINT_FILL bytes=536870912 pattern=00 blocking=1 layout=f16:bfzyx:256x512x64x4x8
```

512 MiB **f16** buffers whose dims are the expert geometry — the plugin's
**reorder of the packed u8 weight into a large f16 buffer** (the packed gate
weight is 84 MB u8; the reordered form is 512 MiB, ~6×). The same site, same
class, as the **packed sequential** d4 that also failed. So the chunked+packed
failure is a **packed-route memory/reorder issue**, not the chunked Loop.

**Proof by isolation (`measured-here`):** the chunked core **with the re-laid
experts** (`--expert-format native`, no `--native-packed`) **compiles and
serves** on GPU.1 — `ready True`, t_boot 48.0 s, `device-resident 2.74 GiB`,
prefill 118.5 t/s at 64 tokens. So the chunked body is **viable on the GPU**
as emitted; nothing about the Loop blocks it.

### The matcher's feasibility — CHECKED, and it is infeasible as stated

`matches_linear_attention_loop` (`code`: `fuse_gated_delta_net.cpp`:62) pins the
body to the **token rule**:

- `query/key/value` must be `pattern::shape_matches("[?, head_num, 1, *_head_size]")`
  — **sequence extent exactly 1**;
- `Squeeze(key, {2})` / `Squeeze(value, {2})` require dim 2 = 1;
- the state update is `ReduceSum(gated_state * key_unsqueeze, -2)` — an
  **outer-product token step**;
- the output is `ScatterUpdate(out_buffer, Unsqueeze(step_index, 0), out, 2)`
  — **one row per iteration**.

A chunked body carries **seq = CHUNK**, `cumsum`, the pairwise decay, the
forward-substitution inverse and **matmuls** — it satisfies **none** of those,
and the fused `ov::op::internal::GatedDeltaNet` primitive **is** that token
rule. So "extend the matcher to a chunked Loop" is not an extension: it needs a
**new chunked primitive and its kernel**.

**VERDICT: BLOCKED.** The chunked fusion matcher is infeasible under the
existing primitive's contract.

### And the rate says the in-graph chunked path LOSES to the fused sequential

Same card, same depth-4 rung, same 32k prompt, `ratio 0` + dispatch, chunk
2048, `measured-here`:

| rung | prefill tokens | prefill t/s | decode t/s | digest |
|---|---|---|---|---|
| **chunked + re-laid** | 23,680 | **153.5** | 18.0 | `4ecb1ca8a6b70749` |
| **sequential + re-laid** (control, `d4n`) | 23,680 | **161.8** | **26.7** | `b283fe50f4ef0280` |

The **unfused chunked core is SLOWER** than the fused token-sequential one
(prefill −5 %, decode −33 %): the fusion is what buys the sequential path its
kernel, and the chunked body forfeits it. So LYON's speed target needs a
**chunked fused primitive**, not a matcher extension — a daylight
implementation, not this leg.

**LYON-001 rows 1–3 stay EMPTY.** The d4 rung's answer is degenerate (a 4-layer
artifact is not the model), and no full-depth chunked artifact was built. No
row is filled by inference.

## 5. Pipeline for the increment

Recon (done, §1–3) → **this note** → red-first: the cell in §1 (landed) plus a
cell for the **chunked stateful core's** parity and T-independence (to be
written with the body) → the chunked body implementation → unit parity vs the
token-sequential core and the chunked unroll → **one** card window: `LYON-001`
rows 1–3, no sweep. Packaging bump for the fusion matcher if the chunked Loop
is to fuse; otherwise the chunked core stays in-graph and the row-3b compile
carries the extra nodes.

## 6. Invariants

DESIGN §3.4: the answer is a pure function of (tokens, state), never of chunk
boundaries — the existing parity cell is the guard. The KV-tier digest identity
(window-050 §7 bonus cell) holds. `LYON-001`'s row-2 bars are not traded for
row-3 rate.

## 7. Status

- 2026-09-26 — recon and this note landed. The premise is **half-solved** and
  said so: the compile-once/stateful half is **already in the served path**
  (§1), so the 2.2M-node static cost does not apply there; the open half is the
  **multi-block** stateful core for rate (§2). Red-first cell landed and
  mutation-verified. No card touched.
