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
