# window-054 — 0.5.4 LYON acceptance (LYON-001: every measured row EMPTY)

Recorded 2026-09-26, before any LYON card window exists and before the
compile-once/replay path is built. This file is the acceptance commit of 0.5.4
in the form the roadmap's law demands (`ROADMAP-0.5.x.local.md`:2–6): *the first
commit is the acceptance criteria, with the measured rows EMPTY; prediction
commits precede measurement; gates red-first; counts generated; comparisons to
FreeToken pinned ONLY by our own measured runs.* The commit that fills a row is
a measurement commit and pastes the raw output. The markers are
`docs/window-050.md`'s (`RUN@<sha>`, `RUN@wt+<sha>`, `RUN@unrecorded`, `DRY`,
`UNTESTED`); a row with no marker is EMPTY. Every disposition of fact carries an
evidence class (`paper` / `code` / `measured-here`); the acceptance-cell bullets
are configuration, not dispositions. **Every threshold pinned here is arithmetic
(`code`) or borrowed (`paper`), never `measured-here`.**

## Feature (roadmap 0.5.4)

**Serving-length prefill** — multi-block, context ≥ 32k, spanning the **2051**
boundary — with **stateful prefill retiring the 2.2M-node static cost** and
**compile-once / replay amortization**. Method reference: **llama.cpp #28725**
(`paper`; surveyed, not portable code).

## The defect LYON owns, as measured

The multi-chunk **static** prefill (`perchunk` emission, the 0.5.0 default)
unrolls ~1,900 ops **per chunk**, linear in the chunk count C
(`code`: `docs/window-050.md` §4.2; the growth law and the C1→C2 `concat`
step are asserted in `tests/python/test_gdn_block.py`):

| T | C | batched | perchunk | ×36 GDN blocks (48-layer stack) |
|---|---|---|---|---|
| 256 | 4 | 2,224 | 7,750 | 80,064 → **279,000** |
| 2048 | 32 | 3,680 | 61,062 | 132,480 → **2,198,232** |

A serving 32k prompt at the served chunk 2048 is ~16 chunks, so the **static**
graph scales toward ~1.1M nodes per 16-chunk rung and beyond; the recorded
compile physics is **superlinear** (`code`: `docs/window-050.md` §6, RUN@be57428):

```
nodes    compile s   ms per node
 9,727        5.41       0.56
28,831       15.88       0.55
86,143      204.90       2.38
```

Flat at ~0.55 ms/node to 29k nodes, then **4.3× worse per node** at 86k. The
**2.2M-node static cost is therefore not a route** at serving lengths — it was
recorded as "still not the route at serving prefill lengths; that remains the
chunked stateful-prefill increment" (`code`: same section). That increment is
the feature here, and it does **not** exist in the tree.

**The premise is live, not solved and not refuted.** The emitted stateful GDN
construct is broken on both cards as emitted (`measured-here`: `docs/window-050.md`
§P5, `RUN@wt+8a84598`) — `stateful_short_conv` compiles its `GroupConvolution`
over a **static** `[1, conv_dim, K+blk]`, but the **stateful** form
(`ReadValue [?,1024,4] → Concat → GroupConvolution → Slice`) is refused by the
GPU plugin ("`GroupConvolution_30 … Weights/ifm mismatch`") and by the CPU
plugin after compile (the `Assign` reorder on a dynamic shape). So LYON is a
**construct problem first**, and the record does not contain a working stateful
prefill.

## What LYON-001 owns

Three acceptance rows, all EMPTY:

1. **a 32k prompt answered on-card** — multi-block prefill across the 2051
   boundary, greedy answer pasted raw, digest recorded;
2. **the KLD gate in both regimes**, at a T **below** and a T **above** row
   2051, on the card where the gate **reads**;
3. **the prefill t/s row against the 86k-node compile physics** — one compile
   per process (a T-independent graph), and the served prefill rate at 32k.

The rows are the gate of `docs/campaigns/static-partition-prefill.md`'s sibling
for long prefill and of the design note `docs/design-lyon-stateful-prefill.md`.
This document does not fill them.

## Card — pinned to the A770, with the B60 named as a rate option

**Pinned: A770** (`GPU.1`, PCI `8086:56a0`, 15.11 GiB). The KLD row decides it:
the served path's determinism floor is a **per-card defect** — on the B60 the
clause is UNREADABLE (`F_served` 0.1361/0.1512 nats, ~44× above the bound,
`measured-here` 2026-09-20), while on the A770 the served depth-48 artifact was
**bit-identical** across runs (0/1367 rows moved) and clause (d) **READS**
(`code`+`measured-here`: `docs/window-051.md` A.2; DESIGN §7.0.2cb;
`docs/campaigns/served-prefill-determinism.md`). A gate that cannot read is not
a gate.

**Flagged, not assumed: a B60 rate confirmation is an OPERATOR DECISION.** The
B60 carries 22.71 GiB (vs 15.11) and the long-prefill activation fit is the
binding term, so the **rate** row would read higher there — but the KLD row
cannot read on it. The operator decides whether row 3 is repeated on the B60 as
a scale point, or the gate closes on the A770 alone.

## The acceptance cells (fixed before measuring)

### Row 1 — the 32k prompt, answered

- **Configuration (`code`)**: `--paged-kv u8`, one lane, `--n-ctx ≥ 32768`,
  `--prefill-chunk 2048` unless the fit refuses it, greedy (`temperature 0`),
  `IGEN` tokens pasted raw.
- **PASS**: non-empty greedy text; the last-position logits finite
  (`absmax` printed); rope positions reach ≥ 32,000; the answer's **digest**
  recorded. The answer's *content* is report-only (this is not a quality
  campaign) — the digest is the determinism handle for row 3.
- **Why 32k and not more**: 32k is the roadmap's pinned length (`code`:
  `ROADMAP-0.5.x.local.md`:167) and is the smallest length that both spans 2051
  and forces ≥ 16 chunks at the served 2048 chunk. It is the length where the
  static form's unroll stops being viable.

### Row 2 — the KLD gate, both regimes

- **Reference (`code`)**: llama.cpp master `56b9eb28`, capture `af7993b7…`,
  n_ctx 2735, 684 rows below / 683 at-or-above row 2051 per window, min clamped
  at max−16, uint16 reconstruction, bit-reproducible (`docs/window-051.md`:48).
- **Thresholds (`code`, `docs/window-051.md` A.2, unchanged)**:
  - below 2051: **3.0905e-03 nats** mean per-token KL(P_ref‖P_served);
  - at/above 2051: **2.6946e-02 nats** = the same bar **+ the measured QSA
    price** 2.385560e-02, which is **exact** for T ≤ 2051 and applies over
    **29/2080** rows at T = 2080 (`code`: `docs/window-050.md` §8,
    `RUN@692c0a6`).
- **PASS**: both regime means at or below their bars **on the A770**, with the
  argmax agreement printed beside them. A mean that passes below and fails
  above localises to the QSA→dense seam, which is the one place the price is
  known to change (`docs/window-050.md` §7).
- **Note**: bars are PROVISIONAL on the f16-rows caveat of window-051 A.2 and
  are inherited unchanged here; this document does not re-decide them.

### Row 3 — prefill t/s against the compile physics

The feature's own criterion is **structural**: the prefill graph is
**T-independent** — one compile per process, replayed across prompt lengths.
Row 3 carries both halves:

- **3a, compile-once (`code`, generated)**: the emitted prefill graph's node
  count is **invariant** in T across at least three lengths (e.g. 2048, 8192,
  32768), and the graph compiles **once** per process. Red-first: a cell that
  fails if the node count grows with T. Threshold: **node count(T) == node
  count(T=2048)** for every sampled T. (The static `perchunk` form grows
  ~1,904 nodes per chunk, `code`: window-050 §4.2, so this cell fails on the
  incumbent.)
- **3b, compile time (`code`)**: `compile_s ≤ nodes × 2.38 ms`, the 86k-node
  law's own per-node rate (`code`: window-050 §6). A generated node count makes
  this a number that can fail; it is deliberately the **worst recorded** per-node
  rate, not the flat 0.55, so the cell does not pass by assuming the good regime.
- **3c, served rate**: prefill **≥ 460 t/s** at 32k on the A770. Threshold
  source: DESIGN §7.0.2's operator prefill bar at depth (**460 t/s**, `code` —
  the bar's own arithmetic, restated there, not re-measured here). The static
  form cannot reach it (the 2.2M-node compile is ~minutes of superlinear build
  before any token), which is exactly the defect.
- **Report-only beside it**: the one-time compile seconds, the per-chunk wall
  time, and the amortized compile share over a 32k prompt
  (`compile_s / (compile_s + prefill_s)`).

## Scope — in / out

**In**: a **T-independent, stateful** prefill graph — the chunked stateful
increment (`code`: "the chunked stateful-prefill increment", window-050 §4.2) —
built from the existing state variables (`ReadValue`/`Assign`) with a dynamic
chunk axis; the fix for the stateful GDN construct's compile refusal (§P5,
window-050) so it compiles on the gate card; the compile-once/replay
configuration; the red-first node-count cell; the packaging bump this ships
under.

**Out**: the static-partition prefill defect (`docs/campaigns/static-partition-prefill.md`
— closed 2026-09-16, a different mechanism); the u8:i4 KV price
(`docs/campaigns/u8i4-prefill-price.md` — closed by patch 0020); the deep-prefill
fault (`docs/campaigns/u8i4-deep-prefill-fault.md` — VRAM-side); the 2051 bars
(decided, not re-decided); the 0.5.5 ROMA speculation (a later dot).

## Invariants

DESIGN §3.4 (history-independent greedy output) holds across chunk boundaries:
a stateful prefill's answer is a pure function of (tokens, state), never of how
the chunking fell. The KV-tier digest identity of the existing gate holds
(bonus cell, window-050 §7). Row 2's bars are not traded for row 3's rate
(`CLAUDE.md`'s measurement discipline: quality gates hold together with speed).

## Pipeline for this campaign

Recon (done — this document's tables are `code`-class from window-050 §4.2/§6
and §P5) → design note `docs/design-lyon-stateful-prefill.md` (the
compile-once/replay shape, the 2.2M-node census, which nodes become dynamic or
reused, and the stateful GDN construct's fix) → red-first: the node-count
invariant cell (fails on the incumbent `perchunk`) and a construct cell (fails
on the unpatched stateful emission) → the implementation → **one** card window
at the end, not many. **No measurement before the feature exists.**

## Status

- 2026-09-26 — LYON-001 written as the 0.5.4 acceptance commit; all three rows
  EMPTY. Recon complete; the design note and the red-first cells are the next
  commit. No card touched.
- 2026-09-26, later — the multi-block stateful core landed device-free
  (`stateful_gdn_core_chunked`), byte-exact against the chunked algebra at
  T=128/192/224/256, and the growth cell now covers both served cores
  (164 sequential / 206 chunked nodes, T-invariant). The **card gate is OWED**:
  a depth-4 chunked artifact was compiled on the A770 and the load **REFUSED**
  it — `Model references undeclared parameters: beam_idx` (`measured-here`;
  the chunked Loop does not fuse, so its `ReadValue → Gather(beam_idx)` chain
  survives the pass that drops the declaration).
- 2026-09-26, latest — the **beam-free fix landed**: the chunked core gathers
  the state with a constant row 0, so no `beam_idx` reference survives the
  paged-attention rewrite. Red-first cell
  `test_the_chunked_served_core_leaves_no_dangling_beam_idx` (mutation-verified;
  chunked consumers `[]`, sequential `['Gather']`). The A770 load check then
  **advanced past the parameter error** and fails later, at the GPU program
  build: `CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST` (`measured-here`). The
  isolated chunked GDN block compiles on GPU.1, so the Loop is fine; the full
  graph's failure is **not localized** and is the finding. The alternative — a
  **chunked fusion matcher** — is the larger next change.
- 2026-09-26, bisect leg — the interaction is **not the Loop itself**: the
  chunked GDN alone (no MoE) compiles on GPU.1, the sequential MoE artifact is
  the standing control, and chunked + MoE fails with
  `CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST`; a depth-1 discriminator is
  refused earlier by `SDPAToPagedAttention` (0 attention layers). Not pinned
  further; the plugin names no last primitive. No fix attempted.
- 2026-09-26, chunked-matcher leg (operator decision) — **localized and
  BLOCKED**. Instrumenting `gpu_usm::fill` shows the throw is a **fill** with
  rank-5 f16 weight layouts (512 MiB, the plugin's reorder of the packed u8
  weight) — a **packed-route** issue, not the GDN. Proof by isolation: the
  chunked core **with re-laid experts compiles and serves** on GPU.1 (t_boot
  48.0 s, prefill 118.5 t/s at 64 tokens). The matcher itself is **infeasible
  as stated** (`matches_linear_attention_loop`, `fuse_gated_delta_net.cpp`:62,
  pins seq extent 1 / Squeeze(2) / ReduceSum(-2) / row-wise ScatterUpdate — the
  token rule; a chunked body satisfies none; the fused primitive IS that rule,
  so a **new chunked primitive + kernel** would be needed). Mechanism numbers,
  same card/rung/32k: chunked prefill **153.5** vs sequential **161.8** t/s;
  decode 18.0 vs **26.7** — the unfused chunked **loses** to the fused
  sequential.
- 2026-09-26, packed-reorder leg (operator instruction) — **BLOCKED, premise
  disproven**. `KeepMOE3GemmConstPrecision` (the u4-only pass that marks MoE
  weight Constants keep-precision) **never fires** for the packed route: 0
  `ARCINT_PASS` lines over a full d4packed load — the packed route is not an
  `MOECompressed`/`GEMM3_SWIGLU` at all. Extending it to any Constant changes
  nothing; reverted (unverified). The real site is
  `ProgramBuilder::build` → `program::build_program` →
  `build_implementations` → `kernels_cache::build_all` →
  `_builder->build_kernels(..., KernelFormat::SOURCE, batch.options)`: the
  throw is the **OCL kernel compilation of a batch** (`CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST`
  surfacing through `gpu_usm::fill`'s wait), not a reorder. **None of the three
  candidate fixes has an object**: there is no reorder in the packed route to
  skip, retarget or bound. **No red-first cell** (naming the failing batch
  needs a plugin instrument this leg did not land). No artifact byte changed,
  no gate run; plugin reinstalled clean (`f9eb7ffdc5d83ee7`). **Rows 1–3 stay
  EMPTY (OWED).**
- 2026-09-26, packed-load leg 2 (operator time-box) — **RULE-OUT + NAMED CALL,
  failing kernel not named**. Instruments landed (patch 0053): on a d4packed
  A770 load, `kernels_cache::build_batch` = 5 batches / **0 exceptions** and
  `ocl_kernel_builder::build_kernels` = 5 programs / **0 failures** → the
  packed route's OCL kernel **build is exonerated**. The artifact is
  **3.63 GiB** (smaller than the 3.99 GiB re-laid control that serves) → fit
  exonerated. `ARCINT_BT` names the call instead:
  `err=-5 msg=[GPU] clEnqueueNDRangeKernel, error code: -5 CL_OUT_OF_RESOURCES`
  — a **kernel launch**, not a compile; the frames are unsymbolized (stripped
  Release plugin), so the failing kernel is **not named**. Hazard recorded
  dated in `contrib/packaging/marfrit-openvino/patches/README.md`: that OV
  measurement tree carries the patch set **uncommitted**, so a file-level
  `git checkout` silently drops a patch (0005 was dropped and re-applied
  verbatim during this leg). No red-first cell, no gate run. **Rows 1–3 stay
  EMPTY (OWED).**
- 2026-09-26, dispatch leg (`measured-here`, A770) — the full-depth Qwen3.6-35B
  now serves all-resident on this card (DESIGN §7.0.2ci), so row 3's rate
  question could be asked of the served path at depth: prefill **12.5 t/s** at
  4096. The mechanism was not the GDN core: the per-expert dispatch launched two
  kernels per (token, expert) pair (16,384 per MoE layer for a 1,024-token
  chunk). Patch 0059 batches them — **143.9 t/s** at 4096, decode 15.2, the same
  digests (DESIGN §7.0.2cj). Row 3c's bar (460 t/s) is not met. The rows'
  MODEL is now an open question: row 2's reference and its 2051 regime belong
  to Flash-Next, which cannot be resident on the A770; the 35B fits and serves
  84,704 tokens of context but has no 2051 boundary. **Rows 1–3 stay EMPTY.**

