# Handoff: M10 (not started) and M14 (nearly closed, three loose threads)

Written 2026-09-04 for whoever — human or model — picks this campaign back
up next. Assume less context than the author of `DESIGN.md` had: this
document exists so you do not have to reconstruct that context from git
history before you can act safely.

## Read this first

1. **`README.md` (scope), `DESIGN.md` (architecture, invariants, milestones),
   `llm.txt` (machine summary) are the source of truth**, in that order.
   `DESIGN.md` §3.4 and §3.8 (equivalence and the offload invariants) and §5
   (the gates) are not negotiable — do not implement around them, do not
   soften them, ask a human if a task seems to require it.
2. **`CLAUDE.md`** (checked into this repo) has the session rules: the
   public-repo no-leak rule, the measurement discipline, and which model
   tier to use for what kind of work. Read it before writing anything.
3. **`CLAUDE.local.md`** (git-ignored, sits next to `CLAUDE.md` in a real
   checkout) has the operator-local infrastructure detail — which host, how
   to reach it, the GPU test-window ritual, the safety rules for a shared
   machine. If your checkout does not have it, ask the operator for it
   before touching any hardware; do not guess at host names or invent a
   procedure. **This document deliberately says nothing infra-specific** —
   that is not an oversight, it is the same rule `CLAUDE.md` states: no host
   names, no addresses, in anything that gets committed.
4. **`docs/milestone-0.3.0.md`** is the actual milestone table. This handoff
   is a derived, task-shaped view of two rows in it (M10, M14) — if the two
   disagree, the milestone doc is correct and this file is stale.

## Ground rules that apply to everything below

- **A test must be able to fail.** Run the red case first.
- **Name the card, the depth, the KV precision and the configuration**
  whenever a number moves. A number without those four is not a
  measurement.
- **A claimed mechanism needs a measurement.** "This should be faster" is
  not a result. If you narrate a mechanism and do not measure it, say so
  explicitly rather than letting it read as verified.
- **No performance milestone closes as "we're ahead" without a recorded
  survey** of comparable stacks (upstream OpenVINO, vLLM/SGLang XPU,
  ik_llama.cpp, independent testbeds). `docs/milestone-0.3.0.md`'s "Where
  the bar sits" table is the last recorded survey (2026-09-01) — check
  whether it is still current before citing it.
- **Never touch the shared GPU host without following the safety ritual in
  `CLAUDE.local.md`.** In short, because it matters enough to repeat here
  even though the specifics live elsewhere: never kill a process on that
  host without tracing its parent first (a live process is more likely
  someone else's legitimate work than an orphan); always independently
  re-verify the production services are healthy after any GPU test window,
  never trust your own earlier log line as proof; large file transfers need
  the method `CLAUDE.local.md` names, not a naive one that fails on big
  files.
- **Do not install anything onto production** (the two services this
  repository serves in production) without being explicitly asked to. Build
  and verify in isolation; deployment is a separate decision every time.
- **Commits end with only a `Co-Authored-By:` trailer** if you are an AI
  model committing — no session IDs, no session URLs, in commits, docs, or
  replies.
- **Reviews are not optional.** Before anything lands on `main`, it needs a
  review pass by a stronger model than the one that implemented it (this
  repository's own convention: Haiku recon → Opus design → Sonnet red-first
  implementation → GPU verification → Opus/Fable-class review → commit). If
  you are the "lesser model" in that chain, your job is recon and
  implementation against a plan someone stronger has signed off on — not
  unreviewed commits to `main`, and not unilateral architecture decisions on
  a milestone this large.

---

## M10: sub-4-bit expert weights — NOT STARTED

Verbatim from `docs/milestone-0.3.0.md`'s M10 row (the authoritative spec;
this section only breaks it into an approach):

> **Sub-4-bit expert weights** — mixed per-tensor quantisation: rarely-routed
> experts below int4 (Q3-class ~3.2–3.5 bpw), dense and shared tensors at
> int4/int8; route is NNCF mixed-precision export or GGUF K-quant import
> with dequant-to-int4-at-load, decided by measurement.
>
> **Gate:** a mixed artifact serves with ≥ 15% more max context than pure
> int4 on the same card at Prüfstand 10/10; per-expert bpw map recorded in
> the artifact; quality delta vs int4 measured, not asserted.

Nothing in this repository implements any part of this. There is no
per-expert bpw map, no NNCF mixed-precision export path exercised, no
GGUF K-quant import path exercised, no measurement of which route is
better. Treat every claim below as "here is the shape of the problem," not
"here is what half-exists."

### The open design decision, and why it is not yours to make alone

The milestone text says the route is **"decided by measurement,"** not
picked in advance. Two candidate paths:

1. **NNCF mixed-precision export**: OpenVINO's own quantization toolkit
   picks per-tensor bit-widths during model conversion. Pro: stays inside
   the toolchain this repository already uses end to end (`export_*.py`,
   whatever produces the IRs `backend_ov.cpp` loads). Con: unknown whether
   NNCF's mixed-precision search even targets MoE expert tensors
   specifically, or treats them like any other linear layer — this needs
   checking against NNCF's actual documentation/source before assuming it
   works the way the milestone text hopes.
2. **GGUF K-quant import with dequant-to-int4-at-load**: import an
   already-quantized K-quant (Q3/IQ3-class) GGUF artifact and dequantize
   sub-4-bit experts up to int4 at load time, keeping the plugin's existing
   int4 kernel path unchanged. Pro: K-quant formats and their calibration
   are mature and well-understood (llama.cpp/ik_llama.cpp lineage). Con:
   a dequant-at-load step is new code on the load path, and "dequant to
   int4" throws away exactly the bit-width win this milestone exists to
   capture unless the *storage* format is what's smaller (i.e., the
   artifact on disk is sub-4-bit even though the served/decoded tensor
   briefly touches int4) — read the milestone text again if this feels
   circular, because it does; get a stronger model or the operator to
   confirm the actual intent before writing code against a
   misunderstanding.

**First move, before any implementation: recon.** Per this repository's own
pipeline convention, the right first step is a bounded research task (the
"Haiku recon" role, even if you are not literally that model): read NNCF's
mixed-precision quantization docs/source for whether it can target MoE
expert tensors selectively; read what K-quant/IQ3-class formats actually
store on disk and whether "dequant to int4 at load" preserves a disk-size
win; check whether the plugin's grouped-int4 weight layout (the same
layout `patches/0018-moe-cpu-tier-static-partition.patch`'s own header
describes arcint as deliberately NOT repacking, for unrelated reasons)
can even accept a sub-4-bit input without a repack. Write up findings before writing any
code. Do not guess at NNCF's or GGUF's behavior — check the actual
libraries/formats.

### Concrete task breakdown

1. **Recon** (above). Deliverable: a short write-up (where NNCF's
   mixed-precision path does or doesn't reach MoE tensors; what a K-quant
   dequant-at-load path would actually cost/save) that a stronger model can
   review before committing to one route.
2. **Design**: once the route is chosen, a design note under `docs/`
   (follow the `design-m8-asymmetric-kv.md` / `design-m9-offload-v2.md`
   naming convention) describing the per-expert bpw decision rule (what
   makes an expert "rarely-routed" — almost certainly ties into the
   routing-histogram instrumentation patch 0013 already added, check
   `DESIGN.md` for `MOE_OTD_ROUTING_HIST` before inventing a new signal),
   the artifact format change (where the bpw map lives), and the load-time
   dequant/repack path if one is needed.
3. **Red-first tests**: a test that fails on today's pure-int4 artifact and
   a fixture defining what "per-expert bpw map recorded in the artifact"
   actually means as a data structure, before writing the code that
   produces it.
4. **Implementation** against the signed-off design.
5. **Gate measurement**: max-context comparison (mixed vs. pure int4, same
   card, Prüfstand 10/10 both), quality delta measured (not asserted — this
   repository's language for "we didn't actually check" is exactly that
   phrase, avoid earning it), per-expert bpw map confirmed present in the
   artifact.

### Landmines specific to this milestone

- `DESIGN.md` §7 has **two retracted headlines** from measuring at the
  wrong chunk/depth (see "Ground rules" above, item 3, and the retraction
  language throughout §7 — search for "RETRACTED" to see the pattern
  before repeating it). Quality-delta and max-context measurements for M10
  must name the same four things every other measurement in this
  repository names: card, depth, KV precision, configuration.
- This milestone's own gate is a **relative** one (≥15% more context than
  pure int4) — measure both arms in the same window, same card, not from
  two different sessions' numbers.

---

## M14: CPU compute tier — mechanism landed, three loose threads before it can be called closed

M14's actual mechanism work is done and measured: the host CPU tier's
hybrid decode split (skip-mask in the fused GPU kernel, host x→y excursion
on an LRU miss, readback decomposed and optimized by patch 0017) is real,
and the `DESIGN.md`/`docs/milestone-0.3.0.md` M14 row has a long, honest
history of what was tried, what was retracted, and what the numbers
actually were. **Do not re-derive any of that** — read
`docs/milestone-0.3.0.md`'s M14 row and `DESIGN.md` §7.0.2x/§7.0.2ae/
§7.0.2af before touching anything here, so you don't repeat work that is
already on the record (including two already-retracted headline numbers —
read why they were retracted, not just that they were).

What is actually still open, in priority order:

### 1. The tier's own decode-rate headline needs re-confirming under the NEW residency mechanism

The 2026-09-02 measured result (`docs/milestone-0.3.0.md` M14 row: tier ON
15.0/15.5 t/s vs. OFF 10.4/10.6 t/s at the reference cell) was taken while
the tier's expert residency was **F0, process-global LRU**. A separate,
more recent line of work (M9, see `DESIGN.md` §7.0.2ae and the commit that
introduced `patches/0018-moe-cpu-tier-static-partition.patch`) replaced
that residency mechanism with **F2, a static per-seed partition**, because
the LRU version violated `DESIGN.md` §3.4 (greedy output depended on
request history). F2 changes *which* experts are resident and when — a
different selection than LRU would have made in the same run. **Nothing
has re-confirmed that M14's own decode-rate headline still holds under
F2.** It plausibly does (the mechanism is still "compute on host instead
of uploading"), but "plausibly" is exactly the word this repository's own
rules forbid substituting for a measurement.

Also still owed, named explicitly in `patches/0018-moe-cpu-tier-static-
partition.patch`'s own header (search it for "still owed"): an E2 check
(one process asked the same prompt twice then a continuation, vs. a fresh
process asked the continuation directly) and a decode-cost cell, both
against the **final**, bug-fixed `+p4` plugin build — earlier runs of both
predate one or more of the five load-time bugs that build fixes.

**Task**: re-run the M14 reference cell's decode-rate comparison (tier ON
vs. OFF, same card, same ratio, same pool size — match the 2026-09-02
cell's configuration exactly, named in the milestone row) against the
`+p4` package (built and verified, not yet installed anywhere — check with
whoever handed you this repo where the built `.deb` currently sits, or
rebuild it from `contrib/packaging/marfrit-openvino/` following that
directory's own `build-openvino.sh`/`build-deb.sh`). Also run the owed E2
and decode-cost checks from the patch header. Record whichever result you
get — win, loss, or unchanged — in `DESIGN.md` (new dated update under
§7.0.2ae or §7.0.2af, whichever the result is more naturally an update to)
and close the loop in `docs/milestone-0.3.0.md`'s M14 row with a dated
line, the way every other entry in that row already does.

### 2. A pre-existing oversized buffer declaration, flagged but not investigated

Found during the M9 investigation, in the plugin source, inside the same
file the M14 tier's own kernel code lives in
(`moe_3gemm_swiglu_opt.cpp`, the resident/device branch of the fused
prefill path near where per-expert oneDNN kernels are dispatched): a line
declaring `routing_weights_size = n_token * max_topk` for a oneDNN scale
tensor, in a code path where the gather kernel that fills it
(`moe_3gemm_swiglu_fuse.cl`) never actually writes that many elements. It
predates every patch this campaign has touched (present verbatim in the
pre-0018 baseline) and is not reached by anything patches 0017/0018
change, so it did not block M9's or M14's own gates — but it sits directly
in M14's own tier code and is worth a dedicated look: read both the
kernel's write pattern and the declaration's actual consumption to
determine whether it's a real (harmless, just imprecise) over-allocation
or masks something worse. Do not assume either answer — check the actual
element counts against `max_topk` at the model shapes this repository
actually serves.

### 3. Known, accepted, dormant risk — do not "fix" without a reason to touch it

`patches/0018-moe-cpu-tier-static-partition.patch`'s header documents (its
own "Not fixed here" section) that `on_load_expert_weights`'s `false`
return is overloaded — it means both "the offload tier is off entirely"
and "this expert is not resident under the static partition" — and its one
caller does not distinguish the two. On a **non**-offloaded load reaching
that caller, the ambiguous case would be undefined behavior. It is
reachable only when a specific plugin flag
(`MOE_USE_GROUPED_GEMM_PREFILL`) is forced off, which defaults `true` and
which nothing in this repository ever sets — so it is dormant in every
configuration this repository actually drives today. **Do not go fix
this unless something in your own work starts touching that flag** — it
is recorded so nobody re-discovers it from scratch, not because it is
blocking anything right now. If a future change does need to touch
`MOE_USE_GROUPED_GEMM_PREFILL`, read the patch header's own description of
the exact call chain first.

### Once 1–3 are done

Update `docs/milestone-0.3.0.md`'s M14 row with a final dated line stating
the milestone's own gate outcome (win or lose against M9's device-tier
path, per the milestone's own redefined gate — re-read the row's
2026-09-02 "scoped, gate redefined" entry for the exact wording of what
"closed" means here before writing the closing line). That is the last
step; nothing else is currently blocking M14.
