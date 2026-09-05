# static-partition-cold-start — a cold sequence's first processes pay minutes of warming under the static partition; find the owner, then remove it

## The defect, as measured

DESIGN §7.0.2ai, the 0.3.0 release gate, M14's reference cell (16 GiB card,
35B int4, ratio 50, 8 GiB device pool, u8 KV, one lane, n_ctx 65,536, the
1,198-token prompt, 64 greedy tokens, two requests per process, patch 0018's
static partition). On the fixed binary (the admission-path fixes described
there already applied), tier-ON loads ran 215–585 s against tier OFF's
30–45 s. With the device term forced to the plateau probe's own figure (a
diagnostic run, not the shipped admission path) the first two tier-ON
processes loaded in 340 and 275 s, served their first request at 2.8/0.1 and
9.9/4.2 t/s prefill/decode and their second at 9.2/4.7 and 26.7/16.6 — while
the fourth tier-ON process of that same night was fast from its first
request. `created_onednn_kernels=325` per process (patch 0004's counter) is
recorded beside this as one candidate owner. Decode itself is not the
defect: warm, the tier holds the record, 16.4 t/s on the second request of
both processes (15.4 already on the first request of the warm-cache one),
at the LRU-era figure of 15.0/15.5 (§7.0.2x) and above tier OFF's 11.3–11.4
on the same request in the same window.

## Known against hypothesised

Known: the fixed-binary range (215–585 s tier ON vs 30–45 s tier OFF), the
first-request decode range (0.1–4.2 t/s against a warm 4.7–16.6 t/s),
`created_onednn_kernels=325` per process, and that the fourth process of the
diagnostic run was already fast from its first request — all §7.0.2ai.
Hypothesised, explicitly not separated on the record: first-use JIT of the
per-expert oneDNN kernels (one candidate reading of the 325-kernel count,
which recurs per process regardless of what an earlier process already
did), a page cache warmed by earlier processes' first-use fills (a
candidate reading of why the fourth process was already fast), and
first-use fills of some other resource neither counter names. §7.0.2ai
states plainly: "the two were not separated."

Prior art, surveyed 2026-09-05 and recorded with URLs, licenses and
Arc applicability: `research-cold-start.md` (same directory). Its "what transfers"
section is the recon's starting point, not a substitute for it.

## Gate

The first request of a cold sequence within 2× of a warm one at the M14
reference cell, decode still at the record (not below §7.0.2ai's 16.4 t/s
warm second-request figure), with the continuation-restore check and E2
(DESIGN §7.0.2ae/§7.0.2ai) still passing — copied, with the record made
numeric, from the 0.3.1 backlog row (`docs/milestone-0.3.0.md`, which states
only "decode still at the record").

## Entry criteria

The warming's owner identified — kernel cache vs. page cache vs. first-use
fills — before anything is built, per the backlog row. Not yet met: no
window has isolated the three candidates from each other; that isolation is
this campaign's first deliverable, not a precondition already on record.

## Scope — in / out

In: an instrumented experiment that separates the three candidates (for
example, a cold-page-cache first process against a warm-page-cache first
process at fixed `created_onednn_kernels`, and a run with the plugin's own
kernel-cache mechanism, if any exists for this graph, forced on/off); once
the owner is named, the lever it implies — a persistent kernel cache or an
explicit pre-warm pass run before the first real request.

Out: the prefill fallback's own throughput cost (`static-partition-prefill`,
same reference cell, a different mechanism — the grouped-GEMM refusal, not
warming); seeding the partition from a routing histogram
(`partition-seeding`, conditional on this campaign and the prefill one);
any change to the static partition's ranking rule itself.

## Where it lives

Patch 0018's header and `static_partition.hpp`
(`contrib/packaging/marfrit-openvino/patches/
0018-moe-cpu-tier-static-partition.patch`) — `bind()`'s `reserve_pinned`
pass, which fills each pinned slot on first use, is the seam the "first-use
fills" candidate points at. Patch 0004's counters
(`created_onednn_kernels`, `grouped_fallbacks`) — the plugin's own
`[OTD_PERF]` log line. `src/exec/backend_ov.cpp` (~2640–2700, the
`MOE_CPU_TIER_STATIC_PARTITION` property read and the `--prefix-cache-mib`
refusal; ~3280–3340, the plateau-probe skip under the static partition and
`static_partition_reported`) and its blob-cache handling for the paged graph
(~2694–2697, `core_.set_property(ov::cache_dir(""))` — the *model* blob
cache is off there by design, a different cache from oneDNN's own kernel
JIT that `created_onednn_kernels` counts; worth distinguishing rather than
conflating when this campaign designs its experiment). The acceptance cell
`tier-reference-cell` (`tests/acceptance/cells/tier_reference.sh`) emits
only the *warm* (second-request) metrics today — `decode-warm-2nd-on/off`,
`prefill-warm-2nd-on/off`, `decode-ratio-on-off`, `grouped-fallbacks-on`
(the script's `metric_value ... 2`, lines 182–189) — it does not currently
emit a first-request/cold metric, so this campaign's gate needs either a
new metric (a `metric_value ... 1` call, already computed positionally by
the same helper) or a dedicated cold-sequence runner.

## Pipeline for this campaign

Recon (read patch 0018's `bind()`/`reserve_pinned` path and confirm whether
the plugin has any kernel-persistence mechanism for the oneDNN primitives
this graph creates) → design note (`docs/design-static-partition-cold-start.
md`, since isolating three candidates is more than a one-line fix) → red
case: a reproducible cold-vs-warm gap at a fixed `created_onednn_kernels`
count, so the test can fail before the owner is separated → one card
window: a controlled process sequence (fresh page cache, then repeated
processes) with the candidate counters read after each, isolating the
owner → the lever (persistent kernel cache or explicit pre-warm) → the same
window's cells: `tier-reference-cell` with the new cold metric, and E2 →
review → commit → DESIGN §7.0.2x record naming the owner and the fix,
CHANGELOG line, the backlog row's closing line.

**The first window, as a ladder** (operator hypothesis, 2026-09-05: "if
the heuristics are saved, the cold start goes away" — testable, because
under the static partition there is no learned state to save, only three
kinds of recomputed work, and each step below moves exactly one of them;
`research-cold-start.md` has the knobs):

1. *Is a kernel cache already doing the "later processes are fast"?* Read
   the compute runtime's on-disk program cache state in the container
   (enabled, directory, size against its cap, populated). If it is the
   mechanism, the cold process is a cache miss and step 2 measures the
   miss.
2. *Compiled kernels.* Two loads of the tier-ON reference cell, the
   runtime cache cleared before the first and intact before the second,
   page cache warm in both (the files read once by timed `dd` before
   either): load time, `created_onednn_kernels`, first-request decode.
   The delta is the JIT term. Then the same with the plugin's own
   persistence, if any, and OpenVINO's `cache_dir` on for this graph.
3. *Page cache.* Two loads with the runtime cache intact, the artifact
   files evicted before the first (measured by a timed read, not
   `fincore` — the root is on ZFS and its ARC is invisible to it) and
   warm before the second. The delta is the I/O term.
4. *First-use fills.* One process, runtime and page cache both warm: the
   first request's decode against the second's, then the same process
   after one explicit pre-warm request. The delta is the fill term.
5. Whichever step swallows the 215→45 s and the first-request cliff
   names the owner. If step 2: the lever is a persistent, pre-populated
   kernel cache (already saveable; make it reliable). If step 3 or 4: no
   file can hold it, and the lever is a pre-warm at service start, kept
   out of the gate's timing. If two steps share it, both levers, and the
   record says which bought what.

Every number: card, artifact, flags, pool bytes, the runtime's patch
level, and which of the four caches was warm — a load time without that
list is the kind of figure §7.0.2ai could not separate.

## Invariants

The static partition's history-independence (DESIGN §3.4, the reason patch
0018 exists) must hold through any lever here — a pre-warm pass or a kernel
cache is a per-process, seed-independent artifact, never a function of
which requests a process has already served. Decode's record is not traded
for a faster cold start.

## Status

- 2026-09-05 — opened from the 0.3.1 backlog row; nothing started.
- 2026-09-05, later — the operator's hypothesis ("save the heuristics
  periodically and the cold start goes away") turned into the first
  window's four-step ladder in the pipeline above; the honest reading is
  that only the compiled-kernel term is saveable, and the runtime may
  already be saving it. Not run yet: needs a quiet card.
