# expert-hot-set-lru — card hot-set + host LRU for expert slots, seeded from a per-token routed-expert census

Charter: 0.5.2 VENICE (`ROADMAP-0.5.x.local.md`). The "smart half" of the
FreeToken-shaped goal — a card-resident hot set of expert slots and a host
LRU for the rest, with the hot set chosen from a measured expert-access
census, not from a random seed.

## The defect, as measured

Not a defect: a **lever**, and on the record today it is a lever whose speed
half is **blocked by a named absence**, not by a missing policy. Three
measured facts frame it:

- **The host-bound baseline is measured.** Served `d48n` on the B60
  (`--offload-ratio 99 --moe-cpu-tier`, KV u8, f16, chunk 512, native
  artifact) decodes **0.5–0.8 t/s** — every routed expert decodes its rows
  on the scalar host path (`measured-here`, `sub4bit-vram-kernel` status
  2026-09-18 night; DESIGN §7.0.2ca). Served decode on this artifact is
  therefore **host-COMPUTE-bound**, not host-feed-bound.
- **The resident-compute path does not exist yet.** Under the native
  artifact every routed expert runs on the host tier because the fused
  kernels refuse the native formats (`code`, patch 0043's in-code assert
  that every routed expert is computed on the CPU tier until the OpenCL
  decode exists `:726-731`, DESIGN §7.0.2ca; design note
  `docs/design-routing-aware-expert-execution.md` §2.3b/§2.3c). [DATED IN
  PLACE 2026-09-22: it exists now — patches 0043/0045 carry the native
  per-expert OpenCL decode, served at the 0045+0047 prefix `ov-0047`; the
  statement above was true when written.] Keeping an
  expert's bytes on the card changes **no** compute in that regime — the
  host kernel still decodes and GEMVs every routed expert from host mmap.
  The rate lever is `sub4bit-vram-kernel`'s step 3, the OpenCL decode of the
  native formats in the per-expert kernel.
- **The census, as it exists, cannot choose a hot set.** Patch 0013's
  `MOE_OTD_ROUTING_HIST` counts routing per `(call, expert)`, **before** the
  hit/miss split (`code`, patch 0013 header: the counting loop is the first
  thing `try_acquire_simultaneous` does, above the dedup map) and dumps an
  aggregate CSV whose contract is `layer,weight_offset,expert,count` ordered
  by `weight_offset`, `layer` a 0-based rank of the weight-file offset
  (`code`, `patches/0013-moe-otd-routing-histogram.patch:279-281,440`). It
  answers "which experts route", not
  "in what order", so it cannot feed the per-layer LRU replay. Run over the
  acceptance prompt alone it left most of 7,360 experts at **0–2 routings**
  — "no distribution to threshold 'rarely-routed' on" (`measured-here`,
  DESIGN §7.0.2ah; the same corpus gap is `partition-seeding`'s entry
  criterion (2)).
- **The offline replay exists and is sha-pinned.** `tools/expert_lru_replay.py`
  consumes one line per `(token, layer)`, `<token_idx> <layer_idx>
  <expert_id> ...`, models both a per-layer and a global LRU, and reproduces
  WP6b's per-layer hit-rate table within ~1.4 points on a trace pinned by
  sha256 (`measured-here`/`code`, tool docstring; `--check`). What it has
  never had is a **served-path trace** in that format.

So the campaign's own first absence is the gap between patch 0013's
aggregate counts and the trace `expert_lru_replay.py` needs. That is the
census instrument this campaign's charter names as its first deliverable.

## Known against hypothesised

**Known.**

- The device slot pool, its async upload ring and the per-layer LRU cache
  exist (patches 0005–0007, 0012; `measured-here`, DESIGN §7.0.2x/§7.0.2ai).
- Patch 0018 pins the host/device split for the process life as a static
  per-`(layer, expert)` partition, ranked by `splitmix64(seed, layer_key,
  expert)` with **no routing-frequency term**, exactly so greedy output is
  history-independent (`code`, patch 0018 header; DESIGN §7.0.2ae "F2",
  §3.4).
- The routing-aware design note deliberately does **not** seed from a
  histogram: "the LRU cache here does not seed from a histogram; it warms by
  demand on the first forward" (`code`, `docs/design-routing-aware-expert-
  execution.md` §7). The demand-warm LRU is the *comparand*, not this
  campaign's policy.
- The short-corpus sparsity above is measured, not assumed.

**Hypothesised** (each falsifiable, none measured):

- A frequency-ranked hot set (top-`S` per layer by census count,
  deterministic tie-break by ascending expert id) beats the random
  `splitmix64` seed at a fixed slot budget, on served decode and on
  rounds-to-plateau.
- The per-layer LRU replay's hit rate at a given resident budget transfers
  to served decode **once the resident-compute path exists**; until then it
  predicts bytes moved, not t/s.
- Seeding the static partition from the census lowers the warm-up cost that
  `static-partition-cold-start` owns, without touching §3.4 (the seed is a
  pure function of the recorded census, not of run history).

**Named dependency, not a hypothesis.** The speed half of this campaign
cannot be measured on the native artifact until either (a)
`sub4bit-vram-kernel`'s OpenCL decode lands, or (b) a resident-compute arm
of the fused path is produced. Stated here so the gate's emptiness is
attributable, the way window-051 clause (d)'s emptiness was. [DATED IN
PLACE 2026-09-22: condition (a) is met — the native per-expert OpenCL
 decode (patches 0043/0045) serves at the 0045+0047 prefix `ov-0047`; the
 dependency is discharged and the speed leg runs.]

## Gate

Copied from `ROADMAP-0.5.x.local.md` 0.5.2 and `docs/window-052.md`
(committed `eaa7a06`), which is the acceptance document:

- **Speed:** warm-up decode ≥ **G × the host-bound baseline** (the gate
  copied from `ROADMAP-0.5.x.local.md` 0.5.2 and `docs/window-052.md`; the
  G basis below is an **amendment** to that copy, dated 2026-09-21). The
  baseline is fixed at the measured `d48n` host-tier rate, **0.5–0.8 t/s**
  (B60, ratio 99 + tier, KV u8, f16, chunk 512). **G is pinned in a dated
  prediction commit before the speed leg runs**, from the census-measured
  per-layer hit rate and the measured resident-compute rate; until the
  resident-compute path exists G is **UNPINNED** and the row reads EMPTY,
  not PASS. No stale figure (the 23.6 t/s HF-exported 35B control) is
  inherited. [DATED IN PLACE 2026-09-22: the resident-compute path now
  exists (native per-expert OpenCL decode, patches 0043/0045; served prefix
  `ov-0047`), so **G is PINNED at 1.10** in `docs/window-052.md` — the
  free-card ceiling `1/(1 − h₉₉) = 1.1106` rounded down, from the census
  top-5 coverage `h₉₉ = 9.9595 %`. The resident-compute ratio `ρ` is NOT
  separately measured: the one counter point (ratio 75) mixes the load
  probe, the warm-up and the 16 served decode tokens, so it is an
  observation, not a fit. The pinned prediction is a V1 shortfall with the
  hot-set *correctly engaged*. The `UNPINNED` sentence above stays as
  written, marked.]
- **Speed hold (operator decision, 2026-09-21):** the speed row waits for
  `sub4bit-vram-kernel` step 3, the OpenCL decode. Census + policy land
  first; the speed measurement is not attempted on the host-compute tier.
  [DATED IN PLACE 2026-09-22: the dependency is met — the native per-expert
  OpenCL decode (patches 0043/0045) serves at the 0045+0047 prefix
  `ov-0047`; the speed leg runs and G is PINNED at 1.10 in
  `docs/window-052.md`. The hold above stays as written, marked.]
- **Stale-byte zero:** `digest(host-bound bytes of expert E) ==
  digest(card-bound bytes of the same E)` for every E in the hot set, with a
  **red-first mutation on eviction** (perturb one byte in the eviction path
  and watch the digest row go RED).
- **Convergence:** rounds-to-plateau printed with the census.
- **Quality:** no greedy digest change against the pre-policy served answer
  (DESIGN §3.4).

Failable clauses (from window-052, V1–V4): V1 speed shortfall, V2 stale
byte, V3 no plateau in the predicted rounds, V4 visible policy change.

## Entry criteria

1. **The census instrument exists** — a per-token routed-expert trace and an
   aggregate histogram, with **storage and format named** and a **device-free
   source** (this campaign's first deliverable; design note
   `docs/design-expert-hot-set-lru.md`).
2. **A census over a corpus long enough to threshold "hot"** — the
   partition-seeding gap; the acceptance prompt alone is not enough.
3. **G pinned before the speed measurement**, per above.
4. **A resident-compute measurement path named** — the current native
   artifact has none. Per the operator decision of 2026-09-21, VENICE's
   **speed leg is HELD** until `sub4bit-vram-kernel` step 3 (the OpenCL
   decode of the native formats in the per-expert kernel) lands: the census
   and policy paths proceed now, and **no speed measurement is taken before
   that patch**. Until then the speed row reads EMPTY, not PASS. [DATED IN
   PLACE 2026-09-22: the path now exists — the native per-expert OpenCL
   decode (patches 0043/0045) serves at the 0045+0047 prefix `ov-0047`
   (plugin `f021de51b5812ee2`) on both cards; the speed leg is no longer
   held and G is PINNED at 1.10 in the acceptance document. The `HELD`
   sentence above stays as written, marked.]

## Scope — in / out

**In:** the census trace/histogram instrument and its storage format; a
hot-set selection function (frequency rank + deterministic tie-break); the
eviction/refresh discipline on top of patches 0005–0019 (seed the static
partition from the census; keep the demand-warm LRU as the comparand); the
stale-byte digest proof; the convergence measurement; one card window at the
end.

**Out:** the OpenCL decode / per-expert kernel (`sub4bit-vram-kernel`);
NVMe (`nvme-direct-expert-tier`, LISBON 0.5.3); the grouped prefill split
(`static-partition-prefill`); the prefix-cache seam (DESIGN §3.4 forbids
`--moe-cpu-tier` with a prefix cache); any FreeToken comparison (GENEVA
0.5.8, and pinned only by our own runs).

## Where it lives

- Acceptance: `docs/window-052.md`.
- Instrument design: `docs/design-expert-hot-set-lru.md`.
- Engine: `src/exec/fit.h` (`expert_slot_bytes`, `expert_slot_bytes_static`),
  `src/exec/flash_next_offload.h` (per-layer slots, projection),
  `src/exec/backend_ov.cpp` (property wiring, plateau probe).
- Plugin patches: 0005–0007 (slot pool + upload), 0012 (decode split), 0013
  (routing histogram), 0018 (static partition / LRU partition), 0043 (native
  formats through the tier).
- Tools: `tools/expert_lru_replay.py`, `tools/flash_next_fit.py`,
  `tools/ref_forward_stream.py` (the device-free reference forward), and the
  new census instrument `tools/hot_set_census.py`.
- Related campaigns: `static-partition-prefill`, `partition-seeding`,
  `static-partition-cold-start`, `sub4bit-vram-kernel`.

## Pipeline for this campaign

recon (this document + the instrument design + the cited headers) → the
census instrument, **red-first and device-free first** (reference-router
trace), then the served-path trace in one card window → census run over a
long corpus; hit-rate and rounds-to-plateau → **G pin** (prediction commit)
→ hot-set selection + eviction discipline, red-first → stale-byte digest
proof, red-first mutation → one card window at the end → review before every
commit → a DESIGN `§7.0.2x` record and a CHANGELOG line when it closes.

## Invariants

DESIGN §3.4 (history-independent greedy output) and §3.8, the §5 ladder, and
`CLAUDE.md`'s measurement discipline are non-negotiable. A policy that would
trade §3.4 for a hit rate does not close; the trade is recorded as a finding
and the campaign stops. Every disposition carries an evidence class
(`paper` / `code` / `measured-here`).

## Status

- 2026-09-21: campaign opened and registered in `docs/campaigns/README.md`.
  The acceptance document is `docs/window-052.md` (committed `eaa7a06`,
  2026-09-19), whose measured rows stay EMPTY. The host-bound baseline was
  corrected in place to the measured `d48n` rate (0.5–0.8 t/s) on the same
  date. The census-instrument design note landed as
  `docs/design-expert-hot-set-lru.md`. **No measured row filled; no policy
  code yet.**
- 2026-09-21: **operator decision — VENICE's speed leg is HELD for the
  patch.** `sub4bit-vram-kernel` step 3 (the OpenCL decode of the native
  formats) is the dependency; the census instrument, hot-set selection,
  eviction/refresh discipline and the stale-byte digest proof proceed on the
  host-tier path in the meantime. The speed row stays EMPTY, and G stays
  UNPINNED, until the resident-compute path exists. Recorded here so no
  window is spent measuring a tier that cannot carry the policy.
- 2026-09-21: **the device-free census instrument landed.**
  `tools/hot_set_census.py` (format-v1 parser, canonical summary, frequency
  rank, budgeted selection, rounds-to-plateau, patch-0013 four-column parser
  and join, seed emission) and `tools/ref_forward_stream.py --router-trace`
  (the device-free reference-router source, sharing the writer). Cells:
  `tools/test_hot_set_census.py`, **31 green, no card**. A census over the
  committed 400-token fixture **does not plateau** (S = 10 per layer:
  coverage 0.636 at prefix 2 falling to 0.356 at prefix 400, and the selected
  set changed at every power-of-two prefix), which confirms entry criterion
  (2) as a measurement — the corpus is too short to threshold "hot" — not an
  assumption. **Not started:** the
  served-path trace (one B60 card window) and the stale-byte digest proof
  (it needs the engine-side host/card readback; per the design it is not
  asserted from code). Outputs on persistent paths under the operator's
  census directory.
- 2026-09-21: **`ref_forward_stream.py --router-trace` verified, and its
  long-corpus limit measured.** `measured-here`: smoke at 1 layer, T = 5,
  `--device cpu`, a format-v1 trace of 5 token-major rows with top-10 ids
  ascending (`card=none device=cpu`), parsed back by
  `tools/hot_set_census.py shape`; trace sha256 `4659806b…49c0`, log sha256
  `3c2404ab…fe4c`. The tool **dequantises every expert tensor** per layer
  (`[forward] T=5 in 119.3s (expert loads 117.8s over 1 layers)`), so the
  cost is per-LAYER and T-independent and repeats each window; the pin's own
  expert loop is sparse (`code`, `tools/q4e/ref_moe.py:18`). The 48-layer
  long-corpus cost is a **projection** (hours per window), not measured, so
  the **reference trace is a short-corpus oracle only** and the long-corpus
  census cannot come from the proxy. **Decision on 0044:** author it at
  `OffloadExpertWeightProvider::try_acquire_simultaneous` as a per-call trace
  (`<call_seq> <layer_key> <top_k> <ids...>`), converted offline to format v1
  by splitting each call's flattened ids into per-token `top_k` chunks and
  mapping `layer_key` to the decoder index by ascending weight-offset
  (export) order (assuming export order equals decoder order, as patch 0018
  already assumes, and that the served op's flattened ids are token-major;
  for decode T=1 this is exact); build to a third prefix on the build host
  against the
  pinned tree, then take **one short B60 window** over the long corpus with
  the emitter on. The measurement plugin stays untouched, and no card has
  been taken yet.
- 2026-09-21: **patch 0044 authored, built, and the offline converter
  landed.** The emitter is `MOE_OTD_ROUTING_TRACE`, a per-call trace at
  `OffloadExpertWeightProvider::try_acquire_simultaneous`, added to the
  canonical series as
  `contrib/packaging/marfrit-openvino/patches/0044-moe-otd-routing-trace.patch`
  (mirrored in `patches/`) and applied on top of the 41 patches against pin
  `71640275`; built via `ninja openvino_intel_gpu_plugin`; the third-prefix
  install reports plugin version
  `2026.4.0-22849-71640275d29-marfrit-p19` (stamp deliberately left at p19,
  which already names patches 0003–0043; the trace build is identified by its
  `routing_trace` symbol). The default-measurement plugin
  and the debug-caps install are untouched. `tools/hot_set_census.py` gained
  `parse_call_trace`, `split_topk_chunks`, `layer_key_index_map`,
  `call_trace_to_v1` and the `from-call-trace` subcommand; cells are 43
  (superseded below by the 2026-09-21 batched-prefill/provenance entry: 52)
  green, including the two silent-if-wrong assumptions (a two-token call
  splits into the right `top_k` chunks / a mis-sized call is refused; the
  `layer_key` -> decoder index map is a bijection over the artifact's own
  layer set or the conversion refuses).
- 2026-09-21: **census-card decision, stated BEFORE the leg: the A770.**
  [decision, not a measurement] A census taken on the B60 would inherit the
  B60's run-to-run varying router input (DESIGN §7.0.2cb: the GDN state
  output is nondeterministic on Xe2), so its counts could move between
  forwards; the A770's served depth-48 path is bit-identical ×8 (the 74bc082
  arm). If a B60 census is ever required, it will be run twice and the
  run-to-run SPREAD of the counts printed beside the census, as the floor
  rule requires for any other reading. A census whose stability is unstated
  is not an instrument for a policy.
- 2026-09-21: **the converter's batched-prefill gap is closed, and the
  artifact/card part of the provenance requirement is enforced.** [code,
  measured-here for the cells] A real served trace opens with a batched prefill
  call, which `call_trace_to_v1` refused, so the captured trace could not
  convert at all. `from-call-trace` now takes **`--skip-batched`**: the batched
  call is skipped and **reported** (call and token counts in the v1 header, and
  on stderr when writing to stdout), because a batch dropped silently is a
  census that under-counts. Two rules hold either way: a trace in which EVERY
  call was batched is refused rather than converted to an empty census, and an
  empty call list stays an empty conversion (not a census). The same change
  makes part of the harness-injected provenance header **mandatory**:
  `from-call-trace` requires `--provenance` and refuses a file without a
  NON-EMPTY `artifact_sha256=` (or `artifact=`) and `card=`, so a census with
  no artifact or card attribution cannot be written. That is exactly what it
  says — artifact and card, nothing more: `capture_sha256`, `kv`, `depth`,
  `chunk`, `offload_ratio`, `tier`, `run` and `utc` from §2 are still the
  harness's responsibility and are **not** machine-checked, so §4b's owed
  item is narrowed, not closed. `tools/test_hot_set_census.py` is now **52
  cells** (was 43): `skip_batched` skips and counts while keeping the decode
  rows, an all-batched trace is refused with the skip ON, the CLI refuses a
  missing, incomplete or empty-valued provenance file, both §2 artifact
  spellings are accepted, and the stdout path still emits a parseable v1
  trace. Note the shape of a skipped opening prefill **as modelled here**: the
  prefill is one batched call, so the decode rows resume mid-sequence and the
  first reconstructed token can be partial — a real depth-48 served prefill
  emits one batched call per layer, all of them skipped, which leaves the
  decode stream at layer 0. Either way the token labels are a reconstruction,
  now stated in the header (`# token_labels=reconstructed`); aggregate counts
  do not depend on them, the LRU/plateau do. The served-path trace cell (§7.3)
  still needs the A770 window and is run against the VENICE plugin prefix
  (patch 0044 built there; the operator-local prefix path is recorded in
  `CLAUDE.local.md`); the speed row stays EMPTY and G UNPINNED.
- 2026-09-21: **corpus-split decision, stated BEFORE the served window.**
  [decision, not a measurement] Two attributions have to be separated before a
  number is read, and both are stated here rather than inferred from the
  result.

  **1. Batched prefill vs decode rows.** The census is the aggregate histogram
  of routed-expert accesses over the long fixed corpus. A served trace opens
  with BATCHED prefill calls (one call per layer, each carrying many tokens'
  ids, `top_k`-flattened token-major), and `from-call-trace --skip-batched`
  drops exactly those. A census derived from the converted v1 rows would
  therefore UNDER-COUNT the corpus it claims to measure. Decision: the census
  (the canonical `layer,expert,count` aggregate) is derived DIRECTLY from the
  patch-0044 call trace, counting every id of every call including the batched
  prefill calls — an aggregate needs no token label. The converted v1 decode
  rows are reserved for the ROW-LEVEL consumers only (LRU replay,
  rounds-to-plateau, token labels), which are explicitly a reconstruction.

  **2. Corpus calls vs load-time probe calls.** With `--offload-ratio > 0` the
  engine runs a distinct-token plateau probe (up to 8 forwards of
  `probe_floor_c` tokens) at load, before the served request (`code`,
  `src/exec/backend_ov.cpp` Phase B), and those forwards route experts through
  the same provider, so the emitter records them too. They are calibration of
  the slot pool, not the corpus. Decision: the census for the hot-set seed is
  taken over CORPUS calls only, selected by `call_seq >= <the trace's call
  count at the instant the corpus request is posted>`; the harness records
  that `call_seq_start` in its provenance file. The FULL-process census is
  still computed and printed, because patch 0013's CSV counts every call
  (probes included) and the `weight_offset` agreement check must join
  like-for-like; the corpus census and the full-process census are both
  reported with their difference (the probe's share) named.

  **Consequence for the instrument.** `tools/hot_set_census.py` gains
  `census_from_call_trace` (count every call's ids; the `layer,expert,count`
  summary in decoder-layer space), a `weight_offset` join on the RAW
  `layer_key` (patch 0013's key, no export-order assumption for the
  cross-check), and a `--from-call-seq` floor on both the census and
  `from-call-trace`, with red-first cells (the ladder is now **66 cells**,
  was 52; design §7.1 and the patch README carry the new count). The corpus
  itself is the pinned KLD capture's window 0 (2735 token ids, a fixed
  published-in-digest document set) posted as a TOKEN-ID prompt with a greedy
  continuation, one A770 window, the VENICE plugin, kv u8, offload 99 + tier.
  The harness must pass `--no-logits-slice`: a load-time probe forward trips
  the default logits slice and the executor refuses to come up (a harness
  fact recorded when the first attempt of this leg failed at load). The
  cause, found 2026-09-26: the slice assumed token axis 0 (`code`), and the
  serving-shape IR's is 1 (`measured-here` on `qwen3_5_moe`). The load now reads the axis (DESIGN §7.0.2cl),
  measured on `qwen3_5_moe` only; this route's IR has not been loaded with it.
- 2026-09-21: **the served-path trace cell (§7.3) is RUN: one A770 window,
  the corpus census, the raw-key CSV agreement, and the stability statement.**
  [measured-here] Window `venice-census-003`, Arc A770 (`GPU.1`, PCI
  8086:56a0), served depth 48, KV u8, `--offload-ratio 99 --moe-cpu-tier`, the
  VENICE plugin, prefill chunk 512, the pinned KLD capture's window 0 as a
  TOKEN-ID prompt (2735 ids) with `ignore_eos` and a 512-token greedy
  continuation (one request = the prompt plus its continuation). Trace
  25,152 calls, 8,754,436 bytes, sha256
  `553480b169c60d653c9892b72609fb4b22c67ed4a69a4b31141c833ce5702e77`;
  patch 0013's CSV sha256
  `67c70f528fe88699355cb06a82b7849ceaaf0cb360dce4c614dd7d2f55d477f5`;
  harness provenance sha256
  `f0e294a01aa4d40d2b3289f3742e03ec9ba5177a75fc3557e022c07638cd8fd2`;
  `call_seq_start=288`.
  - The trace opens with 576 batched calls (288 load-time probe calls, 288
    corpus prefill calls across 6 chunks) followed by 24,576 decode calls
    (512 tokens x 48 layers). Converted with `--skip-batched` and
    `--from-call-seq 288`: 512 tokens x 48 layers, 24,576 rows.
  - **Corpus census** (derived directly from the call trace, calls >= 288,
    batched included): **1,558,560 routed accesses over 23,813
    `(layer, expert)` cells** = 3247 tokens x 48 layers x top-10 exactly.
    The full-process census is 2,172,960 accesses over 23,925 cells; the
    difference (614,400 accesses) is the load-time plateau probe's share,
    **not** the corpus.
  - **Trace vs patch 0013's four-column CSV, joined on the RAW `layer_key`
    (= `weight_offset`): 23,925 keys, mismatches = 0.** The full-process
    totals agree exactly (2,172,960). Under the corpus floor the join shows
    9,343 keys differing by exactly the probe's share: the CSV counts every
    call (probes included), so the like-for-like join is the full one, and
    the floored difference is the probe, named rather than a nuisance.
  - **Rounds-to-plateau**: NO plateau within the 512 decode tokens at S = 6
    or S = 10 (the selected set changed at every prefix; S = 6 coverage
    0.600 at 1 token -> 0.178 at 512, S = 10 1.000 -> 0.259). That is clause
    V3's failing shape, echoed from the device-free 400-token fixture: the
    aggregate census is no longer sparse, but the decode-row ranking does
    not stabilise in this window. `rounds_to_plateau=None`.
  - **Stability statement**: the A770 served depth-48 path is bit-identical
    x8 (`74bc082`); measured here, the 2735-token prefill census counts are
    **BYTE-IDENTICAL across the two independent A770 processes** of windows
    002 and 003 (23,609 cells, 1,312,800 accesses, `diff` of the data rows
    empty), and the probe census (614,400 accesses) is identical too. The
    counts are run-to-run stable on the A770; no spread is required here (a
    B60 census would carry the two-run spread statement, per the card
    decision above).
  - **Hot-set implication** [measured-here]: at the ratio-99 budget
    (ceil(512 x 1%) = 6 slots/layer) the census's top-6 per layer covers
    **9.36%** of the corpus's routed accesses; S = 10 covers 13.48%, S = 16
    18.63%, S = 32 29.23%, S = 64 44.12%. The seed is deterministic (count
    desc, ascending id; e.g. layer 0: 269, 309, 199, 117, 306, 11). The
    speed row stays EMPTY and G UNPINNED: this leg moves no acceptance row,
    and "frequency beats splitmix64" stays HYPOTHESIS.
  - **Harness facts recorded**: the VENICE prefix needs its own
    `runtime/lib/intel64` **plus** `runtime/3rdparty/tbb/lib` on
    `LD_LIBRARY_PATH` (the libdir alone leaves `libtbb.so.12` unresolved);
    the served load requires `--no-logits-slice` (the load-time probe
    forward trips the default logits slice and the executor refuses to come
    up -- window 001 failed at load for exactly this); and a token-id prompt
    over a document that ends at EOS emits 0 decode tokens, so the
    continuation needs `ignore_eos: true` (window 002 was prefill-only for
    exactly this). Window 002's prefill census remains the stability
    comparand.
- 2026-09-21 (night): **"frequency beats splitmix64" is now MEASURED, not a
  hypothesis — and the REGIME of the calibration corpus decides the size of
  the win.** [measured-here] New instrument `tools/expert_policy_compare.py`
  (+ 33 red-first cells in `tools/test_expert_policy_compare.py`) replays three
  per-layer policies at one budget over window 003's served census: the
  INCUMBENT `static-splitmix64` (patch 0018's rule replicated exactly — seed
  `0xF2A17C0DE5EED`, rank `splitmix64(splitmix64(seed ^ layer_key * 0xD6E8FEB86659FD93) ^ expert)`,
  `layer_key` = the weight-file offset, tie-break on the ASCENDING id), the
  CANDIDATE `static-frequency` (top-`slots` per layer from a calibration
  census, ascending-id tie), and the comparand `lru-per-layer` (demand-warm).
  The frequency seed is calibrated on accesses it is NOT scored on (the tool
  REFUSES an in-sample seed), and the analytic chance baseline `slots/512` is
  printed so "the incumbent is at chance" is checkable rather than asserted.
  - **Protocol A — calibrate on the prefill+probe census (seq < 576), score the
    held-out 512-token decode (245,760 accesses).** At 6 slots/layer (the
    ratio-99 budget): splitmix64 **1.289%** (1.10x chance), frequency
    **3.658%** (3.12x), LRU **0.219%**; at 10: 1.934 / 5.445 / 20.954 (chance
    1.953); at 16: 3.087 / 10.205 / 34.348 (chance 3.125); at 32: 5.747 /
    21.163 / 55.586 (chance 6.250); at 64: 11.946 / 38.525 / 69.164 (chance
    12.500). **splitmix64 sits at chance at EVERY budget** (0.92-1.10x) — what a
    frequency-free seed should do — while the census seed reaches 2.79-3.39x
    chance.
  - **Protocol B — calibrate on the FIRST HALF of the decode and score the
    SECOND half (122,880 accesses), so both sides are the same regime.** At 6
    slots/layer frequency reaches **14.591%** (12.45x chance) against splitmix64
    1.262% and LRU 0.206%; at 10, 21.659% vs 1.923% and 18.127%; at 16, 29.563%
    vs 3.073% and 29.548% — a TIE within 0.02 pt, not a win.
  - **The finding that matters most: the calibration regime decides the win.** A
    prefill-derived census under-predicts decode hotness by ~4x (3.658% vs
    14.591% at 6 slots/layer). The corpus a hot set is seeded from must be the
    regime that will be served; seeding from prefill and serving decode left a
    factor of four on the table. Design rule, not a footnote.
  - **The comparand is not decoration.** The demand-warm LRU — a TRUE LRU,
    promoted on hit — is WORST at the ratio-99 budget (0.219%: it starts cold)
    and BEST above ~10 slots/layer in protocol A (20.954% at 10, 69.164% at 64);
    in protocol B the census seed beats it at 10 (21.659% vs 18.127%) and TIES at
    16 (29.563% vs 29.548%). So the policy answer is budget- and
    protocol-dependent — **static census seed at the tight budget, demand-warm
    LRU once the budget can warm** — and the crossover is a measured number, not
    an assumption.
  - **Convergence (protocol A, S = 6, head bounded to the CALIBRATION window,
    fixed decode tail).** Hit% over heads of 73 -> 576 calls: 3.524, 3.824,
    3.196, 2.619, 2.767, 2.899, 3.422, 3.658 — **NOT monotone**, ending at the
    full-census value (3.658%) only because the last head IS the full prefill
    census. Membership churn is large early (**40, 33, 44 of 48 layers**
    changing) and settles to 9-19 changes per step. So at this head length the
    calibration has NOT converged, which agrees with `rounds_to_plateau=None`
    over the decode rows: the two quantities differ (seed membership vs
    decode-row ranking) and neither has plateaued. An earlier same-night draft
    of this entry reported the series as monotone and nearly converged; that
    series was IN-SAMPLE — its heads grew past the calibration window into the
    scored tail — and is withdrawn here (the tool now bounds every head to
    `[cal_lo, cal_hi)`; external review caught it before the commit).
  - **Instrument corrections, dated in place.** This entry REPLACES the first
    pass of the same night, whose hand-rolled comparand was **FIFO, not LRU**
    (no promotion on a hit) and whose convergence series was in-sample; both were
    caught in external review, which returned **NO-GO** with exactly these two as
    the blockers. The LRU numbers above are the true-LRU rerun, and the incumbent
    replication is now **cross-language verified**: the patch's own
    `static_partition.hpp` was reconstructed from `0018-*.patch`, compiled with
    `g++ -std=c++17`, and its `static_partition_rank_key` /
    `static_partition_resident_experts` printed exactly the values the Python
    cells pin (`rank(seed,0,0)=0xcab2b6579e38a8e3`,
    `resident(lk=704,cap=6)=289,321,337,344,499,509`) — so the baseline is the
    real incumbent, not an approximation of it.
  - **Caveats, stated.** The evaluation windows are 512 decode tokens and do not
    plateau (clause V3's failing shape stands), so these are hit fractions over a
    window, not a converged steady state; window 004 (4,096 decode tokens,
    running) is the longer test; the chance baseline is analytic; and NO served
    throughput is claimed — the speed row stays EMPTY, G stays UNPINNED, and no
    acceptance row moves.

- 2026-09-21 (night) — **the long-decode census window (004) ran, and the
  non-plateau is permanent at this corpus length.** [measured-here] One A770
  (GPU.1, PCI 8086:56a0) served window, tree b6dbca5, the pinned KLD capture's
  window 0 (2,735 token ids) with `ignore_eos` and a 4,096-token greedy
  continuation, `--offload-ratio 99 --moe-cpu-tier`, KV u8, chunk 512. Trace
  18,982,949 B, sha256 `bf32c87401d7ac444aa1aae89eb5a641f69cd84b8c43643f7222253ad5d2d304`,
  197,184 calls; patch-0013 CSV sha256
  `ea8b7a6d244b36afda78cb2cf72441bc7ff42e4a8f38b76857556a20c83a70f8`;
  provenance sha256 `00b070cf18d1561e45f044d35881217971805eb93c8834f8b9f9697c2e3ac791`;
  `call_seq_start=288`. Timing: load 1,880 s; prefill 2,735 tok in 3,646.45 s;
  decode 4,096 tok in 7,114.16 s (0.6 t/s).
  - **Census**: 3,893,280 accesses over 24,151 cells RAW; CORPUS (>=288)
    196,896 calls, 3,278,880 accesses over 24,088 cells = 6,831 x 48 x 10
    exactly; probe share 614,400. Raw-`layer_key` join: 24,151 keys,
    **mismatches 0**, totals both 3,893,280. Corpus-floor join mismatches
    9,343 (the probe).
  - **Plateau**: NO plateau within 4,096 decode tokens at S = 6 or S = 10
    (`rounds_to_plateau=None`, `plateau=False`), the selected set changing at
    every prefix including 2,048 -> 4,096. S = 6 coverage at prefixes
    1,2,4,8,...,4096: 0.600, 0.5625, 0.4771, 0.4781, 0.3949, 0.2780, 0.2363,
    0.2217, 0.1964, 0.1781, 0.1582, 0.1565, 0.1632; S = 10: 1.0, 0.7906,
    0.6734, 0.6737, 0.5566, 0.4008, 0.3478, 0.3257, 0.2856, 0.2590, 0.2310,
    0.2258, 0.2317. **Verdict: V3's failing shape is permanent at this corpus
    length.**
  - **Seed**: corpus layer 0 top-6 = 333, 88, 269, 261, 158, 169 (counts 903,
    823, 802, 740, 681, 659), which differs from window 003's (269, 309, 199,
    117, 306, 11); aggregate S = 6 coverage 11.37 % (003: 9.36 %). The corpus
    ranking MOVED with an 8x longer decode.
  - **Stability**: the window-004 prefill-only census (23,609 cells,
    1,312,800 accesses) is BYTE-IDENTICAL to window 003's, so the prefill
    counts remain the stable comparand.
  - **Rate**: decode did not degrade with 8x context (0.576 vs 0.534 t/s in
    003), consistent with the host-COMPUTE-bound reading.
  - Teardown clean: no `arcint` process, units inactive as found, wake lock
    untouched. The speed row stays EMPTY and G UNPINNED.

- 2026-09-22 — **the acceptance document's convergence row is filled (V3
  firing), the seed-implication coverage is recorded, and the remaining rows
  stay EMPTY for named reasons.** [documented fill; the coverage values are
  `measured-here`, re-derived from the window-004 corpus census CSV rather
  than copied]
  - **`docs/window-052.md` convergence row, FILLED.** `rounds_to_plateau=None`,
    `plateau=False` at S = 6 and S = 10, at both 512 decode tokens (window
    003) and 4,096 decode tokens (window 004); the selected set changed at
    every power-of-two prefix including 2,048 → 4,096. **V3 FIRES
    (2026-09-22)** — a failing measurement is still a measurement, so the row
    carries the value and the clause, not EMPTY. This does not change the
    campaign's status: the non-plateau is permanent at this corpus length.
  - **seed implication, ADDED as an input row, FILLED.** On the window-004
    CORPUS census (`w004.corpus-census.csv`, 3,278,880 accesses over 24,088
    cells = 6,831 x 48 x 10 exactly) the census top-S per layer covers
    S = 6 **11.3721 %**, S = 10 **16.2245 %**, S = 16 **22.1925 %**,
    S = 32 **33.8386 %**, S = 64 **49.1955 %** of the corpus's routed
    accesses. Analytic chance (`S/512`) is 1.1719 / 1.9531 / 3.1250 /
    6.2500 / 12.5000 %, so the census seed clears chance by 9.70 / 8.31 /
    7.10 / 5.41 / 3.94x. Command:
    `python3 - <<'PY' ...` over `w004.corpus-census.csv`, grouping by layer,
    sorting `(-count, expert)`, summing the top S; the script and its raw
    output are recorded in the session log. The incumbent `splitmix64` seed
    remains at chance at every budget (0.92–1.10x, `expert_policy_compare.py`)
    — this row is the seed's implied coverage, not a served result.
  - **STILL EMPTY.** (a) *speed* — HELD for `sub4bit-vram-kernel` step 3, G
    UNPINNED, no host-compute-tier speed measurement taken; (b) *stale-byte
    zero* — blocked on the engine-side host/card readback that does not
    exist, not asserted from code; (c) *quality under policy* — to be
    measured in this same session on the census-seeded served path (incumbent
    vs census seed, greedy digest), no earlier value exists; (d) *verdict* —
    REPORT ONLY until the tag.
  - **No policy code is claimed by this entry.** The seed-implication row is
    a census statistic; the seed is not yet consumed by the served static
    partition in this entry.

- 2026-09-22 — **the census seed is wired into the served static partition,
  and the quality row is MEASURED: PASS, no V4.** [code + measured-here,
  one A770 window]
  - **The seed path (patch 0046).** New file `census_seed.hpp` (OpenVINO-free
    parser for "hot-set seed v2", one `<layer_key> <expert> <expert> ...`
    line per layer, MANDATORY `# space=layer_key` header; malformed lines,
    duplicate keys/ids, missing/wrong space header and an empty file are
    REFUSED; `census_seed_resident_experts()` refuses an absent `layer_key`, a
    slot-count mismatch and an out-of-range expert). `expert_weight_providers`
    gains `set_census_seed()`/`census_seed_active()` and `bind()` pins the
    census set when active, else patch 0018's splitmix64 rank. The impl
    constructor reads `MOE_CPU_TIER_SEED=<path>` once per process (cached
    across the 48 layers), validates THIS layer's entry at construction
    (mismatch refuses the load), and logs
    `seed_source=census|census_seed_fp=0x...`. With the env var unset, patch
    0018 is unchanged. `tools/hot_set_census.py select` emits the v2 format
    from `--census <layer,expert,count CSV>` (the CORPUS census) with
    `--layer-keys <JSON decoder-index -> layer_key>`; a map-less seed declares
    `# space=layer` and the plugin parser refuses it. Files: `patches/0046-moe-cpu-tier-census-seed.patch`
    (+ mirrored in `contrib/packaging/marfrit-openvino/patches/` and its
    README), `tools/hot_set_census.py`, `tools/test_hot_set_census.py`,
    `tools/test_census_seed.py`.
  - **Cells.** `tools/test_census_seed.py`: **14 green** (extract
    `census_seed.hpp` from the patch, compile with plain g++, run one driver
    case per refusal/success). Each refusal was shown RED with its check
    removed. `tools/test_hot_set_census.py`: **76 green** (was 66), including
    the v2 seed emission, the census-summary parser and the `select --census`
    CLI. Device-free, no card. Raw output in the session's
    `cells-green.txt` / `redfirst-mutation1.txt` on the persistent census path.
  - **Build, measured (device-free).** Patch 0046 applied on top of patches
    0003–0045 against pin `71640275`; `ninja openvino_intel_gpu_plugin` clean;
    the plugin carries `MOE_CPU_TIER_SEED` / `seed_source=` /
    `census seed: layer_key` strings. Plugin sha256 prefix `d72c00bfc341e398`.
  - **Quality row, MEASURED (A770, GPU.1, PCI 8086:56a0).** Native d48n
    artifact, `--offload-ratio 99 --moe-cpu-tier`, KV u8, chunk 512, one fresh
    process per arm, the SAME 256-token prompt (pinned capture window 0) and
    greedy `max_tokens` 32, temperature 0. Incumbent (`MOE_CPU_TIER_SEED`
    unset; `seed_source=splitmix64`) and census seed
    (`MOE_CPU_TIER_SEED` at the corpus S = 5 seed; `seed_source=census`) both produced
    greedy text sha256 **`2169836b33e8bc74d7965fff867b13c1d3637388a4b52f11f639f381ce7cc36f`**
    — **byte-identical**. The design's expectation holds: under the native
    artifact every routed expert runs on the host tier (patch 0043), so
    residency moves bytes, not arithmetic. **No V4.**
  - **Red-first refusal, measured on the card.** The corpus S = 6 seed was
    run first and refused the load: `census seed: layer_key 284636629 lists 6
    experts but the pool has 5 slots (mismatched budget)`; the process never
    became ready. The device-free cells cover the same class; this is the
    on-hardware instance.
  - **Measured correction (a finding, `code` + `measured-here`).** The
    plugin's actual pool at `--offload-ratio 99` is **5 slots/layer**:
    `prepare_moe_otd_params` uses integer division `512*(100-99)/100 = 5`,
    while the engine's own ledger (`src/exec/fit.h`, `expert_slot_bytes` /
    `expert_slot_bytes_static`) prices `ceil(...) = 6`. The OTD_PERF lines from
    both served arms read `slots=5`. So the campaign's `S = 6` offline
    analysis is one slot larger than the pool the plugin actually pins; the
    served seed is the corpus top-5 (coverage **9.96 %**). This is recorded,
    not smoothed: the offline S = 6…64 coverage numbers in `docs/window-052.md`
    are the engine-priced analysis. The served-pool/ledger off-by-one is open.
  - **Rows.** `docs/window-052.md` now carries: convergence (V3 firing),
    seed implication (with the off-by-one correction), and quality (PASS, no
    V4). STILL EMPTY: *speed* (HELD for `sub4bit-vram-kernel` step 3, G
    UNPINNED; no host-compute-tier speed measurement taken), *stale-byte zero*
    (no engine-side host/card readback exists; not asserted from code),
    *verdict* (REPORT ONLY until the tag). The campaign stays OPEN: the
    resident-compute path that would make the policy pay is still the named
    dependency.

- 2026-09-22 (review) — **external review of the seed-wiring commit; one
  blocker fixed, the rest recorded as carry-forward.** [documented]
  - **Blocker, fixed:** the patch record in
    `contrib/packaging/marfrit-openvino/patches/README.md` still ended
    "OWED-until-run: the served A770 quality row" while the same change set
    records that row as MEASURED/PASS. The OWED sentence is replaced with the
    measured digest pair and the mismatch refusal.
  - **Fixed with it:** the served top-5 coverage (9.96 %) now carries the
    reproducible command and raw output in the session's `quality-report.txt`;
    the CHANGELOG's `+p19` line now discloses that `p19` is also the 0003–0043
    stamp; the design note's "emitted in both spaces" and "consumed at
    `bind()`" sentences are corrected (the env is read and validated at
    construction, applied at `bind()`); `seed_text` now REFUSES a
    `layer_key` map with a duplicate value instead of writing a file the
    plugin would refuse at load (one new Python cell: 76 green, was 75).
  - **Carry-forward, deliberately NOT fixed in the plugin patch.** The
    reviewer flagged two source-level items: `census_seed_active()` has no
    caller in the patch, and `cached_census_seed()`'s function-local static
    map is unsynchronized (safe only because provider construction is
    single-threaded today). Both are behaviour-neutral, but editing the patch
    now would change the source that produced the measured plugin
    (`d72c00bf`) without re-running the card, breaking the measured-artifact
    match. They are recorded as open rather than patched-and-unmeasured; a
    later leg that re-runs the window may clean them in the same commit as the
    measurement.
  - **Also carry-forward:** `resident_slot_count()` and `_capacity` are equal
    by construction (`resident_slot_count()` returns
    `_weight_provider->resident_capacity()`, which returns `_capacity`), so
    the validation/bind pairing is not a divergence; stated here because the
    reviewer could not see the provider source.

- 2026-09-22 (G pin, PREDICTION commit) — **G is PINNED at 1.10 before any
  speed measurement; the speed row's HOLD is discharged.** [documented pin;
  `measured-here` inputs, `derived` ceiling] The native per-expert OpenCL
  decode (patches 0043/0045) serves at the 0045+0047 prefix `ov-0047` on both
  cards, so the speed leg this campaign HELD is runnable. The gate `warm-up
  decode ≥ G × host-bound baseline` now has its G, pinned in
  `docs/window-052.md`: census-measured hit fraction `h₉₉ = 9.9595 %`
  (window-004 CORPUS census top-5 at the plugin's measured 5 slots/layer —
  also transcribed here, S = 5; the in-sample seed is disclosed against the
  held-out decode-regime value 14.59 % at S = 6), and the one existing
  resident-compute datum (B60, ratio 75, 0047: `per_expert_gpu_invocations`
  135,874 → 67,937 card pairs vs `cpu_tier_pairs` 187,903, `h₇₅ = 0.2655`,
  decode 0.5672 t/s). That datum is a phase MIXTURE (255,840 pairs = 533
  tokens × 48 × 10, against a 21-token served request: the load probe and the
  activation ladder dominate), so it is an observation (no win at
  `h₇₅ ≈ 0.27`) and NOT a fit. The assumption-free ceiling is
  `1/(1 − h₉₉) = 1.1106`, so the pinned gate is **G = 1.10** and the
  prediction is a shortfall (V1) with the hot-set *correctly engaged*. A
  measured decode at or above the gate overturns the prediction and G is
  corrected in place with the measurement's date.

- 2026-09-22 (rate leg, cross-reference) — **the speed leg ran: V1 at the
  ratio-99 VENICE budget, and the dispatch route shows a §3.4/V4
  answer-dependence.** [measured-here] A770, native `d48n`, plugin `ov-0047`,
  ratio 99: census S5 **0.556 t/s** against the same-day host-tier comparand
  0.526 (< the pinned 0.579), B60 0.555 against 0.88 — **V1**; the speed row
  stays EMPTY. The residency sweep puts the win at ratio 75: census top-128
  **0.842 t/s** against the same-config host control **0.465 t/s** = 1.81×
  (`ρ = 0.073 [derived]`). **The quality row's PASS was measured without
  `--moe-per-expert-dispatch`** (every routed expert on the host tier,
  residency moves bytes not arithmetic) and therefore does not cover the
  dispatch route; there the greedy answer changes with the resident seed
  (splitmix64 `55dff6f2…` vs census `2e7c508f…` at ratio 99), so the policy is
  **visible** on that route: **V4 FIRES (RED)** and the dispatch-route quality
  is **OPEN**, not PASS. Full raw evidence and the §3.4 finding:
  `sub4bit-vram-kernel.md`, status 2026-09-22 (rate leg); DESIGN §7.0.2ce.
- 2026-09-22 (late) — **the ratio-99 shortfall is a REGIME shortfall, not a policy
  failure: the same seed covers 4.11% of PREFILL and 13.86% of DECODE, and the
  served request was 80% prefill.** [measured-here] Re-derived in one pass from
  window 004's own trace (the corpus top-5 seed fixed, four views of the same
  call trace):
  | view | coverage of the S=5 corpus seed |
  |---|---|
  | corpus census (2,735 prefill + 4,096 decode = 6,831 tok) | 326,559/3,278,880 = **9.96%** |
  | prefill only (2,735 tok) | 54,024/1,312,800 = **4.11%** |
  | decode only (4,096 tok) | 272,535/1,966,080 = **13.86%** |
  | decode, first 64 tokens | 2,026/30,720 = **6.60%** |
  | prefill + the first 250 decode tokens (2,985 tok) | 63,751/1,432,800 = **4.45%** |
  | prefill + the first 314 decode tokens (3,049 tok) | 65,777/1,463,520 = **4.49%** |

  **Why the last two rows are not "the request":** the rate leg served a
  **256-token prompt + 64 greedy tokens**, but this trace CANNOT be sliced
  per-token through the prefill — the emitter's prefill calls are BATCHED (one
  call per 512-token chunk, with that chunk's ids aggregated into one row), so a
  range expressed in CALLS is not a range expressed in TOKENS. The two rows above
  are the closest views the trace admits, and both are prefill-dominated, landing
  at 4.4–4.5%; that is the band the served arm's realized 3.77% sits in.

  So the ratio-99 card share of 3.77% is neither a defect nor a mis-set: a request
  that is 80% prefill sees the PREFILL coverage (~4.1%), and the residual to
  3.77% is attributed — as a HYPOTHESIS, not a measurement — to the load-time
  probe's own mix plus the served counters' accounting (the probe's accesses enter
  both hits and misses). **The corpus's 9.96% is a MIXTURE average over a
  decode-heavy corpus and must never be quoted as what a prefill-heavy request
  will realize.**
  - This is the SAME regime law the policy comparison established from the other
    direction (2026-09-21 entry: a prefill-derived census under-predicts DECODE
    hotness ~4x). Coverage is not a property of the seed; it is a property of the
    **(seed x regime)** pair. The acceptance text therefore carries a coverage
    PAIR (prefill, decode), or one figure for a STATED mix — never a single
    number, which is what `docs/window-052.md`'s seed-implication row now says.
  - **Consequence for the gate (named, not decided here):** the ratio-99 V1 is
    structural FOR A PREFILL-HEAVY REQUEST. The lever is therefore not more slots
    but the right calibration — seed per regime (two sets, or a set chosen by the
    request's prefill/decode shape), or calibrate on the serving mix. Nothing here
    moves the pinned G or the V1/V4 records; it explains WHY V1 fired and names
    the lever.
  - **Per-layer texture:** the request's coverage ranges from **1.3%**
    (layer_key 9143804121) to **20.6%** (73272898913) across the 48 layers, so the
    shortfall is per-layer, not a single global deficit — a per-layer calibration
    problem, not a budget problem.
- 2026-09-22 (late, second) — **regime-matched calibration PAYS: a prefill-seeded
  S=5 set doubles the prefill coverage, and the seed we ship is decode-leaning.**
  [measured-here] Same trace (window 004), same 5-slot budget, three calibration
  sources, coverage measured on BOTH regimes:
  | seed calibrated on | coverage on PREFILL | coverage on DECODE |
  |---|---|---|
  | mixture (2,735 prefill + 4,096 decode) — what we serve today | 4.11% | 13.86% |
  | **prefill only (2,735 tokens)** | **8.11%** | 5.36% |
  | decode only (4,096 tokens) | 2.86% | 14.34% |
  - **The shipped seed is DECODE-leaning**: it agrees with the decode seed on
    **18 of 48 layers** and with the prefill seed on **0 of 48** — exactly what a
    mixture dominated by decode tokens should produce.
  - **The lever has a measured size**: for a prefill-heavy serving shape,
    calibrating on the prefill census buys **1.97x the prefill coverage at the
    SAME 5-slot budget**, with no extra VRAM; the price is decode coverage
    (13.86 -> 5.36%), which the serving mix must decide.
  - This is the same law as the 2026-09-21 policy comparand (a prefill-derived
    census under-predicts DECODE hotness ~4x) and the regime entry above, now
    quantified in the direction that matters for the served workload: coverage is
    a property of the **(seed x regime)** pair, so the seed is calibrated on the
    mix you serve.
  - Caveats, stated: this is coverage over one window, not a converged steady
    state (the decode ranking does not plateau; V3 fires); it is a device-free
    replay of the trace, NOT a served measurement; and it moves no acceptance row
    — the pinned G and the V1/V4 records are untouched. The three seeds disagree
    per layer (layer_key 284636629: mixture [88,158,261,269,333], prefill
    [11,117,199,269,306], decode [88,169,261,333,439]), so this is a different
    membership, not a tie-break effect.
- 2026-09-22 (late, third) — **the regime-calibrated seed is now EMITTABLE with the
  tool, not merely measurable.** [code + measured-here]
  `census-from-call-trace` gained an EXCLUSIVE ceiling (`--to-call-seq`) beside its
  floor, so a REGIME is a half-open call-seq range, and `join-plugin-call-trace`
  takes the same range; an inverted range or one that selects NO call is REFUSED
  (a census that selects nothing is not a census). The recipe, run end-to-end on
  window 004 with the shipped tool:
  1. `census-from-call-trace --call-trace … --from-call-seq 288 --to-call-seq 576`
     -> **23,609 cells, 1,312,800 accesses over 288 batched prefill calls** — the
     prefill regime, matching the prefill census already on the record.
  2. `select --census <that> --slots-per-layer 5 --layer-keys <map>` ->
     `venice-seed-prefill-S5.txt`, a v2 seed keyed by the structural `layer_key`.
  The prefill seed differs from the shipped mixture seed on **all 48 layers**
  (96 differing data rows out of 48+48), so the lever is a different membership
  rather than a tie-break. +5 cells (`tools/test_hot_set_census.py`, now 81 census
  cells; 134 tests green across the four ladders). This makes the measured V1
  lever actionable — whether to serve the prefill- or mixture-calibrated seed is
  still the operator's decision, and it is now a one-line flag, not a new tool.

- 2026-09-22 (V4 quantification leg, cross-reference) — **the dispatch route's
  quality row is scoped out of the PASS, and its divergence is now quantified.**
  [measured-here] This campaign's quality row (PASS, no V4) was measured
  WITHOUT `--moe-per-expert-dispatch`: every routed expert ran on the host
  tier, so residency moved bytes, not arithmetic, and the census seed was
  invisible. Under `--moe-per-expert-dispatch` the served greedy answer
  DEPENDS on the resident seed, so **V4 fires on that route** and its quality
  is OPEN, not PASS. The V4 leg quantified it: the two ratio-99 answers branch
  at greedy token index 3 of 64 (61 of 64 token positions then differ), each
  seed reproduces its own digest (incumbent `55dff6f2…`, census `2e7c508f…`,
  both repeated), and a one-layer native MoE block measures the card-vs-host
  arithmetic directly — **affine per-expert dispatch is bit-identical to the
  host tier; native per-expert dispatch is not** (12.5 / 37.5 / 75.0 % of
  output elements move as the resident fraction grows; max |diff| up to
  1.1e-2; deterministic and card-independent). DESIGN §7.0.2cf,
  `sub4bit-vram-kernel` status 2026-09-22 (V4 leg). **No acceptance row moves
  here**: the speed row stays EMPTY, G stays pinned at 1.10, V1 stands, and the
  ratio-75 point (0.842 t/s, 1.81×) stays a sweep point, not a gate. The scope
  of that sweep point is undecided pending the V4 answer.

- 2026-09-22 (slot off-by-one, CLOSED as intentional) — **the plugin's 5-slot
  pool at ratio 99 is the SERVED TRUTH.** [measured-here + code, operator
  decision] The plugin's integer division (`512*(100-ratio)/100` = 5) sizes
  every served reading in this campaign, including the S = 5 census seed; the
  engine's `fit.h` `ceil` (= 6) is a **fit-side ledger ceiling only**, never a
  plugin buffer size (the ratio-75 run showed the two agree there and the
  pre-0047 fault still reproduced, then patch 0047's resident-sized pool fixed
  it). CLOSED as an intentional divergence: no plugin change to `ceil`, no
  ledger change to integer division, no code movement, and no future session
  should re-open it.
