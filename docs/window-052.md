# window-052 — 0.5.2 VENICE acceptance (the BERLIN→VENICE gate)

Drafted 2026-09-19 as the acceptance commit for VENICE-001, BEFORE any
hot-set/LRU code, per `ROADMAP-0.5.x.local.md`'s law: *the first commit is
the acceptance criteria, with the measured rows EMPTY; prediction commits
precede measurement.* Every row below is EMPTY until measured; no number in
the tables is a result.

## Feature (roadmap 0.5.2)

Card hot-set + host LRU for expert slots — FreeToken's smart half. Hot-set
selection from an expert-access census; eviction/refresh discipline.

## Entry criteria

- **BERLIN closed.** [CORRECTED 2026-09-20: `F_served` at depth 48 **is**
  measured — window-051 clause (d) carries 0.1361/0.1512 on the B60 and 0 on
  the A770, where the row reads READABLE — so this criterion is MET, not
  pending.] The roadmap's falsifiable late-layer question instantiated and run
  (it was never designed).
- **The census instrument exists** — a per-token routed-expert histogram;
  its storage and format are this campaign's first deliverable.
- **G pinned from BERLIN's OWN measured rates, not invented:** [CORRECTED
  2026-09-20: the cited **23.6 t/s** is the **HF-exported 35B control**
  (`campaigns/serving-shape-logits.md:30`), not a d48g decode rate — no d48g
  decode rate is on the record, and the 48-layer tier measures 0.5-0.8 t/s.
  Re-pin G from a measured d48 rate before stating it.] The native
  scalar-tier route is 0.6 t/s (`d48n`) [2026-09-21: that 0.6 t/s point lies
  inside the measured 0.5–0.8 t/s band — it is not a third baseline]; G is a
  multiple of a host-bound
  baseline and is
  stated here before measuring.
  [CORRECTED 2026-09-21: the host-bound baseline is now FIXED at the measured
  `d48n` host-tier rate, **0.5–0.8 t/s** (B60, ratio 99 + tier, KV u8, f16,
  chunk 512; `measured-here`). **G itself stays UNPINNED**, and the speed row
  stays EMPTY rather than PASS, because the native artifact runs every routed
  expert on the scalar host tier (patch 0043: the fused kernels refuse the
  native formats), so residency alone moves no compute. G is pinned in a dated
  prediction commit before the speed leg, once a measured resident-compute rate
  exists — named dependency `sub4bit-vram-kernel` step 3, the OpenCL decode.
  Campaign: `docs/campaigns/expert-hot-set-lru.md`. No measured row is filled
  here.]
  [HELD 2026-09-21, operator decision: **VENICE's speed leg waits for
  `sub4bit-vram-kernel` step 3, the OpenCL decode.** The census instrument
  and the policy land on the host-tier path first; the speed row stays EMPTY
  and G stays UNPINNED until the resident-compute patch exists. No speed
  window is spent on the host-compute tier.]
  [CORRECTED 2026-09-22, PREDICTION: **G is PINNED at 1.10, before any speed
  measurement of this leg.** The dependency the HOLD named is discharged: the
  native per-expert OpenCL decode exists (patches 0043/0045) and the served
  path is the 0045+0047 prefix `ov-0047` (plugin `f021de51b5812ee2`), which
  serves on both cards (`sub4bit-vram-kernel` status 2026-09-22 late
  morning). The pin, its arithmetic and the predicted outcome are in the
  section below. This paragraph supersedes the two `UNPINNED` lines above;
  those stay as written, marked.]

## G pin — the prediction commit, before the speed leg (2026-09-22)

The gate is `warm-up decode ≥ G × the host-bound baseline`. G is pinned from
the measured baseline and the census-measured hit fraction; the one existing
resident-compute datum is a phase mixture and is reported as such, not fitted.

- **baseline** `H`: the `d48n` host-tier decode, **0.5–0.8 t/s** (B60, ratio
  99 + tier, KV u8, f16, chunk 512; `measured-here`, 2026-09-18). The gate is
  read against the band's **upper edge 0.8 t/s**, so a pass cannot be an
  artefact of the friendlier end; the midpoint 0.65 is quoted beside it and
  decides nothing. On the A770 (the rate leg's card) the same-day host-tier
  comparand is **0.526 t/s** `[derived: 32 decode tokens / 60.84 s; measured
  2026-09-22, the quality-window incumbent arm, run without
  `--moe-per-expert-dispatch`]`.
- **served hit fraction** `h`: at the ratio-99 budget the plugin's pool is
  **5 slots/layer** (integer division, `measured-here` 2026-09-22), and the
  window-004 CORPUS census's top-5 per layer covers **9.9595 %** of the
  corpus's routed accesses (`measured-here`; 3,278,880 accesses over 24,088
  nonzero `(layer, expert)` cells; 6,831 tokens × 48 layers × 10 experts).
  `h₉₉ = 0.0996`. **Disclosure:** the seed IS the top-5 of the census it is
  served over (in-sample), so 9.96 % is its realized coverage for that corpus
  by construction; the campaign's held-out protocol B measured the
  decode-regime hit at **14.59 %** for S = 6 (`measured-here`), ~4× the
  prefill-calibrated value. No held-out S = 5 number exists, so the
  held-out value at this budget is unverified; if the served hit were the
  larger 14.59 %, the free-card ceiling would rise with it
  (`1/(1 − 0.1459) = 1.17` `[derived]`) and G = 1.10 would be 94 % of that
  ceiling rather than 99 %. The pin is therefore computed from the smaller
  in-sample hit, which is the stricter of the two readings: it yields the
  lower ceiling and the harder gate.
- **resident-compute rate** `ρ` (card per-pair time ÷ host per-pair time):
  **NOT separately measured, and the gap is stated rather than smoothed.**
  The only counter reading with a card/host split is the B60 ratio-75 run
  under the 0047 plugin (`per_expert_gpu_invocations = 135874` → 67,937 card
  pairs; `cpu_tier_pairs = 187903`; decode 16 tok in 28.21 s = **0.5672 t/s**,
  `measured-here`). Those counters span the whole process: 255,840 pairs =
  533 tokens × 48 layers × 10 experts, while the served request is 21 tokens
  (5 prefill + 16 decode) — so ~96 % of the counted pairs are the load-time
  plateau probe and the activation ladder, a phase mixture whose served-decode
  split is not readable from these counters. Taken at face value under the
  midpoint baseline the mixture implies `ρ ≈ 1.55` `[derived]`; over
  `H ∈ {0.5, 0.8}` it spans 0.55–2.55 `[derived]`, so the point cannot even
  decide the sign of the win. It is used only as an observation: **at
  `h₇₅ ≈ 0.27` the served decode (0.5672 t/s) sat inside the baseline band,
  so no win is demonstrated at that residency.**

Model: `R(h) = H / (1 − h·(1 − ρ))`. With `ρ` unmeasured, the only
assumption-free statement is the free-card ceiling `ρ = 0`:

    R₉₉ ≤ H / (1 − h₉₉) = H / 0.9004  →  G_max = 1.1106   [derived]

**The pinned gate is G = 1.10** (the free-card ceiling at the in-sample hit
`h₉₉ = 9.96 %`, rounded down from 1.1106). It requires the served decode to
reach 99 % of the speedup a zero-cost card could buy at that hit, and the
ratio is invariant across the baseline band (the baseline cancels).
Thresholds: B60 `≥ 1.10 × 0.8 =` **0.88 t/s**; A770
`≥ 1.10 × 0.526 =` **0.579 t/s** `[derived]`. The prediction is that
the gate will **NOT** be met and clause V1 fires: no measured point
demonstrates a card/host per-pair win (the ratio-75 observation), and the
ceiling itself is only 1.1106, so a realized win at `h₉₉ = 9.96 %` is at
best 11 %. The prediction is falsifiable — a measured decode at or above the
gate overturns it, and G is then corrected in place with the measurement's
date. **A V1 firing with the hot-set correctly engaged (the census seed
serving at `seed_source=census`, its top-5 pinned) is the predicted shape;
V1's "the hot-set did not engage" branch is not implied.**

[CORRECTED 2026-09-22 (measured), after the rate leg: the pin's `ρ ≈ 1.55`
was drawn from the ratio-75 counters *before* their phase composition was
attributed; the served-only measurement reverses it. The A770 ratio-75
ledger-hit arm (census top-128, `h = 0.4836`) decodes **0.842 t/s** against
the same-config host control **0.465 t/s**, which solves
`ρ = 1 − (1 − H/R)/h` on the unrounded rates to **ρ = 0.073 `[derived]`** —
the card computes a resident expert pair ~13× faster than the host. The pin
`G = 1.10` and the predicted verdict stand; only the reason is corrected:
the ratio-99 budget fails because just **3.77 %** of its served pairs are
resident (not the corpus's 9.96 % top-5 coverage), whose free-card ceiling
(`ρ = 0`) is `1/(1 − 0.0377) = 1.039`, so no 1.10× win is reachable there.
At ratio 75 the fit reaches 1.81× against a free-card ceiling of 1.936. The
resident-compute rate the gate wanted is therefore **ρ = 0.073 `[derived]`**,
no longer "not separately measured"; the ratio-75 counters' phase mixture is
the reason the prediction commit could not see it. **New §3.4/V4 finding,
V4 FIRES (RED) (same leg):** on the dispatch route the served answer depends
on the resident seed (splitmix64 `55dff6f2…` vs census `2e7c508f…` at
ratio 99), because the GPU per-expert kernel and the host tier are not
bit-identical; recorded in `docs/campaigns/sub4bit-vram-kernel.md`.]

## The bar in force

`bar_0.5.1 = 100 × F_ref` = **3.0905e-03 nats below row 2051 /
2.6946e-02 at or above it** (`docs/window-051.md` clause (d), BERLIN-001
`5d4dd59`; `F_ref` = the capture's uint16 reconstruction error =
3.0905e-05 nats mean). PROVISIONAL on one caveat: the rows are
f16-served. The inherited 0.0599 is SUPERSEDED BY LINEAGE and decides
nothing here. Any KL reading must print `F_served` beside it; a bar below
`F_served` is UNREADABLE, not PASS.

[DATED 2026-09-28: the "above 2051" bar 2.6946e-02 adds a max-abs block-output difference (the QSA price) to a KL bar in nats; the units do not mix and the bar is **withdrawn**. The served KL above 2051 stays owed as a measurement: `docs/campaigns/qsa.md`.]

## Acceptance rows (EMPTY until measured)

| quantity | predicted | measured |
|---|---|---|
| warm-up decode vs the host-bound baseline | ≥ G × (measured d48n 0.5–0.8 t/s), **G = 1.10** (pinned 2026-09-22, PREDICTION; see Entry criteria and the G-pin section). Thresholds: B60 **< 0.88 t/s** fails (1.10 × the band's upper edge 0.8), A770 **< 0.579 t/s** fails (1.10 × the same-day host-tier comparand 0.526) | **MEASURED — V1 FIRES (2026-09-22), speed row NOT filled.** `measured-here`, one fresh process per arm, native `d48n`, plugin `ov-0047`, KV u8, one lane, the capture window-0's first 256 ids, greedy 64, temperature 0: at the ratio-99 VENICE budget the census-seeded resident route is **0.556 t/s** on the A770 (splitmix64 seed 0.547; same-day host-tier comparand 0.526) — **< 0.579 t/s** — and **0.555 t/s** on the B60 — **< 0.88 t/s**. The shortfall is recorded here and the row is not filled. The residency sweep puts the win at ratio 75: census top-128 gives **0.842 t/s** against the same-config host control **0.465 t/s** = **1.81×** (`ρ = 0.073 [derived]`). Raw logs in `docs/campaigns/sub4bit-vram-kernel.md`, status 2026-09-22 (rate leg). **Scope caveat:** the same leg measured a §3.4 answer-dependence on the dispatch route (the served answer changes with the resident seed); the quality row below was measured on the non-dispatch path and does not cover it. |
| stale-byte zero proof | digest(host-bound bytes of expert E) == digest(card-bound bytes of the same E), every E in the hot set | EMPTY |
| convergence | rounds-to-plateau printed with the census | **MEASURED 2026-09-22** (`measured-here`): `rounds_to_plateau = None`, `plateau = False` at **S = 6 and S = 10**, at both **512 decode tokens** (window 003) and **4,096 decode tokens** (window 004); the selected resident set changed at every power-of-two prefix including 2,048 → 4,096. **V3 FIRES (2026-09-22):** the census does not plateau within the rounds this corpus admits, so the policy is not converging at this corpus length. Raw series in `docs/campaigns/expert-hot-set-lru.md` (status, 2026-09-21 night) and the window-004 plateau JSONs. |
| seed implication at the ratio-99 budget | coverage of the corpus's routed accesses by the census-seeded static partition (input to the seed decision; not a V-clause) | **MEASURED 2026-09-22** (`measured-here`): window 004 corpus census (3,278,880 accesses, 6,831 × 48 × 10): S = 6 **11.37 %**, S = 10 **16.22 %**, S = 16 **22.19 %**, S = 32 **33.84 %**, S = 64 **49.20 %**; analytic chance S/512 = 1.17 / 1.95 / 3.13 / 6.25 / 12.50 %. The incumbent `splitmix64` seed sits at chance at every budget (0.92–1.10× `slots/512`), so the census seed reaches 9.70 / 8.31 / 7.10 / 5.41 / 3.94× chance here. Coverage is a hit fraction over a non-plateauing window, not a converged steady state. **Measured correction (2026-09-22):** the plugin's actual pool at ``--offload-ratio 99`` is **5 slots/layer**, not 6 — `prepare_moe_otd_params` uses integer division `512*(100-99)/100 = 5`, while the engine's own ledger (`src/exec/fit.h`) prices `ceil(...) = 6`. The S = 6 seed is therefore REFUSED at load (measured: `census seed: layer_key … lists 6 experts but the pool has 5 slots (mismatched budget)`), and the served seed uses the corpus top-5 (coverage **9.96 %**, computed by grouping the corpus census CSV by layer, sorting `(-count, expert)` and summing the top 5: raw output `S=5: 9.9595%`, `S=6: 11.3721%`, in the session's `quality-report.txt`). The S = 6…64 numbers above are the engine-priced offline analysis; the served pool is one slot smaller. [DATED IN PLACE 2026-09-22 (operator decision, V4 leg): this divergence is **CLOSED as intentional**. The plugin's integer division (`512*(100-ratio)/100` = **5** slots/layer at ratio 99) is the **served truth** — every served reading is taken against that pool. The engine's `fit.h` `ceil` (= 6) is a **fit-side ledger ceiling only**, never a plugin buffer size, as the ratio-75 run already showed (both formulae give 128 there and the pre-0047 fault reproduced unchanged, then patch 0047's resident-sized pool removed it). No plugin change to `ceil`, no ledger change to integer division, no code movement; a later session must not "fix" it.] **Regime correction (2026-09-22, `measured-here`):** coverage is not one number per budget. The corpus's **9.96 %** is a MIXTURE average over a decode-heavy corpus (4,096 decode vs 2,735 prefill tokens), and the SAME S = 5 seed covers **4.11 % of the prefill** (54,024/1,312,800) and **13.86 % of the decode** (272,535/1,966,080); the rate leg's request was 80 % prefill, and a prefill-dominated view of the trace (prefill + the first 314 decode tokens, 3,049 tokens) realizes **4.49 %** (65,777/1,463,520) — which is the band that arm's ratio-99 card share of **3.77 %** sits in. NOTE: the trace cannot be sliced per-token through the prefill (the emitter's prefill calls are BATCHED per 512-token chunk), so the request's own 256+64 shape is not directly extractable; the residual from 4.49 % to 3.77 % is a HYPOTHESIS (the load-time probe's mix plus the served counters' accounting), not a measurement. Quote coverage as a **(prefill, decode) pair** or for a stated mix, never as a single figure. Raw views and the per-layer spread are in `docs/campaigns/expert-hot-set-lru.md` (2026-09-22 late entry). |
| quality under policy | no greedy digest change vs the pre-policy served answer | **MEASURED 2026-09-22** (`measured-here`): **PASS, no V4.** On the A770 (GPU.1, PCI 8086:56a0), native d48n artifact, `--offload-ratio 99 --moe-cpu-tier`, KV u8, chunk 512, one served window per arm, same 256-token prompt and greedy 32 tokens (temperature 0): incumbent `splitmix64` seed → greedy text sha256 `2169836b33e8bc74d7965fff867b13c1d3637388a4b52f11f639f381ce7cc36f`; corpus census seed (S = 5, the plugin's measured pool) → the **same** `2169836b…336f`. Byte-identical greedy output, so the census-seeded static partition is not visible under DESIGN §3.4. Plugin for the window: VENICE tree + patches 0003–0045 + 0046 (`d72c00bf…`); raw evidence in the session's `quality-report.txt` on the persistent census path. **[SCOPE 2026-09-22: this PASS was measured WITHOUT `--moe-per-expert-dispatch`, so every routed expert ran on the host tier and residency moved bytes, not arithmetic. It does NOT cover the dispatch route: there the 2026-09-22 rate leg measured a §3.4 answer-dependence by resident seed (splitmix64 `55dff6f2…` vs census `2e7c508f…`), so **V4 FIRES (RED) on that route** and its quality is OPEN, not PASS (`docs/campaigns/sub4bit-vram-kernel.md`).] [DATED IN PLACE 2026-09-22 (V4 quantification leg): the dispatch route's divergence is quantified — the answer branches at greedy token index 3 of 64 (61 of 64 token positions differ), each seed reproduces its digest, and the card-vs-host numeric arms read: affine per-expert dispatch **bit-identical** (0 elements moved), native per-expert dispatch **not** (12.5 / 37.5 / 75.0 % of output elements move as the resident fraction grows to 1 / 16 / 32 slots, max |diff| 8.3e-3…1.1e-2, mean |diff| 1.8e-4…1.1e-3 against a reference rms of 3.55e-3), deterministic and card-independent. DESIGN §7.0.2cf. The quality row stays PASS only for the non-dispatch path; the dispatch route's quality stays OPEN.]
| verdict | REPORT ONLY until the tag | EMPTY |

## Falsifiable clauses (each can fail)

- **V1** — if warm-up decode < G × baseline, the hot-set did not engage (or
  the baseline moved, or the resident-compute path has no per-pair win). The
  parenthesis names causes, not the only cause: a shortfall with the hot-set
  correctly engaged is V1 too. The row says so, not PASS.
- **V2** — if any expert's host-bound and card-bound bytes differ by digest,
  eviction/refresh is stale: RED, not PASS.
- **V3** — if the census does not plateau within the rounds the prediction
  states, the policy is not converging: the row names the rounds it took.
- **V4** — if any served answer's greeddy digest changes under the policy,
  the policy is visible: RED per DESIGN §3.4 (history-independent greedy
  output).

## Out of scope

- No FreeToken comparison here — that is GENEVA (0.5.8) and is pinned only
  by our own runs.
- No NVMe path — that is LISBON (0.5.3), blocked on the ext4 expert store.
- No segmented-chain compile; that route is `sub4bit-vram-kernel` +
  window-051 §2.

## Where it lives

`docs/window-052.md`. Candidate dependencies named in the campaign index:
`docs/campaigns/static-partition-prefill.md`,
`docs/campaigns/partition-seeding.md`,
`docs/campaigns/sub4bit-vram-kernel.md`.

## Status

- 2026-09-19: drafted as VENICE-001's acceptance commit; every measured row
  EMPTY. **NOT committed** — `CLAUDE.md` requires Fable review before every
  commit, and the tag path is the operator's.
- 2026-09-21: registered the campaign `docs/campaigns/expert-hot-set-lru.md`
  (row added to `docs/campaigns/README.md`) and the census-instrument design
  `docs/design-expert-hot-set-lru.md`. Corrected the host-bound baseline in
  place to the measured `d48n` rate (0.5–0.8 t/s) and marked **G UNPINNED**
  until a resident-compute path exists (patch 0043 runs every native-format
  expert on the host tier). This document was committed as `eaa7a06`
  (2026-09-19) — the 2026-09-19 entry's "NOT committed" is superseded by
  that commit, recorded here rather than rewritten. **No measured row filled;
  no policy code.**
- 2026-09-21: operator decision recorded — **VENICE's speed leg is HELD for
  `sub4bit-vram-kernel` step 3 (the OpenCL decode)**. The census instrument,
  hot-set selection, eviction/refresh discipline and the stale-byte digest
  proof proceed; the speed row stays EMPTY and G UNPINNED until the
  resident-compute patch lands. See `docs/campaigns/expert-hot-set-lru.md`.
- 2026-09-22 — **two rows filled from the now-complete measurements; the
  remaining three stay EMPTY, and why.** [documents the fill, `measured-here`
  for the values]

  **FILLED.**
  - *convergence.* `rounds_to_plateau=None`, `plateau=False` at S = 6 and
    S = 10, at both 512 decode tokens (window 003) and 4,096 decode tokens
    (window 004); the selected set changed at every prefix including
    2,048 → 4,096. This is a failing measurement, and it is recorded as one:
    **V3 FIRES (2026-09-22)**. Raw series and the two window-004 plateau
    JSONs are on the persistent census path; the numbers are transcribed in
    `docs/campaigns/expert-hot-set-lru.md` (status, 2026-09-21 night).
  - *seed implication at the ratio-99 budget.* On the window-004 CORPUS
    census (3,278,880 accesses, 6,831 × 48 × 10 exactly) the census top-S
    per layer covers S = 6 **11.37 %**, S = 10 **16.22 %**, S = 16
    **22.19 %**, S = 32 **33.84 %**, S = 64 **49.20 %** of the corpus's
    routed accesses — 9.70 / 8.31 / 7.10 / 5.41 / 3.94× the analytic
    S/512 chance. This is an input to the seed decision, not a V-clause; it
    is recorded because it is measured, and it is a hit fraction over a
    non-plateauing window, not a converged steady state.

  **STILL EMPTY.**
  - *warm-up decode vs the host-bound baseline.* Speed row, HELD for
    `sub4bit-vram-kernel` step 3 (the OpenCL decode); no speed measurement
    is taken on the host-compute tier. G stays UNPINNED.
  - *stale-byte zero proof.* Blocked on the engine-side host/card readback
    that does not exist; per the design it is not asserted from code.
  - *quality under policy.* Not yet measured at the moment of this entry;
    filled in the same session once the census-seeded served path has run
    twice against the incumbent seed (see the appended entry below when it
    lands).
  - *verdict.* REPORT ONLY until the tag, unchanged.
- 2026-09-22 (cont.) — **the quality row is filled: PASS, no V4.** The
  census-seeded static partition was wired (patch 0046) and served on the
  A770 twice, incumbent vs census seed, same prompt and greedy settings.
  Raw evidence, both digests and both OTD_PERF lines are in the session's
  `quality-report.txt` on the persistent census path; the summary:

  - **Card/config:** A770 (GPU.1, PCI 8086:56a0), native d48n artifact,
    `--offload-ratio 99 --moe-cpu-tier`, KV u8, chunk 512, n_ctx 8192; prompt
    256 token ids from the pinned capture, greedy `max_tokens` 32,
    temperature 0. One fresh process per arm.
  - **Incumbent** (`MOE_CPU_TIER_SEED` unset): `seed_source=splitmix64`,
    `slots=5`, `resident_checksum=0xa3dd88ad84a30530` (layer shown), prefill
    256 tok in 324.70 s, decode 32 tok in 60.84 s. Greedy sha256
    `2169836b33e8bc74d7965fff867b13c1d3637388a4b52f11f639f381ce7cc36f`.
  - **Census seed** (env `MOE_CPU_TIER_SEED` pointing at the corpus S = 5
    seed; the operator-local path is in the local notes): `seed_source=census`,
    `slots=5`, `census_seed_fp=0x3ae78e143cabbfb`,
    `resident_checksum=0x3dd7053a56e79968`, prefill 256 tok in 333.68 s,
    decode 32 tok in 60.20 s. Greedy sha256 **the same**
    `2169836b33e8bc74d7965fff867b13c1d3637388a4b52f11f639f381ce7cc36f`.
  - **Verdict:** byte-identical greedy output. The design's expectation holds:
    under the native artifact every routed expert runs on the host tier
    (patch 0043), so residency moves bytes, not arithmetic. **No V4.**
  - **Red-first refusal, measured on the card:** the S = 6 corpus seed (one
    slot over the pool) was run first and refused the load —
    `census seed: layer_key 284636629 lists 6 experts but the pool has 5 slots
    (mismatched budget)` — and the process never became ready. The device-free
    cells (`tools/test_census_seed.py`, 14) cover the malformed/mismatched
    cases; this is the same refusal on real hardware.
  - **Measured correction:** the plugin's pool at ratio 99 is 5 slots/layer
    (`prepare_moe_otd_params` integer division), while the engine's ledger
    prices 6 (`ceil`). Recorded in the seed-implication row above; it is why
    the served seed is the corpus top-5, not top-6.
  - The speed row stays EMPTY (HELD), G stays UNPINNED, and the stale-byte
    proof stays EMPTY (no engine-side host/card readback). No speed number is
    claimed.
- 2026-09-22 (G pin, PREDICTION commit) — **G is PINNED at 1.10 before any
  speed measurement of this leg, and the HOLD is discharged.**
  [documented pin; `measured-here` for its inputs, `derived` for the ceiling]
  The native per-expert decode the HOLD named exists (patches 0043/0045) and
  serves at the 0045+0047 prefix `ov-0047` (plugin `f021de51b5812ee2`) on both
  cards, so the speed leg is runnable. The pin, written here before the run:
  baseline `H = 0.5–0.8 t/s` (d48n host tier, B60, ratio 99 + tier, KV u8,
  f16, chunk 512; gate read at its upper edge, 0.8) and the A770 same-day
  host-tier comparand 0.526 t/s `[derived: 32 decode tokens / 60.84 s; the
  quality-window incumbent arm, which did NOT pass
  `--moe-per-expert-dispatch`]`; served hit fraction `h₉₉ = 9.9595 %`
  (window-004 corpus census top-5 at the plugin's measured 5 slots/layer;
  in-sample, with the held-out decode-regime value 14.59 % at S = 6 also
  disclosed). The resident-compute ratio `ρ` is **NOT
  separately measured**: the one counter point (B60, ratio 75, 0047:
  `per_expert_gpu_invocations=135874` → 67,937 card pairs vs
  `cpu_tier_pairs=187903`, `h₇₅ = 0.2655`, decode 0.5672 t/s) spans the whole
  process (255,840 pairs = 533 tokens × 48 × 10) and is a probe/warm-up/decode
  mixture, so it is an observation only — at `h₇₅ ≈ 0.27` the served decode sat
  inside the baseline band, no win demonstrated. The assumption-free ceiling
  `1/(1−h₉₉) = 1.1106` `[derived]` is the pin's basis. **The pinned gate is
  G = 1.10 (the ceiling); the prediction is that the gate is NOT met and
  clause V1 fires** — no measured point demonstrates a per-pair win. The
  predicted shape is a V1 shortfall with the hot-set *correctly engaged*.
  Falsifiable: a measured decode at or above the gate overturns it, and G is
  corrected in place with the measurement's date. The speed row itself stays
  EMPTY until the measurement.
- 2026-09-22 (rate leg, MEASURED) — **V1 fires at the ratio-99 VENICE budget;
the speed row stays EMPTY, and the sweep locates the win at ratio 75.**
[measured-here] Native `d48n`, plugin `ov-0047` (`f021de51b5812ee2`), one
fresh process per arm, KV u8, one lane, the capture window-0's first 256 ids,
greedy 64, temperature 0. `--fit-ledger-dir` skipped the load probes on the
second matching run (same greedy answer reproduced). The gate at ratio 99:
A770 census **0.556 t/s** < 1.10 × 0.526 = 0.579 (incumbent 0.547; same-day
host-tier comparand 0.526); B60 census **0.555 t/s** < 1.10 × 0.8 = 0.88.
The residency sweep: A770 ratio 75 census top-128 **0.842 t/s** against the
same-config host control **0.465 t/s** = **1.81×**, `ρ = 0.073` (the card
~13× the host per pair). The pin's `ρ ≈ 1.55` premise is corrected in place
above; `G = 1.10` and the V1 verdict stand. The ratio-50 point is refused on
the A770 (16 GiB card) and did not return on the B60. Raw evidence in
`docs/campaigns/sub4bit-vram-kernel.md`, status 2026-09-22 (rate leg). The
stale-byte proof stays EMPTY (no engine-side host/card readback).
