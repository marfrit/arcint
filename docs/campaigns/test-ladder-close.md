# test-ladder-close — close the 0.3.1 lead item: references filled from the runners' own windows, the first real run on the record

## The defect, as measured

Not an engine defect: the last owed steps of the item that split the test
ladder (`docs/design-0.3.1-test-ladder.md`, Increments 1–3 on `main`,
DESIGN §5/§5.1). The mechanism is complete and self-tested device-free
(41 assertions); the acceptance target ran for real once, on the
Increment-2 commit, and produced these facts:

- Passed: the coder at M9's offload cell (equivalence at one and two
  lanes, concurrency) and served on both cards (equivalence), the dense
  agent (equivalence with its MTP section), the n-gram determinism cell
  (six processes, one hash, 33.3 % drafts accepted), the depth ladder's
  first cell (24 GB card, u8, 98,187 sized tokens).
- The tier reference cell (16 GiB card, 35B int4, ratio 50, 8 GiB pool,
  u8 KV, one lane, n_ctx 65,536, 1,167 sized tokens, 64 greedy tokens):
  tier OFF warm decode 12.4/12.5 t/s, prefill 86.2/86.1; tier ON warm
  decode 16.6/16.4, prefill 26.3/26.3; E2 held. Every tier-ON output
  differed from tier OFF; all four ON outputs identical to each other, all
  four OFF likewise. The runner gated ON == OFF and failed; corrected on
  `main` the same day to gate ON-vs-ON, OFF-vs-OFF and E2.
- Four enumeration corrections, all on `main`: declared expected skips
  (the coder artifact has no MTP head), two decode-probe cells replacing
  gate text no suite measured, two concurrency cells the 0.3.0 gate ran
  but the list lacked, the tier gate above. A fifth runner correction:
  the depth-ladder runner sized its prompt through the server in every
  cell — two full prefills per cell at this depth (546 s and 812 s on the
  24 GB card at u8:i4) — changing the cell's shape from the one prefill
  per process the 0.3.0 gate measured to three; corrected on `main` to
  size the prompt once per artifact and reuse it (DESIGN §7.0.2aj,
  correction 5). And one schema correction: a report-only reference form
  (`gate_at: null`, §8.9).

Every cell that prints metrics still carries `references: null`: nothing
regresses a run yet.

## Known against hypothesised

Known: everything above is on the log of that window. Hypothesised:
that the tier cell's `grouped_fallbacks` counter is 40 per process
independent of request count (§8.1 asks the fill window to confirm; the
first window's per-process logs were cleaned by the runner before the
counter was read).

## Gate

`tests/acceptance/run.py --all` on the filled enumeration, with only the
external cell named, exits 0 on the card window that follows the fill,
with every reference either gating (`gate_at` a number derived by §8.3
from the runners' own samples) or report-only (`gate_at: null`) with the
reason in the design note; DESIGN carries the record of the first real
run (§7.0.2aj) and of the fill window (§7.0.2ak); the milestone row for
the item has its closing line; `ctest -L unit` stays green device-free.

## Entry criteria

Met. The first window's log; the corrected runners on `main`; the
report-only form on `main`.

## Scope — in / out

In:
1. DESIGN §7.0.2aj: the first real run, cell by cell, the tier numbers
   above, the depth ladder's four cells, sanitizers and package-build,
   the four corrections, the schema correction.
2. Card windows, `--cell` runs on the corrected runners: the tier cell,
   the two decode probes, the two new concurrency cells, and the depth
   ladder's u8:i4 cell on both cards with one prefill per process (the
   shape the 0.3.0 gate measured; the first run's two u8:i4 results are
   void, §7.0.2aj). Every metric prints as `REPORT` (references still
   null). Only the unit owning the card in use is stopped, and restored
   after, verified fresh.
3. The fill, from both windows' samples per §8.3: `decode-warm-2nd-on`
   gates (worst warm sample, band by spread); `decode-ratio-on-off` gates
   (the same rule on the ratio samples; the runner prints two decimals);
   `decode-warm-2nd-off` and both prefills are report-only (OFF decode
   drifted 9 % between the two windows on the same card and flags, and
   prefill is a defect 0.3.1's campaigns are chartered to fix — gating it
   would freeze it, §8.1); `grouped-fallbacks-on` gates at 40 per process
   (`higher-is-worse`) if the fill window prints it. The large-card decode
   probe's warm decode gates at §5's ≥ 60 t/s with `samples: 1`; the
   small-card probe's by §8.3 from its samples; both probes' cold
   first-request metrics report-only. The depth ladder stays `null` (one
   sample per cell, no independent bar) with its four numbers in
   §7.0.2aj. Regenerate the §5.1 span and the checklist (`tools/
   acceptance_manifest.py`), which the unit test `acceptance-enumeration`
   then checks. A text correction rides with the fill: `cells.json`'s
   `tier-reference-cell` `flags` field still reads "1,198-tok prompt"
   while the sizer produced 1,167 tokens on the real coder-35b artifact
   (§7.0.2aj); the fill commit corrects the flags text to "prompt sized
   to ~1,198 tokens (measured 1,167 on the coder-35b artifact)".
4. DESIGN §7.0.2ak for the fill window; the milestone row's closing line;
   `CHANGELOG.md` under the unreleased heading; `llm.txt` counts.

Out: any engine change; new cells; the external cell's remote invocation
(its own campaign, `pruefstand-cell-remote`).

## Where it lives

`tests/acceptance/cells.json` (references, `metric_qualifiers`,
`expected_skips`), `tests/acceptance/run.py` (`compare_metrics`),
`tests/acceptance/cells/{tier_reference,decode_probe,depth_ladder}.sh`
(the `emit_metric` call sites name the metrics), `tools/
acceptance_manifest.py` (`validate_cells`, `--check`), `tests/acceptance/
selftest.py`, `docs/design-0.3.1-test-ladder.md` §8, `docs/
release-checklist.md` (generated), DESIGN §5.1's generated span,
`docs/milestone-0.3.0.md`'s last backlog row.

## Pipeline for this campaign

Record → window → fill → review of the fill commit (a reviewer checks
every `value`/`gate_at` against the window logs and §8.3's arithmetic,
and that no host name entered the tree) → commit → the milestone row.

## Invariants

The runners carry nothing operator-local. A reference's `value` is a
sample the committed runner printed, never a number from an ad-hoc script
(§8.4). A gate derived from a spread over 10 % is not a gate (§8.3).

## Status

- 2026-09-05 — Increments 1–3 on `main`; first real run of the target
  on the dev host (Increment-2 commit, `+p4` runtime): eight cells passed
  as listed, the tier cell failed by its own too-strong gate with valid
  numbers, depth ladder in progress at the time of writing; the four
  corrections and the report-only form landed the same day. Open: the
  §7.0.2aj record, the fill window, the fill, §7.0.2ak, the row's close.
- 2026-09-05, later — the first window completed: `12 cell(s) run, 9
  passed, 1 skipped, 4 named skip(s) promoted, 2 failed`, 8,512 s; the
  depth ladder's two u8:i4 cells failed for reasons that became runner
  corrections 5 and 6 (sizing shape; a flat 4,096-token margin the 16 GiB
  card refused by 1,011 tokens) and one engine-side event (a GT reset on
  the 24 GB card with a second client present, handed to
  `u8i4-deep-prefill-fault` and `direct-submission-fault`). §7.0.2aj
  written. The follow-up became two windows: the small-card one (tier
  cell, small decode probe, small concurrency) running the same day, the
  two 24 GB-card cells and the ladder's u8:i4 rerun (one prefill per
  process, both cards) once that card is free.
- 2026-09-05, follow-up window (tier cell): runner correction 7. The
  `metric_value` helper matched "prefill|decode" plus "t/s", and the
  server's load banner ("62.7 t/s decode at 53.5k, 1584 t/s prefill", the
  artifact's recorded rates) matched too, shifting every fixed index by
  one: the emitted "warm 2nd" metrics were the first request's (4.9 t/s
  decode ON where the second request read 16.5). Shown red against the
  fake server once it printed the banner and distinct per-request rates;
  all three runners now match the server's own `slot N:` request lines.
  The tier metrics now come from each arm's SECOND process: the first
  process of this window (`off1`) decoded at 0.3 then 1.7 t/s where
  `off2`, seconds later, read 11.7 / 11.9 and both ON processes hit the
  record on their second request (26.3 / 16.5, 26.3 / 16.4; E2 25.4–26.3 /
  15.5–16.5) — the cold-sequence first process the record already knows,
  which is `static-partition-cold-start`'s cost, not a decode reference.
  This window's rates are therefore usable for the fill from the second
  processes, with the first processes reported beside them.
