<!-- 0.3.1, lead item. Produced by the campaign's design pass on 2026-09-05
against docs/milestone-0.3.0.md's backlog row "Unit tests and acceptance
tests differentiated" and DESIGN §5.1; accepted for implementation in the
increments of §6. The 0.3.0 release gate (DESIGN §7.0.2ai) was run from
prose and ad-hoc window scripts; this note exists so the next one is run
from an enumeration. -->

# Design note: differentiating unit and acceptance tests (0.3.1)

Spec: `docs/milestone-0.3.0.md:91` — device-free `ctest` is the unit gate, a second target is the card-requiring acceptance set with its cells enumerated in one place, and the release checklist is generated from that enumeration. Ladder: `DESIGN.md:1399-1418` (§5.1). Today's assembly: `CMakeLists.txt:167-216`.

## 0. What is actually true today

- `ctest` registers three tests: `unit` (`CMakeLists.txt:202`), `roundtrip` (`:204`), `stress` (`:212`). All three are device-free; §5.1 puts all three in the Unit class.
- Only `tests/test_rope_precision.cpp` is compile-gated on `ARCINT_OPENVINO` (`:9`, `:213`); it holds 4 `TEST()` blocks, exactly the 413−409 of `DESIGN.md:1273`. It builds a synthetic `ov::Model` and never touches `ov::Core` — it needs OV *headers*, not a card. `tests/test_config.cpp:47-58` switches branch on the same macro. Nothing else in the unit binary is OpenVINO- or card-dependent.
- `contrib/packaging/arcint/build-deb.sh:82` runs `ctest -R unit`, which selects the single test named `unit` — `roundtrip` and `stress` are silently outside the package gate, and the comment above it ("GPU-dependent suites skip themselves when no card is present") describes nothing that exists.
- The motivating example, `tests/test_affinity.cpp`, pinned to CPU 0 by literal until `eae32d1` and failed in a container whose cpuset excludes it — a unit case asserting a host property.
- Prose drifts: `llm.txt:790-792` says "306 unit cases … 48 curl checks" against DESIGN's 409/64. This is the argument for generating, not re-reading.

## 1. Mechanism: labels, plus a configure-time guard

**Chosen.** `LABELS unit` on `unit`/`roundtrip`/`stress`; acceptance cells registered only inside `if(ARCINT_ACCEPTANCE)` (new option, default OFF, refused unless `ARCINT_OPENVINO` is ON), each with `LABELS acceptance`. Bare `ctest` in the default build therefore runs exactly the unit set — not by convention but because nothing else is registered. The labels make the partition *assertable* (`ctest -N -L acceptance` must list zero tests in a default build).

**Rejected:** labels alone — bare `ctest` would then run acceptance cells on a card-less machine, which is the gate's own counter-example. Separate targets/directories — duplicates ctest's timeouts and reporting and gives the recipe nothing to select on. A bare CMake option without labels — no way to assert the partition.

**Recipe change:** `build-deb.sh:82` becomes `ctest --output-on-failure --no-tests=error -L unit`, and its comment is corrected. This widens the package gate from one test to three, which is the honest reading of §5.1's Unit row; cost and risk in §7.

## 2. Card-requiring cells: parameters and skip semantics

**Parameters** come from CMake cache variables resolved at configure time (`ARCINT_ACCEPTANCE_MODEL_ROOT`, `ARCINT_ACCEPTANCE_DEVICE_LARGE`, `ARCINT_ACCEPTANCE_DEVICE_SMALL`) into a generated run manifest that also carries `$<TARGET_FILE:arcint>`. Environment is rejected as the primary channel: it does not appear in the ctest log, and §7.0.2ai already shows an env term (`ARCINT_MOE_DEVICE_POOL_BYTES` at 8 GiB per offload arm) that has to be *stated* per cell. A checked-in config file alone is rejected because it cannot know the build's binary path.

**Skip semantics.** ctest's `SKIP_RETURN_CODE` marks a skip as not-failed, so `ctest -L acceptance` on a card-less machine would go **green with every cell skipped**. That is the failure this row exists to close. So the acceptance target is `tests/acceptance/run.py --all`, not bare ctest: a cell exits 0 / 1 / 77(skip, with a printed reason), and the runner exits non-zero unless every 77 is named on the command line as `--allow-skip <cell>`. Symmetrically it exits non-zero when `--allow-skip` names a cell that ran, or a name absent from the enumeration — so the allow-list cannot rot. The per-cell ctest tests call the runner, so its exit code is ctest's.

The suites carry the same defect internally: `tests/equivalence/run.sh:225` prints "no MTP head in $MODEL; skipping the MTP gates" and still exits 0 at `:267`. Increment 2 makes it print a machine-readable `ACCEPTANCE-SKIP <check> <reason>` line that the runner promotes to a named skip -- not a bare `SKIP`, because `tests/harness.h`'s unit-test harness already prints its own `SKIP %s: %s` lines (`tests/test_main.cpp`), and the sanitizers cell tails its ctest log to its own stdout, so a bare `SKIP` would let a unit-test skip get promoted as an acceptance one by accident.

**Increment 3 amendment (2026-09-05): declared expected skips.** The MTP
section fires on *every* coder-artifact cell running the equivalence suite
(`coder-offload-1lane`, `coder-offload-2lane`, `coder-served-large`,
`coder-served-small`) for the same reason every time -- the coder artifact
itself carries no MTP head -- so naming it on every `--all` command line is
a fixed incantation, not a decision. `cells.json` gains an optional
`"expected_skips": [{"check", "reason"}]` per cell; a declared skip is
pre-named (no `--allow-skip` needed), prints `SKIP <cell>/<check>
(declared: <reason>)`, and its *absence* fails the cell -- the enumeration
would be stale, the artifact having grown the capability the skip was
declared against. `--allow-skip` naming a declared one is an error: it is
already named by definition, so naming it again is the same rot as an
unused entry. `agent-dense` (the dense artifact, which does carry an MTP
head) does not declare it.

## 3. The enumeration

One file, `tests/acceptance/cells.json` (JSON, matching the `models/allowlist-raw.json` precedent that `tests/test_provenance.cpp:18` already reads from the source tree). Per cell: `name`, `runner` + args, `artifact_class`, `card_class`, `flags`, `design` (defining section), `gates`, `reports`, `external`.

| cell | runner | artifact / card | flags | defines | gates vs reports |
|---|---|---|---|---|---|
| `coder-offload-1lane` | `tests/equivalence/run.sh` | coder int4, 24 GB | `--offload-ratio 20 --paged-kv u8:i4`, 8 GiB pool | §5, §3.4 | gates byte-equality; reports chunk sweep |
| `coder-offload-2lane` | same, `ARCINT_EXTRA_ARGS="--parallel 2"` | " | " | §5 | gates |
| `coder-offload-concurrency` | `tests/concurrency/run.py` | " | " | §4.1, §5 | gates |
| `coder-served-large` | equivalence + decode sample | coder, 24 GB, no offload | `--paged-kv u8` | §5 | gates ≥ 60 t/s at steady state (53.4 cold / 69.2 warm, §7.0.2ai) |
| `coder-served-small` | equivalence + concurrency | coder, 16 GiB | `--paged-kv u8 --n-ctx 98304`, 2 GiB prefix cache | §5 | gates suites; reports 48.0/49.5 t/s |
| `agent-dense` | equivalence (incl. MTP section) + concurrency | dense 27B, 24 GB | `--paged-kv u8` | §3.5.2, §5 | gates MTP identity + acceptance > 10% |
| `tier-reference-cell` | window script | 35B int4, 16 GiB | ratio 50, 8 GiB pool, u8 KV, 1 lane, n_ctx 65536, 1,198-tok prompt, 64 greedy tokens, 2 requests/process | §3.4, §7.0.2ai | gates tier ON byte-identical to itself and tier OFF likewise (§3.4 is history-independence, not ON-equals-OFF -- device f16 vs host f32 are not bit-equal, §7.0.2ae/§7.0.2af) and E2 (CONT after history = CONT fresh); reports ON-vs-OFF identity and the first divergence, and the warm rates |
| `ngram-determinism-repeat` | window script | 35B tier config | `--draft 4 --draft-ngram 3`, six fresh processes | §7.0.2ae | gates six identical outputs |
| `depth-ladder` | window script | both cards | 98,147-token prefill at u8 and u8:i4 | §5.1 | gates load; reports t/s |
| `pruefstand` | **external** | coder, through the deployed package | greedy, thinking off | §5 | gates 10/10 |
| `sanitizers` | ASan+UBSan (x86_64), UBSan elsewhere, over unit+roundtrip+stress | none | `-fno-sanitize-recover` | §5 | gates zero reports |
| `package-build` | `contrib/packaging/arcint/build-deb.sh` | none | version stamp (`:60`), RPATH probe | §5.1 | gates; post-deploy smoke follows |

**Increment 3 correction (2026-09-05): a cell's `gates` text must not claim
more than its own runner enforces.** `coder-served-large`'s "equivalence +
decode sample" and `coder-served-small`'s "equivalence + concurrency" were
prose describing what the 0.3.0 window ran by hand, not what these cells'
actual runner (`tests/equivalence/run.sh` alone) gates; both rows now gate
byte-equality only, and the decode-rate claim moves to two new sibling cells
(`coder-served-large-decode`, `coder-served-small-decode`, runner
`tests/acceptance/cells/decode_probe.sh`) that actually measure it, with
`references: null` until the fill window. The concurrency claim moves the
same way, into `coder-served-small-concurrency` and a new
`agent-dense-concurrency` (mirroring `coder-offload-concurrency`, both
running `tests/concurrency/run.py`) — `agent-dense`'s own row keeps only
what its equivalence run gates (MTP identity and acceptance).

**External dependency, honestly.** The Prüfstand harness is fleet tooling outside this repository. Its cell carries `"external": {"env": "ARCINT_PRUEFSTAND", "invocation": …, "score_parse": …}` and the reference score — never the harness. Unset env ⇒ the cell reports `SKIPPED external-harness-not-configured`, which must then be named in `--allow-skip`, so a release that ran without it says so in its own log rather than passing.

**DESIGN §5.1 and the checklist are generated.** `tools/acceptance_manifest.py` renders the §5.1 acceptance block and `docs/release-checklist.md` between marker comments; `--check` regenerates and diffs. Rejected: hand-maintained plus a consistency test — `llm.txt:790` is the measured outcome of that policy.

## 4. Unit-side hygiene

Rule: **a unit case may assert only properties of arcint's own code; where it needs a host capability it probes first and SKIPs with a reason.** Present offenders:

- `tests/test_affinity.cpp:10-35` — post-`eae32d1` it picks the first allowed core, but still fails where `sched_setaffinity` is denied outright. Gets `SKIP_UNLESS(pin permitted, …)`.
- `tests/test_turnstile.cpp:59,63,84` — 30/50 ms sleeps; timing-sensitive on a loaded builder and under ASan. Keep, but documented as the one timing-dependent case.
- `tests/test_artifact.cpp:27` reads `TMPDIR`; `tests/test_provenance.cpp:18` reads the source tree via `ARCINT_SOURCE_DIR` (`CMakeLists.txt:198-199`) — both fine in-tree, both recorded.
- `tests/roundtrip.sh:14-15` and `tests/concurrency/stress.sh:20` exit 2 when `curl`/`python3` are missing, which ctest reports as a *failure*. Becomes exit 77 with a reason.

**Harness SKIP primitive** (`tests/harness.h`): `t::skip(reason)` and `SKIP_UNLESS(cond, reason)`, recording `{case, reason}`. `tests/test_main.cpp:47` prints `N cases run, M failed, K skipped` plus one line per skip. Exit stays 0 on skips, but ctest registers `unit` with `--max-skips 0`: a skip is a failure unless the case is named with `--allow-skip <case>` — the same rule as the acceptance runner, so there is one rule in two places.

## 5. Red-first plan

| mechanism | red case (fails before) | green after |
|---|---|---|
| enumeration ↔ §5.1/checklist | remove `ngram-determinism-repeat` from `cells.json`; separately, hand-edit the generated block in `DESIGN.md` | `acceptance-enumeration` (`--check`) red in both directions, green when list and generated files agree |
| runner skip discipline | fixture manifest `tests/acceptance/fixtures/cells-fake.json` with `/bin/true`, `/bin/false`, and a 77-exiting script: run before the allow-skip logic exists → unnamed 77 returns 0 | five assertions: all-pass ⇒ 0; unnamed 77 ⇒ ≠0; named ⇒ 0; naming a passed cell ⇒ ≠0; naming an unknown cell ⇒ ≠0 |
| label partition | register an acceptance cell outside the `ARCINT_ACCEPTANCE` guard | `ctest -N -L acceptance` lists 0 in a default build |
| harness SKIP | a case calling `SKIP_UNLESS(false, …)` under `--max-skips 0` while the flag is stubbed out | exits non-zero; named in `--allow-skip` ⇒ 0 |
| recipe gate | `ctest --no-tests=error -L unit` before labels exist | selects exactly 3 today |

Every one of these runs device-free.

## 6. Scope and order

**Increment 1 — meets the row's gate, no card needed.**

- `CMakeLists.txt:167-216`: labels, `ARCINT_ACCEPTANCE` option + configure-time refusal without `ARCINT_OPENVINO`, per-cell `add_test` from the manifest, `--max-skips 0` on `unit`, two new unit-labelled tests (`acceptance-enumeration`, `acceptance-runner`). ~45 lines.
- `tests/acceptance/cells.json` — the 12 cells above. ~220 lines.
- `tests/acceptance/run.py` — manifest loader, `--cell`/`--all`/`--allow-skip`, skip accounting, exit rules. ~250 lines.
- `tests/acceptance/selftest.py` + `fixtures/cells-fake.json` — ~120 lines.
- `tools/acceptance_manifest.py` — generator with `--check`. ~150 lines.
- `docs/release-checklist.md` — generated, committed. ~80 lines.
- `tests/harness.h`, `tests/test_main.cpp` — SKIP primitive and accounting. ~50 lines.
- `tests/test_affinity.cpp`, `tests/roundtrip.sh`, `tests/concurrency/stress.sh` — skip-with-reason. ~15 lines.
- `contrib/packaging/arcint/build-deb.sh:82` and its comment; `DESIGN.md` §5.1 acceptance row → generated block; `DEVELOPMENT.md:23-38`, `README.md:26-28`, `llm.txt:790-792`.

**Increment 2 — one GPU window.** Move the 0.3.0 gate's window scripts into `tests/acceptance/cells/` as each cell's runner, and add the `SKIP` line to `tests/equivalence/run.sh`. One window, both card classes, per the ritual for stopping and restoring the resident services; the only thing being confirmed is that each runner starts and its console parse matches, not new numbers.

**Increment 3.** Reference values in `reports` so a regression is caught rather than merely recorded.

## 7. Risks

- **Wrongly green.** "ctest green" must never be quoted as a release gate — it says nothing about acceptance by construction. The checklist names the target that produced each line. The larger hazard is ctest's `SKIP_RETURN_CODE`: if acceptance cells are ever registered with it and run via bare `ctest -L acceptance`, a card-less machine goes green with everything skipped. The runner is the only entry point for that reason.
- **Wrongly red.** `--no-tests=error` requires a recent enough ctest; check the package builder's CMake before relying on it, and fall back to asserting the count from `ctest -N -L unit`. Widening the recipe to three tests adds up to ~7 minutes and binds an ephemeral port twice — a builder with a restrictive network namespace turns a green package red. That failure is real and should stand.
- **`ARCINT_OPENVINO`.** The OV build has 4 more cases and `test_config.cpp:47-58` on the other branch. Never assert a bare case *total*; assert skips. `ARCINT_ACCEPTANCE=ON` with `ARCINT_OPENVINO=OFF` must be refused at configure time — a stub binary would "pass" a card cell by serving stub bytes, the worst available outcome.
- **python3** becomes a hard dependency of two more unit tests. It already is one for `roundtrip` and `stress`, so this changes degree, not kind.


## 8. Increment 3 — references, so a regression fails instead of scrolling past

A measured number is prose in `reports` today (`cells.json:99`, `:124`, `:181`),
rendered into the checklist and DESIGN §5.1 by `acceptance_manifest.py:36-38`
(`DESIGN.md:1408`), and compared to nothing. This increment gates a subset of
them under the one-entry-point discipline §2 gave skips.

### 8.1 What a reference is

Per cell, `"references": [ … ]`:

    {"metric": "decode-warm-on", "value": 16.4, "gate_at": 14.8, "unit": "t/s",
     "direction": "lower-is-worse", "samples": 2, "spread_pct": 0.0,
     "config": "16 GiB card, 35B int4, ratio 50, 8 GiB pool, u8 KV, 1 lane,
                n_ctx 65536, --moe-cpu-tier, 2nd request of a fresh process",
     "prompt_tokens": 1197, "design": "§7.0.2ai", "measured": "2026-09-…",
     "binary": "arcint <sha>, +p4"}

`value` is the record; `gate_at` is what run.py compares, derived by §8.3 and
written out, not computed at run time — a threshold you cannot read in the file
is not reviewable. `direction` sets the sense (`lower-is-worse` for t/s,
`higher-is-worse` for seconds and counters). `config` and `prompt_tokens` are
mandatory: CLAUDE.md's rule that a number names card, depth, precision and
configuration, as a schema constraint `acceptance-enumeration` enforces.

Gated vs reported, decided (all §7.0.2ai):

| number | verdict | why |
|---|---|---|
| tier ON decode, 2nd request 16.4/16.4 | **gate** 14.8 t/s | two agreeing samples |
| tier ON/OFF decode ratio, same window (16.4 vs 11.3–11.4) | **gate** ≥ 1.30 | cancels process drift; this is M14's actual claim |
| tier OFF decode 9.5–11.4 | report | 20 % spread, unexplained |
| first-request decode 0.1–4.2, load 215–585 s | report | 42× spread; §7.0.2ai names two owners, separates neither |
| prefill, static partition, 26.6–26.7 warm | report + gate `grouped_fallbacks` ≤ 40/process | below |
| coder-served-large warm 69.2 | gate at §5's ≥ 60 bar, `samples: 1` | §8.3 |
| coder-served-small 48.0/49.5 | **gate** 43.2 t/s | 3.1 % spread |
| depth-ladder t/s ×4 | report | one sample each |

**Amendment (2026-09-05): the ON/OFF decode ratio row above is a rate
comparison, not a byte-identity claim, and stands unchanged.** A card-window
finding on the committed seed prompt showed every tier-ON output differing
from tier OFF, which is permitted (§7.0.2ae/§7.0.2af: device f16 and host f32
arithmetic are not bit-equal, and the 0.3.0 gate's ON-equals-OFF agreement
was a property of that one sample). `tier_reference.sh`'s identity check was
gating ON-equals-OFF anyway -- a claim the engine never made. §3.4's actual
invariant is history-independence: tier ON identical to itself across
processes and requests, tier OFF likewise, and E2. The runner now gates
exactly that (two groups, not one baseline for all seven outputs) and
reports whether ON agrees with OFF this time, with the first divergence
point if not. The ratio row's own claim -- that the ON/OFF *rate* gap
cancels process drift -- was never about byte content and needed no change.

**Amendment (2026-09-05, the fill):** two rows above were demoted to
report-only when the references were written (DESIGN §7.0.2al). The
small-card coder decode (48.0/49.5 in the table) came from §7.0.2ai's ad-hoc
script; the committed runner has one sample (47.6) and this card has no
independently stated bar, so §8.3's single-sample rule makes it a report
until a second window. `grouped_fallbacks` read 400 per process after two
requests, not the 40 the row assumed, and its unit is unread; gating it at
40 would have failed the first real run for a number nobody understands.
Gated as planned: the tier warm decode (14.8 from 16.4/16.4/16.4), the
ON/OFF ratio (1.17 from 1.31/1.38/1.31) and the large-card warm decode at
§5's 60.

**Prefill stays a report.** Gating 26.6 freezes a defect as a specification: the
loss is a known mechanism (`grouped_fallbacks=40`, the per-expert fallback),
0.3.1 is chartered to fix it, and the fixing commit would rewrite the reference
in the same breath — a ratchet that moves only for the wrong reason. What can
worsen unnoticed is the mechanism, so gate the counter, not the rate: integer,
deterministic, and it *is* the 3×. Confirm in the fill window that 40 is per
process independent of request count; if it scales with requests, report it.

### 8.2 Wire format, and who compares

Runners print `ACCEPTANCE-METRIC <metric> <value> <unit>`, mirroring
`ACCEPTANCE-SKIP` (`run.py:50`); run.py scans the stream it already scans
(`run.py:126-131`), qualifies to `<cell>/<metric>`, compares against
`cells.json`. **run.py compares, not the runner.** §2's principle is that one
entry point may tolerate: a comparing runner would keep a copy of the threshold
in shell, decide its own exit code, and leave `--allow-regress` nothing to
attach to — the `SKIP_RETURN_CODE` hole again. *Rejected:* run.py parsing the
server's own `prefill|decode … t/s` lines (`tier_reference.sh:100-102`,
`depth_ladder.sh:117-119`) — a log format is not a contract, and only the runner
knows which request was warm.

Rules, symmetric with skips:

- worse than `gate_at` ⇒ **FAIL**, unless named `--allow-regress <cell>/<metric>`
  in `--all`; no such flag in `--cell` (`run.py:203-213`);
- `--allow-regress` naming a metric that held, or an unknown one ⇒ non-zero
  (`run.py:256-259`'s rule, second domain);
- a printed metric with no reference ⇒ **FAIL**;
- a declared reference the runner never printed ⇒ **FAIL** (missing metric) —
  an error, yes: a reference that can be silently absent is a gate that measures
  nothing, this increment's premise;
- better than the band ⇒ pass, printed `IMPROVED <delta>`, never auto-ratcheted;
  the summary (`run.py:264-266`) gains `N compared, M regressed, K improved`.

### 8.3 Tolerance from a spread

**Gate on the worst sample, band by measurement noise, refuse to gate an
unexplained spread.** With `s` = spread of the warm samples over their minimum:
`s ≤ 5 %` ⇒ `gate_at = min × 0.90`; `5 < s ≤ 10 %` ⇒ `× (1 − 2s)`; `s > 10 %` ⇒
**not a reference** — that wide is an unmeasured mechanism, which CLAUDE.md does
not let one encode as a tolerance. `tol_abs` floor 0.5 t/s.

Applied: tier warm decode 16.4/16.4, `s = 0` ⇒ 14.8 — which sits *below* the
LRU-era record 15.0/15.5 (§7.0.2x): the band is sized by noise, not by the
mechanism's margin, which is why the ON/OFF ratio is the second gate.
coder-served-small 48.0/49.5 ⇒ 43.2.

**A single sample is never a band.** 69.2 gates against §5's independently
stated ≥ 60 t/s bar and records `samples: 1`: that tolerance came from a
commitment, not from measured variance. A single sample with no such bar stays a
report until a second exists (the depth-ladder four).

### 8.4 Where the values come from, and how they change

From this window's runs of the committed runners, never the 0.3.0 ad-hoc
scripts. The prompt is tiled from `prompts/filler-seed.txt` and sized against
the server's own `usage.prompt_tokens` (`size_prompt.py:96`; ±3 % tier, ±2 %
ladder), so the depth is this artifact's, not the 1,198-token document —
§7.0.2ai's figures are the window's sanity envelope, not its references. Each
reference stores the `prompt_tokens` the sizer reported.

**Which process (2026-09-05).** `tier_reference.sh` takes each arm's metrics
from the arm's *second* process, second request. The first process of a
window is the cold one on the record (§7.0.2ai's warming; §7.0.2ak, where the
first OFF process decoded at 0.3 t/s and the next at 11.9 seconds later), and
that cost has its own owner, `static-partition-cold-start`; a decode
reference that included it would gate that campaign's number, not decode.
The first process's lines are still reported beside it. Consequence, stated:
this gate cannot see a regression that affects only a window's first
process. A reference's `config` names the process as well as the request.

A commit moving `value`/`gate_at` cites, in `design` and in the message, the
DESIGN §7.0.2x that measured the new number, and regenerates: references render
beside their cell in `docs/release-checklist.md` (`acceptance_manifest.py:66-86`)
and into the §5.1 span, so the number moving is a visible diff and
`acceptance-enumeration` fails otherwise. Demoting a reference to a report takes
the same citation — §7.0's retraction culture applied to thresholds.

### 8.5 Red-first, device-free

Extend `fixtures/cells-fake.json` and `selftest.py:73-91` (8 assertions → 18),
each fixture a two-line script printing a metric line: at `gate_at` ⇒ 0; below
band, lower-is-worse ⇒ ≠ 0, and 0 when named; above band, higher-is-worse
seconds ⇒ ≠ 0 (proves `direction` is honoured, not assumed); far better ⇒ 0 with
`IMPROVED`; unknown name ⇒ ≠ 0; reference declared, nothing printed ⇒ ≠ 0;
`NaN`/wrong unit ⇒ ≠ 0; `--allow-regress` on a metric that held, and on an
unknown one ⇒ ≠ 0 each. Run them all once with the comparison stubbed out: every
one green there, or they measure nothing.

### 8.6 Files, sizes, and what the fill window records

`cells.json` +~90; `run.py` +~120 (regex, compare, `--allow-regress`,
accounting); `selftest.py` + 7 fixtures +~130; `tier_reference.sh` +~30 and
`depth_ladder.sh` +~25 (emit per request from a known index, not `tail -4` of a
log); `acceptance_manifest.py` +~30; regenerated checklist and §5.1 span. No
CMake change — `acceptance-runner` already runs the selftest.

The window records per metric: card, artifact, KV precision, lanes, n_ctx,
offload ratio and pool bytes, the sizer's `prompt_tokens`, which process and
which request, the arcint commit and the `+pN`, and the raw server line — both
processes, both requests, never one number.

### 8.7 Risks

- **A cold/warm mix in a reference.** The `lines()` helpers tail four matching
  lines of a log holding both requests. The name must state the request
  (`decode-warm-2nd`) and the runner emit it from a known index, or the first
  gate to fire will be a process that logged differently.
- **Sizing tolerance against a t/s reference.** ±3 % of prompt tokens moves
  prefill directly and decode slightly — inside a 10 % band, not a tight one. No
  reference tightens below ~5 % without tightening `--tol` first, and
  `prompt_tokens` is stored so the interaction is auditable. 64 decoded tokens
  is a short window: quantisation, not noise.
- **Both cards under one name.** `depth-ladder` is `card_class: both` and emits
  four numbers; a bare `decode` would average two cards. Names carry card and
  precision (`decode-large-u8`); the schema check refuses an unqualified name in
  a `both` cell — likewise tier ON/OFF.
- **`--allow-regress` rot**, worse than a named skip: a slide once excused stays
  excused. The delta prints into the summary; the checklist carries a line for
  the reason.

### 8.8 Amendment (2026-09-05): `references: null` as the interim state

§8.2 says a printed metric with no reference is a FAIL. Taken literally, that
rule fires the moment `tier_reference.sh` and `depth_ladder.sh` start
printing `ACCEPTANCE-METRIC` lines, since every cell's `references` is `[]`
until the fill window this increment hands off to writes real numbers in —
which would make landing the wire format and landing the numbers one
inseparable change, the opposite of a red-first mechanism. The schema
distinguishes two JSON shapes instead of one: `"references": []` means the
cell is expected to print nothing comparable, so any `ACCEPTANCE-METRIC`
line from it is still the FAIL §8.2 describes; `"references": null` means
"not yet filled" — every printed metric is a report, exempt from both the
unknown-metric and the missing-reference rule, until a later commit replaces
the `null` with a real list and cites the `design`/`measured` provenance
§8.4 requires. Four cells carry `null` now, because they print metrics today
with no values to gate against yet: `tier-reference-cell` and `depth-ladder`
(this increment's own runners), and `coder-served-large-decode` and
`coder-served-small-decode` (added by the same review that split the rate
claim out of `coder-served-large`/`coder-served-small`, §3). Every other
cell carries `[]`, because none of the others has a runner that emits
`ACCEPTANCE-METRIC` at all.

### 8.9 Amendment (2026-09-05): `gate_at: null` is the report-only reference

§8.1's table has rows that are *reports*, not gates — tier OFF decode, the
static-partition prefill, the depth-ladder singles — and §8.4 speaks of
demoting a reference to a report. The schema as landed in Increment 3 had no
such thing: once a cell's `references` is a list, §8.2's "a printed metric with
no reference ⇒ FAIL" applies to every `ACCEPTANCE-METRIC` line the runner
prints, so filling `tier-reference-cell` would have had to either gate its
prefill (freezing a defect, §8.1's own objection) or stop printing it (losing
the record). The first real fill hit exactly that.

A reference whose `gate_at` is `null` is **recorded and reported, never
compared**: it carries every other mandatory field — `value`, `unit`,
`direction`, `samples`, `spread_pct`, `config`, `prompt_tokens`, `design`,
`measured`, `binary` — so the number the run prints has a record to be read
against, run.py prints `REPORT <cell>/<metric>: <value> <unit> (recorded
<value> <unit>, not gated)` with the runner's own number string, counts it
neither as compared nor as a regression, and `--allow-regress` on it is
refused as "did not regress". Four rules stay armed on a report-only metric:
it must still be printed (a declared report the runner no longer emits is the
same stale enumeration a missing gated reference is), its unit must match,
its value must parse as a finite number, and `gate_at` must otherwise be a
number — a `"14.8"` string, like a string `value`, is refused by `--check` and
by run.py's own shape check rather than compared as text. Promoting a report
to a gate, or demoting one, is the `gate_at` field moving under §8.4's
citation rule, visible in the checklist and the §5.1 span as "report only,
not gated" against "gate … at …".

Proven device-free (`selftest.py`, 41 assertions): a report-only reference
with a value far below its record ⇒ exit 0, `REPORT` printed, no `REGRESSED`,
`0 compared`; `--allow-regress` naming it ⇒ non-zero for the did-not-regress
reason; the reference declared but never printed ⇒ non-zero by the
missing-metric rule and not a schema error; `gate_at` or `value` as a string
⇒ schema error, `gate_at` `null` ⇒ none. Red before the mechanism landed: the
exit-0/`REPORT` pair and the string-`gate_at` check. The other two run-level
cases were already non-zero — one by a `TypeError` comparing a float against
`null`, one by the missing-metric rule that applied regardless — which is why
each asserts its reason in run.py's output and not the exit alone; a
re-introduced crash cannot keep either green. Review-found while landing:
the "not counted as compared" half and the `value` type check had no red
case of their own until the count and the string-`value` assertions were
added.
