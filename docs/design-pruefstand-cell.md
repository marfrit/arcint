<!-- Campaign pruefstand-cell-remote (docs/campaigns/pruefstand-cell-remote.md),
design pass 2026-09-05. Changes the run manifest's schema and the external
cell's contract; short, because the mechanism it reuses already exists. -->

# Design note: the Prüfstand cell reaches its harness through the run manifest

Spec: `docs/campaigns/pruefstand-cell-remote.md` — the acceptance target's
one external cell can only be skipped by name where the harness does not
live; make it runnable from the card window, with nothing host-shaped in a
tracked file.

## 0. What is actually true today

- `tests/acceptance/cells.json`'s `pruefstand` cell carries an `external`
  block: an environment variable (`ARCINT_PRUEFSTAND`), an invocation
  template (`$ARCINT_PRUEFSTAND --greedy --no-thinking`), a `score_parse`
  regex and `reference_score: "10/10"`.
- `run.py`'s `run_external` reads the variable, substitutes it into the
  template, runs it through a shell and maps the exit code (0/77/other).
  It never reads `score_parse` or `reference_score`: a wrapper that exits
  0 after printing `score: 3/10` passes the cell. The score contract is
  inert.
- The invocation names a wrapper that does not exist. The harness is two
  steps — a request script against an endpoint (greedy, thinking off, a
  token cap, the served model name, all by environment) that saves the
  answer, then a Lua scorer that executes the answer against ten RFC 4180
  cases and prints `PUNKTE N/10`. Both live on the session host, not the
  dev host, and neither is in this repository (by rule: never the harness).
- Every other card-requiring parameter moved into the generated run
  manifest at configure time (`docs/design-0.3.1-test-ladder.md` §2:
  environment "does not appear in the ctest log"); this cell's parameter
  did not. The manifest today: `binary`, `model_root`, `device_large`,
  `device_small`.
- The cell's card class is `deployed-package`: it measures the coder as
  served by the installed package on its own port, so running it stops no
  unit and touches no card directly.

## 1. Mechanism

**One new manifest key, `pruefstand`.** A command line, empty by default.
`CMakeLists.txt`'s `ARCINT_ACCEPTANCE` block gains
`ARCINT_ACCEPTANCE_PRUEFSTAND` (a `STRING` cache variable, `""`), written
into `run_manifest.json` next to the four existing keys. The value lives in
`CMakeCache.txt` and the generated manifest under the build directory —
never in a tracked file. It is the operator's wrapper: whatever reaches the
harness from that host (a local copy of the harness's three files — the
task text, the request script, the scorer — an ssh hop, anything), invoked
with no arguments.

**The cell's `external` block names the key, nothing else:**
`{"manifest_key": "pruefstand"}`. `run_external` reads
`manifest[manifest_key]`; empty → `SKIP external-harness-not-configured`
(the same reason as today, so a release log that ran without the harness
still says so); non-empty → run through the shell, exit code mapped as
before. The environment variable is gone as a channel, not kept as a
fallback: a second channel that does not appear in the ctest log is the
thing §2 rejected, and an operator who sets it would be silently ignored
otherwise — better that the variable means nothing at all.

**The score rides the metric channel that already exists.** The wrapper's
contract is one line on stdout: `ACCEPTANCE-METRIC score <n> points`. The
cell's `references` carry `score` with `value 10`, `gate_at 10`,
`lower-is-worse`, so `compare_metrics` (§8.2) does the comparison: 9 is a
regression (FAIL under `--cell`, `--allow-regress pruefstand/score` under
`--all` — which a release should never pass), no line at all is the
missing-metric rule (FAIL), a non-numeric score is a hard fail. Nothing
new is parsed; `score_parse` and `reference_score` are deleted rather than
implemented, because implementing them would be a second comparison path
beside §8.2's.

**Schema.** `validate_cells` learns the external shape: `external`, when
present, must be an object with a non-empty string `manifest_key`, and the
cell's `runner` must be null (a cell cannot be both). A malformed external
cell fails the device-free `acceptance-enumeration` unit test.

**Timeout.** The cell gains `timeout_seconds: 1800` like every other cell
(the harness's own request timeout is longer; the first run through the
cell answered in 9.9 s, DESIGN §7.0.2ao, and the bound leaves room for a
cold package), so the generated per-cell ctest entry and `run.py --all`
both bound it.

**Order.** The cell moves to the end of the enumeration, after
`package-build`: the install is outside every cell, and a release that
wants the candidate measured through this cell deploys it between the two.

## 2. Rejected

- *Keep `ARCINT_PRUEFSTAND` as a fallback.* Two channels, one invisible in
  the log; and the fallback would be the one operators reach for.
- *Implement `score_parse`.* A regex-and-compare path parallel to §8.2 for
  one cell; the wrapper printing one metric line is smaller and lands the
  score in the same REPORT/REGRESSED/IMPROVED output as every other number.
- *Ship a generic wrapper in the repository.* It would have to know the
  harness's file names, its environment switches and its `PUNKTE` line —
  the harness, by another name.
- *The "recorded external run" close* (the campaign's shape 2). Defensible,
  but it verifies a record, not the served endpoint; shape 1 costs one
  request against the deployed package (9.9 s on the first run) and
  exercises the real thing.

## 3. Red-first plan

| mechanism | red case (fails before) | green after |
|---|---|---|
| manifest key is the channel | selftest: a temp manifest whose `pruefstand` names a fixture script printing `score 10` → exit 0 | fails before (the runner reads the environment and skips), passes after |
| the environment is not a channel | selftest: `ARCINT_PRUEFSTAND` set in the environment, manifest key empty → unnamed skip, exit != 0 | fails before (the variable runs the fixture and passes) |
| the score is gated | fixture prints `score 9` → exit != 0 with REGRESSED; prints nothing → exit != 0 (missing metric); exits 1 → exit != 0 | the existing §8.2 code, reached through the new external path |
| schema | `validate_cells`: `external` without `manifest_key` → error; `external` plus a `runner` → error | fails before (no such check) |
| the real cell | `run.py --cell pruefstand` on the dev host, the manifest's key pointing at the operator's wrapper, against the deployed package | PASS with `score 10 points`; the reference's `binary`/`measured` fields filled from that run |

## 4. Files

`tests/acceptance/cells.json` (the cell), `tests/acceptance/run.py`
(`run_external`), `tools/acceptance_manifest.py` (`validate_cells`, the
checklist's external line), `CMakeLists.txt` (cache variable, manifest),
`tests/acceptance/selftest.py` and `fixtures/` (manifest key, fixture
script, fake cells), `docs/release-checklist.md` and the DESIGN §5.1 span
(regenerated), `docs/design-0.3.1-test-ladder.md` §3 (the "external
dependency" paragraph superseded by this note).
