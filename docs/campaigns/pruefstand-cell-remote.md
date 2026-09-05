# pruefstand-cell-remote — the Prüfstand acceptance cell can only be skipped by name where the harness does not live; make it runnable from the card window

## The defect, as measured

The fleet's 10-point code-generation harness (DESIGN §5's Prüfstand gate:
10/10 greedy on the coder artifact is the bar) lives outside this
repository, on the machine the sessions run from, not on the dev host
with the cards. `tests/acceptance/cells.json`'s `pruefstand` cell
(`artifact_class: coder-int4`, `card_class: deployed-package`, `gates:
"10/10 on the production coder artifact through the deployed package"`)
reflects this: `"runner": null`, and instead an `"external"` block naming
an environment variable (`env: ARCINT_PRUEFSTAND`), an invocation
template (`$ARCINT_PRUEFSTAND --greedy --no-thinking`), a score-parse
regex and `reference_score: "10/10"`.

That block is read by `run_external()` (`tests/acceptance/run.py:185–205`):
if `ARCINT_PRUEFSTAND` is unset, the cell exits `SKIP`
(`external-harness-not-configured`); if set, it runs the invocation and
maps its exit code (0/77/other) like any other cell. A card window has no
reason to have that variable set, so `run.py --all` there can only pass
by naming `--allow-skip pruefstand` — the release gate then depends on
an out-of-band Prüfstand run the enumeration cannot see or check.

This is the one cell whose *parameters* are still environment-driven:
every other card-requiring parameter (binary path, model root, the two
device strings) was deliberately moved into a generated run manifest at
configure time (`docs/design-0.3.1-test-ladder.md` §2: "Environment is
rejected as the primary channel: it does not appear in the ctest log");
`pruefstand`'s invocation was not brought under that rule when the
manifest was introduced (Increment 1, `docs/milestone-0.3.0.md`'s "Unit
tests and acceptance tests differentiated" row).

## Known against hypothesised

Known: the cell's exact schema, `run_external`'s three outcomes, and that
`ARCINT_ACCEPTANCE_MANIFEST` (`CMakeLists.txt:272–280`) today carries
only `binary`, `model_root`, `device_large`, `device_small` — no field
for an external harness's invocation. Hypothesised: nothing about the
mechanism; open is which of the two shapes below is the better fit, a
design-note decision not yet made.

## Gate

The cell runnable from the card window against the deployed package's own
endpoint or the window's own server. Either close is valid:

1. **Reachable invocation.** The harness invocation made reachable
   through an operator-supplied command in the run manifest — a new
   field (parallel to `model_root`/`device_*`: an `ARCINT_ACCEPTANCE_*`
   CMake cache variable, empty by default) that `run_external` reads
   instead of, or as a fallback to, `ARCINT_PRUEFSTAND`. No repository
   file names the host or path; the cache variable's value lives in the
   build directory's `CMakeCache.txt`, never in a tracked file.
2. **Recorded external run.** The cell's contract redefined so a recorded
   external run (score, artifact identity, date, the binary it measured)
   is what the runner checks instead of a live invocation — verified
   against the artifact and binary the window is actually running, gated
   on score and freshness, with its own staleness rule stated.

Either way: `run.py --all` on the release gate exits 0 with nothing named
on `--allow-skip` for `pruefstand`.

## Entry criteria

Met. `run_external` and the `pruefstand` cell are read and understood
(above); `CMakeLists.txt`'s `ARCINT_ACCEPTANCE` block (261–316) is the
precedent for an operator-supplied, configure-time parameter that puts
nothing host-shaped in a tracked file.

## Scope — in / out

In: the `pruefstand` cell's schema, `run_external`'s reading of it, the
manifest-generation block in `CMakeLists.txt`, and (for the "recorded
run" close) whatever small schema a record needs and where it is
checked. `validate_cells()` (`tools/acceptance_manifest.py`) gets
whatever field either close adds, so a malformed cell fails the
device-free `acceptance-enumeration` unit test, not only on a card.

Out: the harness itself (its target, its own scoring, `--greedy
--no-thinking`); any other external-shaped cell (there is only this one);
the Prüfstand score itself, which DESIGN §5 already records as "10/10 at
the artifact's sampling defaults" — not reopened here, only how the cell
reaches the harness.

## Where it lives

`tests/acceptance/cells.json` (the `pruefstand` cell, `"external"`
block); `tests/acceptance/run.py` (`run_external`, 185–205; `run_cell`
dispatch, 208–215); `tools/acceptance_manifest.py` (`validate_cells` line
56; `external` reporting, 239–243); `CMakeLists.txt` (`ARCINT_ACCEPTANCE`
guards, 173–187; the generated manifest, 265–280; the per-cell
`timeout_seconds` lookup that already treats this cell specially — no
key at all, 300–313); `tests/acceptance/selftest.py` (`FAKE_MANIFEST` /
`tests/acceptance/fixtures/manifest.json`); DESIGN §5 and §5.1
(`pruefstand`'s row); `docs/design-0.3.1-test-ladder.md` §2/§3.

## Pipeline for this campaign

Recon (this document, confirmed against `run.py`/`CMakeLists.txt`) →
design note (changes a manifest schema and possibly a cell contract) →
red-first: a fixture exercising the new field/shape in `selftest.py`
before it exists → the change to `cells.json`, `run.py`,
`CMakeLists.txt`, `acceptance_manifest.py` → a card window running
`--cell pruefstand` for real, producing a genuine record → `run.py --all`
with nothing named on `--allow-skip` → review (a host name, path or
credential in any tracked file) → commit → DESIGN §7.0.2x record,
CHANGELOG line, the milestone backlog row closed.

## Invariants

No host name, address, unit name or credential enters any tracked file —
restated because it is most at risk here: an operator-supplied command
lives in a CMake cache variable and a generated manifest under the build
directory, never in `cells.json`, `CMakeLists.txt`, or any committed
fixture. The cell's own gate (10/10) does not change; only how the
runner reaches or verifies it does.

## Status

- 2026-09-05 — opened from the 0.3.1 window; nothing started.
- 2026-09-05, closed (shape 1, DESIGN §7.0.2ao; design note
  `docs/design-pruefstand-cell.md`). Recon found the score contract inert
  (`run_external` never read `score_parse`) and the invocation naming a
  wrapper that does not exist. Change: run-manifest key `pruefstand` from
  `ARCINT_ACCEPTANCE_PRUEFSTAND` (empty by default); the cell's external
  block names the key only; the environment is no channel; the wrapper
  prints `ACCEPTANCE-METRIC score <n> points` and the cell's reference
  gates it at 10 through §8.2's existing comparison; `validate_cells`
  checks the external shape; `timeout_seconds` 1800. Red first: eight
  self-test assertions failed on the old runner, 58 pass after. Real run
  from the dev host against the deployed package (0.2.12-1+p3, its own
  port, units untouched): `PUNKTE 10/10`, `PASS pruefstand/score: 10
  points`, exit 0. `run.py --all` no longer needs `--allow-skip pruefstand`
  where the wrapper is configured.
