# The test ladder: unit and acceptance (built, 0.3.1)

## 1. Two label sets

`ctest` registers the device-free tests with `LABELS unit` (`arcint-test`, the
HTTP round trip, the stub-only stress test, and the acceptance enumeration's
own checks `acceptance-enumeration` and `acceptance-runner`). Card-requiring
cells are registered only under `-DARCINT_ACCEPTANCE=ON`, which configure
refuses without `-DARCINT_OPENVINO=ON`, each with `LABELS acceptance`. Bare
`ctest` in a default build therefore runs exactly the unit set; the package
recipe runs `ctest -L unit`. A unit case asserts only properties of arcint's
own code; where it needs a host capability it probes and skips with a reason,
and `unit` runs with `--max-skips 0`.

## 2. Card-requiring cells and skips

The cells are enumerated in `tests/acceptance/cells.json` (runner, artifact and
card class, flags, gates, reports, references, timeout). Their parameters come
from CMake cache variables (`ARCINT_ACCEPTANCE_MODEL_ROOT`,
`ARCINT_ACCEPTANCE_DEVICE_LARGE`, `ARCINT_ACCEPTANCE_DEVICE_SMALL`,
`ARCINT_ACCEPTANCE_PRUEFSTAND`) written into a generated `run_manifest.json`.
The acceptance entry point is `tests/acceptance/run.py`, never bare `ctest`: a
cell exits 0, 1 or 77 (skip, with a printed reason), and the run fails unless
**every skip is named on the command line with `--allow-skip <cell>`**. Naming
a cell that ran, or one absent from the enumeration, also fails the run. A
cell may declare `expected_skips` (pre-named; their absence fails the cell).
Runners print `ACCEPTANCE-SKIP <check> <reason>` for a skipped section.

## 3. References

A runner prints `ACCEPTANCE-METRIC <metric> <value> <unit>`; `run.py` compares
it against the cell's `references` (`value`, `gate_at`, `direction`,
`samples`, `spread_pct`, `config`, `prompt_tokens`, `design`, `measured`,
`binary`). Worse than `gate_at` fails unless named with
`--allow-regress <cell>/<metric>` in `--all`; a printed metric with no
reference, or a declared reference never printed, fails. `gate_at: null`
records and reports without gating; `references: null` marks a cell not yet
filled.

## 4. Generated files

`tools/acceptance_manifest.py` renders DESIGN §5.1's acceptance block and
`docs/release-checklist.md` from `cells.json`; `--check` regenerates and diffs
(the `acceptance-enumeration` unit test).

Full history: `git show b0447b8:docs/design-0.3.1-test-ladder.md`.
