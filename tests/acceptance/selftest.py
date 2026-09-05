#!/usr/bin/env python3
"""Self-test for tests/acceptance/run.py (docs/design-0.3.1-test-ladder.md §5, §8).

Assertions against fixtures/cells-fake.json, run through run.py itself
rather than reimplementing its logic. Grouped by what they exercise, not
hand-numbered: TOTAL below counts check() calls as they run and the final
line reports the true count, because a hand-maintained number drifts the
moment one assertion grows a second check() call -- as "far better" below
already has -- and a stale count is worse than none (Increment-3 review,
2026-09-05, item 11).

Skip discipline (§5):

  every cell passing                              -> exit 0
  an unnamed skip                                 -> exit != 0
  the same skip, named via --allow-skip           -> exit 0
  --allow-skip naming a cell that passed          -> exit != 0
  --allow-skip naming an unknown cell             -> exit != 0
  a failing cell                                  -> exit != 0
  a promoted named skip (cell/check), unnamed     -> exit != 0
  the same, named via --allow-skip <cell>/<check> -> exit 0

This is arcint-test's --allow-skip discipline in miniature, run over
run.py's own domain (cells) rather than harness.h's (cases) -- one rule,
checked in two places. The last two are the same discipline applied to a
runner's own internal skip (tests/equivalence/run.sh's MTP section is the
motivating case): a cell can pass overall and still owe a name for a part
of it that was skipped.

Metric comparison (§8.5), each fixture a script printing one or more
`ACCEPTANCE-METRIC` lines against a reference in cells-fake.json:

  at gate_at                                       -> exit 0
  below band, lower-is-worse                       -> exit != 0
  the same, named via --allow-regress <cell/metric> -> exit 0
  above band, higher-is-worse (direction honoured)  -> exit != 0
  far better than the recorded value                -> exit 0, IMPROVED printed
  an unknown metric name, alongside a known one that holds
    (the known metric isolates the unknown-name rule -- item 5: re-proven
    by temporarily deleting that branch in compare_metrics and observing
    this go green there)                            -> exit != 0
  a declared reference the runner never printed     -> exit != 0
  NaN                                               -> exit != 0
  wrong unit                                        -> exit != 0
  --allow-regress naming a metric that held          -> exit != 0
  --allow-regress naming an unknown metric           -> exit != 0
  references: null, runner prints a metric anyway    -> exit 0 (report, not a gate)
  gate_at null (§8.9), value far below the record    -> exit 0, REPORT printed,
                                                        0 compared
  --allow-regress naming a report-only metric        -> exit != 0, "did not regress"
  gate_at null, runner never prints the metric       -> exit != 0, missing-metric rule

Declared expected skips (§2's amendment: a coder-artifact cell's MTP section
is a KNOWN skip, not an anomaly to keep re-naming on every command line):

  a declared expected skip that fires    -> exit 0, no --allow-skip needed
  the same, declared but never fires     -> exit != 0 (stale enumeration)
  --allow-skip naming a declared one     -> exit != 0 (redundant)

Increment-3 review fixes (2026-09-05), each proving one MAJOR:

  MAJOR 2 -- --cell on a declared, occurred skip stays exit 0 (only an
             UNDECLARED promoted skip may degrade --cell to 77)
  MAJOR 3 -- a cell with a real, declared reference whose runner skips (77)
             before printing anything stays SKIP, not FAIL: unnamed it still
             fails the run (the ordinary unnamed-skip rule), but for the
             skip's own reason, never a fabricated "metric comparison
             failed"; named via --allow-skip it is exit 0
  item  7 -- a cell with no "references" key at all (not null, absent) is a
             schema error at run time too, not only at --check

Plus one schema assertion, exercised directly against
tools/acceptance_manifest.py's validate_cells() rather than through run.py
(that function's authority is --check, not the runner):

  a reference missing a mandatory field -> a non-empty error list
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_PY = os.path.join(HERE, "run.py")
FAKE_CELLS = os.path.join(HERE, "fixtures", "cells-fake.json")
FAKE_MANIFEST = os.path.join(HERE, "fixtures", "manifest.json")

sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import acceptance_manifest  # noqa: E402  (path set up just above)

FAILED = []
TOTAL = 0  # incremented by check() itself, so the final count can never drift
           # from what actually ran (item 11: a hand-maintained number is the
           # defect the docstring above stopped doing on purpose)


def subset(names):
    with open(FAKE_CELLS, encoding="utf-8") as f:
        cells = json.load(f)
    picked = [c for c in cells if c["name"] in names]
    assert len(picked) == len(names), f"fixture is missing one of {names}"
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(picked, f)
    return path


def run(cells_path, allow_skip=(), allow_regress=()):
    return run_capture(cells_path, allow_skip, allow_regress)[0]


def run_capture(cells_path, allow_skip=(), allow_regress=()):
    """Like run(), but also returns run.py's own stderr (where its PASS /
    FAIL / REGRESSED / IMPROVED lines land -- the runner's own raw output
    goes to run.py's stdout instead, via _run_streaming's passthrough)."""
    cmd = [RUN_PY, "--cells", cells_path, "--manifest", FAKE_MANIFEST,
           "--source-dir", HERE, "--all"]
    for name in allow_skip:
        cmd += ["--allow-skip", name]
    for name in allow_regress:
        cmd += ["--allow-regress", name]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stderr


def run_cell_mode(cells_path, cell_name):
    """run.py --cell <cell_name> against cells_path -- MAJOR 2 needs this
    mode specifically: --all's per-cell outcome is never escalated to 77 by
    a promoted skip in the first place (only the aggregate unnamed-skip
    check can fail the whole run), so only --cell exercises the PASS->SKIP
    degradation declared_seen must be subtracted from."""
    cmd = [RUN_PY, "--cells", cells_path, "--manifest", FAKE_MANIFEST,
           "--source-dir", HERE, "--cell", cell_name]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc.returncode


def check(label, got_zero, expect_zero):
    global TOTAL
    TOTAL += 1
    ok = got_zero == expect_zero
    status = "ok  " if ok else "FAIL"
    print(f"  {status} {label}")
    if not ok:
        FAILED.append(label)


def main():
    pass_only = subset(["always-pass"])
    pass_skip = subset(["always-pass", "always-skip"])
    fail_only = subset(["always-fail"])
    named_skip_only = subset(["prints-a-named-skip"])
    at_gate = subset(["metric-at-gate"])
    below_band = subset(["metric-below-band"])
    above_band = subset(["metric-above-band"])
    far_better = subset(["metric-far-better"])
    unknown_name = subset(["metric-unknown-name"])
    missing = subset(["metric-missing"])
    nan = subset(["metric-nan"])
    wrong_unit = subset(["metric-wrong-unit"])
    null_refs = subset(["metric-null-refs"])
    declared_fires = subset(["declared-skip-fires"])
    declared_missing = subset(["declared-skip-missing"])
    ref_skip = subset(["metric-ref-but-skips"])
    no_refs_key = subset(["no-references-key"])
    report_only = subset(["metric-report-only"])
    report_only_missing = subset(["metric-report-only-missing"])
    all_metric_fixtures = [pass_only, pass_skip, fail_only, named_skip_only,
                           at_gate, below_band, above_band, far_better,
                           unknown_name, missing, nan, wrong_unit, null_refs,
                           declared_fires, declared_missing, ref_skip, no_refs_key,
                           report_only, report_only_missing]
    try:
        check("all cells pass -> exit 0",
              run(pass_only) == 0, True)
        check("an unnamed skip -> exit != 0",
              run(pass_skip) == 0, False)
        check("the same skip, named -> exit 0",
              run(pass_skip, allow_skip=["always-skip"]) == 0, True)
        # pass_only, not pass_skip: with an unnamed skip in the manifest the
        # exit would be non-zero for that reason alone, and these two rules
        # would measure nothing (review, 2026-09-05).
        check("--allow-skip naming a cell that passed -> exit != 0",
              run(pass_only, allow_skip=["always-pass"]) == 0, False)
        check("--allow-skip naming an unknown cell -> exit != 0",
              run(pass_only, allow_skip=["totally-unknown"]) == 0, False)
        check("a failing cell -> exit != 0",
              run(fail_only) == 0, False)
        check("a promoted named skip (cell/check), unnamed -> exit != 0",
              run(named_skip_only) == 0, False)
        check("the same, named via --allow-skip <cell>/<check> -> exit 0",
              run(named_skip_only, allow_skip=["prints-a-named-skip/mtp-section"]) == 0, True)

        check("metric at gate_at -> exit 0",
              run(at_gate) == 0, True)
        check("metric below band (lower-is-worse) -> exit != 0",
              run(below_band) == 0, False)
        check("the same, named via --allow-regress -> exit 0",
              run(below_band, allow_regress=["metric-below-band/fake-decode"]) == 0, True)
        check("metric above band (higher-is-worse; proves direction honoured) -> exit != 0",
              run(above_band) == 0, False)
        rc, stderr = run_capture(far_better)
        check("metric far better than recorded -> exit 0", rc == 0, True)
        check("... and IMPROVED is printed", "IMPROVED" in stderr, True)
        check("an unknown metric name, alongside a known one that holds -> exit != 0",
              run(unknown_name) == 0, False)
        check("a declared reference the runner never printed -> exit != 0",
              run(missing) == 0, False)
        check("NaN value -> exit != 0",
              run(nan) == 0, False)
        check("wrong unit -> exit != 0",
              run(wrong_unit) == 0, False)
        check("--allow-regress naming a metric that held -> exit != 0",
              run(at_gate, allow_regress=["metric-at-gate/fake-decode"]) == 0, False)
        check("--allow-regress naming an unknown metric -> exit != 0",
              run(at_gate, allow_regress=["totally-unknown-cell/unknown-metric"]) == 0, False)
        check("references: null, runner prints a metric anyway -> exit 0",
              run(null_refs) == 0, True)

        check("a declared expected skip that fires -> exit 0, no --allow-skip needed",
              run(declared_fires) == 0, True)
        check("the same, declared but never fires -> exit != 0 (stale enumeration)",
              run(declared_missing) == 0, False)
        check("--allow-skip naming a declared one -> exit != 0 (redundant)",
              run(declared_fires, allow_skip=["declared-skip-fires/mtp-section"]) == 0, False)

        # MAJOR 2: --cell on a declared, occurred skip must stay exit 0 --
        # --all never exercises this path (its per-cell outcome is never
        # escalated to 77 by a promoted skip; only the aggregate unnamed-skip
        # check can fail the run), so this needs --cell mode specifically.
        check("--cell on a declared expected skip that fires -> exit 0",
              run_cell_mode(declared_fires, "declared-skip-fires") == 0, True)

        # MAJOR 3: a cell with a real, declared reference whose runner skips
        # (77) before printing any metric stays SKIP, not FAIL -- "a cell
        # that exited 77 has no metrics by definition" is not a new defect.
        rc, stderr = run_capture(ref_skip)
        check("a reference'd cell whose runner skips, unnamed -> exit != 0 (skip rule, not metric)",
              rc == 0, False)
        check("... for the skip's own reason, never a fabricated metric failure",
              "metric comparison failed" not in stderr, True)
        check("... named via --allow-skip -> exit 0",
              run(ref_skip, allow_skip=["metric-ref-but-skips"]) == 0, True)

        # item 7: "references" absent (not null) is a schema error at run
        # time too, matching --check's own authority.
        check("a cell with no \"references\" key at all -> exit != 0",
              run(no_refs_key) == 0, False)

        # §8.9: a reference with gate_at null is recorded and reported,
        # never gated -- a value far below the record passes with a REPORT
        # line and no REGRESSED; but the metric must still be printed, a
        # declared report that vanished is the same stale enumeration a
        # missing gated reference is.
        rc, stderr = run_capture(report_only)
        check("report-only reference (gate_at null), value far below record -> exit 0",
              rc == 0, True)
        check("... and REPORT is printed, never REGRESSED",
              "REPORT metric-report-only/fake-prefill" in stderr and "REGRESSED" not in stderr, True)
        check("... and it is not counted as compared",
              "0 compared, 0 regressed" in stderr, True)
        # The next two were non-zero before the mechanism existed too -- one
        # by a TypeError comparing a float against None, one by the missing-
        # metric rule that already applied -- so each asserts the REASON in
        # stderr, not just the exit: a crash must not keep either green
        # (review, 2026-09-05).
        rc, stderr = run_capture(report_only, allow_regress=["metric-report-only/fake-prefill"])
        check("--allow-regress naming a report-only metric -> exit != 0 (nothing to excuse)",
              rc == 0, False)
        check("... for the did-not-regress reason, not a crash",
              "--allow-regress metric-report-only/fake-prefill: did not regress" in stderr, True)
        rc, stderr = run_capture(report_only_missing)
        check("report-only reference the runner never printed -> exit != 0",
              rc == 0, False)
        check("... by the missing-metric rule, not a schema error",
              "reference declared but the runner never printed it" in stderr
              and "schema error" not in stderr, True)
    finally:
        for path in all_metric_fixtures:
            os.unlink(path)

    # --check's schema authority (§8.1's mandatory fields), exercised
    # directly rather than through a subprocess: a reference missing a
    # mandatory field must be rejected, and a well-formed one must not.
    valid_cell = {
        "name": "schema-fixture", "references": [
            {"metric": "m", "value": 1.0, "gate_at": 1.0, "unit": "t/s",
             "direction": "lower-is-worse", "samples": 1, "spread_pct": 0.0,
             "config": "c", "prompt_tokens": 1, "design": "§0",
             "measured": "1970-01-01", "binary": "b"}
        ]
    }
    broken_cell = {
        "name": "schema-fixture", "references": [
            {"metric": "m", "value": 1.0, "unit": "t/s",  # gate_at missing
             "direction": "lower-is-worse", "samples": 1, "spread_pct": 0.0,
             "config": "c", "prompt_tokens": 1, "design": "§0",
             "measured": "1970-01-01", "binary": "b"}
        ]
    }
    report_cell = json.loads(json.dumps(valid_cell))
    report_cell["references"][0]["gate_at"] = None
    text_gate_cell = json.loads(json.dumps(valid_cell))
    text_gate_cell["references"][0]["gate_at"] = "14.8"
    text_value_cell = json.loads(json.dumps(valid_cell))
    text_value_cell["references"][0]["value"] = "16.4"
    check("--check: a well-formed reference -> no schema errors",
          len(acceptance_manifest.validate_cells([valid_cell])) == 0, True)
    check("--check: a reference missing a mandatory field -> schema error",
          len(acceptance_manifest.validate_cells([broken_cell])) == 0, False)
    check("--check: gate_at null (report-only, §8.9) -> no schema errors",
          len(acceptance_manifest.validate_cells([report_cell])) == 0, True)
    check("--check: gate_at as a string -> schema error",
          len(acceptance_manifest.validate_cells([text_gate_cell])) == 0, False)
    check("--check: value as a string -> schema error",
          len(acceptance_manifest.validate_cells([text_value_cell])) == 0, False)

    if FAILED:
        print(f"\n{len(FAILED)} of {TOTAL} assertion(s) failed: {', '.join(FAILED)}",
              file=sys.stderr)
        return 1
    print(f"\nall {TOTAL} assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
