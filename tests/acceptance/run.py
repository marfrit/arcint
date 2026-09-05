#!/usr/bin/env python3
"""The acceptance-cell runner (DESIGN.md §5.1, docs/design-0.3.1-test-ladder.md §2-3).

ctest's SKIP_RETURN_CODE marks a skip as not-failed, so a bare
`ctest -L acceptance` on a card-less machine would go green with every cell
skipped. This script is the one place that closes that hole: a cell exits
0 (pass), 1 (fail) or 77 (skip, with a printed reason), and in --all mode the
runner itself exits non-zero unless every 77 is named on the command line
with --allow-skip <cell> -- and non-zero, symmetrically, if --allow-skip
names a cell that ran (did not skip) or a name absent from the enumeration,
so the allow-list cannot rot into a blanket exemption.

A cell can also pass overall (exit 0) while a section of it was skipped
internally -- tests/equivalence/run.sh's MTP section, absent a model with an
MTP head, prints a machine-readable `ACCEPTANCE-SKIP <check> <reason>` line
and still exits 0. This runner promotes such a line into a named skip of the
form <cell>/<check>, subject to the same --allow-skip discipline as a
whole-cell skip: unnamed, it fails the --all run; named, it clears. In
--cell mode (what the generated per-cell ctest tests call) a promoted skip
turns an otherwise-passing cell's exit code into 77 too -- there is no
--allow-skip in that mode, so a per-cell ctest entry can never go green on a
skip it never named, whole-cell or promoted alike. The sentinel is
deliberately not "SKIP" alone: tests/harness.h's unit-test harness prints
bare "SKIP %s: %s" lines of its own (tests/test_main.cpp), and the
sanitizers cell tails its ctest log to its own stdout, so a plain "SKIP"
would get a unit-test skip promoted as an acceptance one by accident.

A cell may declare a check's skip in advance: cells.json's `expected_skips`
(docs/design-0.3.1-test-ladder.md §2's amendment -- the coder artifact
carries no MTP head, so tests/equivalence/run.sh's MTP section fires on
every coder-artifact cell that runs it, not as an anomaly but as a known
fact of that artifact). A declared skip is pre-named: it needs no
--allow-skip, prints as `SKIP <cell>/<check> (declared: <reason>)`, and its
ABSENCE is itself a failure -- the enumeration going stale, because the
artifact grew the capability the skip was declared against. Naming a
declared skip on --allow-skip anyway is an error: it is already named, by
definition, so asking to name it again is redundant rot the same way an
unused --allow-skip entry is.

Every runner this script starts (and its own subprocess, for the external
cell) runs in its own process group, so a `timeout_seconds` in cells.json
can be enforced by killing the whole group -- a cell's server survives a
SIGTERM to the wrong process otherwise.

A runner may also print `ACCEPTANCE-METRIC <metric> <value> <unit>` lines
(docs/design-0.3.1-test-ladder.md §8.2), scanned out of the same stream as
ACCEPTANCE-SKIP. run.py -- never the runner -- qualifies each to
`<cell>/<metric>` and compares it against the cell's own `references` in
cells.json: worse than a reference's `gate_at` is a FAIL unless named
`--allow-regress <cell>/<metric>` in --all (--cell has no such flag: a
regression there is exit 1 unconditionally); a metric with no reference, or
a reference nothing printed, is a FAIL regardless of mode -- a gate that can
go silently uncompared measures nothing, which is this mechanism's entire
premise. A cell whose `references` is JSON `null` (not `[]`) opts out
entirely: its metrics are reports, not gates, until the fill window writes
real numbers in (§8's amendment).

    run.py --cells cells.json --manifest manifest.json --cell NAME
    run.py --cells cells.json --manifest manifest.json --all [--allow-skip NAME]... [--allow-regress CELL/METRIC]...
"""
import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading

PASS, FAIL, SKIP = 0, 1, 77

# A runner's own promoted-skip line: "ACCEPTANCE-SKIP <check> <reason>",
# printed to stdout or stderr alongside its normal output. <check> has no
# spaces; the rest of the line is the reason.
SKIP_LINE_RE = re.compile(r"^ACCEPTANCE-SKIP (\S+) (.+)$")

# A runner's own measurement line: "ACCEPTANCE-METRIC <metric> <value> <unit>".
# <metric> and <unit> have no spaces; <value> is parsed as a float below (a
# non-numeric token, including "nan"/"inf", is caught there, not here).
METRIC_LINE_RE = re.compile(r"^ACCEPTANCE-METRIC (\S+) (\S+) (\S+)$")

# Mandatory fields of one cells.json reference entry (§8.1). A reference
# missing any of these is a schema error -- tools/acceptance_manifest.py
# enforces this at --check time; run.py assumes cells.json already passed it,
# but does not trust that blindly (see _reference_shape_error below).
REFERENCE_FIELDS = ("metric", "value", "gate_at", "unit", "direction",
                     "samples", "spread_pct", "config", "prompt_tokens",
                     "design", "measured", "binary")
DIRECTIONS = ("lower-is-worse", "higher-is-worse")

# Two levels up from tests/acceptance/run.py.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_cells(path):
    cells = load_json(path)
    if not isinstance(cells, list):
        raise SystemExit(f"{path}: expected a JSON array of cells")
    by_name = {}
    for cell in cells:
        name = cell.get("name")
        if not name:
            raise SystemExit(f"{path}: a cell is missing its name")
        if name in by_name:
            raise SystemExit(f"{path}: duplicate cell name {name!r}")
        by_name[name] = cell
    return cells, by_name


def resolve(path, source_dir):
    return path if os.path.isabs(path) else os.path.join(source_dir, path)


def format_args(args, manifest):
    out = []
    for a in args:
        try:
            out.append(a.format(**manifest))
        except KeyError as e:
            raise SystemExit(f"cell references unknown manifest key {e}")
    return out


def _killpg(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def _run_streaming(cmd, env, cwd, timeout_seconds, shell=False):
    """Starts cmd in its own process group, streams its combined stdout+stderr
    to our own stdout while scanning for ACCEPTANCE-SKIP and ACCEPTANCE-METRIC
    lines, and -- if timeout_seconds is given -- kills the whole group on
    expiry: SIGTERM, then SIGKILL 30s later if it is still alive (the same
    two-step a wedged test process on this project needs, per the
    card-runner stop_server functions).

    Returns (returncode, named_skips, metrics, timed_out). metrics is a list
    of (metric, value_str, unit) in print order.
    """
    proc = subprocess.Popen(cmd, env=env, cwd=cwd, shell=shell,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True)

    timed_out = threading.Event()
    timer = None
    if timeout_seconds:
        def _on_timeout():
            timed_out.set()
            _killpg(proc.pid, signal.SIGTERM)
            escalate = threading.Timer(30, lambda: _killpg(proc.pid, signal.SIGKILL))
            escalate.daemon = True
            escalate.start()

        timer = threading.Timer(timeout_seconds, _on_timeout)
        timer.daemon = True
        timer.start()

    named_skips = []
    metrics = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        stripped = line.rstrip("\n")
        m = SKIP_LINE_RE.match(stripped)
        if m:
            named_skips.append((m.group(1), m.group(2)))
            continue
        m = METRIC_LINE_RE.match(stripped)
        if m:
            metrics.append((m.group(1), m.group(2), m.group(3)))
    proc.wait()
    if timer:
        timer.cancel()
    return proc.returncode, named_skips, metrics, timed_out.is_set()


def run_external(cell, manifest):
    ext = cell["external"]
    env_var = ext["env"]
    value = os.environ.get(env_var, "")
    if not value:
        return SKIP, "external-harness-not-configured", [], []
    invocation = ext["invocation"].replace(f"${env_var}", value)
    env = dict(os.environ)
    timeout_seconds = cell.get("timeout_seconds")
    try:
        rc, named_skips, metrics, timed_out = _run_streaming(
            invocation, env, REPO_ROOT, timeout_seconds, shell=True)
    except OSError as e:
        return FAIL, f"could not start {env_var}: {e}", [], []
    if timed_out:
        return FAIL, f"timeout ({timeout_seconds}s)", named_skips, metrics
    if rc == SKIP:
        return SKIP, f"{env_var} exited 77", named_skips, metrics
    if rc != 0:
        return FAIL, f"{env_var} exited {rc}", named_skips, metrics
    return PASS, None, named_skips, metrics


def run_cell(cell, manifest, source_dir):
    """Returns (outcome, reason, named_skips, metrics): named_skips is a list
    of (check, reason) pairs promoted from the runner's own `ACCEPTANCE-SKIP
    <check> <reason>` lines; metrics is a list of (metric, value_str, unit)
    parsed from its `ACCEPTANCE-METRIC <metric> <value> <unit>` lines -- both
    regardless of the cell's overall outcome."""
    if cell.get("external"):
        return run_external(cell, manifest)

    runner = cell.get("runner")
    if not runner or not runner.get("cmd"):
        return FAIL, "cell has no runner and is not external", [], []

    cmd = [resolve(runner["cmd"], source_dir)] + format_args(runner.get("args", []), manifest)
    env = dict(os.environ)
    env.update(runner.get("env", {}))
    timeout_seconds = cell.get("timeout_seconds")

    try:
        rc, named_skips, metrics, timed_out = _run_streaming(cmd, env, source_dir, timeout_seconds)
    except OSError as e:
        return FAIL, f"could not start {cmd[0]}: {e}", [], []

    if timed_out:
        return FAIL, f"timeout ({timeout_seconds}s)", named_skips, metrics
    if rc == SKIP:
        return SKIP, "runner exited 77 (no reason line captured on stdout/stderr above)", named_skips, metrics
    if rc == 0:
        return PASS, None, named_skips, metrics
    return FAIL, f"runner exited {rc}", named_skips, metrics


def _reference_shape_error(cell_name, ref):
    """None if ref has every mandatory field and a recognized direction,
    else a one-line description. Schema shape belongs to
    tools/acceptance_manifest.py's --check; this is a second, cheap check so
    a malformed reference fails the run that hits it rather than only the
    next --check (defence in depth, not a duplicate authority)."""
    missing = [f for f in REFERENCE_FIELDS if f not in ref]
    if missing:
        return f"reference {ref.get('metric', '?')!r} missing field(s): {', '.join(missing)}"
    if ref["direction"] not in DIRECTIONS:
        return f"reference {ref['metric']!r} has unknown direction {ref['direction']!r}"
    return None


def compare_metrics(cell_name, cell, metrics):
    """Compares a cell's printed `metrics` (list of (metric, value_str, unit))
    against its declared `references` (docs/design-0.3.1-test-ladder.md §8.2).

    Returns (always_fail, hard_fail, regressed, improved, compared, messages):
      always_fail -- bool: cells.json's own schema is broken for this cell
                    (the "references" key is absent, not null or a list; or
                    one of its reference entries is malformed). This is a
                    defect in the enumeration itself, independent of what the
                    runner did or didn't do, so the caller fails the cell on
                    it unconditionally -- never gated by whether the process
                    itself passed (unlike hard_fail, below).
      hard_fail  -- bool: true if a PRINTED metric is unknown, wrong-unit or
                    non-numeric, or a declared reference was never printed.
                    This one IS about what the runner did (or a skipped/failed
                    runner not reaching the point of printing anything), so
                    the caller only lets it fail an otherwise-PASSing cell
                    (docs/design-0.3.1-test-ladder.md, Increment-3 review
                    2026-09-05, MAJOR 3: "a cell that exited 77 has no metrics
                    by definition" -- a runner that skipped or failed early
                    never got the chance to print its declared references,
                    which is not itself a new defect to report).
      regressed  -- list of qualified "<cell>/<metric>" names that are worse
                    than gate_at. The caller decides, from --allow-regress,
                    whether that is tolerated (also gated on PASS, same as
                    hard_fail: a regression is a kind of hard_fail's opposite
                    number, and equally impossible on a cell that never ran
                    far enough to print the metric).
      improved   -- list of qualified names better than the recorded value.
      compared   -- count of metrics actually compared (excludes hard-fails
                    and cells whose references is null).
      messages   -- printable lines, one per metric plus one per missing
                    reference, in the order to print them.

    `references: null` (as opposed to `[]`) means "not yet filled": every
    printed metric is a report, and a declared-but-unprinted reference cannot
    exist because there are no references. `references: []` means the cell is
    expected to print nothing comparable -- any ACCEPTANCE-METRIC line from it
    is therefore a metric with no reference, i.e. a hard fail.
    """
    if "references" not in cell:
        return (True, False, [], [], 0,
                [f"FAIL {cell_name}: cells.json schema error: cell is missing "
                 "\"references\" (must be a list, or null if not yet filled)"])

    references = cell["references"]
    messages = []
    if references is None:
        for metric, value_str, unit in metrics:
            messages.append(f"REPORT {cell_name}/{metric}: {value_str} {unit} (no reference yet)")
        return False, False, [], [], 0, messages

    by_metric = {}
    for ref in references:
        shape_error = _reference_shape_error(cell_name, ref)
        if shape_error:
            messages.append(f"FAIL {cell_name}: cells.json schema error: {shape_error}")
            return True, False, [], [], 0, messages
        by_metric[ref["metric"]] = ref

    hard_fail = False
    regressed, improved = [], []
    compared = 0
    seen = set()

    for metric, value_str, unit in metrics:
        qualified = f"{cell_name}/{metric}"
        ref = by_metric.get(metric)
        if ref is None:
            messages.append(f"FAIL {qualified}: printed metric has no declared reference")
            hard_fail = True
            continue
        seen.add(metric)
        if unit != ref["unit"]:
            messages.append(f"FAIL {qualified}: unit {unit!r} != reference unit {ref['unit']!r}")
            hard_fail = True
            continue
        try:
            value = float(value_str)
        except ValueError:
            messages.append(f"FAIL {qualified}: non-numeric value {value_str!r}")
            hard_fail = True
            continue
        if math.isnan(value) or math.isinf(value):
            messages.append(f"FAIL {qualified}: non-numeric value {value_str!r}")
            hard_fail = True
            continue

        compared += 1
        gate_at, target, direction = ref["gate_at"], ref["value"], ref["direction"]
        if direction == "lower-is-worse":
            is_regression = value < gate_at
            is_improvement = value > target
        else:  # "higher-is-worse", the only other DIRECTIONS member
            is_regression = value > gate_at
            is_improvement = value < target

        if is_regression:
            regressed.append(qualified)
            messages.append(f"REGRESSED {qualified}: {value} {unit} against gate {gate_at} {unit}")
        elif is_improvement:
            delta = value - target
            improved.append(qualified)
            messages.append(f"PASS {qualified}: {value} {unit}; "
                             f"IMPROVED {delta:+.2f} {unit} vs recorded {target} {unit}")
        else:
            messages.append(f"PASS {qualified}: {value} {unit}")

    for metric in by_metric:
        if metric not in seen:
            messages.append(f"FAIL {cell_name}/{metric}: reference declared but the runner never printed it")
            hard_fail = True

    return False, hard_fail, regressed, improved, compared, messages


def expected_skip_reasons(cell):
    """{check: reason} for a cell's declared `expected_skips` (docs/design-
    0.3.1-test-ladder.md §2's amendment). A declared skip is pre-named: it
    needs no --allow-skip, and the enumeration itself asserts it WILL
    happen -- so its absence (the artifact grew the capability the skip was
    declared against) is a stale-enumeration failure, not silence."""
    return {es["check"]: es["reason"] for es in (cell.get("expected_skips") or [])}


def cmd_cell(args, cells_by_name, manifest, source_dir):
    cell = cells_by_name.get(args.cell)
    if cell is None:
        print(f"unknown cell: {args.cell}", file=sys.stderr)
        return 2
    outcome, reason, named_skips, metrics = run_cell(cell, manifest, source_dir)
    # Captured before any escalation below: whether the process itself
    # passed, independent of anything metrics or declared skips later decide.
    # MAJOR 1/3 of the Increment-3 review (2026-09-05): a cell that skipped
    # or failed on its own terms keeps that outcome and that reason -- a
    # runner that never reached a section cannot be accused of a stale
    # declaration for it, and a runner with no metrics by definition (it
    # exited 77 before printing any) cannot be accused of a missing one.
    outcome_was_pass = (outcome == PASS)

    declared = expected_skip_reasons(cell)
    seen_checks = set()
    for check, check_reason in named_skips:
        seen_checks.add(check)
        if check in declared:
            print(f"SKIP {args.cell}/{check} (declared: {declared[check]})", file=sys.stderr)
        else:
            print(f"SKIP {args.cell}/{check}: {check_reason}", file=sys.stderr)
    missing_declared = [check for check in declared if check not in seen_checks]
    # MAJOR 2: an occurred, declared skip is pre-named -- subtract it from
    # what still needs naming before degrading an otherwise-PASSing cell to
    # 77, exactly as --all subtracts declared_seen from unnamed_skips. A
    # coder cell whose only promoted skip is its declared MTP section stays
    # exit 0 under --cell (so its generated per-cell ctest entry is green).
    undeclared_named_skips = [(check, r) for check, r in named_skips if check not in declared]

    always_fail, hard_fail, regressed, improved, compared, messages = compare_metrics(
        args.cell, cell, metrics)
    for line in messages:
        print(line, file=sys.stderr)
    if always_fail:
        # cells.json's own schema is broken for this cell -- unconditional,
        # never gated on outcome_was_pass (docs/design-0.3.1-test-ladder.md,
        # Increment-3 review item 7): a missing/malformed "references" is a
        # defect in the enumeration, not a claim about what the runner did.
        outcome = FAIL
        reason = "cells.json schema error (see messages above)"
    # No --allow-regress in --cell mode either (docs/design-0.3.1-test-ladder.md
    # §8.2): any regression here fails the cell outright -- but only a cell
    # that otherwise passed; see outcome_was_pass above.
    elif outcome_was_pass and (hard_fail or regressed):
        outcome = FAIL
        extra = "; ".join(regressed) if regressed else "malformed metric(s)"
        reason = f"metric comparison failed: {extra}"

    # A declared expected skip that did not occur is the enumeration going
    # stale -- the artifact this cell serves grew the capability the skip
    # was declared against -- and that is a FAIL, not a quieter outcome
    # (§2's amendment): unlike a metric regression, there is no
    # --allow-skip-shaped escape from it in either mode. Only asserted on a
    # cell that otherwise passed (MAJOR 1): if the runner itself skipped or
    # failed, it may never have reached the section the skip is declared
    # against, and that is not this enumeration's fault.
    if outcome_was_pass and missing_declared:
        for check in missing_declared:
            print(f"FAIL {args.cell}/{check}: declared expected skip never occurred "
                 "(stale enumeration?)", file=sys.stderr)
        outcome = FAIL
        stale = f"declared expected skip(s) never occurred: {', '.join(missing_declared)}"
        reason = f"{reason}; {stale}" if reason else stale

    # No --allow-skip in --cell mode (what the generated per-cell ctest tests
    # call): a promoted, UNDECLARED skip on an otherwise-passing cell still
    # has to come back as 77, or a per-cell ctest entry could go green on a
    # skip nobody ever named or declared (docs/design-0.3.1-test-ladder.md
    # §2). A declared, occurred skip alone (MAJOR 2) does not trigger this.
    if undeclared_named_skips and outcome == PASS:
        outcome = SKIP
        reason = (f"{len(undeclared_named_skips)} promoted skip(s): "
                  f"{', '.join(check for check, _ in undeclared_named_skips)}")
    if outcome == SKIP:
        print(f"SKIP {args.cell}: {reason}", file=sys.stderr)
    elif outcome == FAIL:
        print(f"FAIL {args.cell}: {reason}", file=sys.stderr)
    else:
        print(f"PASS {args.cell}", file=sys.stderr)
    # Printed on every branch (Increment-3 review item 9), not only PASS:
    # a FAILed or SKIPped cell may still have compared metrics worth seeing
    # (e.g. it failed for an unrelated reason after printing them).
    if compared:
        print(f"{args.cell}: {compared} compared, {len(regressed)} regressed, "
              f"{len(improved)} improved", file=sys.stderr)
    return outcome


def cmd_all(args, cells, cells_by_name, manifest, source_dir):
    allow_skip = list(args.allow_skip or [])
    allow_regress = list(args.allow_regress or [])
    # Only a plain cell name can be validated before anything has run; a
    # qualified name (cell/check, from a runner's own promoted SKIP line) is
    # checked against what actually happened, below, like any other skip.
    unknown_allow = [name for name in allow_skip
                     if "/" not in name and name not in cells_by_name]
    for name in unknown_allow:
        print(f"--allow-skip {name}: no such cell", file=sys.stderr)

    # An --allow-regress entry is always a qualified <cell>/<metric> name --
    # unlike --allow-skip, there is no bare-cell form, because a regression is
    # never whole-cell. A name with no "/" is malformed on its face; it cannot
    # match anything the run produces, so it is treated the same as one that
    # named a metric that held (misnamed_allow_regress, below).
    malformed_allow_regress = [name for name in allow_regress if "/" not in name]
    for name in malformed_allow_regress:
        print(f"--allow-regress {name}: not a <cell>/<metric> name", file=sys.stderr)

    # Every declared expected skip, across every cell, qualified. Naming one
    # of these on --allow-skip is always redundant rot -- a declared skip is
    # pre-named by definition, so asking to also name it on the command line
    # cannot mean anything but "I did not know it was already declared" or
    # "this declaration is copy-pasted cruft" (§2's amendment).
    declared_qualified = {f"{cell['name']}/{check}"
                          for cell in cells for check in expected_skip_reasons(cell)}
    redundant_allow_skip = [name for name in allow_skip if name in declared_qualified]
    for name in redundant_allow_skip:
        print(f"--allow-skip {name}: names a declared expected skip "
             "(redundant -- it is already pre-named)", file=sys.stderr)

    outcomes = {}
    named_skips_seen = set()
    declared_seen = set()
    regressed_seen = set()
    total_compared = total_regressed = total_improved = 0
    any_failed = False
    for cell in cells:
        name = cell["name"]
        outcome, reason, named_skips, metrics = run_cell(cell, manifest, source_dir)
        # Captured before any escalation below -- see cmd_cell's own comment
        # (Increment-3 review 2026-09-05, MAJOR 1/3): a cell that skipped or
        # failed on its own terms keeps that outcome and reason.
        outcome_was_pass = (outcome == PASS)

        declared = expected_skip_reasons(cell)
        seen_checks = set()
        for check, check_reason in named_skips:
            qualified = f"{name}/{check}"
            named_skips_seen.add(qualified)
            seen_checks.add(check)
            if check in declared:
                declared_seen.add(qualified)
                print(f"SKIP {qualified} (declared: {declared[check]})", file=sys.stderr)
            else:
                print(f"SKIP {qualified}: {check_reason}", file=sys.stderr)
        missing_declared = [check for check in declared if check not in seen_checks]

        always_fail, hard_fail, regressed, improved, compared, messages = compare_metrics(
            name, cell, metrics)
        for line in messages:
            print(line, file=sys.stderr)
        regressed_seen.update(regressed)
        total_compared += compared
        total_regressed += len(regressed)
        total_improved += len(improved)
        # A regression named on --allow-regress does not turn the cell back
        # to PASS by itself -- it is still worth seeing in the log -- but it
        # must not fail the run for that reason alone; an unnamed one, or any
        # hard_fail (unknown metric, missing metric, bad value/unit),
        # always does -- but, per MAJOR 3, only on a cell that otherwise
        # passed: a skipped or failed cell has no metrics to have gotten
        # wrong. always_fail (cells.json's own schema broken for this cell)
        # is unconditional, same as in cmd_cell (item 7).
        unallowed_regressions = [q for q in regressed if q not in allow_regress]
        if always_fail:
            outcome = FAIL
            reason = "cells.json schema error (see messages above)"
        elif outcome_was_pass and (hard_fail or unallowed_regressions):
            outcome = FAIL
            # Item 8: lists only the regressions actually failing the run,
            # not ones already excused by --allow-regress.
            reason = f"metric comparison failed: {'; '.join(unallowed_regressions) or 'malformed metric(s)'}"

        # A declared expected skip that never happened is the enumeration
        # gone stale (the artifact grew the capability), never a quieter
        # outcome -- there is no --allow-skip-shaped escape from it. Only
        # asserted on a cell that otherwise passed (MAJOR 1): a cell that
        # skipped or failed early may never have reached the section.
        if outcome_was_pass and missing_declared:
            for check in missing_declared:
                print(f"FAIL {name}/{check}: declared expected skip never occurred "
                     "(stale enumeration?)", file=sys.stderr)
            outcome = FAIL
            stale = f"declared expected skip(s) never occurred: {', '.join(missing_declared)}"
            reason = f"{reason}; {stale}" if reason else stale

        outcomes[name] = (outcome, reason)
        if outcome == SKIP:
            print(f"SKIP {name}: {reason}", file=sys.stderr)
        elif outcome == FAIL:
            print(f"FAIL {name}: {reason}", file=sys.stderr)
            any_failed = True
        else:
            print(f"PASS {name}", file=sys.stderr)

    skipped = {name for name, (outcome, _) in outcomes.items() if outcome == SKIP}
    skipped |= named_skips_seen
    # Declared-and-occurred skips are pre-named: they drop out of what
    # unnamed_skips demands naming for, the same way as if --allow-skip had
    # already listed them.
    unnamed_skips = skipped - set(allow_skip) - declared_seen
    for name in unnamed_skips:
        print(f"unnamed skip: {name} (pass --allow-skip {name} to acknowledge it)",
              file=sys.stderr)

    # Symmetric with unnamed_skips: an --allow-skip entry that named
    # something recognizable (a known cell, or any qualified name -- those
    # cannot be validated up front) but that did not actually skip is just as
    # much rot as an unacknowledged skip, whole-cell or promoted alike.
    # Declared names are excluded here too -- they get their own, more
    # specific "redundant" message above regardless of whether they skipped.
    misnamed_allow = [name for name in allow_skip
                      if name not in unknown_allow and name not in skipped
                      and name not in declared_qualified]
    for name in misnamed_allow:
        print(f"--allow-skip {name}: did not skip", file=sys.stderr)

    # Same rule, second domain (docs/design-0.3.1-test-ladder.md §8.2): a
    # regression once excused stays excused only for as long as it keeps
    # happening -- an --allow-regress entry that named a metric which held,
    # or one the run never even saw, is rot and fails the run.
    unnamed_regressions = regressed_seen - set(allow_regress)
    for name in unnamed_regressions:
        print(f"unnamed regression: {name} (pass --allow-regress {name} to acknowledge it)",
              file=sys.stderr)
    misnamed_allow_regress = [name for name in allow_regress
                              if name not in malformed_allow_regress
                              and name not in regressed_seen]
    for name in misnamed_allow_regress:
        print(f"--allow-regress {name}: did not regress", file=sys.stderr)

    ran = len(outcomes)
    passed = sum(1 for outcome, _ in outcomes.values() if outcome == PASS)
    whole_cell_skips = {name for name, (outcome, _) in outcomes.items() if outcome == SKIP}
    print(f"\n{ran} cell(s) run, {passed} passed, {len(whole_cell_skips)} skipped, "
          f"{len(named_skips_seen)} named skip(s) promoted, "
          f"{ran - passed - len(whole_cell_skips)} failed; "
          f"{total_compared} compared, {total_regressed} regressed, "
          f"{total_improved} improved", file=sys.stderr)

    if (any_failed or unnamed_skips or misnamed_allow or unknown_allow
            or unnamed_regressions or misnamed_allow_regress or malformed_allow_regress
            or redundant_allow_skip):
        return 1
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "cells.json"),
                   help="path to the cell enumeration (default: cells.json beside this script)")
    p.add_argument("--manifest", required=True,
                   help="path to the run manifest CMake generates at configure time")
    p.add_argument("--source-dir", default=REPO_ROOT,
                   help="root relative runner paths in cells.json are resolved against")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cell", help="run exactly this cell")
    mode.add_argument("--all", action="store_true", help="run every cell in the enumeration")
    p.add_argument("--allow-skip", action="append", default=[],
                   help="acknowledge a named cell's skip (repeatable; --all only)")
    p.add_argument("--allow-regress", action="append", default=[],
                   help="acknowledge a named <cell>/<metric> regression (repeatable; --all only)")
    args = p.parse_args()

    if args.cell and args.allow_skip:
        p.error("--allow-skip applies to --all, not --cell")
    if args.cell and args.allow_regress:
        p.error("--allow-regress applies to --all, not --cell")

    cells, cells_by_name = load_cells(args.cells)
    manifest = load_json(args.manifest)

    if args.cell:
        return cmd_cell(args, cells_by_name, manifest, args.source_dir)
    return cmd_all(args, cells, cells_by_name, manifest, args.source_dir)


if __name__ == "__main__":
    sys.exit(main())
