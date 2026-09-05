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

Every runner this script starts (and its own subprocess, for the external
cell) runs in its own process group, so a `timeout_seconds` in cells.json
can be enforced by killing the whole group -- a cell's server survives a
SIGTERM to the wrong process otherwise.

    run.py --cells cells.json --manifest manifest.json --cell NAME
    run.py --cells cells.json --manifest manifest.json --all [--allow-skip NAME]...
"""
import argparse
import json
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
    to our own stdout while scanning for ACCEPTANCE-SKIP lines, and -- if
    timeout_seconds is given -- kills the whole group on expiry: SIGTERM,
    then SIGKILL 30s later if it is still alive (the same two-step a wedged
    test process on this project needs, per the card-runner stop_server
    functions).

    Returns (returncode, named_skips, timed_out).
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
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        m = SKIP_LINE_RE.match(line.rstrip("\n"))
        if m:
            named_skips.append((m.group(1), m.group(2)))
    proc.wait()
    if timer:
        timer.cancel()
    return proc.returncode, named_skips, timed_out.is_set()


def run_external(cell, manifest):
    ext = cell["external"]
    env_var = ext["env"]
    value = os.environ.get(env_var, "")
    if not value:
        return SKIP, "external-harness-not-configured", []
    invocation = ext["invocation"].replace(f"${env_var}", value)
    env = dict(os.environ)
    timeout_seconds = cell.get("timeout_seconds")
    try:
        rc, named_skips, timed_out = _run_streaming(invocation, env, REPO_ROOT,
                                                     timeout_seconds, shell=True)
    except OSError as e:
        return FAIL, f"could not start {env_var}: {e}", []
    if timed_out:
        return FAIL, f"timeout ({timeout_seconds}s)", named_skips
    if rc == SKIP:
        return SKIP, f"{env_var} exited 77", named_skips
    if rc != 0:
        return FAIL, f"{env_var} exited {rc}", named_skips
    return PASS, None, named_skips


def run_cell(cell, manifest, source_dir):
    """Returns (outcome, reason, named_skips): named_skips is a list of
    (check, reason) pairs promoted from the runner's own `ACCEPTANCE-SKIP
    <check> <reason>` lines, regardless of the cell's overall outcome."""
    if cell.get("external"):
        return run_external(cell, manifest)

    runner = cell.get("runner")
    if not runner or not runner.get("cmd"):
        return FAIL, "cell has no runner and is not external", []

    cmd = [resolve(runner["cmd"], source_dir)] + format_args(runner.get("args", []), manifest)
    env = dict(os.environ)
    env.update(runner.get("env", {}))
    timeout_seconds = cell.get("timeout_seconds")

    try:
        rc, named_skips, timed_out = _run_streaming(cmd, env, source_dir, timeout_seconds)
    except OSError as e:
        return FAIL, f"could not start {cmd[0]}: {e}", []

    if timed_out:
        return FAIL, f"timeout ({timeout_seconds}s)", named_skips
    if rc == SKIP:
        return SKIP, "runner exited 77 (no reason line captured on stdout/stderr above)", named_skips
    if rc == 0:
        return PASS, None, named_skips
    return FAIL, f"runner exited {rc}", named_skips


def cmd_cell(args, cells_by_name, manifest, source_dir):
    cell = cells_by_name.get(args.cell)
    if cell is None:
        print(f"unknown cell: {args.cell}", file=sys.stderr)
        return 2
    outcome, reason, named_skips = run_cell(cell, manifest, source_dir)
    for check, check_reason in named_skips:
        print(f"SKIP {args.cell}/{check}: {check_reason}", file=sys.stderr)
    # No --allow-skip in --cell mode (what the generated per-cell ctest tests
    # call): a promoted skip on an otherwise-passing cell still has to come
    # back as 77, or a per-cell ctest entry could go green on a skip nobody
    # ever named (docs/design-0.3.1-test-ladder.md §2).
    if named_skips and outcome == PASS:
        outcome = SKIP
        reason = (f"{len(named_skips)} promoted skip(s): "
                  f"{', '.join(check for check, _ in named_skips)}")
    if outcome == SKIP:
        print(f"SKIP {args.cell}: {reason}", file=sys.stderr)
    elif outcome == FAIL:
        print(f"FAIL {args.cell}: {reason}", file=sys.stderr)
    else:
        print(f"PASS {args.cell}", file=sys.stderr)
    return outcome


def cmd_all(args, cells, cells_by_name, manifest, source_dir):
    allow_skip = list(args.allow_skip or [])
    # Only a plain cell name can be validated before anything has run; a
    # qualified name (cell/check, from a runner's own promoted SKIP line) is
    # checked against what actually happened, below, like any other skip.
    unknown_allow = [name for name in allow_skip
                     if "/" not in name and name not in cells_by_name]
    for name in unknown_allow:
        print(f"--allow-skip {name}: no such cell", file=sys.stderr)

    outcomes = {}
    named_skips_seen = set()
    any_failed = False
    for cell in cells:
        name = cell["name"]
        outcome, reason, named_skips = run_cell(cell, manifest, source_dir)
        outcomes[name] = (outcome, reason)
        for check, check_reason in named_skips:
            qualified = f"{name}/{check}"
            named_skips_seen.add(qualified)
            print(f"SKIP {qualified}: {check_reason}", file=sys.stderr)
        if outcome == SKIP:
            print(f"SKIP {name}: {reason}", file=sys.stderr)
        elif outcome == FAIL:
            print(f"FAIL {name}: {reason}", file=sys.stderr)
            any_failed = True
        else:
            print(f"PASS {name}", file=sys.stderr)

    skipped = {name for name, (outcome, _) in outcomes.items() if outcome == SKIP}
    skipped |= named_skips_seen
    unnamed_skips = skipped - set(allow_skip)
    for name in unnamed_skips:
        print(f"unnamed skip: {name} (pass --allow-skip {name} to acknowledge it)",
              file=sys.stderr)

    # Symmetric with unnamed_skips: an --allow-skip entry that named
    # something recognizable (a known cell, or any qualified name -- those
    # cannot be validated up front) but that did not actually skip is just as
    # much rot as an unacknowledged skip, whole-cell or promoted alike.
    misnamed_allow = [name for name in allow_skip
                      if name not in unknown_allow and name not in skipped]
    for name in misnamed_allow:
        print(f"--allow-skip {name}: did not skip", file=sys.stderr)

    ran = len(outcomes)
    passed = sum(1 for outcome, _ in outcomes.values() if outcome == PASS)
    whole_cell_skips = {name for name, (outcome, _) in outcomes.items() if outcome == SKIP}
    print(f"\n{ran} cell(s) run, {passed} passed, {len(whole_cell_skips)} skipped, "
          f"{len(named_skips_seen)} named skip(s) promoted, "
          f"{ran - passed - len(whole_cell_skips)} failed", file=sys.stderr)

    if any_failed or unnamed_skips or misnamed_allow or unknown_allow:
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
    args = p.parse_args()

    if args.cell and args.allow_skip:
        p.error("--allow-skip applies to --all, not --cell")

    cells, cells_by_name = load_cells(args.cells)
    manifest = load_json(args.manifest)

    if args.cell:
        return cmd_cell(args, cells_by_name, manifest, args.source_dir)
    return cmd_all(args, cells, cells_by_name, manifest, args.source_dir)


if __name__ == "__main__":
    sys.exit(main())
