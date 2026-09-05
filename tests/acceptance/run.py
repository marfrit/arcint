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

    run.py --cells cells.json --manifest manifest.json --cell NAME
    run.py --cells cells.json --manifest manifest.json --all [--allow-skip NAME]...
"""
import argparse
import json
import os
import subprocess
import sys

PASS, FAIL, SKIP = 0, 1, 77

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


def run_external(cell, manifest):
    ext = cell["external"]
    env_var = ext["env"]
    value = os.environ.get(env_var, "")
    if not value:
        return SKIP, "external-harness-not-configured"
    invocation = ext["invocation"].replace(f"${env_var}", value)
    proc = subprocess.run(invocation, shell=True, cwd=REPO_ROOT)
    if proc.returncode == SKIP:
        return SKIP, f"{env_var} exited 77"
    if proc.returncode != 0:
        return FAIL, f"{env_var} exited {proc.returncode}"
    return PASS, None


def run_cell(cell, manifest, source_dir):
    if cell.get("external"):
        return run_external(cell, manifest)

    runner = cell.get("runner")
    if not runner or not runner.get("cmd"):
        return FAIL, "cell has no runner and is not external"

    cmd = [resolve(runner["cmd"], source_dir)] + format_args(runner.get("args", []), manifest)
    env = dict(os.environ)
    env.update(runner.get("env", {}))

    try:
        proc = subprocess.run(cmd, env=env, cwd=source_dir)
    except OSError as e:
        return FAIL, f"could not start {cmd[0]}: {e}"

    rc = proc.returncode
    if rc == SKIP:
        return SKIP, "runner exited 77 (no reason line captured on stdout/stderr above)"
    if rc == 0:
        return PASS, None
    return FAIL, f"runner exited {rc}"


def cmd_cell(args, cells_by_name, manifest, source_dir):
    cell = cells_by_name.get(args.cell)
    if cell is None:
        print(f"unknown cell: {args.cell}", file=sys.stderr)
        return 2
    outcome, reason = run_cell(cell, manifest, source_dir)
    if outcome == SKIP:
        print(f"SKIP {args.cell}: {reason}", file=sys.stderr)
    elif outcome == FAIL:
        print(f"FAIL {args.cell}: {reason}", file=sys.stderr)
    else:
        print(f"PASS {args.cell}", file=sys.stderr)
    return outcome


def cmd_all(args, cells, cells_by_name, manifest, source_dir):
    allow_skip = list(args.allow_skip or [])
    unknown_allow = [name for name in allow_skip if name not in cells_by_name]
    for name in unknown_allow:
        print(f"--allow-skip {name}: no such cell", file=sys.stderr)

    outcomes = {}
    any_failed = False
    for cell in cells:
        name = cell["name"]
        outcome, reason = run_cell(cell, manifest, source_dir)
        outcomes[name] = (outcome, reason)
        if outcome == SKIP:
            print(f"SKIP {name}: {reason}", file=sys.stderr)
        elif outcome == FAIL:
            print(f"FAIL {name}: {reason}", file=sys.stderr)
            any_failed = True
        else:
            print(f"PASS {name}", file=sys.stderr)

    skipped = {name for name, (outcome, _) in outcomes.items() if outcome == SKIP}
    unnamed_skips = skipped - set(allow_skip)
    for name in unnamed_skips:
        print(f"unnamed skip: {name} (pass --allow-skip {name} to acknowledge it)",
              file=sys.stderr)

    misnamed_allow = [name for name in allow_skip
                      if name in cells_by_name and name not in skipped]
    for name in misnamed_allow:
        print(f"--allow-skip {name}: did not skip", file=sys.stderr)

    ran = len(outcomes)
    passed = sum(1 for outcome, _ in outcomes.values() if outcome == PASS)
    print(f"\n{ran} cell(s) run, {passed} passed, {len(skipped)} skipped, "
          f"{ran - passed - len(skipped)} failed", file=sys.stderr)

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
