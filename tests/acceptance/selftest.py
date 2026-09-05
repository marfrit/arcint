#!/usr/bin/env python3
"""Self-test for tests/acceptance/run.py (docs/design-0.3.1-test-ladder.md §5).

Eight assertions against the fixture manifest in fixtures/cells-fake.json
(/bin/true, /bin/false, a 77-exiting script, and a script that passes while
printing a promoted-skip line), run through run.py itself rather than
reimplementing its logic, in the order main() below runs them:

  1. every cell passing                              -> exit 0
  2. an unnamed skip                                 -> exit != 0
  3. the same skip, named via --allow-skip           -> exit 0
  4. --allow-skip naming a cell that passed          -> exit != 0
  5. --allow-skip naming an unknown cell             -> exit != 0
  6. a failing cell                                  -> exit != 0
  7. a promoted named skip (cell/check), unnamed     -> exit != 0
  8. the same, named via --allow-skip <cell>/<check> -> exit 0

This is arcint-test's --allow-skip discipline in miniature, run over
run.py's own domain (cells) rather than harness.h's (cases) -- one rule,
checked in two places. Assertions 7-8 are the same discipline applied to a
runner's own internal skip (tests/equivalence/run.sh's MTP section is the
motivating case): a cell can pass overall and still owe a name for a part
of it that was skipped.
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

FAILED = []


def subset(names):
    with open(FAKE_CELLS, encoding="utf-8") as f:
        cells = json.load(f)
    picked = [c for c in cells if c["name"] in names]
    assert len(picked) == len(names), f"fixture is missing one of {names}"
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(picked, f)
    return path


def run(cells_path, allow_skip=()):
    cmd = [RUN_PY, "--cells", cells_path, "--manifest", FAKE_MANIFEST,
           "--source-dir", HERE, "--all"]
    for name in allow_skip:
        cmd += ["--allow-skip", name]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc.returncode


def check(label, got_zero, expect_zero):
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
    finally:
        os.unlink(pass_only)
        os.unlink(pass_skip)
        os.unlink(fail_only)
        os.unlink(named_skip_only)

    if FAILED:
        print(f"\n{len(FAILED)} assertion(s) failed: {', '.join(FAILED)}", file=sys.stderr)
        return 1
    print("\nall assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
