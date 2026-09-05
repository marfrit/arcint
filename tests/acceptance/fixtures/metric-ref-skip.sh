#!/usr/bin/env bash
# A fixture runner for tests/acceptance/selftest.py: a cell with a real,
# declared reference whose runner skips (77) before ever printing a metric
# line -- the case Increment-3's review (2026-09-05, MAJOR 3) named: "a
# cell that exited 77 has no metrics by definition", so the missing-metric
# rule must not turn this SKIP into a FAIL.
echo "fixture: skipping before printing any ACCEPTANCE-METRIC line" >&2
exit 77
