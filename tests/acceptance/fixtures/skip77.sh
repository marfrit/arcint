#!/usr/bin/env bash
# A fixture runner for tests/acceptance/selftest.py: always skips, with a
# fixed reason, so the runner's skip accounting has something to name.
echo "fixture: skipping on purpose" >&2
exit 77
