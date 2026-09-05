#!/usr/bin/env bash
# A fixture runner for tests/acceptance/selftest.py: passes overall, but
# also prints a promoted-skip line -- the same mechanism
# tests/equivalence/run.sh's MTP section uses when the artifact under test
# carries no MTP head -- so run.py's promotion logic has something to name.
echo "fixture: printing a promoted skip on purpose, then passing" >&2
echo "ACCEPTANCE-SKIP mtp-section fixture-always-lacks-a-head"
exit 0
