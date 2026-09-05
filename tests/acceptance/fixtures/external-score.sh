#!/usr/bin/env bash
# Selftest stand-in for an operator's Prüfstand wrapper (docs/design-
# pruefstand-cell.md §1): prints the one metric line the contract asks for,
# or nothing, and exits as told.
#   external-score.sh <score|none> [exit-code]
score="${1:-10}"
rc="${2:-0}"
if [[ "$score" != "none" ]]; then
  echo "ACCEPTANCE-METRIC score ${score} points"
fi
exit "$rc"
