#!/usr/bin/env bash
# The lane gates (DESIGN.md §4.1, §5). Same invocation as the equivalence
# suite, and the same requirement: it runs where the card is.
#
#   run.sh /path/to/arcint <model.gguf> [device]
set -uo pipefail
exec python3 "$(dirname "$0")/run.py" "$@"
