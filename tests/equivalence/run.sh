#!/usr/bin/env bash
# The equivalence suite (DESIGN.md §5): the arms and their gates are in run.py.
#   run.sh <arcint> <model.gguf> [device]
# ARCINT_EXTRA_ARGS is appended to every server start (the unit's KV and MTP
# flags), which is how the acceptance cells run it per service.
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: run.sh <arcint> <model.gguf> [device]" >&2; exit 2; }
exec python3 "$(dirname "$0")/run.py" "$@"
