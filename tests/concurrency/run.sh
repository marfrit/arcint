#!/usr/bin/env bash
# The M6 concurrency gates. Same shape of invocation as the equivalence suite,
# and the same requirement: it runs where the card is.
#
#   run.sh /path/to/arcint /models/ov/<artifact> [device]
set -uo pipefail
exec python3 "$(dirname "$0")/run.py" "$@"
