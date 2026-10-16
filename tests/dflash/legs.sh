#!/usr/bin/env bash
# DFlash2 on the dense agent, the card legs: the arms side by side (served)
# and the red cases (red); the cells and their gates are in legs.py.
#   legs.sh served <arcint> <agent.gguf> <drafter.gguf> <assets-dir> [options]
#   legs.sh red <red-dir> <arcint> <agent.gguf> <drafter.gguf> <assets-dir> [options]
# Stop the unit that holds the card first (docs/sop-card-window.md).
set -euo pipefail
[ $# -ge 5 ] || { python3 "$(dirname "$0")/legs.py" --help; exit 2; }
exec python3 "$(dirname "$0")/legs.py" "$@"
