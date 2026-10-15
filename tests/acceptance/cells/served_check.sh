#!/usr/bin/env bash
# The served check of one service (DESIGN.md §5): served_check.py has the
# protocol and the assets it needs.
#   served_check.sh <agent|coder|flashnext> <arcint> <model.gguf> <device> <assets-dir>
set -euo pipefail
exec python3 "$(dirname "$0")/served_check.py" "$@"
