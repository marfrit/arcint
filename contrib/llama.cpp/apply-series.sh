#!/bin/bash
# Apply arcint's llama.cpp patch series to a tree at the pin, one patch at a time, in byte order.
#
#   apply-series.sh <llama.cpp tree>    apply every patch in patches/, stop at the first that fails
#   apply-series.sh --list              print the patches in the order they are applied
#
# The order is the byte order of the file names (LC_ALL=C), whatever the caller's locale.
# Patch names such as 0129-, 0129a- ... 0129k- and 0130b-, 0130b1- rely on it: a locale's
# collation (en_US.UTF-8, for one) ignores punctuation, sorts 0129-* after 0129k-* and
# 0130b1-* before 0130b-*, and the series then stops at 0129b (GitHub issue #6).
set -euo pipefail

# bash re-reads the locale when LC_ALL is assigned, so the glob below sorts by bytes.
LC_ALL=C
export LC_ALL

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
patches=("$here"/patches/*.patch)
[ -e "${patches[0]}" ] || { echo "apply-series: no patches in $here/patches" >&2; exit 1; }

if [ "${1:-}" = --list ]; then
    printf '%s\n' "${patches[@]}"
    exit 0
fi

tree=${1:?usage: apply-series.sh <llama.cpp tree> | --list}
[ -d "$tree" ] || { echo "apply-series: no directory $tree" >&2; exit 1; }
for p in "${patches[@]}"; do
    (cd "$tree" && git apply --whitespace=nowarn "$p") || { echo "patch does not apply to the pin: $p" >&2; exit 1; }
done
echo "apply-series: ${#patches[@]} patches applied to $tree"
