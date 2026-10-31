#!/usr/bin/env bash
# Re-capture the chat and completions routes (review R5) and compare them with
# the checked-in capture. Byte-identical except the two normalised fields named
# in normalise.py. Runs under the unit ctest label.
set -uo pipefail

BIN="${1:-}"
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "usage: compare.sh /path/to/arcint" >&2
  exit 2
fi

HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if ! "$HERE/capture.sh" "$BIN" "$TMP/capture" >"$TMP/capture.log" 2>&1; then
  echo "feyntalk golden capture failed to run:" >&2
  cat "$TMP/capture.log" >&2
  exit 1
fi

fail=0
for body in "$HERE"/expected/*.body; do
  name=$(basename "$body" .body)
  if ! cmp -s "$HERE/expected/${name}.body" "$TMP/capture/${name}.body"; then
    echo "FAIL ${name}: response body differs from the golden capture" >&2
    diff -u "$HERE/expected/${name}.body" "$TMP/capture/${name}.body" >&2 || true
    fail=$((fail + 1))
  fi
  if ! cmp -s "$HERE/expected/${name}.status" "$TMP/capture/${name}.status"; then
    echo "FAIL ${name}: status differs ($(cat "$HERE/expected/${name}.status") -> $(cat "$TMP/capture/${name}.status"))" >&2
    fail=$((fail + 1))
  fi
done

count=$(ls "$HERE"/expected/*.body 2>/dev/null | wc -l)
if [[ "$fail" == 0 ]]; then
  echo "feyntalk golden: $count route captures byte-identical (ids/created normalised)"
else
  echo "feyntalk golden: $fail of $((count * 2)) checks failed"
  exit 1
fi
