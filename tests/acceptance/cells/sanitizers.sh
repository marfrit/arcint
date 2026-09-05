#!/usr/bin/env bash
# sanitizers cell (§5; docs/design-0.3.1-test-ladder.md Increment 2). The
# unit set (`unit`, `roundtrip`, `stress`) is entirely device-free, so it can
# run instrumented on any box with a compiler -- which is also why this is
# its own cell rather than folded into a plain `ctest`: a report here is a
# memory-safety or undefined-behaviour defect in arcint's own code, never a
# card behaving oddly.
#
#   sanitizers.sh <arcint>
#
# The binary argument locates the source tree only (derived from this
# script's own path, below) -- its own build need not be sanitizer-instrumented,
# and this cell configures and builds a separate one from scratch.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

BIN="${1:?usage: sanitizers.sh <arcint>}"   # location only, see header
if [[ ! -e "$BIN" ]]; then
  echo "sanitizers: '$BIN' does not exist (only used to confirm the caller passed something)" >&2
fi

for tool in cmake ctest curl python3; do
  # curl and python3 are not used by this script directly, but the
  # roundtrip/stress sub-tests this cell's ctest run includes need them, and
  # a missing tool there is this cell's environmental precondition to name,
  # not a sanitizer report to bury it under.
  command -v "$tool" >/dev/null || { echo "sanitizers: $tool is required" >&2; exit 77; }
done
CXX="${CXX:-c++}"
command -v "$CXX" >/dev/null || { echo "sanitizers: a C++ compiler ($CXX) is required" >&2; exit 77; }

ARCH="$(uname -m)"
if [[ "$ARCH" == "x86_64" ]]; then
  SANFLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer"
  KIND="ASan+UBSan (x86_64)"
else
  SANFLAGS="-fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer"
  KIND="UBSan only ($ARCH; this cell's ASan arm is x86_64-only)"
fi

echo "== sanitizers cell: $KIND"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Probe first: a missing sanitizer runtime on this host is an environmental
# precondition, not a defect in arcint, so it is a skip rather than a failure.
PROBE_SRC="$WORK/probe.cpp"
cat > "$PROBE_SRC" <<'EOF'
int main() { return 0; }
EOF
# shellcheck disable=SC2086 -- SANFLAGS is deliberately unquoted, multi-word
if ! "$CXX" $SANFLAGS "$PROBE_SRC" -o "$WORK/probe" > "$WORK/probe.log" 2>&1; then
  echo "sanitizers: $CXX does not support '$SANFLAGS' on this host:" >&2
  tail -20 "$WORK/probe.log" >&2
  exit 77
fi

# LeakSanitizer needs ptrace, which some containers deny; a probe answers
# whether it can even start here, rather than the ctest run below discovering
# that the hard way and misreporting an environmental limitation as a leak.
# UBSan-only hosts never load LSan at all, so the probe is moot there.
DETECT_LEAKS=0
LEAK_STATUS="not applicable (this host's arm is UBSan-only, no ASan/LSan)"
if [[ "$ARCH" == "x86_64" ]]; then
  LEAK_PROBE_SRC="$WORK/leak_probe.cpp"
  cat > "$LEAK_PROBE_SRC" <<'EOF'
#include <cstdlib>
int main() { void* p = malloc(64); (void)p; return 0; }
EOF
  if "$CXX" -fsanitize=address -fno-omit-frame-pointer "$LEAK_PROBE_SRC" -o "$WORK/leak_probe" \
       > "$WORK/leak_probe_build.log" 2>&1; then
    if "$WORK/leak_probe" > "$WORK/leak_probe_run.log" 2>&1; then
      DETECT_LEAKS=1
      LEAK_STATUS="enabled (a trivial ASan program ran cleanly, so LeakSanitizer starts here)"
    elif grep -aqi "leaksanitizer" "$WORK/leak_probe_run.log"; then
      DETECT_LEAKS=0
      LEAK_STATUS="disabled (LeakSanitizer could not start: $(grep -a -m1 -i "leaksanitizer" "$WORK/leak_probe_run.log"))"
    else
      DETECT_LEAKS=1
      LEAK_STATUS="enabled (the probe exited nonzero for an unrelated reason, not an LSan startup failure)"
    fi
  else
    DETECT_LEAKS=0
    LEAK_STATUS="disabled (could not build the probe, see $WORK/leak_probe_build.log)"
  fi
fi
echo "-- LeakSanitizer probe: $LEAK_STATUS"

BUILD="$WORK/build"
echo "-- configuring a fresh device-free build in a temp dir"
if ! cmake -S "$REPO_ROOT" -B "$BUILD" \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_CXX_FLAGS="$SANFLAGS" \
       -DARCINT_TESTS=ON \
       -DARCINT_OPENVINO=OFF \
       -DARCINT_ACCEPTANCE=OFF \
       > "$WORK/configure.log" 2>&1; then
  echo "  FAIL configure"
  tail -40 "$WORK/configure.log"
  exit 1
fi
echo "  ok   configure"

echo "-- building"
if ! cmake --build "$BUILD" -j "$(nproc)" > "$WORK/build.log" 2>&1; then
  echo "  FAIL build"
  tail -60 "$WORK/build.log"
  exit 1
fi
echo "  ok   build"

echo "-- ctest --no-tests=error -L unit"
if [[ "$ARCH" == "x86_64" ]]; then
  ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=$DETECT_LEAKS}" \
    ctest --test-dir "$BUILD" --output-on-failure --no-tests=error -L unit \
    > "$WORK/ctest.log" 2>&1
else
  ctest --test-dir "$BUILD" --output-on-failure --no-tests=error -L unit \
    > "$WORK/ctest.log" 2>&1
fi
CTEST_RC=$?
tail -40 "$WORK/ctest.log"

REPORTS=$(grep -a -c -E 'SUMMARY: (Address|Undefined|Leak)Sanitizer' \
              "$WORK/ctest.log" "$WORK/build.log" 2>/dev/null \
          | awk -F: '{s += $2} END{print s + 0}')

if [[ "$CTEST_RC" -eq 0 ]]; then
  echo "  ok   ctest --no-tests=error -L unit is green"
else
  echo "  FAIL ctest --no-tests=error -L unit exited $CTEST_RC"
fi

if [[ "$REPORTS" -eq 0 ]]; then
  echo "  ok   zero sanitizer reports"
else
  echo "  FAIL zero sanitizer reports ($REPORTS found)"
  grep -a -n -E 'ERROR: (Address|Leak|Undefined)|runtime error:|SUMMARY: (Address|Undefined|Leak)Sanitizer' \
       "$WORK/ctest.log" "$WORK/build.log" 2>/dev/null | head -20
fi

echo
if [[ "$CTEST_RC" -eq 0 && "$REPORTS" -eq 0 ]]; then
  echo "sanitizers: all checks passed"
  exit 0
fi
echo "sanitizers: check(s) failed"
exit 1
