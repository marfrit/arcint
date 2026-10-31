#!/usr/bin/env bash
# The Open Responses compliance gate inside the repository
# (docs/campaigns/feyntalk.md Gate 3, W5).
#
#   run-compliance.sh <arcint-binary>
#
# Runs the OFFICIAL compliance CLI vendored under openresponses/ (the
# 2026-04-24 release's bin/compliance-test.ts and its reachable library),
# filtered to the seven ids of the 1.3.0 gate, against a --stub server on a
# free local port. The stub gets --queue-timeout 30: the CLI runs its tests in
# parallel and a one-slot stub without a queue answers 429 and flakes.
#
# Dependencies: node >= 22 and the two pinned packages (tsx 4.19.2, zod
# 3.25.76) installed with `npm ci --ignore-scripts` into $FEYNTALK_COMPLIANCE_
# DEPS (default: tests/feyntalk/.deps, git-ignored). When node is absent, or
# npm cannot install, the run is SKIPPED LOUDLY and exits 0 — the offline unit
# gate stays offline. Set FEYNTALK_COMPLIANCE_REQUIRED=1 to make a skip a
# non-zero failure wherever the gate is claimed.
set -uo pipefail

BIN="${1:-}"
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "usage: run-compliance.sh /path/to/arcint" >&2
  exit 2
fi

HERE=$(cd "$(dirname "$0")" && pwd)
IDS="basic-response,streaming-response,system-prompt,tool-calling,multi-turn,assistant-phase,response-output-phase-schema"

skip() {
  echo "SKIP feyntalk-compliance: $*" >&2
  if [[ "${FEYNTALK_COMPLIANCE_REQUIRED:-0}" == "1" ]]; then
    echo "FEYNTALK_COMPLIANCE_REQUIRED=1: the compliance gate is required, failing" >&2
    exit 1
  fi
  exit 0
}

NODE=$(command -v node 2>/dev/null || true)
[[ -n "$NODE" ]] || skip "node is not installed (needs node >= 22, tsx 4.19.2, zod 3.25.76)"
MAJOR=$("$NODE" -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)
[[ "$MAJOR" -ge 22 ]] 2>/dev/null || skip "node $("$NODE" --version 2>/dev/null) is older than 22"

DEPS="${FEYNTALK_COMPLIANCE_DEPS:-$HERE/.deps}"
TSX="$DEPS/node_modules/.bin/tsx"
if [[ ! -x "$TSX" ]]; then
  NPM=$(command -v npm 2>/dev/null || true)
  [[ -n "$NPM" ]] || skip "npm is not installed; cannot install tsx/zod into $DEPS"
  mkdir -p "$DEPS"
  cp "$HERE/package.json" "$HERE/package-lock.json" "$DEPS/"
  if ! "$NPM" ci --ignore-scripts --prefix "$DEPS" >"$DEPS/install.log" 2>&1; then
    echo "npm ci failed; log:" >&2
    tail -20 "$DEPS/install.log" >&2
    skip "could not install the pinned tsx/zod (network or cache unavailable)"
  fi
fi

for tool in curl python3; do
  command -v "$tool" >/dev/null || skip "$tool is required for the stub port/health probe"
done

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
[[ -n "$PORT" ]] || skip "could not pick a free local port"

WORK=$(mktemp -d)
SRV_PID=""
cleanup() {
  if [[ -n "$SRV_PID" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
    kill "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT

"$BIN" --stub --host 127.0.0.1 --port "$PORT" --queue-timeout 30 >"$WORK/stub.log" 2>&1 &
SRV_PID=$!
for _ in $(seq 1 100); do
  curl -fsS "http://127.0.0.1:${PORT}/health" -o /dev/null 2>/dev/null && break
  kill -0 "$SRV_PID" 2>/dev/null || { echo "the stub exited during boot:" >&2; cat "$WORK/stub.log" >&2; exit 1; }
  sleep 0.1
done

echo "feyntalk-compliance: 7 ids against the stub on 127.0.0.1:${PORT} (queue-timeout 30)"
NODE_PATH="$DEPS/node_modules" "$TSX" "$HERE/openresponses/bin/compliance-test.ts" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --api-key sk-arcint --no-bearer --model stub \
  --filter "$IDS" 2>&1 | tee "$WORK/compliance.out"
rc=${PIPESTATUS[0]}
if [[ "$rc" != 0 ]]; then
  echo "feyntalk-compliance: FAILED (exit $rc)" >&2
  exit "$rc"
fi
echo "feyntalk-compliance: 7/7 passed"
