#!/usr/bin/env bash
# ngram-determinism-repeat (DESIGN §7.0.2ae; docs/design-0.3.1-test-ladder.md
# Increment 2). The n-gram lookup drafter is only safe to ship if it cannot
# introduce process-to-process variance: six fresh processes, same prompt,
# greedy, must all produce identical bytes. A drafter that occasionally
# accepts a token the verifier would not have chosen on its own would pass
# every other gate in this repository and fail only here. Six identical
# outputs from a drafter that never actually drafted proves nothing, so
# every process is also required to report a "draft accept" line -- absence
# of one is a FAIL, not a note.
#
#   ngram_determinism.sh <arcint> <model-dir> <device> [server args...] [--repeats N]
#
# --repeats is consumed by this script, not forwarded to the server. The
# cell's own configuration (the 35B artifact's offload + host-tier setup:
# --offload-ratio 50 --moe-cpu-tier) comes through ARCINT_EXTRA_ARGS, the
# same channel tests/equivalence/run.sh honours -- this cell only adds the
# drafter flags on top of whatever ARCINT_EXTRA_ARGS already configures.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

BIN="${1:?usage: ngram_determinism.sh <arcint> <model-dir> <device> [args...]}"
MODEL="${2:?usage: ngram_determinism.sh <arcint> <model-dir> <device> [args...]}"
DEV="${3:?usage: ngram_determinism.sh <arcint> <model-dir> <device> [args...]}"
shift 3

REPEATS=6
SERVER_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeats) REPEATS="${2:?--repeats needs a value}"; shift 2 ;;
    *) SERVER_ARGS+=("$1"); shift ;;
  esac
done

for tool in python3 curl; do
  command -v "$tool" >/dev/null || { echo "ngram_determinism: $tool is required" >&2; exit 77; }
done
[[ -d "$MODEL" ]] || { echo "ngram_determinism: model dir $MODEL not found" >&2; exit 77; }
if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [[ "$REPEATS" -lt 1 ]]; then
  echo "ngram_determinism: --repeats must be a positive integer, got '$REPEATS'" >&2
  exit 2
fi

WORK=$(mktemp -d)
trap 'stop_server; rm -rf "$WORK"' EXIT
SRV=""
FAILED=0

pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1"; FAILED=$((FAILED + 1)); }

stop_server() {  # SIGTERM, a bounded wait, then SIGKILL -- a wedged server on
  # this project has ignored SIGTERM before; a bare kill -TERM; wait would
  # hang this cleanup forever.
  [[ -n "$SRV" ]] || return 0
  kill -TERM "$SRV" 2>/dev/null
  for _ in $(seq 1 30); do
    kill -0 "$SRV" 2>/dev/null || break
    sleep 1
  done
  kill -KILL "$SRV" 2>/dev/null
  wait "$SRV" 2>/dev/null
  SRV=""
}

start_server() {  # start_server <log> <extra args...>
  local log="$1"; shift
  stop_server
  PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()")
  # shellcheck disable=SC2086 -- word splitting is what ARCINT_EXTRA_ARGS is for
  "$BIN" --model "$MODEL" --device "$DEV" --host 127.0.0.1 --port "$PORT" \
         ${ARCINT_EXTRA_ARGS:-} "${SERVER_ARGS[@]}" "$@" > "$log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 2400); do
    curl -fsS "http://127.0.0.1:$PORT/health" -o /dev/null 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "server died:"; tail -20 "$log"; return 1; }
    sleep 1
  done
  echo "server never became ready"; return 1
}

ask() {  # ask <outfile> <prompt-file> [max_tokens] -- exits non-zero on any
  # HTTP or network failure; callers must check it.
  python3 - "$PORT" "$1" "$2" "${3:-64}" <<'PY'
import hashlib, json, sys, urllib.request
port, out, pf, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
prompt = open(pf, encoding="utf-8").read()
d = {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
     "max_tokens": n, "chat_template_kwargs": {"enable_thinking": False}}
r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                           data=json.dumps(d).encode(),
                           headers={"Content-Type": "application/json"})
o = json.load(urllib.request.urlopen(r, timeout=1800))
txt = o["choices"][0]["message"]["content"] or ""
u = o.get("usage", {})
open(out, "w", encoding="utf-8").write(txt)
print(f"       prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')} "
      f"sha256={hashlib.sha256(txt.encode()).hexdigest()[:16]}")
PY
}

# Use the exact prompt tests/equivalence/run.sh uses for its own determinism
# check (its "two greedy runs are byte-identical" gate), read out of that
# script rather than duplicated here -- a raw seed paragraph is not the same
# claim, and a copy would drift the moment run.sh's own prompt changes. The
# assignment is a fixed, known shape (a `PROMPT="..." \` continuation ending
# in a line that does not continue): pull exactly those lines and evaluate
# them, nothing else out of run.sh runs.
EQUIV_RUN_SH="$REPO_ROOT/tests/equivalence/run.sh"
PROMPT_DEF="$(awk '/^PROMPT="/{p=1} p{print; if (!/\\$/) exit}' "$EQUIV_RUN_SH")"
if [[ -z "$PROMPT_DEF" ]]; then
  echo "ngram_determinism: could not find PROMPT= in $EQUIV_RUN_SH" >&2
  exit 1
fi
eval "$PROMPT_DEF"
PROMPT_FILE="$WORK/prompt.txt"
printf '%s' "$PROMPT" > "$PROMPT_FILE"

echo "== ngram-determinism-repeat on $DEV, $REPEATS fresh processes (${ARCINT_EXTRA_ARGS:-} ${SERVER_ARGS[*]})"

for i in $(seq 1 "$REPEATS"); do
  tag="run$i"
  echo "  -- $tag"
  if ! start_server "$WORK/$tag.log"; then
    fail "$tag: server started"
    continue
  fi
  ask "$WORK/$tag.txt" "$PROMPT_FILE" 64 || fail "$tag: request succeeded"
  acc=$(grep -a -o 'draft accept [0-9.]*%' "$WORK/$tag.log" | tail -1)
  if [[ -n "$acc" ]]; then
    echo "       $acc"
  else
    fail "$tag: the drafter fired (no 'draft accept' line in the log)"
  fi
  stop_server
done

echo "  -- byte-identity across all $REPEATS processes"
if [[ -s "$WORK/run1.txt" ]]; then
  for i in $(seq 1 "$REPEATS"); do
    h=$(sha256sum "$WORK/run$i.txt" 2>/dev/null | cut -d' ' -f1)
    echo "       run$i sha256=${h:-MISSING}"
  done
  for i in $(seq 2 "$REPEATS"); do
    if [[ -f "$WORK/run$i.txt" ]] && cmp -s "$WORK/run1.txt" "$WORK/run$i.txt"; then
      pass "run$i byte-identical to run1"
    else
      fail "run$i byte-identical to run1"
    fi
  done
else
  fail "run1 produced a baseline output to compare the rest against"
fi

echo
if [[ $FAILED -eq 0 ]]; then echo "ngram-determinism-repeat: all checks passed"; exit 0; fi
echo "ngram-determinism-repeat: ${FAILED} check(s) failed"; exit 1
