#!/usr/bin/env bash
# depth-ladder (DESIGN §5.1; docs/design-0.3.1-test-ladder.md Increment 2).
# The coder artifact must be able to *load and serve* a long prefill at both
# KV precisions on both cards. This cell does not claim a fit is fast, only
# that it happens: a refusal to fit the context is a real failure of this
# cell, not an environmental skip, because refusing to fit is exactly the
# failure mode this ladder exists to catch before a release.
#
# Runs against whichever devices it is given -- it must never stop or start
# anything by its own initiative, unlike the other cells here, because it is
# the one cell that runs on both cards in the same invocation.
#
#   depth_ladder.sh <arcint> <model-dir> <device-large> <device-small> \
#                    [--prefill-tokens N]
#
# <model-dir> is a specific artifact directory (cells.json's own job is to
# say which one: "{model_root}/qwen36-coder-b5-ov"), not a root this script
# joins a hardcoded name onto -- every cell's artifact belongs in the
# enumeration, not baked into a runner.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

BIN="${1:?usage: depth_ladder.sh <arcint> <model-dir> <device-large> <device-small> [--prefill-tokens N]}"
MODEL="${2:?usage: depth_ladder.sh <arcint> <model-dir> <device-large> <device-small> [--prefill-tokens N]}"
DEV_LARGE="${3:?usage: depth_ladder.sh <arcint> <model-dir> <device-large> <device-small> [--prefill-tokens N]}"
DEV_SMALL="${4:?usage: depth_ladder.sh <arcint> <model-dir> <device-large> <device-small> [--prefill-tokens N]}"
shift 4

PREFILL_TOKENS=98147
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefill-tokens) PREFILL_TOKENS="${2:?--prefill-tokens needs a value}"; shift 2 ;;
    *) echo "depth_ladder: unrecognized argument '$1'" >&2; exit 2 ;;
  esac
done
if ! [[ "$PREFILL_TOKENS" =~ ^[0-9]+$ ]] || [[ "$PREFILL_TOKENS" -lt 1 ]]; then
  echo "depth_ladder: --prefill-tokens must be a positive integer, got '$PREFILL_TOKENS'" >&2
  exit 2
fi

for tool in python3 curl; do
  command -v "$tool" >/dev/null || { echo "depth_ladder: $tool is required" >&2; exit 77; }
done

[[ -d "$MODEL" ]] || { echo "depth_ladder: model dir $MODEL not found" >&2; exit 77; }

# Room for the prompt plus the 32-token completion plus block-alignment
# slack; the fit deciding this is not enough is the very thing this cell
# gates, so the margin here is generous on purpose.
N_CTX=$((PREFILL_TOKENS + 4096))

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

start_server() {  # start_server <log> <device> <extra args...>
  local log="$1" dev="$2"; shift 2
  stop_server
  PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()")
  "$BIN" --model "$MODEL" --device "$dev" --host 127.0.0.1 --port "$PORT" \
         --n-ctx "$N_CTX" "$@" > "$log" 2>&1 &
  SRV=$!
  # Generous: a cold compile at this depth is minutes even before the fit
  # question is settled.
  for _ in $(seq 1 2400); do
    curl -fsS "http://127.0.0.1:$PORT/health" -o /dev/null 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "server died:"; tail -20 "$log"; return 1; }
    sleep 1
  done
  echo "server never became ready"; return 1
}

ask() {  # ask <outfile> <prompt-file> [max_tokens]
  python3 - "$PORT" "$1" "$2" "${3:-32}" <<'PY'
import json, sys, urllib.error, urllib.request
port, out, pf, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
prompt = open(pf, encoding="utf-8").read()
d = {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
     "max_tokens": n, "chat_template_kwargs": {"enable_thinking": False}}
r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                           data=json.dumps(d).encode(),
                           headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(r, timeout=3600) as resp:
        o = json.load(resp)
except urllib.error.HTTPError as e:
    print(f"       HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}", file=sys.stderr)
    sys.exit(1)
txt = o["choices"][0]["message"]["content"] or ""
u = o.get("usage", {})
open(out, "w", encoding="utf-8").write(txt)
print(f"       prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')}")
sys.exit(0 if txt else 1)
PY
}

lines() {  # lines <log>
  grep -a -E "prefill|decode" "$1" | grep -a "t/s" | tail -4 | sed 's/^/       /'
}

echo "== depth-ladder: ${PREFILL_TOKENS}-token prefill at u8 and u8:i4, on both cards (n-ctx $N_CTX)"

SEED="$REPO_ROOT/tests/acceptance/prompts/filler-seed.txt"

run_cell() {  # run_cell <tag> <device>
  local tag="$1" dev="$2" precision
  precision="${tag#*-}"
  local log="$WORK/$tag.log"
  echo "  -- $tag (device $dev, --paged-kv $precision)"

  if ! start_server "$log" "$dev" --paged-kv "$precision"; then
    fail "$tag: the fit was accepted (server failed to load at n-ctx $N_CTX)"
    return
  fi

  local promptfile="$WORK/$tag-prompt.txt"
  if ! python3 "$REPO_ROOT/tests/acceptance/size_prompt.py" \
         --seed "$SEED" --target "$PREFILL_TOKENS" --tol 2 \
         --url "http://127.0.0.1:$PORT" --out "$promptfile"; then
    fail "$tag: the reference prompt could be sized"
    stop_server
    return
  fi

  local outfile="$WORK/$tag-out.txt"
  if ask "$outfile" "$promptfile" 32 && [[ -s "$outfile" ]]; then
    pass "$tag: served a non-empty completion (HTTP 200)"
  else
    fail "$tag: served a non-empty completion (HTTP 200)"
  fi

  # The fault signatures on the record for this exact prefill depth
  # (tests/test_fit.cpp's packed_values_prefill_scratch_bytes comment,
  # DESIGN §7.0.2ad): a GPU page-fault storm reported as "VM worker error:
  # -12" and "exec queue reset detected", the driver's own "Engine memory
  # CAT error" lines, and the runtime's CL_OUT_OF_RESOURCES. Any of them is
  # a real failure of this cell, not a note.
  fault=$(grep -a -o -E 'CL_OUT_OF_RESOURCES|error: -12|exec queue reset|page fault|CAT error' \
              "$log" | sort -u | tr '\n' ';' | sed 's/;$//')
  if [[ -n "$fault" ]]; then
    fail "$tag: no GPU-fault / out-of-resources line in the log (found: $fault)"
  else
    pass "$tag: no GPU-fault / out-of-resources line in the log"
  fi

  lines "$log"
  stop_server
}

run_cell "large-u8"     "$DEV_LARGE"
run_cell "large-u8:i4"  "$DEV_LARGE"
run_cell "small-u8"     "$DEV_SMALL"
run_cell "small-u8:i4"  "$DEV_SMALL"

echo
if [[ $FAILED -eq 0 ]]; then echo "depth-ladder: all checks passed"; exit 0; fi
echo "depth-ladder: ${FAILED} check(s) failed"; exit 1
