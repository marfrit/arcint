#!/usr/bin/env bash
# tier-reference-cell (DESIGN §3.4, §7.0.2ai; docs/design-0.3.1-test-ladder.md
# Increment 2). The static device/host partition the plugin uses under
# --moe-cpu-tier must not change what the model says: two OFF processes and
# two ON processes, all given the same prompt, must all produce the same
# bytes. E2 additionally checks a turn: CONT (the prompt plus one added
# sentence) served from the SAME tier-on process that was already asked
# PROMPT twice -- so its KV state for that prefix is warm -- must match CONT
# served from a FRESH process that has never seen PROMPT at all. That is
# proof that whatever the tier keeps resident travels with the rest of a
# process's KV state, not just the on-device pages.
#
#   tier_reference.sh <arcint> <model-dir> <device> [server args...]
#
# ARCINT_MOE_DEVICE_POOL_BYTES is honoured the same way ARCINT_EXTRA_ARGS is
# in tests/equivalence/run.sh: forwarded through the environment, unparsed,
# to every server this script starts. ARCINT_EXTRA_ARGS itself is NOT
# honoured here and is unset below -- this cell's configuration is entirely
# its own positional args (cells.json), and an ambient ARCINT_EXTRA_ARGS left
# over from another cell in the same shell must not leak into it.
set -uo pipefail
unset ARCINT_EXTRA_ARGS

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

BIN="${1:?usage: tier_reference.sh <arcint> <model-dir> <device> [server args...]}"
MODEL="${2:?usage: tier_reference.sh <arcint> <model-dir> <device> [server args...]}"
DEV="${3:?usage: tier_reference.sh <arcint> <model-dir> <device> [server args...]}"
shift 3
SERVER_ARGS=("$@")

for tool in python3 curl; do
  command -v "$tool" >/dev/null || { echo "tier_reference: $tool is required" >&2; exit 77; }
done
[[ -d "$MODEL" ]] || { echo "tier_reference: model dir $MODEL not found" >&2; exit 77; }

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
  # Prefix cache off throughout: this cell is about the tier, not the disk
  # cache, and E2 needs the same "no prefix cache" configuration to isolate
  # what --moe-cpu-tier itself keeps resident across a turn.
  "$BIN" --model "$MODEL" --device "$DEV" --host 127.0.0.1 --port "$PORT" \
         --prefix-cache-mib 0 "${SERVER_ARGS[@]}" "$@" \
         > "$log" 2>&1 &
  SRV=$!
  # Generous: a cold compile of a 35B MoE artifact under a static partition is
  # minutes, and a suite that times out looks like a failing gate when it is not.
  for _ in $(seq 1 2400); do
    curl -fsS "http://127.0.0.1:$PORT/health" -o /dev/null 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "server died:"; tail -20 "$log"; return 1; }
    sleep 1
  done
  echo "server never became ready"; return 1
}

ask() {  # ask <outfile> <prompt-file> [max_tokens] -- exits non-zero on any
  # HTTP or network failure, so callers must check it rather than let a
  # missing output file be silently read later as "not compared".
  python3 - "$PORT" "$1" "$2" "${3:-64}" <<'PY'
import json, sys, urllib.request
port, out, pf, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
prompt = open(pf, encoding="utf-8").read()
d = {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
     "max_tokens": n, "chat_template_kwargs": {"enable_thinking": False}}
r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                           data=json.dumps(d).encode(),
                           headers={"Content-Type": "application/json"})
o = json.load(urllib.request.urlopen(r, timeout=1800))
u = o.get("usage", {})
open(out, "w", encoding="utf-8").write(o["choices"][0]["message"]["content"] or "")
print(f"       prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')}")
PY
}

lines() {  # lines <log> -- the server's own reported prefill/decode rates (reported, not gated)
  grep -a -E "prefill|decode" "$1" | grep -a "t/s" | tail -4 | sed 's/^/       /'
}

echo "== tier-reference-cell on $DEV (${SERVER_ARGS[*]})"

# ------------------------------------------------------------- prompt sizing
#
# 1,198 tokens is the reference depth (DESIGN §7.0.2ai). Size it against this
# device's own tokenizer -- a byte or word count does not carry over between
# artifacts -- and print what was actually measured rather than assuming it.
SEED="$REPO_ROOT/tests/acceptance/prompts/filler-seed.txt"
PROMPT="$WORK/prompt.txt"
if ! start_server "$WORK/size.log"; then
  fail "a server came up to size the reference prompt"
  exit 1
fi
if ! python3 "$REPO_ROOT/tests/acceptance/size_prompt.py" \
       --seed "$SEED" --target 1198 --tol 3 --url "http://127.0.0.1:$PORT" --out "$PROMPT"; then
  fail "the reference prompt could be sized"
  exit 1
fi
stop_server

CONT="$WORK/cont.txt"
python3 - "$PROMPT" "$CONT" <<'PY'
import sys
prompt = open(sys.argv[1], encoding="utf-8").read()
open(sys.argv[2], "w", encoding="utf-8").write(prompt + " Then add one more sentence about block alignment.")
PY

# ----------------------------------------------------------- OFF / ON arms
#
# Two requests per (fresh) process, four fresh processes: off1, off2 (the
# baseline, twice, to know what "identical" means before the tier is even
# involved) and on1, on2 (--moe-cpu-tier, twice).
for arm in off1 off2 on1 on2; do
  extra=()
  [[ "$arm" == on* ]] && extra=(--moe-cpu-tier)
  echo "  -- $arm"
  if ! start_server "$WORK/$arm.log" "${extra[@]}"; then
    fail "$arm: server started"
    continue
  fi
  ask "$WORK/$arm-run1.txt" "$PROMPT" 64 || fail "$arm run1: request succeeded"
  ask "$WORK/$arm-run2.txt" "$PROMPT" 64 || fail "$arm run2: request succeeded"
  lines "$WORK/$arm.log"
  stop_server
done

# ------------------------------------------------- byte-identity, all arms
#
# Every one of the seven non-baseline outputs must be present AND identical
# to the baseline: a missing file is a FAIL here, never silently dropped
# from the comparison (a run that produced nothing to compare is not a run
# that agreed with anything), and the baseline is never compared with
# itself -- that would always "pass" even if every other request failed.
echo "  -- byte-identity across all arms and both requests, against off1's first run"
BASE="$WORK/off1-run1.txt"
if [[ -s "$BASE" ]]; then
  others=(off1-run2 off2-run1 off2-run2 on1-run1 on1-run2 on2-run1 on2-run2)
  compared=0
  for name in "${others[@]}"; do
    cand="$WORK/$name.txt"
    if [[ ! -f "$cand" ]]; then
      fail "$name byte-identical to off1-run1 (no output was produced)"
      continue
    fi
    compared=$((compared + 1))
    if cmp -s "$BASE" "$cand"; then
      pass "$name byte-identical to off1-run1"
    else
      fail "$name byte-identical to off1-run1"
    fi
  done
  echo "       compared $compared of ${#others[@]} outputs against the off1-run1 baseline"
else
  fail "off1-run1 produced a baseline output to compare everything else against"
fi

# --------------------------------------------------------------------- E2
#
# Tier ON, prefix cache off (already the default above): PROMPT, PROMPT,
# CONT in one process (e2a); a fresh process asked only CONT (e2b).
echo "  -- E2 (tier on, prefix cache off)"
if start_server "$WORK/e2a.log" --moe-cpu-tier; then
  ask "$WORK/e2a-p1.txt" "$PROMPT" 64 || fail "E2: PROMPT request 1 (warm process) succeeded"
  ask "$WORK/e2a-p2.txt" "$PROMPT" 64 || fail "E2: PROMPT request 2 (warm process) succeeded"
  ask "$WORK/e2a-cont.txt" "$CONT" 64 || fail "E2: CONT request (warm process) succeeded"
  lines "$WORK/e2a.log"
  stop_server
else
  fail "E2: the tier-on process for PROMPT/PROMPT/CONT started"
fi

if start_server "$WORK/e2b.log" --moe-cpu-tier; then
  ask "$WORK/e2b-cont.txt" "$CONT" 64 || fail "E2: CONT request (fresh process) succeeded"
  lines "$WORK/e2b.log"
  stop_server
else
  fail "E2: the fresh process asked only CONT started"
fi

if [[ -f "$WORK/e2a-p1.txt" && -f "$WORK/e2a-p2.txt" ]]; then
  cmp -s "$WORK/e2a-p1.txt" "$WORK/e2a-p2.txt" \
    && pass "E2: PROMPT == PROMPT (same process)" \
    || fail "E2: PROMPT == PROMPT (same process)"
else
  fail "E2: PROMPT == PROMPT (same process) (one or both outputs missing)"
fi
if [[ -f "$WORK/e2a-cont.txt" && -f "$WORK/e2b-cont.txt" ]]; then
  cmp -s "$WORK/e2a-cont.txt" "$WORK/e2b-cont.txt" \
    && pass "E2: CONT from the warm process == CONT from a fresh process" \
    || fail "E2: CONT from the warm process == CONT from a fresh process"
else
  fail "E2: CONT from the warm process == CONT from a fresh process (one or both outputs missing)"
fi

echo
if [[ $FAILED -eq 0 ]]; then echo "tier-reference-cell: all checks passed"; exit 0; fi
echo "tier-reference-cell: ${FAILED} check(s) failed"; exit 1
