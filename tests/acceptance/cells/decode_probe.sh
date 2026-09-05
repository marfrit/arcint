#!/usr/bin/env bash
# decode-probe (DESIGN §5, §7.0.2ai; docs/design-0.3.1-test-ladder.md, part
# of the served-cell split noted in §3). coder-served-large and
# coder-served-small run tests/equivalence/run.sh, which gates byte-equality
# and gates no decode rate at all -- the 60 t/s bar in DESIGN §5 was being
# claimed by cells.json's prose alone. This runner is the actual instrument
# for that bar: one fresh process, the served configuration (no offload),
# two same-prompt requests, and the ACCEPTANCE-METRIC lines run.py can
# compare against a reference. It reuses the same-prompt-twice idiom every
# other cell here already gates on (§3.4), so the decode-rate figure and the
# cold/warm identity claim ride the same two requests, not a third.
#
#   decode_probe.sh <arcint> <model-dir> <device>
#
# ARCINT_EXTRA_ARGS is honoured exactly as tests/equivalence/run.sh honours
# it: appended, unparsed, to the one server this script starts -- the
# served precision (cells.json's own job to state) lives there, not in a
# positional arg, so this runner stays generic across both served cells.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

BIN="${1:?usage: decode_probe.sh <arcint> <model-dir> <device>}"
MODEL="${2:?usage: decode_probe.sh <arcint> <model-dir> <device>}"
DEV="${3:?usage: decode_probe.sh <arcint> <model-dir> <device>}"

for tool in python3 curl; do
  command -v "$tool" >/dev/null || { echo "decode_probe: $tool is required" >&2; exit 77; }
done
[[ -d "$MODEL" ]] || { echo "decode_probe: model dir $MODEL not found" >&2; exit 77; }

WORK=$(mktemp -d)
trap 'stop_server; rm -rf "$WORK"' EXIT
SRV=""
FAILED=0

pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1"; FAILED=$((FAILED + 1)); }

stop_server() {  # SIGTERM, a bounded wait, then SIGKILL -- see the tier and
  # depth-ladder cells' own stop_server for why a bare kill -TERM; wait is
  # not enough on this project.
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

start_server() {  # start_server <log>
  local log="$1"
  stop_server
  PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()")
  # shellcheck disable=SC2086 -- word splitting is what ARCINT_EXTRA_ARGS is for
  "$BIN" --model "$MODEL" --device "$DEV" --host 127.0.0.1 --port "$PORT" \
         ${ARCINT_EXTRA_ARGS:-} > "$log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 2400); do
    curl -fsS "http://127.0.0.1:$PORT/health" -o /dev/null 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "server died:"; tail -20 "$log"; return 1; }
    sleep 1
  done
  echo "server never became ready"; return 1
}

ask() {  # ask <outfile> <prompt-file>
  python3 - "$PORT" "$1" "$2" <<'PY'
import json, sys, urllib.request
port, out, pf = sys.argv[1], sys.argv[2], sys.argv[3]
prompt = open(pf, encoding="utf-8").read()
d = {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
     "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False}}
r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                           data=json.dumps(d).encode(),
                           headers={"Content-Type": "application/json"})
o = json.load(urllib.request.urlopen(r, timeout=1800))
u = o.get("usage", {})
open(out, "w", encoding="utf-8").write(o["choices"][0]["message"]["content"] or "")
print(f"       prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')}")
PY
}

metric_value() {  # metric_value <log> <prefill|decode> <last|last2>
  # A KNOWN index counted from the end, not a blind tail
  # (docs/design-0.3.1-test-ladder.md §8.7): size_prompt.py's own
  # calibration pings share this cell's one server and log, so a fixed
  # index from the START would name a calibration round on a slow-to-
  # converge run. Nothing asks this server anything after the second real
  # request, so "last" and "last2" (the one before it) are sound regardless
  # of how many calibration rounds ran first -- the same reasoning
  # depth_ladder.sh's own metric_value uses for its single request.
  local log="$1" kind="$2" n
  case "$3" in
    last)  n=1 ;;
    last2) n=2 ;;
    *) echo "decode_probe: metric_value: bad index '$3'" >&2; return 1 ;;
  esac
  grep -a -E "$kind" "$log" | grep -a "t/s" | tail -n "$n" | head -n 1 \
    | grep -a -o -E '[0-9]+\.[0-9]+ t/s' | grep -a -o -E '^[0-9]+\.[0-9]+'
}

emit_metric() {  # emit_metric <metric> <value> <unit> -- silent if <value> is empty
  [[ -n "$2" ]] && printf 'ACCEPTANCE-METRIC %s %s %s\n' "$1" "$2" "$3"
}

echo "== decode-probe on $DEV ${ARCINT_EXTRA_ARGS:+(${ARCINT_EXTRA_ARGS})}"

SEED="$REPO_ROOT/tests/acceptance/prompts/filler-seed.txt"
PROMPT="$WORK/prompt.txt"

if ! start_server "$WORK/probe.log"; then
  fail "the server started"
  exit 1
fi

if ! python3 "$REPO_ROOT/tests/acceptance/size_prompt.py" \
       --seed "$SEED" --target 1050 --tol 3 --url "http://127.0.0.1:$PORT" --out "$PROMPT"; then
  fail "the reference prompt could be sized"
  exit 1
fi

req1_ok=1; ask "$WORK/run1.txt" "$PROMPT" || { fail "request 1 (cold) succeeded"; req1_ok=0; }
req2_ok=1; ask "$WORK/run2.txt" "$PROMPT" || { fail "request 2 (warm) succeeded"; req2_ok=0; }

if [[ -f "$WORK/run1.txt" && -f "$WORK/run2.txt" ]]; then
  cmp -s "$WORK/run1.txt" "$WORK/run2.txt" \
    && pass "the two same-prompt requests are byte-identical" \
    || fail "the two same-prompt requests are byte-identical"
else
  fail "the two same-prompt requests are byte-identical (one or both outputs missing)"
fi

# A request's own metrics are emitted only if that request actually
# succeeded (Increment-3 review, 2026-09-05, MAJOR 4): the cell fails
# anyway on a failed request (above), and a failed request's log line, if
# it has one at all, is not a rate to compare against anything -- never a
# calibration line standing in for it either, which is what "last"/"last2"
# already refuse by construction.
if [[ "$req1_ok" == 1 ]]; then
  emit_metric decode-cold-1st  "$(metric_value "$WORK/probe.log" decode  last2)" t/s
  emit_metric prefill-cold-1st "$(metric_value "$WORK/probe.log" prefill last2)" t/s
fi
if [[ "$req2_ok" == 1 ]]; then
  emit_metric decode-warm-2nd  "$(metric_value "$WORK/probe.log" decode  last)"  t/s
  emit_metric prefill-warm-2nd "$(metric_value "$WORK/probe.log" prefill last)"  t/s
fi

echo
if [[ $FAILED -eq 0 ]]; then echo "decode-probe: all checks passed"; exit 0; fi
echo "decode-probe: ${FAILED} check(s) failed"; exit 1
