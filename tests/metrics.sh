#!/usr/bin/env bash
# GET /metrics exists only with --metrics and serves the Prometheus text
# counters/gauges; arcint.timings adds the per-response timing object. Both are
# opt-in, so the default surface is unchanged. Stub-only: this is the HTTP half
# of the device-free unit gate for ops item 1.
set -uo pipefail

BIN="${1:-}"
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "usage: metrics.sh /path/to/arcint" >&2
  exit 2
fi

for tool in curl python3; do
  command -v "$tool" >/dev/null || { echo "metrics: $tool is required" >&2; exit 77; }
done

free_port() {
  python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'
}

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

FAILED=0
pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1"; FAILED=$((FAILED + 1)); }
check() { if [[ "$2" == "0" ]]; then pass "$1"; else fail "$1"; fi; }

jassert() {  # jassert <file> <python expr over `d`>
  python3 - "$1" "$2" <<'PY'
import json, sys
path, expr = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except Exception as e:
    print("parse error:", e, file=sys.stderr)
    sys.exit(1)
sys.exit(0 if eval(expr) else 1)
PY
}

boot() {  # boot <log> <extra args...>
  local log="$1"; shift
  for _ in $(seq 1 100); do
    curl -fsS "http://127.0.0.1:${PORT}/health" -o /dev/null 2>/dev/null && return 0
    kill -0 "$SRV_PID" 2>/dev/null || { echo "server exited during boot:"; cat "$log"; exit 1; }
    sleep 0.1
  done
  return 1
}

PORT=$(free_port)
"$BIN" --stub --host 127.0.0.1 --port "$PORT" --n-ctx 256 -v >"$WORK/off.log" 2>&1 &
SRV_PID=$!
boot "$WORK/off.log" || { echo "off server did not boot" >&2; exit 1; }
BASE="http://127.0.0.1:${PORT}"

echo "== default surface (no --metrics)"
code=$(curl -sS -o "$WORK/off-metrics.body" -w '%{http_code}' "$BASE/metrics")
check "GET /metrics is 404 without --metrics" "$([[ "$code" == "404" ]] && echo 0 || echo 1)"

kill "$SRV_PID"; wait "$SRV_PID" 2>/dev/null; SRV_PID=""

echo "== --metrics"
PORT=$(free_port)
"$BIN" --stub --host 127.0.0.1 --port "$PORT" --n-ctx 256 --metrics -v >"$WORK/on.log" 2>&1 &
SRV_PID=$!
boot "$WORK/on.log" || { echo "metrics server did not boot" >&2; exit 1; }
BASE="http://127.0.0.1:${PORT}"

curl -sS -o "$WORK/metrics.body" -w '%{http_code}' "$BASE/metrics" >"$WORK/metrics.code"
check "GET /metrics is 200" "[[ $(cat "$WORK/metrics.code") == "200" ]] && echo 0 || echo 1"
ctype=$(curl -sS -o /dev/null -w '%{content_type}' "$BASE/metrics")
check "Content-Type is the Prometheus text format" "$([[ "$ctype" == text/plain* ]] && echo 0 || echo 1)"
check "the body carries the request counter" "$(grep -q 'arcint_requests_total' "$WORK/metrics.body" && echo 0 || echo 1)"
check "the body carries the model gauge" "$(grep -q 'arcint_model_loaded' "$WORK/metrics.body" && echo 0 || echo 1)"
curl -fsS "$BASE/health" -o "$WORK/health.json"
check "/health still answers" "$(jassert "$WORK/health.json" 'd["status"] == "ok"' && echo 0 || echo 1)"

cat >"$WORK/chat.json" <<'JSON'
{"model":"stub","messages":[{"role":"user","content":"hello"}],"arcint":{"timings":true}}
JSON
curl -sS -o "$WORK/chat.body" -w '%{http_code}' -H 'Content-Type: application/json' \
     --data-binary @"$WORK/chat.json" "$BASE/v1/chat/completions" >"$WORK/chat.code"
check "chat with arcint.timings is 200" "[[ $(cat "$WORK/chat.code") == "200" ]] && echo 0 || echo 1"
check "the response carries the seven timing fields" \
  "$(jassert "$WORK/chat.body" 'all(k in d.get("timings", {}) for k in ("prompt_n","prompt_ms","predicted_n","predicted_ms","cache_n","draft_n","draft_accepted"))' && echo 0 || echo 1)"
curl -sS "$BASE/metrics" -o "$WORK/metrics2.body"
check "the request counter moved" "$(grep -q 'route="/v1/chat/completions",status="200"' "$WORK/metrics2.body" && echo 0 || echo 1)"

cat >"$WORK/chat-off.json" <<'JSON'
{"model":"stub","messages":[{"role":"user","content":"hello"}]}
JSON
curl -sS -o "$WORK/chat-off.body" -H 'Content-Type: application/json' \
     --data-binary @"$WORK/chat-off.json" "$BASE/v1/chat/completions" >/dev/null
check "an absent arcint.timings adds no timings key" \
  "$(jassert "$WORK/chat-off.body" '"timings" not in d' && echo 0 || echo 1)"

echo
if [[ "$FAILED" == 0 ]]; then
  echo "metrics: all checks green"
else
  echo "metrics: $FAILED check(s) failed"
  exit 1
fi
