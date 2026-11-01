#!/usr/bin/env bash
# --api-key / --api-key-file: every route but /health needs
# `Authorization: Bearer <key>`, a 401 otherwise; CORS preflight is answered
# without the key. Stub-only: the HTTP half of the device-free unit gate for
# ops item 2.
set -uo pipefail

BIN="${1:-}"
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "usage: auth.sh /path/to/arcint" >&2
  exit 2
fi

for tool in curl python3; do
  command -v "$tool" >/dev/null || { echo "auth: $tool is required" >&2; exit 77; }
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

wait_boot() {
  for _ in $(seq 1 100); do
    curl -fsS "http://127.0.0.1:${PORT}/health" -o /dev/null 2>/dev/null && return 0
    kill -0 "$SRV_PID" 2>/dev/null || { echo "server exited during boot:"; cat "$1"; exit 1; }
    sleep 0.1
  done
  return 1
}

KEY="test-key-not-real"
AUTH="Authorization: Bearer ${KEY}"

echo "== no key: the default surface is unchanged"
PORT=$(free_port)
"$BIN" --stub --host 127.0.0.1 --port "$PORT" --n-ctx 256 -v >"$WORK/off.log" 2>&1 &
SRV_PID=$!
wait_boot "$WORK/off.log" || { echo "off server did not boot" >&2; exit 1; }
BASE="http://127.0.0.1:${PORT}"
code=$(curl -sS -o /dev/null -w '%{http_code}' "$BASE/props")
check "GET /props is 200 without a key" "$([[ "$code" == "200" ]] && echo 0 || echo 1)"
kill "$SRV_PID"; wait "$SRV_PID" 2>/dev/null; SRV_PID=""

echo "== --api-key"
PORT=$(free_port)
"$BIN" --stub --host 127.0.0.1 --port "$PORT" --n-ctx 256 --api-key "$KEY" --metrics -v >"$WORK/key.log" 2>&1 &
SRV_PID=$!
wait_boot "$WORK/key.log" || { echo "key server did not boot" >&2; exit 1; }
BASE="http://127.0.0.1:${PORT}"

code=$(curl -sS -o /dev/null -w '%{http_code}' "$BASE/health")
check "/health is exempt" "$([[ "$code" == "200" ]] && echo 0 || echo 1)"

code=$(curl -sS -o "$WORK/props401.body" -w '%{http_code}' "$BASE/props")
check "GET /props without a key is 401" "$([[ "$code" == "401" ]] && echo 0 || echo 1)"
check "the 401 is an OpenAI-shaped invalid_api_key error" \
  "$(jassert "$WORK/props401.body" 'd["error"]["code"] == "invalid_api_key" and d["error"]["type"] == "invalid_request_error"' && echo 0 || echo 1)"

code=$(curl -sS -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer wrong' "$BASE/props")
check "GET /props with a wrong key is 401" "$([[ "$code" == "401" ]] && echo 0 || echo 1)"

code=$(curl -sS -o "$WORK/props.body" -w '%{http_code}' -H "$AUTH" "$BASE/props")
check "GET /props with the key is 200" "$([[ "$code" == "200" ]] && echo 0 || echo 1)"
check "the key never appears on /props" "$(grep -q "$KEY" "$WORK/props.body" && echo 1 || echo 0)"

code=$(curl -sS -o /dev/null -w '%{http_code}' "$BASE/metrics")
check "GET /metrics without a key is 401" "$([[ "$code" == "401" ]] && echo 0 || echo 1)"
code=$(curl -sS -o /dev/null -w '%{http_code}' -H "$AUTH" "$BASE/metrics")
check "GET /metrics with the key is 200" "$([[ "$code" == "200" ]] && echo 0 || echo 1)"

code=$(curl -sS -o /dev/null -w '%{http_code}' -H "$AUTH" "$BASE/v1/models")
check "GET /v1/models with the key is 200" "$([[ "$code" == "200" ]] && echo 0 || echo 1)"

code=$(curl -sS -o /dev/null -w '%{http_code}' -X OPTIONS "$BASE/props")
check "CORS preflight is answered without the key (204)" "$([[ "$code" == "204" ]] && echo 0 || echo 1)"
allow=$(curl -sS -D - -o /dev/null -X OPTIONS "$BASE/props" | tr -d '\r' | grep -i '^access-control-allow-origin:' | head -1)
check "the preflight allows the origin" "$(echo "$allow" | grep -q '\*' && echo 0 || echo 1)"

cat >"$WORK/chat.json" <<'JSON'
{"model":"stub","messages":[{"role":"user","content":"hello"}]}
JSON
code=$(curl -sS -o /dev/null -w '%{http_code}' -H "$AUTH" -H 'Content-Type: application/json' \
       --data-binary @"$WORK/chat.json" "$BASE/v1/chat/completions")
check "POST /v1/chat/completions with the key is 200" "$([[ "$code" == "200" ]] && echo 0 || echo 1)"

kill "$SRV_PID"; wait "$SRV_PID" 2>/dev/null; SRV_PID=""

echo "== --api-key-file"
printf '%s\n' "$KEY" >"$WORK/key.txt"
PORT=$(free_port)
"$BIN" --stub --host 127.0.0.1 --port "$PORT" --n-ctx 256 --api-key-file "$WORK/key.txt" -v >"$WORK/file.log" 2>&1 &
SRV_PID=$!
wait_boot "$WORK/file.log" || { echo "key-file server did not boot" >&2; exit 1; }
BASE="http://127.0.0.1:${PORT}"
code=$(curl -sS -o /dev/null -w '%{http_code}' -H "$AUTH" "$BASE/props")
check "a key read from a file authenticates" "$([[ "$code" == "200" ]] && echo 0 || echo 1)"
code=$(curl -sS -o /dev/null -w '%{http_code}' "$BASE/props")
check "the file key still refuses a bare request" "$([[ "$code" == "401" ]] && echo 0 || echo 1)"

echo
if [[ "$FAILED" == 0 ]]; then
  echo "auth: all checks green"
else
  echo "auth: $FAILED check(s) failed"
  exit 1
fi
