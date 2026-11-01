#!/usr/bin/env bash
# --api-key / --api-key-file: every route but /health needs
# `Authorization: Bearer <key>` (a 401 otherwise), the key is never logged,
# the CORS preflight is answered without it, and a key file that cannot work
# is refused at boot. Stub-only: the HTTP half of the device-free unit gate
# for ops item 2 and its review round.
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
  # A refusal-server that unexpectedly kept running is killed by the test, but
  # guard the common paths too.
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

# expect_code <name> <expected> <method> <path> [data]
expect_code() {
  local name="$1" expected="$2" method="$3" path="$4" data="${5:-}"
  local code
  if [[ "$method" == "POST" ]]; then
    code=$(curl -sS -o /dev/null -w '%{http_code}' -X POST \
                -H 'Content-Type: application/json' --data "${data:-"{}"}" "$BASE$path")
  elif [[ "$method" == "HEAD" ]]; then
    code=$(curl -sS -o /dev/null -w '%{http_code}' -I "$BASE$path")
  else
    code=$(curl -sS -o /dev/null -w '%{http_code}' "$BASE$path")
  fi
  check "$name" "$([[ "$code" == "$expected" ]] && echo 0 || echo 1)"
}

# expect_refusal <name> <keyfile> <grep>  -- the server must exit non-zero
# before it listens, with a clear message naming the problem.
expect_refusal() {
  local name="$1" keyfile="$2" pattern="$3"
  local port; port=$(free_port)
  local log="$WORK/refusal-${name}.log"
  "$BIN" --stub --host 127.0.0.1 --port "$port" --n-ctx 256 --api-key-file "$keyfile" >"$log" 2>&1 &
  local pid=$!
  sleep 0.7
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    fail "$name: the server kept running (auth silently off)"
  elif grep -q "$pattern" "$log"; then
    pass "$name"
  else
    fail "$name: exited but without the expected message"
  fi
}

# expect_warning <name> <key> <grep>  -- the server starts and warns; the
# warning must not echo the key.
expect_warning() {
  local name="$1" key="$2" pattern="$3"
  local port; port=$(free_port)
  local log="$WORK/warn-${name}.log"
  "$BIN" --stub --host 127.0.0.1 --port "$port" --n-ctx 256 --api-key "$key" >"$log" 2>&1 &
  local pid=$!
  sleep 0.7
  if ! kill -0 "$pid" 2>/dev/null; then
    fail "$name: the server exited instead of warning"
    return
  fi
  check "$name" "$(grep -q "$pattern" "$log" && echo 0 || echo 1)"
  check "$name: the warning does not echo the key" "$(grep -qF -- "$key" "$log" && echo 1 || echo 0)"
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
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

expect_code "/health is exempt" "200" GET "/health"

# Every route family, unauthenticated: a 401 before any body is parsed.
expect_code "POST /v1/chat/completions needs the key" "401" POST "/v1/chat/completions"
expect_code "POST /v1/completions needs the key" "401" POST "/v1/completions"
expect_code "POST /v1/responses needs the key" "401" POST "/v1/responses"
expect_code "POST /v1/responses/{id}/cancel needs the key" "401" POST "/v1/responses/resp-x/cancel"
expect_code "POST /v1/audio/speech needs the key" "401" POST "/v1/audio/speech"
expect_code "GET /v1/audio/voices needs the key" "401" GET "/v1/audio/voices"
expect_code "GET /v1/responses/renderer needs the key" "401" GET "/v1/responses/renderer"
expect_code "GET /v1/models needs the key" "401" GET "/v1/models"
expect_code "GET /props needs the key" "401" GET "/props"
expect_code "GET /metrics needs the key" "401" GET "/metrics"
expect_code "HEAD /props needs the key" "401" HEAD "/props"
expect_code "an unknown path 401s (no route enumeration before auth)" "401" GET "/no-such-route"

code=$(curl -sS -o "$WORK/props401.body" -w '%{http_code}' "$BASE/props")
check "GET /props without a key is 401" "$([[ "$code" == "401" ]] && echo 0 || echo 1)"
check "the 401 is an OpenAI-shaped invalid_api_key error" \
  "$(jassert "$WORK/props401.body" 'd["error"]["code"] == "invalid_api_key" and d["error"]["type"] == "invalid_request_error"' && echo 0 || echo 1)"
www=$(curl -sS -D - -o /dev/null "$BASE/props" | tr -d '\r' | grep -i '^www-authenticate:' | head -1)
check "the 401 offers WWW-Authenticate: Bearer (RFC 6750)" \
  "$(echo "$www" | grep -qi '^www-authenticate: *Bearer' && echo 0 || echo 1)"

code=$(curl -sS -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer wrong' "$BASE/props")
check "GET /props with a wrong key is 401" "$([[ "$code" == "401" ]] && echo 0 || echo 1)"
code=$(curl -sS -o /dev/null -w '%{http_code}' -H 'Bearer '"$KEY" "$BASE/props")
check "a bare token without the Bearer scheme is 401" "$([[ "$code" == "401" ]] && echo 0 || echo 1)"

code=$(curl -sS -o "$WORK/props.body" -w '%{http_code}' -H "$AUTH" "$BASE/props")
check "GET /props with the key is 200" "$([[ "$code" == "200" ]] && echo 0 || echo 1)"
check "the key never appears on /props" "$(grep -q "$KEY" "$WORK/props.body" && echo 1 || echo 0)"

code=$(curl -sS -o /dev/null -w '%{http_code}' -H "$AUTH" "$BASE/metrics")
check "GET /metrics with the key is 200" "$([[ "$code" == "200" ]] && echo 0 || echo 1)"
code=$(curl -sS -o /dev/null -w '%{http_code}' -H "$AUTH" "$BASE/v1/models")
check "GET /v1/models with the key is 200" "$([[ "$code" == "200" ]] && echo 0 || echo 1)"
code=$(curl -sS -o /dev/null -w '%{http_code}' -H "$AUTH" "$BASE/no-such-route")
check "an unknown path with the key is 404" "$([[ "$code" == "404" ]] && echo 0 || echo 1)"

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
check "the key never appears in the server log" "$(grep -q "$KEY" "$WORK/key.log" && echo 1 || echo 0)"

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
kill "$SRV_PID"; wait "$SRV_PID" 2>/dev/null; SRV_PID=""

echo "== a key file that can never work is refused at boot"
printf '' >"$WORK/empty.txt"
printf '\n' >"$WORK/newline.txt"
printf 'a\rb\n' >"$WORK/cr.txt"
expect_refusal "an empty key file" "$WORK/empty.txt" "empty"
expect_refusal "a newline-only key file" "$WORK/newline.txt" "empty"
expect_refusal "a CR key file" "$WORK/cr.txt" "carriage return"

echo "== key hygiene warnings (the key is still served)"
expect_warning "leading whitespace warns" " spaced-key" "whitespace"
expect_warning "trailing whitespace warns" "spaced-key " "whitespace"
expect_warning "a percent warns" "key%41" "percent"

echo
if [[ "$FAILED" == 0 ]]; then
  echo "auth: all checks green"
else
  echo "auth: $FAILED check(s) failed"
  exit 1
fi
