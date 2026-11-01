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
OP_PID=""
cleanup() {
  local pid
  for pid in "$SRV_PID" "$OP_PID"; do
    stop_pid "$pid"
  done
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

# pid_alive: false once the process is gone or a zombie. `kill -0` alone is
# true for a zombie, which made a refused server read as "kept running".
pid_alive() {
  local pid="$1"
  [[ -n "$pid" && -r "/proc/$pid/stat" ]] || return 1
  [[ "$(cut -d' ' -f3 "/proc/$pid/stat" 2>/dev/null)" != "Z" ]]
}

# stop_pid: SIGTERM, a bounded wait, then SIGKILL -- a server whose signal
# handler raced its own listen() must not hang the test in `wait`.
stop_pid() {
  local pid="$1" i
  pid_alive "$pid" || { wait "$pid" 2>/dev/null || true; return; }
  kill "$pid" 2>/dev/null
  for ((i = 0; i < 50; i++)); do
    pid_alive "$pid" || break
    sleep 0.1
  done
  pid_alive "$pid" && kill -9 "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null || true
}

# wait_for <pid> <log> <pattern> <seconds>: bounded, so a slow runner cannot
# read a refusal as "kept running". Returns 0 once the pattern is in the log or
# the process has exited, 1 on the timeout.
wait_for() {
  local pid="$1" log="$2" pattern="$3" seconds="${4:-5}"
  local i
  for ((i = 0; i < seconds * 10; i++)); do
    grep -q "$pattern" "$log" 2>/dev/null && return 0
    pid_alive "$pid" || return 0
    sleep 0.1
  done
  return 1
}

# wait_exit <pid> <seconds>: 0 when the process is gone (or a zombie), 1 on the
# timeout. A refusal prints its message and THEN exits, so the exit is what a
# refusal test must confirm -- a warning line can match the grep while the
# server keeps running.
wait_exit() {
  local pid="$1" seconds="${2:-3}" i
  for ((i = 0; i < seconds * 10; i++)); do
    pid_alive "$pid" || return 0
    sleep 0.1
  done
  return 1
}

# expect_refusal <name> <keyfile> <grep>  -- the server must exit non-zero
# before it listens, with a clear message naming the problem.
expect_refusal() {
  local name="$1" keyfile="$2" pattern="$3"
  local port; port=$(free_port)
  local log="$WORK/refusal-${name}.log"
  "$BIN" --stub --host 127.0.0.1 --port "$port" --n-ctx 256 --api-key-file "$keyfile" >"$log" 2>&1 &
  local pid=$!
  wait_for "$pid" "$log" "$pattern" 5 || true
  if wait_exit "$pid" 3; then
    wait "$pid" 2>/dev/null || true
    if grep -q "$pattern" "$log"; then
      pass "$name"
    else
      fail "$name: exited but without the expected message"
    fi
  else
    stop_pid "$pid"
    fail "$name: the server kept running (auth silently off)"
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
  wait_for "$pid" "$log" "$pattern" 5 || true
  if ! pid_alive "$pid"; then
    fail "$name: the server exited instead of warning"
    return
  fi
  check "$name" "$(grep -q "$pattern" "$log" && echo 0 || echo 1)"
  check "$name: the warning does not echo the key" "$(grep -qF -- "$key" "$log" && echo 1 || echo 0)"
  stop_pid "$pid"
}

# The pre-body refusal: a request whose headers declare a large body but send
# almost none must get its 401 promptly, the socket must close, only
# `Connection: close` is advertised, and no second response appears on the
# connection (the unread body is never parsed).
body_not_read_check() {
  PORT="$PORT" python3 - <<'PY'
import os, socket, sys, time
port = int(os.environ["PORT"])
try:
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
except OSError as e:
    print("  FAIL could not connect:", e)
    sys.exit(1)
head = ("POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Content-Length: 1000000\r\n"
        "Content-Type: application/json\r\n"
        "\r\n")
s.sendall(head.encode() + b"{")   # two body bytes, 999998 missing, no auth
s.settimeout(3.0)
start = time.time()
data = b""
try:
    while b"\r\n\r\n" not in data:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
except socket.timeout:
    pass
elapsed = time.time() - start
header_blob = data.split(b"\r\n\r\n", 1)[0].decode("latin1", "replace")
lower = header_blob.lower()
ok_401 = header_blob.startswith("HTTP/1.1 401") or header_blob.startswith("HTTP/1.0 401")
ok_close = "\r\nconnection: close" in ("\r\n" + lower)
no_keep = "keep-alive" not in lower
# Drain to EOF: it must end, and hold no second response.
rest = b""
eof = False
try:
    while True:
        chunk = s.recv(4096)
        if not chunk:
            eof = True
            break
        rest += chunk
except socket.timeout:
    eof = False
s.close()
second = rest.count(b"HTTP/1.1 ") + rest.count(b"HTTP/1.0 ")
checks = [
    ("the 401 arrives without the missing body (under 2 s)", elapsed < 2.0 and ok_401),
    ("the server closes the connection after the 401", eof),
    ("the response says Connection: close", ok_close),
    ("the response does not also say Keep-Alive", no_keep),
    ("no second response on the connection", second == 0),
]
rc = 0
for name, ok in checks:
    print(("  ok   " if ok else "  FAIL ") + name)
    if not ok:
        rc = 1
sys.exit(rc)
PY
}

# Expect: 100-continue is answered above routing in httplib, so an
# unauthenticated request with that header must still see the 401 first (never
# a 100), and the connection must close.
expect_100_check() {
  PORT="$PORT" python3 - <<'PY'
import os, socket, sys
port = int(os.environ["PORT"])
try:
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
except OSError as e:
    print("  FAIL could not connect:", e)
    sys.exit(1)
head = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        b"Expect: 100-continue\r\nContent-Length: 1000000\r\n"
        b"Content-Type: application/json\r\n\r\n")
s.sendall(head)  # no body, no Authorization
s.settimeout(3.0)
data = b""
try:
    while b"\r\n\r\n" not in data:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
except socket.timeout:
    pass
header = data.split(b"\r\n\r\n", 1)[0].decode("latin1", "replace")
rest = b""
eof = False
try:
    while True:
        chunk = s.recv(4096)
        if not chunk:
            eof = True
            break
        rest += chunk
except socket.timeout:
    eof = False
s.close()
checks = [
    ("Expect: 100-continue with no key gets the 401, not a 100",
     header.startswith("HTTP/1.1 401") and "100 continue" not in header.lower()),
    ("the Expect connection closes after the 401", eof),
]
rc = 0
for name, ok in checks:
    print(("  ok   " if ok else "  FAIL ") + name)
    if not ok:
        rc = 1
sys.exit(rc)
PY
}

# Two requests on ONE socket, both answered: an authenticated (or key-less)
# keep-alive connection survives. AUTH="" for the no-key server.
keepalive_check() {
  PORT="$PORT" AUTH="${1:-}" python3 - <<'PY'
import os, socket, sys
port = int(os.environ["PORT"])
auth = os.environ.get("AUTH", "")
req = ("GET /props HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n"
       + ("Authorization: " + auth + "\r\n" if auth else "")
       + "\r\n").encode()
def read_response(s):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = s.recv(4096)
        if not chunk:
            return None
        data += chunk
    header, _, rest = data.partition(b"\r\n\r\n")
    length = 0
    for line in header.split(b"\r\n")[1:]:
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
    while len(rest) < length:
        chunk = s.recv(4096)
        if not chunk:
            break
        rest += chunk
    return header
try:
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
except OSError as e:
    print("  FAIL could not connect:", e)
    sys.exit(1)
s.sendall(req)
h1 = read_response(s)
s.sendall(req)
h2 = read_response(s)
s.close()
ok1 = bool(h1) and h1.startswith(b"HTTP/1.1 200")
ok2 = bool(h2) and h2.startswith(b"HTTP/1.1 200")
checks = [
    ("the first keep-alive request is answered", ok1),
    ("the second request on the same socket is answered", ok2),
]
rc = 0
for name, ok in checks:
    print(("  ok   " if ok else "  FAIL ") + name)
    if not ok:
        rc = 1
sys.exit(rc)
PY
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
# No key: the response is today's -- no Connection: close, Keep-Alive as
# before -- and a keep-alive connection survives two requests.
hdr=$(curl -sS -D - -o /dev/null "$BASE/props" | tr -d '\r')
check "no key: no Connection: close is sent" "$(echo "$hdr" | grep -qi '^connection: *close' && echo 1 || echo 0)"
check "no key: Keep-Alive is advertised" "$(echo "$hdr" | grep -qi '^keep-alive:' && echo 0 || echo 1)"
keepalive_check "" || FAILED=$((FAILED + 1))
stop_pid "$SRV_PID"; SRV_PID=""

echo "== --api-key"
PORT=$(free_port)
"$BIN" --stub --host 127.0.0.1 --port "$PORT" --n-ctx 256 --api-key "$KEY" --metrics -v >"$WORK/key.log" 2>&1 &
SRV_PID=$!
wait_boot "$WORK/key.log" || { echo "key server did not boot" >&2; exit 1; }
BASE="http://127.0.0.1:${PORT}"

expect_code "/health is exempt for GET" "200" GET "/health"
expect_code "/health is exempt for HEAD" "200" HEAD "/health"
expect_code "POST /health is not exempt (401, not 404)" "401" POST "/health"

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
check "a bare OPTIONS is not a preflight and needs the key (401)" "$([[ "$code" == "401" ]] && echo 0 || echo 1)"
code=$(curl -sS -o /dev/null -w '%{http_code}' -X OPTIONS \
            -H 'Access-Control-Request-Method: POST' "$BASE/props")
check "CORS preflight (Access-Control-Request-Method) is answered (204)" "$([[ "$code" == "204" ]] && echo 0 || echo 1)"
allow=$(curl -sS -D - -o /dev/null -X OPTIONS -H 'Access-Control-Request-Method: POST' "$BASE/props" | tr -d '\r' | grep -i '^access-control-allow-origin:' | head -1)
check "the preflight allows the origin" "$(echo "$allow" | grep -q '\*' && echo 0 || echo 1)"

cat >"$WORK/chat.json" <<'JSON'
{"model":"stub","messages":[{"role":"user","content":"hello"}]}
JSON
code=$(curl -sS -o /dev/null -w '%{http_code}' -H "$AUTH" -H 'Content-Type: application/json' \
       --data-binary @"$WORK/chat.json" "$BASE/v1/chat/completions")
check "POST /v1/chat/completions with the key is 200" "$([[ "$code" == "200" ]] && echo 0 || echo 1)"

keepalive_check "Bearer $KEY" || FAILED=$((FAILED + 1))
expect_100_check || FAILED=$((FAILED + 1))
body_not_read_check || FAILED=$((FAILED + 1))

stop_pid "$SRV_PID"; SRV_PID=""
check "the key never appears in the server log" "$(grep -q "$KEY" "$WORK/key.log" && echo 1 || echo 0)"

echo "== --api-key-file"
printf '%s\n' "$KEY" >"$WORK/key.txt"
chmod 0600 "$WORK/key.txt"
PORT=$(free_port)
"$BIN" --stub --host 127.0.0.1 --port "$PORT" --n-ctx 256 --api-key-file "$WORK/key.txt" -v >"$WORK/file.log" 2>&1 &
SRV_PID=$!
wait_boot "$WORK/file.log" || { echo "key-file server did not boot" >&2; exit 1; }
BASE="http://127.0.0.1:${PORT}"
code=$(curl -sS -o /dev/null -w '%{http_code}' -H "$AUTH" "$BASE/props")
check "a key read from a file authenticates" "$([[ "$code" == "200" ]] && echo 0 || echo 1)"
code=$(curl -sS -o /dev/null -w '%{http_code}' "$BASE/props")
check "the file key still refuses a bare request" "$([[ "$code" == "401" ]] && echo 0 || echo 1)"
check "a private (0600) key file gets no mode warning" "$(grep -q -- '--api-key-file mode' "$WORK/file.log" && echo 1 || echo 0)"
stop_pid "$SRV_PID"; SRV_PID=""

echo "== a group/world-readable key file is warned about"
printf '%s\n' "$KEY" >"$WORK/key-open.txt"
chmod 0644 "$WORK/key-open.txt"
PORT=$(free_port)
log="$WORK/key-open.log"
"$BIN" --stub --host 127.0.0.1 --port "$PORT" --n-ctx 256 --api-key-file "$WORK/key-open.txt" -v >"$log" 2>&1 &
OP_PID=$!
mode_pat='--api-key-file mode'
wait_for "$OP_PID" "$log" "$mode_pat" 5 || true
if ! pid_alive "$OP_PID"; then
  fail "the group/world-readable key file made the server exit"
else
  check "a group/world-readable key file warns" "$(grep -q -- "$mode_pat" "$log" && echo 0 || echo 1)"
  check "the warning names the mode (644)" "$(grep -q '644' "$log" && echo 0 || echo 1)"
  check "the mode warning does not echo the key" "$(grep -qF -- "$KEY" "$log" && echo 1 || echo 0)"
  stop_pid "$OP_PID"
fi
OP_PID=""

echo "== a key file that can never work is refused at boot"
printf '' >"$WORK/empty.txt"
printf '\n' >"$WORK/newline.txt"
printf 'a\rb\n' >"$WORK/cr.txt"
printf 'a\nb\n' >"$WORK/lf.txt"
printf 'a\tb\n' >"$WORK/tab.txt"
printf 'a\177b\n' >"$WORK/del.txt"
printf ' spaced\n' >"$WORK/lead-space.txt"
printf 'trailing \n' >"$WORK/trail-space.txt"
expect_refusal "an empty key file" "$WORK/empty.txt" "empty"
expect_refusal "a newline-only key file" "$WORK/newline.txt" "empty"
expect_refusal "a CR key file" "$WORK/cr.txt" "control character"
expect_refusal "a key with an embedded LF" "$WORK/lf.txt" "control character"
expect_refusal "a key with a TAB" "$WORK/tab.txt" "control character"
expect_refusal "a key with a DEL" "$WORK/del.txt" "control character"
expect_refusal "a key with a leading space" "$WORK/lead-space.txt" "starts or ends with a space"
expect_refusal "a key with a trailing space" "$WORK/trail-space.txt" "starts or ends with a space"

echo "== key hygiene warnings (the key is still served)"
expect_warning "a percent warns" "key%41" "percent"

echo
if [[ "$FAILED" == 0 ]]; then
  echo "auth: all checks green"
else
  echo "auth: $FAILED check(s) failed"
  exit 1
fi
