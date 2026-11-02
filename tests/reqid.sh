#!/usr/bin/env bash
# X-Request-Id: one id per request, readable by the caller and carried through
# that request's own log lines. The inbound value is kept only when it is
# 1-128 characters of [A-Za-z0-9._:-]; anything else is replaced by a generated
# `req-<16 hex>` and the value the client sent appears nowhere in the answer or
# the log. Stub-only: the HTTP half of the device-free unit gate.
set -uo pipefail

BIN="${1:-}"
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "usage: reqid.sh /path/to/arcint" >&2
  exit 2
fi

for tool in curl python3; do
  command -v "$tool" >/dev/null || { echo "reqid: $tool is required" >&2; exit 77; }
done

free_port() {
  python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'
}

WORK=$(mktemp -d)
SRV_PID=""
cleanup() { stop_pid "$SRV_PID"; rm -rf "$WORK"; }
trap cleanup EXIT

FAILED=0
pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1"; FAILED=$((FAILED + 1)); }
check() { if [[ "$2" == "0" ]]; then pass "$1"; else fail "$1"; fi; }

pid_alive() {
  local pid="$1"
  [[ -n "$pid" && -r "/proc/$pid/stat" ]] || return 1
  [[ "$(cut -d' ' -f3 "/proc/$pid/stat" 2>/dev/null)" != "Z" ]]
}

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

wait_boot() {
  for _ in $(seq 1 100); do
    curl -fsS "http://127.0.0.1:${PORT}/health" -o /dev/null 2>/dev/null && return 0
    kill -0 "$SRV_PID" 2>/dev/null || { echo "server exited during boot:"; cat "$1"; exit 1; }
    sleep 0.1
  done
  return 1
}

# wait_log <log> <fixed-string> <seconds>: the log line may land after the
# response, so the read is bounded rather than immediate.
wait_log() {
  local log="$1" needle="$2" seconds="${3:-5}" i
  for ((i = 0; i < seconds * 10; i++)); do
    grep -qF -- "$needle" "$log" 2>/dev/null && return 0
    sleep 0.1
  done
  return 1
}

# The header line's value, case-insensitively, first match.
hdr() { printf '%s\n' "$1" | tr -d '\r' | grep -i "^$2:" | head -1 | sed "s/^[^:]*:[[:space:]]*//"; }

GENERIC='^req-[0-9a-f]{16}$'

# ---------------------------------------------------------------- no key ----
echo "== no key: every response carries the id"
PORT=$(free_port)
"$BIN" --stub --host 127.0.0.1 --port "$PORT" --n-ctx 256 -v >"$WORK/nokey.log" 2>&1 &
SRV_PID=$!
wait_boot "$WORK/nokey.log" || { echo "server did not boot" >&2; exit 1; }
BASE="http://127.0.0.1:${PORT}"

SUPPLIED="gateway-42.7:abc_XYZ-9"
h=$(curl -sS -D - -o /dev/null -H "X-Request-Id: $SUPPLIED" "$BASE/props" | tr -d '\r')
check "a supplied valid id is echoed" \
  "$([[ "$(hdr "$h" 'x-request-id')" == "$SUPPLIED" ]] && echo 0 || echo 1)"

h=$(curl -sS -D - -o /dev/null "$BASE/props" | tr -d '\r')
id1=$(hdr "$h" 'x-request-id')
check "an absent id is generated as req-<16 hex>" "$(printf '%s' "$id1" | grep -Eq "$GENERIC" && echo 0 || echo 1)"
h=$(curl -sS -D - -o /dev/null "$BASE/props" | tr -d '\r')
id2=$(hdr "$h" 'x-request-id')
check "two requests get different ids" "$([[ -n "$id2" && "$id1" != "$id2" ]] && echo 0 || echo 1)"

h=$(curl -sS -D - -o /dev/null "$BASE/health" | tr -d '\r')
check "/health carries the id too" "$(printf '%s' "$(hdr "$h" 'x-request-id')" | grep -Eq "$GENERIC" && echo 0 || echo 1)"

h=$(curl -sS -D - -o /dev/null "$BASE/no-such-route" | tr -d '\r')
check "a 404 carries the id" "$(printf '%s' "$(hdr "$h" 'x-request-id')" | grep -Eq "$GENERIC" && echo 0 || echo 1)"

printf 'not json at all' >"$WORK/bad.json"
h=$(curl -sS -D - -o /dev/null -H 'Content-Type: application/json' \
       --data-binary @"$WORK/bad.json" "$BASE/v1/chat/completions" | tr -d '\r')
check "a 400 body error carries the id" "$(printf '%s' "$(hdr "$h" 'x-request-id')" | grep -Eq "$GENERIC" && echo 0 || echo 1)"
check "a 400 body error echoes a supplied id" \
  "$(h=$(curl -sS -D - -o /dev/null -H 'Content-Type: application/json' -H "X-Request-Id: $SUPPLIED" \
        --data-binary @"$WORK/bad.json" "$BASE/v1/chat/completions" | tr -d '\r'); \
     [[ "$(hdr "$h" 'x-request-id')" == "$SUPPLIED" ]] && echo 0 || echo 1)"

# SSE: the header is sent before the first chunk, so it is read off the header
# block of a streamed chat response.
cat >"$WORK/stream.json" <<'JSON'
{"model":"stub","stream":true,"messages":[{"role":"user","content":"hello"}]}
JSON
h=$(curl -sS -N -D - -o "$WORK/stream.body" -H 'Content-Type: application/json' \
       -H "X-Request-Id: sse-1" --data-binary @"$WORK/stream.json" "$BASE/v1/chat/completions" | tr -d '\r')
check "a streaming chat response carries the id" \
  "$([[ "$(hdr "$h" 'x-request-id')" == "sse-1" ]] && echo 0 || echo 1)"
check "the stream itself still produced events" "$(grep -qc 'data:' "$WORK/stream.body" && echo 0 || echo 1)"

# ------------------------------------------------------- the log carries it --
echo "== the log lines of one request carry its id"
CHAT_ID="logline-7"
cat >"$WORK/chat.json" <<'JSON'
{"model":"stub","messages":[{"role":"user","content":"hello"}]}
JSON
curl -sS -o /dev/null -H 'Content-Type: application/json' -H "X-Request-Id: $CHAT_ID" \
     --data-binary @"$WORK/chat.json" "$BASE/v1/chat/completions"
wait_log "$WORK/nokey.log" "rid=$CHAT_ID" 5 || true
# grep -c prints 0 AND exits 1, so its output is the only source: no `|| echo`.
n=$(grep -cF -- "rid=$CHAT_ID" "$WORK/nokey.log" 2>/dev/null); n=${n:-0}
check "the request's log lines carry rid=$CHAT_ID (at least the http line, the prefill line and the decode line)" \
  "$([[ "$n" -ge 3 ]] && echo 0 || echo 1)"
check "the prefill/decode summary lines carry the id" \
  "$(awk -v id="rid=$CHAT_ID" '$0 ~ id && /prefill|decode/ {found=1} END {exit found?0:1}' "$WORK/nokey.log" && echo 0 || echo 1)"

# --------------------------------------------- invalid inbound values ------
# Sent on a raw socket: a header value with a control character, a 129-character
# value and a non-ASCII value are not expressible with curl's header parser.
# The safety property being tested is "the client's bytes are neither echoed in
# the response header nor written to the log". A server that refuses such a
# request outright (no X-Request-Id, nothing echoed) satisfies it too; a value
# containing CR LF cannot be a header value at all, since CR LF ends the line.
echo "== an invalid inbound id is replaced, never echoed, never logged raw"
PORT="$PORT" LOG="$WORK/nokey.log" python3 - <<'PY' || FAILED=$((FAILED + 1))
import os, re, socket, sys

port = int(os.environ["PORT"])
log = os.environ["LOG"]
GEN = re.compile(rb"^req-[0-9a-f]{16}$")

CASES = [
    ("a 129-character value", b"X-Request-Id: " + b"a" * 129),
    ("a value with a space", b"X-Request-Id: ab cd"),
    ("a value with a TAB", b"X-Request-Id: ab\tcd"),
    ("a value with a CR", b"X-Request-Id: ab\rcd"),
    ("a value with CR LF inside", b"X-Request-Id: ab\r\rcd"),
    ("a non-ASCII value", "X-Request-Id: abä".encode()),
    ("an empty value", b"X-Request-Id: "),
    ("a value that is only separators", b"X-Request-Id: ab;cd|ef"),
]

def request(raw_header):
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(b"GET /props HTTP/1.1\r\nHost: localhost\r\n" + raw_header + b"\r\n\r\n")
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
    s.close()
    return data

rc = 0
for name, hdr in CASES:
    blob = request(hdr)
    lines = blob.split(b"\r\n")
    status = lines[0].split()[1] if len(lines[0].split()) >= 2 else b"?"
    got = [ln for ln in lines[1:] if ln.lower().startswith(b"x-request-id:")]
    value = got[0].split(b":", 1)[1].strip() if got else None
    tail = hdr[len(b"X-Request-Id: "):]
    # Nothing of the client's value may come back, whichever way the server
    # handled it. An empty value cannot be echoed, so it is not checked.
    echoed = tail in blob if tail else False
    refused = status.isdigit() and int(status) >= 400
    if value is None:
        # Only a refusal of the request excuses a missing header: a 2xx answer
        # without X-Request-Id breaks the contract for every response.
        ok = refused and not echoed
        verdict = (f"refused ({status.decode()}), nothing echoed" if ok else
                   f"answered {status.decode()} with no X-Request-Id at all")
    else:
        ok = (not echoed) and bool(GEN.match(value))
        verdict = ("replaced by a generated id, not echoed" if ok
                   else f"echoed back verbatim or malformed: {value[:40]!r}")
    print(("  ok   " if ok else "  FAIL ") + f"{name}: {verdict}")
    if not ok:
        rc = 1

# The longest and the control-character values must not reach the log either.
with open(log, "rb") as fh:
    logged = fh.read()
for name, hdr in CASES:
    tail = hdr[len(b"X-Request-Id: "):]
    if len(tail) < 3:
        continue
    if tail in logged:
        print(f"  FAIL {name} appears in the server log raw: {tail[:40]!r}")
        rc = 1
    else:
        print(f"  ok   {name} does not appear in the server log")
sys.exit(rc)
PY

stop_pid "$SRV_PID"; SRV_PID=""

# ------------------------------------------------------------------ key -----
KEY="test-key-not-real"
AUTH="Authorization: Bearer ${KEY}"
echo "== --api-key: the auth refusal and CORS carry it"
PORT=$(free_port)
"$BIN" --stub --host 127.0.0.1 --port "$PORT" --n-ctx 256 --api-key "$KEY" -v >"$WORK/key.log" 2>&1 &
SRV_PID=$!
wait_boot "$WORK/key.log" || { echo "key server did not boot" >&2; exit 1; }
BASE="http://127.0.0.1:${PORT}"

h=$(curl -sS -D - -o /dev/null "$BASE/props" | tr -d '\r')
check "the pre-routing 401 carries a generated id" \
  "$(printf '%s' "$(hdr "$h" 'x-request-id')" | grep -Eq "$GENERIC" && echo 0 || echo 1)"
h=$(curl -sS -D - -o /dev/null -H "X-Request-Id: $SUPPLIED" "$BASE/props" | tr -d '\r')
check "the pre-routing 401 echoes a supplied id" \
  "$([[ "$(hdr "$h" 'x-request-id')" == "$SUPPLIED" ]] && echo 0 || echo 1)"

# Expect: 100-continue is answered above routing: httplib calls the expect-100
# handler before the pre-routing handler, so the auth refusal happens there and
# the id has to be resolved there too. A caller that sent a usable id must get
# that id back on the 401, not a generated one.
PORT="$PORT" python3 - <<'PY' || FAILED=$((FAILED + 1))
import os, re, socket, sys

port = int(os.environ["PORT"])
GEN = re.compile(rb"^req-[0-9a-f]{16}$")

def with_expect(id_bytes):
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(b"POST /chat/completions HTTP/1.1\r\nHost: localhost\r\n"
              b"Expect: 100-continue\r\nContent-Length: 2\r\n"
              b"X-Request-Id: " + id_bytes + b"\r\n\r\n")
    s.settimeout(3.0)
    blob = b""
    try:
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            blob += chunk
    except socket.timeout:
        pass
    s.close()
    return blob

rc = 0
for name, sent, want in (("a valid id", b"abc-123", b"abc-123"),
                        ("an invalid id", b"ab cd", None)):
    blob = with_expect(sent)
    lines = blob.split(b"\r\n")
    status = lines[0].split()[1] if len(lines[0].split()) >= 2 else b"?"
    got = [ln for ln in lines[1:] if ln.lower().startswith(b"x-request-id:")]
    value = got[0].split(b":", 1)[1].strip() if got else None
    ok = (status == b"401"
          and not any(ln.startswith(b"HTTP/1.1 100") for ln in lines))
    if want is None:
        ok = ok and value is not None and bool(GEN.match(value))
    else:
        ok = ok and value == want
    verdict = (f"401 with x-request-id: {value.decode() if value else '<none>'}"
               if ok else f"status={status.decode()} x-request-id="
                         f"{value.decode() if value else '<none>'} blob={blob[:80]!r}")
    print(("  ok   " if ok else "  FAIL ") + f"Expect: 100-continue with {name}: {verdict}")
    if not ok:
        rc = 1
sys.exit(rc)
PY
h=$(curl -sS -D - -o /dev/null -H 'Access-Control-Request-Method: POST' -X OPTIONS "$BASE/props" | tr -d '\r')
check "a CORS preflight carries the id" "$(printf '%s' "$(hdr "$h" 'x-request-id')" | grep -Eq "$GENERIC" && echo 0 || echo 1)"
check "the preflight allows the X-Request-Id header" \
  "$(hdr "$h" 'access-control-allow-headers' | grep -qi 'x-request-id' && echo 0 || echo 1)"
h=$(curl -sS -D - -o /dev/null -H "$AUTH" "$BASE/props" | tr -d '\r')
check "a browser client may read it: Access-Control-Expose-Headers names X-Request-Id" \
  "$(hdr "$h" 'access-control-expose-headers' | grep -qi 'x-request-id' && echo 0 || echo 1)"
check "an authenticated response still carries the id" \
  "$(printf '%s' "$(hdr "$h" 'x-request-id')" | grep -Eq "$GENERIC" && echo 0 || echo 1)"

stop_pid "$SRV_PID"; SRV_PID=""

echo
if [[ "$FAILED" == 0 ]]; then
  echo "reqid: all checks green"
else
  echo "reqid: $FAILED check(s) failed"
  exit 1
fi
