#!/usr/bin/env bash
# ops item 3: POST /tokenize and POST /detokenize on the stub. The HTTP half of
# the device-free unit gate: the llama-server shapes, the pieces, the
# chat/Responses count against a real generation's usage, the model-field lane
# routing and the auth coverage. The invalid-UTF-8 piece representation is a
# unit case (the stub vocabulary glues a multi-byte character into one piece).
set -uo pipefail

BIN="${1:-}"
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "usage: tokenize.sh /path/to/arcint" >&2
  exit 2
fi

for tool in curl python3; do
  command -v "$tool" >/dev/null || { echo "tokenize: $tool is required" >&2; exit 77; }
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

boot() {
  local log="$1"
  for _ in $(seq 1 100); do
    curl -fsS "http://127.0.0.1:${PORT}/health" -o /dev/null 2>/dev/null && return 0
    kill -0 "$SRV_PID" 2>/dev/null || { echo "server exited during boot:"; cat "$log"; exit 1; }
    sleep 0.1
  done
  return 1
}

post() {  # post <path> <json> <out>; prints the status
  curl -sS -o "$3" -w '%{http_code}' -H 'Content-Type: application/json' -d "$2" "$BASE$1"
}

PORT=$(free_port)
"$BIN" --stub --host 127.0.0.1 --port "$PORT" --n-ctx 4096 -v >"$WORK/server.log" 2>&1 &
SRV_PID=$!
boot "$WORK/server.log" || { echo "tokenize server did not boot" >&2; exit 1; }
BASE="http://127.0.0.1:${PORT}"

echo "== /tokenize, the llama-server shape"
code=$(post /tokenize '{"content":"hello world"}' "$WORK/tok.json")
check "/tokenize content is 200" "$([[ "$code" == 200 ]] && echo 0 || echo 1)"
jassert "$WORK/tok.json" 'isinstance(d["tokens"], list) and len(d["tokens"]) > 0 and all(isinstance(t, int) for t in d["tokens"])'
check "/tokenize returns an integer id list" $?
jassert "$WORK/tok.json" '"count" not in d'
check "the llama-server shape carries no count" $?

code=$(post /tokenize '{"content":"hello","with_pieces":true}' "$WORK/tokp.json")
check "/tokenize with_pieces is 200" "$([[ "$code" == 200 ]] && echo 0 || echo 1)"
jassert "$WORK/tokp.json" 'all(set(t) == {"id", "piece"} and isinstance(t["id"], int) for t in d["tokens"])'
check "with_pieces returns {id, piece} objects" $?
jassert "$WORK/tokp.json" '"".join(t["piece"] if isinstance(t["piece"], str) else "".join(chr(b) for b in t["piece"]) for t in d["tokens"]) == "hello"'
check "the pieces concatenate to the content" $?

code=$(post /tokenize '{"content":"hello","add_special":true}' "$WORK/tokbos.json")
check "/tokenize add_special is accepted" "$([[ "$code" == 200 ]] && echo 0 || echo 1)"

echo "== /detokenize"
IDS=$(python3 -c 'import json; print(json.dumps(json.load(open("'"$WORK/tok.json"'"))["tokens"]))')
code=$(post /detokenize "{\"tokens\":$IDS}" "$WORK/detok.json")
check "/detokenize is 200" "$([[ "$code" == 200 ]] && echo 0 || echo 1)"
jassert "$WORK/detok.json" 'd["content"] == "hello world"'
check "detokenize round-trips the content" $?
code=$(post /detokenize '{"tokens":["x"]}' "$WORK/detokbad.json")
check "a non-integer token is a 400" "$([[ "$code" == 400 ]] && echo 0 || echo 1)"

echo "== the count on a chat body"
CHAT='{"model":"stub","messages":[{"role":"user","content":"hello world"}],"max_tokens":4}'
code=$(post /tokenize "$CHAT" "$WORK/tokchat.json")
check "/tokenize on a chat body is 200" "$([[ "$code" == 200 ]] && echo 0 || echo 1)"
jassert "$WORK/tokchat.json" 'isinstance(d["count"], int) and len(d["tokens"]) == d["count"] and d["count"] > 0'
check "the chat count equals the token vector's size" $?
curl -sS -o "$WORK/genchat.json" -H 'Content-Type: application/json' -d "$CHAT" \
     "$BASE/v1/chat/completions"
jassert "$WORK/genchat.json" 'd["usage"]["prompt_tokens"] > 0'
check "the chat generation reports its prompt tokens" $?
python3 - "$WORK/tokchat.json" "$WORK/genchat.json" <<'PY'
import json, sys
t = json.load(open(sys.argv[1])); g = json.load(open(sys.argv[2]))
sys.exit(0 if t["count"] == g["usage"]["prompt_tokens"] else 1)
PY
check "the chat count equals the generation's usage.prompt_tokens" $?

echo "== the count on a Responses body"
RESP='{"model":"stub","instructions":"be terse","input":"hello world","max_output_tokens":16}'
code=$(post /tokenize "$RESP" "$WORK/tokresp.json")
check "/tokenize on a Responses body is 200" "$([[ "$code" == 200 ]] && echo 0 || echo 1)"
jassert "$WORK/tokresp.json" 'isinstance(d["count"], int) and len(d["tokens"]) == d["count"] and d["count"] > 0'
check "the Responses count equals the token vector's size" $?
curl -sS -o "$WORK/genresp.json" -H 'Content-Type: application/json' -d "$RESP" \
     "$BASE/v1/responses"
jassert "$WORK/genresp.json" 'd["usage"]["input_tokens"] > 0'
check "the Responses generation reports its input tokens" $?
python3 - "$WORK/tokresp.json" "$WORK/genresp.json" <<'PY'
import json, sys
t = json.load(open(sys.argv[1])); g = json.load(open(sys.argv[2]))
sys.exit(0 if t["count"] == g["usage"]["input_tokens"] else 1)
PY
check "the Responses count equals the generation's usage.input_tokens" $?

echo "== bad bodies"
code=$(post /tokenize '{"nonsense":true}' "$WORK/tokbad.json")
check "a body without content/messages/input is a 400" "$([[ "$code" == 400 ]] && echo 0 || echo 1)"
code=$(post /tokenize '{"content":7}' "$WORK/tokbad2.json")
check "a non-string content is a 400" "$([[ "$code" == 400 ]] && echo 0 || echo 1)"
code=$(curl -sS -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' \
       -d 'not json' "$BASE/tokenize")
check "malformed JSON is a 400" "$([[ "$code" == 400 ]] && echo 0 || echo 1)"

kill "$SRV_PID"; wait "$SRV_PID" 2>/dev/null; SRV_PID=""

echo "== the model field picks the lane"
PORT=$(free_port)
"$BIN" --stub --host 127.0.0.1 --port "$PORT" \
       --served-model-name agent,sub --lane-ctx 512,256 >"$WORK/lanes.log" 2>&1 &
SRV_PID=$!
boot "$WORK/lanes.log" || { echo "lane server did not boot" >&2; exit 1; }
BASE="http://127.0.0.1:${PORT}"

code=$(post /tokenize '{"model":"sub","messages":[{"role":"user","content":"x"}]}' "$WORK/lane-sub.json")
check "/tokenize routes a named lane" "$([[ "$code" == 200 ]] && echo 0 || echo 1)"
code=$(post /tokenize '{"model":"nope","content":"x"}' "$WORK/lane-404.json")
check "an unknown model on /tokenize is a 404" "$([[ "$code" == 404 ]] && echo 0 || echo 1)"
jassert "$WORK/lane-404.json" 'd["error"]["code"]=="model_not_found" and d["error"]["param"]=="model"'
check "the unknown-model error is the generation routes' shape" $?
code=$(post /detokenize '{"model":"nope","tokens":[]}' "$WORK/lane-detok.json")
check "an unknown model on /detokenize is a 404" "$([[ "$code" == 404 ]] && echo 0 || echo 1)"

kill "$SRV_PID"; wait "$SRV_PID" 2>/dev/null; SRV_PID=""

echo "== behind the API key"
PORT=$(free_port)
"$BIN" --stub --host 127.0.0.1 --port "$PORT" --n-ctx 4096 --api-key secret-key \
       >"$WORK/auth.log" 2>&1 &
SRV_PID=$!
for _ in $(seq 1 100); do
  curl -fsS "http://127.0.0.1:${PORT}/health" -o /dev/null 2>/dev/null && break
  sleep 0.1
done
BASE="http://127.0.0.1:${PORT}"
code=$(post /tokenize '{"content":"x"}' "$WORK/auth-tok.json")
check "/tokenize needs the key" "$([[ "$code" == 401 ]] && echo 0 || echo 1)"
code=$(post /detokenize '{"tokens":[]}' "$WORK/auth-detok.json")
check "/detokenize needs the key" "$([[ "$code" == 401 ]] && echo 0 || echo 1)"
code=$(curl -sS -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer secret-key' \
       -H 'Content-Type: application/json' -d '{"content":"x"}' "$BASE/tokenize")
check "/tokenize serves with the key" "$([[ "$code" == 200 ]] && echo 0 || echo 1)"

echo
if [[ "$FAILED" == 0 ]]; then
  echo "tokenize: all checks green"
else
  echo "tokenize: $FAILED check(s) failed"
  exit 1
fi
