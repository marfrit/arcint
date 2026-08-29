#!/usr/bin/env bash
# M0 exit criterion (DESIGN.md §7): a curl round-trip against the skeleton.
#
# Every check below is an assertion about the serving contract in §3.7, §3.8 and
# §4, not about model quality — the stub backend has no model in it.
set -uo pipefail

BIN="${1:-}"
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "usage: roundtrip.sh /path/to/arcint" >&2
  exit 2
fi

for tool in curl python3; do
  command -v "$tool" >/dev/null || { echo "roundtrip: $tool is required" >&2; exit 2; }
done

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
BASE="http://127.0.0.1:${PORT}"
WORK=$(mktemp -d)
LOG="${WORK}/server.log"

cleanup() {
  local pid
  for pid in "${SRV_PID:-}" "${CANCEL_PID:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
    fi
  done
  rm -rf "$WORK"
}
trap cleanup EXIT

FAILED=0
pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1"; FAILED=$((FAILED + 1)); }

check() {  # check <name> <condition-exit-code>
  if [[ "$2" == "0" ]]; then pass "$1"; else fail "$1"; fi
}

# Runs a python expression over a JSON file; prints "1" when it holds.
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

"$BIN" --stub --host 127.0.0.1 --port "$PORT" --n-ctx 256 -v >"$LOG" 2>&1 &
SRV_PID=$!

for _ in $(seq 1 100); do
  curl -fsS "${BASE}/health" -o "${WORK}/boot.json" 2>/dev/null && break
  kill -0 "$SRV_PID" 2>/dev/null || { echo "server exited during boot:"; cat "$LOG"; exit 1; }
  sleep 0.1
done

echo "== arcint M0 round-trip on ${BASE}"

# ------------------------------------------------------------------ /health
code=$(curl -sS -o "${WORK}/health.json" -w '%{http_code}' "${BASE}/health")
check "/health returns 200" "$([[ $code == 200 ]] && echo 0 || echo 1)"
jassert "${WORK}/health.json" 'd["status"]=="ok" and d["loaded"] is True and d["stub"] is True'
check "/health reports a loaded stub model" $?
jassert "${WORK}/health.json" 'd["slots_total"]==1 and d["slots_free"]>=0 and d["queue_depth"]==0'
check "/health reports slots and queue depth" $?

# ------------------------------------------------------------------- /props
curl -sS "${BASE}/props" -o "${WORK}/props.json"
jassert "${WORK}/props.json" 'd["model"]["id"]=="qwen3.6-27b-a3b-coder" and d["model"]["n_ctx"]==256'
check "/props reports the served model and context" $?
jassert "${WORK}/props.json" 'd["cache"]["kv_block_size"]==32 and d["cache"]["kv_dtype"]=="fp16"'
check "/props reports the cache configuration" $?
jassert "${WORK}/props.json" 'd["sampler_defaults"]["provenance"]=="provisional"'
check "/props marks provisional sampler defaults as such" $?
jassert "${WORK}/props.json" 'd["model"]["arch_hash"]=="6745cfe3d57e3f0f" and d["model"]["template_hash"]=="e84f32a23fdda276"'
check "/props reports the pinned artifact hashes" $?
jassert "${WORK}/props.json" 'd["model"]["n_layer"]==40 and d["model"]["n_gdn_layer"]==30 and d["model"]["n_attn_layer"]==10'
check "/props reports the measured layer split" $?
jassert "${WORK}/props.json" '"10/10" in d["model"]["status"]'
check "/props carries the artifact's Pruefstand status" $?
jassert "${WORK}/props.json" 'bool(d["build"]["version"]) and bool(d["build"]["compiler"])'
check "/props carries build info" $?

# --------------------------------------------------------------- /v1/models
curl -sS "${BASE}/v1/models" -o "${WORK}/models.json"
jassert "${WORK}/models.json" 'len(d["data"])==1 and d["data"][0]["id"]=="qwen3.6-27b-a3b-coder"'
check "/v1/models lists exactly the served model" $?

# ------------------------------------------------------- chat, non-streaming
curl -sS "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hallo"}],"temperature":0}' \
  -o "${WORK}/chat.json"
jassert "${WORK}/chat.json" 'd["object"]=="chat.completion" and d["choices"][0]["message"]["role"]=="assistant"'
check "chat completion has the OpenAI shape" $?
jassert "${WORK}/chat.json" 'd["usage"]["total_tokens"]==d["usage"]["prompt_tokens"]+d["usage"]["completion_tokens"] and d["usage"]["completion_tokens"]>0'
check "chat completion reports consistent usage" $?
jassert "${WORK}/chat.json" 'd["choices"][0]["finish_reason"]=="stop"'
check "chat completion finishes with reason stop" $?

# ------------------------------------------------------------ chat, streaming
curl -sSN "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hallo"}],"temperature":0,"stream":true}' \
  -o "${WORK}/chat.sse"

python3 - "${WORK}/chat.sse" "${WORK}/chat.json" <<'PY'
import json, sys
sse, whole = sys.argv[1], sys.argv[2]
frames = []
for line in open(sse, encoding="utf-8"):
    line = line.strip()
    if line.startswith("data: "):
        frames.append(line[6:])
assert frames and frames[-1] == "[DONE]", "stream must end with [DONE]"
objs = [json.loads(f) for f in frames[:-1]]
assert all(o["object"] == "chat.completion.chunk" for o in objs), "chunk object type"
text = "".join(c["delta"]["content"] for o in objs for c in o["choices"]
               if "content" in c.get("delta", {}))
finish = [c["finish_reason"] for o in objs for c in o["choices"] if c.get("finish_reason")]
assert finish == ["stop"], f"exactly one finish_reason, got {finish}"
usage = [o["usage"] for o in objs if "usage" in o]
assert len(usage) == 1 and usage[0]["completion_tokens"] > 0, "final chunk carries usage"

reference = json.load(open(whole))["choices"][0]["message"]["content"]
assert text == reference, f"streamed text differs from non-streamed:\n{text!r}\n{reference!r}"
assert "ü" in text and "·" in text, "two-byte characters survived the stream"
# A four-byte code point is the case that actually breaks naive chunking: the
# stub emits it split across two callbacks.
assert "\U0001F9E9" in text, f"four-byte code point did not survive: {text!r}"
text.encode("utf-8").decode("utf-8")  # raises if anything got torn
PY
check "stream reassembles byte-identically to the non-streamed body" $?

# ---------------------------------------------------------------- tool calls
curl -sS "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"weather?"}],
       "tools":[{"type":"function","function":{"name":"get_weather",
                 "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}' \
  -o "${WORK}/tools.json"
jassert "${WORK}/tools.json" 'd["choices"][0]["finish_reason"]=="tool_calls"'
check "declared tools produce finish_reason tool_calls" $?
jassert "${WORK}/tools.json" 'd["choices"][0]["message"]["tool_calls"][0]["function"]["name"]=="get_weather"'
check "tool call is parsed into the OpenAI shape" $?
jassert "${WORK}/tools.json" '"<tool_call>" not in (d["choices"][0]["message"]["content"] or "")'
check "tool-call syntax does not leak into content" $?

# A request that declares no tools gets its raw text back untouched (§3.7).
jassert "${WORK}/chat.json" '"tool_calls" not in d["choices"][0]["message"]'
check "tool-less request has no tool_calls field" $?

curl -sSN "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"weather?"}],"stream":true,
       "tools":[{"type":"function","function":{"name":"get_weather","parameters":{}}}]}' \
  -o "${WORK}/tools.sse"
python3 - "${WORK}/tools.sse" <<'PY'
import json, sys
objs = [json.loads(l[6:]) for l in open(sys.argv[1], encoding="utf-8")
        if l.startswith("data: ") and l.strip() != "data: [DONE]"]
content = "".join(c["delta"]["content"] for o in objs for c in o["choices"]
                  if "content" in c.get("delta", {}))
assert "<tool_call>" not in content, f"tool syntax leaked into the stream: {content!r}"
calls = [c["delta"]["tool_calls"] for o in objs for c in o["choices"]
         if "tool_calls" in c.get("delta", {})]
assert len(calls) == 1 and calls[0][0]["function"]["name"] == "get_weather"
finish = [c["finish_reason"] for o in objs for c in o["choices"] if c.get("finish_reason")]
assert finish == ["tool_calls"], finish
PY
check "streamed tool calls arrive as a delta, not as content" $?

# ------------------------------------------------------------------ sampling
curl -sS "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":3}' -o "${WORK}/len.json"
jassert "${WORK}/len.json" 'd["choices"][0]["finish_reason"]=="length" and d["usage"]["completion_tokens"]==3'
check "max_tokens truncates with finish_reason length" $?

curl -sS "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"stop":["backend"]}' -o "${WORK}/stop.json"
jassert "${WORK}/stop.json" '"backend" not in d["choices"][0]["message"]["content"] and d["choices"][0]["finish_reason"]=="stop"'
check "stop sequence truncates before the match" $?

# ---------------------------------------------------------------- completions
curl -sS "${BASE}/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt":"once upon a time","temperature":0}' -o "${WORK}/comp.json"
jassert "${WORK}/comp.json" 'd["object"]=="text_completion" and len(d["choices"][0]["text"])>0'
check "/v1/completions round-trips" $?

curl -sSN "${BASE}/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt":"once upon a time","temperature":0,"stream":true}' -o "${WORK}/comp.sse"
python3 - "${WORK}/comp.sse" "${WORK}/comp.json" <<'PY'
import json, sys
lines = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.startswith("data: ")]
assert lines[-1] == "data: [DONE]"
objs = [json.loads(l[6:]) for l in lines[:-1]]
text = "".join(c["text"] for o in objs for c in o["choices"])
assert text == json.load(open(sys.argv[2]))["choices"][0]["text"], repr(text)
PY
check "streamed completion matches the non-streamed body" $?

# ------------------------------------------------- regressions from the review
#
# Each of these was reproduced against a running server before it was fixed.

code=$(curl -sS -o "${WORK}/badpart.json" -w '%{http_code}' "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":[{"type":123}]}]}')
check "a malformed content part is a 400, not a 500" "$([[ $code == 400 ]] && echo 0 || echo 1)"
jassert "${WORK}/badpart.json" '"json.exception" not in d["error"]["message"]'
check "parser internals do not leak into the error body" $?

code=$(curl -sS -o "${WORK}/bigmax.json" -w '%{http_code}' "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":8589934596}')
check "an out-of-range max_tokens is rejected, not wrapped" "$([[ $code == 400 ]] && echo 0 || echo 1)"

code=$(curl -sS -o /dev/null -w '%{http_code}' "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":null,"tool_calls":[{"id":"c0","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c0","content":""}]}')
check "an empty tool result is accepted" "$([[ $code == 200 ]] && echo 0 || echo 1)"

curl -sSN "${BASE}/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt":"x","temperature":0,"stream":true,"stream_options":{"include_usage":true}}' \
  -o "${WORK}/compusage.sse"
python3 - "${WORK}/compusage.sse" <<'PY2'
import json, sys
objs = [json.loads(l[6:]) for l in open(sys.argv[1], encoding="utf-8")
        if l.startswith("data: ") and l.strip() != "data: [DONE]"]
usage = [o["usage"] for o in objs if "usage" in o]
assert len(usage) == 1, f"expected exactly one usage frame, got {len(usage)}"
assert usage[0]["completion_tokens"] > 0
PY2
check "a streamed completion honours stream_options.include_usage" $?

python3 - "${WORK}/tools.sse" <<'PY2'
import json, sys
objs = [json.loads(l[6:]) for l in open(sys.argv[1], encoding="utf-8")
        if l.startswith("data: ") and l.strip() != "data: [DONE]"]
deltas = [c["delta"]["tool_calls"] for o in objs for c in o["choices"]
          if "tool_calls" in c.get("delta", {})]
assert deltas, "no tool_calls delta"
for group in deltas:
    for i, call in enumerate(group):
        # The OpenAI SDKs model ChoiceDeltaToolCall.index as a required int.
        assert call.get("index") == i, f"missing or wrong index: {call}"
PY2
check "streamed tool_calls deltas carry the required index" $?

# The SSE content provider must not report a normal end as a cancellation, or
# httplib tears the connection down after every stream.
curl -sSN "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"a"}],"temperature":0,"stream":true}' -o /dev/null \
  --next -sSN "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"b"}],"temperature":0,"stream":true}' -o /dev/null \
  -v 2>"${WORK}/keepalive.log"
if grep -q 'Re-using existing' "${WORK}/keepalive.log" && \
   ! grep -q 'shutting down connection' "${WORK}/keepalive.log"; then
  pass "streaming keeps the connection alive for the next request"
else
  fail "streaming keeps the connection alive for the next request"
fi

# ----------------------------------------------------- context overflow (§3.8)
python3 -c 'import json;print(json.dumps({"messages":[{"role":"user","content":"word "*4000}]}))' \
  > "${WORK}/big.json"
code=$(curl -sS -o "${WORK}/overflow.json" -w '%{http_code}' "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' --data-binary "@${WORK}/big.json")
check "context overflow is rejected with 400" "$([[ $code == 400 ]] && echo 0 || echo 1)"
jassert "${WORK}/overflow.json" 'd["error"]["code"]=="context_length_exceeded"'
check "overflow error carries the documented code" $?
jassert "${WORK}/overflow.json" 'd["error"]["overflow"]==d["error"]["prompt_tokens"]-d["error"]["n_ctx"] and d["error"]["overflow"]>0'
check "overflow error carries prompt_tokens, n_ctx and overflow" $?

# --------------------------------------------------------------- bad requests
code=$(curl -sS -o "${WORK}/bad.json" -w '%{http_code}' "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' -d 'not json at all')
check "malformed JSON is rejected with 400" "$([[ $code == 400 ]] && echo 0 || echo 1)"
jassert "${WORK}/bad.json" 'd["error"]["type"]=="invalid_request_error"'
check "errors use the OpenAI envelope" $?

code=$(curl -sS -o /dev/null -w '%{http_code}' "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' -d '{"messages":[{"role":"wizard","content":"x"}]}')
check "unknown role is rejected with 400" "$([[ $code == 400 ]] && echo 0 || echo 1)"

code=$(curl -sS -o /dev/null -w '%{http_code}' "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"x"}],"temperature":9}')
check "out-of-range temperature is rejected with 400" "$([[ $code == 400 ]] && echo 0 || echo 1)"

code=$(curl -sS -o "${WORK}/404.json" -w '%{http_code}' "${BASE}/v1/embeddings")
check "an unimplemented route returns 404" "$([[ $code == 404 ]] && echo 0 || echo 1)"
jassert "${WORK}/404.json" 'd["error"]["code"]=="not_found"'
check "404 uses the error envelope too" $?

# ------------------------------------------- cancellation on disconnect (§3.7)
#
# Needs its own server: the zero-latency stub finishes before a client could
# possibly hang up, so a disconnect against it would prove nothing.
CANCEL_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
CANCEL_LOG="${WORK}/cancel.log"
"$BIN" --stub --host 127.0.0.1 --port "$CANCEL_PORT" --n-ctx 4096 --stub-delay-ms 40 -v \
  >"$CANCEL_LOG" 2>&1 &
CANCEL_PID=$!
for _ in $(seq 1 100); do
  curl -fsS "http://127.0.0.1:${CANCEL_PORT}/health" -o /dev/null 2>/dev/null && break
  if ! kill -0 "$CANCEL_PID" 2>/dev/null; then
    echo "cancel stub exited during boot:"; cat "$CANCEL_LOG"
    exit 1
  fi
  sleep 0.1
done

curl -sN --max-time 0.5 "http://127.0.0.1:${CANCEL_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"x"}],"stream":true,"max_tokens":500}' \
  -o "${WORK}/cancelled.sse" >/dev/null 2>&1
rc=$?
check "a mid-stream disconnect ends the client request" "$([[ $rc == 28 ]] && echo 0 || echo 1)"

# The stub emits ~50 tokens at 40 ms; half a second cannot have collected them.
python3 - "${WORK}/cancelled.sse" <<'PY2'
import sys
frames = [l for l in open(sys.argv[1], encoding="utf-8", errors="replace") if l.startswith("data: ")]
assert frames, "the stream produced nothing at all before the cut"
assert not any(l.strip() == "data: [DONE]" for l in frames), "stream completed; it was not cut short"
PY2
check "the cut stream carries partial output and no [DONE]" $?

sleep 0.5
grep -q 'aborted: client gone' "$CANCEL_LOG"
check "the server logs the abort at a scheduler boundary" $?

curl -sS "http://127.0.0.1:${CANCEL_PORT}/health" -o "${WORK}/after-cancel.json"
jassert "${WORK}/after-cancel.json" 'd["slots_free"]==d["slots_total"] and d["queue_depth"]==0'
check "the aborted request released its slot" $?

kill -TERM "$CANCEL_PID" 2>/dev/null
wait "$CANCEL_PID" 2>/dev/null

# ------------------------------------------------------------ console output
grep -q '^lgc  slot 0: prefill' "$LOG"
check "console prints a per-slot prefill line" $?
grep -q '^lgc  slot 0: decode' "$LOG"
check "console prints a per-slot decode line" $?
grep -q '^lgc  http: listening on' "$LOG"
check "console prints the listen line" $?

# ----------------------------------------------------------- graceful shutdown
kill -TERM "$SRV_PID"
for _ in $(seq 1 50); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 0.1; done
if kill -0 "$SRV_PID" 2>/dev/null; then fail "SIGTERM shuts the server down"; else pass "SIGTERM shuts the server down"; fi
SRV_PID=""

echo
if [[ $FAILED -eq 0 ]]; then
  echo "round-trip: all checks passed"
  exit 0
fi
echo "round-trip: ${FAILED} check(s) failed"
echo "--- server log ---"
cat "$LOG"
exit 1
