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
  command -v "$tool" >/dev/null || { echo "roundtrip: $tool is required" >&2; exit 77; }
done

# A port the kernel says is free right now. Every server below picks its own
# this way: a port derived from another one (PORT + 3) was never checked, and
# the script's own curl connections draw source ports from the same ephemeral
# range, so under back-to-back runs (a client socket's TIME-WAIT lasts 60 s)
# it was taken out from under the alias server (DESIGN §7.0.2an). The
# pick-then-bind window that remains is the server's start-up, the same for
# all four.
free_port() {
  python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'
}
need_port() {  # need_port <value>: an empty pick would fail four checks later, confusingly
  [[ -n "$1" ]] || { echo "roundtrip: could not pick a free port" >&2; exit 1; }
}

PORT=$(free_port); need_port "$PORT"
BASE="http://127.0.0.1:${PORT}"
WORK=$(mktemp -d)
LOG="${WORK}/server.log"

cleanup() {
  local pid
  for pid in "${SRV_PID:-}" "${CANCEL_PID:-}" "${ALIAS_PID:-}" "${OP_PID:-}" "${HOLD_PID:-}" "${LANES_PID:-}"; do
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
jassert "${WORK}/props.json" 'd["model"]["id"]=="stub" and d["model"]["n_ctx"]==256'
check "/props reports the served model and context" $?
# A stub loads nothing: the KV precision, the quant and the architecture are
# unknowable, so they must read null rather than a default the stub never ran
# (the M3 defect this contract replaces -- every stub and every real server
# used to report the identical hardcoded block).
jassert "${WORK}/props.json" \
  'd["cache"]["kv_dtype"] is None and d["model"]["quant"] is None and d["model"]["architecture"] is None'
check "/props reports the cache configuration (stub: nothing served, no guessing)" $?
jassert "${WORK}/props.json" 'd["sampler_defaults"]["provenance"]=="provisional"'
check "/props marks provisional sampler defaults as such" $?
jassert "${WORK}/props.json" 'not any(k in d for k in ("reservation",)) and not any(k in d["cache"] for k in ("path", "kv_block_size", "prefix_cache")) and not any(k in d["model"] for k in ("ov_arch", "arch_hash", "qsa"))'
check "/props carries none of the retired engine's fields" $?
jassert "${WORK}/health.json" 'not any(k in d for k in ("kv_blocks_free", "kv_blocks_total", "cache"))'
check "/health carries none of the retired engine's fields" $?
jassert "${WORK}/props.json" 'bool(d["build"]["version"]) and bool(d["build"]["compiler"])'
check "/props carries build info" $?

# --------------------------------------------------------------- /v1/models
curl -sS "${BASE}/v1/models" -o "${WORK}/models.json"
jassert "${WORK}/models.json" 'len(d["data"])==1 and d["data"][0]["id"]=="stub"'
check "/v1/models lists exactly the served model" $?
# The context has to be here and nowhere else: a discovering proxy reads it from
# this object and asks /props for nothing but template capabilities, so a
# context published only on /props reaches no client at all.
jassert "${WORK}/models.json" 'd["data"][0]["n_ctx"]==256'
check "/v1/models carries the context the server is running with" $?
jassert "${WORK}/models.json" 'd["data"][0]["n_ctx_train"]==262144 and d["data"][0]["n_ctx"]!=d["data"][0]["n_ctx_train"]'
check "/v1/models tells the served context from the trained one" $?
jassert "${WORK}/models.json" 'd["data"][0]["quant"] is None and d["data"][0]["lanes"]==1 and d["data"][0]["canonical_id"]=="stub"'
check "/v1/models carries quant, lanes and the canonical id" $?

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
if grep -Eq 'Re-?using existing' "${WORK}/keepalive.log" && \
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

# ------------------------------------------- images and speech (1.2.0, D4)
#
# The stub loads no projector: an image part and the speech route are the
# client's 400 with the reason, streamed or not, and the process lives on.
IMG='{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}'
code=$(curl -sS -o "${WORK}/img.json" -w '%{http_code}' "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":[{"type":"text","text":"what is this?"},'"$IMG"']}]}')
check "an image part without --mmproj is a 400" "$([[ $code == 400 ]] && echo 0 || echo 1)"
jassert "${WORK}/img.json" '"--mmproj" in d["error"]["message"]'
check "the image refusal names the projector" $?
code=$(curl -sS -o "${WORK}/img-stream.txt" -w '%{http_code}' "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"stream":true,"messages":[{"role":"user","content":['"$IMG"']}]}')
check "a streamed image part without --mmproj is a 400 before any event" "$([[ $code == 400 ]] && echo 0 || echo 1)"
code=$(curl -sS -o "${WORK}/speech.json" -w '%{http_code}' "${BASE}/v1/audio/speech" \
  -H 'Content-Type: application/json' -d '{"input":"Guten Tag","voice":"anyone"}')
check "/v1/audio/speech without --tts is a 400" "$([[ $code == 400 ]] && echo 0 || echo 1)"
jassert "${WORK}/speech.json" '"--tts" in d["error"]["message"]'
check "the speech refusal names --tts" $?
curl -sS -o "${WORK}/voices.json" "${BASE}/v1/audio/voices"
jassert "${WORK}/voices.json" 'd["voices"]==[]'
check "/v1/audio/voices lists no voice without --tts" $?
code=$(curl -sS -o /dev/null -w '%{http_code}' "${BASE}/health")
check "the server answers after the refusals" "$([[ $code == 200 ]] && echo 0 || echo 1)"
# bounded: a stub that took the flag would serve on and never return
timeout 10 "$BIN" --stub --mmproj /nonexistent.gguf --port "$(free_port)" >"${WORK}/stub-mmproj.log" 2>&1
rc=$?
grep -q -- '--mmproj' "${WORK}/stub-mmproj.log"
named=$?
check "--stub refuses --mmproj (a projector it could never load)" \
  "$([[ $rc != 0 && $rc != 124 && $named == 0 ]] && echo 0 || echo 1)"

# ------------------------------------------- cancellation on disconnect (§3.7)
#
# Needs its own server: the zero-latency stub finishes before a client could
# possibly hang up, so a disconnect against it would prove nothing.
CANCEL_PORT=$(free_port); need_port "$CANCEL_PORT"
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

# The abort is logged at the next scheduler boundary, not at the disconnect,
# so wait for the line itself, bounded, rather than for a fixed half-second
# that a loaded builder can outlast (DESIGN §7.0.2am).
for _ in $(seq 1 50); do
  grep -q 'aborted: client gone' "$CANCEL_LOG" && break
  sleep 0.1
done
grep -q 'aborted: client gone' "$CANCEL_LOG"

# ---------------------------------------------------- cancel by id (W4, arcint)
# The same slow stub: a streamed response is cancelled from a second connection
# as soon as response.created gives its id. Issue-to-terminal latency is not
# asserted here; the event order and the cause are.
CANCEL_ID=$(python3 - "$CANCEL_PORT" <<'PY3'
import http.client, json, sys, threading, queue, urllib.request, urllib.error, time
port = int(sys.argv[1])
q = queue.Queue()
result = {}
def canceller():
    rid = q.get()
    t0 = time.time()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/responses/{rid}/cancel", method="POST")
        r = urllib.request.urlopen(req)
        result["cancel"] = (r.status, json.loads(r.read()).get("arcint"))
        result["cancel_seconds"] = time.time() - t0
    except urllib.error.HTTPError as e:
        result["cancel"] = (e.code, e.read().decode())
th = threading.Thread(target=canceller)
th.start()
c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
c.request("POST", "/v1/responses", json.dumps({"model": "stub", "input": "x", "stream": True,
                                              "max_output_tokens": 500}),
          {"Content-Type": "application/json"})
r = c.getresponse()
types = []
done = False
rid = ""
while True:
    line = r.fp.readline()
    if not line: break
    s = line.decode("utf-8", errors="replace").strip()
    if s.startswith("event: "): types.append(s[7:])
    if s.startswith("data: {"):
        d = json.loads(s[6:])
        if d["type"] == "response.created":
            rid = d["response"]["id"]; q.put(rid)
        if d["type"] == "response.incomplete":
            assert d["response"]["incomplete_details"]["reason"] == "cancelled", d["response"]
            assert d["response"]["output"][-1]["status"] == "incomplete", d["response"]["output"]
    if s == "data: [DONE]": done = True; break
th.join()
assert result.get("cancel", (0, None))[0] == 200, result
assert result["cancel"][1] == {"cancelling": True, "v": 1}, result["cancel"]
assert done and types[-1] == "response.incomplete", types[-3:]
assert "arcint:response.cancelled" in types and types[-2] == "arcint:response.cancelled", types[-4:]
print(rid)
PY3
)
check "a cancel by id answers 200 with arcint.cancelling and the stream ends incomplete(cancelled)" \
  "$([[ -n "$CANCEL_ID" ]] && echo 0 || echo 1)"

code=$(curl -sS -o "${WORK}/cancel-late.json" -w '%{http_code}' -X POST -d '' \
       "http://127.0.0.1:${CANCEL_PORT}/v1/responses/${CANCEL_ID}/cancel")
check "a terminal response cannot be cancelled again (409)" "$([[ $code == 409 ]] && echo 0 || echo 1)"
jassert "${WORK}/cancel-late.json" 'd["error"]["type"]=="invalid_request" and d["error"]["code"]=="response_not_cancellable"'
check "the 409 carries response_not_cancellable" $?

code=$(curl -sS -o "${WORK}/cancel-unknown.json" -w '%{http_code}' -X POST -d '' \
       "http://127.0.0.1:${CANCEL_PORT}/v1/responses/resp-nope/cancel")
check "an unknown cancel id is 404" "$([[ $code == 404 ]] && echo 0 || echo 1)"
jassert "${WORK}/cancel-unknown.json" 'd["error"]["type"]=="not_found" and d["error"]["code"]=="response_not_found"'
check "the 404 carries response_not_found" $?

for _ in $(seq 1 50); do
  grep -q 'aborted: cancel requested' "$CANCEL_LOG" && break
  sleep 0.1
done
grep -q 'aborted: cancel requested' "$CANCEL_LOG"
check "the slot line names the cancel, not a gone client" $?
check "the server logs the abort at a scheduler boundary" $?

curl -sS "http://127.0.0.1:${CANCEL_PORT}/health" -o "${WORK}/after-cancel.json"
jassert "${WORK}/after-cancel.json" 'd["slots_free"]==d["slots_total"] and d["queue_depth"]==0'
check "the aborted request released its slot" $?

kill -TERM "$CANCEL_PID" 2>/dev/null
wait "$CANCEL_PID" 2>/dev/null

# --------------------------------------------------------- --served-model-name
#
# A second server, because the name is fixed at startup. What is being gated is
# that the presented name moves and artifact identity does not: a roster that
# discovers names from /v1/models pins itself to whatever it finds, so renaming
# has to be possible without touching which file is served.
ALIAS_PORT=$(free_port); need_port "$ALIAS_PORT"
ALIAS_LOG="${WORK}/alias.log"
"$BIN" --stub --host 127.0.0.1 --port "$ALIAS_PORT" --n-ctx 40960 \
       --served-model-name qwen3.6-coder >"$ALIAS_LOG" 2>&1 &
ALIAS_PID=$!
ALIAS_BASE="http://127.0.0.1:${ALIAS_PORT}"
for _ in $(seq 1 100); do
  curl -fsS "${ALIAS_BASE}/health" -o /dev/null 2>/dev/null && break
  kill -0 "$ALIAS_PID" 2>/dev/null || { echo "alias server died:"; tail -20 "$ALIAS_LOG"; break; }
  sleep 0.1
done

curl -sS "${ALIAS_BASE}/v1/models" -o "${WORK}/alias-models.json"
jassert "${WORK}/alias-models.json" 'd["data"][0]["id"]=="qwen3.6-coder"'
check "--served-model-name renames the id on /v1/models" $?
jassert "${WORK}/alias-models.json" 'd["data"][0]["canonical_id"]=="stub"'
check "the file behind the name is still visible" $?
jassert "${WORK}/alias-models.json" 'd["data"][0]["n_ctx"]==40960'
check "the renamed entry carries the served context, not the trained one" $?

curl -sS "${ALIAS_BASE}/props" -o "${WORK}/alias-props.json"
jassert "${WORK}/alias-props.json" 'd["model"]["id"]=="qwen3.6-coder" and d["model"]["canonical_id"]=="stub"'
check "/props agrees with /v1/models about the name" $?
jassert "${WORK}/alias-props.json" 'set(d["model"]["answers_to"])=={"qwen3.6-coder","stub"} and d["model"]["enforces_model_field"] is False'
check "/props says which names a request may use" $?

curl -sS "${ALIAS_BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-coder","messages":[{"role":"user","content":"hallo"}],"temperature":0}' \
  -o "${WORK}/alias-chat.json"
jassert "${WORK}/alias-chat.json" 'd["choices"][0]["message"]["content"] and d["model"]=="qwen3.6-coder"'
check "a request naming the served name is served, and echoes it back" $?

grep -q "serving 'qwen3.6-coder'" "$ALIAS_LOG"
check "the console says which name is served" $?

kill -TERM "$ALIAS_PID" 2>/dev/null; wait "$ALIAS_PID" 2>/dev/null; ALIAS_PID=""

# ------------------------------------------------------------- named lanes
#
# DESIGN §4.2 (amended 2026-10-07), docs/campaigns/lanes-agent-subagent.md:
# --served-model-name A,B --lane-ctx CA,CB. The model field picks the lane;
# what is gated here is the HTTP wiring the unit cases cannot reach: the
# 404, the 400 at the lane's cap, and a busy lane refused on its own while
# the other one serves.
LANES_PORT=$(free_port); need_port "$LANES_PORT"
LANES_LOG="${WORK}/lanes.log"
"$BIN" --stub --host 127.0.0.1 --port "$LANES_PORT" --stub-delay-ms 40 \
       --served-model-name agent,sub --lane-ctx 512,256 >"$LANES_LOG" 2>&1 &
LANES_PID=$!
LANES_BASE="http://127.0.0.1:${LANES_PORT}"
for _ in $(seq 1 100); do
  curl -fsS "${LANES_BASE}/health" -o /dev/null 2>/dev/null && break
  kill -0 "$LANES_PID" 2>/dev/null || { echo "lanes server died:"; tail -20 "$LANES_LOG"; break; }
  sleep 0.1
done

curl -sS "${LANES_BASE}/v1/models" -o "${WORK}/lanes-models.json"
jassert "${WORK}/lanes-models.json" \
  '[e["id"] for e in d["data"]]==["agent","sub"] and [e["n_ctx"] for e in d["data"]]==[512,256] and all(e["lanes"]==1 for e in d["data"])'
check "named lanes: /v1/models lists each name at its own context" $?
curl -sS "${LANES_BASE}/props" -o "${WORK}/lanes-props.json"
jassert "${WORK}/lanes-props.json" \
  'd["model"]["enforces_model_field"] is True and {"agent","sub"} <= set(d["model"]["answers_to"])'
check "named lanes: /props enforces the model field and lists the names" $?

lanes_post() {  # lanes_post <endpoint> <body> <out>; prints the status
  curl -sS -o "$3" -w '%{http_code}' "${LANES_BASE}$1" -H 'Content-Type: application/json' -d "$2"
}
code=$(lanes_post /v1/chat/completions '{"model":"sub","messages":[{"role":"user","content":"hallo"}],"max_tokens":4}' "${WORK}/lanes-sub.json")
jassert "${WORK}/lanes-sub.json" 'd["model"]=="sub"'
r=$?; check "named lanes: a request for the subagent lane is served and echoes its name" "$([[ $code == 200 && $r == 0 ]] && echo 0 || echo 1)"
code=$(lanes_post /v1/chat/completions '{"messages":[{"role":"user","content":"hallo"}],"max_tokens":4}' "${WORK}/lanes-noname.json")
jassert "${WORK}/lanes-noname.json" 'd["model"]=="agent"'
r=$?; check "named lanes: a request without a name goes to the first lane" "$([[ $code == 200 && $r == 0 ]] && echo 0 || echo 1)"
code=$(lanes_post /v1/chat/completions '{"model":"nope","messages":[{"role":"user","content":"hallo"}]}' "${WORK}/lanes-404.json")
jassert "${WORK}/lanes-404.json" 'd["error"]["code"]=="model_not_found" and d["error"]["param"]=="model"'
r=$?; check "named lanes: an unknown name is a 404" "$([[ $code == 404 && $r == 0 ]] && echo 0 || echo 1)"
# 300 tokens (the stub counts words and single bytes): inside the agent
# lane's 512, over the subagent lane's 256
LONG=$(python3 -c 'print("x " * 150, end="")')
code=$(lanes_post /v1/completions "{\"model\":\"sub\",\"prompt\":\"${LONG}\",\"max_tokens\":1}" "${WORK}/lanes-400.json")
jassert "${WORK}/lanes-400.json" 'd["error"]["n_ctx"]==256 and d["error"]["prompt_tokens"]==300'
r=$?; check "named lanes: the overflow 400 quotes the subagent lane's cap" "$([[ $code == 400 && $r == 0 ]] && echo 0 || echo 1)"
code=$(lanes_post /v1/completions "{\"model\":\"agent\",\"prompt\":\"${LONG}\",\"max_tokens\":1}" "${WORK}/lanes-agent-long.json")
check "named lanes: the same prompt fits the agent lane" "$([[ $code == 200 ]] && echo 0 || echo 1)"

# Hold the subagent lane with a slow stream, then ask for it again: a 503
# for that lane, while the agent lane still serves.
curl -sN --max-time 40 "${LANES_BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"sub","messages":[{"role":"user","content":"x"}],"stream":true,"max_tokens":200}' \
  -o /dev/null >/dev/null 2>&1 &
HOLD_PID=$!
# up to 30 s for the held stream to take the lane and the agent lane to be
# free again: a loaded host (a parallel build) took longer than 5 s once, and
# the 1.2.0 release build once caught the moment in between (the sub lane
# held, the agent lane not yet reported free). The check still fails when the
# state never arrives.
LANES_STATE='d["slots_total"]==2 and d["slots_free"]==1 and [l["name"] for l in d["lanes"]]==["agent","sub"] and d["lanes"][0]["slots_free"]==1 and d["lanes"][1]["slots_free"]==0'
for _ in $(seq 1 300); do
  curl -sS "${LANES_BASE}/health" -o "${WORK}/lanes-health.json"
  jassert "${WORK}/lanes-health.json" "$LANES_STATE" && break
  sleep 0.1
done
jassert "${WORK}/lanes-health.json" "$LANES_STATE"
check "named lanes: /health reports each lane's free count" $?
code=$(lanes_post /v1/chat/completions '{"model":"sub","messages":[{"role":"user","content":"hallo"}],"max_tokens":4}' "${WORK}/lanes-503.json")
jassert "${WORK}/lanes-503.json" 'd["slots"]["lane"]=="sub" and d["slots"]["free"]==0 and [l["slots_free"] for l in d["lanes"]]==[1,0]'
r=$?; check "named lanes: a busy subagent lane is a 503 for that lane" "$([[ $code == 503 && $r == 0 ]] && echo 0 || echo 1)"
code=$(lanes_post /v1/chat/completions '{"model":"agent","messages":[{"role":"user","content":"hallo"}],"max_tokens":4}' "${WORK}/lanes-agent.json")
jassert "${WORK}/lanes-agent.json" 'd["model"]=="agent"'
r=$?; check "named lanes: the agent lane serves while the subagent lane is busy" "$([[ $code == 200 && $r == 0 ]] && echo 0 || echo 1)"
kill "$HOLD_PID" 2>/dev/null; wait "$HOLD_PID" 2>/dev/null; HOLD_PID=""

grep -q "lane 1 'sub': n_ctx 256" "$LANES_LOG"
check "named lanes: the console names each lane and its context" $?

kill -TERM "$LANES_PID" 2>/dev/null; wait "$LANES_PID" 2>/dev/null; LANES_PID=""

# ------------------------------------------------- operator serving defaults
# A third server, because the operator layer is fixed at startup. What is
# being gated: the flags override only what they set, /props says the regime
# came from the operator, and the usage block carries the draft counters that
# tell a caller which performance regime its request actually got.
OP_PORT=$(free_port); need_port "$OP_PORT"
OP_LOG="${WORK}/op.log"
"$BIN" --stub --host 127.0.0.1 --port "$OP_PORT" --n-ctx 4096 \
       --temp 0 --think-eog-mask --tool-grammar --chat-template-kwarg enable_thinking=false >"$OP_LOG" 2>&1 &
OP_PID=$!
OP_BASE="http://127.0.0.1:${OP_PORT}"
for _ in $(seq 1 100); do
  curl -fsS "${OP_BASE}/health" -o /dev/null 2>/dev/null && break
  kill -0 "$OP_PID" 2>/dev/null || { echo "operator server died:"; tail -20 "$OP_LOG"; break; }
  sleep 0.1
done

curl -sS "${OP_BASE}/props" -o "${WORK}/op-props.json"
jassert "${WORK}/op-props.json" 'd["sampler_defaults"]["temperature"]==0 and d["sampler_defaults"]["provenance"]=="operator"'
check "--temp 0 flips the served default and its provenance to operator" $?
jassert "${WORK}/op-props.json" 'd["sampler_defaults"]["top_k"]==20'
check "flags override only what they set" $?
jassert "${WORK}/op-props.json" 'd["sampler_defaults"]["think_eog_mask"]==True'
check "--think-eog-mask is on /props, so a different turn shape is explained there" $?
jassert "${WORK}/op-props.json" 'd["sampler_defaults"]["tool_grammar"]==True'
check "--tool-grammar is on /props, so a constrained call is explained there" $?

curl -sS "${OP_BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":8}' -o "${WORK}/op-chat.json"
jassert "${WORK}/op-chat.json" '"accepted_prediction_tokens" in d["usage"]["completion_tokens_details"] and "rejected_prediction_tokens" in d["usage"]["completion_tokens_details"]'
check "usage says which regime the request got (draft counters present)" $?

# ------------------------------------------------------------- /v1/responses
# Open Responses, non-stream (docs/campaigns/feyntalk.md W1); the SSE writer is
# W2. The required-key set is the 2026-04-24 ResponseResource: a client parses a
# resource only when all of it is present.
code=$(curl -sS -o "${WORK}/resp.json" -w '%{http_code}' "${OP_BASE}/v1/responses" \
       -H 'Content-Type: application/json' -d '{"model":"stub","input":"hello"}')
check "/v1/responses answers a plain string input" "$([[ $code == 200 ]] && echo 0 || echo 1)"
jassert "${WORK}/resp.json" 'd["object"]=="response" and d["status"]=="completed" and d["output"][0]["type"]=="message"'
check "/v1/responses returns a completed response with a message item" $?
jassert "${WORK}/resp.json" \
  'all(k in d for k in ["id","object","created_at","completed_at","status","incomplete_details","model","previous_response_id","instructions","output","error","tools","tool_choice","truncation","parallel_tool_calls","text","top_p","presence_penalty","frequency_penalty","top_logprobs","temperature","reasoning","usage","max_output_tokens","max_tool_calls","store","background","service_tier","metadata","safety_identifier","prompt_cache_key"])'
check "/v1/responses carries every required ResponseResource key" $?
jassert "${WORK}/resp.json" \
  'd["usage"]["input_tokens"]>0 and d["usage"]["total_tokens"]==d["usage"]["input_tokens"]+d["usage"]["output_tokens"] and d["usage"]["input_tokens_details"]["cached_tokens"]==0'
check "/v1/responses usage is filled" $?

# SSE (W2): every frame has an `event:` line equal to the body's type, the
# sequence numbers run 1..N without a gap, the terminal event is last and
# `[DONE]` follows it.
curl -sN "${OP_BASE}/v1/responses" -H 'Content-Type: application/json' \
     -d '{"model":"stub","input":"hello","stream":true,"max_output_tokens":256}' \
     -o "${WORK}/resp-sse.txt"
check "/v1/responses streams an SSE body on request" $?
ssecheck() {  # ssecheck <file> <python assertions over events/types/done>
  python3 - "$1" "$2" <<'PY'
import json, sys
raw = open(sys.argv[1], encoding="utf-8").read()
events, done = [], False
for frame in [f for f in raw.split("\n\n") if f.strip()]:
    lines = frame.split("\n")
    ev = [l[6:] for l in lines if l.startswith("data: ")]
    names = [l[7:] for l in lines if l.startswith("event: ")]
    if ev and ev[0] == "[DONE]":
        done = True
        continue
    if len(names) != 1 or len(ev) != 1:
        print("frame without one event:/data: pair:", frame, file=sys.stderr)
        sys.exit(1)
    body = json.loads(ev[0])
    if names[0] != body["type"]:
        print("event: line", names[0], "!= type", body["type"], file=sys.stderr)
        sys.exit(1)
    events.append(body)
seq = [e["sequence_number"] for e in events]
if seq != list(range(1, len(events) + 1)):
    print("sequence is not 1..N:", seq, file=sys.stderr)
    sys.exit(1)
types = [e["type"] for e in events]
sys.exit(0 if eval(sys.argv[2], {"events": events, "types": types, "done": done}) else 1)
PY
}
ssecheck "${WORK}/resp-sse.txt" \
  'types[:2]==["response.created","response.in_progress"] and types[-1]=="response.completed" and done and any(t=="response.output_text.delta" for t in types) and events[-1]["response"]["status"]=="completed"'
check "/v1/responses SSE framing, order, sequence and terminal event" $?

code=$(curl -sS -o "${WORK}/resp-tools.json" -w '%{http_code}' \
       -H 'Content-Type: application/json' --data-binary '{"model":"stub","instructions":"be terse","input":[{"type":"message","role":"user","content":"read a.txt"},{"type":"function_call","call_id":"call_7","name":"read_file","arguments":"{\"path\":\"a.txt\"}"},{"type":"function_call_output","call_id":"call_7","output":"alpha"}],"tools":[{"type":"function","name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}],"max_output_tokens":256}' \
       "${OP_BASE}/v1/responses")
check "/v1/responses serves an instructions + tool + tool-output turn" "$([[ $code == 200 ]] && echo 0 || echo 1)"
jassert "${WORK}/resp-tools.json" 'd["instructions"]=="be terse" and d["tools"][0]["name"]=="read_file" and any(i["type"]=="function_call" and i["name"]=="read_file" for i in d["output"])'
check "/v1/responses echoes instructions and tools and emits the function_call item" $?

curl -sN "${OP_BASE}/v1/responses" -H 'Content-Type: application/json' \
     -d '{"model":"stub","input":"call it","stream":true,"max_output_tokens":256,"tools":[{"type":"function","name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}]}' \
     -o "${WORK}/resp-sse-tools.txt"
check "/v1/responses streams a tool turn" $?
ssecheck "${WORK}/resp-sse-tools.txt" \
  'done and types[-1]=="response.completed" and any(t=="response.function_call_arguments.delta" for t in types)'
check "/v1/responses SSE emits the function_call argument delta and terminal event" $?

# The state chain (W3): turn 1 stores; turn 2 continues with previous_response_id
# and a function_call_output. The stub backend reports no prefix reuse
# (cache_hit_tokens stays 0), so the affinity number is the live leg's (W5);
# here the continuation, the echo and the /props count are the contract.
curl -sS "${OP_BASE}/v1/responses" -H 'Content-Type: application/json' \
     -d '{"model":"stub","input":"read it","max_output_tokens":256,"tools":[{"type":"function","name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}]}' \
     -o "${WORK}/resp-t1.json"
RESP_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "${WORK}/resp-t1.json")
CALL_ID=$(python3 -c 'import json,sys; print(next(i["call_id"] for i in json.load(open(sys.argv[1]))["output"] if i["type"]=="function_call"))' "${WORK}/resp-t1.json")
[[ -n "$RESP_ID" && -n "$CALL_ID" ]] && pass "responses turn 1 returns an id and a function_call" || fail "responses turn 1 returns an id and a function_call"
python3 - "$WORK/resp-t2-request.json" "$RESP_ID" "$CALL_ID" <<'PY'
import json, sys
body = {"model": "stub", "previous_response_id": sys.argv[2], "max_output_tokens": 256,
        "tools": [{"type": "function", "name": "read_file",
                   "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}],
        "input": [{"type": "function_call_output", "call_id": sys.argv[3], "output": "alpha"}]}
open(sys.argv[1], "w").write(json.dumps(body))
PY
code=$(curl -sS -o "${WORK}/resp-t2.json" -w '%{http_code}' \
       -H 'Content-Type: application/json' --data-binary @"${WORK}/resp-t2-request.json" \
       "${OP_BASE}/v1/responses")
check "/v1/responses continues a stored response on previous_response_id" "$([[ $code == 200 ]] && echo 0 || echo 1)"
python3 - "$WORK/resp-t2.json" "$RESP_ID" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if (d["previous_response_id"] == sys.argv[2] and d["status"] == "completed"
               and d["store"] is True) else 1)
PY
check "responses turn 2 echoes previous_response_id and keeps the chain" $?
curl -sS "${OP_BASE}/props" -o "${WORK}/resp-props.json"
jassert "${WORK}/resp-props.json" 'd["responses"]["entries"] >= 1 and d["responses"]["limit_entries"] >= 32 and d["responses"]["bytes"] > 0'
check "/props reports the Responses store's live entries and bytes" $?

kill -TERM "$OP_PID" 2>/dev/null; wait "$OP_PID" 2>/dev/null; OP_PID=""

# The retired engine's options are refused by name, and its engine value by a
# message that names the history tag.
if "$BIN" --stub --model /models/ov/x --port "$ALIAS_PORT" >"${WORK}/refuse.log" 2>&1; then
  fail "a retired flag is refused"
else
  grep -q -- '--model was retired with the OpenVINO engine in 0.8.0' "${WORK}/refuse.log"
  check "a retired flag is refused by name" $?
fi
if "$BIN" --engine ov --gguf /m/q.gguf --port "$ALIAS_PORT" >"${WORK}/refuse-ov.log" 2>&1; then
  fail "--engine ov is refused"
else
  grep -q 'openvino-history' "${WORK}/refuse-ov.log"
  check "--engine ov is refused with the history tag named" $?
fi
if "$BIN" --stub --served-model-name "" --port "$ALIAS_PORT" >"${WORK}/blank.log" 2>&1; then
  fail "--served-model-name refuses a blank name"
else
  pass "--served-model-name refuses a blank name"
fi

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
