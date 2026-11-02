#!/usr/bin/env bash
# tests/slots.sh - GET /slots (slot state for monitoring), the HTTP half.
#
# What it pins:
#   - without --slots the route does not exist (404) and /props does not list it;
#   - with --slots it sits behind the same --api-key as every other route (401);
#   - one entry per slot, and per entry exactly the state fields - id, lane,
#     n_ctx, is_processing, n_cache, plus id_task while the slot serves;
#   - a busy slot says is_processing and carries the request's own X-Request-Id
#     (1.4.2's request id), and after it ends both are back to idle;
#   - nothing the user sent appears in the body: a unique marker in the prompt
#     of the busy request is absent from /slots.
#
# The stub backend only: no model file, no GPU, so it belongs to the unit gate.
#
# Run standalone: BIN=/path/to/arcint PORT=8097 ./tests/slots.sh
set -u
BIN=${BIN:?BIN must be set to the arcint binary}
PORT=${PORT:-8097}
BASE=http://127.0.0.1:$PORT
KEY='slots-test-key'
RID='req-slotprobe-7'
MARKER='zqx-needle-9c4f7a2e'
WORK=$(mktemp -d)
PIDS=''
fail=0

ok()   { echo "ok   - $1"; }
bad()  { echo "FAIL - $1"; fail=$((fail + 1)); }
check(){ if [ "$2" = "$3" ]; then ok "$1 [$2]"; else bad "$1: expected [$3], got [$2]"; fi; }

cleanup() {
    for p in $PIDS; do kill "$p" 2>/dev/null; done
    wait 2>/dev/null
    rm -rf "$WORK"
}
trap cleanup EXIT INT TERM

stop_server() {
    for p in $PIDS; do kill "$p" 2>/dev/null; done
    wait 2>/dev/null
    PIDS=''
    sleep 0.3
}

boot() {
    name=$1; shift
    "$BIN" --stub --host 127.0.0.1 --port "$PORT" "$@" >"$WORK/$name.log" 2>&1 &
    PIDS="$PIDS $!"
    for _ in $(seq 1 150); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health" 2>/dev/null)" = "200" ] && return 0
        sleep 0.1
    done
    echo "server $name did not come up:"
    cat "$WORK/$name.log"
    exit 2
}

# get <path> [curl args...] - sets CODE and BODY (no subshell: a subshell would
# lose the status, and the status is half of what these cases assert)
get() {
    local path=$1; shift
    CODE=$(curl -s -o "$WORK/body" -w '%{http_code}' "$@" "$BASE$path" 2>/dev/null)
    BODY=$(cat "$WORK/body" 2>/dev/null)
}

echo "== the flag is off: the route does not exist =="
boot off --api-key "$KEY"
get /slots -H "Authorization: Bearer $KEY"
check "GET /slots without --slots is 404" "$CODE" "404"
get /props -H "Authorization: Bearer $KEY"
if printf '%s' "$BODY" | python3 -c 'import json,sys; p=json.load(sys.stdin); sys.exit(0 if "/slots" not in p.get("endpoints", []) else 1)'; then
    ok "/props does not list /slots"
else
    bad "/props lists /slots although the flag is off"
fi
stop_server

echo "== the flag is on: the key gates it =="
boot on --api-key "$KEY" --slots --parallel 2 --stub-delay-ms 150 \
        --served-model-name probe-model
get /slots
check "GET /slots without a key is 401" "$CODE" "401"
get /slots -H "Authorization: Bearer not-the-key"
check "GET /slots with a wrong key is 401" "$CODE" "401"
get /slots -H "Authorization: Bearer $KEY"
check "GET /slots with the key is 200" "$CODE" "200"
printf '%s' "$BODY" > "$WORK/idle.json"
if python3 - "$WORK/idle.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert isinstance(d, list), "the body is not an array"
assert len(d) == 2, "expected one entry per slot (2), got %d" % len(d)
want = {"id", "lane", "n_ctx", "is_processing", "n_cache"}
for e in d:
    assert set(e) == want, "keys %s != %s" % (sorted(set(e)), sorted(want))
    assert e["lane"] == "probe-model", "lane %r" % (e["lane"],)
    assert e["is_processing"] is False, "an idle slot says is_processing"
    assert e["n_cache"] == 0, "an idle slot holds %r cells" % (e["n_cache"],)
    assert isinstance(e["n_ctx"], int) and e["n_ctx"] > 0, "n_ctx %r" % (e["n_ctx"],)
assert sorted(e["id"] for e in d) == [0, 1], "slot ids %s" % ([e["id"] for e in d],)
PY
then
    ok "idle: 2 entries, exactly the state fields, no id_task"
else
    bad "idle shape: $BODY"
fi
get /props -H "Authorization: Bearer $KEY"
if printf '%s' "$BODY" | python3 -c 'import json,sys; p=json.load(sys.stdin); sys.exit(0 if "/slots" in p.get("endpoints", []) else 1)'; then
    ok "/props lists /slots when it is served"
else
    bad "/props does not list /slots although the flag is on"
fi

echo "== a busy slot: state, and no content =="
curl -s -o "$WORK/chat.json" -D "$WORK/chat.headers" \
     -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -H "X-Request-Id: $RID" \
     -d "{\"model\":\"probe-model\",\"messages\":[{\"role\":\"user\",\"content\":\"recite this marker slowly: $MARKER and then keep talking\"}],\"max_tokens\":12,\"stream\":false}" \
     "$BASE/v1/chat/completions" &
CHAT=$!
PIDS="$PIDS $CHAT"

busy_seen=no
for _ in $(seq 1 60); do
    get /slots -H "Authorization: Bearer $KEY"
    printf '%s' "$BODY" > "$WORK/busy.json"
    if python3 - "$WORK/busy.json" "$RID" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
busy = [e for e in d if e.get("is_processing")]
assert len(busy) == 1, "expected exactly one busy slot, got %d" % len(busy)
e = busy[0]
assert e.get("id_task") == sys.argv[2], "id_task %r != %r" % (e.get("id_task"), sys.argv[2])
assert e["n_cache"] > 0, "a busy slot holds %r cells" % (e["n_cache"],)
want = {"id", "lane", "n_ctx", "is_processing", "n_cache", "id_task"}
assert set(e) == want, "busy keys %s != %s" % (sorted(set(e)), sorted(want))
PY
    then
        busy_seen=yes
        ok "busy slot: is_processing true, id_task is the request's X-Request-Id, n_cache > 0"
        break
    fi
    sleep 0.1
done
[ "$busy_seen" = yes ] || bad "no busy slot seen while the request ran: $(cat "$WORK/busy.json" 2>/dev/null)"

case "$BODY" in
    *"$MARKER"*) bad "the prompt's marker appears in /slots" ;;
    *)           ok "the prompt's marker is absent from /slots" ;;
esac
for word in prompt generated content token temperature top_p max_tokens messages; do
    case "$BODY" in
        *"$word"*) bad "/slots mentions the forbidden field or content word '$word'" ;;
    esac
done

wait $CHAT
if grep -qi "^X-Request-Id: $RID" "$WORK/chat.headers"; then
    ok "the chat response echoed the request id (the id /slots reports is the request's own)"
else
    bad "the chat response did not echo X-Request-Id"
fi

get /slots -H "Authorization: Bearer $KEY"
printf '%s' "$BODY" > "$WORK/done.json"
if python3 - "$WORK/done.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert all(e["is_processing"] is False for e in d), "a slot is still processing"
assert all("id_task" not in e for e in d), "a finished slot still carries id_task"
PY
then
    ok "after the request: is_processing false and no id_task"
else
    bad "the finished slot still looks busy: $BODY"
fi
case "$BODY" in
    *"$MARKER"*) bad "the prompt's marker appears in /slots after the request" ;;
    *)           ok "the marker stays absent after the request" ;;
esac
stop_server

echo
if [ "$fail" = 0 ]; then
    echo "slots.sh: all checks passed"
    exit 0
fi
echo "slots.sh: $fail check(s) FAILED"
exit 1
