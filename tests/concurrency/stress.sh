#!/usr/bin/env bash
# Slot accounting under load, against the stub backend — no card needed, which
# is the point: this is the part of §4.1/§4.2 that can run under ASan+UBSan in
# CI, where a use-after-free in the lease or the queue shows up as a report
# rather than as a rare hang on the GPU box.
#
# Two regimes, because M6 gave the server both:
#   queueing  --queue-timeout 30: every request is served, eventually
#   refusing  --queue-timeout 0 : the surplus is refused with 503, and the
#                                 lanes all come back afterwards
#
#   stress.sh /path/to/arcint [requests] [concurrency] [slots]
set -uo pipefail

BIN="${1:?usage: stress.sh <arcint> [requests] [concurrency] [slots]}"
N="${2:-200}"
CONC="${3:-24}"
SLOTS="${4:-8}"

command -v python3 >/dev/null || { echo "python3 required" >&2; exit 2; }
WORK=$(mktemp -d); trap 'rm -rf "$WORK"; [[ -n "${SRV:-}" ]] && kill -9 "$SRV" 2>/dev/null' EXIT
SRV=""
FAILED=0
pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1"; FAILED=$((FAILED+1)); }

start() {  # start <log> <extra args...>
  local log="$1"; shift
  [[ -n "$SRV" ]] && { kill -TERM "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
  PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()")
  # The HTTP worker pool bounds in-flight requests independently of the lanes,
  # and it defaults to the core count: on an 8-core box, --parallel 8 can never
  # see a refusal because only 8 requests are ever inside the server. Sizing it
  # above the client's concurrency is what makes this a test of the lane pool.
  "$BIN" --stub --host 127.0.0.1 --port "$PORT" --n-ctx 4096 --parallel "$SLOTS" \
         --http-threads "$((CONC + 4))" --stub-delay-ms 2 "$@" > "$log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 100); do
    curl -fsS "http://127.0.0.1:$PORT/health" -o /dev/null 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "server died:"; tail -20 "$log"; return 1; }
    sleep 0.2
  done
  echo "server never became ready"; return 1
}

hammer() {  # hammer <port> <n> <concurrency> -> "<ok> <refused> <other>"
  python3 - "$1" "$2" "$3" <<'PY'
import json, sys, threading, urllib.error, urllib.request
port, n, conc = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
counts = {"ok": 0, "refused": 0, "other": 0}
lock = threading.Lock()
work = list(range(n))

def one(i):
    body = json.dumps({"messages": [{"role": "user", "content": f"request {i}"}],
                       "max_tokens": 16, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            json.load(r)
        return "ok"
    except urllib.error.HTTPError as e:
        payload = e.read()
        if e.code == 503 and b"no_slot_available" in payload:
            return "refused"
        return "other"
    except Exception:                                        # noqa: BLE001
        return "other"

def worker():
    while True:
        with lock:
            if not work:
                return
            i = work.pop()
        r = one(i)
        with lock:
            counts[r] += 1

ts = [threading.Thread(target=worker) for _ in range(conc)]
for t in ts: t.start()
for t in ts: t.join()
print(counts["ok"], counts["refused"], counts["other"])
PY
}

echo "== arcint slot stress: $N requests, $CONC at a time, $SLOTS lanes"

# ---------------------------------------------------------------- queueing
start "$WORK/queue.log" --queue-timeout 30 || exit 1
read -r OK REFUSED OTHER <<<"$(hammer "$PORT" "$N" "$CONC")"
[[ "$OK" -eq "$N" ]] && pass "with --queue-timeout 30 every request is served ($OK/$N)" \
                     || fail "with --queue-timeout 30 every request is served (ok=$OK refused=$REFUSED other=$OTHER)"

free_now=$(curl -fsS "http://127.0.0.1:$PORT/health" | python3 -c "import json,sys; print(json.load(sys.stdin)['slots_free'])")
[[ "$free_now" -eq "$SLOTS" ]] && pass "every lane came back ($free_now/$SLOTS)" \
                               || fail "every lane came back ($free_now/$SLOTS)"

used=$(grep -c 'slot [0-9]*: decode' "$WORK/queue.log")
[[ "$used" -ge "$N" ]] && pass "every request reached a lane ($used decode lines)" \
                       || fail "every request reached a lane (only $used decode lines for $N)"
lanes_seen=$(grep -oE 'slot [0-9]+:' "$WORK/queue.log" | sort -u | wc -l)
[[ "$lanes_seen" -eq "$SLOTS" ]] && pass "all $SLOTS lanes were exercised" \
                                 || fail "all $SLOTS lanes were exercised (saw $lanes_seen)"

# ---------------------------------------------------------------- refusing
start "$WORK/refuse.log" || exit 1     # --queue-timeout 0 is the default
read -r OK REFUSED OTHER <<<"$(hammer "$PORT" "$N" "$CONC")"
[[ "$OTHER" -eq 0 ]] && pass "nothing failed in an unexpected way (ok=$OK refused=$REFUSED)" \
                     || fail "nothing failed in an unexpected way (other=$OTHER)"
[[ "$REFUSED" -gt 0 ]] && pass "the surplus was refused with a numbered 503 ($REFUSED of $N)" \
                       || fail "the surplus was refused with a numbered 503 (none were, so the \
refusal path never ran)"
[[ "$OK" -gt 0 ]] && pass "and the lanes kept serving meanwhile ($OK of $N)" \
                  || fail "and the lanes kept serving meanwhile (none did)"

free_now=$(curl -fsS "http://127.0.0.1:$PORT/health" | python3 -c "import json,sys; print(json.load(sys.stdin)['slots_free'])")
[[ "$free_now" -eq "$SLOTS" ]] && pass "every lane came back after the refusals ($free_now/$SLOTS)" \
                               || fail "every lane came back after the refusals ($free_now/$SLOTS)"

kill -TERM "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""
echo
if [[ $FAILED -eq 0 ]]; then echo "stress: all checks passed"; exit 0; fi
echo "stress: ${FAILED} check(s) failed"; exit 1
