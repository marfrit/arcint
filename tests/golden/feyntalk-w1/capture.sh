#!/usr/bin/env bash
# Golden capture of the chat and completions routes on the stub backend
# (review R5 of docs/campaigns/feyntalk.md). W1 splits prepare_chat; a green
# suite does not prove the served bytes stayed the same, so this records them
# before the split and compares them after it.
#
#   capture.sh <arcint-binary> <output-dir>
#
# One server, one fixed request set, every response body and its status code
# into <output-dir>. The bodies are normalised afterwards (normalise.py):
# response ids and `created` are the only variable fields, and they are named
# there. Nothing else is touched — whitespace, key order and SSE framing are
# the contract.
set -uo pipefail

BIN="${1:-}"
OUT="${2:-}"
if [[ -z "$BIN" || ! -x "$BIN" || -z "$OUT" ]]; then
  echo "usage: capture.sh /path/to/arcint <output-dir>" >&2
  exit 2
fi
for tool in curl python3; do
  command -v "$tool" >/dev/null || { echo "capture: $tool is required" >&2; exit 77; }
done

HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
[[ -n "$PORT" ]] || { echo "capture: could not pick a free port" >&2; exit 1; }
BASE="http://127.0.0.1:${PORT}"

# The chat-template markers the request files carry as placeholders are built
# from parts here, never spelled out: the reasoning parser in front of an agent
# session cuts its own turns on a literal marker.
THINK_OPEN="<""thi""nk"">"
THINK_CLOSE="</""thi""nk"">"
TOOL_CALL_OPEN="<""tool""_call"">"
TOOL_CALL_CLOSE="</""tool""_call"">"
TURN_START="<""|im_start|"">"
TURN_END="<""|im_end|"">"

SRV_PID=""
cleanup() {
  if [[ -n "$SRV_PID" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
    kill "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null
  fi
}
trap cleanup EXIT

"$BIN" --stub --host 127.0.0.1 --port "$PORT" --n-ctx 256 >"$OUT/server.log" 2>&1 &
SRV_PID=$!

for _ in $(seq 1 100); do
  curl -fsS "${BASE}/health" -o /dev/null 2>/dev/null && break
  kill -0 "$SRV_PID" 2>/dev/null || { echo "capture: server exited during boot" >&2; cat "$OUT/server.log" >&2; exit 1; }
  sleep 0.1
done

for req in "$HERE"/requests/*.json; do
  name=$(basename "$req" .json)
  case "$name" in
    chat_*)        route="/v1/chat/completions" ;;
    completions_*) route="/v1/completions" ;;
    *) echo "capture: unknown request prefix in $name" >&2; exit 1 ;;
  esac
  # Expand the request file's markers at capture time; the file on disk stays
  # placeholder-only.
  expanded="$OUT/.request-${name}.json"
  sed -e "s#@@THINK_OPEN@@#${THINK_OPEN}#g" \
      -e "s#@@THINK_CLOSE@@#${THINK_CLOSE}#g" \
      -e "s#@@TOOL_CALL_OPEN@@#${TOOL_CALL_OPEN}#g" \
      -e "s#@@TOOL_CALL_CLOSE@@#${TOOL_CALL_CLOSE}#g" \
      -e "s#@@TURN_START@@#${TURN_START}#g" \
      -e "s#@@TURN_END@@#${TURN_END}#g" "$req" >"$expanded"
  code=$(curl -sS --max-time 30 -o "$OUT/${name}.body" -w '%{http_code}' \
             -H 'Content-Type: application/json' --data-binary @"$expanded" "${BASE}${route}")
  rm -f "$expanded"
  printf '%s\n' "$code" >"$OUT/${name}.status"
done

python3 "$HERE/normalise.py" "$OUT"
