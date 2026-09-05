#!/usr/bin/env bash
# depth-ladder (DESIGN §5.1; docs/design-0.3.1-test-ladder.md Increment 2).
# The coder artifact must be able to *load and serve* a long prefill at both
# KV precisions on both cards. This cell does not claim a fit is fast, only
# that it happens: a refusal to fit the context is a real failure of this
# cell, not an environmental skip, because refusing to fit is exactly the
# failure mode this ladder exists to catch before a release.
#
# Runs against whichever devices it is given -- it must never stop or start
# anything by its own initiative, unlike the other cells here, because it is
# the one cell that runs on both cards in the same invocation.
#
#   depth_ladder.sh <arcint> <model-dir> <device-large> <device-small> \
#                    [--prefill-tokens N]
#
# <model-dir> is a specific artifact directory (cells.json's own job is to
# say which one: "{model_root}/qwen36-coder-b5-ov"), not a root this script
# joins a hardcoded name onto -- every cell's artifact belongs in the
# enumeration, not baked into a runner.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

BIN="${1:?usage: depth_ladder.sh <arcint> <model-dir> <device-large> <device-small> [--prefill-tokens N]}"
MODEL="${2:?usage: depth_ladder.sh <arcint> <model-dir> <device-large> <device-small> [--prefill-tokens N]}"
DEV_LARGE="${3:?usage: depth_ladder.sh <arcint> <model-dir> <device-large> <device-small> [--prefill-tokens N]}"
DEV_SMALL="${4:?usage: depth_ladder.sh <arcint> <model-dir> <device-large> <device-small> [--prefill-tokens N]}"
shift 4

PREFILL_TOKENS=98147
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefill-tokens) PREFILL_TOKENS="${2:?--prefill-tokens needs a value}"; shift 2 ;;
    *) echo "depth_ladder: unrecognized argument '$1'" >&2; exit 2 ;;
  esac
done
if ! [[ "$PREFILL_TOKENS" =~ ^[0-9]+$ ]] || [[ "$PREFILL_TOKENS" -lt 1 ]]; then
  echo "depth_ladder: --prefill-tokens must be a positive integer, got '$PREFILL_TOKENS'" >&2
  exit 2
fi

for tool in python3 curl; do
  command -v "$tool" >/dev/null || { echo "depth_ladder: $tool is required" >&2; exit 77; }
done

[[ -d "$MODEL" ]] || { echo "depth_ladder: model dir $MODEL not found" >&2; exit 77; }

# Room for the prompt (sized to within +/-2% of PREFILL_TOKENS), the
# 32-token completion and block-alignment slack -- and no more. The first
# real run asked for a flat +4,096 and was refused on the 16 GiB card at
# u8:i4 by 1,011 tokens: the honest fit admitted 101,232 tokens at chunk
# 128 with unbounded partials, which covers the prompt with room to spare,
# so the refusal measured the runner's margin, not the card. The fit
# deciding the prompt itself does not fit is what this cell gates; the
# margin must not be what makes it fail. (The record's 16 GiB u8:i4 serves
# of this prompt, DESIGN §7.0.2ac/§7.0.2ad, ran with the partition bound at
# 32; this cell runs the unbounded default on purpose, the configuration a
# user gets without the knob.)
# One tolerance for the sizer and the margin: changing --tol without the
# margin (or the reverse) would silently under-provision, the very failure
# being fixed here.
TOL_PCT=2
N_CTX=$((PREFILL_TOKENS + PREFILL_TOKENS * TOL_PCT / 100 + 32 + 512))

WORK=$(mktemp -d)
trap 'stop_server; rm -rf "$WORK"' EXIT
SRV=""
FAILED=0

pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1"; FAILED=$((FAILED + 1)); }

stop_server() {  # SIGTERM, a bounded wait, then SIGKILL -- a wedged server on
  # this project has ignored SIGTERM before; a bare kill -TERM; wait would
  # hang this cleanup forever.
  [[ -n "$SRV" ]] || return 0
  kill -TERM "$SRV" 2>/dev/null
  for _ in $(seq 1 30); do
    kill -0 "$SRV" 2>/dev/null || break
    sleep 1
  done
  kill -KILL "$SRV" 2>/dev/null
  wait "$SRV" 2>/dev/null
  SRV=""
}

start_server() {  # start_server <log> <device> <extra args...>
  local log="$1" dev="$2"; shift 2
  stop_server
  PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()")
  "$BIN" --model "$MODEL" --device "$dev" --host 127.0.0.1 --port "$PORT" \
         --n-ctx "$N_CTX" "$@" > "$log" 2>&1 &
  SRV=$!
  # Generous: a cold compile at this depth is minutes even before the fit
  # question is settled.
  for _ in $(seq 1 2400); do
    curl -fsS "http://127.0.0.1:$PORT/health" -o /dev/null 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "server died:"; tail -20 "$log"; return 1; }
    sleep 1
  done
  echo "server never became ready"; return 1
}

ask() {  # ask <outfile> <prompt-file> [max_tokens]
  python3 - "$PORT" "$1" "$2" "${3:-32}" <<'PY'
import json, sys, urllib.error, urllib.request
port, out, pf, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
prompt = open(pf, encoding="utf-8").read()
d = {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
     "max_tokens": n, "chat_template_kwargs": {"enable_thinking": False}}
r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                           data=json.dumps(d).encode(),
                           headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(r, timeout=3600) as resp:
        o = json.load(resp)
except urllib.error.HTTPError as e:
    print(f"       HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}", file=sys.stderr)
    sys.exit(1)
txt = o["choices"][0]["message"]["content"] or ""
u = o.get("usage", {})
open(out, "w", encoding="utf-8").write(txt)
print(f"       prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')}")
sys.exit(0 if txt else 1)
PY
}

lines() {  # lines <log>
  grep -a -E "prefill|decode" "$1" | grep -a "t/s" | tail -4 | sed 's/^/       /'
}

metric_value() {  # metric_value <log> <prefill|decode> last
  # Unlike tier_reference.sh's own metric_value, this cell has no separate
  # sizing server: size_prompt.py's own calibration pings (max_tokens=1, one
  # to five rounds -- read: it, too, prints a prefill/decode line) share the
  # SAME log as the one real completion this cell asks. A fixed index (say,
  # 1) would therefore name a calibration round, not the request -- the
  # exact "differently-shaped log" failure docs/design-0.3.1-test-ladder.md
  # §8.7 warns about. The known index that IS stable here is the last one:
  # nothing asks this server anything after the real completion, so the
  # last matching line is always it, regardless of how many calibration
  # rounds size_prompt.py needed this time.
  grep -a -E "$2" "$1" | grep -a "t/s" | tail -1 \
    | grep -a -o -E '[0-9]+\.[0-9]+ t/s' | grep -a -o -E '^[0-9]+\.[0-9]+'
}

emit_metric() {  # emit_metric <metric> <value> <unit> -- silent if <value> is empty
  [[ -n "$2" ]] && printf 'ACCEPTANCE-METRIC %s %s %s\n' "$1" "$2" "$3"
}

echo "== depth-ladder: ${PREFILL_TOKENS}-token prefill at u8 and u8:i4, on both cards (n-ctx $N_CTX)"

SEED="$REPO_ROOT/tests/acceptance/prompts/filler-seed.txt"

run_cell() {  # run_cell <tag> <device>
  local tag="$1" dev="$2" precision
  precision="${tag#*-}"
  local log="$WORK/$tag.log"
  echo "  -- $tag (device $dev, --paged-kv $precision)"

  if ! start_server "$log" "$dev" --paged-kv "$precision"; then
    fail "$tag: the fit was accepted (server failed to load at n-ctx $N_CTX)"
    return
  fi

  # Size the prompt ONCE, in the first cell whose server starts, and reuse
  # it: the token count is the tokenizer's, the same for every cell of one
  # artifact, while each sizing round is itself a full prefill at this depth
  # (the first real run measured two rounds on the 24 GB card at u8:i4,
  # 78,867 then 98,187 tokens: 546 s and 812 s before the measured request
  # even started). The ladder's order puts u8 first on purpose: its rounds
  # are the cheap ones, and only if that server fails to start does the
  # next cell size instead (size_prompt.py writes its output only after
  # passing tolerance, so a partial prompt is never reused). Re-sizing per cell also
  # changed the cell's shape from the 0.3.0 gate's one prefill per process
  # to three, and the third faulted (CL_OUT_OF_RESOURCES) where one had
  # passed -- a finding for the u8i4-deep-prefill-fault campaign, not a
  # shape this runner should impose. Only the sizing cell's process carries
  # the rounds; every later cell asks exactly one request.
  local promptfile="$WORK/prompt.txt"
  if [[ ! -s "$promptfile" ]]; then
    if ! python3 "$REPO_ROOT/tests/acceptance/size_prompt.py" \
           --seed "$SEED" --target "$PREFILL_TOKENS" --tol "$TOL_PCT" \
           --url "http://127.0.0.1:$PORT" --out "$promptfile"; then
      fail "$tag: the reference prompt could be sized"
      stop_server
      return
    fi
  else
    echo "       prompt reused from the first cell ($(wc -c < "$promptfile") bytes, sized once per artifact)"
  fi

  local outfile="$WORK/$tag-out.txt" req_ok=1
  if ask "$outfile" "$promptfile" 32 && [[ -s "$outfile" ]]; then
    pass "$tag: served a non-empty completion (HTTP 200)"
  else
    fail "$tag: served a non-empty completion (HTTP 200)"
    req_ok=0
  fi

  # The fault signatures on the record for this exact prefill depth
  # (tests/test_fit.cpp's packed_values_prefill_scratch_bytes comment,
  # DESIGN §7.0.2ad): a GPU page-fault storm reported as "VM worker error:
  # -12" and "exec queue reset detected", the driver's own "Engine memory
  # CAT error" lines, and the runtime's CL_OUT_OF_RESOURCES. Any of them is
  # a real failure of this cell, not a note.
  fault=$(grep -a -o -E 'CL_OUT_OF_RESOURCES|error: -12|exec queue reset|page fault|CAT error' \
              "$log" | sort -u | tr '\n' ';' | sed 's/;$//')
  if [[ -n "$fault" ]]; then
    fail "$tag: no GPU-fault / out-of-resources line in the log (found: $fault)"
  else
    pass "$tag: no GPU-fault / out-of-resources line in the log"
  fi

  lines "$log"

  # Names carry card and precision (docs/design-0.3.1-test-ladder.md §8.7:
  # "both cards under one name" -- a bare `decode` would average two cards).
  # <tag> is "<card>-<precision>" (e.g. "large-u8:i4"); the colon in a
  # u8:i4 precision is stripped so the metric name has none.
  local card="${tag%%-*}" prec_token="${precision//:/}"
  # Emitted only if the one request this cell asks actually succeeded
  # (Increment-3 review, 2026-09-05, MAJOR 4) -- a failed request has no
  # rate to report, and the cell fails anyway (above) without one.
  if [[ "$req_ok" == 1 ]]; then
    local decode_v prefill_v
    decode_v=$(metric_value "$log" decode last)
    prefill_v=$(metric_value "$log" prefill last)
    emit_metric "decode-${card}-${prec_token}"  "$decode_v"  t/s
    emit_metric "prefill-${card}-${prec_token}" "$prefill_v" t/s
  fi

  stop_server
}

run_cell "large-u8"     "$DEV_LARGE"
run_cell "large-u8:i4"  "$DEV_LARGE"
run_cell "small-u8"     "$DEV_SMALL"
run_cell "small-u8:i4"  "$DEV_SMALL"

echo
if [[ $FAILED -eq 0 ]]; then echo "depth-ladder: all checks passed"; exit 0; fi
echo "depth-ladder: ${FAILED} check(s) failed"; exit 1
