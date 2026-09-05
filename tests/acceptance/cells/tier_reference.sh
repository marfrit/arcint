#!/usr/bin/env bash
# tier-reference-cell (DESIGN §3.4, §7.0.2ae/§7.0.2af/§7.0.2ai;
# docs/design-0.3.1-test-ladder.md Increment 2, corrected 2026-09-05 by the
# Increment-3 review). §3.4's actual invariant is HISTORY-INDEPENDENCE, not
# ON-equals-OFF: the static partition must not change what a given arm says
# across processes and requests -- two OFF processes and two ON processes,
# each asked twice, must each agree WITH ITSELF. Device f16 vs host f32
# arithmetic is not bit-equal (DESIGN §7.0.2ae/§7.0.2af), so tier ON is
# permitted to differ from tier OFF on a given prompt; the 0.3.0 gate's
# ON-equals-OFF agreement was a property of that one sample, never a claim
# the engine makes. This cell gates ON self-identity and OFF self-identity
# separately and only REPORTS whether ON agrees with OFF this time (and
# where the first difference falls, if not) -- a report, because a
# same-or-different reading here is not itself a defect either way.
# E2 additionally checks a turn: CONT (the prompt plus one added
# sentence) served from the SAME tier-on process that was already asked
# PROMPT twice -- so its KV state for that prefix is warm -- must match CONT
# served from a FRESH process that has never seen PROMPT at all. That is
# proof that whatever the tier keeps resident travels with the rest of a
# process's KV state, not just the on-device pages.
#
#   tier_reference.sh <arcint> <model-dir> <device> [server args...]
#
# ARCINT_MOE_DEVICE_POOL_BYTES is honoured the same way ARCINT_EXTRA_ARGS is
# in tests/equivalence/run.sh: forwarded through the environment, unparsed,
# to every server this script starts. ARCINT_EXTRA_ARGS itself is NOT
# honoured here and is unset below -- this cell's configuration is entirely
# its own positional args (cells.json), and an ambient ARCINT_EXTRA_ARGS left
# over from another cell in the same shell must not leak into it.
set -uo pipefail
unset ARCINT_EXTRA_ARGS

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

BIN="${1:?usage: tier_reference.sh <arcint> <model-dir> <device> [server args...]}"
MODEL="${2:?usage: tier_reference.sh <arcint> <model-dir> <device> [server args...]}"
DEV="${3:?usage: tier_reference.sh <arcint> <model-dir> <device> [server args...]}"
shift 3
SERVER_ARGS=("$@")

for tool in python3 curl; do
  command -v "$tool" >/dev/null || { echo "tier_reference: $tool is required" >&2; exit 77; }
done
[[ -d "$MODEL" ]] || { echo "tier_reference: model dir $MODEL not found" >&2; exit 77; }

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

start_server() {  # start_server <log> <extra args...>
  local log="$1"; shift
  stop_server
  PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()")
  # Prefix cache off throughout: this cell is about the tier, not the disk
  # cache, and E2 needs the same "no prefix cache" configuration to isolate
  # what --moe-cpu-tier itself keeps resident across a turn.
  "$BIN" --model "$MODEL" --device "$DEV" --host 127.0.0.1 --port "$PORT" \
         --prefix-cache-mib 0 "${SERVER_ARGS[@]}" "$@" \
         > "$log" 2>&1 &
  SRV=$!
  # Generous: a cold compile of a 35B MoE artifact under a static partition is
  # minutes, and a suite that times out looks like a failing gate when it is not.
  for _ in $(seq 1 2400); do
    curl -fsS "http://127.0.0.1:$PORT/health" -o /dev/null 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "server died:"; tail -20 "$log"; return 1; }
    sleep 1
  done
  echo "server never became ready"; return 1
}

ask() {  # ask <outfile> <prompt-file> [max_tokens] -- exits non-zero on any
  # HTTP or network failure, so callers must check it rather than let a
  # missing output file be silently read later as "not compared".
  python3 - "$PORT" "$1" "$2" "${3:-64}" <<'PY'
import json, sys, urllib.request
port, out, pf, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
prompt = open(pf, encoding="utf-8").read()
d = {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
     "max_tokens": n, "chat_template_kwargs": {"enable_thinking": False}}
r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                           data=json.dumps(d).encode(),
                           headers={"Content-Type": "application/json"})
o = json.load(urllib.request.urlopen(r, timeout=1800))
u = o.get("usage", {})
open(out, "w", encoding="utf-8").write(o["choices"][0]["message"]["content"] or "")
print(f"       prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')}")
PY
}

lines() {  # lines <log> -- the server's own reported prefill/decode rates (reported, not gated)
  # The server's per-request lines are "slot N: prefill|decode ... t/s". A
  # looser "prefill|decode" + "t/s" also matched the load banner ("62.7 t/s
  # decode at 53.5k, 1584 t/s prefill" -- the artifact's recorded rates),
  # which shifted every fixed index by one on the first real run of this
  # runner: the "2nd request" metrics were the first request's.
  grep -a -E "slot [0-9]+: (prefill|decode) " "$1" | tail -4 | sed 's/^/       /'
}

metric_value() {  # metric_value <log> <prefill|decode> <request-index, 1-based>
  # Pulls the t/s figure out of the <index>-th matching line of <log>, in the
  # order the server printed them -- a KNOWN request index, not a tail of the
  # log (docs/design-0.3.1-test-ladder.md §8.7: "a differently-shaped log ...
  # the first gate to fire will be a process that logged differently"). Each
  # request here produces exactly one prefill line and one decode line, so
  # index 2 names the second (warm) request unambiguously.
  grep -a -E "slot [0-9]+: $2 " "$1" | sed -n "${3}p" \
    | grep -a -o -E '[0-9]+\.[0-9]+ t/s' | grep -a -o -E '^[0-9]+\.[0-9]+'
}

emit_metric() {  # emit_metric <metric> <value> <unit> -- silent if <value> is empty
  [[ -n "$2" ]] && printf 'ACCEPTANCE-METRIC %s %s %s\n' "$1" "$2" "$3"
}

echo "== tier-reference-cell on $DEV (${SERVER_ARGS[*]})"

# ------------------------------------------------------------- prompt sizing
#
# 1,198 tokens is the reference depth (DESIGN §7.0.2ai). Size it against this
# device's own tokenizer -- a byte or word count does not carry over between
# artifacts -- and print what was actually measured rather than assuming it.
SEED="$REPO_ROOT/tests/acceptance/prompts/filler-seed.txt"
PROMPT="$WORK/prompt.txt"
if ! start_server "$WORK/size.log"; then
  fail "a server came up to size the reference prompt"
  exit 1
fi
if ! python3 "$REPO_ROOT/tests/acceptance/size_prompt.py" \
       --seed "$SEED" --target 1198 --tol 3 --url "http://127.0.0.1:$PORT" --out "$PROMPT"; then
  fail "the reference prompt could be sized"
  exit 1
fi
stop_server

CONT="$WORK/cont.txt"
python3 - "$PROMPT" "$CONT" <<'PY'
import sys
prompt = open(sys.argv[1], encoding="utf-8").read()
open(sys.argv[2], "w", encoding="utf-8").write(prompt + " Then add one more sentence about block alignment.")
PY

# ----------------------------------------------------------- OFF / ON arms
#
# Two requests per (fresh) process, four fresh processes: off1, off2 (the
# baseline, twice, to know what "identical" means before the tier is even
# involved) and on1, on2 (--moe-cpu-tier, twice).
for arm in off1 off2 on1 on2; do
  extra=()
  [[ "$arm" == on* ]] && extra=(--moe-cpu-tier)
  echo "  -- $arm"
  if ! start_server "$WORK/$arm.log" "${extra[@]}"; then
    fail "$arm: server started"
    continue
  fi
  ask "$WORK/$arm-run1.txt" "$PROMPT" 64 || fail "$arm run1: request succeeded"
  ask "$WORK/$arm-run2.txt" "$PROMPT" 64 || fail "$arm run2: request succeeded"
  lines "$WORK/$arm.log"
  stop_server
done

# --------------------------------------------------------- ACCEPTANCE-METRIC
#
# off2 and on2 stand for their arm here: the FIRST process of a window is
# the cold one on the record (DESIGN §7.0.2ai's cold-sequence warming, and
# the follow-up window of §7.0.2ak, where off1 decoded at 0.3 t/s after the
# host's file cache had been churned and off2 at 11.9 seconds later) -- that
# cost belongs to the static-partition-cold-start campaign, not to a decode
# reference; the first processes' lines are still reported above. The 2nd
# request of each (index 2, docs/design-0.3.1-test-ladder.md §8.7) is the
# warm one DESIGN §7.0.2ai's reference cell reports. run.py, not this
# script, decides whether any of these numbers is a regression -- it holds
# the reference and the gate (§8.2).
decode_off=$(metric_value "$WORK/off2.log" decode 2)
prefill_off=$(metric_value "$WORK/off2.log" prefill 2)
decode_on=$(metric_value "$WORK/on2.log" decode 2)
prefill_on=$(metric_value "$WORK/on2.log" prefill 2)
emit_metric decode-warm-2nd-off  "$decode_off"  t/s
emit_metric prefill-warm-2nd-off "$prefill_off" t/s
emit_metric decode-warm-2nd-on   "$decode_on"   t/s
emit_metric prefill-warm-2nd-on  "$prefill_on"  t/s
if [[ -n "$decode_off" && -n "$decode_on" ]]; then
  # Two decimals: the first real run put the ratio at 1.31-1.34 against a
  # band derived by §8.3 from those samples, and one decimal would quantise
  # both the sample and its gate to the same 1.3 -- a gate with no margin
  # to read. On/off from the SAME window (this invocation), which is why it
  # cancels the process-to-process drift a lone decode-warm-2nd-on figure
  # cannot (§8.1's own reason for gating the ratio as well as the rate).
  ratio=$(python3 -c "print(f'{${decode_on}/${decode_off}:.2f}')" 2>/dev/null || true)
  emit_metric decode-ratio-on-off "$ratio" ratio
fi

# grouped_fallbacks is the plugin's own [OTD_PERF] counter (DESIGN §7.0.2ai,
# §4561-4580); it exists only on a patched runtime, so its absence from the
# log is not this cell's failure to detect -- §8's amendment says a cell may
# print nothing rather than invent a zero, and the missing-metric rule stays
# silent here because tier-reference-cell's references are still null.
grouped_fallbacks=$(grep -a -o -E 'grouped_fallbacks=[0-9]+' "$WORK/on2.log" | tail -1 | grep -a -o -E '[0-9]+$')
emit_metric grouped-fallbacks-on "$grouped_fallbacks" count

# ------------------------------------------- byte-identity, WITHIN one arm
#
# §3.4's invariant, corrected (Increment-3 review, 2026-09-05): each arm
# must agree with ITSELF across its own two fresh processes and two
# requests, not with the other arm -- device f16 vs host f32 arithmetic is
# not bit-equal (DESIGN §7.0.2ae/§7.0.2af), so tier ON differing from tier
# OFF on a given prompt is permitted, and gating that would gate a claim the
# engine never made. Every non-baseline output in a group must be present
# AND identical to that group's own baseline: a missing file is a FAIL here,
# never silently dropped from the comparison, and the baseline is never
# compared with itself -- that would always "pass" even if every other
# request in the group failed.
identity_group() {  # identity_group <label> <baseline-name> <other-name>...
  local label="$1" baseline_name="$2"; shift 2
  local base="$WORK/$baseline_name.txt"
  echo "  -- $label: byte-identity across its own processes and requests, against $baseline_name"
  if [[ ! -s "$base" ]]; then
    fail "$baseline_name produced a baseline output for $label to compare the rest against"
    return
  fi
  local total=$# compared=0
  for name in "$@"; do
    local cand="$WORK/$name.txt"
    if [[ ! -f "$cand" ]]; then
      fail "$name byte-identical to $baseline_name (no output was produced)"
      continue
    fi
    compared=$((compared + 1))
    if cmp -s "$base" "$cand"; then
      pass "$name byte-identical to $baseline_name"
    else
      fail "$name byte-identical to $baseline_name"
      # The work dir is gone when this script exits, so a divergence must be
      # diagnosable from the log alone (the follow-up window of 2026-09-05
      # found on2 differing from on1 and had nothing left to read but the
      # verdict): both texts, verbatim, and where they first part.
      python3 - "$base" "$cand" "$baseline_name" "$name" <<'PY'
import sys, hashlib
a = open(sys.argv[1], encoding="utf-8").read(); b = open(sys.argv[2], encoding="utf-8").read()
n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
print(f"       first differing character at index {n} of {max(len(a), len(b))}")
for label, t in ((sys.argv[3], a), (sys.argv[4], b)):
    print(f"       {label} sha256={hashlib.sha256(t.encode()).hexdigest()[:16]} text={t!r}")
PY
    fi
  done
  echo "       compared $compared of $total outputs against the $baseline_name baseline"
  # Every output's hash, pass or fail, so two windows can be compared later
  # without either's work dir.
  for name in "$baseline_name" "$@"; do
    if [[ -f "$WORK/$name.txt" ]]; then
      echo "       $name sha256=$(sha256sum "$WORK/$name.txt" | cut -c1-16)"
    fi
  done
}

identity_group "tier ON"  on1-run1  on1-run2 on2-run1 on2-run2
identity_group "tier OFF" off1-run1 off1-run2 off2-run1 off2-run2

# ----------------------------------------------------- REPORT: ON vs OFF
#
# Not gated (see the header comment): whether tier ON agrees with tier OFF
# on this prompt is data about the arithmetic, not a pass/fail claim. Real
# per-token boundaries are not available here (the API returns text, not a
# token array), so a divergence is reported as the first differing
# CHARACTER, said explicitly rather than dressed up as a token index.
if [[ -s "$WORK/on1-run1.txt" && -s "$WORK/off1-run1.txt" ]]; then
  python3 - "$WORK/on1-run1.txt" "$WORK/off1-run1.txt" <<'PY'
import sys
on = open(sys.argv[1], encoding="utf-8").read()
off = open(sys.argv[2], encoding="utf-8").read()
if on == off:
    print("  --   tier ON vs OFF: identical")
else:
    n = 0
    for a, b in zip(on, off):
        if a != b:
            break
        n += 1
    length = max(len(on), len(off))
    print(f"  --   tier ON vs OFF: differ (first differing character at index {n} "
          f"of {length}; permitted, DESIGN §7.0.2ae)")
PY
else
  echo "  --   tier ON vs OFF: not compared (on1-run1 or off1-run1 missing; see the FAILs above)"
fi

# --------------------------------------------------------------------- E2
#
# Tier ON, prefix cache off (already the default above): PROMPT, PROMPT,
# CONT in one process (e2a); a fresh process asked only CONT (e2b).
echo "  -- E2 (tier on, prefix cache off)"
if start_server "$WORK/e2a.log" --moe-cpu-tier; then
  ask "$WORK/e2a-p1.txt" "$PROMPT" 64 || fail "E2: PROMPT request 1 (warm process) succeeded"
  ask "$WORK/e2a-p2.txt" "$PROMPT" 64 || fail "E2: PROMPT request 2 (warm process) succeeded"
  ask "$WORK/e2a-cont.txt" "$CONT" 64 || fail "E2: CONT request (warm process) succeeded"
  lines "$WORK/e2a.log"
  stop_server
else
  fail "E2: the tier-on process for PROMPT/PROMPT/CONT started"
fi

if start_server "$WORK/e2b.log" --moe-cpu-tier; then
  ask "$WORK/e2b-cont.txt" "$CONT" 64 || fail "E2: CONT request (fresh process) succeeded"
  lines "$WORK/e2b.log"
  stop_server
else
  fail "E2: the fresh process asked only CONT started"
fi

if [[ -f "$WORK/e2a-p1.txt" && -f "$WORK/e2a-p2.txt" ]]; then
  cmp -s "$WORK/e2a-p1.txt" "$WORK/e2a-p2.txt" \
    && pass "E2: PROMPT == PROMPT (same process)" \
    || fail "E2: PROMPT == PROMPT (same process)"
else
  fail "E2: PROMPT == PROMPT (same process) (one or both outputs missing)"
fi
if [[ -f "$WORK/e2a-cont.txt" && -f "$WORK/e2b-cont.txt" ]]; then
  cmp -s "$WORK/e2a-cont.txt" "$WORK/e2b-cont.txt" \
    && pass "E2: CONT from the warm process == CONT from a fresh process" \
    || fail "E2: CONT from the warm process == CONT from a fresh process"
else
  fail "E2: CONT from the warm process == CONT from a fresh process (one or both outputs missing)"
fi

echo
if [[ $FAILED -eq 0 ]]; then echo "tier-reference-cell: all checks passed"; exit 0; fi
echo "tier-reference-cell: ${FAILED} check(s) failed"; exit 1
