#!/usr/bin/env bash
# clean-turns step 1: the endpoint pair, on a card seat.
#
# What it runs: one serving process built from this branch, serving the coder
# artifact with the served unit's own flags, and tools/clean_turns_gate.py
# against it. Off and on are two values of the SAME request field, so one
# process serves both arms and the artifact loads once.
#
# It refuses to run without a seat named on the command line. Stopping a
# serving unit is the operator's decision (docs/sop-card-window.md §3, and
# mneme's record of an engineer's own rollout killing his own agent loop at the
# unit he was building on), so the grant is an argument, not a guess:
#
#   tools/clean_turns_leg.sh --seat "operator <name>, <iso time>" \
#       --binary /home/.../build/arcint --gguf /models/.../x.gguf \
#       --device GPU.1 --port 18080
#
# The SOP is the script's, not the operator's: the sampler starts BEFORE the
# leg and the leg aborts if its log is not growing; card identity is printed
# from PCI ids, never from the DRM number (DRM numbering is inverted against
# arcint's GPU.N on this hardware); the process is swept by pid at the end,
# never pkill -f (a pattern inside the command line matches its own chain).
set -uo pipefail

SEAT="" BIN="" GGUF="" DEVICE="GPU.1" PORT="18080" NCTX="98304" MTP="4"
KV="q8_0:q4_0" EXTRA="" MODEL="qwen3.6-coder"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seat)    SEAT="$2"; shift 2 ;;
    --binary)  BIN="$2"; shift 2 ;;
    --gguf)    GGUF="$2"; shift 2 ;;
    --device)  DEVICE="$2"; shift 2 ;;
    --port)    PORT="$2"; shift 2 ;;
    --n-ctx)   NCTX="$2"; shift 2 ;;
    --mtp)     MTP="$2"; shift 2 ;;
    --kv)      KV="$2"; shift 2 ;;
    --model)   MODEL="$2"; shift 2 ;;
    --extra)   EXTRA="$2"; shift 2 ;;
    *) echo "clean_turns_leg: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$SEAT" ]]; then
  echo "clean_turns_leg: no seat named. Pass --seat 'operator <name>, <iso time>'." >&2
  echo "  The seat covers BOTH cards being free of serving units; this script does" >&2
  echo "  not stop or start any unit, and does not decide to occupy a card." >&2
  exit 2
fi
for tool in curl python3 awk; do
  command -v "$tool" >/dev/null || { echo "clean_turns_leg: $tool is required" >&2; exit 77; }
done
[[ -x "$BIN" ]] || { echo "clean_turns_leg: no binary at $BIN" >&2; exit 2; }
[[ -f "$GGUF" ]] || { echo "clean_turns_leg: no artifact at $GGUF" >&2; exit 2; }

here="$(cd "$(dirname "$0")" && pwd)"
stamp="$(date -u +%Y%m%d-%H%M%SZ)"
logdir="${TMPDIR:-/tmp}/clean-turns-leg-$stamp"
mkdir -p "$logdir"
sam="$logdir/sampler.log"; srv="$logdir/server.log"; gate="$logdir/gate.out"

# SOP §2: the card is named by PCI id. DRM numbering is inverted against
# arcint's GPU.N (OpenCL enumeration) on this hardware, so the mapping is
# printed, never assumed.
{
  echo "=== clean-turns step 1 leg, $stamp"
  echo "seat: $SEAT"
  echo "binary: $BIN"
  echo "  sha256 $(sha256sum "$BIN" | cut -d' ' -f1)"
  echo "artifact: $GGUF"
  echo "  sha256(head 1MiB+size) $( { head -c 1048576 "$GGUF"; stat -c%s "$GGUF"; } | sha256sum | cut -d' ' -f1)"
  echo "arcint device: $DEVICE (n_ctx $NCTX, mtp $MTP, kv $KV, port $PORT)"
  echo "cards, by PCI id (arcint GPU.N is the OpenCL order, NOT this order):"
  for c in /sys/class/drm/card[0-9]; do
    printf '  %-8s vendor %s device %s\n' "$(basename "$c")" \
      "$(cat "$c/device/vendor" 2>/dev/null)" "$(cat "$c/device/device" 2>/dev/null)"
  done
  echo "host memory at start:"
  awk '/MemTotal|MemAvailable/ {printf "  %s %s\n", $1, $2}' /proc/meminfo
  echo "arc stats: $(awk '/^size/{print "  arc size " $2}' /proc/spl/kstat/zfs/arcstats 2>/dev/null || echo '  arcstats absent')"
  echo "processes holding the devices at start:"
  ps -eo pid,comm,args | awk '/[a]rcint --engine/ {print "  pid " $1 " " substr($0, index($0,$2))}' || true
} | tee "$logdir/header.txt"

# SOP §1: the sampler starts here, before the leg, and the leg aborts if the
# log is not growing. The process match is anchored to the binary path.
(
  bin_re="^${BIN}\b"
  while :; do
    ts=$(date -u +%H:%M:%S)
    pid=$(pgrep -f "$bin_re" | head -1 || true)
    if [[ -n "$pid" ]]; then
      rss=$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null || echo 0)
      drm=$(grep -h -E 'drm-resident' /proc/$pid/fdinfo/* 2>/dev/null | awk '{s+=$2} END{print s+0}')
      echo "$ts rss_kb=$rss drm_resident_kb=$drm"
    else
      echo "$ts no-leg-process"
    fi
    awk '/MemAvailable/ {print $1" "$2}' /proc/meminfo | tr ' ' '=' | tr '\n' ' '
    echo
    sleep 2
  done
) > "$sam" 2>&1 &
sampid=$!
sleep 5
[[ -s "$sam" ]] || { echo "clean_turns_leg: sampler log not growing, refusing to run" >&2; kill "$sampid" 2>/dev/null; exit 1; }

# The watchdog: physical MemAvailable below 4 GiB kills the leg, not the box's
# luck. Driver/USM-host memory is charged to the physical host, so a container
# fence would not bound it (SOP §1).
(
  while [[ -d /proc/$sampid ]]; do
    avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if (( avail < 4194304 )); then
      pids=$(pgrep -f "^${BIN}\b" || true)
      [[ -n "$pids" ]] && kill -9 $pids
      echo "WATCHDOG: MemAvailable $avail kB, leg killed" >> "$sam"
      break
    fi
    sleep 2
  done
) &
wdpid=$!

# The server. No operator flag: the gate's off arm is the default, its on arm
# asks the field per request, which is what the default-off claim needs.
"$BIN" --engine llama --gguf "$GGUF" --served-model-name "$MODEL" \
       --device "$DEVICE" --host 127.0.0.1 --port "$PORT" --n-ctx "$NCTX" \
       --llama-mtp "$MTP" --llama-kv "$KV" --llama-checkpoints 0 \
       --queue-timeout 60 $EXTRA > "$srv" 2>&1 &
srvpid=$!

ready=0
for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$srvpid" 2>/dev/null || break
  sleep 2
done
if (( ! ready )); then
  echo "clean_turns_leg: the server never became healthy; last 20 lines:" >&2
  tail -20 "$srv" >&2
  kill -9 "$srvpid" 2>/dev/null
  kill "$sampid" "$wdpid" 2>/dev/null
  exit 1
fi
echo "server up after load; /props reports the switch:"
curl -fsS "http://127.0.0.1:$PORT/props" | python3 -c '
import json,sys
d=json.load(sys.stdin)
sd=d.get("sampler_defaults") or {}
print("  think_eog_mask =", sd.get("think_eog_mask", "ABSENT"))
print("  build:", (d.get("build") or {}).get("git"), (d.get("build") or {}).get("version"))
'

# The pair. 4096-token cap: at 64 the same shape comes back as `length`, which
# is why the earlier arms hid it (campaign note, step 1 at the endpoint).
timeout 1800 python3 "$here/clean_turns_gate.py" "http://127.0.0.1:$PORT" "$MODEL" \
  > "$gate" 2>&1
grc=$?
cat "$gate"

# SOP §5: sweep by pid, never pkill -f with a pattern that lives in this line.
kill -9 "$srvpid" 2>/dev/null
for p in $(pgrep -f "^${BIN}\b" || true); do kill -9 "$p" 2>/dev/null; done
kill "$sampid" "$wdpid" 2>/dev/null
sleep 3
left=$(pgrep -cf "^${BIN}\b" || true)
{
  echo "=== release $(date -u +%Y%m%d-%H%M%SZ): leftover leg processes: $left"
  awk '/MemAvailable/ {print "  " $1 " " $2}' /proc/meminfo
} | tee -a "$logdir/header.txt"

echo
echo "log: $logdir (header.txt, sampler.log, server.log, gate.out)"
echo "gate exit: $grc  (0 = the pair paired; 1 = FAILED or NOT SHOWN, see gate.out)"
exit $grc
