#!/usr/bin/env bash
# guard_check.sh <dir with llama-perplexity> <Inkling 2 Mini GGUF> <text file> <out dir> [ngl]
#
# The layer guard of the port arm on the REAL graph builder and the REAL scheduler
# (fa_port_gate.h: fa_port_layer_attaches; src/models/inkling2.cpp; build_attn_mha).
# Runs of llama-perplexity on the same text:
#   base    the knob unset, batches of 64 rows; its logits are saved
#   ctrl    the knob unset again, against base: what "the same run" reads as (the saved
#           logits are quantised, so a run against itself is not exactly KL 0)
#   knob    GGML_OPENCL_FA_PORT=1, batches of 64 rows (a prompt batch: the port's shape)
#   decode  GGML_OPENCL_FA_PORT=1, batches of 1 row (a decode step)
#   red     batches of 64 with GGML_OPENCL_FA_PORT_RED=5 (the form before the guard: every
#           layer gets the sources)
# knob, decode and red run with GGML_SCHED_DEBUG=2, and the script reads the scheduler's own
# node list: one row a FLASH_ATTN node - its ordinal in the graph (the attention layers in
# order), the backend the scheduler assigned it to (where it RAN), and whether it carries
# the port's sources (the positions among its sources).
#
# THE RULE, any ngl: no FLASH_ATTN node that ran on the CPU carries the port's sources, in
# the prompt batch or in the decode step; the script exits non-zero otherwise. A layer whose
# backend does not declare the sources never gets them; a layer whose backend declares them
# but refuses the op gets today's op back before the scheduler sees it.
# With ngl 0 (the default: every layer on the CPU) it also requires: the DOWNGRADE line in
# knob; knob's statistics against base equal to ctrl's line for line (the run with the knob
# is the run without it); and the red breaking THE RULE (the guard is what holds it).
# With ngl > 0 on a card host (some or all layers on the card) it requires THE RULE and
# prints how many nodes ran on the card with the sources (the port in use).
# A model-sized job: run it under the host's lock for such jobs, if it has one.
set -uo pipefail
BIN=${1:?dir with llama-perplexity}; M=${2:?model}; TXT=${3:?text}; O=${4:?out dir}; NGL=${5:-0}
mkdir -p "$O"
# every run sets the port's knobs itself, whatever the caller exported: base and ctrl run
# with the knob UNSET, knob and decode with the knob and without the red, red with both
off() { env -u GGML_OPENCL_FA_PORT -u GGML_OPENCL_FA_PORT_RED -u GGML_OPENCL_FA_PORT_TRACE -u GGML_SCHED_DEBUG "$@"; }
on()  { env -u GGML_OPENCL_FA_PORT_RED GGML_OPENCL_FA_PORT=1 GGML_SCHED_DEBUG=2 "$@"; }
P64="-c 256 -b 64 --chunks 2"; P1="-c 32 -b 1 --chunks 1"
off "$BIN/llama-perplexity" -m "$M" -f "$TXT" -ngl "$NGL" -fa on $P64 --kl-divergence-base "$O/base.kld" > "$O/base.log" 2>&1 || { echo "FAIL: base run"; exit 2; }
off "$BIN/llama-perplexity" -m "$M" -f "$TXT" -ngl "$NGL" -fa on $P64 --kl-divergence-base "$O/base.kld" --kl-divergence > "$O/ctrl.log" 2>&1 || { echo "FAIL: ctrl run"; exit 2; }
on "$BIN/llama-perplexity" -m "$M" -f "$TXT" -ngl "$NGL" -fa on $P64 -v --kl-divergence-base "$O/base.kld" --kl-divergence > "$O/knob.log" 2>&1 || { echo "FAIL: knob run"; exit 2; }
on "$BIN/llama-perplexity" -m "$M" -f "$TXT" -ngl "$NGL" -fa on $P1 -v > "$O/decode.log" 2>&1 || { echo "FAIL: decode run"; exit 2; }
env GGML_OPENCL_FA_PORT=1 GGML_SCHED_DEBUG=2 GGML_OPENCL_FA_PORT_RED=5 "$BIN/llama-perplexity" -m "$M" -f "$TXT" -ngl "$NGL" -fa on $P64 -v --kl-divergence-base "$O/base.kld" --kl-divergence > "$O/red.log" 2>&1 || { echo "FAIL: red run"; exit 2; }
# the scheduler's node list -> "ordinal backend sources" a FLASH_ATTN node, distinct rows with their count
tab() { awk '
    /## SPLIT #0:/ { n = 0 }
    /node #[ 0-9]+ \(FLASH_ATTN/ {
        b = $0; sub(/^[^\[]*\[ */, "", b); sub(/ .*/, "", b)
        s = ($0 ~ /attn_inp_port_pos/) ? "yes" : "no"
        printf "%3d %-6s %s\n", n++, b, s
    }' "$1" | sort -k1,1n -k2 -k3 | uniq -c | awk '{printf "  attention %3d  ran on %-6s  port sources %-3s  (%d graph prints)\n", $2, $3, $4, $1}'; }
viol() { tab "$1" | grep -c "ran on CPU *port sources yes" || true; }
used() { tab "$1" | grep -v "ran on CPU" | grep -c "port sources yes" || true; }
rows() { tab "$1" | wc -l; }
stats() { sed -n '/Perplexity statistics/,$p' "$1" | sed 's/^[0-9.]* [A-Z] //' | grep -E "KLD|top p|Δp|PPL|percentile|Maximum|Minimum|Median|RMS"; }
for r in knob decode red; do echo "$r: FLASH_ATTN rows $(rows "$O/$r.log"); ran on the CPU WITH the port's sources: $(viol "$O/$r.log"); ran off the CPU with them: $(used "$O/$r.log")"; done
echo "the prompt batch (knob), a FLASH_ATTN node:"; tab "$O/knob.log"
echo "the decode step, a FLASH_ATTN node:"; tab "$O/decode.log"
ok=1
[ "$(rows "$O/knob.log")" -gt 0 ] && [ "$(rows "$O/decode.log")" -gt 0 ] || { echo "no FLASH_ATTN node in the scheduler's list (is the debug log on?)"; ok=0; }
[ "$(viol "$O/knob.log")" = 0 ] || ok=0
[ "$(viol "$O/decode.log")" = 0 ] || ok=0
if [ "$NGL" = 0 ]; then
  dg=$(grep -c 'DOWNGRADE src=llama op=FLASH_ATTN port=off' "$O/knob.log" || true)
  stats "$O/ctrl.log" > "$O/ctrl.stats"; stats "$O/knob.log" > "$O/knob.stats"
  same=no; [ -s "$O/ctrl.stats" ] && cmp -s "$O/ctrl.stats" "$O/knob.stats" && same=yes
  echo "DOWNGRADE lines in knob: $dg; knob's statistics against base are ctrl's, line for line ($(wc -l < "$O/ctrl.stats") lines): $same"
  echo "ctrl: $(grep -E 'Mean +KLD' "$O/ctrl.stats" | tr -s ' ') | red: $(stats "$O/red.log" | grep -E 'Mean +KLD' | tr -s ' ')"
  [ "$dg" -ge 1 ] || ok=0
  [ $same = yes ] || ok=0
  [ "$(viol "$O/red.log")" -gt 0 ] || { echo "the red did not break the rule"; ok=0; }
fi
[ $ok = 1 ] && echo "guard check: PASS" || echo "guard check: FAIL"
[ $ok = 1 ]
