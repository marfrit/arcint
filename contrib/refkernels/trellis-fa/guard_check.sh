#!/usr/bin/env bash
# guard_check.sh <dir with llama-perplexity> <Inkling 2 Mini GGUF> <text file> <out dir> [ngl]
#
# The layer guard of the port arm on the REAL graph builder (fa_port_gate.h:
# fa_port_layer_attaches; src/models/inkling2.cpp). Four runs of llama-perplexity on the
# same text, batches of 64 rows (a prompt batch: the port's shape):
#   base   the knob unset; its logits are saved
#   ctrl   the knob unset again, compared with base: what "the same run" reads as (the saved
#          logits are quantised, so the KL of a run against itself is not exactly 0)
#   knob   GGML_OPENCL_FA_PORT=1, traced: which layers got the port's sources, compared with base
#   red    the same with GGML_OPENCL_FA_PORT_RED=5 (the form before the guard: every layer
#          gets the sources), compared with base
# With ngl 0 (the default; every layer on the CPU, a backend that does not declare the
# port's sources) the check passes when: knob attaches the sources on NO prompt-batch layer
# and prints the DOWNGRADE line; knob's statistics against base are, line for line, ctrl's
# (the run with the knob is the run without it); and red attaches them (the guard is what
# keeps them off).
# With ngl > 0 on a host with a card (a mixed run: some layers on OpenCL, the rest on the
# CPU) it prints the per-layer table and passes when no layer on the CPU carries the port's
# sources; the layers on the card must show sources=attached.
# A model-sized job: run it under the host's lock for such jobs, if it has one.
set -uo pipefail
BIN=${1:?dir with llama-perplexity}; M=${2:?model}; TXT=${3:?text}; O=${4:?out dir}; NGL=${5:-0}
mkdir -p "$O"
run() { "$BIN/llama-perplexity" -m "$M" -f "$TXT" -c 256 -b 64 --chunks 2 -ngl "$NGL" -fa on "$@"; }
run --kl-divergence-base "$O/base.kld" > "$O/base.log" 2>&1 || { echo "FAIL: base run"; exit 2; }
run --kl-divergence-base "$O/base.kld" --kl-divergence > "$O/ctrl.log" 2>&1 || { echo "FAIL: ctrl run"; exit 2; }
GGML_OPENCL_FA_PORT=1 GGML_OPENCL_FA_PORT_TRACE=1 run --kl-divergence-base "$O/base.kld" --kl-divergence > "$O/knob.log" 2>&1 || { echo "FAIL: knob run"; exit 2; }
GGML_OPENCL_FA_PORT=1 GGML_OPENCL_FA_PORT_TRACE=1 GGML_OPENCL_FA_PORT_RED=5 run --kl-divergence-base "$O/base.kld" --kl-divergence > "$O/red.log" 2>&1 || { echo "FAIL: red run"; exit 2; }
# a prompt-batch layer's decisions: layer, device, sources (rows > 8), one line a distinct triple
tab() { grep "FA_PORT_TRACE" "$1" | sed 's/.*FA_PORT_TRACE //' | awk '{r=$2; sub("rows=","",r); if (r+0 > 8) print $1, $3, $4}' | sort -t= -k2 -n | uniq; }
cnt() { tab "$1" | grep -c "$2" || true; }
stats() { sed -n '/Perplexity statistics/,$p' "$1" | sed 's/^[0-9.]* [A-Z] //' | grep -vE "perf|time|ms" ; }
kl()  { grep -E "Mean +KLD" "$1" | head -1 | tr -s ' ' | sed 's/^[0-9.]* [A-Z] //'; }
top() { grep -E "Same top p:" "$1" | head -1 | tr -s ' ' | sed 's/^[0-9.]* [A-Z] //'; }
dg=$(grep -c 'DOWNGRADE src=llama op=FLASH_ATTN port=off' "$O/knob.log" || true)
echo "ctrl: $(kl "$O/ctrl.log"); $(top "$O/ctrl.log")"
echo "knob: prompt-batch layers with the port's sources $(cnt "$O/knob.log" sources=attached), without $(cnt "$O/knob.log" sources=none); DOWNGRADE lines $dg; $(kl "$O/knob.log"); $(top "$O/knob.log")"
echo "red : prompt-batch layers with the port's sources $(cnt "$O/red.log" sources=attached), without $(cnt "$O/red.log" sources=none); $(kl "$O/red.log"); $(top "$O/red.log")"
stats "$O/ctrl.log" > "$O/ctrl.stats"; stats "$O/knob.log" > "$O/knob.stats"
same=no; [ -s "$O/ctrl.stats" ] && cmp -s "$O/ctrl.stats" "$O/knob.stats" && same=yes
echo "knob's statistics against base are ctrl's, line for line ($(wc -l < "$O/ctrl.stats") lines): $same"
echo "the knob run, a prompt-batch layer:"; tab "$O/knob.log" | sed 's/^/  /'
if [ "$NGL" = 0 ]; then
  ok=1
  [ "$(cnt "$O/knob.log" sources=attached)" = 0 ] || ok=0
  [ "$(cnt "$O/knob.log" sources=none)" -gt 0 ] || ok=0
  [ "$dg" -ge 1 ] || ok=0
  [ $same = yes ] || ok=0
  [ "$(cnt "$O/red.log" sources=attached)" -gt 0 ] || ok=0
  [ $ok = 1 ] && echo "guard check: PASS" || echo "guard check: FAIL"
  [ $ok = 1 ]
else
  bad=$(tab "$O/knob.log" | awk '{dev=$2; sub("device=","",dev); if ((dev ~ /CPU|none/) && $3 == "sources=attached") print}' | wc -l)
  echo "mixed: layers on the CPU with the port's sources: $bad (must be 0); layers with them: $(cnt "$O/knob.log" sources=attached)"
  [ "$bad" = 0 ] && echo "guard check (mixed): PASS" || echo "guard check (mixed): FAIL"
  [ "$bad" = 0 ]
fi
