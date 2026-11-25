#!/usr/bin/env bash
# guard_sweep.sh <dir with llama-perplexity> <Inkling 2 Mini GGUF> <text file> <out dir> <ngl> [extra llama args]
#
# THE RULE of guard_check.sh, swept over batch shapes on a card host: with
# GGML_OPENCL_FA_PORT=1, no FLASH_ATTN node that the scheduler ran on the CPU may carry
# the port's sources. A case is one llama-perplexity run (-c <tokens> -b <ub> -ub <ub>
# --chunks 1) with GGML_SCHED_DEBUG=2 whose last micro-batch has a given number of rows:
#   ub 512:  tails 1 2 8 9 12 16 17 20 24 31 33 63 64 65 100 127 128 129 255 256 257 511 512
#   ub 1024: tails 9 64 65 1023        ub 2048: tails 9 64 65 1023 2047
#   each alone (the prompt is the tail: -c <tail> -b <tail> -ub <tail>, because
#   llama-perplexity packs n_batch / n_ctx sequences into a batch and a -b of the full ub
#   would make the "tail" a batch of many short sequences) and after one full micro-batch
#   of earlier context (-c <ub + tail> -b <ub> -ub <ub>: a batch of ub rows, then the tail);
#   and the drafter-style micro-batches 2..8 over a 32-token prompt.
# Per case it prints: the n_ubatch, the context and the batch size llama-perplexity reports
# and the row counts the graph builder saw (GGML_OPENCL_FA_PORT_TRACE: the proof of the
# shape), the FLASH_ATTN nodes in the scheduler's list, those that ran on the CPU WITH the
# port's sources (a hit: must be 0), and those that ran off the CPU with them (the port in
# use). The last line counts the cases that RAN. It exits non-zero on any hit, on a case
# that did not show its tail among the row counts, and at once on a case that did not run -
# except a case named in EXPECTED_NOT_RUN below, which is listed and not counted as run
# (the list is empty: every case of the dry run ran).
# GUARD_SWEEP_CASES=<regex> runs only the cases whose name matches (a dry run). The red of
# the sweep itself: with ngl 0 and GGML_OPENCL_FA_PORT_RED=5 in the environment (the model
# without its guard) every case that runs must be a hit.
# The text file must hold at least 8,200 tokens for the whole sweep. A model-sized job a
# case: run it under the host's lock for such jobs, if it has one.
set -uo pipefail
BIN=${1:?dir with llama-perplexity}; M=${2:?model}; TXT=${3:?text}; O=${4:?out dir}; NGL=${5:?ngl}; shift 5
EXTRA=("$@")
ONLY=${GUARD_SWEEP_CASES:-.}
EXPECTED_NOT_RUN=" "   # none: with -b <tail> the 1- and 2-token contexts run too (the dry run); name a case here only with its reason
mkdir -p "$O"
count() { awk -v want="$2" '
    /node #[ 0-9]+ \(FLASH_ATTN/ {
        b = $0; sub(/^[^\[]*\[ */, "", b); sub(/ .*/, "", b)
        s = ($0 ~ /attn_inp_port_pos/)
        n++; if (b == "CPU" && s) cpu++; if (b != "CPU" && s) dev++
    }
    END { if (want == "n") print n + 0; else if (want == "cpu") print cpu + 0; else print dev + 0 }' "$1"; }
bad=""; n_case=0; n_ran=0; notrun=""
one() {  # name, n_ctx, n_batch (= n_ubatch), the tail the case is about
    local name=$1 c=$2 b=$3 tail=$4 log="$O/$1.log"
    echo "$name" | grep -qE "$ONLY" || return 0
    n_case=$((n_case + 1))
    GGML_OPENCL_FA_PORT=1 GGML_OPENCL_FA_PORT_TRACE=1 GGML_SCHED_DEBUG=2 "$BIN/llama-perplexity" -m "$M" -f "$TXT" -ngl "$NGL" -fa on \
        -c "$c" -b "$b" -ub "$b" --chunks 1 -v ${EXTRA[@]+"${EXTRA[@]}"} > "$log" 2>&1
    local rc=$? n cpu dev ub used rows
    n=$(count "$log" n); cpu=$(count "$log" cpu); dev=$(count "$log" dev)
    ub=$(grep -oE "n_ubatch += +[0-9]+" "$log" | head -1 | grep -oE "[0-9]+$" || true)
    used=$(grep -oE "n_ctx=[0-9]+, batch_size=[0-9]+" "$log" | head -1 || true)
    rows=$(grep -oE "FA_PORT_TRACE layer=[0-9]+ rows=[0-9]+" "$log" | grep -oE "rows=[0-9]+" | sort -t= -k2 -n -u | sed 's/rows=//' | tr '\n' ' ')
    if [ $rc -ne 0 ] || [ "$n" = 0 ] || [ -z "$used" ]; then
        local why; why=$( { grep -E "^[0-9.]+ E " "$log"; grep -iE "invalid argument|you need at least|failed to" "$log"; } | head -1 | sed 's/^[0-9.]* [A-Z] //' | cut -c1-110)
        if echo "$EXPECTED_NOT_RUN" | grep -q " $name "; then
            echo "$name: NOT RUN, expected (rc $rc): $why"; notrun="$notrun $name"
            return 0
        fi
        echo "$name: NOT RUN (rc $rc, FLASH_ATTN nodes $n): $why"
        echo "guard sweep: FAIL: $name did not run (not an expected one); stopped after $n_case cases, $n_ran ran"
        exit 1
    fi
    n_ran=$((n_ran + 1))
    local shape=ok
    echo " $rows" | grep -q " $tail " || shape="TAIL-NOT-SEEN"
    echo "$name: n_ubatch ${ub:-?}, used ${used}, rows seen: ${rows}| FLASH_ATTN nodes $n, HITS (on the CPU with the port's sources) $cpu, off the CPU with them $dev [$shape]"
    [ "$cpu" = 0 ] || bad="$bad $name($cpu)"
    [ "$shape" = ok ] || bad="$bad $name(tail-not-seen)"
}
sweep() {  # ub, tails...
    local ub=$1; shift
    for t in "$@"; do
        one "ub${ub}-tail${t}-alone" "$t" "$t" "$t"
        one "ub${ub}-tail${t}-after-context" "$((ub + t))" "$ub" "$t"
    done
}
sweep 512 1 2 8 9 12 16 17 20 24 31 33 63 64 65 100 127 128 129 255 256 257 511 512
sweep 1024 9 64 65 1023
sweep 2048 9 64 65 1023 2047
for b in 2 3 4 5 6 7 8; do one "drafter-ub${b}" 32 "$b" "$b"; done
if [ -n "$bad" ]; then echo "guard sweep: FAIL: $n_ran of $n_case ran; hits or shape misses:$bad; NOT RUN (expected):${notrun:- none}"; exit 1; fi
echo "guard sweep: PASS: $n_ran of $n_case ran, 0 hits; NOT RUN (expected):${notrun:- none}"
