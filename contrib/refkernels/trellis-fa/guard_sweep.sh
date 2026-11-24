#!/usr/bin/env bash
# guard_sweep.sh <dir with llama-perplexity> <Inkling 2 Mini GGUF> <text file> <out dir> <ngl> [extra llama args]
#
# THE RULE of guard_check.sh, swept over batch shapes on a card host: with
# GGML_OPENCL_FA_PORT=1, no FLASH_ATTN node that the scheduler ran on the CPU may carry
# the port's sources. A case is one llama-perplexity run with GGML_SCHED_DEBUG=2 whose
# last micro-batch has a given number of rows:
#   ub 512:  tails 1 2 8 9 12 16 17 20 24 31 33 63 64 65 100 127 128 129 255 256 257 511 512
#   ub 1024: tails 9 64 65 1023        ub 2048: tails 9 64 65 1023 2047
#   each alone (the prompt is the tail) and after one full micro-batch of earlier context;
#   and the drafter-style micro-batches 2..8 over a 32-token prompt.
# Per case it prints: FLASH_ATTN nodes in the scheduler's list, those that ran on the CPU
# WITH the port's sources (must be 0), and those that ran off the CPU with them (the port in
# use). It exits non-zero if any case is above 0 or did not run, and lists them.
# The text file must hold at least 8,200 tokens. A model-sized job a case: run it under
# the host's lock for such jobs, if it has one.
set -uo pipefail
BIN=${1:?dir with llama-perplexity}; M=${2:?model}; TXT=${3:?text}; O=${4:?out dir}; NGL=${5:?ngl}; shift 5
mkdir -p "$O"
count() { awk -v want="$2" '
    /node #[ 0-9]+ \(FLASH_ATTN/ {
        b = $0; sub(/^[^\[]*\[ */, "", b); sub(/ .*/, "", b)
        s = ($0 ~ /attn_inp_port_pos/)
        n++; if (b == "CPU" && s) cpu++; if (b != "CPU" && s) dev++
    }
    END { if (want == "n") print n + 0; else if (want == "cpu") print cpu + 0; else print dev + 0 }' "$1"; }
bad=""; n_case=0
one() {  # name, n_ctx, n_batch
    local name=$1 c=$2 b=$3 log="$O/$1.log"
    n_case=$((n_case + 1))
    GGML_OPENCL_FA_PORT=1 GGML_SCHED_DEBUG=2 "$BIN/llama-perplexity" -m "$M" -f "$TXT" -ngl "$NGL" -fa on -c "$c" -b "$b" --chunks 1 -v "$@" > "$log" 2>&1
    local rc=$? n cpu dev
    n=$(count "$log" n); cpu=$(count "$log" cpu); dev=$(count "$log" dev)
    if [ $rc -ne 0 ] || [ "$n" = 0 ]; then
        echo "$name: NOT RUN (rc $rc, FLASH_ATTN nodes $n)"; bad="$bad $name(not-run)"
    else
        echo "$name: FLASH_ATTN nodes $n, on the CPU with the port's sources $cpu, off the CPU with them $dev"
        [ "$cpu" = 0 ] || bad="$bad $name($cpu)"
    fi
}
sweep() {  # ub, tails...
    local ub=$1; shift
    for t in "$@"; do
        one "ub${ub}-tail${t}-alone" "$t" "$ub"
        one "ub${ub}-tail${t}-after-context" "$((ub + t))" "$ub"
    done
}
sweep 512 1 2 8 9 12 16 17 20 24 31 33 63 64 65 100 127 128 129 255 256 257 511 512
sweep 1024 9 64 65 1023
sweep 2048 9 64 65 1023 2047
for b in 2 3 4 5 6 7 8; do one "drafter-ub${b}" 32 "$b"; done
echo "cases: $n_case"
if [ -n "$bad" ]; then echo "guard sweep: FAIL:$bad"; exit 1; fi
echo "guard sweep: PASS (0 nodes on the CPU with the port's sources in every case)"
