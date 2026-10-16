#!/usr/bin/env bash
# The DFlash2 red-case binaries for `legs.sh red`: one build a mutant
# (ARCINT_DFLASH_RED=N: src/exec/llama_dflash.cpp, and case 7 in the verify
# loop of src/exec/backend_llama.cpp), each copied to <out-dir>/arcint-red-N.
# One build tree for all: a case recompiles two files.
#   build-red.sh <arcint-source> <llama.cpp-tree> <out-dir> [cases, default "1 2 3 4 5 6 7"]
# The llama.cpp tree is the one the served binary builds against (the pin
# with contrib/llama.cpp's patches); the build tree is <out-dir>/build.
set -euo pipefail
[ $# -ge 3 ] || { sed -n '2,9p' "$0"; exit 2; }
src=$1
llama=$2
out=$3
cases=${4:-1 2 3 4 5 6 7}
mkdir -p "$out"
for n in $cases; do
  cmake -S "$src" -B "$out/build" -DARCINT_LLAMA=ON -DARCINT_LLAMA_DIR="$llama" -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
        -DARCINT_TESTS=OFF -DARCINT_DFLASH_RED="$n" > "$out/configure-$n.log" 2>&1
  cmake --build "$out/build" --target arcint -j"$(nproc)" > "$out/build-$n.log" 2>&1
  cp "$out/build/arcint" "$out/arcint-red-$n"
  # the mutant is in the artefact: its banner is compiled in
  grep -q "THIS BUILD IS THE DFLASH RED CASE" "$out/arcint-red-$n" || { echo "red $n: no banner in the binary" >&2; exit 1; }
  echo "red $n: $out/arcint-red-$n sha256 $(sha256sum "$out/arcint-red-$n" | cut -c1-16)"
done
