#!/bin/bash
# build-tools.sh -- build llama-perplexity and llama-loan-kld from the patched
# series, without rebuilding the deb.
#
#   build-tools.sh <patched-llama-tree>
#   build-tools.sh --fresh <empty-work-dir> [--arcint <arcint-tree>]
#
# The two binaries are NOT standalone: both resolve their shared libs through the
# RPATH CMake writes into build-tools/bin (libllama, libggml, the OpenCL backend).
# Copy them alone and they will not start; run them from <tree>/build-tools/bin,
# or copy the whole directory's libs with them.
# --fresh needs an EMPTY work dir: it extracts the pin tarball there and builds
# the tree in place; an existing tree or a stale extraction is refused.
#
# The recipe (~/bin/arcint-recipe.sh) builds arcint's libllama executor with
# LLAMA_BUILD_TOOLS OFF (arcint's CMakeLists.txt), so no tool of the series is
# built. Worse, the recipe's llama tree does not survive it: build-deb.sh works
# in `mktemp -d` with `trap "rm -rf $work" EXIT`, so there is no tree to point at
# afterwards. Two ways to get one:
#
#   - a patched tree that already exists on the host (the campaign's card
#     windows build the served series there): give it as the argument;
#   - --fresh assembles one exactly as the recipe does: the pin tarball (the
#     recipe's LLAMA_SRC_TARBALL override if it is set or present, else the
#     pinned URL) checked against build-deb.sh's sha256, the series applied with
#     `git apply` in order, then the build.
#
# The build itself is one cmake line against the tree, with the recipe's own
# OpenCL settings and the kernels embedded from that tree:
#
#   cmake -S <tree> -B <tree>/build-tools -DCMAKE_BUILD_TYPE=Release \
#     -DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
#     -DGGML_OPENCL_EMBED_KERNELS=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
#     -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
#   cmake --build <tree>/build-tools --target llama-perplexity llama-loan-kld -j
#
# It copies the driver to <tree>/tools/loan-kld, registers `add_subdirectory`
# once, prints `TOOL <name>: <sha256>` for each binary, and logs to
# <tree>/build-tools/build.log.
#
#   build-tools.sh --fresh /tmp/loan-tools
#   build-tools.sh /tmp/loan-tools/llama.cpp-3e4dfa8f39f904e681737d9f512a6e6338d848aa
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
ARCINT=$(cd "$HERE/../.." && pwd)
DEB_SH="$ARCINT/contrib/packaging/arcint/build-deb.sh"

if [ "${1:-}" = "--fresh" ]; then
    WORK=${2:?usage: build-tools.sh --fresh <work-dir> [--arcint <arcint-tree>]}
    shift 2
    while [ $# -gt 0 ]; do
        case "$1" in
            --arcint) ARCINT=${2:?}; DEB_SH="$ARCINT/contrib/packaging/arcint/build-deb.sh"; shift 2 ;;
            *) echo "unknown argument: $1" >&2; exit 2 ;;
        esac
    done
    [ -f "$DEB_SH" ] || { echo "no build-deb.sh under $ARCINT" >&2; exit 2; }
    COMMIT=$(sed -n 's/^LLAMA_COMMIT=//p' "$DEB_SH")
    SHA=$(sed -n 's/^LLAMA_TARBALL_SHA256=//p' "$DEB_SH")
    if [ -e "$WORK" ] && [ -n "$(ls -A "$WORK" 2>/dev/null)" ]; then
        echo "--fresh needs an empty work dir: $WORK is not empty" >&2
        exit 2
    fi
    mkdir -p "$WORK"
    TARBALL=${LLAMA_SRC_TARBALL:-$HOME/src/llama-${COMMIT:0:7}.tar.gz}
    if [ ! -f "$TARBALL" ]; then
        TARBALL="$WORK/llama.tar.gz"
        curl --connect-timeout 10 --max-time 600 --retry 3 --retry-delay 5 -sSLfo "$TARBALL" \
            "https://github.com/ggml-org/llama.cpp/archive/${COMMIT}.tar.gz"
    fi
    echo "$SHA  $TARBALL" | sha256sum -c --quiet || { echo "the llama tarball's sha256 differs from build-deb.sh's" >&2; exit 2; }
    tar xzf "$TARBALL" -C "$WORK"
    TREE="$WORK/llama.cpp-$COMMIT"
    PATCHES=("$ARCINT"/contrib/llama.cpp/patches/*.patch)
    [ -e "${PATCHES[0]}" ] || { echo "no contrib/llama.cpp/patches under $ARCINT" >&2; exit 2; }
    ( cd "$TREE" && for p in "${PATCHES[@]}"; do
          git apply --whitespace=nowarn "$p" || { echo "patch does not apply to the pin: $p" >&2; exit 2; }
      done )
else
    TREE=${1:?usage: build-tools.sh <patched-llama-tree>  |  build-tools.sh --fresh <work-dir> [--arcint <arcint-tree>]}
fi

[ -f "$TREE/include/llama.h" ] || { echo "not a llama.cpp tree: $TREE" >&2; exit 2; }
[ -f "$TREE/src/llama-loan-book.h" ] || { echo "$TREE has no patch 0068 (the slot loan)" >&2; exit 2; }

# the driver is arcint's file, built inside the tree: tools/loan-kld, registered
# once. A fresh tree has no registration, so this is the first.
rm -rf "$TREE/tools/loan-kld"
cp -R "$HERE" "$TREE/tools/loan-kld"
if ! grep -qx 'add_subdirectory(loan-kld)' "$TREE/tools/CMakeLists.txt"; then
    printf '\nadd_subdirectory(loan-kld)\n' >> "$TREE/tools/CMakeLists.txt"
fi

B="$TREE/build-tools"
mkdir -p "$B"
{
    cmake -S "$TREE" -B "$B" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_OPENCL=ON \
        -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
        -DGGML_OPENCL_EMBED_KERNELS=ON \
        -DGGML_NATIVE=ON \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF
    cmake --build "$B" --target llama-perplexity llama-loan-kld -j"$(nproc)"
} > "$B/build.log" 2>&1 || { echo "build failed; log: $B/build.log" >&2; tail -20 "$B/build.log" >&2; exit 2; }

for t in llama-perplexity llama-loan-kld; do
    [ -x "$B/bin/$t" ] || { echo "$t was not built; log: $B/build.log" >&2; exit 2; }
    echo "TOOL $t: $(sha256sum "$B/bin/$t" | cut -d' ' -f1)"
done
echo "log: $B/build.log"
