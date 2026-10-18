#!/usr/bin/env bash
# Build stable-diffusion.cpp in upstream mode against arcint's patched ggml tree.
#
# The image-generation reference for 1.1.0 is stable-diffusion.cpp at f0aac52
# (docs/milestone-1.1.0.md, docs/imggen.md). It is built with
#   -DSD_USE_UPSTREAM_GGML=ON -DSD_GGML_SOURCE_DIR=<ggml subtree>
#   -DSD_OPENCL=ON -DSD_VULKAN=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF
# so it uses the upstream ggml API (INT8/SageAttention off, FP8 to F16 at load,
# the generic RoPE path) together with arcint's OpenCL kernels from the series.
#
# sd.cpp defines -DGGML_MAX_NAME=160 for the whole build (its CMakeLists sets it
# before adding the ggml subproject), so sd.cpp's own ggml is compiled at 160
# while arcint's libllama build stays at the pin's 64. The two are separate
# binaries; they must not share one process.
#
# Usage: build-sdcpp.sh <sdcpp-src> <llama-tree> <build-dir> [--server]
#   ARCINT_BUILD (environment) is the build string sd-server's /props reports;
#   SD_VULKAN=OFF leaves the Vulkan backend out (the package: OpenCL only).
#   <sdcpp-src>   stable-diffusion.cpp source tree, checked out at f0aac52
#   <llama-tree>  llama.cpp tree at the pin 3e4dfa8 with contrib/llama.cpp/patches applied
#   <build-dir>   out-of-tree build directory
#
# --server also builds sd-server; it needs the examples/server/frontend submodule
# (or HAVE_INDEX_HTML off) and is the D2(a) artifact.
set -euo pipefail

SD_COMMIT=f0aac524d3f29140e60ba78f3cfb16d6acea87db
LLAMA_PIN=3e4dfa8f39f904e681737d9f512a6e6338d848aa
LLAMA_COMMIT=3e4dfa8

usage() { echo "usage: $0 <sdcpp-src> <llama-tree> <build-dir> [--server]" >&2; exit 2; }
[ $# -ge 3 ] || usage
SRC=$1; LLAMA=$2; BUILD=$3; shift 3
SERVER=0
for a in "$@"; do case "$a" in --server) SERVER=1;; *) usage;; esac; done

[ -d "$SRC" ]  || { echo "no sd.cpp source tree: $SRC" >&2; exit 1; }
[ -d "$LLAMA" ] || { echo "no llama.cpp tree: $LLAMA" >&2; exit 1; }
GGML="$LLAMA/ggml"
[ -f "$GGML/CMakeLists.txt" ] || { echo "no ggml at $GGML (pass the llama.cpp tree)" >&2; exit 1; }

# A git checkout (a worktree's .git is a file, so ask git) is built from a
# pristine export of its HEAD: the patches below never land in the caller's tree,
# and a dirty tree cannot pass for f0aac52. A tree without git (the release
# tarball) is used as it is.
if git -C "$SRC" rev-parse --git-dir >/dev/null 2>&1; then
    got=$(git -C "$SRC" rev-parse HEAD)
    [ "$got" = "$SD_COMMIT" ] || { echo "sd.cpp at $got, expected $SD_COMMIT" >&2; exit 1; }
    mkdir -p "$BUILD"
    rm -rf "$BUILD/sdcpp-src" && mkdir -p "$BUILD/sdcpp-src"
    git -C "$SRC" archive HEAD | tar -x -C "$BUILD/sdcpp-src"
    SRC="$BUILD/sdcpp-src"
fi
if git -C "$LLAMA" rev-parse --git-dir >/dev/null 2>&1; then
    git -C "$LLAMA" merge-base --is-ancestor "$LLAMA_PIN" HEAD \
        || { echo "$LLAMA is not at/after the pin $LLAMA_COMMIT" >&2; exit 1; }
fi

# The arcint series has to be compiled in: the kernels are its own files, which
# upstream ggml does not carry. 0057 (ROPE_APPLY) and 0058 (the q8_0 XMX GEMM)
# are the image service's own patches.
[ -f "$GGML/src/ggml-opencl/kernels/flash_attn_dpas.cl" ] \
    || { echo "arcint's OpenCL kernels are not in $GGML -- apply contrib/llama.cpp/patches first" >&2; exit 1; }
grep -q 'ggml_rope_apply' "$GGML/include/ggml.h" \
    || { echo "$GGML lacks GGML_OP_ROPE_APPLY -- apply 0057" >&2; exit 1; }
grep -q 'kernel_mul_mm_q8_0_f16' "$GGML/src/ggml-opencl/kernels/mul_mm_kq_f16.cl" \
    || { echo "$GGML lacks the q8_0 XMX GEMM -- apply 0058" >&2; exit 1; }
grep -Eq '^#\s*define\s+GGML_MAX_NAME\s+64$' "$GGML/include/ggml.h" \
    || { echo "$GGML/include/ggml.h is not the pin's (GGML_MAX_NAME 64); sd.cpp will set its own 160" >&2; exit 1; }

# arcint's sd.cpp patches, in order (contrib/sdcpp/README.md):
#   upstream-rope-apply.patch       lifts the upstream-mode guard around ggml_rope_apply
#                                   (rope.hpp): since 0057 the op exists in arcint's tree;
#   server-health-props.patch       sd-server's /health and /props (charter A4);
#   conv-direct-probe-opencl.patch  times the direct conv on OpenCL as on Vulkan, so
#                                   --conv-direct auto can pick it (campaign imggen-vae-conv).
# Each is applied unless it is already in (its reverse applies), so a re-run is a no-op.
HERE="$(cd "$(dirname "$0")" && pwd)"
for p in upstream-rope-apply server-health-props conv-direct-probe-opencl; do
    f="$HERE/$p.patch"
    [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
    if (cd "$SRC" && git apply --reverse --check "$f" 2>/dev/null); then
        continue
    fi
    (cd "$SRC" && git apply --whitespace=nowarn "$f") || { echo "$p.patch does not apply to $SRC" >&2; exit 1; }
done

BUILD_STRING=${ARCINT_BUILD:-sd.cpp ${SD_COMMIT:0:7} upstream mode, llama.cpp $LLAMA_COMMIT + arcint series}
cmake -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSD_USE_UPSTREAM_GGML=ON \
    -DSD_GGML_SOURCE_DIR="$GGML" \
    -DSD_OPENCL=ON \
    -DSD_VULKAN="${SD_VULKAN:-ON}" \
    -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
    -DGGML_NATIVE=ON \
    -DARCINT_BUILD="$BUILD_STRING"

# Assert the cache really carries A1's asks, not only the command line.
for k in "SD_USE_UPSTREAM_GGML:BOOL=ON" "GGML_OPENCL_USE_ADRENO_KERNELS:BOOL=OFF" "SD_OPENCL:BOOL=ON"; do
    grep -q "^$k\$" "$BUILD/CMakeCache.txt" || { echo "cache lacks $k" >&2; exit 1; }
done

JOBS=$(nproc)
cmake --build "$BUILD" -j "$JOBS" --target sd-cli
if [ "$SERVER" = 1 ]; then
    cmake --build "$BUILD" -j "$JOBS" --target sd-server
    # /props reports the build string; only server-health-props.patch compiles it
    # in (short literals such as /health's body are immediates, not greppable).
    grep -qaF "$BUILD_STRING" "$BUILD/bin/sd-server" \
        || { echo "$BUILD/bin/sd-server lacks the /props build string (server-health-props.patch)" >&2; exit 1; }
fi
CLI="$BUILD/bin/sd-cli"
[ -x "$CLI" ] || { echo "no $CLI" >&2; exit 1; }

# The arcint kernels and the 160 width are the two facts A1 rests on ('code').
grep -qa kernel_mul_mm_kq2d_q4_K_i8 "$CLI" \
    || { echo "$CLI carries no arcint XMX kernel" >&2; exit 1; }
grep -qa flash_attn_f32_f16_dpas "$CLI" \
    || { echo "$CLI carries no arcint DPAS attention kernel" >&2; exit 1; }
for k in kernel_mul_mm_q8_0_f16 kernel_rope_apply_f32; do
    grep -qa "$k" "$CLI" || { echo "$CLI lacks $k (0057/0058)" >&2; exit 1; }
done
n160=$(grep -o -- '-DGGML_MAX_NAME=160' "$BUILD/build.ninja" | wc -l)
[ "$n160" -gt 0 ] || { echo "GGML_MAX_NAME=160 was not applied to the build" >&2; exit 1; }

echo "sdcpp upstream build ok: $CLI ($(sha256sum "$CLI" | cut -c1-16))"
echo "  maxname-160 compile units: $n160"
echo "  report it as: sd.cpp upstream mode, arcint series, GGML_OPENCL_USE_ADRENO_KERNELS=OFF"
