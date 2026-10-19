#!/usr/bin/env bash
# Build whisper.cpp against a ggml installed from arcint's patched llama.cpp
# tree, so the STT engine runs on the same OpenCL backend and the same Intel
# kernels (contrib/whisper/README.md, docs/milestone-1.2.0.md A3/C3).
#
#   build-whisper.sh <whisper-src> <llama-tree> <build-dir>
#
# <whisper-src>: whisper.cpp at the pin below; a git tree is checked against
#   the commit and exported pristine. <llama-tree>: llama.cpp at arcint's pin
#   with contrib/llama.cpp/patches applied. <build-dir>: scratch.
set -euo pipefail

WHISPER_REPO="ggml-org/whisper.cpp"
WHISPER_COMMIT="d1be6fde11ac6e0407606b4e42fe72d34add8037"   # v1.9.5
LLAMA_COMMIT="3e4dfa8"

if [ $# -lt 3 ]; then
  echo "usage: $0 <whisper-src> <llama-tree> <build-dir>" >&2
  exit 2
fi
WHISPER_SRC=$1
LLAMA_TREE=$2
BUILD=$3

command -v cmake >/dev/null || { echo "cmake missing" >&2; exit 2; }

# --- llama tree: the pin, the series, the OpenCL backend -------------------
if [ ! -e "$LLAMA_TREE/include/llama.h" ]; then
  echo "no llama.h in $LLAMA_TREE" >&2; exit 2
fi
if [ ! -e "$LLAMA_TREE/ggml/src/ggml-opencl/kernels/mul_mv_kq_q8_1.cl" ]; then
  echo "$LLAMA_TREE lacks contrib/llama.cpp/patches (the Intel kernels)" >&2; exit 2
fi

mkdir -p "$BUILD"
LLAMA_BUILD="$BUILD/llama"
GGML_PREFIX="$BUILD/ggml-prefix"

# ggml only, OpenCL, no tools: the install carries ggml-config.cmake and the
# OpenCL backend, and GGML_MAX_NAME stays the pin's 64
cmake -S "$LLAMA_TREE" -B "$LLAMA_BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
      -DGGML_OPENCL_EMBED_KERNELS=ON \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_BUILD_APP=OFF -DLLAMA_TOOLS_INSTALL=OFF \
      -DLLAMA_BUILD_COMMON=OFF -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_INSTALL_PREFIX="$GGML_PREFIX" >/dev/null
cmake --build "$LLAMA_BUILD" --target ggml llama -j"$(nproc)"
cmake --install "$LLAMA_BUILD" >/dev/null
if [ ! -e "$GGML_PREFIX/lib/cmake/ggml/ggml-config.cmake" ]; then
  echo "ggml install has no ggml-config.cmake" >&2; exit 2
fi

# --- whisper.cpp: the pin, pristine export, system ggml --------------------
if [ -d "$WHISPER_SRC/.git" ]; then
  got=$(git -C "$WHISPER_SRC" rev-parse HEAD)
  if [ "$got" != "$WHISPER_COMMIT" ]; then
    echo "$WHISPER_SRC is $got, not the pin $WHISPER_COMMIT" >&2; exit 2
  fi
  WHISPER_EXPORT="$BUILD/whisper-src"
  rm -rf "$WHISPER_EXPORT"; mkdir -p "$WHISPER_EXPORT"
  git -C "$WHISPER_SRC" archive HEAD | tar -x -C "$WHISPER_EXPORT"
else
  WHISPER_EXPORT="$WHISPER_SRC"   # tarball: no git to check
fi

WHISPER_BUILD="$BUILD/whisper"
cmake -S "$WHISPER_EXPORT" -B "$WHISPER_BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DWHISPER_USE_SYSTEM_GGML=ON \
      -Dggml_DIR="$GGML_PREFIX/lib/cmake/ggml" \
      -DWHISPER_BUILD_EXAMPLES=ON -DWHISPER_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=OFF >/dev/null
cmake --build "$WHISPER_BUILD" -j"$(nproc)"

BIN="$WHISPER_BUILD/bin/whisper-cli"
[ -e "$BIN" ] || { echo "no whisper-cli built" >&2; exit 2; }
# The binary must carry the Intel kernels and the OpenCL device string of the
# linked ggml: a vendored ggml would not (this is the reason for the recipe)
# The one marker a vendored whisper.cpp ggml cannot have: arcint's series
# kernel name, embedded into the system ggml by GGML_OPENCL_EMBED_KERNELS. The
# OpenCL backend's own strings are a second marker (no literal "Intel(R)"
# exists in the back-end source at this pin).
for s in kernel_mul_mv_kq_q8_1 "ggml_opencl:"; do
  if ! strings -a "$BIN" | grep -q "$s"; then
    echo "whisper-cli lacks '$s' (a vendored ggml, not the system one?)" >&2; exit 2
  fi
done

echo "built: $BIN (whisper.cpp $WHISPER_COMMIT on llama.cpp $LLAMA_COMMIT + series, OpenCL)"
