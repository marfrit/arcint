#!/bin/bash
# Build arcint-stt_<ver>_amd64.deb — speech to text for Intel Arc: whisper.cpp's
# whisper-server built against a ggml installed from arcint's patched llama.cpp
# tree (contrib/whisper/build-whisper.sh), so the STT engine runs the same
# OpenCL backend and the same Intel kernels as the language services
# (docs/milestone-1.2.0.md D3/A3/C3, docs/speech-service.md).
#
# The model is whisper.cpp's own `.bin` container, not a GGUF, and it is staged
# by the operator outside this package (contrib/whisper/README.md).
#
# MUST run on Debian trixie amd64, like arcint's own recipe (no debian-amd64
# runner yet). Build dependencies: cmake, ninja-build, g++, opencl-headers,
# ocl-icd-opencl-dev, git (git apply), python3 (embeds the OpenCL kernels).
set -euo pipefail

PKGVER=1.2.0
PKGREL=1
UPSTREAM_TAG=v${PKGVER}
# The release tarball's sha256 is recorded after the tag was pushed (the
# follow-up commit, as for arcint); before that the pre-tag check builds from
# ARCINT_SRC_TARBALL (a git archive of the staged commit) and this line warns.
SRC_URL="https://github.com/marfrit/arcint/archive/refs/tags/${UPSTREAM_TAG}.tar.gz"
ARCINT_TARBALL_SHA256=${ARCINT_TARBALL_SHA256:-}
LLAMA_COMMIT=3e4dfa8f39f904e681737d9f512a6e6338d848aa
LLAMA_URL="https://github.com/ggml-org/llama.cpp/archive/${LLAMA_COMMIT}.tar.gz"
LLAMA_TARBALL_SHA256=f9046074c9b71eaf9abd5491001e4d09c327591fed6b9e2bfcfddbb575dc5255
WHISPER_COMMIT=d1be6fde11ac6e0407606b4e42fe72d34add8037   # v1.9.5
WHISPER_URL="https://github.com/ggml-org/whisper.cpp/archive/${WHISPER_COMMIT}.tar.gz"
WHISPER_TARBALL_SHA256=645445da884df6dbd52b30f82d7be7c01d1767c938714e477e786b88d529e9f4
HERE=$(dirname "$(readlink -f "$0")")

export SOURCE_DATE_EPOCH=1787990400

for t in curl git python3 cmake ninja objdump strings; do
    command -v "$t" >/dev/null || { echo "missing build dependency: $t" >&2; exit 1; }
done
[ -f /usr/include/CL/cl.h ] || { echo "missing opencl-headers (/usr/include/CL/cl.h)" >&2; exit 1; }

work=$(mktemp -d)
trap "rm -rf $work" EXIT
cd "$work"

fetch() { curl --connect-timeout 10 --max-time 600 --retry 3 --retry-delay 5 -sSLfo "$1" "$2"; }

if [ -n "${ARCINT_SRC_TARBALL:-}" ]; then
    cp "$ARCINT_SRC_TARBALL" arcint.tar.gz
else
    fetch arcint.tar.gz "$SRC_URL"
fi
if [ -n "$ARCINT_TARBALL_SHA256" ]; then
    echo "$ARCINT_TARBALL_SHA256  arcint.tar.gz" | sha256sum -c
else
    echo "WARNING: no ARCINT_TARBALL_SHA256 (pre-tag build) — $(sha256sum arcint.tar.gz | cut -d' ' -f1)"
fi
mkdir arcint && tar xzf arcint.tar.gz -C arcint --strip-components=1
[ -x arcint/contrib/whisper/build-whisper.sh ] || [ -f arcint/contrib/whisper/build-whisper.sh ] \
    || { echo "no contrib/whisper/build-whisper.sh in the arcint tarball" >&2; exit 1; }

fetch llama.tar.gz "$LLAMA_URL"
echo "$LLAMA_TARBALL_SHA256  llama.tar.gz" | sha256sum -c
mkdir llama && tar xzf llama.tar.gz -C llama --strip-components=1
for p in arcint/contrib/llama.cpp/patches/*.patch; do
    (cd llama && git apply --whitespace=nowarn "$work/$p") || { echo "patch does not apply to the pin: $p" >&2; exit 1; }
done

fetch whisper.tar.gz "$WHISPER_URL"
echo "$WHISPER_TARBALL_SHA256  whisper.tar.gz" | sha256sum -c
mkdir whisper && tar xzf whisper.tar.gz -C whisper --strip-components=1

NPATCH=$(ls arcint/contrib/llama.cpp/patches/*.patch | wc -l)
LAST=$(basename "$(ls arcint/contrib/llama.cpp/patches/*.patch | tail -1)" | cut -c1-4)
export ARCINT_BUILD="arcint-stt ${PKGVER}-${PKGREL}: whisper.cpp ${WHISPER_COMMIT:0:7} on llama.cpp ${LLAMA_COMMIT:0:7} + ${NPATCH} arcint patches (to ${LAST}), OpenCL"
bash arcint/contrib/whisper/build-whisper.sh "$work/whisper" "$work/llama" "$work/build"

ROOT="$work/pkgroot"
mkdir -p "$ROOT/DEBIAN" "$ROOT/usr/bin" "$ROOT/usr/share/doc/arcint-stt" "$ROOT/usr/share/arcint-stt"
install -m 755 build/whisper/bin/whisper-server "$ROOT/usr/bin/arcint-whisper"
[ -x build/whisper/bin/whisper-cli ] && install -m 755 build/whisper/bin/whisper-cli "$ROOT/usr/bin/arcint-whisper-cli"
install -m 644 arcint/contrib/systemd/arcint-stt.service "$ROOT/usr/share/arcint-stt/"
install -m 644 arcint/tools/stt_gate.py "$ROOT/usr/share/arcint-stt/"

# The binary must carry the Intel kernels and the OpenCL backend string of the
# linked ggml (a vendored whisper.cpp ggml would not; contrib/whisper/README.md)
for s in kernel_mul_mv_kq_q8_1 "ggml_opencl:"; do
    strings -a "$ROOT/usr/bin/arcint-whisper" | grep -q "$s" || { echo "arcint-whisper lacks '$s'" >&2; exit 1; }
done
LDD_OUT=$(ldd "$ROOT/usr/bin/arcint-whisper")
case "$LDD_OUT" in
    *"not found"*) echo "arcint-whisper: unresolved libraries" >&2; printf '%s\n' "$LDD_OUT" >&2; exit 1 ;;
    *libvulkan*) echo "arcint-whisper links Vulkan (the package is OpenCL only)" >&2; exit 1 ;;
esac

for f in LICENSE; do install -m 644 "arcint/$f" "$ROOT/usr/share/doc/arcint-stt/"; done
install -m 644 whisper/LICENSE "$ROOT/usr/share/doc/arcint-stt/LICENSE.whisper.cpp"
install -m 644 llama/LICENSE "$ROOT/usr/share/doc/arcint-stt/LICENSE.llama.cpp"
cp "$HERE/debian/copyright" "$ROOT/usr/share/doc/arcint-stt/copyright"
cp "$HERE/debian/changelog" "$ROOT/usr/share/doc/arcint-stt/changelog.Debian"
gzip -9 -n "$ROOT/usr/share/doc/arcint-stt/changelog.Debian"

INSTALLED_KB=$(du -sk --apparent-size "$ROOT" | cut -f1)
cat > "$ROOT/DEBIAN/control" <<CTL
Package: arcint-stt
Version: ${PKGVER}-${PKGREL}
Section: sound
Priority: optional
Architecture: amd64
Installed-Size: ${INSTALLED_KB}
Depends: libc6 (>= 2.34), libstdc++6 (>= 13), libgomp1, ocl-icd-libopencl1
Recommends: intel-opencl-icd
Maintainer: Markus Fritsche <mfritsche@reauktion.de>
Homepage: https://github.com/marfrit/arcint
Description: Speech to text on Intel Arc GPUs (whisper.cpp on arcint's ggml)
 whisper.cpp's whisper-server built against a ggml installed from llama.cpp
 at arcint's pin with arcint's Intel OpenCL kernels, so the encoder and
 decoder run on the same backend as the language services. Serves the OpenAI
 shape at /v1/audio/transcriptions through the server's own
 --inference-path flag, plus /health. The whisper model (.bin, not GGUF) is
 staged by the operator outside this package; the systemd user unit ships as
 a template in /usr/share/arcint-stt/ with the STT gate tool.
 .
 The binary is arcint-whisper; German first.
 .
 LAN use only: no authentication, permissive CORS.
CTL

DEB_OUT=arcint-stt_${PKGVER}-${PKGREL}_amd64.deb
dpkg-deb --root-owner-group --build "$ROOT" "$HERE/$DEB_OUT"
echo "built: $HERE/$DEB_OUT"
