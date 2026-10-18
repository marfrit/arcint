#!/bin/bash
# Build arcint-imggen_<ver>_amd64.deb — text-to-image for Intel Arc: sd.cpp's
# sd-server and sd-cli, built against arcint's patched ggml tree.
#
# The route is the 1.1.0 charter's D2(a) (docs/milestone-1.1.0.md,
# docs/imggen.md): stable-diffusion.cpp at a pinned commit in upstream mode,
# its ggml taken from llama.cpp at arcint's pin with contrib/llama.cpp/patches
# applied, so the image service runs the same Intel kernels as the language
# services (0057 ROPE_APPLY and 0058 q8_0 XMX GEMM were written for it). The
# build itself is contrib/sdcpp/build-sdcpp.sh, which applies arcint's sd.cpp
# patches (contrib/sdcpp/README.md) and asserts the kernels are compiled in.
#
# sd.cpp compiles its ggml with GGML_MAX_NAME=160 while arcint's libllama
# build keeps the pin's 64: the binaries are separate, the package is separate.
#
# MUST run on Debian trixie amd64, like arcint's own recipe (no debian-amd64
# runner yet). Build dependencies: cmake, ninja-build, g++, opencl-headers,
# ocl-icd-opencl-dev, git (git apply), python3 (embeds the OpenCL kernels).
set -euo pipefail

PKGVER=1.1.0
PKGREL=1
UPSTREAM_TAG=v${PKGVER}
# arcint's release tarball carries the llama.cpp patches, the sd.cpp patches
# and the build script; the package is built from what anyone can check. Its
# sha256 is taken after the tag was pushed (the follow-up commit, as for arcint).
SRC_URL="https://github.com/marfrit/arcint/archive/refs/tags/${UPSTREAM_TAG}.tar.gz"
ARCINT_TARBALL_SHA256=${ARCINT_TARBALL_SHA256:-aafe69b20ab185ce550cd29336c01a658d4aea64cf284d19efa9a92c3c3ccb6a}
LLAMA_COMMIT=3e4dfa8f39f904e681737d9f512a6e6338d848aa
LLAMA_URL="https://github.com/ggml-org/llama.cpp/archive/${LLAMA_COMMIT}.tar.gz"
LLAMA_TARBALL_SHA256=f9046074c9b71eaf9abd5491001e4d09c327591fed6b9e2bfcfddbb575dc5255
SD_COMMIT=f0aac524d3f29140e60ba78f3cfb16d6acea87db
SD_URL="https://github.com/leejet/stable-diffusion.cpp/archive/${SD_COMMIT}.tar.gz"
SD_TARBALL_SHA256=37c186d685b4096e4861bbfa594767691d613ac7f5a8cec4d6058086f58eb370
HERE=$(dirname "$(readlink -f "$0")")

export SOURCE_DATE_EPOCH=1787990400

for t in curl git python3 cmake ninja objdump; do
    command -v "$t" >/dev/null || { echo "missing build dependency: $t" >&2; exit 1; }
done
[ -f /usr/include/CL/cl.h ] || { echo "missing opencl-headers (/usr/include/CL/cl.h)" >&2; exit 1; }

work=$(mktemp -d)
trap "rm -rf $work" EXIT
cd "$work"

fetch() { curl --connect-timeout 10 --max-time 600 --retry 3 --retry-delay 5 -sSLfo "$1" "$2"; }

# ARCINT_SRC_TARBALL builds an unpublished tree (the pre-tag check); the normal
# path is the tag's tarball, checked against the sha recorded after the tag.
if [ -n "${ARCINT_SRC_TARBALL:-}" ]; then
    cp "$ARCINT_SRC_TARBALL" arcint.tar.gz
else
    fetch arcint.tar.gz "$SRC_URL"
fi
if [ -n "$ARCINT_TARBALL_SHA256" ]; then
    echo "$ARCINT_TARBALL_SHA256  arcint.tar.gz" | sha256sum -c
else
    echo "WARNING: no ARCINT_TARBALL_SHA256 — $(sha256sum arcint.tar.gz | cut -d' ' -f1)"
fi
mkdir arcint && tar xzf arcint.tar.gz -C arcint --strip-components=1
[ -x arcint/contrib/sdcpp/build-sdcpp.sh ] || [ -f arcint/contrib/sdcpp/build-sdcpp.sh ] \
    || { echo "no contrib/sdcpp/build-sdcpp.sh in the arcint tarball" >&2; exit 1; }

fetch llama.tar.gz "$LLAMA_URL"
echo "$LLAMA_TARBALL_SHA256  llama.tar.gz" | sha256sum -c
mkdir llama && tar xzf llama.tar.gz -C llama --strip-components=1
for p in arcint/contrib/llama.cpp/patches/*.patch; do
    (cd llama && git apply --whitespace=nowarn "$work/$p") || { echo "patch does not apply to the pin: $p" >&2; exit 1; }
done

fetch sd.tar.gz "$SD_URL"
echo "$SD_TARBALL_SHA256  sd.tar.gz" | sha256sum -c
mkdir sdcpp && tar xzf sd.tar.gz -C sdcpp --strip-components=1

NPATCH=$(ls arcint/contrib/llama.cpp/patches/*.patch | wc -l)
LAST=$(basename "$(ls arcint/contrib/llama.cpp/patches/*.patch | tail -1)" | cut -c1-4)
export ARCINT_BUILD="arcint-imggen ${PKGVER}-${PKGREL}: sd.cpp ${SD_COMMIT:0:7} upstream mode, llama.cpp ${LLAMA_COMMIT:0:7} + ${NPATCH} arcint patches (to ${LAST})"
SD_VULKAN=OFF bash arcint/contrib/sdcpp/build-sdcpp.sh "$work/sdcpp" "$work/llama" "$work/build" --server

ROOT="$work/pkgroot"
mkdir -p "$ROOT/DEBIAN" "$ROOT/usr/bin" "$ROOT/usr/share/doc/arcint-imggen" "$ROOT/usr/share/arcint-imggen"
install -m 755 build/bin/sd-server "$ROOT/usr/bin/arcint-sd-server"
install -m 755 build/bin/sd-cli "$ROOT/usr/bin/arcint-sd-cli"
install -m 644 arcint/contrib/systemd/arcint-imggen.service "$ROOT/usr/share/arcint-imggen/"
install -m 644 arcint/tools/imggen_gate.py "$ROOT/usr/share/arcint-imggen/"

# The kernels the series exists for, in the binary that serves (the build
# script checks sd-cli; the package ships sd-server too). No pipes: grep -q
# under pipefail takes SIGPIPE.
for k in kernel_mul_mm_q8_0_f16 rope_apply_f32 flash_attn_f32_f16_dpas; do
    grep -qa "$k" "$ROOT/usr/bin/arcint-sd-server" || { echo "arcint-sd-server lacks $k" >&2; exit 1; }
done
grep -qa "$ARCINT_BUILD" "$ROOT/usr/bin/arcint-sd-server" || { echo "/props build string missing" >&2; exit 1; }
for b in arcint-sd-server arcint-sd-cli; do
    LDD_OUT=$(ldd "$ROOT/usr/bin/$b")
    case "$LDD_OUT" in
        *"not found"*) echo "$b: unresolved libraries" >&2; printf '%s\n' "$LDD_OUT" >&2; exit 1 ;;
        *libvulkan*) echo "$b links Vulkan (the package is OpenCL only)" >&2; exit 1 ;;
    esac
done

for f in LICENSE; do install -m 644 "arcint/$f" "$ROOT/usr/share/doc/arcint-imggen/"; done
install -m 644 sdcpp/LICENSE "$ROOT/usr/share/doc/arcint-imggen/LICENSE.stable-diffusion.cpp"
install -m 644 llama/LICENSE "$ROOT/usr/share/doc/arcint-imggen/LICENSE.llama.cpp"
cp "$HERE/debian/copyright" "$ROOT/usr/share/doc/arcint-imggen/copyright"
cp "$HERE/debian/changelog" "$ROOT/usr/share/doc/arcint-imggen/changelog.Debian"
gzip -9 -n "$ROOT/usr/share/doc/arcint-imggen/changelog.Debian"

INSTALLED_KB=$(du -sk --apparent-size "$ROOT" | cut -f1)
cat > "$ROOT/DEBIAN/control" <<CTL
Package: arcint-imggen
Version: ${PKGVER}-${PKGREL}
Section: graphics
Priority: optional
Architecture: amd64
Installed-Size: ${INSTALLED_KB}
Depends: libc6 (>= 2.34), libstdc++6 (>= 13), libgomp1, ocl-icd-libopencl1
Recommends: intel-opencl-icd
Maintainer: Markus Fritsche <mfritsche@reauktion.de>
Homepage: https://github.com/marfrit/arcint
Description: Text-to-image on Intel Arc GPUs (sd.cpp with arcint's kernels)
 stable-diffusion.cpp's sd-server and sd-cli, built in upstream mode against
 llama.cpp at arcint's pin with arcint's Intel OpenCL kernels (XMX GEMMs,
 the fused RoPE, DPAS attention). Serves an OpenAI-compatible
 /v1/images/generations, plus /health and /props. Measured with Z-Image
 Turbo on the Arc Pro B60. Needs Intel's compute runtime (intel-opencl-icd)
 and the xe kernel driver.
 .
 The binaries are arcint-sd-server and arcint-sd-cli; the systemd user unit
 ships as a template in /usr/share/arcint-imggen/ with the image gate tool.
 .
 LAN use only: no authentication, permissive CORS.
CTL

DEB_OUT=arcint-imggen_${PKGVER}-${PKGREL}_amd64.deb
dpkg-deb --root-owner-group --build "$ROOT" "$HERE/$DEB_OUT"
echo "built: $HERE/$DEB_OUT"
