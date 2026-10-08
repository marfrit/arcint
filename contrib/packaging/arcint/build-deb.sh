#!/bin/bash
# Build arcint_<ver>_amd64.deb — LLM inference engine for Intel Arc.
#
# MUST run on Debian trixie amd64. arcint is a C++20 binary; building it on the
# Arch runner and installing it on trixie is the same cross-distro ABI skew that
# already forced the debian-aarch64 runner to exist (see the va-driver job's
# comment in the CI workflow). There is no debian-amd64 runner yet —
# until there is, this recipe is run on a trixie amd64 host by hand.
#
# Build dependency: marfrit-openvino must be INSTALLED, not merely available.
# It carries the CMake package, the headers and the runtime that arcint links.
# Since 0.5.5 also the second engine's: opencl-headers, ocl-icd-opencl-dev,
# git (applies contrib/llama.cpp/patches) and python3 (embeds the OpenCL
# kernels).
set -euo pipefail

PKGVER=0.7.0
UPSTREAM_TAG=v${PKGVER}
PKGREL=1
# The public repository, not the fleet one. The fleet repo (still named
# "ligence", arcint's working title before the ligence.io collision) is private
# and carries operator-local notes; the published tree is the same code without
# them, so the package is built from what anyone can check.
SRC_URL="https://github.com/marfrit/arcint/archive/refs/tags/${UPSTREAM_TAG}.tar.gz"
# sha256 of https://github.com/marfrit/arcint/archive/refs/tags/v0.7.0.tar.gz,
# taken after the tag was pushed (recorded in the follow-up commit, as for every tag).
ARCINT_TARBALL_SHA256=${ARCINT_TARBALL_SHA256:-cee8d9c51a253b425587c78b4ee27796e88b7cb6fc81a8d8503edfb91c68bcab}
# The libllama engine (--engine llama) builds against llama.cpp at this pin
# with contrib/llama.cpp/patches applied (contrib/llama.cpp/README.md): the
# GitHub tarball of the commit, checked by sha256.
LLAMA_COMMIT=bed0a856606ee4a24a164066f73d2379447033f5
LLAMA_URL="https://github.com/ggml-org/llama.cpp/archive/${LLAMA_COMMIT}.tar.gz"
LLAMA_TARBALL_SHA256=0984123c33b7e959f8003f9169e9109897112d9448b681737813911083eca4cc
OV_PREFIX=/usr/lib/marfrit-openvino
# The ABI is the nightly, not the patch level: floor the patch level, cap at
# the next nightly. An exact pin (Depends: = +p1-1) made apt REMOVE arcint when
# the runtime was upgraded to +p3 on 2026-09-04; never render "=" here again.
# +p25-2: the same +p25 binaries with their RUNPATH set to $ORIGIN (+p25-1 found its
# own libtbb.so.12 and plugins only on the host it was built on; fresh trixie, 2026-10-07).
OV_DEP_VERSION="2026.4.0~dev20260821+p25-2"
# 0.5.5 to 0.6.0 keep the +p25 floor: what they add over 0.5.4 needs no newer
# runtime by default. Patches 0076 (+p26) and 0077 (+p27) serve opt-in switches of the
# Flash-Next CPU tier (MOE_CPU_TIER_ADAPTIVE=1, MOE_DOORBELL=1); those need a
# +p27 runtime built from marfrit-openvino/.
# The +p25 floor is 0.5.4's (patches 0068-0074: the CPU tier's decode and prefill
# kernels, the host expert bank, and patch 0073's QSA selection input to
# PagedAttention, which the served QSA route needs).
# The +p20 floor was 0.5.0.1's (patches 0044-0067: the native expert formats' OpenCL
# decode, the all-resident native pool and its dispatch; an older runtime has no
# native per-expert route at all). +p18 was the floor after patch 0042 (patch 0037's hybrid prefill patch 0037's hybrid prefill
# launched its gather over every token-expert pair while the tables held only the
# resident ones — a page fault on Xe2; +p16 and +p17 carry 0037 without the fix).
# The +p15 floor is 0.4.4's (patch 0033: with four-bit values the verify pass read the
# value rows through the f16 row's alignment, the agent configuration's alternating
# text; DESIGN §7.0.2bu -- every +p14 runtime serves that configuration wrong, so the
# floor is what makes apt pull the fix); +p12 was 0.4.3's (patch 0030), +p11 0.4.1's and
# 0.4.2's (patches 0022-0029), +p7 0.4.0's (patch 0021). Building an older tag with it
# would re-issue a released version string under different Depends. Refused.
case "$PKGVER" in 0.3.*|0.4.0|0.4.1|0.4.2|0.4.3) echo "the +p15 floor is 0.4.4's; bump PKGVER and the tarball sha at the tag" >&2; exit 1 ;; esac
case "$PKGVER" in 0.4.*|0.5.0) echo "the +p20 floor is 0.5.0.1's; bump PKGVER and the tarball sha at the tag" >&2; exit 1 ;; esac
case "$PKGVER" in 0.5.0.*|0.5.1|0.5.2|0.5.3) echo "the +p25 floor is 0.5.4's; bump PKGVER and the tarball sha at the tag" >&2; exit 1 ;; esac
OV_DEP_NEXT_NIGHTLY="2026.4.0~dev20260822"
HERE=$(dirname "$(readlink -f "$0")")

export SOURCE_DATE_EPOCH=1787990400

# Fail in a second, not after the downloads: the tools and the second
# engine's build dependencies.
for t in curl git python3 cmake; do
    command -v "$t" >/dev/null || { echo "$t fehlt (Build-Abhaengigkeit)" >&2; exit 1; }
done
[ -f /usr/include/CL/cl.h ] || { echo "opencl-headers fehlen (/usr/include/CL/cl.h)" >&2; exit 1; }
[ -n "${ARCINT_GIT_SHA:-}" ] || { echo "ARCINT_GIT_SHA fehlt — /props wuerde 'unknown' melden" >&2; exit 1; }

work=$(mktemp -d)
trap "rm -rf $work" EXIT
cd "$work"

# ARCINT_SRC_TARBALL stays as an override for building an unpublished tree; the
# normal path is the anonymous URL above.
if [ -n "${ARCINT_SRC_TARBALL:-}" ]; then
    cp "$ARCINT_SRC_TARBALL" arcint.tar.gz
else
    curl --connect-timeout 10 --max-time 600 --retry 3 --retry-delay 5 \
         -sSLfo arcint.tar.gz "$SRC_URL"
fi
if [ -n "$ARCINT_TARBALL_SHA256" ]; then
    echo "$ARCINT_TARBALL_SHA256  arcint.tar.gz" | sha256sum -c
else
    echo "WARNUNG: keine Pruefsumme gesetzt (ARCINT_TARBALL_SHA256) — $(sha256sum arcint.tar.gz | cut -d' ' -f1)"
fi
tar xzf arcint.tar.gz
SRC=$(find . -maxdepth 1 -mindepth 1 -type d | head -1)
[ -f "$SRC/CMakeLists.txt" ] || { echo "Tarball-Layout unerwartet: kein CMakeLists.txt" >&2; exit 1; }

# The second engine: llama.cpp at the pin, the patches from the arcint tarball
# itself, so the package and its source agree on them by construction.
mkdir llama && cd llama
curl --connect-timeout 10 --max-time 600 --retry 3 --retry-delay 5 -sSLfo llama.tar.gz "$LLAMA_URL"
echo "$LLAMA_TARBALL_SHA256  llama.tar.gz" | sha256sum -c
tar xzf llama.tar.gz
LLAMA_DIR="$work/llama/llama.cpp-${LLAMA_COMMIT}"
cd "$LLAMA_DIR"
for p in "$work/$SRC"/contrib/llama.cpp/patches/*.patch; do
    [ -e "$p" ] || { echo "no contrib/llama.cpp/patches in the source tarball" >&2; exit 1; }
    git apply --whitespace=nowarn "$p" || { echo "patch does not apply to the pin: $p" >&2; exit 1; }
done
cd "$work"

# The build must be able to say which commit it is. The tarball has no .git, so
# the sha is handed in; without it the binary would report "unknown" and every
# /props answer would be unattributable.
GIT_SHA=${ARCINT_GIT_SHA:-}
[ -n "$GIT_SHA" ] || { echo "ARCINT_GIT_SHA fehlt — /props wuerde 'unknown' melden" >&2; exit 1; }

[ -f "$OV_PREFIX/openvino/cmake/OpenVINOConfig.cmake" ] || {
    echo "marfrit-openvino ist nicht installiert ($OV_PREFIX fehlt)" >&2; exit 1; }

cmake -S "$SRC" -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DARCINT_OPENVINO=ON \
    -DARCINT_LLAMA=ON \
    -DARCINT_LLAMA_DIR="$LLAMA_DIR" \
    -DARCINT_WERROR=ON \
    -DARCINT_TESTS=ON \
    -DARCINT_GIT_SHA="$GIT_SHA" \
    -DOpenVINO_DIR="$OV_PREFIX/openvino/cmake" \
    -DARCINT_TOKENIZERS_SO="$OV_PREFIX/openvino_tokenizers/lib/libopenvino_tokenizers.so" \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DCMAKE_INSTALL_RPATH="$OV_PREFIX/openvino/libs" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_SKIP_BUILD_RPATH=OFF
cmake --build build -j"$(nproc)"

# The gates are the product. A package built from a tree whose unit tests fail
# is a package that ships a claim nobody checked. -L unit selects the whole
# device-free set (unit, roundtrip, stress, acceptance-enumeration,
# acceptance-runner) -- not just the test named "unit" -- and --no-tests=error
# turns "ctest found nothing to run" into a build failure instead of a silent
# no-op, which is what happened before the ladder was labelled at all.
( cd build && ctest --output-on-failure --no-tests=error -L unit )

ROOT="$work/pkgroot"
mkdir -p "$ROOT/DEBIAN" "$ROOT/usr/share/doc/arcint" "$ROOT/usr/share/arcint"
DESTDIR="$ROOT" cmake --install build >/dev/null

[ -x "$ROOT/usr/bin/arcint" ] || { echo "FEHLT: /usr/bin/arcint" >&2; exit 1; }
# llama.cpp is a static subproject: none of its install rules may land in the
# package (headers, static libraries, CMake and pkg-config files).
STRAY=$(cd "$ROOT" && find . -path ./usr/include -prune -print -o \( -name '*.a' -o -name '*.pc' -o -path '*/cmake/*' -o -name 'libggml*' -o -name 'libllama*' \) -print)
[ -z "$STRAY" ] || { echo "llama.cpp-Installationsreste im Paket:" >&2; echo "$STRAY" >&2; exit 1; }
# The binary carries the second engine (its OpenCL kernels are embedded).
# No pipe: `strings | grep -q` takes SIGPIPE under pipefail (see the RPATH
# probe below).
grep -qa kernel_mul_mm_kq2d_q4_K_i8 "$ROOT/usr/bin/arcint" || {
    echo "die libllama-Engine fehlt im Binary (ARCINT_LLAMA?)" >&2; exit 1; }

# RPATH-Probe. This is the whole reason the unit file carries no
# LD_LIBRARY_PATH: if the binary cannot find its runtime by itself, the failure
# shows up minutes into a model load on the target host.
# No pipe here reads part of its input and walks away, and that is deliberate.
# The old form was `objdump -x … | awk '…{print $2; exit}'`: awk exits at the
# first match, objdump takes SIGPIPE, and under `set -o pipefail` that kills the
# script with 141 — after a full build, with no message. It survived 0.2.0 only
# because objdump's output still fit in the 64 KiB pipe buffer; a slightly
# larger binary made it deterministic.
RP=$(objdump -x "$ROOT/usr/bin/arcint" | awk '/RUNPATH|RPATH/ && !seen {seen=1; print $2}')
case "$RP" in
    *"$OV_PREFIX/openvino/libs"*) ;;
    *) echo "RPATH zeigt nicht auf $OV_PREFIX/openvino/libs (ist: '$RP')" >&2; exit 1 ;;
esac
# And prove it resolves, here, where it is cheap. Captured once and matched
# twice: `ldd | grep -q` carries the same SIGPIPE hazard as the probe above.
LDD_OUT=$(ldd "$ROOT/usr/bin/arcint")
case "$LDD_OUT" in
    *"not found"*)
        echo "ungeloeste Bibliotheken:" >&2
        printf '%s\n' "$LDD_OUT" | grep "not found" >&2
        exit 1 ;;
esac
case "$LDD_OUT" in
    *"$OV_PREFIX/openvino/libs/libopenvino.so"*) ;;
    *)  echo "libopenvino wird nicht aus $OV_PREFIX geladen" >&2
        printf '%s\n' "$LDD_OUT" >&2
        exit 1 ;;
esac

# The unit TEMPLATE comes from `cmake --install` since 0.2.2, not from a copy
# here: two places installing the same file is how the packaged unit and the
# source unit drift apart. It stays a template — a package must not write into
# a user's ~/.config, and on a managed host that directory belongs to the unit
# manager. Assert it landed, because a silently missing unit turns into a
# puzzled operator on the target host.
[ -f "$ROOT/usr/share/arcint/arcint.service" ] || {
    echo "FEHLT: /usr/share/arcint/arcint.service — installiert CMake die Vorlage nicht mehr?" >&2
    exit 1; }
# The separate env file is gone as of 0.2.2 (flags are literal in ExecStart, so
# a unit manager and a journal can both see the port). If it comes back, it has
# to be installed by CMake too, not by this recipe.
if [ -e "$SRC/packaging/arcint.env" ]; then
    echo "packaging/arcint.env ist zurueck — CMake-Installationsregel pruefen" >&2
    exit 1
fi
for f in README.md llm.txt LICENSE; do
    [ -f "$SRC/$f" ] && install -m 644 "$SRC/$f" "$ROOT/usr/share/doc/arcint/"
done
cp "$HERE/debian/copyright" "$ROOT/usr/share/doc/arcint/copyright"
cp "$HERE/debian/changelog" "$ROOT/usr/share/doc/arcint/changelog.Debian"
gzip -9 -n "$ROOT/usr/share/doc/arcint/changelog.Debian"

# --apparent-size: on ZFS a freshly written tree reports its compressed/unflushed
# allocation (9 KiB for ~3 MiB), and apt shows that as the installed size.
INSTALLED_KB=$(du -sk --apparent-size "$ROOT" | cut -f1)
cat > "$ROOT/DEBIAN/control" <<EOF
Package: arcint
Version: ${PKGVER}-${PKGREL}
Section: misc
Priority: optional
Architecture: amd64
Installed-Size: ${INSTALLED_KB}
Depends: libc6 (>= 2.34), libstdc++6 (>= 13), libgomp1, ocl-icd-libopencl1, marfrit-openvino (>= ${OV_DEP_VERSION}), marfrit-openvino (<< ${OV_DEP_NEXT_NIGHTLY})
Recommends: intel-opencl-icd
Maintainer: Markus Fritsche <mfritsche@reauktion.de>
Homepage: https://github.com/marfrit/arcint
Description: Narrow LLM inference engine for Intel Arc GPUs
 arcint serves exactly three Qwen models on exactly two Intel Arc cards over an
 OpenAI-compatible HTTP surface. It owns its scheduler, paged KV cache, GDN
 ledger, exact prefix cache, speculation and sampling; OpenVINO supplies the
 compiler and kernels. A second engine (--engine llama) runs GGUF models
 through libllama with ggml's OpenCL backend and arcint's Intel kernels.
 .
 LAN use only: no authentication, permissive CORS — the same warning class
 llama-server prints.
 .
 The systemd user unit ships as a template in /usr/share/arcint/; copy it to
 ~/.config/systemd/user/ and edit the flags in ExecStart (on a host with a unit
 manager, hand it the file rather than installing it by hand).
EOF

DEB_OUT=arcint_${PKGVER}-${PKGREL}_amd64.deb
dpkg-deb --root-owner-group --build "$ROOT" "$HERE/$DEB_OUT"
echo "built: $HERE/$DEB_OUT"
