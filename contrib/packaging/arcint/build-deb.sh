#!/bin/bash
# Build arcint_<ver>_amd64.deb — LLM inference engine for Intel Arc.
#
# MUST run on Debian trixie amd64. arcint is a C++20 binary; building it on the
# Arch runner and installing it on trixie is the same cross-distro ABI skew that
# already forced the debian-aarch64 runner to exist (see the va-driver job's
# comment in the CI workflow). There is no debian-amd64 runner yet —
# until there is, this recipe is run on a trixie amd64 host by hand.
#
# Build dependencies: opencl-headers, ocl-icd-opencl-dev, git (applies
# contrib/llama.cpp/patches), python3 (embeds the OpenCL kernels), cmake, g++.
# Until 0.7.12 the package also linked the OpenVINO engine against an
# installed marfrit-openvino; that engine was retired in 0.8.0
# (docs/history.md), and older tags are built with the recipe at their tag.
set -euo pipefail

PKGVER=1.2.6
UPSTREAM_TAG=v${PKGVER}
PKGREL=1
# The public repository, not the fleet one. The fleet repo (still named
# "ligence", arcint's working title before the ligence.io collision) is private
# and carries operator-local notes; the published tree is the same code without
# them, so the package is built from what anyone can check.
SRC_URL="https://github.com/marfrit/arcint/archive/refs/tags/${UPSTREAM_TAG}.tar.gz"
# sha256 of https://github.com/marfrit/arcint/archive/refs/tags/v1.2.6.tar.gz,
# taken after the tag was pushed (recorded in the follow-up commit, as for every tag).
ARCINT_TARBALL_SHA256=${ARCINT_TARBALL_SHA256:-efc0e4fa87c2e7729add12601b57334713786149ae43e0778edf0cf66a899831}
# The libllama engine (--engine llama) builds against llama.cpp at this pin
# with contrib/llama.cpp/patches applied (contrib/llama.cpp/README.md): the
# GitHub tarball of the commit, checked by sha256.
LLAMA_COMMIT=3e4dfa8f39f904e681737d9f512a6e6338d848aa
LLAMA_URL="https://github.com/ggml-org/llama.cpp/archive/${LLAMA_COMMIT}.tar.gz"
LLAMA_TARBALL_SHA256=f9046074c9b71eaf9abd5491001e4d09c327591fed6b9e2bfcfddbb575dc5255
# 0.7.12 and older link the OpenVINO engine and depend on marfrit-openvino:
# building one of them here would re-issue a released version without that
# Depends. Refused; their recipe is the one at their tag.
case "$PKGVER" in 0.[0-7].*) echo "$PKGVER links OpenVINO: build it with the recipe at its tag" >&2; exit 1 ;; esac
HERE=$(dirname "$(readlink -f "$0")")

export SOURCE_DATE_EPOCH=1787990400

# Fail in a second, not after the downloads: the tools and the engine's
# build dependencies.
for t in curl git python3 cmake objdump; do
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

# The engine: llama.cpp at the pin, the patches from the arcint tarball itself,
# so the package and its source agree on them by construction.
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

cmake -S "$SRC" -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DARCINT_LLAMA=ON \
    -DARCINT_LLAMA_DIR="$LLAMA_DIR" \
    -DARCINT_WERROR=ON \
    -DARCINT_TESTS=ON \
    -DARCINT_GIT_SHA="$GIT_SHA" \
    -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DGGML_NATIVE=ON
# ggml turns GGML_NATIVE off by default when SOURCE_DATE_EPOCH is set (it reads that as a reproducible
# build), which left every package's ggml-cpu at the SSE2 baseline although docs/docker.md says the .deb is
# built native on its host: Flash-Next with the CPU misses decoded 45.6 against 49.4 t/s (measured-here,
# 2026-10-15). Asked for explicitly, and checked.
grep -q '^GGML_NATIVE:BOOL=ON$' build/CMakeCache.txt || { echo "GGML_NATIVE ist nicht ON (ggml-cpu ohne -march=native)" >&2; exit 1; }
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
# The binary carries the engine (its OpenCL kernels are embedded).
# No pipe: `strings | grep -q` takes SIGPIPE under pipefail (see the ldd
# probe below).
grep -qa kernel_mul_mm_kq2d_q4_K_i8 "$ROOT/usr/bin/arcint" || {
    echo "die libllama-Engine fehlt im Binary (ARCINT_LLAMA?)" >&2; exit 1; }
# The result of GGML_NATIVE=ON, not only the ask: this host is the target host, so a ggml-cpu dot product
# without AVX2 here is the SSE-only ggml-cpu of the packages before 0.7.12 (0 ymm registers in 0.7.11's,
# 58 in a native build). A ggml-cpu function, because the CPU miss rows carry AVX2 in any build. Captured
# once (no pipe, see the ldd probe).
DIS=$(objdump -d --no-show-raw-insn --disassemble=ggml_vec_dot_q4_K_q8_K "$ROOT/usr/bin/arcint")
case "$DIS" in
    *"%ymm"*) ;;
    *) echo "ggml_vec_dot_q4_K_q8_K ohne AVX2 (ggml-cpu ohne -march=native)" >&2; exit 1 ;;
esac
unset DIS

# The binary must resolve its libraries by itself (the unit file carries no
# LD_LIBRARY_PATH): prove it here, where it is cheap, not minutes into a model
# load on the target host. Captured once and matched twice: `ldd | grep -q`
# under `set -o pipefail` takes SIGPIPE when grep exits at its first match,
# and the script dies with 141 after a full build, with no message.
LDD_OUT=$(ldd "$ROOT/usr/bin/arcint")
case "$LDD_OUT" in
    *"not found"*)
        echo "ungeloeste Bibliotheken:" >&2
        printf '%s\n' "$LDD_OUT" | grep "not found" >&2
        exit 1 ;;
esac
# The retired engine's runtime must not come back through a stale build cache.
case "$LDD_OUT" in
    *openvino*|*libtbb*)
        echo "das Binary linkt OpenVINO (seit 0.8.0 entfernt)" >&2
        printf '%s\n' "$LDD_OUT" >&2
        exit 1 ;;
esac

# The unit TEMPLATE comes from `cmake --install` since 0.2.2, not from a copy
# here: two places installing the same file is how the packaged unit and the
# source unit drift apart. It stays a template — a package must not write into
# a user's ~/.config, and on a managed host that directory belongs to the unit
# manager. Assert it landed, because a silently missing unit turns into a
# puzzled operator on the target host.
for f in arcint.service arcint-vlm.service arcint-tts.service tts_enrol.py; do
    [ -f "$ROOT/usr/share/arcint/$f" ] || {
        echo "FEHLT: /usr/share/arcint/$f — installiert CMake die Vorlage nicht mehr?" >&2
        exit 1; }
done
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
Depends: libc6 (>= 2.34), libstdc++6 (>= 13), libgomp1, ocl-icd-libopencl1
Recommends: intel-opencl-icd
Maintainer: Markus Fritsche <mfritsche@reauktion.de>
Homepage: https://github.com/marfrit/arcint
Description: LLM inference engine for Intel Arc GPUs
 arcint serves a short list of hybrid Qwen models (and one Mistral Small
 finetune) on the Intel Arc A770 and Arc Pro B60 over an OpenAI-compatible
 HTTP surface. GGUF files run through libllama (llama.cpp at a pinned commit,
 compiled in) with ggml's OpenCL backend and arcint's Intel kernels; arcint
 owns the lanes, MTP speculation, context checkpoints, the expert cache's
 policy and sampling. Needs Intel's compute runtime (intel-opencl-icd) and
 the xe kernel driver.
 .
 LAN use only: no authentication, permissive CORS — the same warning class
 llama-server prints.
 .
 The systemd user unit ships as a template in /usr/share/arcint/; copy it to
 ~/.config/systemd/user/ and edit the flags in ExecStart (on a host with a unit
 manager, hand it the file rather than installing it by hand). Since 1.2.0 the
 same directory holds the vision (arcint-vlm) and speech (arcint-tts) unit
 templates and tts_enrol.py, which enrols a voice on the host: the speech API
 takes enrolled voices only, never a reference recording.
EOF

DEB_OUT=arcint_${PKGVER}-${PKGREL}_amd64.deb
dpkg-deb --root-owner-group --build "$ROOT" "$HERE/$DEB_OUT"
echo "built: $HERE/$DEB_OUT"
