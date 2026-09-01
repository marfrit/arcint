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
set -euo pipefail

PKGVER=0.2.12
UPSTREAM_TAG=v${PKGVER}
PKGREL=1
# The public repository, not the fleet one. The fleet repo (still named
# "ligence", arcint's working title before the ligence.io collision) is private
# and carries operator-local notes; the published tree is the same code without
# them, so the package is built from what anyone can check.
SRC_URL="https://github.com/marfrit/arcint/archive/refs/tags/${UPSTREAM_TAG}.tar.gz"
ARCINT_TARBALL_SHA256=${ARCINT_TARBALL_SHA256:-681b1edb610ac91635c4abf173c3c2f6e6e3423f4d7336e91a0a3484f4e78a59}
OV_PREFIX=/usr/lib/marfrit-openvino
OV_DEP_VERSION="2026.4.0~dev20260821+p1-1"
HERE=$(dirname "$(readlink -f "$0")")

export SOURCE_DATE_EPOCH=1787990400

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
# is a package that ships a claim nobody checked. GPU-dependent suites skip
# themselves when no card is present; the unit suite must pass regardless.
( cd build && ctest --output-on-failure -R unit )

ROOT="$work/pkgroot"
mkdir -p "$ROOT/DEBIAN" "$ROOT/usr/share/doc/arcint" "$ROOT/usr/share/arcint"
DESTDIR="$ROOT" cmake --install build >/dev/null

[ -x "$ROOT/usr/bin/arcint" ] || { echo "FEHLT: /usr/bin/arcint" >&2; exit 1; }

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

INSTALLED_KB=$(du -sk "$ROOT" | cut -f1)
cat > "$ROOT/DEBIAN/control" <<EOF
Package: arcint
Version: ${PKGVER}-${PKGREL}
Section: misc
Priority: optional
Architecture: amd64
Installed-Size: ${INSTALLED_KB}
Depends: libc6 (>= 2.34), libstdc++6 (>= 13), marfrit-openvino (= ${OV_DEP_VERSION})
Recommends: intel-opencl-icd
Maintainer: Markus Fritsche <mfritsche@reauktion.de>
Homepage: https://github.com/marfrit/arcint
Description: Narrow LLM inference engine for Intel Arc GPUs
 arcint serves exactly three Qwen models on exactly two Intel Arc cards over an
 OpenAI-compatible HTTP surface. It owns its scheduler, paged KV cache, GDN
 ledger, exact prefix cache, speculation and sampling; OpenVINO supplies the
 compiler and kernels.
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
