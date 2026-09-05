#!/bin/bash
# Build marfrit-openvino_<ver>_amd64.deb — the pinned OpenVINO runtime that
# arcint links against.
#
# Why this package exists at all: arcint is built against an OpenVINO *nightly*
# (2026.4 dev), because that is the stack every measurement in the arcint repo
# was taken on. Debian ships no OpenVINO, and the nightly index prunes old
# builds, so "just pip install it at deploy time" is a dependency that expires.
# Repacking the wheel into a versioned .deb makes the pin an artifact instead
# of an instruction.
#
# Deliberately NOT registered in /etc/ld.so.conf.d/: arcint finds these libs
# through its RPATH. A system-wide entry would put a nightly OpenVINO ahead of
# anything else on the host that ever links the same sonames.
#
# SINCE +p1 THIS IS NO LONGER A PURE REPACK. The wheels still supply the layout,
# the CMake package, the headers and the tokenizers library; the OpenVINO
# libraries themselves come from a source build of the SAME upstream commit with
# the patch series in patches/ applied. Run build-openvino.sh first. The wheel
# stays in the recipe because its directory layout and CMake glue are what arcint
# builds against, and reproducing those by hand is how you get a package that
# configures and then fails at link time.
#
# The one component still taken from a wheel binary is libopenvino_tokenizers.so.
# That is a separate upstream project, it is not patched, and it links only
# against the public OpenVINO API of the very commit we build — so the ABI it was
# compiled against is the ABI we ship.
set -euo pipefail

# Upstream nightly version, exactly as pip names it. The Debian version uses a
# tilde so that a future 2026.4.0 release sorts ABOVE this dev build.
OV_VER=2026.4.0.dev20260821
OV_TOK_VER=2026.4.0.0.dev20260821
PKGVER=2026.4.0~dev20260821+p5
PKGREL=1
# Sorts above the unpatched 2026.4.0~dev20260821-1 and still below a real
# 2026.4.0 release, because the tilde keeps the whole thing under it.

# Where build-openvino.sh left its output. Libraries come from the build tree's
# bin/ rather than from its install prefix: upstream does not install the JAX
# frontend in this configuration, and shipping fewer libraries than the previous
# package did would be a silent capability change.
OV_BUILD_DIR=${OV_BUILD_DIR:-${OV_SRC:-$HOME/ovsrc-pkg}}   # OpenVINO writes bin/intel64/Release under the SOURCE tree, whatever the build dir
OV_BIN=$OV_BUILD_DIR/bin/intel64/Release
OV_TBB=${OV_BUILD_PREFIX:-$HOME/ovinstall}/runtime/3rdparty/tbb/lib
PATCHLEVEL=marfrit-p5
PYTAG=cp313
OV_WHEEL=openvino-${OV_VER}-22849-${PYTAG}-${PYTAG}-manylinux_2_28_x86_64.whl
OV_TOK_WHEEL=openvino_tokenizers-${OV_TOK_VER}-py3-none-manylinux_2_28_x86_64.whl
OV_WHEEL_SHA256=54e64b62ad0fc56ed7963992e53c871673d622643adfebcc0d5d989319fb1d5d
OV_TOK_WHEEL_SHA256=67c2f70f2f4eb860e48a8b7d3b0a6f0748fd7f132db90a3444c3fbbf0ee01dc0
# NOT the /simple/ index path: that one serves the PEP 503 HTML listing, whose
# anchors point here. Fetching the /simple/ URL directly hands you an HTML page
# that then fails the checksum — which is exactly how this line got written.
INDEX=https://storage.openvinotoolkit.org/wheels/nightly
PREFIX=/usr/lib/marfrit-openvino
HERE=$(dirname "$(readlink -f "$0")")

# Reproducible build (see the lmcp recipe for why reprepro insists).
export SOURCE_DATE_EPOCH=1787990400

work=$(mktemp -d)
trap "rm -rf $work" EXIT
cd "$work"

# The wheels come from the nightly index by exact filename. If the nightly has
# been pruned, MIRROR_DIR lets a local copy stand in — keep one, that is the
# insurance this whole package is about.
hole() { # $1=file $2=index-project
    if [ -n "${MIRROR_DIR:-}" ] && [ -f "$MIRROR_DIR/$1" ]; then
        cp "$MIRROR_DIR/$1" "$1"
    else
        curl --connect-timeout 10 --max-time 900 --retry 3 --retry-delay 5 \
             -sSLfo "$1" "$INDEX/${2}/$1"
    fi
}
hole "$OV_WHEEL" openvino
hole "$OV_TOK_WHEEL" openvino-tokenizers
echo "$OV_WHEEL_SHA256  $OV_WHEEL"         | sha256sum -c
echo "$OV_TOK_WHEEL_SHA256  $OV_TOK_WHEEL" | sha256sum -c

# python3 rather than unzip: a wheel is a zip, python3 is on every Debian, and
# adding a build dependency for one extraction is a bad trade.
mkdir -p unpack
python3 -m zipfile -e "$OV_WHEEL" unpack/
python3 -m zipfile -e "$OV_TOK_WHEEL" unpack/

ROOT="$work/pkgroot"
mkdir -p "$ROOT/DEBIAN" "$ROOT$PREFIX" "$ROOT/usr/share/doc/marfrit-openvino"

# The wheel's INTERNAL LAYOUT IS LOAD-BEARING and must not be flattened.
# OpenVINOTargets.cmake resolves every library as
# "${_IMPORT_PREFIX}/openvino/libs/<soname>", with _IMPORT_PREFIX derived two
# levels up from the cmake directory. Move the .so files into a lib/ of our own
# taste and find_package(OpenVINO) still succeeds — and then fails at link time
# pointing at paths that do not exist. So: copy openvino/ and
# openvino_tokenizers/ verbatim under one prefix, exactly as site-packages had
# them.
cp -a unpack/openvino "$ROOT$PREFIX/"
mkdir -p "$ROOT$PREFIX/openvino_tokenizers"
cp -a unpack/openvino_tokenizers/lib "$ROOT$PREFIX/openvino_tokenizers/"

# --- the patched libraries replace the wheel's -----------------------------
[ -d "$OV_BIN" ] || {
    echo "No build at $OV_BIN. Run build-openvino.sh first — this package is" >&2
    echo "no longer a pure repack; see the header." >&2; exit 1; }

# Every library the wheel shipped must exist in the build. A missing one is a
# capability regression, and the loud failure is the whole point of this loop.
missing=0
for f in $(cd "$ROOT$PREFIX/openvino/libs" && ls); do
    case "$f" in
        cache.json) continue ;;                 # AUTO plugin data, not a binary
        tbb.pc)     src="$OV_TBB/pkgconfig/$f" ;;
        libtbb*|libhwloc*) src="$OV_TBB/$f" ;;
        *)          src="$OV_BIN/$f" ;;
    esac
    if [ ! -e "$src" ]; then
        echo "the build does not provide $f (wheel has it)" >&2
        missing=1
        continue
    fi
    # -L: the build keeps soname symlink chains, the wheel is flat. Copy the
    # real file under the wheel's name.
    cp -Lf "$src" "$ROOT$PREFIX/openvino/libs/$f"
done
[ "$missing" = 0 ] || { echo "refusing to ship a package with fewer libraries than the last one" >&2; exit 1; }

# The wheel ships stripped libraries; a plain Release build does not, and the
# difference is about a quarter of the package.
find "$ROOT$PREFIX/openvino/libs" -name '*.so*' -type f -exec strip --strip-unneeded {} + 2>/dev/null || true

# The shipped runtime must say out loud that it is patched, so that any log line
# naming an OpenVINO version answers "which one" without a detour.
# NOT `| grep -q`: under `set -o pipefail` grep -q exits at the first match,
# strings takes SIGPIPE, the pipeline reports 141, and the `!` turns that into
# a "failure" exactly when the marker IS present. grep without -q drains its
# input, so the pipeline's status is grep's.
if ! strings -a "$ROOT$PREFIX/openvino/libs/libopenvino.so.2640" | grep -F -- "$PATCHLEVEL" >/dev/null; then
    echo "the packaged libopenvino does not carry $PATCHLEVEL — wrong build tree?" >&2
    exit 1
fi

# Python is not what we want from these wheels. Dropping the bindings keeps the
# package to the runtime, the headers and the CMake glue.
rm -rf "$ROOT$PREFIX/openvino/__pycache__" \
       "$ROOT$PREFIX/openvino"/*.py "$ROOT$PREFIX/openvino"/*.pyi \
       "$ROOT$PREFIX/openvino"/_pyopenvino*.so
find "$ROOT$PREFIX" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$ROOT$PREFIX" -name '*.py' -delete 2>/dev/null || true
find "$ROOT$PREFIX" -name '*.pyi' -delete 2>/dev/null || true

# python3's extractor drops permission bits; normalise to Debian's convention
# for shared libraries so lintian-style expectations and repeat builds agree.
find "$ROOT$PREFIX" -type f -name '*.so*' -exec chmod 644 {} +
find "$ROOT$PREFIX" -type d -exec chmod 755 {} +

# Zaehlproben. An incomplete runtime fails at model-load time on the target
# host, minutes later, with a message about a plugin — not here, where it would
# be cheap to see. Check the pieces that actually get loaded, and the CMake
# glue that the arcint build needs.
for f in libopenvino.so.2640 libopenvino_intel_gpu_plugin.so libtbb.so.12 \
         libopenvino_ir_frontend.so.2640; do
    [ -f "$ROOT$PREFIX/openvino/libs/$f" ] || { echo "FEHLT im Paket: openvino/libs/$f" >&2; exit 1; }
done
[ -f "$ROOT$PREFIX/openvino_tokenizers/lib/libopenvino_tokenizers.so" ] || {
    echo "FEHLT: openvino_tokenizers/lib/libopenvino_tokenizers.so" >&2; exit 1; }
[ -f "$ROOT$PREFIX/openvino/cmake/OpenVINOConfig.cmake" ] || {
    echo "FEHLT: openvino/cmake/OpenVINOConfig.cmake — ohne das kann arcint nicht bauen" >&2
    exit 1; }
[ -d "$ROOT$PREFIX/openvino/include/openvino" ] || {
    echo "FEHLT: openvino/include — der CMake-Zielsatz zeigt darauf" >&2; exit 1; }
# The cmake package computes _IMPORT_PREFIX two levels up; assert that this
# lands on our prefix and not somewhere outside the package.
grep -q '\${_IMPORT_PREFIX}/openvino/libs/libopenvino.so' \
     "$ROOT$PREFIX/openvino/cmake/OpenVINOTargets-release.cmake" || {
    echo "Wheel-Layout hat sich geaendert: OpenVINOTargets zeigt nicht mehr auf openvino/libs" >&2
    exit 1; }

install -m 644 unpack/openvino-${OV_VER}.dist-info/licenses/LICENSE \
    "$ROOT/usr/share/doc/marfrit-openvino/LICENSE" 2>/dev/null || \
  install -m 644 unpack/openvino-${OV_VER}.dist-info/LICENSE \
    "$ROOT/usr/share/doc/marfrit-openvino/LICENSE"
cp "$HERE/debian/copyright" "$ROOT/usr/share/doc/marfrit-openvino/copyright"
cp "$HERE/debian/changelog" "$ROOT/usr/share/doc/marfrit-openvino/changelog.Debian"
gzip -9 -n "$ROOT/usr/share/doc/marfrit-openvino/changelog.Debian"

INSTALLED_KB=$(du -sk "$ROOT" | cut -f1)
cat > "$ROOT/DEBIAN/control" <<EOF
Package: marfrit-openvino
Version: ${PKGVER}-${PKGREL}
Section: libs
Priority: optional
Architecture: amd64
Installed-Size: ${INSTALLED_KB}
Depends: libc6 (>= 2.28), libstdc++6, ocl-icd-libopencl1 | libopencl1
Recommends: intel-opencl-icd
Maintainer: Markus Fritsche <mfritsche@reauktion.de>
Homepage: https://github.com/openvinotoolkit/openvino
Description: OpenVINO ${OV_VER} runtime (patched) for the marfrit fleet
 The exact OpenVINO nightly that arcint is built and measured against, with
 the patch series in debian/marfrit-openvino/patches/ applied to the same
 upstream commit. Layout, CMake package, headers and the tokenizers library
 come from the upstream wheels; the OpenVINO libraries are built from source.
 .
 The runtime reports itself as ${OV_VER}-...-${PATCHLEVEL} so a version string
 in a log says whether it is the patched build or the stock nightly.
 .
 Not registered with the dynamic linker on purpose: consumers find these
 libraries through their own RPATH, so a nightly runtime never shadows a
 system library for unrelated software.
 .
 Ships the CMake package and headers as well, so it is also the build
 dependency for arcint. The wheel's directory layout is preserved verbatim
 because the CMake targets resolve libraries relative to it.
EOF

DEB_OUT=marfrit-openvino_${PKGVER}-${PKGREL}_amd64.deb
dpkg-deb --root-owner-group --build "$ROOT" "$HERE/$DEB_OUT"
echo "built: $HERE/$DEB_OUT"
