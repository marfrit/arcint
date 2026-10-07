#!/bin/bash
# Repack a published marfrit-openvino .deb with its libraries' RUNPATH set to $ORIGIN, under the next Debian
# revision: the same binaries, found where the package puts them.
#
# Why: up to +p25-1 build-deb.sh copied the patched libraries out of the OpenVINO build tree with the build
# tree's RUNPATH (<build>/temp/Linux_x86_64/tbb/lib, <build>/bin/intel64/Release). On the build host those
# paths exist and every library resolved; on a fresh host libopenvino.so and the plugins found neither their
# bundled libtbb.so.12 nor each other, and arcint failed to start (fresh trixie container, 2026-10-07).
# build-deb.sh now sets the RUNPATH itself; this script fixes an already published package.
#
# usage: repack-runpath.sh <marfrit-openvino_<ver>-<rev>_amd64.deb> "<changelog line>"
#   needs patchelf, dpkg-deb, readelf (binutils)
set -euo pipefail
IN=$(readlink -f "${1:?input .deb}"); NOTE=${2:?changelog line}
PREFIX=/usr/lib/marfrit-openvino
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
dpkg-deb -R "$IN" "$work/root"
OLDVER=$(sed -n 's/^Version: //p' "$work/root/DEBIAN/control")
REV=${OLDVER##*-}; UP=${OLDVER%-*}; NEWVER="$UP-$((REV + 1))"

n=0
while IFS= read -r f; do
    r=$(readelf -d "$f" 2>/dev/null | sed -n 's/.*(RUNPATH).*\[\(.*\)\].*/\1/p')
    [ -n "$r" ] || continue
    case "$r" in
        '$ORIGIN'|'$ORIGIN/'*) continue ;;
    esac
    patchelf --set-rpath '$ORIGIN' "$f"
    n=$((n + 1))
done < <(find "$work/root$PREFIX" -type f -name '*.so*')
echo "RUNPATH set to \$ORIGIN in $n libraries"

# no library may still point outside the package
bad=$(find "$work/root$PREFIX" -type f -name '*.so*' -exec sh -c \
      'readelf -d "$1" 2>/dev/null | grep -E "\((RUNPATH|RPATH)\)" | grep -v "\$ORIGIN" | sed "s|^|$1: |"' _ {} \;)
[ -z "$bad" ] || { echo "RUNPATH outside the package left:" >&2; echo "$bad" >&2; exit 1; }

sed -i "s/^Version: .*/Version: $NEWVER/" "$work/root/DEBIAN/control"
DOC="$work/root/usr/share/doc/marfrit-openvino/changelog.Debian.gz"
if [ -f "$DOC" ]; then
    { printf 'marfrit-openvino (%s) bookworm trixie; urgency=medium\n\n  * %s\n\n -- Markus Fritsche <mfritsche@reauktion.de>  %s\n\n' \
        "$NEWVER" "$NOTE" "$(date -R)"; zcat "$DOC"; } | gzip -9 -n > "$DOC.new"
    mv "$DOC.new" "$DOC"
fi
# md5sums follow the patched files
(cd "$work/root" && find . -path ./DEBIAN -prune -o -type f -print | sed 's|^\./||' | sort | xargs md5sum > DEBIAN/md5sums)
OUT="marfrit-openvino_${NEWVER}_amd64.deb"
dpkg-deb --root-owner-group --build "$work/root" "$OUT" >/dev/null
echo "built: $(readlink -f "$OUT")"
