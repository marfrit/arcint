#!/bin/bash
# Build the pinned OpenVINO from source with the patch series in patches/.
#
# Why this exists: the .deb used to be a pure repack of two upstream wheels,
# which built in seconds and carried no divergence at all. It stops being
# enough the moment we need a plugin fix that upstream has not shipped. Per
# DESIGN.md §1 in the arcint repository, rung 2 of "smallest sufficient
# divergence" is a numbered patch set applied AT BUILD TIME here — not a
# divergent checkout somebody has to keep rebased.
#
# The pin is a commit, not a branch. `2026.4.0-22849-71640275d29` is what the
# upstream nightly wheel reports; 71640275 is the commit it was cut from, and
# every measurement in the arcint repository was taken on that code.
#
# This must run on x86_64 with a glibc userland. The packaging CI cannot do it
# (its only amd64 runner is host-exec on musl with no container runtime), so
# this is a by-hand build like the engine's own package. That is a known and
# stated gap, not an oversight.
set -euo pipefail

PIN=71640275
PATCHLEVEL=marfrit-p6         # appended to the version string, so a loaded
                              # runtime says out loud that it is patched.
                              # The build-number FIELD must stay numeric:
                              # ov_parse_ci_build_number wants -([0-9]+)- and
                              # dies on anything else. The tail after it is free.
UPSTREAM=https://github.com/openvinotoolkit/openvino.git
SRC=${OV_SRC:-$HOME/ovsrc-pkg}
BUILD=$SRC/build-prod
PREFIX=${OV_BUILD_PREFIX:-$HOME/ovinstall}
HERE=$(dirname "$(readlink -f "$0")")
JOBS=${JOBS:-$(nproc)}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

if [ ! -d "$SRC/.git" ]; then
    log "Cloning upstream (this is a few GB with submodules)"
    git clone "$UPSTREAM" "$SRC"
fi

cd "$SRC"
log "Checking out the pin: $PIN"
git fetch --quiet origin "$PIN" 2>/dev/null || git fetch --quiet origin
git checkout --quiet --force "$PIN"
git submodule update --init --recursive --quiet

# Everything that is not the pinned commit has to go before the patches are
# applied. This is the guard that keeps a measurement instrument out of a
# package: the arcint session's own tree carries per-stage timing accumulators
# (network.cpp, primitive_inst.cpp, stage_acc.hpp) that are how the MoE churn
# was found, and none of that belongs in a shipped runtime.
log "Resetting to a pristine tree"
git clean -qfdx -e build-prod
git reset --quiet --hard "$PIN"

log "Applying the patch series"
shopt -s nullglob
for p in "$HERE"/patches/*.patch; do
    log "  $(basename "$p")"
    git apply --check "$p" || { echo "does not apply to $PIN — fix patches/, not the pin" >&2; exit 1; }
    git apply "$p"
done
# Exactly the patches, nothing else.
git status --porcelain

log "Configuring (Release, debug caps OFF, plugin set matching the wheel)"
cmake -S . -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCI_BUILD_NUMBER="2026.4.0-22849-${PIN}d29-${PATCHLEVEL}" \
  -DENABLE_INTEL_GPU=ON -DENABLE_INTEL_CPU=ON -DENABLE_INTEL_NPU=ON \
  -DENABLE_DEBUG_CAPS=OFF -DENABLE_GPU_DEBUG_CAPS=OFF \
  -DENABLE_PYTHON=OFF -DENABLE_SAMPLES=OFF -DENABLE_TESTS=OFF -DENABLE_WHEEL=OFF

log "Building with $JOBS jobs (about 35 minutes on 8 cores)"
cmake --build "$BUILD" -j"$JOBS"
cmake --install "$BUILD"

V=$(strings -a "$PREFIX/runtime/lib/intel64/libopenvino.so.2640" \
    | grep -oE "2026\.4\.0-[0-9]+-[0-9a-z-]+" | head -1 || true)
log "Built: ${V:-<no version string found>}"
case "$V" in
    *"$PATCHLEVEL"*) log "Version string names the patch level, good" ;;
    *) echo "Version string does not name the patch level: $V" >&2; exit 1 ;;
esac
log "Installed to $PREFIX — now run build-deb.sh"
