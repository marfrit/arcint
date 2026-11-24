#!/usr/bin/env bash
# ocloc-check.sh — the offline check of the port kernel on the card host, with no call
# into the driver: compile flash_attn_port.cl (and, optionally, a module from
# build-spirv.sh) ahead of time with ocloc, then read the compiled binary's ISA
# and report the GRF mode, the spills, and the kernel's presence. Nothing here
# enqueues; nothing is committed (`out/` and `*.bin`/`*.asm` are ignored).
#
#   bash ocloc-check.sh <llama-tree> [outdir]
#
# Needs ocloc with the compute runtime (the card host has /usr/bin/ocloc,
# ocloc-26.27.1). The recipe host does not, so this script is for the card host.
set -euo pipefail
export LC_ALL=C

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tree="${1:?usage: bash ocloc-check.sh <llama-tree> [outdir]}"
out="${2:-$here/out/ocloc}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"

command -v ocloc >/dev/null || { echo "ocloc-check: no ocloc on this host (the card host has /usr/bin/ocloc)" >&2; exit 3; }
ocloc --version | head -2

# the device: an Arc Pro B60 is BMG-G21 = ocloc's bmg-g21 (B70 = bmg-g31; never
# `-device bmg`, which resolves to the B580-class config — exl3xpu issue #1 §1)
dev="${OCLOC_DEV:-bmg-g21}"

# the port kernel, the same defines the run-time build uses
src="$tree/ggml/src/ggml-opencl/kernels/flash_attn_port.cl"
bin="$out/flash_attn_port.$dev.bin"
# -vc-codegen is required for an ESIMD image (exl3xpu issue #1 §1-2: without it
# IGC's VLD path splits the module and rejects a helper without the VC decoration)
# 256 GRF is the backend's default here (README §GRF): at 128 the kernel spills
# 8,192 B. OCLOC_GRF=128 checks the alternative arm, which must fail on spread.
grf="${OCLOC_GRF:-256}"
opts="-DDKQ=128 -DBK=32 -vc-codegen -cl-fp32-correctly-rounded-divide-sqrt -ze-opt-greater-than-4GB-buffer-required"
[ "$grf" = 256 ] && opts="$opts -cl-intel-256-GRF-per-thread"

echo "ocloc compile: $src -> $bin (device $dev)"
ocloc compile -device "$dev" -input "$src" -output "$bin" -options "$opts" || {
    echo "ocloc-check: compile failed; the log above is the whole answer" >&2; exit 4;
}

# ISA and register state, from the compiled binary alone
ocloc disasm -device "$dev" -input "$bin" -dump "$out/port.$dev.asm" >/dev/null 2>&1 || true
asm="$out/port.$dev.asm"
[ -f "$asm" ] || { echo "ocloc-check: no disassembly (ocloc disasm did not write $asm)" >&2; exit 5; }

# the GRF mode: a 128-GRF kernel says simd 16 / grf_count 128 (or no grf_count), a
# 256-GRF one carries grf_count 256 — the number the target's arithmetic assumes
grf=$(grep -m1 -o "grf_count[^0-9]*[0-9]*" "$asm" | grep -o "[0-9]*$" || echo "unstated")
spill=$(grep -c -i "spill" "$asm" || true)
calls=$(grep -m1 -o "simd_size[^0-9]*[0-9]*" "$asm" | grep -o "[0-9]*$" || echo "unstated")
echo "GRF mode: ${grf}   SIMD: ${calls}   spill lines: ${spill}"

# the kernel is present: its name in the header block (kernel void kernel_flash_attn_port)
grep -q "kernel_flash_attn_port" "$asm" && echo "kernel present: kernel_flash_attn_port" || {
    echo "ocloc-check: kernel_flash_attn_port not in the disassembly" >&2; exit 6; }

# the target: 256 GRF (the default) and 0 spill lines; OCLOC_GRF=128 is the arm
# the gate measured at 8,192 B of scratch, so it is a comparison, not a default.
if [ "$spill" != "0" ]; then
    echo "ocloc-check: $spill spill line(s): the 128-GRF plan does not hold this kernel as written" >&2
    echo "  (carry the 256-GRF arm next: add -cl-intel-256-GRF-per-thread and re-run this check)" >&2
    exit 7
fi
echo "ocloc-check: 0 spill, GRF ${grf} — the offline bar is met"
