#!/usr/bin/env bash
# 0227: the offline check of flash_attn_dec27.cl, no enqueue. Compiles the text with the options the
# backend builds it with (ggml-opencl.cpp ggml_cl_dec27_kernels: the backend's compile options plus
# -cl-intel-256-GRF-per-thread and -DDEC27_MT=<mask slots>) for bmg-g21 (B60)
# and bmg-g31 (B70), dumps IGC's ISA and reports per build and kernel: the GRF mode, scratch (spill)
# use, an IGC recompile (a second dump of the kernel, the `_1_` retry with rolled loops), DPAS, the
# sends (2D block loads, other global loads and stores, local ones: there must be none) and the md5
# of the kernel's ISA text (comment lines dropped), so a red build can be shown to change the kernel
# and an unchanged one to be unchanged.
#   tools/ocloc_fa_dec27.sh <tree>/ggml/src/ggml-opencl/kernels/flash_attn_dec27.cl <outdir> ["extra -D options"]
# Builds: MT 2 (24 query heads on 4, the dense 27B hybrid), MT 1 (16 on 2) and MT 8 (the most); with
# DEC27_REDS=1 also the seven red builds of MT 2, whose main or merge kernel must differ from the
# green one (IGC's two hash moves, which change with the options, are left out of the md5).
# The main kernel does not fit 256 registers: its P V phase (the scale normalised a block) spills. That is the
# form's known price (measured with it on the card); the tool prints the spill size and the instruction count and
# fails above DEC27_SPILL_MAX bytes (default 4608; measured: MT 1 3,584, MT 2 3,648, MT 8 4,416) - a tripwire for
# a change that spills more. The prepass and the merge must not spill at all.
# Needs ocloc (the compute runtime's offline compiler). PASS: every build 256 GRF, the main kernel's scratch within
# the bound and none elsewhere, one dump a kernel (no IGC retry), no local-memory send, the main kernel with DPAS,
# on both targets.
set -uo pipefail
export LC_ALL=C
src=$1; out=$2; extra=${3:-}
mkdir -p "$out"
base="-cl-std=CL3.0 -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math"
base="$base -cl-intel-256-GRF-per-thread $extra"
kernels="kernel_fa_dec27_q kernel_fa_dec27 kernel_fa_dec27_merge"
variants="mt2 mt1 mt8"
[ "${DEC27_REDS:-0}" = 1 ] && variants="$variants red1 red2 red3 red4 red5 red6 red7"
spill_max=${DEC27_SPILL_MAX:-4608}
bad=0
declare -A green
for dev in bmg-g21 bmg-g31; do
  for v in $variants; do
    case $v in
      mt2) opt="$base -DDEC27_MT=2" ;;
      mt1) opt="$base -DDEC27_MT=1" ;;
      mt8) opt="$base -DDEC27_MT=8" ;;
      red1) opt="$base -DDEC27_MT=2 -DDEC27_RED_V_SCALE" ;;
      red2) opt="$base -DDEC27_MT=2 -DDEC27_RED_TAIL" ;;
      red3) opt="$base -DDEC27_MT=2 -DDEC27_RED_SEAM" ;;
      red4) opt="$base -DDEC27_MT=2 -DDEC27_RED_NOMAX" ;;
      red5) opt="$base -DDEC27_MT=2 -DDEC27_RED_K_SCALE2" ;;
      red6) opt="$base -DDEC27_MT=2 -DDEC27_RED_PVNORM_IC" ;;
      red7) opt="$base -DDEC27_MT=2 -DDEC27_RED_MASK_ALIGN" ;;
    esac
    d="$out/$dev-$v"; rm -rf "$d"; mkdir -p "$d"
    ( cd "$d" && IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$d/dump" \
        ocloc compile -q -device $dev -file "$src" -options "$opt" -output dec27 > build.log 2>&1 ); rc=$?
    echo "== $dev $v compile rc=$rc ($(ocloc --version 2>/dev/null | head -1))"
    [ $rc = 0 ] || { tail -30 "$d/build.log"; bad=1; continue; }
    zi=$(ls "$d"/dump/*.zeinfo | head -1)
    scratch=$(grep -c -E "per_thread_memory_buffers|spill_size|private_size" "$zi")
    differs=0
    for kn in $kernels; do
      files=$(grep -l -x "//.kernel $kn" "$d"/dump/*.asm)
      n=$(echo "$files" | grep -c .)
      f=$(echo "$files" | head -1)
      [ -n "$f" ] || { echo "$kn: no ISA dump"; bad=1; continue; }
      grf=$(grep -m1 -o "numGRF=[0-9]*" "$f")
      dpas=$(grep -c " dpas" "$f")
      l2d=$(grep -cE " load_block2d" "$f")
      ldg=$(grep -cE " load\.ugm" "$f"); lds=$(grep -cE " load\.slm" "$f")
      stg=$(grep -cE " store\.ugm" "$f"); sts=$(grep -cE " store\.slm" "$f")
      spill=$(grep -c -E "^//\.spill size|Scratch\[" "$f")
      fsp=$(grep -c -E "^//\.spill flag" "$f")
      lines=$(grep -c . "$f")
      isa=$(grep -v '^ *//' "$f" | grep -v -E "mov \([0-9]+\|M0\) +null<1>:ud +0x[0-9A-F]+:ud" | md5sum | cut -c1-12)
      echo "$kn: dumps=$n $grf dpas=$dpas load_block2d=$l2d load.ugm=$ldg load.slm=$lds store.ugm=$stg store.slm=$sts scratch_lines=$spill flag_spill_lines=$fsp asm_lines=$lines isa_md5=$isa"
      [ "$n" = 1 ] || bad=1
      [ "$grf" = numGRF=256 ] || bad=1
      [ "$lds" = 0 ] && [ "$sts" = 0 ] || { echo "$kn: a local-memory send"; bad=1; }
      if [ $kn = kernel_fa_dec27 ]; then
        # the main kernel spills (the form's price): its size against the bound; an IGC retry dump stays a failure
        sz=$(grep -m1 '^//.spill size' "$f" | grep -oE '[0-9]+' | head -1); sz=${sz:-0}
        echo "$kn: scratch: spill size $sz bytes (bound $spill_max) $(grep -m1 '^//.instCount' "$f")"
        [ "$sz" -le "$spill_max" ] || { echo "$kn: the scratch is above the bound"; bad=1; }
      else
        [ "$spill" = 0 ] || { echo "$kn: scratch (spill) in the ISA"; bad=1; }
      fi
      [ $kn = kernel_fa_dec27 ] && [ "$dpas" -eq 0 ] && { echo "the main kernel without DPAS"; bad=1; }
      case $v in
        mt2) green[$dev-$kn]=$isa ;;
        red*) [ "${green[$dev-$kn]}" != "$isa" ] && differs=1 ;;
      esac
    done
    case $v in red*) [ $differs = 1 ] && echo "$v: the ISA differs from the green build's" || { echo "$v: the ISA is the green build's - the red is not in the build"; bad=1; } ;; esac
    echo "scratch/spill entries in zeinfo: $scratch"
    ls "$d"/dump | grep -q "_1_" && { echo "an IGC retry dump (_1_) exists"; bad=1; }
  done
done
[ $bad = 0 ] && echo "OCLOC-FA-DEC27: PASS (256 GRF, the main kernel's scratch within $spill_max bytes and none elsewhere, no recompile, no local memory, both targets, every build)" || echo "OCLOC-FA-DEC27: FAIL"
