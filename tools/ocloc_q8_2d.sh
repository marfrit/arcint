#!/usr/bin/env bash
# 0220: the offline check of mul_mm_kq_2d.cl's Q8_0 program, no enqueue. Compiles the
# text with the options the backend builds it with (ggml-opencl.cpp: the K-quant 2D
# program's options plus -DQ8_2D_ONLY -DQ8_2D_KERNELS) for bmg-g21 (B60) and bmg-g31
# (B70), dumps IGC's ISA and reports per kernel: the GRF mode, scratch (spill) use,
# an IGC recompile (a second dump of the same kernel, the `_1_` retry with rolled
# loops), DPAS, 2D block loads and other loads/stores.
#   tools/ocloc_q8_2d.sh <tree>/ggml/src/ggml-opencl/kernels/mul_mm_kq_2d.cl <outdir> ["extra -D options"]
# Needs ocloc (the compute runtime's offline compiler). PASS: 256 GRF, no scratch, one dump a kernel,
# on both targets. Red of the check itself: -DTMM=128 spills, retries and FAILs.
set -uo pipefail
export LC_ALL=C
src=$1; out=$2; extra=${3:-}
mkdir -p "$out"
base="-cl-std=CL3.0 -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math"
base="$base -cl-intel-256-GRF-per-thread -DTM=64 -DWG=16 -DAT=32 -DQ8_2D_ONLY -DQ8_2D_KERNELS $extra"
bad=0
for dev in bmg-g21 bmg-g31; do
  d="$out/$dev"; rm -rf "$d"; mkdir -p "$d"
  ( cd "$d" && IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$d/dump" \
      ocloc compile -q -device $dev -file "$src" -options "$base" -output q8 > build.log 2>&1 ); rc=$?
  echo "== $dev compile rc=$rc ($(ocloc --version 2>/dev/null | head -1))"
  [ $rc = 0 ] || { tail -30 "$d/build.log"; bad=1; continue; }
  zi=$(ls "$d"/dump/*.zeinfo | head -1)
  scratch=$(grep -c -E "per_thread_memory_buffers|spill_size|private_size" "$zi")
  for k in kernel_mul_mm_kq2d_q8_0_i8 kernel_mul_mm_kq2d_q8_0 kernel_quant_i8; do
    files=$(grep -l -x "//.kernel $k" "$d"/dump/*.asm)
    n=$(echo "$files" | grep -c .)
    f=$(echo "$files" | head -1)
    grf=$(grep -m1 -o "numGRF=[0-9]*" "$f")
    dpas=$(grep -c " dpas" "$f")
    b2d=$(grep -c "load_block2d" "$f")
    ld=$(grep -cE " load\.(ugm|slm)" "$f")
    st=$(grep -cE " store\.(ugm|slm)" "$f")
    echo "$k: dumps=$n $grf dpas=$dpas block2d_loads=$b2d other_loads=$ld stores=$st"
    [ "$n" = 1 ] || bad=1
    [ "$grf" = numGRF=256 ] || bad=1
  done
  echo "scratch/spill entries in zeinfo: $scratch"
  [ "$scratch" = 0 ] || bad=1
  ls "$d"/dump | grep -q "_1_" && { echo "an IGC retry dump (_1_) exists"; bad=1; }
done
[ $bad = 0 ] && echo "OCLOC-Q8-2D: PASS (256 GRF, no scratch, no recompile, both targets)" || echo "OCLOC-Q8-2D: FAIL"
