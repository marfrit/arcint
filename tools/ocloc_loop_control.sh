#!/usr/bin/env bash
# 0221: the offline check of loop_control.cl, no enqueue. Compiles the text with the options the
# backend builds it with (ggml-opencl.cpp ggml_cl_loop_control_kernel: the backend's compile
# options plus -cl-intel-256-GRF-per-thread, and -DLC_BF16 for the bf16 variant) for bmg-g21 (B60)
# and bmg-g31 (B70), dumps IGC's ISA and reports per build: the GRF mode, scratch (spill) use, an
# IGC recompile (a second dump of the kernel, the `_1_` retry with rolled loops), DPAS, sends
# (loads / stores, global and local) and the waits on a load's token (sync / $N.dst before a use).
#   tools/ocloc_loop_control.sh <tree>/ggml/src/ggml-opencl/kernels/loop_control.cl <outdir> ["extra -D options"]
# Needs ocloc (the compute runtime's offline compiler). PASS: every build 256 GRF, no scratch, one
# dump, on both targets; the bf16 build has DPAS, the f32 build none.
set -uo pipefail
export LC_ALL=C
src=$1; out=$2; extra=${3:-}
mkdir -p "$out"
base="-cl-std=CL3.0 -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math"
base="$base -cl-intel-256-GRF-per-thread $extra"
bad=0
for dev in bmg-g21 bmg-g31; do
  for v in f32 bf16; do
    opt="$base"; [ $v = bf16 ] && opt="$opt -DLC_BF16"
    d="$out/$dev-$v"; rm -rf "$d"; mkdir -p "$d"
    ( cd "$d" && IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$d/dump" \
        ocloc compile -q -device $dev -file "$src" -options "$opt" -output lc > build.log 2>&1 ); rc=$?
    echo "== $dev $v compile rc=$rc ($(ocloc --version 2>/dev/null | head -1))"
    [ $rc = 0 ] || { tail -30 "$d/build.log"; bad=1; continue; }
    zi=$(ls "$d"/dump/*.zeinfo | head -1)
    scratch=$(grep -c -E "per_thread_memory_buffers|spill_size|private_size" "$zi")
    files=$(grep -l -x "//.kernel kernel_loop_control" "$d"/dump/*.asm)
    n=$(echo "$files" | grep -c .)
    f=$(echo "$files" | head -1)
    grf=$(grep -m1 -o "numGRF=[0-9]*" "$f")
    dpas=$(grep -c " dpas" "$f")
    ldg=$(grep -cE " load\.ugm" "$f"); lds=$(grep -cE " load\.slm" "$f")
    stg=$(grep -cE " store\.ugm" "$f"); sts=$(grep -cE " store\.slm" "$f")
    pf=$(grep -cE "load\.ugm.* null:0" "$f")
    sync=$(grep -cE "sync\.(allrd|allwr|nop)" "$f")
    lines=$(grep -c . "$f")
    echo "kernel_loop_control: dumps=$n $grf dpas=$dpas load.ugm=$ldg load.slm=$lds store.ugm=$stg store.slm=$sts prefetch=$pf sync=$sync asm_lines=$lines"
    [ "$n" = 1 ] || bad=1
    [ "$grf" = numGRF=256 ] || bad=1
    [ $v = bf16 ] && [ "$dpas" -eq 0 ] && { echo "bf16 build without DPAS"; bad=1; }
    [ $v = f32 ] && [ "$dpas" -ne 0 ] && { echo "f32 build with DPAS"; bad=1; }
    echo "scratch/spill entries in zeinfo: $scratch"
    [ "$scratch" = 0 ] || bad=1
    ls "$d"/dump | grep -q "_1_" && { echo "an IGC retry dump (_1_) exists"; bad=1; }
  done
done
[ $bad = 0 ] && echo "OCLOC-LOOP-CONTROL: PASS (256 GRF, no scratch, no recompile, both targets, both maths)" || echo "OCLOC-LOOP-CONTROL: FAIL"
