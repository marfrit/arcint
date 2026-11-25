#!/usr/bin/env bash
# build-spirv.sh — compile the vendored trellis-serve prefill-attention kernel
# (fa::Kern<128>, fa_kern.sycl) to the SPIR-V image an Intel GPU driver can build
# an ESIMD kernel from, and dump the kernel name, the entry point's argument ABI
# (order, type, size) and the SPIR-V capabilities/extensions/decorations involved.
#
#   bash build-spirv.sh <outdir>
#
# Writes, from one source:
#   fa_kern.precise.spv   the ORACLE arm: -fp-model=precise -O2
#   fa_kern.ref.spv       the TIMING arm: trellis-serve's own flags
#   abi.txt / abi.ref.txt, kernel-names.txt / kernel-names.ref.txt,
#   the disassemblies, the per-module post-link tables and commands.txt.
#
# The pipeline is the driver's own for a spir64 device image, step by step:
#   icpx -fsycl-device-only  -> the pre-post-link device bitcode
#   sycl-post-link ... -split-esimd -lower-esimd -> the ESIMD device image (.bc):
#       the VC IR (llvm.genx.*) plus the VectorComputeFunctionINTEL decorations
#   llvm-spirv -spirv-allow-unknown-intrinsics=llvm.genx. ... -> the module the
#       SYCL runtime itself loads (byte-identical to icpx's own device image)
# The reason it is not one icpx call: -fsycl-device-obj=spirv emits the device
# code BEFORE sycl-post-link, so it carried no VectorComputeFunctionINTEL
# decoration. Measured on the card (B60, driver 26.27.39122.11): that module
# built, but CL_PROGRAM_KERNEL_NAMES came back empty and clCreateKernel of its
# entry point returned CL_INVALID_KERNEL_NAME. See README §post-link.
#
# Needs: Intel oneAPI DPC++ 2026.1 (icpx), its sycl-post-link and llvm-spirv,
# plus python3 for the ABI dump. No GPU, no Level Zero, no torch. Everything is
# written under <outdir>; the repository stays untouched.
set -euo pipefail
export LC_ALL=C

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="$here/fa_kern.sycl"

if [ $# -lt 1 ]; then
    echo "usage: bash build-spirv.sh <outdir>" >&2
    exit 2
fi
out="$1"
mkdir -p "$out"
out="$(cd "$out" && pwd)"

oneapi_roots=( "${ONEAPI_ROOT:-}" /opt/intel/oneapi )

find_tool() {   # find_tool <name> <extra-relative-locations...>
    local name="$1"; shift
    if command -v "$name" >/dev/null 2>&1; then command -v "$name"; return 0; fi
    local r rel hit
    for r in "${oneapi_roots[@]}"; do
        [ -n "$r" ] && [ -d "$r" ] || continue
        for rel in "$@"; do
            if [ -x "$r/$rel" ]; then echo "$r/$rel"; return 0; fi
        done
        hit="$(ls -1d "$r"/compiler/*/bin/$name 2>/dev/null | tail -1 || true)"
        [ -n "$hit" ] && { echo "$hit"; return 0; }
    done
    return 1
}

icpx="$(find_tool icpx)" || {
    echo "build-spirv: icpx not found (PATH, \$ONEAPI_ROOT, /opt/intel/oneapi/compiler/*/bin)." >&2
    echo "             with oneAPI installed, 'source /opt/intel/oneapi/setvars.sh' fixes it." >&2
    exit 3
}
icpx_bin_dir="$(dirname "$icpx")"

# llvm-spirv and sycl-post-link are NOT in the public bin dir in 2026.1: they
# live one level deeper, <root>/compiler/<ver>/bin/compiler/.
locate_deep() {   # locate_deep <name> <dest-var-name>
    local name="$1" var="$2" cand found=""
    found="$(command -v "$name" || true)"
    if [ -z "$found" ]; then
        for cand in "$icpx_bin_dir/compiler/$name" "$icpx_bin_dir/$name"; do
            [ -x "$cand" ] && found="$cand" && break
        done
    fi
    [ -z "$found" ] && { echo "build-spirv: $name not found (expected next to icpx: $icpx_bin_dir/compiler/$name)" >&2; exit 3; }
    printf -v "$var" '%s' "$found"
}
locate_deep llvm-spirv llvm_spirv
locate_deep sycl-post-link post_link

spirv_dis="$(command -v spirv-dis || true)"   # oneAPI ships none; llvm-spirv --to-text is the fallback

precise_spv="$out/fa_kern.precise.spv"
ref_spv="$out/fa_kern.ref.spv"

compile_lines=()
vc_report=()

# ---------------------------------------------------------------------------
# build_module <label> <output.spv> <compile flags...>
#
# The compile flags below, each named once:
# -fsycl                 the source is SYCL C++ (esimd.hpp, SYCL_ESIMD_KERNEL).
# -fsycl-device-only     device code only: no host object, no loader needed.
# -fsycl-targets=spir64  generic SPIR-V, the reference's own target and what
#                        clCreateProgramWithIL consumes.
# -x c++                 .sycl is not a source extension icpx knows; without it
#                        icpx treats the file as a linker input and emits nothing
#                        ('linker' input unused [-Wunused-command-line-argument]).
# -c                     no link step.
# The FP/optimisation flags are the per-arm ones (precise: -fp-model=precise -O2;
# reference: -O3 -ffast-math -std=c++17 -fsycl-device-code-split=per_kernel).
# -fsycl-device-obj=spirv is NOT used: it is the pre-post-link form (README
# §post-link), and it produces a module whose ESIMD entry point never reaches the
# VC backend.
# ---------------------------------------------------------------------------
build_module() {
    local label="$1" dst="$2"; shift 2
    local -a extra=("$@")
    local work="$out/postlink-$label"
    rm -rf "$work"; mkdir -p "$work"
    local devbc="$work/dev.bc"

    # 1. pre-post-link device bitcode, the compiler's own device compile
    local -a cmd=("$icpx" -fsycl -fsycl-device-only -fsycl-targets=spir64
                  "${extra[@]}" -c -x c++ "$src" -o "$devbc")
    "${cmd[@]}"
    compile_lines+=("$label icpx | $(printf '%s ' "${cmd[@]}")")

    # 2. sycl-post-link: the driver's own option set, taken from `icpx -###` on
    #    the link step (2026.1.1): -split-esimd separates the ESIMD entry points,
    #    -lower-esimd lowers ESIMD to the VC IR (llvm.genx.*) and attaches the
    #    VectorComputeFunctionINTEL decorations. -emit-param-info is what carries
    #    the kernel-arg ABI, -properties writes the .prop that marks the image
    #    isEsimdImage=1. -O is the arm's own opt level.
    local opt="-O2"
    case " ${extra[*]} " in *" -O3 "*) opt="-O3" ;; esac
    local devlib="$icpx_bin_dir/../lib"
    local -a cmd2=("$post_link" "$opt" -device-globals "--device-lib-dir=$devlib"
                   -properties -spec-const=native -split=auto
                   -emit-only-kernels-as-entry-points -emit-param-info
                   -symbols -emit-exported-symbols -emit-imported-symbols
                   -split-esimd -lower-esimd -o table.txt dev.bc)
    ( cd "$work" && "${cmd2[@]}" )
    compile_lines+=("$label sycl-post-link | (cd $work && $(printf '%s ' "${cmd2[@]}"))")

    local mod
    mod="$(awk -F'|' '/esimd/ {print $1; exit}' "$work/table.txt")"
    if [ -z "$mod" ] || [ ! -f "$work/$mod" ]; then
        echo "build-spirv: $label: sycl-post-link produced no ESIMD module (table:" >&2
        cat "$work/table.txt" >&2 || true
        echo "); stopping rather than shipping a module the driver cannot expose a kernel from" >&2
        exit 4
    fi
    local sym="${mod%.bc}.sym" prop="${mod%.bc}.prop" names=""
    [ -f "$work/$sym" ] && names="$(tr '\n' ' ' < "$work/$sym")"
    local esimd_prop=""
    [ -f "$work/$prop" ] && esimd_prop="$(grep -o 'isEsimdImage=[01]' "$work/$prop" || true)"

    # 3. VC IR -> SPIR-V, with the translator flags the driver uses (same `icpx
    #    -###` link step): -spirv-allow-unknown-intrinsics=llvm.genx. is what makes
    #    the lowered VC intrinsics Import declarations instead of a hard error
    #    ("InvalidFunctionCall: Unexpected llvm intrinsic:
    #     llvm.genx.rdregionf.v16f16.v128f16.i16" is what it reads as without it,
    #    measured 2026-11-24); -spirv-max-version=1.5 and the explicit extension
    #    list are the driver's own. The list is verbatim from the link step.
    local ext="-all,+SPV_EXT_shader_atomic_float_add,+SPV_EXT_shader_atomic_float_min_max,+SPV_KHR_no_integer_wrap_decoration,+SPV_KHR_float_controls,+SPV_KHR_bit_instructions,+SPV_KHR_expect_assume,+SPV_KHR_linkonce_odr,+SPV_INTEL_subgroups,+SPV_INTEL_media_block_io,+SPV_INTEL_device_side_avc_motion_estimation,+SPV_INTEL_fpga_loop_controls,+SPV_INTEL_unstructured_loop_controls,+SPV_INTEL_fpga_reg,+SPV_INTEL_blocking_pipes,+SPV_INTEL_function_pointers,+SPV_INTEL_kernel_attributes,+SPV_INTEL_io_pipes,+SPV_INTEL_inline_assembly,+SPV_INTEL_arbitrary_precision_integers,+SPV_INTEL_float_controls2,+SPV_INTEL_vector_compute,+SPV_INTEL_fast_composite,+SPV_INTEL_arbitrary_precision_fixed_point,+SPV_INTEL_arbitrary_precision_floating_point,+SPV_INTEL_variable_length_array,+SPV_INTEL_fp_fast_math_mode,+SPV_INTEL_long_composites,+SPV_INTEL_arithmetic_fence,+SPV_INTEL_global_variable_decorations,+SPV_INTEL_cache_controls,+SPV_INTEL_bindless_images,+SPV_INTEL_task_sequence,+SPV_INTEL_int4,+SPV_EXT_float8,+SPV_INTEL_optnone,+SPV_INTEL_bfloat16_conversion,+SPV_INTEL_joint_matrix,+SPV_INTEL_hw_thread_queries,+SPV_KHR_uniform_group_instructions,+SPV_INTEL_masked_gather_scatter,+SPV_INTEL_tensor_float32_conversion,+SPV_KHR_non_semantic_info,+SPV_KHR_cooperative_matrix,+SPV_EXT_shader_atomic_float16_add,+SPV_INTEL_memory_access_aliasing"
    local -a cmd3=("$llvm_spirv" -o "$dst" -spirv-max-version=1.5
                   -spirv-debug-info-version=nonsemantic-shader-200
                   -spirv-allow-unknown-intrinsics=llvm.genx.
                   -spirv-allow-unknown-intrinsics=llvm.genx.
                   -spirv-ext="$ext" "$work/$mod")
    "${cmd3[@]}"
    compile_lines+=("$label llvm-spirv | $(printf '%s ' "${cmd3[@]}")")
    compile_lines+=("$label device image | $work/$mod, kernel names (.sym): ${names:-<none>}, .prop: ${esimd_prop:-<none>}")

    # 4. the gate the card taught: the VC backend keys on VectorComputeFunctionINTEL,
    #    and the image the SYCL runtime loads carries it. Zero means the module is
    #    not a VC image and the driver will list no kernel.
    local chk="$work/check.spvasm"
    "$llvm_spirv" --to-text "$dst" -o "$chk"
    local vc vccap
    vc="$(grep -c 'VectorComputeFunctionINTEL' "$chk" || true)"
    vccap="$(grep -c 'Capability VectorComputeINTEL' "$chk" || true)"
    vc_report+=("$label: VectorComputeFunctionINTEL decorations ${vc}, VectorComputeINTEL capability ${vccap}, isEsimdImage property ${esimd_prop:-<none>}, device-image kernel names: ${names:-<none>}, $(stat -c%s "$dst") bytes, sha256 $(sha256sum "$dst" | cut -d' ' -f1)")
    if [ "${vc:-0}" -lt 1 ] || [ "${vccap:-0}" -lt 1 ]; then
        echo "build-spirv: $label: the module is not a VC image (VectorComputeFunctionINTEL ${vc}, VectorComputeINTEL capability ${vccap})." >&2
        echo "  That is the shape the card rejected (no kernels listed). Do not ship it." >&2
        exit 5
    fi

    # the disassembly the ABI dump reads
    if [ -n "$spirv_dis" ]; then
        "$spirv_dis" "$dst" > "${dst%.spv}.spvasm"
    else
        cp "$chk" "${dst%.spv}.spvasm"
    fi
}

# ---------------------------------------------------------------------------
# Two modules, two arms:
#   precise: the oracle arm, -fp-model=precise -O2 (no reassociation, IEEE div).
#   ref:     trellis-serve's own compile flags, from the reference tree,
#            xpu/scripts/build_inkling2.sh:7-8:
#              icpx -fsycl -fsycl-targets=spir64 -O3 -ffast-math -fPIC
#              -std=c++17 -shared -fsycl-device-code-split=per_kernel
#              -D_GLIBCXX_USE_CXX11_ABI=1
#            The device-relevant part is -O3 -ffast-math -std=c++17
#            -fsycl-device-code-split=per_kernel; -fPIC/-shared/the ABI define
#            are host-object and link options.
# ---------------------------------------------------------------------------
build_module precise "$precise_spv" -fp-model=precise -O2
build_module ref "$ref_spv" -O3 -ffast-math -std=c++17 -fsycl-device-code-split=per_kernel

# ---------------------------------------------------------------------------
# The ABI dump, per module, read out of the SPIR-V itself (entry point name,
# execution model, the entry point's function type, each OpFunctionParameter in
# order, capabilities/extensions/execution modes). run_abi=1 for the precise
# module writes abi.txt and kernel-names.txt, which fa_ref_host checks against.
# ---------------------------------------------------------------------------
python3 "$here/abi-dump.py" "${precise_spv%.spv}.spvasm" "$precise_spv" "$out"
python3 "$here/abi-dump.py" "${ref_spv%.spv}.spvasm" "$ref_spv" "$out" \
    --abi-name abi.ref.txt --names-name kernel-names.ref.txt

if ! diff -q <(grep -E '^(entry_point|arg)' "$out/abi.txt") \
             <(grep -E '^(entry_point|arg)' "$out/abi.ref.txt") >/dev/null; then
    echo "build-spirv: WARNING: the two modules disagree on the entry point or the argument ABI" >&2
    diff "$out/abi.txt" "$out/abi.ref.txt" >&2 || true
fi

{
  printf '%s\n' "# the exact command lines build-spirv.sh ran"
  for l in "${compile_lines[@]}"; do printf '%s\n' "$l"; done
  printf 'disassembler: %s\n' "${spirv_dis:-llvm-spirv --to-text (spirv-dis not installed)}"
  for l in "${vc_report[@]}"; do printf 'check: %s\n' "$l"; done
} > "$out/commands.txt"

echo
echo "post-link pipeline, per module:"
for l in "${vc_report[@]}"; do echo "  $l"; done
echo
echo "artifacts in $out:"
ls -l "$out"
echo
for f in "$precise_spv" "$ref_spv"; do
    echo "sha256 $(sha256sum "$f" | cut -d' ' -f1)  $(basename "$f")  $(stat -c%s "$f") bytes"
done
