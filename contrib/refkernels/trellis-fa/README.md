# trellis-serve's prefill attention kernel, as a reference arm

Stage 1b of the FA port: run **trellis-serve's own ESIMD prefill-attention
kernel**, verbatim, on our card through our OpenCL path, to measure it and to
serve as the correctness oracle for the port. Nothing here hooks into arcint's
engine; this directory is a measurement arm and a yardstick.

Two modules are built from the one source, and the host runs both:

| module | build flags | arm |
|---|---|---|
| `out/fa_kern.precise.spv` | `-fp-model=precise -O2` | correctness: no reassociation, IEEE division |
| `out/fa_kern.ref.spv` | trellis-serve's own: `-O3 -ffast-math -std=c++17 -fsycl-device-code-split=per_kernel` | timing: what the reference actually ships |

Both are checked against the same CPU oracle. The timing module gets a stated
tolerance (§tolerance), so a fast-math difference is seen rather than assumed
away, and both modules' timings are reported.

Both are built along the **driver's own pipeline** for a spir64 ESIMD image —
device bitcode, `sycl-post-link -split-esimd -lower-esimd`, `llvm-spirv
-spirv-allow-unknown-intrinsics=llvm.genx.` — because the one-step
`-fsycl-device-obj=spirv` form stops before post-link and the card rejected it:
it built, then listed **no kernel** (§post-link). The precise module's output is
**byte-identical** (sha256) to the device image `icpx` itself dumps for the same
build.

## Source and licence

| | |
|---|---|
| repository | https://github.com/sybil-solutions/trellis-serve |
| commit | `8f8e2fccf0c82c9715b774812f30d9d52f78a5fb` |
| file | `xpu/csrc/inkling2_ops.sycl` |
| block | lines **509-654**, `namespace fa` (the ESIMD kernel `fa::Kern<128>`) |
| launch site | lines 655-671, `inkling2::prefill_attn` |
| caller | `xpu/exl3xpu/inkling2/attn_triton.py:658-670` (`attention_prefill_esimd`), reached from `xpu/exl3xpu/inkling2/sgl_model.py:415,427`; the brief's `vllm_model.py:166` is the same call in the vLLM-era tree |
| licence | MIT, Copyright (c) 2026 0xSero — the notice is in `fa_kern.sycl`'s header and in `THIRD_PARTY.md` |

`fa_kern.sycl` holds the block **byte-for-byte**; the only additions are the
include set the block needs (the reference's own `xpu/csrc/inkling2_ops.sycl:12-16`)
and the wrapper afterwards. The check, run 2026-11-24:

    sed -n '45,190p' fa_kern.sycl > <tmp>/a
    sed -n '509,654p' <ref>/xpu/csrc/inkling2_ops.sycl > <tmp>/b
    diff <tmp>/a <tmp>/b    # identical

`fa_ref::launch` is the reference's launch (lines 655-671) with the torch tensor
calls replaced by the pointers those tensors' `data_ptr()` returned; the
initializer order, `qtiles`/`qgroups`/`n`/`wg` arithmetic and
`scale_log2 = LOG2E / sqrt(D)` are the reference's, unchanged.

## What each file is

| file | what |
|---|---|
| `fa_kern.sycl` | the vendored kernel (verbatim) + `fa_ref::launch` + MIT header |
| `build-spirv.sh` | `bash build-spirv.sh <outdir>`: icpx → device bitcode → `sycl-post-link --split-esimd` → `llvm-spirv`, for both arms; then the kernel name, the argument ABI, the extensions and the VectorCompute-decoration gate |
| `abi-dump.py` | the ABI/kernel-name reader `build-spirv.sh` calls; it reads the compiled SPIR-V, not this repository's claims |
| `fa_ref_host.cpp` | standalone OpenCL 3.0 host: `clCreateProgramWithIL` (or `…WithBinary` for an AOT fallback), build, the shapes, CPU oracle, max abs error / per-row KL per module, timing per module, `--dry` |
| `binding-notes.md` | (no code) what our side must feed this kernel from a `GGML_OP_FLASH_ATTN_EXT` call, with the cost of each item |
| `opencl-plan.md` | (no code) 1a's plan: `fa::Kern`'s tile structure in OpenCL C, each ESIMD construct against the builtin ours already uses, what has no equivalent, and the binding plan |
| `fa_port_gate.h` | a copy of the port's pure gate (the llama tree's `ggml/src/ggml-opencl/fa_port_gate.h`, patch 0198), so the test beside it builds from this directory |
| `port_gate_test.cpp` | the gate's CPU unit test, fed the model's graph export lines; red with `-DFA_PORT_GATE_RED` |
| `port_abi_check.py` | the kernel's parameter list against the launch's `clSetKernelArg` sequence, no device |
| `port_kernel_model.py` | the kernel's key-block loop a lane at a time against a direct softmax, with three reds, no device |
| `port_arithmetic_check.py` | the first session's check of the bias run read (still the kernel's) and of a sink epilogue the kernel no longer has (the model above covers the sink) |
| `out/` | build products; git-ignored |

## Build

    bash build-spirv.sh out

`icpx` is found on `PATH`, then `$ONEAPI_ROOT`, then `/opt/intel/oneapi`
(`source /opt/intel/oneapi/setvars.sh` first if it is not on `PATH`). It needs
nothing else: **no GPU, no Level Zero, no torch**. It writes
`out/fa_kern.precise.spv` and `out/fa_kern.ref.spv`, plus `abi.txt`,
`abi.ref.txt`, the kernel-name files, the disassemblies, the per-module
post-link trees `out/postlink-<label>/` (device bitcode, split table, the ESIMD
image's `.sym`) and `commands.txt` (the exact command lines it ran, with the
VectorCompute-decoration check). The host program is separate and links only
OpenCL:

    g++ -std=c++17 -O2 -o fa_ref_host fa_ref_host.cpp -lOpenCL

## The pipeline, and what each flag is for

The reference's own build of this source (`code`), for the flags the timing arm
uses — `xpu/scripts/build_inkling2.sh:7-8`:

    icpx -fsycl -fsycl-targets=spir64 -O3 -ffast-math -fPIC -std=c++17 -shared \
      -fsycl-device-code-split=per_kernel -D_GLIBCXX_USE_CXX11_ABI=1 ${INKLING2_FLAGS:-} \
      ... -x c++ csrc/inkling2_ops.sycl -x none -o exl3xpu/_inkling2.so

so the device-relevant part is `-O3 -ffast-math -fsycl-device-code-split=per_kernel`
(`-std=c++17`) — `-fPIC`, `-shared`, the ABI define and the link line are not.
It sets no `-fp-model`, i.e. the timing arm is in icpx's fast FP model, spelled
`-ffast-math` as the reference spells it.

`build-spirv.sh` runs three steps per module, and each is here with its reason:

| step | command | why |
|---|---|---|
| device bitcode | `icpx -fsycl -fsycl-device-only -fsycl-targets=spir64 <arm flags> -c -x c++ fa_kern.sycl -o dev.bc` | `-fsycl` for the SYCL/ESIMD source, `-fsycl-device-only` for a device image with no host object and no loader, `-fsycl-targets=spir64` for generic SPIR-V (the reference's target, and what `clCreateProgramWithIL` consumes), `-x c++` because `.sycl` is not a source extension icpx knows (without it: `'linker' input unused`), `-c` for no link. Arm flags: precise `-fp-model=precise -O2`; reference `-O3 -ffast-math -std=c++17 -fsycl-device-code-split=per_kernel` |
| ESIMD device image | `sycl-post-link -O2 -device-globals --device-lib-dir=<oneAPI lib> -properties -spec-const=native -split=auto -emit-only-kernels-as-entry-points -emit-param-info -symbols -emit-exported-symbols -emit-imported-symbols -split-esimd -lower-esimd -o table.txt dev.bc` | the driver's own option set (`icpx -###` on the link step, 2026.1.1). `-split-esimd` separates the ESIMD entry points; `-lower-esimd` lowers ESIMD to the VC IR (`llvm.genx.*`) and is where the **`VectorComputeFunctionINTEL` decorations** the VC backend keys on are attached; `-emit-param-info` carries the kernel-param ABI, `-properties` writes the `.prop` with `isEsimdImage=1`; `-emit-imported-symbols` keeps the VC imports |
| SPIR-V module | `llvm-spirv -o out.spv -spirv-max-version=1.5 -spirv-debug-info-version=nonsemantic-shader-200 -spirv-allow-unknown-intrinsics=llvm.genx. -spirv-allow-unknown-intrinsics=llvm.genx. -spirv-ext=<the driver's list> table_esimd_0.bc` | same link step: `-spirv-allow-unknown-intrinsics=llvm.genx.` turns each lowered VC intrinsic into an `Import` declaration (the whole module in `Import`s: 70 `llvm.genx.` names) instead of the hard error `InvalidFunctionCall: Unexpected llvm intrinsic: llvm.genx.rdregionf.v16f16.v128f16.i16` (measured without it); `-spirv-max-version=1.5` and the explicit `-spirv-ext=-all,+…` list (which includes `SPV_INTEL_vector_compute`, `SPV_INTEL_joint_matrix`, `SPV_INTEL_media_block_io`) are the driver's |

Flags and their reasons that are still worth naming:

| flag | why |
|---|---|
| `-fp-model=precise` (precise module only) | icpx defaults to the fast FP model (reciprocal approximations, reassociation; Strata's SYCL build records it, `src/sycl/CMakeLists.txt:94-97`). An oracle must not reassociate the softmax arithmetic |
| `-O2` (precise module only) | a release-grade build without the fast-math model |
| **not** `-fno-sycl-esimd-force-stateless-mem` | it is not the default (oneAPI UsersManual, `-f[no-]sycl-esimd-force-stateless-mem`), the reference does not force it, and the kernel takes raw pointers, no accessors |
| **not** `-cl-fast-relaxed-math -cl-finite-math-only -cl-mad-enable` | llama.cpp's OpenCL build options (`ggml-opencl.cpp:1998-2000`); they are build options, not icpx flags, and they stay out of the oracle arm |
| **not** `-fsycl-device-obj=spirv` | measured dead end, §post-link: it emits the device code before post-link |

### The one-step form, and the pipeline that replaces it {#post-link}

`-fsycl-device-obj=spirv` (or the bitcode + `llvm-spirv` pair) produced, at the
first attempt, these modules:

| module | bytes | sha256 |
|---|---|---|
| precise (one-step) | 98,364 | `3700abe92a6bfe5445d5ebe847038f7b26dfe2f35d5d03c28e73c8d49d125688` |
| reference-flags (one-step) | 98,704 | `358115286ebf24db6a959aba43ae5aca13992b8935a558324f30b7ad06c46bfb` |

and their disassembly carries **no `VectorComputeFunctionINTEL` decoration at
all** — only the `VectorAnyINTEL` capability, the ESIMD intrinsics as
`__esimd_*` imported functions, execution modes `ContractionOff` and
`SubgroupSize 1`, one entry point `_ZTSN2fa4KernILi128EEE`, and the 17 flat
arguments below.

**Card result, measured on a B60 (driver 26.27.39122.11, 2026-11-24):**
`clCreateProgramWithIL` + `clBuildProgram` with
`-vc-codegen -cl-fp32-correctly-rounded-divide-sqrt -ze-opt-greater-than-4GB-buffer-required`
**built**, with only `warning: Linking two modules of different target triples:
''`, but `CL_PROGRAM_KERNEL_NAMES` was **empty** and `clCreateKernel("_ZTSN2fa4KernILi128EEE")`
returned **`CL_INVALID_KERNEL_NAME` (-46)**. No enqueue happened, no fault; the
host program, the bounds check and the oracle were right (checksum 494.011836).

**Cause, now measured rather than inferred.** The one-step form stops before
`sycl-post-link`. The driver's own link step (`icpx -###`, 2026.1.1) runs
`sycl-post-link … -split-esimd -lower-esimd` and then `llvm-spirv
-spirv-allow-unknown-intrinsics=llvm.genx. …`. Running those two steps on the
same source reproduces the image `icpx` itself embeds, **byte for byte**:

    icpx's own dump:  <tmp>/…/host_main-…-7997a3.spv   28,320 B  sha256 fd8af4a93e08f59941f3e5e2cb334803382b8f1977b6f0e179549b985c2360ab
    build-spirv.sh:   out/fa_kern.precise.spv          28,320 B  sha256 fd8af4a93e08f59941f3e5e2cb334803382b8f1977b6f0e179549b985c2360ab

(the dump is `icpx -fsycl -fsycl-targets=spir64 -fp-model=precise -O2
-fsycl-dump-device-code=<dir> … -o prog`, deprecated in favour of
`-save-offload-code=<dir>`, measured here.) What the module now contains:

- `Capability VectorComputeINTEL` **and** `VectorAnyINTEL`, plus
  `Addresses, Linkage, Kernel, Vector16, Float16Buffer, Int64, Int16,
  GenericPointer, Int8, SubgroupDispatch, MemoryAccessAliasingINTEL`;
  extensions `SPV_INTEL_vector_compute`, `SPV_INTEL_memory_access_aliasing`,
  `SPV_KHR_expect_assume`, `SPV_KHR_linkonce_odr`;
- **36 `VectorComputeFunctionINTEL` decorations** on the entry point and every
  ESIMD helper function;
- 70 `llvm.genx.*` names as `Import` declarations (`Decorate … LinkageAttributes
  "llvm.genx.rdregionf.v16f16.v128f16.i16" Import` and siblings) — the VC IR,
  linked by IGC at compile time;
- execution modes `ContractionOff`, `SubgroupSize 1`, and two INTEL modes
  `5618 0` and `6417 0` (values as printed; the mode names are INTEL extension
  enumerants and llvm-spirv's text form prints them numerically);
- entry point `_ZTSN2fa4KernILi128EEE`, the same 17 flat arguments
  (unchanged ABI: `q k v o table sink` pointers 8 B with distinct pointee types
  now, then the 10 ints and the float, 92 B, no implicit argument);
- and its `.prop` says `isEsimdImage=1`, the same property the SYCL runtime
  reads to add `-vc-codegen` itself (0xSero/exl3xpu issue #1 §2).

`build-spirv.sh` gates on that decoration and capability, so a module without
them is never shipped:

    build-spirv: <label>: the module is not a VC image (VectorComputeFunctionINTEL <n>, VectorComputeINTEL capability <n>).

The gate's first run passed for both arms (§verified).

### The kernel-name question {#kernel-name}

Card result, measured (B60, driver 26.27.39122.11, 2026-11-24), second attempt
with the post-link modules, both sha256 matching `build-spirv.sh`'s output:

```
build log: warning: Linking two modules of different target triples: '' is 'spir64'
           whereas 'Deserialized LLVM Module' is 'genx64-unknown-unknown'
CL_PROGRAM_KERNEL_NAMES: _ZTSN2fa4KernILi128EEE
clCreateKernel("_ZTSN2fa4KernILi128EEE") in precise -> CL_INVALID_KERNEL_NAME (-46)
```

So the module is listed but not creatable. Two things can look identical from
`clCreateKernel` alone — a byte-level name mismatch (trailing NUL, `;`,
whitespace) and a kernel the driver lists but will not create — and
`fa_ref_host` now separates them in one run:

- `CL_PROGRAM_KERNEL_NAMES` is dumped **byte-exactly**: its size, a hex dump, and
  each `;`-separated token with its own length and hex; the name handed to
  `clCreateKernel` is dumped the same way (length + hex), so a trailing NUL or a
  space is visible instead of being trimmed by stdio;
- if `clCreateKernel` fails, `clCreateKernelsInProgram` is called for the count
  and then for the list, and **each** created kernel prints its own
  `CL_KERNEL_FUNCTION_NAME` (length + hex), `CL_KERNEL_NUM_ARGS` and
  `CL_KERNEL_ATTRIBUTES`; the one whose name matches is then used;
- if `clCreateKernelsInProgram` creates nothing, the host says so and names the
  AOT route as the next step. `CL_KERNEL_FUNCTION_NAME` of the kernel that was
  created is dumped too.

**What the code and the docs say (`code`, plus `inferred` for the mechanism).**
If the names match byte for byte, the difference is not the module — it is
byte-identical to `icpx`'s own device image (§post-link) — it is the *creation
path*:

- NEO's OpenCL program takes its kernel list from the **compiled binary's**
  `programInfo.kernelInfos`, not from the SPIR-V entry points: `Program` builds
  from `decodedSingleDeviceBinary.programInfo.kernelInfos`
  (compute-runtime `opencl/source/program/program.cpp:207-212`), and a kernel
  whose `heapInfo.kernelHeapSize` is 0 is filtered out of the ISA setup
  (`program.cpp:355-362`, `DEBUG_BREAK_IF(validKernelInfos.size() !=
  kernelInfoArray.size())`). The names OpenCL reports come from the module's
  entry points, so a VC-codegen'd ESIMD entry can be listed and still have no
  OpenCL kernel-heap entry to create: `CL_INVALID_KERNEL_NAME` with the name
  listed. That is `inferred` from those two code paths, not stepped through.
- The SYCL runtime does not take the OpenCL path on an Intel GPU: it goes
  through Unified Runtime's Level Zero adapter, which creates kernels with
  `zeKernelCreate` (`oneapi-src/unified-runtime`,
  `source/adapters/level_zero/kernel.cpp:574-607`), on the same module with
  `-vc-codegen` added from the image property `isEsimdImage`
  (`intel/llvm`, `sycl/source/detail/program_manager/program_manager.cpp:321-327`);
  the ESIMD extension's own README states the runtime "automatically
  recognizes ESIMD kernels and dispatches their execution", "including OpenCL
  and Level Zero backends". So the image is built for a VC kernel object, and
  our loader asks the OpenCL path for one.
- The runtime's option string differs from ours: it adds `-vc-codegen` from the
  property, GRF options from `sycl-grf-size` (`-doubleGRF` for an ESIMD image,
  `-ze-opt-large-register-file` for SPMD, `program_manager.cpp:274-288`),
  `-disable-finalizer-msg` when the warning level is 0 (323-329), and for an
  ESIMD image it deliberately passes **no** `-O` level: "Passing these options
  to vector compiler causes build failure in backend" (`345-346`). Our options
  `-cl-fp32-correctly-rounded-divide-sqrt` and
  `-ze-opt-greater-than-4GB-buffer-required` are additions of ours. This is
  where a difference could also live, and the `--dry` run's build log would show
  it if IGC complains.

**Two routes forward, and the next card command for each** (no card work is
queued for this arm now; these are for the operator):

1. **Diagnosis, OpenCL, IL** — one case, no shapes:
   `./fa_ref_host --spv out/fa_kern.precise.spv --abi out/abi.txt --dry`
   It prints the byte dumps and the `clCreateKernelsInProgram` enumeration, then
   either creates the kernel (and the case could be measured) or reports that
   nothing is creatable. This is the run that answers the question with bytes.
2. **AOT binary, OpenCL** (§AOT-fallback, ocloc on the card host):
   ```
   ocloc compile -device bmg-g21 \
         -input out/fa_kern.precise.spv -output out/fa_kern.bmg_g21.bin \
         -options "-vc-codegen -cl-fp32-correctly-rounded-divide-sqrt -ze-opt-greater-than-4GB-buffer-required"
   ./fa_ref_host --spv out/fa_kern.bmg_g21.bin --abi out/abi.txt --dry
   ./fa_ref_host --spv out/fa_kern.bmg_g21.bin --abi out/abi.txt --reps 20
   ```
   `clCreateProgramWithBinary` is the path where NEO's kernel list does come
   from the binary's own metadata, which is why it may succeed where the IL path
   fails (`inferred`, not measured — no ocloc on this host).
3. **Level Zero** is the runtime's path and would need `libze_loader` and
   `zeModuleCreate`/`zeKernelCreate` in the host; **not** available on this
   recipe host (no Level Zero package among the 23, §Toolchain), and it is not
   what this arm's brief asked for. Named, not built.

### Flags older oneAPI docs name, and what 2026.1.1 does with them

Measured here, both accepted-but-not-what-they-say:

- `-fsycl-explicit-simd` (the pre-2023 way to enable ESIMD) —
  `warning: option '-fsycl-explicit-simd' is deprecated and will be removed in
  a future release`, then `the flag '-fsycl-explicit-simd' has been deprecated
  and will be ignored`. ESIMD is on by default in 2026.1; the flag compiles
  nothing differently (rc=0).
- `-fno-sycl-use-bitcode` (the recipe in intel/llvm discussion #5608) —
  `warning: option '-fno-sycl-use-bitcode' is deprecated and will be removed in
  a future release, use '-fsycl-device-obj=spirv' instead`. It still produced a
  valid 98,768-byte module here — the same pre-post-link shape §post-link
  rejects.
- `-fno-sycl-esimd-force-stateless-mem` is accepted (rc=0) and is not needed:
  the kernel takes raw pointers, no accessors, so it is stateless already.

An older recipe, kept only as a measured dead end: device bitcode + `llvm-spirv
--spirv-ext=+all` produced a 99,240 B module carrying `SPV_KHR_untyped_pointers`
(pointers untyped, ABI unchanged: 8 B, still a `cl_mem` at `clSetKernelArg`) —
i.e. the same pre-post-link shape, without `VectorComputeFunctionINTEL`. Also
measured: `llvm-spirv` without any `--spirv-ext` dies on this kernel with
`LLVM ERROR: Invalid vector size for fixup: 1` (pass
`SPIRVLowerBitCastToNonStandardTypePass`), because the pre-lowering input uses
`SPV_INTEL_vector_compute`. Neither is what `build-spirv.sh` does.

## Tools used, and what they printed

- `llvm-spirv` and `sycl-post-link` — **ship with oneAPI**, but not in the
  compiler's public bin dir. In 2026.1 both are in
  `/opt/intel/oneapi/compiler/2026.1/bin/compiler/` (one level below `icpx`'s
  `bin/`). `build-spirv.sh` finds them there.
- `spirv-dis` / `spirv-val` — **not shipped by oneAPI 2026.1** (searched the
  whole install; no `SPIRV-Tools` binary). `build-spirv.sh` uses `spirv-dis` when
  the host has SPIRV-Tools and `llvm-spirv --to-text` otherwise; the latter is
  what ran, and its own assembly form (no `Op` prefix, a word-count column) is
  what `abi-dump.py` reads — it accepts both forms.
- Round-trip check of a module: `llvm-spirv -r` back to bitcode succeeds
  (44,656 B of bitcode for the one-step module), the only structural validation
  available without `spirv-val`.
- `llvm-spirv` **without** `-spirv-allow-unknown-intrinsics=llvm.genx.` refuses
  the post-link VC IR: `InvalidFunctionCall: Unexpected llvm intrinsic:
  llvm.genx.rdregionf.v16f16.v128f16.i16` (measured 2026-11-24). With the flag
  every VC intrinsic becomes an `Import` declaration and the module translates.

`build-spirv.sh out` printed (2026-11-24, this host), both modules:

    precise:  28,320 bytes, sha256 fd8af4a93e08f59941f3e5e2cb334803382b8f1977b6f0e179549b985c2360ab
    ref:      28,672 bytes, sha256 31865371a934e259e96de165e1decf7925bd4862a2ca2716828d8830ea416eba

    entry point         "_ZTSN2fa4KernILi128EEE"
    execution model     Kernel (6)
    interface variables 1: "__spirv_BuiltInGlobalInvocationId"

    capabilities (both): Addresses, Linkage, Kernel, Vector16, Float16Buffer,
                  Int64, Int16, GenericPointer, Int8, SubgroupDispatch,
                  VectorComputeINTEL, VectorAnyINTEL, MemoryAccessAliasingINTEL
    extensions (both):  SPV_INTEL_memory_access_aliasing, SPV_INTEL_vector_compute,
                  SPV_KHR_expect_assume, SPV_KHR_linkonce_odr
    execution modes: ContractionOff, SubgroupSize = 1, 5618 0, 6417 0
    decorations:      36 × VectorComputeFunctionINTEL (entry point + every ESIMD helper)
    imports:          70 × llvm.genx.* (LinkageAttributes … Import)

For comparison, the **one-step modules the card rejected** (§post-link), kept
as the evidence for the pipeline change:

    precise:  98,364 bytes, sha256 3700abe92a6bfe5445d5ebe847038f7b26dfe2f35d5d03c28e73c8d49d125688
    ref:      98,704 bytes, sha256 358115286ebf24db6a959aba43ae5aca13992b8935a558324f30b7ad06c46bfb

and neither declared `VectorComputeFunctionINTEL`, `VectorComputeINTEL`, or any
`llvm.genx.*` name; it also declared neither `FPFastMathModeINTEL` nor
`SPV_INTEL_fp_fast_math_mode`, and the post-link modules do not either.

The **kernel name** is `_ZTSN2fa4KernILi128EEE` — DPC++'s mangled *type* name for
the unnamed SYCL kernel `fa::Kern<128>` — read from the device image's `.sym`
(post-link) and from the entry point (SPIR-V); both modules agree.
`out/kernel-names.txt` holds the same string (and `kernel-names.ref.txt` for the
other module); `fa_ref_host` reads it from `out/abi.txt` and lists the live
`CL_PROGRAM_KERNEL_NAMES` on the card to confirm the two agree. The two modules'
ABI dumps are identical (entry point and all 17 argument types), compared by
`build-spirv.sh`.

### The argument ABI, in order

17 arguments, one per functor field (the `sycl::nd_item<1> it` parameter is not
an argument; it is the builtin). Read out of the module's
`OpFunctionParameter`s — not asserted by hand:

| # | type | bytes | `fa::Kern<128>` field |
|---|---|---|---|
| 0 | ptr (`CrossWorkgroup`) | 8 | `q` `const fp16*` |
| 1 | ptr | 8 | `k` `const fp16*` |
| 2 | ptr | 8 | `v` `const fp16*` |
| 3 | ptr | 8 | `o` `float*` |
| 4 | ptr | 8 | `table` `const uint16_t*` |
| 5 | ptr | 8 | `sink` `const float*` |
| 6 | int (i32) | 4 | `Lq` |
| 7 | int | 4 | `Lk` |
| 8 | int | 4 | `Hq` |
| 9 | int | 4 | `Hk` |
| 10 | int | 4 | `G` (= `Hq / Hk`; the reference passes it, it is not derived in the kernel) |
| 11 | int | 4 | `qpos0` |
| 12 | int | 4 | `kpos0` |
| 13 | int | 4 | `window` |
| 14 | int | 4 | `E` (the table's bin count; 0 with no table) |
| 15 | int | 4 | `QTW` |
| 16 | float (f32) | 4 | `scale_log2` (= `LOG2E / sqrt(D)`) |

Total: 6×8 + 10×4 + 4 = **92 B**, flat — no argument struct, no padding.
**Implicit arguments: none** beyond `__spirv_BuiltInGlobalInvocationId`; the
module declares no local-size, global-offset, printf or work-group-count
builtin. The six pointers are `CrossWorkgroup`, so the host passes `cl_mem`
objects, not `cl_mem` + offset.

## Build options the OpenCL host uses, and where each comes from

`fa_ref_host` builds an IL program with

    -vc-codegen -cl-fp32-correctly-rounded-divide-sqrt -ze-opt-greater-than-4GB-buffer-required

| option | source |
|---|---|
| `-vc-codegen` | **required for an ESIMD image**. The SYCL runtime adds it itself for images tagged `isEsimdImage` (intel/llvm `program_manager.cpp`, `appendCompileOptionsFromImage`), and 0xSero/exl3xpu issue #1 states it: "JIT works because the SYCL runtime adds -vc-codegen for images marked isEsimdImage". Without it IGC has no ESIMD codegen path for this module |
| `-cl-fp32-correctly-rounded-divide-sqrt` | the kernel divides `1.0f / l`; Intel's fp32 divide is 2.5 ulp unless told otherwise. Strata measured a quantiser going byte-non-exact without it (`src/sycl/CMakeLists.txt:69-72`). Both arms get it: the comparison then isolates the compile flags, not the driver's rounding choice |
| `-ze-opt-greater-than-4GB-buffer-required` | keeps pointer arguments on 64-bit addresses; without it IGC turns pointers into 32-bit surface offsets and misreads beyond 4 GiB (Strata's measured defect, `src/sycl/CMakeLists.txt:79-85`). This case's buffers are all under 4 GiB, a served binding is not |

The options are the **same for both modules**; a precompiled binary (the AOT
fallback) gets none — the implementation rejects build options for binaries.

Not used, deliberately: `-cl-fast-relaxed-math`, `-cl-finite-math-only`,
`-cl-mad-enable` (llama.cpp's `ggml-opencl.cpp:1998-2000`), and
`-cl-intel-256-GRF-per-thread` (llama.cpp adds it per DPAS kernel,
`ggml-opencl.cpp:2741`). The last is a *performance* knob that changes register
allocation, not numerics; if the card's IGC wants it for this kernel, add it by
hand for the timing arm and keep the oracle arm without it.

## Tolerance for the reference-flags module {#tolerance}

The timing module is compiled with `-ffast-math`, so its arithmetic is allowed
to differ from the oracle's. It is still compared, and the stated bar is:

| measure | tolerance | why |
|---|---|---|
| max abs error | ≤ **5e-3** | `-ffast-math` permits reciprocal approximation (2.5 ulp on the `1.0f/l` divide), reassociation and flush-to-zero. The kernel's output magnitudes are order 0.1-1, and fp16 inputs already carry ~1e-3 relative quantization, so 5e-3 is above the compile-flag effect and below a real defect |
| mean row KL | ≤ **5e-4** nats | same reasoning on the row's distribution |
| NaN count | **0** | fast-math must not turn a finite output into a NaN |

Both are `inferred` here — no card was available — and can be overridden with
`--ref-tol-abs` / `--ref-tol-kl`. Exceeding them is printed as `EXCEEDED` and a
warning, never converted into a pass: then the timing module's numbers are a
separate arm, not a comparison.

## Toolchain

What the recipe host has installed for this directory, and nothing else that
this work needs:

- **Intel oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)**, from Intel's
  apt repository, as these packages and versions:

      intel-oneapi-compiler-dpcpp-cpp-2026.1                 2026.1.1-325
      intel-oneapi-compiler-dpcpp-cpp-runtime-2026.1         2026.1.1-325
      intel-oneapi-compiler-shared-2026.1                    2026.1.1-325
      intel-oneapi-compiler-shared-runtime-2026.1            2026.1.1-325
      intel-oneapi-openmp-2026.1                             2026.1.1-325
      intel-oneapi-tbb-2023.1                                2023.1.0-151
      intel-oneapi-umf-1.1                                  1.1.0-340
      intel-oneapi-tcm-1.5                                  1.5.0-489
      intel-oneapi-dev-utilities-2026.0                      2026.0.1-16
      intel-oneapi-libdpstd-devel-2022.13                    2022.13.0-107
      intel-oneapi-dpcpp-debugger-2026.1                     2026.1.0-70
      intel-oneapi-common / intel-oneapi-licensing / intel-oneapi-vars   2026.0.0-235

- **23 packages, about 1.4 GB.** **No OpenCL / IGC / Level Zero package is among
  them**: the SPIR-V is produced, but nothing here can build or run a device
  image. `ocloc`, the OpenCL ICD loader and the compute runtime are absent, so
  the AOT fallback (§AOT-fallback) is a card-host step, not this one's.
- **The recipe does not call icpx.** arcint's build and its package do not
  depend on it: this directory is a reference arm, and `build-spirv.sh` is run by
  the operator when a measurement is wanted. `icpx` is only needed for that.
- **The SPIR-V is built from this source and not checked in.** `out/` is
  git-ignored; the modules are reproducible from `fa_kern.sycl` with the command
  above (the two sha256 in §tools are what this toolchain produced).

## AOT fallback for the reference arm {#AOT-fallback}

If the card's driver refuses the SPIR-V module, the reference arm may be built
ahead of time **on the card host, from the checked-in source**, and loaded with
`clCreateProgramWithBinary`. This is a fallback for the reference arm only: it
is never the default, and **the binary is never committed** — it is a
card-and-driver-specific artefact, `out/` is git-ignored and so is `*.bin` /
`*.zebin`.

**Device name for an Arc Pro B60: Battlemage BMG-G21.** Two spellings, both from
documentation:

- DPC++ AOT target list (oneAPI DPC++ UsersManual, "Ahead of Time
  Compilation", the `intel_gpu_*` values): `intel_gpu_bmg_g21, intel_gpu_20_1_4
  - Battlemage G21 Intel graphics architecture` (`intel_gpu_bmg_g31` is G31,
  the B70's part).
- ocloc/`-Xsycl-target-backend` spelling: `bmg-g21` — the form Strata passes for
  the B70, `src/sycl/CMakeLists.txt:66-68` (`-device bmg-g31`), and the form
  0xSero/exl3xpu issue #1 describes ocloc being called with
  (`ocloc -spirv_input -device bmg-g31`). **Do not use `-device bmg`**: the same
  issue records that `bmg` resolves to product config 20.1.0 (B580-class) and
  that NEO requires an exact product-config match, so a `bmg` binary would not
  load on the target card.

**A measured correction first.** The command that suggests itself — `icpx
-fsycl -fsycl-device-only -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen
"-device bmg-g21 …" -c` — does **not** produce a native binary on this host. It
produced **LLVM IR bitcode**: 48,392 B, sha256
`5575fc9e593dbae5830fd3b3685a34f49409a852654b58c058de531bcefe450e`, `file`
reports `LLVM IR bitcode`, and an invalid device string (`-device bmg-g99`) was
accepted with rc=0 and no diagnostic — so the device is not resolved at that
step. `-fsycl-device-only` stops before the step that would hand the module to
the AOT compiler. Do not use that command.

The AOT step the reference describes is **ocloc consuming the SPIR-V image**
(issue #1 §2: "icpx sends every image to ocloc with the same options
(`llvm-foreach … ocloc -spirv_input -device bmg-g31, no -options`)"), and for an
ESIMD image `-vc-codegen` must be in its options: the same issue's root cause is
that without it "IGC's VLD path splits each SPIR-V module into SPMD and ESIMD
parts" and rejects a module whose helper lacks the VC decoration; passing
`-vc-codegen` "does the same and is correct" (issue #1 lines 86-98).

`ocloc` **is installed on the card host** (`/usr/bin/ocloc`, ocloc-26.27.1 from
the compute runtime 26.27, `libocloc.so` present), so the route is executable
there. Check it with:

    command -v ocloc && ocloc --version        # expect ocloc-26.27.1 or newer
    ldconfig -p | grep -F libocloc             # the IGC/ocloc library it needs

The command, on the card host, from the checked-in source (nothing committed;
`out/` and `*.bin` are ignored, and the file is named in the report):

    ocloc compile -device bmg-g21 \
          -input out/fa_kern.precise.spv -output out/fa_kern.bmg_g21.bin \
          -options "-vc-codegen -cl-fp32-correctly-rounded-divide-sqrt -ze-opt-greater-than-4GB-buffer-required"

`compile` with `-device -input -output -options` is the documented ocloc CLI
(`paper`: Intel's "Graphics Offline Compiler for OpenCL Code"); ocloc's internal
call form, as the driver makes it, is the same with a bare `-spirv_input` flag
instead of the `compile` subcommand (`code`: issue #1 §2). `-device bmg-g21` is
the B60's BMG-G21 spelling (`-device bmg-g31` for a B70; do **not** use
`-device bmg`, which resolves to the B580-class config). `-vc-codegen` first in
the `-options` string is the ESIMD requirement; the other two are the same
driver options the IL route uses (§build-options), so the two artefacts differ
only in delivery.

`fa_ref_host` loads anything whose path does **not** end in `.spv` with
`clCreateProgramWithBinary`, passes no build options (the implementation rejects
them for binaries) and reports `CL_PROGRAM_BINARY_TYPE`:

    ./fa_ref_host --spv out/fa_kern.bmg_g21.bin --abi out/abi.txt --shape all

Both routes are the same kernel and the same ABI dump, so comparison and timing
stay comparable; what changes is which artefact the driver saw.

## Shapes

`--shape` selects the comparison shapes, all with Hq 24, Hk 8, D 128, window
1,024, E 1,024, sinks, a bf16 table, and contiguous positions with `kpos0 = 0`,
`qpos0 = Lk - Lq` (each chunk's queries at the end of its keys). The default is
the stage-1b case; `--shape all` runs the other two as well:

| shape | Lq | Lk | qpos0 | launch: global / work-group |
|---|---|---|---|---|
| `q512xkv1536` (default) | 512 | 1,536 | 1,024 | 1,536 / 3 |
| `q2048xkv3072` | 2,048 | 3,072 | 1,024 | 6,144 / 3 |
| `q2048xkv2048` | 2,048 | 2,048 | 0 | 6,144 / 3 |

The global size is the reference's own `qgroups·qtw·Hq`, the work-group is
`G·qtw` (`G = Hq/Hk = 3`, `qtw = 1`), so the three shapes are one launch each
with the same geometry per head. For each shape, and for each module:

- the host-side **bounds check**, before any enqueue;
- the **oracle comparison** — max abs error, max row relative L2, mean row KL;
- the **median, min and max** of `--reps` (default 20) timed launches, device
  time from `clGetEventProfilingInfo`, plus TFLOP/s over the pairs the shape
  actually multiplies.

The oracle prints its own wall time. Measured here with plain `g++ -O2` on one
core (2026-11-24), which is why it is affordable to run it before the card:

| shape | oracle wall time | MAC pairs it does (2 flops each) |
|---|---|---|
| `q2048xkv3072` | 6,584 ms | 2048·24·1024·128 = 6.44e9 |
| `q2048xkv2048` | 4,855 ms | 2048·24·1024·128 = 6.44e9 |
| `q512xkv1536` | 1,798 ms | 512·24·1024·128 = 1.61e9 |

(The window, not Lk, sets the work: a row has at most 1,024 valid keys in every
shape, so `q2048xkv3072` and `q2048xkv2048` do the same 6.44e9 MAC pairs — the
earlier version decoded f16 per element per row and took 94.5 s / 70.8 s /
30.3 s; decoding once up front cut that ~14× with the identical checksums,
494.011836 / 2834.350616 / −117.452736.)

Do **not** build the host with `-march=native`: it enables FMA, which shifts the
oracle's dot-product rounding — measured here as checksum 494.021848 against
494.011836 for the same fixture. The oracle must be host-independent.

## What is verified here, and what needs the card host

Verified on this host (no GPU, no Level Zero device), 2026-11-24:

1. the block is byte-identical to the reference (`diff`, above);
2. icpx 2026.1.1 compiles it by the **post-link pipeline** for both arms, and
   the Gate passes for both (§post-link): 36 `VectorComputeFunctionINTEL`
   decorations and `VectorComputeINTEL` in each module, `.prop`
   `isEsimdImage=1`, the device-image kernel `_ZTSN2fa4KernILi128EEE`;
3. the precise module is **byte-identical to the device image `icpx` itself
   embeds** for the same build: 28,320 B, sha256
   `fd8af4a93e08f59941f3e5e2cb334803382b8f1977b6f0e179549b985c2360ab` on both
   sides. The ref arm: 28,672 B, sha256
   `31865371a934e259e96de165e1decf7925bd4862a2ca2716828d8830ea416eba`;
4. the ABI dump for both modules: 17 flat arguments, same order and types,
   identical between the arms, no implicit arguments, kernel name
   `_ZTSN2fa4KernILi128EEE`;
5. the card result that motivated the pipeline change (§post-link): the
   one-step module builds with `-vc-codegen`, but exposes no kernel (empty
   `CL_PROGRAM_KERNEL_NAMES`, `CL_INVALID_KERNEL_NAME` on its entry point);
6. `llvm-spirv -r` round-trips a module;
7. `fa_ref_host` builds with g++ and `-lOpenCL` alone;
8. `fa_ref_host --dry --shape all` runs to the platform query: each shape's
   case, launch geometry, **bounds table** (every maximal byte index inside its
   allocation), fixture, oracle + its time, and the ABI check; then it stops at
   `clGetPlatformIDs` with `-1001` (no vendor ICD here) — the documented `--dry`
   limit on this host. The per-shape, per-module `[dry]` launch lines are printed
   after the platform step, so they are seen on a host with a device;
9. the **oracle is cross-checked against an independent implementation**: with
   `--dry --dump out.bin --dump-fixture fx`, the C++ oracle's output and the
   exact fixture bytes went to a numpy implementation of the same formula
   written from `attn_triton.py:142-162`. Max |difference| **1.5e-5** on the
   512×24×128 output, checksums 494.012091 (numpy) against 494.011836 (C++).
   That is f32 summation order, not a formula difference. The fixture's q row
   norms read 19.996-20.004 (the model's L2-clip to 20).

Left for the card host — **the first card command is the one below**, and it
needs the operator:

    g++ -std=c++17 -O2 -o fa_ref_host fa_ref_host.cpp -lOpenCL
    ./fa_ref_host --spv out/fa_kern.precise.spv --ref-spv out/fa_kern.ref.spv \
                  --abi out/abi.txt --shape all --dry    # builds + kernel names, no enqueue
    ./fa_ref_host --spv out/fa_kern.precise.spv --ref-spv out/fa_kern.ref.spv \
                  --abi out/abi.txt --shape all --reps 20

What that will settle, and what it may fail on:

- **whether the OpenCL build exposes the kernel from the post-link module.** The
  module is now the exact artefact the SYCL runtime loads (byte-identical to
  `icpx`'s own device image, `.prop isEsimdImage=1`, 36 VC decorations), so the
  `-vc-codegen` build option is the same path the runtime takes. `inferred`
  until the card's `[dry]` step prints a non-empty `CL_PROGRAM_KERNEL_NAMES` and
  `clCreateKernel(_ZTSN2fa4KernILi128EEE)` succeeds; if a driver refuses it,
  §AOT-fallback consumes the same module.
- whether IGC's ESIMD codegen accepts the kernel on a B60 (Xe2, DPAS fp16
  8×8×16 — supported; the reference targets Xe3/B390 with the same DPAS shape).
- the numbers: per shape and per module, max abs error and mean row KL against
  the oracle, the reference-flags tolerance verdict, and median/min/max of 20
  launches.
- The `.spv` files are portable: the same sha256 can be built and run on the card
  host, so a card-host rebuild is optional (rebuild if the icpx version there
  differs and you want the recorded sha to match the toolchain).

## Reading the comparison output

- `max abs error`: elementwise over all `Lq·Hq·D` outputs, against the f32
  oracle. The kernel writes f32, so this is the kernel's own error.
- `mean row KL`: **the row's 128 output components, each normalised to a
  simplex**, KL(oracle row ‖ kernel row) averaged over `Lq·Hq` rows. **This is
  not a quality number**: it normalises a raw output row into a simplex, it says
  nothing about the model's answer, and it is not comparable across shapes (49-68
  nats at the 2048 shapes is that artefact, not a distance). Quality is the
  campaign's answer-level KL on real logits; here the comparison is max abs error
  and row relative L2. The kernel
  writes `o` only and no attention weights, so a KL over keys is not observable
  from its output; this is the row's shape and it is labelled that way in the
  output. The operator-level KL gate for the port is a different measurement on
  real logits (campaign `inkling2-mini-b60`).
- `timing`: median, min and max of `--reps` launches, from
  `clGetEventProfilingInfo` (`CL_PROFILING_COMMAND_START`/`END` on the launch's
  own event), plus TFLOP/s over the pairs the shape actually multiplies — the
  reference's own bench arithmetic (`xpu/tests/t_inkling2_prefill_esimd.py:26`).
  `spread` is `max − min`.

## Bounds, per buffer, per shape

The maximal index each buffer can be touched at, from the kernel's own index
expressions (`fa_kern.sycl` lines 60-190), checked before any enqueue. The
tables are printed per shape; the formulas:

| buffer | maximal byte index | allocation |
|---|---|---|
| q | 2·Lq·Hq·D − 1 | 2·Lq·Hq·D |
| k | 2·Lk·Hk·D − 1 | 2·Lk·Hk·D |
| v | 2·Lk·Hk·D − 1 | 2·Lk·Hk·D |
| o | 4·Lq·Hq·D − 1 | 4·Lq·Hq·D |
| table | 2·Lq·Hq·E − 1 | 2·Lq·Hq·E |
| sink | 4·Lq·Hq − 1 | 4·Lq·Hq |

For the default shape: q/k/v 3,145,727 of 3,145,728; o 6,291,455 of 6,291,456;
table 25,165,823 of 25,165,824; sink 49,151 of 49,152. For `q2048xkv3072`: q
12,582,911 of 12,582,912; k/v 6,291,455 of 6,291,456; o 25,165,823 of
25,165,824; table 100,663,295 of 100,663,296; sink 196,607 of 196,608.

The k/v bound is the one worth explaining: the kernel's 2D loads
(`lsc_load_2d`, `fa_kern.sycl` lines 96-100 and 145-149) take a surface
descriptor `surfW = D*2-1`, `surfH = Lk-1`, `pitch = Hk*D*2-1` — byte width,
row count and row pitch minus one, the LSC 2D-block convention — so a load from
row `r` of kv head `kvh` spans `r·Hk·D·2 + kvh·D·2 … + D·2`. The key-block loop
(`kb0 < hi`, `hi ≤ Lk`, step `KB`) never names a row ≥ Lk, so the last byte
reachable is `Lk·Hk·D·2 − 1`, exactly the allocation. The `x` argument
(`c·8` / `c·16` uint32 lanes) stays inside the row because `c < NC = D/16`.
If a later shape breaks this bound, the check fails *before* the driver sees it.

## Third-party record

`THIRD_PARTY.md` carries the row: trellis-serve, commit `8f8e2fcc`, MIT,
Copyright (c) 2026 0xSero, the files in this directory. No file here is derived
from another project.

# 1a: the port arm (code), its targets, and the card recipe

The port of `fa::Kern`'s structure into our OpenCL kernels — plan, review and
changes: `opencl-plan.md` (revised for all 14 findings of the second-model
review). The code is four patches on our llama pin, `contrib/llama.cpp/patches/`:

| patch | what |
|---|---|
| `0193-flash-attn-port-op-sources` | `GGML_MAX_SRC` 20 and `GGML_MAX_OP_PARAMS` 80; `GGML_FA_SRC_KPOS/QPOS/CFLAG/RFLAG/SINK_Q` and `GGML_FA_PARAM_WINDOW`, with setters and static asserts |
| `0194-flash-attn-port-sink-positions` | the model's graph: the sink **unfolded** (no `ggml_sub`, the table is bias only, the pad bin is `extent` and zero), the table stays bin-fastest `[n_bins, n_head, rows]`, `attn_rel_bias`'s columns are **not** reversed; the kv-cache walk emits `kpos`, `qpos` and one linearity flag a 16-key and a 8-row group; the CPU's indexed-bias getter reads both table layouts |
| `0198-opencl-flash-attn-port-kernel` | `flash_attn_port.cl` and the OpenCL dispatch under `GGML_OPENCL_FA_PORT=1`: one sub-group a (head, 8-row) tile, 32-key blocks, DPAS S/PV, the f32 bias run + shuffle, positions and compares, the sink as the initial (m, l), the general path (mask), a bounds check of every buffer under `GGML_OPENCL_BOUNDS=1`, and a `DOWNGRADE … port=declined reason=` line for every shape or layout the port does not cover |
| `0199-flash-attn-port-tests` | `test-backend-ops` cases at the three perf shapes and Inkling's real ones (SWA 1024, global 1025, wrapped ring 1024 and 1088, the pad bin), positions/flags and the mask/bins derived from them, and the red procedure |

`GGML_OPENCL_FA_PORT` unset (or `0`) = the dispatch exactly as it was.

## The bar is the oracle (the oracle window, measured-here)

The reference arm ran (the oracle window, B60 at 165 W, `measured-here`, local only - no
binary in the repo or the product): `fa::Kern` compiled ahead of time by ocloc
26.27.1 from `out/fa_kern.{ref,precise}.spv` (`-spirv_input -device bmg-g21`,
`"-vc-codegen -cl-fp32-correctly-rounded-divide-sqrt -ze-opt-greater-than-4GB-buffer-required"`,
with and without `-doubleGRF`), loaded by `fa_ref_host` with
`clCreateProgramWithBinary`; device time from `clGetEventProfilingInfo`, median of
20 launches:

| shape | fa::Kern SYCL / L0 (coordinator) | oracle 128 GRF (default; 2,112 B scratch) | oracle 256 GRF (`-doubleGRF`; no scratch) | today's kernel |
|---|---|---|---|---|
| q2048 × kv3072 | 2,473-3,500 | 2,271 | **1,909** | 12,583 |
| q2048 × kv2048 | 1,828 | 1,795 | **1,457** | 9,938 |
| q512 × kv1536 | 856 | 863 | **533** | 2,293 |

Max abs error against the CPU oracle 4.3e-4, row relative L2 ≤ 3.4e-4, 0 NaN; the
ref and the precise binaries time the same. So 256 GRF is 16-38 % faster despite
half the threads a core, and **the port's bar is the 256 oracle: 1,909 / 1,457 /
533 us**. `-doubleGRF` is thus the default by measurement, not by the spill alone.

Two corrections that run forced, both in `fa_ref_host`'s README below: its "mean
row KL" normalises a raw 128-component output row to a simplex and is **not a
quality number** (49-68 nats at the 2048 shapes says nothing about quality; the
comparison's numbers are max abs error and row relative L2, and quality is the
campaign's answer-level KL). And `--abi` needed no file: `out/abi.txt` is a
SPIR-V dump the parser reads as zero arguments - the parser now says which of the
two files it got and how to regenerate it.

## The targets, written before any card run

`measured-here` (the operator, B60, 2026-11-24; the reference kernel in its own
SYCL stack; ours through our OpenCL path; one call a shape, Hq 24, Hk 8, D 128,
window 1,024, E 1,025, sinks, the table):

| shape | theirs, with the table | theirs, table removed | ours, before the port |
|---|---|---|---|
| q512 × kv1536 | **856 µs** | 537 µs | 2,293 µs |
| q2048 × kv2048 | **1,828 µs** | 1,201 µs | 9,938 µs |
| q2048 × kv3072 | **2,500-3,500 µs** | 1,630 µs | 12,583 µs |

The bias lookup's own cost, theirs 843 / 627 / 319 µs against ours 8,894 / 6,844
/ 1,275 µs. The port must reach the "theirs, with the table" column at these
shapes; the "table removed" column is the floor its base loop has to approach.
No card run has been made for this code; these are the bars it is measured
against, not results.

## The card recipe

**Superseded for the port's cases by §commands2 below** (the knobs, the cases and
the filters changed in the second session); the standing arm and the served legs
here still hold.

**The standard arm (the operator, 2026-11-24 11:31).** Every Inkling *prefill*
number here is measured and predicted with the unit at the 1.8.1 deploy:

    GGML_OPENCL_KQ_MM_ID_I8=1 GGML_OPENCL_KQ_MM_ID_I8_WIDE=1   # the expert tile
    ub 1024                                                     # main batch; 512 and 2048 as extra columns
    --llama-swa-window                                          # 262k context, the SWA ring

The wide tile is a prefill path; decode is unaffected (the arm is still named
beside a decode number). Reference, `measured-here`, an earlier window, B60: pp4096 with
WIDE=1 at ub 512 / 1024 / 2048 = **722.2 / 765.7 / 723.4 t/s**. The port's bars
are the µs of §targets below; a served t/s comparison runs on this arm, not on an
unset or wide-off build, and the port must not change decode.

1. Build the pin + the series in a tree of its own (the four patches are inside
   the series):

       <arcint>/contrib/llama.cpp/apply-series.sh <llama-tree>
       cmake -S <llama-tree> -B <build> -DGGML_OPENCL=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
       cmake --build <build> -j

2. The offline check, no call into the driver (oclc is on the card host):

       bash contrib/refkernels/trellis-fa/ocloc-check.sh <llama-tree>

   It compiles `flash_attn_port.cl` for `bmg-g21` (= the B60's BMG-G21) with
   `-vc-codegen` and `-cl-intel-256-GRF-per-thread` (the backend's default, §GRF),
   reads the disassembly and fails on any spill line; `OCLOC_GRF=128` checks the
   128 arm, which must fail on scratch. `ocloc compile` with
   `-device -input -output -options` is its documented CLI (`paper`); the
   `ocloc disasm -input -dump` form and the "spill" wording in the ISA are not
   verified here (no ocloc on the recipe host), so the first card run may need to
   adjust the check's two greps - the compile and the 0-spill verdict are the
   point, not the reader. A 256-GRF arm
   (`-cl-intel-256-GRF-per-thread`) is only a measured follow-up.

3. The cases, on the card, with the port on:

       GGML_OPENCL_FA_PORT=1 GGML_OPENCL_BOUNDS=1 <build>/bin/test-backend-ops -o FLASH_ATTN_EXT -b OpenCL
       GGML_OPENCL_FA_PORT=1 GGML_OPENCL_BOUNDS=1 <build>/bin/test-backend-ops -o FLASH_ATTN_EXT -b OpenCL -p flash_attn  # one case at a time

   Red first, as the tests carry it: `GGML_OPENCL_FA_PORT_RED=2` (the sink term
   dropped) must FAIL on the q512 case, and `GGML_OPENCL_FA_PORT_RED=1` (every
   K/V read shifted one key row up) must fail on the answers - the hardware 2D
   block read clamps at the surface height, so a shifted row is an answer
   difference, not a fault. Then unset and green.

4. The two timing arms, one call a shape, on the standard arm:

       GGML_OPENCL_KQ_MM_ID_I8=1 GGML_OPENCL_KQ_MM_ID_I8_WIDE=1 GGML_OPENCL_FA_PORT=1 \
         <build>/bin/llama-bench -m <Inkling GGUF> -p 4096 -ub 1024
       # and -ub 512 / -ub 2048 for the extra columns; the same three with GGML_OPENCL_FA_PORT unset
       # are today's arm, and GGML_OPENCL_FA_PORT_QTW=1 / =4 are the two geometry arms
       # (production runs 4, SGL:39)

   The reference on this arm is pp4096 722.2 / 765.7 / 723.4 t/s at ub 512 / 1024 /
   2048 (an earlier window, B60, WIDE=1); decode (tg128) is named on the same arm and must not
   move. The dense-prefill bounds (`GGML_OPENCL_FA_PORT_HINT=1`) are a speed lever
   only for a graph that guarantees a contiguous dense kv run; the default walks
   the whole kv axis and is safe on a wrapped ring.

5. The correctness oracle stays 1b's: `fa_ref_host` (this directory) measures the
   same shapes against the same CPU oracle; its per-module comparison and the
   port's `test-backend-ops` cases are the two references for this kernel.

## What the code is believed to do, and what is owed

The port's call ABI is checked without a device: `port_abi_check.py` parses the
kernel's 42-parameter list out of `flash_attn_port.cl` and the launch's 42
`clSetKernelArg` expressions out of `ggml-opencl.cpp`, and compares each position
against the expression that parameter must receive (a head stride, a row pitch,
an offset, a scalar). It caught a real defect before any card run: a duplicated
V head-stride argument had shifted the whole tail of the argument list. Run
against the patched series tree, 42 positions, 0 failures.

The port's two arithmetic tricks are also checked without a device:
`port_arithmetic_check.py` (this directory) runs 25,000 checks with 0 failures —
(1) the bias run read (one sub-group block read at `(d0-15)` plus the lane
shuffle) against a direct per-bin read, for linear groups and at both clamped
ends including the pad bin; (2) the sink accounting (a per-lane partial sum
reduced once, the sink term added once) against the reference's `exp(sink - m)`
in the denominator. Both checks carry a red variant that must be caught (no
shuffle; the sink in every lane), and the script fails if either red is not
caught. The `.spv`/SPIR-V tools are untouched by it.

The test cases' plumbing runs on the CPU alone (`test-backend-ops -b CPU -o
FLASH_ATTN_EXT`): every `FLASH_ATTN_EXT` case including the seven `port=1` ones
(the three perf shapes, SWA 1024, the wrapped ring, and global 1025 and 1088, the
pad bin) constructs, initializes and executes, CPU against CPU. That run found a real
defect first: `ggml_flash_attn_ext_add_indexed_bias` asserted upstream's
`[n_head, n_bins, rows]` only and aborted on the port's `[n_bins, n_head, rows]`;
the setter now accepts both, with `rows` at dim 2 in either order. It does not
compare the port against anything - the device is the other half.

### The sink folded or unfolded: the knob decides {#sink}

With `GGML_OPENCL_FA_PORT` unset the Inkling graph is exactly today's: the sink
logit is folded into every table bin (the model's `exp(s-t)/(1+sum exp(s-t))`
identity), the op gets a zero per-head sink, and the table is bin-fastest. With
the knob on the graph unfolds it: the table is bias only (the pad bin is then
zero), the per-token logit rides at `src[18]`, and the port uses it as the
initial `(m, l)`. The CPU reads `src[18]` when present, so a call the port declines
still gets the per-token sink in the CPU fallback. The env is read once per
process (a `getenv` in the model's graph builder, static-initialised).

### The layout rule: the strides the kernel indexes {#layout}

The card's first port run (the first card window, B60, `measured-here`) declined every call:
`DOWNGRADE src=opencl op=FLASH_ATTN port=declined reason=layout`. The gate had
asked for `ggml_is_contiguous(q/k/v)`; the model's FLASH_ATTN_EXT inputs are
**permuted views**. From the graph export (`inkling-ub512.txt`, lines 222 / 223;
`code`): q `[128, Lq, 24]` nb `[4, 12288, 512]` (row stride nb1, head stride nb2),
K/V `[128, n_kv, 8]` nb `[2, 2048, 256]`. For K/V that is the reference's own
`[Lk, Hk, D]` memory order, and the kernel's 2D block reads take it as it is
(pitch nb1, base `kvh·nb2`). For Q it is f32 where the kernel's DPAS wants f16:
the 256-GRF build converts a tile's rows in registers, the 128-GRF build runs a
prepass (§structure).

The rule is one pure function, `ggml/src/ggml-opencl/fa_port_gate.h` (no `cl_`
call, no backend state; a copy is beside this file):

| term | rule | what reads it |
|---|---|---|
| q | f32, nb0 = 4, nb1 / nb2 / view offset multiples of 4 | 16 (or 8) contiguous floats at `head·nb2 + row·nb1` |
| k / v | f16, nb0 = 2, nb1 % 16 = 0, 64 ≤ nb1 < 2^24, nb1 ≥ D·2, nb2 % 64 = 0, view offset % 64 = 0, buffer alignment % 64 = 0 | the 2D block reads, surface `(D·2-1, n_kv-1, nb1-1)` at `base + kvh·nb2` (the same terms `flash_attn_dpas2.cl`'s reads have) |
| mask | f16 `[≥ n_kv, ≥ rows, 1, 1]`, nb0 = 2, nb1 ≥ n_kv·2 | `row·nb1/2 + key` (the general path) |
| table | f32 `[n_bins ≥ 17, n_head, ≥ rows]`, nb = 4, n_bins·4, n_head·n_bins·4 | `(row·n_head + head)·n_bins + bin`, a run of 16 bins |
| sink | f32 `[n_head, ≥ rows]`, nb = 4, n_head·4 | `row·n_head + head` |
| dst | f32 `[D, n_head, rows]`, nb = 4, D·4, n_head·D·4 | `head·nb1 + row·nb2 + 4·col` |
| rows | ≥ 9, one stream, head 128, no softcap, no ALiBi, not paged, Intel | the scope |

Both the model's permuted views and a contiguous q/k/v pass it; a row pitch of
300 bytes, a head stride of 200, a 32-byte view offset, a `[head, bin, row]`
table, a per-head sink, two streams, a q8_0 K do not.

**The unit test, on the export's own lines** (`port_gate_test.cpp`, CPU, no
OpenCL). It reads the three export files, takes each FLASH_ATTN_EXT line's
q / k / v / mask **ne and nb** from the line (lines 220-223), the bias table from
the MUL_MAT that produces `[1025, 24, rows]` (line 91 / 92) with the strides it has
as a source of the SUB that consumes it (line 11 / 12), the per-query sink from the
MUL_MAT that produces `[24, rows]` (line 79 / 80), and synthesizes only what 0193 /
0194 add and no export of today's graph can hold (indices' shape, positions,
group classes, the op's output). Expected: lines 222 / 223 (the prompt batch: the
SWA ring `q·ub × kv·(ub+1024)` and the global cache `q·ub × kv4096`) ACCEPT; lines
220 / 221 (the decode step) refuse `decode`; every accepted line, bent one property
at a time (18 bends), refuses with that property's reason.

Red first, then green (`measured-here`, the recipe host):

    g++ -DFA_PORT_GATE_RED ...   (0198's contiguity rule)
        rc 1, "FAIL: 6 failure(s)": lines 222 and 223 of each of the three files, "refuse layout"
    g++ ...                      (the stride rules)
        rc 0, "PASS: 0 failure(s)": 6 lines ACCEPT, 6 decode lines refuse "decode",
        108 bends refused each with its reason, 6 contiguous variants ACCEPT

An earlier version of this section said the table came from "the MUL_MAT line
216" and that the test was red-proven. Line 216 is the PAD of `attn_rel_bias`;
the test that existed looked for op 6, which is SUB, had not been compiled, and
had not been run. Retracted.

### The offline gate, run on the committed kernel {#gate}

`ocl-gate.sh` (the recipe host's offline gate) (ocloc 26.27.1 with IGC 2.38.2, `bmg-g21`, the backend's
options, `IGC_ShaderDumpEnable` read of each kernel's `.zeinfo`) on
`flash_attn_port.cl`, one argument a variant (base, `GB_PAD_KEYS_OFF=1`,
`GB_SKIP_SINK`):

    OCL_EXTRA="-cl-intel-256-GRF-per-thread":  ocloc 3/3 variants  (PASS)
      kernel_flash_attn_port: grf 256 simd 16 slm - scratch 0 stack_calls -   (each variant)
    no OCL_EXTRA (the 128 arm):                ocloc 0/3 variants  (FAIL)
      kernel_flash_attn_port: grf 128 simd 16 slm - scratch 8192 stack_calls -  <-- FAIL (each variant)

The 256 arm is the default for that reason (§GRF). The 128 arm is the measured
alternative and the gate says it spills, so it is not shippable and not the
default.

### GRF: 256 is the default, measured {#GRF}

(The first session's kernel; the current kernel's gate lines are in §gate2.)

The offline gate (`ocloc` 26.27.1 with IGC 2.38.2, `bmg-g21`, the backend's
options; `ocl-gate.sh` (the recipe host's offline gate)) on this kernel, three variants (base,
`GB_PAD_KEYS_OFF=1`, `GB_SKIP_SINK`):

    flash_attn_port: grf 128 simd 16 scratch 8192  -> 0/3 variants (FAIL: spill)
    flash_attn_port: grf 256 simd 16 scratch 0     -> 3/3 variants (PASS, no calls, no retry)

And the reference's own kernel, compiled from its exact SPIR-V by the same IGC
(`ocloc compile -spirv_input -device bmg-g21 fa_kern.precise.spv`):

    fa::Kern: grf_count 128, simd_size 1, scratch 2112 B (IGC's default; it spills)
    fa::Kern, -doubleGRF: grf_count 256, no scratch entry (0)

So the reference's measured 2,500-3,500 / 1,828 / 856 us came from a **spilling**
build, and the 8-row tile with its O accumulator and DPAS operands does not fit
128 GRF. The port defaults to 256 (`-cl-intel-256-GRF-per-thread`); the cost is
thread occupancy, 256 halves the threads a core (640 against 1,280 on the B60,
`exl3_ops.sycl:176`, `CL:5341`). `GGML_OPENCL_FA_PORT_GRF=128` builds the
measured alternative, which the gate reports as spilling.

Also verified, no card: the CPU-only `test-backend-ops -b CPU -o FLASH_ATTN_EXT`
run after the sink/CPU edits passes (1/1 backends, “OK”, the seven `port=1` cases
included), and a fresh tree of the pin plus the 145-patch series with the four
current patches applied configures and builds `test-backend-ops` and `llama-cli`.

`code` here: the four patches were checked against a fresh tree of the pin plus
the 145-patch series — `git apply --check` clean for all four, and that tree
builds `ggml-opencl`, `llama-cli` and `test-backend-ops`; the `.cl` passes the
frontend with the four Intel builtins stubbed, since
this frontend has no `intel_sub_group_f16_f16_matrix_mad_k16` or
`__builtin_IB_subgroup_block_read_*` — the dedicated kernels fail the same way,
see §tools): the patches apply cleanly to a fresh pin + series. **Owed, and the
first card step:** a device build and a run — no enqueue has happened for this
kernel, so its numbers, its spills and its answers are all unmeasured. The
linearity flags, the GENERAL path on a ring seam, and the `kv_lo` hint are the
three places a first card run is most likely to correct.

One correction made after the first commit, before any card run: the kernel's
surfaces. Q is `[D, Lq, Hq]` and K/V `[D, Lk, Hk]` with D fastest, so a head's
slice is a base offset (`head * nb[2]`, `kvh * nb[2]`) and a row pitch is
`nb[1]` — the first form used `nb[2]` as the Q row pitch and started every head
at offset 0, which would read head 0's rows and the wrong pitch. The shim now
passes `q_nb1/q_nb2`, `k_nb1/k_nb2`, `v_nb1/v_nb2`, `o_nb1/o_nb2` and the kernel
takes the head slice from `nb[2]`; the sink is attached as the raw
`[n_head, n_tokens]` tensor (head fastest, which is the layout the kernel reads),
without the permute/copy the first form built.

## Prediction before the first card run (the coordinator, inference)

(Kept as written. The implementer's prediction for the restructured kernel, per
register mode, is §prediction2.)

Written 2026-11-24, before the first card run, by the coordinator, not the port session. Inferred from the plan: the port
follows fa::Kern's structure, so its base loop should sit near the reference's no-table times, plus our table read.
The 256-GRF build halves the threads a core against a 128-GRF build. fa::Kern's own measured numbers came from its
default build, which spills at 128 GRF.

| shape | predicted, port | fa::Kern | fa::Kern without the table | today's kernel |
|---|---|---|---|---|
| q2048 x kv3072 | ~1,700-2,600 us | 2,473-3,500 | 1,630 | 12,583 |
| q2048 x kv2048 | ~1,300-2,000 us | 1,828 | 1,201 | 9,938 |
| q512 x kv1536 | ~600-1,000 us | 856 | 537 | 2,293 |

The first window stops at the single case if it is slower than today's kernel.

# 1a, second session: the gate, the graph, the structure {#second}

## What the handed-over state was, claim by claim

Checked against the code, the export and the reference before anything was built
on it (`code` unless marked):

| handover claim | verdict |
|---|---|
| the first card window's `reason=layout` comes from `ggml_is_contiguous(q/k/v)` against permuted views | **true** (export lines 222 / 223) |
| "Those are exactly the strides the kernel indexes" | **true for K/V** (the reference's `[Lk, Hk, D]` order, pitch nb1, base `kvh·nb2`); **not for Q**: the op's Q is f32, the kernel read f16, which the handover names as its second mismatch |
| the gate is now a pure function, still carrying the red rule | **true** |
| `gate_v2.py` holds the stride rules | **partly**: its Q terms (`% 32`) guard a `vload8` that needs no alignment; it has no rule for the view offset or the buffer alignment (the 2D block read's 64-byte base); not applied here, the header was written directly |
| `port_gate_test.cpp` takes the table "from the MUL_MAT line 216" | **false**: line 216 is PAD; the MUL_MAT is line 91 / 92; the test matched op 6, which is SUB (lines 11 / 12) |
| the README's "It is red-proven" | **false**: the test had never been compiled |
| "n_q <= 8 keeps 0143's split kernels" (the gate's `decode` refusal) | **false**: with the knob on, a declined call returned `false` from `supports_op`, so every decode step went to the CPU, on an unfolded graph |
| 0194: the port's sources are attached after `build_attn` | **a defect the handover did not know**: `build_attn_mha` asks the layer's backend `supports_op` *before* that, so the gate saw no positions, refused `no-positions`, and the graph fell to the dense gather, which cannot read the port's `[n_bins, n_head, rows]` table (`ggml_reshape_4d` asserts). With the knob on the model could not build a prompt graph at all, and the port could never have been taken |
| 0194: `kq_b = ggml_get_rows(table, rel_pos)` built unconditionally | **the same abort** by itself: on the unfolded table the reshape's element count does not match |
| 0198: one linearity flag a block (`cfl[t0/16]`) | **a defect**: the block's second 16-key group took the first group's flag |
| 0198: the `kv_lo` hint | **a defect**: not rounded to the 32-key block, so the group flags were read for the wrong groups; and no causal upper bound at all |
| 0198: `intel_sub_group_block_read(… + base + lane)` | worked only because the pointer of lane 0 is the one used; now a uniform pointer |
| 0198: the softmax | **a defect**: `m` was the sink logit for the whole walk and never updated, so `p = exp2(s - m)` overflows f16 for a score 16 (base 2) above the sink; fa::Kern updates `m` a block |
| 0198: `qps[q0 + r]` for a tile past `n_q` | read up to 7 ints past the buffer (the host's bounds check cannot see a kernel's index) |
| 0198: no TAKEN line | **true**, added |
| 0199: "the mask and bins are built from the positions" | **false for the bins and worse for the mask**: the port's branch sat behind the generic `"bi"` branch (the port cases got pseudo-random bins) and, for every non-port case, replaced `init_tensor_kq_mask` by a uniform fill - today's FLASH_ATTN_EXT cases would have lost their masks |
| 0199: the cases' layout | contiguous q and a kv view: the layout the first card window declined, not the model's |
| 0199: "RED=2 must FAIL on the q512 case" | **would not have failed**: a sink in [-1, 1] beside ~1,000 keys moves the answer by about 1e-6 NMSE against a 5e-4 tolerance (`inference`); the sink is now in [3, 6] |
| 0199: the pad bin "is read, and not added" | the test's table had a random pad bin, the kernel adds nothing past the extent: the far-key cases would have failed; the port's table now has a zero pad bin, as the model's has |
| the oracle table (the oracle window) | taken as recorded (`measured-here`, local only, reference-arm evidence): it is fa::Kern's own SPIR-V built by ocloc and run by `fa_ref_host`; nothing of it ships. `fa_ref_host`'s "mean row KL" is **not a quality number** (it normalises a raw output row to a simplex), and `out/abi.txt` is a SPIR-V dump that `read_abi` does not parse |

## What changed {#structure}

**The graph (0194).** The port's graph is a prompt batch's graph only:
`GGML_OPENCL_FA_PORT=1`, flash attention on, more than 8 rows, one stream. A
decode or verify step keeps today's graph, today's kernels, bit for bit. The
port's sources (per-query sinks, positions, classes, window) ride to
`build_attn_mha` in `llm_graph_context::fa_port_next` and are on the op before the
backend is asked; an op that carries them and is declined stays an indexed-bias op
for the CPU (which reads the port's table and `src[18]`), it is never bent into
the dense gather. Key-group classes replace the flag: 1 = consecutive positions,
all cells of the first row's sequence, none empty; 2 = every cell empty; 0 = the
rest. A row group is 1 when its rows are consecutive positions of that sequence.

**The dispatch (0198).** The port branch is taken for an op that carries
`src[18]`; any other indexed-bias op takes 0143 / 0144's dispatch as before, and an
op with `src[18]` and the knob unset is declined (`reason=knob-off`) and not
handed to a kernel that would read its table another way. A refusal after
`supports_op` accepted is an abort, not a silent return with the output unwritten.
Once a process: `TAKEN src=opencl op=FLASH_ATTN port=fa_kern grf=… qhoist=… qf32=…
skip=… qtw=…`.

**The kernel, against fa::Kern** (`inkling2_ops.sycl:509-654`), three structural
reasons it would have landed well above the oracle, and one of correctness:

1. *The keys a tile visits.* fa::Kern walks `[lo, hi)` a tile: the window's
   lower bound and the causal upper bound, about 33 of the 96 blocks at
   q2048 × kv3072. The port walked every block of the kv axis. Positions here
   are arrays (a ring has a seam, a cache is presented beyond its fill), so the
   bound is a test a block: skipped when both of its groups are empty, past
   `n_kv`, or linear under linear rows with every key in the future of the tile's
   last row or behind the window of its first. A skip is two flag reads and two
   position reads. The export's global line (q512 × kv4096 with the cache part
   filled) is the case this decides: without the class 2 skip a first batch
   would walk 128 blocks for 16 used. `GGML_OPENCL_FA_PORT_SKIP=0` is the A/B.
2. *Q in registers.* fa::Kern holds a tile's Q (`Qc`, 32 registers); the port
   re-read 8 × 16 × 8 halves a key block, 8 loads beside the block's 32. The
   256-GRF build holds it (`PORT_QHOIST`).
3. *No prepass.* The oracle is handed f16 Q; our op hands f32. A separate
   conversion launch moves 37.8 MB at q2048 before the kernel starts. The
   256-GRF build reads the f32 rows itself, 64 sub-group block reads a tile, and
   converts in registers (`PORT_QF32`); `GGML_OPENCL_FA_PORT_QF32=0` keeps the
   prepass (the port's own `kernel_flash_attn_port_q16`, built with the kernel -
   no dependence on `flash_attn_dpas2.cl` having been built).
4. *The online softmax* as fa::Kern has it: a block's row maxima (8 sub-group
   reductions), `alpha`, `l` and `O` rescaled, `p ≤ 1` to f16. The sink's term
   is 1 in lane 0 from the start and is rescaled with the rest.

What stays as it was: one sub-group a (head, 8-row) tile, 32-key blocks, the
DPAS pair, the K/V 2D reads, the bias run (one block read + a shuffle) and the
clamped read, the general path through the mask.

`port_kernel_model.py` transcribes the kernel's block loop a lane at a time
(classes, skip, compares, mask path, bias run, online softmax with f16 `p`) and
compares it with a direct masked softmax on the op test's own position layouts
(ring seam in either group, empty tail, window, a short last tile, a group past
`n_kv`): 13 cases, NMSE ≤ 1e-7, and three bent kernels (the window skip decided
from the tile's last row, a mixed group taken as linear, the sink in every lane)
each caught (`measured-here`, the recipe host). It is a model of the control flow,
not the kernel: the device run is still the proof.

## Knobs (all read once a process; unset = the default named)

| knob | default | what |
|---|---|---|
| `GGML_OPENCL_FA_PORT` | off | the arm; unset = today's dispatch and today's graph exactly |
| `GGML_OPENCL_FA_PORT_GRF` | 256 | `128` or `256`: the register mode of the port's program |
| `GGML_OPENCL_FA_PORT_QHOIST` | 1 at 256, 0 at 128 | Q held in registers a tile |
| `GGML_OPENCL_FA_PORT_QF32` | = QHOIST | Q read from the op's f32 rows, no prepass |
| `GGML_OPENCL_FA_PORT_SKIP` | 1 | `0` walks every key block |
| `GGML_OPENCL_FA_PORT_QTW` | 1 | adjacent query tiles a work-group (the reference's server runs 4) |
| `GGML_OPENCL_FA_PORT_RED` | - | `1` K/V one key row up, `2` the sink's term dropped: both must FAIL |
| `GGML_OPENCL_BOUNDS` | off | every buffer of the launch against its allocation, abort past it |

## The offline gate on this kernel {#gate2}

`ocl-gate.sh` (ocloc 26.27.1, IGC, `bmg-g21`, the backend's options, each kernel's
`.zeinfo`), `measured-here` on the recipe host, no device:

    256 GRF (-cl-intel-256-GRF-per-thread), five variants - QHOIST+QF32 (the default),
    QHOIST with the prepass, GB_NO_SKIP, GB_PAD_KEYS_OFF=1, GB_SKIP_SINK:
      ocloc 5/5 variants
      kernel_flash_attn_port:     grf 256 simd 16 scratch 0 stack_calls -   (each variant)
      kernel_flash_attn_port_q16: grf 256 simd 32 scratch 0 stack_calls -
    128 GRF (no option; QHOIST=0, QF32=0):
      ocloc 0/1 variants
      kernel_flash_attn_port:     grf 128 simd 16 scratch 8192 stack_calls -   <-- FAIL (spill)
      kernel_flash_attn_port_q16: grf 128 simd 32 scratch 0

So Q in registers and the f32 read fit 256 GRF without a spill, and the 128 build
spills as the first session's did. It builds and is selectable for the timing;
the gate does not pass it.

Also without a device (`measured-here`, the recipe host): the applied tree builds
`test-backend-ops` and `llama-cli` with the OpenCL backend and embedded kernels;
`GGML_OPENCL_FA_PORT=1 test-backend-ops -b CPU -o FLASH_ATTN_EXT -p "port=1"` runs
the ten port cases, 10/10 (CPU against CPU: the cases construct with the export's
layout, the layout asserts hold, the graph executes - it compares the port with
nothing); `port_abi_check.py` 42 positions, 0 failures.

## Prediction before the card run (inference) {#prediction2}

Written before any card run of this code. The oracle's time is proportional to
the key blocks its tiles visit (`code`: 33.25 blocks a tile inside a 1,024
window; `measured-here`, the oracle window: 533 / 1,457 / 1,909 us over 51k / 153k / 204k
block visits = 10.4 / 9.5 / 9.4 ns a visit). The port now visits the same blocks
with the same loads and DPAS; what it has on top is a header a skipped block
(63 of 96 at kv3072), two position gathers a block, eight reductions, and the
per-row bias logic as SIMD-16 code where fa::Kern has hand-laid vectors.

| shape (window 1,024 unless said) | oracle 256 | **port 256 GRF, us a call** | oracle 128 | **port 128 GRF, us a call** | today's kernel |
|---|---|---|---|---|---|
| q512 × kv1536 | 533 | **560-720** (620) | 863 | **1,100-1,550** | 2,293 |
| q2048 × kv2048 | 1,457 | **1,530-1,970** (1,700) | 1,795 | **2,300-3,200** | 9,938 |
| q2048 × kv3072 | 1,909 | **2,050-2,600** (2,250) | 2,271 | **2,950-4,100** | 12,583 |
| q512 × kv4096, 1,536 filled, global | - | **650-900** | - | **1,300-1,900** | - |

The 128 column is a spilling build (the gate: 8 KiB a thread against the
oracle's 2.1 KiB), with Q re-read a block and the prepass launch; it is expected
to lose to 256 by more than the oracle's own 128 does.

ms a pass (24 attention layers, 20 SWA and 4 global, `code`: binding-notes.md;
pp4096, the global layers at their mean depth), port at 256 GRF: **ub 512 about
18 ms (16-23), ub 1024 about 31 ms (27-38), ub 2048 about 63 ms (55-75)**;
today's kernel is 46 ms for the 20 SWA calls alone at ub 512 (20 × 2,293 us).

If the single case lands above 900 us at 256 GRF, the gap is in the per-row
score pass (validity, bias, the scalarised 8 × 2 loop), and that loop is the
candidate for the inline-vISA rule; the ISA dump decides.

## The card commands, in order {#commands2}

`T` is `<build>/bin/test-backend-ops`, the build a fresh pin + `apply-series.sh`
(write-tree in the commit message), kernels embedded. `P1` is the single case's
filter, `"kv=1536,nb=512,.*port=1"`.

    # 1. the single case under the bounds check; expect 1 case run, OK, the TAKEN line, no DOWNGRADE port=
    GGML_OPENCL_FA_PORT=1 GGML_OPENCL_BOUNDS=1 T -b <OpenCL device> -o FLASH_ATTN_EXT -p "$P1"
    # 2. the suite: 0 FAIL, and the ten port=1 cases must have RUN (0 cases ran = red)
    GGML_OPENCL_FA_PORT=1 GGML_OPENCL_BOUNDS=1 T -b <OpenCL device> -o FLASH_ATTN_EXT
    #    and once with the knob unset: today's cases as before, the port=1 cases "not supported" (reason=knob-off)
    T -b <OpenCL device> -o FLASH_ATTN_EXT
    # 3. the two reds: each must FAIL
    GGML_OPENCL_FA_PORT=1 GGML_OPENCL_FA_PORT_RED=1 T -b <OpenCL device> -o FLASH_ATTN_EXT -p "port=1"
    GGML_OPENCL_FA_PORT=1 GGML_OPENCL_FA_PORT_RED=2 T -b <OpenCL device> -o FLASH_ATTN_EXT -p "port=1"
    # 4. op perf, the single shape: the port, then today's kernel on the same shape (port=0)
    GGML_OPENCL_FA_PORT=1 T perf -b <OpenCL device> -o FLASH_ATTN_EXT -p "kv=1536,nb=512,.*n_bins=1025.*port=1"
    T perf -b <OpenCL device> -o FLASH_ATTN_EXT -p "kv=1536,nb=512,.*n_bins=1025.*port=0"
    #    stop here if the port is slower than today's kernel
    # 5. the other shapes, both register modes
    for g in 256 128; do GGML_OPENCL_FA_PORT=1 GGML_OPENCL_FA_PORT_GRF=$g T perf -b <OpenCL device> -o FLASH_ATTN_EXT -p "n_bins=1025.*port=1"; done
    T perf -b <OpenCL device> -o FLASH_ATTN_EXT -p "n_bins=1025.*port=0"
    # 6. the A/Bs worth one run each at 256: the prepass, the skip, four tiles a work-group
    GGML_OPENCL_FA_PORT=1 GGML_OPENCL_FA_PORT_QF32=0 T perf ... -p "n_bins=1025.*port=1"
    GGML_OPENCL_FA_PORT=1 GGML_OPENCL_FA_PORT_SKIP=0 T perf ... -p "n_bins=1025.*port=1"
    GGML_OPENCL_FA_PORT=1 GGML_OPENCL_FA_PORT_QTW=4  T perf ... -p "n_bins=1025.*port=1"

The perf mode's time is the op's graph (the launch and, at 128 GRF or with
QF32=0, the prepass); the oracle's is device time of one launch.
