# Packaging recipes

The Debian recipe this project is deployed with, published because "build it
yourself" is otherwise the only instruction an outside reader gets, and
because the numbers in the README depend on a *patched* llama.cpp that
nobody can reproduce without knowing exactly which commit and which patches.

It is not a supported product. The built packages live in the operator's own
repository; what is here is everything needed to build the same thing.

The top-level `CHANGELOG.md` is the human record per release; the Debian
changelog here is the package record.

Kept in step by hand: every tagged release updates `arcint/build-deb.sh` and
`arcint-imggen/build-deb.sh`
(PKGVER, PKGREL, the release tarball's sha256; the llama.cpp commit and its
tarball's sha256 when the pin moves) and the changelog here in the same
breath. A stale recipe is a broken instruction, which is the one thing this
directory exists to prevent.

## What is in here

    arcint/
      build-deb.sh          builds arcint from a GitHub release tarball with a
                            pinned sha256 against llama.cpp's tarball at the
                            pinned commit (sha256-checked), with the release's
                            own contrib/llama.cpp/patches applied. Then gates:
                            unit tests, ldd resolution, a unit-template
                            assertion, no llama.cpp install residue, the
                            engine's kernels in the binary, ggml-cpu built
                            native (GGML_NATIVE in the CMake cache, AVX2 in
                            ggml_vec_dot_q4_K_q8_K)
      debian/               control, copyright, changelog
    arcint-imggen/
      build-deb.sh          builds the image service (sd.cpp's sd-server and
                            sd-cli as arcint-sd-server / arcint-sd-cli) from
                            the same arcint tarball, llama.cpp's at the pin
                            and stable-diffusion.cpp's at f0aac52, each
                            sha256-checked, through contrib/sdcpp/build-sdcpp.sh;
                            then gates: 0057's, 0058's and the DPAS attention's
                            kernels in sd-server, the /props build string, ldd
                            resolution, no Vulkan
      debian/               control, copyright, changelog

## Two things worth knowing before you try

**ggml-cpu must be native.** The recipe exports `SOURCE_DATE_EPOCH` for a
reproducible build, and ggml reads that as a request for a portable build
(`GGML_NATIVE=OFF`): every package from 0.5.5 to 0.7.11 shipped an SSE-only
ggml-cpu, which cost Flash-Next with the CPU misses 7.7 %. The recipe asks for
`-DGGML_NATIVE=ON`, checks the CMake cache, and disassembles a ggml-cpu dot
product for `%ymm` registers. The package therefore runs only on a CPU with
the build host's instruction set (AVX2 at least).

**The GPU runtime is not a dependency the package can name.** Debian trixie
ships no Intel compute runtime in any component; the package depends on the
OpenCL ICD loader, and the host needs Intel's compute-runtime and IGC
release `.deb`s installed beside it (README, "Building"). A fresh container
with those and this package served the coder at the measured speed.

## Retired

`marfrit-openvino/`, the recipe for the patched OpenVINO runtime the
OpenVINO engine needed (the pinned nightly `71640275` with plugin patches
0003-0077), was retired with that engine in 0.8.0. It is at the tag
`openvino-history` (`docs/history.md`).

## Provenance

These recipes come from a private packaging repository and were scrubbed of
fleet-internal host names before publication. Maintainer addresses in the
changelogs are left as they are: a Debian changelog has an author field.
