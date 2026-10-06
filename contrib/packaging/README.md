# Packaging recipes

These are the Debian recipes this project is actually deployed with, published
because "build it yourself" is otherwise the only instruction an outside reader
gets, and because the numbers in the README and FURTHER-READING.md depend on a *patched*
OpenVINO that nobody can reproduce without knowing exactly which patch.

They are not a supported product. No `.deb` is published anywhere — the built
packages live in a private fleet repository. What is here is everything needed
to build the same thing.

The top-level `CHANGELOG.md` is the human record per release; the Debian
changelogs here are the package record.

Kept in step by hand: every tagged release updates `arcint/build-deb.sh`
(PKGVER, PKGREL, tarball sha256, the marfrit-openvino pin) and the changelog
here in the same breath as the fleet copy — a stale recipe is a broken
instruction, which is the one thing this directory exists to prevent.

## What is in here

    arcint/
      build-deb.sh          builds arcint from a GitHub release tarball with a
                            pinned sha256, with both engines: the libllama
                            engine against llama.cpp's tarball at the pinned
                            commit (sha256-checked), with the release's own
                            contrib/llama.cpp/patches applied. Then gates:
                            unit tests, a RUNPATH probe, ldd resolution, a
                            unit-template assertion, no llama.cpp install
                            residue, the engine's kernels in the binary
      debian/               control, copyright, changelog

    marfrit-openvino/
      build-openvino.sh     clones upstream OpenVINO at the pinned commit,
                            resets hard, applies patches/, builds Release
      build-openvino-debugcaps.sh
                            the same with the GPU plugin's debug
                            capabilities on, into an explicitly named
                            prefix (a measurement build, never a package)
      build-deb.sh          repacks the upstream wheels for their layout and
                            CMake package, then overwrites the libraries with
                            the patched build
      patches/              the patch series 0003–0067, and README.md: what
                            each patch does and its standing measured effect
      debian/               control, copyright, changelog

## The pin

`2026.4.0-22849-71640275d29` — upstream commit `71640275`. Every measurement in
the README and FURTHER-READING.md was taken on that code. A patch that does not apply
cleanly to that commit is a bug in `patches/`, not a reason to move the pin.

The patched build reports itself as `…-71640275d29-marfrit-pN`, so a version
string in a log says whether it is the patched runtime and at which level.
`+p20` (patches 0003–0067) is the floor arcint 0.5.0.1 depends on. Earlier
levels worth knowing when reading an old log: `+p6` is the first with the
`u8:i4` prefill on micro-SDPA (patch 0020), `+p7` the first carrying the
K-quant kernel `--gguf` needs (patch 0021), `+p15` (0003–0033) what arcint
0.4.4 through 0.5.0 depend on, and `+p16`/`+p17` carry patch 0037 without
its fix (0042, in `+p18`) and fault at the first CPU-tier prefill on an
Arc Pro B60. Several measurement builds between `+p19` and `+p20` kept the
`+p19` stamp and are identified by their symbols, not the version string
(the per-patch record says which). arcint reads the level off the plugin's
version string at load and sizes its scratch terms by it.

The build-number field stays numeric because `ov_parse_ci_build_number` rejects
anything else; the patch level rides in the free-form tail.

## Two things worth knowing before you try

**`build-openvino.sh` is a real compile**, roughly 35 minutes on 8 cores, and
it needs an x86_64 glibc userland. The recipe resets the checkout hard before
applying patches — that is not ceremony. The session that found the MoE bug
carried per-stage timing instruments in its own tree, and without the reset
those would have shipped inside a "release" build.

**`build-deb.sh` still unpacks the upstream wheels.** The wheel layout is
load-bearing: `OpenVINOTargets-release.cmake` resolves every library as
`${_IMPORT_PREFIX}/openvino/libs/<soname>`. Flatten it and `find_package`
succeeds, then the link fails pointing at paths that do not exist. So the
layout, the CMake package, the headers and `libopenvino_tokenizers.so` come
from the wheels; only the OpenVINO libraries are replaced.

Every library the wheel shipped must exist in the build, checked one by one —
a missing one is a loud failure rather than a silent capability regression.
That check exists because `--no-install-recommends`-style thinking once cost a
window manager elsewhere in this fleet, and the same class of mistake here
would ship a runtime quietly missing a plugin.

## Provenance

These recipes come from a private packaging repository and were scrubbed of
fleet-internal host names before publication. Maintainer addresses in the
changelogs are left as they are — a Debian changelog has an author field.
