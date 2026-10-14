# Container images

arcint can be packed as a pull-and-run container set the way other Arc
serving stacks are. Two images, because the two halves of the dependency
have very different change rates:

| Tier | Image | Rebuilds when | Build cost |
|---|---|---|---|
| 1 | `arcint-ov` — the **patched** OpenVINO runtime | the patch series or the pin changes | heavy (~35 min) |
| 2 | `arcint` — the engine, both engines in one binary | every commit | light |

Splitting them is the whole point. Building them together would recompile a
pinned OpenVINO nightly on every engine commit.

The split is *per layer of divergence*, not per backend. The patch series
in `contrib/packaging/marfrit-openvino/patches/` is why a single
"install OpenVINO" Dockerfile is not available: the runtime arcint is
measured against is a source build, not a release.

Tier 2 builds **both** engines the `.deb` builds — `-DARCINT_OPENVINO=ON`
and `-DARCINT_LLAMA=ON` with llama.cpp at the recipe's pin and
`contrib/llama.cpp/patches` applied. Since 0.5.5 the units this project
serves run `--engine llama`, so an image carrying the OpenVINO path alone
would not be the thing the project ships.

## Building

From the repo root:

```bash
# tier 1 — the patched OpenVINO base
docker build -t arcint-ov:2026.4.0-ov71640275-marfrit-p27 \
  -f docker/Dockerfile.openvino-patched .

# tier 2 — the engine on top of it
docker build -t arcint:local \
  --build-arg OV_BASE=arcint-ov:2026.4.0-ov71640275-marfrit-p27 \
  -f docker/Dockerfile.arcint .
```

Or dispatch `.github/workflows/build-toolbox.yml`, which builds both and
publishes to GHCR under the running repo's own namespace. It takes a
`tiers` input (`ov`, `arcint`, or `all`) and a `channel` input. No
configuration is needed to run it from a fork — the namespace is derived
from `GITHUB_REPOSITORY_OWNER`.

### Operational notes

**First run must include tier 1.** Tier 2 takes its base with `FROM` on the
registry, so a `tiers=arcint` dispatch with no tier-1 image published yet
fails at the first `FROM`. Use `tiers=all` (or `ov,arcint`) the first time.

**GHCR packages are private on first push.** To let others pull the images,
set the package visibility to public after the first push, or have them log
in with a PAT carrying `read:packages`.

**Tier 1 needs runner disk space freed first.** The OpenVINO clone and build
is larger than a GitHub runner ships with, so the tier-1 job removes the
unused SDK trees (`dotnet`, `android`, `ghc`, `CodeQL`) and prunes the
Docker image and builder caches before building. A tier-1 build on a
runner that has not been cleaned can fail partway with no space left on the
device.

## Tagging: the tag names the pin and the package level

The tier-1 tag is the OpenVINO pin plus the package level the recipe is at:

```
2026.4.0-ov71640275-marfrit-p27
```

The `ov71640275` component is the recipe's `PIN` verbatim — eight hex
characters, as `build-openvino.sh` names it. The full 40-char commit the
image was actually built from is in the `ai.arcint.ov.pin` label.

The level is a usable identity: since the `+p20` package, a release names
its package level (`contrib/packaging/marfrit-openvino/patches/README.md`).

The **patch ceiling** — the highest patch actually applied — is still
recorded, as an OCI label, so a running container can be asked without
opening it:

```bash
docker inspect --format '{{index .Config.Labels "ai.arcint.ov.patchceiling"}}' <image>
```

Labels carried by the tier-1 image: `ai.arcint.ov.pin`,
`ai.arcint.ov.patchlevel`, `ai.arcint.ov.patchceiling`.

### The tag is derived, and the build args are gated

The workflow does not have the tag written down: it reads `PIN` and
`PATCHLEVEL` out of
`contrib/packaging/marfrit-openvino/build-openvino.sh` and names the tier-1
tag from them. The Dockerfiles then gate their own build args against the
same recipes — tier 1 checks `OV_PIN`/`PATCHLEVEL`/`CEILING` against
`build-openvino.sh` and the contents of `patches/`, tier 2 checks
`LLAMA_COMMIT`/`LLAMA_TARBALL_SHA256` against
`contrib/packaging/arcint/build-deb.sh` and the base image's version string
against the level it expects.

That is deliberate: if the recipe and the image ever disagree, the build
fails in seconds instead of publishing a base whose tag describes a
runtime nobody built.

## Running

Device access follows the standard Intel Arc recipe — the host supplies the
kernel driver and `/dev/dri`, the image supplies the userspace compute
stack:

```bash
docker run --rm -it \
  --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -v "$MODELS:/models:ro" \
  -p 8080:8080 \
  arcint:local \
  --model /models/ov/<ir-dir> --model-id <id> \
  --device GPU.0 --host 0.0.0.0 --port 8080
```

`docker/docker-compose.example.yaml` is the same thing in compose form,
including the writable blob-cache volume — a cold kernel cache turns every
start into a recompile, so make that volume persistent.

Models are never baked into an image. They are mounted read-only.

The same image serves the GGUF path — the engine the served units use:

```bash
docker run --rm -it \
  --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -v "$MODELS:/models:ro" \
  -p 8080:8080 \
  arcint:local \
  --engine llama --gguf /models/gguf/<model>.gguf \
  --device GPU.0 --host 0.0.0.0 --port 8080
```

`--engine` selects the executor; `--engine llama` serves a GGUF through
ggml's OpenCL backend, which is why the image carries the ICD loader and
the Intel compute-runtime driver (`ocl-icd`, `intel-compute-runtime`).
See `docs/llama-engine.md` for that engine's own flags.

### The three services as the project runs them (0.7.8)

The command lines below are the ones the project's own services use;
`docker/docker-compose.example.yaml` carries all three as compose profiles.
Paths are placeholders under the read-only `/models` mount.

**The coder** (Qwen3.6-35B-A3B coder GGUF, the A770 or any 16 GB card):

```bash
--engine llama --gguf /models/<coder>.gguf --served-model-name qwen3.6-coder \
--device GPU.0 --n-ctx 98304 --llama-mtp 4 --llama-kv q8_0:q4_0 --queue-timeout 30
```

**The dense agent** (Qwen3.8-27B Q4_K_M, a 24 GB card):

```bash
--engine llama --gguf /models/<qwen3.8-27b-q4_k_m>.gguf --served-model-name qwen3.8-agent \
--device GPU.0 --n-ctx 245760 --llama-kv q8_0:q4_0 --llama-mtp 5 \
--llama-mtp-vocab /models/<draft-vocab>.json --queue-timeout 30 --repetition-penalty 1.0
```

`--llama-mtp-vocab` restricts the MTP head's output to a token subset, so the
draft head costs a fraction of the full vocabulary. It is optional; the
project's agent runs with it, and an arm without it was not measured for
this release.

**Qwen3.8-Flash-Next** (IQ2_XS GGUF, every expert through an expert cache on
a 24 GB card, the rest in host memory):

```bash
# environment: GGML_OPENCL_BANK_STREAM=1
--engine llama --gguf /models/<flash-next-iq2_xs-00001-of-00002>.gguf \
--served-model-name qwen3.8-flash-next --device GPU.0 --n-ctx 131072 \
--llama-cpu-moe 48 --llama-threads 8 --llama-expert-cache 11300 \
--llama-expert-profile /models/<expert-profile>.bin --llama-mtp 2 \
--llama-mtp-gguf /models/<flash-next-mtp-head>.gguf --llama-kv q8_0 --queue-timeout 30
```

Flash-Next needs more from the host than the other two:
- **pinned host memory:** the expert bank (~22 GiB for IQ2_XS) lives in
  USM host memory that the driver pins. The host's TTM page limit must
  allow it (`/sys/module/ttm/parameters/pages_limit`; the default is half
  of system RAM, which on a 64 GB host leaves little room beside the bank),
  and the container needs no extra capability for it;
- **the GGUF on fast local storage:** experts not in the bank or the cache
  are read from the file; with the file on a ZFS pool the record has a
  Flash-Next quant decoding at 4-5 t/s (README, current numbers);
- **a VRAM margin:** 11,300 MiB of expert slots peaks at ~24.2 GB of a
  24.5 GB card. Do not run anything else on that card.

## CPU baseline: the image is portable, and that costs something

A container is built once and run anywhere, so the CPU instruction set has to
be chosen at build time for a machine that is not present. The engine image
is built at the **x86-64-v3** baseline:

| enabled | disabled |
|---|---|
| SSE4.2, AVX, AVX2, BMI2, FMA, F16C | AVX-512 (F/CD/BW/DQ/VL), AVX512-VBMI, AVX512-VNNI, AVX512-BF16, AVX-VNNI, AMX |

That covers essentially every x86 server CPU from Haswell (2013) and Zen 1
(2017) onward. It is **not** the bare SSE2 baseline — AVX2 and FMA are on.

**What is left on the table:** on a host that *does* have AVX-512 or AMX —
Skylake-SP and later Xeon, EPYC 9004 and later — ggml has wider kernels for
the CPU-side work and this image will not use them. How much that matters
here is **not measured**. The served workload is GPU-resident: the CPU
backend carries tokenisation, sampling, any op without an OpenCL kernel, and
tensors the planner did not place on the device. So the expected cost is
small, but it is a real cost and it has not been quantified against a
native-tuned build on the same host.

**Why not build every variant and dispatch at runtime.** `GGML_CPU_ALL_VARIANTS=ON`
is the portable-and-fast option, but it requires `GGML_BACKEND_DL=ON` —
without it CMake fails outright — and that makes the CPU backend a
dynamically loaded shared library. arcint links llama.cpp statically as a
subproject (`CMakeLists.txt:165`, `target_link_libraries(arcint_core PUBLIC
llama OpenCL::OpenCL)`), so DL backends do not fit the current link model
without rework. The floor was the cheap correct answer; all-variants is a
possible follow-up.

**The `.deb` does not have this problem, and that is the distinction worth
keeping.** `contrib/packaging/arcint/build-deb.sh` is run by hand on the
target trixie host, so `-march=native` there optimises for the machine that
will actually run it. The package and the image differ in *who owns the ISA
decision*: the packager, or the image. If you need the last of the CPU-side
performance on a specific host, build the `.deb` there.

The build asserts the baseline rather than trusting the flag: `objdump` over
the installed binary must show **no `zmm`** registers (AVX-512 would SIGILL
elsewhere) and **must show `ymm`** (their absence means the build fell back
to SSE-only, which would be a silent performance cliff of its own).

## The patched runtime finds itself through `$ORIGIN`

The upstream wheel's libraries are self-locating (`$ORIGIN`). The source
build's libraries carry the build tree as `DT_RUNPATH`
(`/opt/ov/temp/Linux_x86_64/tbb/lib`, `/opt/ov/bin/intel64/Release`), paths
that do not exist in the final image, so `libopenvino` would not find its
bundled `libtbb.so.12` or the plugins each other. Tier 1 sets the RUNPATH of
every patched library to `$ORIGIN` with `patchelf` -- they sit side by side in
`openvino/libs`, so that is the whole lookup -- and fails the build if any
library under the prefix still points outside it. No `ld.so.conf.d` entry is
needed, which keeps the image on the deb's discipline: the nightly runtime
never shadows anything on the system's search path.

The `.deb` had the same defect until `marfrit-openvino +p25-2` (2026-10-07):
it worked on the host that built it, where the build tree exists, and nowhere
else. Its recipe now sets `$ORIGIN` the same way.

## The GPU runtime

The image takes Fedora's `intel-compute-runtime` (26.18.38308.4 with IGC
2.34.4 in Fedora 43, 2026-10-07). The numbers in this repository were
measured on compute-runtime 26.27.39122.11 with IGC 2.38.2 (Intel's release
packages, on Debian trixie). Debian trixie ships no Intel compute runtime at
all, which is one reason the image is Fedora-based.

Served from these images (`measured-here`, 2026-10-07, tier 1 and tier 2
built from this tree in a privileged Debian container with podman, the A770
bound in): the coder (`--engine llama`, its production GGUF, MTP 4, q8_0:q4_0
KV) answered the capital check and decoded the 500-token long answer at
43.4 t/s with 42.7 % of drafts accepted -- the same as the `.deb` on the
host (43.3-43.5 t/s, 42.7 %). That image predates 0.7.8: the `.deb` has
since gained llama.cpp patches 0042-0048, among them 0047, which raised the
coder's decode on the A770 by 14-15 % (`contrib/llama.cpp/README.md`). An
image built from the 0.7.8 tree carries the same patches; it has not been
re-measured from a container.

**No image is published yet.** `.github/workflows/build-toolbox.yml` has
not been dispatched on this repository; build the two tiers locally as
above, or dispatch the workflow from a fork.

## What CI covers, and what it cannot

Covered, no GPU required:

- the full device-free unit ladder (`ctest -L unit`), run inside the tier-2 build;
- the tier-1 gates: the pin fetch, the whole patch series under
  `git apply --check`, and the shipped `libopenvino` carrying the level;
- the tier-2 gates: the llama.cpp tarball's sha256, its patch series, the
  RPATH/`ldd` probes, and `nm` on the installed binary for the libllama
  symbols (the `--help` text advertises `--engine llama` whether or not the
  backend was compiled in, so the usage string proves nothing);
- a stub + `/props` smoke test against the **final** image, before the
  moving channel tag is published. The smoke test has to run inside the
  container — the stub binds `127.0.0.1` in the container's network
  namespace, so a host-side probe connects to nothing.

Not covered, by design: anything that touches a real card. Card-requiring
acceptance cells live in the test ladder (`docs/design-0.3.1-test-ladder.md`)
and need a card. A green container build means the engine compiles against
the patched runtime and serves without one. It is not a claim about served
quality or throughput.
