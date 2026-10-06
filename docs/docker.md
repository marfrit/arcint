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

## Tagging: the tag names the pin and the package level

The tier-1 tag is the OpenVINO pin plus the package level the recipe is at:

```
2026.4.0-ov71640275-marfrit-p27
```

The `ov71640275` component is the recipe's `PIN` verbatim — eight hex
characters, as `build-openvino.sh` names it. The full 40-char commit the
image was actually built from is in the `ai.arcint.ov.pin` label.

The level is a usable identity. It was not for a while — builds of
0044–0067 kept the `marfrit-p19` stamp, which is also 0003–0043's, so one
string covered several patch sets. `contrib/packaging/marfrit-openvino/patches/README.md`
discloses that era and closes it: *"A release names its package level."*
Since `+p20` the level in the tag is the package.

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

That is deliberate. This toolbox spent a review cycle carrying its own
`p15`/`ceiling0033` while the recipes had moved to `p27`; the gate means the
next time that happens the build fails in seconds instead of publishing a
base whose tag describes a runtime nobody built.

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

## Known issue resolved: `patches/0037`

An earlier revision of this toolbox refused to build past patch `0033`,
because `0037-moe-hybrid-prefill-split.patch` on `main` was corrupt: its
last hunk declared 82 new / 6 old and carried 78 / 3, with the empty
context lines stripped of their leading space — `git apply` refused the file
outright (`corrupt patch at line 436`).

That was fixed upstream in `ab19b09`, an hour after the base this PR was cut
from. The toolbox now applies the whole series with no ceiling arg. Checked
here rather than assumed: `git apply --summary` over every patch in
`contrib/packaging/marfrit-openvino/patches/` and
`contrib/llama.cpp/patches/` parses clean, and the same command against the
old base's `0037` still reports the corruption. What that check does **not**
cover is whether each patch applies to the pinned OpenVINO *tree* — that is
`git apply --check` inside the tier-1 build, which CI runs.

