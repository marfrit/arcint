# Container image

arcint can be packed as a pull-and-run container the way other Arc serving
stacks are: one image, `arcint`, carrying the engine built against llama.cpp
at the recipe's pin with `contrib/llama.cpp/patches` applied, and the
userspace GPU stack (the OpenCL ICD loader and Intel's compute runtime). The
host supplies the kernel driver and `/dev/dri`; models are mounted, never
baked in.

Until 0.7.12 there were two images: a tier-1 base carrying the patched
OpenVINO runtime (`arcint-ov`) and the engine on top. That base was retired
with the OpenVINO engine in 0.8.0 (`docs/history.md`); the published
`arcint-ov` images stay as they are.

## Building

From the repo root:

```bash
docker build -t arcint:local -f docker/Dockerfile.arcint .
```

Or dispatch `.github/workflows/build-toolbox.yml`, which builds the image
and publishes it to GHCR under the running repo's own namespace, with a
`channel` input for the moving tag. No configuration is needed to run it
from a fork: the namespace is derived from `GITHUB_REPOSITORY_OWNER`.

### Operational notes

**GHCR packages are private on first push.** To let others pull the image,
set the package visibility to public after the first push, or have them log
in with a PAT carrying `read:packages`.

## Tagging and the gated build args

The image is tagged with the build's UTC timestamp and the channel
(`:edge`); its labels carry the arcint version and the llama.cpp commit.
The Dockerfile gates its build args against the recipe: `LLAMA_COMMIT` and
`LLAMA_TARBALL_SHA256` must equal `contrib/packaging/arcint/build-deb.sh`'s,
so if the recipe and the image ever disagree the build fails in seconds
instead of publishing an image built from another base.

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
  --engine llama --gguf /models/gguf/<model>.gguf \
  --device GPU.0 --host 0.0.0.0 --port 8080
```

`--engine llama` serves the GGUF through ggml's OpenCL backend, which is why
the image carries the ICD loader and the Intel compute-runtime driver
(`ocl-icd`, `intel-compute-runtime`). See `docs/llama-engine.md` for the
engine's flags. `docker/docker-compose.example.yaml` is the same thing in
compose form; a cold OpenCL kernel cache recompiles the kernels at every
start, so keep `GGML_OPENCL_KERNEL_CACHE_DIR` on a persistent volume.

### The three services as the project runs them (0.7.12)

The command lines below are the ones the project's own services use;
`docker/docker-compose.example.yaml` carries all three as compose profiles.
Paths are placeholders under the read-only `/models` mount.

**The coder** (Qwen3.6-27B-A3B coder GGUF, the A770 or any 16 GB card):

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

**Qwen3.8-Flash-Next** (the IQ3_XXS GGUF through an expert cache on a 24 GB
card, the rest in host memory; for the IQ2_XS file: `--llama-expert-cache
13400`):

```bash
# environment: GGML_OPENCL_BANK_STREAM=1 LLAMA_EXPERT_CACHE_CPU_MISSES=1
--engine llama --gguf /models/<flash-next-iq3_xxs-00001-of-00002>.gguf \
--served-model-name qwen3.8-flash-next --device GPU.0 --n-ctx 131072 \
--llama-cpu-moe 48 --llama-threads 8 --llama-expert-cache 12000 \
--llama-reserve-cells 20480 --llama-expert-profile /models/<expert-profile>.bin \
--llama-mtp 3 --llama-mtp-min-p 0.5 --llama-mtp-gguf /models/<flash-next-mtp-head>.gguf --llama-kv q8_0 --queue-timeout 30
```

Flash-Next needs more from the host than the other two:
- **pinned host memory:** the expert bank (28.72 GiB for IQ3_XXS, 20.32 for IQ2_XS) lives in
  USM host memory that the driver pins. The host's TTM page limit must
  allow it (`/sys/module/ttm/parameters/pages_limit`; the default is half
  of system RAM, which on a 64 GB host leaves little room beside the bank),
  and the container needs no extra capability for it;
- **the GGUF on fast local storage:** experts not in the bank or the cache
  are read from the file; with the file on a ZFS pool the record has a
  Flash-Next quant decoding at 4-5 t/s (README, current numbers);
- **a VRAM margin:** 13,400 MiB of expert slots with `--llama-reserve-cells
  20480` (0052) peaks at 23,787 MiB of the card's 24,480 (0.7.12) (the earlier 11,300 MiB
  without the reserve peaked at ~24.2 GB). Do not run anything else on that
  card;
- **CPU threads:** with `LLAMA_EXPERT_CACHE_CPU_MISSES=1` (0056) a
  coordinator and 7 workers (`LLAMA_EXPERT_CACHE_CPU_MISS_THREADS`, 8 in
  all) spin at 100 % while Flash-Next decodes; the coder decoding on a second
  card of the same host loses ~1 % (`measured-here`).

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
will actually run it. Until 0.7.12 it did not: the recipe exports
`SOURCE_DATE_EPOCH`, ggml reads that as a reproducible build and turns
`GGML_NATIVE` off, so every package's ggml-cpu since 0.5.5 was SSE-only.
With Flash-Next's CPU misses that cost 7.7 % on one answer (45.6 against
49.4 t/s); without them the package matched a native dev build on six
answers (`measured-here`, `docs/campaigns/flash-next-llama-engine.md` on the
`qfndev` branch). The recipe now asks for
`GGML_NATIVE=ON` and checks the CMake cache. The package and the image differ in *who owns the ISA
decision*: the packager, or the image. If you need the last of the CPU-side
performance on a specific host, build the `.deb` there.

The build asserts the baseline rather than trusting the flag: `objdump` over
the installed binary must show **no `zmm`** registers (AVX-512 would SIGILL
elsewhere) and **must show `ymm`** (their absence means the build fell back
to SSE-only, which would be a silent performance cliff of its own).

## The GPU runtime

The image takes Fedora's `intel-compute-runtime` (26.18.38308.4 with IGC
2.34.4 in Fedora 43, 2026-10-07). The numbers in this repository were
measured on compute-runtime 26.27.39122.11 with IGC 2.38.2 (Intel's release
packages, on Debian trixie). Debian trixie ships no Intel compute runtime at
all, which is why the image is Fedora-based.

Served from the two-tier images of 0.7.x (`measured-here`, 2026-10-07, both
built from this tree in a privileged Debian container with podman, the A770
bound in): the coder (`--engine llama`, its production GGUF, MTP 4, q8_0:q4_0
KV) answered the capital check and decoded the 500-token long answer at
43.4 t/s with 42.7 % of drafts accepted -- the same as the `.deb` on the
host (43.3-43.5 t/s, 42.7 %). That image predates 0.7.8: the `.deb` has
since gained llama.cpp patches 0042-0048, among them 0047, which raised the
coder's decode on the A770 by 14-15 % (`contrib/llama.cpp/README.md`). An
image built from the 0.7.8 tree carries the same patches; it has not been
re-measured from a container.

**Published images**: 0.8.0's single image (2026-10-15,
`.github/workflows/build-toolbox.yml` on the tag v0.8.0):
`ghcr.io/marfrit/arcint:20261015T184539`, `:edge` (digest
`sha256:aa74d93ffc35…`). The last two-tier set is 0.7.12's
(`ghcr.io/marfrit/arcint-ov:2026.4.0-ov71640275-marfrit-p27` and
`ghcr.io/marfrit/arcint:20261015T141859`). The workflow's smoke test is
device-free, and no GPU run of a published image is recorded. GHCR publishes
a new package as private; pulling it from another account needs its
visibility set by the owner.

## What CI covers, and what it cannot

Covered, no GPU required:

- the full device-free unit ladder (`ctest -L unit`), run inside the build;
- the gates: the llama.cpp tarball's sha256, its patch series, the `ldd`
  probe, and `nm` on the installed binary for the libllama symbols (the `--help` text advertises `--engine llama` whether or not the
  backend was compiled in, so the usage string proves nothing);
- a stub + `/props` smoke test against the **final** image, before the
  moving channel tag is published. The smoke test has to run inside the
  container — the stub binds `127.0.0.1` in the container's network
  namespace, so a host-side probe connects to nothing.

Not covered, by design: anything that touches a real card. Card-requiring
acceptance cells live in the test ladder (`docs/design-0.3.1-test-ladder.md`)
and need a card. A green container build means the engine compiles against
the patched llama.cpp and serves without a card. It is not a claim about served
quality or throughput.
