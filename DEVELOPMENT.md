# Development

## Hardware and driver assumptions

Development and every published measurement ran on:

- Intel Arc A770 16 GB (Xe-HPG, ACM-G10) and Intel Arc Pro B60 24 GB
  (Xe2, BMG-G21), both behind the **xe** kernel driver. i915 is untested.
- Intel compute-runtime 26.27.39122.11 with IGC 2.38.2 (OpenCL), from
  Intel's release `.deb`s; the OpenCL ICD loader (`ocl-icd`).
- Linux, an x86_64 host with AVX2 for GPU work (the expert cache's CPU
  misses need AVX2, FMA and F16C). The stub backend and the unit tests
  build and run on any Linux, aarch64 included (ASan aborts at startup on
  aarch64: run sanitizer builds on x86_64).

## Dependencies

- C++20 compiler, CMake >= 3.20.
- For a serving build: a llama.cpp tree at the pinned commit (`3e4dfa8`,
  `contrib/packaging/arcint/build-deb.sh` names it with the tarball's
  sha256) with `contrib/llama.cpp/patches/*.patch` applied in order, and the
  OpenCL headers and ICD loader. llama.cpp builds as a static subproject
  with `GGML_OPENCL=ON` and the kernels embedded; configure with
  `-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF` (the default build drops the Arc
  cards silently). Configure warns when the patches are missing.
- No network at build time: `third_party/` is vendored.

## Build and test

Two ctest targets (`docs/design-0.3.1-test-ladder.md`), not one:

- **Unit**: `LABELS unit`, device-free, every commit: `arcint-test` (the
  hand-rolled harness), the HTTP round trip, the stub-only lane stress
  test, and the acceptance enumeration's own consistency checks.
- **Acceptance**: `LABELS acceptance`, needs the cards and a serving build,
  enumerated in `tests/acceptance/cells.json`; run through
  `tests/acceptance/run.py`, never bare `ctest`, because `ctest`'s own skip
  semantics would let a card-less run go green with every cell skipped.

Unit:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    ctest --test-dir build --output-on-failure -L unit   # the unit gate
    ./build/arcint --stub --port 8090 -v                  # serving skeleton, no model

Serving build:

    git -C <llama.cpp> apply <arcint>/contrib/llama.cpp/patches/*.patch
    cmake -S . -B build-ll -DCMAKE_BUILD_TYPE=Release -DARCINT_LLAMA=ON \
          -DARCINT_LLAMA_DIR=<llama.cpp> -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
          -DARCINT_GIT_SHA=$(git rev-parse --short=12 HEAD)
    cmake --build build-ll -j"$(nproc)"

Acceptance, with the cards and the GGUFs:

    cmake -S . -B build-accept -DCMAKE_BUILD_TYPE=Release -DARCINT_LLAMA=ON \
          -DARCINT_LLAMA_DIR=<llama.cpp> -DARCINT_ACCEPTANCE=ON \
          -DARCINT_ACCEPTANCE_GGUF_AGENT=<agent.gguf> \
          -DARCINT_ACCEPTANCE_GGUF_CODER=<coder.gguf> \
          -DARCINT_ACCEPTANCE_GGUF_FLASHNEXT=<flash-next shard 1>
    cmake --build build-accept
    ctest --test-dir build-accept -N -L acceptance        # lists the enumerated cells
    tests/acceptance/run.py --manifest build-accept/acceptance/run_manifest.json \
          --all --allow-skip pruefstand   # name every 77, or the run does not count

A cell's runner may print `ACCEPTANCE-METRIC <metric> <value> <unit>` lines;
run.py compares each against the cell's own `references` in `cells.json`
and fails the run on a regression unless it is named with `--allow-regress
<cell>/<metric>` (only in `--all`; a regression in `--cell` mode fails
outright). A `references: null` cell has none yet and reports its numbers
instead of gating them. `-DARCINT_ACCEPTANCE=ON` is refused without
`-DARCINT_LLAMA=ON`: a stub binary would "pass" a card cell by serving stub
bytes.

The kernels have their own ladder: `test-backend-ops` from the llama.cpp
tree (build it with `-DLLAMA_BUILD_TESTS=ON` and Adreno kernels off) on
both cards, each patch's red case, and KL as served
(`contrib/llama.cpp/README.md`). A wrong `-b` backend name skips every case
and prints OK: read the case count.

Correctness is judged at the answer (`CLAUDE.md`, DESIGN §3.4): the answers
stay right (facts, needle, task battery), mean KL against the reference is
at most 0.03 nats worse than the baseline arm's on the same card and
window, and argmax agreement drops by at most 1 point. The suites'
byte-equality checks are tripwires: a red one is reported, and a
configuration that passes the answer-level bar may become a default with it
red.

## Card windows

Measuring on the cards follows `docs/sop-card-window.md`: card identity by
PCI id, the owning unit stopped and restored, the sampler running, one fresh
process per cell, zombies swept by pid.

## Security posture

arcint binds plainly and answers everyone: **no authentication, permissive
CORS, no TLS**. It is built for a LAN or a reverse proxy that provides all
three. Do not expose it to the internet as-is.
