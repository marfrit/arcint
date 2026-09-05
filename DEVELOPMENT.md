# Development

## Hardware and driver assumptions

Development and every published measurement ran on:

- Intel Arc A770 16 GB (Xe-HPG) and Intel Arc Pro B60 24 GB (Xe2-HPG),
  both behind the **xe** kernel driver (i915 legacy paths untested and,
  for SYCL-era runtimes, known broken).
- Intel compute-runtime / Level Zero as shipped with the OpenVINO build below.
- Linux, x86_64 host for GPU work. The stub backend and the unit tests build
  and run on any Linux including aarch64 (ASan aborts at startup on aarch64 —
  run sanitizer builds on x86_64).

## Dependencies

- C++20 compiler, CMake ≥ 3.20.
- **OpenVINO 2026.4 dev** (the pinned measurement stack) including
  `openvino_tokenizers`. `-DARCINT_OPENVINO=ON` needs its CMake package;
  without it only the stub backend builds.
- No network at build time: `third_party/` is vendored.

## Build and test

Two ctest targets (docs/design-0.3.1-test-ladder.md), not one:

- **Unit** — `LABELS unit`, device-free, every commit: `arcint-test` (the
  hand-rolled harness), the HTTP round trip, the stub-only concurrency stress
  test, and the acceptance enumeration's own consistency checks.
- **Acceptance** — `LABELS acceptance`, needs a card and `-DARCINT_OPENVINO=ON`,
  enumerated in `tests/acceptance/cells.json`; run through
  `tests/acceptance/run.py`, never bare `ctest`, because `ctest`'s own skip
  semantics would let a card-less run go green with every cell skipped.

Unit:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    ctest --test-dir build --output-on-failure -L unit   # the unit gate
    ./build/arcint --stub --port 8090 -v                  # serving skeleton, no model

Acceptance, with a card and a model root:

    cmake -S . -B build-accept -DCMAKE_BUILD_TYPE=Release -DARCINT_OPENVINO=ON \
          -DARCINT_ACCEPTANCE=ON -DARCINT_ACCEPTANCE_MODEL_ROOT=/path/to/models/ov
    cmake --build build-accept
    ctest --test-dir build-accept -N -L acceptance        # lists the twelve cells
    tests/acceptance/run.py --manifest build-accept/acceptance/run_manifest.json \
          --all --allow-skip pruefstand   # name every 77, or the run does not count

A cell's runner may also print `ACCEPTANCE-METRIC <metric> <value> <unit>`
lines; run.py compares each against the cell's own `references` in
`cells.json` and fails the run on a regression unless it is named
`--allow-skip`-style with `--allow-regress <cell>/<metric>` (only in --all;
a regression in --cell mode fails outright) -- a `references: null` cell has
none yet and reports its numbers instead of gating them.

`-DARCINT_ACCEPTANCE=ON` is refused at configure time without
`-DARCINT_OPENVINO=ON`: a stub binary would "pass" a card cell by serving stub
bytes. The equivalence suite behind several cells is the contract:
byte-equality gates for cache, chunking, and speculation. A configuration that
cannot pass it does not become a default.

## Security posture

arcint binds plainly and answers everyone: **no authentication, permissive
CORS, no TLS**. It is built for a LAN or a reverse proxy that provides all
three. Do not expose it to the internet as-is.
