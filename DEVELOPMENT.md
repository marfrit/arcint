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

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    ctest --test-dir build --output-on-failure     # unit tests + HTTP round trip
    ./build/arcint --stub --port 8090 -v          # serving skeleton, no model

With a card and a model directory:

    cmake -S . -B build-ov -DCMAKE_BUILD_TYPE=Release -DARCINT_OPENVINO=ON
    cmake --build build-ov
    ./build-ov/arcint --model /path/to/ir-dir --device GPU.0 --port 8090
    tests/equivalence/run.sh ./build-ov/arcint /path/to/ir-dir GPU.0

The equivalence suite is the contract: byte-equality gates for cache, chunking,
and speculation. A configuration that cannot pass it does not become a default.

## Security posture

arcint binds plainly and answers everyone: **no authentication, permissive
CORS, no TLS**. It is built for a LAN or a reverse proxy that provides all
three. Do not expose it to the internet as-is.
