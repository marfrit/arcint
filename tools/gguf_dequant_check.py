#!/usr/bin/env python3
"""Compare arcint's host dequantizer (src/core/gguf_dequant.cpp) against
gguf-py's own `quants.dequantize` on a real GGUF tensor.

docs/design-gguf-native.md §3.4's "host reference" gate wants this checked
against a real Q4_K_M/Q5_K/Q6_K/Q8_0 file, not only the tiny synthetic
fixture in tests/fixtures/. arcint itself has no tensor-dump flag yet at
stage 0 (that lands with the reader's first real caller in stage 1), so
this script's job today is the comparison side: given a real .gguf and a
raw float32 dump of one of its tensors -- produced however arcint ends up
exposing that later (a `--dump-tensor NAME OUT.bin` flag is the expected
shape) -- it checks the dump against gguf-py's reference and reports where
they diverge. Until that flag exists, run tests/test_gguf.cpp's
`ARCINT_GGUF_REAL=<path>` case instead for a device-free sanity check with
no external dump required (finite values, plausible range, a checksum).

Dump format expected here: raw float32, native-endian, row-major in the
tensor's numpy shape (i.e. dims reversed from the GGUF file's ne[] order --
the same convention tools/gguf_fixture.py's reference .bin uses), exactly
`n_elements` values, no header.

Usage:
    python3 tools/gguf_dequant_check.py MODEL.gguf TENSOR_NAME ARCINT_DUMP.bin
    python3 tools/gguf_dequant_check.py MODEL.gguf --list

Needs the `gguf` package (pip install gguf, or the OpenVINO tooling venv
this repo's other tools/*.py scripts use). Exit status is 0 only if every
value matches within the tolerance (default 1e-6 relative, 1e-6 absolute
floor for near-zero values); nonzero otherwise, with the first few
differing indices printed.
"""
import argparse
import sys

import numpy as np


def list_tensors(reader):
    for t in reader.tensors:
        print(f"{t.name}\t{t.tensor_type.name}\t{list(t.shape)}\t{t.n_elements} elements")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf_path")
    ap.add_argument("tensor_name", nargs="?")
    ap.add_argument("dump_path", nargs="?")
    ap.add_argument("--list", action="store_true", help="list tensor names/types/shapes and exit")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--max-print", type=int, default=8, help="differing indices to print")
    args = ap.parse_args()

    from gguf.gguf_reader import GGUFReader
    from gguf import quants

    reader = GGUFReader(args.gguf_path)

    if args.list or not args.tensor_name:
        list_tensors(reader)
        return 0

    found = None
    for t in reader.tensors:
        if t.name == args.tensor_name:
            found = t
            break
    if found is None:
        print(f"no tensor named '{args.tensor_name}' in {args.gguf_path}", file=sys.stderr)
        return 2

    if args.dump_path is None:
        print("dump_path is required unless --list is given", file=sys.stderr)
        return 2

    reference = quants.dequantize(found.data, found.tensor_type).astype(np.float32).reshape(-1)

    dump = np.fromfile(args.dump_path, dtype=np.float32)
    if dump.size != reference.size:
        print(f"size mismatch: dump has {dump.size} floats, {args.tensor_name} "
              f"({found.tensor_type.name}) has {reference.size}", file=sys.stderr)
        return 2

    diff = np.abs(dump - reference)
    tol = args.atol + args.rtol * np.abs(reference)
    bad = np.nonzero(diff > tol)[0]

    print(f"{args.tensor_name} ({found.tensor_type.name}, {reference.size} values): "
          f"max abs diff {diff.max():.3e}, {bad.size} of {reference.size} exceed tolerance "
          f"(rtol={args.rtol}, atol={args.atol})")
    for i in bad[:args.max_print]:
        print(f"  [{i}] arcint={dump[i]!r} gguf-py={reference[i]!r} diff={diff[i]!r}")

    return 0 if bad.size == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
