#!/usr/bin/env python3
"""Rank the OpenCL kernels by device time from sd.cpp's profiling dump.

sd.cpp built with -DGGML_OPENCL_PROFILING=ON writes `cl_profiling.csv` in its
cwd when the backend frees (at process end): one row a kernel launch -- op name,
kernel name, device duration (ms), global/local/output size. This is B1's
material: the per-op device time of a card leg (`docs/milestone-1.1.0.md` B1,
`docs/imggen.md`).

    imggen_profile.py cl_profiling.csv [--top 30] [--sort op|kernel] [--min-ms 1]

The kernel launches are in execution order, so two files -- a 1-step and an
8-step run of the same configuration -- bracket the first step's extra work
(kernel compilation and graph warm-up) by subtraction.

Exit 0 on a readable dump; 2 on a usage or format error.
"""
import argparse
import collections
import csv
import sys


def read_rows(path):
    rows = []
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        header = next(r, None)
        if not header or "kernel name" not in header[1].lower():
            raise ValueError("%s: not a cl_profiling.csv (header %r)" % (path, header))
        for line in r:
            if len(line) < 3:
                continue
            try:
                ms = float(line[2])
            except ValueError:
                continue
            rows.append((line[0].strip(), line[1].strip(), ms))
    if not rows:
        raise ValueError("%s: no rows" % path)
    return rows


def summarise(rows, key):
    total = sum(r[2] for r in rows)
    agg = collections.defaultdict(lambda: [0.0, 0])
    for op, kern, ms in rows:
        k = op if key == "op" else kern
        agg[k][0] += ms
        agg[k][1] += 1
    ranked = sorted(agg.items(), key=lambda kv: -kv[1][0])
    return total, ranked


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", help="one or more cl_profiling.csv dumps")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--sort", choices=["op", "kernel"], default="op")
    ap.add_argument("--min-ms", type=float, default=0.0,
                    help="drop entries below this device time")
    a = ap.parse_args()
    try:
        rows = []
        for p in a.csvs:
            rows += read_rows(p)
        rows = [r for r in rows if r[2] >= a.min_ms]
        if not rows:
            raise ValueError("no rows at or above --min-ms %g" % a.min_ms)
        total, ranked = summarise(rows, a.sort)
        print("rows %d, device total %.3f ms, ranked by %s" % (len(rows), total, a.sort))
        for k, (ms, n) in ranked[:a.top]:
            print("  %8.3f ms  %6.2f %%  n=%-5d %s" % (ms, 100.0 * ms / total, n, k))
        print("ACCEPTANCE-METRIC cl_device_total_ms %.3f" % total)
        if len(ranked) > a.top:
            rest = sum(ms for _, (ms, _) in ranked[a.top:])
            print("  (%d more entries, %.3f ms, %.2f %%)" % (len(ranked) - a.top, rest, 100.0 * rest / total))
        return 0
    except (ValueError, FileNotFoundError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
