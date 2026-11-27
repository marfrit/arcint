#!/usr/bin/env python3
"""Compare two GGML_TEST_FA_PORT_DUMP directories: the external compare of patch 0223's red of record.

test-backend-ops writes, for each flash-attention port case, <dir>/<hash>.f32 (the backend's output, raw
little-endian f32) and a line "<hash> <floats> <vars>" in <dir>/index.txt (the file is appended to). Run the
binary once with the key split on and once off, each into its own empty directory, then:

    fa_port_dump_compare.py SPLIT_DIR UNSPLIT_DIR [--bound 1e-5]

Cases are paired by hash. Exit 1, one line per problem, on: a case on one side only, unequal float counts, a
missing or short .f32 file, non-finite values, NMSE above the bound, or a hash listed twice in one index.txt with
different lines (a re-used directory; identical repeats are reported and accepted). NMSE = sum((a-b)^2) / sum(b^2),
b the second directory. Exit 2: usage or unreadable input.
"""
import argparse
import math
import struct
import sys
from pathlib import Path


def read_index(d, problems, side):
    """{hash: (n, vars)}; duplicates with different lines are problems, identical ones are noted."""
    idx = {}
    seen = {}
    path = Path(d) / "index.txt"
    try:
        lines = path.read_text().splitlines()
    except OSError as e:
        raise SystemExit("cannot read %s: %s" % (path, e))
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(" ", 2)
        if len(parts) < 2 or not parts[1].isdigit():
            problems.append("%s: malformed index line: %r" % (side, line))
            continue
        h, n, key = parts[0], int(parts[1]), parts[2] if len(parts) > 2 else ""
        if h in seen:
            if seen[h] == line:
                print("note: %s: hash %s listed again, identical line" % (side, h))
            else:
                problems.append("%s: hash %s listed twice with different lines" % (side, h))
            continue
        seen[h] = line
        idx[h] = (n, key)
    return idx


def read_floats(d, h, n, problems, side):
    p = Path(d) / (h + ".f32")
    try:
        raw = p.read_bytes()
    except OSError as e:
        problems.append("%s: %s: cannot read %s" % (side, h, e))
        return None
    if len(raw) != 4 * n:
        problems.append("%s: %s: file holds %d bytes, index says %d floats" % (side, h, len(raw), n))
        return None
    return struct.unpack("<%df" % n, raw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--bound", type=float, default=1e-5, help="largest NMSE a case may have (default 1e-5)")
    a = ap.parse_args(argv)

    problems = []
    ia = read_index(a.dir_a, problems, "A")
    ib = read_index(a.dir_b, problems, "B")
    for h in sorted(set(ia) - set(ib)):
        problems.append("%s: case only in A (%s)" % (h, ia[h][1]))
    for h in sorted(set(ib) - set(ia)):
        problems.append("%s: case only in B (%s)" % (h, ib[h][1]))

    worst = 0.0
    ncases = 0
    for h in sorted(set(ia) & set(ib)):
        na, nb = ia[h][0], ib[h][0]
        if na != nb:
            problems.append("%s: float counts differ, A %d, B %d" % (h, na, nb))
            continue
        fa = read_floats(a.dir_a, h, na, problems, "A")
        fb = read_floats(a.dir_b, h, nb, problems, "B")
        if fa is None or fb is None:
            continue
        ncases += 1
        bad = sum(1 for x in fa if not math.isfinite(x)) + sum(1 for x in fb if not math.isfinite(x))
        if bad:
            problems.append("%s: %d non-finite values" % (h, bad))
            print("%s n=%d nmse=nan maxabs=nan" % (h, na))
            continue
        num = sum((x - y) ** 2 for x, y in zip(fa, fb))
        den = sum(y * y for y in fb)
        nmse = 0.0 if num == 0 else (num / den if den > 0 else math.inf)
        mx = max((abs(x - y) for x, y in zip(fa, fb)), default=0.0)
        worst = max(worst, nmse)
        print("%s n=%d nmse=%.3g maxabs=%.3g" % (h, na, nmse, mx))
        if nmse > a.bound:
            problems.append("%s: nmse %.3g over the bound %.3g" % (h, nmse, a.bound))
    if ncases == 0 and not problems:
        problems.append("no cases in common")
    for p in problems:
        print("FAIL: " + p)
    print("FA PORT DUMP COMPARE: %s (%d cases, worst nmse %.3g)" % ("FAIL" if problems else "PASS", ncases, worst))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
