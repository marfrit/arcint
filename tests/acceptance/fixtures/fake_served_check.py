#!/usr/bin/env python3
"""Stands in for cells/served_check.py in ab_selftest.py.

    fake_served_check.py <service> <binary> <gguf> <device> <assets>

<binary> is a file whose name is a comma-separated list of decode rates
("100,101"; ab.py hashes the file, so it must exist): the n-th
run with the same <binary> string prints the n-th (the last repeats), counted
in the directory named by $AB_FAKE_STATE. A rate of "x" prints a failed
answer gate and exits 1.
"""
import os
import sys

binary = sys.argv[2]
state = os.path.join(os.environ["AB_FAKE_STATE"], binary.replace(",", "_").replace("/", "_"))
n = int(open(state).read()) if os.path.exists(state) else 0
open(state, "w").write(str(n + 1))
rates = os.path.basename(binary).split(",")
rate = rates[min(n, len(rates) - 1)]
print("== fake served check")
if rate == "x":
    print("  FAIL   the capital")
    sys.exit(1)
print("  ok   the capital")
print(f"ACCEPTANCE-METRIC decode {rate} t/s")
print("  texts md5 aaaaaaaa")
