#!/usr/bin/env python3
"""Stands in for cells/served_check.py in ab_selftest.py.

    fake_served_check.py <service> <binary> <gguf> <device> <assets>

<binary> is a file whose name is a comma-separated list of decode rates
("100,101"; ab.py hashes the file, so it must exist): the n-th
run with the same <binary> string prints the n-th (the last repeats), counted
in the directory named by $AB_FAKE_STATE. A rate of "x" prints a failed
answer gate and exits 1. A rate written "37.3:4b3a6d8c" also prints the
decode answer line with that md5 (others get "aaaaaaaa"), as the real check
does since 1.2.6.
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
rate, _, md5 = rate.partition(":")
print(f"ACCEPTANCE-METRIC decode {rate} t/s")
if md5 or any(":" in r for r in rates):
    print(f"ACCEPTANCE-DECODE decode md5 {md5 or 'aaaaaaaa'} tokens 300 accept 65.0 verifies 116 verify_s 6.90")
print("  texts md5 aaaaaaaa")
