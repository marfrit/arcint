#!/usr/bin/env python3
"""Diff two ARCINT_LOGITS_DUMP files record by record: the same prompt served
by two configurations (fused vs unfused, tier on vs off, one card vs the
other). Prints, per aligned record, the argmax agreement over rows, the max
|logit diff|, and the mean KL(A‖B) of the row softmaxes; then a summary.

Only the records BOTH files have are compared, in file order; once greedy
paths diverge, later records feed different tokens and the comparison
stops meaning "numerics" -- read the first prefill record first.

Usage: logits_dump_diff.py A.bin B.bin [--records N] [--rows R]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kld_served import read_dump  # noqa: E402


def load(path, rec):
    lane, past, n, rows, vocab, off = rec
    return np.fromfile(path, dtype=np.float32, count=rows * vocab, offset=off).reshape(rows, vocab)


def log_softmax(x):
    m = x.max(axis=1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--records", type=int, default=0, help="compare only the first N records (0 = all shared)")
    ap.add_argument("--from-n", type=int, default=0,
                    help="align on the LAST record with this many new tokens in EACH file (the request's prefill; the load ladder's 128..2048-token forwards and the slot-pool probe come BEFORE it and can share its length, so the first such record may be a ladder rung), then compare it and the records after it; the number of candidates is printed so an ambiguous prompt length is visible")
    a = ap.parse_args()
    ra, rb = read_dump(a.a), read_dump(a.b)
    if a.from_n:
        ca = [i for i, r in enumerate(ra) if r[2] == a.from_n]
        cb = [i for i, r in enumerate(rb) if r[2] == a.from_n]
        if not ca or not cb:
            print(f"no record with n={a.from_n} in A ({len(ca)}) or B ({len(cb)})"); return 1
        ia, ib = ca[-1], cb[-1]
        print(f"aligned: A record {ia} of {len(ca)} with n={a.from_n}, B record {ib} of {len(cb)} (the last each)")
        ra, rb = ra[ia:], rb[ib:]
    n = min(len(ra), len(rb))
    if a.records:
        n = min(n, a.records)
    print(f"records: A {len(ra)}  B {len(rb)}  compared {n}")
    worst_kl = 0.0; worst_diff = 0.0; agree_all = 0; rows_all = 0; rc = 0
    for i in range(n):
        (la, pa, na, rowsa, va, _), (lb, pb, nb, rowsb, vb, _) = ra[i], rb[i]
        if (rowsa, va) != (rowsb, vb) or pa != pb:
            print(f"#{i}: shape/past mismatch A past={pa} rows={rowsa} vocab={va} | B past={pb} rows={rowsb} vocab={vb}")
            rc = 1; break
        A, B = load(a.a, ra[i]), load(a.b, rb[i])
        agree = int((A.argmax(axis=1) == B.argmax(axis=1)).sum())
        diff = float(np.abs(A - B).max())
        la_, lb_ = log_softmax(A.astype(np.float64)), log_softmax(B.astype(np.float64))
        kl = float((np.exp(la_) * (la_ - lb_)).sum(axis=1).mean())
        print(f"#{i}: past={pa} n={na} rows={rowsa} | argmax agree {agree}/{rowsa} | max|diff| {diff:.4e} | KL(A||B) mean {kl:.4e}"
              f" | argmax A {A.argmax(axis=1)[:8].tolist()} B {B.argmax(axis=1)[:8].tolist()}")
        worst_kl = max(worst_kl, kl); worst_diff = max(worst_diff, diff); agree_all += agree; rows_all += rowsa
    if rows_all:
        print(f"SUMMARY: argmax agreement {agree_all}/{rows_all}, max|diff| {worst_diff:.4e}, worst mean KL {worst_kl:.4e}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
