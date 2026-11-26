#!/usr/bin/env python3
# R4: what a lost seam block (RED=13: the first 32-key block of every split >= 1 skipped) does to the six
# split cases, in NMSE against the case's own answer. A numpy model of the harness's port case: the same
# deterministic fill (det01 by logical index), scale 1/sqrt(128), the bias table [-4, 4] for distances
# below 1024 and the pad bin 0 beyond, the per-query sink 6..8, causal, window 0, n_empty cells at the end.
# q8_0 cases: the fill's amplitude 0.25 .. 2 a (head, key, 32-block) for K and V (no quantization: its
# rounding is far below the effect). Exact attention in float64 a head and a row chunk at a time.
import sys
import numpy as np

M64 = np.uint64(0xFFFFFFFFFFFFFFFF)

def det01(salt, idx):
    x = (np.uint64(salt + 1) * np.uint64(0x9E3779B97F4A7C15) + idx.astype(np.uint64)) & M64
    x ^= x >> np.uint64(30); x = (x * np.uint64(0xBF58476D1CE4E5B9)) & M64
    x ^= x >> np.uint64(27); x = (x * np.uint64(0x94D049BB133111EB)) & M64
    x ^= x >> np.uint64(31)
    return (x >> np.uint64(40)).astype(np.float64) / 16777216.0

def case(kv, nb, n_empty, q8, hides):
    D, n_head, n_kvh, n_bins = 128, 24, 8, 1025
    n_used = kv - n_empty
    kpos = np.where(np.arange(kv) < n_used, np.arange(kv), -1)
    hid = kpos.copy()
    for lo, hi in hides:
        hid[lo:hi] = -1
    qpos = n_used - nb + np.arange(nb)
    def fill(salt, ne1, ne2):
        i = np.arange(ne2 * ne1 * D, dtype=np.uint64)
        x = -1.0 + 2.0 * det01(salt, i)
        if q8 and salt in (1, 2):
            blk = (np.arange(ne2 * ne1 * (D // 32), dtype=np.uint64))
            amp = 0.25 * np.power(2.0, 3.0 * det01(salt + 16, blk))
            x = x * np.repeat(amp, 32)
        return x.reshape(ne2, ne1, D)
    q = fill(0, nb, n_head); k = fill(1, kv, n_kvh); v = fill(2, kv, n_kvh)
    sink = 6.0 + 2.0 * det01(3, np.arange(nb * n_head, dtype=np.uint64)).reshape(nb, n_head)
    bias = -4.0 + 8.0 * det01(4, np.arange(nb * n_head * n_bins, dtype=np.uint64)).reshape(nb, n_head, n_bins)
    bias[:, :, n_bins - 1] = 0.0
    num = den = 0.0
    sc = 1.0 / np.sqrt(D)
    for h in range(n_head):
        kh = h // (n_head // n_kvh)
        s = (q[h] @ k[kh].T) * sc                                   # [nb, kv]
        dist = qpos[:, None] - kpos[None, :]
        b = np.take_along_axis(bias[:, h, :], np.clip(dist, 0, n_bins - 1), axis=1)
        s = s + b
        outs = []
        for pos in (kpos, hid):
            vis = (pos[None, :] >= 0) & (dist >= 0)
            t = np.where(vis, s, -np.inf)
            m = np.maximum(t.max(axis=1), sink[:, h])
            w = np.exp(t - m[:, None])
            outs.append((w @ v[kh]) / (np.exp(sink[:, h] - m) + w.sum(axis=1))[:, None])
        num += ((outs[1] - outs[0]) ** 2).sum(); den += (outs[0] ** 2).sum()
    return num / den

def seams(kv, p):
    blocks = (kv + 31) // 32; sb = (blocks + p - 1) // p
    return [(s * sb * 32, s * sb * 32 + 32) for s in range(1, p)]

CASES = [(16384, 1024, 0, {200: 3, 320: 3}), (12288, 36, 0, {200: 3, 320: 3}), (16384, 64, 8000, {200: 3, 320: 4})]
for q8 in (False, True):
    bound = 1e-5 if q8 else 5e-4
    for kv, nb, ne, ps in CASES:
        for wave, p in ps.items():
            e = case(kv, nb, ne, q8, seams(kv, p))
            print(f"{'q8_0' if q8 else 'f16 '} kv {kv} nb {nb} n_empty {ne} wave {wave} p {p}: seam nmse {e:.3g} "
                  f"(bound {bound:g}: {'visible' if e >= 10 * bound else ('over the bound, < 10x' if e > bound else 'HIDDEN')})", flush=True)
