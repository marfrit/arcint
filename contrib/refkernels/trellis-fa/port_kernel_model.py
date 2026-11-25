#!/usr/bin/env python3
"""port_kernel_model.py - flash_attn_port.cl's control flow and softmax, modelled a lane at a
time and checked against a direct masked softmax, without a device.

What it models, statement for statement from the kernel's key-block loop: the group classes
(1 linear, 2 empty, 0 mixed), the block skip, the linear path's compares, the general path's
mask read, the bias run read (one block read at bin d0-15 plus the lane shuffle) and the
clamped read, the online softmax (block maxima, alpha, l a lane with the sink's term in
lane 0, p to f16), and O = O alpha + P V. What it does not model: the DPAS and the 2D block
reads (S and V are given), register allocation, and anything the compiler does.

The inputs are built as the op's test cases build them (tests/test-backend-ops.cpp, the
port's cases): positions with an optional ring seam and an empty tail, the mask and the
classes derived from them. Red: three bent kernels (the window skip decided from the tile's last
row, a mixed group taken as linear, the sink's term in every lane) must each be caught.

`measured-here` by its run on the recipe host; `code` in what it transcribes.
"""
import sys
import numpy as np

M_INIT = -3.0e38
LOG2E = 1.4426950408889634
QR, BK = 8, 32


def build(n_kv, n_q, wrap, window, n_empty, n_bins, seed):
    rng = np.random.default_rng(seed)
    n_used = n_kv - n_empty
    shift = 0 if wrap == 0 else n_used // 2 + (24 if wrap == 1 else 8)
    kpos = np.array([-1 if j >= n_used else (j + shift) % n_used for j in range(n_kv)])
    qpos = np.array([n_used - n_q + i for i in range(n_q)])
    cf = np.zeros((n_kv + 15) // 16, dtype=int)
    for g in range(len(cf)):
        ks = kpos[g * 16:min(g * 16 + 16, n_kv)]
        lin = all(ks >= 0) and all(ks[1:] == ks[:-1] + 1)
        cf[g] = 1 if lin else (2 if all(ks < 0) else 0)
    rf = np.ones((n_q + 7) // 8, dtype=int)
    dist = qpos[:, None] - kpos[None, :]
    vis = (kpos[None, :] >= 0) & (dist >= 0) & ((window == 0) | (dist < window))
    mask = np.where(vis, 0.0, -np.inf)
    table = rng.uniform(-4, 4, (n_q, n_bins)); table[:, n_bins - 1] = 0.0   # one head; the pad bin is zero
    S = rng.normal(0, 0.5, (n_q, n_kv))            # scale q.k, natural-log units (the op test: std 0.33, the bias +-4)
    V = rng.uniform(-1, 1, (n_kv, 4))
    sink = rng.uniform(3, 6, n_q)
    return dict(n_kv=n_kv, n_q=n_q, window=window, n_bins=n_bins, kpos=kpos, qpos=qpos, cf=cf, rf=rf,
                mask=mask, table=table, S=S, V=V, sink=sink, vis=vis, dist=dist)


def direct(c):
    """the op: softmax over the visible keys and the sink, bias by clamped distance"""
    b = c['table'][np.arange(c['n_q'])[:, None], np.clip(c['dist'], 0, c['n_bins'] - 1)]
    s = np.where(c['vis'], c['S'] + b, -np.inf)
    m = np.maximum(s.max(axis=1), c['sink'])
    p = np.exp(s - m[:, None]); den = p.sum(axis=1) + np.exp(c['sink'] - m)
    return (p @ c['V']) / den[:, None], den


def kernel(c, bend=None):
    n_kv, n_q, window, n_bins = c['n_kv'], c['n_q'], c['window'], c['n_bins']
    kps, qps, cfl, rfl, tb = c['kpos'], c['qpos'], c['cf'], c['rf'], c['table']
    out = np.zeros((n_q, c['V'].shape[1])); visited = 0
    lane = np.arange(16)
    for q0 in range(0, n_q, QR):
        rg_lin = rfl[q0 // QR] != 0
        nrow = min(QR, n_q - q0)
        rowi = [min(q0 + r, n_q - 1) for r in range(QR)]
        qpr = [qps[x] for x in rowi]
        m = [LOG2E * c['sink'][rowi[r]] if r < nrow else M_INIT for r in range(QR)]
        # the sink's term: 1 in lane 0 (the bent kernel puts it in every lane)
        l = [np.where(((lane == 0) | (bend == 'sink-all')) & (r < nrow), 1.0, 0.0) for r in range(QR)]
        qmin, qmax = qpr[0], qpr[0] + nrow - 1
        o = np.zeros((QR, c['V'].shape[1]))
        for t0 in range(0, n_kv, BK):
            glin, gdead, gk0, skip = [], [], [], True
            for gb in range(2):
                kg = t0 + gb * 16
                kgc = min(kg, n_kv - 1)
                cf = cfl[kgc >> 4] if kg < n_kv else 2
                if bend == 'mixed-as-linear' and cf == 0:
                    cf = 1
                k0 = kps[kgc]; cnt = min(16, n_kv - kg)
                dead = cf == 2
                if cf == 1 and rg_lin:
                    behind = window > 0 and qmin - (k0 + cnt - 1) >= window
                    if bend == 'window-of-last-row':
                        behind = window > 0 and qmax - (k0 + cnt - 1) >= window
                    dead = k0 > qmax or behind
                glin.append(cf == 1 and rg_lin); gdead.append(dead); gk0.append(k0); skip = skip and dead
            if skip:
                continue
            visited += 1
            sc = np.full((2, QR, 16), M_INIT)
            for gb in range(2):
                kg = t0 + gb * 16
                kk = np.minimum(kg + lane, n_kv - 1)
                live = (kg + lane < n_kv) & (not gdead[gb])
                kpl = kps[kk]
                for r in range(QR):
                    row = rowi[r]
                    d = qpr[r] - kpl
                    if glin[gb]:
                        vis = (r < nrow) & live & (d >= 0) & ((window == 0) | (d < window)); mv = 0.0
                    else:
                        mv = c['mask'][row, kk]
                        vis = (r < nrow) & live & (mv > -6.0e4)
                        mv = np.where(np.isfinite(mv), mv, 0.0)
                    bv = np.zeros(16)
                    binc = np.where(d < 0, n_bins - 1, np.minimum(d, n_bins - 1))
                    if glin[gb]:
                        d0 = qpr[r] - gk0[gb]
                        if d0 - 15 < n_bins - 1:
                            if 15 <= d0 <= n_bins - 1:
                                tv = tb[row, d0 - 15:d0 + 1]          # the block read, lane = bin offset
                                bv = tv[15 - lane]                    # the shuffle
                            else:
                                bv = tb[row, binc]
                    else:
                        bv = tb[row, binc]
                    sc[gb, r] = np.where(vis, LOG2E * c['S'][row, kk] + mv + LOG2E * bv, M_INIT)
            alpha = np.ones(QR)
            for r in range(QR):
                mx = max(sc[0, r].max(), sc[1, r].max())
                mn = max(m[r], mx)
                alpha[r] = 2.0 ** max(m[r] - mn, -2000.0)
                m[r] = mn; l[r] = l[r] * alpha[r]
            o *= alpha[:, None]
            for gb in range(2):
                kg = t0 + gb * 16
                kk = np.minimum(kg + lane, n_kv - 1)
                for r in range(QR):
                    p = np.where(sc[gb, r] > 0.5 * M_INIT, 2.0 ** np.maximum(sc[gb, r] - m[r], -2000.0), 0.0)
                    p = p.astype(np.float16).astype(np.float64)
                    l[r] = l[r] + p
                    o[r] += p @ c['V'][kk]          # lanes past n_kv hold p = 0
        for r in range(nrow):
            lv = l[r].sum()
            out[q0 + r] = o[r] * (1.0 / lv if lv > 0 else 0.0)
    return out, visited


CASES = [  # n_kv, n_q, wrap, window, n_empty  (the op test's cases, and small ones around every edge)
    (1536, 512, 2, 1024, 0), (4096, 512, 0, 0, 2560), (2048, 1024, 1, 1024, 0), (1024, 512, 0, 1024, 0),
    (1025, 512, 0, 0, 0), (1088, 64, 2, 0, 0), (1040, 20, 0, 0, 17),
    (96, 40, 0, 0, 0), (96, 40, 1, 24, 0), (100, 13, 2, 40, 0), (160, 9, 0, 50, 60), (160, 12, 0, 0, 60), (2200, 24, 0, 0, 0),
]


def nmse(a, b):
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def main():
    fails = 0
    full = {}
    for i, (n_kv, n_q, wrap, window, n_empty) in enumerate(CASES):
        if n_kv * n_q > 300000 and '--all' not in sys.argv:
            n_q = max(9, min(n_q, 40))      # the large cases with their last 40 rows (the same key axis)
        c = build(n_kv, n_q, wrap, window, n_empty, 1025 if n_kv > 200 else 33, 100 + i)
        ref, _ = direct(c)
        got, visited = kernel(c)
        e = nmse(got, ref)
        blocks = ((n_kv + BK - 1) // BK) * ((n_q + QR - 1) // QR)
        ok = e < 1e-6
        fails += not ok
        print(f"kv {n_kv:5d} q {n_q:4d} wrap {wrap} window {window:4d} empty {n_empty:4d}: nmse {e:.2e} "
              f"blocks visited {visited}/{blocks} {'ok' if ok else 'FAIL'}")
        full[(n_kv, n_q, wrap, window, n_empty)] = c
    # the reds: each bent kernel must be caught by at least one case
    for bend in ('window-of-last-row', 'mixed-as-linear', 'sink-all'):
        caught = 0
        for key, c in full.items():
            ref, _ = direct(c)
            got, _ = kernel(c, bend)
            caught += nmse(got, ref) > 5e-4
        print(f"red {bend}: caught by {caught} case(s) {'ok' if caught else 'FAIL (not caught)'}")
        fails += caught == 0
    print("FAIL" if fails else "PASS")
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
