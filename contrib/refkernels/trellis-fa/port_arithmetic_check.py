#!/usr/bin/env python3
"""port_arithmetic_check.py — the port kernel's two arithmetic tricks, checked
without a device: (1) the bias run read on a bin-fastest f32 table - one
sub-group block read at (d0-15) plus the lane shuffle, against a direct per-bin
read, for linear groups and at both clamped ends; (2) the sink accounting - a
per-lane partial sum reduced once with the sink term in lane 0, against the
reference's `exp(sink - m)` in the denominator.

Both are `code` from flash_attn_port.cl (the run/read and the epilogue) and
`measured-here` by this script's run. It is not the kernel: it is the arithmetic
the kernel holds, on the same bin and lane layout, so a sign or an off-by-one in
the trick is caught before a card run.
"""
import random

QBINS = 16          # keys a group (KT)
NL = 16             # lanes a sub-group
LANES = range(NL)


def bias_run(table_row, d0):
    """flash_attn_port.cl's run read: lane j's key has distance d0-j, the run is
    read at (d0-15) with lane = position, and lane j takes lane (15-j)'s value."""
    if d0 - (QBINS - 1) >= len(table_row) - 1:      # whole group at the pad bin
        return [0.0] * QBINS
    if QBINS - 1 <= d0 <= len(table_row) - 1:
        start = d0 - (QBINS - 1)
        read = [table_row[start + i] for i in LANES]      # the block read
        return [read[NL - 1 - j] for j in LANES]          # the shuffle
    # mixed end: the lane's own clamped bin
    out = []
    for j in LANES:
        d = d0 - j
        out.append(table_row[min(max(d, 0), len(table_row) - 1)])
    return out


def bias_direct(table_row, d0):
    """the reference's per-key read: clamp(dist, 0, E-1), straight from the row."""
    return [table_row[min(max(d0 - j, 0), len(table_row) - 1)] for j in LANES]


def sink_accounting(scores, sink2):
    """l per lane, reduced once, + exp2(sink - m) in lane 0 (flash_attn_port.cl's
    epilogue), against the reference's denominator with the sink as a key."""
    m_ref = max([sink2] + list(scores))
    l_ref = sum(2.0 ** (s - m_ref) for s in scores) + 2.0 ** (sink2 - m_ref)
    # the kernel: each lane holds its own keys' p for one tile row; here one key a lane
    l_lane = [2.0 ** (scores[l] - m_ref) for l in LANES]
    l_kernel = sum(l_lane) + 2.0 ** (sink2 - m_ref)      # lane 0 adds the sink once
    return l_ref, l_kernel


def main():
    random.seed(2026)
    nb = 1025                       # extent 1024 + the pad bin, reversed-free
    fails = 0
    cases = 0
    for trial in range(20000):
        table_row = [random.uniform(-4.0, 4.0) for _ in range(nb)]
        # the pad bin (extent) is zero bias after the sink is unfolded
        table_row[nb - 1] = 0.0
        d0 = random.choice([random.randint(15, nb - 1), random.randint(0, 14), random.randint(nb - 1, nb + 40)])
        a = bias_run(table_row, d0)
        b = bias_direct(table_row, d0)
        cases += 1
        if any(abs(x - y) > 1e-6 for x, y in zip(a, b)):
            fails += 1
            if fails <= 5:
                print(f"FAIL run d0={d0}: {[round(x,3) for x in a]} vs {[round(x,3) for x in b]}")
    # the sink: random scores on 16 lanes, a sink in the range of the scores
    for _ in range(5000):
        scores = [random.uniform(-20.0, 20.0) for _ in LANES]
        sink2 = random.uniform(-20.0, 20.0)
        lr, lk = sink_accounting(scores, sink2)
        cases += 1
        if abs(lr - lk) > 1e-9 * max(1.0, abs(lr)):
            fails += 1
            if fails <= 10:
                print(f"FAIL sink: ref {lr} kernel {lk}")
    # the two checks can fail: the red variants below must be caught
    table_row = [random.uniform(-4.0, 4.0) for _ in range(nb)]
    d0 = nb // 2
    read = [table_row[d0 - (QBINS - 1) + i] for i in LANES]
    no_shuffle = [read[j] for j in LANES]                # lane j keeps lane j's value
    ok_direct  = bias_direct(table_row, d0)
    if not any(abs(x - y) > 1e-6 for x, y in zip(no_shuffle, ok_direct)):
        print("FAIL: the run check cannot fail (the no-shuffle variant agrees)")
        fails += 1
    scores = [0.0] * NL
    sink2 = 0.0
    l_ref, l_kernel = sink_accounting(scores, sink2)
    l_all_lanes = sum(2.0 ** 0.0 for _ in LANES) + NL * 2.0 ** (sink2)
    if abs(l_ref - l_all_lanes) < 1e-9:
        print("FAIL: the sink check cannot fail (sink in every lane agrees)")
        fails += 1
    print(f"{cases} checks, {fails} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
