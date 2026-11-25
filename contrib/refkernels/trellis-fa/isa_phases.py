#!/usr/bin/env python3
"""isa_phases.py <kernel .asm from IGC's shader dump> ... : the key-block loop of an attention
kernel, cut into phases by its own landmarks, with instruction counts a phase.

Landmarks (program order): the first transposed 2D block load (K) starts S = Q K^T; the 16th
DPAS ends it; the first math.exp starts the softmax; the first transforming 2D block load (V)
starts P V; the 32nd DPAS ends it. What lies between S and the softmax is the score pass
(validity, mask, bias). The header is what precedes S back to the last instruction of the
tile's setup (the last load before the first K load that is not part of the loop is not
knowable from the text alone, so the header is cut at the nearest preceding label run of
the loop: the lines after the last pre-loop Q/sink load). Counts are static: a branch arm
that a given block does not take is still counted, so the score pass is an upper bound.

--pre-dpas N: the kernel runs N DPAS a tile before its key-block loop (the q8_0 build's
sums of Q, 8 of them), so the loop's 16th and 32nd are the (N+16)th and (N+32)th. P V then
starts at the first 2D block load after the softmax's first math.exp, whatever its form
(the q8_0 build reads V's codes as bytes). --loop-dpas M: the DPAS a key block runs (32;
the q8_0 build 40: two more a q8_0 block, the sums of its P).
"""
import re, sys

def classify(m):
    if m.startswith('dpas'): return 'dpas'
    if m.startswith('load_block2d'): return 'load2d'
    if m.startswith('load'): return 'load'
    if m.startswith('store'): return 'store'
    if m.startswith('sync'): return 'sync'
    if m.startswith('math'): return 'math'
    if m in ('mov', 'movi'): return 'mov'
    if m in ('goto', 'join', 'jmpi', 'while', 'if', 'else', 'endif', 'break', 'cont', 'brd', 'brc'): return 'cf'
    if m in ('mul', 'macl', 'mach', 'mad', 'madm', 'mulh'): return 'mul'
    if m in ('cmp', 'sel', 'csel', 'cmpn') or m.startswith('bfn'): return 'cmp/sel'
    return 'other'

COLS = ['dpas', 'load2d', 'load', 'store', 'sync', 'math', 'mov', 'mul', 'cmp/sel', 'cf', 'other']

def parse(path):
    ins = []
    for line in open(path, errors='replace'):
        t = line.strip()
        if not t or t.startswith('//') or t.endswith(':'): continue
        t = re.sub(r'^\([^)]*\)\s*', '', t)          # the predicate / (W)
        m = t.split()[0]
        w = re.search(r'\((\d+)\|', line)
        ins.append((m, int(w.group(1)) if w else 0, line))
    return ins

def main():
    args = sys.argv[1:]
    pre_d, loop_d = 0, 32
    while args and args[0] in ('--pre-dpas', '--loop-dpas'):
        if args[0] == '--pre-dpas': pre_d = int(args[1])
        else: loop_d = int(args[1])
        args = args[2:]
    for path in args:
        ins = parse(path)
        k = [i for i, x in enumerate(ins) if x[0].startswith('load_block2d') and 'd32t' in x[0]]
        v = [i for i, x in enumerate(ins) if x[0].startswith('load_block2d') and 'd16' in x[0]]
        d = [i for i, x in enumerate(ins) if x[0].startswith('dpas')]
        d = d[pre_d:]
        if pre_d and len(d) >= loop_d:
            e0 = [i for i, x in enumerate(ins) if x[0] == 'math.exp' and i > d[15]]
            v = [i for i, x in enumerate(ins) if x[0].startswith('load_block2d') and e0 and i > e0[0]]
        if not k or not v or len(d) < loop_d:
            print(path, ': landmarks missing', len(k), len(v), len(d)); continue
        s0, s1 = k[0], d[15] + 1
        e = [i for i, x in enumerate(ins) if x[0] == 'math.exp' and i >= s1]
        m0 = e[0] if e else v[0]
        p0, p1 = v[0], d[loop_d - 1] + 1
        pre = [i for i, x in enumerate(ins) if x[0].startswith('load') and i < s0 and i < k[0]]
        # the header: after the last load that belongs to the tile's setup: the last load in the
        # longest run of loads before S (the Q rows / the sink), else 60 instructions back
        h0 = max(s0 - 200, 0)
        runs = [i for i in pre if 'x16t' in ins[i][0] or 'd16x' in ins[i][0] or 'block' in ins[i][0]]
        if runs: h0 = max(h0, runs[-1] + 1)
        phases = [('header', h0, s0), ('S = Q K^T', s0, s1), ('score pass', s1, m0), ('softmax', m0, p0), ('P V', p0, p1)]
        print(path.split('/')[-1])
        print(f"  total instructions {len(ins)}; simd widths in the loop: "
              + ", ".join(f"{w}:{sum(1 for x in ins[h0:p1] if x[1] == w)}" for w in sorted({x[1] for x in ins[h0:p1]})))
        print("  %-12s %5s " % ('phase', 'insts') + " ".join("%7s" % c for c in COLS))
        tot = {c: 0 for c in COLS}; n = 0
        for name, a, b in phases:
            cnt = {c: 0 for c in COLS}
            for x in ins[a:b]: cnt[classify(x[0])] += 1
            for c in COLS: tot[c] += cnt[c]
            n += b - a
            print("  %-12s %5d " % (name, b - a) + " ".join("%7d" % cnt[c] for c in COLS))
        print("  %-12s %5d " % ('loop', n) + " ".join("%7d" % tot[c] for c in COLS))
        wide = {}
        for x in ins[h0:p1]:
            if classify(x[0]) == 'load': wide[(x[0], x[1])] = wide.get((x[0], x[1]), 0) + 1
        print("  loads by form:", ", ".join(f"{k[0]} ({k[1]}) x{n}" for k, n in sorted(wide.items())))

if __name__ == '__main__':
    main()
