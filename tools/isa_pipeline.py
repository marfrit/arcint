#!/usr/bin/env python3
"""0221: read an IGC ISA dump (the .asm of ocloc's IGC_ShaderDumpEnable) for the software
pipelining of the loops: for every loop (a block range closed by a backward jump) and every
global or local load in it, the compute instructions (mad / dpas / mul / add on floats) between
the load's issue and the first instruction that waits on its SWSB token ($N.dst, or a sync on
it). A load whose data is first waited for in the NEXT iteration, after the whole body, is
pipelined ("next"); a load waited for with no compute in between is a stall in the making.

    tools/isa_pipeline.py <kernel .asm>

Prints one line a loop: its lines, its loads (ugm / slm), the compute count of the body, and the
gap distribution (min / median / max, and how many loads are first used only in the next
iteration). Static only: it says what the schedule allows, not what the card measures."""
import re
import sys

src = open(sys.argv[1]).read().splitlines()
lab = {}
for i, l in enumerate(src):
    m = re.match(r'^(_0_\d+|L\d+):', l)
    if m:
        lab[m.group(1)] = i

compute_re = re.compile(r'^\s*(\(W[^)]*\)\s*)?(mad|dpas|mul|add|fma)\b.*:f\b|^\s*(\(W[^)]*\)\s*)?dpas')
load_re = re.compile(r'^\s*(\(W[^)]*\)\s*)?load\.(ugm|slm)\S*\s.*\{[^}]*\$(\d+)\}')
jump_re = re.compile(r'^\s*(\(W[^)]*\)\s*)?(jmpi|goto|while)\b.*?(_0_\d+|L\d+)')

loops = []
for i, l in enumerate(src):
    m = jump_re.match(l)
    if m and m.group(3) in lab and lab[m.group(3)] < i:
        loops.append((lab[m.group(3)], i))


def waits_on(line, tok):
    return re.search(r'\$%s\.dst' % tok, line) is not None or \
        (re.search(r'\{[^}]*\$%s[,}]' % tok, line) is not None and 'sync' in line)


for a, b in loops:
    body = src[a:b + 1]
    n_comp = sum(1 for l in body if compute_re.match(l))
    if n_comp == 0:
        continue
    gaps, nxt, kinds = [], 0, {'ugm': 0, 'slm': 0}
    for j, l in enumerate(body):
        m = load_re.match(l)
        if not m or 'null:0' in l:
            continue
        kinds[m.group(2)] += 1
        tok = m.group(3)
        gap, found = 0, False
        for l2 in body[j + 1:]:
            if waits_on(l2, tok):
                found = True
                break
            if compute_re.match(l2):
                gap += 1
        if not found:
            nxt += 1
            for l2 in body[:j]:
                if waits_on(l2, tok):
                    break
                if compute_re.match(l2):
                    gap += 1
        gaps.append(gap)
    if not gaps:
        continue
    g = sorted(gaps)
    print("loop lines %d-%d: loads ugm=%d slm=%d, compute=%d, gap load->first wait min=%d median=%d max=%d, "
          "first waited in the next iteration: %d of %d" %
          (a + 1, b + 1, kinds['ugm'], kinds['slm'], n_comp, g[0], g[len(g) // 2], g[-1], nxt, len(g)))
