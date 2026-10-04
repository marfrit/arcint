#!/usr/bin/env python3
"""Compare end-to-end search runs (GA vs random) at equal budget.

  e2e_analysis.py run1.json [run2.json ...]

Per run: best normalized fitness (ref_tps / tps, lower is better) so far after
10, 20, 40, 60, 80, 100, 120 evaluations, and the best genome with its tps.
Then the 5 best distinct genomes over all runs (each a single measurement;
re-measure before believing any of them).
"""
import json, sys

marks = [10, 20, 40, 60, 80, 100, 120]
pool = {}
for path in sys.argv[1:]:
    d = json.load(open(path))
    ev = [e for e in d['evals']]
    best, curve = float('inf'), {}
    for i, e in enumerate(ev, 1):
        if e['norm'] is not None:
            best = min(best, e['norm'])
        if i in marks:
            curve[i] = best
        if e['norm'] is not None and (e['key'] not in pool or e['norm'] < pool[e['key']][0]):
            pool[e['key']] = (e['norm'], e['tps'], path)
    good = [e for e in ev if e['norm'] is not None]
    if not good:
        print(f'{path}: no successful evaluation'); continue
    b = min(good, key=lambda e: e['norm'])
    fails = sum(1 for e in ev if e['norm'] is None)
    print(f"{path}: {len(ev)} evals ({fails} failed), refs {[round(r['tps'], 1) for r in d['refs']]}")
    print('   best so far: ' + '  '.join(f'{k}:{v:.4f}' for k, v in curve.items()))
    print(f"   best {b['key']} norm {b['norm']:.4f} tps {b['tps']}")
print('top 5 distinct genomes:')
for k, (n, t, p) in sorted(pool.items(), key=lambda x: x[1][0])[:5]:
    print(f'   {k}  norm {n:.4f}  tps {t}  ({p})')
