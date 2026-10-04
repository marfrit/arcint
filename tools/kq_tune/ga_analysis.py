#!/usr/bin/env python3
"""Score microbenchmark GA runs against the exhaustive ground truth.

  ga_analysis.py exh-all.json ga-all-s1.json [ga-all-s2.json ...]

Ground truth: each genome's normalized fitness from the exhaustive run. For a
GA run, the genomes it evaluated, in order, are looked up in that table (so
both are judged on the same measurements, not the GA's own noisy re-runs).
Reported per run: the best genome found, its exhaustive rank, its gap to the
optimum, the evaluations until within 1 % and 2 % of the optimum. Random
search at the same budget: the same statistics over 10,000 random orders of
the exhaustive table (sampling without replacement).
"""
import json, random, statistics, sys

exh = json.load(open(sys.argv[1]))
truth = {e['key']: e['norm'] for e in exh['evals'] if e['norm'] is not None}
ranked = sorted(truth.values())
opt = ranked[0]
print(f'exhaustive: {len(truth)} genomes, optimum {opt:.4f} '
      f'({min(truth, key=truth.get)}), median {statistics.median(ranked):.4f}')

def first_within(seq, tol):
    for i, v in enumerate(seq, 1):
        if v <= opt * (1 + tol):
            return i
    return None

budgets = []
for path in sys.argv[2:]:
    ga = json.load(open(path))
    seq = [truth.get(e['key']) for e in ga['evals']]
    seq = [v for v in seq if v is not None]
    if not seq:
        print(f'{path}: no genome of this run is in the exhaustive table'); continue
    best = min(seq)
    rank = ranked.index(best) + 1
    budgets.append(len(seq))
    print(f'{path}: {len(seq)} evaluations, best {best:.4f} (rank {rank} of {len(ranked)}, '
          f'+{100 * (best / opt - 1):.2f} %), within 1 % after {first_within(seq, 0.01)}, '
          f'within 2 % after {first_within(seq, 0.02)}')

rng = random.Random(0)
vals = list(truth.values())
for n in sorted(set(budgets)):
    gaps, w1 = [], []
    for _ in range(10000):
        s = rng.sample(vals, n)
        gaps.append(min(s) / opt - 1)
        w1.append(first_within(s, 0.01) is not None)
    print(f'random search, {n} evaluations: median gap +{100 * statistics.median(gaps):.2f} %, '
          f'P(within 1 %) {sum(w1) / len(w1):.2f}')
