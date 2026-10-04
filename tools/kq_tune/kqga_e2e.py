#!/usr/bin/env python3
"""End-to-end GA for the B60 2D K-quant GEMMs, per-type programs (search harness).

  kqga_e2e.py ga     <out.json> [--seed S] [--budget N] [--pop P]
  kqga_e2e.py random <out.json> [--seed S] [--budget N]
  kqga_e2e.py eval   <out.json> <genome key> [...]   (re-measure given genomes)

Genome: per type (0 Q4_K, 1 Q5_K, 2 Q6_K) TM, WG, AT, KSYNC; Q4_K also I8 (the
int8 kernel; its TM and AT do not apply and are canonicalized). Each type gets
its own OpenCL program (GGML_OPENCL_KQ_2D_T<i>), so the space is the product
of the three: ~1.3e7 points.

Fitness (TUNE): llama-bench prompt processing, 512 tokens, the dense 27B on the
B60, -r 2: lower is better, as ref_tps / tps with a reference genome (0014's
defaults, seeded into the first population) re-measured every REF_EVERY
evaluations. The REPORT set (4,096 tokens; ubatch 1,024) is never used here.
"""
import json, os, random, re, subprocess, sys, time

BENCH = os.path.join(os.environ['ARCINT_TUNE_LLAMA_BIN'], 'llama-bench')
MODEL = os.environ['ARCINT_TUNE_MODEL']   # the dense 27B Q4_K_M GGUF
TMS, WGS, ATS, KSS = [32, 64, 96, 128], [4, 8, 16, 32], [8, 32], [1, 2, 4, 8, 16, 32, 100000]
ORD = {'TM': TMS, 'WG': WGS, 'AT': ATS, 'KSYNC': KSS, 'I8': [0, 1]}
# the reference: contrib/llama.cpp 0014's defaults (Q4_K on int8 with its barrier every 32 blocks)
REF = [{'TM': 64, 'WG': 16, 'AT': 32, 'KSYNC': 32, 'I8': 1}] + [{'TM': 64, 'WG': 16, 'AT': 32, 'KSYNC': 16, 'I8': 0}] * 2
REF_EVERY, MAX_FAILS, TIMEOUT = 8, 5, 600
PLATFORM = os.environ.get('ARCINT_TUNE_PLATFORM', '0')   # ggml-opencl's platform index of the B60

def clean_env(**over):
    """The caller's environment without any GGML_OPENCL_KQ_* switch, plus `over`:
    a stale override would silently replace a gene."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('GGML_OPENCL_KQ_')}
    env.update(over)
    return env

# the host fell back to a default: not this genome. ggml-opencl's warnings can
# start mid-line (after the kernel loader's progress dots); llama-bench prints
# them only with -v
FALLBACK = re.compile(r'ggml_opencl: [^\n]*(ignored|not built|does not fit)')

def canon(g):
    g = [dict(x) for x in g]
    g[1]['I8'] = g[2]['I8'] = 0
    if g[0]['I8'] == 1:
        g[0]['TM'], g[0]['AT'] = 64, 32
    return g

def valid(g):
    return all(x['TM'] % x['AT'] == 0 for x in g)

def key(g):
    return '|'.join(f"{x['TM']},{x['WG']},{x['AT']},{x['KSYNC']},{x['I8']}" for x in g)

def unkey(k):
    return [dict(zip(['TM', 'WG', 'AT', 'KSYNC', 'I8'], map(int, p.split(',')))) for p in k.split('|')]

def measure(g, prompt=512, ub=512):
    env = clean_env(GGML_OPENCL_PLATFORM=PLATFORM, GGML_OPENCL_KQ_2D_I8=str(g[0]['I8']))
    for i, x in enumerate(g):
        env[f'GGML_OPENCL_KQ_2D_T{i}'] = f"{x['TM']},{x['WG']},{x['AT']},{x['KSYNC']}"
    try:
        r = subprocess.run([BENCH, '-v', '-m', MODEL, '-ngl', '99', '-fa', '1', '-p', str(prompt), '-n', '0', '-r', '2', '-ub', str(ub)],
                           env=env, capture_output=True, text=True, timeout=TIMEOUT)
        out = r.stdout
    except subprocess.TimeoutExpired:
        return None
    if FALLBACK.search(r.stderr) or 'B60' not in r.stderr + r.stdout:
        return None
    m = re.search(r'pp%d\s*\|\s*([0-9.]+)' % prompt, out)
    return float(m.group(1)) if m else None

class Evaluator:
    def __init__(self, path):
        self.path = path
        self.db = json.load(open(path)) if os.path.exists(path) else {'evals': [], 'refs': []}
        self.seen = {e['key']: e for e in self.db['evals']}
        self.n, self.fails = len(self.db['evals']), 0
        self.ref = self.db['refs'][-1]['tps'] if self.db['refs'] else None
        if self.ref is None:
            self.refresh()
        if self.ref is None:
            sys.exit('the reference genome did not run')

    def refresh(self):
        r = measure(canon(REF))
        if r:
            self.ref = r
            self.db['refs'].append({'at': self.n, 'tps': r, 'time': time.time()})

    def save(self):
        tmp = self.path + '.tmp'
        json.dump(self.db, open(tmp, 'w'), indent=1)
        os.replace(tmp, self.path)   # an interrupt never leaves a torn file

    def __call__(self, g):
        g = canon(g)
        if not valid(g):
            return None
        k = key(g)
        if k in self.seen:
            return self.seen[k]['norm']
        if self.n % REF_EVERY == REF_EVERY - 1:
            self.refresh()
        tps = measure(g)
        self.n += 1
        e = {'key': k, 'tps': tps, 'ref': self.ref, 'norm': None if not tps else self.ref / tps, 'order': self.n, 'time': time.time()}
        self.db['evals'].append(e); self.seen[k] = e; self.save()
        self.fails = self.fails + 1 if not tps else 0
        if self.fails >= MAX_FAILS:
            sys.exit(f'{MAX_FAILS} consecutive failures, last {k}')
        print(f"[{self.n}] {k} -> {tps} ({e['norm']})", flush=True)
        return e['norm']

def rand_genome(rng):
    while True:
        g = canon([{n: rng.choice(v) for n, v in ORD.items()} for _ in range(3)])
        if valid(g):
            return g

def mutate(g, rng, p=0.15):
    while True:
        h = [dict(x) for x in g]
        for x in h:
            for n, v in ORD.items():
                if rng.random() < p:
                    i = v.index(x[n]) if x[n] in v else 0
                    x[n] = v[max(0, min(len(v) - 1, i + rng.choice([-1, 1])))] if rng.random() < 0.75 else rng.choice(v)
        h = canon(h)
        if valid(h):
            return h

def cross(a, b, rng):
    # by role: each type's block comes whole from one parent
    return canon([dict((a if rng.random() < 0.5 else b)[i]) for i in range(3)])

def ga(ev, seed, budget, pop_n, noseed=False):
    rng = random.Random(seed)
    fit = lambda g: (lambda f: float('inf') if f is None else f)(ev(g))
    pop = ([] if noseed else [canon(REF)]) + [rand_genome(rng) for _ in range(pop_n - (0 if noseed else 1))]
    scored = [(fit(g), g) for g in pop]
    gen = 0
    while ev.n < budget and gen < 4 * budget // pop_n:
        gen += 1
        scored.sort(key=lambda x: x[0])
        nxt = [g for _, g in scored[:3]]   # elitism
        while len(nxt) < pop_n:
            a = min(rng.sample(scored, 3), key=lambda x: x[0])[1]
            b = min(rng.sample(scored, 3), key=lambda x: x[0])[1]
            c = cross(a, b, rng) if rng.random() < 0.8 else [dict(x) for x in a]
            nxt.append(mutate(c, rng))
        scored = [(fit(g), g) for g in nxt]
        best = min(scored, key=lambda x: x[0])
        ev.db.setdefault('generations', []).append({'gen': gen, 'evals': ev.n, 'best': key(best[1]), 'norm': best[0],
                                                     'mean': sum(s for s, _ in scored if s != float('inf')) / max(1, sum(1 for s, _ in scored if s != float('inf')))})
        ev.save()
        print(f'gen {gen} evals {ev.n} best {key(best[1])} {best[0]:.4f}', flush=True)

def random_search(ev, seed, budget):
    rng = random.Random(seed)
    ev(canon(REF))
    while ev.n < budget:
        ev(rand_genome(rng))

if __name__ == '__main__':
    mode, path, args = sys.argv[1], sys.argv[2], sys.argv[3:]
    opt = lambda n, d: int(args[args.index(n) + 1]) if n in args else d
    ev = Evaluator(path)
    if mode == 'ga':
        ga(ev, opt('--seed', 1), opt('--budget', 120), opt('--pop', 12), '--noseed' in args)
    elif mode == 'random':
        random_search(ev, opt('--seed', 1), opt('--budget', 120))
    elif mode == 'eval':
        for k in args:
            print(k, measure(unkey(k)), flush=True)
    good = [e for e in ev.db['evals'] if e['norm'] is not None]
    if good:
        b = min(good, key=lambda e: e['norm'])
        print('BEST', b['key'], b['norm'], b['tps'], 'evals', ev.n)
