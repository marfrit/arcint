#!/usr/bin/env python3
"""End-to-end GA for the A770's K-quant prefill GEMMs (the coder MoE), tile knobs.

  kqga_a770.py ga     <out.json> [--seed S] [--budget N] [--pop P] [--noseed]
  kqga_a770.py random <out.json> [--seed S] [--budget N]
  kqga_a770.py eval   <out.json> <genome key> [...]

Genome: [ID2 kernel on/off, XT, XSG | the MUL_MAT_ID tile kernel's RG, TG,
NSGM, NSGN | the plain GEMM's RG, TG, NSGM, NSGN], through ggml-opencl's
GGML_OPENCL_KQ_MM_* switches. Fitness: llama-bench pp512 on the coder, as
ref_tps / tps against the defaults re-measured every REF_EVERY evaluations.
"""
import json, os, random, re, subprocess, sys, time

BENCH = os.path.join(os.environ['ARCINT_TUNE_LLAMA_BIN'], 'llama-bench')
MODEL = os.environ['ARCINT_TUNE_MODEL']   # the coder Q4_K_M GGUF
LOCAL_MEM, MAX_WG, SG = 65536, 1024, 8
# genome: [ID2 kernel (on, XT, XSG), ID tile (RG, TG, NSGM, NSGN), plain tile (RG, TG, NSGM, NSGN)]
ORD = {'ON': [0, 1], 'XT': [8, 16, 24, 32, 48, 64, 96], 'XSG': [2, 4, 8, 16, 32],
       'RG': [1, 2, 4, 8], 'TG': [1, 2, 4, 8], 'NSGM': [1, 2, 4], 'NSGN': [1, 2, 4, 8, 16]}
BLOCKS = [['ON', 'XT', 'XSG'], ['RG', 'TG', 'NSGM', 'NSGN'], ['RG', 'TG', 'NSGM', 'NSGN']]
# the reference: contrib/llama.cpp 0014's defaults on SG 8 (0013's were ON 1, 2,4,1,1, 4,4,1,16)
REF = [{'ON': 0, 'XT': 32, 'XSG': 16}, {'RG': 2, 'TG': 4, 'NSGM': 2, 'NSGN': 2}, {'RG': 4, 'TG': 4, 'NSGM': 2, 'NSGN': 8}]
REF_EVERY, MAX_FAILS, TIMEOUT = 8, 5, 600
PLATFORM = os.environ.get('ARCINT_TUNE_PLATFORM', '1')   # ggml-opencl's platform index of the A770

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
    if g[0]['ON'] == 1:
        g[1] = dict(REF[1])        # the tile kernel is not used beside ID2
    else:
        g[0]['XT'], g[0]['XSG'] = REF[0]['XT'], REF[0]['XSG']
    return g

def tile_ok(t):
    mt, nsg = t['NSGM'] * t['RG'] * 8, t['NSGM'] * t['NSGN']
    return mt * 256 * 2 <= LOCAL_MEM and nsg * SG <= MAX_WG

def valid(g):
    x = g[0]
    id2 = (x['XT'] * 32) % (x['XSG'] * SG) == 0 and x['XSG'] * SG <= MAX_WG and x['XT'] * (256 * 2 + 16 * 4) <= LOCAL_MEM
    return id2 and tile_ok(g[1]) and tile_ok(g[2])

def key(g):
    return '|'.join(','.join(str(x[n]) for n in blk) for x, blk in zip(g, BLOCKS))

def unkey(k):
    return [dict(zip(blk, map(int, p.split(',')))) for p, blk in zip(k.split('|'), BLOCKS)]

def measure(g, prompt=512, ub=512):
    env = clean_env(GGML_OPENCL_PLATFORM=PLATFORM, GGML_OPENCL_KQ_MM_ID2=str(g[0]['ON']),
               GGML_OPENCL_KQ_MM_ID2_SHAPE=f"{g[0]['XT']},{g[0]['XSG']}",
               GGML_OPENCL_KQ_MM_ID_TILE=','.join(str(g[1][n]) for n in BLOCKS[1]),
               GGML_OPENCL_KQ_MM_TILE=','.join(str(g[2][n]) for n in BLOCKS[2]))
    try:
        r = subprocess.run([BENCH, '-v', '-m', MODEL, '-ngl', '99', '-fa', '1', '-p', str(prompt), '-n', '0', '-r', '2', '-ub', str(ub)],
                           env=env, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    if FALLBACK.search(r.stderr) or 'A770' not in r.stderr + r.stdout:
        return None
    m = re.search(r'pp%d\s*\|\s*([0-9.]+)' % prompt, r.stdout)
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
        g = canon([{n: rng.choice(ORD[n]) for n in blk} for blk in BLOCKS])
        if valid(g):
            return g

def mutate(g, rng, p=0.2):
    for _ in range(1000):
        h = [dict(x) for x in g]
        for x, blk in zip(h, BLOCKS):
            for n in blk:
                if rng.random() < p:
                    v = ORD[n]
                    i = v.index(x[n]) if x[n] in v else 0
                    x[n] = v[max(0, min(len(v) - 1, i + rng.choice([-1, 1])))] if rng.random() < 0.75 else rng.choice(v)
        h = canon(h)
        if valid(h):
            return h
    return canon(g)

def cross(a, b, rng):
    return canon([dict((a if rng.random() < 0.5 else b)[i]) for i in range(len(BLOCKS))])

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
