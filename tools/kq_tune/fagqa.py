#!/usr/bin/env python3
"""Structural search over the B60's decode/verify attention kernel
(flash_attn_gqa_dpas.cl, contrib/llama.cpp 0016/0017).

  fagqa.py exhaustive <out.json>
  fagqa.py ga <out.json> [--seed S] [--budget N] [--pop P] [--noseed]
  fagqa.py gate <out.json> [--top N]

Note: KMODE reaches every type build of the kernel (GGML_OPENCL_FA_GQA_OPTS
is appended to all four), so a "direct" genome also runs quantized K direct,
while the shipped kernel stages quantized K. The fitness is f16 only; the
served q8_0 configuration is measured end to end, not here.

Genome: the kernel's structural switches, which are source alternatives
selected at compile time (docs/campaigns/kernel-autotune-ga.md, "code-level
genetic optimization"): SIDES (sub-groups sharing an 8-row tile), KMODE (K
read straight from the cache or staged in local memory), BK (keys a staged
tile), and the host's split size for this kernel (KVPS,
GGML_OPENCL_FA_GQA_KV_PER_SPLIT). Further genes are added as the kernel
grows switches (GENES below).

Fitness (TUNE set): one attention layer at the dense 27B's geometry (24 query
heads on 4, head size 256), f16 K/V, test-backend-ops perf us/run, at 4, 6
and 8 query rows and 32k and 131k keys; the geometric mean of the six. A
fixed reference genome (REF: 0016/0017's structure) is re-measured every REF_EVERY
evaluations and each result divided by the latest reference. Lower is
better.

Gate (`gate`): the best genomes are run through FLASH_ATTN_EXT at head sizes
256 and 128 with the kernel taking every row count it can
(GGML_OPENCL_FA_GQA_MIN_ROWS=1); a genome that fails any case is out.
q8_0 K/V is not in the fitness: test-backend-ops stores quantized K/V as SoA
and reconstructs it per call, which swamps the kernel; the served
configuration is measured end to end instead (llama-bench, the agent).

Results are written after every evaluation (resumable).
"""
import itertools, json, math, os, random, re, subprocess, sys, time

BIN = os.path.join(os.environ['ARCINT_TUNE_LLAMA_BIN'], 'test-backend-ops')
PLATFORM = os.environ.get('ARCINT_TUNE_PLATFORM', '0')   # ggml-opencl's platform index of the B60
GENES = {
    'SIDES': [1, 2, 4],
    'KMODE': ['direct', 'staged'],
    'BK':    [16, 32, 64],
    'KVPS':  [64, 128, 256, 512, 1024, 2048],
    'SSPLIT': [0, 1],   # GQA_S_SPLIT: the sides of a tile split Q K^T and exchange S
}   # GQA_PF (V prefetch, a gene of the 2026-10-04 search) was removed from the kernel by 0019
REF = {'SIDES': 2, 'KMODE': 'direct', 'BK': 32, 'KVPS': 128, 'SSPLIT': 0}   # 0016/0017's structure (the 2026-10-04 reference); the shipped defaults are 0018's
CASES = [(kv, nb) for kv in (32768, 131072) for nb in (4, 6, 8)]
REF_EVERY, MAX_FAILS, TIMEOUT = 8, 5, 600
# 'kernel compile error': a type or head-size variant that fails to build is dropped
# silently and its cases run on the split kernel
FALLBACK = re.compile(r'ggml_opencl: [^\n]*(ignored|not built|does not fit|kernel compile error)')

def clean_env(**over):
    """The caller's environment without any GGML_OPENCL_FA_* switch, plus `over`."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('GGML_OPENCL_FA_')}
    env.update(over)
    return env

def genome_env(g):
    opts = ('-DGQA_K_DIRECT' if g['KMODE'] == 'direct' else '-DGQA_K_STAGED') + f" -DBK={g['BK']}"
    opts += ' -DGQA_S_SPLIT' if g.get('SSPLIT') else ' -DGQA_NO_S_SPLIT'   # explicit: 0018 made it the default
    return clean_env(GGML_OPENCL_PLATFORM=PLATFORM, GGML_OPENCL_FA_GQA_SIDES=str(g['SIDES']),
                     GGML_OPENCL_FA_GQA_OPTS=opts, GGML_OPENCL_FA_GQA_KV_PER_SPLIT=str(g['KVPS']))

def norm(g):
    """One side has nothing to split: SSPLIT is meaningless at SIDES 1."""
    return dict(g, SSPLIT=0) if g['SIDES'] == 1 else g

def key(g):
    return ','.join(str(norm(g).get(n, REF[n])) for n in GENES)

def measure(g):
    rx = '|'.join(f'kv={kv},nb={nb},' for kv, nb in CASES)
    try:
        r = subprocess.run([BIN, 'perf', '-o', 'FLASH_ATTN_EXT', '-p',
                            rf'hsk=256,hsv=256,nh=4,nr23=\[6,1\],({rx}).*type_K=f16'],
                           env=genome_env(g), capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    if FALLBACK.search(r.stderr + r.stdout) or 'B60' not in r.stderr + r.stdout:
        return None
    us = {}
    for line in r.stdout.splitlines():
        m = re.search(r'kv=(\d+),nb=(\d+),.*type_K=f16.*?([0-9.]+) us/run', line)
        if m:
            us[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
    if any(c not in us for c in CASES):
        return None
    return math.exp(sum(math.log(us[c]) for c in CASES) / len(CASES))

def gate(g):
    """FLASH_ATTN_EXT at head sizes 256 and 128 with every row count on this kernel.
    A run that fell back to a default or did not run on the B60 is a failure,
    not a pass: a dropped variant would otherwise pass on the split kernel."""
    env = genome_env(g)
    env['GGML_OPENCL_FA_GQA_MIN_ROWS'] = '1'
    try:
        r = subprocess.run([BIN, 'test', '-o', 'FLASH_ATTN_EXT', '-p', r'hsk=(256|128),hsv=(256|128)'],
                           env=env, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return 'timed out', -1
    if FALLBACK.search(r.stderr + r.stdout) or 'B60' not in r.stderr + r.stdout:
        return 'fell back or not on the B60', -1
    m = re.search(r'(\d+)/(\d+) tests passed', r.stdout)
    fails = [l for l in r.stdout.splitlines() if 'FAIL' in l and 'FLASH_ATTN_EXT(' in l
             and 'logit_softcap=0.000000' in l]   # the pin's own softcap failures are not this kernel's
    return (m.group(0) if m else 'no result'), len(fails)

class Evaluator:
    def __init__(self, path):
        self.path = path
        self.db = json.load(open(path)) if os.path.exists(path) else {'evals': [], 'refs': []}
        self.seen = {e['key']: e for e in self.db['evals']}
        self.fails, self.n = 0, len(self.db['evals'])
        self.ref = self.db['refs'][-1]['us'] if self.db['refs'] else None
        if self.ref is None:
            self.refresh()
        if self.ref is None:
            sys.exit('the reference genome did not run')

    def save(self):
        tmp = self.path + '.tmp'
        json.dump(self.db, open(tmp, 'w'), indent=1)
        os.replace(tmp, self.path)

    def refresh(self):
        r = measure(REF)
        if r is not None:
            self.ref = r
            self.db['refs'].append({'at': self.n, 'us': r, 'time': time.time()})

    def __call__(self, g):
        g = norm(g)   # the key, the measured build and the stored genome agree
        k = key(g)
        if k in self.seen:
            return self.seen[k]['norm']
        if self.n % REF_EVERY == REF_EVERY - 1:
            self.refresh()
        us = measure(g)
        self.n += 1
        e = {'key': k, 'genome': g, 'us': us, 'ref': self.ref, 'norm': None if us is None else us / self.ref,
             'order': self.n, 'time': time.time()}
        self.db['evals'].append(e)
        self.seen[k] = e
        self.save()
        self.fails = self.fails + 1 if us is None else 0
        if self.fails >= MAX_FAILS:
            sys.exit(f'{MAX_FAILS} consecutive failures, last {k}')
        print(f"[{self.n}] {k} -> {us} ({e['norm']})", flush=True)
        return e['norm']

def space():
    seen = set()
    for vals in itertools.product(*GENES.values()):
        g = norm(dict(zip(GENES, vals)))
        if key(g) not in seen:
            seen.add(key(g))
            yield g

def ga(ev, seed, budget, pop_n, noseed=False):
    rng = random.Random(seed)
    def fit(g):
        f = ev(g)
        return float('inf') if f is None else f
    def mutate(g, p=0.25):
        h = dict(g)
        for n, v in GENES.items():
            if rng.random() < p:
                i = v.index(h[n]) if h[n] in v else 0
                h[n] = v[max(0, min(len(v) - 1, i + rng.choice([-1, 1])))] if rng.random() < 0.7 else rng.choice(v)
        return h
    def cross(a, b):
        return {n: (a if rng.random() < 0.5 else b)[n] for n in GENES}
    pop = ([] if noseed else [dict(REF)]) + [{n: rng.choice(v) for n, v in GENES.items()}
                                             for _ in range(pop_n - (0 if noseed else 1))]
    scored = [(fit(g), g) for g in pop]
    gen, n0 = 0, ev.n   # the budget counts this run's evaluations, also on a resumed file
    while ev.n - n0 < budget and gen < 3 * budget // pop_n:
        gen += 1
        scored.sort(key=lambda x: x[0])
        nxt = [g for _, g in scored[:2]]
        while len(nxt) < pop_n:
            a = min(rng.sample(scored, 3), key=lambda x: x[0])[1]
            b = min(rng.sample(scored, 3), key=lambda x: x[0])[1]
            nxt.append(mutate(cross(a, b)))
        scored = [(fit(g), g) for g in nxt]
        best = min(scored, key=lambda x: x[0])
        ev.db.setdefault('generations', []).append({'gen': gen, 'evals': ev.n, 'best': key(best[1]), 'norm': best[0]})
        ev.save()
        print(f'gen {gen} evals {ev.n} best {key(best[1])} {best[0]:.4f}', flush=True)

if __name__ == '__main__':
    mode, path = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    opt = lambda n, d: int(args[args.index(n) + 1]) if n in args else d
    if mode == 'gate':
        db = json.load(open(path))
        good = sorted((e for e in db['evals'] if e['norm'] is not None), key=lambda e: e['norm'])
        for e in good[:opt('--top', 5)]:
            print('GATE', e['key'], f"{e['norm']:.4f}", *gate(e['genome']), flush=True)
        sys.exit(0)
    ev = Evaluator(path)
    if mode == 'exhaustive':
        for g in space():
            ev(g)
    else:
        ga(ev, opt('--seed', 1), opt('--budget', 60), opt('--pop', 10), '--noseed' in args)
    good = [e for e in ev.db['evals'] if e['norm'] is not None]
    if good:
        best = min(good, key=lambda e: e['norm'])
        print('BEST', best['key'], best['norm'], 'evals', ev.n)
