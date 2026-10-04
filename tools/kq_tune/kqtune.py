#!/usr/bin/env python3
"""Parameter search for the B60 2D K-quant GEMMs (contrib/llama.cpp 0013, hooks of 0014).

  kqtune.py exhaustive <type> <out.json>
  kqtune.py ga <type> <out.json> [--seed S] [--budget N] [--pop P]

Genome: TM (fp16 token tile), WG (sub-groups a work-group), AT (activation
read rows), KSYNC (blocks between work-group barriers), and for q4_K I8 (the
int8 kernel; TM and AT do not apply to it and are canonicalized).

Fitness (TUNE set): the dense 27B's own shapes of that type at 512 tokens,
test-backend-ops perf us/run (conversion included), weighted by how many
tensors of that shape the model has. A fixed reference genome is re-measured
every REF_EVERY evaluations and each result is divided by the nearest
reference (clock and thermal drift). Lower is better.

Invalid genomes are rejected before any run. A run that fails or times out
scores None; MAX_FAILS consecutive failures abort. Results are written to the
out file after every evaluation (resumable: a rerun skips measured genomes).
"""
import itertools, json, os, random, re, subprocess, sys, time

BIN = os.path.join(os.environ['ARCINT_TUNE_LLAMA_BIN'], 'test-backend-ops')
SHAPES = {  # (m, k, tensors in the model)
    'q4_K': [(17408, 5120, 130), (5120, 17408, 32), (6144, 5120, 48), (10240, 5120, 24),
             (12288, 5120, 17), (5120, 6144, 17), (1024, 5120, 25)],
    'q5_K': [(5120, 6144, 48)],
    'q6_K': [(5120, 17408, 33), (10240, 5120, 24), (1024, 5120, 9)],
}
JOINT = ['q4_K', 'q5_K', 'q6_K']   # type 'all': the knobs are program-wide, shared by the three
TMS, WGS, ATS, KSS = [32, 64, 96, 128], [4, 8, 16, 32], [8, 32], [1, 2, 4, 8, 16, 32, 100000]
# the reference: 0013's defaults (one KSYNC for all kernels: 0014's 16 / 32 split is not a genome here)
REF = {'TM': 64, 'WG': 16, 'AT': 32, 'KSYNC': 16, 'I8': 1}
REF_EVERY, MAX_FAILS, TIMEOUT = 8, 5, 300
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

def canon(g, t):
    g = dict(g)
    if t not in ('q4_K', 'all'):
        g['I8'] = 0
    elif g['I8'] == 1 and t == 'q4_K':
        g['TM'], g['AT'] = 64, 32   # not used by the int8 kernel
    return g

def valid(g):
    return g['TM'] % g['AT'] == 0

def key(g):
    return f"{g['TM']},{g['WG']},{g['AT']},{g['KSYNC']},{g['I8']}"

def measure(g, t):
    """One test-backend-ops run over every weighted case of type t ('all': the three)."""
    cases = [(u, m, k, w) for u in (JOINT if t == 'all' else [t]) for m, k, w in SHAPES[u]]
    rx = '|'.join(f'type_a={u},type_b=f32,m={m},n=512,k={k},' for u, m, k, _ in cases)
    env = clean_env(GGML_OPENCL_PLATFORM=PLATFORM, GGML_OPENCL_KQ_2D_I8=str(g['I8']),
               GGML_OPENCL_KQ_2D_SHAPE=f"{g['TM']},{g['WG']},{g['AT']}",
               GGML_OPENCL_KQ_2D_OPTS=f"-DKSYNC={g['KSYNC']} -DKSYNC_I8={g['KSYNC']}")
    try:
        r = subprocess.run([BIN, 'perf', '-o', 'MUL_MAT', '-p', f'({rx})'],
                           env=env, capture_output=True, text=True, timeout=TIMEOUT)
        out = r.stdout
    except subprocess.TimeoutExpired:
        return None
    if FALLBACK.search(r.stderr) or 'B60' not in r.stderr + r.stdout:
        return None
    us = {}
    for line in out.splitlines():
        m = re.search(r'type_a=(\w+),type_b=f32,m=(\d+),n=512,k=(\d+),.*?([0-9.]+) us/run', line)
        if m:
            us[(m.group(1), int(m.group(2)), int(m.group(3)))] = float(m.group(4))
    if any((u, m, k) not in us for u, m, k, _ in cases):
        return None
    return sum(us[(u, m, k)] * w for u, m, k, w in cases)

class Evaluator:
    def __init__(self, t, path):
        self.t, self.path = t, path
        self.db = json.load(open(path)) if os.path.exists(path) else {'type': t, 'evals': [], 'refs': []}
        if self.db.get('type') != t:
            sys.exit(f"{path} holds a search of type {self.db.get('type')}, not {t}")
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
        os.replace(tmp, self.path)   # an interrupt never leaves a torn file

    def refresh(self):
        r = measure(canon(REF, self.t), self.t)
        if r is not None:
            self.ref = r
            self.db['refs'].append({'at': self.n, 'us': r, 'time': time.time()})

    def __call__(self, g):
        g = canon(g, self.t)
        if not valid(g):
            return None
        k = key(g)
        if k in self.seen:
            return self.seen[k]['norm']
        if self.n % REF_EVERY == REF_EVERY - 1:
            self.refresh()
        us = measure(g, self.t)
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

def space(t):
    for tm, wg, at, ks, i8 in itertools.product(TMS, WGS, ATS, KSS, [0, 1] if t in ('q4_K', 'all') else [0]):
        g = canon({'TM': tm, 'WG': wg, 'AT': at, 'KSYNC': ks, 'I8': i8}, t)
        if valid(g):
            yield g

def exhaustive(ev):
    done = set()
    for g in space(ev.t):
        k = key(g)
        if k not in done:
            done.add(k)
            ev(g)

def ga(ev, seed, budget, pop_n, noseed=False):
    rng = random.Random(seed)
    genes = {'TM': TMS, 'WG': WGS, 'AT': ATS, 'KSYNC': KSS, 'I8': [0, 1] if ev.t in ('q4_K', 'all') else [0]}
    def rand():
        while True:
            g = canon({n: rng.choice(v) for n, v in genes.items()}, ev.t)
            if valid(g):
                return g
    def fit(g):
        f = ev(g)
        return float('inf') if f is None else f
    def mutate(g, p=0.25):
        while True:
            h = dict(g)
            for n, v in genes.items():
                if rng.random() < p:
                    i = v.index(h[n]) if h[n] in v else 0
                    h[n] = v[max(0, min(len(v) - 1, i + rng.choice([-1, 1])))] if rng.random() < 0.7 else rng.choice(v)
            h = canon(h, ev.t)
            if valid(h):
                return h
    def cross(a, b):
        while True:
            h = canon({n: (a if rng.random() < 0.5 else b)[n] for n in genes}, ev.t)
            if valid(h):
                return h
    pop = ([] if noseed else [canon(REF, ev.t)]) + [rand() for _ in range(pop_n - (0 if noseed else 1))]
    scored = [(fit(g), g) for g in pop]
    gen = 0
    while ev.n < budget and gen < 3 * budget // pop_n:   # cached children cost no evaluation
        gen += 1
        scored.sort(key=lambda x: x[0])
        nxt = [g for _, g in scored[:2]]   # elitism
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
    mode, t, path = sys.argv[1], sys.argv[2], sys.argv[3]
    args = sys.argv[4:]
    opt = lambda n, d: int(args[args.index(n) + 1]) if n in args else d
    ev = Evaluator(t, path)
    if mode == 'exhaustive':
        exhaustive(ev)
    else:
        ga(ev, opt('--seed', 1), opt('--budget', 60), opt('--pop', 10), '--noseed' in args)
    best = min((e for e in ev.db['evals'] if e['norm'] is not None), key=lambda e: e['norm'])
    print('BEST', best['key'], best['norm'], 'evals', ev.n)
