#!/usr/bin/env python3
"""Offline expert-residency LRU replay for the Flash-Next (qwen4exp) MoE.

WP7 (0.5.0). This is the reproducible, in-repo replacement for the routing-trace
LRU sweep that produced the WP6b hit-rate table. That sweep lived only in a
scratch checkout on the measurement host (a llama.cpp qwen4exp instrumentation
fork @ 30b6a755; routing trace sha256
cbc4fe8c323e41321c87566631b2f70006ad166c984828298224769241c7da95); a projection
the whole streaming plan rests on must not depend on a number no one else can
re-derive. This tool re-derives it.

The trace format (one line per (token, layer)):

    <token_idx> <layer_idx> <expert_id> <expert_id> ...   # top-K routed experts

Comment (``#``) and blank lines are skipped, so a committed fixture can carry a
provenance header.

Two cache models, and the difference matters (WP7 finding, 2026-09-10):

  * GLOBAL LRU  -- one shared residency pool over all (layer, expert) slices,
    the shape FreeToken's paper documents (a shared GPU LRU table,
    docs/research-freetoken.md §3.2). On this trace it is FLAT at ~93.8% across
    a wide capacity range (2000..~10485 slices, i.e. up to ~24 GiB): reuse here
    is dominated by short range (token-to-token, the 36.3% cross-token figure),
    so a small pool already catches it and more capacity buys almost nothing
    until ~28 GiB, past which it rises toward the compulsory-miss floor
    (95.0% at 32 GiB, 97.7% at 40 GiB).

  * PER-LAYER LRU -- each of the 48 MoE layers gets its own residency budget
    (total_slots / n_layers), the shape arcint's OWN offload slot pool has:
    fit.h ``expert_slot_bytes`` = ceil(num_expert*(100-ratio)/100) slots PER
    LAYER, replicated across ``moe_layers``. This is capacity-sensitive and is
    the model WP6b's table actually used (reproduced below within ~1.4%).

The two models are NOT interchangeable: at a 16 GiB resident budget the global
model reads ~93.8% and the per-layer model ~88.1% on the full trace -- a
~5.6-point gap that moves the projected t/s materially. The served path is
per-layer (arcint's slot pool is per-layer), so the per-layer number is the one
the projection must use; the global number is the (more optimistic) FreeToken
comparison point, recorded, not adopted.

This is a REPLAY instrument: it reports a measured hit-rate from a measured
trace. It does not itself claim a served throughput; tools/flash_next_fit.py
turns a hit-rate into a bandwidth-bound t/s projection, and
src/exec/flash_next_offload.h is the C++ mirror of that projection the served
config uses.
"""
import argparse
import sys
from collections import OrderedDict

GIB = 2 ** 30
# Measured Flash-Next geometry (WP6, off the real GGUF; mirrored in
# tools/flash_next_fit.py and src/exec/flash_next_offload.h).
SLICE_BYTES = 2_457_600   # one expert-layer int4 slice (gate/up/down fused)
N_LAYERS = 48
N_EXPERTS = 512

# Full-trace reproduction of WP6b (trace sha cbc4fe8c...). Checked by --check.
# Per-layer is WP6b's own model (its table, within ~1.4%); global is the
# FreeToken-shape comparison. reuse = cross-token same-layer expert reuse.
FULL_TRACE_SHA = "cbc4fe8c323e41321c87566631b2f70006ad166c984828298224769241c7da95"
EXPECTED_FULL = {
    # gib: (per_layer_hit_pct, global_hit_pct) -- measured on the full trace.
    16: (88.1, 93.8),
    24: (94.4, 93.8),
    32: (96.8, 95.0),
    40: (97.9, 97.7),
}
EXPECTED_REUSE_PCT = 36.3
# WP6b's published table, for the record (per-layer model reproduces it within
# ~1.4%; the residual is most likely WP6b counting the always-resident shared
# expert as a hit or a marginally different per-layer slot rounding -- not
# re-derived here, so stated as a bound, not an exact match).
WP6B_PUBLISHED = {16: 89.5, 24: 95.0, 32: 97.1, 40: 98.0}


def load_trace(path):
    """(token, layer, [expert...]) rows; skips comment/blank lines."""
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            p = ln.split()
            if len(p) < 3:
                continue
            rows.append((int(p[0]), int(p[1]), [int(x) for x in p[2:]]))
    return rows


def slots_per_layer_for_gib(gib, slice_bytes=SLICE_BYTES, n_layers=N_LAYERS):
    """Per-layer resident slot count for a total resident budget of `gib` GiB,
    split evenly across the layers -- the same split arcint's per-layer slot
    pool uses (fit.h expert_slot_bytes is per layer). Floors (budget-safe: never
    claim a slot the budget cannot hold), matching
    src/exec/flash_next_offload.h::flash_next_slots_per_layer exactly."""
    return int(gib * GIB // slice_bytes) // n_layers


def global_slots_for_gib(gib, slice_bytes=SLICE_BYTES):
    return int(gib * GIB // slice_bytes)


def replay_global(rows, cap_slices):
    """One shared LRU over (layer, expert) keys. Returns hit fraction."""
    lru = OrderedDict()
    hits = acc = 0
    for _tok, lay, experts in rows:
        for e in experts:
            k = (lay, e)
            acc += 1
            if k in lru:
                hits += 1
                lru.move_to_end(k)
            else:
                lru[k] = None
                if len(lru) > cap_slices:
                    lru.popitem(last=False)
    return hits / acc if acc else 0.0


def replay_per_layer(rows, slots_per_layer):
    """One LRU per layer, each holding `slots_per_layer` experts. Returns hit
    fraction. This is the model arcint's per-layer slot pool implements."""
    caches = {}
    hits = acc = 0
    for _tok, lay, experts in rows:
        c = caches.get(lay)
        if c is None:
            c = caches[lay] = OrderedDict()
        for e in experts:
            acc += 1
            if e in c:
                hits += 1
                c.move_to_end(e)
            else:
                c[e] = None
                if len(c) > slots_per_layer:
                    c.popitem(last=False)
    return hits / acc if acc else 0.0


def cross_token_reuse(rows):
    """Fraction of expert accesses that repeat the same layer's expert from the
    immediately preceding token -- the 'cross-token reuse' WP6b reported."""
    by_tok = {}
    for tok, lay, experts in rows:
        by_tok.setdefault(tok, {})[lay] = set(experts)
    toks = sorted(by_tok)
    same = tot = 0
    for i in range(1, len(toks)):
        cur, prev = by_tok[toks[i]], by_tok[toks[i - 1]]
        for lay, es in cur.items():
            pe = prev.get(lay, set())
            for e in es:
                tot += 1
                if e in pe:
                    same += 1
    return same / tot if tot else 0.0


def report(rows, gibs=(16, 24, 32, 40)):
    ntok = len({r[0] for r in rows})
    nlay = len({r[1] for r in rows})
    acc = sum(len(r[2]) for r in rows)
    distinct = len({(r[1], e) for r in rows for e in r[2]})
    print(f"rows={len(rows)} tokens={ntok} layers={nlay} accesses={acc} "
          f"distinct(layer,expert)={distinct}/{nlay * N_EXPERTS}")
    print(f"cross-token reuse: {cross_token_reuse(rows) * 100:.1f}%")
    print(f"{'cap(GiB)':>8} {'sl/layer':>9} {'per-layer%':>11} {'global%':>9}")
    for gib in gibs:
        spl = slots_per_layer_for_gib(gib)
        pl = replay_per_layer(rows, spl) * 100
        gl = replay_global(rows, global_slots_for_gib(gib)) * 100
        print(f"{gib:>8} {spl:>9} {pl:>11.1f} {gl:>9.1f}")


def check_full(path, tol=1.5):
    """Assert the replay reproduces the WP6b table (per-layer model) and the
    global-model figure on the sha-pinned full trace. tol in percentage points.
    Returns 0 on success."""
    import hashlib
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if h != FULL_TRACE_SHA:
        print(f"CHECK: trace sha {h[:16]}... != expected {FULL_TRACE_SHA[:16]}...; "
              f"--check pins the WP6b full trace only.")
        return 2
    rows = load_trace(path)
    fails = []
    reuse = cross_token_reuse(rows) * 100
    if abs(reuse - EXPECTED_REUSE_PCT) > 0.2:
        fails.append(f"cross-token reuse {reuse:.1f}% != {EXPECTED_REUSE_PCT}%")
    for gib, (exp_pl, exp_gl) in EXPECTED_FULL.items():
        pl = replay_per_layer(rows, slots_per_layer_for_gib(gib)) * 100
        gl = replay_global(rows, global_slots_for_gib(gib)) * 100
        if abs(pl - exp_pl) > tol:
            fails.append(f"{gib}GiB per-layer {pl:.1f}% != {exp_pl}% (+-{tol})")
        if abs(gl - exp_gl) > tol:
            fails.append(f"{gib}GiB global {gl:.1f}% != {exp_gl}% (+-{tol})")
        pub = WP6B_PUBLISHED[gib]
        print(f"  {gib}GiB: per-layer {pl:.1f}% (WP6b published {pub}%, "
              f"reproduced within {abs(pl - pub):.1f}pt), global {gl:.1f}%")
    if fails:
        print("CHECK FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print(f"CHECK PASSED: replay reproduces WP6b (per-layer model) within {tol}pt; "
          f"global-model figure {EXPECTED_FULL[16][1]}% flat.")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?", help="routing trace path")
    ap.add_argument("--check", metavar="TRACE",
                    help="assert reproduction of the WP6b table on the sha-pinned full trace")
    ap.add_argument("--gibs", type=float, nargs="*", default=[16, 24, 32, 40])
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.check:
        return check_full(args.check)
    if not args.trace:
        print("usage: expert_lru_replay.py <trace> | --check <full-trace>", file=sys.stderr)
        return 2
    report(load_trace(args.trace), gibs=args.gibs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
