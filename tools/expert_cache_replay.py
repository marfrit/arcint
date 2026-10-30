#!/usr/bin/env python3
"""Offline replay of the libllama engine's expert cache over a routing trace.

The trace comes from LLAMA_EXPERT_CACHE_TRACE=<file> (a measurement switch on the
llama tree's branch p0057-trace, not in contrib/llama.cpp/patches): a header
("ECT1", the layer count, then per layer il, n_expert, n_slots, one expert's bytes
as u64 and the slotted experts at load) and per counted decode step its tokens
and each layer's routed ids (u32 count, i32 ids).

Policies (strata as src/llama-expert-cache.cpp runs it, patches 0021 and 0042;
lfru as patch 0055 ran it, dropped from the series in 0.8.1):
  strata  per-layer usage counts decayed by 0.7 an adapt, a swap while the bank
          expert's usage beats the slot's by 1.5 (Strata's adapt);
  lfru    Infernix's score uses / (now - last + 1) per layer, the round's
          experts protected, up to --budget admissions a layer;
  glfru   the same score over one pool of frames for all layers, so a swap may
          move a frame from one layer to another (Infernix's global LFRU).
"exact" uses Infernix's clock (a round ages an expert by the layer count) and
only the step's misses as candidates. Every 6 decode tokens a plan, capped at
96 swaps; its swaps land --delay counted steps later and no plan runs while
some are in flight (the engine's two-phase non-blocking swaps). Hits count
against the residency at the step, as the engine counts them. Frames are
counted, not bytes: the layers' experts are not one size.

--bounds adds what no online policy can beat on the trace: the best fixed set per
layer in hindsight, a planner inside the same mechanism that knows the next H steps'
routing, and Belady's MIN per layer (every miss admitted at once).

Usage: expert_cache_replay.py <trace> [--budget N] [--delay N] [--cap N] [--every N]
       [--bounds]   (needs numpy)
The older replay tools for the retired engine's traces and policies are at the
tag openvino-history (docs/history.md).
"""
import struct, argparse
import numpy as np


def load(path):
    b = open(path, "rb").read()
    assert b[:4] == b"ECT1"
    o = 4
    (nl,) = struct.unpack_from("<I", b, o); o += 4
    layers = []
    for _ in range(nl):
        il, ne, ns = struct.unpack_from("<iii", b, o); o += 12
        (eb,) = struct.unpack_from("<Q", b, o); o += 8
        ex = np.frombuffer(b, "<i4", ns, o).copy(); o += 4 * ns
        layers.append(dict(il=il, n_expert=ne, n_slots=ns, eb=eb, slots=ex))
    steps = []
    while o + 4 <= len(b):
        (nt,) = struct.unpack_from("<I", b, o); o += 4
        ids = []
        for _ in range(nl):
            (c,) = struct.unpack_from("<I", b, o); o += 4
            ids.append(np.frombuffer(b, "<i4", c, o).copy()); o += 4 * c
        steps.append((nt, ids))
    return layers, steps


def simulate(layers, steps, policy, adapt_every=6, max_swaps=96, budget=4, delay=2,
             min_usage=2.0, min_gain=1.5, decay=0.7, exact=False):
    nl = len(layers); ne = layers[0]["n_expert"]
    res = np.zeros((nl, ne), bool)
    for i, L in enumerate(layers):
        res[i, L["slots"][L["slots"] >= 0]] = True
    usage = np.zeros((nl, ne)); cnt = np.zeros((nl, ne)); last = np.zeros((nl, ne))
    now = 0; since = 0; pending = []          # (land_step, [(layer, expert, resident)])
    hits = misses = 0; per_step = []
    age = float(nl) if exact else 1.0
    for si, (nt, ids) in enumerate(steps):
        keep = []
        for land, flips in pending:
            if land <= si:
                for l, e, r in flips: res[l, e] = r
            else:
                keep.append((land, flips))
        pending = keep
        now += 1
        group = []
        h_step = m_step = 0
        for l, x in enumerate(ids):
            x = x[(x >= 0) & (x < ne)]
            h = int(res[l, x].sum()); h_step += h; m_step += len(x) - h
            np.add.at(usage[l], x, 1.0)
            g = np.unique(x); group.append(g); cnt[l, g] += 1; last[l, g] = now
        hits += h_step; misses += m_step; per_step.append((h_step, m_step))
        since += nt
        if since < adapt_every:
            continue
        since = 0
        if pending:                            # no plan while swaps are in flight; the engine still decays
            if policy == "strata": usage *= decay
            continue
        plan = []                              # (gain, (layer_in, e_in), (layer_out, e_out))
        if policy == "strata":
            for l in range(nl):
                c = np.where(~res[l] & (usage[l] >= min_usage))[0]
                v = np.where(res[l])[0]
                c = c[np.argsort(-usage[l, c], kind="stable")]; v = v[np.argsort(usage[l, v], kind="stable")]
                for j in range(min(len(c), len(v))):
                    if usage[l, c[j]] < usage[l, v[j]] + min_gain: break
                    plan.append((usage[l, c[j]] - usage[l, v[j]], (l, c[j]), (l, v[j])))
            usage *= decay
        else:
            score = cnt / (age * (now - last) + 1.0)
            prot = np.zeros((nl, ne), bool)
            for l in range(nl): prot[l, group[l]] = True
            if exact:
                cand = np.zeros((nl, ne), bool)
                for l in range(nl): cand[l, group[l]] = True
                cand &= ~res
            else:
                cand = ~res & (cnt > 0)
            if policy == "lfru":
                for l in range(nl):
                    c = np.where(cand[l])[0]; v = np.where(res[l] & ~prot[l])[0]
                    nb = min(budget, len(c), len(v))
                    c = c[np.argsort(-score[l, c], kind="stable")][:nb]; v = v[np.argsort(score[l, v], kind="stable")][:nb]
                    for j in range(nb):
                        if not score[l, c[j]] > score[l, v[j]]: break
                        plan.append((score[l, c[j]] - score[l, v[j]], (l, c[j]), (l, v[j])))
            else:  # glfru
                cl, ce = np.where(cand); vl, ve = np.where(res & ~prot)
                nb = min(budget * nl, len(cl), len(vl))
                ci = np.argsort(-score[cl, ce], kind="stable")[:nb]; vi = np.argsort(score[vl, ve], kind="stable")[:nb]
                for j in range(nb):
                    a = (cl[ci[j]], ce[ci[j]]); b = (vl[vi[j]], ve[vi[j]])
                    if not score[a] > score[b]: break
                    plan.append((score[a] - score[b], a, b))
        plan.sort(key=lambda t: -t[0]); plan = plan[:max_swaps]
        if plan:
            flips = []
            for _, (li, ei), (lo, eo) in plan:
                flips.append((lo, eo, False)); flips.append((li, ei, True))
            pending.append((si + delay, flips))
    return hits / max(1, hits + misses), per_step, res.sum(1)


def bounds(layers, steps, delay=2, every=6, cap=96):
    nl = len(layers); ne = layers[0]["n_expert"]; S = len(steps)
    use = np.zeros((S, nl, ne), np.int64)
    for si, (_, ids) in enumerate(steps):
        for l, x in enumerate(ids):
            np.add.at(use[si, l], x[(x >= 0) & (x < ne)], 1)
    tot = use.sum()
    init = np.zeros((nl, ne), bool)
    for i, L in enumerate(layers): init[i, L["slots"][L["slots"] >= 0]] = True
    ns = init.sum(1)
    out = {"fill-only (the load-time profile)": (use * init).sum() / tot}
    best = np.zeros((nl, ne), bool); tu = use.sum(0)
    for l in range(nl): best[l, np.argsort(-tu[l], kind="stable")[:ns[l]]] = True
    out["best fixed set per layer, in hindsight"] = (use * best).sum() / tot
    cum = np.concatenate([np.zeros((1, nl, ne), np.int64), np.cumsum(use, 0)])
    for H in (6, 24, 96):                      # the mechanism's cadence, cap and landing; the next H steps known
        res = init.copy(); pending = []; since = 0; h = 0
        for si, (nt, _) in enumerate(steps):
            keep = []
            for land, flips in pending:
                if land <= si:
                    for l, e, r in flips: res[l, e] = r
                else: keep.append((land, flips))
            pending = keep
            h += (use[si] * res).sum(); since += nt
            if since < every: continue
            since = 0
            if pending: continue
            fut = cum[min(S, si + delay + H)] - cum[min(S, si + delay)]
            plan = []
            for l in range(nl):
                c = np.where(~res[l])[0]; v = np.where(res[l])[0]
                c = c[np.argsort(-fut[l, c], kind="stable")]; v = v[np.argsort(fut[l, v], kind="stable")]
                for j in range(min(len(c), len(v))):
                    if fut[l, c[j]] <= fut[l, v[j]]: break
                    plan.append((fut[l, c[j]] - fut[l, v[j]], l, c[j], v[j]))
            plan.sort(key=lambda t: -t[0])
            if plan[:cap]:
                flips = []
                for _, l, ei, eo in plan[:cap]: flips += [(l, eo, False), (l, ei, True)]
                pending.append((si + delay, flips))
        out["foresight planner in the mechanism, next %d steps" % H] = h / tot
    h = 0                                       # Belady's MIN per layer
    for l in range(nl):
        res = set(np.where(init[l])[0].tolist()); seq = [np.where(use[si, l] > 0)[0].tolist() for si in range(S)]
        nxt = {}; nexts = [None] * S
        for si in range(S - 1, -1, -1):
            nexts[si] = dict(nxt)
            for e in seq[si]: nxt[e] = si
        for si in range(S):
            row = seq[si]
            h += sum(int(use[si, l, e]) for e in row if e in res)
            for e in row:
                if e in res: continue
                if len(res) >= ns[l]:
                    v = max((r for r in res if r not in row), key=lambda r: nexts[si].get(r, S + 1), default=None)
                    if v is None: continue
                    res.discard(v)
                res.add(e)
    out["Belady MIN per layer"] = h / tot
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--delay", type=int, default=2)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--cap", type=int, default=96, help="swaps a plan (the engine's LLAMA_EXPERT_CACHE_MAX_SWAPS)")
    ap.add_argument("--every", type=int, default=6, help="decode tokens a plan (LLAMA_EXPERT_CACHE_ADAPT_EVERY)")
    ap.add_argument("--bounds", action="store_true")
    a = ap.parse_args()
    layers, steps = load(a.trace)
    tot = sum(L["n_slots"] for L in layers)
    print("layers %d, slots %d (%d-%d a layer), expert bytes %s, steps %d (decode %d), routed %d" % (
        len(layers), tot, min(L["n_slots"] for L in layers), max(L["n_slots"] for L in layers),
        sorted(set(L["eb"] for L in layers)), len(steps), sum(1 for s in steps if s[0] <= 8),
        sum(sum(len(x) for x in s[1]) for s in steps)))
    for pol, ex in (("strata", False), ("lfru", False), ("lfru", True), ("glfru", False), ("glfru", True)):
        h, per, occ = simulate(layers, steps, pol, delay=a.delay, budget=a.budget, exact=ex,
                               max_swaps=a.cap, adapt_every=a.every)
        print("%-6s%-6s %.2f %%  slots a layer at the end %d-%d" % (pol, " exact" if ex else "", 100 * h, occ.min(), occ.max()))
    if a.bounds:
        for k, v in bounds(layers, steps, delay=a.delay, every=a.every, cap=a.cap).items():
            print("%-50s %.2f %%" % (k, 100 * v))
