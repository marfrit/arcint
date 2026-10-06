#!/usr/bin/env python3
"""Device-free convergence measurement for a per-layer expert-residency policy.

Campaign: docs/campaigns/nvme-direct-expert-tier.md, entry criterion 4
(RED-C-02: "does the residency policy converge at the admission ratio a large
MoE needs"). This tool consumes a SERVED per-call routing stream (patch 0044's
`<call_seq> <layer_key> <top_k> <ids...>`, or a format-v1 trace) that is already
on disk, and reports the four convergence quantities the campaign defines --
with NO card leg. It measures nothing itself: every number is a deterministic
function of the trace it is handed.

Definitions, all in CALL-POSITION order over a regime of the served sequence:

  * seed:  the incumbent `static-splitmix64` (patch 0018, frequency-free) or a
           `static-frequency` seed calibrated on one REGIME of the trace.
  * regime: a half-open call-seq range. On the served window-004 trace the
           regimes are probe=[0,288) (load-time forwards), prefill=[288,576)
           (288 batched calls), decode=[576,end) (single-token calls) and
           corpus=[288,end) (the served request: prefill+decode).
  * hit rate over position: deciles of ACCESS position (a monotone proxy for
           token position; a batched prefill call contributes many accesses).
           `early` = decile 1, `steady` = mean of deciles 2..n-1, `tail` =
           decile n; `convergence_decile` = the first decile from which every
           later decile stays within `eps` of the final one.
  * resident-set composition vs position: (a) the SERVED static partition's
           set is a pure function of configuration, fixed at bind -> zero
           membership changes for all positions (patch 0018: no history input);
           (b) a ROLLING census recomputed from a growing call prefix ->
           `layers_changed` per power-of-two prefix and `rounds_to_plateau`.
  * eviction/thrash per 1000 routed calls: the static partition evicts nothing
           (no eviction path exists; non-resident experts take the host tier).
           The demand-warm LRU comparand's evictions are reported per 1000
           calls AND per 1000 accesses (prefill calls are batched, so per-call
           rates are not comparable across regimes).
  * fill / plateau-probe proxy: the fraction of pinned `(layer, expert)` slots
           demanded at least once, as a function of call. Under the static
           partition an expert uploads on first demand and is never freed, so
           the device-byte high-water stops rising exactly when no NEW pinned
           expert is demanded. `probe_plateau` reports the per-forward deltas
           of that demand count (48 calls = one forward across 48 layers), the
           device-free analogue of the engine's `plateaued` counter.

Evidence class: `code` for the policy arithmetic (the seeds are the patches'
own functions, imported from `tools/expert_policy_compare.py`) and
`measured-here` for every number this tool prints from a trace.
"""
import argparse
import json
import sys
from collections import OrderedDict

import expert_policy_compare as epc   # noqa: E402
import hot_set_census as hc           # noqa: E402


# ---------------------------------------------------------------- regimes

def regime_slice(calls, lo, hi):
    """Calls with `lo <= call_seq < hi` (hi=None means to the end)."""
    return [c for c in calls
            if c[0] >= lo and (hi is None or c[0] < hi)]


def access_rows(calls_subset):
    """(layer_key, [expert...]) per call, ACCESS-level: a batched call's whole
    flattened id list is kept (one routed access per id)."""
    return [(lk, ids) for _seq, lk, _top_k, ids in calls_subset]


# ------------------------------------------------------------------ seeds

def seed_splitmix(layer_keys, slots):
    """Patch 0018's incumbent resident set per layer key."""
    return epc.static_sets_splitmix64(layer_keys, slots)


def seed_census(calls, cal_lo, cal_hi, slots):
    """A frequency seed calibrated on the half-open call range."""
    counts = epc.census_from_calls(calls, cal_lo, cal_hi)
    return epc.static_sets_frequency(counts, slots)


# ----------------------------------------------------------------- metrics

def decile_hit(rows, sets, n=10, eps=0.01):
    """Hit fraction per decile of ACCESS position, plus early/steady/tail and
    the first decile from which the series stays within `eps` of the final
    value (`convergence_decile`, or None if it never settles)."""
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    total = sum(len(ids) for _lk, ids in rows)
    if total == 0:
        return {"series": [], "early": 0.0, "steady": 0.0, "tail": 0.0,
                "tail_minus_steady_pt": 0.0, "convergence_decile": None,
                "accesses": 0}
    # Bucket by the access ORDINAL, not by a running accumulator: with fewer
    # accesses than deciles an accumulator would advance several buckets per
    # access and mis-place it. `(ordinal-1)*n//total` partitions the ordinal
    # range [1, total] into n contiguous buckets and degenerates cleanly when
    # total < n (some buckets stay empty).
    sums = [0.0] * n
    accs = [0] * n
    ordinal = 0
    for lk, ids in rows:
        res = sets.get(lk, ())
        for e in ids:
            ordinal += 1
            b = (ordinal - 1) * n // total
            accs[b] += 1
            if e in res:
                sums[b] += 1
    vals = [(sums[i] / accs[i]) if accs[i] else 0.0 for i in range(n)]
    early = vals[0]
    middle = vals[1:-1] if n > 2 else vals
    steady = sum(middle) / len(middle) if middle else 0.0
    tail = vals[-1]
    conv = None
    for i in range(n):
        if all(abs(vals[j] - tail) <= eps for j in range(i, n)):
            conv = i + 1
            break
    return {"series": [{"decile": i + 1, "hit_pct": vals[i] * 100.0,
                        "accesses": accs[i]} for i in range(n)],
            "early": early, "steady": steady, "tail": tail,
            "tail_minus_steady_pt": (tail - steady) * 100.0,
            "convergence_decile": conv, "accesses": total}


def pinned_total(sets):
    return sum(len(v) for v in sets.values())


def fill_curve(calls_subset, sets):
    """Cumulative distinct pinned (layer, expert) demanded per call.

    `calls_to_last_new` is the call index of the LAST first-demand of a pinned
    expert; after it the device-byte high-water cannot rise (uploads are never
    freed under the static partition). `fill_at_end` is the fraction of pinned
    slots ever demanded -- below 1.0 when a pinned expert is never routed,
    which is normal for a frequency-free seed at a tight budget.
    """
    total = pinned_total(sets)
    if total == 0:
        return {"per_call": [], "calls_to_last_new": 0, "monotone": True,
                "total": 0, "fill_at_end": 0.0, "fill_fraction_end": 0.0}
    demanded = set()
    per_call = []
    calls_to_last_new = 0
    for idx, (_seq, lk, _tk, ids) in enumerate(calls_subset):
        res = sets.get(lk, ())
        if res:
            before = len(demanded)
            for e in ids:
                if e in res:
                    demanded.add((lk, e))
            if len(demanded) > before:
                calls_to_last_new = idx
        per_call.append([idx, len(demanded)])
    monotone = all(per_call[i][1] >= per_call[i - 1][1]
                   for i in range(1, len(per_call)))
    return {"per_call": per_call, "calls_to_last_new": calls_to_last_new,
            "monotone": monotone, "total": total, "fill_at_end": len(demanded),
            "fill_fraction_end": len(demanded) / total}


def probe_plateau(calls_subset, sets, calls_per_forward=48):
    """Device-free analogue of the engine's plateau probe.

    Splits the calls into consecutive forwards of `calls_per_forward` calls
    (48 = one forward across 48 layers), counts NEW pinned experts demanded per
    forward, and reports the deltas. `plateau` is True iff the last two deltas
    are both zero -- the engine stops once `plateaued` reaches 2. The CUMULATIVE
    fill is monotone (uploads never free), but the per-forward DELTAS need not
    be: a quiet forward can be followed by a forward that demands many new
    pinned experts, which is exactly why one quiet forward is not a plateau."""
    if calls_per_forward <= 0:
        raise ValueError(f"calls_per_forward must be > 0, got {calls_per_forward}")
    demanded = set()
    deltas = []
    for start in range(0, len(calls_subset), calls_per_forward):
        block = calls_subset[start:start + calls_per_forward]
        before = len(demanded)
        for _seq, lk, _tk, ids in block:
            res = sets.get(lk, ())
            if res:
                for e in ids:
                    if e in res:
                        demanded.add((lk, e))
        deltas.append(len(demanded) - before)
    plateau = len(deltas) >= 2 and deltas[-1] == 0 and deltas[-2] == 0
    last_new = None
    for i, d in enumerate(deltas):
        if d > 0:
            last_new = i + 1
    return {"forwards": len(deltas), "deltas": deltas,
            "last_new_forward": last_new, "plateau": plateau,
            "filled": len(demanded), "total": pinned_total(sets)}


def lru_thrash(rows, slots):
    """True demand-warm per-layer LRU comparand: hits and evictions, with the
    eviction rate per 1000 CALLS and per 1000 ACCESSES. Every insertion into a
    full cache is one eviction (the LRU victim)."""
    if slots < 0:
        raise ValueError(f"slots must be >= 0, got {slots}")
    caches = {}
    hits = evictions = acc = calls = 0
    for lk, ids in rows:
        c = caches.get(lk)
        if c is None:
            c = caches[lk] = OrderedDict()
        calls += 1
        for e in ids:
            acc += 1
            if e in c:
                hits += 1
                c.move_to_end(e)              # LRU: a hit refreshes recency
            elif slots > 0:
                if len(c) >= slots:
                    c.popitem(last=False)     # evict least-recently-used
                    evictions += 1
                c[e] = None
    return {"hits": hits, "accesses": acc, "calls": calls,
            "evictions": evictions,
            "evictions_per_1000_calls": (evictions / calls * 1000.0) if calls else 0.0,
            "evictions_per_1000_accesses": (evictions / acc * 1000.0) if acc else 0.0,
            "hit_pct": (hits / acc * 100.0) if acc else 0.0}


def rolling_prefix_series(calls_subset, slots, n_points=13):
    """Recompute a frequency seed from a growing CALL prefix and count the
    layers whose membership changed vs the previous prefix. This is the
    ROLLING-census view (what a periodic re-calibration would do) -- NOT the
    served static partition, whose set is fixed at bind."""
    if slots <= 0:
        raise ValueError(
            f"slots must be > 0 for a rolling policy (0 gives an always-empty, "
            f"falsely-plateauing set), got {slots}")
    n = len(calls_subset)
    if n == 0:
        return {"series": [], "rounds_to_plateau": None, "plateau": False}
    lengths = []
    k = 1
    while k < n:
        lengths.append(k)
        k *= 2
    lengths.append(n)
    if len(lengths) > n_points:
        keep = sorted(set(lengths[:n_points] + [n]))
        lengths = [x for x in keep if 0 < x <= n]
    prev = None
    series = []
    stable = 0
    plateau_at = None
    for L in lengths:
        counts = epc.census_from_calls(calls_subset[:L], 0, None)
        sets = epc.static_sets_frequency(counts, slots)
        changed = None
        if prev is not None:
            changed = sum(1 for lk in set(prev) | set(sets)
                          if prev.get(lk, set()) != sets.get(lk, set()))
        series.append({"prefix_calls": L, "layers_changed": changed,
                       "hot_cells": pinned_total(sets)})
        if prev is not None and changed == 0:
            stable += 1
        else:
            stable = 0
        if stable >= 1 and plateau_at is None and len(series) >= 2:
            plateau_at = series[-2]["prefix_calls"]
        prev = sets
    return {"series": series, "rounds_to_plateau": plateau_at,
            "plateau": plateau_at is not None}


# --------------------------------------------------------------- reporting

def measure(trace_path, budgets, regimes, seeds, calls_per_forward=48):
    """Run the full matrix and return a JSON-able dict."""
    calls = sorted(hc.parse_call_trace(trace_path), key=lambda c: c[0])
    layer_keys = sorted({c[1] for c in calls})
    reg_calls = {name: regime_slice(calls, lo, hi) for name, (lo, hi) in regimes.items()}
    reg_rows = {name: access_rows(v) for name, v in reg_calls.items()}
    out = {"trace": trace_path, "budgets": list(budgets),
           "regimes": {k: list(v) for k, v in regimes.items()},
           "seeds": {}, "rolling": {}, "lru": {}}
    for slots in budgets:
        seed_sets = {}
        for sname, cal in seeds.items():
            seed_sets[sname] = (seed_splitmix(layer_keys, slots) if cal is None
                                else seed_census(calls, cal[0], cal[1], slots))
        out["seeds"][str(slots)] = {}
        for rname, rows in reg_rows.items():
            per_seed = {}
            for sname, sets in seed_sets.items():
                dec = decile_hit(rows, sets)
                fill = fill_curve(reg_calls[rname], sets)
                per_seed[sname] = {
                    "calibrated_on": seeds[sname],
                    "hit": dec,
                    "fill_calls_to_last_new": fill["calls_to_last_new"],
                    "fill_fraction_end": fill["fill_fraction_end"],
                    "fill_total": fill["total"],
                    "fill_monotone": fill["monotone"],
                    "pinned_cells": pinned_total(sets),
                    "probe_plateau": probe_plateau(reg_calls[rname], sets,
                                                   calls_per_forward),
                }
            out["seeds"][str(slots)][rname] = per_seed
        out["lru"][str(slots)] = {rname: lru_thrash(rows, slots)
                                  for rname, rows in reg_rows.items()}
        out["rolling"][str(slots)] = {
            rname: rolling_prefix_series(reg_calls[rname], slots)
            for rname in regimes}
    return out


DEFAULT_REGIMES = OrderedDict([("probe", (0, 288)), ("prefill", (288, 576)),
                               ("decode", (576, None)), ("corpus", (288, None))])
DEFAULT_SEEDS = OrderedDict([("splitmix64", None), ("census-probe", (0, 288)),
                             ("census-prefill", (288, 576)),
                             ("census-decode", (576, None)),
                             ("census-mixture", (288, None))])


def _parse_regimes(pairs):
    if not pairs:
        return OrderedDict(DEFAULT_REGIMES)
    out = OrderedDict()
    for p in pairs:
        name, rng = p.split("=", 1)
        lo_s, hi_s = rng.split(":", 1)
        out[name] = (int(lo_s), None if hi_s.lower() in ("", "end", "none") else int(hi_s))
    return out


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True,
                    help="patch-0044 per-call trace (or format-v1)",)
    ap.add_argument("--slots", default="5,16,128",
                    help="comma-separated slots/layer budgets")
    ap.add_argument("--regime", action="append", default=None,
                    metavar="NAME=LO:HI",
                    help="override/add a regime; default probe/prefill/decode/corpus")
    ap.add_argument("--calls-per-forward", type=int, default=48,
                    help="calls in one load-time probe forward (default 48 = layers)")
    ap.add_argument("--out", default="",
                    help="write the full matrix JSON here (persistent path)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    budgets = [int(x) for x in args.slots.split(",")]
    regimes = _parse_regimes(args.regime)
    res = measure(args.trace, budgets, regimes, DEFAULT_SEEDS,
                  args.calls_per_forward)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, sort_keys=True)
        print(f"wrote {args.out}")
    for slots in budgets:
        print(f"### slots/layer={slots}")
        for rname in regimes:
            for sname in DEFAULT_SEEDS:
                x = res["seeds"][str(slots)][rname][sname]
                h = x["hit"]
                pp = x["probe_plateau"]
                print(f"{rname:<8} {sname:<15} "
                      f"early={h['early'] * 100:6.2f}% "
                      f"steady={h['steady'] * 100:6.2f}% "
                      f"tail={h['tail'] * 100:6.2f}% "
                      f"tail-steady={h['tail_minus_steady_pt']:+6.2f}pt "
                      f"conv@d{h['convergence_decile']} "
                      f"fill={x['fill_fraction_end'] * 100:5.1f}% "
                      f"lastnew={x['fill_calls_to_last_new']} "
                      f"plateau={pp['plateau']}")
            l = res["lru"][str(slots)][rname]
            print(f"{rname:<8} {'lru-per-layer':<15} hit={l['hit_pct']:6.2f}% "
                  f"evict={l['evictions']} "
                  f"per1kcalls={l['evictions_per_1000_calls']:10.1f} "
                  f"per1kacc={l['evictions_per_1000_accesses']:7.1f}")
        for rname in regimes:
            rr = res["rolling"][str(slots)][rname]
            print(f"rolling {rname:<8} rounds_to_plateau={rr['rounds_to_plateau']} "
                  f"plateau={rr['plateau']} "
                  f"changed={[s['layers_changed'] for s in rr['series']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
