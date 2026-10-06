#!/usr/bin/env python3
"""Offline comparison of per-layer expert-residency policies on a routed trace.

0.5.2 VENICE, the policy half. The census instrument
(``tools/hot_set_census.py``) answers WHICH experts route; this tool answers
whether a seed built from that census beats the incumbent seed, and by how
much, at the same per-layer budget.

Three policies, all per layer, all at ``slots`` residents per layer:

  * ``static-splitmix64`` -- the INCUMBENT (patch 0018, DESIGN Section 7.0.2ae):
    for each layer the resident set is the ``slots`` experts with the smallest
    ``splitmix64`` rank over ``(seed, layer_key, expert)``, fixed once at
    ``bind()`` and independent of request history. Replicated here exactly from
    the patch's C++ -- same seed constant, same rank key, tie-break on the
    ASCENDING expert id -- so the comparison measures the real incumbent and not
    an approximation of it (evidence class: code).
  * ``static-frequency`` -- the CANDIDATE: for each layer the ``slots`` experts
    with the most routed accesses in a CALIBRATION census, ties on the
    ASCENDING expert id. This is what a census-seeded static partition buys.
  * ``lru-per-layer`` -- the demand-warm PER-LAYER LRU comparand (the model
    ``tools/expert_lru_replay.py`` implements), which needs no calibration
    because it is causal: it only ever uses accesses it has already seen.

HONESTY RULE, enforced: the frequency seed is calibrated on a DIFFERENT set of
accesses than the one it is scored on. Scoring an in-sample seed would flatter
it by exactly the amount the campaign wants to measure, so the tool REFUSES a
calibration/evaluation overlap unless ``--allow-overlap`` is given, and prints
the in-sample number beside the held-out one when it is (a seed chosen from the
evaluation trace is a look-ahead, not a policy). Two ways to stay honest:

  * two traces: ``--calibrate-call A --eval-call B``;
  * one trace, split: ``--calibrate-call T --calibrate-seq-hi N --eval-call T
    --eval-seq-lo N`` (calibrate on the head, score the tail) -- the shape used
    for a served trace, where the prefill calls are the calibration census and
    the decode calls are the held-out evaluation.

``lru-per-layer`` is scored on the evaluation accesses only (it warms inside
the scored window); ``static-*`` are priced on the same accesses, so the three
numbers are like-for-like.

Input is a patch-0044 per-call trace (``<call_seq> <layer_key> <top_k> <ids...>``)
or a format-v1 trace (``<token> <layer> <expert...>``). Call traces are used
directly: the CALIBRATION side counts aggregate ids (including batched prefill
calls -- the aggregate needs no token labels), the EVALUATION side keeps only
single-``top_k`` decode calls. Layers are identified by the raw ``layer_key``
(the weight-file offset) throughout, which is the key patch 0018 ranks on.

This is a REPLAY instrument: it reports measured hit fractions from measured
traces. It claims no served throughput.
"""
import argparse
import json
import sys
from collections import OrderedDict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import hot_set_census as hc  # noqa: E402
import expert_lru_replay as lru  # noqa: E402

# Patch 0018's own constants (contrib/packaging/marfrit-openvino/patches/0018-*).
STATIC_PARTITION_SEED = 0xF2A17C0DE5EED
LAYER_MIX = 0xD6E8FEB86659FD93
MASK64 = (1 << 64) - 1
N_EXPERTS = lru.N_EXPERTS  # 512, the Flash-Next per-layer expert count


def splitmix64(x):
    """The patch's mix, verbatim (64-bit unsigned arithmetic)."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def rank_key(seed, layer_key, expert):
    """``static_partition_rank_key``: pure function of (seed, layer_key, expert)."""
    h = splitmix64((seed ^ ((layer_key * LAYER_MIX) & MASK64)) & MASK64)
    return splitmix64((h ^ expert) & MASK64)


def static_set_splitmix64(layer_key, slots, num_expert=N_EXPERTS,
                          seed=STATIC_PARTITION_SEED):
    """The incumbent's resident set for one layer: the `slots` smallest ranks,
    returned sorted by ASCENDING expert id (the patch's own tie-break)."""
    if slots <= 0:
        return []
    ranked = sorted((rank_key(seed, layer_key, e), e) for e in range(num_expert))
    return sorted(e for _h, e in ranked[:min(slots, num_expert)])


def static_sets_splitmix64(layer_keys, slots, num_expert=N_EXPERTS,
                           seed=STATIC_PARTITION_SEED):
    return {lk: set(static_set_splitmix64(lk, slots, num_expert, seed))
            for lk in layer_keys}


def static_sets_frequency(counts, slots):
    """The candidate: per layer, the `slots` most-routed experts in `counts`
    (a {(layer_key, expert): n} census), ties on the ASCENDING expert id."""
    per_layer = {}
    for (lk, e), n in counts.items():
        per_layer.setdefault(lk, []).append((n, e))
    out = {}
    for lk, items in per_layer.items():
        items.sort(key=lambda t: (-t[0], t[1]))
        out[lk] = {e for _n, e in items[:max(slots, 0)]}
    return out


def census_from_calls(calls, lo=0, hi=None):
    """Aggregate routed-access counts from a per-call trace, batched calls
    INCLUDED (the aggregate histogram needs no token labels)."""
    counts = {}
    for seq, lk, _top_k, ids in calls:
        if seq < lo or (hi is not None and seq >= hi):
            continue
        for e in ids:
            counts[(lk, e)] = counts.get((lk, e), 0) + 1
    return counts


def eval_rows_from_calls(calls, lo=0, hi=None):
    """Decode (single-`top_k`-chunk) accesses in [lo, hi) as (layer_key, ids)
    rows. A batched call is skipped: its accesses cannot be attributed to a
    token, and the evaluation is per-access, so it is counted only on the
    calibration side. Returns `(rows, skipped_calls)`."""
    rows, skipped = [], 0
    for seq, lk, top_k, ids in sorted(calls, key=lambda r: r[0]):
        if seq < lo or (hi is not None and seq >= hi):
            continue
        chunks = hc.split_topk_chunks(ids, top_k)
        if len(chunks) != 1:
            skipped += 1
            continue
        rows.append((lk, sorted(chunks[0])))
    return rows, skipped


def hit_rate(rows, sets):
    """Fraction of routed accesses whose (layer_key, expert) is resident.
    `rows` are (layer_key, [expert...]) pairs WITHOUT token labels, so a
    missing layer's set is treated as empty (the layer keeps no residents)."""
    hits = acc = 0
    for lk, experts in rows:
        res = sets.get(lk, ())
        for e in experts:
            acc += 1
            if e in res:
                hits += 1
    return (hits / acc) if acc else 0.0, hits, acc


def lru_hit_rate(rows, slots):
    """TRUE demand-warm per-layer LRU over the evaluation accesses, at the same
    per-layer budget. Causal: only prior accesses inform it, and a hit is
    PROMOTED to most-recently-used -- exactly the model
    `tools/expert_lru_replay.py::replay_per_layer` implements (an OrderedDict
    with `move_to_end` on a hit). FIFO without promotion is NOT this model and
    understates the comparand, which would flatter the static policies.
    """
    caches, hits, acc = {}, 0, 0
    for lk, experts in rows:
        c = caches.get(lk)
        if c is None:
            c = caches[lk] = OrderedDict()
        for e in experts:
            acc += 1
            if e in c:
                hits += 1
                c.move_to_end(e)          # LRU: a hit refreshes recency
            elif slots > 0:
                c[e] = None
                if len(c) > slots:
                    c.popitem(last=False)  # evict least-recently-used
    return (hits / acc) if acc else 0.0, hits, acc


def convergence_series(calls, eval_rows, slots, n_points=8, cal_lo=0, cal_hi=None):
    """Calibrate the frequency seed on progressively longer HEADS of the
    calibration window and score the fixed evaluation tail: the set-stability
    curve the campaign's convergence clause wants.

    Honesty rule, as everywhere in this tool: the head is bounded by the
    CALIBRATION window `[cal_lo, cal_hi)`, so no point of the series ever
    calibrates on an access it is scored on. Returns a list of dicts.
    """
    seqs = sorted({c[0] for c in calls
                   if c[0] >= cal_lo and (cal_hi is None or c[0] < cal_hi)})
    if not seqs:
        return []
    out = []
    prev = None
    for i in range(1, n_points + 1):
        cut = seqs[min(len(seqs) - 1, (len(seqs) * i) // n_points)]
        counts = census_from_calls(calls, lo=cal_lo, hi=cut + 1)
        seed_sets = static_sets_frequency(counts, slots)
        changed = None if prev is None else sum(1 for lk in set(prev) | set(seed_sets)
                                                if prev.get(lk, set()) != seed_sets.get(lk, set()))
        prev = seed_sets
        hr, _h, acc = hit_rate(eval_rows, seed_sets)
        out.append({"calib_calls": cut + 1, "layers_changed": changed,
                    "hit_rate": hr, "accesses": acc,
                    "hot_cells": sum(len(v) for v in seed_sets.values())})
    return out


def compare(cal_calls, cal_lo, cal_hi, eval_calls, eval_lo, eval_hi, slots,
            seed=STATIC_PARTITION_SEED, num_expert=N_EXPERTS):
    """The three-policy comparison. Returns a dict of measured numbers."""
    cal_counts = census_from_calls(cal_calls, cal_lo, cal_hi)
    eval_rows, skipped = eval_rows_from_calls(eval_calls, eval_lo, eval_hi)
    layer_keys = sorted({lk for lk, _e in cal_counts} | {lk for lk, _ids in eval_rows})
    freq_sets = static_sets_frequency(cal_counts, slots)
    split_sets = static_sets_splitmix64(layer_keys, slots, num_expert, seed)
    freq_hr, freq_h, freq_a = hit_rate(eval_rows, freq_sets)
    split_hr, split_h, split_a = hit_rate(eval_rows, split_sets)
    lru_hr, lru_h, lru_a = lru_hit_rate(eval_rows, slots)
    identical_layers = sum(1 for lk in layer_keys
                           if freq_sets.get(lk, set()) == split_sets.get(lk, set()))
    return {
        "slots_per_layer": slots, "seed": hex(seed), "n_experts": num_expert,
        "calibration": {"calls": sum(1 for c in cal_calls
                                    if c[0] >= cal_lo and (cal_hi is None or c[0] < cal_hi)),
                        "cells": len(cal_counts),
                        "accesses": sum(cal_counts.values()),
                        "seq_lo": cal_lo, "seq_hi": cal_hi},
        "evaluation": {"decode_calls": len(eval_rows),
                       "batched_calls_skipped": skipped, "accesses": freq_a,
                       "seq_lo": eval_lo, "seq_hi": eval_hi},
        # Analytic context, not a measurement: a frequency-free policy that
        # pins `slots` of `num_expert` per layer hits about slots/num_expert
        # of accesses by chance. It is printed so "the incumbent is at
        # chance" is checkable rather than asserted.
        "chance_hit_pct": (slots / num_expert) * 100.0,
        "policies": {
            "static-splitmix64": {"hit_rate": split_hr, "hits": split_h,
                                  "accesses": split_a,
                                  "pinned_cells": sum(len(v) for v in split_sets.values())},
            "static-frequency": {"hit_rate": freq_hr, "hits": freq_h,
                                 "accesses": freq_a,
                                 "pinned_cells": sum(len(v) for v in freq_sets.values())},
            "lru-per-layer": {"hit_rate": lru_hr, "hits": lru_h, "accesses": lru_a,
                              "pinned_cells": None},
        },
        "layers_where_seeds_agree": identical_layers,
        "layers": len(layer_keys),
        "frequency_minus_splitmix64_hit_pct":
            (freq_hr - split_hr) * 100.0,
        "splitmix64_minus_frequency_hit_pct":
            (split_hr - freq_hr) * 100.0,
    }


def detect_format(path, default="call"):
    """`call` or `v1`, from the file's own header marker.

    The converter writes `# source=plugin-0044-call-trace`; a format-v1 trace
    carries `# arcint routing trace v1`. A row-based guess is NOT used: in
    format v1 the third field is an expert id, which can divide the id count
    exactly and masquerade as a `top_k`, so a wrong guess would silently read
    v1 rows as calls (or the reverse). Header, else the explicit flag.
    """
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            if not s.startswith("#"):
                break
            if "plugin-0044-call-trace" in s:
                return "call"
            if "arcint routing trace v1" in s:
                return "v1"
    return default


def _load(path, fmt):
    """Load a call trace or a v1 trace."""
    if path is None:
        return None
    fmt = detect_format(path, fmt)
    if fmt == "call":
        return hc.parse_call_trace(path)
    # format v1: <token> <layer> <expert...> -> one call per row, top_k = len-2
    calls = []
    for i, (tok, lay, experts) in enumerate(lru.load_trace(path)):
        calls.append((i, lay, len(experts), list(experts)))
    return calls


def _sha256(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _ranges_overlap(cal_lo, cal_hi, eval_lo):
    """True when the calibration range reaches into the evaluation range."""
    hi = cal_hi if cal_hi is not None else (1 << 62)
    return not (eval_lo >= hi)


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate-call", default=None,
                    help="calibration corpus: patch-0044 per-call trace or v1 trace")
    ap.add_argument("--calibrate-format", choices=("call", "v1"), default="call",
                    help="used only when the file has no header marker")
    ap.add_argument("--calibrate-seq-lo", type=int, default=0)
    ap.add_argument("--calibrate-seq-hi", type=int, default=None)
    ap.add_argument("--eval-call", required=True,
                    help="evaluation corpus: patch-0044 per-call trace or v1 trace")
    ap.add_argument("--eval-format", choices=("call", "v1"), default="call",
                    help="used only when the file has no header marker")
    ap.add_argument("--eval-seq-lo", type=int, default=0)
    ap.add_argument("--eval-seq-hi", type=int, default=None)
    ap.add_argument("--slots-per-layer", type=int, required=True)
    ap.add_argument("--seed", default=hex(STATIC_PARTITION_SEED),
                    help="the incumbent's seed (patch 0018 default 0xF2A17C0DE5EED)")
    ap.add_argument("--num-expert", type=int, default=N_EXPERTS)
    ap.add_argument("--allow-overlap", action="store_true",
                    help="permit an in-sample seed; the in-sample number is then "
                         "printed beside the held-out one and flagged")
    ap.add_argument("--convergence", action="store_true",
                    help="also print the head-calibration convergence series")
    ap.add_argument("--json", default="", help="write the result JSON here")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    seed = int(args.seed, 16) if args.seed.lower().startswith("0x") \
        else int(args.seed)

    cal_path = args.calibrate_call or args.eval_call
    eval_path = args.eval_call
    cal_calls = _load(cal_path, args.calibrate_format)
    eval_calls = cal_calls if (eval_path == cal_path and args.calibrate_call is not None) \
        else _load(eval_path, args.eval_format)
    cal_lo, cal_hi = args.calibrate_seq_lo, args.calibrate_seq_hi
    eval_lo, eval_hi = args.eval_seq_lo, args.eval_seq_hi

    # honesty guard: the calibration and evaluation access sets must not overlap.
    # Two paths can be the SAME accesses (a copy or an alias), so the guard keys
    # on content, not only on the path string.
    same_accesses = cal_path == eval_path or _sha256(cal_path) == _sha256(eval_path)
    if same_accesses and _ranges_overlap(cal_lo, cal_hi, eval_lo):
        if not args.allow_overlap:
            print(f"REFUSED: calibration and evaluation are the same accesses "
                  f"(same path or identical sha256) with overlapping call ranges "
                  f"(cal < {cal_hi if cal_hi is not None else 'end'}, eval >= {eval_lo}); "
                  f"an in-sample frequency seed flatters itself. Split the trace "
                  f"(e.g. --calibrate-seq-hi N --eval-seq-lo N) or pass "
                  f"--allow-overlap to see the look-ahead number.", file=sys.stderr)
            return 2
        print("# WARNING: in-sample seed (--allow-overlap): the "
              "static-frequency number is a LOOK-AHEAD, not a policy.",
              file=sys.stderr)

    res = compare(cal_calls, cal_lo, cal_hi, eval_calls, eval_lo, eval_hi,
                  args.slots_per_layer, seed, args.num_expert)
    print(f"# calibration: {cal_path} calls={res['calibration']['calls']} "
          f"cells={res['calibration']['cells']} accesses={res['calibration']['accesses']} "
          f"seq=[{cal_lo},{cal_hi})")
    print(f"# evaluation:  {eval_path} decode_calls={res['evaluation']['decode_calls']} "
          f"batched_skipped={res['evaluation']['batched_calls_skipped']} "
          f"accesses={res['evaluation']['accesses']} seq=[{eval_lo},{eval_hi})")
    print(f"# slots/layer={args.slots_per_layer} seed={res['seed']} "
          f"layers={res['layers']} seeds_agree_on={res['layers_where_seeds_agree']} layers")
    print(f"{'policy':<20} {'hit%':>8} {'hits':>12} {'accesses':>12} {'pinned_cells':>13}")
    for name in ("static-splitmix64", "static-frequency", "lru-per-layer"):
        p = res["policies"][name]
        pc = "-" if p["pinned_cells"] is None else str(p["pinned_cells"])
        print(f"{name:<20} {p['hit_rate'] * 100:>8.3f} {p['hits']:>12} "
              f"{p['accesses']:>12} {pc:>13}")
    print(f"# frequency - splitmix64 = {res['frequency_minus_splitmix64_hit_pct']:+.3f} pt "
          f"| chance baseline (analytic, slots/num_expert) = "
          f"{res['chance_hit_pct']:.3f}%")
    if res["chance_hit_pct"] > 0 and res["n_experts"] > 0:
        print(f"# ratio vs chance: splitmix64 = "
              f"{res['policies']['static-splitmix64']['hit_rate'] * 100 / res['chance_hit_pct']:.2f}x, "
              f"frequency = "
              f"{res['policies']['static-frequency']['hit_rate'] * 100 / res['chance_hit_pct']:.2f}x")
    else:
        print("# ratio vs chance: undefined (zero slots or zero experts)")
    if args.convergence:
        if not args.calibrate_call:
            print("# convergence skipped: needs an explicit calibration corpus "
                  "(--calibrate-call) so the head cannot reach the scored tail")
        else:
            print("# convergence (frequency seed calibrated on a growing head of the "
                  "CALIBRATION window, scored on the fixed eval tail):")
            print(f"# {'calib_calls':>11} {'layers_changed':>14} {'hot_cells':>9} {'hit%':>8}")
            rows_eval = eval_rows_from_calls(eval_calls, eval_lo, eval_hi)[0]
            for row in convergence_series(cal_calls, rows_eval, args.slots_per_layer,
                                          cal_lo=cal_lo, cal_hi=cal_hi):
                ch = "-" if row["layers_changed"] is None else str(row["layers_changed"])
                print(f"# {row['calib_calls']:>11} {ch:>14} {row['hot_cells']:>9} "
                      f"{row['hit_rate'] * 100:>8.3f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, sort_keys=True)
        print(f"# wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
