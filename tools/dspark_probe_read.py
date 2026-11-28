#!/usr/bin/env python3
"""The DSpark shortlist's probe log, read (arcint --llama-dflash-shortlist-probe FILE, patch 0243).

A probe log has one JSON line a verified draft of a lane:
  {"seq": 0, "pos0": 812, "id_last": 1234, "a": 2, "sl": 256, "d": [...], "f": [...], "in": [1, 0, ...]}
pos0 the block's first position (id_last's), d the drafts offered, a how many the verify kept; with the shortlist
on (sl = K > 0) f is the full chain's pick at each position on the same state and in whether it was among the
position's K candidates; with it off (sl = 0) f = d and in is empty.

The question (the shortlist's accept loss against DSpark without it): is the loss the shortlist's design price, the
positions whose full-chain pick lies outside the top K, or more?

What the target drew. The verify samples each row and keeps a draft while it draws that same token
(src/exec/verify_walk.h), so the target's draw at a row does not depend on the draft. The next line of the same
lane gives it: a = next.pos0 - pos0 - 1 drafts kept, and the target's token at row a + 1 (the first refused) is
next.id_last. The line's own "a" (the driver's count) checks the pair: where they differ, or the next line does not
lie after this one (another request), the next line is not this draft's next cycle and the refused row's draw stays
unknown (the line is out of the prediction).

The counterfactual, per line, j0 the first position where f and d differ (0-based):
  - no j0, or j0 > a: the full chain drafts the same up to the refused row: the same count;
  - j0 < a: the shortlist's d[j0] was kept, so the target drew d[j0] != f[j0]: the full chain is refused at j0;
  - j0 = a (the refused row): the full chain is kept there iff f[a] equals the target's draw, and then goes on as
    the full chain does: the further accepts after a kept row come from the probe run WITHOUT the shortlist
    (--off), its per-position rates; without --off they count 0 (the prediction is then a lower bound).
The lines are split by where f[j0] lies:
  - outside the K candidates: the shortlist's design price. Their sum is the PREDICTION (points of the accept rate,
    over the drafts offered on the lines used);
  - inside the K candidates (f[j0] among them, yet not the shortlist's pick): not a price of the design but a
    defect signal (a wrong fold row, bias or base gather, an ordering error, or rounding between the gathered and
    the full product). Their sum is INSIDE_LOSS, read on its own; it must be about 0.
The membership TRIPWIRE: a position with in = 0 and f = d is impossible (the shortlist's own pick is always among
its candidates); a nonzero count means the read-back of the candidates is wrong (llama_get_dspark_probe's rows).
First order: the outside first divergences the verify reached (kept by the shortlist, or at the refused row with a
known draw), their share of the drafts offered x their net rate (the full chain kept minus the shortlist kept,
further accepts not counted).

The reading, pre-registered (printed as RULE: the coordinator's rule, plus one explanatory clause; RULE_AS: how it is
computed here):
  measured = the mean paired difference of sp0's timed pairs (shortlist - no shortlist, the same build), with its
  SE_d (--measured-delta, --measured-se); prediction = pr0's probe logs, outside only.
  GREEN iff all three hold: |prediction - measured| <= max(2 pt, 2 SE_d); inside_loss <= 0.5 pt absolute (above
  that is a gather defect, regardless of the outside agreement); the membership tripwire = 0.
  pr0's own difference (the two probe runs) is printed beside it as data, not gated.

DSPARK_PROBE_READ_RED (test only): 1 the kept count one too many (a = next.pos0 - pos0); 2 the inside divergences
folded back into the prediction (inside_loss 0); 3 the tripwire not counted; 4 the validity limits not applied.

INVALID: above 5 % of an arm's lines unpaired (either log), or above 5 % of the shortlist's lines with a refused-row
draw unknown, the sample the prediction rests on is not the run's: the reading gives no verdict (neither GREEN nor
DEFECT). Without a measured value (sp0 skipped: no --measured-delta) the reading is NOT READ, the same class.

Usage: dspark_probe_read.py SHORT.jsonl [--off OFF.jsonl] [--measured-delta PTS --measured-se PTS] [--json OUT]
Exit 0 with a reading, 2 when the shortlist log has no usable line, 3 when --off is not a log without the shortlist.
"""
import argparse
import json
import math
import os
import sys

BAND = 2.0          # points: the band's floor around the prediction
INSIDE_MAX = 0.5    # points: inside_loss, absolute
VALID_MAX = 5.0     # % of an arm's lines: above it (unpaired lines in either log, refused rows with an unknown draw in
                    # the shortlist's) the reading is INVALID: no verdict
RULE = ("PROBE READING (pre-registered: the coordinator's rule, plus one explanatory clause; sp0 is the window's timed depth-0 row, where DSpark off, the "
        "shortlist and width 3 without it run the same continuations, TABLEACC its per-continuation accept difference "
        "of the shortlist against width 3 without it, and pr0 the two untimed runs that write the probe logs): Measured \u0394accept = the mean paired difference of sp0's 10 timed "
        "pairs (shortlist \u2212 off, the same build), with its SE_d. Prediction = pr0's probe logs, outside-only. GREEN "
        "iff all three hold: |prediction \u2212 measured| \u2264 max(2 pt, 2\u00b7SE_d); inside_loss \u2264 0.5 pt absolute. "
        "Above that is a gather defect, regardless of the outside agreement; the membership tripwire = 0. pr0's own "
        "\u0394 is printed beside it as data, not gated.")
RULE_AS = ("PROBE READING, as computed: \"off\" in sp0's pairs is width 3 without the shortlist (the window's TABLEACC; "
           "sp0's DSpark-off arm has no accept rate); measured and prediction compared as accept lost; inside_loss by "
           "its absolute value; INVALID (no verdict) when an arm's unpaired lines or the shortlist's refused rows with "
           "an unknown draw exceed 5 % of its lines; NOT READ (no verdict) without a measured value (sp0 not run)")


def load(path):
    """the lines of a probe log, by lane, in order; bad lines counted"""
    lanes, bad = {}, 0
    with open(path) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                r = json.loads(raw)
                d = [int(x) for x in r["d"]]
                rec = {"seq": int(r["seq"]), "pos0": int(r["pos0"]), "id_last": int(r["id_last"]),
                       "a": int(r["a"]) if "a" in r else None, "sl": int(r.get("sl", 0)), "d": d,
                       "f": [int(x) for x in r.get("f", d)], "in": [int(x) for x in r.get("in", [])]}
            except (ValueError, KeyError, TypeError):
                bad += 1
                continue
            if len(rec["f"]) != len(d) or (rec["sl"] > 0 and len(rec["in"]) != len(d)):
                bad += 1
                continue
            lanes.setdefault(rec["seq"], []).append(rec)
    return lanes, bad


def steps(lanes, red=0):
    """each line with its kept count a and, when known, the target's draw at the refused row t"""
    out, mismatch = [], 0
    for seq in sorted(lanes):
        ls = lanes[seq]
        for k, rec in enumerate(ls):
            nxt = ls[k + 1] if k + 1 < len(ls) else None
            a, t = rec["a"], None
            if nxt is not None:
                ad = nxt["pos0"] - rec["pos0"] - (0 if red == 1 else 1)
                if ad >= 0 and (rec["a"] is None or ad == rec["a"]):
                    a = ad
                    t = nxt["id_last"]
                else:
                    mismatch += 1
            if a is None or a < 0 or a > len(rec["d"]):
                mismatch += 1
                continue
            out.append(dict(rec, a=a, t=t if a < len(rec["d"]) else None))
    return out, mismatch


def off_rates(off_steps):
    """P(kept at position q | kept through q - 1), q = 1.., from the run without the shortlist"""
    reach, kept = {}, {}
    for s in off_steps:
        n, a = len(s["d"]), s["a"]
        for q in range(1, n + 1):
            if a >= q - 1:
                reach[q] = reach.get(q, 0) + 1
                if a >= q:
                    kept[q] = kept.get(q, 0) + 1
    return {q: kept.get(q, 0) / reach[q] for q in reach if reach[q] > 0}


def further(rates, m, n):
    """the expected further kept drafts after m kept, of n offered, at the run-without's rates"""
    e, p = 0.0, 1.0
    for q in range(m + 1, n + 1):
        p *= rates.get(q, 0.0)
        e += p
    return e


def read(short_steps, rates=None, red=0):
    c = dict(lines=0, offered=0, kept=0, outside=0, outside_reached=0, inside_differs=0, tripwire=0,
             div_unreached=0, offered_used=0, short_kept_used=0)
    for w in ("out", "in"):
        c.update({f"kept_by_short_{w}": 0, f"refused_{w}": 0, f"refused_unknown_{w}": 0, f"hits_{w}": 0,
                  f"delta_{w}": 0.0})
    for s in short_steps:
        if s["sl"] <= 0:
            continue
        d, f, ins, a, n = s["d"], s["f"], s["in"], s["a"], len(s["d"])
        c["lines"] += 1
        c["offered"] += n
        c["kept"] += a
        for j in range(n):
            if not ins[j]:
                c["outside"] += 1
                if j <= a:
                    c["outside_reached"] += 1
                if f[j] == d[j] and red != 3:
                    c["tripwire"] += 1     # impossible: the shortlist's own pick is among its candidates
            elif f[j] != d[j]:
                c["inside_differs"] += 1
        j0 = next((j for j in range(n) if f[j] != d[j]), None)
        full = float(a)
        where = None
        if j0 is not None and j0 > a:
            c["div_unreached"] += 1
        elif j0 is not None:
            where = "in" if ins[j0] else "out"
            if j0 < a:
                c[f"kept_by_short_{where}"] += 1
                full = float(j0)
            else:   # j0 == a < n: the refused row
                c[f"refused_{where}"] += 1
                if s["t"] is None:
                    c[f"refused_unknown_{where}"] += 1
                    continue   # its draw unknown: out of the prediction
                hit = f[a] == s["t"]
                c[f"hits_{where}"] += int(hit)
                if hit:
                    full = a + 1 + further(rates or {}, a + 1, n)
        c["offered_used"] += n
        c["short_kept_used"] += a
        if where is not None:
            c[f"delta_{'out' if red == 2 else where}"] += full - a
    res = dict(c)
    pct = lambda x, y: 100.0 * x / y if y else float("nan")
    res["short_rate"] = pct(c["kept"], c["offered"])
    res["share_outside"] = pct(c["outside"], c["offered"])
    res["predicted"] = pct(c["delta_out"], c["offered_used"])
    res["inside_loss"] = pct(c["delta_in"], c["offered_used"])
    first_out = c["kept_by_short_out"] + c["refused_out"] - c["refused_unknown_out"]
    res["first_out"] = first_out
    res["first_out_share"] = pct(first_out, c["offered_used"])
    res["first_out_net_rate"] = pct(c["hits_out"] - c["kept_by_short_out"], first_out)
    res["first_order"] = pct(c["hits_out"] - c["kept_by_short_out"], c["offered_used"])
    res["unknown"] = c["refused_unknown_out"] + c["refused_unknown_in"]
    res["bound"] = rates is None
    return res


def reading(predicted, measured_loss, se, inside_loss, tripwire):
    """the pre-registered reading: (green, the reasons it is not)"""
    why = []
    se_ok = se is not None and se == se
    band = max(BAND, 2.0 * se) if se_ok else BAND
    if measured_loss is None or measured_loss != measured_loss or predicted != predicted:
        why.append("no measured difference or no prediction")
    elif abs(predicted - measured_loss) > band:
        why.append("|prediction - measured| %.2f > the band %.2f" % (abs(predicted - measured_loss), band))
    if inside_loss != inside_loss or abs(inside_loss) > INSIDE_MAX:
        why.append("inside_loss %.2f pt above %.1f pt: a gather defect" % (inside_loss, INSIDE_MAX))
    if tripwire != 0:
        why.append("the membership tripwire is %d (the candidates' read-back)" % tripwire)
    return not why, why, band


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("short")
    ap.add_argument("--off", help="the probe log of the run without the shortlist (its per-position rates)")
    ap.add_argument("--measured-delta", type=float,
                    help="sp0's mean paired accept difference, shortlist - no shortlist, points (a loss is negative)")
    ap.add_argument("--measured-se", type=float, help="its SE_d, points")
    ap.add_argument("--earlier-gate", help="an earlier measurement to print beside the measured one, as data only")
    ap.add_argument("--json", help="write the numbers here")
    o = ap.parse_args(argv)
    red = int(os.environ.get("DSPARK_PROBE_READ_RED") or 0)
    lanes, bad = load(o.short)
    st, mm = steps(lanes, red)
    rates, off_rate, off_lines, omm = None, float("nan"), 0, 0
    if o.off:
        olanes, obad = load(o.off)
        if any(rec["sl"] > 0 for ls in olanes.values() for rec in ls):
            print("PROBE REFUSED: --off %s carries shortlist lines (sl > 0): it must be the run without the shortlist"
                  % o.off)
            return 3
        bad += obad
        ost, omm = steps(olanes, red)
        rates = off_rates(ost)
        off_lines = len(ost)
        offered = sum(len(s["d"]) for s in ost)
        off_rate = 100.0 * sum(s["a"] for s in ost) / offered if offered else float("nan")
    r = read(st, rates, red)
    nl = r["lines"]
    share = lambda x, y: 100.0 * x / y if y else float("nan")
    r.update(bad_lines=bad, unpaired=mm, unpaired_off=omm, off_lines=off_lines, off_rate=off_rate, red=red,
             unpaired_share=share(mm, nl), unpaired_off_share=share(omm, off_lines) if o.off else 0.0,
             unknown_share=share(r["unknown"], nl))
    invalid = []
    if red != 4:
        if not r["unpaired_share"] <= VALID_MAX:
            invalid.append("the shortlist log's unpaired lines %d (%.1f %%) above %.0f %%" % (mm, r["unpaired_share"], VALID_MAX))
        if not r["unpaired_off_share"] <= VALID_MAX:
            invalid.append("the run-without's unpaired lines %d (%.1f %%) above %.0f %%" % (omm, r["unpaired_off_share"], VALID_MAX))
        if not r["unknown_share"] <= VALID_MAX:
            invalid.append("refused rows with an unknown draw %d (%.1f %%) above %.0f %%" % (r["unknown"], r["unknown_share"], VALID_MAX))
    own_delta = r["short_rate"] - off_rate if o.off else float("nan")   # shortlist - without, points
    measured_loss = -o.measured_delta if o.measured_delta is not None else None
    green, why, band = reading(r["predicted"], measured_loss, o.measured_se, r["inside_loss"], r["tripwire"])
    r.update(own_delta=own_delta, measured_delta=o.measured_delta, measured_se=o.measured_se, band=band,
             green=green and not invalid and measured_loss is not None, why=why, invalid=invalid,
             reading="INVALID" if invalid else "NOT READ" if measured_loss is None else "GREEN" if green else "DEFECT")
    print("PROBE lines %d | offered %d kept %d (%.1f %%) | outside the top K: %d (%.1f %% of offered), %d reached | "
          "inside but differing %d | tripwire (in 0, f = d) %d | bad lines %d | unpaired: shortlist %d (%.1f %%), "
          "without %s | refused-row draw unknown %d (%.1f %%)%s"
          % (nl, r["offered"], r["kept"], r["short_rate"], r["outside"], r["share_outside"], r["outside_reached"],
             r["inside_differs"], r["tripwire"], bad, mm, r["unpaired_share"],
             "%d (%.1f %%)" % (omm, r["unpaired_off_share"]) if o.off else "none (no --off)", r["unknown"],
             r["unknown_share"], " | RED reader %d" % red if red else ""))
    print("PROBE first divergence outside: kept by the shortlist %d, at the refused row %d (draw unknown %d, the full "
          "pick = the draw %d) | inside: kept by the shortlist %d, at the refused row %d (unknown %d, hits %d) | "
          "unreached %d"
          % (r["kept_by_short_out"], r["refused_out"], r["refused_unknown_out"], r["hits_out"], r["kept_by_short_in"],
             r["refused_in"], r["refused_unknown_in"], r["hits_in"], r["div_unreached"]))
    print("PROBE prediction (outside only) %.2f points of accept lost%s | inside_loss %.2f points | first order: the "
          "outside first divergences reached, %.2f %% of offered x their net rate %.1f %% = %.2f points (further "
          "accepts not counted)"
          % (r["predicted"], " (a lower bound: no --off)" if r["bound"] else "", r["inside_loss"],
             r["first_out_share"], r["first_out_net_rate"], r["first_order"]))
    print("PROBE measured (sp0's timed pairs, shortlist - no shortlist): %s | pr0's own difference (data, not gated): "
          "%.2f points (shortlist %.1f %%, without %.1f %% over %d lines)%s"
          % ("%.2f points, SE_d %s" % (o.measured_delta, "%.2f" % o.measured_se if o.measured_se is not None else "none")
             if o.measured_delta is not None else "none given", own_delta, r["short_rate"], off_rate, off_lines,
             " | the earlier gate (data only): %s" % o.earlier_gate if o.earlier_gate else ""))
    print(RULE)
    print(RULE_AS)
    if invalid:
        print("PROBE READING: INVALID (no verdict: " + "; ".join(invalid) + ")")
    elif measured_loss is None:
        print("PROBE READING: NOT READ (no measured value: sp0's timed pairs were not run or not given)")
    else:
        print("PROBE READING: %s" % ("GREEN (band %.2f pt)" % band if green else "DEFECT (" + "; ".join(why) + ")"))
    if o.json:
        with open(o.json, "w") as fh:
            json.dump(r, fh, indent=1, sort_keys=True, default=lambda x: None)
    return 0 if nl > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
