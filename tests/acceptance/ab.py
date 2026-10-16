#!/usr/bin/env python3
"""The release speed gate: the candidate against the last release, same window.

    ab.py <agent|coder|flashnext> --old <arcint> --new <arcint> --gguf <model>
          --device GPU.N --assets <dir> [--rounds 2] [--out DIR]

Serves the last release's binary (--old) and the release candidate (--new)
alternately in one window -- old, new, old, new for two rounds -- each run a
fresh process with the service's unit flags (cells/served_check.py, which
this script runs unchanged). The candidate may not be slower than the old
binary beyond that window's own run-to-run spread (operator, 2026-10-16,
DESIGN.md §5 "Speed against the last release"). The absolute floors in
cells.json are a report, not a veto.

Per metric (`ACCEPTANCE-METRIC <name> <value> <unit>` lines of the runs):

    spread  = the larger, over the two arms, of (max - min) / mean within
              the arm: what the window itself varies by
    passes  = mean(new) >= mean(old) * (1 - spread - TOLERANCE)
              (for a unit where less is better -- s, ms -- mirrored)

An answer gate (a `  FAIL` line of served_check.py) or a run that dies, in
any run of either arm, fails the A/B. The `texts md5` line (the answers'
bytes) is compared and reported, never gated: the byte tripwire.

Exit status: 0 the candidate is not slower and every answer is right; 1
otherwise. With one round there is one run an arm and no spread to measure:
the spread is 0 and the table says so; use --rounds 2 or more for a gate.
"""
import argparse
import os
import re
import subprocess
import sys

# Slack added to the spread on top of the window's own run-to-run variation,
# as a fraction (0.01 = 1 %). Default 0.0: the spread alone is the allowance.
# The operator may set it (DESIGN.md §5); nothing else in this file does.
TOLERANCE = 0.0

# Units where a smaller number is better; every other unit (t/s) is a rate.
LESS_IS_BETTER_UNITS = ("s", "ms")

METRIC_RE = re.compile(r"^ACCEPTANCE-METRIC (\S+) (\S+) (\S+)\s*$")
GATE_RE = re.compile(r"^\s+(ok|FAIL)\s+(.*?)\s*$")
TEXTS_RE = re.compile(r"^\s+texts md5\s*(.*?)\s*$")

HERE = os.path.dirname(os.path.abspath(__file__))
SERVED_CHECK = os.path.join(HERE, "cells", "served_check.py")


def parse_run(text, returncode=0):
    """One served_check.py run's output -> {"metrics": {name: (value, unit)},
    "failures": [str], "texts": [str]}. A run that exited non-zero without
    printing a FAIL gate (the server died at start, a traceback) still
    carries a failure: a run that did not complete is not a pass."""
    metrics, failures, texts = {}, [], []
    for line in text.splitlines():
        m = METRIC_RE.match(line)
        if m:
            try:
                value = float(m.group(2))
            except ValueError:
                failures.append(f"non-numeric metric {m.group(1)} {m.group(2)!r}")
                continue
            if value != value or value in (float("inf"), float("-inf")):
                failures.append(f"non-finite metric {m.group(1)} {m.group(2)!r}")
                continue
            if m.group(1) in metrics:
                failures.append(f"metric {m.group(1)} printed twice")
            metrics[m.group(1)] = (value, m.group(3))
            continue
        g = GATE_RE.match(line)
        if g and g.group(1) == "FAIL":
            failures.append(g.group(2))
            continue
        t = TEXTS_RE.match(line)
        if t:
            texts = t.group(1).split()
    if returncode != 0 and not failures:
        failures.append(f"the run exited {returncode} without a FAIL gate line")
    return {"metrics": metrics, "failures": failures, "texts": texts}


def arm_stats(values):
    """(mean, spread) of one arm's values; spread = (max - min) / mean, 0
    for a single run (nothing to measure)."""
    mean = sum(values) / len(values)
    spread = (max(values) - min(values)) / mean if mean else 0.0
    return mean, spread


def decide(old_runs, new_runs, tolerance=None):
    """Per metric verdicts from the parsed runs of both arms.

    old_runs, new_runs: lists of parse_run() results. Returns (rows,
    failures): rows is a list of dicts (metric, unit, old, new, mean_old,
    mean_new, spread_old, spread_new, spread, limit, ok, note), failures a
    list of strings (answer gates and incomplete runs, each naming its arm
    and run) that fail the A/B independently of the numbers."""
    tol = TOLERANCE if tolerance is None else tolerance
    failures = []
    for arm, runs in (("old", old_runs), ("new", new_runs)):
        for i, r in enumerate(runs, 1):
            failures.extend(f"{arm} run {i}: {f}" for f in r["failures"])
    names = []
    for r in old_runs + new_runs:
        for n in r["metrics"]:
            if n not in names:
                names.append(n)
    rows = []
    for name in names:
        row = {"metric": name, "ok": False, "note": ""}
        arms = {}
        for arm, runs in (("old", old_runs), ("new", new_runs)):
            vals = [r["metrics"][name][0] for r in runs if name in r["metrics"]]
            units = {r["metrics"][name][1] for r in runs if name in r["metrics"]}
            arms[arm] = (vals, units)
            row[arm] = vals
        units = arms["old"][1] | arms["new"][1]
        row["unit"] = "/".join(sorted(units))
        if len(units) != 1:
            row["note"] = "units differ between runs"
            rows.append(row)
            continue
        if any(len(arms[a][0]) != len(runs) for a, runs in (("old", old_runs), ("new", new_runs))):
            row["note"] = "not printed by every run"
            rows.append(row)
            continue
        mo, so = arm_stats(arms["old"][0])
        mn, sn = arm_stats(arms["new"][0])
        spread = max(so, sn)
        if row["unit"] in LESS_IS_BETTER_UNITS:
            limit = mo * (1 + spread + tol)
            ok = mn <= limit
        else:
            limit = mo * (1 - spread - tol)
            ok = mn >= limit
        row.update(mean_old=mo, mean_new=mn, spread_old=so, spread_new=sn,
                   spread=spread, limit=limit, ok=ok)
        if len(arms["old"][0]) < 2:
            row["note"] = "one run an arm: no spread measured"
        rows.append(row)
    if not rows:
        failures.append("no ACCEPTANCE-METRIC line in any run: nothing was compared")
    return rows, failures


def compare_texts(old_runs, new_runs):
    """A one-line report on the answer bytes (md5 prefixes): not gated."""
    olds = {tuple(r["texts"]) for r in old_runs}
    news = {tuple(r["texts"]) for r in new_runs}
    if olds == news and len(olds) == 1:
        return "texts md5: identical in every run of both arms"
    if len(olds) == 1 and len(news) == 1:
        return "texts md5: DIFFER between the arms (reported, not gated)"
    return ("texts md5: runs within an arm differ (old %d distinct, new %d distinct; "
            "reported, not gated)" % (len(olds), len(news)))


def fmt(vals):
    return " ".join(f"{v:.1f}" for v in vals)


def render(rows, failures, texts_line):
    out = [f"{'metric':<16} {'unit':<5} {'old runs':<22} {'new runs':<22} "
           f"{'mean old':>9} {'mean new':>9} {'spread':>7} {'limit':>9}  verdict"]
    for r in rows:
        if "mean_old" not in r:
            out.append(f"{r['metric']:<16} {r['unit']:<5} {fmt(r['old']):<22} {fmt(r['new']):<22} "
                       f"{'-':>9} {'-':>9} {'-':>7} {'-':>9}  FAIL ({r['note']})")
            continue
        verdict = "ok" if r["ok"] else "FAIL (slower than the old beyond the spread)"
        if r["note"] and r["ok"]:
            verdict += f" ({r['note']})"
        out.append(f"{r['metric']:<16} {r['unit']:<5} {fmt(r['old']):<22} {fmt(r['new']):<22} "
                   f"{r['mean_old']:>9.2f} {r['mean_new']:>9.2f} {r['spread'] * 100:>6.2f}% "
                   f"{r['limit']:>9.2f}  {verdict}")
    out.append(texts_line)
    for f in failures:
        out.append(f"FAIL {f}")
    return "\n".join(out)


def run_one(args, binary, arm, n):
    cmd = [sys.executable, args.served_check, args.service, binary, args.gguf, args.device, args.assets]
    print(f"== {arm} run {n}: {os.path.basename(binary)}", flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if args.out:
        with open(os.path.join(args.out, f"{args.service}-{arm}-{n}.log"), "w") as f:
            f.write(proc.stdout)
    print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    return parse_run(proc.stdout, proc.returncode)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("service", choices=("agent", "coder", "flashnext"))
    p.add_argument("--old", required=True, help="the last release's arcint binary")
    p.add_argument("--new", required=True, help="the release candidate's arcint binary")
    p.add_argument("--gguf", required=True)
    p.add_argument("--device", required=True, help="e.g. GPU.0")
    p.add_argument("--assets", required=True, help="the served cells' assets directory")
    p.add_argument("--rounds", type=int, default=2, help="old/new pairs (default 2: old,new,old,new)")
    p.add_argument("--out", help="directory for each run's full output")
    p.add_argument("--served-check", dest="served_check", default=SERVED_CHECK, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.rounds < 1:
        p.error("--rounds must be at least 1")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    old_runs, new_runs = [], []
    for n in range(1, args.rounds + 1):
        old_runs.append(run_one(args, args.old, "old", n))
        new_runs.append(run_one(args, args.new, "new", n))
    rows, failures = decide(old_runs, new_runs)
    print()
    print(render(rows, failures, compare_texts(old_runs, new_runs)))
    bad = bool(failures) or not all(r["ok"] for r in rows)
    print("\nA/B " + ("FAIL" if bad else "PASS") + f": {args.service}, {args.rounds} round(s), "
          f"tolerance {TOLERANCE * 100:.1f} %")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
