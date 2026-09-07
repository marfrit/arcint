#!/usr/bin/env python3
# Per-decode-step device timeline from an OpenCL intercept layer chrome trace
# (https://github.com/intel/opencl-intercept-layer, CLI_ChromePerformanceTiming=1).
# Used for DESIGN §7.0.2bg: one served process traced, the tail segmented into
# decode steps on the per-step logits copy (--marker "DtoH; 993280" for a
# 248,320-entry f32 vocabulary), device busy per step against the untraced step.
# The device durations are the card's own timestamps; the span is the traced wall
# and carries the tracer's host overhead (measured +4 % on the IR, +18 % on the
# GGUF-opened forms at 1k).
"""Per-decode-step device timeline from an OpenCL intercept layer chrome trace.

Usage: tl_steps.py <clintercept_trace.json> [--steps 64] [--tail 40] [--top 25] [--marker NAME]

Segments the trace's tail into decode steps by the largest host gaps, then reports
per step: launches, device busy (union of intervals), span, idle, and the device
time per kernel name. Device durations are what the card reported; the span is the
traced wall and carries the tracer's overhead.
"""
import json, sys, argparse, collections, re

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--steps", type=int, default=64)
ap.add_argument("--tail", type=int, default=40, help="steady-state window: the last N steps")
ap.add_argument("--top", type=int, default=25)
ap.add_argument("--marker", default=None, help="kernel name occurring once per step; boundaries are placed before it")
ap.add_argument("--dump-steps", action="store_true")
ap.add_argument("--prefill", action="store_true", help="also report the launches before the first decode step back to the last gap > --prefill-gap-ms (the served prefill)")
ap.add_argument("--prefill-gap-ms", type=float, default=1000.0)
a = ap.parse_args()

raw = open(a.trace, "rb").read().decode("utf-8", "replace").strip()
try:
    ev = json.loads(raw)
except json.JSONDecodeError:
    # a process that did not close the array: cut at the last complete record
    cut = raw.rfind("}")
    ev = json.loads(raw[:cut + 1].rstrip().rstrip(",") + "]")
if isinstance(ev, dict):
    ev = ev.get("traceEvents", [])

dev = []
for e in ev:
    if e.get("ph") != "X":
        continue
    # device commands sit on queue threads (tid "N.1"); the host thread is tid 0
    if str(e.get("tid", 0)) in ("0", "0.0"):
        continue
    ts = float(e["ts"]); dur = float(e.get("dur", 0.0))
    dev.append((ts, ts + dur, e.get("name", "?"), e.get("tid", 0), e.get("pid", 0)))
dev.sort()
print(f"device events: {len(dev)}; host events skipped: {len(ev) - len(dev)}")
if not dev:
    sys.exit(1)
names = collections.Counter(n for _, _, n, _, _ in dev)
print(f"distinct names: {len(names)}; queues (tid): {sorted(set(t for *_, t, _ in dev))}")

def gaps(seq):
    out = []
    for i in range(1, len(seq)):
        out.append((seq[i][0] - max(s[1] for s in seq[max(0, i - 4):i]), i))
    return out

# Boundaries: N-1 largest gaps in the tail, iterated: first guess from the last 25 %.
if a.marker:
    idx = [i for i, e in enumerate(dev) if a.marker in e[2]]
    idx = idx[-a.steps:]
    bounds = [idx[0]] + [i for i in idx[1:]]
    # a step runs from just after the previous marker to this marker: use marker positions as ends
    ends = idx
    starts = [0 if k == 0 else ends[k - 1] + 1 for k in range(len(ends))]
    # the first step's start: the first event after the previous (prefill) marker
    prev = [i for i, e in enumerate(dev) if a.marker in e[2]]
    if len(prev) > a.steps:
        starts[0] = prev[-a.steps - 1] + 1
    steps = [dev[s:e + 1] for s, e in zip(starts, ends)]
else:
    n = len(dev)
    tail = dev[int(n * 0.75):]
    g = sorted(gaps(tail), reverse=True)[: max(8, a.steps // 4)]
    cut_idx = sorted(i for _, i in g)
    lens = [b - a_ for a_, b in zip(cut_idx, cut_idx[1:])]
    lens.sort()
    K = lens[len(lens) // 2]
    region_start = max(0, n - int(a.steps * K * 1.15) - K)
    region = dev[region_start:]
    g = sorted(gaps(region), reverse=True)[: a.steps - 1]
    cuts = sorted(i for _, i in g)
    # keep only the last `steps` segments
    segs = []
    prev = 0
    for c in cuts:
        segs.append(region[prev:c]); prev = c
    segs.append(region[prev:])
    steps = segs[-a.steps:]
    print(f"segmentation: median launches per step from the tail {K}; region of {len(region)} events; {len(steps)} steps")

def busy(seq):
    tot = 0.0; cur_s, cur_e = None, None
    for s, e, *_ in sorted(seq):
        if cur_e is None or s > cur_e:
            if cur_e is not None: tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None: tot += cur_e - cur_s
    return tot

rows = []
for k, st in enumerate(steps):
    span = st[-1][1] - st[0][0]
    b = busy(st)
    summed = sum(e - s for s, e, *_ in st)
    rows.append((k, len(st), span / 1000, b / 1000, summed / 1000))
print("\nstep  launches  span_ms  busy_ms  sum_ms   (span = traced wall of the launches, busy = union of device intervals)")
for r in rows[:3] + [None] + rows[-3:]:
    if r is None: print("  ..."); continue
    print(f"{r[0]:4d}  {r[1]:8d}  {r[2]:7.2f}  {r[3]:7.2f}  {r[4]:6.2f}")
tail = rows[-a.tail:]
def mean(xs): return sum(xs) / len(xs)
print(f"\nsteady (last {len(tail)} steps): launches {mean([r[1] for r in tail]):.0f}, span {mean([r[2] for r in tail]):.2f} ms, "
      f"busy {mean([r[3] for r in tail]):.2f} ms, sum {mean([r[4] for r in tail]):.2f} ms, idle in span {mean([r[2]-r[3] for r in tail]):.2f} ms")
if len(rows) >= 3:
    print(f"first steps: busy {rows[0][3]:.2f} / {rows[1][3]:.2f} / {rows[2][3]:.2f} ms, span {rows[0][2]:.2f} / {rows[1][2]:.2f} / {rows[2][2]:.2f} ms")

# the inter-step host gap (from the last launch's end to the next step's first start)
if len(steps) > 1:
    igaps = [steps[k + 1][0][0] - steps[k][-1][1] for k in range(len(steps) - 1)]
    tg = igaps[-a.tail:]
    print(f"inter-step gap (device idle between the last kernel and the next step's first): mean {mean(tg)/1000:.2f} ms, min {min(tg)/1000:.2f}, max {max(tg)/1000:.2f}")

# per kernel name, steady state
agg = collections.defaultdict(lambda: [0, 0.0])
for st in steps[-a.tail:]:
    for s, e, n, *_ in st:
        agg[n][0] += 1; agg[n][1] += e - s
ntail = len(steps[-a.tail:])
tot = sum(v[1] for v in agg.values())
print(f"\nper step (steady), device time by kernel name; total {tot/ntail/1000:.2f} ms over {sum(v[0] for v in agg.values())/ntail:.0f} launches")
print(f"{'count':>6} {'ms/step':>8} {'share':>6} {'us/launch':>9}  name")
for n, (c, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: a.top]:
    print(f"{c/ntail:6.0f} {t/ntail/1000:8.3f} {100*t/tot:5.1f}% {t/c:9.1f}  {n[:110]}")

# the served prefill: everything before the first full decode step, back to the last host gap > 200 ms
if a.prefill and len(steps) > 1:
    end = steps[1][0][0]
    pre = [e for e in dev if e[1] <= end]
    k = len(pre) - 1
    while k > 0 and pre[k][0] - pre[k - 1][1] < a.prefill_gap_ms * 1000:
        k -= 1
    pre = pre[k:]
    span = pre[-1][1] - pre[0][0]; b = busy(pre)
    print(f"\nprefill (+ the first decode step), back to the last gap > {a.prefill_gap_ms:.0f} ms: {len(pre)} launches, span {span/1000:.1f} ms, busy {b/1000:.1f} ms, idle {(span-b)/1000:.1f} ms")
    pagg = collections.defaultdict(lambda: [0, 0.0])
    for s_, e_, n, *_ in pre:
        pagg[n][0] += 1; pagg[n][1] += e_ - s_
    ptot = sum(v[1] for v in pagg.values())
    print(f"{'count':>6} {'ms':>8} {'share':>6} {'us/launch':>9}  name")
    for n, (c, t) in sorted(pagg.items(), key=lambda kv: -kv[1][1])[: a.top]:
        print(f"{c:6d} {t/1000:8.2f} {100*t/ptot:5.1f}% {t/c:9.1f}  {n[:110]}")

# the largest intra-step gaps (host stalls inside a step)
if a.dump_steps:
    st = steps[-1]
    gl = sorted(((st[i][0] - st[i-1][1], i) for i in range(1, len(st))), reverse=True)[:10]
    print("\nlargest intra-step gaps in the last step (us, before kernel):")
    for g, i in gl:
        print(f"  {g:8.0f}  before {st[i][2][:80]}  after {st[i-1][2][:60]}")
