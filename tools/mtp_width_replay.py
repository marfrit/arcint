#!/usr/bin/env python3
"""Offline replay of Strata's cost-aware MTP window over arcint's trace.

Campaign research-mtp-width-policy, section 6 step 1: price the width policy
before building it into the serving path. The trace is arcint's own
(ARCINT_MTP_TRACE, src/exec/mtp_trace.h, off by default), one line a verify
round; this tool replays `DraftPolicy::keep_draft` + `observe_probs` + the
measured per-width round cost over it and reports what a fixed width and the
policy would each get, plus the draft-probability -> acceptance table.

The port is Strata's own rule, constants cited from the checked-out reference
source (`Strata-ref` at REV `strata 4b9e05a2`), every path `Strata-ref:<path>`:

  kShape          {0, 1, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6}  Strata-ref:src/spec/draft_policy.cpp:13
  kCostAlpha      0.1     EMA of a round time                            Strata-ref:src/spec/draft_policy.cpp:14
  kTokAlpha       0.05    EMA of an MTP window outcome                  Strata-ref:src/spec/draft_policy.cpp:15
  kCalDecay       0.995   the probability -> acceptance table's fade    Strata-ref:src/spec/draft_policy.cpp:16
  kPriorN         4.0     a bucket's prior is worth 4 observations      Strata-ref:src/spec/draft_policy.cpp:21
  buckets         8       p_bucket = int(p * 8) clamped 0..7            Strata-ref:src/spec/draft_policy.cpp:174-179
  draft_rate      (pa_ok[b] + kPriorN*p) / (pa_n[b] + kPriorN)          Strata-ref:src/spec/draft_policy.cpp:182-188
  observe_probs   reached j <= accepted, held j < accepted, decayed     Strata-ref:src/spec/draft_policy.cpp:190-196
  cost_ms         measured, else scaled by kShape from measured widths  Strata-ref:src/spec/draft_policy.cpp:40-53
  clamp_sample    a first-measured width takes the other widths' guess,
                  later samples capped at 2x the known value             Strata-ref:src/spec/draft_policy.cpp:55-65
  seed_costs      a per-width table counted as `weight` rounds           Strata-ref:src/spec/draft_policy.cpp:66-74
  set_context     <6K, <24K, longer; an empty bucket takes its nearest  Strata-ref:src/spec/draft_policy.cpp:76-93
  mtp_tokens      measured, else 1 + 0.7*(t-1)                          Strata-ref:src/spec/draft_policy.cpp:118-121
  keep_draft      the E/cost rule, margin/3 per width, probe of an
                  unmeasured width while q >= 0.15                      Strata-ref:src/spec/draft_policy.cpp:199-224
  margin          0.03                                                  Strata-ref:include/strata/spec/draft_policy.hpp:32
  kMaxT           8, the widest window in rows                           Strata-ref:include/strata/spec/draft_policy.hpp:25
  kCtxBuckets     3                                                     Strata-ref:include/strata/spec/draft_policy.hpp:27

kProbes (draft_policy.cpp:22) is `keep_draft`'s own unmeasured-width probe
(:216, :223) and is ported; kPriorQ and kDecay belong to the lookup drafter
(draft_policy.cpp:20-26) and are not. Not ported, because the MTP rule does not
use them: `stale()` only guards `choose()`/`chain()` (draft_policy.cpp:116,
:142, :147, :287, :291), and robust_/clamp_sample's cap only matters when a
width's first use captures its graph - the arm below sets robust as
STRATA_SPEC_COST=1 does (generate.cpp:16509) and then reads the capped sample,
exactly as observe() does (draft_policy.cpp:155-166).

What the replay CANNOT be exact on, and what covers it:

  1. Drafts never made beyond the ceiling. `p` ends at the round's own nmax;
     a policy that would want a wider window cannot be priced from a run that
     never drafted it. Cover: run ceiling+1 (the doc's 5.3/6 step 2).
  2. Drafts the min-p floor never made. With --llama-mtp-min-p > 0 the chain
     stops at one low head probability, so no later draft's probability and no
     wider window's outcome exist in the trace. Cover: a min-p 0 arm (this
     tool prints a warning when a trace carries minp > 0).
  3. The cost of a width is estimated from the rounds that ran at it. A fixed
     width run measures exactly one width; every other width's cost is Strata's
     kShape scaling, a guess. Cover: the per-width legs of doc 6 step 2, or a
     policy arm whose own rounds visit several widths (the online EMA then
     learns them as the reference does).
  4. The window the policy would verify is not the window the trace verified.
     The trace's `acc` is the full window's walk outcome; the truncated
     window's outcome is inferred as min(acc, chosen drafts) + 1, not observed.
     A width whose acceptance differs from its rounds' full-window acceptance
     cannot be seen here.
  5. Start-to-start vs Strata's cost. `ms` is the round's own duration: its
     start to the NEXT round's start, the request's last round to the loop's
     exit (the doc's open question); Strata's cost(t) is start-to-end inside
     the round (generate.cpp:16752-16767). The replay prints tp + tv against
     ms so the gap is visible; it feeds `ms` to the policy, as the whole-round
     cost Strata learns is the honest one. A line with `ms` <= 0 is unknown:
     the replay learns no cost from it and says so.
  6. The trace's own host cost. On, the writer computes the head's top
     probability per draft (src/exec/mtp_trace.h): one host pass over the
     head's logits a draft, ~0.8 ms a draft for a 248,320-row head at the
     code's own comment (llama_spec.cpp:354-361). That round time is in the
     trace and the serving path does not pay it, so a trace-on arm prices its
     widths high and the replay leans narrower than a trace-off arm. Cover: a
     trace-off arm with the same flags per unit in the same window.
  7. T1 is never measured by a trace. A round's `rows` is 1 + its drafts and
     a chunk drafts at least one while the budget allows, so width 1 appears
     only at a budget end. `--llama-mtp 0` drafts nothing and writes no trace.
     Cover: the per-unit MTP-off leg (`fn_sweep.py --no-mtp`); its decode t/s
     is the ms a token, i.e. T1's round ms = 1000/t_s, passed by --seed-costs
     (`inferred`).
  8. Process-wide state. Strata's policy lives for the process
     (generate.cpp:16492), so cost and a(p) accumulate across requests; the
     replay does the same. A per-request restart would price differently. The
     trace's request counter restarts per process, so the replay keys requests
     by (file, req).

A measured cost table: `--seed-costs "T1,T2,..;T1,..;T1,.."` takes Strata's
STRATA_SPEC_COST_TABLE form (generate.cpp:16495-16511): comma-separated round
ms per width, one list a context bucket (<6K, <24K, longer), weight 6 rounds
each (DraftPolicy::seed_costs, draft_policy.cpp:66-74). Without it, every
width the trace did not run is Strata's kShape guess; the report says so.

`--costs-from <leg traces>` measures the per-width mean round ms of the
fixed-width legs by context bucket and seeds those (weight 6). Replay the legs
ALONE first to read their means, then the arm with `--costs-from` the legs: an
arm and its legs in ONE history would learn the arm's first rounds by kShape
and pour the legs' rows into the arm's histogram and tables. An explicit
`--seed-costs` value for a width wins over `--costs-from` (its T1, see
limitation 7) and both lose to a round of the arm's own that measured it.

Standard library only: the recipe's card host has python3 and no packages.
"""

import argparse
import json
import math
import sys

KMAXT = 8            # draft_policy.hpp:25, window rows
KCTXBUCKETS = 3      # draft_policy.hpp:27, <6K / <24K / longer
KPBUCKETS = 8        # draft_policy.cpp:174-179, p_bucket = int(p * 8)
KSHAPE = (0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6)   # draft_policy.cpp:13
KCOSTALPHA = 0.1     # draft_policy.cpp:14
KTOKALPHA = 0.05     # draft_policy.cpp:15
KCALDECAY = 0.995    # draft_policy.cpp:16
KPRIORN = 4.0        # draft_policy.cpp:21
KPROBES = 3          # draft_policy.cpp:22, an unmeasured width is probed this often
MARGIN = 0.03        # draft_policy.hpp:32

# the trace's field order (src/exec/mtp_trace.h, format mtp2); the parser
# rejects any other
KEYS = ("req", "seq", "pos", "rows", "acc", "nmax", "minp", "tp", "tv", "ms", "p")
INTKEYS = ("req", "seq", "pos", "rows", "acc", "nmax")


class TraceError(Exception):
    pass


def p_bucket(p):
    """Strata's bucket: int(p * 8) clamped 0..7 (draft_policy.cpp:174-179)."""
    b = int(p * 8.0)
    return 0 if b < 0 else 7 if b > 7 else b


def parse_line(line):
    """One trace line -> a record (dict). Malformed input fails loudly."""
    if not line.startswith("mtp2 "):
        raise TraceError("no mtp2 prefix: %r" % line[:24])
    fields = line[5:].split(" ")
    if len(fields) != len(KEYS):
        raise TraceError("%d fields, %d expected: %r" % (len(fields), len(KEYS), line))
    rec = {}
    for key, fld in zip(KEYS, fields):
        if not fld.startswith(key + "="):
            raise TraceError("field %r out of order (wanted %s=): %r" % (fld, key, line))
        v = fld[len(key) + 1:]
        if key in INTKEYS:
            try:
                rec[key] = int(v)
            except ValueError:
                raise TraceError("field %s not an integer: %r" % (key, v))
        elif key == "p":
            rec["probs"] = [float(x) for x in v.split(",") if x != ""]
        else:
            try:
                rec[key] = float(v)
            except ValueError:
                raise TraceError("field %s not a number: %r" % (key, v))
    if rec["rows"] < 1 or rec["acc"] <= -1:
        raise TraceError("rows/acc out of range: %r" % line)
    # the probability list's two shapes (exec/mtp_trace.h): rows - 1 kept drafts,
    # or rows when the min-p floor stopped the chain with a stopped draft's own
    # probability appended. Anything else is not a line this writer makes.
    if len(rec["probs"]) not in (rec["rows"] - 1, rec["rows"]):
        raise TraceError("%d probabilities for %d rows: %r" % (len(rec["probs"]), rec["rows"], line))
    # a kept draft is never below the floor: the writer stops the chain there
    # (llama_spec.cpp:628). Only the floor-stopping draft (the last, when the
    # list is `rows` long) may sit below minp.
    for i, p in enumerate(rec["probs"][:rec["rows"] - 1]):
        if p < rec["minp"]:
            raise TraceError("kept draft %d at %.6f below the floor %.6f: %r" % (i, p, rec["minp"], line))
    return rec


def render(rec):
    """A record -> the writer's line, byte-identical to parse-then-render."""
    probs = ",".join("%.6f" % p for p in rec["probs"])
    return ("mtp2 req=%d seq=%d pos=%d rows=%d acc=%d nmax=%d minp=%.6f tp=%.3f tv=%.3f ms=%.3f p=%s"
            % (rec["req"], rec["seq"], rec["pos"], rec["rows"], rec["acc"], rec["nmax"], rec["minp"],
               rec["tp"], rec["tv"], rec["ms"], probs))


def read_trace(path):
    """A file -> its records, in order. Blank lines are skipped; a bad line
    raises. Each record carries `src`, the file it came from: a request counter
    restarts per process, so request identity is (file, req)."""
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rec = parse_line(line)
            except TraceError as e:
                raise TraceError("%s:%d: %s" % (path, n, e))
            rec["src"] = path
            recs.append(rec)
    return recs


class ChainState:
    """DraftPolicy::ChainState (draft_policy.hpp:57-62)."""
    __slots__ = ("q", "e", "best_v", "best_e", "best_rows", "kept")

    def __init__(self):
        self.q = 1.0
        self.e = 1.0
        self.best_v = 0.0
        self.best_e = 1.0     # E at best_rows: rows 1 before any draft
        self.best_rows = 1
        self.kept = 0


class DraftPolicy:
    """The MTP half of Strata's DraftPolicy: cost by width and context bucket,
    the probability -> acceptance table, and keep_draft. The lookup drafter's
    choose/chain and stale() are not ported (the MTP rule does not call them)."""

    def __init__(self, max_t=KMAXT, margin=MARGIN, robust=True, freeze_costs=False):
        self.max_t = max(1, min(max_t, KMAXT))
        self.margin = margin
        self.robust = robust            # STRATA_SPEC_COST=1 sets set_robust(true) (generate.cpp:16509)
        self.freeze_costs = freeze_costs
        self.cost = [[0.0] * (KMAXT + 1) for _ in range(KCTXBUCKETS)]
        self.cost_n = [[0.0] * (KMAXT + 1) for _ in range(KCTXBUCKETS)]
        self.bk = 0
        self.mtp_tok = [0.0] * (KMAXT + 1)
        self.mtp_n = [0.0] * (KMAXT + 1)
        self.pa_ok = [0.0] * KPBUCKETS
        self.pa_n = [0.0] * KPBUCKETS
        self.rounds = 0.0

    # -- cost ---------------------------------------------------------------
    def cost_ms(self, t):
        """draft_policy.cpp:40-53: the measured value, else the kShape scaling
        of the measured widths, weighted by how often each was seen."""
        t = max(1, min(t, KMAXT))
        if self.cost_n[self.bk][t] > 0:
            return self.cost[self.bk][t]
        num = den = 0.0
        for u in range(1, KMAXT + 1):
            if self.cost_n[self.bk][u] > 0:
                w = min(self.cost_n[self.bk][u], 20.0)
                num += w * self.cost[self.bk][u] * KSHAPE[t] / KSHAPE[u]
                den += w
        return num / den if den > 0 else KSHAPE[t]

    def clamp_sample(self, t, ms):
        """draft_policy.cpp:55-65 with robust on: a width's first measurement is
        replaced by the guess from the other widths; later ones cap at 2x."""
        if not self.robust:
            return ms
        known = self.cost_ms(t)
        if self.cost_n[self.bk][t] == 0:
            for u in range(1, KMAXT + 1):
                if self.cost_n[self.bk][u] > 0:
                    return known
            return ms
        return min(ms, 2.0 * known)

    def seed_costs(self, ms, weight, bucket=0):
        """draft_policy.cpp:66-74 (STRATA_SPEC_COST_TABLE, generate.cpp:16495-16511)."""
        bucket = max(0, min(bucket, KCTXBUCKETS - 1))
        for t in range(1, min(len(ms), KMAXT) + 1):
            if ms[t - 1] > 0.0 and self.cost_n[bucket][t] == 0:
                self.cost[bucket][t] = ms[t - 1]
                self.cost_n[bucket][t] = weight

    def set_context(self, pos):
        """draft_policy.cpp:76-93: the context bucket of the next round's rows."""
        if not self.robust:
            return
        nb = 0 if pos < 6000 else 1 if pos < 24000 else 2
        if nb == self.bk:
            return
        has = any(self.cost_n[nb][t] > 0 for t in range(1, KMAXT + 1))
        if not has:
            for d in range(1, KCTXBUCKETS):
                if has:
                    break
                for src in (nb - d, nb + d):
                    if src < 0 or src >= KCTXBUCKETS or has:
                        continue
                    for t in range(1, KMAXT + 1):
                        if self.cost_n[src][t] > 0:
                            self.cost[nb][t] = self.cost[src][t]
                            self.cost_n[nb][t] = 1.0
                            has = True
        self.bk = nb

    def cost_table_text(self):
        """draft_policy.cpp:98-113, the log line Strata prints at a run's end."""
        out = []
        for k in range(KCTXBUCKETS):
            if not any(self.cost_n[k][t] > 0 for t in range(1, KMAXT + 1)):
                continue
            name = "ctx<6K" if k == 0 else "ctx<24K" if k == 1 else "ctx>=24K"
            fields = name + ":" + "".join(" T%d:%.2f(n=%.0f)" % (t, self.cost[k][t], self.cost_n[k][t])
                                          for t in range(1, KMAXT + 1) if self.cost_n[k][t] > 0)
            out.append(fields)
        return " | ".join(out)

    # -- round outcome ------------------------------------------------------
    def mtp_tokens(self, t):
        """draft_policy.cpp:118-121."""
        if self.mtp_n[t] > 0:
            return self.mtp_tok[t]
        return 1.0 + 0.7 * (t - 1)

    def observe(self, t, accepted, round_ms):
        """draft_policy.cpp:155-170 with lookup=false: the measured round cost of
        a window of `t` rows and the tokens it committed."""
        t = max(1, min(t, KMAXT))
        if round_ms > 0 and not self.freeze_costs:
            ms = self.clamp_sample(t, round_ms)
            alpha = max(KCOSTALPHA, 1.0 / (self.cost_n[self.bk][t] + 1.0)) if self.robust else KCOSTALPHA
            cn = self.cost_n[self.bk][t]
            self.cost[self.bk][t] = (1.0 - alpha) * self.cost[self.bk][t] + alpha * ms if cn > 0 else ms
            self.cost_n[self.bk][t] += 1.0
        self.rounds += 1.0
        got = accepted + 1.0
        n = self.mtp_n[t]
        self.mtp_tok[t] = (1.0 - KTOKALPHA) * self.mtp_tok[t] + KTOKALPHA * got if n > 0 else got
        self.mtp_n[t] += 1.0

    def draft_rate(self, p):
        """draft_policy.cpp:182-188: until a bucket has data, a draft holds as
        often as the MTP says (the prior is `p` itself, worth kPriorN)."""
        b = p_bucket(p)
        centre = max(0.0, min(p, 1.0))
        return (self.pa_ok[b] + KPRIORN * centre) / (self.pa_n[b] + KPRIORN)

    def observe_probs(self, probs, n_window_drafts, accepted):
        """draft_policy.cpp:190-196: draft j was reached when the drafts before
        it held (j <= accepted) and held when j < accepted."""
        for j in range(n_window_drafts):
            if j > accepted:
                break
            b = p_bucket(probs[j])
            self.pa_ok[b] = KCALDECAY * self.pa_ok[b] + (1.0 if j < accepted else 0.0)
            self.pa_n[b] = KCALDECAY * self.pa_n[b] + 1.0

    # -- the rule -----------------------------------------------------------
    def keep_draft(self, st, idx, p, cap):
        """draft_policy.cpp:199-224, verbatim. Returns whether to compute the
        next draft; st.best_rows is the window to verify."""
        cap = max(1, min(cap, self.max_t))
        if idx == 0:
            st.__init__()
            st.best_v = 1.0 / self.cost_ms(1)
        a = self.draft_rate(p)
        st.q *= a
        st.e += st.q
        rows = idx + 2
        if rows <= cap:
            v = st.e / self.cost_ms(rows)
            if v > st.best_v * (1.0 + self.margin / 3.0):
                st.best_v = v
                st.best_e = st.e     # E of the width it would verify (not the chain's end)
                st.best_rows = rows
            if self.cost_n[self.bk][rows] < KPROBES and st.q >= 0.15:
                st.best_e = st.e
                st.best_rows = rows
        st.kept = st.best_rows - 1
        if rows + 1 > cap:
            return False
        v1 = (st.e + st.q * a) / self.cost_ms(rows + 1)
        v2 = ((st.e + st.q * a + st.q * a * a) / self.cost_ms(rows + 2)) if rows + 2 <= cap else 0.0
        return max(v1, v2) > st.best_v * (1.0 + self.margin / 3.0) or \
            (self.cost_n[self.bk][rows + 1] < KPROBES and st.q * a >= 0.15)


def replay(records, max_t=KMAXT, margin=MARGIN, constant_cost=None, respects_floor=False,
           seed_costs=None, costs_from=None):
    """One trace as a history: the fixed-width numbers from what ran, and the
    policy's would-be numbers from the same rounds' probabilities and times.

    `seed_costs` is Strata's STRATA_SPEC_COST_TABLE form already parsed: one
    list of round ms per width a context bucket, seeded as `weight` rounds
    (DraftPolicy::seed_costs, generate.cpp:16495-16511). `costs_from` is the
    same form measured from the fixed-width legs (see measured_costs): it fills
    only a width the seed or an arm's own rounds did not, so an explicitly
    seeded T1 stands.
    """
    policy = DraftPolicy(max_t=max_t, margin=margin, robust=True,
                         freeze_costs=constant_cost is not None)
    if constant_cost is not None:
        # the test hook: one cost for every width of every bucket, so the
        # policy's arithmetic is checkable and nothing is measured by accident
        for b in range(KCTXBUCKETS):
            policy.seed_costs([constant_cost] * KMAXT, 6.0, b)
    if seed_costs:
        for b, ms in enumerate(seed_costs[:KCTXBUCKETS]):
            policy.seed_costs(ms, 6.0, b)
    if costs_from:
        for b, ms in enumerate(costs_from[:KCTXBUCKETS]):
            policy.seed_costs(ms, 6.0, b)   # skips a width already seeded/measured

    widths = {}          # rows actually verified -> [rounds, tokens, ms, rounds with ms]
    hist = {}            # rows the policy would verify -> count
    exp_tok = []         # the policy's own E at its chosen width, a round
    exp_v = []           # the policy's own E/cost, tokens a ms (best_v)
    trunc_tok = []       # min(acc, chosen drafts) + 1, the truncated round inferred
    trunc_ms = []        # that round's chosen width's round time
    trunc_v = []         # that round's tokens / its chosen width's cost
    cal = [[0, 0] for _ in range(KPBUCKETS)]   # reached, held, from the trace
    sums = {"ms": 0.0, "tp": 0.0, "tv": 0.0}
    wider_than_floor = 0
    never_drafted = 0
    unknown_ms = 0
    guessed_width = 0
    requests = set()

    for r in records:
        requests.add((r.get("src"), r["req"]))   # the counter restarts per process
        policy.set_context(r["pos"])
        rows = r["rows"]
        acc = r["acc"]
        known_ms = r["ms"] > 0
        widths.setdefault(rows, [0, 0, 0.0, 0])
        widths[rows][0] += 1
        widths[rows][1] += acc + 1            # tokens a round: 1 + drafts kept
        if known_ms:
            widths[rows][2] += r["ms"]
            widths[rows][3] += 1
            sums["ms"] += r["ms"]
        else:
            unknown_ms += 1                       # ms <= 0: no cost from this round
        sums["tp"] += r["tp"]
        sums["tv"] += r["tv"]

        # a(p) from the trace's own reached/held counts (raw, not the decayed
        # table the policy keeps - the two are printed side by side)
        for j, p in enumerate(r["probs"][:rows - 1]):
            b = p_bucket(p)
            cal[b][0] += 1
            if j < acc:
                cal[b][1] += 1

        # the chain: every probability the trace holds, in order, through
        # keep_draft, until it says stop or the recorded chain ends
        st = ChainState()
        cap = max(1, min(r["nmax"] + 1, max_t))    # *_llama-mtp N drafts -> N + 1 rows
        nxt = 0
        for idx, p in enumerate(r["probs"]):
            if respects_floor and p < r["minp"]:
                break                                # the floor keeps its veto
            go = policy.keep_draft(st, idx, p, cap)
            nxt += 1
            if not go:
                break
        chosen = max(1, st.best_rows)
        hist[chosen] = hist.get(chosen, 0) + 1
        exp_tok.append(st.best_e)   # E at the width it would verify, not the chain's end
        if r["probs"]:
            exp_v.append(st.best_v)   # a round with no drafts has no rate
        if chosen - 1 > rows - 1:
            wider_than_floor += 1                    # the policy wanted a width the round did not run
        if len(r["probs"]) < r["nmax"] and rows - 1 < chosen - 1:
            never_drafted += 1                       # drafts it needs were never made
        wc = widths.get(chosen)
        if not (wc and wc[3] > 0) and policy.cost_n[policy.bk][chosen] == 0:
            guessed_width += 1                       # that width's cost is the kShape guess

        # the truncated round inferred: the walked prefix kept, one more token,
        # at the measured cost of that width when the trace has one
        kept = min(acc, chosen - 1)
        trunc_tok.append(kept + 1)
        tms = _width_mean_ms(widths, chosen, policy)
        trunc_ms.append(tms)

        # the observer: the round's own window and its outcome. A line with
        # `ms` <= 0 has no round time, so observe() is skipped ENTIRELY - no
        # cost and no window-token EMA; the a(p) table below still learns, as
        # Strata has no round without a time.
        if known_ms:
            policy.observe(rows, acc, r["ms"])
        policy.observe_probs(r["probs"], max(0, min(rows - 1, len(r["probs"]))), acc)

    top = max(hist, key=lambda k: hist[k]) if hist else 1
    exp_tok_mean = _mean(exp_tok)
    mode_cost = _width_mean_ms(widths, top, policy)
    return {
        "rounds": len(records),
        "requests": len(requests),
        "widths": widths,
        "hist": hist,
        "exp_tok": exp_tok_mean,
        "exp_e_rounds": exp_tok,
        # the policy's own E/cost at the width it would verify, averaged over
        # the rounds: not the mean E over the mode width's cost (that mixed two
        # widths' numbers)
        "exp_v": _mean(exp_v),
        "exp_v_rounds": exp_v,
        "mode_cost": mode_cost,
        "trunc_tok": _mean(trunc_tok),
        "trunc_ms": _mean(trunc_ms),
        # the truncated round's own rate: its inferred tokens over its chosen
        # width's cost, summed (a ratio of means, not a mean of ratios)
        "trunc_v": (sum(trunc_tok) / sum(trunc_ms)) if sum(trunc_ms) > 0 else 0.0,
        "cal": cal,
        "sums": sums,
        "wider_than_floor": wider_than_floor,
        "never_drafted": never_drafted,
        "unknown_ms": unknown_ms,
        "guessed_width": guessed_width,
        "policy": policy,
        "minp_seen": sorted({r["minp"] for r in records}),
    }


def parse_cost_table(text):
    """Strata's STRATA_SPEC_COST_TABLE form -> one ms list a context bucket:
    "T1,T2,..;T1,..;T1,..", the round ms per width (generate.cpp:16495-16511)."""
    buckets = []
    for part in text.split(";"):
        if len(buckets) >= KCTXBUCKETS:
            break
        vals = []
        for x in part.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                vals.append(float(x))
            except ValueError:
                raise TraceError("--seed-costs %r: %r is not a number" % (text, x))
        buckets.append(vals)
    if not buckets:
        raise TraceError("--seed-costs %r: no lists" % text)
    return buckets


def measured_costs(records):
    """The fixed-width legs' measured round ms per width a context bucket: the
    mean over the rounds that ran that width with a known time, 0 where none
    did. Feed it to replay(..., costs_from=...) so the arm's replay prices a
    width by a measured leg instead of Strata's kShape guess."""
    acc = [[[0.0, 0] for _ in range(KMAXT + 1)] for _ in range(KCTXBUCKETS)]
    for r in records:
        if r["ms"] <= 0:
            continue
        b = 0 if r["pos"] < 6000 else 1 if r["pos"] < 24000 else 2
        t = max(1, min(r["rows"], KMAXT))
        acc[b][t][0] += r["ms"]
        acc[b][t][1] += 1
    return [[(acc[b][t][0] / acc[b][t][1]) if acc[b][t][1] else 0.0 for t in range(1, KMAXT + 1)]
            for b in range(KCTXBUCKETS)]


def _width_mean_ms(widths, rows_wanted, policy):
    """The measured mean round ms of a width the trace ran, else the policy's
    own estimate (seeded, EMA-learned, or kShape-scaled) - limitation 3/6."""
    w = widths.get(rows_wanted)
    if w and w[3] > 0:
        return w[2] / w[3]
    return policy.cost_ms(rows_wanted)


def _mean(v):
    return sum(v) / len(v) if v else 0.0


def report(path, rep, out):
    p = rep["policy"]
    w = rep["widths"]
    out.append("trace %s: %d rounds, %d requests, %d widths, min-p seen %s"
               % (path, rep["rounds"], rep["requests"], len(w), ",".join("%.3f" % m for m in rep["minp_seen"])))
    # the fixed widths that ran: tokens a round, tokens a ms
    dom = max(w, key=lambda k: w[k][0])
    out.append("  fixed (what ran):")
    for rows in sorted(w):
        n, tok, ms, msn = w[rows]
        if msn > 0:
            out.append("    width %d: %d rounds (%d with a round time), %.2f tokens a round, %.2f ms a round,"
                       " %.4f tokens a ms%s"
                       % (rows, n, msn, tok / n, ms / msn, (tok / n) / (ms / msn),
                          "  <- dominant" if rows == dom else ""))
        else:
            out.append("    width %d: %d rounds, %.2f tokens a round, no measured round time%s"
                       % (rows, n, tok / n, "  <- dominant" if rows == dom else ""))
    n0, tok0, ms0, msn0 = w[dom]
    if msn0 > 0:
        out.append("  fixed dominant width %d: %.2f tokens a round, %.4f tokens a ms"
                   % (dom, tok0 / n0, (tok0 / n0) / (ms0 / msn0)))
    # the policy: its own expectation, and the truncated round inferred from the
    # rounds' acceptance
    hist = rep["hist"]
    top = max(hist, key=lambda k: hist[k]) if hist else 1
    out.append("  policy (cost-aware keep_draft):")
    out.append("    width histogram: %s (mode %d)"
               % (" ".join("%d:%d" % (k, hist[k]) for k in sorted(hist)), top))
    out.append("    expected tokens a round: %.2f (the policy's own E at its chosen width, mean)" % rep["exp_tok"])
    out.append("    expected tokens a ms:   %.4f (mean of the policy's own E over the cost of the width it would"
               " verify, per round)" % rep["exp_v"])
    out.append("    truncated round inferred: %.2f tokens, %.2f ms, %.4f tokens a ms (mean of that round's"
               " tokens / its chosen width's cost)"
               % (rep["trunc_tok"], rep["trunc_ms"], rep["trunc_v"]))
    # a(p)
    out.append("  a(p), the accept rate by draft-probability bucket (raw trace, then the policy's table):")
    for b in range(KPBUCKETS):
        reached, held = rep["cal"][b]
        raw = ("%.3f" % (held / reached)) if reached else "  -  "
        out.append("    p in [%.2f,%.2f): reached %5d, held %5d, accept %s | policy draft_rate %.3f"
                   % (b / 8.0, (b + 1) / 8.0, reached, held, raw, p.draft_rate((b + 0.5) / 8.0)))
    # the open question: tp + tv against the round's own duration
    s = rep["sums"]
    measured = rep["rounds"] - rep["unknown_ms"]
    if s["ms"] > 0 and measured > 0:
        out.append("  open question: mean round duration (its own start to the next start) %.2f ms,"
                   " mean tp+tv %.2f ms, gap %.2f ms (%.1f %% of the round)"
                   % (s["ms"] / measured, (s["tp"] + s["tv"]) / rep["rounds"],
                      (s["ms"] - s["tp"] - s["tv"]) / measured,
                      100.0 * (s["ms"] - s["tp"] - s["tv"]) / s["ms"]))
    out.append("  learned cost table: %s" % (p.cost_table_text() or "(nothing measured)"))
    if not any(p.cost_n[b][1] > 0 for b in range(KCTXBUCKETS)):
        out.append("  NOTE: T1 is never measured by a trace (rows = 1 only at a budget end; --llama-mtp 0"
                   " writes none). The MTP-off leg (fn_sweep.py --no-mtp) decode t/s gives it: T1 round ms"
                   " = 1000/t_s, passed by --seed-costs (limitation 7).")
    if rep["rounds"]:
        out.append("  NOTE: on, a round pays the head's top probability per draft (~0.8 ms a draft on the host"
                   " for a 248,320-row head); the serving path pays none of it, so these widths price high -"
                   " compare a trace-off arm (limitation 6).")
    if rep["guessed_width"]:
        out.append("  WARNING: %d rounds priced their chosen width by Strata's kShape guess (no round of the"
                   " trace ran it and none is measured or seeded): pass --seed-costs with the fixed-width legs'"
                   " round ms (limitation 3)." % rep["guessed_width"])
    if rep["unknown_ms"]:
        out.append("  WARNING: %d rounds carry ms <= 0 (unknown round time): no cost is learned from them."
                   % rep["unknown_ms"])
    if rep["minp_seen"] and any(m > 0 for m in rep["minp_seen"]):
        out.append("  WARNING: this trace's min-p > 0: drafts past the floor were never made, so a width the"
                   " policy would want is unpriced (limitation 2). Run the min-p 0 arm.")
    if rep["wider_than_floor"]:
        out.append("  WARNING: %d rounds where the policy wanted a wider window than the trace ran (limitation 1/3)."
                   % rep["wider_than_floor"])
    if rep["never_drafted"]:
        out.append("  WARNING: %d rounds where the policy needed drafts the ceiling never made: run ceiling+1"
                   % rep["never_drafted"])


def main(argv):
    ap = argparse.ArgumentParser(description="replay Strata's cost-aware MTP window over an arcint trace")
    ap.add_argument("trace", nargs="+", help="ARCINT_MTP_TRACE file(s), one history")
    ap.add_argument("--max-t", type=int, default=KMAXT, help="the widest window in rows (Strata kMaxT 8)")
    ap.add_argument("--margin", type=float, default=MARGIN, help="Strata's switch margin (0.03)")
    ap.add_argument("--constant-cost", type=float, default=None,
                    help="test hook: force every width's cost to this many ms")
    ap.add_argument("--seed-costs", default=None,
                    help="Strata's STRATA_SPEC_COST_TABLE form: \"T1,T2,..;T1,..;T1,..\" round ms per width,"
                         " one list a context bucket (<6K, <24K, longer)")
    ap.add_argument("--costs-from", nargs="+", default=None, metavar="TRACE",
                    help="fixed-width leg traces: seed each width's cost from its measured round ms"
                         " (per context bucket) instead of Strata's kShape guess; an explicit"
                         " --seed-costs value wins")
    ap.add_argument("--respect-floor", action="store_true",
                    help="keep --llama-mtp-min-p's veto (default: the min-p 0 arm)")
    ap.add_argument("--json", action="store_true", help="the report as JSON")
    a = ap.parse_args(argv)

    recs = []
    for t in a.trace:
        recs.extend(read_trace(t))
    seeds = parse_cost_table(a.seed_costs) if a.seed_costs else None
    from_legs = measured_costs([r for t in a.costs_from for r in read_trace(t)]) if a.costs_from else None
    rep = replay(recs, max_t=a.max_t, margin=a.margin, constant_cost=a.constant_cost,
                 respects_floor=a.respect_floor, seed_costs=seeds, costs_from=from_legs)
    if a.json:
        out = {
            "rounds": rep["rounds"], "requests": rep["requests"],
            "fixed": {str(k): {"rounds": v[0], "measured_rounds": v[3], "tokens": v[1] / v[0],
                               "ms": v[2] / v[3] if v[3] else None,
                               "tokens_per_ms": (v[1] / v[0]) / (v[2] / v[3]) if v[3] else None}
                      for k, v in rep["widths"].items() if v[0]},
            "policy": {"histogram": {str(k): v for k, v in rep["hist"].items()},
                       "expected_tokens": rep["exp_tok"], "expected_tokens_per_ms": rep["exp_v"],
                       "truncated_tokens": rep["trunc_tok"], "truncated_ms": rep["trunc_ms"],
                       "truncated_tokens_per_ms": rep["trunc_v"],
                       "cost_table": rep["policy"].cost_table_text(),
                       "expected_e_rounds": rep["exp_e_rounds"]},
            "calibration": [{"bucket": b, "reached": v[0], "held": v[1]} for b, v in enumerate(rep["cal"])],
            "tp_tv_vs_round_duration": {k: rep["sums"][k] for k in ("ms", "tp", "tv")},
            "warnings": {"wider_than_ran": rep["wider_than_floor"], "never_drafted": rep["never_drafted"],
                         "unknown_ms": rep["unknown_ms"], "guessed_width": rep["guessed_width"],
                         "minp": rep["minp_seen"],
                         "t1_unmeasured": not any(rep["policy"].cost_n[b][1] > 0 for b in range(KCTXBUCKETS))},
        }
        print(json.dumps(out, indent=2))
        return 0
    lines = []
    report(", ".join(a.trace), rep, lines)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
