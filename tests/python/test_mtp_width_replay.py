#!/usr/bin/env python3
"""tools/mtp_width_replay.py, device-free: the trace format's round trip, the
fixed-width arithmetic, and the port of Strata's cost-aware MTP window.

Campaign research-mtp-width-policy, section 6 step 1: the replay is what prices
the policy before any card leg. Every cell was red first:

1. The format round trip. The shared fixture (tests/python/fixtures/
   mtp_trace_lines.txt, also read by tests/test_mtp_trace.cpp) is parsed by
   this tool's parser and rendered back; every line is byte-identical, and the
   fields mean what they say (rows, acc, the probability list's two shapes).
   Red mutation seen failing: render prints `acc` before `rows` -- all four
   lines differ. Second mutation seen failing: parse_line accepts any prefix
   (the `mtp2 ` check dropped, an `mtp1` line parses) -- the malformed-line
   cell stops raising.
   Third, in the bucket map: p_bucket uses int(p * 8) + 1 -- the 0.812345
   fixture value lands in bucket 7 instead of 6 and the calibration cell fails.

2. The fixed-width numbers. A trace whose rounds all ran one width and one
   constant round time (the round's own duration) must give tokens a round =
   acc + 1 exactly and
   tokens a ms = that over the round time. Red mutation seen failing: tokens =
   acc instead of acc + 1 -- 3.0 where 4.0 is expected. Also: with one constant
   cost for every width (--constant-cost), the policy's chain runs to the cap,
   which is the ceiling's own arithmetic (nmax + 1 rows) -- the mutation
   `cap = nmax` (off by one row) puts the mode width at 4 instead of 5. Third,
   the truncated round: `kept = acc` instead of `min(acc, chosen - 1)` emits 5
   tokens where the cell expects 1.

3. The keep_draft port against Strata's own vectors, where they apply. Strata's
   src/spec/draft_policy_test.cpp has NO keep_draft vectors (its five blocks
   exercise choose()/lookup only); what its observe() calls produce does apply:
   the cost(t) = 19 + 10.5*(t-1) shape and the committed-token EMA, plus the
   prior `1 + 0.7*(t-1)` (draft_policy.cpp:118-121). The rule's own vectors are
   derived here from draft_policy.cpp:199-224 with exact arithmetic.
   Red mutations seen failing: the margin factor `1 + margin/3` -> `1 + margin`
   (`the_margin_is_three_percent_over_three` turns best_rows from 2 to 1); the
   reached rule `j < accepted` -> `j <= accepted` (`observe_probs_reached...`);
   the EMA weight `max(kCostAlpha, 1/(n+1))` -> kCostAlpha
   (`the_cost_ema_is_a_running_mean_for_the_first_samples`).

4. The limits are reported, not silently absorbed: a min-p > 0 trace warns that
   drafts past the floor were never made (limitation 2), a round where the
   policy wanted more drafts than the ceiling made warns for ceiling+1, and
   every report names the trace's own host cost (limitation 6).

5. Round 2's cells. A kept draft below min-p is rejected at parse (red: the
   floor check dropped accepts a line the writer cannot make); a round with
   `ms` <= 0 learns no cost and is counted as unknown (red: `known_ms = True`
   prices the 0.125 ms first round as a width cost); `--seed-costs` takes
   Strata's STRATA_SPEC_COST_TABLE form at weight 6 (red: weight 1 changes the
   EMA after seeding); the policy's rate is the mean of its own E/cost per
   round, not E over the mode width's cost (red: that substitution breaks the
   cell's equality against the per-round list).

6. Round 3's cells. E is reported at the width the policy would verify, not at
   the chain's end (red: append st.e gives 2.764 where the reviewer's chain has
   E 2.62 at chosen width 3); an unmeasured width is probed through the ratio
   (red: `KPROBES = 0` leaves the policy on width 2 where the chain chose 3);
   the legs' measured means seed the arm's cost table per context bucket and
   an explicit T1 seed wins (red: seed_costs overwriting whatever it finds lets
   the legs' T1 replace the seed); a request counter restarts per process, so
   requests are keyed by (file, req) (red: keyed by req alone, two files with
   req 1 count as one); the cost table names a bucket once (red: per-width
   naming repeats it); an MTP-only trace says T1 is never measured.

Standard library only (unittest) so the recipe's `ctest -L unit` runs it on a
build host with python3 and nothing else; pytest collects the class too.
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import mtp_width_replay as mw  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mtp_trace_lines.txt"
POLICY_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mtp_policy_trace.txt"


def write_trace(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(mw.render(r) + "\n")


def rec(req=1, seq=1, pos=4096, rows=5, acc=3, nmax=4, minp=0.0, tp=4.0, tv=44.0, ms=50.0,
        probs=(0.9, 0.9, 0.9, 0.9)):
    return dict(req=req, seq=seq, pos=pos, rows=rows, acc=acc, nmax=nmax, minp=minp, tp=tp, tv=tv,
                ms=ms, probs=list(probs))


class TraceFormat(unittest.TestCase):
    def test_fixture_round_trips_byte_identical(self):
        lines = [l.rstrip("\n") for l in FIXTURE.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertGreaterEqual(len(lines), 3)
        for line in lines:
            self.assertEqual(mw.render(mw.parse_line(line)), line)

    def test_the_fields_mean_what_they_say(self):
        lines = [l.rstrip("\n") for l in FIXTURE.read_text(encoding="utf-8").splitlines() if l.strip()]
        r0 = mw.parse_line(lines[0])           # ceiling end: rows - 1 probabilities
        self.assertEqual((r0["rows"], r0["acc"], r0["nmax"]), (4, 3, 3))
        self.assertEqual(len(r0["probs"]), r0["rows"] - 1)
        r1 = mw.parse_line(lines[1])           # floor stop: rows probabilities, the last below minp
        self.assertEqual(len(r1["probs"]), r1["rows"])
        self.assertLess(r1["probs"][-1], r1["minp"])
        r2 = mw.parse_line(lines[2])           # no drafts: rows 1, empty list
        self.assertEqual((r2["rows"], r2["nmax"], r2["probs"]), (1, 0, []))
        # the value precision survives (0.812345 must not read back as 0.812)
        self.assertAlmostEqual(r0["probs"][0], 0.812345, places=6)

    def test_malformed_lines_fail_loudly(self):
        # red: without the `mtp2 ` prefix check this parses as a record
        with self.assertRaises(mw.TraceError):
            # otherwise valid, another writer's prefix: only the prefix check refuses it
            mw.parse_line("mtp1 req=1 seq=1 pos=0 rows=1 acc=0 nmax=0 minp=0.000000 tp=0.000 tv=0.000 ms=1.000 p=")
        # red: without the field-order check, swapped rows/acc parses silently wrong
        with self.assertRaises(mw.TraceError):
            mw.parse_line("mtp2 req=1 seq=1 pos=0 acc=3 rows=4 nmax=3 minp=0.000000 tp=1.000 tv=1.000 ms=1.000 p=0.5")
        with self.assertRaises(mw.TraceError):
            mw.parse_line("mtp2 req=1 seq=1 pos=0 rows=4 acc=3 nmax=3 minp=0.000000 tp=1.000 tv=1.000 ms=1.000 p=0.5")
        # red: without the floor check, a kept draft below minp parses as a line
        # the writer cannot make (llama_spec.cpp:628 stops the chain there)
        with self.assertRaises(mw.TraceError):
            mw.parse_line("mtp2 req=1 seq=1 pos=0 rows=3 acc=2 nmax=2 minp=0.500000 tp=1.000 tv=2.000"
                          " ms=3.000 p=0.900000,0.400000")


class CalibrationBuckets(unittest.TestCase):
    def test_a_draft_probability_lands_in_its_own_bucket(self):
        # p_bucket = int(p * 8) clamped 0..7 (draft_policy.cpp:174-179): 0.812345*8
        # is 6.4988, so bucket 6, not 7. Red: `int(p*8) + 1` puts it in 7.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.txt"
            write_trace(path, [rec(rows=2, acc=1, nmax=1, probs=(0.812345,))])
            rep = mw.replay(mw.read_trace(path))
        self.assertEqual(rep["cal"][6], [1, 1])
        self.assertEqual(rep["cal"][7], [0, 0])


class PolicyExpectedTokens(unittest.TestCase):
    def test_e_is_at_the_chosen_width_not_the_chain_end(self):
        # the reviewer's chain: a(p) has no data, so a(p) = p (the prior), and
        # probs 0.9, 0.8, 0.2 give E 1.90, 2.62, 2.764; best_rows 3 (a probe
        # at idx1, then q = 0.144 < 0.15 and E4/cost4 does not beat it)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.txt"
            write_trace(path, [rec(rows=4, acc=2, nmax=3, ms=0.0, probs=(0.9, 0.8, 0.2))])
            rep = mw.replay(mw.read_trace(path))
        self.assertEqual(rep["hist"], {3: 1})
        self.assertAlmostEqual(rep["exp_tok"], 2.62, places=2)
        self.assertEqual(rep["exp_e_rounds"], [rep["exp_tok"]])
        # the chain's own E, step by step: keep_draft's st.e is 1.90, 2.62,
        # 2.764; the window it verifies is 3 rows with E 2.62
        st = mw.ChainState()
        p = mw.DraftPolicy()
        es = []
        for idx, pp in enumerate((0.9, 0.8, 0.2)):
            p.keep_draft(st, idx, pp, 4)
            es.append(st.e)
        for got, want in zip(es, (1.90, 2.62, 2.764)):
            self.assertAlmostEqual(got, want, places=3)
        self.assertEqual(st.best_rows, 3)
        self.assertAlmostEqual(st.best_e, 2.62, places=3)

    def test_an_unmeasured_width_is_probed_through_the_ratio(self):
        # p 0.39 each: E/cost falls going to width 3 (kShape grows faster than
        # E), but q after idx1 is 0.152 >= 0.15 and width 3 is unmeasured, so
        # the probe takes it: chosen 3 with E 1.5421, not 2 with E 1.39
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.txt"
            write_trace(path, [rec(rows=4, acc=2, nmax=3, ms=0.0, probs=(0.39, 0.39, 0.39))])
            rep = mw.replay(mw.read_trace(path))
        self.assertEqual(rep["hist"], {3: 1})
        self.assertAlmostEqual(rep["exp_tok"], 1.5421, places=3)


class CostsFrom(unittest.TestCase):
    def test_the_legs_measured_mean_is_per_width_and_context_bucket(self):
        legs = [rec(pos=1000, rows=1, ms=10.0), rec(pos=1000, rows=2, ms=20.0),
                rec(pos=30000, rows=3, ms=90.0), rec(pos=30000, rows=3, ms=50.0),
                rec(pos=8000, rows=1, ms=0.0)]   # unknown ms: no sample
        t = mw.measured_costs(legs)
        self.assertEqual(t[0][:2], [10.0, 20.0])
        self.assertEqual(t[1][1], 0.0)            # <24K has no sample
        self.assertAlmostEqual(t[2][2], 70.0)     # 90 and 50 mean
        self.assertEqual(t[2][1], 0.0)

    def test_an_explicit_t1_seed_wins_over_the_legs(self):
        # seed_costs only fills an empty width: the MTP-off leg's T1 (99 ms) must
        # not replace the explicit 20 ms; red: seed_costs overwrites what it finds
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "arm.txt"
            write_trace(path, [rec(rows=1, acc=0, nmax=0, ms=10.0, probs=())])
            rep = mw.replay(mw.read_trace(path), seed_costs=[[20.0]],
                            costs_from=[[99.0]])
        self.assertAlmostEqual(rep["policy"].cost_n[0][1], 7.0)
        self.assertAlmostEqual(rep["policy"].cost_ms(1), (6 * 20.0 + 10.0) / 7.0, places=2)

    def test_the_cli_wires_the_legs_into_the_arm(self):
        # red: --costs-from parsed but not passed, the arm's cost table keeps
        # only its own round (or the kShape guess) and T2 stays unmeasured
        import io
        import json
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as d:
            arm, legs = Path(d) / "arm.txt", Path(d) / "legs.txt"
            write_trace(arm, [rec(rows=2, acc=1, nmax=1, ms=0.0, probs=(0.9, 0.9))])
            write_trace(legs, [rec(rows=2, acc=1, nmax=1, ms=44.0, probs=(0.9, 0.9))])
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mw.main([str(arm), "--costs-from", str(legs), "--json"])
        self.assertEqual(rc, 0)
        out = json.loads(buf.getvalue())
        self.assertIn("T2:44.00(n=6)", out["policy"]["cost_table"])

    def test_requests_are_keyed_by_file_not_by_the_counter_alone(self):
        # the counter restarts per process; two files with req 1 are two requests
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.txt", Path(d) / "b.txt"
            write_trace(a, [rec(req=1), rec(req=1, pos=0)])
            write_trace(b, [rec(req=1)])
            recs = mw.read_trace(a) + mw.read_trace(b)
            rep = mw.replay(recs)
        self.assertEqual(rep["rounds"], 3)
        self.assertEqual(rep["requests"], 2)


class UnknownTimes(unittest.TestCase):
    def test_a_round_without_a_time_learns_no_cost(self):
        # red: without the ms <= 0 guard, observe(0.0) still counts a round and
        # a committed-token outcome with no cost; with it, observe() is skipped
        # ENTIRELY (no cost and no token EMA) and the report says the round is
        # unpriced
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.txt"
            write_trace(path, [rec(ms=0.0), rec(ms=0.0, pos=0)])
            recs = mw.read_trace(path)
            rep = mw.replay(recs)
        self.assertEqual(rep["unknown_ms"], 2)
        self.assertTrue(rep["policy"].cost_table_text() in ("", None) or "T" not in rep["policy"].cost_table_text())
        out = []
        mw.report(str(path), rep, out)
        self.assertIn("ms <= 0", "\n".join(out))


class SeedCosts(unittest.TestCase):
    def test_the_table_form_is_stratas(self):
        # generate.cpp:16495-16511: "T1,T2,..;T1,..;T1,..", the 3 context
        # buckets; weight 6 rounds a width (draft_policy.cpp:66-74)
        self.assertEqual(mw.parse_cost_table("10,20;30;40,50,60"), [[10.0, 20.0], [30.0], [40.0, 50.0, 60.0]])
        with self.assertRaises(mw.TraceError):
            mw.parse_cost_table("10,x;30")

    def test_a_seeded_table_prices_a_width_the_trace_never_ran(self):
        # red: without --seed-costs the width is Strata's kShape guess and the
        # cost table stays empty; with it, T2 carries 6 rounds of weight
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.txt"
            write_trace(path, [rec(rows=2, acc=1, nmax=1, probs=(0.9, 0.9), ms=30.0)])
            recs = mw.read_trace(path)
            plain = mw.replay(recs)
            seeded = mw.replay(recs, seed_costs=mw.parse_cost_table("10,20;30;40"))
        # plain: the round's own 30 ms is the only T2 entry
        self.assertIn("T2:30.00(n=1)", plain["policy"].cost_table_text())
        # seeded: T2 starts at the seeded 20 ms with 6 rounds of weight, then
        # the round's observation moves it by alpha = 1/(6+1) -- 21.43
        self.assertAlmostEqual(seeded["policy"].cost_n[0][2], 7.0)
        self.assertAlmostEqual(seeded["policy"].cost_ms(2), (6 * 20.0 + 30.0) / 7.0)
        self.assertNotIn("cost_n", "x")   # documentation, not a check


class CostTableText(unittest.TestCase):
    def test_a_bucket_is_named_once_like_stratas_log_line(self):
        # draft_policy.cpp:105-111: "ctx<6K:" then " T1:.. T2:.."; red: the name
        # is repeated before every width
        p = mw.DraftPolicy()
        p.seed_costs([10.0, 20.0], 6.0, 0)
        text = p.cost_table_text()
        self.assertTrue(text.startswith("ctx<6K:"))
        self.assertEqual(text.count("ctx<6K"), 1)
        self.assertIn("T1:10.00(n=6)", text)
        self.assertIn("T2:20.00(n=6)", text)


class FixedWidthNumbers(unittest.TestCase):
    def test_the_policy_rate_is_its_own_e_over_cost_per_round(self):
        # red: exp_v = mean E / the MODE width's cost mixes two widths' numbers.
        # Here the mode width (8 rows) is 80 ms and the 2-row rounds are 1 ms,
        # so the mean of the policy's own best_v is far above E / 80.
        rounds = [rec(rows=2, acc=1, nmax=1, ms=1.0, probs=(0.99, 0.99)) for _ in range(4)]
        rounds += [rec(rows=8, acc=7, nmax=7, ms=80.0, probs=tuple([0.99] * 7)) for _ in range(6)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.txt"
            write_trace(path, rounds)
            rep = mw.replay(mw.read_trace(path))
        self.assertGreaterEqual(len(rep["hist"]), 2)   # the rounds chose different widths
        self.assertGreater(rep["exp_v"], 0.5)
        self.assertAlmostEqual(rep["exp_v"], sum(rep["exp_v_rounds"]) / len(rep["exp_v_rounds"]))
        # under the mutation (E / the mode width's cost) this equality breaks
        self.assertNotAlmostEqual(rep["exp_v"], rep["exp_tok"] / rep["mode_cost"], places=2)
    def test_constant_cost_run_reproduces_the_fixed_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.txt"
            write_trace(path, [rec() for _ in range(6)])
            recs = mw.read_trace(path)
            rep = mw.replay(recs, constant_cost=50.0)
        n, tok, ms, msn = rep["widths"][5]
        self.assertEqual((n, msn), (6, 6))
        self.assertAlmostEqual(tok / n, 4.0)          # acc 3 + the target's own token
        self.assertAlmostEqual(ms / msn, 50.0)
        self.assertAlmostEqual((tok / n) / (ms / msn), 4.0 / 50.0)
        # one constant cost for every width: the chain runs to the cap (nmax + 1 rows)
        self.assertEqual(list(rep["hist"]), [5])
        self.assertEqual(rep["hist"][5], 6)
        self.assertAlmostEqual(rep["trunc_tok"], 4.0)  # min(acc, 4) + 1

    def test_the_truncated_round_keeps_only_the_chosen_prefix(self):
        # a policy that settles on 1 row while the trace walked 4 accepted
        # drafts: the truncated round emits 1 token, not 5. Red: kept = acc.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.txt"
            write_trace(path, [rec(rows=5, acc=4, nmax=4, probs=(0.02, 0.02, 0.02, 0.02))])
            rep = mw.replay(mw.read_trace(path))
        self.assertEqual(max(rep["hist"]), 1)
        self.assertAlmostEqual(rep["trunc_tok"], 1.0)
        self.assertAlmostEqual(rep["trunc_v"], 1.0 / rep["trunc_ms"])

    def test_the_cap_is_the_ceiling_plus_one_row(self):
        # red: cap = nmax (drafts, not rows) makes the mode width 4, not 5
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.txt"
            write_trace(path, [rec(nmax=4, rows=5) for _ in range(3)])
            rep = mw.replay(mw.read_trace(path), constant_cost=40.0)
        self.assertEqual(max(rep["hist"]), 5)


class KeepDraftPort(unittest.TestCase):
    """Strata's own test vectors that apply (see the module header): none of its
    keep_draft vectors exist; its observe() shape and the derived rule vectors
    are asserted here."""

    def test_strata_test_cost_shape_and_committed_tokens(self):
        # Strata-ref:src/spec/draft_policy_test.cpp's cost(t) = 19 + 10.5*(t-1)
        # with p.observe(false, 4, 2, 0, cost(4)) 50 times
        def cost(t):
            return 19.0 + 10.5 * (t - 1)
        p = mw.DraftPolicy()
        for _ in range(50):
            p.observe(4, 2, cost(4))
        self.assertAlmostEqual(p.cost_ms(4), cost(4), places=6)
        self.assertAlmostEqual(p.mtp_tokens(4), 3.0, places=6)   # accepted 2 + 1
        # the file's own for-loop (draft_policy_test.cpp block 2) settles t = 2..6 at 1
        for t in range(2, 7):
            for _ in range(20):
                p.observe(t, 0, cost(t))
        self.assertAlmostEqual(p.mtp_tokens(5), 1.0, places=6)
        # before any window of that size: the typical acceptance (draft_policy.cpp:118-121);
        # 3 was measured above, 7 never was
        self.assertAlmostEqual(p.mtp_tokens(7), 1.0 + 0.7 * 6, places=6)

    def test_the_cost_ema_is_a_running_mean_for_the_first_samples(self):
        # observe()'s alpha = max(kCostAlpha, 1/(n+1)) (draft_policy.cpp:158) with
        # robust on; clamp_sample takes the guess only at a width's first use.
        p = mw.DraftPolicy()
        p.observe(4, 3, 41.875)          # first use: no other width measured -> the sample stands
        self.assertAlmostEqual(p.cost_ms(4), 41.875, places=6)
        p.observe(4, 2, 40.0)            # alpha 0.5
        self.assertAlmostEqual(p.cost_ms(4), (41.875 + 40.0) / 2.0, places=6)
        p.observe(4, 1, 20.0)            # alpha 1/3
        want = (1 - 1.0 / 3.0) * ((41.875 + 40.0) / 2.0) + (1.0 / 3.0) * 20.0
        self.assertAlmostEqual(p.cost_ms(4), want, places=6)
        # a stalled sample caps at twice the known value (draft_policy.cpp:55-65):
        # the EMA moves towards 2x the known, never towards 10000
        known = want
        p.observe(4, 1, 10000.0)
        alpha = max(mw.KCOSTALPHA, 1.0 / (3.0 + 1.0))
        self.assertAlmostEqual(p.cost_ms(4), (1.0 - alpha) * known + alpha * min(10000.0, 2.0 * known), places=6)
        self.assertLess(p.cost_ms(4), 100.0)

    def test_the_margin_is_three_percent_over_three(self):
        # With cost(1) == cost(2) == 100 ms, the window of 2 rows is worth
        # (1 + a(p))/100 against 1/100. p = 0.02 gives a = 0.02 (the prior is p
        # itself, draft_policy.cpp:182-188), so v = 0.0102 and best_v*1.01 =
        # 0.010302: the 3 %/3 margin keeps the width; a 3 % margin would not.
        p = mw.DraftPolicy()
        p.seed_costs([100.0, 100.0, 100.0], 6.0, 0)   # weight 6, generate.cpp:18176
        st = mw.ChainState()
        self.assertFalse(p.keep_draft(st, 0, 0.02, 3))
        self.assertEqual(st.best_rows, 2)
        self.assertEqual(st.kept, 1)

    def test_a_constant_cost_chain_runs_to_the_cap(self):
        p = mw.DraftPolicy()
        p.seed_costs([40.0] * mw.KMAXT, 6.0, 0)
        st = mw.ChainState()
        go = True
        for idx in range(mw.KMAXT):
            go = p.keep_draft(st, idx, 0.9, 8)
            if not go:
                break
        self.assertFalse(go)
        self.assertEqual(st.best_rows, 8)
        self.assertAlmostEqual(st.kept, 7)

    def test_observe_probs_reached_by_the_prefix(self):
        # draft j is reached when j <= accepted and held when j < accepted
        # (draft_policy.cpp:190-196). Three 0.8 drafts, one accepted: j = 0 held,
        # j = 1 reached and not held, j = 2 never reached. Bucket 6 is
        # int(0.8*8) and its prior is p = 0.8, worth kPriorN = 4.
        p = mw.DraftPolicy()
        p.observe_probs([0.8, 0.8, 0.8], 3, 1)
        # the counts decay as they are added (KCALDECAY 0.995, draft_policy.cpp:16):
        # j = 0 sets ok 1, n 1; j = 1 (reached, not held) makes them 0.995 and 1.995
        self.assertAlmostEqual(p.draft_rate(0.8), (0.995 + 4 * 0.8) / (1.995 + 4), places=6)
        # red: counting the unreached j = 2 as reached-and-not-held gives
        # (0.990025 + 3.2) / (2.985 + 4), about 0.6 instead of about 0.7
        self.assertNotAlmostEqual(p.draft_rate(0.8), (0.995 ** 2 + 4 * 0.8) / (0.995 * 1.995 + 1 + 4),
                                  places=2)


class Limits(unittest.TestCase):
    def test_a_floor_stopped_trace_warns(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.txt"
            # the chain ran 4 rows at minp 0.5; probabilities include the stopped draft
            # the floor stopped after 3 kept drafts at a 0.6 probability the policy
            # would verify; with min-p 0 it wants 5 rows, which the ceiling made
            write_trace(path, [rec(rows=4, acc=2, nmax=5, minp=0.7, probs=(0.99, 0.99, 0.99, 0.6))])
            recs = mw.read_trace(path)
            rep = mw.replay(recs)
        self.assertTrue(any(m > 0 for m in rep["minp_seen"]))
        self.assertGreater(rep["wider_than_floor"] + rep["never_drafted"], 0)
        out = []
        mw.report(str(path), rep, out)
        text = "\n".join(out)
        self.assertIn("min-p > 0", text)
        self.assertIn("ceiling+1", text)

    def test_a_minp0_trace_does_not_warn_about_the_floor(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.txt"
            write_trace(path, [rec(minp=0.0)])
            rep = mw.replay(mw.read_trace(path))
        out = []
        mw.report(str(path), rep, out)
        self.assertNotIn("min-p > 0", "\n".join(out))


class DecisionPin(unittest.TestCase):
    # the cross-language pin: this file and src/exec/draft_policy.h's test both
    # compare their bytes to the checked-in decisions fixture, so the C++ class
    # and this replay have ONE behaviour round by round.
    FIXTURE_DEC = Path(__file__).resolve().parent / "fixtures" / "mtp_policy_decisions.txt"

    def test_the_replay_emits_the_checked_in_decisions(self):
        # red: `st.best_e = st.e` unconditionally at the end of keep_draft (E at
        # the chain's end, not at the width it would verify) moves the
        # truncating round's e= off 1.966277143 and the comparison fails.
        rep = mw.replay(mw.read_trace(FIXTURE) + mw.read_trace(POLICY_FIXTURE))
        want = self.FIXTURE_DEC.read_text(encoding="utf-8")
        self.assertEqual(mw.decision_text(rep), want)

    def test_the_decisions_fixture_names_its_own_table(self):
        # red: a table line dropped or the bucket name changed once a bucket -
        # the fifth line disappears (the C++ side fails the same way)
        text = self.FIXTURE_DEC.read_text(encoding="utf-8")
        lines = [l for l in text.splitlines() if l.strip()]
        self.assertEqual(len(lines), 6)
        self.assertTrue(lines[-1].startswith("table ctx<6K:"))
        self.assertIn(" | ctx<24K:", lines[-1])


class T1Unmeasured(unittest.TestCase):
    def test_an_mtp_only_trace_says_t1_is_never_measured(self):
        # red: the note dropped, the report presents a cost table with no T1 and
        # the card procedure cannot know width 1 needs the MTP-off leg
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.txt"
            write_trace(path, [rec(rows=2, acc=1, nmax=1, ms=30.0, probs=(0.9, 0.9))])
            rep = mw.replay(mw.read_trace(path))
        out = []
        mw.report(str(path), rep, out)
        text = "\n".join(out)
        self.assertIn("T1 is never measured", text)
        self.assertIn("--no-mtp", text)


if __name__ == "__main__":
    unittest.main()
