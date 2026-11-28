#!/usr/bin/env python3
"""tools/dspark_probe_read.py (patch 0243's probe log), device-free.

The fixture tests/python/fixtures/dspark_probe_short.jsonl, by hand (0-based positions, 3 drafts a line):
  lane 0  pos0 100  kept 2, no divergence; the next line at 103 = 100 + 2 + 1: paired      full 2
          pos0 103  kept 1, f differs at 1 (outside) = the refused row; the draw there is the next
                    id_last 99 = f[1]: the full chain is kept, then the run-without's rate  full 1 + 1 + p3
          pos0 105  kept 3, f differs at 2 (outside) < 3: the shortlist's kept draft, the full chain refused there  full 2
          pos0 109  kept 0, f differs at 0 (outside) = the refused row, the draw 5 != 88     full 0
          pos0 110  kept 1, f differs at 2 (outside) > 1: unreached; the last line          full 1
  lane 1  pos0 200  kept 0; the next line at 205 != 201: not its next cycle (unpaired)      full 0
          pos0 205  kept 1, f differs at 1 (inside: rounding) = the refused row, no next line: the draw
                    unknown, out of the prediction
  and one line that is not JSON.
The run without (dspark_probe_off.jsonl): kept 3, 1, 2, 0 of 3: p1 = 3/4, p2 = 2/3, p3 = 1/2.

dspark_probe_inside.jsonl, the inside divergences (f among the K candidates, not the shortlist's pick):
  pos0 300  kept 1, f differs at 1 (inside) = the refused row, the draw 30 = f[1]: kept   full 1 + 1 + p3
  pos0 302  kept 2, f differs at 0 (inside) < 2: the full chain refused at 0                full 0
  pos0 305  kept 0, no divergence, the last line                                            full 0
  inside_loss = (1 + p3 - 2) / 9 drafts offered; the prediction (outside only) 0.

The reading's reds (synthetic logs, a lane of 3-draft lines, a = 0 each): inside_loss 0.6 points (9 inside hits in
1,500 drafts offered) is a defect, 0.47 is not; the tripwire 1 is a defect; |prediction - measured| outside
max(2 pt, 2 SE_d) is a defect, inside it is not. DSPARK_PROBE_READ_RED: 1 (a one too many), 2 (the inside
divergences folded into the prediction), 3 (the tripwire not counted): each fails a case here.
"""
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "..", "tools", "dspark_probe_read.py")
spec = importlib.util.spec_from_file_location("dspark_probe_read", TOOL)
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

SHORT = os.path.join(HERE, "fixtures", "dspark_probe_short.jsonl")
INSIDE = os.path.join(HERE, "fixtures", "dspark_probe_inside.jsonl")
OFF = os.path.join(HERE, "fixtures", "dspark_probe_off.jsonl")
RED = int(os.environ.get("DSPARK_PROBE_READ_RED") or 0)


def run(*args):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "r.json")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = pr.main(list(args) + ["--json", out])
        with open(out) as fh:
            return rc, json.load(fh), buf.getvalue()


def run_rc(*args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = pr.main(list(args))
    return rc, None, buf.getvalue()


def synth2(path, n_lines, jumps=0, alone=0, sl=256):
    """a lane of 3-draft lines, a = 0 each, no divergence; after each of the first `jumps` lines the next pos0 lies 2
    further (the driver's count 0 disagrees: unpaired); then `alone` lines each on a lane of its own, diverging
    outside at row 0 (the refused row, its draw unknown: no next line)"""
    with open(path, "w") as fh:
        pos = 10
        for k in range(n_lines):
            fh.write(json.dumps({"seq": 0, "pos0": pos, "id_last": 5, "a": 0, "sl": sl, "d": [1, 2, 3],
                                 "f": [1, 2, 3], "in": [1, 1, 1] if sl else []}) + "\n")
            pos += 3 if k < jumps else 1
        for k in range(alone):
            fh.write(json.dumps({"seq": 1 + k, "pos0": 10, "id_last": 5, "a": 0, "sl": sl, "d": [1, 2, 3],
                                 "f": [99, 2, 3] if sl else [1, 2, 3], "in": [0, 1, 1] if sl else []}) + "\n")


def synth(path, n_lines, inside_hits=0, trip=0):
    """a lane of 3-draft lines, a = 0 each; the first inside_hits lines diverge inside at row 0 and the target draws
    the full pick there; the first trip lines also carry an impossible in = 0 with f = d at row 2"""
    with open(path, "w") as fh:
        for k in range(n_lines):
            d = [1, 2, 3]
            f = [99, 2, 3] if k < inside_hits else list(d)
            ins = [1, 1, 0] if k < trip else [1, 1, 1]
            fh.write(json.dumps({"seq": 0, "pos0": 10 + k, "id_last": 5 if k == 0 or k > inside_hits else 99, "a": 0,
                                 "sl": 256, "d": d, "f": f, "in": ins}) + "\n")


class Reading(unittest.TestCase):
    def reading(self, n_lines, delta="0", se="0.3", **kw):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.jsonl")
            synth(p, n_lines + 1, **kw)   # + 1: the last line has no next (no draw), out of nothing
            return run(p, "--measured-delta", delta, "--measured-se", se)

    def test_inside_loss_0_6_points_is_a_defect(self):
        _, r, out = self.reading(500, inside_hits=9)
        self.assertAlmostEqual(r["inside_loss"], 100 * 9 / 1503)
        self.assertGreater(r["inside_loss"], 0.5)
        self.assertFalse(r["green"])
        self.assertIn("gather defect", out)

    def test_inside_loss_under_half_a_point_passes(self):
        _, r, _ = self.reading(500, inside_hits=7)
        self.assertLess(r["inside_loss"], 0.5)
        self.assertTrue(r["green"])

    def test_the_tripwire_is_a_defect(self):
        _, r, out = self.reading(100, trip=1)
        self.assertEqual(r["tripwire"], 1)
        self.assertFalse(r["green"])
        self.assertIn("tripwire", out)

    def test_a_measured_loss_outside_the_band_is_a_defect(self):
        _, r, _ = self.reading(100, delta="-2.5", se="0.5")
        self.assertAlmostEqual(r["band"], 2.0)
        self.assertFalse(r["green"])

    def test_the_band_widens_with_the_se(self):
        _, r, _ = self.reading(100, delta="-2.5", se="1.5")
        self.assertAlmostEqual(r["band"], 3.0)
        self.assertTrue(r["green"])


class Validity(unittest.TestCase):
    """INVALID: above 5 % of an arm's lines unpaired (either log), or of the shortlist's lines with a refused-row draw
    unknown: no verdict; NOT READ without a measured value. DSPARK_PROBE_READ_RED=4 drops the limits."""
    def read(self, short, off=None, measured=True):
        with tempfile.TemporaryDirectory() as d:
            ps, po = os.path.join(d, "s.jsonl"), os.path.join(d, "o.jsonl")
            synth2(ps, **short)
            args = [ps]
            if off is not None:
                synth2(po, sl=0, **off)
                args += ["--off", po]
            if measured:
                args += ["--measured-delta", "0", "--measured-se", "0.3"]
            return run(*args)

    def test_a_clean_run_reads_green_and_prints_the_rule_verbatim(self):
        _, r, out = self.read({"n_lines": 200}, {"n_lines": 200})
        self.assertEqual(r["reading"], "GREEN")
        self.assertIn("PROBE READING: GREEN", out)
        self.assertIn("Measured \u0394accept = the mean paired difference of sp0's 10 timed pairs", out)
        self.assertIn("GREEN iff all three hold: |prediction \u2212 measured| \u2264 max(2 pt, 2\u00b7SE_d)", out)
        self.assertIn("TABLEACC its per-continuation accept difference", out)

    def test_unpaired_shortlist_lines_above_5_percent_are_invalid(self):
        _, r, out = self.read({"n_lines": 100, "jumps": 6}, {"n_lines": 100})
        self.assertEqual(r["unpaired"], 6)
        self.assertEqual(r["reading"], "INVALID")
        self.assertIn("PROBE READING: INVALID", out)
        self.assertNotIn("DEFECT", out.split("PROBE READING: ")[-1])

    def test_unpaired_lines_at_5_percent_are_valid(self):
        _, r, _ = self.read({"n_lines": 100, "jumps": 5}, {"n_lines": 100})
        self.assertEqual(r["reading"], "GREEN")

    def test_unpaired_lines_of_the_run_without_above_5_percent_are_invalid(self):
        _, r, out = self.read({"n_lines": 100}, {"n_lines": 100, "jumps": 6})
        self.assertEqual((r["unpaired"], r["unpaired_off"]), (0, 6))
        self.assertEqual(r["reading"], "INVALID")
        self.assertIn("run-without's unpaired lines 6", out)

    def test_unknown_draws_above_5_percent_are_invalid(self):
        _, r, out = self.read({"n_lines": 94, "alone": 6}, {"n_lines": 100})
        self.assertEqual((r["unpaired"], r["unknown"]), (0, 6))
        self.assertEqual(r["reading"], "INVALID")
        self.assertIn("unknown draw 6", out)

    def test_no_measured_value_is_not_read(self):
        _, r, out = self.read({"n_lines": 100}, {"n_lines": 100}, measured=False)
        self.assertEqual(r["reading"], "NOT READ")
        self.assertIn("PROBE READING: NOT READ", out)


class Counts(unittest.TestCase):
    def test_the_fixture_as_counted_by_hand(self):
        rc, r, _ = run(SHORT)
        self.assertEqual(rc, 0)
        self.assertEqual(r["bad_lines"], 1)
        self.assertEqual(r["unpaired"], 1)
        self.assertEqual(r["lines"], 7)
        self.assertEqual(r["offered"], 21)
        self.assertEqual(r["kept"], 8)
        self.assertEqual(r["outside"], 4)
        self.assertEqual(r["outside_reached"], 3)
        self.assertEqual(r["inside_differs"], 1)
        self.assertEqual(r["tripwire"], 0)
        self.assertEqual((r["kept_by_short_out"], r["refused_out"], r["refused_unknown_out"], r["hits_out"]), (1, 2, 0, 1))
        self.assertEqual((r["kept_by_short_in"], r["refused_in"], r["refused_unknown_in"], r["hits_in"]), (0, 1, 1, 0))
        self.assertEqual(r["div_unreached"], 1)
        self.assertAlmostEqual(r["share_outside"], 100 * 4 / 21)
        # first order: the outside first divergences reached (pos0 103, 105, 109) of the 18 drafts used, their net
        # rate (1 kept by the full chain - 1 kept by the shortlist) / 3
        self.assertEqual(r["first_out"], 3)
        self.assertAlmostEqual(r["first_out_share"], 100 * 3 / 18)
        self.assertAlmostEqual(r["first_order"], 0.0)
        self.assertAlmostEqual(r["inside_loss"], 0.0)

    def test_without_the_run_without_the_prediction_is_a_lower_bound(self):
        _, r, out = run(SHORT)
        self.assertTrue(r["bound"])
        self.assertAlmostEqual(r["delta_out"], 0.0)   # +1 (pos0 103) - 1 (pos0 105)
        self.assertAlmostEqual(r["predicted"], 0.0)
        self.assertIn("a lower bound", out)

    def test_with_the_run_without(self):
        _, r, _ = run(SHORT, "--off", OFF)
        self.assertFalse(r["bound"])
        self.assertEqual(r["off_lines"], 4)
        self.assertAlmostEqual(r["off_rate"], 50.0)
        self.assertAlmostEqual(r["delta_out"], 0.5)          # the kept outside row goes on at p3 = 1/2
        self.assertAlmostEqual(r["predicted"], 100 * 0.5 / 18)
        self.assertAlmostEqual(r["own_delta"], 100 * 8 / 21 - 50.0)   # data, not gated
        # the fixture's 1 unpaired line and 1 unknown draw of 7 lines (14.3 %) put it past the validity limits
        self.assertEqual(r["reading"], "INVALID")
        _, r, out = run(SHORT, "--off", OFF, "--measured-delta", "-1.0", "--measured-se", "0.3",
                        "--earlier-gate", "54.3 -> 49.0 %")
        self.assertEqual(r["reading"], "INVALID")
        self.assertFalse(r["green"])
        self.assertIn("PROBE READING: INVALID (no verdict", out)
        self.assertIn("the earlier gate (data only): 54.3 -> 49.0 %", out)

    def test_the_inside_divergences_are_inside_loss_not_the_prediction(self):
        _, r, out = run(INSIDE, "--off", OFF, "--measured-delta", "0", "--measured-se", "0.3")
        self.assertEqual((r["kept_by_short_in"], r["refused_in"], r["hits_in"]), (1, 1, 1))
        self.assertAlmostEqual(r["predicted"], 0.0)
        self.assertAlmostEqual(r["inside_loss"], 100 * (1 + 0.5 - 2) / 9)
        self.assertFalse(r["green"])
        self.assertIn("gather defect", out)

    def test_off_refuses_a_shortlist_log(self):
        rc, _, out = run_rc(SHORT, "--off", SHORT)
        self.assertEqual(rc, 3)
        self.assertIn("PROBE REFUSED", out)

    def test_a_log_without_the_drivers_count_derives_it_from_the_positions(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "noa.jsonl")
            with open(SHORT) as src, open(p, "w") as dst:
                for line in src:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if rec["seq"] == 0:
                        rec.pop("a")
                        dst.write(json.dumps(rec) + "\n")
            # the last line has no next and no count: dropped (unpaired); the rest as in the fixture
            _, r, _ = run(p)
            self.assertEqual(r["unpaired"], 1)
            self.assertEqual(r["lines"], 4)
            self.assertEqual(r["kept"], 2 + 1 + 3 + 0)

    def test_an_empty_log_exits_2(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "e.jsonl")
            open(p, "w").close()
            rc, r, _ = run(p)
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
