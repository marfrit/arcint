#!/usr/bin/env python3
"""Unit ladder for tools/expert_convergence.py (stdlib only, device-free).

Red-first: every cell below fails against a plausible wrong implementation
before it passes against the one in the tool.

  * `decile_hit` buckets ACCESS position (not call position) into equal-access
    deciles, and `convergence_decile` is the first decile after which the
    series stays within `eps` of the final value -- a wrong bucket boundary or
    a first-bucket "convergence" is caught.
  * `fill_curve` counts only PINNED experts demanded (an unpinned id must not
    move the fill) and `calls_to_last_new` is the LAST first-demand -- the
    device-byte plateau point under the static partition.
  * `probe_plateau` splits calls into forwards of `calls_per_forward` and a
    plateau requires TWO consecutive zero deltas (the engine's `plateaued < 2`
    test), so a single quiet forward cannot declare convergence.
  * `lru_thrash` is a TRUE LRU (promotion on hit) -- a FIFO without promotion
    is distinguishable on the crafted sequence, and it understates the
    comparand.
  * `rolling_prefix_series` reports a plateau only when the recomputed set is
    unchanged across prefixes; a set that changes at every prefix reports None.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expert_convergence as ec  # noqa: E402


def write_call_trace(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# source=plugin-0044-call-trace\n")
        for seq, lk, top_k, ids in rows:
            f.write(" ".join([str(seq), str(lk), str(top_k)] +
                             [str(x) for x in ids]) + "\n")


class TestDecileHit(unittest.TestCase):
    def test_exact_two_bucket_split(self):
        rows = [(0, [1, 1, 1, 1, 1, 2, 2, 2, 2, 2])]
        sets = {0: {1}}
        d = ec.decile_hit(rows, sets, n=2)
        self.assertEqual(d["series"][0]["hit_pct"], 100.0)
        self.assertEqual(d["series"][1]["hit_pct"], 0.0)
        self.assertEqual(d["early"], 1.0)
        self.assertEqual(d["tail"], 0.0)
        self.assertEqual(d["tail_minus_steady_pt"], -50.0)

    def test_convergence_decile(self):
        # hit series (n=4): 0.00, 1.00, 1.00, 1.00 -> last three within 1pt of
        # the final value, so convergence begins at decile 2.
        rows = [(0, [2] * 5 + [1] * 15)]
        sets = {0: {1}}
        d = ec.decile_hit(rows, sets, n=4)
        self.assertEqual([round(s["hit_pct"], 1) for s in d["series"]],
                         [0.0, 100.0, 100.0, 100.0])
        self.assertEqual(d["convergence_decile"], 2)

    def test_empty_is_zero(self):
        d = ec.decile_hit([], {0: {1}})
        self.assertEqual(d["accesses"], 0)
        self.assertIsNone(d["convergence_decile"])

    def test_n_below_two_refused(self):
        with self.assertRaises(ValueError):
            ec.decile_hit([(0, [1])], {0: {1}}, n=1)

    def test_fewer_accesses_than_deciles(self):
        # 3 accesses into 10 deciles: the ordinal formula places them in
        # buckets 0, 3, 6 -- it must NOT advance several buckets per access.
        d = ec.decile_hit([(0, [1, 2, 3])], {0: {1}}, n=10)
        self.assertEqual([s["accesses"] for s in d["series"]],
                         [1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        self.assertEqual([round(s["hit_pct"], 1) for s in d["series"]],
                         [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class TestFillCurve(unittest.TestCase):
    def test_only_pinned_ids_count(self):
        # expert 4 is routed but NOT pinned; it must not move the fill.
        calls = [(0, 0, 1, [1]), (1, 0, 1, [4]), (2, 0, 1, [2]), (3, 0, 1, [3])]
        fill = ec.fill_curve(calls, {0: {1, 2, 3}})
        self.assertEqual(fill["total"], 3)
        self.assertEqual(fill["fill_at_end"], 3)
        self.assertEqual(fill["fill_fraction_end"], 1.0)
        self.assertEqual(fill["calls_to_last_new"], 3)
        self.assertEqual(fill["per_call"], [[0, 1], [1, 1], [2, 2], [3, 3]])
        self.assertTrue(fill["monotone"])

    def test_never_routed_pinned_expert_leaves_fill_below_one(self):
        fill = ec.fill_curve([(0, 0, 1, [1])], {0: {1, 2}})
        self.assertEqual(fill["total"], 2)
        self.assertEqual(fill["fill_at_end"], 1)
        self.assertEqual(fill["fill_fraction_end"], 0.5)
        self.assertEqual(fill["calls_to_last_new"], 0)


class TestProbePlateau(unittest.TestCase):
    def test_two_zero_deltas_plateau(self):
        # forward 1 demands every pinned expert; forwards 2 and 3 demand none.
        rows = []
        seq = 0
        for lk in range(3):
            for ids in ([1, 2, 3], [], []):
                rows.append((seq, lk, len(ids) if ids else 1,
                             ids if ids else [99]))
                seq += 1
        # rebuild as three forwards of 3 calls: one call per layer per forward
        rows = [(0, 0, 3, [1, 2, 3]), (1, 1, 3, [1, 2, 3]), (2, 2, 3, [1, 2, 3]),
                (3, 0, 1, [99]), (4, 1, 1, [99]), (5, 2, 1, [99]),
                (6, 0, 1, [99]), (7, 1, 1, [99]), (8, 2, 1, [99])]
        sets = {0: {1, 2, 3}, 1: {1, 2, 3}, 2: {1, 2, 3}}
        p = ec.probe_plateau(rows, sets, calls_per_forward=3)
        self.assertEqual(p["deltas"], [9, 0, 0])
        self.assertTrue(p["plateau"])
        self.assertEqual(p["last_new_forward"], 1)
        self.assertEqual(p["filled"], 9)

    def test_single_quiet_forward_is_not_a_plateau(self):
        rows = [(0, 0, 1, [1]), (1, 0, 1, [99]), (2, 0, 1, [2])]
        sets = {0: {1, 2}}
        p = ec.probe_plateau(rows, sets, calls_per_forward=1)
        self.assertEqual(p["deltas"], [1, 0, 1])
        self.assertFalse(p["plateau"])
        self.assertEqual(p["last_new_forward"], 3)


class TestLruThrash(unittest.TestCase):
    def test_true_lru_promotes_on_hit(self):
        # TRUE LRU at 2 slots: hits {1} then {1}; FIFO (no promotion) would hit
        # only {1} and evict twice.
        rows = [(0, [1]), (0, [2]), (0, [1]), (0, [3]), (0, [1])]
        l = ec.lru_thrash(rows, 2)
        self.assertEqual(l["hits"], 2)
        self.assertEqual(l["evictions"], 1)
        self.assertEqual(l["accesses"], 5)

    def test_zero_slots_never_inserts(self):
        l = ec.lru_thrash([(0, [1, 2])], 0)
        self.assertEqual(l["hits"], 0)
        self.assertEqual(l["evictions"], 0)

    def test_negative_slots_refused(self):
        with self.assertRaises(ValueError):
            ec.lru_thrash([(0, [1])], -1)

    def test_rate_units(self):
        # Same layer key, or the per-layer cache would never fill.
        rows = [(0, [1]), (0, [2]), (0, [1])]
        l = ec.lru_thrash(rows, 1)
        self.assertEqual(l["calls"], 3)
        self.assertEqual(l["accesses"], 3)
        self.assertEqual(l["evictions"], 2)
        self.assertEqual(l["evictions_per_1000_calls"], 2000.0 / 3.0)
        self.assertEqual(l["evictions_per_1000_accesses"], 2000.0 / 3.0)


class TestRollingPrefixSeries(unittest.TestCase):
    def test_stable_set_plateaus(self):
        calls = [(0, 0, 3, [1, 2, 3]), (1, 0, 3, [1, 2, 3]),
                 (2, 0, 3, [1, 2, 3]), (3, 0, 3, [1, 2, 3])]
        r = ec.rolling_prefix_series(calls, 2)
        self.assertTrue(r["plateau"])
        self.assertEqual(r["rounds_to_plateau"], 1)
        self.assertEqual([s["layers_changed"] for s in r["series"]],
                         [None, 0, 0])

    def test_changing_set_never_plateaus(self):
        calls = [(0, 0, 1, [4]), (1, 0, 1, [3]), (2, 0, 1, [2]), (3, 0, 1, [1])]
        r = ec.rolling_prefix_series(calls, 1)
        self.assertFalse(r["plateau"])
        self.assertIsNone(r["rounds_to_plateau"])
        self.assertEqual([s["layers_changed"] for s in r["series"]],
                         [None, 1, 1])

    def test_zero_slots_refused(self):
        with self.assertRaises(ValueError):
            ec.rolling_prefix_series([(0, 0, 1, [1])], 0)


class TestEndToEnd(unittest.TestCase):
    def test_measure_on_small_trace(self):
        rows = []
        seq = 0
        # probe regime: 2 forwards x 2 layers, ids 1,2
        for _fwd in range(2):
            for lk in (10, 20):
                rows.append((seq, lk, 2, [1, 2]))
                seq += 1
        # prefill: one batched call per layer, ids 1,2,3
        for lk in (10, 20):
            rows.append((seq, lk, 3, [1, 2, 3]))
            seq += 1
        # decode: three single-token calls per layer
        for _tok in range(3):
            for lk in (10, 20):
                rows.append((seq, lk, 1, [1]))
                seq += 1
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.trace")
            write_call_trace(p, rows)
            regimes = ec._parse_regimes(
                ["probe=0:4", "prefill=4:6", "decode=6:12", "corpus=4:12"])
            res = ec.measure(p, [1, 2], regimes, ec.DEFAULT_SEEDS,
                             calls_per_forward=2)
        self.assertEqual(res["budgets"], [1, 2])
        self.assertIn("probe", res["seeds"]["1"])
        x = res["seeds"]["1"]["decode"]["splitmix64"]
        self.assertEqual(x["pinned_cells"], 2)          # 1 slot x 2 layers
        self.assertIn("rolling", res)
        self.assertIn("lru", res)

    def test_parse_regimes(self):
        r = ec._parse_regimes(["a=0:5", "b=5:end"])
        self.assertEqual(r["a"], (0, 5))
        self.assertEqual(r["b"], (5, None))


if __name__ == "__main__":
    unittest.main()
