#!/usr/bin/env python3
"""Unit ladder for tools/expert_policy_compare.py (stdlib only, device-free).

Red-first: each cell fails against a plausible wrong implementation before it
passes against the one here.

  * the splitmix64 rank key is pinned to GOLDEN values computed from patch
    0018's own C++ constants, so a refactor that changes the mix, the layer
    mix constant, or the seed is caught rather than silently changing which
    experts the incumbent pins.
  * the incumbent's resident set is deterministic, ascending, and edge-correct
    (0 slots -> empty, >= num_expert -> everything), and its tie-break is the
    ASCENDING expert id -- a wrong tie-break is invisible until it moves a
    served answer.
  * the frequency seed is the top-`slots` by count with an ascending-id tie.
  * the calibration census counts BATCHED calls (their ids are aggregates, and
    dropping them under-counts the corpus) while the evaluation keeps only
    single-chunk decode calls and COUNTS the skipped batched ones.
  * the honesty guard: an in-sample frequency seed is refused by the CLI.
"""
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expert_policy_compare as epc  # noqa: E402
import expert_lru_replay as lru  # noqa: E402


def write_call_trace(path, rows, header="# source=plugin-0044-call-trace\n"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for seq, lk, top_k, ids in rows:
            f.write(" ".join([str(seq), str(lk), str(top_k)] + [str(x) for x in ids]) + "\n")


def write_v1_trace(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# arcint routing trace v1\n")
        for tok, lk, ids in rows:
            f.write(" ".join([str(tok), str(lk)] + [str(x) for x in ids]) + "\n")


class TestSplitmix64(unittest.TestCase):
    def test_rank_key_golden_values(self):
        # ORACLE, cross-language (2026-09-21): the patch's own
        # static_partition.hpp was reconstructed from 0018-*.patch and
        # compiled (`g++ -std=c++17`), and its static_partition_rank_key /
        # static_partition_resident_experts printed EXACTLY these values:
        #   rank(seed,0,0)=0xcab2b6579e38a8e3
        #   rank(seed,0,1)=0xe2492761c8141779
        #   resident(lk=0,cap=3)=22,189,359
        #   resident(lk=704,cap=3)=321,344,499
        #   resident(lk=704,cap=6)=289,321,337,344,499,509
        # A Python-only transcription could be wrong in the same way twice;
        # a cross-language check cannot, so these are oracle values and not
        # the function under test restated.
        self.assertEqual(epc.rank_key(epc.STATIC_PARTITION_SEED, 0, 0),
                         0xCAB2B6579E38A8E3)
        self.assertEqual(epc.rank_key(epc.STATIC_PARTITION_SEED, 0, 1),
                         0xE2492761C8141779)

    def test_seed_and_layer_mix_constants(self):
        self.assertEqual(epc.STATIC_PARTITION_SEED, 0xF2A17C0DE5EED)
        self.assertEqual(epc.LAYER_MIX, 0xD6E8FEB86659FD93)


class TestStaticSet(unittest.TestCase):
    def test_golden_small_set(self):
        self.assertEqual(epc.static_set_splitmix64(0, 3), [22, 189, 359])

    def test_deterministic_and_ascending(self):
        a = epc.static_set_splitmix64(704, 6)
        b = epc.static_set_splitmix64(704, 6)
        self.assertEqual(a, b)
        self.assertEqual(a, sorted(a))
        self.assertEqual(a, [289, 321, 337, 344, 499, 509])

    def test_zero_and_full_capacity(self):
        self.assertEqual(epc.static_set_splitmix64(100, 0), [])
        full = epc.static_set_splitmix64(100, epc.N_EXPERTS)
        self.assertEqual(len(full), epc.N_EXPERTS)
        self.assertEqual(full, list(range(epc.N_EXPERTS)))

    def test_capacity_above_num_expert_is_clamped(self):
        self.assertEqual(len(epc.static_set_splitmix64(100, epc.N_EXPERTS + 500)),
                         epc.N_EXPERTS)

    def test_tie_break_is_ascending_expert_id(self):
        # with a constant rank every expert ties: the tie-break must pick the
        # LOWEST ids, not whatever the sort happened to do.
        with mock.patch.object(epc, "rank_key", return_value=0):
            self.assertEqual(epc.static_set_splitmix64(7, 4), [0, 1, 2, 3])


class TestFrequencySeed(unittest.TestCase):
    def test_top_slots_with_ascending_tie(self):
        counts = {(100, 5): 9, (100, 6): 9, (100, 7): 3, (200, 1): 4}
        sets = epc.static_sets_frequency(counts, 3)
        self.assertEqual(sets[100], {5, 6, 7})
        self.assertEqual(sets[200], {1})

    def test_tie_breaks_on_the_lower_expert_id(self):
        counts = {(100, 8): 5, (100, 3): 5, (100, 9): 5}
        self.assertEqual(epc.static_sets_frequency(counts, 2)[100], {3, 8})

    def test_zero_slots_is_empty(self):
        self.assertEqual(epc.static_sets_frequency({(1, 1): 9}, 0)[1], set())


class TestCorpusSides(unittest.TestCase):
    def test_calibration_counts_batched_calls(self):
        # a batched call's ids are aggregate accesses: they must be counted,
        # or the census under-counts the corpus it is meant to measure.
        calls = [(0, 100, 2, [1, 2, 3, 4]), (1, 100, 2, [1, 5])]
        counts = epc.census_from_calls(calls)
        self.assertEqual(counts[(100, 1)], 2)
        self.assertEqual(counts[(100, 4)], 1)
        self.assertEqual(sum(counts.values()), 6)

    def test_evaluation_skips_batched_and_counts_it(self):
        calls = [(0, 100, 2, [1, 2, 3, 4]), (1, 100, 2, [1, 5]), (2, 100, 2, [1, 6])]
        rows, skipped = epc.eval_rows_from_calls(calls)
        self.assertEqual(skipped, 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], (100, [1, 5]))


class TestHitRate(unittest.TestCase):
    def test_known_case(self):
        rows = [(100, [1, 2]), (100, [3, 4]), (200, [1, 2])]
        sets = {100: {1, 2}}
        hr, hits, acc = epc.hit_rate(rows, sets)
        self.assertEqual((hits, acc), (2, 6))
        self.assertAlmostEqual(hr, 2 / 6)

    def test_missing_layer_keeps_no_residents(self):
        hr, hits, acc = epc.hit_rate([(999, [1, 2])], {100: {1, 2}})
        self.assertEqual((hits, acc), (0, 2))


class TestLru(unittest.TestCase):
    def test_lru_matches_the_repo_replay_model_on_the_same_accesses(self):
        # the docstring claims parity with tools/expert_lru_replay.py's
        # replay_per_layer; assert it rather than assert it in prose. Feed the
        # SAME access sequence through both implementations.
        rows = [(100, [1]), (100, [2]), (100, [1]), (100, [3]), (100, [1]),
                (200, [4]), (200, [4]), (100, [3])]
        tokens = [(i, lk, ids) for i, (lk, ids) in enumerate(rows)]
        for slots in (0, 1, 2, 3, 8):
            ours, _h, _a = epc.lru_hit_rate(rows, slots)
            theirs = lru.replay_per_layer(tokens, slots)
            self.assertAlmostEqual(ours, theirs, places=12)

    def test_lru_promotes_on_hit_unlike_fifo(self):
        # capacity 2, accesses 1,2,1,3,1: TRUE LRU promotes 1 on the hit, so 3
        # evicts 2 and the last 1 hits -> 2 hits. FIFO (no promotion) keeps 1
        # as the oldest, so 3 evicts 1 and the last access misses -> 1 hit.
        # The capacity MUST be exceeded for the two to differ: at capacity 3
        # this same sequence is 2 hits under BOTH, so a capacity-3 cell pins
        # nothing (a vacuous cell, caught in review before the commit).
        rows = [(100, [1]), (100, [2]), (100, [1]), (100, [3]), (100, [1])]
        hr, hits, acc = epc.lru_hit_rate(rows, 2)
        self.assertEqual((hits, acc), (2, 5))

    def test_full_repeat_is_a_hit_after_the_first(self):
        rows = [(100, [1, 2]), (100, [1, 2]), (100, [1, 2])]
        hr, hits, acc = epc.lru_hit_rate(rows, 2)
        self.assertEqual((hits, acc), (4, 6))

    def test_all_distinct_never_hits(self):
        rows = [(100, [i]) for i in range(5)]
        hr, hits, acc = epc.lru_hit_rate(rows, 2)
        self.assertEqual((hits, acc), (0, 5))

    def test_zero_budget_is_all_misses_and_does_not_crash(self):
        hr, hits, acc = epc.lru_hit_rate([(100, [1, 2])], 0)
        self.assertEqual((hits, acc), (0, 2))


class TestFormatDetection(unittest.TestCase):
    def test_header_marks_call_trace(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t")
            write_call_trace(p, [(0, 100, 2, [1, 2])])
            self.assertEqual(epc.detect_format(p), "call")

    def test_header_marks_v1_trace(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t")
            write_v1_trace(p, [(0, 100, [1, 2])])
            self.assertEqual(epc.detect_format(p), "v1")

    def test_v1_row_is_not_read_as_a_call_row(self):
        # "0 100 2 3" -- a v1 row whose third field (2) would divide the id
        # count; header detection keeps it v1, and the ids stay ids.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t")
            write_v1_trace(p, [(0, 100, [2, 3])])
            calls = epc._load(p, "call")   # explicit wrong flag: header wins
            self.assertEqual(calls, [(0, 100, 2, [2, 3])])
            self.assertEqual(epc.detect_format(p), "v1")


class TestHonestyGuard(unittest.TestCase):
    def _cm(self, args):
        return epc.main(args)

    def test_in_sample_calibration_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t")
            write_call_trace(p, [(i, 100, 2, [1, 2]) for i in range(6)])
            rc = self._cm(["--calibrate-call", p, "--eval-call", p,
                           "--slots-per-layer", "2"])
            self.assertEqual(rc, 2)

    def test_disjoint_ranges_are_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t")
            write_call_trace(p, [(i, 100, 2, [1, 2]) for i in range(6)])
            buf = io.StringIO()
            with mock.patch("sys.stdout", buf):
                rc = self._cm(["--calibrate-call", p, "--calibrate-seq-hi", "3",
                               "--eval-call", p, "--eval-seq-lo", "3",
                               "--slots-per-layer", "2"])
            self.assertEqual(rc, 0)
            self.assertIn("static-splitmix64", buf.getvalue())
            self.assertIn("static-frequency", buf.getvalue())
            self.assertIn("lru-per-layer", buf.getvalue())

    def test_zero_slots_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t")
            write_call_trace(p, [(i, 100, 2, [1, 2]) for i in range(6)])
            buf = io.StringIO()
            with mock.patch("sys.stdout", buf):
                rc = self._cm(["--calibrate-call", p, "--calibrate-seq-hi", "3",
                               "--eval-call", p, "--eval-seq-lo", "3",
                               "--slots-per-layer", "0"])
            self.assertEqual(rc, 0)
            self.assertIn("undefined", buf.getvalue())

    def test_uppercase_seed_literal_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t")
            write_call_trace(p, [(i, 100, 2, [1, 2]) for i in range(6)])
            with mock.patch("sys.stdout", io.StringIO()):
                rc = self._cm(["--calibrate-call", p, "--calibrate-seq-hi", "3",
                               "--eval-call", p, "--eval-seq-lo", "3",
                               "--slots-per-layer", "2", "--seed", "0XF2A17C0DE5EED"])
            self.assertEqual(rc, 0)

    def test_convergence_without_calibration_is_announced(self):
        # without an explicit calibration corpus the run is in-sample, so the
        # guard refuses unless --allow-overlap is given; then --convergence
        # must say it was skipped rather than silently print nothing.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t")
            write_call_trace(p, [(i, 100, 2, [1, 2]) for i in range(6)])
            buf = io.StringIO()
            with mock.patch("sys.stdout", buf), mock.patch("sys.stderr", io.StringIO()):
                rc = self._cm(["--eval-call", p, "--slots-per-layer", "2",
                               "--allow-overlap", "--convergence"])
            self.assertEqual(rc, 0)
            self.assertIn("convergence skipped", buf.getvalue())

    def test_convergence_without_calibration_is_refused_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t")
            write_call_trace(p, [(i, 100, 2, [1, 2]) for i in range(6)])
            with mock.patch("sys.stdout", io.StringIO()), \
                    mock.patch("sys.stderr", io.StringIO()):
                rc = self._cm(["--eval-call", p, "--slots-per-layer", "2",
                               "--convergence"])
            self.assertEqual(rc, 2)

    def test_allow_overlap_runs_but_warns(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t")
            write_call_trace(p, [(i, 100, 2, [1, 2]) for i in range(6)])
            err = io.StringIO()
            with mock.patch("sys.stderr", err), mock.patch("sys.stdout", io.StringIO()):
                rc = self._cm(["--calibrate-call", p, "--eval-call", p,
                               "--slots-per-layer", "2", "--allow-overlap"])
            self.assertEqual(rc, 0)
            self.assertIn("LOOK-AHEAD", err.getvalue())


class TestComparison(unittest.TestCase):
    def test_seeds_are_not_the_same_policy(self):
        # a comparison where both policies pin the same experts would be
        # vacuous; on a skewed census at a small budget they must differ.
        calls = [(i, 100, 2, [1, 2]) for i in range(20)]
        calls += [(i, 200, 2, [7, 8]) for i in range(20, 40)]
        res = epc.compare(calls, 0, None, calls, 0, None, 2)
        self.assertLess(res["layers_where_seeds_agree"], res["layers"])

    def test_frequency_beats_splitmix64_in_sample_only_by_construction(self):
        # in-sample, the frequency seed must be >= the incumbent on the very
        # accesses it was fitted to -- this is why the tool refuses in-sample
        # scoring. The cell pins that the guard is load-bearing.
        calls = [(i, 100, 2, [1, 2]) for i in range(20)]
        res = epc.compare(calls, 0, None, calls, 0, None, 2)
        self.assertGreaterEqual(res["frequency_minus_splitmix64_hit_pct"], 0.0)

    def test_convergence_heads_stay_inside_the_calibration_window(self):
        # honesty rule: every head must be a prefix of the CALIBRATION window,
        # never reaching into the scored tail. With calibration [0,4) and eval
        # [4,None), the largest head must be 3 calls, i.e. calib_calls <= 4.
        calls = [(i, 100, 2, [1, 2]) for i in range(8)]
        rows, _ = epc.eval_rows_from_calls(calls, lo=4)
        series = epc.convergence_series(calls, rows, 2, n_points=4, cal_lo=0, cal_hi=4)
        self.assertTrue(all(r["calib_calls"] <= 4 for r in series))

    def test_convergence_series_is_bounded_and_shaped(self):
        calls = [(i, 100, 2, [1, 2]) for i in range(40)]
        rows, _ = epc.eval_rows_from_calls(calls)
        series = epc.convergence_series(calls, rows, 2, n_points=4)
        self.assertEqual(len(series), 4)
        for row in series:
            self.assertGreaterEqual(row["hit_rate"], 0.0)
            self.assertLessEqual(row["hit_rate"], 1.0)
        self.assertIn("layers_changed", series[0])


if __name__ == "__main__":
    unittest.main()
