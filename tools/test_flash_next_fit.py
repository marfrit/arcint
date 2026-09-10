#!/usr/bin/env python3
"""Unit ladder for tools/flash_next_fit.py (stdlib only, no numpy/torch).

Pins the streaming-fit projection model's regime ceilings and monotonicity so
a mis-edited formula is caught. The tool prints a PROJECTION (bandwidth-bound
estimate) fed by measured inputs; these tests check the arithmetic, not a
served throughput.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flash_next_fit as ff  # noqa: E402


class TestFitModel(unittest.TestCase):
    def test_self_test_passes(self):
        self.assertEqual(ff.self_test(), 0)

    def test_full_resident_hits_dram_ceiling(self):
        t = ff.project_tps(1.0, 44.4, 4.66, amortization=1.0)
        expect = 1000.0 / ((ff.TRAFFIC_GIB / 44.4) * 1000.0)
        self.assertAlmostEqual(t, expect, places=3)

    def test_full_miss_hits_nvme_ceiling(self):
        t = ff.project_tps(0.0, 44.4, 4.66, amortization=1.0)
        expect = 1000.0 / ((ff.TRAFFIC_GIB / 4.66) * 1000.0)
        self.assertAlmostEqual(t, expect, places=3)

    def test_monotonic_in_hit_rate(self):
        ts = [ff.project_tps(h, 44.4, 4.66, 1.0) for h in (0.0, 0.3, 0.6, 0.9, 1.0)]
        self.assertEqual(ts, sorted(ts))

    def test_amortization_helps_only_misses(self):
        # At h=1 (all resident) amortization changes nothing; at h<1 it raises t/s.
        self.assertAlmostEqual(ff.project_tps(1.0, 44.4, 4.66, 1.0),
                               ff.project_tps(1.0, 44.4, 4.66, 5.0), places=6)
        self.assertGreater(ff.project_tps(0.5, 44.4, 4.66, 3.0),
                           ff.project_tps(0.5, 44.4, 4.66, 1.0))

    def test_table_reserved_from_dram_first(self):
        _, _, dram_for_experts = ff.resident_expert_gib(15, 44, 2.3, 3.0, 2.0)
        self.assertLessEqual(dram_for_experts, 44 - ff.TABLE_GIB)

    def test_geometry_constants_match_measured(self):
        # WP6 measured: 56.25 GiB pool, 1.0986 GiB/token, 26.82 GiB table.
        self.assertAlmostEqual(ff.EXPERT_POOL_GIB, 56.25, places=2)
        self.assertAlmostEqual(ff.TRAFFIC_GIB, 1.0986, places=3)
        self.assertAlmostEqual(ff.TABLE_GIB, 26.82, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
