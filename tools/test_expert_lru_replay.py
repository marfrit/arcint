#!/usr/bin/env python3
"""Unit ladder for tools/expert_lru_replay.py (stdlib only).

Runs entirely offline against the committed fixture
tools/testdata/qwen4exp_moe_trace_sample.txt (a contiguous 400-token slice of
the sha-pinned WP6b routing trace) -- no external checkout, no full trace needed.
The full-trace WP6b reproduction is a separate check
(``expert_lru_replay.py --check <full-trace>``), runnable where that trace is
present, the same split gen_ngram_vectors.py uses.

What these tests pin, red-first-capable:
  * the replay parses the fixture and reports the expected shape;
  * the two cache models are DISTINGUISHABLE on the fixture (per-layer LRU reads
    materially below global LRU) -- the WP7 finding, so a regression that
    conflated them fails here;
  * the per-layer hit fed into the projection (flash_next_fit) yields a
    materially LOWER t/s than the global figure would -- the model choice is
    not cosmetic;
  * the geometry constants agree across the replay, the projection instrument
    and (by their shared values) the C++ policy header.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import expert_lru_replay as r  # noqa: E402
import flash_next_fit as ff  # noqa: E402

FIXTURE = os.path.join(HERE, "testdata", "qwen4exp_moe_trace_sample.txt")

# Fixture operating point at a 16 GiB resident budget (145 slots/layer under the
# budget-safe floor). Measured 2026-09-10 on the committed fixture; exact, so a
# tight tolerance. These are the fixture's OWN numbers (a 400-token window), not
# the full-trace WP6b table -- that lives in --check.
FIX_SLOTS_16 = 145
FIX_PER_LAYER_16 = 91.39
FIX_GLOBAL_16 = 93.81


class TestReplayFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = r.load_trace(FIXTURE)

    def test_fixture_shape(self):
        self.assertEqual(len(self.rows), 19200)                 # 400 tokens x 48 layers
        self.assertEqual(len({x[0] for x in self.rows}), 400)
        self.assertEqual(len({x[1] for x in self.rows}), 48)
        # top-10 routed experts per (token, layer)
        self.assertTrue(all(len(x[2]) == 10 for x in self.rows))

    def test_comment_header_is_skipped(self):
        # the fixture carries a '#'-prefixed provenance header; the loader must
        # skip it (otherwise the first data row would be misparsed)
        with open(FIXTURE) as f:
            first = f.readline()
        self.assertTrue(first.startswith("#"))
        self.assertEqual(self.rows[0][0], 0)                    # first data row is token 0

    def test_slots_per_layer_floors_budget_safe(self):
        # matches src/exec/flash_next_offload.h::flash_next_slots_per_layer
        self.assertEqual(r.slots_per_layer_for_gib(16), FIX_SLOTS_16)
        self.assertEqual(r.slots_per_layer_for_gib(24), 218)
        self.assertEqual(r.slots_per_layer_for_gib(40), 364)

    def test_per_layer_and_global_are_distinguishable(self):
        pl = r.replay_per_layer(self.rows, FIX_SLOTS_16) * 100
        gl = r.replay_global(self.rows, r.global_slots_for_gib(16)) * 100
        self.assertAlmostEqual(pl, FIX_PER_LAYER_16, delta=0.3)
        self.assertAlmostEqual(gl, FIX_GLOBAL_16, delta=0.3)
        # THE WP7 FINDING: per-layer (arcint's slot-pool shape) reads below the
        # global (FreeToken) shape by a real margin -- not interchangeable.
        self.assertLess(pl, gl)
        self.assertGreater(gl - pl, 2.0)

    def test_model_choice_moves_the_projection(self):
        pl = r.replay_per_layer(self.rows, FIX_SLOTS_16)
        gl = r.replay_global(self.rows, r.global_slots_for_gib(16))
        tps_pl = ff.project_tps(pl, 44.4, 1.68)   # measured DRAM / NVMe feeds
        tps_gl = ff.project_tps(gl, 44.4, 1.68)
        # adopting the optimistic global model would overstate the projection;
        # the served per-layer pool is materially slower.
        self.assertLess(tps_pl, tps_gl)
        self.assertGreater(tps_gl - tps_pl, 2.0)

    def test_geometry_constants_agree_across_modules(self):
        # the replay and the projection instrument must share the measured
        # geometry (the C++ header pins the same values in its own test).
        self.assertEqual(r.SLICE_BYTES, ff.SLICE_BYTES)
        self.assertEqual(r.N_LAYERS, ff.L_LAYERS)
        self.assertEqual(r.N_EXPERTS, ff.E_EXPERTS)


if __name__ == "__main__":
    unittest.main()
