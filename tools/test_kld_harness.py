#!/usr/bin/env python3
"""Unit ladder for tools/kld_harness.py -- the KLD acceptance instrument.

Torch-free and model-free (numpy only), so it runs on the build host. These
tests pin the KL core's arithmetic against hand-computed values and assert the
red probe actually fires -- an acceptance instrument that cannot go red is not
admitted (docs/design-qwen-flash-next.md, WP3.5 / WP5).
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kld_harness as kh  # noqa: E402


class TestKLCore(unittest.TestCase):
    def test_identical_is_zero(self):
        import numpy as np
        rng = np.random.default_rng(1)
        logits = rng.standard_normal((32, 100)) * 3.0
        per = kh.kl_per_token(logits, logits.copy())
        self.assertLess(float(per.max()), 1e-12)

    def test_matches_hand_computed_two_class(self):
        # Two-class case with known probabilities. logits -> softmax:
        # ref [0, log3] -> p=[0.25, 0.75]; cand [0,0] -> q=[0.5,0.5].
        # KL = 0.25*ln(0.25/0.5) + 0.75*ln(0.75/0.5).
        import numpy as np
        ref = np.array([[0.0, math.log(3.0)]])
        cand = np.array([[0.0, 0.0]])
        expect = 0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5)
        got = float(kh.kl_per_token(ref, cand)[0])
        self.assertAlmostEqual(got, expect, places=12)

    def test_kl_is_nonnegative(self):
        import numpy as np
        rng = np.random.default_rng(2)
        a = rng.standard_normal((64, 256)) * 2.0
        b = a + rng.standard_normal((64, 256)) * 0.5
        per = kh.kl_per_token(a, b)
        self.assertTrue((per >= -1e-12).all())

    def test_shape_mismatch_refused(self):
        import numpy as np
        with self.assertRaises(ValueError):
            kh.kl_per_token(np.zeros((4, 10)), np.zeros((4, 11)))

    def test_gate_boundary(self):
        # At threshold -> PASS (<=); just above -> RED.
        below = {"mean": kh.THRESHOLD_NATS - 1e-6, "max": 0, "p95": 0, "tokens": 1}
        at = {"mean": kh.THRESHOLD_NATS, "max": 0, "p95": 0, "tokens": 1}
        above = {"mean": kh.THRESHOLD_NATS + 1e-6, "max": 0, "p95": 0, "tokens": 1}
        self.assertEqual(kh.gate(below)[1], "PASS")
        self.assertEqual(kh.gate(at)[1], "PASS")
        self.assertEqual(kh.gate(above)[1], "RED")


class TestRedProbe(unittest.TestCase):
    def test_self_test_passes(self):
        # The self-test asserts its own red-probe fires; rc 0 means the
        # instrument read zero, stayed green under the bar, and went red above.
        self.assertEqual(kh.self_test(), 0)

    def test_large_drift_actually_red(self):
        # Independent of self_test(): a coarse perturbation must exceed the bar.
        import numpy as np
        rng = np.random.default_rng(7)
        ref = rng.standard_normal((200, 2048)) * 4.0
        large = ref + rng.standard_normal((200, 2048)) * 0.6
        s = kh.summarize_kl(kh.kl_per_token(ref, large))
        self.assertGreater(s["mean"], kh.THRESHOLD_NATS)
        self.assertEqual(kh.gate(s)[1], "RED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
