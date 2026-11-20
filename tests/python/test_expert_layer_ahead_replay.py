#!/usr/bin/env python3
"""tools/expert_layer_ahead_replay.py, device-free: the arms, the metrics, the
online order and the cache metric, on synthetic captures whose answer is known.

Campaign expert-foresight, "Queued (operator, 2026-11-19): a layer-ahead
predictor", step 2. Every cell was red first, with the mutation named:

1. The router read and the capture read. A synthetic GGUF (F32 and Q8_0
   `blk.<il>.ffn_gate_inp.weight`) loads as (n_layer, n_embd, n_expert) with
   logits = W.T x; a synthetic LAC1 file reads back positions, phases, ids and
   f16 inputs. Red mutation seen failing: `load_routers` ignores `tensor_type`
   and reads Q8_0's bytes as F32 (shape 528, not 512 x ...: loud, or garbage).
   Skipped loudly when gguf-py is not importable (ARCINT_GGUF_PY / --gguf-py).

2. Arm A is exact when the next layer's router equals its own. Five layers, one
   fixed router M an every layer, the target layers' captured inputs equal the
   source layer's, so `W_lt x_l` is the target's own logits and recall@1 and
   recall_used are 1.0 at d = 1, 2 and 4 and every K >= n_used. Red mutation
   seen failing: the reuse logits take `W` transposed -- M is not symmetric,
   the ranks change and the 1.0 cells fail.

3. The first layers have no source and are reported. d = 2 lists layers
   [0, 1] in `layers_without_source` and scores only target layers 2..4; d = 4
   lists [0, 1, 2, 3]. Red mutation seen failing: the target range starts at 0
   and `z`/`base` index the negative source -- or the list is reported as [] and
   the cell fails.

4. The metrics on hand-counted rows. Token 0 layer 0: real ids [3, 1], predicted
   top-2 [3, 7] -- recall@1 1, used 1/2. Token 1 layer 0: real [0, 4],
   predicted [5, 6] -- 0. The prompt/decode split is pinned the same way. Red
   mutation seen failing: used_n counts distinct real ids instead of n_used --
   the row whose real ids repeat gives (2, 1) where the cell expects (2, 2).

5. The non-resident metric. The same rows with a residency where real expert 3
   is resident and 1 non-resident: `recall_nonres_used` is 0 / 1 where the
   plain recall is 1 / 2. Red mutation seen failing: `resid_row[e]` inverted
   (`not` dropped) -- residency and non-residency swap and the cell fails.

6. The cache model's residency view. ecr.simulate's `resid_log` collects the
   residency at each step BEFORE the step's hits and before its plan lands; the
   default (no log) leaves the returned numbers byte-identical. Red mutation
   seen failing: the log appends after the hits count and after a plan lands --
   a swap that lands at step 1 shows up at step 1's own view and the cell fails.

7. The slot mode and the slot split. A run without a profile names itself
   `flat` in the JSON, a run with one `profile`. Red mutation seen failing:
   slot_mode returns "flat" for a profile too. Then The production 5,915 over 48 layers: 124 on the first 11,
   123 on the rest, sum 5,915; a shard discount divides it. Red mutation seen
   failing: `base, extra = divmod(share, ...)` replaced by a flat base -- the
   sum is 5,904 and the cell fails.

8. Test-then-train. A correction must learn a fixed permutation: the target
   layer's router is M, its captured input is x_0 permuted, so the target's real
   logits are `M perm x_0` and the reuse logits `M x_0`; rank 8 (= n_embd) can
   represent the map. The FIRST token, scored before the first update, misses
   the real rank-0; after the online steps over identical tokens the last score
   hits it. Red mutations seen failing: `run` updates before it scores (the
   first token's score is 1 instead of 0), and lr = 0 (the last score never
   reaches the real rank-0).

Standard library unittest; numpy is needed for the arrays, and the cells skip
loudly without it, so the recipe runs the file with python3 alone.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

if np is not None:
    import expert_cache_replay as ecr
    import expert_router_capture_read as capture
    import expert_layer_ahead_replay as replay


def _skip_numpy():
    if np is None:
        raise unittest.SkipTest("numpy is not installed: the replay's arrays need it")


def _have_gguf():
    try:
        replay.resolve_gguf()
    except SystemExit as e:
        raise unittest.SkipTest(str(e))


def bits(vals):
    """f16 bit patterns, the capture writer's input."""
    return np.asarray(vals, np.float16).view(np.uint16).tolist()


def write_lac1(path, X, ids, phases, positions=None, token_ids=None, n_expert=None):
    """X (n, nl, e) floats; ids (n, nl, nu) ints; phases 0/1."""
    n, nl, e = X.shape
    nu = ids.shape[2]
    if n_expert is None:
        n_expert = max(e, int(ids.max()) + 1) if ids.size else e
    header = capture.Header(capture.MAGIC, capture.VERSION, nl, e, n_expert, nu,
                            capture.DTYPE_F16)
    if positions is None:
        positions = list(range(n))
    if token_ids is None:
        token_ids = list(range(n))
    tokens = []
    for t in range(n):
        layers = [(bits(X[t, il]), list(ids[t, il])) for il in range(nl)]
        tokens.append((int(positions[t]), int(token_ids[t]), int(phases[t]), layers))
    capture.write_capture(str(path), header, tokens)
    return header


def ids_of(logits, nu):
    return np.argsort(-logits, axis=-1, kind="stable")[..., :nu]


class RouterRead(unittest.TestCase):
    def test_synthetic_gguf_f32_and_q8_0(self):
        _skip_numpy()
        _have_gguf()
        import gguf
        nl, e, ne = 3, 32, 512
        W = np.random.default_rng(1).standard_normal((nl, e, ne)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.gguf")
            w = gguf.GGUFWriter(path, "llama")
            w.add_name("synthetic")
            for il in range(nl):
                w.add_tensor("blk.%d.ffn_gate_inp.weight" % il, W[il])
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()
            got = replay.load_routers(path, nl, e, ne)
        self.assertEqual(got.shape, (nl, e, ne))
        np.testing.assert_allclose(got, W, rtol=0, atol=0)

    def test_q8_0_router_is_dequantized(self):
        _skip_numpy()
        _have_gguf()
        import gguf
        from gguf import quants
        nl, e, ne = 1, 32, 512
        # one Q8_0 row (32 values, 34 bytes) of the router per expert column
        rng = np.random.default_rng(2)
        vals = rng.standard_normal((ne, e)).astype(np.float32) * 0.05
        rows = []
        for j in range(ne):
            q = quants.quantize(vals[j], gguf.GGMLQuantizationType.Q8_0)
            rows.append(q.view(np.uint8))
        raw = np.stack(rows)                     # (ne, 34) bytes, one row an expert
        dq = quants.dequantize(raw.view(np.uint8), gguf.GGMLQuantizationType.Q8_0)
        dq = np.asarray(dq, np.float32).reshape(ne, e)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "q.gguf")
            w = gguf.GGUFWriter(path, "llama")
            w.add_name("synthetic")
            w.add_tensor("blk.0.ffn_gate_inp.weight", raw,
                         raw_shape=(ne, 34), raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()
            got = replay.load_routers(path, nl, e, ne)
        np.testing.assert_allclose(got[0], dq.T, rtol=0, atol=0)

    def test_capture_reads_back(self):
        _skip_numpy()
        nl, e, nu, n = 2, 4, 2, 3
        X = np.random.default_rng(3).standard_normal((n, nl, e)).astype(np.float16)
        ids = np.random.default_rng(4).integers(0, 8, (n, nl, nu)).astype(np.int16)
        phases = np.array([0, 1, 1], np.int8)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.lac1")
            write_lac1(path, X.astype(np.float32), ids, phases, positions=[7, 8, 9],
                       token_ids=[11, 12, 13], n_expert=8)
            cap = replay.Capture(path)
            got = np.concatenate([Xb for _, Xb, _, _ in cap.blocks(2)])
            got_ids = np.concatenate([ib for _, _, ib, _ in cap.blocks(2)])
        np.testing.assert_array_equal(got, X.astype(np.float32))
        np.testing.assert_array_equal(got_ids, ids.astype(np.int32))
        np.testing.assert_array_equal(cap.phases, phases)
        np.testing.assert_array_equal(cap.positions, [7, 8, 9])
        np.testing.assert_array_equal(cap.token_ids, [11, 12, 13])


class ArmReuse(unittest.TestCase):
    def test_next_router_equals_own_is_exact(self):
        _skip_numpy()
        nl, e, ne, nu = 5, 8, 16, 4
        M = np.random.default_rng(5).standard_normal((e, ne)).astype(np.float32)
        x0 = np.random.default_rng(6).standard_normal((3, e)).astype(np.float32)
        X = np.repeat(x0[:, None, :], nl, axis=1)          # every layer the same input
        ids = np.repeat(ids_of(x0 @ M, nu)[:, None, :], nl, axis=1)  # every layer's own routing
        W = np.repeat(M[None], nl, axis=0)
        phases = np.array([0, 0, 1], np.int8)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.lac1")
            write_lac1(path, X, ids, phases, n_expert=ne)
            cap = replay.Capture(path)
            slots = replay.flat_slots(nl, ne, slots=nl * 4)
            stats, _first, _cache, _corr, _fs = replay.run(cap, W, slots, [1, 2, 4], [4], 0.05, 64)
        for dep in (1, 2, 4):
            for K in (4, 8, 16):
                b = stats.block("A", dep, K)
                self.assertEqual(b["decode"]["recall_top1"], 1.0, (dep, K))
                self.assertEqual(b["decode"]["recall_used"], 1.0, (dep, K))

    def test_first_layers_have_no_source(self):
        _skip_numpy()
        nl, e, ne, nu = 5, 4, 8, 2
        M = np.random.default_rng(7).standard_normal((e, ne)).astype(np.float32)
        X = np.ones((2, nl, e), np.float32)
        ids = np.repeat(ids_of(X[:, 0] @ M, nu)[:, None, :], nl, axis=1)
        W = np.repeat(M[None], nl, axis=0)
        phases = np.array([0, 1], np.int8)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.lac1")
            write_lac1(path, X, ids, phases, n_expert=ne)
            cap = replay.Capture(path)
            stats, first, _cache, _corr, _fs = replay.run(cap, W, replay.flat_slots(nl, ne, slots=8),
                                                        [1, 2, 4], [4], 0.05, 64)
        self.assertEqual(first[1], [0])
        self.assertEqual(first[2], [0, 1])
        self.assertEqual(first[4], [0, 1, 2, 3])


class Metrics(unittest.TestCase):
    def test_hand_counted_rows(self):
        _skip_numpy()
        pred0 = np.array([0.0, 0.1, 0.2, 3.0, 0.3, 0.4, 0.5, 0.6])
        pred1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 5.0, 0.0])
        self.assertEqual(replay.top_k(pred0, 2).tolist(), [3, 7])
        self.assertEqual(replay.top_k(pred1, 2).tolist(), [5, 6])
        r0 = replay.row_stats(replay.top_k(pred0, 2), [3, 1])
        r1 = replay.row_stats(replay.top_k(pred1, 2), [0, 4])
        self.assertEqual(r0["top1_hit"], 1)
        self.assertEqual((r0["used_hit"], r0["used_n"]), (1, 2))
        self.assertEqual(r1["top1_hit"], 0)
        self.assertEqual((r1["used_hit"], r1["used_n"]), (0, 2))
        # a real row with a repeated id counts n_used, not the set's size
        r2 = replay.row_stats(replay.top_k(pred0, 2), [3, 3])
        self.assertEqual((r2["used_hit"], r2["used_n"]), (2, 2))

    def test_nonresident_recall_hand_counted(self):
        _skip_numpy()
        pred = np.array([0.0, 0.1, 0.2, 3.0, 0.3, 0.4, 0.5, 0.6])
        resid = np.zeros(8, bool)
        resid[3] = True
        r = replay.row_stats(replay.top_k(pred, 2), [3, 1], resid)
        self.assertEqual((r["nres_hit"], r["nres_n"]), (0, 1))
        self.assertEqual((r["nres_top1_n"], r["nres_top1_hit"]), (0, 0))
        # the real rank-0 non-resident and hit
        resid2 = np.zeros(8, bool)
        r2 = replay.row_stats(replay.top_k(pred, 2), [3, 1], resid2)
        self.assertEqual((r2["nres_top1_n"], r2["nres_top1_hit"]), (1, 1))
        self.assertEqual((r2["nres_hit"], r2["nres_n"]), (1, 2))


class CacheView(unittest.TestCase):
    def test_resid_log_is_the_step_view_and_default_unchanged(self):
        _skip_numpy()
        layers = [dict(il=0, n_expert=4, n_slots=2, slots=np.array([0, 1]))]
        steps = [(1, np.array([[2, 3]])), (1, np.array([[0, 1]])),
                 (1, np.array([[2, 3]])), (1, np.array([[0, 1]]))]
        a = ecr.simulate(layers, steps, "strata", adapt_every=1, max_swaps=2, delay=0,
                         min_usage=1.0, min_gain=0.5)
        log = []
        b = ecr.simulate(layers, steps, "strata", adapt_every=1, max_swaps=2, delay=0,
                         min_usage=1.0, min_gain=0.5, resid_log=log)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[3], b[3])
        self.assertEqual(len(log), len(steps))
        # step 0: the first token is all misses, the plan lands at step 0's end
        # step 0 plans the swap; it lands at step 0's end, so step 1 sees it
        self.assertEqual(log[0][0].tolist(), [True, True, False, False])
        self.assertEqual(log[1][0].tolist(), [False, False, True, True])

    def test_slot_mode_is_named(self):
        _skip_numpy()
        self.assertEqual(replay.slot_mode(None), "flat")
        self.assertEqual(replay.slot_mode("expert-profile-strata.bin"), "profile")

    def test_slot_split(self):
        _skip_numpy()
        layers = replay.flat_slots(48, 512, 5915)
        self.assertEqual(sum(int(L["n_slots"]) for L in layers), 5915)
        self.assertEqual(sum(1 for L in layers if L["n_slots"] == 124), 11)
        self.assertEqual(sum(1 for L in layers if L["n_slots"] == 123), 37)
        shard = replay.flat_slots(48, 512, 5915, discount_shard=8)
        self.assertEqual(sum(int(L["n_slots"]) for L in shard), 5915 // 8)

    def test_profile_walk_spends_the_budget(self):
        _skip_numpy()
        base = replay.flat_slots(2, 4, slots=3)
        out = ecr.profile_slots(base, [(0, 0), (0, 1), (1, 1)])
        self.assertEqual(sum(int(L["n_slots"]) for L in out), 3)
        self.assertEqual(out[0]["n_slots"], 2)
        self.assertEqual(out[1]["n_slots"], 1)


class OnlineOrder(unittest.TestCase):
    def _perm_case(self):
        nl, e, ne, nu = 2, 6, 6, 2
        M = np.random.default_rng(8).standard_normal((e, ne)).astype(np.float32)
        x = np.random.default_rng(9).standard_normal(e).astype(np.float32)
        perm = np.array([3, 4, 5, 0, 1, 2])
        x1 = x[perm]                                     # target's input is the permutation
        real = x1 @ M                                    # the target's own logits
        base = x @ M                                     # reuse: the same weight, other input
        ids = ids_of(real, nu)
        self.assertNotEqual(int(np.argmax(real)), int(np.argmax(base)),
                            "the synthetic must differ: reuse is wrong on this token")
        X = np.zeros((40, nl, e), np.float32)
        X[:, 0] = x
        X[:, 1] = x1
        phases = np.array([0] + [1] * 39, np.int8)
        W = np.repeat(M[None], nl, axis=0)
        W[1] = M
        idrows = np.zeros((40, nl, nu), np.int32)
        idrows[:, 0, :] = ids_of(x @ M, nu)              # layer 0's own routing
        idrows[:, 1, :] = ids                             # layer 1's routing
        return X, idrows, phases, W, M, e, ne

    def test_first_score_is_before_the_first_update_and_learning_reaches_it(self):
        _skip_numpy()
        X, idrows, phases, W, M, e, ne = self._perm_case()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.lac1")
            write_lac1(path, X, idrows, phases, n_expert=ne)
            cap = replay.Capture(path)
            slots = replay.flat_slots(2, ne, slots=4)
            rid = replay.run(cap, W, slots, [1], [8], 0.05, 64, seed=0)
        stats, first, cache, corr, first_score = rid
        c = corr[(1, 8)]
        # exactly one prediction and one update per token, in that order: a
        # token's label enters the parameters only after its own score
        self.assertEqual(len(c.events), 2 * 40)
        self.assertEqual(c.events[:6], [("predict", 0), ("update", 0),
                                        ("predict", 1), ("update", 1),
                                        ("predict", 2), ("update", 2)])
        self.assertEqual(first_score[(1, 8)], 0, "reuse misses the first token's rank-0")
        b = stats.block("B8", 1, 32)
        self.assertEqual(b["decode"]["recall_top1"], 1.0, "the correction learns the permutation")


if __name__ == "__main__":
    unittest.main()
