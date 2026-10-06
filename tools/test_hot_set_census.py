#!/usr/bin/env python3
"""Unit ladder for tools/hot_set_census.py (stdlib only, device-free).

Red-first: each cell fails against a plausible wrong implementation before it
passes against the one here.

  * parse_trace REFUSES a malformed row (fewer than three fields) where the
    replay loader silently drops it -- a census that drops a row undercounts,
    so the refusal is the point.
  * the canonical summary is a pure function of the trace's own layer-index
    space, sorted, and its `# total,` equals the routed-access count.
  * selection is deterministic, budgeted (exactly `slots_per_layer` per layer
    when the layer has that many distinct experts) and tie-breaks on the
    ASCENDING expert id -- a wrong tie-break is invisible until it moves a
    served answer, so it is pinned here instead.
  * rounds_to_plateau returns the first stable prefix and REFUSES to claim a
    plateau for a ranking that never stabilises (clause V3's failing shape).
  * patch 0013's four-column CSV parses, and a three-column row, a
    wrong `# total,`, and a perturbed join count are all refused or caught
    (the schema blocker the campaign review caught is pinned as a cell).

The fixture is the committed WP7 routing sample
`tools/testdata/qwen4exp_moe_trace_sample.txt` (400 tokens x 48 layers,
top-10), the same trace tools/test_expert_lru_replay.py uses. The full-trace
WP6b reproduction stays a separate check.
"""
import contextlib
import inspect
import io
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hot_set_census as hc  # noqa: E402

FIXTURE = os.path.join(HERE, "testdata", "qwen4exp_moe_trace_sample.txt")


def write_trace(path, rows, header=("# arcint routing trace v1",)):
    lines = list(header)
    for tok, lay, experts in rows:
        lines.append(" ".join([str(tok), str(lay)] + [str(e) for e in experts]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


class TestParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = hc.parse_trace(FIXTURE)

    def test_fixture_shape(self):
        self.assertEqual(len(self.rows), 19200)                    # 400 x 48
        self.assertEqual(len({r[0] for r in self.rows}), 400)
        self.assertEqual(len({r[1] for r in self.rows}), 48)
        self.assertTrue(all(len(r[2]) == 10 for r in self.rows))   # top-10

    def test_comment_and_blank_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            write_trace(p, [(0, 0, [1, 2])], header=("# a comment", "", "# k=v"))
            rows = hc.parse_trace(p)
            self.assertEqual(rows, [(0, 0, [1, 2])])

    def test_malformed_row_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            with open(p, "w") as f:
                f.write("0 0 1 2\n0 1\n")                          # 2-field row
            with self.assertRaises(ValueError):
                hc.parse_trace(p)

    def test_negative_index_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            with open(p, "w") as f:
                f.write("0 0 -1 2\n")
            with self.assertRaises(ValueError):
                hc.parse_trace(p)

    def test_non_integer_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            with open(p, "w") as f:
                f.write("0 0 1 x\n")
            with self.assertRaises(ValueError):
                hc.parse_trace(p)

    def test_multi_pair_provenance_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            write_trace(p, [(0, 0, [1, 2])],
                        header=("# arcint routing trace v1",
                                "# card=none device=cpu depth=48 kv=u8"))
            prov = hc.read_provenance(p)
            self.assertEqual(prov.get("card"), "none")
            self.assertEqual(prov.get("device"), "cpu")
            self.assertEqual(prov.get("depth"), "48")
            self.assertEqual(prov.get("kv"), "u8")

    def test_non_strict_drops_malformed(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            with open(p, "w") as f:
                f.write("0 0 1 2\n0 1\n1 0 3\n")
            rows = hc.parse_trace(p, strict=False)
            self.assertEqual(rows, [(0, 0, [1, 2]), (1, 0, [3])])


class TestSummary(unittest.TestCase):
    def test_summary_is_sorted_and_total_matches(self):
        rows = hc.parse_trace(FIXTURE)
        summ = hc.canonical_summary(rows)
        self.assertEqual(summ, sorted(summ))
        self.assertEqual(sum(c for _l, _e, c in summ),
                         sum(len(r[2]) for r in rows))

    def test_summary_is_pure_of_row_order(self):
        import random
        rows = hc.parse_trace(FIXTURE)
        shuffled = list(rows)
        random.Random(7).shuffle(shuffled)
        self.assertEqual(hc.canonical_summary(rows), hc.canonical_summary(shuffled))

    def test_summary_text_has_four_column_free_header_and_total(self):
        rows = hc.parse_trace(FIXTURE)
        text = hc.write_summary(rows, None, {"card": "none", "device": "cpu"})
        self.assertIn("layer,expert,count", text)
        self.assertIn("# total,", text)
        self.assertIn("# card=none", text)
        self.assertNotIn("weight_offset", text)


class TestSelection(unittest.TestCase):
    def test_budget_is_exact_per_layer(self):
        rows = hc.parse_trace(FIXTURE)
        hot = hc.select_hot_set(rows, 10)
        self.assertEqual(sorted(hot), list(range(48)))
        for lay, es in hot.items():
            self.assertEqual(len(es), 10)
            self.assertEqual(len(set(es)), 10)                     # distinct

    def test_zero_budget_is_empty(self):
        rows = hc.parse_trace(FIXTURE)
        hot = hc.select_hot_set(rows, 0)
        self.assertTrue(all(v == [] for v in hot.values()))

    def test_tie_break_is_ascending_expert_id(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            # expert 7 and expert 3 both routed once -> 3 must win
            write_trace(p, [(0, 0, [7]), (1, 0, [3])])
            hot = hc.select_hot_set(hc.parse_trace(p), 1)
            self.assertEqual(hot[0], [3])

    def test_clearly_hot_expert_beats_barely_routed(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            rows = [(t, 0, [5] if t < 9 else [9]) for t in range(10)]
            write_trace(p, rows)
            hot = hc.select_hot_set(hc.parse_trace(p), 1)
            self.assertEqual(hot[0], [5])

    def test_selection_is_deterministic(self):
        rows = hc.parse_trace(FIXTURE)
        self.assertEqual(hc.select_hot_set(rows, 17), hc.select_hot_set(rows, 17))


class TestCoverage(unittest.TestCase):
    def test_hot_set_cover_exceeds_random_baseline(self):
        rows = hc.parse_trace(FIXTURE)
        for s in (1, 10, 50):
            hot = hc.select_hot_set(rows, s)
            cov = hc.hot_coverage(rows, hot)
            # a random set of s of 512 experts expects s/512 coverage
            self.assertGreater(cov, s / 512.0)

    def test_full_budget_covers_everything(self):
        rows = hc.parse_trace(FIXTURE)
        hot = hc.select_hot_set(rows, 512)
        self.assertAlmostEqual(hc.hot_coverage(rows, hot), 1.0)


class TestPlateau(unittest.TestCase):
    def test_stable_ranking_plateaus_early(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            rows = [(t, 0, [5]) for t in range(16)]                 # always expert 5
            write_trace(p, rows)
            rep = hc.rounds_to_plateau(hc.parse_trace(p), 1)
            self.assertTrue(rep["plateau"])
            self.assertEqual(rep["rounds"], 1)

    def test_never_stabilising_ranking_has_no_plateau(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            # each power-of-two prefix has a NEW argmax, so no two consecutive
            # prefixes are unchanged -> no plateau (clause V3's failing shape)
            rows = [(0, 0, [6]), (1, 0, [5])]
            rows += [(t, 0, [4]) for t in range(2, 4)]
            rows += [(t, 0, [3]) for t in range(4, 8)]
            rows += [(t, 0, [2]) for t in range(8, 16)]
            rows += [(t, 0, [1]) for t in range(16, 32)]
            write_trace(p, rows)
            rep = hc.rounds_to_plateau(hc.parse_trace(p), 1)
            self.assertFalse(rep["plateau"])
            self.assertIsNone(rep["rounds"])
            self.assertEqual([s["changed"] for s in rep["series"][1:]],
                             [True] * (len(rep["series"]) - 1))

    def test_stable_window_below_two_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            write_trace(p, [(t, 0, [5]) for t in range(4)])
            with self.assertRaises(ValueError):
                hc.rounds_to_plateau(hc.parse_trace(p), 1, stable_window=1)

    def test_series_is_monotone_in_prefix_length(self):
        rows = hc.parse_trace(FIXTURE)
        rep = hc.rounds_to_plateau(rows, 10)
        lens = rep["prefix_tokens"]
        self.assertEqual(lens, sorted(lens))
        self.assertEqual(lens, sorted(set(lens)))
        self.assertEqual(lens[-1], 400)


class TestPluginCsv(unittest.TestCase):
    def _write(self, path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_four_column_csv_parses_with_total(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write(os.path.join(d, "h.csv"),
                            "layer,weight_offset,expert,count\n"
                            "0,100,3,2\n0,100,7,1\n1,200,2,4\n# total,7\n")
            rows = hc.read_plugin_csv(p)
            self.assertEqual(rows, [(0, 100, 3, 2), (0, 100, 7, 1), (1, 200, 2, 4)])

    def test_three_column_row_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write(os.path.join(d, "h.csv"),
                            "layer,expert,count\n0,3,2\n")
            with self.assertRaises(ValueError):
                hc.read_plugin_csv(p)

    def test_wrong_total_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write(os.path.join(d, "h.csv"),
                            "layer,weight_offset,expert,count\n0,100,3,2\n# total,99\n")
            with self.assertRaises(ValueError):
                hc.read_plugin_csv(p)

    def test_join_matches_when_counts_agree(self):
        rows = hc.parse_trace(FIXTURE)
        # build a plugin CSV in weight-offset space from the same trace
        layer_keys = {lay: 1000 + lay for lay in range(48)}
        lines = ["layer,weight_offset,expert,count"]
        total = 0
        for lay, e, c in hc.canonical_summary(rows):
            off = layer_keys[lay]
            lines.append(f"{lay},{off},{e},{c}")
            total += c
        lines.append(f"# total,{total}")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "h.csv")
            with open(p, "w") as f:
                f.write("\n".join(lines) + "\n")
            plugin = hc.read_plugin_csv(p)
        _p, _t, mism = hc.join_plugin_to_trace(rows, plugin, layer_keys)
        self.assertEqual(mism, [])

    def test_join_mismatch_fires_on_perturbed_count(self):
        rows = hc.parse_trace(FIXTURE)
        layer_keys = {lay: 1000 + lay for lay in range(48)}
        plugin = [(lay, layer_keys[lay], e, c + 1)
                  for lay, e, c in hc.canonical_summary(rows)]   # every count +1
        _p, _t, mism = hc.join_plugin_to_trace(rows, plugin, layer_keys)
        self.assertTrue(mism)
        self.assertTrue(all(a - b == 1 for _k, a, b in mism))

    def test_seed_text_carries_layer_key_map(self):
        hot = {0: [3, 7], 1: [2]}
        text = hc.seed_text(hot, {0: 1000, 1: 1001})
        self.assertIn("# space=layer_key", text)
        self.assertIn("# layer_key_by_index", text)
        # one line per layer, keyed by the structural layer_key, not the
        # decoder index -- v2's whole point over the ambiguous v1 form.
        self.assertIn("1000 3 7", text)
        self.assertIn("1001 2", text)
        self.assertNotIn("\n0 3 7", text)

    def test_seed_text_declares_decoder_space_without_the_map(self):
        text = hc.seed_text({0: [3, 7], 1: [2]})
        self.assertIn("# space=layer", text)
        self.assertIn("0 3 7", text)
        self.assertIn("1 2", text)
        self.assertNotIn("space=layer_key", text)

    def test_seed_text_refuses_a_layer_missing_from_the_map(self):
        # an incomplete map is a mismatch, not a partial seed: refused.
        with self.assertRaises(ValueError):
            hc.seed_text({0: [3], 1: [2]}, {0: 1000})

    def test_seed_text_refuses_a_duplicate_layer_key(self):
        # two decoder layers mapping to one layer_key would collide in the
        # plugin's layer_key-keyed map: refused, not written.
        with self.assertRaises(ValueError):
            hc.seed_text({0: [3], 1: [2]}, {0: 1000, 1: 1000})

    def test_census_summary_parses_and_sums(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.csv")
            with open(p, "w") as f:
                f.write("# census summary\nlayer,expert,count\n0,3,5\n0,7,2\n1,2,9\n# total,16\n")
            counts = hc.read_census_summary(p)
        self.assertEqual(counts, [(0, 3, 5), (0, 7, 2), (1, 2, 9)])

    def test_census_summary_refuses_a_wrong_total(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.csv")
            with open(p, "w") as f:
                f.write("layer,expert,count\n0,3,5\n# total,99\n")
            with self.assertRaises(ValueError):
                hc.read_census_summary(p)

    def test_census_summary_refuses_a_four_column_row(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.csv")
            with open(p, "w") as f:
                f.write("layer,expert,count\n0,3,5,1\n")
            with self.assertRaises(ValueError):
                hc.read_census_summary(p)

    def test_select_from_counts_ranks_count_desc_id_asc(self):
        counts = [(0, 9, 3), (0, 4, 3), (0, 7, 5), (1, 2, 1)]
        hot = hc.select_hot_set_from_counts(counts, 2)
        self.assertEqual(hot, {0: [7, 4], 1: [2]})
        # ties broken by ascending id, not by file order
        self.assertLess(hot[0][1], hot[0][2] if len(hot[0]) > 2 else 99)

    def test_select_from_counts_zero_slots_is_empty_not_error(self):
        self.assertEqual(hc.select_hot_set_from_counts([(0, 3, 1)], 0), {0: []})

    def test_select_cli_refuses_trace_and_census_together(self):
        with self.assertRaises(ValueError):
            hc.main(["select", "--trace", "x", "--census", "y",
                     "--slots-per-layer", "1"])

    def test_select_cli_emits_a_layer_key_seed_from_a_census(self):
        with tempfile.TemporaryDirectory() as d:
            csv = os.path.join(d, "c.csv")
            mapf = os.path.join(d, "map.json")
            out = os.path.join(d, "seed.txt")
            with open(csv, "w") as f:
                f.write("layer,expert,count\n0,3,5\n0,7,2\n")
            with open(mapf, "w") as f:
                f.write('{"0": 704}')
            rc = hc.main(["select", "--census", csv, "--layer-keys", mapf,
                          "--slots-per-layer", "2", "--out", out])
            self.assertEqual(rc, 0)
            with open(out) as f:
                text = f.read()
        self.assertIn("# space=layer_key", text)
        self.assertIn("704 3 7", text)


class TestRouterTraceWriter(unittest.TestCase):
    def test_format_v1_token_major_and_ascending(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.trace")
            by_layer = {0: [[9, 3], [4, 1]],
                        1: [[2, 7], [5, 6]]}
            hc.write_router_trace(p, by_layer, 2, {"card": "none", "device": "cpu"})
            rows = hc.parse_trace(p)
            self.assertEqual(rows, [(0, 0, [3, 9]), (0, 1, [2, 7]),
                                    (1, 0, [1, 4]), (1, 1, [5, 6])])
            prov = hc.read_provenance(p)
            self.assertEqual(prov.get("card"), "none")
            self.assertEqual(prov.get("device"), "cpu")

    def test_missing_layer_row_for_a_short_array_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.trace")
            # layer 1 has only one token; the token-2 row must be absent, not
            # an IndexError, and the trace must still parse
            hc.write_router_trace(p, {0: [[1]] * 5, 1: [[2]]}, 5)
            rows = hc.parse_trace(p)
            self.assertEqual(len(rows), 6)          # 5 x layer0 + 1 x layer1


class TestCallTrace(unittest.TestCase):
    def test_split_two_tokens_into_chunks(self):
        self.assertEqual(hc.split_topk_chunks([1, 2, 3, 4], 2), [[1, 2], [3, 4]])
        self.assertEqual(hc.split_topk_chunks([1, 2, 3, 4, 5, 6], 3),
                         [[1, 2, 3], [4, 5, 6]])

    def test_split_missized_is_refused_not_truncated(self):
        with self.assertRaises(ValueError):
            hc.split_topk_chunks([1, 2, 3], 2)
        with self.assertRaises(ValueError):
            hc.split_topk_chunks([], 2)

    def test_split_zero_top_k_is_refused(self):
        with self.assertRaises(ValueError):
            hc.split_topk_chunks([1, 2], 0)

    def test_layer_key_map_is_ascending_export_order(self):
        self.assertEqual(hc.layer_key_index_map([300, 100, 200]),
                         {100: 0, 200: 1, 300: 2})

    def test_explicit_map_non_injective_is_refused(self):
        with self.assertRaises(ValueError):
            hc.layer_key_index_map([100, 200], {0: 100, 1: 200, 2: 200})

    def test_explicit_map_missing_key_is_refused(self):
        with self.assertRaises(ValueError):
            hc.layer_key_index_map([100, 200, 300], {0: 100, 1: 200})

    def test_decode_calls_reconstruct_token_major_v1(self):
        call_rows = [(0, 100, 2, [3, 4]), (1, 200, 2, [5, 6]),
                     (2, 100, 2, [7, 8]), (3, 200, 2, [9, 10])]
        rows, rep = hc.call_trace_to_v1(call_rows)
        self.assertEqual(rows, [(0, 0, [3, 4]), (0, 1, [5, 6]),
                                (1, 0, [7, 8]), (1, 1, [9, 10])])
        self.assertEqual(rep["tokens"], 2)
        self.assertEqual(rep["layer_keys"], 2)
        self.assertNotIn("prefill", rep)

    def test_batched_call_is_refused(self):
        with self.assertRaises(ValueError):
            hc.call_trace_to_v1([(0, 100, 2, [1, 2, 3, 4])])

    def test_skip_batched_keeps_decode_rows_and_counts_the_skip(self):
        # a served trace opens with a batched prefill call: skipped, counted,
        # and the decode rows still convert frame-for-frame. As modelled here
        # the prefill is ONE call, so the decode stream resumes mid-sequence:
        # token 0 = seq 1 (layer 200) + seq 2 (layer 100), token 1 = seq 3
        # (layer 200) ONLY -- token 1 is the partial one. A real depth-48
        # prefill emits one batched call per layer; skipped, they leave the
        # decode stream at layer 0.
        call_rows = [(0, 100, 2, [1, 2, 3, 4]),          # prefill, 2 tokens
                     (1, 200, 2, [5, 6]),
                     (2, 100, 2, [7, 8]),
                     (3, 200, 2, [9, 10])]
        rows, rep = hc.call_trace_to_v1(call_rows, skip_batched=True)
        self.assertEqual(rows, [(0, 0, [7, 8]), (0, 1, [5, 6]), (1, 1, [9, 10])])
        self.assertEqual(rep["batched_calls_skipped"], 1)
        self.assertEqual(rep["batched_tokens_skipped"], 2)

    def test_all_batched_is_refused_even_with_skip_batched(self):
        # the skip must not turn a prefill-only trace into an empty census.
        with self.assertRaises(ValueError):
            hc.call_trace_to_v1([(0, 100, 2, [1, 2, 3, 4])], skip_batched=True)

    def test_skip_batched_default_is_off(self):
        # the name says the DEFAULT is off, so pin the default itself: the
        # empty-list count would be zero whichever way it were set.
        sig = inspect.signature(hc.call_trace_to_v1)
        self.assertIs(sig.parameters["skip_batched"].default, False)

    def test_empty_call_trace_reports_zero_tokens(self):
        rows, rep = hc.call_trace_to_v1([])
        self.assertEqual(rows, [])
        self.assertEqual(rep["tokens"], 0)

    def test_call_trace_parse_refuses_negative(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.trace")
            with open(p, "w") as f:
                f.write("0 -5 2 3 4\n")
            with self.assertRaises(ValueError):
                hc.parse_call_trace(p)

    def test_call_trace_parse_refuses_missing_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.trace")
            with open(p, "w") as f:
                f.write("0 100 2\n")          # top_k declared, no ids
            with self.assertRaises(ValueError):
                hc.parse_call_trace(p)

    def test_census_range_selects_one_regime(self):
        # a REGIME is a half-open call-seq range: floor 2, ceiling 5 selects
        # calls 2, 3 and 4 only -- the mechanism the prefill/decode split needs.
        calls = [(i, 100, 2, [i, i + 1]) for i in range(10)]
        summary, meta = hc.census_from_call_trace(calls, None, 2, 5)
        self.assertEqual(meta["calls_selected"], 3)
        self.assertEqual(meta["to_call_seq"], 5)
        self.assertEqual(sum(c for _l, _e, c in summary), 6)

    def test_census_range_without_a_ceiling_still_counts_the_tail(self):
        calls = [(i, 100, 2, [1, 2]) for i in range(5)]
        _s, meta = hc.census_from_call_trace(calls, None, 2, None)
        self.assertEqual(meta["calls_selected"], 3)

    def test_census_inverted_range_is_refused(self):
        calls = [(i, 100, 2, [1, 2]) for i in range(5)]
        with self.assertRaises(ValueError):
            hc.census_from_call_trace(calls, None, 4, 2)

    def test_census_range_selecting_none_is_refused(self):
        calls = [(i, 100, 2, [1, 2]) for i in range(5)]
        with self.assertRaises(ValueError):
            hc.census_from_call_trace(calls, None, 9, 12)

    def test_join_honours_the_same_regime_range(self):
        # the join must be takable over one regime, so the ceiling applies there
        # too; an inverted range is refused exactly as in the census.
        calls = [(i, 100, 2, [1, 2]) for i in range(6)]
        plugin = [(0, 100, 1, 4), (0, 100, 2, 2)]
        _p, trace, _m = hc.join_plugin_to_call_trace(calls, plugin, 2, 5)
        self.assertEqual(sum(trace.values()), 6)
        with self.assertRaises(ValueError):
            hc.join_plugin_to_call_trace(calls, plugin, 4, 2)

    def test_from_call_trace_cli_writes_format_v1(self):
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            out = os.path.join(d, "v1.trace")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 3 4\n1 200 2 5 6\n2 100 2 7 8\n3 200 2 9 10\n")
            with open(prov, "w") as f:
                f.write("# artifact=d48n card=A770 kv=u8\n")
            rc = hc.main(["from-call-trace", "--call-trace", calls,
                          "--provenance", prov, "--out", out])
            self.assertEqual(rc, 0)
            rows = hc.parse_trace(out)
            self.assertEqual(rows, [(0, 0, [3, 4]), (0, 1, [5, 6]),
                                    (1, 0, [7, 8]), (1, 1, [9, 10])])
            prov_out = hc.read_provenance(out)
            self.assertEqual(prov_out["artifact"], "d48n")
            self.assertEqual(prov_out["card"], "A770")

    def test_from_call_trace_cli_refuses_missing_provenance(self):
        # an empty provenance file carries neither artifact= nor card=.
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 3 4\n")
            with open(prov, "w") as f:
                f.write("")
            with self.assertRaises(ValueError):
                hc.main(["from-call-trace", "--call-trace", calls,
                         "--provenance", prov])

    def test_from_call_trace_cli_refuses_incomplete_provenance(self):
        # artifact without card is still unattributable: refused, not written.
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 3 4\n")
            with open(prov, "w") as f:
                f.write("# artifact=d48n\n")
            with self.assertRaises(ValueError):
                hc.main(["from-call-trace", "--call-trace", calls,
                         "--provenance", prov])

    def test_from_call_trace_cli_refuses_empty_provenance_values(self):
        # an empty value is not an attribution: refused, not written.
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 3 4\n")
            with open(prov, "w") as f:
                f.write("# artifact_sha256= card=\n")
            with self.assertRaises(ValueError):
                hc.main(["from-call-trace", "--call-trace", calls,
                         "--provenance", prov])

    def test_from_call_trace_cli_accepts_the_artifact_sha256_spelling(self):
        # §2's canonical key is artifact_sha256=; both spellings are accepted
        # and the emitted header carries the one that was injected.
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            out = os.path.join(d, "v1.trace")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 3 4\n1 200 2 5 6\n")
            with open(prov, "w") as f:
                f.write("# artifact_sha256=deadbeef card=8086:56A0\n")
            rc = hc.main(["from-call-trace", "--call-trace", calls,
                          "--provenance", prov, "--out", out])
            self.assertEqual(rc, 0)
            prov_out = hc.read_provenance(out)
            self.assertEqual(prov_out["artifact_sha256"], "deadbeef")
            self.assertEqual(prov_out["token_labels"], "reconstructed")

    def test_from_call_trace_cli_stdout_is_a_valid_v1_trace(self):
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 3 4\n1 200 2 5 6\n")
            with open(prov, "w") as f:
                f.write("# artifact=d48n card=A770\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = hc.main(["from-call-trace", "--call-trace", calls,
                              "--provenance", prov])
            self.assertEqual(rc, 0)
            out = os.path.join(d, "stdout.trace")
            with open(out, "w") as f:
                f.write(buf.getvalue())
            self.assertEqual(hc.read_provenance(out)["card"], "A770")
            self.assertEqual(len(hc.parse_trace(out)), 2)

    def test_from_call_trace_cli_skip_batched_reports_the_skip(self):
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            out = os.path.join(d, "v1.trace")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 1 2 3 4\n1 200 2 5 6\n2 100 2 7 8\n3 200 2 9 10\n")
            with open(prov, "w") as f:
                f.write("# artifact=d48n card=A770\n")
            rc = hc.main(["from-call-trace", "--call-trace", calls,
                          "--provenance", prov, "--skip-batched", "--out", out])
            self.assertEqual(rc, 0)
            prov_out = hc.read_provenance(out)
            self.assertEqual(prov_out["batched_calls_skipped"], "1")
            self.assertEqual(prov_out["batched_tokens_skipped"], "2")
            self.assertEqual(len(hc.parse_trace(out)), 3)


class TestCallTraceCensus(unittest.TestCase):
    """The aggregate census is derived from the CALL TRACE, not the
    decode-only v1 rows: batched prefill ids count, and the load-time probe
    calls are excluded by a call_seq floor."""

    ROWS = [(0, 100, 2, [1, 2, 3, 4]),   # batched prefill, 2 tokens
            (1, 100, 2, [5, 6]),         # decode
            (2, 200, 2, [7, 8])]         # decode

    def test_census_counts_every_batched_id(self):
        # A census that kept only the first chunk of the batched call (the
        # `call_trace_to_v1` behaviour) would count 6 accesses, not 8; ids 3
        # and 4 are the second chunk and MUST be present.
        summary, meta = hc.census_from_call_trace(self.ROWS)
        self.assertEqual(meta["accesses"], 8)
        self.assertEqual(meta["tokens_observed"], 4)
        self.assertEqual(meta["batched_calls"], 1)
        self.assertEqual(meta["decode_calls"], 2)
        counts = {(lay, e): c for lay, e, c in summary}
        self.assertEqual(counts[(0, 3)], 1)
        self.assertEqual(counts[(0, 4)], 1)
        self.assertEqual(counts[(0, 5)], 1)
        self.assertEqual(counts[(0, 6)], 1)
        # layer 200 is decoder index 1 (export order of the observed keys)
        self.assertEqual(counts[(1, 7)], 1)
        self.assertEqual(counts[(1, 8)], 1)

    def test_census_from_seq_excludes_the_probe_calls(self):
        # call_seq 0 is the load-time probe call; from_seq=1 drops it, and the
        # dropped ids must be ABSENT from the summary -- a share the corpus
        # census must not inherit.
        summary, meta = hc.census_from_call_trace(self.ROWS, from_seq=1)
        self.assertEqual(meta["calls"], 3)
        self.assertEqual(meta["calls_selected"], 2)
        self.assertEqual(meta["accesses"], 4)
        self.assertEqual(meta["batched_calls"], 0)
        ids = {e for _l, e, _c in summary}
        self.assertEqual(ids, {5, 6, 7, 8})

    def test_census_and_v1_summary_diverge_when_a_batched_call_is_present(self):
        # The census counts the batched call's ids 1..4; the decode-only v1
        # summary (skip_batched) does not. An implementation that derived the
        # census from the v1 rows would wrongly AGREE here, so this cell is
        # red against exactly that implementation.
        summary, _meta = hc.census_from_call_trace(self.ROWS)
        v1, _rep = hc.call_trace_to_v1(self.ROWS, skip_batched=True)
        self.assertNotEqual(summary, hc.canonical_summary(v1))
        self.assertLess(sum(c for _l, _e, c in hc.canonical_summary(v1)),
                        sum(c for _l, _e, c in summary))

    def test_census_floor_beyond_all_calls_is_refused_not_written_empty(self):
        # A one-too-high harness call_seq floor must fail loudly, not write a
        # `# total,0` "census" -- the same rule the converter applies to an
        # all-batched trace.
        with self.assertRaises(ValueError):
            hc.census_from_call_trace(self.ROWS, from_seq=99)

    def test_census_on_an_empty_call_trace_is_refused(self):
        # An emitter that wrote nothing is not a census; refuse rather than
        # exit 0 with `# total,0`.
        with self.assertRaises(ValueError):
            hc.census_from_call_trace([])

    def test_join_plugin_to_call_trace_matches_on_layer_key(self):
        plugin_rows = [(0, 100, 1, 1), (0, 100, 2, 1), (0, 100, 3, 1),
                       (0, 100, 4, 1), (0, 100, 5, 1), (0, 100, 6, 1),
                       (1, 200, 7, 1), (1, 200, 8, 1)]
        _p, _t, mismatches = hc.join_plugin_to_call_trace(self.ROWS, plugin_rows)
        self.assertEqual(mismatches, [])

    def test_join_plugin_to_call_trace_fires_on_one_wrong_count(self):
        plugin_rows = [(0, 100, 1, 1), (0, 100, 2, 1), (0, 100, 3, 1),
                       (0, 100, 4, 1), (0, 100, 5, 1), (0, 100, 6, 1),
                       (1, 200, 7, 1), (1, 200, 8, 2)]   # perturbed
        _p, _t, mismatches = hc.join_plugin_to_call_trace(self.ROWS, plugin_rows)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0][0], (200, 8))

    def test_join_plugin_to_call_trace_from_seq_shows_the_probe_delta(self):
        # With the corpus floor on, the probe-only ids are on the CSV side
        # only: the mismatch is exactly the probe's share, by key.
        plugin_rows = [(0, 100, 1, 1), (0, 100, 2, 1), (0, 100, 3, 1),
                       (0, 100, 4, 1), (0, 100, 5, 1), (0, 100, 6, 1),
                       (1, 200, 7, 1), (1, 200, 8, 1)]
        _p, _t, mismatches = hc.join_plugin_to_call_trace(
            self.ROWS, plugin_rows, from_seq=1)
        self.assertEqual(sorted(k for k, _pc, _tc in mismatches),
                         [(100, 1), (100, 2), (100, 3), (100, 4)])

    def test_census_from_call_trace_cli_writes_a_summary(self):
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            out = os.path.join(d, "census.csv")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 1 2 3 4\n1 100 2 5 6\n2 200 2 7 8\n")
            with open(prov, "w") as f:
                f.write("# artifact=d48n card=A770 kv=u8\n")
            rc = hc.main(["census-from-call-trace", "--call-trace", calls,
                          "--provenance", prov, "--out", out])
            self.assertEqual(rc, 0)
            text = open(out).read()
            self.assertIn("layer,expert,count", text)
            self.assertIn("# total,8", text)
            self.assertIn("# batched_calls=1", text)
            self.assertIn("# calls_selected=3", text)

    def test_census_from_call_trace_cli_refuses_missing_attribution(self):
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 3 4\n")
            with open(prov, "w") as f:
                f.write("# artifact=d48n\n")          # no card=
            with self.assertRaises(ValueError):
                hc.main(["census-from-call-trace", "--call-trace", calls,
                         "--provenance", prov])

    def test_from_call_trace_cli_from_call_seq_drops_a_pre_floor_decode_call(self):
        # The pre-floor call (seq 0) is a DECODE call, so --skip-batched does
        # not remove it; only the floor does. An implementation that ignored
        # from_seq in call_trace_to_v1 would emit three rows here, not two.
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            out = os.path.join(d, "v1.trace")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 1 2\n1 100 2 3 4 5 6\n2 100 2 7 8\n3 200 2 9 10\n")
            with open(prov, "w") as f:
                f.write("# artifact=d48n card=A770\n")
            rc = hc.main(["from-call-trace", "--call-trace", calls,
                          "--provenance", prov, "--skip-batched",
                          "--from-call-seq", "1", "--out", out])
            self.assertEqual(rc, 0)
            self.assertEqual(hc.parse_trace(out), [(0, 0, [7, 8]), (0, 1, [9, 10])])
            self.assertEqual(hc.read_provenance(out)["from_call_seq"], "1")

    def test_census_from_call_trace_cli_from_call_seq_reports_the_floor(self):
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            out = os.path.join(d, "census.csv")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 1 2 3 4\n1 100 2 5 6\n2 200 2 7 8\n")
            with open(prov, "w") as f:
                f.write("# artifact=d48n card=A770\n")
            rc = hc.main(["census-from-call-trace", "--call-trace", calls,
                          "--provenance", prov, "--from-call-seq", "1",
                          "--out", out])
            self.assertEqual(rc, 0)
            text = open(out).read()
            self.assertIn("# total,4", text)
            self.assertIn("# calls_selected=2", text)
            self.assertIn("# from_call_seq=1", text)
            self.assertNotIn("0,3,1", text)          # the probe ids are gone

    def test_join_plugin_call_trace_cli_reports_zero_mismatches(self):
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            csv = os.path.join(d, "hist.csv")
            with open(calls, "w") as f:
                f.write("0 100 2 1 2 3 4\n1 100 2 5 6\n2 200 2 7 8\n")
            with open(csv, "w") as f:
                f.write("layer,weight_offset,expert,count\n")
                for e in (1, 2, 3, 4, 5, 6):
                    f.write(f"0,100,{e},1\n")
                f.write("1,200,7,1\n1,200,8,1\n# total,8\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = hc.main(["join-plugin-call-trace", "--call-trace", calls,
                              "--csv", csv])
            self.assertEqual(rc, 0)
            self.assertIn("mismatches=0", buf.getvalue())

    def test_join_plugin_call_trace_cli_flags_the_floor_delta(self):
        # With the corpus floor on, the probe ids sit only on the CSV side:
        # the CLI must report a nonzero mismatch count, not a silent agree.
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            csv = os.path.join(d, "hist.csv")
            with open(calls, "w") as f:
                f.write("0 100 2 1 2 3 4\n1 100 2 5 6\n2 200 2 7 8\n")
            with open(csv, "w") as f:
                f.write("layer,weight_offset,expert,count\n")
                for e in (1, 2, 3, 4, 5, 6):
                    f.write(f"0,100,{e},1\n")
                f.write("1,200,7,1\n1,200,8,1\n# total,8\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = hc.main(["join-plugin-call-trace", "--call-trace", calls,
                              "--csv", csv, "--from-call-seq", "1"])
            self.assertEqual(rc, 0)
            self.assertIn("mismatches=4", buf.getvalue())


class TestCli(unittest.TestCase):
    def test_summary_subcommand_writes_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "s.csv")
            rc = hc.main(["summary", "--trace", FIXTURE, "--out", out])
            self.assertEqual(rc, 0)
            text = open(out).read()
            self.assertIn("layer,expert,count", text)
            self.assertIn("# total,", text)

    def test_select_subcommand_is_parseable(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "seed.txt")
            rc = hc.main(["select", "--trace", FIXTURE,
                          "--slots-per-layer", "4", "--out", out])
            self.assertEqual(rc, 0)
            text = open(out).read()
            self.assertIn("# space=layer", text)
            data = [l for l in open(out) if l.strip() and not l.startswith("#")]
            # v2: ONE line per layer, key + slots_per_layer experts
            self.assertEqual(len(data), 48)
            for line in data:
                self.assertEqual(len(line.split()), 1 + 4)


if __name__ == "__main__":
    unittest.main()
