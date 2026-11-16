"""Patches 0129x/0129y/0129z's device-free gate: the dead next-graph lead's cause,
its named early returns, and the verify-only request totals.

Window 292 (`measured-here`, 65k B60): OVERLAP+DEBUG printed 2,240 "commands (lead)"
lines and not one "next-graph lead" line; every request-end summary read "0 reused, 0
re-copied"; and the VERIFY=1 arm with OVERLAP unset printed no totals at all.

Red mutations, each run and seen failing before it was restored:
  (a) the tail call in the split-decode branch dropped (the `return` kept, the
      `ggml_cl_fa_paged_overlap_after` call removed): the branch cell fails -- that is
      window 292's code, and it is why the lead was dead;
  (b) the `runs < 0` early return made silent (its WARN removed): the named-fallbacks
      cell fails (CLAUDE.md: no silent fallbacks);
  (c) the roll-up gated on `overlap_on()` only (the verify switch dropped): the
      verify-only cell fails -- window 292's A-uverify arm.
Source-text assertions, not numerics.
"""
import os
import unittest
from pathlib import Path

OPENCL = "ggml/src/ggml-opencl/ggml-opencl.cpp"


def _llama_root():
    raw = os.environ.get("LLAMA_SRC")
    if not raw:
        raise unittest.SkipTest("LLAMA_SRC is unset: the patched llama tree is not available")
    root = Path(raw)
    for cand in (root, root.parent):
        if (cand / OPENCL).exists():
            return cand
    raise unittest.SkipTest("LLAMA_SRC has no ggml-opencl.cpp: it lacks patch 0119")


class FaPagedLead(unittest.TestCase):

    def setUp(self):
        self.root = _llama_root()
        self.t = {f: (self.root / f).read_text() for f in (OPENCL,)}

    def test_the_split_branch_calls_the_tail(self):
        t = self.t[OPENCL]
        i = t.find("this branch returns here, before the tail call below")
        self.assertTrue(i >= 0, "0129x's note is absent")
        j = t.rfind("const size_t merge_wg", 0, i)
        self.assertTrue(j >= 0)
        block = t[j:t.find("        return;\n", i)]
        self.assertIn("if (paged_overlap_copied && paged_overlap_parity >= 0) {", block,
                      "the split branch must call the tail before it returns")
        self.assertIn("ggml_cl_fa_paged_overlap_after(backend_ctx, paged_overlap_parity);", block)
        # the served decode takes this branch; the tail call below it is unreachable there
        self.assertIn("backend_ctx->enqueue_ndrange_kernel(k_merge, 3, merge_gws, merge_lws, dst);", block,
                      "the call must follow both kernels")

    def test_every_early_return_names_itself(self):
        t = self.t[OPENCL]
        i = t.find("static void ggml_cl_fa_paged_prefetch_head(ggml_backend_opencl_context * ctx) {")
        self.assertTrue(i >= 0)
        block = t[i:t.find("static bool ggml_cl_fa_paged_prefetch_same", i)]
        for reason in ("no paged FA in the graph's chain",
                       "ring slot is not cached",
                       "plan is not USM host memory",
                       "enqueue_pages -1"):
            self.assertIn(reason, block, f"the early return '{reason}' is silent")
        self.assertIn("bool pre_head_dead_warned = false;", t, "the WARNs are one-shot a process")
        self.assertIn("o.pre_head_dead_warned = true;", block)

    def test_the_summary_carries_issued_and_prints_for_verify_only(self):
        t = self.t[OPENCL]
        self.assertIn("next-graph lead: %d issued, %d reused, %d re-copied", t)
        self.assertIn("int issued = 0, fell_back = 0, reused = 0, recopied = 0, pre_issued = 0;", t)
        self.assertIn("g_fa_paged_totals.pre_issued += o.pre_issued;", t)
        i = t.find("if (ggml_cl_fa_paged_overlap_on() || ggml_cl_fa_paged_verify_on()) {")
        self.assertTrue(i >= 0, "0129z's roll-up gate is absent")
        block = t[i:t.find("backend_ctx->pair_perm_t = nullptr;", i)]
        self.assertIn("g_fa_paged_totals.v_nodes    += o.v_nodes;", block,
                      "the verify totals roll up without OVERLAP")
        self.assertIn("o.pre_issued = o.pre_reused = o.pre_recopied = 0;", block)
        self.assertIn("static bool ggml_cl_fa_paged_verify_on();   // 0129z", t,
                      "the roll-up needs the verify switch declared above graph_compute")


if __name__ == "__main__":
    unittest.main()
