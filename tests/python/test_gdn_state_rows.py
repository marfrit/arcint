"""Patch 0081's device-free gate cell: the GDN state read straight from the cache.

The numeric half is tests/test_gdn_state_rows.cpp (the CPU backend, both forms,
byte equality). This cell reads the patched tree and asserts the parts the C
cell cannot: that the build gate is behind LLAMA_GDN_STATE_ROWS, OFF by default,
that a multi-sequence ubatch declines the fold, that the gate also requires the
fused GDN op the batch takes, that build_rs_cache_view gathers only the extra
rows and returns the cache itself for the live row (inp->s_copy_main), that the
three served archs pass that row index to the op, that the rows constructor sets
src[6] over a 2D cache view, that the CPU op reads state_rows[iv3] * the row
stride from nb[1], and that both OpenCL kernels read rows[seq] when use_rows is
1. The cards' own arm (the same comparison through the OpenCL kernel, on the
B60/A770 in a card window) is in the patch record.

Red mutations, each run and seen failing before it was restored (command in the
patch README section):
  (a) the switch's default flipped on: the default-off cell fails;
  (b) the `n_seqs != 1` decline dropped: the multi-sequence cell fails;
  (c) the fused-op term dropped: the gate cell fails;
  (d) build_rs_cache_view gathering the main rows again: the no-gather cell fails;
  (e) the CPU op's row index replaced by the sequence ordinal: the CPU cell fails;
  (f) the intel kernel's rows term dropped from state_base: the kernel cell fails;
  (g) the dispatch's use_rows forced 0: the dispatch cell fails.
These are source-text assertions (the cells read the patched tree), not
numerics. Standard library only (unittest), so `ctest -L unit` runs it on a
build host with python3 and nothing else.
"""
import os
import unittest
from pathlib import Path

SCRATCH = "src/llama-graph.cpp"
GRAPH_H = "src/llama-graph.h"
GGML_C = "ggml/src/ggml.c"
GGML_H = "ggml/include/ggml.h"
CPU_OPS = "ggml/src/ggml-cpu/ops.cpp"
OPENCL = "ggml/src/ggml-opencl/ggml-opencl.cpp"
INTEL_CL = "ggml/src/ggml-opencl/kernels/gated_delta_net_intel.cl"
GENERIC_CL = "ggml/src/ggml-opencl/kernels/gated_delta_net.cl"
MODELS = ("src/models/qwen35.cpp", "src/models/qwen35moe.cpp", "src/models/qwen4exp.cpp")


def _llama_root():
    raw = os.environ.get("LLAMA_SRC")
    if not raw:
        raise unittest.SkipTest("LLAMA_SRC is unset: the patched llama tree is not available")
    root = Path(raw)
    for cand in (root, root.parent):
        if (cand / SCRATCH).exists() and (cand / INTEL_CL).exists():
            return cand
    raise unittest.SkipTest("LLAMA_SRC has no patched llama tree: it lacks patch 0081")


def _body(text, marker, end="\n}\n"):
    i = text.find(marker)
    if i < 0:
        return None
    j = text.find(end, i)
    return text[i:j if j >= 0 else len(text)]


class GdnStateRows(unittest.TestCase):

    def setUp(self):
        self.root = _llama_root()
        self.t = {f: (self.root / f).read_text() for f in
                  (SCRATCH, GRAPH_H, GGML_C, GGML_H, CPU_OPS, OPENCL, INTEL_CL, GENERIC_CL) + MODELS}

    def test_gate_is_default_off(self):
        t = self.t[SCRATCH]
        self.assertIn('getenv("LLAMA_GDN_STATE_ROWS")', t)
        self.assertRegex(t, r"e\s*!=\s*nullptr\s*&&\s*atoi\(e\)\s*!=\s*0",
                         "the fold must be off unless LLAMA_GDN_STATE_ROWS is 1")
        self.assertIn("LLAMA_GDN_STATE_ROWS=1 (off by", self.t[GRAPH_H])

    def test_multi_sequence_declines(self):
        b = _body(self.t[SCRATCH], "bool llama_gdn_state_rows_enabled(")
        self.assertTrue(b, "the gate function is absent")
        self.assertTrue("if (n_seqs != 1)" in b and "return false;" in b,
                        "a multi-sequence ubatch must decline the fold")

    def test_gate_needs_the_fused_op(self):
        b = _body(self.t[SCRATCH], "bool llama_gdn_state_rows_enabled(")
        self.assertIn("n_seq_tokens == 1 ? cparams.fused_gdn_ar : cparams.fused_gdn_ch", b,
                      "the autoregressive path needs a 4D state, so the gate must take it out")

    def test_cache_view_has_no_main_gather(self):
        b = _body(self.t[SCRATCH], "llm_graph_context::build_rs_cache_view(")
        self.assertTrue(b, "build_rs_cache_view is absent")
        self.assertIn("inp->s_copy_tail", b, "the extra rows are gathered")
        self.assertNotIn("inp->s_copy,", b, "the live rows must not be gathered")
        self.assertIn("ggml_scale_inplace", b, "the zero state, as build_rs does")
        self.assertIn("(rs_head + n_seqs)*s->nb[1]", b, "the extra relocation, as build_rs does")
        self.assertIn("return states;", b, "the 2D cache view is returned")

    def test_models_pass_the_row_index(self):
        for f in MODELS:
            t = self.t[f]
            self.assertIn("llama_gdn_state_rows_enabled(cparams, n_seqs, n_seq_tokens)", t, f)
            self.assertIn("build_rs_cache_view(inp, ssm_states_all, hparams.n_embd_s(), n_seqs)", t, f)
            self.assertIn("state_rows = inp->s_copy_main;", t, f)
            self.assertRegex(t, r"build_recurrent_attn\(inp, ssm_states_all, q_conv, k_conv, v_conv, "
                                r"gate, beta, state, il, state_rows\);", f)

    def test_rows_op_over_a_2d_cache(self):
        t = self.t[GGML_C]
        b = _body(t, "struct ggml_tensor * ggml_gated_delta_net_rows(")
        self.assertTrue(b, "ggml_gated_delta_net_rows is absent")
        self.assertIn("result->src[6] = rows;", b)
        self.assertIn("states->ne[0] == S_v * S_v * H", b, "the cache rows are D wide")
        self.assertIn("rows->ne[0] == n_seqs", b)
        self.assertIn("ggml_gated_delta_net_rows(", self.t[GGML_H])

    def test_cpu_reads_the_named_row(self):
        t = self.t[CPU_OPS]
        self.assertIn("const ggml_tensor * src_rows = dst->src[6];", t)
        self.assertIn("const int64_t state_row_stride = src_rows ? (int64_t) (src_state->nb[1] / sizeof(float))",
                      t, "the row stride is the cache's nb[1]")
        self.assertIn("state_rows ? (int64_t) state_rows[iv3] * state_row_stride : iv3 * state_seq_stride",
                      t, "rows mode reads row state_rows[iv3]")

    def test_opencl_kernels_read_rows(self):
        for f in (INTEL_CL, GENERIC_CL):
            t = self.t[f]
            self.assertIn("rows_buf", t, f)
            self.assertIn("state_row_stride", t, f)
            self.assertIn("use_rows", t, f)
            self.assertRegex(t, r"use_rows\s*\n?\s*\?[^;]*rows_buf[^;]*state_row_stride[^;]*head",
                             f + ": state_base must take the row in rows mode")
        i = self.t[INTEL_CL]
        self.assertEqual(i.count("const ulong state_base = use_rows"), 2,
                         "both Intel kernels (tg and pp) read the row")

    def test_dispatch_passes_rows(self):
        t = self.t[OPENCL]
        self.assertIn("const ggml_tensor * src_rows = dst->src[6];", t)
        self.assertIn("const cl_uint  use_rows        = src_rows != nullptr ? 1 : 0;", t)
        self.assertRegex(t, r"src_state->nb\[1\] / sizeof\(float\)", "the row stride from nb[1]")


if __name__ == "__main__":
    unittest.main()
