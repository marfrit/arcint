"""Patch 0082's device-free cell: the joint q/k l2_norm and the bf16 gate dispatch.

The numeric half is tests/test_gdn_qk_l2_norm.cpp (CPU backend, both forms, byte
equality). This cell reads the patched tree and asserts the parts the C cell
cannot: the switch LLAMA_FUSE_QK_L2_NORM is OFF by default, the three served
archs build the joint form under it and keep the two separate norms otherwise,
the helper's joint view keeps the conv output's group stride and its k view
takes the head-group offset, and the second half of item 8 -- the bf16 48-row
gate matvec -- is already what the dispatcher gives: BF16 is stored as F16 on
device and the F16 nc/ksplit matvec takes 1-8 columns with <= 64 rows
(a work-group a row), so no kernel work was added for it (code; README 0082).

Red mutations, each run and seen failing before it was restored:
  (a) the switch's default flipped on: the default-off cell fails;
  (b) a model's else arm dropped (the two separate norms gone): the two-norms
      cell fails;
  (c) the k view's offset loses `num_k_heads *`: the head-group cell fails;
  (d) the joint view's group stride set contiguous: the stride cell fails;
  (e) the bf16 matvec condition narrowed (ne01 <= 64 -> 48): the dispatch cell
      fails.
Source-text assertions, not numerics.
"""
import os
import unittest
from pathlib import Path

HELPER = "src/models/gdn-qk-l2-norm.h"
MODELS_H = "src/models/models.h"
OPENCL = "ggml/src/ggml-opencl/ggml-opencl.cpp"
MODELS = ("src/models/qwen35.cpp", "src/models/qwen35moe.cpp", "src/models/qwen4exp.cpp")


def _llama_root():
    raw = os.environ.get("LLAMA_SRC")
    if not raw:
        raise unittest.SkipTest("LLAMA_SRC is unset: the patched llama tree is not available")
    root = Path(raw)
    for cand in (root, root.parent):
        if (cand / HELPER).exists():
            return cand
    raise unittest.SkipTest("LLAMA_SRC has no src/models/gdn-qk-l2-norm.h: it lacks patch 0082")


class GdnQkL2Norm(unittest.TestCase):

    def setUp(self):
        self.root = _llama_root()
        self.t = {f: (self.root / f).read_text() for f in (HELPER, MODELS_H, OPENCL) + MODELS}

    def test_switch_is_default_off(self):
        for f in MODELS:
            t = self.t[f]
            i = t.find('static const bool fuse_qk_l2 = [] {')
            self.assertTrue(i >= 0, f + ": the fusion switch is absent")
            j = t.find("}();", i)
            block = t[i:j]
            self.assertIn('getenv("LLAMA_FUSE_QK_L2_NORM")', block, f)
            self.assertIn("e != nullptr && atoi(e) != 0", block,
                          f + ": the fusion must be off unless LLAMA_FUSE_QK_L2_NORM is 1")
            self.assertIn("if (fuse_qk_l2) {", t, f)

    def test_two_separate_norms_are_kept(self):
        for f in MODELS:
            t = self.t[f]
            self.assertIn("q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);", t,
                          f + ": the switch-off arm must keep the two separate norms")
            self.assertIn("k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);", t, f)
        self.assertIn('#include "gdn-qk-l2-norm.h"', self.t[MODELS_H])

    def test_joint_view_keeps_the_group_stride(self):
        t = self.t[HELPER]
        i = t.find("ggml_view_4d(ctx, conv_qkv_mix, head_k_dim, 2 * num_k_heads")
        self.assertTrue(i >= 0, "the joint view is absent")
        block = t[i:t.find(");", i)]
        # the view's nb arguments: dim0 rows are contiguous at head_k_dim, dim1's
        # group stride is the conv output's row stride nb1_qkv, dim2's is nb1_qkv*T
        self.assertIn("ggml_row_size(conv_qkv_mix->type, head_k_dim)", block)
        self.assertIn("nb1_qkv,", block, "the joint view's nb[2] is the conv output's group stride")
        self.assertIn("nb1_qkv * n_seq_tokens", block)
        self.assertIn("qk = build_gdn_l2_norm(ctx, qk, eps);", t, "one norm over the joint view")

    def test_k_view_takes_the_head_group_offset(self):
        t = self.t[HELPER]
        self.assertIn("num_k_heads * nb1", t, "k starts at the second head group of the norm's result")
        self.assertIn("const size_t nb1 = qk->nb[1];", t, "q and k are views of the dense result")

    def test_bf16_gate_already_takes_a_matvec(self):
        t = self.t[OPENCL]
        self.assertIn("const enum ggml_type src0t = (src0->type == GGML_TYPE_BF16) ? GGML_TYPE_F16 : src0->type;", t,
                      "bf16 is stored as f16 on device")
        # the F16 nc/ksplit matvec: 1-8 columns, a work-group (or sub-group) a row
        self.assertIn("const bool f16_ksplit = backend_ctx->kernel_mul_mv_f16_f32_ksplit != nullptr && src0->ne[0] >= 1024 &&",
                      t)
        self.assertIn("src1->ne[1] <= 8 && src0->ne[0] % 8 == 0", t, "2-8 columns take it")
        self.assertIn("(src0->ne[1] <= 64 ||", t, "a 48-row weight is under the 64-row term")
        self.assertIn("kernel_mul_mv_f16_f32_ksplit", t)


if __name__ == "__main__":
    unittest.main()
