"""Patches 0088/0089's device-free gate: the switch, the builds and the routing.

The numeric half is tests/test_fa_gqa_i8.cpp (the q8_0 block offsets, the RNE Q
quantizer, the S selftest against the f16 form), built against the kernel's own
.inc. This cell reads the patched llama tree and asserts the parts that cell
cannot: GGML_OPENCL_FA_GQA_I8 is OFF by default, the int8 helper text is
prepended and -DGQA_I8=1 passed only for a q8_0 K pair, the staged form
(-DGQA_ST=1) is a second build only when the switch is on and only for the KT 1
pairs, and the dispatch takes the staged kernel for 1-3 rows under
GGML_OPENCL_FA_GQA_MIN_ROWS and the direct kernel otherwise. The kernel's int8
branch is gated on `GQA_I8 && KT == 1`, the B operand is the raw block's codes
(read at the helper's offset), and P V stays f16 DPAS.

Red mutations, each run and seen failing before it was restored:
  (a) the switch's default flipped on (`fi8_env != nullptr` without atoi):
      the default-off cell fails;
  (b) the KT 1 guard dropped (`true` for the q8_0 term): the build cell fails;
  (c) the staged dispatch dropped (`n_q < 4` -> false): the routing cell fails;
  (d) fa_gqa_i8_qs reads 34 * b + 18 (the block's second half): the offset
      assertion fails here and the C++ cell's S selftest fails.
Source-text assertions, not numerics.
"""
import os
import unittest
from pathlib import Path

OPENCL = "ggml/src/ggml-opencl/ggml-opencl.cpp"
KERNEL = "ggml/src/ggml-opencl/kernels/flash_attn_gqa_dpas.cl"
OPS = "ggml/src/ggml-opencl/kernels/fa_gqa_i8_ops.inc"
BACKEND = "tests/test-backend-ops.cpp"


def _llama_root():
    raw = os.environ.get("LLAMA_SRC")
    if not raw:
        raise unittest.SkipTest("LLAMA_SRC is unset: the patched llama tree is not available")
    root = Path(raw)
    for cand in (root, root.parent):
        if (cand / OPS).exists():
            return cand
    raise unittest.SkipTest("LLAMA_SRC has no fa_gqa_i8_ops.inc: it lacks patch 0088")


class FaGqaI8(unittest.TestCase):

    def setUp(self):
        self.root = _llama_root()
        self.t = {f: (self.root / f).read_text() for f in (OPENCL, KERNEL, OPS, BACKEND)}

    def test_switch_is_default_off(self):
        t = self.t[OPENCL]
        i = t.find('const char * fi8_env = getenv("GGML_OPENCL_FA_GQA_I8");')
        self.assertTrue(i >= 0, "the switch is absent")
        block = t[i:t.find("\n", t.find("const bool   fi8_on", i))]
        self.assertIn("atoi(fi8_env) != 0", block,
                      "the switch must be off unless GGML_OPENCL_FA_GQA_I8 is a nonzero number")
        self.assertIn("const bool   fi8_on", block, "the switch is read once at init")
        # the int8 build is a separate arm; the f16 program's source and options
        # are untouched when the switch is off
        j = t.find("const bool i8_arm =")
        self.assertTrue(j >= 0, "the int8 arm is absent")
        arm = t[j:t.find(";", j)]
        self.assertIn("fi8_on", arm)
        self.assertIn("gkv[i][0] == 1", arm, "the int8 QK is built for a q8_0 K (KT 1) only")
        k = t.find("if (i8_arm && !gqa_i8_inc.empty())")
        self.assertTrue(k >= 0, "the -DGQA_I8 build option is not gated on the arm")
        self.assertIn('gqa_opts += " -DGQA_I8=1";', t)
        self.assertIn("gqa_prog_src = gqa_i8_inc + gqa_src;", t,
                      "the helper text is prepended only to an int8 arm")

    def test_staged_form_is_a_second_build_only_with_the_switch(self):
        t = self.t[OPENCL]
        i = t.find("if (fi8_on && !gqa_i8_inc.empty()) {")
        self.assertTrue(i >= 0, "the staged build is not gated on the switch")
        block = t[i:t.find("if (backend_ctx->kernel_fa_gqa_st_dk[0][1] == nullptr)", i)]
        self.assertIn("if (gkv[i][0] != 1) continue;", block, "only the q8_0 K pairs have a staged form")
        self.assertIn('" -DGQA_I8=1 -DGQA_ST=1"', block, "the staged form is GQA_ST")
        self.assertIn("kernel_fa_gqa_st_dk[di][i] = kg;", block)
        self.assertIn("cl_kernel kernel_fa_gqa_st_dk[2][4] = {};", t,
                      "the staged kernels have their own array, so switch-off leaves it empty")

    def test_routing_takes_the_staged_form_for_1_to_3_rows(self):
        t = self.t[OPENCL]
        i = t.find("bool use_fd_gqa = false;")
        self.assertTrue(i >= 0)
        block = t[i:t.find("    if (fd_k_split == NULL &&", i)]
        self.assertIn("n_q >= fg_min_rows", block, "GGML_OPENCL_FA_GQA_MIN_ROWS still routes")
        self.assertIn("(n_head / n_head_kv) * n_q <= 64", block, "the 8-tile cap")
        self.assertIn("n_q < 4", block, "1-3 rows take the staged form")
        self.assertIn("kernel_fa_gqa_st_dk[fa_dki][gi]", block)
        self.assertIn("if (kg == nullptr && gi >= 0) kg = backend_ctx->kernel_fa_gqa_dk[fa_dki][gi];", block,
                      "a missing staged kernel falls back to the direct one")

    def test_kernel_int8_branch_and_operands(self):
        k = self.t[KERNEL]
        self.assertIn("#if GQA_I8 && KT == 1", k, "the int8 branch is compile-time and q8_0 K only")
        self.assertIn("intel_sub_group_i8_i8_matrix_mad_k32(a, b, (int8)(0))", k,
                      "the int8 DPAS is the public s8 x s8 k32 builtin")
        self.assertIn("fa_gqa_i8_quant32(x, &dq, q);", k, "Q is quantized once a call")
        self.assertIn("intel_sub_group_block_read_us8(", k,
                      "the s8 x s8 a operand is the strided block read over a row-major tile")
        self.assertIn("cl_intel_subgroup_local_block_io", k,
                      "the local-memory block read's extension is enabled")
        self.assertIn("fa_gqa_i8_blk(b)", k, "the B operand is read at the helper's block offset")
        self.assertIn("fa_gqa_i8_b32_g(bp + 2)", k, "the raw codes start two bytes into the block")
        self.assertIn("acc += convert_float8(DPAS8(av, bv))", k, "the integer S is scaled in float")
        # P V stays the f16 DPAS: the P V loop calls DPAS(, not DPAS8(
        pv = k[k.find("for (int i = 0; i < NDH; ++i) {"):]
        self.assertIn("acc = DPAS(pa[g], b, acc);", pv, "P V must stay f16 DPAS")
        self.assertIn("#if GQA_ST && GQA_I8 && KT == 1", k, "the staged raw K copy is 0089's")
        self.assertIn("ks8[key * KSTR + d] = ((const global uint *)(kb + kk * k_nb1))[d];", k,
                      "the staged copy is coalesced dwords of the raw row")

    def test_ops_arithmetic(self):
        o = self.t[OPS]
        self.assertIn("static inline int fa_gqa_i8_blk(int b) { return 34 * b; }", o,
                      "a q8_0 block is 34 bytes")
        self.assertIn("static inline int fa_gqa_i8_qs(int b) { return 34 * b + 2; }", o,
                      "the codes sit behind the half scale")
        self.assertIn("static inline int fa_gqa_i8_half(int lane) { return 16 + lane; }", o,
                      "dim l+16 is the block's second half")
        self.assertIn("return (dk / 32) * 34 / 4;", o, "a q8_0 row in dwords")
        self.assertIn("FA_GQA_I8_RINT(x)", o, "the code is RNE")
        self.assertIn("const float d    = amax / 127.f;", o, "one scale per row per 32-block")

    def test_card_cases_at_both_geometries(self):
        b = self.t[BACKEND]
        i = b.find("0088/0089: the int8 Q K^T")
        self.assertTrue(i >= 0, "the new eval cases are absent")
        block = b[i:b.find("\n    }\n", i)]
        self.assertIn("{{4, 6}, {2, 8}}", block, "the agent's 24 on 4 and the coder's 16 on 2")
        self.assertIn("{{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0}, {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}}", block)
        self.assertIn("for (int64_t nb : {1, 2, 3, 4, 6})", block)
        self.assertIn("for (int64_t kv : {1013, 4096, 131072})", block)


if __name__ == "__main__":
    unittest.main()
