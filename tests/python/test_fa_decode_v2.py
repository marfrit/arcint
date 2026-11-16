"""Patch 0130a/0130b's device-free gate: the switch, the builds, the kernel's
double-buffered V stage, and the depth op cases it is red against.

The numeric half is on the card (the op suite at q8_0 K / q4_0 V, head 256,
24 query heads on 4, 4-7 rows, 4,096 / 65,536 / 245,760 keys), first with the
broken variant, then with the fix. This cell asserts the parts the card cannot
show: GGML_OPENCL_FA_DECODE_V2 is off by default; the V2 programme is a second
build that replaces a pair's kernel only after it built and fits local memory;
the kernel stages tile t+1 into the buffer this tile does not read, at the
loop's top (before QK^T), and P V reads the buffer of this tile; the op cases
carry the quantized depths.

Red mutations, each run and seen failing before it was restored:
  (a) the switch's default flipped on (`fdv_env != nullptr` without atoi):
      the default-off cell fails;
  (b) the prefetch moved after the tile's pre-P V barrier (the pre-0130b place,
      `FA_GQA_STAGE_V(k0 + BK,...)` just before `barrier(CLK_LOCAL_MEM_FENCE);`):
      the placement cell fails -- the loads are fenced and no longer overlap the
      products;
  (c) the consumed buffer not toggled (`vcur = 1 - vcur;` dropped): the toggle
      cell fails, and on the card every arm reads the previous tile's V (the
      kernel's own red);
  (d) the broken variant targeting the consumed buffer (`const int vn = vcur;`
      outside `#if GQA_V2_BROKEN`): the broken-gate cell fails.
Source-text assertions, not numerics.
"""
import os
import unittest
from pathlib import Path

OPENCL = "ggml/src/ggml-opencl/ggml-opencl.cpp"
KERNEL = "ggml/src/ggml-opencl/kernels/flash_attn_gqa_dpas.cl"
BACKEND = "tests/test-backend-ops.cpp"


def _llama_root():
    raw = os.environ.get("LLAMA_SRC")
    if not raw:
        raise unittest.SkipTest("LLAMA_SRC is unset: the patched llama tree is not available")
    root = Path(raw)
    for cand in (root, root.parent):
        if (cand / KERNEL).exists():
            return cand
    raise unittest.SkipTest("LLAMA_SRC has no flash_attn_gqa_dpas.cl: it lacks patch 0016")


class FaDecodeV2(unittest.TestCase):

    def setUp(self):
        self.root = _llama_root()
        self.t = {f: (self.root / f).read_text() for f in (OPENCL, KERNEL, BACKEND)}

    def test_switch_is_default_off(self):
        t = self.t[OPENCL]
        i = t.find('const char * fdv_env    = getenv("GGML_OPENCL_FA_DECODE_V2");')
        self.assertTrue(i >= 0, "the switch is absent")
        block = t[i:t.find("\n", t.find("const bool   fdv_on", i))]
        self.assertIn("atoi(fdv_env) != 0", block,
                      "the switch must be off unless GGML_OPENCL_FA_DECODE_V2 is a nonzero number")
        j = t.find("if (fdv_on) {")
        self.assertTrue(j >= 0, "the V2 build is not gated on the switch")
        # the base kernels are built first and replaced only on success
        self.assertIn("CL_CHECK(clReleaseKernel(backend_ctx->kernel_fa_gqa_dk[di][i]));", t)
        self.assertIn("backend_ctx->kernel_fa_gqa_dk[di][i] = kg;", t)
        self.assertIn("did not build; the base kernel stays", t,
                      "a refused V2 programme must leave the base kernel in the slot")
        self.assertIn('" -DGQA_V2=1"', t)
        self.assertIn('(fdv_broken ? " -DGQA_V2_BROKEN=1" : "")', t,
                      "the broken variant is a separate build, not a runtime flag")

    def test_kernel_has_two_stages_and_the_prefetch_at_the_top(self):
        k = self.t[KERNEL]
        self.assertIn("local half vt[2 * DV * BK];", k,
                      "two V stages: 16 KB each at head 256, BK 32")
        i = k.find("#if defined(GQA_V2) || defined(GQA_PRE)\n    // the first tile staged before the loop")
        self.assertTrue(i >= 0, "the prologue stage is absent")
        loop = k.find("for (int k0 = kv_start; k0 < kv_end; k0 += BK) {", i)
        self.assertTrue(loop >= 0)
        # the prefetch sits before the int8 stage and before the QK^T products
        pref = k.find("FA_GQA_STAGE_V(k0 + BK,", loop)
        qkt = k.find("// QK^T", loop)
        self.assertTrue(pref >= 0 and qkt > pref,
                      "tile t+1 must be staged before tile t's QK^T, so the loads overlap it")
        # the next barrier after the prefetch is the one before this tile's P V
        self.assertLess(pref, k.find("barrier(CLK_LOCAL_MEM_FENCE);", pref),
                        "the prefetch must precede a barrier, so its loads are not fenced by it")
        self.assertIn("#if GQA_V2_BROKEN", k)
        self.assertIn("const int vn = vcur;", k, "the broken variant overwrites the consumed stage")

    def test_pv_reads_the_consumed_buffer_and_the_toggle_exists(self):
        k = self.t[KERNEL]
        self.assertIn("vt + vcur * (DV * BK) + col * BK + g * 16", k,
                      "P V must read this tile's buffer")
        self.assertIn("vcur = 1 - vcur;", k, "the consumed buffer toggles at the tile's end")
        # the base form is untouched: the non-V2 path stages k0 into vt at the old place
        self.assertIn("#if !defined(GQA_V2) && !defined(GQA_PRE)\n        FA_GQA_STAGE_V(k0, vt)\n#endif", k)
        self.assertIn("local half vt[DV * BK];     // [column][key]", k)

    def test_depth_op_cases_exist_for_the_quantized_decode(self):
        t = self.t[BACKEND]
        i = t.find("// 0130a: the 4-7 row decode kernel at the depths the agent serves")
        self.assertTrue(i >= 0, "the 0130a op cases are absent")
        block = t[i:t.find("// 0119/0122: paged K/V at the agent's geometry", i)]
        self.assertIn("for (int64_t kv : {4096, 65536, 245760})", block, "the served depth")
        self.assertIn("for (int64_t nb : {4, 5, 6, 7})", block, "the 4-7 row form")
        self.assertIn("new test_flash_attn_ext(256, 256, 4, {6, 1}, kv, nb", block,
                      "head 256, 24 query heads on 4 (G 6)")
        self.assertIn("{GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}", block, "the served K/V pair")


if __name__ == "__main__":
    unittest.main()


class FaDecodePre(unittest.TestCase):
    """Patch 0130f's device-free gate: the register prefetch of the V stage, its
    broken variant, and 0130e's perf cases.

    0130b's V2 stage writes the loaded halves straight to local memory, so the
    thread stalls at the store before tile t's products; 0130f loads tile t+1's
    halves into private registers at the loop's top and stores them to the other
    buffer only after P V, so the value is consumed after the products.

    Red mutations, each run and seen failing before it was restored:
      (a) the register store moved before the tile's P V (into the other buffer):
          the placement cell fails -- the loads are consumed before the products;
      (b) the register loads moved after QK^T: the placement cell fails -- no
          load is in flight during the products;
      (c) the broken store removed: the broken-gate cell fails -- the red op
          cases would have nothing to fail against.
    """

    def setUp(self):
        self.root = _llama_root()
        self.t = {f: (self.root / f).read_text() for f in (OPENCL, KERNEL, BACKEND)}

    def test_switch_is_default_off_and_a_separate_build(self):
        t = self.t[OPENCL]
        i = t.find('const char * fdp_env    = getenv("GGML_OPENCL_FA_DECODE_PRE");')
        self.assertTrue(i >= 0, "the PRE switch is absent")
        block = t[i:t.find("\n", t.find("const bool   fdp_on", i))]
        self.assertIn("atoi(fdp_env) != 0", block,
                      "PRE must be off unless GGML_OPENCL_FA_DECODE_PRE is a nonzero number")
        self.assertIn("if (fdp_on) {", t, "the PRE build is not gated on the switch")
        self.assertIn('" -DGQA_PRE=1"', t)
        self.assertIn('(fdp_broken ? " -DGQA_PRE_BROKEN=1" : "")', t,
                      "the broken variant is a separate build, not a runtime flag")
        self.assertIn("did not build; the base kernel stays", t)

    def test_loads_are_at_the_top_and_the_store_after_pv(self):
        k = self.t[KERNEL]
        loop = k.find("for (int k0 = kv_start; k0 < kv_end; k0 += BK) {")
        self.assertTrue(loop >= 0)
        load = k.find("FA_GQA_PRE_V(k0 + BK, pre)", loop)
        qkt = k.find("// QK^T", loop)
        pv = k.find("for (int i = 0; i < NDH; ++i) {\n            const int col", loop)
        store = k.find("FA_GQA_STORE_V(pre, vt + (1 - vcur) * (DV * BK))", loop)
        self.assertTrue(load >= 0 and qkt >= 0 and pv >= 0 and store >= 0,
                        "PRE's load, QK^T, P V and store must all be present")
        self.assertLess(load, qkt, "the loads must be issued before the products")
        self.assertLess(qkt, pv)
        self.assertGreater(store, pv, "the store must follow the tile's P V")
        self.assertIn("half16 pre[2][4];", k, "private registers, not a local buffer")
        self.assertIn("local half vt[2 * DV * BK];", k, "V2's two stages are reused")

    def test_the_broken_variant_overwrites_the_consumed_stage(self):
        k = self.t[KERNEL]
        i = k.find("#ifdef GQA_PRE_BROKEN")
        self.assertTrue(i >= 0, "the broken variant is absent")
        block = k[i:k.find("#endif", i)]
        self.assertIn("FA_GQA_STORE_V(pre, vt + vcur * (DV * BK))", block,
                      "the broken store must target the buffer P V reads")
        j = k.find("barrier(CLK_LOCAL_MEM_FENCE);", i)
        self.assertGreater(j, i, "the broken store precedes the P V barrier")

    def test_the_perf_list_carries_the_depth_cases(self):
        t = self.t[BACKEND]
        i = t.find("static std::vector<std::unique_ptr<test_case>> make_test_cases_perf()")
        self.assertTrue(i >= 0)
        perf = t[i:t.find("// SWIGLU at a 27B-class FFN width", i)]
        self.assertIn("// 0130e: 0130a's depth cases in the PERF list too.", perf)
        self.assertIn("for (int64_t kv : {4096, 65536, 245760})", perf, "the served depth")
        self.assertIn("for (int64_t nb : {4, 5, 6, 7})", perf, "the 4-7 row form")
        self.assertIn("new test_flash_attn_ext(256, 256, 4, {6, 1}, kv, nb", perf)
        self.assertIn("{GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}", perf)
