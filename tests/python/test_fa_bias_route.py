"""Patch 0212's device-free gate: a failed GQA bias build falls back, never aborts.

The decision is tests/test_fa_bias_route.cpp (the pure function in
ggml-opencl-fa-bias-route.h, with its red twin). This cell reads the patched llama tree
and asserts the wiring that cell cannot see:
  - the dispatcher's indexed-bias block asks ggml_cl_fa_bias_route; its split branch
    clears use_fd_gqa (the launch then takes the split kernel's split size, grid and
    arguments) and leaves a DOWNGRADE line; the abort is left only for the NONE route;
  - the GQA bias variant is still memoised with its failure, so a later call does not
    rebuild it; a variant that builds but cannot take the launch (work-group cap, local
    memory: the plain build's rule) is released and counts as unbuilt;
  - supports_op still builds the split variant for a decode-row bias call and declines
    the call when that fails (reason no-bias-kernel), so the fallback has a kernel;
  - GGML_OPENCL_FA_BIAS_GQA_FAIL (debug) adds -DFA_BIAS_GQA_FORCE_FAIL to the GQA bias
    build only, and flash_attn_gqa_dpas.cl turns that define into a compile error.

Red: the applied series without 0212 (the tree the dispatcher aborted in) fails the
dispatcher, switch and kernel cells. Source-text assertions, not numerics.
"""
import os
import re
import unittest
from pathlib import Path

OPENCL = "ggml/src/ggml-opencl/ggml-opencl.cpp"
KERNEL = "ggml/src/ggml-opencl/kernels/flash_attn_gqa_dpas.cl"


def _llama_root():
    raw = os.environ.get("LLAMA_SRC")
    if not raw:
        raise unittest.SkipTest("LLAMA_SRC is unset: the patched llama tree is not available")
    root = Path(raw)
    for cand in (root, root.parent):
        if (cand / OPENCL).exists():
            return cand
    raise unittest.SkipTest("LLAMA_SRC has no ggml-opencl.cpp")


def _between(text, start, end):
    i = text.find(start)
    if i < 0:
        return None
    j = text.find(end, i + len(start))
    return text[i:j] if j >= 0 else None


class FaBiasRoute(unittest.TestCase):

    def setUp(self):
        self.root = _llama_root()
        self.cpp = (self.root / OPENCL).read_text()
        self.cl = (self.root / KERNEL).read_text()

    def _dispatch_block(self):
        block = _between(self.cpp, "const bool fa_bias = ggml_flash_attn_ext_has_indexed_bias(dst);",
                         "use_fd_mq = false;")
        self.assertIsNotNone(block, "the dispatcher's indexed-bias block (0143) is absent")
        return block

    def _builder(self):
        block = _between(self.cpp, "static cl_kernel ggml_opencl_fa_bias_kernel(", "\n}\n")
        self.assertIsNotNone(block, "ggml_opencl_fa_bias_kernel (0143) is absent")
        return block

    def test_header_included(self):
        self.assertTrue('#include "ggml-opencl-fa-bias-route.h"' in self.cpp, "ggml-opencl.cpp does not include the route header")

    def test_dispatch_routes(self):
        b = self._dispatch_block()
        self.assertIn("ggml_cl_fa_bias_route(use_fd_gqa,", b,
                      "the dispatcher must ask the route function with the rows' GQA choice")
        split = _between(b, "bias_route == GGML_CL_FA_BIAS_ROUTE_SPLIT", "} else {")
        self.assertIsNotNone(split, "the split route's branch is absent")
        self.assertIn("use_fd_gqa = false;", split,
                      "the fallback must clear use_fd_gqa, else the split kernel runs on the GQA launch")
        self.assertIn('ggml_cl_downgrade("op=FLASH_ATTN indexed-bias path=split reason=gqa-variant-unbuilt")', split)
        self.assertIn("fd_k_split = k_bias_split;", split)
        # the abort only after the last branch
        abort = b.find('GGML_ABORT("ggml_opencl: an indexed-bias FLASH_ATTN_EXT reached the dispatcher without a kernel")')
        self.assertGreater(abort, 0, "the NONE route keeps its abort")
        self.assertGreater(abort, b.rfind("} else {"), "the abort must sit in the NONE branch only")
        self.assertEqual(b.count("GGML_ABORT"), 1, "one abort in the block")
        self.assertIn("TAKEN src=opencl op=FLASH_ATTN indexed-bias form=gqa", b)
        self.assertIn("std::call_once", b, "the TAKEN line is once a process")

    def test_failure_is_memoised(self):
        b = self._builder()
        memo = b.find("backend_ctx->fa_bias_kernels[key] = k;")
        self.assertGreater(memo, 0, "the result, a failure included, must be memoised")
        self.assertLess(b.find("return it->second;"), memo, "the memo is read before any build")

    def test_switch_only_in_gqa_build(self):
        b = self._builder()
        gqa = _between(b, "if (kind == 0", "} else if (kind == 1")
        self.assertIsNotNone(gqa, "the GQA branch of the bias builder is absent")
        self.assertIn('ggml_cl_fa_bias_gqa_fail(getenv("GGML_OPENCL_FA_BIAS_GQA_FAIL"))', gqa)
        self.assertIn('opts += " -DFA_BIAS_GQA_FORCE_FAIL";', gqa)
        self.assertEqual(b.count("FA_BIAS_GQA_FORCE_FAIL"), gqa.count("FA_BIAS_GQA_FORCE_FAIL"),
                         "the forced failure belongs to the GQA variant only")
        self.assertEqual(self.cpp.count("GGML_OPENCL_FA_BIAS_GQA_FAIL\")"), 1, "the switch is read in one place")

    def test_variant_must_fit_the_launch(self):
        b = self._builder()
        gqa = _between(b, "if (kind == 0", "} else if (kind == 1")
        self.assertIsNotNone(gqa, "the GQA branch of the bias builder is absent")
        create = gqa.find("k = clCreateKernel(prog, name.c_str(), &err);")
        fits = gqa.find("ggml_cl_fa_gqa_fits(wg, lm, lm_dev, backend_ctx->fa_gqa_sides)")
        self.assertGreater(create, 0, "the GQA bias kernel is not created")
        self.assertGreater(fits, create, "the fit rule must run on the created kernel")
        for q in ("CL_KERNEL_WORK_GROUP_SIZE", "CL_KERNEL_LOCAL_MEM_SIZE", "CL_DEVICE_LOCAL_MEM_SIZE"):
            self.assertIn(q, gqa[create:fits], "the fit rule's input " + q + " is not queried")
        refuse = gqa[fits:gqa.find("CL_CHECK(clReleaseProgram(prog));", fits)]
        self.assertIn("CL_CHECK(clReleaseKernel(k));", refuse, "a refused variant must be released")
        self.assertIn("k = nullptr;", refuse, "a refused variant must count as unbuilt")

    def test_kernel_fails_on_the_define(self):
        m = re.search(r"#ifdef FA_BIAS_GQA_FORCE_FAIL\s*\n\s*#error[^\n]*\n\s*#endif", self.cl)
        self.assertIsNotNone(m, "flash_attn_gqa_dpas.cl must #error under FA_BIAS_GQA_FORCE_FAIL")
        self.assertFalse(')"' in self.cl, "the kernel is embedded as a raw string: no )\" in it")

    def test_supports_op_builds_the_split(self):
        sup = _between(self.cpp, "const bool decode_rows = q->ne[1] <= 8;", "ggml_cl_downgrade(\"op=FLASH_ATTN indexed-bias path=declined")
        self.assertIsNotNone(sup, "supports_op's indexed-bias block (0143) is absent")
        self.assertIn("ggml_opencl_fa_bias_kernel(backend_ctx, f16kv ? 1 : 2,", sup)
        self.assertIn('why = "no-bias-kernel";', sup)


if __name__ == "__main__":
    unittest.main()
