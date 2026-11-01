"""Patch 0074's device-free cell: the q/k norm + IMRoPE fusion's terms and its gate.

The fused kernel runs only under an OpenCL card, so the device-free half of the
patch's proof is (a) that the kernel's arithmetic carries every term the two
kernels it replaces carry — the epsilon, the per-column norm weight at both
rotated positions, the norm-then-weight rounding order, the rotation's cos/sin
and the interleaved M-RoPE's position layout — and (b) that the dispatch is a
3-node pattern checked before the 2-node norm+mul prefix, behind a switch that
defaults OFF. The numerics are `test-backend-ops -o RMS_NORM_MUL_ROPE` on the
cards (the patch adds the served models' shapes there, a red case drops one
term), run by Claude in an announced window.

Red mutations, each run and seen failing before it was restored (command in the
patch README section):
  (a) the norm's epsilon dropped (`sqrt(sumf / ne00 + eps)` -> `sqrt(sumf / ne00)`):
      the epsilon cell fails;
  (b) the weight read at the wrong index (`w[n_offs + ic]` -> `w[ic]`): the
      weight-at-rotation cell fails;
  (c) the norm weighted before rounding (`as_float(as_uint(src[0] * scale) ^ opaque) * w[`
      -> `src[0] * scale * w[`): the rounding-order cell fails;
  (d) the switch's default flipped (`fuse_qk_norm_rope = false` -> `= true`):
      the default-off cell fails;
  (e) the fused dispatch moved below the 2-node norm+mul check: the
      pattern-order cell fails;
  (f) the sections guard dropped (`sections[0] + ... <= 0` removed), and the
      operand-order guard dropped (`rope->src[0] != mul` removed): the guards
      cell fails.
These are source-text assertions (the cells read the patched tree), not
numerics; `test-backend-ops`' NMSE cannot see the fused rounding order either,
so the rounding gate is the answer-level bar with the md5/argmax tripwire.
Standard library only (unittest), so the recipe's `ctest -L unit` runs it on a
build host with python3 and nothing else.
"""
import os
import unittest
from pathlib import Path



def _llama_root():
    raw = os.environ.get("LLAMA_SRC")
    if not raw:
        raise unittest.SkipTest("LLAMA_SRC is unset: the patched llama tree is not available")
    root = Path(raw)
    for cand in (root, root.parent):
        if (cand / "ggml" / "src" / "ggml-opencl" / "kernels" / "rms_norm.cl").exists():
            return cand
    raise unittest.SkipTest("LLAMA_SRC has no ggml-opencl kernels: the tree lacks patch 0074")


def _kernel_body(text):
    start = text.find("kernel void kernel_rms_norm_mul_rope_multi_f32(")
    if start < 0:
        return None
    # the body runs to the next kernel declaration or the end of the file
    nxt = text.find("kernel void ", start + 1)
    return text[start:nxt if nxt >= 0 else len(text)]


class FuseQkNormRopeTerms(unittest.TestCase):
    """The fused kernel carries the norm's epsilon and weight, the rounding
    order, the rotation and the IMROPE position layout (mutations a-c)."""

    def setUp(self):
        self.root = _llama_root()
        self.cl = (self.root / "ggml" / "src" / "ggml-opencl" / "kernels" / "rms_norm.cl").read_text()
        body = _kernel_body(self.cl)
        if body is None:
            self.fail("rms_norm.cl has no kernel_rms_norm_mul_rope_multi_f32")

    def test_epsilon_is_in_the_norm_scale(self):
        self.assertRegex(self.cl,
                         r"scale\s*=\s*1\.0f\s*/\s*sqrt\(\s*sumf\s*/\s*ne00\s*\+\s*eps\s*\)",
                         "the norm's scale must add the epsilon, as kernel_rms_norm_mul does")

    def test_weight_is_applied_at_both_rotated_positions(self):
        body = _kernel_body(self.cl)
        self.assertIn("w[n_offs + ic]", body, "the first rotated position's weight")
        self.assertIn("w[n_offs + ic + n_dims/2]", body, "the second rotated position's weight")

    def test_norm_is_rounded_before_the_weight(self):
        # kernel_rms_norm_mul's rounding: (x * scale) stored, then * w; the opaque
        # guard stops the compiler regrouping them into x * (scale * w)
        self.assertRegex(self.cl,
                         r"x0\s*=\s*as_float\(as_uint\(src\[0\] \* scale\) \^ opaque\) \* w\[",
                         "the normed value must round before the weight multiply")

    def test_rotation_is_the_mrope_formula(self):
        body = _kernel_body(self.cl)
        self.assertIn("x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1", body)
        self.assertIn("x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0", body)
        self.assertIn("rope_yarn_fused", body, "the YaRN cos/sin path, as rope.cl's")

    def test_positions_offset_is_bytes(self):
        # window 43: src1 is an int pointer and `src1 + offset1` stepped the tensor
        # offset in ELEMENTS (x4 bytes), so the kernel read zeros for every position
        # stream. Every operand's offset is bytes, as kernel_rope_multi_f32 casts it.
        self.assertRegex(self.cl, r"src1 = \(global int \*\)\(\(global char \*\) src1 \+ offset1\);")

    def test_sect_dims_guarded(self):
        # ic % sect_dims is undefined at zero: the host refuses an empty sections
        # sum and the kernel keeps a defined divisor if the argument ever arrives so
        self.assertRegex(self.cl, r"sect_sum > 0 \? sect_sum : 1")

    def test_imrope_position_layout(self):
        body = _kernel_body(self.cl)
        for s in ('pos[i02 + ne02 * 0]', 'pos[i02 + ne02 * 1]', 'pos[i02 + ne02 * 2]', 'pos[i02 + ne02 * 3]'):
            self.assertIn(s, body, s)


class FuseQkNormRopeGate(unittest.TestCase):
    """The dispatch: 3-node pattern, before the 2-node prefix, switch default off
    (mutations d-e)."""

    def setUp(self):
        self.root = _llama_root()
        self.cpp = (self.root / "ggml" / "src" / "ggml-opencl" / "ggml-opencl.cpp").read_text()

    def test_switch_defaults_off(self):
        self.assertRegex(self.cpp, r"fuse_qk_norm_rope\s*=\s*false\s*;")
        self.assertIn('getenv("GGML_OPENCL_FUSE_QK_NORM_ROPE")', self.cpp)

    def test_dispatch_pattern_and_order(self):
        fused = self.cpp.find("fuse_qk_norm_rope &&")
        self.assertGreaterEqual(fused, 0, "the fused dispatch is not guarded by its switch")
        three = self.cpp.find("GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }", fused)
        self.assertGreater(three, fused, "the 3-node pattern is not the fused one")
        # the 2-node prefix must come after, or it swallows the same nodes
        two = self.cpp.find("ggml_opencl_can_fuse(backend_ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })", fused)
        self.assertGreater(two, three, "the norm+mul pair is checked before the 3-node pattern")

    def test_lws_cap(self):
        # the work-group is one row; lws is the pairs rounded to a whole subgroup and
        # capped at sgs*sgs, so the number of subgroups fits the local `sum` (sgs floats)
        self.assertRegex(self.cpp, r"const int pairs = n_dims / 2;")
        self.assertRegex(self.cpp, r"nth = MIN\(nth, \(int\) sgs \* \(int\) sgs\);")

    def test_kernel_is_created(self):
        self.assertIn('clCreateKernel(backend_ctx->program_rms_norm, "kernel_rms_norm_mul_rope_multi_f32"', self.cpp)

    def test_gate_guards(self):
        # the rotation's input is the mul (can_fuse also accepts it as rope's src[1])
        self.assertIn("rope->src[0] != mul", self.cpp)
        # the sections must carry a dimension: the per-op path asserts the sum, the
        # fused kernel's ic % sect_dims would divide by zero on the GPU silently
        self.assertRegex(self.cpp, r"sections\[0\] \+ sections\[1\] \+ sections\[2\] \+ sections\[3\] <= 0")


class FuseQkNormRopeCardCases(unittest.TestCase):
    """The card cases exist: the served shapes, the 1-D weight, IMROPE, the
    strided Q layout (one term dropped is the red arm on the card)."""

    def test_backend_ops_cases(self):
        # the cases live in the patched llama tree, not in arcint
        llama = _llama_root()
        ops = (llama / "tests" / "test-backend-ops.cpp").read_text()
        self.assertIn("GGML_ROPE_TYPE_IMROPE, true, true, GGML_TYPE_F16, true", ops,
                      "the strided Q + 1-D weight IMROPE cases")
        self.assertIn("bool gap = false)", ops, "the case's strided-input option")


if __name__ == "__main__":
    unittest.main()
