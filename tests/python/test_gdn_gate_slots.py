"""Patch 0081's device-free gate cell: the GDN family's source slots.

The numeric half is tests/test_gdn_gate_slots.cpp (the CPU backend: the served
gate chain with 0051's raw projections attached computes the plain path's bytes,
with rows on and with replay on). This cell reads the patched tree and asserts
the parts the C cell cannot:

  - op_params[1] is the mode word of the GDN family (GGML_GDN_MODE_GATHERED /
    ROWS / REPLAY), defined in ggml.h;
  - both GATED_DELTA_NET builders set it; the replay builder sets REPLAY;
  - the CPU op, the OpenCL supports_op and the OpenCL dispatch select their
    state path from the mode word, never from whether src[6] is set -- the
    defect was that src[6] was taken three times (0051's alpha, the rows index,
    the replay's records), so a gathered node with 0051's fold attached looked
    like the rows form and the CPU op asserted on a F32;
  - src[6] is the rows form's row index, src[6]/src[7] the replay's records and
    control;
  - gdn-gate-inputs.h (the header the builders and the C cell share) writes
    src[8] and src[9] and nothing else, delta-net-base.cpp uses it, and 0051's
    OpenCL prepass matches those slots.

Red mutations, each run and seen failing before it was restored (commands in the
patch README section):
  (a) gdn-gate-inputs.h writing src[6]/src[7] (the served pre-fix form): the
      sheet cell and the C cell fail (the C cell aborts in the CPU op);
  (b) the plain builder announcing ROWS (src[6] unset): the mode cell and the
      C cell fail;
  (c) the CPU op reading `dst->src[6] != nullptr` again: the mode cell fails;
  (d) 0051's prepass comparing src[6]/src[7]: the slot cell fails.
These are source-text assertions (the cells read the patched tree), not
numerics. Standard library only (unittest), so `ctest -L unit` runs it on a
build host with python3 and nothing else.
"""
import os
import unittest
from pathlib import Path

GGML_C = "ggml/src/ggml.c"
GGML_H = "ggml/include/ggml.h"
CPU_OPS = "ggml/src/ggml-cpu/ops.cpp"
OPENCL = "ggml/src/ggml-opencl/ggml-opencl.cpp"
GATE_H = "src/models/gdn-gate-inputs.h"
DELTA_BASE = "src/models/delta-net-base.cpp"


def _llama_root():
    raw = os.environ.get("LLAMA_SRC")
    if not raw:
        raise unittest.SkipTest("LLAMA_SRC is unset: the patched llama tree is not available")
    root = Path(raw)
    for cand in (root, root.parent):
        if (cand / GGML_C).exists() and (cand / GATE_H).exists():
            return cand
    raise unittest.SkipTest("LLAMA_SRC has no patched llama tree: it lacks patch 0081")


def _body(text, marker, end="\n}\n"):
    i = text.find(marker)
    if i < 0:
        return None
    j = text.find(end, i)
    return text[i:j if j >= 0 else len(text)]


class GdnGateSlots(unittest.TestCase):

    def setUp(self):
        self.root = _llama_root()
        self.t = {f: (self.root / f).read_text() for f in (GGML_C, GGML_H, CPU_OPS, OPENCL, GATE_H, DELTA_BASE)}

    def test_mode_word_is_defined(self):
        h = self.t[GGML_H]
        for name, val in (("GGML_GDN_MODE_GATHERED", "0"), ("GGML_GDN_MODE_ROWS", "1"),
                          ("GGML_GDN_MODE_REPLAY", "2")):
            self.assertRegex(h, rf"#define\s+{name}\s+{val}\b", name + " must be defined in ggml.h")
        self.assertIn("op_params[1] is the state mode", h, "the contract is documented")

    def test_builders_announce_the_mode(self):
        t = self.t[GGML_C]
        plain = _body(t, "struct ggml_tensor * ggml_gated_delta_net(\n")
        rows = _body(t, "struct ggml_tensor * ggml_gated_delta_net_rows(")
        replay = _body(t, "struct ggml_tensor * ggml_gated_delta_net_replay(")
        for b, want, name in ((plain, "GGML_GDN_MODE_GATHERED", "plain"),
                              (rows, "GGML_GDN_MODE_ROWS", "rows"),
                              (replay, "GGML_GDN_MODE_REPLAY", "replay")):
            self.assertTrue(b, name + " builder is absent")
            self.assertIn(f"ggml_set_op_params_i32(result, 1, {want});", b,
                          name + " builder must set the mode word")
        self.assertIn("result->src[6] = rows;", rows, "the rows index stays on src[6]")
        self.assertIn("result->src[6] = rec;", replay, "the replay's records stay on src[6]")
        self.assertIn("result->src[7] = ctrl;", replay, "the replay's control stays on src[7]")
        for b, name in ((plain, "plain"), (rows, "rows"), (replay, "replay")):
            for slot in ("src[8]", "src[9]"):
                self.assertNotIn("result->" + slot + " =", b,
                                 name + " builder must not write the gating slots")

    def test_cpu_selects_by_mode(self):
        t = self.t[CPU_OPS]
        self.assertIn("const int gdn_mode = ggml_get_op_params_i32(dst, 1);", t)
        self.assertIn("const bool rows_mode = gdn_mode == GGML_GDN_MODE_ROWS;", t)
        self.assertIn("const ggml_tensor * src_rows = rows_mode ? dst->src[6] : nullptr;", t)
        self.assertNotIn("const ggml_tensor * src_rows = dst->src[6];", t,
                         "the rows path must not be inferred from src[6] being set")
        self.assertIn("GGML_ASSERT(ggml_get_op_params_i32(dst, 1) == GGML_GDN_MODE_REPLAY);", t,
                      "the replay op checks its own mode")

    def test_opencl_selects_by_mode(self):
        t = self.t[OPENCL]
        self.assertIn("const int gdn_mode = ggml_get_op_params_i32(op, 1);", t)
        self.assertIn("if (gdn_mode == GGML_GDN_MODE_ROWS)", t)
        self.assertIn("} else if (gdn_mode != GGML_GDN_MODE_GATHERED) {", t,
                      "an unknown mode is declined, not guessed")
        self.assertIn("const ggml_tensor * src_rows = gdn_mode == GGML_GDN_MODE_ROWS ? dst->src[6] : nullptr;", t)
        self.assertIn("ggml_get_op_params_i32(op, 1) != GGML_GDN_MODE_REPLAY", t,
                      "the replay's supports_op checks its mode")
        self.assertNotIn("if (op->src[6] != nullptr) {   // rows mode", t,
                         "the rows path must not be inferred from src[6] being set")

    def test_gating_inputs_ride_slots_8_and_9(self):
        g = self.t[GATE_H]
        self.assertIn("gdn->src[8] = sp->src[0]->src[0];", g)
        self.assertIn("gdn->src[9] = b->src[0];", g)
        self.assertNotIn("gdn->src[6]", g)
        self.assertNotIn("gdn->src[7]", g)
        d = self.t[DELTA_BASE]
        self.assertIn('#include "gdn-gate-inputs.h"', d, "the builders use the same header the cell does")
        self.assertIn("gdn_keep_gating_inputs(result, g, b);", d)
        self.assertNotIn("static void gdn_keep_gating_inputs", d, "the helper lives in the header only")

    def test_0051_matches_the_gating_slots(self):
        t = self.t[OPENCL]
        self.assertIn("if (gd->src[8] == al) part_ok |= 1;", t,
                      "0051's fold matches the alpha on src[8]")
        self.assertIn("if (gd->src[9] == br) part_ok |= 2;", t,
                      "0051's fold matches the b projection on src[9]")
        self.assertNotIn("if (gd->src[6] == al)", t)
        self.assertNotIn("if (gd->src[7] == br)", t)


if __name__ == "__main__":
    unittest.main()
