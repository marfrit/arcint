"""Patch 0086/0087's device-free cell: GDN rollback by replay.

The op and its CPU reference (0086) and the memory/graph/OpenCL wiring (0087)
are read out of the patched tree (LLAMA_SRC): the replay prologue repeats the
step's own update in the same order as the snapshot path, the record epilogue
stores the three small vectors, the switch defaults OFF and the memory sizes S
to two planes only when it is on, and the OpenCL kernel is built only with the
switch. The numerics are the replay-chain cases in test-backend-ops; the card
run is owed (commands in the patch README).

Red mutations, each run and seen failing before it was restored:
  (a) the CPU prologue's `ggml_vec_mad_f32` dropped: the prologue cell fails;
  (b) the record's `r[2 * H * S_v + iv1] = g_val` -> `g0`: the exactness cell fails;
  (c) `n_rows_s = rpl ? mem_size * 2 : n_rows` -> `n_rows`: the plane cell fails;
  (d) the OpenCL prologue line's `fma(` -> `+`: the kernel-order cell fails;
  (e) the switch's `atoi(env) != 0` -> `atoi(env) == 0`: the default-off cell fails.
Standard library only (unittest); the recipe's `ctest -L unit` runs it.
"""
import os
import re
import unittest
from pathlib import Path


def _root():
    raw = os.environ.get("LLAMA_SRC")
    if not raw:
        raise unittest.SkipTest("LLAMA_SRC is unset: the patched llama tree is not available")
    root = Path(raw)
    for cand in (root, root.parent):
        if (cand / "ggml" / "src" / "ggml.c").exists() and (cand / "src" / "llama-memory-recurrent.cpp").exists():
            return cand
    raise unittest.SkipTest("LLAMA_SRC has no patched ggml/llama tree")


ROOT = None


def read(rel):
    return (ROOT / rel).read_text(errors="replace")


def setUpModule():
    global ROOT
    ROOT = _root()


class OpAndCPU(unittest.TestCase):
    def test_op_declared_and_dispatched(self):
        h = read("ggml/include/ggml.h")
        self.assertRegex(h, r"GGML_OP_GATED_DELTA_NET_REPLAY,\s*\n\s*GGML_OP_COUNT")
        self.assertIn("ggml_gated_delta_net_replay(", h)
        c = read("ggml/src/ggml.c")
        self.assertRegex(c, r'"GATED_DELTA_NET_REPLAY",\s*\n\};\s*\n\s*static_assert\(GGML_OP_COUNT == 103')
        self.assertIn("R = H * (2 * S_v + 1)", c)
        self.assertRegex(c, r"result->src\[7\] = ctrl")
        cpu = read("ggml/src/ggml-cpu/ops.cpp")
        self.assertIn("GGML_OP_GATED_DELTA_NET_REPLAY", read("ggml/src/ggml-cpu/ggml-cpu.c"))
        self.assertIn("ggml_compute_forward_gated_delta_net_replay", cpu)

    def test_prologue_repeats_the_step(self):
        cpu = read("ggml/src/ggml-cpu/ops.cpp")
        # the one_chunk scalar-gate step: scale by g, then mad k * delta
        i = cpu.index("ggml_compute_forward_gated_delta_net_replay_f32")
        j = cpu.index("void ggml_compute_forward_gated_delta_net_replay(", i)
        body = cpu[i:j]
        self.assertIn("ggml_vec_scale_f32(S_v * S_v, s, r[2 * H * S_v + iv1]);", body)
        self.assertIn("ggml_vec_mad_f32(S_v, &s[jj * S_v], rk, rd[jj]);", body)

    def test_records_are_the_step_vectors(self):
        cpu = read("ggml/src/ggml-cpu/ops.cpp")
        i = cpu.index("ggml_compute_forward_gated_delta_net_replay_f32")
        j = cpu.index("void ggml_compute_forward_gated_delta_net_replay(", i)
        body = cpu[i:j]
        self.assertIn("memcpy(r + iv1 * S_v,           delta, S_v * sizeof(float));", body)
        self.assertIn("memcpy(r + H * S_v + iv1 * S_v, k_d,   S_v * sizeof(float));", body)
        self.assertIn("r[2 * H * S_v + iv1] = g_val;", body)

    def test_chain_case_present(self):
        t = read("tests/test-backend-ops.cpp")
        self.assertIn("test_gdn_replay_chain", t)
        self.assertRegex(t, r"mx = std::max\(mx, \(double\) std::fabs\(a\[i\] - b\[i\]\)\);")
        self.assertIn("return exact ? 0.0 : 1e-7;", t)


class MemoryAndOpenCL(unittest.TestCase):
    def test_switch_defaults_off_and_two_planes(self):
        m = read("src/llama-memory-recurrent.cpp")
        self.assertIn('getenv("LLAMA_GDN_REPLAY")', m)
        self.assertIn("atoi(env) != 0", m)
        self.assertIn("const uint32_t n_rows_s = rpl ? mem_size * 2 : n_rows;", m)
        self.assertIn("ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_rows_s)", m)

    def test_pending_records_and_rollback(self):
        m = read("src/llama-memory-recurrent.cpp")
        i = m.index("const llama_pos rollback = cell.pos - (p0 - 1);")
        j = m.index("set_rs_idx(seq_id, (uint32_t) rollback);", i)
        self.assertIn("if (rpl) {", m[i:j])
        self.assertIn("rpl_n[seq_id] -= (uint32_t) rollback;", m[i:j])
        self.assertIn("mem->rpl_n[seq]    = std::min(n_seq_tokens - 1, mem->n_rs_seq);", m)
        self.assertIn("mem->rpl_bank[seq] = (uint8_t) (bank ^ 1);", m)
        self.assertIn("const uint32_t s_trans = rpl_pending > 0 ? 2 : 0;", m)
        self.assertIn("io.write(&rpl_pending, sizeof(rpl_pending));", m)
        self.assertIn("io.write(&rpl_R,       sizeof(rpl_R));", m)

    def test_rows_form_is_disabled_by_replay(self):
        q = read("src/models/qwen35.cpp")
        self.assertIn("const bool gdn_rpl = inp->s_rpl_ctrl != nullptr;", q)
        self.assertIn("const bool gdn_state_rows = !gdn_rpl && llama_gdn_state_rows_enabled", q)
        self.assertIn("build_rs(ssm_states_all, inp->s_rpl_rows, inp->s_rpl_main", q)
        d = read("src/models/delta-net-base.cpp")
        self.assertIn("ggml_gated_delta_net_replay(ctx0, q, k, v, g, b, s, rec, inp->s_rpl_ctrl, (int) K)", d)
        self.assertIn("ggml_set_rows(ctx0,", d)

    def test_opencl_kernel_and_build_gate(self):
        k = read("ggml/src/ggml-opencl/kernels/gated_delta_net_intel.cl")
        i = k.index("#ifdef GDN_RPL")
        j = k.index("#endif // GDN_RPL", i)
        body = k[i:j]
        self.assertIn("uint  RK,", body)
        self.assertIn("S[rr][m] = fma(", body)
        self.assertIn("rt[2 * H_v * S_V + head] = g;", body)
        self.assertIn("if (lane < R) rt[head * S_V + row0 + lane] = dl[lane];", body)
        c = read("ggml/src/ggml-opencl/ggml-opencl.cpp")
        self.assertIn('getenv("LLAMA_GDN_REPLAY")', c)
        self.assertIn("kernel_gdn_intel_rpl", c)
        i = c.index("case GGML_OP_GATED_DELTA_NET_REPLAY:")
        j = c.index("case GGML_OP_CONCAT:", i)
        self.assertIn("kernel_gdn_intel_rpl != nullptr", c[i:j])


if __name__ == "__main__":
    unittest.main()
