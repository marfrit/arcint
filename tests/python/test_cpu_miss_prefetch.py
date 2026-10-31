"""Patch 0075's device-free cell: the CPU-miss prefetch arm's switch, its hints, its touch.

The prefetch and the cold-page touch run only where the expert cache hands misses
to the CPU's AVX2 rows, so the device-free half of the proof is: the switch
defaults OFF and is read once at registration; every row function carries one
block-ahead hint; the touch reads a byte a 4 KiB page through a volatile pointer
(so the page fault is not optimised away); and the touch pass runs in both the
fast and the slow compute path, under level >= 2 only. The residency and parity
arms are C cells in `tests/test-opencl-miss-cpu.cpp` (a touch that strides 8192
leaves half the pages absent, a zero ahead distance fails). The card arms are
the cold/warm served pair and `perf stat` on one decode, run by Claude in an
announced window (the handover carries the commands).

Red mutations, each run and seen failing before it was restored:
  (a) the hint dropped from one row function: the per-function cell fails;
  (b) the ahead distance set to 0: the distance cell fails (and the C cell's
      "prefetch ahead" line);
  (c) the touch stride 4096 -> 8192: the stride cell fails (and the C residency
      cell);
  (d) the touch's pointer not volatile: the volatile cell fails;
  (e) the default level 1: the default-off cell fails;
  (f) the touch pass's `>= 2` guard removed: the guard cell fails.
Standard library only (unittest); the recipe's `ctest -L unit` runs it.
"""
import os
import unittest
from pathlib import Path

ROW_FUNCTIONS = ["row_iq(", "row_iq2xs(", "row_iq3xxs(", "row_iq3s_1(", "row_q2(", "row_iq4nl("]


def _llama_root():
    raw = os.environ.get("LLAMA_SRC")
    if not raw:
        raise unittest.SkipTest("LLAMA_SRC is unset: the patched llama tree is not available")
    root = Path(raw)
    for cand in (root, root.parent):
        if (cand / "ggml" / "src" / "ggml-opencl" / "ggml-opencl-miss-cpu.cpp").exists():
            return cand
    raise unittest.SkipTest("LLAMA_SRC has no ggml-opencl-miss-cpu.cpp: the tree lacks patch 0075")


class CpuMissPrefetch(unittest.TestCase):
    def setUp(self):
        root = _llama_root()
        base = root / "ggml" / "src" / "ggml-opencl"
        self.miss = (base / "ggml-opencl-miss-cpu.cpp").read_text()
        self.hdr = (base / "ggml-opencl-miss-cpu.h").read_text()
        self.cpp = (base / "ggml-opencl.cpp").read_text()

    def test_switch_defaults_off(self):
        self.assertIn('getenv("GGML_OPENCL_CPU_MISS_PREFETCH"); return e != nullptr ? atoi(e) : 0;', self.cpp)
        self.assertIn("static int g_cl_miss_pf = 0;", self.miss)

    def test_ahead_distance(self):
        self.assertRegex(self.miss, r"GGML_CL_MISS_PF_AHEAD\s*=\s*2;")
        self.assertIn("int ggml_cl_miss_cpu_prefetch_ahead(void)", self.miss)
        self.assertIn("int    ggml_cl_miss_cpu_prefetch_ahead(void)", self.miss)  # the no-AVX2 stub

    def test_every_row_function_has_the_hint(self):
        for name in ROW_FUNCTIONS:
            start = self.miss.find(name)
            self.assertGreater(start, 0, name)
            end = self.miss.find("\n}\n", start)
            self.assertGreater(end, start, name)
            body = self.miss[start:end]
            self.assertIn("GGML_CL_MISS_PF(", body, "%s lacks the block-ahead hint" % name)

    def test_touch_is_volatile_and_one_byte_a_page(self):
        self.assertRegex(self.miss, r"volatile unsigned char \* q")
        self.assertRegex(self.miss, r"o \+= 4096\)")
        self.assertIn("void   ggml_cl_miss_cpu_touch(const void * p, size_t bytes);", self.hdr)

    def test_touch_pass_is_gated_at_level_2(self):
        # both compute paths (the grouped fast path and the single-token slow path)
        self.assertGreaterEqual(self.cpp.count("if (L.prefetch >= 2) {"), 2)
        self.assertIn("ggml_cl_miss_touch(L,", self.cpp)

    def test_level_reaches_the_rows(self):
        self.assertIn("ggml_cl_miss_cpu_set_prefetch(pf);", self.cpp)
        self.assertIn("L->prefetch = pf;", self.cpp)


if __name__ == "__main__":
    unittest.main()
