"""Patch 0224's device-free cell: every 1.8.3 item's TAKEN / reach line reaches a served unit's log, once a process.

arcint routes libllama's and ggml's log (route_llama_log, src/exec/backend_llama.cpp): ERROR and WARN to its error
and warn lines, everything else to its verbose level, which a unit started without -v does not show. 0211 raised the
depth mix's TAKEN line to WARN for that reason. The 1.8.3 items' lines, read from the applied tree:

  0211 depth mix     "depth_mix: TAKEN opencl f32 token-a-group"                 WARN, call_once (0211)
  0212 FA GQA bias   "TAKEN src=opencl op=FLASH_ATTN indexed-bias form=gqa"      WARN, call_once (0212)
  0220 Q8_0 2D GEMM  "TAKEN src=opencl op=MUL_MAT type=q8_0 path=2d-"            WARN; a static bool array before
                                                                                  0224, call_once a mode since
  0221 loop control  "loop_control: TAKEN opencl "                               INFO under a static bool before
                                                                                  0224; WARN, call_once since
  0222 64-pair tile  "TAKEN src=opencl op=MUL_MAT_ID type=%s path=xt64"         WARN, call_once (two sites)

The cell asserts for each: its GGML_LOG_* macro is one route_llama_log shows outside -v, and every call site is
inside a std::call_once on a static std::once_flag (an element of a static once_flag array for a line a mode). It
reads LLAMA_SRC (the applied tree) and arcint's routing; source text, not a run (the paths need the card).
Red on the series without 0224 (aa0744b9): 0221's level and guard and 0220's guard fail.
"""
import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROUTING = ROOT / "src" / "exec" / "backend_llama.cpp"
OPENCL = "ggml/src/ggml-opencl/ggml-opencl.cpp"

# (patch, the FULL format string as the source writes it (a tail change fails the cell), call sites, the guard's index:
# None for one flag, else the expression the flag array must be indexed with)
LINES = [
    ("0211", r"depth_mix: TAKEN opencl f32 token-a-group\n", 1, None),
    ("0212", r"TAKEN src=opencl op=FLASH_ATTN indexed-bias form=gqa\n", 1, None),
    ("0220", r"TAKEN src=opencl op=MUL_MAT type=q8_0 path=2d-%s k=%d m=%d n=%d wg=%dx%d\n", 1, "mode"),
    ("0221", r"loop_control: TAKEN opencl %s one-launch rows_wg=%d n_wg=%d\n", 1, None),
    ("0222", r"TAKEN src=opencl op=MUL_MAT_ID type=%s path=xt64 tile=64 skip=1\n", 2, None),
]


def _llama_root():
    raw = os.environ.get("LLAMA_SRC")
    if not raw:
        raise unittest.SkipTest("LLAMA_SRC is unset: the patched llama tree is not available")
    root = Path(raw)
    for cand in (root, root.parent):
        if (cand / OPENCL).exists():
            return cand
    raise AssertionError("LLAMA_SRC names no llama tree with " + OPENCL)


def _routed_levels(src):
    i = src.find("void route_llama_log(")
    assert i >= 0, "route_llama_log is absent"
    body = src[i:src.find("\n}\n", i)]
    assert "log::verbose" in body, "route_llama_log no longer has a verbose fallback: re-read this cell"
    return set(re.findall(r"level == GGML_LOG_LEVEL_([A-Z]+)\)\s*log::(?:error|warn)\(", body))


def _sites(text, needle):
    """(position, macro) of every GGML_LOG_* call whose format string starts with needle."""
    return [(m.start(), m.group(1)) for m in re.finditer(r"(GGML_LOG_[A-Z]+)\(\s*\"" + re.escape(needle) + "\"", text)]


def _guard(index):
    """std::call_once on a static once_flag right before the call; with an index, a static flag ARRAY indexed by it
    (one flag a mode: a flag shared across the modes would print one mode's line only)."""
    if index is None:
        return re.compile(r"static std::once_flag (\w+);\s*std::call_once\(\1,[^{]*\{\s*$")
    return re.compile(r"static std::once_flag (\w+)\[\d+\];\s*std::call_once\(\1\[" + re.escape(index) + r"\],[^{]*\{\s*$")


class TakenLines183(unittest.TestCase):

    def setUp(self):
        self.cl = (_llama_root() / OPENCL).read_text()
        self.shown = _routed_levels(ROUTING.read_text())

    def test_each_line_present(self):
        for patch, needle, n, _ in LINES:
            with self.subTest(patch=patch):
                self.assertEqual(len(_sites(self.cl, needle)), n, "%s: %d call site(s) of %r expected" % (patch, n, needle))

    def test_each_line_reaches_the_served_log(self):
        for patch, needle, _, _ in LINES:
            for pos, macro in _sites(self.cl, needle):
                with self.subTest(patch=patch, at=pos):
                    self.assertIn(macro[len("GGML_LOG_"):], self.shown,
                                  "%s: %r is %s, which route_llama_log keeps for -v" % (patch, needle, macro))

    def test_each_line_once_a_process(self):
        for patch, needle, _, index in LINES:
            for pos, macro in _sites(self.cl, needle):
                with self.subTest(patch=patch, at=pos):
                    head = self.cl[max(0, pos - 500):pos]
                    self.assertIsNotNone(_guard(index).search(head),
                                         "%s: %r is not printed inside a std::call_once on a static once_flag" % (patch, needle))


if __name__ == "__main__":
    unittest.main()
