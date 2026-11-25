"""Patch 0211's device-free cell: the depth mix's TAKEN line reaches a served unit's log.

0205 printed `depth_mix: TAKEN opencl f32 token-a-group` with GGML_LOG_INFO. arcint routes
libllama's and ggml's log (route_llama_log, src/exec/backend_llama.cpp): ERROR and WARN to
its error and warn lines, everything else to its verbose level. So on a served Inkling unit
with GGML_OPENCL_DEPTH_MIX=1 the FA port's TAKEN line (GGML_LOG_WARN) showed and the depth
mix's did not: production could not see that the mix was on. 0211 prints it at the port's
level, under the port's once-a-process guard (std::call_once; the bool guard it replaces was
a data race when two backends launch at once).

The cell reads the patched tree (LLAMA_SRC) and arcint's routing and asserts:
  - the depth mix's TAKEN line is printed with the same GGML_LOG_* macro as the FA port's
    TAKEN lines, and that macro is one route_llama_log sends to a non-verbose level;
  - it is printed once a process: inside a std::call_once on a static once_flag;
  - its text is unchanged (`depth_mix: TAKEN opencl f32 token-a-group`: the README's grep).

Red on the series without 0211 (the 1.8.2 tree): the line is GGML_LOG_INFO under a static
bool, and the level and guard cases fail. Source-text assertions, not a run: the op needs
the card.
"""
import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROUTING = ROOT / "src" / "exec" / "backend_llama.cpp"
OPENCL = "ggml/src/ggml-opencl/ggml-opencl.cpp"
TEXT = "depth_mix: TAKEN opencl f32 token-a-group"


def _llama_root():
    raw = os.environ.get("LLAMA_SRC")
    if not raw:
        raise unittest.SkipTest("LLAMA_SRC is unset: the patched llama tree is not available")
    root = Path(raw)
    for cand in (root, root.parent):
        if (cand / OPENCL).exists():
            return cand
    raise AssertionError("LLAMA_SRC names no llama tree with " + OPENCL)


def _macro_of(text, needle):
    """The GGML_LOG_* macro of every call whose format string starts with `needle`."""
    return re.findall(r"(GGML_LOG_[A-Z]+)\(\s*\"" + re.escape(needle), text)


def _routed_levels(src):
    """The GGML log levels route_llama_log sends to arcint's error or warn line."""
    i = src.find("void route_llama_log(")
    assert i >= 0, "route_llama_log is absent"
    body = src[i:src.find("\n}\n", i)]
    shown = set(re.findall(r"level == GGML_LOG_LEVEL_([A-Z]+)\)\s*log::(?:error|warn)\(", body))
    assert "log::verbose" in body, "route_llama_log no longer has a verbose fallback: re-read this cell"
    return shown


class DepthMixTakenLine(unittest.TestCase):

    def setUp(self):
        self.cl = (_llama_root() / OPENCL).read_text()
        self.routing = ROUTING.read_text()

    def test_text_is_unchanged_and_printed_once_in_source(self):
        self.assertEqual(self.cl.count('"' + TEXT + "\\n\""), 1,
                         "one call prints the depth mix's TAKEN line, with the README's text")

    def test_level_is_the_fa_ports(self):
        port = set(_macro_of(self.cl, "TAKEN src=opencl op=FLASH_ATTN port=fa_kern"))
        self.assertEqual(len(port), 1, "the FA port's TAKEN lines share one level: %s" % sorted(port))
        mix = _macro_of(self.cl, TEXT)
        self.assertEqual(mix, list(port), "the depth mix's TAKEN line is at %s, the FA port's at %s"
                         % (mix, sorted(port)))

    def test_level_reaches_the_served_log(self):
        mix = _macro_of(self.cl, TEXT)
        self.assertEqual(len(mix), 1)
        level = mix[0][len("GGML_LOG_"):]
        self.assertIn(level, _routed_levels(self.routing),
                      "route_llama_log sends GGML_LOG_%s to the verbose level only" % level)

    def test_once_a_process(self):
        i = self.cl.find('"' + TEXT)
        self.assertGreaterEqual(i, 0)
        head = self.cl[max(0, i - 400):i]
        m = re.search(r"static std::once_flag (\w+);\s*std::call_once\(\1,[^{]*\{\s*GGML_LOG_[A-Z]+\($", head)
        self.assertIsNotNone(m, "the line is not printed inside a std::call_once on a static once_flag")


if __name__ == "__main__":
    unittest.main()
