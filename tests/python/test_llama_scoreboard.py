#!/usr/bin/env python3
"""tools/llama_scoreboard.py: what makes arcint's row comparable, device-free.

Red first, each by a mutation of the tool:
  - a prefill the prompt cache helped is refused (red: the cache-hit check removed:
    the fastest number the tool could print, and a false one);
  - a prompt of the wrong length is refused (red: the token check removed);
  - a generation that stopped short is refused (red: the n_gen check removed);
  - a counter missing from /metrics is refused (red: the missing check removed, a
    KeyError or a row from nothing);
  - the rate divides by the phase's own timer (red: tg_rate reads the prefill timer);
  - the server runs the tables' context and KV, and only an arm asks for another
    (red: --n-ctx changed, or --llama-kv always passed).
"""
import importlib.util
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("llama_scoreboard", os.path.join(HERE, "..", "..", "tools", "llama_scoreboard.py"))
sb = importlib.util.module_from_spec(spec)
sys.modules["llama_scoreboard"] = sb
spec.loader.exec_module(sb)


def counters(prompt=0.0, completion=0.0, hit=0.0, prefill=0.0, decode=0.0):
    return {"arcint_prompt_tokens_total": prompt, "arcint_completion_tokens_total": completion,
            "arcint_prompt_cache_hit_tokens_total": hit, "arcint_prefill_seconds_total": prefill,
            "arcint_decode_seconds_total": decode}


class Rates(unittest.TestCase):
    def test_pp_from_the_prefill_timer(self):
        self.assertAlmostEqual(sb.pp_rate(counters(prompt=100, prefill=1.0, decode=9.0),
                                          counters(prompt=612, prefill=1.256, decode=30.0)), 2000.0, places=6)

    def test_cache_hit_refused(self):
        with self.assertRaises(sb.NotComparable):
            sb.pp_rate(counters(), counters(prompt=512, hit=500, prefill=0.01))

    def test_wrong_prompt_length_refused(self):
        with self.assertRaises(sb.NotComparable):
            sb.pp_rate(counters(), counters(prompt=513, prefill=0.2))

    def test_tg_from_the_decode_timer(self):
        self.assertAlmostEqual(sb.tg_rate(counters(completion=1, decode=1.0, prefill=5.0),
                                          counters(completion=129, decode=2.28, prefill=99.0)), 100.0, places=6)

    def test_short_generation_refused(self):
        with self.assertRaises(sb.NotComparable):
            sb.tg_rate(counters(), counters(completion=60, decode=1.0))


class Metrics(unittest.TestCase):
    TEXT = """# HELP arcint_prompt_tokens_total Prompt tokens processed.
# TYPE arcint_prompt_tokens_total counter
arcint_prompt_tokens_total 512
arcint_completion_tokens_total 1
arcint_prompt_cache_hit_tokens_total 0
arcint_prefill_seconds_total 0.25
arcint_decode_seconds_total{slot="0"} 0.5
arcint_something_else 7
"""

    def test_parse(self):
        m = sb.parse_metrics(self.TEXT)
        self.assertEqual(m["arcint_prompt_tokens_total"], 512.0)
        self.assertEqual(m["arcint_decode_seconds_total"], 0.5)
        self.assertNotIn("arcint_something_else", m)

    def test_missing_counter_refused(self):
        with self.assertRaises(sb.NotComparable):
            sb.parse_metrics("arcint_prompt_tokens_total 512\n")


class Server(unittest.TestCase):
    def test_the_tables_configuration(self):
        cmd = sb.server_argv("arcint", "m.gguf", "GPU.1", 8095, "f16")
        self.assertEqual(cmd[cmd.index("--n-ctx") + 1], "4096")
        self.assertEqual(cmd[cmd.index("--device") + 1], "GPU.1")
        self.assertNotIn("--llama-kv", cmd)
        self.assertNotIn("--llama-mtp", cmd)

    def test_an_arm_names_its_kv(self):
        cmd = sb.server_argv("arcint", "m.gguf", "GPU.0", 8095, "q8_0:q4_0")
        self.assertEqual(cmd[cmd.index("--llama-kv") + 1], "q8_0:q4_0")

    def test_row(self):
        self.assertEqual(sb.row("Arc A770", "16 GB", (2000.0, 1.5), (90.0, 0.25), "arcint 1.5.0 (98619b0a)", "f16"),
                         "| Arc A770 | 16 GB | 2000.00 ± 1.50 | 90.00 ± 0.25 | arcint 1.5.0 (98619b0a) | f16 |")


if __name__ == "__main__":
    unittest.main()
