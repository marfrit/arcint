#!/usr/bin/env python3
"""tools/fa_port_dump_compare.py on synthetic dump directories: green cases and one red per failure class."""
import contextlib
import io
import os
import struct
import tempfile
import unittest
from pathlib import Path

import importlib.util

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("fa_port_dump_compare", ROOT / "tools" / "fa_port_dump_compare.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

VALS = [0.5 + 0.01 * i for i in range(64)]


def dump(d, cases, dup=None):
    """cases: {hash: floats}; dup: extra raw index text appended."""
    d.mkdir()
    with open(d / "index.txt", "w") as ix:
        for h, v in cases.items():
            (d / (h + ".f32")).write_bytes(struct.pack("<%df" % len(v), *v))
            ix.write("%s %d case vars=%s\n" % (h, len(v), h))
        if dup:
            ix.write(dup)


class Compare(unittest.TestCase):

    def run_cmp(self, a, b, bound=None):
        with tempfile.TemporaryDirectory() as t:
            da, db = Path(t) / "a", Path(t) / "b"
            dump(da, a[0], a[1] if len(a) > 1 else None)
            dump(db, b[0], b[1] if len(b) > 1 else None)
            out = io.StringIO()
            args = [str(da), str(db)] + (["--bound", str(bound)] if bound else [])
            with contextlib.redirect_stdout(out):
                rc = tool.main(args)
            return rc, out.getvalue()

    def test_equal_passes(self):
        rc, out = self.run_cmp(({"h1": VALS, "h2": VALS},), ({"h1": VALS, "h2": VALS},))
        self.assertEqual(rc, 0, out)
        self.assertIn("FA PORT DUMP COMPARE: PASS (2 cases", out)

    def test_small_noise_passes(self):
        noisy = [v * (1 + 1e-5) for v in VALS]   # nmse 1e-10
        rc, out = self.run_cmp(({"h1": noisy},), ({"h1": VALS},))
        self.assertEqual(rc, 0, out)

    def test_missing_case_fails(self):
        rc, out = self.run_cmp(({"h1": VALS, "h2": VALS},), ({"h1": VALS},))
        self.assertEqual(rc, 1)
        self.assertIn("case only in A", out)
        self.assertIn("COMPARE: FAIL", out)

    def test_count_mismatch_fails(self):
        rc, out = self.run_cmp(({"h1": VALS},), ({"h1": VALS[:-1]},))
        self.assertEqual(rc, 1)
        self.assertIn("float counts differ", out)

    def test_nan_fails(self):
        bad = list(VALS)
        bad[3] = float("nan")
        rc, out = self.run_cmp(({"h1": bad},), ({"h1": VALS},))
        self.assertEqual(rc, 1)
        self.assertIn("non-finite", out)

    def test_over_bound_fails(self):
        off = [v * 1.01 for v in VALS]            # nmse 1e-4
        rc, out = self.run_cmp(({"h1": off},), ({"h1": VALS},))
        self.assertEqual(rc, 1)
        self.assertIn("over the bound", out)
        rc, out = self.run_cmp(({"h1": off},), ({"h1": VALS},), bound=1e-3)
        self.assertEqual(rc, 0, out)

    def test_differing_duplicate_fails(self):
        rc, out = self.run_cmp(({"h1": VALS}, "h1 %d other vars\n" % len(VALS)), ({"h1": VALS},))
        self.assertEqual(rc, 1)
        self.assertIn("listed twice with different lines", out)

    def test_identical_duplicate_is_reported_not_failed(self):
        rc, out = self.run_cmp(({"h1": VALS}, "h1 %d case vars=h1\n" % len(VALS)), ({"h1": VALS},))
        self.assertEqual(rc, 0, out)
        self.assertIn("listed again, identical", out)


if __name__ == "__main__":
    unittest.main()
