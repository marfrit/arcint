#!/usr/bin/env python3
"""tools/downgrade_check.py: what fails a unit, device-free."""
import importlib.util
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "..", "tools", "downgrade_check.py")
spec = importlib.util.spec_from_file_location("downgrade_check", TOOL)
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)


class Compare(unittest.TestCase):
    def test_none_is_ok(self):
        self.assertEqual(dc.compare({"downgrades": []}, set()), (True, [], []))

    def test_a_new_key_fails_and_is_named(self):
        ok, new, _ = dc.compare({"downgrades": [{"key": "a", "count": 7}]}, set())
        self.assertFalse(ok)
        self.assertEqual(new, [("a", 7)])

    def test_an_accepted_key_passes(self):
        self.assertTrue(dc.compare({"downgrades": [{"key": "a", "count": 7}]}, {"a"})[0])

    def test_a_new_key_beside_an_accepted_one_fails(self):
        ok, new, _ = dc.compare({"downgrades": [{"key": "a", "count": 1}, {"key": "b", "count": 2}]}, {"a"})
        self.assertFalse(ok)
        self.assertEqual(new, [("b", 2)])

    def test_an_accepted_key_that_did_not_occur_is_said_and_passes(self):
        ok, new, absent = dc.compare({"downgrades": []}, {"a"})
        self.assertTrue(ok)
        self.assertEqual(new, [])
        self.assertEqual(absent, ["a"])

    def test_a_unit_without_the_field_fails(self):
        # an old binary must not read as clean
        ok, new, _ = dc.compare({"build": {}}, set())
        self.assertFalse(ok)
        self.assertIsNone(new)


class Table(unittest.TestCase):
    def test_the_table_names_the_three_units(self):
        with open(os.path.join(HERE, "..", "acceptance", "expected-downgrades.json")) as f:
            t = json.load(f)
        self.assertEqual(sorted(t), ["agent", "coder", "flashnext"])
        # the operator accepted Flash-Next's keys; these two carry nothing
        self.assertEqual(t["coder"], [])
        self.assertEqual(t["agent"], [])


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
