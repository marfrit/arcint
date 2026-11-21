"""The patch series applies in byte order, whatever the caller's locale (GitHub issue #6).

contrib/llama.cpp/apply-series.sh lists and applies contrib/llama.cpp/patches/*.patch. Its
order must be the byte order of the names: 0129- before 0129a- ... 0129k-, and 0130b- before
0130b1-, because each later patch is written against the tree the earlier one leaves. Under
en_US.UTF-8 a plain shell glob ignores the punctuation and puts 0129-* after 0129k-* and
0130b1-* before 0130b-*; the v1.7.0 series then stops at patch 55, 0129b (measured on the
recipe host: en_US.utf8 applied 55, first failure 0129b; C applied all 98).

Red first: with the LC_ALL=C lines removed from apply-series.sh, test_list_is_byte_order fails
under en_US.UTF-8 with 0129b listed before 0129-. Green: the list equals the sorted byte order.
A host without a non-C locale skips the locale cases (there is nothing to sort differently).
"""
import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "contrib" / "llama.cpp" / "apply-series.sh"
PATCHES = ROOT / "contrib" / "llama.cpp" / "patches"


def available_locales():
    try:
        out = subprocess.run(["locale", "-a"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [l.strip() for l in out.splitlines() if l.strip()]


def pick_locale(names):
    have = {l.lower().replace("-", ""): l for l in available_locales()}
    for n in names:
        key = n.lower().replace("-", "")
        if key in have:
            return have[key]
    return None


def listed(locale_name):
    env = {k: v for k, v in os.environ.items() if not k.startswith("LC_") and k != "LANG"}
    env["LANG"] = locale_name
    env["LC_ALL"] = locale_name
    out = subprocess.run(["bash", str(SCRIPT), "--list"], capture_output=True, text=True, env=env, check=True)
    return [Path(l).name for l in out.stdout.splitlines()]


class PatchSeriesOrder(unittest.TestCase):
    def setUp(self):
        self.names = sorted(p.name for p in PATCHES.glob("*.patch"))  # Python sorts str by code point = bytes for ASCII
        self.assertTrue(self.names, "no patches found")

    def test_list_in_c_locale(self):
        self.assertEqual(listed("C"), self.names)

    def test_list_is_byte_order(self):
        loc = pick_locale(["en_US.utf8", "en_US.UTF-8", "de_DE.utf8", "de_DE.UTF-8", "C.utf8"])
        if loc is None or loc.lower().startswith("c."):
            self.skipTest("no en_US/de_DE locale on this host: no collation that differs from byte order")
        self.assertEqual(listed(loc), self.names, f"order under {loc} differs from byte order")

    def test_known_pairs(self):
        order = {n.split("-")[0]: i for i, n in enumerate(self.names)}
        for first, then in [("0129", "0129a"), ("0129", "0129b"), ("0130b", "0130b1")]:
            if first in order and then in order:
                self.assertLess(order[first], order[then], f"{first} must precede {then}")

    def test_apply_refuses_missing_tree(self):
        r = subprocess.run(["bash", str(SCRIPT), "/nonexistent-tree-for-the-test"], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no directory", r.stderr)


if __name__ == "__main__":
    unittest.main()
