#!/usr/bin/env python3
"""Red-first, device-free cells for patch 0046's census-seed parser.

The parser lives in the OpenVINO plugin patch
`patches/0046-moe-cpu-tier-census-seed.patch` (new file `census_seed.hpp`).
This test EXTRACTS that file from the patch, compiles it with a plain g++
against a small driver, and runs the driver once per case. No OpenVINO build,
no card: the header is deliberately dependency-free, so the refusal behaviour
is exercised on the exact bytes the plugin will compile, not on a Python
re-implementation.

Every case is one cell. The malformed and mismatched cases are the red-first
refusals: each must throw, not default to the frequency-free seed.
"""

import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, "..", "patches",
                     "0046-moe-cpu-tier-census-seed.patch")
HEADER_NAME = "census_seed.hpp"

DRIVER = r"""
#include "census_seed.hpp"
#include <iostream>
#include <string>
#include <vector>

using namespace ov::intel_gpu::ocl::moe;

static int g_failures = 0;
#define CHECK(name, cond) do { \
    if (cond) { std::cout << "PASS " << (name) << "\n"; } \
    else { std::cout << "FAIL " << (name) << "\n"; ++g_failures; } \
} while (0)

static bool parse_ok(const std::string& text) {
    try { parse_census_seed(text); return true; } catch (...) { return false; }
}

static std::string valid() {
    return "# hot-set seed v2 (frequency rank, id tie-break)\n"
           "# space=layer_key\n"
           "# layer_key_by_index={\"0\": 704}\n"
           "704 5 3 9\n"
           "705 1 2\n";
}

int main(int argc, char** argv) {
    const std::string c = (argc > 1) ? argv[1] : "";
    if (c == "valid_parse") {
        CensusSeed s = parse_census_seed(valid());
        CHECK("valid_parse.membership",
              s.by_layer_key.count(704) && s.by_layer_key.count(705) &&
              s.by_layer_key[704] == std::vector<uint32_t>({5, 3, 9}));
    } else if (c == "missing_space_header_refused") {
        CHECK("missing_space_header_refused",
              !parse_ok("704 5 3\n"));
    } else if (c == "wrong_space_refused") {
        CHECK("wrong_space_refused",
              !parse_ok("# space=layer\n704 5 3\n"));
    } else if (c == "duplicate_layer_key_refused") {
        CHECK("duplicate_layer_key_refused",
              !parse_ok("# space=layer_key\n704 5\n704 3\n"));
    } else if (c == "duplicate_expert_refused") {
        CHECK("duplicate_expert_refused",
              !parse_ok("# space=layer_key\n704 5 5\n"));
    } else if (c == "non_numeric_layer_key_refused") {
        CHECK("non_numeric_layer_key_refused",
              !parse_ok("# space=layer_key\nabc 5\n"));
    } else if (c == "non_numeric_expert_refused") {
        CHECK("non_numeric_expert_refused",
              !parse_ok("# space=layer_key\n704 5x\n"));
    } else if (c == "empty_data_refused") {
        CHECK("empty_data_refused",
              !parse_ok("# space=layer_key\n# only comments\n"));
    } else if (c == "missing_expert_refused") {
        CHECK("missing_expert_refused",
              !parse_ok("# space=layer_key\n704\n"));
    } else if (c == "resident_valid_sorted") {
        CensusSeed s = parse_census_seed(valid());
        auto r = census_seed_resident_experts(s, 704, 512, 3);
        CHECK("resident_valid_sorted", r == std::vector<uint32_t>({3, 5, 9}));
    } else if (c == "resident_missing_layer_refused") {
        CensusSeed s = parse_census_seed(valid());
        bool threw = false;
        try { census_seed_resident_experts(s, 999, 512, 3); }
        catch (const std::runtime_error&) { threw = true; }
        CHECK("resident_missing_layer_refused", threw);
    } else if (c == "resident_capacity_mismatch_refused") {
        CensusSeed s = parse_census_seed(valid());
        bool threw = false;
        try { census_seed_resident_experts(s, 704, 512, 2); }  // seed has 3
        catch (const std::runtime_error&) { threw = true; }
        CHECK("resident_capacity_mismatch_refused", threw);
    } else if (c == "resident_expert_out_of_range_refused") {
        CensusSeed s = parse_census_seed(valid());
        bool threw = false;
        try { census_seed_resident_experts(s, 704, 6, 3); }  // expert 9 >= 6
        catch (const std::runtime_error&) { threw = true; }
        CHECK("resident_expert_out_of_range_refused", threw);
    } else if (c == "fingerprint_stable") {
        const std::string a = valid();
        CHECK("fingerprint_stable",
              census_seed_fingerprint(a) == census_seed_fingerprint(a) &&
              census_seed_fingerprint(a) != census_seed_fingerprint(a + "\n"));
    } else {
        std::cout << "FAIL unknown case " << c << "\n";
        ++g_failures;
    }
    return g_failures == 0 ? 0 : 1;
}
"""


def _extract_header(patch_path):
    """The added `census_seed.hpp` body, from the patch's `+` lines."""
    with open(patch_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("+++ b/") and ln.endswith(HEADER_NAME):
            start = i + 1
            break
    if start is None:
        raise RuntimeError(
            f"{patch_path}: no added {HEADER_NAME} in the patch")
    body = []
    for ln in lines[start:]:
        if ln.startswith("diff --git "):
            break
        if ln.startswith("@@"):
            continue
        if ln.startswith("+"):
            body.append(ln[1:])
        elif ln.startswith("\\"):
            continue
    if not body:
        raise RuntimeError(f"{patch_path}: {HEADER_NAME} body is empty")
    return "\n".join(body) + "\n"


class TestCensusSeedParser(unittest.TestCase):
    _bin = None

    @classmethod
    def setUpClass(cls):
        patch = os.path.abspath(PATCH)
        if not os.path.exists(patch):
            raise unittest.SkipTest(f"patch not found: {patch}")
        cls._tmp = tempfile.TemporaryDirectory()
        header = _extract_header(patch)
        with open(os.path.join(cls._tmp.name, HEADER_NAME), "w") as f:
            f.write(header)
        with open(os.path.join(cls._tmp.name, "driver.cpp"), "w") as f:
            f.write(DRIVER)
        cxx = os.environ.get("CXX", "g++")
        cls._bin = os.path.join(cls._tmp.name, "driver")
        subprocess.run([cxx, "-std=c++17", "-Wall", "-Wextra", "-O0",
                        "-I", cls._tmp.name,
                        os.path.join(cls._tmp.name, "driver.cpp"),
                        "-o", cls._bin],
                       check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _run(self, case):
        p = subprocess.run([self._bin, case], capture_output=True, text=True)
        return p.returncode, p.stdout

    def test_valid_parse(self):
        rc, out = self._run("valid_parse")
        self.assertEqual(rc, 0, out)

    def test_missing_space_header_refused(self):
        rc, out = self._run("missing_space_header_refused")
        self.assertEqual(rc, 0, out)

    def test_wrong_space_refused(self):
        rc, out = self._run("wrong_space_refused")
        self.assertEqual(rc, 0, out)

    def test_duplicate_layer_key_refused(self):
        rc, out = self._run("duplicate_layer_key_refused")
        self.assertEqual(rc, 0, out)

    def test_duplicate_expert_refused(self):
        rc, out = self._run("duplicate_expert_refused")
        self.assertEqual(rc, 0, out)

    def test_non_numeric_layer_key_refused(self):
        rc, out = self._run("non_numeric_layer_key_refused")
        self.assertEqual(rc, 0, out)

    def test_non_numeric_expert_refused(self):
        rc, out = self._run("non_numeric_expert_refused")
        self.assertEqual(rc, 0, out)

    def test_empty_data_refused(self):
        rc, out = self._run("empty_data_refused")
        self.assertEqual(rc, 0, out)

    def test_missing_expert_refused(self):
        rc, out = self._run("missing_expert_refused")
        self.assertEqual(rc, 0, out)

    def test_resident_valid_sorted(self):
        rc, out = self._run("resident_valid_sorted")
        self.assertEqual(rc, 0, out)

    def test_resident_missing_layer_refused(self):
        rc, out = self._run("resident_missing_layer_refused")
        self.assertEqual(rc, 0, out)

    def test_resident_capacity_mismatch_refused(self):
        rc, out = self._run("resident_capacity_mismatch_refused")
        self.assertEqual(rc, 0, out)

    def test_resident_expert_out_of_range_refused(self):
        rc, out = self._run("resident_expert_out_of_range_refused")
        self.assertEqual(rc, 0, out)

    def test_fingerprint_stable(self):
        rc, out = self._run("fingerprint_stable")
        self.assertEqual(rc, 0, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
