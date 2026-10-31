"""Patch 0070's STRP save: the C++ round trip, read back by the STRP reader.

The cell compiles tools/expert_profile_save_selftest.cpp against the llama
tree's src directory given by EXPERT_PROFILE_SRC, runs it, and reads the file
it leaves with tools/expert_profile_from_trace.py's reader (the same STRP
header + pairs the cache reads; a struct fallback if that tool has no reader).
Skipped with a clear message when EXPERT_PROFILE_SRC is unset, so the suite is
green without the llama tree.

Standard library only (unittest), so the recipe's `ctest -L unit` runs it on a
build host with python3 and nothing else: pytest is not a build dependency.
pytest collects the unittest class too, for anyone running the whole tests/
tree with it.

The selftest itself carries the red mutations (see its header): (a) rank's
heat comparator `>` -> `<`, the same-placement cell fails; (b) drop the version
check, the version cell fails; (c) accept a truncated record, the truncation
cell fails; (h) heat counts a decayed weight, the raw-count cell fails; (i) heat
narrowed to float, the 2^24 cell fails.
"""
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

# the selftest's geometry and its expected ranking (Strata's
# rank_learned_profile on the synthetic resident/heat/prior it builds)
N_LAYER, N_EXPERT = 3, 4
EXPECTED = [
    (1, 1), (0, 0), (0, 2), (2, 3), (2, 0), (1, 3), (0, 1),
    (0, 3), (1, 0), (1, 2), (2, 1), (2, 2),
]


def _src_dir():
    raw = os.environ.get("EXPERT_PROFILE_SRC")
    if not raw:
        raise unittest.SkipTest("EXPERT_PROFILE_SRC is unset: the llama tree with patch 0070 is not available")
    root = Path(raw)
    for cand in (root, root / "src"):
        if (cand / "llama-expert-profile.cpp").exists() and (cand / "llama-expert-profile.h").exists():
            return cand
    raise unittest.SkipTest("EXPERT_PROFILE_SRC has no llama-expert-profile.{cpp,h}: the tree lacks patch 0070")


def _read_profile(path):
    """(ranked pairs, slots hint). The trace tool's reader when it exists."""
    try:
        import expert_profile_from_trace as epft
    except Exception:
        epft = None
    if epft is not None and hasattr(epft, "read_profile"):
        return epft.read_profile(path, N_LAYER, N_EXPERT)
    blob = Path(path).read_bytes()
    assert blob[:4] == b"STRP", "magic"
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    assert (ver, nl, ne) == (1, N_LAYER, N_EXPERT), (ver, nl, ne)
    ranked = [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]
    return ranked, slots


class ExpertProfileSave(unittest.TestCase):
    def test_expert_profile_save_round_trip(self):
        src = _src_dir()
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            exe = tmp / "expert_profile_save_selftest"
            r = subprocess.run(["g++", "-std=c++17", "-O1", "-I", str(src),
                                str(REPO_ROOT / "tools" / "expert_profile_save_selftest.cpp"),
                                str(src / "llama-expert-profile.cpp"), "-o", str(exe)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            out = tmp / "saved-profile.bin"
            run = subprocess.run([str(exe), str(out)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("expert_profile_save_selftest: OK", run.stdout)
            self.assertTrue(out.exists())
            self.assertFalse(Path(str(out) + ".tmp").exists())

            ranked, slots = _read_profile(out)
            self.assertEqual(tuple(EXPECTED), tuple(ranked), "the saved ranking is not Strata's rule")
            self.assertEqual(slots, len(ranked))
            self.assertEqual(len(ranked), N_LAYER * N_EXPERT)

            blob = out.read_bytes()
            ver, nl, ne, hdr_slots, n = struct.unpack_from("<5I", blob, 4)
            self.assertEqual((blob[:4], ver, nl, ne, hdr_slots, n),
                             (b"STRP", 1, N_LAYER, N_EXPERT, len(EXPECTED), len(EXPECTED)))
            self.assertEqual(len(blob), 24 + 4 * n + 4 * N_LAYER * N_EXPERT,
                             "header, pairs, then the rank table")
            table = struct.unpack_from("<%di" % (N_LAYER * N_EXPERT), blob, 24 + 4 * n)
            want = [-1] * (N_LAYER * N_EXPERT)
            for rank, (layer, e) in enumerate(ranked):
                want[layer * N_EXPERT + e] = rank
            self.assertEqual(list(table), want,
                             "the trailing table is the ranking's own ranks (Strata's writer)")

            # the same ranking through the Python writer (make_profile.py's format):
            # byte for byte, so a saved file and a trace-built file are one format
            try:
                import expert_profile_from_trace as ept
            except Exception:
                raise unittest.SkipTest("expert_profile_from_trace is not importable from tools/")
            py = tmp / "py-profile.bin"
            ept.write_profile(py, ranked, N_LAYER, N_EXPERT)
            self.assertEqual(py.read_bytes(), blob,
                             "the C++ save and make_profile.py's writer agree byte for byte")

    def test_heat_counts_in_count_not_adapt(self):
        """The save's heat is counted in the cache's count(), one a routed pair,
        never in adapt(): LLAMA_EXPERT_CACHE_ADAPT_EVERY=0 must still accumulate
        it. Red mutation (seen failing after the move): put the
        llama_expert_profile_add_heat call back inside adapt() -- the assert on
        count() fails, and the adapt() assert fails too."""
        src = _src_dir()
        text = (src / "llama-expert-cache.cpp").read_text()

        def body(name):
            i = text.index("void llama_expert_cache::%s(" % name)
            j = text.find("\nvoid llama_expert_cache::", i + 1)
            return text[i:j if j >= 0 else len(text)]

        self.assertIn("llama_expert_profile_add_heat", body("count"),
                      "the save's heat is counted in count()")
        self.assertNotIn("llama_expert_profile_add_heat", body("adapt"),
                         "the save's heat is not counted in adapt(): adapt-off must still count")


if __name__ == "__main__":
    unittest.main()
