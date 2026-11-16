"""Patch 0130l's gate: no kernel source 0130a-0130k touched may fail the C preprocessor.

0130k put `#ifdef GQA_NOPRIV` / `#else` / `#endif` inside the backslash-continued body of
FA_GQA_STAGE_V and FA_GQA_STORE_V. In C a `#` inside a macro body is the stringize operator,
so `cpp` fails the whole file at 8236e31d -- and the file is the production GQA decode
kernel, NOPRIV off. This gate runs `cpp -x c -P` over flash_attn_gqa_dpas.cl under every
define set the host builds (base, I8, ST, V2, PRE, PRE+BROKEN, NOPRIV, NOPRIV+BROKEN,
NOPRIV+PRE) at DK=256, GQA_SIDES=2 and the host's K/V codes q8_0 K / q4_0 V (KT=1, VT=2) and
f16 (KT=0, VT=0), and fails on any nonzero rc or any output on stderr.

Red first: on 8236e31d's file the base set prints
  "flash_attn_gqa_dpas.cl:276:32: error: '#' is not followed by a macro parameter"
and rc 1. Green: rc 0 for every set.
"""
import re
import shutil
import subprocess
import unittest
from pathlib import Path

KERNELS = ["ggml/src/ggml-opencl/kernels/flash_attn_gqa_dpas.cl"]
SETS = [
    ("base", []),
    ("i8", ["-DGQA_I8=1"]),
    ("st", ["-DGQA_I8=1", "-DGQA_ST=1"]),
    ("v2", ["-DGQA_V2=1"]),
    ("pre", ["-DGQA_PRE=1"]),
    ("pre_broken", ["-DGQA_PRE=1", "-DGQA_PRE_BROKEN=1"]),
    ("nopriv", ["-DGQA_NOPRIV=1"]),
    ("nopriv_broken", ["-DGQA_NOPRIV=1", "-DGQA_NOPRIV_BROKEN=1"]),
    ("nopriv_pre", ["-DGQA_NOPRIV=1", "-DGQA_PRE=1"]),
]
KV = [("q8q4", ["-DKT=1", "-DVT=2"]), ("f16", ["-DKT=0", "-DVT=0"])]


def _llama_root():
    import os
    raw = os.environ.get("LLAMA_SRC")
    if not raw:
        raise unittest.SkipTest("LLAMA_SRC is unset: the patched llama tree is not available")
    root = Path(raw)
    for cand in (root, root.parent):
        if (cand / KERNELS[0]).exists():
            return cand
    raise unittest.SkipTest("LLAMA_SRC has no flash_attn_gqa_dpas.cl")


@unittest.skipUnless(shutil.which("cpp"), "cpp is not installed")
class KernelPreproc(unittest.TestCase):

    def setUp(self):
        self.root = _llama_root()

    def test_every_define_set_preprocesses(self):
        base = ["-DDK=256", "-DGQA_SIDES=2", "-x", "c", "-P"]
        bad = []
        for k in KERNELS:
            for kvname, kv in KV:
                for name, d in SETS:
                    p = self.root / k
                    r = subprocess.run(["cpp", *base, *kv, *d, str(p)], capture_output=True, text=True)
                    if r.returncode != 0 or r.stderr.strip():
                        bad.append((k, kvname, name, r.returncode, r.stderr.strip()[:160]))
                    # 0130o: a macro parameter left in the output = a body that ended early
                    # (dafb1a4b: the NOPRIV stores outside FA_GQA_STAGE_V/STORE_V, which cpp
                    # accepts and ocloc rejects: "use of undeclared identifier 'dst'")
                    elif re.search(r"\b(dst|k0i)\b", r.stdout):
                        bad.append((k, kvname, name, "residue", "macro parameter dst/k0i after cpp"))
        self.assertEqual(bad, [], "a kernel failed the preprocessor: %r" % (bad,))


if __name__ == "__main__":
    unittest.main()
