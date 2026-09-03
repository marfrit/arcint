#!/usr/bin/env python3
"""CPU-only red/green probe for the DFlash2 stateful IR's KV-state trim.

Drives a compiled `openvino_dflash_draft_stateful.xml` on device="CPU" only
(no GPU device anywhere) and checks two things after every step:

  1. inference succeeds (this is the red case: the GPU-only layout-mismatch
     bug this probe was written against may or may not reproduce on CPU --
     see the script's own printed verdict);
  2. every per-layer KV state variable's sequence length (state.shape[2])
     equals min(total_rows_appended_so_far, window) -- i.e. the trim keeps
     exactly the last `window` rows and no more, before and after the state
     fills.

Four step patterns, each starting from a freshly reset state:
  - pattern A: three steps of 900 rows (900, 900, 900 -> 2700 total; crosses
    `window` partway through the third step)
  - pattern B: one single step of 2300 rows (a prompt over `window` tokens
    fed in a single dflash_append call)
  - pattern C: two steps, 433 then 1,615 rows, landing the concatenated
    length exactly on `window` (2,048) after the second step -- the
    measured trigger shape for the GPU-only "Layout mismatch" (see
    patches/0014-gpu-assign-adopts-output-layout.patch and
    docs/dflash-pairing-probe.md); CPU does not reproduce that failure, so
    this pattern's value here is the state-length assertion at the exact
    boundary, not the red case itself
  - pattern D: one single step of exactly `window` (2,048) rows, the same
    boundary reached in one call instead of two

Patterns A and B run well past `window` (2,700 and 2,300 total rows) and
so pin the "one row too many" direction: an off-by-one that kept
`window + 1` rows instead of exactly `window` once the state has long
since filled would show up as a state length one over `expect` on every
step after the crossing. Patterns C and D land the concatenated length on
exactly `window`, pinning the opposite direction, "one too few": an
off-by-one in the runtime start (`max(0, len(cat) - window)` computed one
row short) would drop the newest row right at the boundary and show a
state length one under `expect`, a case A/B's later, larger totals cannot
distinguish from a correct trim. All four patterns have been run on the
container this probe targets (CPU, python3, no GPU device) -- see
docs/dflash-pairing-probe.md for the recorded output, kept honest rather
than asserted from reading the code.

This probe checks CPU *behavior* (does inference run, do the state lengths
come out right), not IR *content*. For a byte-level check that a re-export
did not change anything beyond what it was supposed to change -- the
"bit-exact old vs new" comparison -- use tools/dflash_compare_ir.py against
the two stateful IRs' .xml files; that is the reproducible form of that
check, this probe does not repeat it.

Usage:
    python3 tools/dflash_state_probe.py --dir <export-dir> \
        [--xml openvino_dflash_draft_stateful.xml] [--window 2048]
"""
import argparse
import os
import sys

import numpy as np
import openvino as ov

HIDDEN, HEAD_DIM = 5120, 128
N_FEATS = 5 * HIDDEN


def run_pattern(compiled, label, steps, window):
    print(f"  -- {label}: steps={steps} --")
    req = compiled.create_infer_request()
    req.reset_state()
    rng = np.random.default_rng(0)
    total = 0
    all_ok = True
    for i, p in enumerate(steps):
        feats = rng.standard_normal((1, p, N_FEATS)).astype(np.float32)
        noise = rng.standard_normal((1, 1, HIDDEN)).astype(np.float32)
        positions = np.arange(total, total + p + 1, dtype=np.int64)
        try:
            req.infer({"new_feats": feats, "noise": noise, "positions": positions})
        except Exception as e:  # noqa: BLE001 -- this failure IS the measurement
            print(f"     step {i} (+{p} rows, total would be {total + p}): "
                  f"INFER FAILED: {type(e).__name__}: {e}")
            return False
        total += p
        expect = min(total, window)
        seqs = sorted({int(st.state.shape[2]) for st in req.query_state()})
        status = "OK" if seqs == [expect] else "MISMATCH"
        print(f"     step {i} (+{p} rows, total {total}): infer OK, "
              f"state seq lens = {seqs} (expect [{expect}]) -- {status}")
        if seqs != [expect]:
            all_ok = False
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="export directory")
    ap.add_argument("--xml", default="openvino_dflash_draft_stateful.xml")
    ap.add_argument("--window", type=int, default=2048)
    args = ap.parse_args()

    path = os.path.join(args.dir, args.xml)
    core = ov.Core()
    model = core.read_model(path)
    compiled = core.compile_model(model, "CPU")

    print(f"=== {path} on CPU (window={args.window}) ===")
    a_ok = run_pattern(compiled, "pattern A (900,900,900)", [900, 900, 900], args.window)
    b_ok = run_pattern(compiled, "pattern B (single 2300-row step)", [2300], args.window)
    c_ok = run_pattern(compiled, "pattern C (433,1615 -- lands exactly on window)",
                        [433, 1615], args.window)
    d_ok = run_pattern(compiled, "pattern D (single window-row step)", [args.window], args.window)

    if a_ok and b_ok and c_ok and d_ok:
        print("RESULT: PASS on CPU (infer succeeded and state lengths matched "
              "min(total, window) at every step -- if this is the OLD export, "
              "that means the CPU plugin does not reproduce the GPU's layout "
              "refusal; the GPU run remains the actual red case)")
    else:
        print("RESULT: FAIL on CPU")
        sys.exit(1)


if __name__ == "__main__":
    main()
