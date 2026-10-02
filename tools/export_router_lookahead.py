#!/usr/bin/env python3
"""Write the router matrices of a MoE GGUF for the CPU tier's router lookahead.

The plugin's lookahead (MOE_CPU_TIER_LOOKAHEAD, patch 0076) predicts layer
L+1's experts by applying L+1's router to layer L's MoE input, and asks the
kernel for the pages of the predicted experts the card and the host bank do
not hold -- Strata's routing-aware prefetch of the file tier
(src/program/generate.cpp:3001-3030, src/core/expert_source.cpp:814-930). The
plugin knows a layer by its layer_key (the .bin offset of its first expert
weight), so the matrices are written keyed by it.

Output: "ARCRLA01", then uint32 n_layers, n_expert, n_embd, 0; then per layer
uint64 layer_key and n_expert x n_embd float32 (row e = expert e's router row).

The layer_key of each block comes from a seed file's
`# layer_key_by_index={...}` line (tools/hot_set_census.py writes it).
"""
import argparse
import glob
import json
import re
import struct
import sys

import numpy as np


def layer_keys(seed_path):
    with open(seed_path) as f:
        for line in f:
            m = re.match(r"#\s*layer_key_by_index=(\{.*\})\s*$", line)
            if m:
                return {int(k): int(v) for k, v in json.loads(m.group(1)).items()}
    sys.exit(f"{seed_path}: no '# layer_key_by_index=' line")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="a GGUF file, or a glob over the shards of one")
    ap.add_argument("--seed", required=True, help="a seed file carrying # layer_key_by_index=")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import gguf

    keys = layer_keys(args.seed)
    routers = {}
    for path in sorted(glob.glob(args.gguf)):
        for t in gguf.GGUFReader(path).tensors:
            m = re.fullmatch(r"blk\.(\d+)\.ffn_gate_inp\.weight", t.name)
            if m:
                if t.tensor_type.name not in ("F32", "F16", "BF16"):
                    sys.exit(f"{t.name}: {t.tensor_type.name} is not a float router")
                routers[int(m.group(1))] = np.asarray(t.data, dtype=np.float32)
    if not routers:
        sys.exit(f"{args.gguf}: no blk.N.ffn_gate_inp.weight tensors")
    missing = sorted(set(routers) - set(keys))
    if missing:
        sys.exit(f"blocks {missing} have a router but no layer_key in {args.seed}")
    shapes = {r.shape for r in routers.values()}
    if len(shapes) != 1:
        sys.exit(f"router shapes differ: {shapes}")
    n_expert, n_embd = shapes.pop()
    with open(args.out, "wb") as f:
        f.write(b"ARCRLA01")
        f.write(struct.pack("<4I", len(routers), n_expert, n_embd, 0))
        for blk in sorted(routers):
            f.write(struct.pack("<Q", keys[blk]))
            f.write(np.ascontiguousarray(routers[blk], dtype="<f4").tobytes())
    print(f"{args.out}: {len(routers)} routers, {n_expert} x {n_embd}, keyed by layer_key")


if __name__ == "__main__":
    main()
