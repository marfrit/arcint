#!/usr/bin/env python3
"""The native per-expert decoders outside the plugin, in f32, against f64 host
models (docs/design-native-dpas-expert-kernel.md, gate 1).

Takes a program source the plugin built (captured with tools/cldump.c: the
batch that holds the grouped native kernels, every JIT constant resolved),
appends a harness kernel that calls the plugin's own IQ2_S-packed row decoder
(`native_gu_iq2s_packed_rows`, patch 0061) and stores its f32 sums, and runs
it on random IQ2_S blocks. The host models are gguf-py's IQ2_S dequantisation
of the same bytes, exact in f32 (d, the sub-scale and the grid need 22 bits):

  y_W   = x . W            the exact weights, in f64
  y_W16 = x . RNE-f16(W)   the weights a matrix-unit B operand carries
  S     = |x| . |W|        the per-element scale of an accumulation error

and prints, per arm, max |y - y_W| / S and max |y - y_W16| / S. The arms edit
the captured `NATIVE_W_ROUND` (patch 0063): 0 is the kernel as served, 1
rounds each decoded weight to f16, 2 to bf16. A matrix-unit kernel under
development is appended the same way (--dpas), so it is timed and checked
without a plugin rebuild.

With --dpas <file.cl> the file is appended after the served decoders and its
kernel `h_dpas_gu` (same arguments as h_scalar_gu; local size (1, 64, 1), one
work-group per tile of --dpas-tm pairs, 16 as shipped, and 64 columns; the
m1k arm's 64 work-items are 8 subgroups, one per 256-value block of K = 2048) runs as arm "dpas", checked
against the same models. --repeat N times every kernel with profiling events,
median of N after one warm launch.

Usage: native_kernel_harness.py <captured.cl> [--device A770] [--pairs 32]
       [--rows 512] [--seed 0] [--arms 0,1,2] [--dpas f.cl] [--dpas-defs "-D..."]
       [--repeat N]   (0 by default: no timing)
"""
import argparse
import re

import numpy as np
import pyopencl as cl
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

K = 2048          # the hidden size: gate/up's K
BLOCK = 256
PACKED = 80       # IQ2_S without its f16 d (patch 0052's packed row)

HARNESS = r"""
// arcint harness: the served IQ2_S-packed row decoder, f32 sums out. One
// work-group row per (pair tile, row pair); the tile is TM pairs of x rows.
#define H_TM 4
#define H_TN 2
#ifndef H_TFILL
#  define H_TFILL H_TM
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void h_scalar_gu(const __global uchar* gw, const __global half* gs,
                          const __global uchar* uw, const __global half* us,
                          const __global half* x, int pairs, int n_rows,
                          __global float* og, __global float* ou) {
    const int tile = sub_group_broadcast((int)get_global_id(0), 0);
    const int n0 = sub_group_broadcast((int)get_global_id(2), 0) * H_TN;
    const int lane = get_sub_group_local_id();
    const int cnt = min(H_TFILL, pairs - tile * H_TFILL);
    int xb[H_TM];
    for (int t = 0; t < H_TM; t++) xb[t] = (tile * H_TFILL + min(t, cnt - 1)) * HARNESS_K;
    float su[H_TN][H_TM], sg[H_TN][H_TM];
    native_gu_iq2s_packed_rows(uw, us, gw, gs, n0, H_TN, HARNESS_K, x, xb, cnt, lane, su, sg);
    if (lane == 0)
        for (int r = 0; r < H_TN; r++)
            for (int t = 0; t < H_TM; t++)
                if (t < cnt) {
                    og[(tile * H_TFILL + t) * n_rows + n0 + r] = sg[r][t];
                    ou[(tile * H_TFILL + t) * n_rows + n0 + r] = su[r][t];
                }
}
"""


def random_iq2s(rng, rows):
    """rows x (K/256) IQ2_S blocks: the 80-byte packed part and an f16 d kept in
    the normal range (a subnormal d is a separate case, design note §5)."""
    nblk = K // BLOCK
    packed = rng.integers(0, 256, size=(rows, nblk, PACKED), dtype=np.uint8)
    d = (rng.uniform(0.05, 1.0, size=(rows, nblk)) * (0.3 / (43.0 * 3.875))).astype(np.float16)
    return packed, d


def dequant(packed, d):
    rows, nblk, _ = packed.shape
    raw = np.concatenate([d.reshape(rows, nblk, 1).view(np.uint8).reshape(rows, nblk, 2), packed], axis=2)
    return dequantize(raw.reshape(rows, nblk * 82), GGMLQuantizationType.IQ2_S).reshape(rows, K)


def grid_h2(src):
    """The IQ2_S grid of the captured source as a __constant uint table of
    half2 pairs (D_GRID_H2): the matrix-unit decoder's exact-integer form reads
    an entry's eight values as four f16 pairs instead of eight bytes."""
    m = re.search(r"NATIVE_IQ2S_GRID\[8192\]\s*=\s*\{([^}]*)\}", src)
    vals = np.array([int(v, 0) for v in m.group(1).replace("\n", " ").split(",") if v.strip()], dtype=np.uint8)
    assert vals.size == 8192
    h = vals.astype(np.float16).view(np.uint16).astype(np.uint32)
    words = h[0::2] | (h[1::2] << 16)
    body = ",".join(f"0x{w:08x}u" for w in words)
    return f"\n__constant uint D_GRID_H2[4096] = {{{body}}};\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--device", default="A770")
    ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="0,1,2")
    ap.add_argument("--dpas", default="")
    ap.add_argument("--dpas-defs", default="")
    ap.add_argument("--dpas-tm", type=int, default=16)  # the shipped kNativeDpasTm
    ap.add_argument("--dpas-wg", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--scalar-tfill", type=int, default=4, help="pairs per scalar tile (served decode: 1)")
    ap.add_argument("--m1", action="store_true", help="also run the candidate's one-pair kernel h_dpas_gu_m1")
    ap.add_argument("--m1k", action="store_true", help="also run its K-split one-pair kernel h_dpas_gu_m1k")
    ap.add_argument("--m1k2", action="store_true", help="also run its K- and projection-split kernel h_dpas_gu_m1k2")
    args = ap.parse_args()

    dev = [d for p in cl.get_platforms() for d in p.get_devices() if args.device in d.name]
    assert dev, f"no OpenCL device matching {args.device!r}"
    ctx = cl.Context([dev[0]])
    q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    base = open(args.src).read()
    assert "native_gu_iq2s_packed_rows" in base, "the source has no IQ2_S-packed row decoder (patch 0061)"

    rng = np.random.default_rng(args.seed)
    n_rows, pairs = args.rows, args.pairs
    gp, gd = random_iq2s(rng, n_rows)
    up, ud = random_iq2s(rng, n_rows)
    x = rng.standard_normal((pairs, K)).astype(np.float16)
    Wg, Wu = dequant(gp, gd).astype(np.float64), dequant(up, ud).astype(np.float64)
    xf = x.astype(np.float64)
    ref = {}
    for name, W in (("gate", Wg), ("up", Wu)):
        W16 = W.astype(np.float16).astype(np.float64)
        ref[name] = (xf @ W.T, xf @ W16.T, np.abs(xf) @ np.abs(W).T)

    mf = cl.mem_flags
    bufs = [cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(a))
            for a in (gp, gd, up, ud, x)]
    og = np.zeros((pairs, n_rows), np.float32)
    ou = np.zeros_like(og)
    og_b = cl.Buffer(ctx, mf.WRITE_ONLY, og.nbytes)
    ou_b = cl.Buffer(ctx, mf.WRITE_ONLY, ou.nbytes)
    def timed(launch):
        ev = launch()
        ev.wait()
        ms = []
        for _ in range(args.repeat):
            ev = launch()
            ev.wait()
            ms.append((ev.profile.end - ev.profile.start) * 1e-6)
        return f"  ms median {np.median(ms):.3f} min {min(ms):.3f}" if ms else ""

    outs = {}
    arms = [a for a in args.arms.split(",") if a] + (["dpas"] if args.dpas else []) + (["m1"] if args.m1 else []) + (["m1k"] if args.m1k else []) + (["m1k2"] if args.m1k2 else [])
    for arm in arms:
        w_round = 0 if arm in ("dpas", "m1", "m1k", "m1k2") else int(arm)
        src, n_sub = re.subn(r"#define NATIVE_W_ROUND \d", f"#define NATIVE_W_ROUND {w_round}", base)
        assert n_sub or w_round == 0, "no NATIVE_W_ROUND in the source (a pre-0063 capture): the arms would be equal"
        src += f"\n#define HARNESS_K {K}\n#define H_TFILL {args.scalar_tfill}\n" + HARNESS
        if arm in ("dpas", "m1", "m1k", "m1k2"):
            src += "\n#define NATIVE_W_FORM(a, b, c) ((a) * (b) * (c))\n" + grid_h2(base) + open(args.dpas).read()
        prg = cl.Program(ctx, src).build(options="-cl-mad-enable -cl-std=CL3.0 " + args.dpas_defs)
        if arm == "m1k2":
            t = timed(lambda: prg.h_dpas_gu_m1k2(q, (pairs, n_rows // 8 * 128, 1), (1, 128, 1), *bufs,
                                                 np.int32(pairs), np.int32(n_rows), og_b, ou_b))
        elif arm == "m1k":
            t = timed(lambda: prg.h_dpas_gu_m1k(q, (pairs, n_rows // 8 * 64, 1), (1, 64, 1), *bufs,
                                                np.int32(pairs), np.int32(n_rows), og_b, ou_b))
        elif arm == "m1":
            t = timed(lambda: prg.h_dpas_gu_m1(q, (pairs, n_rows, 1), (1, args.dpas_wg, 1), *bufs,
                                               np.int32(pairs), np.int32(n_rows), og_b, ou_b))
        elif arm == "dpas":
            tiles = (pairs + args.dpas_tm - 1) // args.dpas_tm
            t = timed(lambda: prg.h_dpas_gu(q, (tiles, n_rows, 1), (1, args.dpas_wg, 1), *bufs,
                                            np.int32(pairs), np.int32(n_rows), og_b, ou_b))
        else:
            tiles = (pairs + args.scalar_tfill - 1) // args.scalar_tfill
            t = timed(lambda: prg.h_scalar_gu(q, (tiles, 16, n_rows // 2), (1, 16, 1), *bufs,
                                              np.int32(pairs), np.int32(n_rows), og_b, ou_b))
        print(f"ARM {arm}{t}")
        cl.enqueue_copy(q, og, og_b)
        cl.enqueue_copy(q, ou, ou_b)
        q.finish()
        outs[arm] = (og.copy(), ou.copy())
        for name, y in (("gate", og), ("up", ou)):
            yw, yw16, s = ref[name]
            e_w = np.abs(y - yw) / s
            e_w16 = np.abs(y - yw16) / s
            print(f"ARM {arm} {name}: max|y-y_W|/S {e_w.max():.3e}  max|y-y_W16|/S {e_w16.max():.3e}  "
                  f"median|y-y_W|/S {np.median(e_w):.3e}")
    for other in ("m1", "m1k", "m1k2"):
        if "dpas" in outs and other in outs:
            same = all(np.array_equal(outs["dpas"][i].view(np.uint32), outs[other][i].view(np.uint32)) for i in (0, 1))
            print(f"{other.upper()} vs tiled: {'BYTE-IDENTICAL' if same else 'DIFFERENT'}")


if __name__ == "__main__":
    main()
