"""The dense projections in the plugin's u8 group-16 form (`--dense-u8`).

The checkpoint stores every dense projection as Q6_K: 16-value groups whose
values are exactly s * q with an integer q in [-32, 31] and a per-group scale
s = d * sc (ggml-quants.c dequantize_row_q6_k). The emitter receives them
dequantised to f32 (q4e.gguf_feed, after its row/column re-orders, which move
whole heads) and `--dense-fp16` stores them as f16: 2 bytes a value, 3.68 GiB
of the full-depth Qwen3.6-35B-A3B artifact's 14.55 GiB (measured 2026-09-26).

This pass recovers (q, s) per group FROM THE VALUES -- no second read of the
file, and no assumption about which re-order the feed applied -- and writes
the plugin's own compressed-weight chain for each projection:

    u8 [N, K/16, 16] -> Convert(f16) -> Subtract(Convert(u8 32)) ->
    Multiply(f16 scale [N, K/16, 1]) -> Reshape [N, K] -> Convert(f32) -> MatMul

the form arcint's C++ Q6_K repack (src/core/gguf_repack.cpp, type 14:
u8 0..63, scalar zero point 32, group 16) already serves through the GPU
plugin's compressed fully-connected (DESIGN 7.0.2ba): 1.125 bytes a value.
The one rounding is the scale's f16, the same 2^-11 relative class as the f16
artifact's per-value rounding.

A group is recovered when some k in 1..32 and a sign make every value an
integer multiple of +-max|w|/k within `tol` and inside [-32, 31]. A tensor is converted
only when EVERY group recovers; otherwise it is left as it was (an f32-sourced
tensor, or a re-order that split a group) and the report names it. The shared
expert's four weights are never converted: the GPU plugin fuses that MLP into
the MoE op (FuseMOESharedExpert), whose kernel reads plain weights; nor are
attention's k/v (with q, k and v all compressed the plugin's horizontal fusion
served wrong; see apply()).
"""
import numpy as np
from openvino import Type, opset13 as op

GROUP = 16
ZP = 32
QMIN, QMAX = -32, 31


def recover_groups(w, tol=1e-3, rows_per_chunk=4096):
    """w: f32 [N, K] with K % 16 == 0. Returns (q u8 [N, K/16, 16] offset by
    ZP, scale f32 [N, K/16], ok bool [N, K/16]): q - ZP times scale reproduces
    w in exact arithmetic for every ok group; a not-ok group's q/scale are 0."""
    w = np.asarray(w, dtype=np.float32)
    n, k = w.shape
    assert k % GROUP == 0, (k, GROUP)
    g = k // GROUP
    q_out = np.zeros((n, g, GROUP), np.uint8)
    s_out = np.zeros((n, g), np.float32)
    ok_out = np.zeros((n, g), bool)
    for r0 in range(0, n, rows_per_chunk):
        wg = w[r0:r0 + rows_per_chunk].reshape(-1, GROUP).astype(np.float64)   # [groups, 16]
        m = np.abs(wg).max(axis=-1)
        q = np.zeros(wg.shape, np.int64)
        s = np.zeros(m.shape, np.float64)
        done = m == 0                                                  # an all-zero group is q = 0
        # Q6_K's sub-scale is a SIGNED int8, so s may be negative: under a
        # positive scale its q in [-32, 31] reads as [-31, 32]. Both signs.
        # k runs from 32 down (the checkpoint's scales use the range, so most
        # groups resolve first); only unresolved groups are recomputed, and
        # ANY (k, sign) that fits is an exact representation.
        for kk in range(32, 0, -1):
            for sign in (1.0, -1.0):
                todo = np.nonzero(~done)[0]
                if todo.size == 0:
                    break
                sk = sign * m[todo] / kk
                qf = wg[todo] / sk[:, None]
                qi = np.rint(qf)
                fits = (np.abs(qf - qi) <= tol).all(axis=-1) & (qi >= QMIN).all(axis=-1) & (qi <= QMAX).all(axis=-1)
                hit = todo[fits]
                q[hit] = qi[fits].astype(np.int64)
                s[hit] = sk[fits]
                done[hit] = True
        # canonical form: the COARSEST scale. A finer valid one (q doubled, s
        # halved) is exact too, but its f16 rounding is worse -- a small s
        # reaches f16's subnormal range sooner (measured 2026-09-26: max
        # relative deviation 1.7e-2 against 9.5e-4 on the real shard)
        gcd = np.gcd.reduce(np.abs(q), axis=-1)
        gcd = np.where(gcd == 0, 1, gcd)
        q //= gcd[:, None]
        s *= gcd
        rows = wg.shape[0] // g
        rs = slice(r0, r0 + rows)
        q_out[rs] = (q + ZP).astype(np.uint8).reshape(rows, g, GROUP)
        s_out[rs] = s.astype(np.float32).reshape(rows, g)
        ok_out[rs] = done.reshape(rows, g)
    return q_out, s_out, ok_out


def decode(q_u8, scale_f16):
    """The chain's own arithmetic in numpy: (f16(q) - f16(32)) * f16(scale)
    in f16, as the plugin's decompression computes it, returned as f32."""
    x = (q_u8.astype(np.float16) - np.float16(ZP)) * scale_f16.astype(np.float16)[..., None]
    return x.reshape(q_u8.shape[0], -1).astype(np.float32)


def _u8_chain(q_u8, scale_f16, name):
    n, g, gs = q_u8.shape
    w = op.constant(np.ascontiguousarray(q_u8))
    w.set_friendly_name(name + "/dense_u8")
    x = op.convert(w, Type.f16)
    zp = op.constant(np.array([[[ZP]]], np.uint8))
    zp.set_friendly_name(name + "/dense_u8/zero_point")
    x = op.subtract(x, op.convert(zp, Type.f16))
    sc = op.constant(np.ascontiguousarray(scale_f16.reshape(n, g, 1)))
    sc.set_friendly_name(name + "/dense_u8/scale")
    x = op.multiply(x, sc)
    x = op.reshape(x, op.constant(np.array([n, g * gs], np.int64)), special_zero=False)
    return op.convert(x, Type.f32)


# A projection is converted only when the f16 rounding of its recovered scales
# keeps every decoded weight within this relative deviation of its f32 value.
# The real shard's worst is 9.47e-4 (2026-09-26); a scale in f16's subnormal
# range (a group of tiny weights) rounds far worse and is kept, not converted.
REL_MAX = 2.0 ** -9


def plan(model, min_elems=1 << 20):
    """Decide and PREPARE every eligible f32 projection of `model` without
    touching the graph: returns (plans, report). A plan is (MatMul node,
    name, q_u8, scale_f16). The graph edit is commit()'s, so a caller can run
    another pass in between -- the exporter's f16 compression, which
    compress_model_to_f16 skips for a WHOLE model once it detects any
    compressed-weight chain (compress_float_constants.cpp,
    is_model_optimized): committing first left every other constant f32
    (measured 2026-09-26: 0.772 GiB of f32 in the u8 artifact against 0.386 as
    f16)."""
    rep = {"converted": [], "kept": [], "f16_bytes": 0, "u8_bytes": 0}
    plans = []
    # measurement knob (bisection only): convert just the listed "NxK" shapes
    import os
    only = {x.strip() for x in os.environ.get("ARCINT_DENSE_U8_SHAPES", "").split(",") if x.strip()}
    for node in model.get_ordered_ops():
        if node.get_type_name() != "Constant" or node.get_output_element_type(0) != Type.f32:
            continue
        shape = list(node.get_output_shape(0))
        if len(shape) != 2 or shape[0] * shape[1] < min_elems:
            continue
        targets = list(node.output(0).get_target_inputs())
        if len(targets) != 1:
            continue
        mm = targets[0].get_node()
        if mm.get_type_name() != "MatMul" or targets[0].get_index() != 1:
            continue
        attrs = mm.get_attributes()
        if not attrs.get("transpose_b", False) or attrs.get("transpose_a", False):
            continue
        name = node.get_friendly_name()
        if name.startswith("shared_expert"):
            # FuseMOESharedExpert hands these to the MoE op as they are and its
            # kernel reads plain weights: a u8 chain there served garbage
            # (measured 2026-09-26, depth-4 logits A/B: argmax 7/1000)
            rep["kept"].append((name, "shared expert: fused into the MoE op, which reads plain weights"))
            continue
        if name.endswith("/k_proj") or name.endswith("/v_proj"):
            # q, k and v all compressed are fused horizontally by the GPU plugin,
            # and in the served paged graph that read garbage (measured
            # 2026-09-26: depth-4 logits A/B argmax 7/1000, KL 2.73 nats; q
            # compressed with k/v plain, or k/v with q plain, both clean). The
            # fused kernel alone is exact, so the defect needs the attention
            # context; keeping k/v plain costs 2 x 512 x 2048 a layer.
            rep["kept"].append((name, "attention k/v: q/k/v all compressed fuse horizontally and serve wrong"))
            continue
        n, k = shape
        if only and f"{n}x{k}" not in only:
            rep["kept"].append((name, f"{n}x{k} not in ARCINT_DENSE_U8_SHAPES"))
            continue
        if k % GROUP:
            rep["kept"].append((name, f"K={k} not a multiple of {GROUP}"))
            continue
        w = node.get_data()
        q, s, ok = recover_groups(w)
        if not ok.all():
            rep["kept"].append((name, f"{int((~ok).sum())} of {ok.size} groups not s*q with q in [{QMIN},{QMAX}]"))
            continue
        s16 = s.astype(np.float16)
        dec = decode(q, s16)
        rel = float(np.max(np.abs(dec - w) / np.maximum(np.abs(w), 1e-30), initial=0.0))
        del w, dec, s
        if rel > REL_MAX:
            rep["kept"].append((name, f"f16 scale rounding {rel:.2e} over {REL_MAX:.2e}"))
            continue
        plans.append((mm, name, q, s16))
        rep["converted"].append((name, n, k, rel))
        rep["f16_bytes"] += n * k * 2
        rep["u8_bytes"] += q.nbytes + s16.nbytes
    return plans, rep


def commit(plans):
    """Splice each plan's u8 chain into its MatMul's weight input. Whatever fed
    it before (the f32 Constant, or the f16 Constant + Convert a compression
    pass put there) loses its only consumer."""
    for i, (mm, name, q, s16) in enumerate(plans):
        mm.input(1).replace_source_output(_u8_chain(q, s16, name).output(0))
        plans[i] = None


def summary(rep):
    return (f"dense-u8: {len(rep['converted'])} projection(s) converted, {len(rep['kept'])} kept; "
            f"{rep['f16_bytes'] / 2**30:.3f} GiB as f16 -> {rep['u8_bytes'] / 2**30:.3f} GiB as u8+scale; "
            f"max rel deviation {max((c[3] for c in rep['converted']), default=0.0):.3e}")


def apply(model, min_elems=1 << 20, log=print):
    """plan() then commit(), nothing between: the cells' entry point. The
    exporter runs its f16 compression between the two instead."""
    plans, rep = plan(model, min_elems)
    commit(plans)
    log(summary(rep))
    return rep
