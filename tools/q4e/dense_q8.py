"""The dense projections in their checkpoint's Q8_0 form (`--dense-q8`).

Flash-Next's checkpoint stores almost every dense projection as Q8_0: 32-value
groups whose values are exactly d * q with an f16 d and an integer q in
[-127, 127] (ggml-quants.c quantize_row_q8_0: d = amax / 127, q = round(x / d),
so the group's largest |q| is 127). The emitter receives them dequantised to
f32 (q4e.gguf_feed; its re-orders only concatenate whole rows or whole
column blocks, never inside a group) and the exporter wrote them as f32
Constants, which the GPU plugin serves as f16: 2 bytes a value where the
checkpoint has 1.0625, and each value rounded once to f16.

This pass recovers (q, d) per group FROM THE VALUES, as q4e.dense_u8 does for
Q6_K, and writes the plugin's compressed-weight chain for each projection:

    i8 [N, K/32, 32] -> Convert(f16) -> Multiply(f16 scale [N, K/32, 1]) ->
    Reshape [N, K] -> Convert(f32) -> MatMul

the symmetric group-32 form arcint's C++ Q8_0 repack (src/core/gguf_repack.cpp,
type 8) serves. d is an f16 in the checkpoint and max|w| = 127 * d is exact in
f32, so the recovered scale IS the checkpoint's d and the chain carries the
checkpoint's bytes exactly; the only rounding left is the plugin's own f16
product, the class of the f16 artifact's per-value rounding.

Everything else is q4e.dense_u8's: the same eligibility (a 2-D f32 Constant,
the only input 1 of a transpose_b MatMul), the same exclusions by name (the
shared expert, which the MoE op reads plain; attention k/v, whose horizontal
fusion with a compressed q served wrong), a tensor converted only when every
group recovers.
"""
import numpy as np
from openvino import Type, opset13 as op

from q4e import dense_u8

GROUP = 32
QMAX = 127


def recover_groups(w, tol=1e-3, rows_per_chunk=4096):
    """w: f32 [N, K] with K % 32 == 0. Returns (q i8 [N, K/32, 32], scale f32
    [N, K/32], ok bool [N, K/32]): q times scale reproduces w in exact
    arithmetic for every ok group; a not-ok group's q/scale are 0."""
    w = np.asarray(w, dtype=np.float32)
    n, k = w.shape
    assert k % GROUP == 0, (k, GROUP)
    g = k // GROUP
    q_out = np.zeros((n, g, GROUP), np.int8)
    s_out = np.zeros((n, g), np.float32)
    ok_out = np.zeros((n, g), bool)
    for r0 in range(0, n, rows_per_chunk):
        wg = w[r0:r0 + rows_per_chunk].reshape(-1, GROUP).astype(np.float64)   # [groups, 32]
        m = np.abs(wg).max(axis=-1)
        q = np.zeros(wg.shape, np.int64)
        s = np.zeros(m.shape, np.float64)
        done = m == 0                                                  # an all-zero group is q = 0
        # Q8_0's largest |q| is 127 by construction; smaller maxima are tried
        # only for what 127 leaves unresolved (a hand-made or re-scaled group),
        # and not at all when 127 leaves more than 1 % unresolved: that is not
        # a Q8_0 tensor (Q6_K's output head, an f32 router), and walking the
        # other 126 candidates over it cost the first full-depth export ~30 min.
        for kk in range(QMAX, 0, -1):
            todo = np.nonzero(~done)[0]
            if todo.size == 0 or (kk < QMAX and todo.size > 0.01 * done.size):
                break
            sk = m[todo] / kk
            qf = wg[todo] / sk[:, None]
            qi = np.rint(qf)
            fits = (np.abs(qf - qi) <= tol).all(axis=-1) & (np.abs(qi) <= QMAX).all(axis=-1)
            hit = todo[fits]
            q[hit] = qi[fits].astype(np.int64)
            s[hit] = sk[fits]
            done[hit] = True
        rows = wg.shape[0] // g
        rs = slice(r0, r0 + rows)
        q_out[rs] = q.astype(np.int8).reshape(rows, g, GROUP)
        s_out[rs] = s.astype(np.float32).reshape(rows, g)
        ok_out[rs] = done.reshape(rows, g)
    return q_out, s_out, ok_out


def decode(q_i8, scale_f16):
    """The chain's own arithmetic in numpy: f16(q) * f16(scale) in f16, as the
    plugin's decompression computes it, returned as f32."""
    x = q_i8.astype(np.float16) * scale_f16.astype(np.float16)[..., None]
    return x.reshape(q_i8.shape[0], -1).astype(np.float32)


def _i8_chain(q_i8, scale_f16, name):
    n, g, gs = q_i8.shape
    w = op.constant(np.ascontiguousarray(q_i8))
    w.set_friendly_name(name + "/dense_q8")
    x = op.convert(w, Type.f16)
    sc = op.constant(np.ascontiguousarray(scale_f16.reshape(n, g, 1)))
    sc.set_friendly_name(name + "/dense_q8/scale")
    x = op.multiply(x, sc)
    x = op.reshape(x, op.constant(np.array([n, g * gs], np.int64)), special_zero=False)
    return op.convert(x, Type.f32)


Q8_0 = dense_u8.Format("q8", GROUP, recover_groups, decode, _i8_chain, f"[-{QMAX}, {QMAX}]",
                       "ARCINT_DENSE_Q8_SHAPES")


def plan(model, min_elems=1 << 20, skip=()):
    """q4e.dense_u8.plan() in the Q8_0 form: (plans, report), graph untouched."""
    return dense_u8.plan(model, min_elems, fmt=Q8_0, skip=skip)


commit = dense_u8.commit
summary = dense_u8.summary


def apply(model, min_elems=1 << 20, log=print):
    """plan() then commit(), nothing between: the cells' entry point."""
    return dense_u8.apply(model, min_elems, log=log, fmt=Q8_0)
