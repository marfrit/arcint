"""OV opset-13 emission of the qwen4_exp GatedDeltaNet (GDN) block,
full-sequence no-cache branch.

Mirrors `tools/q4e/ref_gdn.Qwen4ExpTextGatedDeltaNet.forward` (itself the
pinned transformers reference, modeling_qwen4_exp.py 465-628) as a *static*
graph: batch fixed to 1 and sequence length fixed at build time, so the
chunked gated delta rule -- which the reference writes as two Python loops --
unrolls to a fixed op graph. This is the KLD-harness shape (fixed T); the
paged/stateful serving shape is a later increment.

Op choices where opset-13 differs from torch:
  * cumsum over the 64-wide chunk axis        -> op.cumsum (inclusive).
  * softplus                                  -> op.softplus.
  * causal depthwise conv1d (kernel 4)        -> 4 shifted multiply-adds over a
    left-zero-padded input (bit-for-bit the same taps as F.conv1d(padding=K-1)
    then [:seq_len], no GroupConvolution lowering to trust).
  * the unit-lower-triangular solve that condenses the delta-rule updates
    (reference: solve_triangular / the export-branch forward substitution)
    -> the exact closed form. With L0 = strictly-lower(ut_system), the solve is
    (I + L0)^-1 @ rhs; L0 is 64x64 strictly lower triangular hence nilpotent
    (L0^64 = 0), so (I + L0)^-1 = sum_{p=0}^{63} (-L0)^p exactly. The finite sum
    is built by geometric doubling (6 steps x 2 matmuls = 12 matmuls for
    chunk 64), no solver op.

Entry point: build_gdn_model(config, state, seq_len) -> ov.Model with inputs
`hidden_states` [1, T, H] f32 and `attention_mask` [1, T] f32, and result
`output` [1, T, H] f32.
"""
import numpy as np
from openvino import Model, Type
from openvino import opset13 as op

CHUNK = 64  # torch_chunk_gated_delta_rule default chunk_size


# --- thin op wrappers ------------------------------------------------------
def _c(v):
    return op.constant(np.ascontiguousarray(v, dtype=np.float32))


def _i(v):
    return op.constant(np.ascontiguousarray(v, dtype=np.int64))


def _reshape(x, shape):
    return op.reshape(x, _i(np.array(shape, np.int64)), False)


def _transpose(x, perm):
    return op.transpose(x, _i(np.array(perm, np.int64)))


def _slice(x, start, stop, step, axis):
    return op.slice(x, _i([start]), _i([stop]), _i([step]), _i([axis]))


def _mm(a, b, ta=False, tb=False):
    return op.matmul(a, b, ta, tb)


def _mul(a, b):
    return op.multiply(a, b)


def _add(a, b):
    return op.add(a, b)


def _sub(a, b):
    return op.subtract(a, b)


def _silu(x):
    return op.multiply(x, op.sigmoid(x))


def _rsum(x, axis):
    return op.reduce_sum(x, _i([axis]), True)


def _rmean(x, axis):
    return op.reduce_mean(x, _i([axis]), True)


def _rsqrt_eps(x, eps):
    return op.divide(_c(1.0), op.sqrt(_add(x, _c(np.float32(eps)))))


# --- building blocks -------------------------------------------------------
def _causal_conv_silu(x, conv_w, T, conv_dim, K):
    """Depthwise causal conv1d + silu. x: [1, conv_dim, T]; conv_w: [conv_dim,1,K]."""
    xpad = op.concat([_c(np.zeros((1, conv_dim, K - 1), np.float32)), x], axis=2)
    acc = None
    for j in range(K):
        xs = _slice(xpad, j, j + T, 1, 2)  # [1, conv_dim, T]
        wj = _c(conv_w[:, 0, j].reshape(1, conv_dim, 1))
        term = _mul(xs, wj)
        acc = term if acc is None else _add(acc, term)
    return _silu(acc)


def _repeat_interleave_heads(x, T, n_heads, dim, r):
    """[1,T,n_heads,dim] -> [1,T,n_heads*r,dim], each head repeated r times."""
    if r == 1:
        return x
    x5 = _reshape(x, [1, T, n_heads, 1, dim])
    xr = op.concat([x5] * r, axis=3)  # [1,T,n_heads,r,dim]
    return _reshape(xr, [1, T, n_heads * r, dim])


def _l2norm_last(x, last_axis):
    inv = _rsqrt_eps(_rsum(_mul(x, x), last_axis), 1e-6)
    return _mul(x, inv)


def _ut_inverse(ut_orig, chunk):
    """(I + L0)^-1 with L0 = strictly-lower(ut_orig), via the exact nilpotent
    geometric series sum_{p=0}^{chunk-1} (-L0)^p built by doubling."""
    sl = _c(np.tril(np.ones((chunk, chunk), np.float32), -1))  # strictly lower ones
    eye = _c(np.eye(chunk, dtype=np.float32))
    m = op.negative(_mul(ut_orig, sl))  # -L0
    s = eye  # sum_{p=0}^{0}
    mp = m   # (-L0)^1
    n = 1
    while n < chunk:
        s = _add(s, _mm(mp, s))  # S_{2n} = S_n + (-L0)^n @ S_n
        mp = _mm(mp, mp)         # (-L0)^{2n}
        n *= 2
    return s


def _rmsnorm_gated(core, z, weight_vec, eps, last_axis):
    """Qwen4ExpTextRMSNormGated over the last axis (reference lines 31-47)."""
    h = _mul(core, _rsqrt_eps(_rmean(_mul(core, core), last_axis), eps))
    h = _mul(_c(weight_vec.reshape([1] * last_axis + [-1])), h)
    return _mul(h, _silu(z))


# --- top-level emitter -----------------------------------------------------
def build_gdn_model(config, state, seq_len):
    H = config.hidden_size
    HK = config.linear_num_key_heads
    HV = config.linear_num_value_heads
    Dk = config.linear_key_head_dim
    Dv = config.linear_value_head_dim
    K = config.linear_conv_kernel_dim
    eps = config.rms_norm_eps
    key_dim = Dk * HK
    value_dim = Dv * HV
    conv_dim = key_dim * 2 + value_dim
    ratio = HV // HK
    T = int(seq_len)
    pad = (CHUNK - T % CHUNK) % CHUNK
    Tp = T + pad
    C = Tp // CHUNK

    def w(name):
        return state[name]

    hidden = op.parameter([1, T, H], Type.f32)
    hidden.set_friendly_name("hidden_states")
    amask = op.parameter([1, T], Type.f32)
    amask.set_friendly_name("attention_mask")

    # apply_mask_to_padding_states: hidden * attention_mask[:, :, None]
    x = _mul(hidden, _reshape(amask, [1, T, 1]))

    # in-projections (Linear, bias=False): y = x @ W^T
    qkv = _mm(x, _c(w("in_proj_qkv.weight")), tb=True)  # [1,T,conv_dim]
    z = _mm(x, _c(w("in_proj_z.weight")), tb=True)       # [1,T,value_dim]
    b = _mm(x, _c(w("in_proj_b.weight")), tb=True)       # [1,T,HV]
    a = _mm(x, _c(w("in_proj_a.weight")), tb=True)       # [1,T,HV]

    # depthwise causal conv over the qkv channels, then split
    qkv_t = _transpose(qkv, [0, 2, 1])                    # [1,conv_dim,T]
    conv = _causal_conv_silu(qkv_t, w("conv1d.weight"), T, conv_dim, K)
    conv_t = _transpose(conv, [0, 2, 1])                  # [1,T,conv_dim]
    query = _slice(conv_t, 0, key_dim, 1, 2)
    key = _slice(conv_t, key_dim, 2 * key_dim, 1, 2)
    value = _slice(conv_t, 2 * key_dim, 2 * key_dim + value_dim, 1, 2)

    query = _reshape(query, [1, T, HK, Dk])
    key = _reshape(key, [1, T, HK, Dk])
    value = _reshape(value, [1, T, HV, Dv])
    z = _reshape(z, [1, T, HV, Dv])

    beta = op.sigmoid(b)                                  # [1,T,HV]
    # g = -exp(A_log) * softplus(a + dt_bias)
    g = _mul(
        op.negative(op.exp(_c(w("A_log")))),
        op.softplus(_add(a, _c(w("dt_bias")))),
    )                                                     # [1,T,HV]

    if ratio > 1:
        query = _repeat_interleave_heads(query, T, HK, Dk, ratio)
        key = _repeat_interleave_heads(key, T, HK, Dk, ratio)

    # ---- chunked gated delta rule (transpose to [1,HV,T,*]) ----
    q = _transpose(query, [0, 2, 1, 3])                   # [1,HV,T,Dk]
    k = _transpose(key, [0, 2, 1, 3])
    v = _transpose(value, [0, 2, 1, 3])                   # [1,HV,T,Dv]
    beta_t = _transpose(beta, [0, 2, 1])                  # [1,HV,T]
    decay_t = _transpose(g, [0, 2, 1])                    # [1,HV,T]

    q = _l2norm_last(q, 3)
    k = _l2norm_last(k, 3)
    q = _mul(q, _c(np.float32(Dk ** -0.5)))

    if pad > 0:
        q = op.concat([q, _c(np.zeros((1, HV, pad, Dk), np.float32))], axis=2)
        k = op.concat([k, _c(np.zeros((1, HV, pad, Dk), np.float32))], axis=2)
        v = op.concat([v, _c(np.zeros((1, HV, pad, Dv), np.float32))], axis=2)
        beta_t = op.concat([beta_t, _c(np.zeros((1, HV, pad), np.float32))], axis=2)
        decay_t = op.concat([decay_t, _c(np.zeros((1, HV, pad), np.float32))], axis=2)

    beta_u = _reshape(beta_t, [1, HV, Tp, 1])
    v_beta = _mul(v, beta_u)
    k_beta = _mul(k, beta_u)

    q_c = _reshape(q, [1, HV, C, CHUNK, Dk])
    k_c = _reshape(k, [1, HV, C, CHUNK, Dk])
    kb_c = _reshape(k_beta, [1, HV, C, CHUNK, Dk])
    vb_c = _reshape(v_beta, [1, HV, C, CHUNK, Dv])
    decay_c = _reshape(decay_t, [1, HV, C, CHUNK])

    cum = op.cumsum(decay_c, op.constant(np.int64(3)))    # [1,HV,C,CHUNK] (scalar axis)
    expc = op.exp(cum)
    expc5 = _reshape(expc, [1, HV, C, CHUNK, 1])

    # pairwise decay: exp(cum_i - cum_j), strictly-upper masked to 0
    cd_i = _reshape(cum, [1, HV, C, CHUNK, 1])
    cd_j = _reshape(cum, [1, HV, C, 1, CHUNK])
    add_mask = np.where(np.triu(np.ones((CHUNK, CHUNK), np.float32), 1) > 0, -1e30, 0.0)
    pd = op.exp(_add(_sub(cd_i, cd_j), _c(add_mask.astype(np.float32))))

    ut_orig = _mul(_mm(kb_c, k_c, tb=True), pd)           # [1,HV,C,CHUNK,CHUNK]
    intra = _mul(_mm(q_c, k_c, tb=True), pd)
    dkb = _mul(kb_c, expc5)                               # decayed_k_beta

    inv = _ut_inverse(ut_orig, CHUNK)
    new_values = _mm(inv, vb_c)                           # [1,HV,C,CHUNK,Dv]
    k_cumdecay = _mm(inv, dkb)                            # [1,HV,C,CHUNK,Dk]

    q_dec = _mul(q_c, expc5)
    cum_last = _slice(cum, CHUNK - 1, CHUNK, 1, 3)        # [1,HV,C,1]
    k_dec = _mul(k_c, _reshape(op.exp(_sub(cum_last, cum)), [1, HV, C, CHUNK, 1]))
    chunk_decay = op.exp(_reshape(cum_last, [1, HV, C]))  # [1,HV,C]

    # sequential scan over chunks (unrolled; C is 1 or 2 for the harness Ts)
    last = _c(np.zeros((1, HV, Dk, Dv), np.float32))
    cores = []
    for ci in range(C):
        nv = _reshape(_slice(new_values, ci, ci + 1, 1, 2), [1, HV, CHUNK, Dv])
        kcd = _reshape(_slice(k_cumdecay, ci, ci + 1, 1, 2), [1, HV, CHUNK, Dk])
        qd = _reshape(_slice(q_dec, ci, ci + 1, 1, 2), [1, HV, CHUNK, Dk])
        kd = _reshape(_slice(k_dec, ci, ci + 1, 1, 2), [1, HV, CHUNK, Dk])
        it = _reshape(_slice(intra, ci, ci + 1, 1, 2), [1, HV, CHUNK, CHUNK])
        cd = _reshape(_slice(chunk_decay, ci, ci + 1, 1, 2), [1, HV, 1, 1])

        v_new = _sub(nv, _mm(kcd, last))                 # [1,HV,CHUNK,Dv]
        inter = _mm(qd, last)
        core_i = _add(inter, _mm(it, v_new))
        last = _add(_mul(last, cd), _mm(kd, v_new, ta=True))
        cores.append(_reshape(core_i, [1, HV, 1, CHUNK, Dv]))

    core = op.concat(cores, axis=2) if C > 1 else cores[0]  # [1,HV,C,CHUNK,Dv]
    core = _reshape(core, [1, HV, Tp, Dv])
    if pad > 0:
        core = _slice(core, 0, T, 1, 2)
    core = _transpose(core, [0, 2, 1, 3])                # [1,T,HV,Dv]

    # RMSNormGated(core, z) over Dv, then out_proj
    core = _rmsnorm_gated(core, z, w("norm.weight"), eps, 3)
    core = _reshape(core, [1, T, value_dim])
    out = _mm(core, _c(w("out_proj.weight")), tb=True)   # [1,T,H]

    result = op.result(out)
    result.set_friendly_name("output")
    model = Model([result], [hidden, amask], "qwen4_exp_gdn_block")
    return model


__all__ = ["build_gdn_model"]
