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
    -> the pin's OWN export-branch forward substitution (pin 355-366, the
    `is_torchdynamo_exporting()` branch, written precisely because "not all
    export targets support the fast triangular solver"), transcribed op for op
    and unrolled over the 64 chunk rows. See `_ut_inverse`.

    CORRECTION, 2026-09-11 (FIX-GDN-UTINV; REVIEW 58e3e09 finding 1). This
    header used to claim a closed form: L0 = strictly-lower(ut_system) is 64x64
    strictly lower triangular hence nilpotent, so
    (I + L0)^-1 = sum_{p=0}^{63} (-L0)^p "exactly", built by geometric doubling,
    "no solver op". The identity is true in EXACT arithmetic and FALSE in f32 on
    this checkpoint's weights, by five orders of magnitude. Measured: the
    intermediate powers explode before they cancel --
    |(-L0)^p| for p=1..8 = 5.05e-01 7.79e+00 6.57e+01 3.58e+02 1.42e+03
    4.39e+03 1.11e+04 2.32e+04, peaking at 7.377e+04 while the answer's entries
    are O(1); the built inverse then sat 2.268e-02 from the exact one, against
    1.825e-07 for the forward substitution at the SAME f32 precision. Two
    independent f32 evaluations of the series disagreed with each other by
    1.367e-02 -- the method is not reproducible in f32 better than 1e-2.
    The doubling series was a DEVIATION from E1.5's prescription, introduced at
    E2 inc1 and never tested on real weights until REVIEW 58e3e09; this is a
    return to spec, not a new idea. It costs a much larger static graph (the
    substitution unrolls to ~9 ops per chunk row instead of 12 matmuls total)
    and that is the correct trade: correctness over elegance.
    Random-weight fixtures cannot see any of this -- on them the powers DECAY
    (peak 3.69e-02) and the two algorithms agree exactly.

Entry point: build_gdn_model(config, state, seq_len) -> ov.Model with inputs
`hidden_states` [1, T, H] f32 and `attention_mask` [1, T] f32, and result
`output` [1, T, H] f32.
"""
import os

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


def _key_head_map(config):
    """HOW the HK key heads serve the HV value heads (HV = r * HK):
    `interleave` -- value head h reads key head h // r (the pin's
    `repeat_interleave`, pin line 583; HF's qwen3_5 files alike); `tiled` --
    value head h reads key head h % HK (llama.cpp's qwen35 / qwen4exp:
    `ggml_repeat_4d` on the non-fused path and the fused GATED_DELTA_NET op
    alike). The shipped Flash-Next GGUF computes the model TILED: measured
    2026-09-18 on the dev host against llama.cpp's whole tensors (France ids,
    layer 0) -- interleaved, 4 of 48 core heads agree (exactly the heads where
    h // 3 == h % 16) and no permutation of the pin's heads matches the rest;
    tiled, 48 of 48 agree at token 0 and the layer's output lands at corr
    0.9999 (campaign serving-shape-logits). Whether HF's order or a converter
    permutation of the value side explains the difference is unverified and
    moot for a fill that reads the GGUF. The real geometry says `tiled`
    (piecewise_export.REAL_GEOMETRY); the tiny parity configs keep the pin's
    `interleave`."""
    mode = getattr(config, "gdn_key_head_map", None) or "interleave"
    if mode not in ("interleave", "tiled"):
        raise ValueError(f"unsupported gdn_key_head_map {mode!r} (interleave or tiled)")
    return mode


def _expand_heads(x, T, n_heads, dim, r, mode="interleave"):
    """[1,T,n_heads,dim] -> [1,T,n_heads*r,dim]. `interleave`: each head
    repeated r times in place (head h*r+j <- h); `tiled`: the block of heads
    repeated r times (head c*n_heads+h <- h)."""
    if r == 1:
        return x
    if mode == "tiled":
        return op.concat([x] * r, axis=2)                   # [1,T,r*n_heads,dim]
    x5 = _reshape(x, [1, T, n_heads, 1, dim])
    xr = op.concat([x5] * r, axis=3)  # [1,T,n_heads,r,dim]
    return _reshape(xr, [1, T, n_heads * r, dim])


def _repeat_interleave_heads(x, T, n_heads, dim, r):
    """Kept for callers of the old name: the pin's interleave."""
    return _expand_heads(x, T, n_heads, dim, r, "interleave")


def _l2norm_last(x, last_axis):
    inv = _rsqrt_eps(_rsum(_mul(x, x), last_axis), 1e-6)
    return _mul(x, inv)


# How the chunk axis is presented to the forward-substitution unroll below.
# THE DEFECT THIS EXISTS FOR (frontier card pass 2026-09-12, ~/win-050/FINDINGS;
# reproduced at this tip by `tests/python/test_gdn_block.py` and the T sweep in
# that ledger): on both Arc cards, every multi-chunk static-T GDN graph is
# corrupt from global row 65 onward, exactly T-65 rows, while T=64 (one chunk)
# is clean at the f32 floor. The emitted math is identical in both cases; only
# the extent of the chunk axis changes.
#
#   "batched"    ut_orig stays [1, HV, C, CHUNK, CHUNK] and the unroll runs
#                once, with BOTH HV and C as leading axes. The form shipped up
#                to 2026-09-12, and the one that is corrupt for C > 1. Kept
#                selectable because it is the defect's reproducer.
#   "folded"     the two leading axes are merged to [1, HV*C, CHUNK, CHUNK] by
#                a pure view reshape (HV and C are adjacent and contiguous), so
#                the unroll runs once at RANK 4 instead of rank 5 over the same
#                leading volume. Two reshapes; node count essentially unchanged.
#   "debatched"  the chunk axis is hoisted out OF THE UNROLL ONLY: C separate
#                unroll instances, each [1, HV, CHUNK, CHUNK], concatenated
#                back. Node count scales with C. NOTE this is batch-HV, not
#                batch-1 -- it is exactly the leading shape the clean T=64 case
#                already has. MEASURED USELESS: bit-identical corruption to
#                "batched" (9.7893e-02, 31 bad, first row 65 at T=96), which is
#                what EXCLUDES the unroll's own batching as the cause. Kept
#                because that exclusion is the finding.
#   "perchunk"   the chunk axis is hoisted out of EVERYTHING: no emitted op ever
#                sees a tensor with a live chunk axis. Per chunk, at rank 4:
#                cumsum, the pairwise decay, both k/q contractions, the unroll,
#                and both inverse matmuls; the delta-rule carry stays an
#                ordinary internal edge, as it already was. This is the brief's
#                actual hypothesis; "debatched" was a partial form of it.
#
# DEFAULT IS "perchunk" SINCE 2026-09-12, because it is the only one of the four
# that is correct on this plugin for C > 1 (measured, both cards; the card table
# is in the commit that flipped it and in ~/win-050/FINDINGS). It is not free
# and the cost is structural, not incidental: the unroll is ~1,900 ops and
# "perchunk" emits one per chunk instead of one per graph, so the GDN block
# grows LINEARLY IN C where "batched" was nearly flat --
#
#     T     C   batched   perchunk        x36 GDN blocks (the 48-layer stack)
#      64   1      2054       2037
#     256   4      2224       7750        80,064  ->    279,000
#    2048  32      3680      61062       132,480  ->  2,198,232
#
# so at serving prefill lengths this form is NOT viable, and the route there is
# the chunked STATEFUL prefill increment (the 13 paged-port xfails), not a
# bigger static graph. What the flip buys is correct multi-chunk static graphs
# at the shapes the window actually boots. The growth law is gated by
# `test_the_chunk_emission_modes_agree_on_cpu_and_differ_in_node_count`.
#
# `Q4E_GDN_UT_MODE` overrides it for a whole suite run without editing code,
# which is how the card legs were taken on both sides of the comparison.
UT_EMIT_MODE = os.environ.get("Q4E_GDN_UT_MODE", "perchunk").strip() or "perchunk"

_UT_MODES = ("batched", "folded", "debatched", "perchunk")


def _ut_inverse_chunked(ut_orig, chunk, hv, c, mode):
    """(I + L0)^-1 for every chunk, presenting the chunk axis per `mode`.

    Returns [1, hv, c, chunk, chunk] in every mode; the modes differ only in
    the shape the unroll's ops see, never in the arithmetic they perform.
    """
    if mode not in _UT_MODES:
        raise ValueError(f"unknown ut emit mode {mode!r}; expected one of {_UT_MODES}")
    if mode == "batched":
        return _ut_inverse(ut_orig, chunk, [1, hv, c])
    if mode == "folded":
        # HV and C are adjacent leading axes of a contiguous tensor, so merging
        # them is a view: the unroll sees rank 4 with the same leading volume.
        folded = _reshape(ut_orig, [1, hv * c, chunk, chunk])
        inv = _ut_inverse(folded, chunk, [1, hv * c])
        return _reshape(inv, [1, hv, c, chunk, chunk])
    parts = []
    for ci in range(c):
        one = _reshape(_slice(ut_orig, ci, ci + 1, 1, 2), [1, hv, chunk, chunk])
        inv_i = _ut_inverse(one, chunk, [1, hv])
        parts.append(_reshape(inv_i, [1, hv, 1, chunk, chunk]))
    return op.concat(parts, axis=2) if c > 1 else parts[0]


def _ut_inverse(ut_orig, chunk, lead):
    """(I + L0)^-1 with L0 = strictly-lower(ut_orig), by the pin's OWN
    export-branch FORWARD SUBSTITUTION (pin 355-366), unrolled.

    `lead` is the static leading shape of `ut_orig` ([1, HV, C] here), so the
    two trailing axes are the chunk x chunk system. Transcription, line for
    line against the pin's `is_torchdynamo_exporting()` branch:

      pin 360  ut_system = -ut_system.tril(-1)
               -> A = negative(ut_orig * strictly_lower_ones)
      pin 361  for i in range(1, chunk_size):
               -> this Python loop; unrolled into the static graph
      pin 362  row = ut_system[..., i, :i].clone()
               -> row_i = A[..., i:i+1, :i]        (taken from A: row i has NOT
                  been substituted yet, which is what the pin's in-place write
                  order guarantees)
      pin 363  sub = ut_system[..., :i, :i].clone()
               -> sub = acc[..., :i, :i]           (the ALREADY-substituted rows
                  0..i-1 -- the pin reads them back after writing them in place)
      pin 364  ut_system[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
               -> reshape row to [..., i, 1], multiply by sub, reduce_sum over
                  axis -2 keepdims, add row. The broadcast-multiply-and-reduce
                  is emitted as written rather than folded into a matmul, so the
                  contraction is the pin's, not an equivalent one.
      pin 365  ut_system = ut_system + eye(chunk_size)
               -> add(acc, eye)
      pin 366  new_values, k_cumdecay = ut_system @ v_beta, @ decayed_k_beta
               -> at the call site, unchanged.

    The in-place row assignment of pin 364 has no opset-13 equivalent, so the
    rows are ACCUMULATED: `acc` carries the finished rows 0..i-1 at full chunk
    width (zero-padded past column i-1, which is what a strictly lower row is),
    and each new row is concatenated on. Row 0 is never written by the pin's
    loop, so it is taken from A unchanged.

    ~9 ops per chunk row, ~567 per GDN block against the old series' 12 matmuls.
    That growth is the point: see the module header for the f32 measurement that
    condemned the series."""
    R = len(lead) + 2                       # rank; rows axis R-2, cols axis R-1
    sl = _c(np.tril(np.ones((chunk, chunk), np.float32), -1))  # strictly lower ones
    eye = _c(np.eye(chunk, dtype=np.float32))
    a = op.negative(_mul(ut_orig, sl))      # pin 360: -ut_system.tril(-1)

    acc = _slice(a, 0, 1, 1, R - 2)         # row 0: the pin's loop starts at 1
    for i in range(1, chunk):               # pin 361
        row = _slice(_slice(a, i, i + 1, 1, R - 2), 0, i, 1, R - 1)   # pin 362
        sub = _slice(_slice(acc, 0, i, 1, R - 2), 0, i, 1, R - 1)     # pin 363
        rowc = _reshape(row, lead + [i, 1])          # pin 364: row.unsqueeze(-1)
        upd = _add(row, _rsum(_mul(rowc, sub), R - 2))  # pin 364: + (...).sum(-2)
        pad = _c(np.zeros(lead + [1, chunk - i], np.float32))
        acc = op.concat([acc, op.concat([upd, pad], axis=R - 1)], axis=R - 2)
    return _add(acc, eye)                   # pin 365


def _gate_activation(config):
    """The output gate's activation, the pin's rule (`output_gate_type or
    hidden_act`). The shipped Flash-Next checkpoint gates with a SIGMOID --
    llama.cpp hard-codes it for the architecture (qwen4exp.cpp: "sigmoid
    output gate, not silu") and the GGUF carries no key for it; the pin's
    default is hidden_act = silu, which is what every artifact before
    2026-09-18 was exported with (campaign serving-shape-logits)."""
    act = getattr(config, "output_gate_type", None) or config.hidden_act
    if act not in ("sigmoid", "silu"):
        raise ValueError(f"unsupported GDN output gate activation {act!r} "
                         "(sigmoid or silu)")
    return act


def _rmsnorm_gated(core, z, weight_vec, eps, last_axis, gate_act="silu"):
    """Qwen4ExpTextRMSNormGated over the last axis (reference lines 31-47):
    norm, weight, then the gate `ACT2FN[activation](z)`."""
    h = _mul(core, _rsqrt_eps(_rmean(_mul(core, core), last_axis), eps))
    h = _mul(_c(weight_vec.reshape([1] * last_axis + [-1])), h)
    return _mul(h, op.sigmoid(z) if gate_act == "sigmoid" else _silu(z))


# --- top-level emitter -----------------------------------------------------
def _gdn_subgraph(hidden, amask, config, state, T, ut_mode=None,
                  conv_emitter=None, core_emitter=None):
    """The GDN block body: hidden [1,T,H] f32 + amask [1,T] f32 -> out [1,T,H].
    Shared by build_gdn_model (standalone) and emit_gdn (the assembled
    backbone); the emitted ops are unchanged from the inc1 monolith.

    `ut_mode` selects how the chunk axis reaches the forward-substitution
    unroll -- see UT_EMIT_MODE. None takes the module default.

    `core_emitter` replaces the chunked gated delta rule, and only it: the
    projections, the conv, the norms, the gated RMSNorm and out_proj are the
    same ops either way. It exists for the same reason `conv_emitter` does --
    the serving path's transformation fuses a TOKEN-SEQUENTIAL v5::Loop into
    the op that becomes `gated_delta_state_table.N`, and the chunked rule this
    module writes is a different computation, not a near miss of that pattern.
    None takes the chunked default and nothing here emits a different op.

    `conv_emitter` replaces the depthwise causal conv, and ONLY that step. It
    is called exactly as `_causal_conv_silu` is and must return the same
    [1, conv_dim, T]. It exists so the serving-shape emitter can carry the
    conv's state in an ov Variable -- the construct the serving path's own
    SDPAToPagedAttention turns into `conv_state_table.N` and the four `la.*`
    index ports -- WITHOUT this module, which the parity suites gate, emitting
    one different op when nobody passes it. None takes the unrolled default."""
    mode = UT_EMIT_MODE if ut_mode is None else ut_mode
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
    pad = (CHUNK - T % CHUNK) % CHUNK
    Tp = T + pad
    C = Tp // CHUNK

    def w(name):
        return state[name]

    # apply_mask_to_padding_states: hidden * attention_mask[:, :, None]
    x = _mul(hidden, _reshape(amask, [1, T, 1]))

    # in-projections (Linear, bias=False): y = x @ W^T
    qkv = _mm(x, _c(w("in_proj_qkv.weight")), tb=True)  # [1,T,conv_dim]
    z = _mm(x, _c(w("in_proj_z.weight")), tb=True)       # [1,T,value_dim]
    b = _mm(x, _c(w("in_proj_b.weight")), tb=True)       # [1,T,HV]
    a = _mm(x, _c(w("in_proj_a.weight")), tb=True)       # [1,T,HV]

    # depthwise causal conv over the qkv channels, then split
    qkv_t = _transpose(qkv, [0, 2, 1])                    # [1,conv_dim,T]
    conv = (conv_emitter or _causal_conv_silu)(
        qkv_t, w("conv1d.weight"), T, conv_dim, K)
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
        khm = _key_head_map(config)
        query = _expand_heads(query, T, HK, Dk, ratio, khm)
        key = _expand_heads(key, T, HK, Dk, ratio, khm)

    # ---- chunked gated delta rule (transpose to [1,HV,T,*]) ----
    q = _transpose(query, [0, 2, 1, 3])                   # [1,HV,T,Dk]
    k = _transpose(key, [0, 2, 1, 3])
    v = _transpose(value, [0, 2, 1, 3])                   # [1,HV,T,Dv]
    beta_t = _transpose(beta, [0, 2, 1])                  # [1,HV,T]
    decay_t = _transpose(g, [0, 2, 1])                    # [1,HV,T]

    q = _l2norm_last(q, 3)
    k = _l2norm_last(k, 3)

    def _finish(core_bthd):
        """RMSNormGated over Dv, then out_proj -- the tail EVERY core returns
        through, which is the only way the sentence "the tail cannot drift
        between them" is true.

        It was not true when this was written: the perchunk branch, which is
        the module DEFAULT, kept its own copy of these three lines, so on the
        default configuration the serving core went through here and the
        parity-gated core did not -- exactly the pair the claim was about. The
        copy is gone. The emitted graph is unchanged either way: the arguments
        are built in the same order, so the ops are created in the same order
        with the same operands, and the node-count assertions in
        `test_the_chunk_emission_modes_agree_on_cpu_and_differ_in_node_count`
        would move if they were not."""
        normed = _rmsnorm_gated(core_bthd, z, w("norm.weight"), eps, 3,
                                _gate_activation(config))
        return _mm(_reshape(normed, [1, T, value_dim]),
                   _c(w("out_proj.weight")), tb=True)      # [1,T,H]

    if core_emitter is not None:
        # The chunked core below is skipped ENTIRELY -- not parameterised, not
        # partly reused. `core_emitter` gets the normalised q/k/v and the two
        # per-head sequences and returns the same [1, T, HV, Dv] the chunked
        # path has at its own `core = _transpose(...)`. Note q is handed over
        # UNSCALED: the serving core applies the head-size scale in the form
        # its own matcher demands, which is a Divide by a ShapeOf-derived
        # Power and not this Multiply by a folded constant.
        return _finish(core_emitter(q, k, v, beta_t, decay_t, T, HV, Dk, Dv))

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

    add_mask = np.where(np.triu(np.ones((CHUNK, CHUNK), np.float32), 1) > 0,
                        -1e30, 0.0).astype(np.float32)

    if mode == "perchunk":
        # The chunk axis never becomes a tensor axis: it is a Python loop, and
        # every emitted op below is rank 4 with leading shape [1, HV, ...] --
        # the same leading shape the single-chunk T=64 graph has, which is the
        # only shape measured clean on this plugin. The arithmetic is the
        # batched path's, term for term; only the shapes the ops see differ.
        last = _c(np.zeros((1, HV, Dk, Dv), np.float32))
        cores = []
        for ci in range(C):
            lo, hi = ci * CHUNK, (ci + 1) * CHUNK
            q_i = _slice(q, lo, hi, 1, 2)                 # [1,HV,CHUNK,Dk]
            k_i = _slice(k, lo, hi, 1, 2)
            kb_i = _slice(k_beta, lo, hi, 1, 2)
            vb_i = _slice(v_beta, lo, hi, 1, 2)           # [1,HV,CHUNK,Dv]
            dec_i = _slice(decay_t, lo, hi, 1, 2)         # [1,HV,CHUNK]

            cum_i = op.cumsum(dec_i, op.constant(np.int64(2)))
            expc4 = _reshape(op.exp(cum_i), [1, HV, CHUNK, 1])
            pd_i = op.exp(_add(_sub(_reshape(cum_i, [1, HV, CHUNK, 1]),
                                    _reshape(cum_i, [1, HV, 1, CHUNK])),
                               _c(add_mask)))

            ut_i = _mul(_mm(kb_i, k_i, tb=True), pd_i)    # [1,HV,CHUNK,CHUNK]
            it = _mul(_mm(q_i, k_i, tb=True), pd_i)
            dkb_i = _mul(kb_i, expc4)
            inv_i = _ut_inverse(ut_i, CHUNK, [1, HV])     # pin 355-365
            nv = _mm(inv_i, vb_i)                         # pin 366
            kcd = _mm(inv_i, dkb_i)

            qd = _mul(q_i, expc4)
            cum_last_i = _slice(cum_i, CHUNK - 1, CHUNK, 1, 2)   # [1,HV,1]
            kd = _mul(k_i, _reshape(op.exp(_sub(cum_last_i, cum_i)),
                                    [1, HV, CHUNK, 1]))
            cd = _reshape(op.exp(cum_last_i), [1, HV, 1, 1])

            v_new = _sub(nv, _mm(kcd, last))
            inter = _mm(qd, last)
            cores.append(_add(inter, _mm(it, v_new)))     # [1,HV,CHUNK,Dv]
            last = _add(_mul(last, cd), _mm(kd, v_new, ta=True))

        core = op.concat(cores, axis=2) if C > 1 else cores[0]  # [1,HV,Tp,Dv]
        if pad > 0:
            core = _slice(core, 0, T, 1, 2)
        core = _transpose(core, [0, 2, 1, 3])            # [1,T,HV,Dv]
        return _finish(core)

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
    pd = op.exp(_add(_sub(cd_i, cd_j), _c(add_mask)))

    ut_orig = _mul(_mm(kb_c, k_c, tb=True), pd)           # [1,HV,C,CHUNK,CHUNK]
    intra = _mul(_mm(q_c, k_c, tb=True), pd)
    dkb = _mul(kb_c, expc5)                               # decayed_k_beta

    # pin 355-365: the export branch's forward substitution builds the inverse
    inv = _ut_inverse_chunked(ut_orig, CHUNK, HV, C, mode)
    # pin 366: new_values, k_cumdecay = ut_system @ v_beta, @ decayed_k_beta
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
    return _finish(core)


def emit_gdn(hidden, amask, config, state, seq_len, ut_mode=None,
             conv_emitter=None, core_emitter=None):
    """The GDN subgraph for the assembled backbone (E2 inc5b).

    `seq_len=None` builds the block dynamic in T (reshapes with -1). Only
    legal with BOTH hooks given: the default conv and the chunked core unroll
    over T and need it as a number."""
    if seq_len is None:
        assert conv_emitter is not None and core_emitter is not None, (
            "dynamic T needs the stateful conv and core hooks; the unrolled "
            "defaults bake the block length")
        return _gdn_subgraph(hidden, amask, config, state, -1, ut_mode,
                             conv_emitter, core_emitter)
    return _gdn_subgraph(hidden, amask, config, state, int(seq_len), ut_mode,
                         conv_emitter, core_emitter)


def build_gdn_model(config, state, seq_len, ut_mode=None, core_emitter=None,
                    sinks=None):
    """The standalone GDN block.

    `sinks` is required when `core_emitter` carries state: an emitter that
    appends an Assign has to get it onto the Model, and a Model built without
    it reads a state nothing ever writes. Pass the same list the emitter was
    built with.
    """
    H = config.hidden_size
    T = int(seq_len)
    hidden = op.parameter([1, T, H], Type.f32)
    hidden.set_friendly_name("hidden_states")
    amask = op.parameter([1, T], Type.f32)
    amask.set_friendly_name("attention_mask")
    out = _gdn_subgraph(hidden, amask, config, state, T, ut_mode, None,
                        core_emitter)
    result = op.result(out)
    result.set_friendly_name("output")
    return Model([result], list(sinks or []), [hidden, amask],
                 "qwen4_exp_gdn_block")


__all__ = ["build_gdn_model", "emit_gdn", "UT_EMIT_MODE", "CHUNK"]
