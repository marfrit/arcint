"""OV opset-13 emission of the qwen4_exp PLELayer (n-gram PLE block),
full-sequence, no-cache.

Mirrors `tools/q4e/ref_ple.Qwen4ExpTextPLELayer.forward` (itself the pinned
transformers reference, modeling_qwen4_exp.py 1235-1255) as a *static* graph:
batch fixed to 1 and sequence length fixed at build time.

THE N-GRAM ROW INDEX IS FED, NOT EMITTED (measured OV limitation). The
row-index -> table-row function is a pure INTEGER hash (pin 1131-1180,
byte-identical to arcint's vector-tested src/exec/ngram_row_ids.h): an
XOR-of-(token * splitmix-multiplier) reduced modulo a per-head prime vocab,
with eos-boundary shifting. It requires EXACT int64 arithmetic on values up to
~2^63. The installed OpenVINO build's CPU integer kernels are effectively
32-bit -- MEASURED on the dev host (openvino 2026.4.0-22849-71640275d29):

    i64 Multiply, [100000] * 100000  -> 1410065408  (== 1e10 mod 2^32; np: 1e10)
    i64 Multiply, [200]    * 35888608703715081 -> -200
    i64 Add,      [1] + 9000000000000000000     -> -2147483648 (INT32_MIN)

so both Multiply and Add wrap at 2^32 / saturate at int32 for i64 operands.

  FRONTIER RULING (dated 2026-09-11): in-graph 64-bit hash is NO-GO on OV
  2026.4.0 CPU (measured: multiply wraps mod 2^32, 1e5*1e5 -> 1410065408; add
  breaks 1+9e18 -> INT32_MIN). Design boundary: the hash lives in the runtime
  kernel; the graph consumes row_ids. row_ids are a DECLARED int64 GRAPH INPUT
  `ngram_row_ids` [1, T, num_ngram_heads]; this graph emits ONLY the gather +
  the (float) PLE forward, with NO float on the index path it touches (the fed
  ids are int64 and the only op consuming them is the dtype-agnostic Gather).
  The two PRODUCERS of the input, either side of the parity seam:
    * tests   -- the in-file numpy int64 generator (test_ple_block._gen_row_ids),
                 validated three-ways against the committed Link-3 vectors
                 (tests/ngram_row_ids_vectors.h) before any graph run;
    * serving -- arcint's vector-tested src/exec/ngram_row_ids.h (AVX2-verified;
                 the NEON twin is a separate queue item). Not reimplemented here.
  Deferred (optimization ticket, NEVER a correctness dependency): whether OV i64
  Multiply holds on GPU.0/GPU.1 -- window territory, untested this session.

The (float) forward (pin 1242-1255):
  embeddings = ngram_embedding[row_ids].flatten(-2)          (pin 1242, gather)
  key   = norm_key(key_proj(embeddings)).unflatten(hc, H)    (pin 1243)
  value = value_proj(embeddings)                             (pin 1244)
  query = norm_query(hidden_states).unflatten(hc, H)         (pin 1245)
  gate  = (key * query).sum(-1, keepdim) / sqrt(H)           (pin 1246)
  gate  = sqrt(|gate|.clamp_min(1e-6)) * sign(gate)          (pin 1247, signed sqrt)
  gated_value = sigmoid(gate) * value.unsqueeze(-2)          (pin 1248)
  gated_value_normed = norm_conv(gated_value.flatten(-2))    (pin 1249)
  gated_value        = gated_value.flatten(-2)               (pin 1250)
  (conv_mask, if given, zeroes both -- pin 1251-1253)
  output = gated_value + silu(dilated_depthwise_conv1d(gated_value_normed))  (pin 1254)

The three group-RMSNorms are the pin's Qwen4ExpTextRMSNorm with group_size = H
(the (1+w) zero-init weight in the f64-roundtrip lowering hc.py established);
the conv is depthwise (groups = hc*H), kernel `ple_conv_kernel_size`, dilation
`ngram_size`, causal via a left zero-pad of (K-1)*dilation then K dilated taps
(bit-for-bit F.pad + Conv1d, no GroupConvolution lowering to trust).

Entry points:
  build_ple_model(config, state, seq_len) -> ov.Model, inputs `hidden_states`
    [1, T, hc*H] f32 and `ngram_row_ids` [1, T, num_ngram_heads] i64, result
    `output` [1, T, hc*H] f32.
  emit_ple(hidden_bth, row_ids_node, config, state, seq_len) -> ov node
    [1, T, hc*H]: the same subgraph for the assembled backbone (E2 inc5b).
"""
import math

import numpy as np
from openvino import Model, Type
from openvino import opset13 as op

from .gdn import _c, _i, _mm, _mul, _add, _slice, _transpose, _reshape, _rmean, _rsum, _rsqrt_eps, _silu  # noqa: F401


def _group_rms(x, T, hc, H, weight_vec, eps):
    """Qwen4ExpTextRMSNorm with group_size = H (pin 152-172): reshape the last
    dim into the hc groups, normalize within each, apply (1+w) as hc.py's
    f64-roundtrip lowering (bit-identical to the pin's fp32 add), flatten. x:
    [1,T,hc*H] -> [1,T,hc*H]."""
    xg = _reshape(x, [1, T, hc, H])
    var = _rmean(_mul(xg, xg), 3)            # pin 164: mean(x^2) over the group
    xn = _mul(xg, _rsqrt_eps(var, eps))      # pin 164: x * rsqrt(var + eps)
    w64 = op.convert(_c(np.ascontiguousarray(weight_vec, np.float32).reshape(1, 1, hc, H)), Type.f64)
    ones64 = op.constant(np.ones((1, 1, hc, H), np.float64))
    wn = op.convert(op.add(ones64, w64), Type.f32)  # pin 171: (1 + w)
    xn = _mul(wn, xn)
    return _reshape(xn, [1, T, hc * H])      # pin 165: flatten(-2)


def _short_conv(x, conv_w, T, C, K, dilation):
    """Dilated depthwise causal conv1d + silu (pin 1216-1233). x: [1,T,C] ->
    [1,T,C]. conv_w: [C,1,K]. Causal via a left zero-pad of (K-1)*dilation then
    K dilated taps (F.pad(state_len,0) + Conv1d(dilation), pin 1226-1232)."""
    xt = _transpose(x, [0, 2, 1])            # [1, C, T]   (pin 1218)
    state_len = (K - 1) * dilation
    xpad = op.concat([_c(np.zeros((1, C, state_len), np.float32)), xt], axis=2)  # [1,C,state_len+T]
    acc = None
    for j in range(K):                       # depthwise dilated taps (pin 1230 conv1d)
        xs = _slice(xpad, j * dilation, j * dilation + T, 1, 2)  # [1, C, T]
        wj = _c(conv_w[:, 0, j].reshape(1, C, 1))
        term = _mul(xs, wj)
        acc = term if acc is None else _add(acc, term)
    out = _silu(acc)                         # pin 1230: F.silu
    return _transpose(out, [0, 2, 1])        # [1, T, C]   (pin 1232)


def _ple_subgraph(hidden, row_ids, config, state, T, conv_mask=None):
    """hidden: [1,T,hc*H] f32; row_ids: [1,T,Hn] i64 -> output [1,T,hc*H].
    conv_mask (optional): [1,T] f32; when given, zeroes both gated streams
    before the conv (pin 1251-1253 -- these apply_mask sites belong to THIS
    path, not the final mixer)."""
    H = config.hidden_size
    hc = config.hc_count
    eps = config.rms_norm_eps
    Hn = (config.ngram_size - 1) * config.heads_per_ngram
    ple_embed_dim = config.ple_embed_dim
    head_dim = ple_embed_dim // Hn
    K = config.ple_conv_kernel_size
    dilation = config.ngram_size

    # pin 1242: embeddings = ngram_embedding(ngram_ids).flatten(-2). The fed
    # int64 ids index the table; Gather is dtype-agnostic (no float, no i64
    # arithmetic) -- the index path this graph touches is exact.
    emb_w = _c(state["ple_embedding.ngram_embedding.weight"])  # [V, head_dim] f32
    gathered = op.gather(emb_w, row_ids, op.constant(np.int64(0)))  # [1,T,Hn,head_dim]
    emb = _reshape(gathered, [1, T, Hn * head_dim])                 # [1,T,ple_embed_dim]

    # pin 1243: key = norm_key(key_proj(emb)).unflatten(hc,H)
    key = _mm(emb, _c(state["key_proj.weight"]), tb=True)          # [1,T,hc*H]
    key_n = _group_rms(key, T, hc, H, state["norm_key.weight"], eps)
    key_n4 = _reshape(key_n, [1, T, hc, H])
    # pin 1244: value = value_proj(emb)
    value = _mm(emb, _c(state["value_proj.weight"]), tb=True)      # [1,T,H]
    # pin 1245: query = norm_query(hidden).unflatten(hc,H)
    q_n = _group_rms(hidden, T, hc, H, state["norm_query.weight"], eps)
    q_n4 = _reshape(q_n, [1, T, hc, H])

    # pin 1246: gate = (key*query).sum(-1,keepdim) / sqrt(H)
    gate = _rsum(_mul(key_n4, q_n4), 3)                            # [1,T,hc,1]
    gate = _mul(gate, _c(np.float32(1.0 / math.sqrt(H))))
    # pin 1247: signed sqrt = sqrt(|gate|.clamp_min(1e-6)) * sign(gate)
    ag = op.maximum(op.abs(gate), _c(np.float32(1e-6)))
    gate = _mul(op.sqrt(ag), op.sign(gate))
    # pin 1248: gated_value = sigmoid(gate) * value.unsqueeze(-2)
    sg = op.sigmoid(gate)                                          # [1,T,hc,1]
    value4 = _reshape(value, [1, T, 1, H])                        # unsqueeze(-2)
    gv = _mul(sg, value4)                                          # [1,T,hc,H]
    gv_flat = _reshape(gv, [1, T, hc * H])                         # pin 1250
    # pin 1249: gated_value_normed = norm_conv(gated_value.flatten(-2))
    gv_normed = _group_rms(gv_flat, T, hc, H, state["norm_conv.weight"], eps)

    # pin 1251-1253: zero both gated streams on masked rows (apply_mask_to_
    # padding_states = x * mask[:, :, None]). These sites are the PLE conv
    # path's, not the final mixer's.
    if conv_mask is not None:
        m = _reshape(conv_mask, [1, T, 1])
        gv_flat = _mul(gv_flat, m)
        gv_normed = _mul(gv_normed, m)

    # pin 1254: output = gated_value + silu(short_conv(gated_value_normed))
    conv_out = _short_conv(gv_normed, state["conv1d.weight"], T, hc * H, K, dilation)
    return _add(gv_flat, conv_out)


def emit_ple(hidden_bth, row_ids_node, config, state, seq_len, conv_mask=None):
    """The PLE subgraph for the assembled backbone (E2 inc5b)."""
    return _ple_subgraph(hidden_bth, row_ids_node, config, state, int(seq_len), conv_mask)


def build_ple_model(config, state, seq_len, with_mask=False):
    H = config.hidden_size
    hc = config.hc_count
    Hn = (config.ngram_size - 1) * config.heads_per_ngram
    T = int(seq_len)

    hidden = op.parameter([1, T, hc * H], Type.f32)
    hidden.set_friendly_name("hidden_states")
    row_ids = op.parameter([1, T, Hn], Type.i64)
    row_ids.set_friendly_name("ngram_row_ids")
    params = [hidden, row_ids]
    conv_mask = None
    if with_mask:
        conv_mask = op.parameter([1, T], Type.f32)
        conv_mask.set_friendly_name("conv_mask")
        params.append(conv_mask)

    out = _ple_subgraph(hidden, row_ids, config, state, T, conv_mask)
    res = op.result(out)
    res.set_friendly_name("output")
    return Model([res], params, "qwen4_exp_ple")


__all__ = ["build_ple_model", "emit_ple"]
