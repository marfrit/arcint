#!/usr/bin/env python3
"""Build the OpenVINO IRs for Qwen3.8-Flash-Next's MTP draft layer (P4).

The layer is Strata's (`code`: ~/src/Strata-ref/include/strata/core/mtp.hpp:1-20,
src/core/mtp.cpp:413-585 `MtpDrafter::record_forward`). Cell i pairs the main
model's final multi-stream residual R_i (`layer47/out`, hc*H wide) with the
token at i+1, at rope position i; its output predicts the token at i+2, and its
own residual R_out feeds the next draft step:

    e = fc_embedding(rms(embed(tok)) * (1 + w))
    h = fc_hidden per stream (rms over all hc*H of R, * (1 + w))
    R = h + e (e broadcast to every stream)
    -> attention hyper-connection -> attention (own K/V) -> recombine
    -> MLP hyper-connection -> MoE (512 experts, top-10, gated shared expert)
    -> recombine -> final mixer -> the main model's head

As in Strata, the attention is DENSE over every cell the layer has seen (the
model's sparse selection is identical below 2,051 cells) and needs no indexer
state; its K/V live in two state variables. The `kv_len` input keeps the first
`kv_len` cells of that state before the new ones are appended, so speculative
cells (draft steps, rejected window rows) are dropped by the next call instead
of rolled back -- Strata's "simply overwritten when their positions are
processed again".

Precision: drafts only move acceptance; the verify window decides every
emitted token. The dense projections are stored f16, the 512 routed experts
grouped 4-bit (`--expert-bits 8` for 8-bit), ~1.26 GB on the card; the main
model's experts are untouched (they are not in this IR).

Outputs, two IRs in --out:
  openvino_mtp_layer.xml     inputs  hidden_states [1,T,hc*H] f32, input_embeds
                             [1,T,H] f32, position_ids [1,T] i64, kv_len [1] i64;
                             outputs mtp_hidden [1,T,H] (the mixer row, for the
                             head) and mtp_residual [1,T,hc*H] (R_out, for the
                             next draft step)
  openvino_mtp_lm_head.xml   the main model's head, cut from --base
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import openvino as ov
from openvino import opset13 as op
from openvino import Type
from openvino.op import util as ovutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_mtp import (load_mtp_tensors, extract_lm_head, pick_group_size,  # noqa: E402
                        compressed_weight)
from q4e import hc as qhc  # noqa: E402
from q4e import attention as qattn  # noqa: E402
from q4e import gdn as qgdn  # noqa: E402
from q4e.serving_shape import (_recombine, _repeat_kv_broadcast,  # noqa: E402
                               _additive_causal_mask, emit_moe_tiled, SparseArena,
                               EXPERT_GROUP_SIZE)
from q4e.expert_fill import ExpertFiller  # noqa: E402

i64 = lambda v: op.constant(np.array(v, np.int64))  # noqa: E731
i32 = lambda v: op.constant(np.array(v, np.int32))  # noqa: E731


def f32c(a):
    return op.constant(np.ascontiguousarray(a, dtype=np.float32))


def rms_plus_one(x, weight, eps):
    """Qwen4ExpTextRMSNorm over the last axis, applied as (1 + w)."""
    var = op.reduce_mean(op.multiply(x, x), i32([-1]), keep_dims=True)
    xn = op.multiply(x, op.divide(f32c(1.0), op.sqrt(op.add(var, f32c(eps)))))
    return op.multiply(xn, f32c(1.0 + np.asarray(weight, np.float32)))


def matmul_t(x, w):
    return op.matmul(x, f32c(w), transpose_a=False, transpose_b=True)


def compressed_weight_u4(arr3d, name, group_size=None):
    """The 4-bit form of export_mtp.compressed_weight: the same decompression
    chain (Convert, Subtract zero point, Multiply scale, Reshape rank 4 -> 3,
    Convert f32) with u4 codes and an ASYMMETRIC per-group zero point
    (min/max over the group). OpenVINO packs a u4 Constant two per byte,
    element 0 in the low nibble (probed on this build).

    Not symmetric with a constant zero point 8: those zero-point arrays are
    identical for the gate and up weights, the graph shares one Constant
    between them, and the GPU program builder refuses the reorder it puts in
    front of the fused MoE ("Node which is about to be added in between two
    other nodes should not have any existing dependencies", measured
    2026-10-03 on the B60). Per-group zero points are the checkpoint-style
    form the main model's experts compile with."""
    E, out, in_ = arr3d.shape
    if group_size is None:
        group_size = pick_group_size(in_, preferred=(32, 64, 16, 8, 4, 2))
    groups = in_ // group_size
    g = arr3d.reshape(E, out, groups, group_size)
    lo = np.minimum(g.min(axis=-1, keepdims=True), 0.0)
    hi = np.maximum(g.max(axis=-1, keepdims=True), 0.0)
    scale = np.where(hi > lo, (hi - lo) / 15.0, 1.0).astype(np.float32)
    zp_u8 = np.clip(np.round(-lo / scale), 0, 15).astype(np.uint8)
    q = np.clip(np.round(g / scale) + zp_u8, 0, 15).astype(np.uint8)

    def u4_const(codes, shape):
        flat = codes.reshape(-1)
        if flat.size % 2:
            flat = np.concatenate([flat, np.zeros(1, np.uint8)])
        packed = (flat[0::2] & 0xF) | ((flat[1::2] & 0xF) << 4)
        t = ov.Tensor(Type.u4, list(shape))
        t.data[:] = packed.reshape(t.data.shape)
        return op.constant(t)

    q_c = u4_const(q, q.shape)
    q_c.set_friendly_name(name)
    zp = u4_const(zp_u8, scale.shape)
    zp.set_friendly_name(name + "/zero_point")
    sub16 = op.subtract(op.convert(q_c, "f16"), op.convert(zp, "f16"))
    sc = op.constant(scale.astype(np.float16))
    sc.set_friendly_name(name + "/scale")
    deq4 = op.multiply(sub16, sc)
    deq4.set_friendly_name(name + "/fq_weights_1")
    deq3 = op.reshape(deq4, i32([E, out, in_]), special_zero=False)
    deq3.set_friendly_name(name + "/fq_weights_1/reshape")
    deq = op.convert(deq3, "f32")
    deq.set_friendly_name(name + "/fq_weights_1/convert")
    return deq


def moe_tiled(y, w, p, topk, H, expert_bits):
    """export_mtp.moe_block_tiled's graph (the form the GPU plugin fuses),
    on [1, T, H] with this layer's tensors and 4- or 8-bit experts."""
    gu = w[p + "mlp.experts.gate_up_proj"]          # [E, 2I, H]
    dn = w[p + "mlp.experts.down_proj"]              # [E, H, I]
    E, two_i, _ = gu.shape
    I = two_i // 2
    cw = compressed_weight_u4 if expert_bits == 4 else compressed_weight
    y_flat = op.reshape(y, i32([-1, H]), special_zero=False)                  # [M,H]
    logits = matmul_t(y_flat, w[p + "mlp.gate.weight"])                     # [M,E]
    probs = op.softmax(logits, axis=-1)
    tk = op.topk(probs, i32(topk), axis=-1, mode="max", sort="value", index_element_type="i32")
    vals, idx = tk.output(0), tk.output(1)
    vals = op.divide(vals, op.reduce_sum(vals, i32([-1]), keep_dims=True))  # renormalised top-k
    vals = op.slice(vals, i32([0, 0]), op.shape_of(vals, output_type="i32"), i32([1, 1]), i32([0, 1]))
    zeros = op.multiply(probs, f32c(np.array([0.0], np.float32)))
    weights = op.scatter_elements_update(zeros, idx, vals, i32(-1))        # [M,E]

    tiled = op.tile(y_flat, i32([E, 1]))
    m_h3 = op.reshape(tiled, i32([E, -1, H]), special_zero=False)          # [E,M,H]

    def swish1(x):
        s = op.swish(x)
        s.set_arguments([s.input_value(0)])
        s.validate_and_infer_types()
        return s

    g = swish1(op.matmul(m_h3, cw(gu[:, :I, :], "experts_gate_proj"), transpose_a=False, transpose_b=True))
    u = op.matmul(m_h3, cw(gu[:, I:, :], "experts_up_proj"), transpose_a=False, transpose_b=True)
    outs = op.matmul(op.multiply(g, u), cw(dn, "experts_down_proj"), transpose_a=False, transpose_b=True)

    shape_y = op.shape_of(y, output_type="i32")
    b = op.slice(shape_y, i32([0]), i32([1]), i32([1]), i32([0]))
    outs4 = op.reshape(outs, op.concat([i32([E]), b, i32([-1]), i32([H])], axis=0), special_zero=False)
    w_t = op.transpose(weights, i32([1, 0]))
    w_r = op.reshape(w_t, op.concat([i32([E]), b, i32([-1])], axis=0), special_zero=False)
    mixed = op.reduce_sum(op.multiply(outs4, op.unsqueeze(w_r, i32([-1]))), i32([0]), keep_dims=False)

    sg = matmul_t(y, w[p + "mlp.shared_expert.gate_proj.weight"])
    su = matmul_t(y, w[p + "mlp.shared_expert.up_proj.weight"])
    shared = matmul_t(op.multiply(op.multiply(sg, op.sigmoid(sg)), su), w[p + "mlp.shared_expert.down_proj.weight"])
    shared = op.multiply(shared, op.sigmoid(matmul_t(y, w[p + "mlp.shared_expert_gate.weight"])))
    return op.add(mixed, shared)


def attention(h, pid, kv_len, cfg, w, p, rope_cos, rope_sin, sinks):
    """emit_stateful_attention's projections, norms, rope and gate, over the
    layer's own K/V state sliced to `kv_len` before the new cells join it."""
    H, heads, kv, d = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    eps = cfg.rms_norm_eps
    rotary = int(rope_cos.get_output_shape(0)[-1])
    T = -1
    s = lambda k: w[p + "self_attn." + k]  # noqa: E731
    qg = qgdn._reshape(qgdn._mm(h, qattn._c(s("q_proj.weight")), tb=True), [1, T, heads, 2 * d])
    q = qgdn._slice(qg, 0, d, 1, 3)
    gate = qgdn._reshape(qgdn._slice(qg, d, 2 * d, 1, 3), [1, T, heads * d])
    q = qgdn._transpose(qattn._rmsnorm_hd(q, s("q_norm.weight"), eps, d, True), [0, 2, 1, 3])
    k = qgdn._reshape(qgdn._mm(h, qattn._c(s("k_proj.weight")), tb=True), [1, T, kv, d])
    k = qgdn._transpose(qattn._rmsnorm_hd(k, s("k_norm.weight"), eps, d, True), [0, 2, 1, 3])
    v = qgdn._transpose(qgdn._reshape(qgdn._mm(h, qattn._c(s("v_proj.weight")), tb=True), [1, T, kv, d]),
                        [0, 2, 1, 3])
    q, k = qattn._apply_rope(q, k, rope_cos, rope_sin, pid, rotary, T)    # [1,heads,T,d], [1,kv,T,d]

    full = []
    for tag, cur in (("key", k), ("value", v)):
        info = ovutil.VariableInfo()
        info.data_shape = ov.PartialShape([1, kv, -1, d])
        info.data_type = Type.f32
        info.variable_id = f"mtp.{tag}"
        var = ovutil.Variable(info)
        init = op.broadcast(op.constant(np.array(0.0, np.float32)), i64([1, kv, 0, d]))
        past = op.slice(op.read_value(init, var), i64([0]), kv_len, i64([1]), i64([2]))
        joined = op.concat([past, cur], axis=2)
        sinks.append(op.assign(joined, var))
        full.append(joined)

    n_tok = op.reduce_prod(op.shape_of(pid, output_type="i64"), i64([0]), keep_dims=False)
    total = op.gather(op.shape_of(full[0], output_type="i64"), i64([2]), i64(0))
    mask = _additive_causal_mask(n_tok, total, kv_len)                    # [1,1,T,TOTAL]
    att = op.scaled_dot_product_attention(
        q, _repeat_kv_broadcast(full[0], kv, heads, d), _repeat_kv_broadcast(full[1], kv, heads, d),
        mask, op.constant(np.array(d ** -0.5, np.float32)), causal=False)  # [1,heads,T,d]
    out = qgdn._reshape(qgdn._transpose(att, [0, 2, 1, 3]), [1, T, heads * d])
    out = qgdn._mul(out, op.sigmoid(gate))
    return qgdn._mm(out, qattn._c(s("o_proj.weight")), tb=True)


def build_layer(w, cfg, rope_span, expert_bits):
    H, hc = cfg.hidden_size, cfg.hc_count
    eps = cfg.rms_norm_eps
    hidden = op.parameter(ov.PartialShape([1, -1, hc * H]), Type.f32, name="hidden_states")
    embeds = op.parameter(ov.PartialShape([1, -1, H]), Type.f32, name="input_embeds")
    pid = op.parameter(ov.PartialShape([1, -1]), Type.i64, name="position_ids")
    kv_len = op.parameter(ov.PartialShape([1]), Type.i64, name="kv_len")
    for prm, nm in ((hidden, "hidden_states"), (embeds, "input_embeds"), (pid, "position_ids"),
                    (kv_len, "kv_len")):
        prm.output(0).set_names({nm})

    # the two input branches (mtp.cpp:430-449)
    e2 = matmul_t(rms_plus_one(embeds, w["mtp.pre_fc_norm_embedding.weight"], eps),
                  w["mtp.fc_embedding.weight"])                                   # [1,T,H]
    hn = rms_plus_one(hidden, w["mtp.pre_fc_norm_hidden.weight"], eps)             # over all hc*H
    h2 = matmul_t(op.reshape(hn, i64([1, -1, hc, H]), special_zero=False),
                  w["mtp.fc_hidden.weight"])                                      # [1,T,hc,H] per stream
    R = op.add(h2, op.unsqueeze(e2, i64([2])))
    R = op.reshape(R, i64([1, -1, hc * H]), special_zero=False)                    # [1,T,hc*H]

    p = "mtp.layers.0."
    strip = lambda pre: {k[len(pre):]: v for k, v in w.items() if k.startswith(pre)}  # noqa: E731
    cos_np, sin_np = qattn._freqs_tables(cfg, rope_span)
    rope_cos, rope_sin = op.constant(cos_np), op.constant(sin_np)
    sinks = []

    h, hyper, inj = qhc.emit_combine(R, cfg, strip(p + "attn_hyper_connection."), None)
    a = attention(h, pid, kv_len, cfg, w, p, rope_cos, rope_sin, sinks)
    R = _recombine(hyper, inj, a, cfg, None)
    h, hyper, inj = qhc.emit_combine(R, cfg, strip(p + "mlp_hyper_connection."), None)
    if expert_bits == 4:
        # The main model's own MoE emitter and expert form (u4, asymmetric,
        # group EXPERT_GROUP_SIZE): the block that serves single tokens on the
        # card. moe_tiled() below (export_mtp's lowering, symmetric codes,
        # ShapeOf batch) compiled but failed its first one-token infer with
        # "Unable to cast reference from base to derived type" on the stock
        # and the patched plugin alike (B60, 2026-10-03); the shared expert
        # alone ran at T=1, T=3.
        gu = w[p + "mlp.experts.gate_up_proj"]
        dn = w[p + "mlp.experts.down_proj"]
        I = gu.shape[1] // 2
        rows = {"gate": gu[:, :I, :], "up": gu[:, I:, :], "down": dn}
        filler = ExpertFiller(lambda layer, kind: rows[kind], EXPERT_GROUP_SIZE)
        arena = SparseArena()
        m = emit_moe_tiled(h, cfg, strip(p), arena, None, "mtp/moe", filler=filler, layer=0)
        build_layer.arena = arena   # the constants wrap its pages until saved
    else:
        m = moe_tiled(h, w, p, cfg.num_experts_per_tok, H, expert_bits)
    R = _recombine(hyper, inj, m, cfg, None)

    out = qhc.emit_hc(R, cfg, strip("mtp.hyper_connection_mixer."), None)          # [1,T,H]
    r_hid = op.result(out)
    r_hid.output(0).set_names({"mtp_hidden"})
    r_res = op.result(R)
    r_res.output(0).set_names({"mtp_residual"})
    return ov.Model([r_hid, r_res], sinks, [hidden, embeds, pid, kv_len], "qwen4_exp_mtp")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="directory holding the mtp.* safetensors")
    ap.add_argument("--artifact", required=True, help="the Flash-Next serving artifact (config.json, "
                    "openvino_language_model.xml)")
    ap.add_argument("--out", required=True, help="directory to write the MTP IRs into")
    ap.add_argument("--rope-span", type=int, default=131072, help="positions the rope table covers")
    ap.add_argument("--expert-bits", type=int, choices=(4, 8), default=4)
    a = ap.parse_args()

    c = json.load(open(os.path.join(a.artifact, "config.json")))
    t = c.get("text_config", c)
    cfg = SimpleNamespace(
        hidden_size=int(t["hidden_size"]), hc_count=int(t["hc_count"]), hc_lowrank=int(t["hc_lowrank"]),
        rms_norm_eps=float(t.get("rms_norm_eps", 1e-6)), num_attention_heads=int(t["num_attention_heads"]),
        num_key_value_heads=int(t["num_key_value_heads"]), head_dim=int(t["head_dim"]),
        rope_parameters=dict(t["rope_parameters"]), num_experts_per_tok=int(t["num_experts_per_tok"]),
        num_experts=int(t.get("num_experts", 512)), moe_intermediate_size=int(t["moe_intermediate_size"]),
        shared_expert_intermediate_size=int(t.get("shared_expert_intermediate_size", t["moe_intermediate_size"])),
        norm_topk_prob=True, norm_plus_one=True)
    print(f"hidden {cfg.hidden_size} x hc {cfg.hc_count} (lowrank {cfg.hc_lowrank}), heads "
          f"{cfg.num_attention_heads}/{cfg.num_key_value_heads}x{cfg.head_dim}, top-{cfg.num_experts_per_tok}, "
          f"rope {cfg.rope_parameters}, experts {a.expert_bits}-bit")
    w = load_mtp_tensors(a.weights)
    print(f"loaded {len(w)} mtp tensors")

    os.makedirs(a.out, exist_ok=True)
    layer = build_layer(w, cfg, a.rope_span, a.expert_bits)
    layer.validate_nodes_and_infer_types()
    ov.save_model(layer, os.path.join(a.out, "openvino_mtp_layer.xml"), compress_to_fp16=True)
    print("wrote openvino_mtp_layer.xml")
    head = extract_lm_head(os.path.join(a.artifact, "openvino_language_model.xml"))
    ov.save_model(head, os.path.join(a.out, "openvino_mtp_lm_head.xml"), compress_to_fp16=False)
    print("wrote openvino_mtp_lm_head.xml")


if __name__ == "__main__":
    main()
