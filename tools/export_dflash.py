#!/usr/bin/env python3
"""Export the DFlash2 draft head (incoai/Qwen3.8-27B-DFlash2) to OpenVINO.

Stateless v1: one call per verification cycle.

    inputs:  ctx_feats [1, C, 25600] f32  -- target features (layers 5,19,33,
                                            47,61 concatenated) of the last C
                                            ACCEPTED positions, C <= 2048 (the
                                            head's sliding window)
             noise     [1, Q, 5120]  f32  -- target input embeddings of
                                            [anchor, mask x (Q-1)]
             positions [C+Q]         i64  -- absolute positions, ctx then block
    output:  draft_hidden [1, Q, 5120]    -- final-normed; rows 1..Q-1 feed the
                                            target lm_head + the selector

The selector (top-16 candidate path over codebooks) is host-side; its weights
go to a sidecar (dflash_selector.npz). The reference is z-lab/dflash
dflash/model.py; unlike the 3.6 MTP reconstruction this checkpoint is a
standard HF Qwen3 layout: plain RMSNorm (weight applied directly) and
rotate_half rope (half-split, NOT interleaved).
"""
import argparse
import json
import os

import numpy as np
import openvino as ov
from openvino import opset13 as op
from safetensors import safe_open

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTER = 5120, 32, 8, 128, 17408
N_LAYERS, THETA, EPS = 5, 1e7, 1e-6
CONV_K, GROUP_SIZE = 2, 16
# The draft's residual stream peaks at ~128k (measured per layer, 2026-09-01)
# while the noise input starts at ~0.05 -- more dynamic range than f16 holds
# once the norms square it; upstream serves this head in bf16. Two exact
# transformations fix f16: (1) fold 1/4 into the residual writers (o_proj,
# down_proj) and the noise, bringing the stream tensor under f16's 65504;
# (2) give each norm its own input pre-scale with eps scaled by pre^2 --
# rms(c*x, c^2*eps) == rms(x) identically -- so the squares inside the big
# stream norms (x1/256) and the tiny layer-0 input norm (x64) both land
# mid-range. Every step is an identity; the CPU parity gate proves it.
RESID_SCALE = 4.0
NORM_PRE_L0     = 64.0        # layer-0 input norm: values ~0.003..0.011 after the fold
NORM_PRE_STREAM = 1.0 / 256.0 # later stream norms: values up to ~32k after the fold
GROUPS = HIDDEN // GROUP_SIZE
N_FEATS = 5 * HIDDEN


def const(a, name=None):
    c = op.constant(np.ascontiguousarray(a, dtype=np.float32))
    if name:
        c.set_friendly_name(name)
    return c


def rms(x, weight, name, pre=1.0):
    """RMSNorm with an f16-range pre-scale: rms(pre*x, pre^2*eps) == rms(x)
    exactly, for any pre. The pre only moves the squares into f16 range."""
    if pre != 1.0:
        x = op.multiply(x, const(np.array([pre]), f"{name}_pre"))
    axis = op.constant(np.array([-1], dtype=np.int32))
    mean = op.reduce_mean(op.multiply(x, x), axis, keep_dims=True)
    inv = op.divide(const(np.array([1.0])),
                    op.sqrt(op.add(mean, const(np.array([EPS * pre * pre])))))
    return op.multiply(op.multiply(x, inv), const(weight, name))


def linear(x, weight, name):
    return op.matmul(x, const(weight, name), transpose_a=False, transpose_b=True)


def i32(v):
    return op.constant(np.array(v if isinstance(v, (list, tuple)) else [v], dtype=np.int32))


def rotate_half_rope(x, cos, sin):
    """x [1, heads, L, 128]; cos/sin [L, 128] -> standard Qwen3 rope."""
    half = HEAD_DIM // 2
    x1 = op.slice(x, i32(0), i32(half), i32(1), i32(-1))
    x2 = op.slice(x, i32(half), i32(2**31 - 1), i32(1), i32(-1))
    rot = op.concat([op.negative(x2), x1], axis=-1)
    cs = op.unsqueeze(op.unsqueeze(cos, i32(0)), i32(0))   # [1,1,L,128]
    sn = op.unsqueeze(op.unsqueeze(sin, i32(0)), i32(0))
    return op.add(op.multiply(x, cs), op.multiply(rot, sn))


def dyn_conv_arm(x, dyn_arm, base_arm, name):
    """One GroupedDynamicCausalConv application (prepare or finish).
    x [1,L,5120]; dyn_arm [1,L,K,320]; base_arm np [K,5120]."""
    blocks = op.reshape(x, i32([0, 0, GROUPS, GROUP_SIZE]), special_zero=True)
    out = None
    for off in range(CONV_K):
        if off == 0:
            shifted = blocks
        else:
            zeros = const(np.zeros((1, off, GROUPS, GROUP_SIZE)))
            kept = op.slice(blocks, i32(0), i32(-off), i32(1), i32(1))
            shifted = op.concat([zeros, kept], axis=1)
        d = op.squeeze(op.gather(dyn_arm, i32(off), i32(2)), i32(2))  # [1,L,320]
        d = op.unsqueeze(d, i32(-1))                                  # [1,L,320,1]
        b = const(base_arm[off].reshape(1, 1, GROUPS, GROUP_SIZE), f"{name}_base{off}")
        term = op.multiply(shifted, op.add(b, d))
        out = term if out is None else op.add(out, term)
    return op.reshape(out, i32([0, 0, HIDDEN]), special_zero=True)


def expand_kv(t):
    """[1, KV, L, D] -> [1, HEADS, L, D] for GQA."""
    rep = HEADS // KV_HEADS
    t = op.unsqueeze(t, i32(2))
    sh = op.shape_of(t, ov.Type.i32)
    part = lambda a, b: op.slice(sh, i32(a), i32(b), i32(1), i32(0))
    t = op.broadcast(t, op.concat([part(0, 2), i32(rep), part(3, 5)], axis=0))
    return op.reshape(t, op.concat([part(0, 1), i32(HEADS), part(3, 5)], axis=0),
                      special_zero=False)


def kv_state(t, layer, kind, window):
    """Append [1,KV,P,128] to per-layer state, trim to the last `window`."""
    shape = ov.PartialShape([1, KV_HEADS, -1, HEAD_DIM])
    init = op.broadcast(const(np.array([0.0])),
                        op.constant(np.array([1, KV_HEADS, 0, HEAD_DIM], dtype=np.int32)))
    info = ov.op.util.VariableInfo()
    info.data_shape = shape
    info.data_type = ov.Type.f32
    info.variable_id = f"dflash.{kind}.{layer}"
    var = ov.op.util.Variable(info)
    past = op.read_value(init, var)
    cat = op.concat([past, t], axis=2)
    i64c = lambda v: op.constant(np.array([v], dtype=np.int64))
    trimmed = op.slice(cat, i64c(-window), i64c(2**31 - 1), i64c(1), i64c(2))
    sink = op.assign(trimmed, var)
    return trimmed, sink


def build_stateful(w, window):
    """One call per verification cycle: the context K/V live in graph state and
    only ever receive ACCEPTED positions, so there is nothing to roll back.

    inputs:  new_feats [1,P,25600]  target features of the P newly accepted
                                    positions (first call: the prompt window)
             noise     [1,Q,5120]   [anchor embed, mask embed x (Q-1)]
             positions [P+Q] i64    absolute positions, new ctx then block
    output:  draft_hidden [1,Q,5120]
    """
    feats_in = op.parameter(ov.PartialShape([1, -1, N_FEATS]), ov.Type.f32, name="new_feats")
    noise = op.parameter(ov.PartialShape([1, -1, HIDDEN]), ov.Type.f32, name="noise")
    positions = op.parameter(ov.PartialShape([-1]), ov.Type.i64, name="positions")

    ctx = rms(linear(feats_in, w["fc.weight"], "fc"), w["hidden_norm.weight"], "hidden_norm")

    inv = (1.0 / (THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM))).astype(np.float32)
    ang = op.multiply(op.unsqueeze(op.convert(positions, ov.Type.f32), i32(-1)),
                      const(inv.reshape(1, -1), "inv_freq"))
    emb = op.concat([ang, ang], axis=-1)
    cos_all, sin_all = op.cos(emb), op.sin(emb)
    i64c = lambda v: op.constant(np.array([v], dtype=np.int64))
    p_len = op.gather(op.shape_of(feats_in, ov.Type.i64), i32(1), i32(0))
    q_len = op.gather(op.shape_of(noise, ov.Type.i64), i32(1), i32(0))
    ax0 = i64c(0)
    one = i64c(1)
    cos_ctx = op.slice(cos_all, i64c(0), p_len, one, ax0)
    sin_ctx = op.slice(sin_all, i64c(0), p_len, one, ax0)
    cos_q = op.slice(cos_all, op.negative(q_len), i64c(2**31 - 1), one, ax0)
    sin_q = op.slice(sin_all, op.negative(q_len), i64c(2**31 - 1), one, ax0)

    perm = op.constant(np.array([0, 2, 1, 3], dtype=np.int32))
    sinks = []
    h = op.multiply(noise, const(np.array([1.0 / RESID_SCALE]), "resid_scale"))
    for L in range(N_LAYERS):
        p = f"layers.{L}"
        res = h
        x = rms(h, w[f"{p}.input_layernorm.weight"], f"{p}.in_ln",
                pre=NORM_PRE_L0 if L == 0 else NORM_PRE_STREAM)
        dyn = op.reshape(linear(x, w[f"{p}.attention_conv.kernel_projection.weight"], f"{p}.aconv_proj"),
                         i32([0, 0, 2, CONV_K, GROUPS]), special_zero=True)
        aprep = op.squeeze(op.gather(dyn, i32(0), i32(2)), i32(2))
        afin = op.squeeze(op.gather(dyn, i32(1), i32(2)), i32(2))
        x = dyn_conv_arm(x, aprep, w[f"{p}.attention_conv.base_kernel"][0], f"{p}.aconv0")

        q = op.reshape(linear(x, w[f"{p}.self_attn.q_proj.weight"], f"{p}.q"),
                       i32([0, 0, HEADS, HEAD_DIM]), special_zero=True)
        q = op.transpose(rms(q, w[f"{p}.self_attn.q_norm.weight"], f"{p}.qn"), perm)
        q = rotate_half_rope(q, cos_q, sin_q)

        def kv(src, cs, sn, roped=True):
            kk = op.reshape(linear(src, w[f"{p}.self_attn.k_proj.weight"], None),
                            i32([0, 0, KV_HEADS, HEAD_DIM]), special_zero=True)
            kk = op.transpose(rms(kk, w[f"{p}.self_attn.k_norm.weight"], None), perm)
            kk = rotate_half_rope(kk, cs, sn)
            vv = op.transpose(op.reshape(linear(src, w[f"{p}.self_attn.v_proj.weight"], None),
                                         i32([0, 0, KV_HEADS, HEAD_DIM]), special_zero=True), perm)
            return kk, vv

        k_new, v_new = kv(ctx, cos_ctx, sin_ctx)
        k_state, sk = kv_state(k_new, L, "key", window)
        v_state, sv = kv_state(v_new, L, "value", window)
        sinks += [sk, sv]
        k_noise, v_noise = kv(x, cos_q, sin_q)
        k = op.concat([k_state, k_noise], axis=2)
        v = op.concat([v_state, v_noise], axis=2)

        if L == 0:
            tap(q, "t_q0"); tap(k, "t_k0"); tap(v, "t_v0"); tap(x, "t_x0")
        a = op.scaled_dot_product_attention(q, expand_kv(k), expand_kv(v), causal=False)
        if L == 0:
            tap(a, "t_sdpa0")
        a = op.reshape(op.transpose(a, perm), i32([0, 0, HEADS * HEAD_DIM]), special_zero=True)
        a = linear(a, w[f"{p}.self_attn.o_proj.weight"], f"{p}.o")
        a = dyn_conv_arm(a, afin, w[f"{p}.attention_conv.base_kernel"][1], f"{p}.aconv1")
        h = op.add(res, a)
        if L == 0:
            tap(h, "t_h0")

        res = h
        x = rms(h, w[f"{p}.post_attention_layernorm.weight"], f"{p}.post_ln",
                pre=NORM_PRE_STREAM)
        mdyn = op.reshape(linear(x, w[f"{p}.mlp_conv.kernel_projection.weight"], f"{p}.mconv_proj"),
                          i32([0, 0, 2, CONV_K, GROUPS]), special_zero=True)
        mprep = op.squeeze(op.gather(mdyn, i32(0), i32(2)), i32(2))
        mfin = op.squeeze(op.gather(mdyn, i32(1), i32(2)), i32(2))
        x = dyn_conv_arm(x, mprep, w[f"{p}.mlp_conv.base_kernel"][0], f"{p}.mconv0")
        g = linear(x, w[f"{p}.mlp.gate_proj.weight"], f"{p}.gate")
        u = linear(x, w[f"{p}.mlp.up_proj.weight"], f"{p}.up")
        d = linear(op.multiply(op.multiply(g, op.sigmoid(g)), u),
                   w[f"{p}.mlp.down_proj.weight"], f"{p}.down")
        d = dyn_conv_arm(d, mfin, w[f"{p}.mlp_conv.base_kernel"][1], f"{p}.mconv1")
        h = op.add(res, d)

    out = rms(h, w["norm.weight"], "final_norm", pre=NORM_PRE_STREAM)
    res = op.result(out)
    res.get_output_tensor(0).set_names({"draft_hidden"})
    return ov.Model([res], sinks, [feats_in, noise, positions], "qwen38_dflash2_draft_stateful")


DEBUG_TAPS = []

def tap(node, name):
    if DEBUG_TAPS is not None and isinstance(DEBUG_TAPS, list):
        DEBUG_TAPS.append((name, node))
    return node


def build(w):
    ctx_in = op.parameter(ov.PartialShape([1, -1, N_FEATS]), ov.Type.f32, name="ctx_feats")
    noise = op.parameter(ov.PartialShape([1, -1, HIDDEN]), ov.Type.f32, name="noise")
    positions = op.parameter(ov.PartialShape([-1]), ov.Type.i64, name="positions")

    ctx = tap(rms(linear(ctx_in, w["fc.weight"], "fc"), w["hidden_norm.weight"], "hidden_norm"),
              "t_ctx")

    # rope tables over ctx+block; q takes the last Q rows
    inv = (1.0 / (THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM))).astype(np.float32)
    ang = op.multiply(op.unsqueeze(op.convert(positions, ov.Type.f32), i32(-1)),
                      const(inv.reshape(1, -1), "inv_freq"))
    emb = op.concat([ang, ang], axis=-1)                                # [L,128]
    cos_all, sin_all = tap(op.cos(emb), "t_cos"), op.sin(emb)
    q_len = op.gather(op.shape_of(noise, ov.Type.i64), i32(1), i32(0))  # [1]
    neg_q = op.negative(q_len)
    cos_q = op.slice(cos_all, neg_q, op.constant(np.array([2**31 - 1], dtype=np.int64)),
                     op.constant(np.array([1], dtype=np.int64)), op.constant(np.array([0], dtype=np.int64)))
    sin_q = op.slice(sin_all, neg_q, op.constant(np.array([2**31 - 1], dtype=np.int64)),
                     op.constant(np.array([1], dtype=np.int64)), op.constant(np.array([0], dtype=np.int64)))

    perm = op.constant(np.array([0, 2, 1, 3], dtype=np.int32))
    h = op.multiply(noise, const(np.array([1.0 / RESID_SCALE]), "resid_scale"))
    for L in range(N_LAYERS):
        p = f"layers.{L}"
        res = h
        x = rms(h, w[f"{p}.input_layernorm.weight"], f"{p}.in_ln",
                pre=NORM_PRE_L0 if L == 0 else NORM_PRE_STREAM)
        dyn = op.reshape(linear(x, w[f"{p}.attention_conv.kernel_projection.weight"], f"{p}.aconv_proj"),
                         i32([0, 0, 2, CONV_K, GROUPS]), special_zero=True)
        aprep = op.squeeze(op.gather(dyn, i32(0), i32(2)), i32(2))   # [1,Q,K,320]
        afin = op.squeeze(op.gather(dyn, i32(1), i32(2)), i32(2))
        x = dyn_conv_arm(x, aprep, w[f"{p}.attention_conv.base_kernel"][0], f"{p}.aconv0")

        q = op.reshape(linear(x, w[f"{p}.self_attn.q_proj.weight"], f"{p}.q"),
                       i32([0, 0, HEADS, HEAD_DIM]), special_zero=True)
        q = op.transpose(rms(q, w[f"{p}.self_attn.q_norm.weight"], f"{p}.qn"), perm)
        kin = op.concat([ctx, x], axis=1)
        k = op.reshape(linear(kin, w[f"{p}.self_attn.k_proj.weight"], f"{p}.k"),
                       i32([0, 0, KV_HEADS, HEAD_DIM]), special_zero=True)
        k = op.transpose(rms(k, w[f"{p}.self_attn.k_norm.weight"], f"{p}.kn"), perm)
        v = op.transpose(op.reshape(linear(kin, w[f"{p}.self_attn.v_proj.weight"], f"{p}.v"),
                                    i32([0, 0, KV_HEADS, HEAD_DIM]), special_zero=True), perm)

        q = rotate_half_rope(q, cos_q, sin_q)
        k = rotate_half_rope(k, cos_all, sin_all)

        if L == 0:
            tap(q, "t_q0"); tap(k, "t_k0"); tap(v, "t_v0"); tap(x, "t_x0")
        a = op.scaled_dot_product_attention(q, expand_kv(k), expand_kv(v), causal=False)
        if L == 0:
            tap(a, "t_sdpa0")
        a = op.reshape(op.transpose(a, perm), i32([0, 0, HEADS * HEAD_DIM]), special_zero=True)
        a = linear(a, w[f"{p}.self_attn.o_proj.weight"], f"{p}.o")
        if L == 0:
            tap(a, "t_o0")
        a = dyn_conv_arm(a, afin, w[f"{p}.attention_conv.base_kernel"][1], f"{p}.aconv1")
        h = op.add(res, a)
        if L == 0:
            tap(h, "t_h0")

        res = h
        x = rms(h, w[f"{p}.post_attention_layernorm.weight"], f"{p}.post_ln",
                pre=NORM_PRE_STREAM)
        mdyn = op.reshape(linear(x, w[f"{p}.mlp_conv.kernel_projection.weight"], f"{p}.mconv_proj"),
                          i32([0, 0, 2, CONV_K, GROUPS]), special_zero=True)
        mprep = op.squeeze(op.gather(mdyn, i32(0), i32(2)), i32(2))
        mfin = op.squeeze(op.gather(mdyn, i32(1), i32(2)), i32(2))
        x = dyn_conv_arm(x, mprep, w[f"{p}.mlp_conv.base_kernel"][0], f"{p}.mconv0")
        g = linear(x, w[f"{p}.mlp.gate_proj.weight"], f"{p}.gate")
        u = linear(x, w[f"{p}.mlp.up_proj.weight"], f"{p}.up")
        d = linear(op.multiply(op.multiply(g, op.sigmoid(g)), u),
                   w[f"{p}.mlp.down_proj.weight"], f"{p}.down")
        d = dyn_conv_arm(d, mfin, w[f"{p}.mlp_conv.base_kernel"][1], f"{p}.mconv1")
        h = op.add(res, d)
        if L == 0:
            tap(h, "t_l0out")

    out = rms(h, w["norm.weight"], "final_norm", pre=NORM_PRE_STREAM)
    res = op.result(out)
    res.get_output_tensor(0).set_names({"draft_hidden"})
    results = [res]
    for name, node in DEBUG_TAPS:
        r = op.result(node)
        r.get_output_tensor(0).set_names({name})
        results.append(r)
    return ov.Model(results, [ctx_in, noise, positions], "qwen38_dflash2_draft")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/models/gptq/qwen38-dflash2")
    ap.add_argument("--out", default="/models/gptq/qwen38-dflash2")
    args = ap.parse_args()

    import torch  # bf16 tensors: numpy has no bfloat16, torch casts them
    w = {}
    with safe_open(os.path.join(args.src, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            w[k] = f.get_tensor(k).to(torch.float32).numpy()
            if k.endswith("self_attn.o_proj.weight") or k.endswith("mlp.down_proj.weight"):
                w[k] = w[k] / RESID_SCALE

    model = build(w)
    ov.save_model(model, os.path.join(args.out, "openvino_dflash_draft.xml"),
                  compress_to_fp16=True)
    stateful = build_stateful(w, window=2048)
    ov.save_model(stateful, os.path.join(args.out, "openvino_dflash_draft_stateful.xml"),
                  compress_to_fp16=True)

    # raw sidecars for the C++ selector
    w["candidate_selector.hidden_projection.weight"].astype(np.float32).tofile(
        os.path.join(args.out, "dflash_hidden_projection.f32.bin"))
    w["candidate_selector.predecessor_codebook"].astype(np.float16).tofile(
        os.path.join(args.out, "dflash_predecessor_codebook.f16.bin"))
    w["candidate_selector.successor_codebook"].astype(np.float16).tofile(
        os.path.join(args.out, "dflash_successor_codebook.f16.bin"))

    cfg = json.load(open(os.path.join(args.src, "config.json")))["dflash_config"]
    np.savez(os.path.join(args.out, "dflash_selector.npz"),
             hidden_projection=w["candidate_selector.hidden_projection.weight"].astype(np.float16),
             predecessor_codebook=w["candidate_selector.predecessor_codebook"].astype(np.float16),
             successor_codebook=w["candidate_selector.successor_codebook"].astype(np.float16),
             top_k=np.int64(cfg["selector_top_k"]),
             block_size=np.int64(cfg["block_size"]),
             mask_token_id=np.int64(cfg["mask_token_id"]),
             target_layer_ids=np.asarray(cfg["target_layer_ids"], dtype=np.int64))
    print("saved openvino_dflash_draft.xml + dflash_selector.npz in", args.out)


if __name__ == "__main__":
    main()
