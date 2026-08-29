#!/usr/bin/env python3
"""Build an OpenVINO IR for the Qwen3.5 MTP head.

No public implementation consumes these weights (checked against transformers
5.16.1 and optimum-intel), so the forward pass here is reconstructed from the
tensor shapes and config.  It does not need a reference to be safe: arcint
accepts a drafted token only when it equals what the sampler would have picked
anyway, so a wrong head cannot change the answer -- it can only lower draft
acceptance.  Acceptance is therefore the oracle.  ~0% means this file is wrong;
45-75% means it is right.

    hidden      h_t, the base model's final hidden state at position t
    embedding   e_{t+1}, the embedding of the token just committed
    x  = fc( concat( rms(h_t, pre_fc_norm_hidden), rms(e_{t+1}, pre_fc_norm_embedding) ) )
    x  = x + gated_attn( rms(x, input_layernorm) )
    x  = x + swiglu( rms(x, post_attention_layernorm) )
    logits = lm_head( rms(x, mtp.norm) )          -> predicts token t+2

Text-only: the base export's mrope carries four identical position rows for
text, so plain RoPE over the rotary prefix is equivalent here.
"""
import argparse, glob, json, os
import numpy as np
import openvino as ov
from openvino import opset13 as op
from safetensors import safe_open

HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
HIDDEN = 5120


def load_mtp_tensors(src):
    """The head is stored bf16, which numpy has no dtype for, so widen via torch
    if it is available and by the bit layout (bf16 is the top half of an f32) if
    it is not."""
    try:
        import torch
        framework = "pt"
    except ImportError:
        framework = None

    out = {}
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with safe_open(f, framework=framework or "np") as h:
            for k in h.keys():
                if "mtp" not in k:
                    continue
                if framework == "pt":
                    out[k] = h.get_tensor(k).float().numpy()
                else:
                    raw = np.frombuffer(h.get_tensor(k).tobytes(), dtype=np.uint16)
                    out[k] = (raw.astype(np.uint32) << 16).view(np.float32).reshape(
                        h.get_slice(k).get_shape())
    if not out:
        raise SystemExit(f"no mtp.* tensors in {src}")
    return out


def shape_part(t, start, stop):
    """A slice of shape_of(t) as an i32 vector; Node has no Python indexing."""
    i32 = lambda v: op.constant(np.array([v], dtype=np.int32))
    return op.slice(op.shape_of(t, output_type="i32"), i32(start), i32(stop), i32(1), i32(0))


def const(a, name=None):
    c = op.constant(np.ascontiguousarray(a, dtype=np.float32))
    if name:
        c.friendly_name = name
    return c


def rms_norm(x, weight, eps, name):
    """Zero-centred RMSNorm: the scale is stored centred on 0 and applied as
    (1 + w).  Nothing states this, but the weights do -- pre_fc_norm_embedding
    is entirely negative, which is not a scale.  Under (1 + w) every norm in the
    head becomes a sensible positive one, and acceptance goes 0% -> 66%."""
    axis = op.constant(np.array([-1], dtype=np.int32))
    sq = op.multiply(x, x)
    mean = op.reduce_mean(sq, axis, keep_dims=True)
    inv = op.divide(op.constant(np.array([1.0], dtype=np.float32)),
                    op.sqrt(op.add(mean, op.constant(np.array([eps], dtype=np.float32)))))
    y = op.multiply(x, inv)
    return op.multiply(y, const(1.0 + weight, name))


def linear(x, weight, name):
    """y = x @ W^T for a torch-convention [out, in] weight."""
    return op.matmul(x, const(weight, name), transpose_a=False, transpose_b=True)


def rope(x, cos, sin, rotary_dim, heads):
    """Rotate the first `rotary_dim` channels as interleaved (even, odd) pairs
    and pass the rest through."""
    half = rotary_dim // 2
    i32 = lambda v: op.constant(np.array([v], dtype=np.int32))
    rot = op.slice(x, i32(0), i32(rotary_dim), i32(1), i32(-1))
    passthru = op.slice(x, i32(rotary_dim), i32(2 ** 31 - 1), i32(1), i32(-1))
    pairs = op.reshape(rot, op.constant(np.array([0, 0, heads, half, 2], dtype=np.int32)),
                       special_zero=True)
    ax = op.constant(np.array(4, dtype=np.int32))
    x1 = op.squeeze(op.gather(pairs, op.constant(np.array([0], dtype=np.int32)), ax), ax)
    x2 = op.squeeze(op.gather(pairs, op.constant(np.array([1], dtype=np.int32)), ax), ax)
    even = op.subtract(op.multiply(x1, cos), op.multiply(x2, sin))
    odd  = op.add(op.multiply(x1, sin), op.multiply(x2, cos))
    m1 = op.constant(np.array([-1], dtype=np.int32))
    out = op.concat([op.unsqueeze(even, m1), op.unsqueeze(odd, m1)], axis=-1)
    out = op.reshape(out, op.constant(np.array([0, 0, heads, rotary_dim], dtype=np.int32)),
                     special_zero=True)
    return op.concat([out, passthru], axis=-1)



def build_mtp_layer(w, eps, theta, rotary_dim):
    """hidden_states + input_embeds -> the MTP layer's normed hidden state."""
    dyn = ov.PartialShape([-1, -1, HIDDEN])
    hidden = op.parameter(dyn, ov.Type.f32, name="hidden_states")
    embeds = op.parameter(dyn, ov.Type.f32, name="input_embeds")
    pos    = op.parameter(ov.PartialShape([-1, -1]), ov.Type.f32, name="position_ids")
    mask   = op.parameter(ov.PartialShape([-1, 1, -1, -1]), ov.Type.f32, name="attention_mask")
    beam   = op.parameter(ov.PartialShape([-1]), ov.Type.i32, name="beam_idx")

    # --- fuse the base hidden state with the next token's embedding -----------
    x = op.concat([rms_norm(embeds, w["mtp.pre_fc_norm_embedding.weight"], eps, "pre_fc_e"),
                   rms_norm(hidden, w["mtp.pre_fc_norm_hidden.weight"], eps, "pre_fc_h")],
                  axis=-1)
    x = linear(x, w["mtp.fc.weight"], "fc")

    # --- rotary tables from the positions ------------------------------------
    half = rotary_dim // 2
    inv = (1.0 / (theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / rotary_dim))).astype(np.float32)
    freqs = op.multiply(op.unsqueeze(pos, op.constant(np.array([-1], dtype=np.int32))),
                        op.constant(inv.reshape(1, 1, half)))
    ax2 = op.constant(np.array([2], dtype=np.int32))
    cos = op.unsqueeze(op.cos(freqs), ax2)                    # [B,S,1,half]
    sin = op.unsqueeze(op.sin(freqs), ax2)

    # --- gated attention ------------------------------------------------------
    p = "mtp.layers.0."
    y = rms_norm(x, w[p + "input_layernorm.weight"], eps, "in_ln")

    # q_proj emits q and its gate interleaved *per head*, not as two contiguous
    # halves: [head0_q | head0_gate | head1_q | ...]. Splitting it the obvious
    # way costs 53 points of acceptance.
    qg = linear(y, w[p + "self_attn.q_proj.weight"], "q_proj")   # [B,S,24*2*256]
    qg = op.reshape(qg, op.constant(np.array([0, 0, HEADS, 2, HEAD_DIM], dtype=np.int32)),
                    special_zero=True)
    ax3 = op.constant(np.array(3, dtype=np.int32))
    q = op.squeeze(op.gather(qg, op.constant(np.array([0], dtype=np.int32)), ax3), ax3)
    gate = op.reshape(
        op.squeeze(op.gather(qg, op.constant(np.array([1], dtype=np.int32)), ax3), ax3),
        op.constant(np.array([0, 0, HEADS * HEAD_DIM], dtype=np.int32)), special_zero=True)

    def heads(t, n):
        shape = op.constant(np.array([0, 0, n, HEAD_DIM], dtype=np.int32))
        return op.reshape(t, shape, special_zero=True)

    k = heads(linear(y, w[p + "self_attn.k_proj.weight"], "k_proj"), KV_HEADS)
    v = heads(linear(y, w[p + "self_attn.v_proj.weight"], "v_proj"), KV_HEADS)

    q = rope(rms_norm(q, w[p + "self_attn.q_norm.weight"], eps, "q_norm"), cos, sin,
             rotary_dim, HEADS)
    k = rope(rms_norm(k, w[p + "self_attn.k_norm.weight"], eps, "k_norm"), cos, sin,
             rotary_dim, KV_HEADS)

    perm = op.constant(np.array([0, 2, 1, 3], dtype=np.int32))
    q, k, v = op.transpose(q, perm), op.transpose(k, perm), op.transpose(v, perm)

    # the head's own KV cache, so a draft can attend to the whole prefix
    k, v, sinks = kv_cache(k, v)

    # GQA: 4 kv heads serve 24 query heads
    rep = HEADS // KV_HEADS
    def expand(t):
        t = op.unsqueeze(t, op.constant(np.array([2], dtype=np.int32)))   # [B,KV,1,L,D]
        shape = op.concat([shape_part(t, 0, 2),
                           op.constant(np.array([rep], dtype=np.int32)),
                           shape_part(t, 3, 5)], axis=0)
        t = op.broadcast(t, shape)
        flat = op.concat([shape_part(t, 0, 1),
                          op.constant(np.array([HEADS], dtype=np.int32)),
                          shape_part(t, 3, 5)], axis=0)
        return op.reshape(t, flat, special_zero=False)

    a = op.scaled_dot_product_attention(q, expand(k), expand(v), mask, causal=False)
    a = op.transpose(a, perm)                                              # [B,S,H,D]
    a = op.reshape(a, op.constant(np.array([0, 0, HEADS * HEAD_DIM], dtype=np.int32)),
                   special_zero=True)

    # A plain sigmoid gate. The config says output_gate_type: swish, but swish
    # scores 13% against sigmoid's 66% -- the config field describes something
    # else, and the measurement decides.
    a = op.multiply(a, op.sigmoid(gate))
    x = op.add(x, linear(a, w[p + "self_attn.o_proj.weight"], "o_proj"))

    # --- SwiGLU ---------------------------------------------------------------
    y = rms_norm(x, w[p + "post_attention_layernorm.weight"], eps, "post_ln")
    g = linear(y, w[p + "mlp.gate_proj.weight"], "gate_proj")
    u = linear(y, w[p + "mlp.up_proj.weight"], "up_proj")
    x = op.add(x, linear(op.multiply(op.multiply(g, op.sigmoid(g)), u),
                         w[p + "mlp.down_proj.weight"], "down_proj"))

    out = rms_norm(x, w["mtp.norm.weight"], eps, "final_norm")
    res = op.result(out)
    res.get_output_tensor(0).set_names({"mtp_hidden"})
    return ov.Model([res], sinks, [hidden, embeds, pos, mask, beam], "qwen3_5_mtp")


def kv_cache(k, v):
    """Stateful K/V for the head's single attention layer."""
    outs, sinks = [], []
    for name, t in (("mtp.key", k), ("mtp.value", v)):
        shape = ov.PartialShape([-1, KV_HEADS, -1, HEAD_DIM])
        init = op.broadcast(op.constant(np.array([0.0], dtype=np.float32)),
                            op.concat([shape_part(t, 0, 1),
                                       op.constant(np.array([KV_HEADS, 0, HEAD_DIM],
                                                            dtype=np.int32))], axis=0))
        info = ov.op.util.VariableInfo()
        info.data_shape = shape
        info.data_type = ov.Type.f32
        info.variable_id = name
        var = ov.op.util.Variable(info)
        past = op.read_value(init, var)
        cat = op.concat([past, t], axis=2)
        sinks.append(op.assign(cat, var))
        outs.append(cat)
    return outs[0], outs[1], sinks


def extract_lm_head(base_xml):
    """The base model's LM head as a standalone IR, weights and all.

    Cloning the branch keeps the head int4 with its scales; re-reading it from
    the checkpoint would materialise 248320x5120 at full precision instead.
    """
    m = ov.Core().read_model(base_xml)
    node = m.get_results()[0].input_value(0).get_node()
    for _ in range(8):
        if node.get_type_name() == "MatMul":
            break
        node = node.input_value(0).get_node()
    if node.get_type_name() != "MatMul":
        raise SystemExit("could not find the LM head MatMul in the base IR")

    act = node.input_value(0)
    p = op.parameter(act.get_partial_shape(), act.get_element_type(), name="mtp_hidden")
    node.input(0).replace_source_output(p.output(0))
    res = op.result(node.output(0))
    res.get_output_tensor(0).set_names({"logits"})
    out = ov.Model([res], [p], "qwen3_5_mtp_lm_head")
    out.validate_nodes_and_infer_types()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="checkpoint holding the mtp.* tensors")
    ap.add_argument("--base", required=True, help="the base openvino_language_model.xml")
    ap.add_argument("--out", required=True, help="directory to write the MTP IRs into")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.weights, "config.json")))
    t = cfg.get("text_config", cfg)
    eps = float(t.get("rms_norm_eps", 1e-6))
    theta = float(t.get("rope_parameters", {}).get("rope_theta", t.get("rope_theta", 1e7)))
    rotary = int(HEAD_DIM * float(t.get("partial_rotary_factor", 0.25)))
    if t.get("mtp_num_hidden_layers", 1) != 1:
        raise SystemExit("this exporter builds a single-layer MTP head")

    print(f"eps {eps}  theta {theta:g}  rotary_dim {rotary}  heads {HEADS}/{KV_HEADS}x{HEAD_DIM}")
    w = load_mtp_tensors(a.weights)
    print(f"loaded {len(w)} mtp tensors")

    os.makedirs(a.out, exist_ok=True)
    layer = build_mtp_layer(w, eps, theta, rotary)
    layer.validate_nodes_and_infer_types()
    ov.save_model(layer, os.path.join(a.out, "openvino_mtp_layer.xml"), compress_to_fp16=True)
    print("wrote openvino_mtp_layer.xml")

    head = extract_lm_head(a.base)
    ov.save_model(head, os.path.join(a.out, "openvino_mtp_lm_head.xml"), compress_to_fp16=False)
    print("wrote openvino_mtp_lm_head.xml")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# How the reconstruction was settled, 2026-08-28, against qwen38-b7c1-ov:
#
# Every choice below was measured, not assumed. One base forward supplies h_t
# and the true continuation; the head's job is to predict x_{t+2} from
# (h_t, emb(x_{t+1})); the score is how often it does. The ablations show each
# element is load-bearing rather than fitted:
#
#   zero-centred (1 + w) norms, embedding-first concat,
#   per-head interleaved q/gate, sigmoid gate, q/k-norm before rope,
#   interleaved rotary pairs, post-final-norm hidden      66.0%
#     ... with a swish gate instead of sigmoid              13.2%
#     ... with q and gate as two contiguous halves          13.2%
#     ... with no gate at all                               15.1%
#     ... from the pre-final-norm hidden state              49.1%
#     ... with plain RMSNorm instead of (1 + w)              0.0%
#
# 66% sits inside the 45-75% band the vLLM campaigns report for this head.
