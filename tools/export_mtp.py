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

# Geometry comes from the checkpoint's config (set in main); these are the
# Qwen3.8-27B values the head was first reconstructed against, kept as the
# defaults so the file reads the same as before.
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


ROPE_STYLE = "interleaved"   # --rope: interleaved (even,odd pairs) | half (rotate_half)

def rope(x, cos, sin, rotary_dim, heads):
    """Rotate the first `rotary_dim` channels and pass the rest through.

    Two pairings exist in the wild for the same weights: interleaved
    (even, odd) pairs -- what this exporter measured its acceptance with --
    and HF/llama.cpp's rotate_half (first half, second half). The checkpoint
    trained with exactly one of them; --rope selects, acceptance decides
    (2026-09-01: the GGUF/llama.cpp reference uses half-split on byte-identical
    weights, so the two conventions are A/B'd rather than assumed)."""
    half = rotary_dim // 2
    i32 = lambda v: op.constant(np.array([v], dtype=np.int32))
    rot = op.slice(x, i32(0), i32(rotary_dim), i32(1), i32(-1))
    passthru = op.slice(x, i32(rotary_dim), i32(2 ** 31 - 1), i32(1), i32(-1))
    if ROPE_STYLE == "half":
        x1 = op.slice(rot, i32(0), i32(half), i32(1), i32(-1))
        x2 = op.slice(rot, i32(half), i32(2 ** 31 - 1), i32(1), i32(-1))
        lo = op.subtract(op.multiply(x1, cos), op.multiply(x2, sin))
        hi = op.add(op.multiply(x2, cos), op.multiply(x1, sin))
        return op.concat([lo, hi, passthru], axis=-1)
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



MOE_TOPK = 8
MOE_NORM_TOPK = True   # renormalise the top-k routing weights; a switch the oracle decides
MOE_LOWERING = "batched"   # --moe-lowering: batched (default, current behaviour) | unrolled

def moe_block(y, w, p, topk, norm_topk):
    """The Qwen3.6 MTP layer's MLP: a router over E experts, the top-k of them
    applied and summed, plus a sigmoid-gated shared expert.

    Every expert is computed for every token and the non-selected ones are
    weighted by exactly zero. That is the same arithmetic as a gather over the
    selected experts, without a data-dependent loop in the graph; at one token
    per step it reads the expert weights once (~0.4 GB int4, ~1 ms), and at
    prefill the caller keeps the batch small enough for the [E, M, 2I]
    intermediate. Tensor layouts as stored: gate_up_proj [E, 2I, H] with the
    gate half first, down_proj [E, H, I], gate.weight [E, H] (torch [out, in]).
    """
    gu = w[p + "mlp.experts.gate_up_proj"]          # [E, 2I, H]
    dn = w[p + "mlp.experts.down_proj"]             # [E, H, I]
    E, two_i, H = gu.shape
    I = two_i // 2

    # router: softmax over all experts, keep the top-k, optionally renormalise
    logits = linear(y, w[p + "mlp.gate.weight"], "router")            # [B,S,E]
    probs = op.softmax(logits, axis=-1)
    tk = op.topk(probs, op.constant(np.array(topk, dtype=np.int32)), axis=-1,
                 mode="max", sort="value", index_element_type="i32")
    vals, idx = tk.output(0), tk.output(1)                            # [B,S,k]
    if norm_topk:
        vals = op.divide(vals, op.reduce_sum(vals, op.constant(np.array([-1], dtype=np.int32)),
                                             keep_dims=True))
    zeros = op.multiply(probs, op.constant(np.array([0.0], dtype=np.float32)))
    weights = op.scatter_elements_update(zeros, idx, vals,
                                         op.constant(np.array(-1, dtype=np.int32)))  # [B,S,E]

    # all experts, batched over E: y as [1, M, H] against W^T as [E, H, 2I]
    m_h = op.reshape(y, op.constant(np.array([1, -1, H], dtype=np.int32)), special_zero=False)
    gu_c = op.constant(np.ascontiguousarray(gu, dtype=np.float32))
    gu_c.friendly_name = "experts_gate_up"
    h2 = op.matmul(m_h, gu_c, transpose_a=False, transpose_b=True)     # [E, M, 2I]
    split = op.split(h2, op.constant(np.array(-1, dtype=np.int32)), 2)
    g, u = split.output(0), split.output(1)                            # [E, M, I]
    act = op.multiply(op.multiply(g, op.sigmoid(g)), u)
    dn_c = op.constant(np.ascontiguousarray(dn, dtype=np.float32))
    dn_c.friendly_name = "experts_down"
    outs = op.matmul(act, dn_c, transpose_a=False, transpose_b=True)   # [E, M, H]

    # weights [B,S,E] -> [E, M, 1]
    w_m = op.reshape(weights, op.constant(np.array([-1, E], dtype=np.int32)), special_zero=False)
    w_e = op.unsqueeze(op.transpose(w_m, op.constant(np.array([1, 0], dtype=np.int32))),
                       op.constant(np.array([-1], dtype=np.int32)))
    mixed = op.reduce_sum(op.multiply(outs, w_e), op.constant(np.array([0], dtype=np.int32)),
                          keep_dims=False)                              # [M, H]
    mixed = op.reshape(mixed, op.concat([shape_part(y, 0, 2),
                                         op.constant(np.array([H], dtype=np.int32))], axis=0),
                       special_zero=False)

    # shared expert, gated by a scalar sigmoid per token
    sg = linear(y, w[p + "mlp.shared_expert.gate_proj.weight"], "shared_gate_proj")
    su = linear(y, w[p + "mlp.shared_expert.up_proj.weight"], "shared_up_proj")
    shared = linear(op.multiply(op.multiply(sg, op.sigmoid(sg)), su),
                    w[p + "mlp.shared_expert.down_proj.weight"], "shared_down_proj")
    shared = op.multiply(shared, op.sigmoid(linear(y, w[p + "mlp.shared_expert_gate.weight"],
                                                   "shared_expert_gate")))
    return op.add(mixed, shared)

def moe_block_unrolled(y, w, p, topk, norm_topk):
    """The same MoE MLP as moe_block(), lowered so OpenVINO's FuseMOE pass
    (ov::pass::FuseMOE, fuse_moe_experts.cpp) recognises it and rewrites it
    into the internal MOECompressed op the GPU plugin's moe_3gemm_swiglu
    primitives consume. moe_block() is a re-expression of the identical
    arithmetic as one batched [E,M,*] matmul pair, which is cheap to write but
    invisible to that pass -- it reads every expert for every token instead of
    gathering the selected ones (measured: 1.69 GB/forward). This form does a
    data-dependent per-expert gather instead, at the cost of an E-way unrolled
    graph (one subgraph per expert, so this is expensive to *build* for a
    256-expert checkpoint, though not to *run*).

    Router: Softmax -> TopK -> OneHot(indices, depth=E, axis=2) ->
    Transpose([2,1,0]), giving an [E,k,M] mask FuseMOE's matcher expects.
    Per expert: Gather(mask, i, axis=0) -> Squeeze -> NonZero -> Split(2) ->
    Convert -> Gather selects that expert's tokens out of the flattened
    hidden state; three plain GEMMs (mlp3: gate, up, down -- no batched-over-E
    matmul) replace the SwiGLU; ScatterElementsUpdate(..., reduction="sum")
    accumulates each expert's weighted output back into a zero tensor shaped
    like the flattened hidden state, chained expert to expert.

    gate_up_proj is stored as one fused [E, 2I, H] tensor (gate half first);
    the pass wants gate_proj and up_proj as separate rank-2 Constants, so the
    split happens here, at export time, on the numpy array -- not in the
    graph.
    """
    gu = w[p + "mlp.experts.gate_up_proj"]          # [E, 2I, H]
    dn = w[p + "mlp.experts.down_proj"]              # [E, H, I]
    E, two_i, H = gu.shape
    I = two_i // 2
    gate_np, up_np = gu[:, :I, :], gu[:, I:, :]

    i32 = lambda v: op.constant(np.array(v, dtype=np.int32))
    i32v = lambda v: op.constant(np.array([v], dtype=np.int32))

    # router: softmax -> top-k -> one-hot mask, transposed to [E, k, M]
    logits = linear(y, w[p + "mlp.gate.weight"], "router")             # [B,S,E]
    probs = op.softmax(logits, axis=-1)
    tk = op.topk(probs, i32(topk), axis=-1, mode="max", sort="value",
                index_element_type="i32")
    vals, idx = tk.output(0), tk.output(1)                              # [B,S,k]
    if norm_topk:
        vals = op.divide(vals, op.reduce_sum(vals, i32v(-1), keep_dims=True))

    idx_m = op.reshape(idx, op.constant(np.array([-1, topk], dtype=np.int32)),
                       special_zero=False)                               # [M,k]
    vals_m = op.reshape(vals, op.constant(np.array([-1, topk], dtype=np.int32)),
                        special_zero=False)                              # [M,k]
    onehot = op.one_hot(idx_m, i32(E), op.constant(1.0, dtype=np.float32),
                        op.constant(0.0, dtype=np.float32), axis=2)      # [M,k,E]
    transposed = op.transpose(onehot, op.constant(np.array([2, 1, 0], dtype=np.int32)))  # [E,k,M]
    vals_km = op.transpose(vals_m, op.constant(np.array([1, 0], dtype=np.int32)))        # [k,M]

    # the flattened hidden state every expert gathers its tokens from
    m_h3 = op.reshape(y, op.constant(np.array([1, -1, H], dtype=np.int32)),
                      special_zero=False)                                # [1,M,H]
    m_h2 = op.squeeze(m_h3, i32v(0))                                     # [M,H]
    acc = op.broadcast(op.constant(np.array(0.0, dtype=np.float32)),
                       op.shape_of(m_h2, output_type="i32"))             # [M,H]

    for e in range(E):
        sel = op.gather(transposed, i32v(e), i32(0))                     # [1,k,M]
        sel = op.squeeze(sel, i32v(0))                                   # [k,M]
        nz = op.non_zero(sel, output_type="i64")                         # [2,count]
        spl = op.split(nz, i32(0), 2)
        tok_idx = op.squeeze(op.convert(spl.output(1), "i32"), i32v(0))  # [count]

        tokens = op.gather(m_h3, tok_idx, i32(1))                        # [1,count,H]
        tokens = op.squeeze(tokens, i32v(0))                             # [count,H]

        gate_c = const(gate_np[e], f"expert{e}_gate_proj")
        up_c = const(up_np[e], f"expert{e}_up_proj")
        down_c = const(dn[e], f"expert{e}_down_proj")

        g = op.swish(op.matmul(tokens, gate_c, transpose_a=False, transpose_b=True))
        u = op.matmul(tokens, up_c, transpose_a=False, transpose_b=True)
        out = op.matmul(op.multiply(g, u), down_c, transpose_a=False, transpose_b=True)  # [count,H]

        # this expert's routing weight per selected token, gathered from the
        # top-k values via the same one-hot mask used to pick the tokens
        tok_w = op.reduce_sum(op.multiply(sel, vals_km), i32v(0), keep_dims=False)  # [M]
        tok_w = op.unsqueeze(op.gather(tok_w, tok_idx, i32(0)), i32v(-1))            # [count,1]
        weighted = op.multiply(out, tok_w)                                          # [count,H]

        idx_bcast = op.broadcast(op.unsqueeze(tok_idx, i32v(-1)),
                                 op.shape_of(weighted, output_type="i32"))
        acc = op.scatter_elements_update(acc, idx_bcast, weighted, i32(0), reduction="sum")

    mixed = op.reshape(acc, op.concat([shape_part(y, 0, 2),
                                       op.constant(np.array([H], dtype=np.int32))], axis=0),
                       special_zero=False)                               # [B,S,H]

    # shared expert, gated by a scalar sigmoid per token -- identical to moe_block()
    sg = linear(y, w[p + "mlp.shared_expert.gate_proj.weight"], "shared_gate_proj")
    su = linear(y, w[p + "mlp.shared_expert.up_proj.weight"], "shared_up_proj")
    shared = linear(op.multiply(op.multiply(sg, op.sigmoid(sg)), su),
                    w[p + "mlp.shared_expert.down_proj.weight"], "shared_down_proj")
    shared = op.multiply(shared, op.sigmoid(linear(y, w[p + "mlp.shared_expert_gate.weight"],
                                                   "shared_expert_gate")))
    return op.add(mixed, shared)


def pick_group_size(in_dim, preferred=(64, 32, 16, 8, 4, 2)):
    """The largest candidate group size that (a) is strictly smaller than
    `in_dim` -- so the grouped-quantization Constant has more than one group
    -- and (b) divides `in_dim` evenly. 64 is the group size measured in the
    one IR on record that provably fuses (both its gate/up weights, grouped
    over H=2048, and its down weight, grouped over I=512, use group_size=64
    -- a fixed hyperparameter, not derived from the tensor's own shape), so
    it is preferred; this exporter's reconstructed geometry does not always
    divide by 64 (the tiny synthetic weights in test_moe_lowering.py do
    not), hence the smaller fallbacks. Falls back to `in_dim` itself
    (groups=1, degenerate) only if nothing smaller divides evenly -- e.g.
    in_dim is prime and small -- which earlier was this exporter's only
    option and is suspected of being part of why the first rank-4 fix still
    crashed on GPU (see moe_block_tiled_flat_rank3 in test_moe_lowering.py:
    a degenerate single-group Constant may not exercise the same code path
    in ConvertTiledMoeBlockToGatherMatmuls' rewrite as a genuinely grouped
    one)."""
    for g in preferred:
        if g < in_dim and in_dim % g == 0:
            return g
    return in_dim


def quantize_u8_grouped(arr3d, group_size):
    """Symmetric u8 quantization of a rank-3 [E, out, in] weight, grouped
    over `in` into `in / group_size` groups -- one scale/zero_point per
    (E, out, group), zero_point fixed at 128 as instructed by the on-card
    measurement this mode imitates. This is the real grouping axis and
    granularity measured in the ground-truth IR (see pick_group_size):
    earlier this exporter used one scale per (E, out) row with no groups at
    all (group_size == in, degenerate). Returns (q_u8, scale_f32), both
    shaped [E, out, groups, group_size] and [E, out, groups, 1]."""
    E, out, in_ = arr3d.shape
    assert in_ % group_size == 0, f"in_dim {in_} not divisible by group_size {group_size}"
    groups = in_ // group_size
    grouped = arr3d.reshape(E, out, groups, group_size)
    amax = np.abs(grouped).max(axis=-1, keepdims=True)             # [E,out,groups,1]
    scale = np.where(amax == 0, 1.0, amax / 127.0).astype(np.float32)
    q = np.clip(np.round(grouped / scale) + 128.0, 0, 255).astype(np.uint8)
    return q, scale


def compressed_weight(arr3d, name, group_size=None):
    """A u8-quantized expert weight with the exact decompression chain and
    op count measured in the ground-truth IR (the 35B-A3B export, layer 0's
    mlp.experts.down_proj, byte offsets read directly out of its .bin):

        Constant(u4|u8, [E,out,groups,group_size]) -> Convert(f16)
        Constant(zero_point, u4|u8, [E,out,groups,1])  -> Convert(f16)
        Subtract(weight, zero_point)                                -> f16
        Constant(scale, f16, [E,out,groups,1])
        Multiply(Subtract, scale)                    ("fq_weights_1") -> f16
        Reshape(rank 4 -> rank 3, [E,out,in])                        -> f16
        Convert(f32)                                                 -> f32
        -> MatMul

    Two earlier versions of this function were tried and both compiled fine
    on CPU but crashed a real GPU compile inside
    ConvertTiledMoeBlockToGatherMatmuls' own rewrite (a MatMul batch-dim
    merge failure invisible to CPU/generic shape inference, since that pass
    never runs there): the first skipped the groups dimension and the
    Reshape entirely (flat rank 3, no Reshape at all -- see
    moe_block_tiled_flat_rank3 in test_moe_lowering.py); the second added
    the Reshape but used a degenerate single group (group_size == in,
    groups=1) and did the whole dequant chain, plus the u8->f32 upcast, in a
    single Convert with zero_point stored directly as f32 -- collapsing
    three of the ground truth's Converts into one and skipping the grouped
    quantization the fusing pass's rewrite evidently does math specific to.
    This version imitates the measured op count and grouping literally:
    pick_group_size() chooses a real (>1) number of groups whenever `in`
    allows it, and every Convert the ground-truth chain has is reproduced,
    including keeping the dequant arithmetic in f16 and upcasting to f32
    only after the reshape."""
    E, out, in_ = arr3d.shape
    if group_size is None:
        group_size = pick_group_size(in_)
    q4, scale4 = quantize_u8_grouped(arr3d, group_size)         # [E,out,groups,g] u8, [E,out,groups,1] f32
    zp4 = np.full(scale4.shape, 128, dtype=np.uint8)             # zero_point stored low-precision too, like the weight

    q_c = op.constant(np.ascontiguousarray(q4))
    q_c.friendly_name = name
    q16 = op.convert(q_c, "f16")

    zp_c = op.constant(zp4)
    zp_c.friendly_name = name + "/zero_point"
    zp16 = op.convert(zp_c, "f16")

    sub16 = op.subtract(q16, zp16)                               # f16, [E,out,groups,g]

    sc = op.constant(scale4.astype(np.float16))
    sc.friendly_name = name + "/scale"

    deq4 = op.multiply(sub16, sc)                                # f16, [E,out,groups,g] ("fq_weights_1")
    deq4.friendly_name = name + "/fq_weights_1"

    deq3_f16 = op.reshape(deq4, op.constant(np.array([E, out, in_], dtype=np.int32)),
                          special_zero=False)                    # f16, rank 4 -> rank 3
    deq3_f16.friendly_name = name + "/fq_weights_1/reshape"

    deq = op.convert(deq3_f16, "f32")                            # the final upcast, right before MatMul
    deq.friendly_name = name + "/fq_weights_1/convert"
    return deq


def moe_block_tiled(y, w, p, topk, norm_topk):
    """The same MoE MLP again, lowered to the form actually measured fusing
    on the card: OpenVINO's GPU-plugin pipeline does not go through
    ov::pass::FuseMOE's per-expert-gather rewrite (moe_block_unrolled()'s
    target) -- it fuses through ConvertTiledMoeBlockToGatherMatmuls, whose
    input pattern this function reproduces, extracted from the one IR on
    record that provably fuses (a 35B-A3B checkpoint's own export, layer 0's
    mlp.experts subgraph, walked node by node from its shared upstream
    flatten through the router's final ReduceSum):

        Reshape(hidden [B,S,H] -> [M,H])                     (shared flatten)
        -> Tile(repeats=[E,1]) -> Reshape -> [E,M,H]                  (entry)
        three per-expert-stacked GEMMs, transpose_b=true, no bias:
            gate = BMM([E,M,H], gate_w[E,I,H]^T) -> Swish
            up   = BMM([E,M,H], up_w[E,I,H]^T)
            down = BMM(gate*up, down_w[E,H,I]^T)                 -> [E,M,H]
        router, over the SAME [M,H] flatten as the entry (rank 2 the whole
        way, not rank 3 [B,S,*] -- see below):
            Softmax -> TopK -> ReduceSum/Divide (renorm) ->
            ScatterElementsUpdate(zeros[M,E], idx, vals, axis=-1) ->
            Transpose -> Reshape -> Unsqueeze -> Multiply(with expert
            outputs) -> ReduceSum

    No OneHot, no per-expert NonZero loop -- unlike moe_block_unrolled(),
    every expert is still computed for every token here (same batched-BMM
    cost as moe_block()); what changes is only the shape of the graph the
    GPU-plugin pass is looking for.

    An earlier version of this function computed the router on the
    UNFLATTENED [B,S,H]/[B,S,E] activation (moe_block()'s own convention),
    only flattening to [M,E] after the ScatterElementsUpdate. Diffing this
    function's full op sequence against the ground-truth IR's, node by node
    with every attribute and partial shape, found that ground truth's router
    runs on the exact same rank-2 [M,H] flatten the Tile entry consumes
    (both read the one shared Reshape) and stays at rank 2 all the way
    through Softmax/TopK/ScatterElementsUpdate/Transpose -- it only expands
    back to [E,B,S] after the Transpose. Whether that rank mismatch upstream
    of the expert GEMMs is what the fusing pass's matcher was actually
    tripping on (as opposed to the grouped-quantization gap fixed in
    compressed_weight()) is not confirmed without a GPU re-run; it is
    reproduced here because it is a real, measured divergence and costs
    nothing to fix.

    Two things measured but deliberately NOT changed here, noted for
    whoever GPU-verifies next: ground truth reshapes the down-projection's
    [E,M,H] output to rank-4 [E,B,S,H] (splitting M back into B and S)
    BEFORE the router-weight Multiply/ReduceSum, where this function keeps
    the mixing stage flat at [E,M,H] and reshapes back to [B,S,H] only once,
    at the very end -- mathematically equivalent (the reshape and the
    all-but-axis-0 ReduceSum commute), so left as is, but it is a shape
    difference the pass's matcher could still care about. And ground truth
    keeps the group-quantization dequant math in f16 with a final Convert to
    f32 only after the collapsing Reshape (matched literally in
    compressed_weight() now); this exporter still cannot reproduce
    production's own group_size (no quantized checkpoint to read it from),
    only the grouping *shape* the pass evidently expects.
    """
    gu = w[p + "mlp.experts.gate_up_proj"]          # [E, 2I, H]
    dn = w[p + "mlp.experts.down_proj"]              # [E, H, I]
    E, two_i, H = gu.shape
    I = two_i // 2
    gate_np, up_np = gu[:, :I, :], gu[:, I:, :]

    i32 = lambda v: op.constant(np.array(v, dtype=np.int32))
    i32v = lambda v: op.constant(np.array([v], dtype=np.int32))

    # the shared flatten -- ground truth's Reshape284, consumed by BOTH the
    # router matmul and the Tile entry below. Rank 2 [M,H] throughout.
    y_flat = op.reshape(y, op.constant(np.array([-1, H], dtype=np.int32)),
                        special_zero=False)                              # [M,H]

    # router, entirely at rank 2 [M,*] -- same arithmetic as moe_block()'s
    # (dense per-token weights via a single ScatterElementsUpdate
    # (reduction="none"), no OneHot), just on the flattened activation.
    router_w = const(w[p + "mlp.gate.weight"], "router")
    logits = op.matmul(y_flat, router_w, transpose_a=False, transpose_b=True)  # [M,E]
    probs = op.softmax(logits, axis=-1)
    tk = op.topk(probs, i32(topk), axis=-1, mode="max", sort="value",
                index_element_type="i32")
    vals, idx = tk.output(0), tk.output(1)                              # [M,k]
    if norm_topk:
        vals = op.divide(vals, op.reduce_sum(vals, i32v(-1), keep_dims=True))
    # a full-range Slice on `vals` right before the scatter -- a no-op
    # numerically (begin=(0,0), end=shape_of(vals), step=(1,1)) but ground
    # truth has it (Slice411, between its Divide and its
    # ScatterElementsUpdate) and window-D fusion checking flagged it as
    # missing here, so it is reproduced literally rather than assumed
    # irrelevant.
    vals = op.slice(vals, op.constant(np.array([0, 0], dtype=np.int32)),
                    op.shape_of(vals, output_type="i32"),
                    op.constant(np.array([1, 1], dtype=np.int32)),
                    op.constant(np.array([0, 1], dtype=np.int32)))
    zeros = op.multiply(probs, op.constant(np.array([0.0], dtype=np.float32)))
    weights = op.scatter_elements_update(zeros, idx, vals, i32(-1))    # [M,E]

    # entry: Tile -> Reshape, the shape the fusing pass matches on
    tiled = op.tile(y_flat, op.constant(np.array([E, 1], dtype=np.int32)))  # [E*M,H]
    m_h3 = op.reshape(tiled, op.constant(np.array([E, -1, H], dtype=np.int32)),
                      special_zero=False)                                # [E,M,H]

    gate_w = compressed_weight(gate_np, "experts_gate_proj")             # [E,I,H]
    up_w = compressed_weight(up_np, "experts_up_proj")                   # [E,I,H]
    down_w = compressed_weight(dn, "experts_down_proj")                  # [E,H,I]

    g = op.swish(op.matmul(m_h3, gate_w, transpose_a=False, transpose_b=True))
    u = op.matmul(m_h3, up_w, transpose_a=False, transpose_b=True)
    outs = op.matmul(op.multiply(g, u), down_w, transpose_a=False, transpose_b=True)  # [E,M,H]

    # weighted sum over experts: Transpose -> Reshape -> Unsqueeze -> Multiply -> ReduceSum.
    # Ground truth splits M back into [B,S] on BOTH operands before the
    # Multiply. The FIRST attempt at this (extracting both B and S
    # explicitly via shape_part(y,0,2), with no wildcard dim in either
    # Reshape's target) matched byte-for-byte against ground truth's node
    # types, but check_tiled_pattern.py's constraint walk against the real
    # exported head found it FAILING at E1 (end_reshape.type observed
    # MatMul, expected Reshape): OpenVINO 2026.4.0's own shape-inference/
    # simplification during validate_nodes_and_infer_types() /
    # ov.save_model() silently ELIMINATES that Reshape (and the matching
    # router-weight one), because it can prove -- with every target
    # dimension supplied as an explicit, already-known value -- that the
    # rank-4 reshape carries no information a rank-3 [E,M,H]/[E,M,1] tensor
    # didn't already have, and folds it back to rank 3 before the fusing
    # pass ever runs (reproduced locally on OpenVINO 2026.4.0 with a tiny
    # synthetic layer; 2026.3.1 here does not do this, which is why the
    # earlier local suite never caught it). Ground truth's own Reshape383
    # survives on the SAME 2026.4.0 build (confirmed: the positive-control
    # base model matches 40/40 in check_tiled_pattern.py) -- the difference
    # is that ground truth supplies only ONE dim (B, via a Gather) and
    # leaves the other (S) as a literal -1 wildcard, forcing a genuine
    # runtime division the optimizer cannot fold away. Reproduced here: B is
    # extracted explicitly, S is left as -1.
    outs4 = op.reshape(outs, op.concat([i32v(E), shape_part(y, 0, 1), i32v(-1), i32v(H)], axis=0),
                       special_zero=False)                                # [E,B,-1(S),H]
    w_t = op.transpose(weights, op.constant(np.array([1, 0], dtype=np.int32)))  # [E,M]
    w_r = op.reshape(w_t, op.concat([i32v(E), shape_part(y, 0, 1), i32v(-1)], axis=0),
                     special_zero=False)                                  # [E,B,-1(S)]
    w_u = op.unsqueeze(w_r, i32v(-1))                                     # [E,B,S,1]
    mixed = op.reduce_sum(op.multiply(outs4, w_u), i32v(0), keep_dims=False)  # [B,S,H]

    # shared expert, gated by a scalar sigmoid per token -- identical to moe_block()
    sg = linear(y, w[p + "mlp.shared_expert.gate_proj.weight"], "shared_gate_proj")
    su = linear(y, w[p + "mlp.shared_expert.up_proj.weight"], "shared_up_proj")
    shared = linear(op.multiply(op.multiply(sg, op.sigmoid(sg)), su),
                    w[p + "mlp.shared_expert.down_proj.weight"], "shared_down_proj")
    shared = op.multiply(shared, op.sigmoid(linear(y, w[p + "mlp.shared_expert_gate.weight"],
                                                   "shared_expert_gate")))
    return op.add(mixed, shared)


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

    # --- MLP: dense SwiGLU (Qwen3.8) or MoE (Qwen3.6, 256 experts, top-8) ------
    y = rms_norm(x, w[p + "post_attention_layernorm.weight"], eps, "post_ln")
    if (p + "mlp.experts.gate_up_proj") in w:
        moe_fn = {"batched": moe_block, "unrolled": moe_block_unrolled,
                 "tiled": moe_block_tiled}[MOE_LOWERING]
        x = op.add(x, moe_fn(y, w, p, MOE_TOPK, MOE_NORM_TOPK))
    else:
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
    ap.add_argument("--norm-topk", dest="norm_topk", type=lambda v: v.lower() in ("1", "true", "yes"),
                    default=None, help="MoE heads: renormalise the top-k routing weights "
                    "(default: the config's norm_topk_prob, else true)")
    ap.add_argument("--rope", choices=("interleaved", "half"), default="interleaved",
                    help="rotary pairing: interleaved (even,odd) or rotate_half (default: interleaved)")
    ap.add_argument("--moe-lowering", dest="moe_lowering",
                    choices=("batched", "unrolled", "tiled"),
                    default="batched", help="MoE emission: batched (default, current behaviour); "
                    "unrolled (per-expert gather/scatter graph OpenVINO's FuseMOE pass can match "
                    "onto its internal MOECompressed op); tiled (Reshape/Tile/Reshape entry with "
                    "rank-3 compressed-weight GEMMs, the form measured actually fusing on the GPU "
                    "plugin via ConvertTiledMoeBlockToGatherMatmuls)")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.weights, "config.json")))
    t = cfg.get("text_config", cfg)
    global HEADS, KV_HEADS, HEAD_DIM, HIDDEN, MOE_TOPK, MOE_NORM_TOPK
    HEADS    = int(t.get("num_attention_heads", HEADS))
    KV_HEADS = int(t.get("num_key_value_heads", KV_HEADS))
    HEAD_DIM = int(t.get("head_dim", HEAD_DIM))
    HIDDEN   = int(t.get("hidden_size", HIDDEN))
    MOE_TOPK = int(t.get("num_experts_per_tok", MOE_TOPK))
    if a.norm_topk is not None:
        MOE_NORM_TOPK = a.norm_topk
    elif t.get("norm_topk_prob") is not None:
        MOE_NORM_TOPK = bool(t["norm_topk_prob"])
    global ROPE_STYLE
    ROPE_STYLE = a.rope
    global MOE_LOWERING
    MOE_LOWERING = a.moe_lowering
    eps = float(t.get("rms_norm_eps", 1e-6))
    theta = float(t.get("rope_parameters", {}).get("rope_theta", t.get("rope_theta", 1e7)))
    rotary = int(HEAD_DIM * float(t.get("partial_rotary_factor", 0.25)))
    if t.get("mtp_num_hidden_layers", 1) != 1:
        raise SystemExit("this exporter builds a single-layer MTP head")

    print(f"eps {eps}  theta {theta:g}  rotary_dim {rotary}  heads {HEADS}/{KV_HEADS}x{HEAD_DIM}  "
          f"hidden {HIDDEN}  moe top-{MOE_TOPK} norm_topk={MOE_NORM_TOPK} lowering={MOE_LOWERING}")
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
