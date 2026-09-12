"""OV opset-13 emission of the qwen4_exp DENSE-CAUSAL full-attention block,
static T, no cache.

SCOPE -- CORRECTION, 2026-09-12 (frontier ruling; supersedes any reading of the
causal-scope note that excluded the attention layers). The checkpoint is 48
layers = 36 GDN + 12 full-attention layers (`qwen4exp.attention.compress_ratios`
in the shipped GGUF is 4 at exactly blk 3,7,...,47 and 0 on the other 36). The
frontier rule excused the QSA INDEXER, not the layers: dense causal attention IS
the semantics the indexer approximates, so each full-attention layer is emitted
DENSE CAUSAL from the pin's own `Qwen4ExpTextAttention` (modeling_qwen4_exp.py
819-901, sha256 `ca9f00bbd73cfcfbad7ba6073b5ecc23ca27bdace58b746fe6ead6ab175efc0c`,
re-hashed by the test before every numeric table):

  * KEPT, line for line -- the attention proper: the fused q_proj and its
    per-head [query | gate] chunk (pin 867-870), q_norm / k_norm RMSNorm over
    head_dim (pin 872-873), v_proj (pin 874), rope (pin 876-877), then the
    eager attention interface `eager_attention_forward` (pin 794-816):
    repeat_kv (pin 804-805, itself pin 782-791), scaled q@k^T (pin 807), the
    additive mask (pin 809), f32 softmax (pin 811), @V (pin 813) and the
    transpose (pin 814); finally the reshape (pin 897), the sigmoid gate
    (pin 898) and o_proj (pin 900).
  * DROPPED, the ruled-out part -- the QSA selection branch: the indexer call
    (pin 855) and its mask overlay (pin 857-860). The indexer's selection is a
    per-batch/per-query Python loop around `torch.nonzero` (pin 729-731), not
    statically opset-13-emittable (E1.5 finding 7), so the piece carries NO
    selected-token mask; the causal mask is baked as a constant.
  * RoPE -- the pin's `apply_rotary_pos_emb` (pin 653-668) fed by the pin's OWN
    rotary embedding (`Qwen4ExpTextRotaryEmbedding`, pin 79-149) in its
    degenerate-for-text form. The model expands a 1-row position vector to FOUR
    identical rows (pin 1433-1434, or pin 1436 for a supplied 2-D one), hands
    row 0 to the QSA mask and rows 1-3 to the rotary module (pin 1438-1440), so
    for text the 3-row recomposition (pin 144-149) reads three identical rows
    and is numerically the identity. What remains depends only on the position
    value, so it is emitted as a GATHER-AND-APPLY over constant cos/sin tables
    [T, rotary] (baked in the pin's f32 order) indexed by a declared
    `position_ids` input [1, T] i64; the text-only serving path feeds arange(T).
    A hard non-emittable rope sub-op would be a STOP-and-report finding, never a
    licence to drop the layer -- none has been found.

TWO DEFECTS FIXED HERE, both found by salvage triage of an orphan branch and
both invisible to an emitter-vs-reference leg because the reference shared them
(2026-09-12, see the commit message for the red tables):

  1. The [query | gate] split is PER HEAD, not a leading slice. The pin views
     the fused projection to [B, T, heads, 2*head_dim] and only THEN chunks the
     LAST axis (pin 867-869), so head h's 512 outputs are [256 query | 256
     gate]. Taking the leading heads*head_dim of the flat 12288 axis instead
     hands heads 0..11's FULL blocks to the query and heads 12..23's to the
     gate. This is E2 FIX D's defect class ("narrow by TRUE HALVES, not a
     leading slice") with the halves on the other axis.
  2. RMSNorm divides by head_dim. The pin is
     `x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)` (pin 164) -- a MEAN.
     A `reduce_sum` with no division is sqrt(head_dim) = 16x off at
     head_dim 256.

ACCEPTANCE IS TWO-LEGGED (both are needed; neither substitutes for the other):
  * MATH, equality-shaped: emitted vs the pin's own `Qwen4ExpTextAttention` with
    its indexer NEUTRALISED (the indexer stubbed to return an all-zero additive
    mask, which makes the pin's own code path exactly dense causal). Same float
    width on both sides, so the verdict is the f32 floor. This is the leg that
    can see defects 1 and 2; a reference written from the same misreading cannot.
  * PRICE (frontier ruling, CORRECTED 2026-09-12): emitted vs the pin WITH its
    real QSA indexer, on identical fed real tensors. The superseded text here
    read "PRICE, KLD-shaped ... Non-zero BY DESIGN -- QSA prunes keys the dense
    block keeps"; it is recorded rather than edited away because the premise,
    not the wording, is what was wrong. MEASURED, the indexer does not prune
    below its budget: query i keeps topk(min(block_topk, num_complete_blocks))
    (pin 757) with block_topk = budget // compress_ratio = 2048 // 4 = 512
    (pin 684), so while (i+1)//4 <= 512 the min IS num_complete_blocks, every
    complete block is selected, the incomplete tail is added unconditionally
    (pin 762-763), and the overlaid mask EQUALS the causal mask.

        T=  64   |dense - QSA| max-abs 0.000000e+00   rows differing    0/64
        T=  96   |dense - QSA| max-abs 0.000000e+00   rows differing    0/96
        T=2080   |dense - QSA| max-abs 2.385560e-02   rows differing   29/2080

    So this leg is EQUALITY-shaped at serving prefill lengths -- a stronger gate
    than a KLD-shaped one, not a weaker one -- and the price there is a measured
    0.0 rather than a tolerated residue. The price is non-zero only ABOVE the
    budget (pruning needs (i+1)//4 > 512, i.e. from row i = 2051), where it is
    PASTED and never bounded, the row count being the structural quantity and
    the magnitude a sample. The end-to-end KLD gate decides small-enough for
    T > 2048, later.

`attention_scaling` (pin 133-134) multiplies cos/sin; for rope_type "default"
the pin's own `compute_default_rope_parameters` returns 1.0 (pin 117), so the
baked tables omit it. `_freqs_tables` asserts the rope_type it was handed.

Per-position-ness: every op before the attention core is row-local; the core
mix is causal by the additive mask, whose strictly-upper triangle is
finfo(f32).min (the pin's eager-path mask value) so a future key contributes
exp(-inf) = 0. Row 0 attends to itself only, never to an all-masked row, so no
row softmaxes over an all-min vector. Static batch 1.

State keys (module-relative, the pin `Qwen4ExpTextAttention` state_dict):
q_proj.weight [heads*2*d, H], k_proj.weight [kv*d, H], v_proj.weight [kv*d, H],
o_proj.weight [H, heads*d], q_norm.weight [d], k_norm.weight [d]. These are the
GGUF `attn_q / attn_k / attn_v / attn_output / attn_q_norm / attn_k_norm` (see
gguf_feed; `attn_q` is the FUSED [query|gate] tensor, width heads*2*d). The
indexer's own weights (GGUF `indexer.q_proj / k_proj / q_norm / k_norm`, which
the checkpoint DOES ship) are NEVER fed here -- the selection branch is not
emitted. The test feeds them to the pin-side reference only, to price the
ruling.

Entry points:
  build_dense_attention_model(config, state, seq_len) -> ov.Model, inputs
    `hidden_states` [1, T, H] f32 and `position_ids` [1, T] i64, result
    `output` [1, T, H] f32.
  emit_dense_attention(hidden, position_ids, config, state, seq_len) -> node
    -- the same subgraph for the assembled backbone.
"""
import numpy as np
from openvino import Model, Type
from openvino import opset13 as op

from .gdn import (_add, _c, _i, _mm, _mul, _reshape, _rmean, _rsqrt_eps,  # noqa: F401
                  _slice, _transpose)
from .hc import _add64  # the f64 (1+w) roundtrip lowering


# --- rotary tables (pin 79-149, degenerate-for-text form) -------------------
def _freqs_tables(config, T):
    """The f32 cos/sin tables the pin's rotary embedding produces for the
    text-degenerate positions 0..T-1. Mirrors Qwen4ExpTextRotaryEmbedding
    (pin 79-149) in float32 order:
      * dim = head_dim * partial_rotary_factor                     (pin 115)
      * inv_freq = 1 / (base ** (arange(0, dim, 2) / dim))         (pin 119)
      * inv_freq_expanded [3, T, freq_len, 1] @ positions          (pin 127-128)
      * freqs = (...).transpose(2, 3)                              (pin 132)
      * cos = freqs.cos() * attention_scaling (1.0 here)           (pin 133)
      * sin = freqs.sin() * attention_scaling                      (pin 134)
      * recomposition over the 3 rows, then cat(freqs, freqs)      (pin 144-149)
    Returns (cosT [T, rotary], sinT [T, rotary]) f32 numpy."""
    rope_type = config.rope_parameters.get("rope_type", "default")
    assert rope_type == "default", (
        f"baked rope tables assume the pin's compute_default_rope_parameters "
        f"(attention_scaling 1.0, pin 117); config says rope_type={rope_type!r}")
    theta = float(config.rope_parameters["rope_theta"])
    partial = float(config.rope_parameters.get("partial_rotary_factor", 1.0))
    head_dim = getattr(config, "head_dim", None) or \
        config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial)                  # pin 115
    freq_len = dim // 2
    section = list(config.rope_parameters.get("mrope_section", [11, 11, 10]))

    ar = np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)        # pin 119
    inv = (np.float32(1.0) / (np.float32(theta) ** ar)).astype(np.float32)
    pos = np.arange(T, dtype=np.float32)           # the text rows, all equal

    # pin 127-128: inv_freq [3, T, freq_len, 1] @ positions [3, 1, 1, T].
    freqs = np.matmul(
        np.broadcast_to(inv.reshape(1, 1, freq_len, 1), (3, 1, freq_len, 1)),
        np.broadcast_to(pos.reshape(1, 1, 1, T), (3, 1, 1, T)),
    ).astype(np.float32)                           # [3, 1, freq_len, T]
    freqs = np.transpose(freqs, (0, 1, 3, 2))      # pin 132: .transpose(2, 3)
    freqs = freqs[:, 0, :, :]                      # [3, T, freq_len]

    # pin 144-148: recomposition_frequencies over the (identical) 3 rows. The
    # pin writes freq[0] in place; reading rows 1 and 2 only, so a copy here is
    # equivalent. For text all three rows are equal and this is the identity --
    # it is transcribed anyway, so the degeneracy stays a measured property of
    # the input rather than an assumption baked into the emitter.
    freqs_thw = freqs[0].copy()
    for dim_idx, offset in ((1, 1), (2, 2)):       # pin 145: enumerate((1,2), start=1)
        length = section[dim_idx] * 3              # pin 146
        idx = slice(offset, length, 3)             # pin 147
        freqs_thw[..., idx] = freqs[dim_idx, ..., idx]      # pin 148
    freqs_thw = np.concatenate([freqs_thw, freqs_thw], axis=-1)  # pin 149

    cosT = np.cos(freqs_thw).astype(np.float32)    # pin 133 (scaling 1.0)
    sinT = np.sin(freqs_thw).astype(np.float32)    # pin 134
    return cosT, sinT


def _rmsnorm_hd(x, weight, eps, d):
    """Qwen4ExpTextRMSNorm over the LAST axis, no group: pin 152-172. x is
    [1, T, heads, d]; the norm is a MEAN of squares (pin 164), not a sum. The
    [d] weight (zero-init, pin 156) is applied as the pin's (1 + w) (pin 171)
    through the f64-roundtrip lowering hc.py established -- bit-identical to the
    pin's fp32 add for |w| >= 2^-30, and checkpoint norm weights are zero-init
    plus drift, far above that."""
    var = _rmean(_mul(x, x), 3)                    # pin 164: x.pow(2).mean(-1)
    xn = _mul(x, _rsqrt_eps(var, eps))             # pin 164: x * rsqrt(var+eps)
    w64 = op.convert(
        _c(np.ascontiguousarray(weight, np.float32).reshape([1, 1, 1, d])),
        Type.f64)
    ones64 = op.constant(np.ones((1, 1, 1, d), np.float64))
    wn = op.convert(_add64(ones64, w64), Type.f32)  # pin 171
    return _mul(wn, xn)


def _rotate_half(x):
    """pin 628-632: rotate_half -- x1 = x[..., :d/2], x2 = x[..., d/2:],
    cat((-x2, x1), dim=-1)."""
    half = x.shape[-1] // 2
    x1 = _slice(x, 0, half, 1, -1)                 # pin 630
    x2 = _slice(x, half, half * 2, 1, -1)          # pin 631
    return op.concat([op.negative(x2), x1], axis=-1)   # pin 632


def _apply_rope(q, k, cosT, sinT, pid_node, rotary, T):
    """pin 653-668 (apply_rotary_pos_emb) with cos/sin gathered from the baked
    [T, rotary] tables by `position_ids` -- the degenerate-for-text fixed
    gather-and-apply. q/k: [1, heads, T, d]; returns rotated q, k."""
    cos = op.gather(cosT, pid_node, op.constant(np.int64(0)))   # [1,T,rotary]
    sin = op.gather(sinT, pid_node, op.constant(np.int64(0)))
    cos = _reshape(cos, [1, 1, T, rotary])         # pin 653: unsqueeze(1)
    sin = _reshape(sin, [1, 1, T, rotary])         # pin 654
    # pin 658 / 665: keep the non-rotary tail, rotate the leading rotary band.
    qr, qn = _slice(q, 0, rotary, 1, -1), _slice(q, rotary, q.shape[-1], 1, -1)
    kr, kn = _slice(k, 0, rotary, 1, -1), _slice(k, rotary, k.shape[-1], 1, -1)
    qr = _add(_mul(qr, cos), _mul(_rotate_half(qr), sin))       # pin 660
    kr = _add(_mul(kr, cos), _mul(_rotate_half(kr), sin))       # pin 666
    return (op.concat([qr, qn], axis=-1),                       # pin 662
            op.concat([kr, kn], axis=-1))                       # pin 667


def _repeat_heads_h(x, kv_heads, heads, r, T):
    """pin 782-791 (repeat_kv): [1,kv,T,d] -> [1,heads,T,d] by expand over a
    new axis 2 then reshape, so each kv head's r copies are ADJACENT. Concat of
    r copies along that axis is the same tensor as the pin's expand."""
    if r == 1:                                     # pin 788-789
        return x
    x5 = _reshape(x, [1, kv_heads, 1, T, x.shape[3]])
    return _reshape(op.concat([x5] * r, axis=2),   # pin 790
                    [1, heads, T, x.shape[3]])     # pin 791


def _attention_subgraph(hidden, pid_node, config, state, T):
    """hidden [1,T,H] f32 + position_ids [1,T] i64 -> attn_out [1,T,H]. The
    dense-causal block: pin 864-900 minus the indexer (pin 855-860)."""
    H = config.hidden_size
    heads = config.num_attention_heads
    kv = config.num_key_value_heads
    d = getattr(config, "head_dim", None) or H // heads
    eps = config.rms_norm_eps
    cosT, sinT = _freqs_tables(config, T)
    rotary = cosT.shape[-1]

    # pin 867-869: q_proj -> view [1, T, heads, 2*d] -> chunk(2, dim=-1). The
    # split is PER HEAD on the LAST axis: head h's 2*d outputs are
    # [d query | d gate]. A leading slice of the flat heads*2*d axis is a
    # different tensor (see the header's defect 1).
    qg = _mm(hidden, _c(state["q_proj.weight"]), tb=True)   # [1,T,heads*2d]
    qg = _reshape(qg, [1, T, heads, 2 * d])                 # pin 868: .view
    q = _slice(qg, 0, d, 1, 3)                              # pin 867: chunk[0]
    gate = _slice(qg, d, 2 * d, 1, 3)                       # pin 867: chunk[1]
    gate = _reshape(gate, [1, T, heads * d])                # pin 870: .reshape

    # pin 872: q_norm(query_states.view(hidden_shape)).transpose(1, 2). The
    # chunk already yields [1,T,heads,d], so the pin's .view is a no-op here.
    q = _transpose(_rmsnorm_hd(q, state["q_norm.weight"], eps, d), [0, 2, 1, 3])
    # pin 873: k_norm(k_proj(hidden).view(hidden_shape)).transpose(1, 2)
    k = _mm(hidden, _c(state["k_proj.weight"]), tb=True)    # [1,T,kv*d]
    k = _reshape(k, [1, T, kv, d])
    k = _transpose(_rmsnorm_hd(k, state["k_norm.weight"], eps, d), [0, 2, 1, 3])
    # pin 874: v_proj(hidden).view(hidden_shape).transpose(1, 2) -- no norm
    v = _transpose(_reshape(_mm(hidden, _c(state["v_proj.weight"]), tb=True),
                            [1, T, kv, d]), [0, 2, 1, 3])

    # pin 876-877: apply_rotary_pos_emb on the transposed q/k
    q, k = _apply_rope(q, k, _c(cosT), _c(sinT), pid_node, rotary, T)

    # eager_attention_forward (pin 794-816)
    r = heads // kv
    k = _repeat_heads_h(k, kv, heads, r, T)                 # pin 804
    v = _repeat_heads_h(v, kv, heads, r, T)                 # pin 805
    scaling = np.float32(d ** -0.5)                         # pin 828
    scores = _mm(q, k, tb=True)                             # [1,heads,T,T]
    scores = _mul(scores, _c(scaling))                      # pin 807
    # pin 809: attn_weights = attn_weights + attention_mask. The causal
    # additive mask is baked: 0.0 allowed, finfo(f32).min strictly above the
    # diagonal -- the value the pin's eager path carries (it converts the bool
    # mask with torch.finfo(dtype).min).
    causal = np.triu(np.full((T, T), np.float32(np.finfo(np.float32).min),
                             np.float32), 1)
    scores = _add(scores, _c(causal.reshape(1, 1, T, T)))
    att = op.softmax(scores, -1)                            # pin 811 (f32)
    out = _mm(att, v)                                       # pin 813
    out = _transpose(out, [0, 2, 1, 3])                     # pin 814
    out = _reshape(out, [1, T, heads * d])                  # pin 897

    # pin 898: attn_output = attn_output * torch.sigmoid(gate), both
    # [1, T, heads*d] with the same head-major layout.
    out = _mul(out, op.sigmoid(gate))
    return _mm(out, _c(state["o_proj.weight"]), tb=True)     # pin 900: o_proj


def emit_dense_attention(hidden, pid_node, config, state, seq_len):
    """The dense-causal subgraph for the assembled backbone."""
    return _attention_subgraph(hidden, pid_node, config, state, int(seq_len))


def build_dense_attention_model(config, state, seq_len):
    T = int(seq_len)
    H = config.hidden_size
    hidden = op.parameter([1, T, H], Type.f32)
    hidden.set_friendly_name("hidden_states")
    pid = op.parameter([1, T], Type.i64)
    pid.set_friendly_name("position_ids")
    out = _attention_subgraph(hidden, pid, config, state, T)
    res = op.result(out)
    res.set_friendly_name("output")
    return Model([res], [hidden, pid], "qwen4_exp_dense_attention")


__all__ = ["build_dense_attention_model", "emit_dense_attention", "_freqs_tables"]
