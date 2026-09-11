"""OV opset-13 emission of the qwen4_exp GatedResidual (hyper-connection
mixer), in BOTH of its forms:
  * `use_combine=False` (build_hc_model) -- the text model's final
    `hyper_connection_mixer` (pin line 1393); returns only the mixed stream.
  * `use_combine=True`  (build_combine_model) -- the per-layer attn/mlp
    mixers (pin lines 1270-1271; class default, pin 1004); returns the pin's
    3-tuple (mixed, hyper_input passthrough, injection) per pin 1030-1031.

Mirrors `tools/q4e/ref_hc.Qwen4ExpTextGatedResidual.forward` (itself the
pinned transformers reference, modeling_qwen4_exp.py 1003-1031) as a *static*
graph: batch fixed to 1 and sequence length fixed at build time. Every op is
row-local (group RMSNorm with group_size = hidden, row-wise low-rank
projections, per-row mean over the hc streams), so no chunking, scanning or
padding machinery is needed -- a position never reads or is read by any
other position (the test's garbage probe proves the block is exactly inert
on masked rows; see tests/python/test_hc_block.py).

LINE-LINE DISCIPLINE (mandatory for this block; the math is small and easy
to paraphrase wrong, so every emitted op cites the pin line(s) it mirrors).
Pin = modeling_qwen4_exp.py, sha256
`ca9f00bbd73cfcfbad7ba6073b5ecc23ca27bdace58b746fe6ead6ab175efc0c`
(== the installed venv file, re-hashed by the test before every table).

  pin 1021        hc_norm(x), group RMSNorm with group_size = hidden
  pin 152-172     Qwen4ExpTextRMSNorm: weight is ZERO-INITIALIZED (156) and
                  applied as (1 + w) (171); _norm (161-165) reshapes the
                  last dim into groups of `group_size` (163), mean-square
                  over the group + rsqrt(mean + eps) (164), flatten (165) --
                  for group_size = hidden that is one group per (hc stream, row)
  pin 1022        silu( down(x_norm) / hc_count )
  pin 1023        sigmoid( up(.) )
  pin 1024        unflatten(-1) into (hc_count, hidden)
  pin 1025        (w * x_norm.unflatten(-1, (hc, H))) elementwise
  pin 1026        mean over the hc streams (dim=-2; the axis is DROPPED --
                  OV reduce_mean with keep_dims=False, because this is the
                  block's result and torch .mean has no keepdim)
  pin 1027-1029   pin 1027 is the closing paren of the mean call; the
                  None-return branch is 1028-1029: block_inject is None in
                  this form -> return mixed_input ONLY (no 2*sigmoid
                  injection stream)

Op choices where opset-13 differs from torch (and what the first runs cost):
  * the (1 + w) scale of the zero-initialized hc_norm weight (pin 171) is
    held in 64 bits: convert(w, f64) + 1.0 (f64), convert(f32). Why f64:
    the pin's line 171 IS a plain fp32 add (1.0 + w) and torch rounds it
    half-to-even; for f32 addends 1.0 + w with |w| >= 2^-30 the f64 sum is
    EXACT, so convert(f64) + 1.0 -> convert(f32) equals round_f32(1 + w)
    -- bit-identical to the pin's fp32 add (shared round-half-even). There
    is no float64 cast in the pin; the f64 roundtrip is the safe lowering
    of that fp32 add, not a transcription of one (the 160a64a review,
    section 4, measured 0 mismatches over 15,000,000 f32 weights incl. the
    exact-midpoint tie cases on the f32 grid). Cost: f64 convert/add/
    convert nodes.
  * gdn.py's _rsqrt_eps (1/sqrt(x+eps)) is pure fp32 -- nothing to match
    here; the f64 nodes are this block's own (the 160a64a review, section
    4, retracted an earlier claim that gdn.py kept a 64-bit divide).
  * torch .mean(dim=-2) drops its axis -> reduce_mean(keep_dims=False);
    the keep-dims mean (gdn.py's _rmean) yields a 4-D [1,T,1,H] result --
    the first dev-host run returned that shape and the test failed on it
    (see the red in the commit message).

Entry points:
  build_hc_model(config, state, seq_len) -> ov.Model, input `hyper_input`
    [1, T, 4H] f32, single result `output` [1, T, H] f32 (mixed). State keys:
    hc_norm, input_mix_weight_down/up; in the full checkpoint the
    `hyper_connection_mixer.*` tensors (E3).
  build_combine_model(config, state, seq_len) -> ov.Model, same input, THREE
    results in pin order (pin 1031): `mixed` [1,T,H], `hyper_passthrough`
    [1,T,4H] (the raw input unchanged), `injection` [1,T,hc_count]. Adds the
    fourth state key block_inject_weight (pin 1012); in the full checkpoint
    the per-layer `*_hyper_connection.*` tensors (E3).
"""
import numpy as np
from openvino import Model, Type
from openvino import opset13 as op

# Reused verbatim from tools/q4e/gdn.py: the same OV opset-13 thin wrappers
# (no op is re-invented here).
from .gdn import _c, _i, _mm, _mul, _rmean, _reshape, _rsqrt_eps  # noqa: F401


def _silu(x):
    """F.silu (pin 1022) as x * sigmoid(x) -- the same decomposition gdn.py
    uses (opset-13 has no silu node; sigmoid is on the verified-green op
    list, E1.5 env probe)."""
    return _mul(x, op.sigmoid(x))


def _sigmoid(x):
    return op.sigmoid(x)


def _add64(a, b):
    return op.add(a, b)


def _mean_axis_drop(x, axis):
    """reduce_mean over one axis with keep_dims=False (torch .mean(dim) --
    the axis is dropped). gdn.py's _rmean keeps dims (keepdim=True) because
    its reduced axis feeds a broadcast; this block's result must be 3-D."""
    return op.reduce_mean(x, _i([axis]), False)


# --- the block (every op cites its pin line) ------------------------------
def _gated_residual(hyper_input, T, H, hc, lowrank, weight_vec, eps, state):
    """hyper_input: [1, T, hc*H] -> (mixed [1, T, H], xg [1, T, hc*H]).

    xg is the normed+flattened stream (hyper_input_normed, pin 1021 -> the
    flatten of pin 165); it is returned alongside mixed so the use_combine
    emitter can feed it to the block_inject projection (pin 1030) without
    restating the norm/gate body. build_hc_model (use_combine=False) ignores
    it, so its emitted graph is byte-identical to before."""

    # pin 1021 -> Qwen4ExpTextRMSNorm, group_size = hidden (pin line 1009).
    # _norm (pin 161-165): reshape the last dim into (-1, group), then
    # x * rsqrt(mean(x^2) over the group + eps) (pin 164), flatten back
    # (pin 165). For group_size = H the groups are exactly the hc streams:
    # [1, T, hc*H] -> [1, T, hc, H] -> normalize over the last axis ->
    # [1, T, hc, H].
    x = _reshape(hyper_input, [1, T, hc, H])
    var = _rmean(_mul(x, x), 3)  # pin 164: x.pow(2).mean(-1, keepdim=True)
    xn = _mul(x, _rsqrt_eps(var, eps))  # pin 164: x * rsqrt(var + eps)
    # pin 171: output * (1 + w) with the zero-initialized hc_norm weight
    # (pin 156); the "1 +" is part of the pin, not a convenience. Pin lines
    # 169-170 are a Llama-comment about .to(float16) ORDERING, not a 64-bit
    # cast: line 171 is a plain fp32 add. fp32 1 + w is therefore emitted
    # as convert(w, f64) + 1.0 (f64), convert(f32) -- for |w| >= 2^-30 the
    # f64 sum is exact, so this roundtrip equals the pin's fp32 add
    # bit-for-bit (shared round-half-even); the 160a64a review (section 4)
    # measured 0 mismatches over 15,000,000 f32 weights (checkpoint
    # weights are zero-init + drift, far above 2^-30). Any residual
    # double-rounding corner needs |w| < 2^-30 and costs 1 ulp of the
    # gate weight, ~7 orders of magnitude under the 1e-5 gate.
    w64 = op.convert(_c(np.ascontiguousarray(weight_vec, dtype=np.float32).reshape(1, 1, hc, H)), Type.f64)
    ones64 = op.constant(np.ones((1, 1, hc, H), np.float64))
    wn = op.convert(_add64(ones64, w64), Type.f32)
    xn = _mul(wn, xn)
    xg = _reshape(xn, [1, T, hc * H])  # pin 165: flatten(-2) -> [1, T, hc*H]

    # pin 1022: silu(down(x_norm) / hc_count). The down Linear (pin 1010) is
    # [hc_lowrank, hc*H], bias=False: y = x @ W^T.
    down = _mm(xg, _c(state["input_mix_weight_down.weight"]), tb=True)  # [1,T,lowrank]
    down = _mul(down, _c(np.float32(1.0 / hc)))  # pin 1022: "/ self.hc_count"
    silu_d = _silu(down)  # pin 1022: F.silu

    # pin 1023: sigmoid(up(silu_d)). The up Linear (pin 1011) is
    # [hc*H, hc_lowrank], bias=False.
    up = _mm(silu_d, _c(state["input_mix_weight_up.weight"]), tb=True)  # [1,T,hc*H]
    wgt = _sigmoid(up)  # pin 1023: torch.sigmoid

    # pin 1024: unflatten the last dim into (hc, H)
    w5 = _reshape(wgt, [1, T, hc, H])

    # pin 1025: (w * x_norm.unflatten(-1, (hc, H))) elementwise over [1,T,hc,H]
    prod = _mul(w5, xn)

    # pin 1025-1027: .mean(dim=-2) over the hc streams -> [1, T, H]. torch's
    # .mean(dim=-2) DROPS the axis (no keepdim), so the OV reduce_mean uses
    # keep_dims=False (the gdn.py _rmean wrapper keeps dims -- that block
    # feeds the reduced axis straight into a matmul; here it is the result
    # and must be 3-D, not 4-D).
    mixed = _mean_axis_drop(prod, 2)  # [1, T, H]
    return mixed, xg


# --- top-level emitter -----------------------------------------------------
def build_hc_model(config, state, seq_len):
    H = config.hidden_size
    hc = config.hc_count
    lowrank = config.hc_lowrank
    eps = config.rms_norm_eps
    T = int(seq_len)

    hyper_input = op.parameter([1, T, hc * H], Type.f32)
    hyper_input.set_friendly_name("hyper_input")

    # use_combine=False: no block_inject_weight (pin 1012; the None-return
    # branch is pin 1028-1029), so the forward is exactly the norm ->
    # low-rank gate -> weighted mean above.
    mixed, _ = _gated_residual(
        hyper_input, T, H, hc, lowrank, state["hc_norm.weight"], eps, state
    )

    result = op.result(mixed)
    result.set_friendly_name("output")
    model = Model([result], [hyper_input], "qwen4_exp_hc_mixer")
    return model


# --- top-level emitter (use_combine=True, the per-layer mixer) -------------
def build_combine_model(config, state, seq_len):
    """The use_combine=True GatedResidual (pin 1004: the class default is
    True; the per-layer attn/mlp mixers, pin 1270-1271) as a static opset-13
    graph. It adds the block_inject projection (pin 1012) to the
    use_combine=False body and returns the pin's 3-tuple (pin 1030-1031),
    in the pin's return order:

      result 0  mixed        [1, T, H]         -- identical node to build_hc_model
      result 1  hyper_input  [1, T, hc*H]      -- the RAW input, passed through
                                                   unchanged (pin 1031 returns
                                                   `hyper_input` as-is; its
                                                   [1,T,hc*H] width is the model
                                                   entry's repeat, pin 1480)
      result 2  injection    [1, T, hc_count]  -- 2*sigmoid(inject(x_norm)/hc),
                                                   pin 1030

    The injection stream is CONSUMED downstream in the decoder layer (pin
    1302-1303 after attn, 1308-1309 after mlp): `hidden.unsqueeze(-2) *
    inj.unsqueeze(-1)` then `hyper_input + injection.flatten(-2)`. This block
    only PRODUCES the 3-tuple; the combine sites belong to the assembled
    layer and are not emitted here. No pad-row machinery is needed -- every
    op is row-local exactly as in the use_combine=False body (the mixed and
    hyper-passthrough row-locality is already the settled contract of
    test_hc_block); the masked-row injection is exactly 2*sigmoid(0)=1 (a
    zeroed normed row -> inject matmul of zeros -> sigmoid(0)=0.5 -> *2),
    which the combine test asserts like-for-like against the pin.

    State keys: the GatedResidual's own four (hc_norm, input_mix_weight_down/
    up, block_inject_weight); in the full checkpoint these are the per-layer
    `*_hyper_connection.*` tensors.
    """
    H = config.hidden_size
    hc = config.hc_count
    lowrank = config.hc_lowrank
    eps = config.rms_norm_eps
    T = int(seq_len)

    hyper_input = op.parameter([1, T, hc * H], Type.f32)
    hyper_input.set_friendly_name("hyper_input")

    # mixed stream + the normed+flattened stream xg (hyper_input_normed, pin
    # 1021 -> 165), shared op-for-op with the use_combine=False body.
    mixed, xg = _gated_residual(
        hyper_input, T, H, hc, lowrank, state["hc_norm.weight"], eps, state
    )

    # pin 1030: injection_weights = 2 * sigmoid( block_inject(x_norm) / hc ).
    # block_inject (pin 1012) is Linear(hc*H -> hc_count, bias=False): weight
    # is [hc_count, hc*H], so y = xg @ W^T -> [1, T, hc_count].
    inj = _mm(xg, _c(state["block_inject_weight.weight"]), tb=True)  # [1,T,hc]
    inj = _mul(inj, _c(np.float32(1.0 / hc)))        # pin 1030: "/ self.hc_count"
    inj = _mul(_c(np.float32(2.0)), _sigmoid(inj))   # pin 1030: 2 * sigmoid(.)

    # pin 1031: return mixed_input, hyper_input (raw passthrough), injection.
    res_mixed = op.result(mixed)
    res_mixed.set_friendly_name("mixed")
    res_hyper = op.result(hyper_input)
    res_hyper.set_friendly_name("hyper_passthrough")
    res_inj = op.result(inj)
    res_inj.set_friendly_name("injection")
    model = Model(
        [res_mixed, res_hyper, res_inj], [hyper_input], "qwen4_exp_hc_combine"
    )
    return model


def emit_hc(hyper_input, config, state, seq_len):
    """use_combine=False mixer NODE for the assembled backbone (E2 inc5b):
    hyper_input [1,T,hc*H] -> mixed [1,T,H]. Same body as build_hc_model."""
    T = int(seq_len)
    mixed, _ = _gated_residual(
        hyper_input, T, config.hidden_size, config.hc_count, config.hc_lowrank,
        state["hc_norm.weight"], config.rms_norm_eps, state,
    )
    return mixed


def emit_combine(hyper_input, config, state, seq_len):
    """use_combine=True mixer NODES for the assembled backbone: hyper_input
    [1,T,hc*H] -> (mixed [1,T,H], hyper_input passthrough, injection
    [1,T,hc_count]). Same body as build_combine_model (pin 1030-1031)."""
    T = int(seq_len)
    H, hc, lowrank, eps = config.hidden_size, config.hc_count, config.hc_lowrank, config.rms_norm_eps
    mixed, xg = _gated_residual(
        hyper_input, T, H, hc, lowrank, state["hc_norm.weight"], eps, state
    )
    inj = _mm(xg, _c(state["block_inject_weight.weight"]), tb=True)  # pin 1030
    inj = _mul(inj, _c(np.float32(1.0 / hc)))
    inj = _mul(_c(np.float32(2.0)), _sigmoid(inj))
    return mixed, hyper_input, inj


__all__ = ["build_hc_model", "build_combine_model", "emit_hc", "emit_combine"]
