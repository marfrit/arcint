"""OV opset-13 emission of the qwen4_exp GatedResidual (hyper-connection
mixer), in its `use_combine=False` form -- the one the text model builds as
its final `hyper_connection_mixer` (pin line 1393).

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
                  applied as (1 + w) (171); _norm (163-164) reshapes the
                  last dim into groups of `group_size`, mean-square over the
                  group, rsqrt(mean + eps), flatten -- for group_size =
                  hidden that is one group per (hc stream, row)
  pin 1022        silu( down(x_norm) / hc_count )
  pin 1023        sigmoid( up(.) )
  pin 1024        unflatten(-1) into (hc_count, hidden)
  pin 1025        (w * x_norm.unflatten(-1, (hc, H))) elementwise
  pin 1026        mean over the hc streams (dim=-2; the axis is DROPPED --
                  OV reduce_mean with keep_dims=False, because this is the
                  block's result and torch .mean has no keepdim)
  pin 1027-1028   block_inject is None in this form -> return mixed_input
                  ONLY (no 2*sigmoid injection stream)

Op choices where opset-13 differs from torch (and what the first runs cost):
  * the pin's float64 weight-cast (pin 171, "(x * w).to(float16)") ->
    convert(f32 weight, f64) + 1.0 (f64), convert(f32): the 1 + w held in
    64 bits, back to f32 before the multiply (gdn.py keeps the same 64-bit
    divide in _rsqrt_eps). Cost: f64 convert/add/convert nodes.
  * torch .mean(dim=-2) drops its axis -> reduce_mean(keep_dims=False);
    the keep-dims mean (gdn.py's _rmean) yields a 4-D [1,T,1,H] result --
    the first dev-host run returned that shape and the test failed on it
    (see the red in the commit message).

Entry point: build_hc_model(config, state, seq_len) -> ov.Model with input
`hyper_input` [1, T, 4H] f32 and result `output` [1, T, H] f32 (the mixed
stream). State keys are the GatedResidual's own state_dict keys (hc_norm,
input_mix_weight_down/up); in the full checkpoint they are the
`hyper_connection_mixer.*` tensors fetched by E3.
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
    """hyper_input: [1, T, hc*H] -> mixed [1, T, H]."""

    # pin 1021 -> Qwen4ExpTextRMSNorm, group_size = hidden (pin line 1010).
    # _norm (pin 161-165): reshape the last dim into (-1, group), then
    # x * rsqrt(mean(x^2) over the group + eps) (pin 164), flatten back
    # (pin 165). For group_size = H the groups are exactly the hc streams:
    # [1, T, hc*H] -> [1, T, hc, H] -> normalize over the last axis ->
    # [1, T, hc, H].
    x = _reshape(hyper_input, [1, T, hc, H])
    var = _rmean(_mul(x, x), 3)  # pin 164: x.pow(2).mean(-1, keepdim=True)
    xn = _mul(x, _rsqrt_eps(var, eps))  # pin 164: x * rsqrt(var + eps)
    # pin 171: output * (1 + weight) with the zero-initialized hc_norm weight
    # (pin 156); the "1 +" is part of the pin, not a convenience. In fp32
    # 1 + w is emitted as convert(w, f64) + 1.0, convert(f32) -- the pin's
    # float64 weight-cast (pin 171) is the numerically safe way to hold
    # 1 + w, and gdn.py's 1/x form keeps the same 64-bit divide; the
    # weights are O(1) checkpoint values, not 1e-8-scale ones, so the
    # rounding cost is a few ulps of the gate weight, within the 1e-5 gate.
    w64 = op.convert(_c(np.ascontiguousarray(weight_vec, dtype=np.float32).reshape(1, 1, hc, H)), Type.f64)
    ones64 = op.constant(np.ones((1, 1, hc, H), np.float64))
    wn = op.convert(_add64(ones64, w64), Type.f32)
    xn = _mul(wn, xn)
    xg = _reshape(xn, [1, T, hc * H])  # pin 165: flatten(-2) -> [1, T, hc*H]

    # pin 1022: silu(down(x_norm) / hc_count). The down Linear (pin 1011) is
    # [hc_lowrank, hc*H], bias=False: y = x @ W^T.
    down = _mm(xg, _c(state["input_mix_weight_down.weight"]), tb=True)  # [1,T,lowrank]
    down = _mul(down, _c(np.float32(1.0 / hc)))  # pin 1022: "/ self.hc_count"
    silu_d = _silu(down)  # pin 1022: F.silu

    # pin 1023: sigmoid(up(silu_d)). The up Linear (pin 1012) is
    # [hc*H, hc_lowrank], bias=False.
    up = _mm(silu_d, _c(state["input_mix_weight_up.weight"]), tb=True)  # [1,T,hc*H]
    wgt = _sigmoid(up)  # pin 1023: torch.sigmoid

    # pin 1024: unflatten the last dim into (hc, H)
    w5 = _reshape(wgt, [1, T, hc, H])

    # pin 1025: (w * x_norm.unflatten(-1, (hc, H))) elementwise over [1,T,hc,H]
    prod = _mul(w5, xn)

    # pin 1026: .mean(dim=-2) over the hc streams -> [1, T, H]. torch's
    # .mean(dim=-2) DROPS the axis (no keepdim), so the OV reduce_mean uses
    # keep_dims=False (the gdn.py _rmean wrapper keeps dims -- that block
    # feeds the reduced axis straight into a matmul; here it is the result
    # and must be 3-D, not 4-D).
    mixed = _mean_axis_drop(prod, 2)  # [1, T, H]
    return mixed


# --- top-level emitter -----------------------------------------------------
def build_hc_model(config, state, seq_len):
    H = config.hidden_size
    hc = config.hc_count
    lowrank = config.hc_lowrank
    eps = config.rms_norm_eps
    T = int(seq_len)

    hyper_input = op.parameter([1, T, hc * H], Type.f32)
    hyper_input.set_friendly_name("hyper_input")

    # use_combine=False: no block_inject_weight (pin 1013, 1027-1028), so the
    # forward is exactly the norm -> low-rank gate -> weighted mean above.
    mixed = _gated_residual(
        hyper_input, T, H, hc, lowrank, state["hc_norm.weight"], eps, state
    )

    result = op.result(mixed)
    result.set_friendly_name("output")
    model = Model([result], [hyper_input], "qwen4_exp_hc_mixer")
    return model


__all__ = ["build_hc_model"]
