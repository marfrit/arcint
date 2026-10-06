"""--dense-u8 (tools/q4e/dense_u8.py): the dense projections in the plugin's u8
group-16 form, recovered exactly from Q6_K-valued weights.

Q6_K values are (d * sc) * q, d an f16, sc an int8, q in [-32, 31], computed in
f32 as gguf-py does. The cells build such tensors from random fields (every
group different: the blind-fill lesson), and check that the pass recovers
every group, decodes within the f16 scale's rounding, refuses values that are
not of that form, and rewrites only the projections it can carry exactly.
"""
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import pytest
from openvino import Type, opset13 as op

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from q4e import dense_u8 as du  # noqa: E402


def _q6k_like(rng, n, k):
    """f32 [n, k]: per 16-group scale d*sc (f16 d, int8 sc), q in [-32, 31]."""
    g = k // 16
    d = rng.uniform(1e-4, 5e-3, size=(n, g // 16 + 1)).astype(np.float16).astype(np.float32)
    d = np.repeat(d, 16, axis=1)[:, :g]                         # one d per 256 values
    sc = rng.integers(-128, 128, size=(n, g)).astype(np.float32)
    sc[sc == 0] = 1
    q = rng.integers(-32, 32, size=(n, g, 16)).astype(np.float32)
    s = (d * sc).astype(np.float32)                            # exact in f32
    w = (s[..., None] * q).astype(np.float32)
    return w.reshape(n, k), s, q


def test_every_q6k_group_recovers_and_decodes_within_the_f16_scale():
    rng = np.random.default_rng(5)
    w, s, q = _q6k_like(rng, 64, 512)
    # the range edges and a zero group, in known places
    w3 = w.reshape(64, 32, 16)
    w3[0, 0] = s[0, 0] * np.arange(-32, -16)                    # q = -32 present
    w3[0, 1] = s[0, 1] * np.concatenate([np.arange(16, 31), [31]])
    w3[1, 0] = 0.0
    w = w3.reshape(64, 512)
    qu8, sc, ok = du.recover_groups(w)
    assert ok.all()
    assert qu8.min() >= 0 and qu8.max() <= 63
    # exact in f64 under the recovered f32 scale
    exact = (qu8.astype(np.float64) - du.ZP) * sc.astype(np.float64)[..., None]
    assert np.array_equal(exact.reshape(64, 512).astype(np.float32), w)
    dec = du.decode(qu8, sc.astype(np.float16))
    rel = np.abs(dec - w) / np.maximum(np.abs(w), 1e-30)
    assert rel.max() <= 2.0 ** -10, rel.max()


def test_values_not_of_the_q6k_form_are_refused():
    rng = np.random.default_rng(6)
    w = rng.standard_normal((32, 256)).astype(np.float32)
    _, _, ok = du.recover_groups(w)
    assert not ok.any()


def test_the_pass_rewrites_exactly_the_carriable_projections():
    rng = np.random.default_rng(7)
    n, k, T = 1024, 1024, 5
    wq, _, _ = _q6k_like(rng, n, k)
    wr = (rng.standard_normal((n, k)) * 0.02).astype(np.float32)
    x = op.parameter([1, -1, k], Type.f32, name="x")
    a = op.matmul(x, op.constant(wq), False, True)
    a.input_value(1).get_node().set_friendly_name("proj_q6k")
    b = op.matmul(x, op.constant(wr), False, True)
    b.input_value(1).get_node().set_friendly_name("proj_f32")
    model = ov.Model([op.result(a), op.result(b)], [x], "two_projections")
    xin = rng.standard_normal((1, T, k)).astype(np.float32)
    core = ov.Core()
    before = core.compile_model(model, "CPU")({"x": xin})
    rep = du.apply(model, min_elems=1, log=lambda *_: None)
    assert [c[0] for c in rep["converted"]] == ["proj_q6k"]
    assert [kk[0] for kk in rep["kept"]] == ["proj_f32"]
    types = {nd.get_type_name() for nd in model.get_ordered_ops()}
    assert {"Subtract", "Convert"} <= types
    u8 = [nd for nd in model.get_ordered_ops()
          if nd.get_type_name() == "Constant" and nd.get_output_element_type(0) == Type.u8
          and nd.get_friendly_name() == "proj_q6k/dense_u8"]
    assert len(u8) == 1 and list(u8[0].get_output_shape(0)) == [n, k // 16, 16]
    after = core.compile_model(model, "CPU")({"x": xin})
    ya, yb = before[0], after[0]
    bound = np.einsum("tk,nk->tn", np.abs(xin[0]), np.abs(wq)) * 2.0 ** -10 + 1e-6
    assert (np.abs(ya[0] - yb[0]) <= bound).all(), np.abs(ya[0] - yb[0]).max()
    assert np.array_equal(before[1], after[1])                  # the kept projection is untouched


def test_the_shared_expert_is_kept_plain():
    """The plugin fuses the shared-expert MLP into the MoE op and its kernel
    reads plain weights; a u8 chain there served garbage (depth-4 logits A/B,
    2026-09-26: argmax 7/1000). Built through the emitter's own
    emit_shared_expert, with Q6_K-exact weights that WOULD convert."""
    import types
    from q4e import moe
    rng = np.random.default_rng(8)
    H, I = 1024, 1024
    cfg = types.SimpleNamespace(hidden_size=H, shared_expert_intermediate_size=I)
    st = {"shared_expert.gate_proj.weight": _q6k_like(rng, I, H)[0],
          "shared_expert.up_proj.weight": _q6k_like(rng, I, H)[0],
          "shared_expert.down_proj.weight": _q6k_like(rng, H, I)[0],
          "shared_expert_gate.weight": _q6k_like(rng, 16, H)[0][:1]}
    x = op.parameter([1, -1, H], Type.f32, name="x")
    m = ov.Model([op.result(moe.emit_shared_expert(x, cfg, st, None))], [x], "shared")
    rep = du.apply(m, min_elems=1, log=lambda *_: None)
    assert rep["converted"] == []
    assert sorted(k[0] for k in rep["kept"]) == ["shared_expert/down_proj", "shared_expert/gate_proj",
                                                 "shared_expert/up_proj", "shared_expert_gate"]


def test_the_recovered_scale_is_the_coarsest():
    """Any (k, sign) that fits is exact, but a finer scale rounds worse in f16;
    the recovered q of every non-zero group has gcd 1."""
    rng = np.random.default_rng(9)
    w, _, _ = _q6k_like(rng, 16, 256)
    w = (w.reshape(16, 16, 16) * 1.0).reshape(16, 256)
    w[:, :16] = w[:, :16] * 0 + np.float32(0.004) * np.array([2, 4, -6, 8] * 4, np.float32)  # q shares a factor 2
    qu8, sc, ok = du.recover_groups(w)
    assert ok.all()
    q = qu8.astype(np.int64) - du.ZP
    g = np.gcd.reduce(np.abs(q), axis=-1)
    assert (g[np.abs(q).max(axis=-1) > 0] == 1).all()


def test_attention_k_and_v_are_kept_plain():
    """q, k and v all compressed are fused horizontally by the GPU plugin and
    served wrong in the paged graph (depth-4 logits A/B, 2026-09-26); the pass
    keeps the emitter-named k/v projections and still converts q."""
    rng = np.random.default_rng(10)
    x = op.parameter([1, -1, 1024], Type.f32, name="x")
    outs = []
    for nm, n in (("attn3/q_proj", 2048), ("attn3/k_proj", 1024), ("attn3/v_proj", 1024)):
        c = op.constant(_q6k_like(rng, n, 1024)[0])
        c.set_friendly_name(nm)
        outs.append(op.result(op.matmul(x, c, False, True)))
    m = ov.Model(outs, [x], "qkv")
    rep = du.apply(m, min_elems=1, log=lambda *_: None)
    assert [c[0] for c in rep["converted"]] == ["attn3/q_proj"]
    assert sorted(k[0] for k in rep["kept"]) == ["attn3/k_proj", "attn3/v_proj"]


def test_a_scale_whose_f16_rounding_is_too_coarse_is_kept():
    """A group of tiny weights has a scale in f16's subnormal range, where the
    f16 rounding is far above 2^-11: the projection is kept, not converted."""
    rng = np.random.default_rng(11)
    w, _, _ = _q6k_like(rng, 64, 256)
    w[0, :16] = np.float32(3e-8) * np.arange(-8, 8, dtype=np.float32)   # exact s*q, s subnormal in f16
    x = op.parameter([1, -1, 256], Type.f32, name="x")
    c = op.constant(w)
    c.set_friendly_name("tiny")
    m = ov.Model([op.result(op.matmul(x, c, False, True))], [x], "tiny")
    rep = du.apply(m, min_elems=1, log=lambda *_: None)
    assert rep["converted"] == [] and rep["kept"][0][0] == "tiny" and "rounding" in rep["kept"][0][1]


def test_plan_then_compress_then_commit_leaves_no_f32_constant():
    """The exporter's order: plan from the exact f32, compress the rest to f16,
    then splice. compress_model_to_f16 skips a model that already carries a
    compressed-weight chain, so committing first left every other constant f32
    (0.772 GiB in the first full-depth u8 artifact)."""
    from openvino._offline_transformations import compress_model_transformation
    rng = np.random.default_rng(12)
    x = op.parameter([1, -1, 1024], Type.f32, name="x")
    wq, _, _ = _q6k_like(rng, 1024, 1024)
    cq = op.constant(wq)
    cq.set_friendly_name("proj")
    kept = op.constant((rng.standard_normal((1024, 1024)) * 0.02).astype(np.float32))
    kept.set_friendly_name("attn0/k_proj")
    m = ov.Model([op.result(op.matmul(x, cq, False, True)), op.result(op.matmul(x, kept, False, True))], [x], "m")
    plans, rep = du.plan(m, min_elems=1)
    compress_model_transformation(m)
    du.commit(plans)
    f32 = [nd.get_friendly_name() for nd in m.get_ordered_ops()
           if nd.get_type_name() == "Constant" and nd.get_output_element_type(0) == Type.f32
           and np.prod(nd.get_output_shape(0)) > 64]
    assert f32 == [], f32
    assert [c[0] for c in rep["converted"]] == ["proj"]
    assert any(nd.get_friendly_name() == "proj/dense_u8" for nd in m.get_ordered_ops())
