"""--dense-q8 (tools/q4e/dense_q8.py): the dense projections in their
checkpoint's Q8_0 form, i8 group-32 with an f16 scale, recovered exactly from
the f32 values the feed hands the emitter.

Q8_0 values are d * q, d an f16, q in [-127, 127] with the group's largest |q|
127 (quantize_row_q8_0), computed in f32 as gguf-py does. The cells build such
tensors from random fields (every group different), and check that the pass
recovers every group with the checkpoint's own d, decodes within the plugin's
f16 product, refuses values not of that form, rewrites only what it can carry,
keeps the shared expert and attention k/v plain, and hands the Q6_K pass only
what it did not take.
"""
import sys
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import Type, opset13 as op

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from q4e import dense_q8 as dq  # noqa: E402
from q4e import dense_u8 as du  # noqa: E402


def _q8_0_like(rng, n, k):
    """f32 [n, k]: per 32-group f16 d, q in [-127, 127] with max |q| = 127."""
    g = k // 32
    d = rng.uniform(1e-4, 5e-3, size=(n, g)).astype(np.float16).astype(np.float32)
    q = rng.integers(-127, 128, size=(n, g, 32)).astype(np.float32)
    pos = rng.integers(0, 32, size=(n, g))
    sign = np.where(rng.integers(0, 2, size=(n, g)) == 0, -1.0, 1.0)
    np.put_along_axis(q, pos[..., None], (127 * sign)[..., None], axis=-1)   # the group max, as quantize_row_q8_0 leaves it
    w = (d[..., None] * q).astype(np.float32)                                 # exact in f32
    return w.reshape(n, k), d, q


def test_every_q8_0_group_recovers_the_checkpoints_own_scale():
    rng = np.random.default_rng(21)
    w, d, q = _q8_0_like(rng, 64, 1024)
    w3 = w.reshape(64, 32, 32)
    w3[1, 0] = 0.0                                                   # an all-zero group
    w = w3.reshape(64, 1024)
    qi8, sc, ok = dq.recover_groups(w)
    assert ok.all()
    nz = np.abs(w.reshape(64, 32, 32)).max(axis=-1) > 0
    assert np.array_equal(sc[nz], d[nz])                            # the checkpoint's d, bit for bit
    assert np.array_equal(qi8.astype(np.float32)[nz], q[nz])
    exact = qi8.astype(np.float64) * sc.astype(np.float64)[..., None]
    assert np.array_equal(exact.reshape(64, 1024).astype(np.float32), w)
    dec = dq.decode(qi8, sc.astype(np.float16))
    rel = np.abs(dec - w) / np.maximum(np.abs(w), 1e-30)
    assert rel.max() <= 2.0 ** -11, rel.max()


def test_values_not_of_the_q8_0_form_are_refused():
    rng = np.random.default_rng(22)
    w = rng.standard_normal((32, 256)).astype(np.float32)
    _, _, ok = dq.recover_groups(w)
    assert not ok.any()


def test_a_q6k_tensor_is_not_mistaken_for_q8_0_across_its_groups():
    """Q6_K's 16-groups carry two scales per 32 values; a 32-group spanning two
    different scales is not d * q with |q| <= 127 unless the scales happen to
    divide -- the random fields make that vanishingly rare."""
    rng = np.random.default_rng(23)
    g = 16
    d = rng.uniform(1e-4, 5e-3, size=(32, g)).astype(np.float16).astype(np.float32)
    sc = rng.integers(17, 128, size=(32, g)).astype(np.float32)
    q = rng.integers(-32, 32, size=(32, g, 16)).astype(np.float32)
    w = ((d * sc)[..., None] * q).astype(np.float32).reshape(32, 256)
    _, _, ok = dq.recover_groups(w)
    assert ok.mean() < 0.05, ok.mean()


def test_the_pass_rewrites_exactly_the_carriable_projections():
    rng = np.random.default_rng(24)
    n, k, T = 1024, 1024, 5
    wq, _, _ = _q8_0_like(rng, n, k)
    wr = (rng.standard_normal((n, k)) * 0.02).astype(np.float32)
    x = op.parameter([1, -1, k], Type.f32, name="x")
    a = op.matmul(x, op.constant(wq), False, True)
    a.input_value(1).get_node().set_friendly_name("proj_q8")
    b = op.matmul(x, op.constant(wr), False, True)
    b.input_value(1).get_node().set_friendly_name("proj_f32")
    model = ov.Model([op.result(a), op.result(b)], [x], "two_projections")
    xin = rng.standard_normal((1, T, k)).astype(np.float32)
    core = ov.Core()
    before = core.compile_model(model, "CPU")({"x": xin})
    rep = dq.apply(model, min_elems=1, log=lambda *_: None)
    assert [c[0] for c in rep["converted"]] == ["proj_q8"]
    assert [kk[0] for kk in rep["kept"]] == ["proj_f32"]
    i8 = [nd for nd in model.get_ordered_ops()
          if nd.get_type_name() == "Constant" and nd.get_output_element_type(0) == Type.i8
          and nd.get_friendly_name() == "proj_q8/dense_q8"]
    assert len(i8) == 1 and list(i8[0].get_output_shape(0)) == [n, k // 32, 32]
    assert "Subtract" not in {nd.get_type_name() for nd in model.get_ordered_ops()}   # symmetric: no zero point
    after = core.compile_model(model, "CPU")({"x": xin})
    ya, yb = before[0], after[0]
    bound = np.einsum("tk,nk->tn", np.abs(xin[0]), np.abs(wq)) * 2.0 ** -10 + 1e-6
    assert (np.abs(ya[0] - yb[0]) <= bound).all(), np.abs(ya[0] - yb[0]).max()
    assert np.array_equal(before[1], after[1])                  # the kept projection is untouched


def test_the_shared_expert_and_attention_k_v_are_kept_plain():
    rng = np.random.default_rng(25)
    x = op.parameter([1, -1, 1024], Type.f32, name="x")
    outs = []
    for nm, n in (("attn3/q_proj", 2048), ("attn3/k_proj", 1024), ("attn3/v_proj", 1024),
                  ("shared_expert/up_proj", 1024)):
        c = op.constant(_q8_0_like(rng, n, 1024)[0])
        c.set_friendly_name(nm)
        outs.append(op.result(op.matmul(x, c, False, True)))
    m = ov.Model(outs, [x], "qkv_shared")
    rep = dq.apply(m, min_elems=1, log=lambda *_: None)
    assert [c[0] for c in rep["converted"]] == ["attn3/q_proj"]
    assert sorted(k[0] for k in rep["kept"]) == ["attn3/k_proj", "attn3/v_proj", "shared_expert/up_proj"]


def test_both_passes_split_the_projections_and_leave_no_f32_constant():
    """The exporter's order with --dense-q8 --dense-u8: plan Q8_0, plan Q6_K
    over what Q8_0 did not take, compress the rest to f16, splice both."""
    from openvino._offline_transformations import compress_model_transformation
    rng = np.random.default_rng(26)
    x = op.parameter([1, -1, 1024], Type.f32, name="x")
    w8, _, _ = _q8_0_like(rng, 1024, 1024)
    g = 64
    d = rng.uniform(1e-4, 5e-3, size=(1024, g // 16 + 1)).astype(np.float16).astype(np.float32)
    d = np.repeat(d, 16, axis=1)[:, :g]
    sc = rng.integers(-128, 128, size=(1024, g)).astype(np.float32)
    sc[sc == 0] = 1
    w6 = ((d * sc)[..., None] * rng.integers(-32, 32, size=(1024, g, 16))).astype(np.float32).reshape(1024, 1024)
    c8 = op.constant(w8)
    c8.set_friendly_name("proj_q8")
    c6 = op.constant(w6)
    c6.set_friendly_name("proj_q6k")
    m = ov.Model([op.result(op.matmul(x, c8, False, True)), op.result(op.matmul(x, c6, False, True))], [x], "m")
    p8, r8 = dq.plan(m, min_elems=1)
    p6, r6 = du.plan(m, min_elems=1, skip={c[0] for c in r8["converted"]})
    compress_model_transformation(m)
    dq.commit(p8)
    du.commit(p6)
    assert [c[0] for c in r8["converted"]] == ["proj_q8"]
    assert [c[0] for c in r6["converted"]] == ["proj_q6k"]
    assert "proj_q8" not in [k[0] for k in r6["kept"]]        # skipped, not even examined
    names = {nd.get_friendly_name() for nd in m.get_ordered_ops()}
    assert {"proj_q8/dense_q8", "proj_q6k/dense_u8"} <= names
    f32 = [nd.get_friendly_name() for nd in m.get_ordered_ops()
           if nd.get_type_name() == "Constant" and nd.get_output_element_type(0) == Type.f32
           and np.prod(nd.get_output_shape(0)) > 64]
    assert f32 == [], f32
