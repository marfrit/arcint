"""serving_shape._native_expert: the checkpoint's expert blocks decoded in
standard ops, exact on the CPU plugin (design-routing-aware-expert-execution
2.3b). Device-free cells fill the per-role tensors with RANDOM bytes -- a
fill that is NOT alike across rows or blocks (the blind-fill lesson of
2026-09-08: a fill that makes every page alike certifies nothing) -- and
compare a MatMul over the chain with numpy's exact decode; the shard-gated
cell does the same with two real experts through NativeExpertFiller.
"""
import os
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import pytest
from openvino import Type, opset13 as op

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from q4e import native_blocks as nb  # noqa: E402
from q4e import serving_shape as ss  # noqa: E402

_SHARDS = os.environ.get("Q4E_GGUF_SHARDS", "")
_skip = pytest.mark.skipif(not _SHARDS, reason="Q4E_GGUF_SHARDS unset (real GGUF shards absent)")


def _random_raw(rng, rows, inn, fmt):
    """Valid random blocks: every byte random, the f16 scale finite and non-zero."""
    block, nbytes = nb.BLOCK_BYTES[fmt]
    nblk = inn // block
    raw = rng.integers(0, 256, size=(rows, nblk, nbytes), dtype=np.uint8)
    # IQ4_XS multiplies d by a 6-bit sub-block scale of up to 31: a smaller d
    # keeps the random weights at the other formats' magnitude (the exact
    # cells compare at 1e-5, which a 1e4 accumulation would not meet in f32)
    d = rng.uniform(0.01, 0.2, size=(rows, nblk)).astype(np.float32) / (31.0 if fmt == "IQ4_XS" else 1.0)
    d = d.astype("<f2")
    raw[:, :, 0:2] = d.view(np.uint8).reshape(rows, nblk, 2)
    return raw.reshape(rows, nblk * nbytes)


def _matmul_model(arena, e, out, inn, fmt, parts, T):
    x = op.parameter([1, T, inn], Type.f32, name="x")
    w = ss._native_expert(arena, e, out, inn, f"t/experts_{fmt}", fmt, parts)
    y = op.matmul(x, w, transpose_a=False, transpose_b=True)           # [E, T, out]
    res = op.result(y)
    return ov.Model([res], [x], "native_expert_chain")


@pytest.mark.parametrize("fmt,inn", [("IQ4_NL", 64), ("IQ3_XXS", 512), ("IQ4_XS", 512), ("Q8_0", 64)])
def test_the_chain_decodes_random_blocks_exactly(fmt, inn):
    rng = np.random.default_rng({"IQ4_NL": 11, "IQ3_XXS": 13, "IQ4_XS": 17, "Q8_0": 19}[fmt])
    e, out, T = 3, 5, 7
    raw = _random_raw(rng, e * out, inn, fmt)
    parts = nb.SPLIT[fmt](raw)
    # the chain carries the block scale as f16 (serving_shape._native_expert):
    # the reference decodes with the same rounding, and the rounding itself
    # is bounded below
    *codes, scales = parts
    scales16 = scales.astype(np.float16).astype(np.float32)
    w_ref = nb.DECODE[fmt](*codes, scales16).reshape(e, out, inn)
    w_exact = nb.DECODE[fmt](*codes, scales).reshape(e, out, inn)
    rel = np.abs(w_ref - w_exact) / np.maximum(np.abs(w_exact), 1e-30)
    if fmt in ("IQ4_NL", "Q8_0"):
        assert np.array_equal(w_ref, w_exact), f"{fmt}: d is an f16, the f16 scale must be exact"
    else:
        assert rel.max() <= 2.0 ** -11, f"{fmt}: f16 scale rounding {rel.max():.3e} above 2^-11"
    # not alike: every row differs from every other, and no row is constant
    assert len({r.tobytes() for r in w_ref.reshape(e * out, inn)}) == e * out
    assert (w_ref.reshape(e * out, inn).std(axis=1) > 0).all()
    arena = ss.SparseArena()
    try:
        model = _matmul_model(arena, e, out, inn, fmt, parts, T)
        x = rng.standard_normal((1, T, inn)).astype(np.float32)
        compiled = ov.Core().compile_model(model, "CPU")
        y = compiled({"x": x})[compiled.output(0)]
    finally:
        arena.close()
    want = np.einsum("tk,eok->eto", x[0], w_ref)
    d = np.abs(y - want)
    # exact decode, inexact summation: the two f32 dots (the plugin's and
    # numpy's) round in different orders, so the bound is on the dot's own
    # magnitude sum(|x||w|), not on the (possibly cancelled) output
    bound = np.einsum("tk,eok->eto", np.abs(x[0]), np.abs(w_ref))
    print(f"\n[native-chain] {fmt} inn={inn}: max|diff| {d.max():.3e} vs max|want| {np.abs(want).max():.3e}; "
          f"max diff/bound {(d / bound).max():.2e}")
    assert y.shape == (e, T, out)
    assert (d <= 1e-5 * bound + 1e-6).all(), f"{fmt}: max|diff| {d.max():.3e}, max diff/bound {(d / bound).max():.2e}"
    # the chain is standard ops only: nothing the plugin does not know
    types = {n.get_type_name() for n in model.get_ordered_ops()}
    assert types <= {"Parameter", "Constant", "Convert", "Gather", "Reshape", "Multiply", "Unsqueeze",
                     "BitwiseAnd", "Greater", "Select", "MatMul", "Result"}, types


def test_the_packed_chain_decodes_random_blocks_exactly():
    """patch 0052: IQ2_S carried VERBATIM (82 B/256) and decoded in-graph,
    bit-exact against native_blocks.iq2_s_decode."""
    inn = 512
    e, out, T = 3, 5, 7
    rng = np.random.default_rng(23)
    raw = _random_raw(rng, e * out, inn, "IQ2_S")
    w80, d = nb.PACKED["IQ2_S"](raw)
    # the chain carries d as f16 and evaluates d*(0.5+nib)*0.25 in f32:
    # the oracle is iq2_s_packed_decode, the same f32 arithmetic
    w_ref = nb.iq2_s_packed_decode(w80, d).reshape(e, out, inn)
    # not alike: every row differs from every other, and no row is constant
    assert len({r.tobytes() for r in w_ref.reshape(e * out, inn)}) == e * out
    assert (w_ref.reshape(e * out, inn).std(axis=1) > 0).all()
    # 82 B/256 -- the GGUF's own size, against the re-laid IQ2_S's 128 B/256
    assert w80.nbytes / (e * out * inn) == 80.0 / 256.0
    arena = ss.SparseArena()
    try:
        model = _matmul_model(arena, e, out, inn, "IQ2_S_PACKED", (w80, d), T)
        x = rng.standard_normal((1, T, inn)).astype(np.float32)
        compiled = ov.Core().compile_model(model, "CPU")
        y = compiled({"x": x})[compiled.output(0)]
    finally:
        arena.close()
    want = np.einsum("tk,eok->eto", x[0], w_ref)
    bound = np.einsum("tk,eok->eto", np.abs(x[0]), np.abs(w_ref))
    dmax = np.abs(y - want)
    print(f"\n[native-chain packed] IQ2_S_PACKED inn={inn}: max|diff| {dmax.max():.3e} vs "
          f"max|want| {np.abs(want).max():.3e}; max diff/bound {(dmax / bound).max():.2e}")
    assert y.shape == (e, T, out)
    assert (dmax <= 1e-5 * bound + 1e-6).all(), (
        f"IQ2_S_PACKED: max|diff| {dmax.max():.3e}, max diff/bound {(dmax / bound).max():.2e}")
    types = {n.get_type_name() for n in model.get_ordered_ops()}
    assert types <= {"Parameter", "Constant", "Convert", "Gather", "Reshape", "Multiply", "Unsqueeze",
                     "BitwiseAnd", "Greater", "Select", "MatMul", "Result", "Slice", "Floor",
                     "Add", "Subtract", "Broadcast", "Divide"}, types


def test_the_packed_chain_leaves_no_rank6_select():
    """The GPU plugin's layout optimizer has no layout for a rank-6 Select
    (measured 2026-09-26: add_required_reorders.cpp:342 refused
    [256,512,8,8,4,8]). The IQ2_S chain's signs Select is rank 5; the packed
    chain must not add an axis. This cell fails if the extra axis returns."""
    inn = 512
    e, out = 3, 5
    rng = np.random.default_rng(29)
    raw = _random_raw(rng, e * out, inn, "IQ2_S")
    w80, d = nb.PACKED["IQ2_S"](raw)
    arena = ss.SparseArena()
    try:
        model = _matmul_model(arena, e, out, inn, "IQ2_S_PACKED", (w80, d), 2)
        sels = [n for n in model.get_ordered_ops() if n.get_type_name() == "Select"]
        assert sels, "no Select in the packed chain"
        ranks = [len(n.get_output_shape(0)) for n in sels]
        shapes = [list(n.get_output_shape(0)) for n in sels]
        assert max(ranks) == 5, (
            f"packed chain Select is rank {max(ranks)} (want 5, the IQ2_S "
            f"chain's and the only rank the GPU layout optimizer lays out): {shapes}")
    finally:
        arena.close()


def test_an_unknown_format_is_refused():
    arena = ss.SparseArena()
    try:
        with pytest.raises(ValueError):
            ss._native_expert(arena, 1, 1, 32, "t/x", "Q3_K", (None,))
    finally:
        arena.close()


@_skip
@pytest.mark.parametrize("layer,kind,fmt", [(0, "down", "IQ4_NL"), (0, "gate", "IQ3_XXS"),
                                            (2, "gate", "IQ4_XS"), (2, "down", "Q8_0")])
def test_two_real_experts_through_the_native_filler_match_gguf_py(layer, kind, fmt):
    """Layer 2 is the checkpoint's odd one: IQ4_XS gate/up over a Q8_0 down
    (four more layers carry a Q8_0 down under IQ3_XXS; the census of all 48
    is in docs/design-routing-aware-expert-execution.md 2.3d)."""
    from q4e import expert_fill as ef, gguf_feed as gf
    feed = gf.GgufFeed(_SHARDS)
    filler = ef.NativeExpertFiller(feed)
    e, T = 2, 3
    out, inn = (2560, 640) if kind == "down" else (640, 2560)
    got_fmt, parts = filler.native(layer, kind, e, out, inn)
    assert got_fmt == fmt
    ref = np.asarray(feed.dequant(f"blk.{layer}.ffn_{kind}_exps.weight", rows=e), np.float32)   # gguf-py, [e,out,inn]
    rng = np.random.default_rng(0)
    arena = ss.SparseArena()
    try:
        model = _matmul_model(arena, e, out, inn, fmt, parts, T)
        x = rng.standard_normal((1, T, inn)).astype(np.float32)
        compiled = ov.Core().compile_model(model, "CPU")
        y = compiled({"x": x})[compiled.output(0)]
    finally:
        arena.close()
    want = np.einsum("tk,eok->eto", x[0], ref)
    bound = np.einsum("tk,eok->eto", np.abs(x[0]), np.abs(ref))
    d = np.abs(y - want)
    print(f"\n[native-chain real] blk.{layer} {kind} {fmt}: max|diff| {d.max():.3e} vs max|want| {np.abs(want).max():.3e}; "
          f"max diff/bound {(d / bound).max():.2e}; census {filler.census()}")
    # the f16 block scale (exact for IQ4_NL / Q8_0) plus f32 summation order
    assert (d <= (2.0 ** -11 + 1e-5) * bound + 1e-6).all(), f"{kind}: max diff/bound {(d / bound).max():.2e}"
    assert filler.census()["bodies"] == 1 and filler.census()["format"] == "native"
