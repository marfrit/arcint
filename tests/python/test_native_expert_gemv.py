"""q4e.native_expert: the per-expert GEMV/GEMM with the checkpoint's own
blocks decoded in the K-loop (campaign sub4bit-vram-kernel, design note §2.3a).

Device-free: no OpenVINO, no card. The cells pin the arithmetic the OpenCL
kernel must reproduce -- block-scale application, the fused-vs-materialised
equality, partial-block refusal, unknown-format refusal, and a deliberately
wrong decode that must be caught rather than silently absorbed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from q4e import native_blocks as nb  # noqa: E402
from q4e import native_expert as ne  # noqa: E402
from q4e.expert_fill import pack_u4  # noqa: E402


def _codes(rows, K, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 16, size=(rows, K), dtype=np.uint8)


def iq4nl(rows, K, seed=0):
    codes = _codes(rows, K, seed)
    scales = np.linspace(1.0, 2.0, rows * (K // 32), dtype=np.float32).reshape(rows, K // 32)
    return pack_u4(codes).reshape(rows, K // 2), None, scales


def iq3xxs(rows, K, seed=0):
    groups = K // 32
    rng = np.random.default_rng(seed)
    gridix = rng.integers(0, 256, size=(rows, groups * 8), dtype=np.uint8)
    signix = rng.integers(0, 128, size=(rows, groups * 4), dtype=np.uint8)
    scales = np.linspace(1.0, 2.0, rows * groups, dtype=np.float32).reshape(rows, groups)
    return gridix, signix, scales


def q8_0(rows, K, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(-16, 16, size=(rows, K), dtype=np.int8)
    scales = np.linspace(1.0, 2.0, rows * (K // 32), dtype=np.float32).reshape(rows, K // 32)
    return codes, None, scales


def iq2s(rows, K, seed=0):
    """The artifact's IQ2_S per-role layout: weight u8 [rows, K/32, 8] holding
    four 10-bit grid indices as little-endian u16, signix u8 [rows, K/32, 4]
    RAW bytes, scales f32 [rows, K/32, 2] (lo for values 0..15, hi for 16..31)."""
    groups = K // 32
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 1024, size=(rows, groups, 4), dtype=np.uint16)
    weight = np.ascontiguousarray(idx.astype("<u2")).view(np.uint8).reshape(rows, groups, 8)
    signix = rng.integers(0, 256, size=(rows, groups, 4), dtype=np.uint8)
    scales = np.linspace(0.5, 2.0, rows * groups * 2, dtype=np.float32).reshape(rows, groups, 2)
    return weight, signix, scales


# --- block-scale application -------------------------------------------------

def test_iq4nl_block_scale_is_applied_in_the_gemv():
    rows, K = 2, 64
    weight, signix, scales = iq4nl(rows, K, seed=1)
    x = np.ones(K, dtype=np.float32)
    got = ne.gemv(weight, scales, 1, x, rows=rows)
    for r in range(rows):
        want = np.float32(0.0)
        for g in range(K // 32):
            vals = ne.decode_group_iq4_nl(weight[r], g)
            want += np.float32(vals.sum() * scales[r, g])
        assert got[r] == pytest.approx(float(want), abs=1e-4)
    # halving every scale halves every output: the scale is read, not ignored
    half = ne.gemv(weight, (scales * np.float32(0.5)), 1, x, rows=rows)
    assert np.allclose(half, got * np.float32(0.5), atol=1e-5)


def test_q8_0_block_scale_is_applied_in_the_gemv():
    rows, K = 2, 64
    weight, signix, scales = q8_0(rows, K, seed=2)
    x = np.ones(K, dtype=np.float32)
    got = ne.gemv(weight, scales, 3, x, rows=rows)
    for r in range(rows):
        want = np.float32(0.0)
        for g in range(K // 32):
            vals = ne.decode_group_q8_0(weight[r], g)
            want += np.float32(vals.sum() * scales[r, g])
        assert got[r] == pytest.approx(float(want), abs=1e-4)


def test_iq3xxs_block_scale_is_applied_in_the_gemv():
    rows, K = 2, 64
    weight, signix, scales = iq3xxs(rows, K, seed=3)
    x = np.ones(K, dtype=np.float32)
    got = ne.gemv(weight, scales, 2, x, signix=signix, rows=rows)
    for r in range(rows):
        want = np.float32(0.0)
        for g in range(K // 32):
            vals = ne.decode_group_iq3_xxs(weight[r], signix[r], g)
            want += np.float32(vals.sum() * scales[r, g])
        assert got[r] == pytest.approx(float(want), abs=1e-4)


# --- fused == materialised ---------------------------------------------------

@pytest.mark.parametrize("fmt,builder", [(1, iq4nl), (2, iq3xxs), (3, q8_0), (4, iq2s)])
def test_fused_gemv_matches_the_materialised_decode(fmt, builder):
    rows, K = 3, 128
    weight, signix, scales = builder(rows, K, seed=7)
    rng = np.random.default_rng(11)
    x = rng.standard_normal(K).astype(np.float32)
    got = ne.gemv(weight, scales, fmt, x, signix=signix, rows=rows)
    for r in range(rows):
        row = ne.decode_row(weight, scales, fmt, r, K, signix=signix)
        assert got[r] == pytest.approx(float(np.dot(row.astype(np.float64), x.astype(np.float64))), rel=1e-5)


# --- partial-block / unknown-format refusal ---------------------------------

@pytest.mark.parametrize("fmt,builder", [(1, iq4nl), (2, iq3xxs), (3, q8_0)])
def test_a_k_that_is_not_a_multiple_of_32_is_refused(fmt, builder):
    weight, signix, scales = builder(1, 64, seed=5)
    x = np.ones(50, dtype=np.float32)  # 50 is not a multiple of 32
    with pytest.raises(ValueError, match="not a multiple"):
        ne.gemv(weight, scales, fmt, x, signix=signix, rows=1)


def test_an_unknown_format_is_refused_rather_than_decoded():
    weight, signix, scales = iq4nl(1, 64)
    x = np.ones(64, dtype=np.float32)
    for fmt in (0, 5, 99):
        with pytest.raises(ValueError):
            ne.gemv(weight, scales, fmt, x, signix=signix, rows=1)


# --- the wrong decode must be caught ----------------------------------------

def test_the_affine_decode_of_iq4nl_bytes_is_rejected_not_absorbed():
    """The wrong reading of IQ4_NL's packed bytes (nibble as value, table
    ignored) must differ from the correct one -- a cell that silently accepted
    it would be green before the change and measure nothing."""
    rows, K = 2, 64
    weight, signix, scales = iq4nl(rows, K, seed=13)
    x = np.ones(K, dtype=np.float32)
    right = ne.gemv(weight, scales, 1, x, rows=rows)
    wrong = ne.wrong_affine_gemv_iq4_nl(weight, scales, x)
    assert not np.allclose(right, wrong), "the affine reading matched IQ4_NL -- the cell cannot fail"
    # a hand-built block makes the gap exact: all codes 0 -> right = T[0] = -127,
    # wrong = 0. Both sides are hand numbers, so neither can pass by agreement.
    zero_block = np.zeros((1, 32), np.uint8)
    unit = np.zeros(32, dtype=np.float32)
    unit[0] = 1.0
    one = np.array([[1.0]], np.float32)
    assert ne.gemv(zero_block, one, 1, unit, rows=1)[0] == -127.0
    assert ne.wrong_affine_gemv_iq4_nl(zero_block, one, unit)[0] == 0.0


def test_blocks_are_not_uniform_across_rows():
    """The blind-fill lesson: every row decodes to a different value, so a
    fill that repeated one expert's block for all rows would fail this cell."""
    rows, K = 4, 64
    weight, signix, scales = iq4nl(rows, K, seed=17)
    x = np.ones(K, dtype=np.float32)
    got = ne.gemv(weight, scales, 1, x, rows=rows)
    assert len(set(np.round(got, 6).tolist())) == rows, "rows decoded alike -- fill suspected"


# --- the full per-expert MLP -------------------------------------------------

def test_expert_moe_matches_a_materialised_chain():
    """One expert's SwiGLU MLP at flash-next's role split (gate/up IQ3_XXS,
    down IQ4_NL), fused against a materialised decode chain."""
    H, I = 64, 32
    gate = iq3xxs(I, H, seed=21)
    up = iq3xxs(I, H, seed=22)
    down = iq4nl(H, I, seed=23)
    W = H
    for t in range(3):
        rng = np.random.default_rng(30 + t)
        x = rng.standard_normal(H).astype(np.float32)
        got = ne.expert_moe(gate, 2, up, 2, down, 1, x)
        # the materialised oracle: decode every row, dot, then the chain
        def mm(parts, fmt, vec):
            weight, signix, scales = parts
            return np.array([np.dot(ne.decode_row(weight, scales, fmt, r, len(vec), signix=signix).astype(np.float64),
                                    vec.astype(np.float64)) for r in range(weight.shape[0])], dtype=np.float32)
        g = mm(gate, 2, x)
        u = mm(up, 2, x)
        mid = u * ne._silu(g)
        want = mm(down, 1, mid)
        assert np.allclose(got, want, rtol=1e-5, atol=1e-4)


def test_format_name_rejects_the_affine_zero():
    assert ne.format_name(1) == "IQ4_NL"
    assert ne.format_name(2) == "IQ3_XXS"
    assert ne.format_name(3) == "Q8_0"
    assert ne.format_name(4) == "IQ2_S"
    with pytest.raises(ValueError):
        ne.format_name(0)


def test_reference_matches_the_native_blocks_oracle():
    """The INDEPENDENT oracle: native_blocks' own DECODE, pinned to gguf-py on
    the real shards by test_native_blocks.py. Every row's every element is read
    back through the reference GEMV with a unit vector, so a wrong table entry,
    a wrong grid/sign orientation or a swapped group is caught against a decoder
    outside this module. (The packed-nibble extraction is shared with the
    implementation and is covered by test_native_blocks' own cell.)"""
    rows, K = 2, 64
    for fmt, builder in [(1, iq4nl), (2, iq3xxs), (3, q8_0), (4, iq2s)]:
        weight, signix, scales = builder(rows, K, seed=41)
        name = ne.format_name(fmt)
        if name == "IQ4_NL":
            codes = np.zeros((rows, K), np.uint8)
            for r in range(rows):
                for k in range(K):
                    byte = int(weight[r][k >> 1])
                    codes[r, k] = (byte >> 4) if (k & 1) else (byte & 0x0F)
            oracle = nb.iq4_nl_decode(codes, scales)
        elif name == "Q8_0":
            oracle = nb.q8_0_decode(weight, scales)
        elif name == "IQ2_S":
            gi = np.ascontiguousarray(weight).view("<u2").reshape(rows, K // 8)
            oracle = nb.iq2_s_decode(gi, np.ascontiguousarray(signix).reshape(rows, K // 8),
                                     np.repeat(scales, 2, axis=-1).reshape(rows, K // 8))
        else:
            oracle = nb.iq3_xxs_decode(weight, signix, scales)
        for r in range(rows):
            for k in range(K):
                x = np.zeros(K, dtype=np.float32)
                x[k] = 1.0
                got = ne.gemv(weight, scales, fmt, x, signix=signix, rows=r + 1)[r]
                assert got == pytest.approx(float(oracle[r, k]), rel=1e-6, abs=1e-6), (fmt, r, k)


def test_the_device_scale_transpose_convention():
    """The OCL kernel reads scales as S[g*N + n], the [groups, oc] layout
    maybe_transpose_scale_zp writes; the reference reads [rows, groups]. Pin the
    two against each other so the device convention is exercised device-free."""
    rows, K = 3, 64
    weight, signix, scales = iq4nl(rows, K, seed=51)
    x = np.random.default_rng(3).standard_normal(K).astype(np.float32)
    ref = ne.gemv(weight, scales, 1, x, rows=rows)
    dev = np.ascontiguousarray(scales.T)  # [groups, rows]
    out = np.zeros(rows, dtype=np.float32)
    for n in range(rows):
        acc = np.float32(0.0)
        for g in range(K // 32):
            vals = ne.decode_group_iq4_nl(weight[n], g) * np.float32(dev[g, n])
            acc += np.float32(np.dot(vals.astype(np.float64), x[g * 32:(g + 1) * 32].astype(np.float64)))
            acc = np.float32(acc)
        out[n] = acc
    assert np.array_equal(ref, out), "the reference and the device transpose disagree"
