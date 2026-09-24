"""q4e.native_blocks: the checkpoint's expert blocks re-laid per role and
decoded exactly (DESIGN 7.0.2bz, design-routing-aware-expert-execution 2.3a/b).

Two kinds of cell: synthetic blocks whose expected values are derived by
hand from the block layout (the same blocks as tests/test_gguf.cpp's
decoder cells, so the numpy and the C++ side are pinned to one expectation),
and the real shards against gguf-py's own `dequantize` -- an oracle outside
this repository. `Q4E_GGUF_SHARDS` gates the latter, as in test_gguf_feed.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from q4e import native_blocks as nb  # noqa: E402

_SHARDS = os.environ.get("Q4E_GGUF_SHARDS", "")
_skip = pytest.mark.skipif(not _SHARDS, reason="Q4E_GGUF_SHARDS unset (real GGUF shards absent)")


def _f16_bytes(x):
    return np.array([x], dtype="<f2").view(np.uint8)


def test_iq4_nl_split_and_decode_on_a_hand_built_block():
    # d = 1.0, nibble j low / 15-j high in byte j: y[j] = T[j], y[j+16] = T[15-j]
    block = np.concatenate([_f16_bytes(1.0),
                            np.array([(j & 0xF) | ((15 - j) << 4) for j in range(16)], np.uint8)])
    # a second block at d = 0.5
    two = np.concatenate([block, block]); two[18:20] = _f16_bytes(0.5)
    codes, scales = nb.iq4_nl_split(two[None, :])
    assert codes.shape == (1, 64) and codes.dtype == np.uint8 and scales.shape == (1, 2)
    assert list(codes[0, :16]) == list(range(16)) and list(codes[0, 16:32]) == list(range(15, -1, -1))
    y = nb.iq4_nl_decode(codes, scales)
    want = np.concatenate([nb.KVALUES_IQ4NL, nb.KVALUES_IQ4NL[::-1]]).astype(np.float32)
    assert np.array_equal(y[0, :32], want)
    assert np.array_equal(y[0, 32:], want * np.float32(0.5))


def test_iq3_xxs_split_and_decode_on_a_hand_built_block():
    # d = 1.0; sub-block 0: index pair (1, 0) for the first 8 values, sign
    # index 5 (= 0b101: values 0 and 2 negative), scale 3 -> db = 1.75:
    # [-35, 7, -7, 7, 7, 7, 7, 7]; the rest of sub-block 0: grid 0 -> 7;
    # sub-blocks 1..7: scale 0 -> db 0.25, grid 0 -> 1.0
    block = np.zeros(98, np.uint8)
    block[0:2] = _f16_bytes(1.0)
    block[2] = 1
    block[66:70] = np.array([(3 << 28) | 5], "<u4").view(np.uint8)
    gridix, signix, scales = nb.iq3_xxs_split(block[None, :])
    assert gridix.shape == (1, 64) and signix.shape == (1, 32) and scales.shape == (1, 8)
    assert gridix[0, 0] == 1 and signix[0, 0] == 5 and scales[0, 0] == np.float32(1.75)
    assert np.all(scales[0, 1:] == np.float32(0.25))
    y = nb.iq3_xxs_decode(gridix, signix, scales)
    assert np.array_equal(y[0, :8], np.array([-35, 7, -7, 7, 7, 7, 7, 7], np.float32))
    assert np.all(y[0, 8:32] == 7.0) and np.all(y[0, 32:] == 1.0)


def test_iq4_xs_split_and_decode_on_a_hand_built_block():
    # one 256-value block, d = 1.0: sub-block ib gets scale bits ls = ib + 30
    # (low nibble in scales_l[ib//2], the two high bits in scales_h) so the
    # per-32 scale is d * (ls - 32) = ib - 2 (negative, zero and positive
    # scales all covered); the nibbles as in the IQ4_NL cell
    block = np.zeros(136, np.uint8)
    block[0:2] = _f16_bytes(1.0)
    ls = [ib + 30 for ib in range(8)]
    scales_h = sum(((v >> 4) & 3) << (2 * ib) for ib, v in enumerate(ls))
    block[2:4] = np.array([scales_h], "<u2").view(np.uint8)
    for ib, v in enumerate(ls):
        block[4 + ib // 2] |= (v & 0xF) << (4 * (ib % 2))
    block[8:136] = np.tile(np.array([(j & 0xF) | ((15 - j) << 4) for j in range(16)], np.uint8), 8)
    codes, scales = nb.iq4_xs_split(block[None, :])
    assert codes.shape == (1, 256) and codes.dtype == np.uint8 and scales.shape == (1, 8)
    assert list(scales[0]) == [float(ib - 2) for ib in range(8)]
    assert list(codes[0, :16]) == list(range(16)) and list(codes[0, 16:32]) == list(range(15, -1, -1))
    y = nb.iq4_xs_decode(codes, scales)
    want32 = np.concatenate([nb.KVALUES_IQ4NL, nb.KVALUES_IQ4NL[::-1]]).astype(np.float32)
    for ib in range(8):
        assert np.array_equal(y[0, ib * 32:(ib + 1) * 32], want32 * np.float32(ib - 2)), ib
    # the IQ4_XS split is the IQ4_NL layout: the same decode reads it
    assert np.array_equal(nb.iq4_nl_decode(codes, scales), y)


def test_q8_0_split_and_decode_on_a_hand_built_block():
    # two 32-value blocks: d = 0.5 then d = -2.0; qs = j - 16 (covers -16..15)
    qs = np.arange(-16, 16, dtype=np.int8)
    block = np.concatenate([_f16_bytes(0.5), qs.view(np.uint8), _f16_bytes(-2.0), qs.view(np.uint8)])
    codes, scales = nb.q8_0_split(block[None, :])
    assert codes.shape == (1, 64) and codes.dtype == np.int8 and scales.shape == (1, 2)
    assert np.array_equal(codes[0, :32], qs) and list(scales[0]) == [0.5, -2.0]
    y = nb.q8_0_decode(codes, scales)
    assert np.array_equal(y[0, :32], qs.astype(np.float32) * np.float32(0.5))
    assert np.array_equal(y[0, 32:], qs.astype(np.float32) * np.float32(-2.0))


def test_iq2_s_split_and_decode_on_a_hand_built_block():
    # one 256-value block, d = 1.0. ib32 = 0: low indices [0, 1, 2, 3], sign
    # bytes [0x00, 0xFF, 0, 0], qh = 0x01, scales[0] = 0x0F.
    #   l=0: idx = 0 | ((0x01 >> 0) & 3) << 8 = 256, scale low nibble 15
    #   l=1: idx = 1 | ((0x01 >> 2) & 3) << 8 = 1,   sign 0xFF -> all negative
    #   l=2: idx = 2, l=3: idx = 3                    scale high nibble 0
    # scales are lo, lo, hi, hi per 32-value sub-block: 3.875 then 0.125.
    block = np.zeros(82, np.uint8)
    block[0:2] = _f16_bytes(1.0)
    block[2:34].reshape(8, 4)[0] = [0, 1, 2, 3]
    block[34:66].reshape(8, 4)[0] = [0x00, 0xFF, 0, 0]
    block[66:74][0] = 0x01
    block[74:82][0] = 0x0F
    gridix, signix, scales = nb.iq2_s_split(block[None, :])
    assert gridix.shape == (1, 32) and gridix.dtype == np.uint16
    assert signix.shape == (1, 32) and signix.dtype == np.uint8
    assert scales.shape == (1, 32) and scales.dtype == np.float32
    assert [int(x) for x in gridix[0, :4]] == [256, 1, 2, 3]
    assert int(signix[0, 1]) == 0xFF
    assert scales[0, 0] == np.float32(3.875) and scales[0, 2] == np.float32(0.125)
    y = nb.iq2_s_decode(gridix, signix, scales)
    assert np.array_equal(y[0, 0:8], nb.IQ2S_GRID[256].astype(np.float32) * np.float32(3.875))
    assert np.array_equal(y[0, 8:16], -nb.IQ2S_GRID[1].astype(np.float32) * np.float32(3.875))
    assert np.array_equal(y[0, 16:24], nb.IQ2S_GRID[2].astype(np.float32) * np.float32(0.125))
    assert np.array_equal(y[0, 24:32], nb.IQ2S_GRID[3].astype(np.float32) * np.float32(0.125))


def test_every_split_has_its_block_geometry_and_decode():
    assert set(nb.SPLIT) == set(nb.BLOCK_BYTES) == set(nb.DECODE) == {
        "IQ4_NL", "IQ3_XXS", "IQ2_S", "IQ4_XS", "Q8_0"}
    assert nb.BLOCK_BYTES == {"IQ4_NL": (32, 18), "IQ3_XXS": (256, 98),
                             "IQ2_S": (256, 82), "IQ4_XS": (256, 136),
                             "Q8_0": (32, 34)}


@_skip
@pytest.mark.parametrize("name", ["blk.0.ffn_gate_exps.weight",
                                  "blk.0.ffn_up_exps.weight"])
def test_iq2_s_split_decodes_the_real_shard_exactly_as_gguf_py_does(feed, name):
    """IQ2_S, the format the 35B-A3B UD-IQ3_XXS file ships its gate/up experts
    in: gguf-py's own dequantize of the same bytes, every row of two experts.
    The other formats are pinned by the test above; this one skips when the
    gated shards are the Flash-Next checkpoint (whose gate/up are IQ3_XXS)."""
    if feed.gguf_type(name) != "IQ2_S":
        pytest.skip(f"{name} is {feed.gguf_type(name)}, not IQ2_S for these shards")
    raw = feed.raw_rows(name, rows=2)
    ref = np.asarray(feed.dequant(name, rows=2), np.float32)
    E, out, row_bytes = raw.shape
    parts = nb.SPLIT["IQ2_S"](raw.reshape(E * out, row_bytes))
    y = nb.DECODE["IQ2_S"](*parts).reshape(E, out, -1)
    assert y.shape == ref.shape, (y.shape, ref.shape)
    d = np.abs(y - ref)
    print(f"\n[iq2-s] {name}: rows {E * out}, max|diff| {d.max():.3e}, exact {np.array_equal(y, ref)}")
    assert np.array_equal(y, ref), f"{name}: max|diff| {d.max():.3e}"
    gridix, signix, scales = parts
    assert gridix.shape == (E * out, ref.shape[-1] // 8)
    assert signix.shape == (E * out, ref.shape[-1] // 8)
    assert scales.shape == (E * out, ref.shape[-1] // 8)


@pytest.fixture(scope="module")
def feed():
    from q4e import gguf_feed as gf
    return gf.GgufFeed(_SHARDS)


@_skip
@pytest.mark.parametrize("name,fmt", [("blk.0.ffn_down_exps.weight", "IQ4_NL"),
                                      ("blk.0.ffn_gate_exps.weight", "IQ3_XXS"),
                                      ("blk.24.ffn_up_exps.weight", "IQ3_XXS"),
                                      ("blk.2.ffn_gate_exps.weight", "IQ4_XS"),
                                      ("blk.2.ffn_down_exps.weight", "Q8_0")])
def test_the_split_decodes_the_real_shard_exactly_as_gguf_py_does(feed, name, fmt):
    """The oracle outside the repository: gguf-py's dequantize of the same
    bytes. Two experts, every row. The split is a byte re-arrangement, so the
    decode must land on gguf-py's f32 to float rounding, not to a tolerance
    that could hide a wrong table or a swapped nibble."""
    assert feed.gguf_type(name) in nb.SPLIT, (
        f"{name} is {feed.gguf_type(name)}, not a format this module splits")
    if feed.gguf_type(name) != fmt:
        pytest.skip(f"{name} is {feed.gguf_type(name)}, not {fmt} for these shards")
    raw = feed.raw_rows(name, rows=2)                      # [2, out, row_bytes]
    ref = np.asarray(feed.dequant(name, rows=2), np.float32)   # [2, out, in]
    E, out, row_bytes = raw.shape
    flat = raw.reshape(E * out, row_bytes)
    parts = nb.SPLIT[fmt](flat)
    y = nb.DECODE[fmt](*parts).reshape(E, out, -1)
    assert y.shape == ref.shape, (y.shape, ref.shape)
    d = np.abs(y - ref)
    print(f"\n[native-blocks] {name} ({fmt}): rows {E * out}, max|diff| {d.max():.3e}, "
          f"max|ref| {np.abs(ref).max():.4f}, exact {np.array_equal(y, ref)}")
    assert np.allclose(y, ref, rtol=1e-6, atol=0.0), f"{name}: max|diff| {d.max():.3e}"
    # the byte budget per role, the numbers the Fit table is recomputed with
    if fmt in ("IQ4_NL", "IQ4_XS", "Q8_0"):
        codes, scales = parts
        assert codes.shape == (E * out, ref.shape[-1]) and scales.shape == (E * out, ref.shape[-1] // 32)
        assert codes.dtype == (np.int8 if fmt == "Q8_0" else np.uint8)
    else:
        gridix, signix, scales = parts
        assert gridix.shape == (E * out, ref.shape[-1] // 4) and signix.shape == (E * out, ref.shape[-1] // 8)
        assert signix.max() <= 127 and scales.shape == (E * out, ref.shape[-1] // 32)
