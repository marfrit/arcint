"""Per-expert GEMV/GEMM with the checkpoint's own blocks decoded IN the K-loop
(campaign: sub4bit-vram-kernel, design-routing-aware-expert-execution.md §2.3a).

This is the CPU reference for the native-format per-expert kernel: the same
dequant arithmetic the OpenCL kernel does, expressed without materialising a
dequantised row. It exists so the kernel's numbers can be pinned device-free,
before a card is taken. The decoders are the ones of `tools/q4e/native_blocks.py`
(IQ4_NL's 16-entry table, IQ3_XXS's 256-entry grid and 128 sign masks, Q8_0's
plain i8), read from the per-role weight bytes the served artifact carries.

Layouts, exactly the artifact's per-expert slots (design note §2.3c; row-major
per role, the file layout the CPU tier also reads):

    IQ4_NL   weight  packed u4   [rows, K/2]   (element k: byte k/2, even low)
             scale   f16/f32     [rows, K/32]
    IQ3_XXS  weight  u8 grididx  [rows, K/4]   (one grid index per 4 values)
             signix  u8           [rows, K/8]   (one 7-bit sign-mask index per 8)
             scale   f16/f32     [rows, K/32]   (d*(0.5+s)*0.5, precomputed)
    IQ2_S    weight  u8           [rows, K/32, 8]  (four 10-bit grid indices per
             32 values, each index two little-endian bytes)
             signix  u8           [rows, K/32, 4]  (one RAW sign byte per 8
             values; bit j flips value j)
             scale   f16/f32     [rows, K/32, 2]  (the low nibble's scale for
             values 0..15, the high nibble's for 16..31)
    Q8_0     weight  i8          [rows, K]
             scale   f16/f32     [rows, K/32]

Every format uses 32-value blocks. A K that is not a multiple of 32 is refused
rather than decoded with a partial tail, and an unknown format is refused
rather than silently falling through to another decode -- both are what the
kernel's own contract requires (design note §2.3a red-first: "the per-expert
kernel's harness cell fed with block bytes whose decode is NOT alike across
pages").

Evidence: decoders pinned to llama.cpp's ggml-quants.c via
`tests/python/test_native_blocks.py` and `src/core/gguf_dequant.cpp`.
"""
from __future__ import annotations

import numpy as np

from q4e import native_blocks as nb

#: The native formats the served artifact uses, keyed by the plugin's
#: `MOECompressed::Config` weight_format value (patch 0043; IQ2_S added by
#: patch 0050): IQ4_NL == 1, IQ3_XXS == 2, Q8_0 == 3, IQ2_S == 4.
NATIVE_FORMATS = {1: "IQ4_NL", 2: "IQ3_XXS", 3: "Q8_0", 4: "IQ2_S"}

BLOCK = 32


def format_name(fmt: int) -> str:
    if fmt not in NATIVE_FORMATS:
        raise ValueError(f"unsupported native expert format {fmt!r}; known: {sorted(NATIVE_FORMATS)}")
    return NATIVE_FORMATS[fmt]


def _require_block_multiple(K: int, name: str) -> None:
    if K % BLOCK != 0:
        raise ValueError(
            f"{name}: K={K} is not a multiple of the {BLOCK}-value block; "
            "a partial tail is refused (no silent partial-block decode)"
        )


def decode_group_iq4_nl(weight_row: np.ndarray, g: int) -> np.ndarray:
    """One 32-value IQ4_NL group from the row's packed bytes: T[code] (no scale)."""
    out = np.empty(BLOCK, dtype=np.float32)
    for j in range(BLOCK):
        byte = int(weight_row[g * 16 + (j >> 1)])
        code = (byte >> 4) if (j & 1) else (byte & 0x0F)
        out[j] = nb.KVALUES_IQ4NL[code]
    return out


def decode_group_iq3_xxs(weight_row: np.ndarray, signix_row: np.ndarray, g: int) -> np.ndarray:
    """One 32-value IQ3_XXS group: grid magnitude * sign (no scale)."""
    out = np.empty(BLOCK, dtype=np.float32)
    for l in range(4):  # four 8-value sub-blocks per 32-value group
        signs = int(nb.KSIGNS_IQ2XS[int(signix_row[g * 4 + l]) & 127])
        g1 = int(weight_row[g * 8 + 2 * l])
        g2 = int(weight_row[g * 8 + 2 * l + 1])
        for m in range(8):
            idx = g1 if m < 4 else g2
            mag = float(nb.IQ3XXS_GRID[idx][m & 3])
            out[l * 8 + m] = mag * (-1.0 if (signs & int(nb.KMASK_IQ2XS[m])) else 1.0)
    return out


def decode_group_iq2_s(weight_row: np.ndarray, signix_row: np.ndarray,
                       scales_row: np.ndarray, g: int) -> np.ndarray:
    """One 32-value IQ2_S group from the artifact's per-role layout:
    `weight_row` u8 [K/32, 8] (four 10-bit grid indices, two little-endian
    bytes each), `signix_row` u8 [K/32, 4] (one RAW sign byte per 8 values,
    bit j flips value j), `scales_row` f16/f32 [K/32, 2] (the low nibble's
    scale for values 0..15, the high nibble's for 16..31). Returns
    magnitudes*signs*scale (the block scale folded in)."""
    out = np.empty(BLOCK, dtype=np.float32)
    for l in range(4):
        idx = int(weight_row[g, 2 * l]) | (int(weight_row[g, 2 * l + 1]) << 8)
        signs = int(signix_row[g, l])
        s = float(scales_row[g, 0] if l < 2 else scales_row[g, 1])
        for m in range(8):
            mag = float(nb.IQ2S_GRID[idx][m])
            out[l * 8 + m] = s * mag * (-1.0 if (signs & int(nb.KMASK_IQ2XS[m])) else 1.0)
    return out


def decode_group_q8_0(weight_row: np.ndarray, g: int) -> np.ndarray:
    return weight_row[g * BLOCK:(g + 1) * BLOCK].astype(np.float32)


def decode_row(weight, scales, fmt: int, row: int, K: int, signix=None) -> np.ndarray:
    """The materialised decode of one row (the oracle the fused GEMV is compared to)."""
    name = format_name(fmt)
    _require_block_multiple(K, name)
    out = np.empty(K, dtype=np.float32)
    for g in range(K // BLOCK):
        if name == "IQ4_NL":
            vals = decode_group_iq4_nl(weight[row], g)
        elif name == "Q8_0":
            vals = decode_group_q8_0(weight[row], g)
        elif name == "IQ2_S":
            vals = decode_group_iq2_s(weight[row], signix[row], scales[row], g)
        else:
            vals = decode_group_iq3_xxs(weight[row], signix[row], g)
        if name == "IQ2_S":
            out[g * BLOCK:(g + 1) * BLOCK] = vals
        else:
            out[g * BLOCK:(g + 1) * BLOCK] = vals * np.float32(scales[row, g])
    return out


def gemv(weight, scales, fmt: int, x: np.ndarray, signix=None, rows: int | None = None) -> np.ndarray:
    """Per-expert GEMV with the decode fused into the K-loop.

    weight/scales/signix are per-role arrays over all rows of one expert (or a
    stack whose first `rows` are used). Returns one float32 per row.
    """
    name = format_name(fmt)
    K = int(x.shape[0])
    _require_block_multiple(K, name)
    n = int(weight.shape[0]) if rows is None else rows
    out = np.zeros(n, dtype=np.float32)
    for r in range(n):
        acc = np.float32(0.0)
        for g in range(K // BLOCK):
            if name == "IQ4_NL":
                vals = decode_group_iq4_nl(weight[r], g)
            elif name == "Q8_0":
                vals = decode_group_q8_0(weight[r], g)
            elif name == "IQ2_S":
                vals = decode_group_iq2_s(weight[r], signix[r], scales[r], g)
            else:
                vals = decode_group_iq3_xxs(weight[r], signix[r], g)
            if name != "IQ2_S":
                vals = vals * np.float32(scales[r, g])  # the block scale, applied in the K-loop
            acc += np.float32(np.dot(vals.astype(np.float64),
                                     x[g * BLOCK:(g + 1) * BLOCK].astype(np.float64)))
            acc = np.float32(acc)  # one rounding per group, like the kernel's float accumulation
        out[r] = acc
    return out


def _silu(v: np.ndarray) -> np.ndarray:
    # x * sigmoid(x) = x * 0.5 * (1 + tanh(x/2)): no exp overflow on large |x|
    v64 = v.astype(np.float64)
    return (v64 * 0.5 * (1.0 + np.tanh(0.5 * v64))).astype(np.float32)


def expert_gate_up(gate, gate_fmt, up, up_fmt, x: np.ndarray):
    """gate(x) and up(x) for one expert; `gate`/`up` are (weight, signix, scales)."""
    g = gemv(gate[0], gate[2], gate_fmt, x, signix=gate[1])
    u = gemv(up[0], up[2], up_fmt, x, signix=up[1])
    return u, g


def expert_moe(gate, gate_fmt, up, up_fmt, down, down_fmt, x: np.ndarray) -> np.ndarray:
    """down(SiLU(gate(x)) * up(x)) for one expert, routing weight NOT applied."""
    u, g = expert_gate_up(gate, gate_fmt, up, up_fmt, x)
    mid = u * _silu(g)
    return gemv(down[0], down[2], down_fmt, mid, signix=down[1])


# ---------------------------------------------------------------------------
# A deliberately WRONG decode, for the red-first cell: the affine u4 reading of
# IQ4_NL's packed bytes (the nibble used as the value, the 16-entry table
# ignored). A kernel that silently fell into this would still produce finite
# numbers; the cell below must catch it rather than absorb it.
# ---------------------------------------------------------------------------

def wrong_affine_gemv_iq4_nl(weight, scales, x: np.ndarray) -> np.ndarray:
    K = int(x.shape[0])
    out = np.zeros(weight.shape[0], dtype=np.float32)
    for r in range(weight.shape[0]):
        acc = np.float32(0.0)
        for g in range(K // BLOCK):
            for j in range(BLOCK):
                k = g * BLOCK + j
                byte = int(weight[r][k >> 1])
                code = (byte >> 4) if (k & 1) else (byte & 0x0F)
                acc += np.float32(code * float(x[k]))
            acc = np.float32(acc * np.float32(scales[r, g]))
        out[r] = acc
    return out
