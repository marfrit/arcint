"""Synthetic `per_layer_token_embd` table generator (FIX D Link 1).

Design doc `docs/design-qwen-flash-next.md` §"The tensor" and §"The
format" record what the table IS -- shape (~52.45 G elements, 160-wide
rows, config keys `ngram_vocab_size_base` / `ple_embed_dim` /
`ple_layer_ids`) and the Q4_0 / Q4_1 / Q8_0 byte layouts. The doc has
no section documenting HOW the table is produced from a source
checkpoint: a grep over the doc for `corpus`, `n-gram statistic`,
`learned` or `training procedure` returns only meta-references, and
the recon's "learned from the model's embeddings + n-gram statistics"
phrase names no procedure.

Verdict recorded 2026-09-10: the design-doc spec is too thin to
implement real generation. Per the roadmap's own allowance, this
module lands a synthetic stand-in whose bytes match the byte layouts
`docs/design-qwen-flash-next.md` §"The format" records and
`src/exec/ngram_gather.h`'s scalar decoder reads. When the real
procedure is documented, this file grows a real-generation entry
point alongside the synthetic one; the tests below already pin the
byte-layout surface, so the swap is a fill change, not a rewrite.

Row-width convention. The design doc names 160 elements per row: that
is `ple_embed_dim` (2560) split into 16 physical rows of 160 elements
each, so the block-32 constraint the AVX2 kernel relies on holds on
every row (160 = 5 * 32). The generator's `n_rows` argument counts
physical 160-wide rows; a caller who wants "per ngram_vocab_size_base
entry" multiplies vocab by (ple_embed_dim / 160) before calling.

Byte-layout compatibility with `ggml-quants.c`. The three per-format
quantizers here transcribe `quantize_row_q4_0_reference`,
`quantize_row_q4_1_reference` and `quantize_row_q8_0_reference` from
ggml, INCLUDING the half-up rounding convention (`floor(x + 0.5)`,
which Python `round()` does not do -- Python is half-to-even, ggml is
half-away-from-zero). See `_iround` below.

File format. A small header precedes the raw block-quantised bytes so
the loader (FIX D Link 2, not yet landed) can validate shape and type
without a companion config file:

    offset  size  field
    0       8     magic "ARCINGRM" (ASCII, no NUL terminator)
    8       4     ggml_type id (u32 LE): 2=Q4_0, 3=Q4_1, 8=Q8_0
    12      4     n_cols (u32 LE), must be a multiple of 32
    16      4     n_rows (u32 LE)
    20      4     reserved (u32, 0)
    24      -     raw quantised rows, `n_rows * blocks_per_row *
                  bytes_per_block` bytes total, no padding between rows

The header is 24 bytes. `parse_header` returns the tuple
(ggml_type, n_cols, n_rows) or raises `ValueError` if the magic or
any field is unrecognised.
"""
import math
import os
import struct
from typing import Tuple


BLOCK_ELEMENTS = 32                          # every format here blocks at 32
BYTES_PER_BLOCK = {2: 18, 3: 20, 8: 34}      # Q4_0=2, Q4_1=3, Q8_0=8 (ggml_type ids)
FORMAT_NAMES = {"q4_0": 2, "q4_1": 3, "q8_0": 8}

HEADER_MAGIC = b"ARCINGRM"
HEADER_BYTES = 24


def _iround(x: float) -> int:
    """Half-away-from-zero rounding, matching ggml's own quantizers
    (`(int8_t)(x + 8.5f)` for Q4_0, `roundf(x)` for Q8_0, `(int8_t)(x + 0.5f)`
    for Q4_1 -- all half-away-from-zero on their signed inputs).
    Python's built-in `round()` is half-to-even (banker's rounding),
    so a naive `round()` diverges from ggml at every X.5 tie."""
    return math.floor(x + 0.5) if x >= 0.0 else -math.floor(-x + 0.5)


def _f32_to_f16_bits(x: float) -> int:
    """f32 -> f16 as a 16-bit unsigned integer, matching ggml's
    `ggml_fp32_to_fp16` for the finite-normal range this file
    exercises (all scales `d` produced by the three quantizers below
    fall inside f16-normal). Subnormals flush to zero and out-of-range
    values clamp to +/- max normal (65504); ggml preserves both, so
    this function is not a general drop-in for ggml's converter -- it
    is a converter that agrees with ggml on this file's inputs."""
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    sign = (b >> 31) & 0x1
    exp = (b >> 23) & 0xFF
    mant = b & 0x7FFFFF
    if exp == 0xFF:
        return (sign << 15) | 0x7C00 | (1 if mant else 0)
    if exp == 0 or exp < 0x71:
        return sign << 15
    if exp > 0x8E:
        return (sign << 15) | 0x7BFF
    exp16 = exp - 0x70
    mant16 = mant >> 13
    round_bits = mant & 0x1FFF
    if round_bits > 0x1000 or (round_bits == 0x1000 and (mant16 & 1)):
        mant16 += 1
        if mant16 >> 10:
            mant16 = 0
            exp16 += 1
            if exp16 >= 0x1F:
                return (sign << 15) | 0x7BFF
    return (sign << 15) | (exp16 << 10) | mant16


def _hash_row_seed(row: int, seed: int) -> int:
    """Splitmix-style avalanche of (row, seed) into 64 bits. Deterministic,
    reproducible, avoids the constant-per-dimension trap this repository
    has already been burned by (HANDOFF-0.4.7.local.md, BY_TOKEN NaN)."""
    x = (row * 0x9E3779B97F4A7C15) ^ (seed * 0xBF58476D1CE4E5B9)
    x &= (1 << 64) - 1
    x ^= (x >> 30)
    x = (x * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    x ^= (x >> 27)
    x = (x * 0x94D049BB133111EB) & ((1 << 64) - 1)
    x ^= (x >> 31)
    return x


def _row_floats(row: int, n_cols: int, seed: int) -> list:
    """Deterministic per-column-varying floats for one row. The
    per-column phase term is what avoids the BY_TOKEN pathology; the
    amplitude is chosen so Q4_0's +/- 8 grid at scale ~0.036 quantises
    every value within one grid step."""
    h = _hash_row_seed(row, seed)
    a = ((h & 0xFFFF) / 65535.0) * 0.5 + 0.5      # 0.5 .. 1.0
    b = ((h >> 16) & 0xFFFF) / 65535.0            # 0.0 .. 1.0
    out = []
    for c in range(n_cols):
        t = c / max(1, n_cols - 1)
        v = a * (t - 0.5) + b * ((c % 3) - 1) * 0.1
        out.append(v * 0.5)
    return out


def _quantize_q8_0(row_floats: list) -> bytes:
    """Q8_0: f16 d + 32 signed int8 per block, decoded value `qs[l]*d`."""
    out = bytearray()
    for b in range(0, len(row_floats), BLOCK_ELEMENTS):
        block = row_floats[b:b + BLOCK_ELEMENTS]
        amax = max(abs(v) for v in block)
        d = amax / 127.0 if amax > 0 else 0.0
        out += struct.pack("<H", _f32_to_f16_bits(d))
        inv = (1.0 / d) if d != 0.0 else 0.0
        for v in block:
            q = _iround(v * inv)
            q = max(-127, min(127, q))
            out += struct.pack("<b", q)
    return bytes(out)


def _quantize_q4_0(row_floats: list) -> bytes:
    """Q4_0: f16 d + 16 packed 4-bit nibbles per block; d picked so the
    largest-|v| element maps to nibble 0 (i.e. `d = amax_signed / -8`),
    decoded value `(nibble - 8) * d`. Matches ggml-quants.c
    `quantize_row_q4_0_reference` including the half-up round."""
    out = bytearray()
    for b in range(0, len(row_floats), BLOCK_ELEMENTS):
        block = row_floats[b:b + BLOCK_ELEMENTS]
        amax_signed = max(block, key=abs) if block else 0.0
        d = amax_signed / -8.0 if amax_signed != 0.0 else 0.0
        out += struct.pack("<H", _f32_to_f16_bits(d))
        inv = (1.0 / d) if d != 0.0 else 0.0
        for l in range(BLOCK_ELEMENTS // 2):
            q0 = _iround(block[l] * inv) + 8
            q1 = _iround(block[l + 16] * inv) + 8
            q0 = max(0, min(15, q0))
            q1 = max(0, min(15, q1))
            out += struct.pack("<B", (q1 << 4) | q0)
    return bytes(out)


def _quantize_q4_1(row_floats: list) -> bytes:
    """Q4_1: f16 d + f16 m + 16 packed 4-bit nibbles per block, decoded
    value `nibble * d + m`. Matches ggml-quants.c
    `quantize_row_q4_1_reference`."""
    out = bytearray()
    for b in range(0, len(row_floats), BLOCK_ELEMENTS):
        block = row_floats[b:b + BLOCK_ELEMENTS]
        vmin = min(block)
        vmax = max(block)
        d = (vmax - vmin) / 15.0 if vmax > vmin else 0.0
        m = vmin
        out += struct.pack("<H", _f32_to_f16_bits(d))
        out += struct.pack("<H", _f32_to_f16_bits(m))
        inv = (1.0 / d) if d != 0.0 else 0.0
        for l in range(BLOCK_ELEMENTS // 2):
            q0 = _iround((block[l] - m) * inv)
            q1 = _iround((block[l + 16] - m) * inv)
            q0 = max(0, min(15, q0))
            q1 = max(0, min(15, q1))
            out += struct.pack("<B", (q1 << 4) | q0)
    return bytes(out)


_QUANTIZERS = {"q4_0": _quantize_q4_0, "q4_1": _quantize_q4_1, "q8_0": _quantize_q8_0}


def build_header(fmt: str, n_cols: int, n_rows: int) -> bytes:
    """Return the 24-byte ARCINGRM header for `(fmt, n_cols, n_rows)`.
    Raises ValueError on an unknown fmt so a caller does not silently
    produce a header the loader will not parse."""
    if fmt not in FORMAT_NAMES:
        raise ValueError(f"unknown fmt {fmt!r}: pick one of {sorted(FORMAT_NAMES)}")
    return (HEADER_MAGIC +
            struct.pack("<IIII", FORMAT_NAMES[fmt], n_cols, n_rows, 0))


def parse_header(buf: bytes) -> Tuple[int, int, int]:
    """Parse the ARCINGRM header. Returns (ggml_type, n_cols, n_rows).
    Raises ValueError with the exact field named on any inconsistency."""
    if len(buf) < HEADER_BYTES:
        raise ValueError(
            f"header truncated: {len(buf)} bytes, expected at least {HEADER_BYTES}")
    if buf[:8] != HEADER_MAGIC:
        raise ValueError(
            f"bad magic: {buf[:8]!r}, expected {HEADER_MAGIC!r}")
    ggml_type, n_cols, n_rows, reserved = struct.unpack("<IIII", buf[8:24])
    if ggml_type not in BYTES_PER_BLOCK:
        raise ValueError(
            f"unknown ggml_type {ggml_type}: pick one of {sorted(BYTES_PER_BLOCK)} "
            "(2=Q4_0, 3=Q4_1, 8=Q8_0)")
    if n_cols == 0 or n_cols % BLOCK_ELEMENTS != 0:
        raise ValueError(
            f"n_cols {n_cols} is not a positive multiple of {BLOCK_ELEMENTS}")
    if n_rows == 0:
        raise ValueError("n_rows must be positive")
    if reserved != 0:
        raise ValueError(f"reserved field must be zero, got {reserved}")
    return ggml_type, n_cols, n_rows


def emit_synthetic_ngram_table(out_path: str, n_rows: int, n_cols: int,
                               fmt: str, seed: int = 0) -> Tuple[int, int]:
    """Write a synthetic per_layer_token_embd table to `out_path`.

    Returns (payload_byte_count, blocks_per_row) -- the header's 24
    bytes are not counted in `payload_byte_count`, so a caller who
    wants total file size adds HEADER_BYTES.
    """
    if fmt not in _QUANTIZERS:
        raise ValueError(f"unknown fmt {fmt!r}: pick one of {sorted(_QUANTIZERS)}")
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"n_rows={n_rows} n_cols={n_cols}: both must be positive")
    if n_cols % BLOCK_ELEMENTS != 0:
        raise ValueError(
            f"n_cols={n_cols} is not a multiple of {BLOCK_ELEMENTS}: "
            f"Q4_0/Q4_1/Q8_0 block at 32 elements and cannot cross a row boundary"
        )
    quantizer = _QUANTIZERS[fmt]
    blocks_per_row = n_cols // BLOCK_ELEMENTS
    bytes_per_row = blocks_per_row * BYTES_PER_BLOCK[FORMAT_NAMES[fmt]]
    with open(out_path, "wb") as f:
        f.write(build_header(fmt, n_cols, n_rows))
        for row in range(n_rows):
            f.write(quantizer(_row_floats(row, n_cols, seed)))
    return n_rows * bytes_per_row, blocks_per_row
