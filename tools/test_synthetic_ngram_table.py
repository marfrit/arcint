#!/usr/bin/env python3
"""Unit tests for tools/synthetic_ngram_table.py.

Every test is a python-only exercise of the emitter's byte layout
against a python-side reference decoder built from the design doc's
own `docs/design-qwen-flash-next.md` §"The format" byte-layout table.
No C++ / arcint build dependency; the arcint tree's own gguf_dequant
reference is exercised elsewhere (tests/test_ngram_gather.cpp), not
here.

Tolerances are the per-format half-step:
  Q4_0: half a grid step = |d|/2 at scale |d| ~ 0.036 (this file's
        max |v| ~ 0.29 divided by 8) -> ~0.02.
  Q4_1: half a grid step = (range/15)/2. At range ~0.58 -> ~0.02.
  Q8_0: half a grid step = amax/127/2 ~ 0.002 at amax ~ 0.29.
A tolerance loose enough to hide a sign flip, off-by-one offset or
half-step scale error would defeat the purpose of these tests --
Fable review, 2026-09-10, mutation-tested the emitter at that class
of regression and every mutation slipped through the previous 0.05
tolerance. The 0.02 / 0.02 / 0.002 numbers here are the smallest
values that keep the emitter's own output within tolerance while
rejecting each of those regressions.

Run:  python3 tools/test_synthetic_ngram_table.py
"""
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from synthetic_ngram_table import (  # noqa: E402
    BLOCK_ELEMENTS,
    BYTES_PER_BLOCK,
    FORMAT_NAMES,
    HEADER_BYTES,
    HEADER_MAGIC,
    _f32_to_f16_bits,
    _iround,
    _quantize_q4_0,
    _row_floats,
    build_header,
    emit_synthetic_ngram_table,
    parse_header,
)


def _f16_bits_to_f32(bits: int) -> float:
    """Reverse of `_f32_to_f16_bits`. Straight IEEE."""
    sign = (bits >> 15) & 0x1
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF
    if exp == 0x1F:
        return float("inf") if mant == 0 else float("nan")
    if exp == 0:
        if mant == 0:
            return -0.0 if sign else 0.0
        val = mant / 1024.0 * (2.0 ** -14)
        return -val if sign else val
    val = (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))
    return -val if sign else val


def _dequant_q4_0_row(payload: bytes, n_cols: int) -> list:
    """Reference decode for Q4_0 per design doc §"The format"."""
    out = []
    off = 0
    for _ in range(n_cols // BLOCK_ELEMENTS):
        d = _f16_bits_to_f32(struct.unpack("<H", payload[off:off + 2])[0])
        off += 2
        lo = [0] * BLOCK_ELEMENTS
        for l in range(BLOCK_ELEMENTS // 2):
            byte = payload[off + l]
            lo[l] = (byte & 0x0F) - 8
            lo[l + 16] = (byte >> 4) - 8
        for q in lo:
            out.append(q * d)
        off += BLOCK_ELEMENTS // 2
    return out


def _dequant_q4_1_row(payload: bytes, n_cols: int) -> list:
    out = []
    off = 0
    for _ in range(n_cols // BLOCK_ELEMENTS):
        d = _f16_bits_to_f32(struct.unpack("<H", payload[off:off + 2])[0])
        m = _f16_bits_to_f32(struct.unpack("<H", payload[off + 2:off + 4])[0])
        off += 4
        lo = [0] * BLOCK_ELEMENTS
        for l in range(BLOCK_ELEMENTS // 2):
            byte = payload[off + l]
            lo[l] = byte & 0x0F
            lo[l + 16] = byte >> 4
        for q in lo:
            out.append(q * d + m)
        off += BLOCK_ELEMENTS // 2
    return out


def _dequant_q8_0_row(payload: bytes, n_cols: int) -> list:
    out = []
    off = 0
    for _ in range(n_cols // BLOCK_ELEMENTS):
        d = _f16_bits_to_f32(struct.unpack("<H", payload[off:off + 2])[0])
        off += 2
        for _l in range(BLOCK_ELEMENTS):
            q = struct.unpack("<b", payload[off:off + 1])[0]
            out.append(q * d)
            off += 1
    return out


DEQUANT = {"q4_0": _dequant_q4_0_row, "q4_1": _dequant_q4_1_row, "q8_0": _dequant_q8_0_row}
TOLERANCE = {"q4_0": 0.02, "q4_1": 0.02, "q8_0": 0.002}


class TestF16Roundtrip(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(_f32_to_f16_bits(0.0), 0x0000)
        self.assertEqual(_f16_bits_to_f32(0x0000), 0.0)

    def test_one(self):
        self.assertEqual(_f32_to_f16_bits(1.0), 0x3C00)
        self.assertAlmostEqual(_f16_bits_to_f32(0x3C00), 1.0)

    def test_negative_normal(self):
        self.assertEqual(_f32_to_f16_bits(-1.0), 0xBC00)
        self.assertEqual(_f32_to_f16_bits(-0.0625), 0xAC00)

    def test_small_positive_roundtrip(self):
        for x in (0.01, 0.1, 0.5, 2.5, 100.0):
            back = _f16_bits_to_f32(_f32_to_f16_bits(x))
            self.assertAlmostEqual(back, x, delta=abs(x) * 0.01)


class TestIRoundConvention(unittest.TestCase):
    """`_iround` is half-away-from-zero, matching ggml. Python's own
    `round()` is half-to-even; a test that passes on both proves nothing
    about the ggml compatibility this file claims."""

    def test_half_away_from_zero(self):
        self.assertEqual(_iround(0.5), 1)
        self.assertEqual(_iround(1.5), 2)
        self.assertEqual(_iround(2.5), 3)              # Python round(2.5) = 2
        self.assertEqual(_iround(-0.5), -1)            # Python round(-0.5) = 0
        self.assertEqual(_iround(-2.5), -3)

    def test_regular_rounding_unchanged(self):
        self.assertEqual(_iround(1.4), 1)
        self.assertEqual(_iround(-1.4), -1)
        self.assertEqual(_iround(0.6), 1)


class TestByteLayoutFilesystem(unittest.TestCase):
    """On-disk file sizes must match the design doc's stated bytes per
    block. Uses `os.path.getsize` against a written file (as Fable's
    review, 2026-09-10, specifically noted: comparing the emitter's
    arithmetic against itself is not a check)."""

    def test_file_sizes(self):
        for fmt, expected_per_row in (("q4_0", 90), ("q4_1", 100), ("q8_0", 170)):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                name = f.name
            try:
                got_payload, blocks_per_row = emit_synthetic_ngram_table(
                    name, n_rows=3, n_cols=160, fmt=fmt, seed=7)
                self.assertEqual(blocks_per_row, 5)
                self.assertEqual(got_payload, 3 * expected_per_row)
                self.assertEqual(os.path.getsize(name),
                                 HEADER_BYTES + 3 * expected_per_row)
            finally:
                os.unlink(name)


class TestGoldenBlockQ4_0(unittest.TestCase):
    """One golden 32-element block pinned to exact bytes. Any regression
    on Q4_0's sign convention, offset (nibble-8), scale (d = amax/-8) or
    nibble packing (low = q[l], high = q[l+16]) trips this test.

    Input:  [0.5, -0.25, 0, 0, ...] (element 0 = 0.5 = max |v|)
    d      = 0.5 / -8 = -0.0625 => f16 bits 0xAC00
    q[0]   = round(0.5 * -16) + 8 = -8 + 8 = 0
    q[1]   = round(-0.25 * -16) + 8 = 4 + 8 = 12 (0xC)
    q[l>=2]= 0 * ... + 8 = 8 (all)
    q[l+16>=0..15] all zero contribution => nibble 8.

    Packed byte layout (low nibble = q[l], high nibble = q[l+16]):
      byte[0] = (8 << 4) | 0  = 0x80  (l=0)
      byte[1] = (8 << 4) | 12 = 0x8C  (l=1)
      byte[2..15] = (8 << 4) | 8 = 0x88

    Full block bytes:  00 AC 80 8C 88 88 88 88 88 88 88 88 88 88 88 88 88 88
    """
    GOLDEN = bytes([
        0x00, 0xAC,                                    # f16 d = -0.0625
        0x80, 0x8C,
        0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
        0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
    ])

    def test_golden(self):
        block = [0.0] * BLOCK_ELEMENTS
        block[0] = 0.5
        block[1] = -0.25
        got = _quantize_q4_0(block)
        self.assertEqual(len(got), 18)
        self.assertEqual(got, self.GOLDEN,
                         msg=f"Q4_0 golden mismatch:\n"
                             f"got: {got.hex(' ')}\n"
                             f"want: {self.GOLDEN.hex(' ')}")


class TestQuantizeDequantizeRoundtrip(unittest.TestCase):
    """Every row's decoded value must round-trip within TOLERANCE.
    Read the payload bytes (skipping the 24-byte header) and decode
    each row with the python reference."""

    def test_all_three_formats(self):
        n_rows, n_cols, seed = 4, 160, 42
        for fmt in ("q4_0", "q4_1", "q8_0"):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                name = f.name
            try:
                emit_synthetic_ngram_table(name, n_rows=n_rows, n_cols=n_cols,
                                           fmt=fmt, seed=seed)
                with open(name, "rb") as fh:
                    all_bytes = fh.read()
            finally:
                os.unlink(name)
            payload = all_bytes[HEADER_BYTES:]
            per_row = len(payload) // n_rows
            for row in range(n_rows):
                row_bytes = payload[row * per_row:(row + 1) * per_row]
                decoded = DEQUANT[fmt](row_bytes, n_cols)
                truth = _row_floats(row, n_cols, seed)
                self.assertEqual(len(decoded), n_cols)
                for c in range(n_cols):
                    self.assertAlmostEqual(
                        decoded[c], truth[c], delta=TOLERANCE[fmt],
                        msg=f"{fmt} row {row} col {c}: got {decoded[c]}, "
                            f"expected ~ {truth[c]}",
                    )


class TestDeterminism(unittest.TestCase):
    def test_same_seed_identical(self):
        with tempfile.NamedTemporaryFile(delete=False) as a, \
             tempfile.NamedTemporaryFile(delete=False) as b:
            an, bn = a.name, b.name
        try:
            emit_synthetic_ngram_table(an, n_rows=5, n_cols=160, fmt="q4_0", seed=13)
            emit_synthetic_ngram_table(bn, n_rows=5, n_cols=160, fmt="q4_0", seed=13)
            with open(an, "rb") as f:
                a_bytes = f.read()
            with open(bn, "rb") as f:
                b_bytes = f.read()
            self.assertEqual(a_bytes, b_bytes)
        finally:
            os.unlink(an)
            os.unlink(bn)

    def test_different_seed_differs(self):
        with tempfile.NamedTemporaryFile(delete=False) as a, \
             tempfile.NamedTemporaryFile(delete=False) as b:
            an, bn = a.name, b.name
        try:
            emit_synthetic_ngram_table(an, n_rows=5, n_cols=160, fmt="q4_0", seed=1)
            emit_synthetic_ngram_table(bn, n_rows=5, n_cols=160, fmt="q4_0", seed=2)
            with open(an, "rb") as f:
                a_bytes = f.read()
            with open(bn, "rb") as f:
                b_bytes = f.read()
            self.assertNotEqual(a_bytes, b_bytes)
        finally:
            os.unlink(an)
            os.unlink(bn)


class TestArgumentRefusals(unittest.TestCase):
    def test_unknown_format(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            name = f.name
        try:
            with self.assertRaises(ValueError) as cm:
                emit_synthetic_ngram_table(name, n_rows=1, n_cols=160,
                                           fmt="q5_K", seed=0)
            self.assertIn("q5_K", str(cm.exception))
        finally:
            os.unlink(name)

    def test_non_block_multiple_cols(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            name = f.name
        try:
            with self.assertRaises(ValueError) as cm:
                emit_synthetic_ngram_table(name, n_rows=1, n_cols=100,
                                           fmt="q4_0", seed=0)
            self.assertIn("32", str(cm.exception))
        finally:
            os.unlink(name)

    def test_negative_dims(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            name = f.name
        try:
            for bad_rows, bad_cols in ((0, 160), (5, 0), (-1, 160), (5, -32)):
                with self.assertRaises(ValueError):
                    emit_synthetic_ngram_table(name, n_rows=bad_rows,
                                               n_cols=bad_cols, fmt="q4_0", seed=0)
        finally:
            os.unlink(name)


class TestHeader(unittest.TestCase):
    """The 24-byte ARCINGRM header is what Link 2 (loader admission,
    not yet landed) reads to know the file's format and shape. Its
    magic, size and field parses must be exact; any drift makes a
    valid emitter file unloadable by a valid loader."""

    def test_build_and_parse_roundtrip(self):
        for fmt in ("q4_0", "q4_1", "q8_0"):
            hdr = build_header(fmt, n_cols=160, n_rows=7)
            self.assertEqual(len(hdr), HEADER_BYTES)
            self.assertEqual(hdr[:8], HEADER_MAGIC)
            got_type, got_cols, got_rows = parse_header(hdr)
            self.assertEqual(got_type, FORMAT_NAMES[fmt])
            self.assertEqual(got_cols, 160)
            self.assertEqual(got_rows, 7)

    def test_emitted_file_header(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            name = f.name
        try:
            emit_synthetic_ngram_table(name, n_rows=3, n_cols=160,
                                       fmt="q4_1", seed=99)
            with open(name, "rb") as fh:
                head = fh.read(HEADER_BYTES)
            got_type, got_cols, got_rows = parse_header(head)
            self.assertEqual(got_type, 3)                        # Q4_1
            self.assertEqual(got_cols, 160)
            self.assertEqual(got_rows, 3)
        finally:
            os.unlink(name)

    def test_bad_magic_refused(self):
        buf = b"WRONGMAG" + b"\x00" * (HEADER_BYTES - 8)
        with self.assertRaises(ValueError) as cm:
            parse_header(buf)
        self.assertIn("magic", str(cm.exception))

    def test_truncated_refused(self):
        with self.assertRaises(ValueError) as cm:
            parse_header(HEADER_MAGIC + b"\x00" * 4)             # only 12 bytes
        self.assertIn("truncated", str(cm.exception))

    def test_unknown_type_refused(self):
        buf = HEADER_MAGIC + struct.pack("<IIII", 99, 160, 3, 0)
        with self.assertRaises(ValueError) as cm:
            parse_header(buf)
        self.assertIn("ggml_type", str(cm.exception))

    def test_non_block_cols_refused(self):
        buf = HEADER_MAGIC + struct.pack("<IIII", 2, 100, 3, 0)
        with self.assertRaises(ValueError) as cm:
            parse_header(buf)
        self.assertIn("n_cols", str(cm.exception))

    def test_reserved_nonzero_refused(self):
        buf = HEADER_MAGIC + struct.pack("<IIII", 2, 160, 3, 42)
        with self.assertRaises(ValueError) as cm:
            parse_header(buf)
        self.assertIn("reserved", str(cm.exception))


class TestExportFlashNextCliIntegration(unittest.TestCase):
    """The generator must be reachable from tools/export_flash_next.py
    via `--emit-ngram-synthetic`. Byte-equality against a direct library
    call proves the CLI wrapper is not a second implementation."""

    def test_cli_flag_emits_file(self):
        import export_flash_next
        with tempfile.NamedTemporaryFile(delete=False) as f:
            name = f.name
        try:
            rc = export_flash_next.main([
                "--emit-ngram-synthetic",
                "--out", name,
                "--n-rows", "3",
                "--n-cols", "160",
                "--fmt", "q4_0",
                "--seed", "5",
            ])
            self.assertEqual(rc, 0)
            with open(name, "rb") as fh:
                cli_bytes = fh.read()
        finally:
            os.unlink(name)
        with tempfile.NamedTemporaryFile(delete=False) as g:
            gname = g.name
        try:
            emit_synthetic_ngram_table(gname, n_rows=3, n_cols=160,
                                       fmt="q4_0", seed=5)
            with open(gname, "rb") as gh:
                direct_bytes = gh.read()
        finally:
            os.unlink(gname)
        self.assertEqual(cli_bytes, direct_bytes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
