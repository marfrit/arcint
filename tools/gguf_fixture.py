#!/usr/bin/env python3
"""Generate the GGUF v3 fixture for src/core/gguf.* (docs/design-gguf-native.md
stage 0).

Device-free: writes a tiny but complete file with one tensor of each type
the stage 0 host dequantizers cover (Q4_K, Q5_K, Q6_K, Q8_0, plus one F32 and
one F16 tensor), a handful of metadata keys spanning int/float/string/string
array, and a reference dequantization of every quantized tensor computed by
gguf-py's own `quants.dequantize` -- not by arcint -- so tests/test_gguf.cpp
checks the C++ decoders against an independent implementation.

gguf-py's `quants` module implements `quantize_blocks` for Q8_0 but not for
the K-quants (only `dequantize_blocks`, since llama.cpp does K-quant
quantization in C with an importance-matrix search gguf-py does not
reproduce). This script therefore carries its own small, byte-layout-correct
encoders for Q4_K/Q5_K/Q6_K below (`encode_q4_k` etc.) -- not
error-optimal, just valid blocks -- and self-checks each one by decoding
with gguf-py's own `dequantize_blocks` immediately after encoding, before
ever writing to the fixture. The bit layout matches gguf-py's
`Q4_K.get_scale_min` / `Q5_K` / `Q6_K` dequantize_blocks exactly (see the
comments inline); this is also the layout DESIGN and gguf_dequant.cpp cite
as ggml's `dequantize_row_q4_K` &c.

Regenerate with:

    <venv>/bin/python tools/gguf_fixture.py

using a venv with the `gguf` package installed (numpy required). Output is
seeded (NUMPY_SEED below) so a regeneration reproduces byte-identical
tensors; it does not reproduce a byte-identical *file*, since gguf-py does
not guarantee metadata/tensor-info byte order across versions -- the tests
read the file's own tensor list rather than assuming fixed offsets.
"""
import json
import os

import numpy as np

from gguf import GGUFWriter, GGMLQuantizationType
from gguf import quants

NUMPY_SEED = 20260906
QK_K = 256

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(HERE, "..", "tests", "fixtures")
GGUF_PATH = os.path.join(FIXTURES_DIR, "qwen35-tiny.gguf")
DEQUANT_PATH = os.path.join(FIXTURES_DIR, "qwen35-tiny.dequant.json")
DEQUANT_BIN_PATH = os.path.join(FIXTURES_DIR, "qwen35-tiny.dequant.bin")


def _pack_scale_min(sc, m):
    """The 12-byte Q4_K/Q5_K scale field, inverse of gguf-py's
    Q4_K.get_scale_min. sc, m: uint8 arrays of shape (8,), each < 64.
    """
    d_byte  = np.zeros(4, dtype=np.uint8)
    m_byte  = np.zeros(4, dtype=np.uint8)
    md_byte = np.zeros(4, dtype=np.uint8)
    for i in range(4):
        d_byte[i]  = (sc[i] & 0x3F) | (((sc[i + 4] >> 4) & 0x3) << 6)
        m_byte[i]  = (m[i] & 0x3F) | (((m[i + 4] >> 4) & 0x3) << 6)
        md_byte[i] = (sc[i + 4] & 0xF) | ((m[i + 4] & 0xF) << 4)
    return np.concatenate([d_byte, m_byte, md_byte]).tobytes()


def _subblock_scale_min(sub, bits):
    """Per-32-value sub-block affine fit: value ~= step*q + lo, q in
    [0, 2**bits - 1]. Returns (step, lo, q). `lo` is clamped to <= 0
    because the block format only ever subtracts (dmin*m), never adds.
    """
    qmax = (1 << bits) - 1
    lo = min(0.0, float(sub.min()))
    hi = float(sub.max())
    step = (hi - lo) / qmax if hi > lo else 1.0
    q = np.clip(np.round((sub - lo) / step), 0, qmax).astype(np.uint8)
    return step, -lo, q  # (d*sc target, dmin*m target, code)


def encode_k4_or_5(data: np.ndarray, bits: int) -> np.ndarray:
    """Shared Q4_K (bits=4) / Q5_K (bits=5) encoder. `data`: float32 array
    whose last axis is a multiple of QK_K. Returns the uint8 byte array
    gguf-py's Q4_K/Q5_K.dequantize_blocks decodes back out.
    """
    assert data.shape[-1] % QK_K == 0
    rows = data.reshape(-1, QK_K)
    # qs always carries the low 4 bits per value (QK_K/2 bytes); Q5_K's
    # fifth bit lives in the separate qh field, not in a wider qs.
    type_size = 2 + 2 + 12 + (QK_K // 8 if bits == 5 else 0) + QK_K // 2
    out = np.zeros((rows.shape[0], type_size), dtype=np.uint8)

    for bi, row in enumerate(rows):
        raw_scale = np.zeros(8)
        raw_min   = np.zeros(8)
        codes     = np.zeros((8, 32), dtype=np.uint8)
        for k in range(8):
            step, minterm, q = _subblock_scale_min(row[32 * k:32 * k + 32], bits)
            raw_scale[k] = step
            raw_min[k]   = minterm
            codes[k]     = q

        d    = max(raw_scale.max(), 1e-8) / 63.0
        dmin = max(raw_min.max(), 1e-8) / 63.0
        sc = np.clip(np.round(raw_scale / d), 0, 63).astype(np.uint8)
        m  = np.clip(np.round(raw_min / dmin), 0, 63).astype(np.uint8)

        buf = bytearray()
        buf += np.float16(d).tobytes()
        buf += np.float16(dmin).tobytes()
        buf += _pack_scale_min(sc, m)

        qs = np.zeros(QK_K // 2, dtype=np.uint8)
        qh = np.zeros(QK_K // 8, dtype=np.uint8) if bits == 5 else None
        for p in range(4):
            lo_code = codes[2 * p] & 0xF
            hi_code = codes[2 * p + 1] & 0xF
            qs[32 * p:32 * p + 32] = (lo_code | (hi_code << 4)).astype(np.uint8)
            if bits == 5:
                bit_lo = (codes[2 * p] >> 4) & 1
                bit_hi = (codes[2 * p + 1] >> 4) & 1
                qh |= (bit_lo << (2 * p)).astype(np.uint8)
                qh |= (bit_hi << (2 * p + 1)).astype(np.uint8)
        # File order is d, dmin, scales, qh, qs (qh before qs) for Q5_K.
        if qh is not None:
            buf += qh.tobytes()
        buf += qs.tobytes()

        assert len(buf) == type_size, (len(buf), type_size)
        out[bi] = np.frombuffer(bytes(buf), dtype=np.uint8)

    return out.reshape((*data.shape[:-1], data.shape[-1] // QK_K * type_size))


def encode_q6_k(data: np.ndarray) -> np.ndarray:
    assert data.shape[-1] % QK_K == 0
    rows = data.reshape(-1, QK_K)
    type_size = QK_K // 2 + QK_K // 4 + QK_K // 16 + 2
    out = np.zeros((rows.shape[0], type_size), dtype=np.uint8)

    for bi, row in enumerate(rows):
        raw_scale = np.zeros(16)
        codes     = np.zeros((16, 16), dtype=np.uint8)  # 6-bit codes, 0..63
        for k in range(16):
            sub = row[16 * k:16 * k + 16]
            amax = float(np.abs(sub).max())
            step = amax / 32.0 if amax > 0 else 1.0
            raw_scale[k] = step
            q = np.clip(np.round(sub / step) + 32, 0, 63).astype(np.uint8)
            codes[k] = q

        d = max(raw_scale.max(), 1e-8) / 127.0
        scales_i8 = np.clip(np.round(raw_scale / d), -127, 127).astype(np.int8)

        ql = np.zeros(QK_K // 2, dtype=np.uint8)
        qh = np.zeros(QK_K // 4, dtype=np.uint8)
        # gguf-py's Q6_K.dequantize_blocks addresses values as
        # y[half*128 + l], y[half*128+32+l], ... for l in 0..31 -- which
        # works out to sub-block k (16 contiguous values, scales[k]) being
        # exactly flat[16k : 16k+16]. Pack directly against that flat
        # ordering rather than re-deriving the half/l indexing twice.
        flat = codes.reshape(QK_K)  # flat[i] is the 6-bit code for value i
        for half in range(2):
            yoff, qloff, qhoff = half * 128, half * 64, half * 32
            for l in range(32):
                q1 = int(flat[yoff + l])
                q2 = int(flat[yoff + 32 + l])
                q3 = int(flat[yoff + 64 + l])
                q4 = int(flat[yoff + 96 + l])
                ql[qloff + l]      = (q1 & 0xF) | ((q3 & 0xF) << 4)
                ql[qloff + 32 + l] = (q2 & 0xF) | ((q4 & 0xF) << 4)
                qh[qhoff + l] = (((q1 >> 4) & 3) | (((q2 >> 4) & 3) << 2) |
                                 (((q3 >> 4) & 3) << 4) | (((q4 >> 4) & 3) << 6))

        buf = bytearray()
        buf += ql.tobytes()
        buf += qh.tobytes()
        buf += scales_i8.tobytes()
        buf += np.float16(d).tobytes()
        assert len(buf) == type_size, (len(buf), type_size)
        out[bi] = np.frombuffer(bytes(buf), dtype=np.uint8)

    return out.reshape((*data.shape[:-1], data.shape[-1] // QK_K * type_size))


def self_check(name, data, quantized, qtype):
    """Round-trip through gguf-py's OWN dequantize_blocks before this ever
    reaches the fixture: a bug in the hand-rolled encoders above must not
    turn into a fixture whose "reference" and "arcint" numbers agree only
    because both are wrong the same way.
    """
    ref = quants.dequantize(quantized, qtype).astype(np.float32)
    assert ref.shape == data.shape, (name, ref.shape, data.shape)
    finite = np.isfinite(ref).all()
    assert finite, (name, "non-finite in self-check dequant")
    # Not a precision claim (this encoder is not error-minimising) -- just
    # that decode is in the right ballpark, catching a transposed nibble or
    # a wrong scale-index mapping, which would blow this bound wide open.
    err = np.abs(ref - data)
    assert err.max() < 2.0, (name, "encoder self-check exceeds sanity bound",
                              float(err.max()))


# name -> (numpy shape as [out, in] or [n], quant type)
QUANT_TENSORS = {
    "blk.0.ffn_gate.weight": ((64, 512), GGMLQuantizationType.Q4_K),
    "blk.0.ffn_down.weight": ((512, 256), GGMLQuantizationType.Q6_K),
    "blk.0.ssm_out.weight":  ((32, 256), GGMLQuantizationType.Q5_K),
    "nextn.eh_proj.weight":  ((16, 64), GGMLQuantizationType.Q8_0),
    # For the template-pass test (tests/test_gguf_graph.cpp): a value-head
    # row-reordered projection (rows = 4 value heads x 64) and a fused
    # q/k/v projection (2 x 2 key heads x 64 q/k rows, then the 256 value
    # rows), both at the fixture's toy GDN geometry.
    "blk.0.attn_gate.weight": ((256, 256), GGMLQuantizationType.Q4_K),
    "blk.0.attn_qkv.weight":  ((512, 256), GGMLQuantizationType.Q6_K),
}


def quantize_any(data, qtype):
    if qtype == GGMLQuantizationType.Q4_K:
        return encode_k4_or_5(data, bits=4)
    if qtype == GGMLQuantizationType.Q5_K:
        return encode_k4_or_5(data, bits=5)
    if qtype == GGMLQuantizationType.Q6_K:
        return encode_q6_k(data)
    return quants.quantize(data, qtype)  # Q8_0 and friends: gguf-py has it


def main():
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    rng = np.random.default_rng(NUMPY_SEED)

    writer = GGUFWriter(GGUF_PATH, "qwen35")

    # ------------------------------------------------------------- metadata
    # Explicit even though 32 is gguf's own default: exercises the int getter
    # against a real key rather than only a fallback.
    writer.add_uint32("general.alignment", 32)
    writer.add_uint32("qwen35.block_count", 2)
    writer.add_uint32("qwen35.embedding_length", 64)
    writer.add_float32("qwen35.rope.freq_base", 10000.0)
    # The geometry keys the engine's open reads (core/gguf_map.cpp), at the
    # toy sizes the pass test uses: one model layer plus one MTP block, two
    # GDN key heads of 64 against four value heads of 64.
    writer.add_uint32("qwen35.nextn_predict_layers", 1)
    writer.add_uint32("qwen35.attention.head_count", 4)
    writer.add_uint32("qwen35.attention.head_count_kv", 2)
    writer.add_uint32("qwen35.attention.key_length", 16)
    writer.add_uint32("qwen35.ssm.group_count", 2)
    writer.add_uint32("qwen35.ssm.time_step_rank", 4)
    writer.add_uint32("qwen35.ssm.state_size", 64)
    writer.add_uint32("qwen35.ssm.inner_size", 256)
    writer.add_uint32("qwen35.full_attention_interval", 4)
    writer.add_string("tokenizer.ggml.pre", "qwen2")
    tokens = ["<pad>", "<s>", "</s>", "hello", ",", " world", "!", "▁tok"]
    writer.add_array("tokenizer.ggml.tokens", tokens)

    # -------------------------------------------------------------- tensors
    # The manifest (JSON, tiny) records shape/qtype/byte-range; the actual
    # float32 reference values go in a sidecar .bin (raw, native-endian,
    # concatenated in manifest order) so the pair stays commit-sized -- as
    # JSON, four tensors' worth of float32 text runs to several MB.
    manifest = {}
    blob = bytearray()

    # The exact-equality reference covers the four original tensors; the two
    # added for the template-pass test are checked structurally there and
    # would double the reference's size for nothing.
    for name, (shape, qtype) in QUANT_TENSORS.items():
        data = rng.standard_normal(shape).astype(np.float32)
        quantized = quantize_any(data, qtype)
        self_check(name, data, quantized, qtype)
        writer.add_tensor(name, quantized, raw_dtype=qtype)
        if name in ("blk.0.attn_gate.weight", "blk.0.attn_qkv.weight"):
            continue
        reference = quants.dequantize(quantized, qtype).astype(np.float32)
        manifest[name] = {
            "shape": list(reference.shape),
            "qtype": qtype.name,
            "byte_offset": len(blob),
            "n_floats": int(reference.size),
        }
        blob += reference.reshape(-1).astype("<f4").tobytes()

    norm = rng.standard_normal((64,)).astype(np.float32)
    writer.add_tensor("output_norm.weight", norm)
    # A layer norm the pass compares with the template's (the test builds a
    # template whose input_layernorm constant holds these same values).
    attn_norm = (1.0 + 0.01 * np.arange(512, dtype=np.float32)).astype(np.float32)
    writer.add_tensor("blk.0.attn_norm.weight", attn_norm)

    embd = rng.standard_normal((8, 64)).astype(np.float32).astype(np.float16)
    writer.add_tensor("token_embd.weight", embd)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    with open(DEQUANT_BIN_PATH, "wb") as f:
        f.write(bytes(blob))
    with open(DEQUANT_PATH, "w") as f:
        json.dump({"bin": os.path.basename(DEQUANT_BIN_PATH), "tensors": manifest},
                   f, indent=1, sort_keys=True)
        f.write("\n")

    size = os.path.getsize(GGUF_PATH)
    print(f"wrote {GGUF_PATH} ({size} bytes)")
    print(f"wrote {DEQUANT_PATH}")
    print(f"wrote {DEQUANT_BIN_PATH} ({len(blob)} bytes)")
    assert size < 1024 * 1024, f"fixture is {size} bytes, expected well under 1 MiB"


if __name__ == "__main__":
    main()
