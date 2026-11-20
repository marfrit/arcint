#!/usr/bin/env python3
"""Reader (and byte-mirror writer) for the LAC1 router-input capture.

LAC1 is campaign expert-foresight's layer-ahead step 1 (docs/campaigns/
expert-foresight.md, "Queued (operator, 2026-11-19): a layer-ahead predictor"):
tools/expert_router_capture.cpp records, for every token of a prompt and of a
greedy decode, each MoE layer's router input (the hidden state that feeds
ffn_gate_inp, after the layer's ffn norm) and the routed expert ids in rank
order, on the CPU. The format is tools/expert_router_capture_format.h:

    header   b"LAC1", u32 version, u32 n_layer, u32 n_embd, u32 n_expert,
             u32 n_used, u8 dtype (1: f16 inputs)   = 25 bytes, little endian
    token    i32 position, i32 token id, u8 phase (0 prompt, 1 decode), then
             n_layer x { n_embd x u16 f16 input row, n_used x i16 ids }
             inputs token-major (one row a layer), ids in rank order
             (descending, as ffn_moe_topk selected them)

A token record is `8 + 1 + n_layer * (2*n_embd + 2*n_used)` bytes; for
Flash-Next (48 layers, 2560 wide, top-10 of 512) 246,729 bytes, prompt and
decode alike. No trailer, no padding; a torn last record is dropped, never
half-read.

Stdlib only, so the unit cell runs where only python3 is installed (the
recipe's `ctest -L unit`); the replay converts Token.layer[i] to numpy itself.

API:
    read_header(path) -> Header(namedtuple-like fields, plus token_bytes)
    iter_tokens(path, limit=None, strict=False) -> iterator of
        Token(position, token_id, phase, layers)
        layers[i] = Layer(inputs, ids): inputs a tuple of floats (f16 values
        widened), ids a tuple of ints, rank order.
    iter_raw(path, limit=None, strict=False) -> iterator of
        Token with layers[i] = RawLayer(inputs_bytes, ids_bytes) (little-endian
        payloads), for a caller that keeps the bytes.
    write_capture(path, header, tokens) -> the byte mirror of the C++ writer.
        header a Header (or a dict with the same fields), tokens an iterable of
        (position, token_id, phase, layers) with layers[i] = (input_bits, ids):
        input_bits a sequence of f16 bit patterns (ints), ids a sequence of ints.

Red mutations (tests/python/test_expert_router_capture.py runs them):
    - write_capture dropping the phase byte: the reader's next token starts one
      byte early, the position or token id differs (round trip fails).
    - write_capture writing n_used+1 ids a layer: ids shift one i16 into the
      next layer's input row (round trip fails).
    - the reader accepting a short record: a truncated file reads one record
      with short layers instead of stopping at the torn one (strict cell fails).
"""

from __future__ import annotations

import struct
from collections import namedtuple

MAGIC = b"LAC1"
VERSION = 1
DTYPE_F16 = 1
PHASE_PROMPT = 0
PHASE_DECODE = 1

HEADER = struct.Struct("<4sIIIIIB")     # 25 bytes
TOKEN = struct.Struct("<iiB")           # 9 bytes


class CaptureError(ValueError):
    """A file that is not a LAC1 capture, or a writer input of the wrong shape."""


def token_bytes(n_layer: int, n_embd: int, n_used: int) -> int:
    """The record size, 9 + n_layer * (2*n_embd + 2*n_used)."""
    return TOKEN.size + n_layer * (2 * n_embd + 2 * n_used)


class Header(namedtuple("HeaderBase", "magic version n_layer n_embd n_expert n_used dtype")):
    @property
    def record_bytes(self) -> int:
        return token_bytes(self.n_layer, self.n_embd, self.n_used)


Layer = namedtuple("Layer", "inputs ids")
RawLayer = namedtuple("RawLayer", "inputs_bytes ids_bytes")
Token = namedtuple("Token", "position token_id phase layers")


def read_header(path) -> Header:
    """The geometry header of a capture. Raises CaptureError on a bad magic,
    a version this reader does not know, or a file shorter than the header."""
    with open(path, "rb") as f:
        raw = f.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise CaptureError(f"{path}: {len(raw)} bytes, shorter than the {HEADER.size}-byte header")
    magic, version, n_layer, n_embd, n_expert, n_used, dtype = HEADER.unpack(raw)
    if magic != MAGIC:
        raise CaptureError(f"{path}: magic {magic!r}, not {MAGIC!r}")
    if version != VERSION:
        raise CaptureError(f"{path}: version {version}, not {VERSION}")
    if dtype != DTYPE_F16:
        raise CaptureError(f"{path}: dtype {dtype}, not {DTYPE_F16} (f16 inputs)")
    if not (n_layer and n_embd and n_expert and n_used):
        raise CaptureError(f"{path}: geometry n_layer {n_layer} n_embd {n_embd} "
                           f"n_expert {n_expert} n_used {n_used} has a zero")
    return Header(magic, version, n_layer, n_embd, n_expert, n_used, dtype)


def _f16_values(payload: bytes) -> tuple:
    return struct.unpack(f"<{len(payload) // 2}e", payload)


def _ids_values(payload: bytes) -> tuple:
    return struct.unpack(f"<{len(payload) // 2}h", payload)


def iter_raw(path, limit=None, strict=False):
    """Token records as raw layer payloads (little-endian bytes), in file order.
    `limit` stops after that many records. A torn last record (a short read) is
    dropped with `strict=False`; `strict=True` raises CaptureError instead."""
    header = read_header(path)
    rec = header.record_bytes
    ids_bytes = 2 * header.n_used
    in_bytes = 2 * header.n_embd
    layer = in_bytes + ids_bytes
    n = 0
    with open(path, "rb") as f:
        f.seek(HEADER.size)
        while limit is None or n < limit:
            chunk = f.read(rec)
            if not chunk:
                return
            if len(chunk) < rec:
                if strict:
                    raise CaptureError(f"{path}: token {n} is {len(chunk)} of {rec} bytes (torn record)")
                return
            pos, token, phase = TOKEN.unpack_from(chunk, 0)
            if phase not in (PHASE_PROMPT, PHASE_DECODE):
                raise CaptureError(f"{path}: token {n} has phase {phase}, not 0 or 1")
            layers = []
            off = TOKEN.size
            for _ in range(header.n_layer):
                layers.append(RawLayer(chunk[off:off + in_bytes], chunk[off + in_bytes:off + layer]))
                off += layer
            n += 1
            yield Token(pos, token, phase, layers)


def iter_tokens(path, limit=None, strict=False):
    """Token records with the inputs widened to floats and the ids read as ints."""
    for tok in iter_raw(path, limit=limit, strict=strict):
        yield Token(tok.position, tok.token_id, tok.phase,
                    [Layer(_f16_values(raw.inputs_bytes), _ids_values(raw.ids_bytes))
                     for raw in tok.layers])


def write_capture(path, header, tokens) -> int:
    """The byte mirror of lac1::writer (tools/expert_router_capture_format.h).
    Writes the header, then one token record a token; returns the token count.
    A mutation of this writer is how the round-trip cells go red."""
    if isinstance(header, dict):
        header = Header(header["magic"], header["version"], header["n_layer"],
                        header["n_embd"], header["n_expert"], header["n_used"],
                        header["dtype"])
    if header.magic != MAGIC or header.version != VERSION:
        raise CaptureError(f"header magic {header.magic!r} version {header.version}, not LAC1 v{VERSION}")
    if header.dtype != DTYPE_F16:
        raise CaptureError(f"header dtype {header.dtype}, not {DTYPE_F16}")
    n_layer, n_embd, n_used = header.n_layer, header.n_embd, header.n_used
    n = 0
    with open(path, "wb") as f:
        f.write(HEADER.pack(MAGIC, VERSION, n_layer, n_embd, header.n_expert,
                            n_used, header.dtype))
        for position, token_id, phase, layers in tokens:
            if len(layers) != n_layer:
                raise CaptureError(f"token {n}: {len(layers)} layers, not {n_layer}")
            out = [TOKEN.pack(position, token_id, phase)]
            for inputs, ids in layers:
                bits = list(inputs)
                if len(bits) != n_embd:
                    raise CaptureError(f"token {n}: layer input {len(bits)} values, not {n_embd}")
                row = list(ids)
                if len(row) != n_used:
                    raise CaptureError(f"token {n}: {len(row)} ids, not {n_used}")
                out.append(struct.pack(f"<{n_embd}H", *[b & 0xFFFF for b in bits]))
                out.append(struct.pack(f"<{n_used}h", *row))
            f.write(b"".join(out))
            n += 1
    return n


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.stderr.write("usage: expert_router_capture_read.py FILE\n")
        raise SystemExit(2)
    h = read_header(sys.argv[1])
    print(f"LAC1 v{h.version}: n_layer {h.n_layer} n_embd {h.n_embd} "
          f"n_expert {h.n_expert} n_used {h.n_used}, {h.record_bytes} B a token")
    total = 0
    for tok in iter_tokens(sys.argv[1]):
        total += 1
        print(f"  pos {tok.position:>7} token {tok.token_id:>7} phase {tok.phase} "
              f"first ids {tok.layers[0].ids[:4]}")
    print(f"{total} tokens")
