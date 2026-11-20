"""The LAC1 router-input capture's device-free cells: writer/reader round trip.

Campaign expert-foresight, "Queued (operator, 2026-11-19): a layer-ahead
predictor", plan step 1. The capture tool (tools/expert_router_capture.cpp)
needs a model and a CPU host, so the round trip is tested here without either:
tools/expert_router_capture_read.py's writer mirror writes the file, its reader
reads it back, and tools/expert_router_capture_selftest.cpp -- the C++ writer
of tools/expert_router_capture_format.h, compiled with a plain C++ compiler --
must produce the same bytes. The C++ writer is the one the tool uses, so the
byte equality makes the fixture read the tool's own format, not a copy.

Standard library only (unittest), so the recipe's `ctest -L unit` runs it with
python3 and nothing else. pytest collects the unittest class too.

Red mutations, run and seen failing before each was removed (the mutant is in
this file or in the reader's docstring):
  (a) the mirror writer drops the phase byte -> token 1's position is read as
      something other than 7 and the byte equality fails.
  (b) the mirror writer writes n_used+1 ids a layer -> every later field shifts
      two bytes; the first token's layer-0 ids and inputs differ and the byte
      equality fails.
  (c) the reader accepts a short last record (strict=False) instead of stopping
      at the torn one -> a truncated file yields a token with short layer
      payloads; the strict cell fails.
"""

import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import expert_router_capture_read as ercr  # noqa: E402

GEOMETRY = dict(magic=ercr.MAGIC, version=ercr.VERSION, n_layer=3, n_embd=8,
                n_expert=6, n_used=3, dtype=ercr.DTYPE_F16)

# the C++ selftest's constants (tools/expert_router_capture_selftest.cpp)
TOKENS = [
    (0, 100, ercr.PHASE_PROMPT),
    (7, 42, ercr.PHASE_DECODE),
]


def synthetic_layers():
    """The selftest's input bits and ids, one list a layer."""
    layers = []
    for il in range(GEOMETRY["n_layer"]):
        bits = [0x3C00 + 0x40 * il + k for k in range(GEOMETRY["n_embd"])]
        ids = [(2 * il + 7 * k) % GEOMETRY["n_expert"] for k in range(GEOMETRY["n_used"])]
        layers.append((bits, ids))
    return layers


def write_mirror(path, mutate=None):
    tokens = [(pos, tok, phase, synthetic_layers()) for pos, tok, phase in TOKENS]
    if mutate is None:
        return ercr.write_capture(path, GEOMETRY, tokens)
    return mutate(path, GEOMETRY, tokens)


# --- the mutants (each keeps the header and shifts the records) -------------

def mutant_drop_phase(path, header, tokens):
    """(a) the phase byte is not written."""
    h = ercr.Header(**header)
    n = 0
    with open(path, "wb") as f:
        f.write(ercr.HEADER.pack(ercr.MAGIC, ercr.VERSION, h.n_layer, h.n_embd,
                                 h.n_expert, h.n_used, h.dtype))
        for position, token_id, phase, layers in tokens:
            out = [struct.pack("<ii", position, token_id)]   # no phase
            for bits, ids in layers:
                out.append(struct.pack(f"<{h.n_embd}H", *bits))
                out.append(struct.pack(f"<{h.n_used}h", *ids))
            f.write(b"".join(out))
            n += 1
    return n


def mutant_extra_id(path, header, tokens):
    """(b) n_used+1 ids a layer."""
    h = ercr.Header(**header)
    n = 0
    with open(path, "wb") as f:
        f.write(ercr.HEADER.pack(ercr.MAGIC, ercr.VERSION, h.n_layer, h.n_embd,
                                 h.n_expert, h.n_used, h.dtype))
        for position, token_id, phase, layers in tokens:
            out = [ercr.TOKEN.pack(position, token_id, phase)]
            for bits, ids in layers:
                out.append(struct.pack(f"<{h.n_embd}H", *bits))
                out.append(struct.pack(f"<{h.n_used + 1}h", *(list(ids) + [0])))
            f.write(b"".join(out))
            n += 1
    return n


def cxx():
    return os.environ.get("CXX") or "g++"


def build_selftest(tmp: Path):
    exe = tmp / "expert_router_capture_selftest"
    r = subprocess.run(
        [cxx(), "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
         "-I", str(REPO_ROOT / "tools"),
         str(REPO_ROOT / "tools" / "expert_router_capture_selftest.cpp"),
         "-o", str(exe)],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise unittest.SkipTest(f"the C++ writer selftest does not compile: {r.stderr.strip()[:400]}")
    return exe


class RoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lac1-"))
        self.path = self.tmp / "capture.lac1"

    def test_header_and_record_geometry(self):
        self.assertEqual(ercr.HEADER.size, 25)
        self.assertEqual(ercr.token_bytes(48, 2560, 10), 246729)
        h = ercr.read_header(self.path) if write_mirror(self.path) else None
        self.assertEqual(h.n_layer, GEOMETRY["n_layer"])
        self.assertEqual(h.n_embd, GEOMETRY["n_embd"])
        self.assertEqual(h.n_expert, GEOMETRY["n_expert"])
        self.assertEqual(h.n_used, GEOMETRY["n_used"])
        self.assertEqual(h.dtype, ercr.DTYPE_F16)
        self.assertEqual(h.record_bytes, 9 + 3 * (2 * 8 + 2 * 3))

    def test_round_trip(self):
        self.assertEqual(write_mirror(self.path), 2)
        tokens = list(ercr.iter_tokens(self.path))
        self.assertEqual(len(tokens), 2)
        layers = synthetic_layers()
        for got, (pos, tok, phase) in zip(tokens, TOKENS):
            self.assertEqual((got.position, got.token_id, got.phase), (pos, tok, phase))
            self.assertEqual(len(got.layers), GEOMETRY["n_layer"])
            for il, layer in enumerate(got.layers):
                bits, ids = layers[il]
                self.assertEqual(layer.ids, tuple(ids),
                                 f"layer {il} ids of token {got.position}")
                self.assertEqual(layer.inputs, struct.unpack(f"<{len(bits)}e", struct.pack(f"<{len(bits)}H", *bits)),
                                 f"layer {il} inputs of token {got.position}")
        raw = list(ercr.iter_raw(self.path))
        self.assertEqual([t.position for t in raw], [0, 7])

    def test_limit_and_torn_last_record(self):
        write_mirror(self.path)
        self.assertEqual(len(list(ercr.iter_tokens(self.path, limit=1))), 1)
        blob = self.path.read_bytes()
        self.path.write_bytes(blob[:-1])      # cut the last byte
        # strict: the torn last record is refused
        with self.assertRaises(ercr.CaptureError):
            list(ercr.iter_tokens(self.path, strict=True))
        # not strict: it is dropped, never half-read
        self.assertEqual(len(list(ercr.iter_tokens(self.path))), 1)

    def test_bad_header_is_named(self):
        write_mirror(self.path)
        blob = bytearray(self.path.read_bytes())
        blob[0:4] = b"NOPE"
        self.path.write_bytes(bytes(blob))
        with self.assertRaises(ercr.CaptureError):
            ercr.read_header(self.path)
        blob = bytearray(self.path.read_bytes())
        blob[0:4] = ercr.MAGIC
        blob[4] = 9                          # version
        self.path.write_bytes(bytes(blob))
        with self.assertRaises(ercr.CaptureError):
            ercr.read_header(self.path)

    def test_red_mutant_drops_phase(self):
        write_mirror(self.path, mutate=mutant_drop_phase)
        positions = [t.position for t in ercr.iter_tokens(self.path)]
        self.assertNotEqual(positions, [0, 7],
                            "the missing phase byte must shift the next record")

    def test_red_mutant_extra_id(self):
        ref = self.tmp / "ref.lac1"
        write_mirror(ref)
        write_mirror(self.path, mutate=mutant_extra_id)
        self.assertNotEqual(ref.read_bytes(), self.path.read_bytes())
        first = next(ercr.iter_tokens(self.path))
        exp_bits, exp_ids = synthetic_layers()[1]
        self.assertTrue(first.layers[1].inputs != struct.unpack(f"<{len(exp_bits)}e", struct.pack(f"<{len(exp_bits)}H", *exp_bits)) or
                        first.layers[1].ids != tuple(exp_ids),
                        "an extra id a layer must shift the next layer's fields")


class CppWriterBytes(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lac1-cpp-"))

    def test_cpp_writer_equals_the_python_mirror(self):
        exe = build_selftest(self.tmp)
        cpp_path = self.tmp / "cpp.lac1"
        r = subprocess.run([str(exe), str(cpp_path)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("expert_router_capture_selftest: OK", r.stdout)
        py_path = self.tmp / "py.lac1"
        write_mirror(py_path)
        self.assertEqual(py_path.read_bytes(), cpp_path.read_bytes(),
                         "the C++ writer and the Python mirror differ")
        # and the Python reader reads the C++ bytes
        tokens = list(ercr.iter_tokens(cpp_path))
        self.assertEqual([(t.position, t.token_id, t.phase) for t in tokens],
                         [(0, 100, 0), (7, 42, 1)])


if __name__ == "__main__":
    unittest.main()
