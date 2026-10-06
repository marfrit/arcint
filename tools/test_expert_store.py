#!/usr/bin/env python3
"""Unit ladder for tools/q4e/expert_store.py -- the artifact-format step's
writer (device-free; needs numpy only, no card, no GGUF, no ext4).

Red-first: every cell below fails against a plausible wrong implementation
before it passes against the one in the tool. The mutants are named in each
docstring; the raw mutant evidence is in the store's build packet.

  * the constant is DERIVED from the geometry, so 2,457,600 is not recited:
    a wrong hidden/ff, a wrong bytes-per-weight, or a record that includes
    scales/zp fails the sum.
  * `device_order_weight_record` is gate|up|down; a swapped role order fails
    because the three test roles carry distinct byte patterns.
  * the u4 nibble contract is the C++ one (`src/core/gguf_repack.cpp:225`),
    read back with a transcription of that code, not the packer's inverse.
  * `transpose_scale_zp_to_device` is checked against a brute-force
    transcription of `maybe_transpose_scale_zp`'s mapping
    `src[o*group_count+g] -> dst[g*oc+o]` on a NON-SQUARE shape, where a
    wrong formula is an O(1) error rather than a coincidence.
  * `write_expert_file` MUST call posix_fallocate (a truncate/write leaves a
    hole; ext4 then reports no/wrong extents) and MUST refuse a short payload.
  * the byte-exactness cell packs a real quantisation, writes it, reads the
    FILE back, unpacks with the C++ transcription, dequantises with the IR's
    own chain and asserts the reconstruction is within half a group step --
    and that the bound is TIGHT, because a bound nothing approaches passes
    vacuously.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "tools"))

from q4e import expert_fill as ef          # noqa: E402
from q4e import expert_store as es          # noqa: E402


class TestGeometryAndOrder(unittest.TestCase):
    def test_slice_constant_derives_from_geometry(self):
        # 3 * 2560 * 640 * 0.5, and the roles sum to it. Mutant: hard-coding a
        # number, or including scales/zp, fails one of the two assertions.
        self.assertEqual(es.EXPERT_SLICE_BYTES, 2_457_600)
        self.assertEqual(es.EXPERT_SLICE_BYTES, es.slice_byte_count())
        self.assertEqual(es.role_byte_count("gate"), 640 * 2560 // 2)
        self.assertEqual(es.role_byte_count("up"), 640 * 2560 // 2)
        self.assertEqual(es.role_byte_count("down"), 2560 * 640 // 2)
        self.assertEqual(es.EXPERT_SLICE_BYTES % 4096, 0)   # 600 pages
        self.assertEqual(es.EXPERT_SLICE_BYTES % 512, 0)    # 4800 LBAs

    def test_record_is_gate_up_down_in_that_order(self):
        # Distinct per-role patterns; a swapped order is a different record.
        g = np.full(es.role_byte_count("gate"), 0x11, np.uint8)
        u = np.full(es.role_byte_count("up"), 0x22, np.uint8)
        d = np.full(es.role_byte_count("down"), 0x33, np.uint8)
        rec = es.device_order_weight_record({"gate": g, "up": u, "down": d})
        self.assertEqual(rec.size, es.EXPERT_SLICE_BYTES)
        n_g, n_u, n_d = (es.role_byte_count(r) for r in ("gate", "up", "down"))
        self.assertTrue((rec[:n_g] == 0x11).all())
        self.assertTrue((rec[n_g:n_g + n_u] == 0x22).all())
        self.assertTrue((rec[n_g + n_u:] == 0x33).all())
        # Mutant: up/down swapped.
        bad = es.device_order_weight_record({"gate": g, "up": d, "down": u})
        self.assertFalse(np.array_equal(rec, bad))

    def test_record_refuses_wrong_role_size(self):
        bad = {"gate": np.zeros(es.role_byte_count("gate") + 1, np.uint8),
               "up": np.zeros(es.role_byte_count("up"), np.uint8),
               "down": np.zeros(es.role_byte_count("down"), np.uint8)}
        with self.assertRaises(ValueError):
            es.device_order_weight_record(bad)

    def test_u4_nibble_contract_roundtrips(self):
        # pack_u4 then the C++ unpack (expert_fill.unpack_u4 is a transcription
        # of gguf_repack.cpp:225, NOT the packer's inverse).
        rng = np.random.default_rng(7)
        q = rng.integers(0, 16, size=2048).astype(np.uint8)
        packed = ef.pack_u4(q)
        self.assertEqual(ef.unpack_u4(packed, q.size).tolist(), q.tolist())
        # Mutant: high/low nibble swapped -- element 0 becomes q[1].
        swapped = (q[0::2] << 4) | q[1::2]
        self.assertNotEqual(ef.unpack_u4(swapped, q.size).tolist(), q.tolist())


class TestDeviceOrderTranspose(unittest.TestCase):
    def test_scale_transpose_matches_moe_otd_mapping(self):
        # Non-square oc != group_count: maybe_transpose_scale_zp maps
        # src[o*gc+g] -> dst[g*oc+o].
        oc, gc = 640, 20
        src = np.arange(oc * gc, dtype=np.uint16).reshape(oc, gc)
        got = es.transpose_scale_to_device(src)
        want = np.empty(oc * gc, dtype=np.uint16)
        for o in range(oc):
            for g in range(gc):
                want[g * oc + o] = src[o, g]
        self.assertEqual(got.tolist(), want.tolist())
        # Mutant: identity (forgetting the transpose) differs.
        self.assertFalse(np.array_equal(got, src.ravel()))

    def test_zp_transpose_unpacks_to_transposed_codes(self):
        oc, gc = 2560, 5
        codes = np.arange(oc * gc, dtype=np.uint8).reshape(oc, gc) % 16
        want_dev = np.ascontiguousarray(np.transpose(codes, (1, 0))).reshape(gc * oc)
        # file order: [oc][group], even linear index low nibble
        packed = ef.pack_u4(codes.ravel())
        # rebuild file codes first, then transpose as the tool does
        self.assertEqual(ef.unpack_u4(packed, oc * gc).tolist(), codes.ravel().tolist())
        dev_bytes = es.transpose_zp_to_device(codes)
        self.assertEqual(ef.unpack_u4(dev_bytes, oc * gc).tolist(),
                         want_dev.tolist())
        # Mutant: file-order bytes (no transpose) differ.
        self.assertNotEqual(ef.unpack_u4(dev_bytes, oc * gc).tolist(),
                            ef.unpack_u4(packed, oc * gc).tolist())


class TestFileGeometry(unittest.TestCase):
    def test_fallocate_is_called_and_size_is_exact(self):
        d = tempfile.mkdtemp(prefix="expert-store-")
        try:
            path = os.path.join(d, "e.bin")
            payload = np.zeros(es.EXPERT_SLICE_BYTES, np.uint8)
            calls = []
            real = os.posix_fallocate

            def spy(fd, off, length):
                calls.append((off, length))
                return real(fd, off, length)

            os.posix_fallocate = spy
            try:
                es.write_expert_file(path, payload)
            finally:
                os.posix_fallocate = real
            self.assertEqual(calls, [(0, es.EXPERT_SLICE_BYTES)])
            self.assertEqual(os.stat(path).st_size, es.EXPERT_SLICE_BYTES)
            # fallocate allocates real blocks; a truncate leaves a hole. The
            # assertion is a geometry consequence and only meaningful on a
            # filesystem that reports fallocated blocks (ext4); tmpfs/overlay
            # may not, so skip there rather than pass vacuously.
            if os.stat(path).st_blocks * 512 < es.EXPERT_SLICE_BYTES:
                self.skipTest("filesystem does not report fallocated blocks "
                              "(not ext4?) -- the real-store FIEMAP is the proof")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_short_payload_is_refused(self):
        d = tempfile.mkdtemp(prefix="expert-store-")
        try:
            path = os.path.join(d, "short.bin")
            with self.assertRaises(ValueError):
                es.write_expert_file(path, np.zeros(10, np.uint8),
                                     size=es.EXPERT_SLICE_BYTES)
            # a payload larger than the declared slice also refuses
            with self.assertRaises(ValueError):
                es.write_expert_file(path, np.zeros(es.EXPERT_SLICE_BYTES + 1,
                                                    np.uint8),
                                     size=es.EXPERT_SLICE_BYTES)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_expert_file_name_is_flat_and_collision_free(self):
        # The arcwell reference tool builds `expert_%04d.bin`.
        self.assertEqual(es.expert_file_name(0), "expert_0000.bin")
        self.assertEqual(es.expert_file_name(3407), "expert_3407.bin")
        self.assertNotEqual(es.expert_file_name(1), es.expert_file_name(10))


class _FakeTensor:
    def __init__(self, arr, name="F32"):
        self.data = arr
        self.tensor_type = type("T", (), {"name": name})()


class _FakeFeed:
    """Minimal `GgufFeed` shape: `_index` of F32 reader tensors."""

    def __init__(self, layer, experts):
        e = len(experts)
        self._index = {
            f"blk.{layer}.ffn_gate_exps.weight":
                _FakeTensor(np.zeros((e, es.ROLE_OUT["gate"],
                                      es.ROLE_IN["gate"]), np.float32)),
            f"blk.{layer}.ffn_up_exps.weight":
                _FakeTensor(np.zeros((e, es.ROLE_OUT["up"],
                                      es.ROLE_IN["up"]), np.float32)),
            f"blk.{layer}.ffn_down_exps.weight":
                _FakeTensor(np.zeros((e, es.ROLE_OUT["down"],
                                      es.ROLE_IN["down"]), np.float32)),
        }


class TestBf16Source(unittest.TestCase):
    """A BF16 expert tensor arrives from gguf-py as raw bytes (uint8, twice
    the row width). Mutant: the pre-2026-09-28 plain cast, which returned the
    bytes as values 0..255 at twice the width."""

    def test_bf16_experts_dequantise_to_their_values(self):
        try:
            from gguf import GGMLQuantizationType
        except ImportError:
            self.skipTest("gguf-py absent")
        want = np.array([[[1.5, -2.0, 0.25, 3.0]], [[-0.5, 8.0, 0.0, -1.0]]],
                        np.float32)                              # [E, out, in]
        raw = (want.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)
        t = _FakeTensor(raw)
        t.tensor_type = GGMLQuantizationType.BF16
        feed = type("F", (), {"_index": {"w": t}})()
        w = es.ExpertStoreWriter(feed, tempfile.gettempdir(), {})
        got = w._dequant_experts("w", [1])
        np.testing.assert_array_equal(got, want[[1]])


class TestWriterLevel(unittest.TestCase):
    """ONE expert through the whole writer: record offsets + sidecar size.

    Mutant: a float-scale sidecar (without the f16 conversion) is 172,800 B,
    not 96,000 B. (The role-order mutant is caught by
    `test_record_is_gate_up_down_in_that_order`; at real geometry all three
    roles are the same size, so this writer-level cell cannot detect a swap.)
    """

    def test_record_offsets_and_sidecar_size(self):
        d = tempfile.mkdtemp(prefix="expert-store-")
        try:
            writer = es.ExpertStoreWriter(
                _FakeFeed(0, [0]), d, {0: [0]}, with_scale_zp=True,
                provenance={"seed": 0xF2A17C0DE5EED})
            m = writer.write()
            self.assertEqual(len(m), 1)
            self.assertEqual(m[0]["file"], "expert_0000.bin")
            rec = np.fromfile(os.path.join(d, "expert_0000.bin"), np.uint8)
            self.assertEqual(rec.size, es.EXPERT_SLICE_BYTES)
            n_g = es.role_byte_count("gate")
            n_u = es.role_byte_count("up")
            n_d = es.role_byte_count("down")
            self.assertEqual((n_g, n_u, n_d), (819200, 819200, 819200))
            # offsets are the contract, not just the order
            self.assertEqual(n_g + n_u + n_d, es.EXPERT_SLICE_BYTES)
            side = os.path.join(d, "expert_0000.scaleszp")
            self.assertEqual(os.stat(side).st_size, 96_000)
            with open(os.path.join(d, "manifest.json")) as f:
                man = json.load(f)
            self.assertEqual(man["provenance"]["seed"], 0xF2A17C0DE5EED)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_sidecar_scales_are_f16_bits(self):
        # A float scale array must be rounded to f16 before the transpose; a
        # float pass-through gives 4 B/element and a wrong length.
        oc, gc = 640, 20
        s = (np.arange(oc * gc, dtype=np.float32) * 1e-4).reshape(oc, gc)
        z = (np.arange(oc * gc, dtype=np.uint8) % 16).reshape(oc, gc)
        s16, zp = es.transpose_scale_zp_to_device(s, z)
        self.assertEqual(s16.dtype, np.uint16)
        self.assertEqual(s16.size, oc * gc)
        want = np.ascontiguousarray(
            np.transpose(s.astype(np.float16).view(np.uint16), (1, 0))).reshape(gc * oc)
        self.assertTrue(np.array_equal(s16, want))
        self.assertEqual(zp.size, oc * gc // 2)
        # A raw float array is refused, not silently re-emitted at 4 B/element.
        with self.assertRaises(ValueError):
            es.transpose_scale_to_device(s)


class TestPinnedSetInputs(unittest.TestCase):
    def test_splitmix64_matches_patch_0018_transcription(self):
        # The C++ splitmix64 constants; a wrong constant produces a wrong set.
        self.assertEqual(es.splitmix64(0), 0xE220A8397B1DCDAF)
        self.assertEqual(es.splitmix64(1), 0x910A2DEC89025CC1)

    def test_static_partition_is_a_pure_function(self):
        a = es.static_partition_resident_experts(0xF2A17C0DE5EED, 284636629, 512, 71)
        b = es.static_partition_resident_experts(0xF2A17C0DE5EED, 284636629, 512, 71)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 71)
        self.assertEqual(a, sorted(a))
        other = es.static_partition_resident_experts(0xF2A17C0DE5EED, 284636630, 512, 71)
        self.assertNotEqual(a, other)

    def test_seed_file_needs_layer_keys(self):
        d = tempfile.mkdtemp(prefix="expert-store-")
        try:
            seed = os.path.join(d, "seed.txt")
            with open(seed, "w", encoding="utf-8") as f:
                f.write("# hot-set seed v2 (frequency rank, id tie-break)\n"
                        "# space=layer_key\n"
                        "100 1 2 3\n"
                        "200 4 5 6\n")
            with self.assertRaises(ValueError):
                es.load_pinned(seed)
            lk = os.path.join(d, "lk.json")
            with open(lk, "w") as f:
                json.dump({"0": 100, "1": 200}, f)
            pinned = es.load_pinned(seed, lk)
            self.assertEqual(pinned, {0: [1, 2, 3], 1: [4, 5, 6]})
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestByteExactness(unittest.TestCase):
    def test_file_roundtrip_dequantises_within_half_a_step(self):
        """The file's bytes, read back and unpacked by the C++ transcription,
        dequantise to the source expert slice within the u4 half-step."""
        rng = np.random.default_rng(11)
        gs = es.DEFAULT_GROUP_SIZE
        # small real-shaped roles: down is [out=8, inn=gs*2]
        src = {}
        for role in es.ROLE_ORDER:
            out = 8
            inn = 2 * gs
            src[role] = (rng.standard_normal((out, inn)) * 0.1).astype(np.float32)

        q, zp, scale = {}, {}, {}
        packed = {}
        for role, x in src.items():
            qq, zz, ss = ef.quantise_group_affine(x, gs)
            q[role], zp[role], scale[role] = qq, zz, ss
            packed[role] = ef.pack_u4(qq)

        # Build a record the same size as a real expert? Here we verify the
        # per-role bytes round-trip, which is the property the writer relies on.
        d = tempfile.mkdtemp(prefix="expert-store-")
        try:
            for role in es.ROLE_ORDER:
                p = os.path.join(d, f"{role}.bin")
                payload = np.ascontiguousarray(packed[role], np.uint8).ravel()
                es.write_expert_file(p, payload)
                raw = np.fromfile(p, dtype=np.uint8)
                codes = ef.unpack_u4(raw, q[role].size).reshape(q[role].shape)
                self.assertTrue(np.array_equal(codes, q[role]))
                rec = ef.dequantise_affine(codes, zp[role], scale[role])
                srcg = src[role].reshape(src[role].shape[0], -1, gs)
                bound = ef.quantisation_step_bound(scale[role])[:, :, 0]
                err = np.abs(rec - srcg).max(-1)
                # within half a step, groupwise
                self.assertTrue((err <= bound + 1e-6).all())
                # AND tight: at least one group approaches the bound, so the
                # assertion is not vacuous.
                self.assertGreater(err.max(), bound.max() * 0.5)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
