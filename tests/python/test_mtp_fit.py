#!/usr/bin/env python3
"""tools/mtp_fit.py and tools/mtp_capture_replay.py, device-free.

Campaign mtp-head-fit, experiment E1. The pure half (the capture format, the
task key, the split, the round-start simulation, the replay's HTTP flow against
a fake server) needs only the standard library. The numeric half (eval and
linear on a synthetic capture, the sampled probabilities, the GGUF decoders)
needs numpy and is skipped where numpy is missing; the recipe host has it.

Every cell was red first; the mutation, the cell it turned red and the
restoration are recorded here so a reader can repeat them:

 1. The format. The fixture tests/python/fixtures/mtp_capture_fixture.{h,z,tok}.bin
    is built here from the spec with struct and compared byte for byte; the same
    files are compared with the C++ writer's output by tests/test_mtp_capture.cpp.
 2. The split. Task keys of one task in several languages and wordings are one
    key; the split is disjoint by key, deterministic, independent of record
    order, and every class keeps a held-out task. Red mutation: the split keyed
    on the whole question (an answer-level split) -> `a task is in one split
    only` fails, the 16-language task lands in train and test.
 3. The round starts. Hand-computed rounds, kept-given-reached and tokens a
    verify. Red mutation: the next start `s + a + 1` -> `s + a` -> the hand
    rounds differ.
 4. The sampled probabilities against hand arithmetic (top-k, then temperature,
    then top-p). Red mutation: the temperature ignored -> the T=0.5 case differs.
 5. The red case for alignment: on a synthetic capture whose head rows predict
    the next token, eval has top-1 near 1; with --shift 1 it collapses to the
    chance level. Without that, a misaligned capture and a bad head look alike.
 6. The linear fit recovers a known map and lowers the held-out KL.
    Red mutation: the target row paired one position off in `gather` -> the
    fitted head's agreement with the target collapses.
"""
import contextlib
import importlib.util
import io
import json
import os
import struct
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import mtp_fit as mf  # noqa: E402
import mtp_capture_replay as rp  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

FIXTURES = REPO_ROOT / "tests" / "python" / "fixtures"


def fixture_bytes():
    """The C++ test's capture, from the spec: dim 4; chunks (lane 0, pos 0,
    toks 11 12 13), (lane 1, pos 0, toks 21 22), (lane 0, pos 3, toks 14 15);
    record k: h = 10k + d, z = -(10k + d) - 0.5."""
    chunks = [(0, 0, [11, 12, 13]), (1, 0, [21, 22]), (0, 3, [14, 15])]
    n = sum(len(c[2]) for c in chunks)
    h = bytearray(mf.render_header(1, 4, 0, n))
    z = bytearray(mf.render_header(2, 4, 0, n))
    t = bytearray(mf.render_header(3, 3, 1, n))
    k, req, nreq = 0, {}, 0
    for seq, pos0, toks in chunks:
        if pos0 == 0:
            req[seq] = nreq
            nreq += 1
        for i, tok in enumerate(toks):
            h += struct.pack("<4f", *[float(10 * k + d) for d in range(4)])
            z += struct.pack("<4f", *[-float(10 * k + d) - 0.5 for d in range(4)])
            t += struct.pack("<3i", req[seq], pos0 + i, tok)
            k += 1
    return bytes(h), bytes(z), bytes(t)


class FormatTests(unittest.TestCase):
    def test_header_round_trip_and_refusals(self):
        b = mf.render_header(2, 5120, 0, 92000)
        self.assertEqual(len(b), mf.HEADER_BYTES)
        self.assertEqual(mf.parse_header(b), {"kind": 2, "dim": 5120, "dtype": 0, "count": 92000})
        with self.assertRaises(ValueError):
            mf.parse_header(b"XRCMTPC1" + b[8:])
        with self.assertRaises(ValueError):
            mf.parse_header(b[:8] + struct.pack("<I", 2) + b[12:])      # a version this tool does not read
        with self.assertRaises(ValueError):
            mf.parse_header(b[:63])

    def test_fixture_is_the_specs_bytes(self):
        h, z, t = fixture_bytes()
        self.assertEqual((FIXTURES / "mtp_capture_fixture.h.bin").read_bytes(), h)
        self.assertEqual((FIXTURES / "mtp_capture_fixture.z.bin").read_bytes(), z)
        self.assertEqual((FIXTURES / "mtp_capture_fixture.tok.bin").read_bytes(), t)


class TaskAndSplitTests(unittest.TestCase):
    LANGS = ["Python", "C#", "TypeScript", "Lua", "Java", "C++", "Common Lisp", "SQL"]

    def test_task_key_removes_language_and_wording(self):
        a = mf.task_key("Write a function that base64-encodes bytes without a library in TypeScript, and explain the design briefly first.", "code")
        b = mf.task_key("Write a function that base64-encodes bytes without a library in C#. Only the code.", "code")
        c = mf.task_key("Write a function that base64-encodes bytes without a library in Common Lisp. Only the code.", "code")
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(a, "Write a function that base64-encodes bytes without a library")
        self.assertNotEqual(a, mf.task_key("Write a simple key-value store persisted to a file in Java. Only the code.", "code"))
        # a lower-case 'in' phrase is not a language; other classes keep their question whole
        self.assertEqual(mf.task_key("Explain in about 250 words why the seasons change.", "prose"),
                         "Explain in about 250 words why the seasons change.")
        self.assertEqual(mf.task_key("Reason it through in Python.", "think"), "Reason it through in Python.")

    def records(self):
        recs = []
        for t in range(10):
            for lang in self.LANGS:
                recs.append({"i": len(recs), "kind": "code", "cls": "code",
                             "task": "Write a function number %d in %s. Only the code." % (t, lang)})
        for t in range(9):
            recs.append({"i": len(recs), "kind": "prose", "cls": "prose", "task": "Explain in about 250 words topic %d." % t})
        for t in range(6):
            recs.append({"i": len(recs), "kind": "think", "cls": "think", "task": "Reason through case %d." % t})
        return recs

    def test_a_task_is_in_one_split_only(self):
        recs = mf.split_records(self.records())
        seen = {}
        for r in recs:
            seen.setdefault((r["kind"], r["task_key"]), set()).add(r["split"])
        for k, v in seen.items():
            self.assertEqual(len(v), 1, "task %r is in %s" % (k, sorted(v)))
        # all eight languages of a task together
        by = {}
        for r in recs:
            if r["kind"] == "code":
                by.setdefault(r["task_key"], []).append(r["split"])
        for k, v in by.items():
            self.assertEqual(len(v), len(self.LANGS))
            self.assertEqual(len(set(v)), 1)

    def test_every_class_keeps_a_held_out_task_and_all_three_splits(self):
        recs = mf.split_records(self.records())
        for kind in ("code", "prose", "think"):
            got = {r["split"] for r in recs if r["kind"] == kind}
            self.assertEqual(got, {"train", "val", "test"}, kind)

    def test_deterministic_and_order_independent(self):
        a = {r["i"]: r["split"] for r in mf.split_records(self.records())}
        shuffled = self.records()
        shuffled.reverse()
        b = {r["i"]: r["split"] for r in mf.split_records(shuffled)}
        self.assertEqual(a, b)
        c = {r["i"]: r["split"] for r in mf.split_records(self.records(), seed="another")}
        self.assertNotEqual(a, c)
        # the exact assignment is part of the data's identity: pinned
        keys = ["task %d" % i for i in range(10)]
        m = mf.assign_splits(keys)
        self.assertEqual(sorted(m.values()).count("test"), 2)
        self.assertEqual(sorted(m.values()).count("val"), 2)
        self.assertEqual(m, mf.assign_splits(list(reversed(keys))))

    def test_small_classes_do_not_lose_train(self):
        self.assertEqual(sorted(mf.assign_splits(["a", "b", "c"]).values()), ["test", "train", "val"])
        self.assertEqual(set(mf.assign_splits(["a", "b"]).values()), {"train"})


class RoundStartTests(unittest.TestCase):
    def test_hand_computed_rounds(self):
        match = [1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1]
        rounds = mf.simulate_rounds(match, 3)
        # start 0: two kept, the third refused -> next start 3; 3: one kept -> 5; 5: none -> 6;
        # 6: the width's three -> 10; 10: two kept, the text ends
        self.assertEqual(rounds, [(0, 2), (3, 1), (5, 0), (6, 3), (10, 2)])
        t = mf.chain_table(rounds, 3)
        self.assertEqual(t["rounds"], 5)
        self.assertEqual(t["reached"], [5, 4, 3])
        self.assertAlmostEqual(t["kept"][0], 4 / 5)
        self.assertAlmostEqual(t["kept"][1], 3 / 4)
        self.assertAlmostEqual(t["kept"][2], 1 / 3)
        self.assertAlmostEqual(t["tokens"], (3 + 2 + 1 + 4 + 3) / 5)

    def test_all_right_and_all_wrong(self):
        self.assertEqual(mf.simulate_rounds([1] * 10, 4), [(0, 4), (5, 4)])
        self.assertEqual(mf.chain_table(mf.simulate_rounds([1] * 10, 4), 4)["tokens"], 5.0)
        wrong = mf.simulate_rounds([0] * 7, 4)
        self.assertEqual([s for s, _ in wrong], list(range(7)))
        self.assertEqual(mf.chain_table(wrong, 4)["tokens"], 1.0)


@unittest.skipIf(np is None, "numpy missing")
class SampledProbabilityTests(unittest.TestCase):
    def test_top_k_then_temperature_then_top_p(self):
        lh = np.array([[3.0, 2.0, 1.0, 0.0]], dtype=np.float32)
        pick = np.array([1])
        # top-k 2 of [3, 2, 1, 0], T=1, no top-p: softmax([3, 2])
        pmax, pp = mf.sampled_stats(lh, 2, 1.0, 1.0, pick)
        e = 1.0 / (1.0 + np.exp(-1.0))
        self.assertAlmostEqual(float(pmax[0]), float(e), places=6)
        self.assertAlmostEqual(float(pp[0]), float(1 - e), places=6)
        # a pick outside the top-k has probability 0
        _, pp = mf.sampled_stats(lh, 2, 1.0, 1.0, np.array([2]))
        self.assertEqual(float(pp[0]), 0.0)
        # T = 0.5 sharpens: softmax([6, 4])
        pmax, _ = mf.sampled_stats(lh, 2, 0.5, 1.0, pick)
        self.assertAlmostEqual(float(pmax[0]), float(1 / (1 + np.exp(-2.0))), places=6)
        # top-p 0.5: the first token already holds 0.73, so it is the whole set
        pmax, pp = mf.sampled_stats(lh, 2, 1.0, 0.5, pick)
        self.assertAlmostEqual(float(pmax[0]), 1.0, places=9)
        self.assertEqual(float(pp[0]), 0.0)


def write_capture(base, h, z, req, pos, tok):
    n, d = h.shape
    with open(base + ".h", "wb") as f:
        f.write(mf.render_header(1, d, 0, n))
        f.write(h.astype("<f4").tobytes())
    with open(base + ".z", "wb") as f:
        f.write(mf.render_header(2, d, 0, n))
        f.write(z.astype("<f4").tobytes())
    with open(base + ".tok", "wb") as f:
        f.write(mf.render_header(3, 3, 1, n))
        f.write(np.stack([req, pos, tok], 1).astype("<i4").tobytes())


def synthetic(dirpath, d=16, noise_z=0.05, mix=None, seed=3, n_tasks=8, length=40, langs=("Lua", "Java")):
    """A capture in which h(p) = 3 W[x_{p+1}] + noise (the target says the next
    token) and z(p) = h(p) mixed by `mix` plus noise (the head says it too, in
    its own basis). Returns (base, records path, W npy, vocab json)."""
    rs = np.random.RandomState(seed)
    V = 40
    ids = sorted(rs.choice(100, size=V, replace=False).tolist())
    W = rs.randn(V, d).astype(np.float32)
    mix = np.eye(d, dtype=np.float32) if mix is None else mix
    rows_h, rows_z, req, pos, tok, recs = [], [], [], [], [], []
    r = 0
    specs = []
    for t in range(n_tasks):
        for lang in langs:
            specs.append(("code", "code", "Write function %d in %s. Only the code." % (t, lang)))
    for t in range(n_tasks):
        specs.append(("prose", "prose", "Explain topic %d." % t))
    for t in range(n_tasks):
        specs.append(("think", "think", "Reason about case %d." % t))
    for i, (kind, cls, q) in enumerate(specs):
        seq = rs.choice(V, size=length)                         # columns of W
        toks = [ids[c] for c in seq]
        h = 3.0 * W[np.roll(seq, -1)] + 0.1 * rs.randn(length, d).astype(np.float32)
        z = h @ mix + noise_z * rs.randn(length, d).astype(np.float32)
        rows_h.append(h)
        rows_z.append(z)
        req += [i] * length
        pos += list(range(length))
        tok += toks
        recs.append({"i": i, "corpus": i, "kind": kind, "cls": cls, "task": q, "tokens": length, "n_ids": length + 1,
                     "ctx": 0, "ids_sha1": rp.ids_sha1(toks + [0])})
    base = os.path.join(dirpath, "cap")
    write_capture(base, np.concatenate(rows_h), np.concatenate(rows_z), np.array(req), np.array(pos), np.array(tok))
    # the replay's ids carry one token the capture does not hold (the last prompt token);
    # here each request has `length` rows, so the stored ids are toks + [0]
    idx = os.path.join(dirpath, "records.idx")
    with open(idx, "w") as f:
        json.dump({"format": "mtp-capture-records-1", "complete": True, "records": recs}, f)
    wnpy = os.path.join(dirpath, "w.npy")
    np.save(wnpy, W)
    vj = os.path.join(dirpath, "vocab.json")
    with open(vj, "w") as f:
        json.dump(ids, f)
    return base, idx, wnpy, vj


def run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = mf.main(argv)
    return rc, out.getvalue()


@unittest.skipIf(np is None, "numpy missing")
class CaptureReaderTests(unittest.TestCase):
    def test_the_fixture_reads_back_aligned(self):
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "fx")
            for ext, name in (("h", "h"), ("z", "z"), ("tok", "tok")):
                (Path(d) / ("fx." + ext)).write_bytes((FIXTURES / ("mtp_capture_fixture.%s.bin" % name)).read_bytes())
            cap = mf.Capture(base)
            self.assertEqual((cap.n, cap.dim), (7, 4))
            self.assertEqual(cap.h[3].tolist(), [30.0, 31.0, 32.0, 33.0])
            self.assertEqual(cap.z[3].tolist(), [-30.5, -31.5, -32.5, -33.5])
            self.assertEqual(cap.req.tolist(), [0, 0, 0, 1, 1, 0, 0])
            self.assertEqual(cap.pos.tolist(), [0, 1, 2, 0, 1, 3, 4])
            self.assertEqual(cap.tok.tolist(), [11, 12, 13, 21, 22, 14, 15])

    def test_a_record_is_found_by_its_token_hash(self):
        with tempfile.TemporaryDirectory() as d:
            base, idx, _, _ = synthetic(d)
            cap = mf.Capture(base)
            recs = json.load(open(idx))["records"]
            # a stray request at the front would shift a counting reader; the hash does not care
            self.assertEqual(mf.resolve_rows(cap, recs), [])
            self.assertEqual([r["row0"] for r in recs[:3]], [0, 40, 80])
            recs[2]["ids_sha1"] = "0" * 40
            self.assertEqual(len(mf.resolve_rows(cap, recs)), 1)


@unittest.skipIf(np is None, "numpy missing")
class EvalTests(unittest.TestCase):
    def eval_json(self, d, shift=0, split="all"):
        base, idx, wnpy, vj = synthetic(d)
        out = os.path.join(d, "eval%d.json" % shift)
        rc, text = run(["eval", "--capture", base, "--records", idx, "--w-npy", wnpy, "--vocab", vj,
                        "--split", split, "--skip-first", "2", "--shift", str(shift), "--json", out])
        self.assertEqual(rc, 0)
        return json.load(open(out)), text

    def test_aligned_capture_has_high_top_one(self):
        with tempfile.TemporaryDirectory() as d:
            res, text = self.eval_json(d)
            for cls in ("code", "prose", "think"):
                c = res["classes"][cls]
                self.assertGreater(c["top1_actual"], 0.9, cls)
                self.assertGreater(c["top1_target"], 0.9, cls)
                self.assertGreater(c["target_top1_actual"], 0.9, cls)
                self.assertEqual(c["coverage"], 1.0)
                self.assertGreater(c["E_max_pT"], 0.0)
            self.assertNotIn("RED CASE", text)

    def test_red_case_shifted_rows_collapse(self):
        with tempfile.TemporaryDirectory() as d:
            res, text = self.eval_json(d, shift=1)
            self.assertIn("RED CASE", text)
            for cls in ("code", "prose", "think"):
                c = res["classes"][cls]
                self.assertLess(c["top1_actual"], 0.15, cls)         # chance is 1 in 40
                self.assertGreater(c["target_top1_actual"], 0.9, cls)  # the target's side is untouched

    def test_default_split_is_the_held_out_one(self):
        with tempfile.TemporaryDirectory() as d:
            base, idx, wnpy, vj = synthetic(d)
            rc, text = run(["eval", "--capture", base, "--records", idx, "--w-npy", wnpy, "--vocab", vj, "--skip-first", "2"])
            self.assertEqual(rc, 0)
            self.assertIn("split=test", text)
            self.assertIn("[code]", text)


@unittest.skipIf(np is None, "numpy missing")
class LinearTests(unittest.TestCase):
    def test_the_fit_recovers_a_known_map(self):
        rs = np.random.RandomState(9)
        d = 16
        # the head's basis is a rotation of the target's: the unfitted head is wrong, a linear map is right
        q, _ = np.linalg.qr(rs.randn(d, d))
        mix = q.astype(np.float32)
        with tempfile.TemporaryDirectory() as dd:
            base, idx, wnpy, vj = synthetic(dd, d=d, mix=mix, noise_z=0.05)
            out = os.path.join(dd, "lin.json")
            rc, text = run(["linear", "--capture", base, "--records", idx, "--w-npy", wnpy, "--vocab", vj,
                            "--skip-first", "2", "--sizes", "200,400", "--val-rows", "512", "--eval-rows", "512",
                            "--lambdas", "0.0001,0.01,1", "--json", out])
            self.assertEqual(rc, 0)
            rep = json.load(open(out))
            self.assertGreaterEqual(len(rep["curve"]), 2)
            for cls in ("code", "prose", "think"):
                base_te = rep["baseline"][cls]["test"]
                fit_te = rep["curve"][-1]["classes"][cls]["test"]
                self.assertLess(base_te["top1_target"], 0.3, cls)      # rotated: near chance
                self.assertGreater(fit_te["top1_target"], 0.9, cls)
                self.assertLess(fit_te["kl"], base_te["kl"] / 3, cls)
            self.assertIn("lambda", text)


@unittest.skipIf(np is None, "numpy missing")
class GgufDecodeTests(unittest.TestCase):
    def test_f16_f32_q8_0_q4_0_blocks(self):
        rs = np.random.RandomState(5)
        x = rs.randn(3, 64).astype(np.float32)
        raw = np.frombuffer(x.astype("<f4").tobytes(), np.uint8).reshape(3, 256)
        self.assertTrue(np.array_equal(mf.dequant_rows(raw, 0, 64), x))
        raw = np.frombuffer(x.astype("<f2").tobytes(), np.uint8).reshape(3, 128)
        self.assertTrue(np.allclose(mf.dequant_rows(raw, 1, 64), x, atol=2e-3))
        # Q8_0: f16 scale, 32 int8
        blocks = []
        for r in range(3):
            row = b""
            for b in range(2):
                v = x[r, 32 * b:32 * b + 32]
                dsc = np.float16(np.abs(v).max() / 127.0)
                q = np.round(v / np.float32(dsc)).astype(np.int8)
                row += dsc.tobytes() + q.tobytes()
            blocks.append(np.frombuffer(row, np.uint8))
        got = mf.dequant_rows(np.stack(blocks), 8, 64)
        self.assertLess(np.abs(got - x).max(), 0.02)
        # Q4_0: f16 scale, 16 bytes; element j low nibble, element j + 16 high nibble
        blocks = []
        for r in range(3):
            row = b""
            for b in range(2):
                v = x[r, 32 * b:32 * b + 32]
                dsc = np.float16(np.abs(v).max() / 8.0 + 1e-9)
                q = np.clip(np.round(v / np.float32(dsc)) + 8, 0, 15).astype(np.uint8)
                row += dsc.tobytes() + (q[:16] | (q[16:] << 4)).astype(np.uint8).tobytes()
            blocks.append(np.frombuffer(row, np.uint8))
        got = mf.dequant_rows(np.stack(blocks), 2, 64)
        self.assertLess(np.abs(got - x).max(), 0.5)

    def test_header_parse_and_row_read(self):
        # a minimal GGUF v3: one F32 tensor of 2 rows x 4 columns
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.gguf")
            def s(t): return struct.pack("<Q", len(t)) + t.encode()
            body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 1, 1)
            body += s("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", 32)
            body += s("output.weight") + struct.pack("<I", 2) + struct.pack("<QQ", 4, 2) + struct.pack("<I", 0) + struct.pack("<Q", 0)
            body += b"\0" * (-len(body) % 32)
            body += np.arange(8, dtype="<f4").tobytes()
            with open(path, "wb") as f:
                f.write(body)
            h = mf.read_gguf_header(path)
            self.assertEqual(h["tensors"]["output.weight"], ([4, 2], 0, 0))
            rows = mf.read_gguf_rows(path, "output.weight", [1, 0])
            self.assertEqual(rows.tolist(), [[4, 5, 6, 7], [0, 1, 2, 3]])


class FakeServer(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
        if self.path == "/tokenize":
            toks = [1 + (sum(w.encode()) % 50) for w in body["content"].split()]
            out = {"tokens": toks}
        elif self.path == "/v1/completions":
            FakeServer.seen.append(body)
            out = {"choices": [{"text": "x"}], "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": 1}}
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ReplayTests(unittest.TestCase):
    def corpus(self):
        return [
            {"kind": "code", "tokens": 5, "q": "Write f in Lua. Only the code.", "think": None, "text": "a b c d e", "calls": None},
            {"kind": "think", "tokens": 6, "q": "Reason it.", "think": "t1 t2 t3 t4 t5 t6", "text": "", "calls": None},
            {"kind": "tool", "tokens": 4, "q": "List.", "think": None, "text": None, "calls": [{"name": "ls"}]},
            {"kind": "prose", "tokens": 9, "q": "Explain.", "think": "p q", "text": "w x y", "calls": None},
        ]

    def test_segments_classes_and_the_dropped_tool(self):
        c = self.corpus()
        self.assertEqual(rp.segments(c[0]), [("code", "a b c d e")])
        self.assertEqual(rp.segments(c[1]), [("think", "t1 t2 t3 t4 t5 t6")])       # empty answer text: no answer segment
        self.assertEqual(rp.segments(c[2]), [])
        self.assertEqual(rp.segments(c[3]), [("think", "p q"), ("prose", "w x y")])  # thinking is its own class

    def test_ids_hash_convention(self):
        ids = [5, 6, 7]
        self.assertEqual(rp.ids_sha1(ids), rp.hashlib.sha1(struct.pack("<2i", 5, 6)).hexdigest())

    def test_end_to_end_against_a_fake_server(self):
        srv = HTTPServer(("127.0.0.1", 0), FakeServer)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        FakeServer.seen.clear()
        try:
            with tempfile.TemporaryDirectory() as d:
                cp = os.path.join(d, "c.jsonl")
                with open(cp, "w") as f:
                    for r in self.corpus():
                        f.write(json.dumps(r) + "\n")
                out = os.path.join(d, "o.idx")
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = rp.main([str(srv.server_address[1]), cp, out])
                self.assertEqual(rc, 0)
                idx = json.load(open(out))
                self.assertTrue(idx["complete"])
                self.assertEqual([(r["kind"], r["cls"], r["n_ids"]) for r in idx["records"]],
                                 [("code", "code", 5), ("think", "think", 6), ("prose", "think", 2), ("prose", "prose", 3)])
                self.assertEqual(len(FakeServer.seen), 4)
                self.assertTrue(all(b["max_tokens"] == 1 and isinstance(b["prompt"], list) for b in FakeServer.seen))
                # code 5 = 5 and think 6 = 6 match; the prose record re-tokenises to 2 + 3, not its 9
                self.assertIn("1 of 3 records", buf.getvalue())
        finally:
            srv.shutdown()
            srv.server_close()

    def test_a_dead_server_leaves_an_incomplete_index(self):
        with tempfile.TemporaryDirectory() as d:
            cp = os.path.join(d, "c.jsonl")
            with open(cp, "w") as f:
                f.write(json.dumps(self.corpus()[0]) + "\n")
            out = os.path.join(d, "o.idx")
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                rc = rp.main(["1", cp, out, "--timeout", "2"])      # nothing listens on port 1
            self.assertEqual(rc, 1)
            self.assertFalse(json.load(open(out))["complete"])



@unittest.skipIf(np is None, "numpy missing")
class TestDistill(unittest.TestCase):
    def setUp(self):
        self.np = np
        rng = np.random.RandomState(3)
        self.W = rng.randn(64, 16).astype(np.float32)
        Hall = rng.randn(1200, 16).astype(np.float32) * 2
        self.R = np.linalg.qr(rng.randn(16, 16))[0].astype(np.float32)
        Z = Hall @ self.R          # the head's rows are the target's, turned
        self.Z, self.H, self.Zv, self.Hv = Z[:800], Hall[:800], Z[800:], Hall[800:]

    def _top(self, A):
        st = mf.row_stats(self.Zv @ A, self.Hv, self.W)
        return float((st["dz"] == st["dh"]).mean())

    def test_the_fit_recovers_a_turned_head(self):
        np = self.np
        base = self._top(np.eye(16, dtype=np.float32))
        A, top, step = mf.fit_distill(self.Z, self.H, self.W, self.Zv, self.Hv, 300, 256, 3e-3, 1.0, 50)
        self.assertLess(base, 0.3)
        self.assertGreater(top, 0.8)
        self.assertGreater(step, 0)
        self.assertAlmostEqual(top, self._top(A), places=6)

    def test_the_identity_stays_when_the_head_is_right(self):
        np = self.np
        A, top, step = mf.fit_distill(self.H, self.H, self.W, self.Hv, self.Hv, 50, 256, 3e-3, 1.0, 10)
        self.assertEqual(step, 0)
        self.assertTrue((A == np.eye(16, dtype=np.float32)).all())
        self.assertEqual(top, 1.0)

    def test_the_gradient_is_the_loss_slope(self):
        np = self.np
        A = np.eye(16, dtype=np.float64) + 0.01
        Z, H, W = self.Z[:32].astype(np.float64), self.H[:32].astype(np.float64), self.W.astype(np.float64)
        l0, g = mf.distill_grad(Z, H, W, A, 1.0)
        e = np.zeros_like(A)
        e[3, 5] = 1e-5
        l1, _ = mf.distill_grad(Z, H, W, A + e, 1.0)
        self.assertAlmostEqual((l1 - l0) / 1e-5, g[3, 5], places=3)

if __name__ == "__main__":
    unittest.main()
