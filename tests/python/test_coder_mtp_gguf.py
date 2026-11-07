#!/usr/bin/env python3
"""tools/coder_mtp_gguf.py, device-free, on synthetic GGUFs and safetensors a few kilobytes large.

Campaign mtp-head-fit, experiment E2. The tool writes the head files whose
arms the card window compares; a tensor it cannot place must fail loudly,
because a head that loads with a wrong tensor is a wrong arm, not an error.
Red cases (each was run red first; the mutation that turned it red is named):

 1. A tensor of the head's layer with a name the spec does not know
    (attn_q -> attn_qq): build refuses. Mutation: the extra-name test dropped
    from build_from_gguf -> the file is written without it and `missing` is
    the only complaint.
 2. A wrong shape (attn_k with one column more): build and check refuse.
    Mutation: the dims comparison removed -> the file is written.
 3. A missing tensor, a trunk tensor in a head file, an F32 matrix where the
    quantizer wants F16: refused.
 4. experts-used outside 1..expert_count: refused; inside: the metadata moves
    and nothing else does (the tokenizer array is kept byte for byte).
 5. The safetensors path: the norms are the base's plus one, gate_up_proj is
    split gate first, up second, the router and the shared gate are F32, an
    unknown mtp.* tensor and a wrong shape are refused. Mutation: up and gate
    swapped in ST split -> the split test fails.
 6. splice replaces a tensor raw and refuses a shape that does not fit;
    compare finds one flipped byte.
 7. strip-output (patch 0105): the file without output.weight keeps every
    other tensor byte for byte and the metadata; check accepts it and
    --no-output requires it; a file that still has output.weight fails
    --no-output (mutation: strip_output writes the tensor back -> the
    names test fails); a file with no output to strip and a file that is not
    head-only are refused. Mutation: the `not in g.names` guard removed -> a
    no-op "strip" is written and the arm is not what it is called.
"""
import importlib.util
import os
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("cm", os.path.join(HERE, "..", "..", "tools", "coder_mtp_gguf.py"))
cm = importlib.util.module_from_spec(spec)
sys.modules["cm"] = cm
spec.loader.exec_module(cm)
try:
    import numpy as np
except ImportError:
    np = None

N, HD, NH, NKV, FF, FFS, NE, V, BLOCKS = 8, 4, 2, 1, 4, 4, 6, 10, 3
HEAD = BLOCKS - 1


def kv_u32(k, v): return [k, cm.T_U32, struct.pack("<I", v)]
def kv_str(k, v): return [k, cm.T_STR, cm._wstr(v)]


def kvs(experts=NE, used=3):
    toks = b"".join(cm._wstr("t%d" % i) for i in range(V))
    return [kv_str("general.architecture", "qwen35moe"),
            kv_u32("qwen35moe.block_count", BLOCKS), kv_u32("qwen35moe.nextn_predict_layers", 1),
            kv_u32("qwen35moe.embedding_length", N), kv_u32("qwen35moe.attention.key_length", HD),
            kv_u32("qwen35moe.attention.value_length", HD), kv_u32("qwen35moe.attention.head_count", NH),
            kv_u32("qwen35moe.attention.head_count_kv", NKV), kv_u32("qwen35moe.expert_count", experts),
            kv_u32("qwen35moe.expert_used_count", used), kv_u32("qwen35moe.expert_feed_forward_length", FF),
            kv_u32("qwen35moe.expert_shared_feed_forward_length", FFS),
            ["tokenizer.ggml.tokens", cm.T_ARR, struct.pack("<IQ", cm.T_STR, V) + toks]]


def tensor(name, dims, ty, fill):
    n = 1
    for d in dims:
        n *= d
    raw = (struct.pack("<f", fill) * n) if ty == cm.F32 else (struct.pack("<e", fill) * n)
    return cm.Tensor(name, dims, ty, lambda r=raw: iter([r]))


def head_tensors(prefix, dims_override=None, rename=None, skip=(), extra=()):
    g = type("G", (), {"scalar": lambda self, k, d=None: dict(
        (e[0], struct.unpack("<I", e[2])[0]) for e in kvs() if e[1] == cm.T_U32)[k]})()
    sp = cm.head_spec(g, NE)
    out = []
    for i, (name, (dims, kind)) in enumerate(sp.items()):
        if name in skip:
            continue
        dims = (dims_override or {}).get(name, dims)
        if name == prefix + "ffn_gate_inp_shexp.weight":
            dims = dims + [1]               # the F16 file's [n, 1]
        out.append(tensor((rename or {}).get(name, name), dims, cm.F32 if kind == "f32" else cm.F16, 0.25 * (i + 1)))
    for n in extra:
        out.append(tensor(n, [N], cm.F32, 1.0))
    return out


def full_model(path, **kw):
    ts = [tensor("token_embd.weight", [N, V], cm.F16, 0.5), tensor("output.weight", [N, V], cm.F16, 0.75),
          tensor("output_norm.weight", [N], cm.F32, 1.0), tensor("blk.0.attn_norm.weight", [N], cm.F32, 1.0)]
    ts += head_tensors("blk.%d." % HEAD, **kw)
    cm.write_gguf(path, kvs(), ts)


class GgufTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.p = lambda n: os.path.join(self.d, n)

    def build(self, experts_used=None, **kw):
        full_model(self.p("src.gguf"), **kw)
        cm.build_from_gguf(self.p("src.gguf"), self.p("head.gguf"), experts_used)

    def test_build_is_head_only_and_exact(self):
        self.build()
        g = cm.check_file(self.p("head.gguf"), NE)
        self.assertNotIn("blk.0.attn_norm.weight", g.names)
        self.assertNotIn("token_embd.weight", g.names)       # the embedding comes by splice
        self.assertEqual(g.info("blk.2.ffn_gate_inp_shexp.weight")[1], [N])
        src = cm.Gguf(self.p("src.gguf"))
        for n in g.names:
            self.assertEqual(g.tensor(n).data(), src.tensor(n).data(), n)
        self.assertEqual([e[2] for e in g.kv if e[0] == "tokenizer.ggml.tokens"],
                         [e[2] for e in src.kv if e[0] == "tokenizer.ggml.tokens"])

    def test_unknown_name_is_refused(self):
        full_model(self.p("src.gguf"), rename={"blk.2.attn_q.weight": "blk.2.attn_qq.weight"})
        with self.assertRaisesRegex(cm.SpecError, "cannot be placed"):
            cm.build_from_gguf(self.p("src.gguf"), self.p("head.gguf"))
        self.assertFalse(os.path.exists(self.p("head.gguf")))

    def test_extra_tensor_in_head_layer_is_refused(self):
        full_model(self.p("src.gguf"), extra=["blk.2.nextn.embed_tokens.weight"])
        with self.assertRaisesRegex(cm.SpecError, "cannot be placed"):
            cm.build_from_gguf(self.p("src.gguf"), self.p("head.gguf"))

    def test_wrong_shape_is_refused(self):
        full_model(self.p("src.gguf"), dims_override={"blk.2.attn_k.weight": [N, NKV * HD + 1]})
        with self.assertRaisesRegex(cm.SpecError, "attn_k.weight has shape"):
            cm.build_from_gguf(self.p("src.gguf"), self.p("head.gguf"))

    def test_missing_tensor_is_refused(self):
        full_model(self.p("src.gguf"), skip=["blk.2.nextn.eh_proj.weight"])
        with self.assertRaisesRegex(cm.SpecError, "eh_proj.weight is missing"):
            cm.build_from_gguf(self.p("src.gguf"), self.p("head.gguf"))

    def test_trunk_tensor_in_a_head_file_is_refused(self):
        self.build()
        trunk = cm.Gguf(self.p("src.gguf")).tensor("blk.0.attn_norm.weight")
        h = cm.Gguf(self.p("head.gguf"))
        cm.write_gguf(self.p("bad.gguf"), h.kv, [h.tensor(n) for n in h.names] + [trunk])
        with self.assertRaisesRegex(cm.SpecError, "trunk tensor"):
            cm.check_file(self.p("bad.gguf"))

    def test_f32_matrix_is_refused_for_the_quantizer(self):
        self.build()
        h = cm.Gguf(self.p("head.gguf"))
        q = h.tensor("blk.2.attn_q.weight")
        f32 = cm.Tensor(q.name, q.dims, cm.F32, lambda: iter([b"\0" * (4 * N * 2 * NH * HD)]))
        cm.write_gguf(self.p("bad.gguf"), h.kv, [f32 if n == q.name else h.tensor(n) for n in h.names])
        with self.assertRaisesRegex(cm.SpecError, "quantizer's input is F16"):
            cm.check_file(self.p("bad.gguf"), floats_only=True)

    def test_experts_used(self):
        self.build(experts_used=2)
        g = cm.Gguf(self.p("head.gguf"))
        self.assertEqual(g.scalar("qwen35moe.expert_used_count"), 2)
        self.assertEqual(g.scalar("qwen35moe.expert_count"), NE)
        for bad in (0, NE + 1):
            with self.assertRaisesRegex(cm.SpecError, "outside"):
                cm.build_from_gguf(self.p("src.gguf"), self.p("x.gguf"), bad)

    def test_splice_and_compare(self):
        self.build()
        full_model(self.p("served.gguf"))
        cm.splice(self.p("head.gguf"), self.p("served.gguf"), ["token_embd.weight", "blk.2.attn_q.weight"], self.p("arm.gguf"))
        a = cm.Gguf(self.p("arm.gguf"))
        self.assertIn("token_embd.weight", a.names)
        self.assertEqual(a.tensor("token_embd.weight").data(), cm.Gguf(self.p("served.gguf")).tensor("token_embd.weight").data())
        lines = []
        same, n = cm.compare(self.p("arm.gguf"), self.p("served.gguf"), "blk.2.", False, lines.append)
        self.assertEqual((same, n), (len(cm.head_spec(a, NE)), len(cm.head_spec(a, NE))))
        # one flipped byte is found
        with open(self.p("arm.gguf"), "rb") as f:
            raw = bytearray(f.read())
        i = raw.index(struct.pack("<e", 0.25 * 3) * 4)
        raw[i] ^= 1
        with open(self.p("flip.gguf"), "wb") as f:
            f.write(raw)
        same, n = cm.compare(self.p("flip.gguf"), self.p("served.gguf"), "blk.2.", False, lines.append)
        self.assertEqual(n - same, 1)

    def test_strip_output(self):
        self.build()
        full_model(self.p("served.gguf"))
        cm.splice(self.p("head.gguf"), self.p("served.gguf"), ["token_embd.weight", "output.weight"], self.p("arm.gguf"))
        with self.assertRaisesRegex(cm.SpecError, "carries output.weight"):     # the red: the with-output arm
            cm.check_file(self.p("arm.gguf"), NE, no_output=True)
        cm.check_file(self.p("arm.gguf"), NE)
        cm.strip_output(self.p("arm.gguf"), self.p("no.gguf"))
        g = cm.check_file(self.p("no.gguf"), NE, no_output=True)
        a = cm.Gguf(self.p("arm.gguf"))
        self.assertEqual(g.names, [n for n in a.names if n != "output.weight"])
        for n in g.names:
            self.assertEqual(g.tensor(n).data(), a.tensor(n).data(), n)
        self.assertEqual(g.kv, a.kv)
        cm.check_file(self.p("head.gguf"), NE, no_output=True)      # a built head has none either

    def test_strip_output_refusals(self):
        self.build()
        with self.assertRaisesRegex(cm.SpecError, "no output.weight to strip"):
            cm.strip_output(self.p("head.gguf"), self.p("x.gguf"))
        full_model(self.p("served.gguf"))
        with self.assertRaisesRegex(cm.SpecError, "trunk tensor"):
            cm.strip_output(self.p("served.gguf"), self.p("x.gguf"))
        self.assertFalse(os.path.exists(self.p("x.gguf")))

    def test_splice_refuses_a_shape_that_does_not_fit(self):
        self.build()
        h = cm.Gguf(self.p("head.gguf"))
        cm.write_gguf(self.p("donor.gguf"), h.kv, [tensor("blk.2.attn_q.weight", [N, 3], cm.F16, 1.0)])
        with self.assertRaisesRegex(cm.SpecError, "shape"):
            cm.splice(self.p("head.gguf"), self.p("donor.gguf"), ["blk.2.attn_q.weight"], self.p("arm.gguf"))

    def test_quantize_refuses_a_matrix_without_a_rule(self):
        self.build()
        with open(self.p("r.ttf"), "w") as f:
            f.write("^blk\\.2\\.attn_k\\.weight$=Q4_K\n")
        with self.assertRaisesRegex(cm.SpecError, "has no rule"):
            cm.quantize(self.p("head.gguf"), self.p("q.gguf"), self.p("r.ttf"), "/bin/true")


def write_safetensors(path, tensors):
    import json
    h, off, blobs = {"__metadata__": {"format": "pt"}}, 0, []
    for k, (shape, u16) in tensors.items():
        b = u16.astype("<u2").tobytes()
        h[k] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    js = json.dumps(h).encode()
    js += b" " * (-len(js) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(js)) + js + b"".join(blobs))


BF1, BF2, BF05 = 0x3F80, 0x4000, 0x3F00


@unittest.skipIf(np is None, "numpy missing")
class SafetensorsTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.p = lambda n: os.path.join(self.d, n)
        full_model(self.p("coder.gguf"))

    def shards(self, drop=(), shapes=None, extra=None):
        t = {}
        full = lambda shape, v: np.full(shape, v, np.uint16)
        shapes = shapes or {}
        for hf, suffix, kind in cm.ST_MAP:
            shp = {"fc.weight": (N, 2 * N), "layers.0.input_layernorm.weight": (N,),
                   "layers.0.post_attention_layernorm.weight": (N,), "layers.0.self_attn.q_proj.weight": (2 * NH * HD, N),
                   "layers.0.self_attn.k_proj.weight": (NKV * HD, N), "layers.0.self_attn.v_proj.weight": (NKV * HD, N),
                   "layers.0.self_attn.o_proj.weight": (N, NH * HD), "layers.0.self_attn.q_norm.weight": (HD,),
                   "layers.0.self_attn.k_norm.weight": (HD,), "layers.0.mlp.gate.weight": (8, N),
                   "layers.0.mlp.experts.down_proj": (8, N, FF), "layers.0.mlp.shared_expert.gate_proj.weight": (FFS, N),
                   "layers.0.mlp.shared_expert.up_proj.weight": (FFS, N), "layers.0.mlp.shared_expert.down_proj.weight": (N, FFS),
                   "layers.0.mlp.shared_expert_gate.weight": (1, N), "norm.weight": (N,),
                   "pre_fc_norm_embedding.weight": (N,), "pre_fc_norm_hidden.weight": (N,)}[hf]
            t["mtp." + hf] = (shapes.get(hf, shp), full(shapes.get(hf, shp), BF05 if kind == "norm" else BF1))
        gu = np.empty((8, 2 * FF, N), np.uint16)
        gu[:, :FF, :] = BF1
        gu[:, FF:, :] = BF2
        t["mtp." + cm.ST_GATE_UP] = (gu.shape, gu)
        for k in drop:
            del t["mtp." + k]
        t.update(extra or {})
        a, b = dict(list(t.items())[:9]), dict(list(t.items())[9:])
        write_safetensors(self.p("s25.safetensors"), a)
        write_safetensors(self.p("s26.safetensors"), b)
        return [self.p("s25.safetensors"), self.p("s26.safetensors")]

    def build(self, *a, **kw):
        cm.build_from_safetensors(self.shards(*a, **kw), self.p("coder.gguf"), self.p("h3.gguf"), 4)
        return cm.check_file(self.p("h3.gguf"))

    def test_values_layout_and_metadata(self):
        g = self.build()
        L = "blk.%d." % HEAD
        self.assertEqual(g.scalar("qwen35moe.expert_count"), 8)
        self.assertEqual(g.scalar("qwen35moe.expert_used_count"), 4)
        norm = np.frombuffer(g.tensor(L + "attn_norm.weight").data(), "<f4")
        self.assertTrue((norm == 1.5).all())                  # 0.5 + 1
        self.assertEqual(g.info(L + "attn_norm.weight")[2], cm.F32)
        self.assertEqual(g.info(L + "ffn_gate_inp.weight")[2], cm.F32)
        self.assertEqual(g.info(L + "ffn_gate_inp_shexp.weight")[1], [N])
        gate = np.frombuffer(g.tensor(L + "ffn_gate_exps.weight").data(), "<f2")
        up = np.frombuffer(g.tensor(L + "ffn_up_exps.weight").data(), "<f2")
        self.assertTrue((gate == 1.0).all() and (up == 2.0).all())   # gate first half, up second
        self.assertEqual(g.info(L + "ffn_gate_exps.weight")[1], [N, FF, 8])
        self.assertEqual(g.info(L + "ffn_down_exps.weight")[1], [FF, N, 8])
        self.assertEqual(g.info(L + "nextn.eh_proj.weight")[1], [2 * N, N])

    def test_unknown_tensor_is_refused(self):
        with self.assertRaisesRegex(cm.SpecError, "cannot be placed"):
            self.build(extra={"mtp.layers.0.mlp.experts.other": ((1,), np.zeros(1, np.uint16))})

    def test_missing_tensor_is_refused(self):
        with self.assertRaisesRegex(cm.SpecError, "missing from the shards"):
            self.build(drop=["layers.0.self_attn.k_proj.weight"])

    def test_wrong_shape_is_refused(self):
        with self.assertRaisesRegex(cm.SpecError, "does not fit"):
            self.build(shapes={"layers.0.self_attn.k_proj.weight": (NKV * HD + 1, N)})

    def test_f16_overflow_is_refused(self):
        s = self.shards()
        # a BF16 value of 1e10 (0x5015) in the first shared expert matrix
        _, _, shape, start, _ = cm.Safetensors(s).t["mtp.layers.0.self_attn.k_proj.weight"]
        path = cm.Safetensors(s).t["mtp.layers.0.self_attn.k_proj.weight"][0]
        with open(path, "r+b") as f:
            f.seek(start)
            f.write(struct.pack("<H", 0x5015))
        with self.assertRaisesRegex(cm.SpecError, "F16 range"):
            cm.build_from_safetensors(s, self.p("coder.gguf"), self.p("h3.gguf"), 4)


if __name__ == "__main__":
    unittest.main()
