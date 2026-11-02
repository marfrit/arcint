"""Patch 0080's device-free cell: the MTP head's KV-only prompt pass.

The draft decode with no output rows exists only to fill the MTP layer's K/V,
so the patch returns after the cache store. The device-free half of the proof
is that the branch (a) keeps exactly the K/V ops the full graph builds, in the
same order, with the same arguments, (b) writes V then K then cpy_k then
cpy_v after the same optional cache rotations, (c) builds nothing after the
store (no attention, gate, output projection, FFN or head), (d) is reached
only for a masked nextn decode with no output rows, on a layer without a fused
wqkv or K/V bias, and (e) sits behind LLAMA_MTP_KV_ONLY, OFF by default.

The bytes themselves are owed to the cards: an arm that runs the same draft
against both graphs and compares the retained K/V rows, plus the served A/B
(same accept counters, KL 0 on the draft's K/V rows) - both in the patch
record, run by Claude in an announced window. This cell is a source-text
assertion (it reads the patched tree), not numerics; with the switch off the
graph is byte-identical to before, which is the other arm.

Red mutations, each run and seen failing before it was restored (commands in
the patch README section):
  (a) cpy_v dropped from the branch: the store-writes-V cell fails;
  (b) the switch's default flipped (`e != nullptr` -> `e == nullptr`): the
      default-off cell fails;
  (c) the K norm dropped in the branch: the K-ops cell fails;
  (d) the expand order swapped (K before V): the order cell fails;
  (e) the early `return;` dropped: the nothing-after-store cell fails.
Standard library only (unittest), so the recipe's `ctest -L unit` runs it on
a build host with python3 and nothing else.
"""
import os
import unittest
from pathlib import Path

FILES = ("src/models/qwen35.cpp", "src/models/qwen35moe.cpp")


def _llama_root():
    raw = os.environ.get("LLAMA_SRC")
    if not raw:
        raise unittest.SkipTest("LLAMA_SRC is unset: the patched llama tree is not available")
    root = Path(raw)
    for cand in (root, root.parent):
        if (cand / "src" / "models" / "qwen35.cpp").exists():
            return cand
    raise unittest.SkipTest("LLAMA_SRC has no src/models/qwen35.cpp: the tree lacks patch 0080")


class MtpKvOnly(unittest.TestCase):
    """The branch's store half equals the full graph's and nothing follows it."""

    def setUp(self):
        self.root = _llama_root()
        self.texts = {f: (self.root / f).read_text() for f in FILES}

    def _branch(self, text):
        start = text.find("if (kv_only) {")
        if start < 0:
            self.fail("the KV-only branch is absent")
        end = text.find("\n    }\n\n    auto [Qcur_full, Kcur, Vcur] = build_qkv(layer, cur,", start)
        if end < 0:
            self.fail("the KV-only branch does not end before build_qkv")
        return text[start:end + len("\n    }")]

    def _full(self, text):
        start = text.find("auto [Qcur_full, Kcur, Vcur] = build_qkv(layer, cur,")
        return text[start:]

    def test_switch_is_default_off(self):
        for f, t in self.texts.items():
            self.assertIn('getenv("LLAMA_MTP_KV_ONLY")', t, f)
            self.assertRegex(t, r"e\s*!=\s*nullptr\s*&&\s*atoi\(e\)\s*!=\s*0",
                             f + ": the switch must be off unless LLAMA_MTP_KV_ONLY is 1")

    def test_gated_on_a_masked_no_output_decode(self):
        for f, t in self.texts.items():
            self.assertRegex(t, r"const bool kv_only = kv_only_on\s*&&\s*n_outputs == 0\s*&&\s*"
                                r"cparams\.embeddings_nextn_masked", f)
            # a fused wqkv or a K/V bias would not be reproduced by the branch
            self.assertIn("layer.wqkv == nullptr && layer.wk_b == nullptr && layer.wv_b == nullptr", t, f)

    def test_k_ops_are_the_full_graphs(self):
        # the full graph's K/V come from build_qkv's separate path with reshape=false:
        # a plain wk/wv mat-mul, then reshape_3d; the branch must build the same
        for f, t in self.texts.items():
            b = self._branch(t)
            full = self._full(t)
            self.assertRegex(full, r"build_qkv\(layer, cur,[^;]*\n\s*il, false\);", f)
            for line in (
                "ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);",
                "ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);",
                "Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);",
                "Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);",
                "Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);",
                "ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,",
            ):
                self.assertIn(line, b, f + ": the branch must build " + line)
            # the norm and rope the full graph runs after build_qkv
            self.assertIn("Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);", full, f)
            self.assertIn("ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,", full, f)

    def test_store_order_and_rotations(self):
        for f, t in self.texts.items():
            b = self._branch(t)
            i_krot = b.find("Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot);")
            i_vrot = b.find("Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot);")
            i_ev = b.find("ggml_build_forward_expand(gf, Vcur);")
            i_ek = b.find("ggml_build_forward_expand(gf, Kcur);")
            i_ck = b.find("cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il)")
            i_cv = b.find("cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il)")
            for name, i in (("k rotation", i_krot), ("v rotation", i_vrot), ("expand v", i_ev),
                            ("expand k", i_ek), ("cpy_k", i_ck), ("cpy_v", i_cv)):
                self.assertTrue(i >= 0, f + ": the branch must have " + name)
            self.assertLess(i_krot, i_ek, f)
            self.assertLess(i_ev, i_ek, f + ": v is expanded before k, as build_attn does")
            self.assertLess(i_ek, i_ck, f)
            self.assertLess(i_ck, i_cv, f + ": k is stored before v")

    def test_nothing_follows_the_store(self):
        for f, t in self.texts.items():
            b = self._branch(t)
            self.assertTrue(b.rstrip().endswith("}"), f)
            after = b[b.find("return;") + len("return;"):]
            for forbidden in ("build_attn(", "build_lora_mm(layer.wo", "build_ffn(",
                              "mtp_shared_head_norm", "ggml_mul(ctx0"):
                self.assertNotIn(forbidden, after, f + ": the branch builds past the store: " + forbidden)


if __name__ == "__main__":
    unittest.main()
