#!/usr/bin/env python3
"""Unit tests for the qwen4_exp export shim: argument plumbing, config
translation, output layout. All synthetic -- no real checkpoint, no
network, no venv beyond the standard library.

Run:  python3 tools/test_export_qwen4_exp.py
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from export_qwen4_exp import (  # noqa: E402
    ARCHITECTURE,
    MODEL_TYPE,
    PASSTHROUGH_FILES,
    REFERENCE_COMMIT,
    REQUIRED_OUTPUTS,
    build_backbone_ir,
    load_config,
    main,
    parse_args,
    translate_config,
    write_output_layout,
)


SYNTHETIC_CONFIG = {
    "architectures": [ARCHITECTURE],
    "model_type": MODEL_TYPE,
    "text_config": {
        "num_hidden_layers": 4,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "intermediate_size": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 8192,
        "tie_word_embeddings": True,
        "layer_types": [
            "linear_attention", "linear_attention", "linear_attention",
            "full_attention",
        ],
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 512,
        "num_shared_experts": 1,
        "norm_topk_prob": True,
        "mtp_num_hidden_layers": 1,
    },
}


def make_checkpoint(root, cfg=None, chat_template="TEMPLATE",
                    tokenizer_json='{"model":"stub"}',
                    tokenizer_config_json='{"tokenizer_class":"stub"}'):
    """Build a synthetic checkpoint directory under `root` with only the
    files the shim's passthrough copies. Missing files must be added by
    a test that wants to check the refusal path."""
    if cfg is None:
        cfg = SYNTHETIC_CONFIG
    ck = Path(root) / "ck"
    ck.mkdir()
    (ck / "config.json").write_text(json.dumps(cfg))
    if chat_template is not None:
        (ck / "chat_template.jinja").write_text(chat_template)
    if tokenizer_json is not None:
        (ck / "tokenizer.json").write_text(tokenizer_json)
    if tokenizer_config_json is not None:
        (ck / "tokenizer_config.json").write_text(tokenizer_config_json)
    return ck


class TestArgumentPlumbing(unittest.TestCase):
    def test_required_flags(self):
        # --out is the only PARSE-level requirement; --checkpoint is optional at
        # parse time (a second mode, --gguf-ir, does not use it) and its
        # presence is enforced in main().
        with self.assertRaises(SystemExit):
            parse_args([])
        with self.assertRaises(SystemExit):
            parse_args(["--checkpoint", "/tmp/x"])  # missing --out
        a = parse_args(["--out", "/tmp/x"])         # valid: checkpoint optional
        self.assertIsNone(a.checkpoint)
        self.assertFalse(a.gguf_ir)
        # main() enforces the per-mode requirements.
        with self.assertRaises(SystemExit):
            main(["--out", "/tmp/x"])                    # no --checkpoint, no --gguf-ir
        with self.assertRaises(SystemExit):
            main(["--gguf-ir", "--out", "/tmp/x"])       # --gguf-ir needs --gguf-shards

    def test_defaults(self):
        a = parse_args(["--checkpoint", "/nowhere", "--out", "/tmp/out"])
        self.assertEqual(a.moe_lowering, "tiled")
        self.assertEqual(a.rope, "half")
        self.assertFalse(a.dry_run)

    def test_choices_enforced(self):
        with self.assertRaises(SystemExit):
            parse_args(["--checkpoint", "/x", "--out", "/y",
                        "--moe-lowering", "invalid"])
        with self.assertRaises(SystemExit):
            parse_args(["--checkpoint", "/x", "--out", "/y",
                        "--rope", "bogus"])

    def test_all_flags_together(self):
        a = parse_args([
            "--checkpoint", "/ck",
            "--out", "/out",
            "--moe-lowering", "unrolled",
            "--rope", "interleaved",
            "--dry-run",
        ])
        self.assertEqual(a.checkpoint, "/ck")
        self.assertEqual(a.out, "/out")
        self.assertEqual(a.moe_lowering, "unrolled")
        self.assertEqual(a.rope, "interleaved")
        self.assertTrue(a.dry_run)


class TestConfigTranslation(unittest.TestCase):
    def test_basic_translation(self):
        geo = translate_config(SYNTHETIC_CONFIG)
        self.assertEqual(geo["arch"], ARCHITECTURE)
        self.assertEqual(geo["model_type"], MODEL_TYPE)
        self.assertEqual(geo["n_layer"], 4)
        self.assertEqual(geo["n_embd"], 64)
        self.assertEqual(geo["n_head"], 4)
        self.assertEqual(geo["n_head_kv"], 2)
        self.assertEqual(geo["head_dim"], 16)
        self.assertEqual(geo["n_ff"], 128)
        self.assertAlmostEqual(geo["rms_norm_eps"], 1e-6)
        self.assertAlmostEqual(geo["rope_theta"], 1000000.0)
        self.assertEqual(geo["max_position_embeddings"], 8192)
        self.assertTrue(geo["tie_word_embeddings"])

    def test_layer_types_and_interval(self):
        geo = translate_config(SYNTHETIC_CONFIG)
        self.assertEqual(len(geo["layer_types"]), 4)
        self.assertEqual(geo["full_attention_interval"], 4)

    def test_gdn_geometry(self):
        geo = translate_config(SYNTHETIC_CONFIG)
        self.assertEqual(geo["linear_num_key_heads"], 16)
        self.assertEqual(geo["linear_num_value_heads"], 48)
        self.assertEqual(geo["linear_conv_kernel_dim"], 4)

    def test_moe_geometry(self):
        geo = translate_config(SYNTHETIC_CONFIG)
        self.assertEqual(geo["num_experts"], 512)
        self.assertEqual(geo["moe_topk"], 10)
        self.assertEqual(geo["moe_intermediate_size"], 512)
        self.assertEqual(geo["num_shared_experts"], 1)
        self.assertTrue(geo["moe_norm_topk"])

    def test_mtp_metadata(self):
        geo = translate_config(SYNTHETIC_CONFIG)
        self.assertEqual(geo["mtp_layers"], 1)

    def test_wrong_model_type_refuses(self):
        cfg = {"model_type": "llama", "text_config": {}}
        with self.assertRaises(ValueError) as cm:
            translate_config(cfg)
        self.assertIn("model_type", str(cm.exception))

    def test_missing_required_key_refuses(self):
        cfg = json.loads(json.dumps(SYNTHETIC_CONFIG))
        del cfg["text_config"]["hidden_size"]
        with self.assertRaises(ValueError) as cm:
            translate_config(cfg)
        self.assertIn("hidden_size", str(cm.exception))


class TestOutputLayout(unittest.TestCase):
    """The shim's job on the layout side: passthrough of HF-native
    config + tokenizer + chat_template, sidecar with arcint's own
    knobs, and REQUIRED_OUTPUTS all present after `component_writer`
    ran. These tests fail if a regression drops a required file, if
    the shim invents a chat template, or if config.json is emitted in
    the wrong shape."""

    def test_passthrough_verbatim(self):
        with tempfile.TemporaryDirectory() as d:
            ck = make_checkpoint(d, chat_template="TEMPLATE-XYZ",
                                 tokenizer_json='{"model":"tok-xyz"}')
            out = Path(d) / "out"
            geo = translate_config(SYNTHETIC_CONFIG)
            write_output_layout(out, ck, geo, {"moe_lowering": "tiled", "rope": "half"},
                                verify=False)
            self.assertEqual((out / "chat_template.jinja").read_text(),
                             "TEMPLATE-XYZ")
            self.assertEqual((out / "tokenizer.json").read_text(),
                             '{"model":"tok-xyz"}')
            emitted_cfg = json.loads((out / "config.json").read_text())
            self.assertEqual(emitted_cfg["model_type"], MODEL_TYPE)
            self.assertEqual(emitted_cfg["text_config"]["num_hidden_layers"], 4)
            self.assertEqual(emitted_cfg["text_config"]["hidden_size"], 64)
            self.assertNotIn("geometry", emitted_cfg)
            self.assertNotIn("exporter", emitted_cfg)

    def test_sidecar_carries_geometry_and_options(self):
        with tempfile.TemporaryDirectory() as d:
            ck = make_checkpoint(d)
            out = Path(d) / "out"
            geo = translate_config(SYNTHETIC_CONFIG)
            write_output_layout(out, ck, geo,
                                {"moe_lowering": "unrolled", "rope": "interleaved"},
                                verify=False)
            sidecar = json.loads((out / "arcint.json").read_text())
            self.assertEqual(sidecar["shim"], "arcint.tools.export_qwen4_exp")
            self.assertEqual(sidecar["arch"], ARCHITECTURE)
            self.assertEqual(sidecar["geometry"]["n_layer"], 4)
            self.assertEqual(sidecar["options"]["moe_lowering"], "unrolled")
            self.assertEqual(sidecar["options"]["rope"], "interleaved")

    def test_refuses_missing_chat_template(self):
        with tempfile.TemporaryDirectory() as d:
            ck = make_checkpoint(d, chat_template=None)
            out = Path(d) / "out"
            geo = translate_config(SYNTHETIC_CONFIG)
            with self.assertRaises(FileNotFoundError) as cm:
                write_output_layout(out, ck, geo, {"moe_lowering": "tiled", "rope": "half"},
                                    verify=False)
            self.assertIn("chat_template.jinja", str(cm.exception))

    def test_refuses_missing_tokenizer(self):
        with tempfile.TemporaryDirectory() as d:
            ck = make_checkpoint(d, tokenizer_json=None)
            out = Path(d) / "out"
            geo = translate_config(SYNTHETIC_CONFIG)
            with self.assertRaises(FileNotFoundError) as cm:
                write_output_layout(out, ck, geo, {"moe_lowering": "tiled", "rope": "half"},
                                    verify=False)
            self.assertIn("tokenizer.json", str(cm.exception))

    def test_verify_flags_missing_required_output(self):
        """verify=True raises when component_writer did not produce
        every REQUIRED_OUTPUTS entry -- the layout is not silently half-
        built. Fails if verify becomes optional or REQUIRED_OUTPUTS
        stops being checked."""
        with tempfile.TemporaryDirectory() as d:
            ck = make_checkpoint(d)
            out = Path(d) / "out"
            geo = translate_config(SYNTHETIC_CONFIG)
            with self.assertRaises(FileNotFoundError) as cm:
                write_output_layout(out, ck, geo, {"moe_lowering": "tiled", "rope": "half"})
            missing = str(cm.exception)
            self.assertIn("openvino_language_model.xml", missing)

    def test_verify_passes_when_component_writer_lands_all(self):
        def writer(out, geo, opts):
            for name in REQUIRED_OUTPUTS:
                (Path(out) / name).write_text("stub")
        with tempfile.TemporaryDirectory() as d:
            ck = make_checkpoint(d)
            out = Path(d) / "out"
            geo = translate_config(SYNTHETIC_CONFIG)
            write_output_layout(out, ck, geo, {"moe_lowering": "tiled", "rope": "half"},
                                component_writer=writer, verify=True)
            for name in REQUIRED_OUTPUTS:
                self.assertTrue((out / name).is_file())

    def test_required_and_passthrough_lists_agree(self):
        """Every passthrough file must be in REQUIRED_OUTPUTS unless the
        loader tolerates its absence (tokenizer_config.json is not in
        artifact.cpp's require list at present -- so PASSTHROUGH_FILES
        is a superset)."""
        required = set(REQUIRED_OUTPUTS)
        passthrough = set(PASSTHROUGH_FILES)
        self.assertTrue(passthrough <= required | {"tokenizer_config.json"})

    def test_reference_commit_is_a_declared_wellformed_provenance_id(self):
        """FIX F: `REFERENCE_COMMIT` became an unused import when 2cd2b2f
        rewrote the refusal message and dropped its only assertion (the old one
        checked that the message quoted REFERENCE_COMMIT[:8] and named
        kld_harness).

        DECISION, recorded: the constant STAYS and the import earns its keep
        here, but only for what this test can honestly gate. Re-quoting a
        commit id inside an error string was never a provenance check -- it
        asserted that a substring appeared in a substring. The SUBSTANTIVE
        oracle gate is the pin FILE's sha256 (ca9f00bb..., plus the config
        module's), asserted by tests/python/test_backbone.py::_assert_pin,
        which every q4e parity cell calls; this file is stdlib-only by design
        (no venv, no transformers), so it cannot reach the pin to hash it.

        What it CAN gate is that the module still declares a well-formed
        upstream provenance id -- so a botched edit or a silent deletion of the
        comment block that names where the pinned reference came from lands
        red here rather than nowhere."""
        self.assertIsInstance(REFERENCE_COMMIT, str)
        self.assertEqual(len(REFERENCE_COMMIT), 40, REFERENCE_COMMIT)
        self.assertTrue(all(c in "0123456789abcdef" for c in REFERENCE_COMMIT),
                        REFERENCE_COMMIT)

    def test_backbone_build_refuses_full_size_with_enumerated_blockers(self):
        # E2 Phase B/C: the emitter EXISTS now (q4e.backbone from q4e.gguf_feed),
        # so the refusal is not "not yet emitted". FIX C: nor is it "window
        # territory" -- that named residency only and read as "get a GPU window
        # and this works", which is false. The refusal must ENUMERATE its real
        # blockers, so this cell asserts all three are named, that the head is
        # recorded as RESOLVED (FIX B wired it), and that the banned framing is
        # gone. The tiny path (tiny=True / --gguf-ir --dry-run) is the CPU one.
        geo = translate_config(SYNTHETIC_CONFIG)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(NotImplementedError) as cm:
                build_backbone_ir(d, geo, "/no/such/shards")  # tiny defaults False
        msg = str(cm.exception)
        self.assertIn("qwen4_exp", msg)
        self.assertIn("n_layer=4", msg)
        # (1) geometry, (2) ruled scope, (3) residency -- each named
        self.assertIn("GEOMETRY", msg)
        self.assertIn("_tiny_config", msg)
        self.assertIn("SCOPE", msg)
        self.assertIn("QSA", msg)
        self.assertIn("indexer", msg)
        self.assertIn("RESIDENCY", msg)
        # residency names its assumption and a number, never a bare "does not fit"
        self.assertIn("f32 ov Constant", msg)
        self.assertIn("656.9 GiB", msg)
        # the head is no longer a blocker -- and the refusal says so
        self.assertIn("RESOLVED", msg)
        self.assertIn("output.weight", msg)
        # the banned framing
        self.assertNotIn("WINDOW TERRITORY", msg)
        self.assertNotIn("window's job", msg)


class TestMain(unittest.TestCase):
    def test_dry_run_lands_passthrough_and_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            ck = make_checkpoint(d, chat_template="MAIN-TEMPLATE")
            out = Path(d) / "out"
            rc = main([
                "--checkpoint", str(ck),
                "--out", str(out),
                "--dry-run",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((out / "config.json").is_file())
            self.assertTrue((out / "chat_template.jinja").is_file())
            self.assertTrue((out / "tokenizer.json").is_file())
            self.assertTrue((out / "arcint.json").is_file())
            self.assertEqual((out / "chat_template.jinja").read_text(),
                             "MAIN-TEMPLATE")

    def test_non_dry_run_refuses_via_component_writer(self):
        with tempfile.TemporaryDirectory() as d:
            ck = make_checkpoint(d)
            out = Path(d) / "out"
            with self.assertRaises(NotImplementedError):
                main(["--checkpoint", str(ck), "--out", str(out)])
            self.assertTrue((out / "config.json").is_file())


class TestLoadConfig(unittest.TestCase):
    def test_missing_dir_raises_named(self):
        with self.assertRaises(FileNotFoundError):
            load_config("/no/such/checkpoint/anywhere")

    def test_reads_synthetic_config(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text(json.dumps(SYNTHETIC_CONFIG))
            cfg = load_config(d)
            self.assertEqual(cfg["model_type"], MODEL_TYPE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
