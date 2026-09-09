#!/usr/bin/env python3
"""Tests for the three export-tooling fronts in export_2b_awq.py.

Each front is tested independently of the actual model export (no GPU,
no checkpoint download, no window).  The full export is tested only by
running export_2b_awq.py itself on the dev host with the checkpoint.

Run:  python3 tools/test_export_2b_awq.py
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(__file__))

# Guard: these tests need optimum-intel + transformers importable.
# On hosts without the venv they skip cleanly.
try:
    from optimum.intel.openvino.configuration import OVWeightQuantizationConfig
    HAS_OPTIMUM = True
except ImportError:
    HAS_OPTIMUM = False


class TestFront2DatasetValidation(unittest.TestCase):
    """Front 2: OVWeightQuantizationConfig.post_init rejects custom datasets."""

    @unittest.skipUnless(HAS_OPTIMUM, "optimum-intel not installed")
    def test_allowlist_rejects_custom_string(self):
        """The bug: a custom dataset path is rejected by post_init."""
        with self.assertRaises((ValueError, Exception)):
            OVWeightQuantizationConfig(
                bits=4, quant_method="awq", dataset="/tmp/my_custom_dataset"
            )

    @unittest.skipUnless(HAS_OPTIMUM, "optimum-intel not installed")
    def test_bypass_sets_arbitrary_dataset(self):
        """The fix: construct with a valid name, then override."""
        from export_2b_awq import bypass_dataset_validation

        config = OVWeightQuantizationConfig(
            bits=4, quant_method="awq", dataset="wikitext2"
        )
        self.assertEqual(config.dataset, "wikitext2")

        bypass_dataset_validation(config, "/tmp/my_custom_corpus")
        self.assertEqual(config.dataset, "/tmp/my_custom_corpus")

    @unittest.skipUnless(HAS_OPTIMUM, "optimum-intel not installed")
    def test_bypass_accepts_none(self):
        """None dataset is valid for OVQuantizer with explicit calibration_dataset."""
        from export_2b_awq import bypass_dataset_validation

        config = OVWeightQuantizationConfig(
            bits=4, quant_method="awq", dataset="wikitext2"
        )
        bypass_dataset_validation(config, None)
        self.assertIsNone(config.dataset)


class TestFront3CalibrationData(unittest.TestCase):
    """Front 3: VL calibration crashes on Qwen3.5 (no video_processor_class)."""

    @unittest.skipUnless(HAS_OPTIMUM, "optimum-intel not installed")
    def test_text_calibration_shape(self):
        """Calibration data has the right shape without touching a processor."""
        from export_2b_awq import make_text_calibration_data
        from unittest.mock import MagicMock

        # Mock tokenizer — no network access needed
        tok = MagicMock()
        tok.vocab_size = 248320

        samples = make_text_calibration_data(tok, n_samples=4, seq_len=64)
        self.assertEqual(len(samples), 4)
        for s in samples:
            self.assertIn("input_ids", s)
            self.assertIn("attention_mask", s)
            self.assertEqual(s["input_ids"].shape, (1, 64))
            self.assertEqual(s["attention_mask"].shape, (1, 64))

    @unittest.skipUnless(HAS_OPTIMUM, "optimum-intel not installed")
    def test_text_calibration_deterministic(self):
        """Same inputs produce same outputs (reproducible calibration)."""
        from export_2b_awq import make_text_calibration_data
        from unittest.mock import MagicMock
        import torch

        tok = MagicMock()
        tok.vocab_size = 248320

        a = make_text_calibration_data(tok, n_samples=8, seq_len=32)
        b = make_text_calibration_data(tok, n_samples=8, seq_len=32)
        for sa, sb in zip(a, b):
            self.assertTrue(torch.equal(sa["input_ids"], sb["input_ids"]))

    @unittest.skipUnless(HAS_OPTIMUM, "optimum-intel not installed")
    def test_text_calibration_ids_in_vocab_range(self):
        """All token IDs are within [0, vocab_size)."""
        from export_2b_awq import make_text_calibration_data
        from unittest.mock import MagicMock

        tok = MagicMock()
        tok.vocab_size = 1000

        samples = make_text_calibration_data(tok, n_samples=4, seq_len=128)
        for s in samples:
            ids = s["input_ids"].squeeze().tolist()
            self.assertTrue(all(0 <= i < 1000 for i in ids),
                            f"token ID out of range: {[i for i in ids if i >= 1000]}")


class TestFront1Markers(unittest.TestCase):
    """Front 1: CausalLM path produces 0 _openvino_orig_weight markers.

    This front is MOOT if fronts 2+3 are fixed (VL path produces markers).
    The test here verifies the marker-counting utility, not the export itself.
    """

    def test_count_markers_on_missing_dir(self):
        from export_2b_awq import count_orig_weight_markers
        markers, total = count_orig_weight_markers("/nonexistent/path")
        self.assertEqual(markers, 0)
        self.assertEqual(total, 0)

    def test_count_markers_on_synthetic_xml(self):
        """Synthetic IR XML with known marker constants."""
        from export_2b_awq import count_orig_weight_markers
        import tempfile, os

        xml = """<?xml version="1.0"?>
<net name="test" version="11">
<layers>
  <layer id="0" name="self.model.layers.0.mlp.gate_proj._openvino_orig_weight" type="Const">
    <data element_type="f32" shape="1,1"/>
    <output><port id="0" precision="FP32"><dim>1</dim></port></output>
  </layer>
  <layer id="1" name="self.model.layers.0.mlp.gate_proj/scale" type="Const">
    <data element_type="f32" shape="1,1"/>
    <output><port id="0" precision="FP32"><dim>1</dim></port></output>
  </layer>
  <layer id="2" name="self.model.layers.0.mlp.up_proj._openvino_orig_weight" type="Const">
    <data element_type="f32" shape="1,1"/>
    <output><port id="0" precision="FP32"><dim>1</dim></port></output>
  </layer>
  <layer id="3" name="embedding_table" type="Const">
    <data element_type="f32" shape="1,1"/>
    <output><port id="0" precision="FP32"><dim>1</dim></port></output>
  </layer>
</layers>
</net>"""
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "openvino_language_model.xml"), "w") as f:
                f.write(xml)
            markers, total = count_orig_weight_markers(d)
            self.assertEqual(markers, 2)
            self.assertEqual(total, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
