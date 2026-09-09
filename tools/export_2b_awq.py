#!/usr/bin/env python3
"""Export a Qwen3.5-2B template IR with AWQ markers (_openvino_orig_weight).

Fixes three export-tooling bugs that block a 2B AWQ export (forensics:
commit 67ef4d9, docs/design-qwen-flash-next.md §6):

  Front 2 — OVWeightQuantizationConfig.post_init() validates dataset names
  against a hardcoded allowlist.  VL models accept only {'contextual'}.
  Fix: override config.dataset after construction (post_init has run).

  Front 3 — The 'contextual' dataset falls back to textvqa, which calls
  processor.video_processor_class on Qwen3.5's processor (Qwen3VLProcessor),
  which returns None and crashes with AttributeError.
  Fix: provide calibration data as a pre-tokenized list[dict], bypassing the
  named-dataset pipeline entirely.  The calibration samples are plain text
  (this IS a text model); no video processor is needed.

  Front 1 — OVModelForCausalLM + AWQ produces 0 _openvino_orig_weight
  constants (the CausalLM path folds weights without FakeQuantize subgraphs).
  Fix: use OVModelForVisualCausalLM (the VL pipeline), which inserts
  FakeQuantize nodes with _openvino_orig_weight suffixes.  Fronts 2+3 make
  this path work.

Usage:
    python3 tools/export_2b_awq.py --checkpoint Qwen/Qwen3.5-2B \\
        --out /models/ov/qwen35-2b-awq

Requires: optimum-intel, transformers, nncf, openvino in the active venv.
The export runs on CPU (no GPU window needed), but is slow (~10-30 min for
AWQ calibration depending on num_samples).  Does NOT need a GPU window.
"""
import argparse
import sys


def bypass_dataset_validation(config, dataset_value):
    """Front 2 fix: set config.dataset to an arbitrary value after post_init
    validation has already run.  post_init checks dataset against a hardcoded
    allowlist; this bypasses it by writing directly to the instance dict."""
    config.__dict__["dataset"] = dataset_value


def make_text_calibration_data(tokenizer, n_samples=32, seq_len=128):
    """Front 3 fix: build calibration data from tokenizer alone, no processor.

    Returns a list of dicts with 'input_ids' and 'attention_mask' tensors,
    the format OVQuantizer.quantize() accepts as calibration_dataset when
    passed as a list.  The content is synthetic (repeated vocab samples);
    AWQ's sensitivity measurement only needs diverse activation patterns,
    not meaningful text.

    This avoids the video_processor_class crash: AutoProcessor is never
    instantiated, and the calibration pipeline never touches the VL
    data-loading path.
    """
    import torch

    samples = []
    vocab_size = tokenizer.vocab_size
    for i in range(n_samples):
        # Deterministic, diverse token sequences (no randomness — the
        # calibration must be reproducible across runs).
        ids = [(i * 7919 + j * 131 + 17) % vocab_size for j in range(seq_len)]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        samples.append({"input_ids": input_ids, "attention_mask": attention_mask})
    return samples


def count_orig_weight_markers(model_dir):
    """Count _openvino_orig_weight constants in the exported IR.

    Reads the language model XML and counts constants whose friendly name
    ends with '._openvino_orig_weight'.  Returns (count, total_constants).
    """
    import xml.etree.ElementTree as ET
    from pathlib import Path

    xml_path = Path(model_dir) / "openvino_language_model.xml"
    if not xml_path.exists():
        xml_path = Path(model_dir) / "openvino_model.xml"
    if not xml_path.exists():
        return 0, 0

    tree = ET.parse(xml_path)
    root = tree.getroot()
    marker_count = 0
    total_const = 0
    for layer in root.iter("layer"):
        if layer.get("type") == "Const":
            total_const += 1
            name = layer.get("name", "")
            if name.endswith("._openvino_orig_weight"):
                marker_count += 1
    return marker_count, total_const


def export_2b_awq(checkpoint, out_dir, num_samples=32, group_size=64):
    """Two-phase export: plain VL export, then AWQ via OVQuantizer.

    Phase 1: OVModelForVisualCausalLM.from_pretrained(export=True) — no
    compression.  This is the path that already worked for the 2B
    (design doc §6 Path (a)).

    Phase 2: OVQuantizer.quantize() with AWQ config and text-only
    calibration data.  This applies weight compression with FakeQuantize
    insertion, producing the _openvino_orig_weight markers.
    """
    from optimum.intel import OVModelForVisualCausalLM
    from optimum.intel.openvino.configuration import OVConfig, OVWeightQuantizationConfig
    from optimum.intel.openvino.quantization import OVQuantizer
    from transformers import AutoTokenizer

    print(f"Phase 1: exporting {checkpoint} via VL pipeline (no compression)")
    model = OVModelForVisualCausalLM.from_pretrained(
        checkpoint,
        export=True,
        trust_remote_code=False,
    )

    print(f"Phase 2: AWQ compression with {num_samples} text-only samples")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False)
    calibration_data = make_text_calibration_data(tokenizer, n_samples=num_samples)

    quant_config = OVWeightQuantizationConfig(
        bits=4,
        quant_method="awq",
        group_size=group_size,
        sym=False,
        ratio=1.0,
        num_samples=num_samples,
        dataset="wikitext2",  # passes validation; overridden below
    )
    bypass_dataset_validation(quant_config, None)

    quantizer = OVQuantizer(model)
    quantizer.quantize(
        calibration_dataset=calibration_data,
        save_directory=out_dir,
        ov_config=OVConfig(quantization_config=quant_config),
    )

    markers, total = count_orig_weight_markers(out_dir)
    print(f"exported to {out_dir}: {markers} _openvino_orig_weight markers "
          f"out of {total} constants")
    if markers == 0:
        print("WARNING: 0 markers — gguf_apply_to_template will not match "
              "any projections. See front 1 in the docstring.", file=sys.stderr)
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="Qwen/Qwen3.5-2B",
                    help="HF repo id (default: Qwen/Qwen3.5-2B)")
    ap.add_argument("--out", default="/models/ov/qwen35-2b-awq",
                    help="output directory for the AWQ-compressed IR")
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    rc = export_2b_awq(args.checkpoint, args.out, args.num_samples, args.group_size)
    sys.exit(rc)


if __name__ == "__main__":
    main()
