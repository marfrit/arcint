#!/usr/bin/env python3
"""Export a Qwen Flash Next (`qwen4_exp`) checkpoint to the arcint IR layout.

Same construction pattern as `tools/export_mtp.py` and
`tools/export_dflash.py`: read `config.json` and the safetensors state
dict directly, build the OpenVINO `ov::Model` by walking the
checkpoint's tensors, write the multi-component IR layout arcint's
loader expects -- bypassing `optimum-intel`'s export pipeline. This
script owns the CLI plumbing, the config-and-tokenizer passthrough,
and the sidecar `arcint.json` that carries the export-time knobs
(`--moe-lowering`, `--rope`); `build_backbone_ir()` is the next
increment.

Usage:
    python3 tools/export_qwen4_exp.py --checkpoint <ckpt-dir> \\
        --out <out-dir> [--dry-run]

The `<ckpt-dir>` is a local snapshot: `config.json`, `*.safetensors`,
`chat_template.jinja`, `tokenizer.json` and `tokenizer_config.json` must
all be present -- absent files cause the shim to refuse by name, never
to synthesise a fallback (arcint hashes the chat template and the
tokenizer; a fabricated file would produce an artifact whose hashes
diverge from every existing pin).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

MODEL_TYPE = "qwen4_exp"
ARCHITECTURE = "Qwen4ExpForConditionalGeneration"

# The files the arcint loader reads at startup (`src/core/artifact.cpp`
# `parse_artifact`): every entry is a hard require, and the shim refuses
# if any of them is absent from `<out-dir>` after the export.
REQUIRED_OUTPUTS = (
    "openvino_language_model.xml",
    "openvino_language_model.bin",
    "openvino_text_embeddings_model.xml",
    "openvino_text_embeddings_model.bin",
    "openvino_tokenizer.xml",
    "openvino_tokenizer.bin",
    "openvino_detokenizer.xml",
    "openvino_detokenizer.bin",
    "config.json",
    "chat_template.jinja",
    "tokenizer.json",
)

# Checkpoint files copied through verbatim. Tokenizer files travel with
# their own hashes; the loader binds the tokenizer to the OpenVINO
# tokenizer IR at load time, so we ship the source files too for
# reproducibility and re-conversion.
PASSTHROUGH_FILES = (
    "config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
)


def translate_config(cfg):
    """Return the arcint-internal geometry dict the shim's kernel and
    layout code keys off.

    Not written into the produced `config.json` -- that file passes
    through with HF-native keys (see `write_output_layout`). This dict
    lives in the shim's `arcint.json` sidecar and is a stable interface
    for the graph builder still to land.
    """
    if cfg.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"config model_type is {cfg.get('model_type')!r}, expected "
            f"{MODEL_TYPE!r}: refusing to translate a non-qwen4_exp checkpoint"
        )

    text = cfg.get("text_config", cfg)

    required = (
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "rms_norm_eps",
    )
    for k in required:
        if k not in text:
            raise ValueError(
                f"config missing required key {k!r} in text_config: "
                f"cannot translate qwen4_exp geometry without it"
            )

    geo = {
        "arch": ARCHITECTURE,
        "model_type": MODEL_TYPE,
        "n_layer": int(text["num_hidden_layers"]),
        "n_embd": int(text["hidden_size"]),
        "n_head": int(text["num_attention_heads"]),
        "n_head_kv": int(text["num_key_value_heads"]),
        "head_dim": int(text["head_dim"]),
        "n_ff": int(text["intermediate_size"]),
        "rms_norm_eps": float(text["rms_norm_eps"]),
        "rope_theta": float(text.get("rope_theta",
                                     text.get("rope_parameters", {})
                                         .get("rope_theta", 1e7))),
        "max_position_embeddings": int(text.get("max_position_embeddings",
                                                262144)),
        "tie_word_embeddings": bool(text.get("tie_word_embeddings", False)),
    }

    layer_types = text.get("layer_types")
    if layer_types is not None:
        geo["layer_types"] = list(layer_types)
        full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        if len(full) >= 2:
            geo["full_attention_interval"] = full[1] - full[0]
        else:
            geo["full_attention_interval"] = len(layer_types)

    for k in ("linear_num_key_heads", "linear_num_value_heads",
              "linear_key_head_dim", "linear_value_head_dim",
              "linear_conv_kernel_dim"):
        if k in text:
            geo[k] = int(text[k])

    if "num_experts" in text:
        geo["num_experts"] = int(text["num_experts"])
    if "num_experts_per_tok" in text:
        geo["moe_topk"] = int(text["num_experts_per_tok"])
    if "moe_intermediate_size" in text:
        geo["moe_intermediate_size"] = int(text["moe_intermediate_size"])
    if "num_shared_experts" in text:
        geo["num_shared_experts"] = int(text["num_shared_experts"])
    if "norm_topk_prob" in text:
        geo["moe_norm_topk"] = bool(text["norm_topk_prob"])

    if "mtp_num_hidden_layers" in text:
        geo["mtp_layers"] = int(text["mtp_num_hidden_layers"])

    return geo


def write_output_layout(out_dir, checkpoint_dir, geometry, options,
                        component_writer=None, verify=True):
    """Write the arcint output directory.

    Passthrough files come from `<checkpoint_dir>` verbatim; arcint
    hashes `chat_template.jinja` and reads `config.json` for geometry,
    so a checkpoint's own copies are what the loader must see -- never
    a synthesised fallback. `arcint.json` sidecar carries the shim's
    own knobs (moe_lowering, rope) and the derived arcint geometry.

    `component_writer(out_dir, geometry, options)` is where the caller
    injects the ov::Model components (openvino_language_model.xml/.bin
    and companions). `verify=True` (the default) asserts that every
    name in `REQUIRED_OUTPUTS` exists after the writer returns.
    """
    out = Path(out_dir)
    src = Path(checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name in PASSTHROUGH_FILES:
        src_path = src / name
        if not src_path.is_file():
            raise FileNotFoundError(
                f"checkpoint {checkpoint_dir!r} is missing {name!r}, which "
                f"the arcint loader hashes or reads at startup -- refusing "
                f"to fabricate one"
            )
        shutil.copyfile(src_path, out / name)

    sidecar = {
        "shim": "arcint.tools.export_qwen4_exp",
        "model_type": geometry["model_type"],
        "arch": geometry["arch"],
        "geometry": geometry,
        "options": options,
    }
    with (out / "arcint.json").open("w") as f:
        json.dump(sidecar, f, indent=2)

    if component_writer is not None:
        component_writer(out, geometry, options)

    if verify:
        missing = [name for name in REQUIRED_OUTPUTS
                   if not (out / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"arcint layout incomplete under {out_dir}: missing "
                f"{missing}. The component_writer or the passthrough "
                f"step did not produce every required file."
            )


def build_backbone_ir(out_dir, geometry, checkpoint_dir):
    """Reconstruct the qwen4_exp backbone as an ov::Model and save it.

    Refuses with a named reason: the linear-attention + full-attention +
    512-expert MoE + 51B n-gram embedding + MTP head graph is the
    watcher-gated increment (`tools/watch_flash_next_export.py`;
    upstream `optimum-intel` needs to lift both its `<5.6` transformers
    cap and its stale `VisionRotaryEmbedding` import, OR the operator
    picks the fallback and this function ships the graph). The
    geometry carried in the message is what the shim already produced,
    so a caller sees the surface reached, not a generic KeyError.
    """
    raise NotImplementedError(
        "qwen4_exp backbone reconstruction is watcher-gated "
        f"({ARCHITECTURE}, tools/watch_flash_next_export.py). Config, "
        "tokenizer and sidecar passthrough succeeded; the ov::Model "
        f"graph is the remaining surface. Geometry: n_layer="
        f"{geometry.get('n_layer')} n_embd={geometry.get('n_embd')} "
        f"num_experts={geometry.get('num_experts')}"
    )


def load_config(checkpoint):
    """Load `config.json` from a local checkpoint directory. No network
    fetch and no HF snapshot resolve: the caller supplies the local
    directory (arcint's tooling accepts only local paths at export
    time to make provenance auditable)."""
    cfg_path = Path(checkpoint) / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"no config.json under {checkpoint} -- pass a local checkpoint "
            "directory (a full snapshot, not just the HF repo id)"
        )
    with cfg_path.open() as f:
        return json.load(f)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="local checkpoint directory containing config.json, "
                    "the safetensors, chat_template.jinja and tokenizer.json")
    ap.add_argument("--out", required=True,
                    help="output directory for the arcint IR layout")
    ap.add_argument("--moe-lowering", dest="moe_lowering",
                    choices=("batched", "unrolled", "tiled"),
                    default="tiled",
                    help="MoE emission (mirrors export_mtp.py --moe-lowering). "
                    "'tiled' is the form the GPU plugin's fusion pass matches.")
    ap.add_argument("--rope", choices=("interleaved", "half"),
                    default="half",
                    help="RoPE pairing. Default 'half' (llama.cpp/GGUF "
                    "reference convention); export_mtp.py defaults to "
                    "'interleaved' because its MTP-head oracle was A/B'd "
                    "under that pairing.")
    ap.add_argument("--dry-run", action="store_true",
                    help="run passthrough + sidecar only, skip the backbone "
                    "build. Test-hook; write_output_layout(verify=False) is "
                    "used so the missing ov::Model does not raise.")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.checkpoint)
    geo = translate_config(cfg)
    options = {"moe_lowering": args.moe_lowering, "rope": args.rope}

    if args.dry_run:
        write_output_layout(args.out, args.checkpoint, geo, options,
                            verify=False)
        print(f"dry-run: passthrough + sidecar under {args.out}")
        print(f"geometry: {json.dumps(geo, indent=2)}")
        return 0

    def writer(out, geometry, opts):
        build_backbone_ir(out, geometry, args.checkpoint)

    write_output_layout(args.out, args.checkpoint, geo, options,
                        component_writer=writer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
