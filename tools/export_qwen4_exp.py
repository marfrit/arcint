#!/usr/bin/env python3
"""Export a Qwen Flash Next (`qwen4_exp`) checkpoint to the arcint IR layout.

Same construction pattern as `tools/export_mtp.py` and
`tools/export_dflash.py`: read `config.json` and the safetensors state
dict directly, build the OpenVINO `ov::Model` by walking the
checkpoint's tensors, write the multi-component IR layout arcint's
loader expects -- bypassing `optimum-intel`'s export pipeline. This
script owns the CLI plumbing, the config-and-tokenizer passthrough,
and the sidecar `arcint.json` that carries the export-time knobs
(`--moe-lowering`, `--rope`).

`build_backbone_ir()` (E2 Phase C) builds the ov::Model from real GGUF weights
(via `q4e.gguf_feed`) + the arcint-original opset-13 emitter (`q4e.backbone`),
replacing the earlier unreachable full-safetensors-checkpoint path. A TINY config
emits end to end on CPU (`--gguf-ir --dry-run`, .xml AND .bin hashes printed --
the .xml carries the graph, the .bin the weights, and a weight change moves only
the latter); the
full-size 48-layer IR materialises the whole backbone and is WINDOW TERRITORY --
refused here on CPU with a named reason.

Usage:
    # config/tokenizer/sidecar passthrough (safetensors provenance path):
    python3 tools/export_qwen4_exp.py --checkpoint <ckpt-dir> --out <out-dir> [--dry-run]
    # tiny end-to-end backbone IR from GGUF weights (windowless dry-run):
    python3 tools/export_qwen4_exp.py --gguf-ir --dry-run \\
        --gguf-shards <shard-dir> --out <out-dir>

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


# The arcint-original emission reads its forward-pass spec from the HF
# transformers reference, PINNED so a later read is against the same code:
#   transformers @ main, src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
#   commit 5b7dcb0d36c242d8d85920a81c564ef3a86ca6dd (2026-09-09), generated
#   from modular_qwen4_exp.py (Apache-2.0, "The Qwen Team and HuggingFace").
# The installed dev-host venv (transformers 5.0.0) does NOT carry qwen4_exp; its
# cousin qwen3_next covers GDN + MoE + RoPE + RMSNorm but NOT the three
# qwen4_exp-specific modules (GatedResidual, QSA, PLE). The module inventory to
# emit, with the reference class that specifies each:
#   Qwen4ExpTextNGramEmbedding  - hashed n-gram row_ids (arcint: exec/ngram_row_ids.h)
#   Qwen4ExpTextPLELayer        - PLE inject: gating + dilated conv. WHICH layer
#     is unreconciled: HF config.json (FIX B) declares ple_layer_ids [2]; the
#     Unsloth GGUF KV (WP2) records qwen4exp.ple.layers [1]. Both are pre-
#     conversion one-indexed claims; settle against the pinned reference's
#     index convention (config.py:149-150 one->zero-based) before injecting.
#   Qwen4ExpTextGatedResidual   - hyper-connection stream mix (hc_count, hc_lowrank)
#   Qwen4ExpTextQSAIndexer      - Qwen Sparse Attention block selection
#   Qwen4ExpTextAttention       - MHA + q-norm + RoPE + gate scaling (full-attn layers)
#   Qwen4ExpTextGatedDeltaNet   - GDN linear attention (interval 4 -> 36 of 48)
#   Qwen4ExpTextTopKRouter / Experts / SparseMoeBlock - 512 experts, top-10, width 640
#   MTP x1 head                 - text_config.mtp (hybrid, layer_types, ...)
# Acceptance is the KLD gate in tools/kld_harness.py (threshold 0.0599 nats),
# NOT "compiles and serves". Every emitted component is validated against the
# torch reference (max-abs + KL drift on synthetic weights) before the whole
# backbone is admitted.
REFERENCE_COMMIT = "5b7dcb0d36c242d8d85920a81c564ef3a86ca6dd"


# --- The gguf_feed + q4e.backbone emission path (E2 Phase C) ------------------
# The backbone IR is now built from (real GGUF weights via q4e.gguf_feed) +
# (the arcint-original opset-13 emitter q4e.backbone), NOT from a full HF
# safetensors checkpoint (the "unreachable full checkpoint path" the earlier
# NotImplementedError named). q4e.backbone / ref_backbone are already parity-
# validated against the pin on random weights (E2 inc1-5b) AND on real GGUF
# tensors (E2 Phase B: transcription-vs-pin 0.0 whole-backbone).
#
# BOUNDARY (windowless prep): a TINY config emits end to end on CPU -- that is
# what `--gguf-ir --dry-run` does, and what this session records the artifact
# hash of. FULL-SIZE emission (48 layers, hidden 2560, 512 experts) materialises
# the whole backbone's weights and is WINDOW TERRITORY (both cards hold resident
# services); this tool refuses it on CPU with a named reason rather than OOM the
# host. The full-size IR is the window's job.

# Mirrors tests/python/test_backbone._make_config (small-but-complete: 4 GDN
# layers, PLE at ple_layer_ids [2], every layer MoE). Kept here so the dry-run
# is self-contained (a tool does not import from tests/).
def _tiny_config():
    from transformers.models.qwen4_exp import configuration_qwen4_exp as pin_cfg
    return pin_cfg.Qwen4ExpTextConfig(
        hidden_size=16, num_hidden_layers=4, hc_count=4, hc_lowrank=8,
        rms_norm_eps=1e-6, layer_types=["linear_attention"] * 4,
        num_experts=8, num_experts_per_tok=2, norm_topk_prob=True,
        moe_intermediate_size=32, shared_expert_intermediate_size=32,
        hidden_act="silu",
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        ple_layer_ids=[2], ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=17, make_ngram_vocab_size_divisible_by=128,
        ple_embed_dim=32, ple_conv_kernel_size=4, seed=1234,
        vocab_size=257, eos_token_id=0, pad_token_id=0,
    )


def gguf_fed_state(config, feed):
    """Numpy state dict (pin state-dict keys -> arrays) for `config`, every
    weight fed from the GGUF via q4e.gguf_feed and sliced to the config's shape;
    derived / non-fed buffers keep the pin's config-recomputed values. GDN/MoE/hc
    keys are fed from a real GDN block (blk.0), PLE from the real PLE block
    (blk.1) -- the tiny config declares layers GDN that the real model may ship
    as QSA, so a slice cell feeds from a real GDN block.

    THE HEAD follows the source, not the pin: when the GGUF declares a separate
    `output.weight` (the shipped UD-Q3_K_XL does -- it is NOT tied, measured;
    see `q4e.ref_backbone`'s header) the state dict carries `lm_head.weight` fed
    from it, and `q4e.backbone` emits that head. Only a source WITHOUT a head
    falls back to the pin's tie to `embed_tokens`."""
    import numpy as np
    import torch
    from q4e import ref_backbone, gguf_feed as _gf

    ref = ref_backbone.Qwen4ExpTextBackbone(
        config, declare_lm_head=feed.has_lm_head())
    sd = ref.state_dict()
    state = {}
    for k, v in sd.items():
        shape = tuple(v.shape)
        if any(k.endswith(s) for s in _gf._DERIVED_SUFFIXES):
            state[k] = v.detach().cpu().numpy()
            continue
        gl = (1 if ".ple." in k else 0) if k.startswith("layers.") else None
        # feed.fitted, never pin_tensor + a hand slice: the fused gate_up's ff
        # axis is two stacked halves and a leading slice of the concat takes
        # both of them from gate (FIX D).
        state[k] = feed.fitted(k, shape, gguf_layer=gl).astype(np.float32)
    return state


def build_backbone_ir(out_dir, geometry, shards, seq_len=64, tiny=False):
    """Build the qwen4_exp backbone as an ov::Model from GGUF weights and save
    it under `out_dir` (openvino_language_model.xml/.bin). Returns
    `{"xml": sha256, "bin": sha256}`.

    BOTH hashes, because the .xml carries the GRAPH and the .bin carries the
    WEIGHTS: the FIX D change (true gate|up halves) moved the .bin and left the
    .xml byte-identical, so an artifact record that quotes only the .xml cannot
    witness a weight change at all.

    `tiny=True` uses the small end-to-end-CPU config (the dry-run). `tiny=False`
    (full-size, from `geometry`) is refused here: the full backbone's residency
    is window territory -- run it in a GPU window, not on the CPU export host.
    """
    if not tiny:
        # Refuse before importing openvino/torch so the refusal stays device-free.
        raise NotImplementedError(
            "full-size qwen4_exp backbone IR emission is WINDOW TERRITORY "
            f"(n_layer={geometry.get('n_layer')} n_embd={geometry.get('n_embd')} "
            f"num_experts={geometry.get('num_experts')}): materialising the whole "
            "backbone's weights will not fit the CPU export host. Use --gguf-ir "
            "--dry-run for the tiny end-to-end emission; run the full IR in a "
            "GPU window (both cards hold resident services)."
        )

    import hashlib
    from pathlib import Path

    import openvino as ov

    from q4e import gguf_feed
    from q4e.backbone import build_backbone

    feed = gguf_feed.GgufFeed(shards)
    config = _tiny_config()
    state = gguf_fed_state(config, feed)
    model = build_backbone(config, state, seq_len=seq_len)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xml = out / "openvino_language_model.xml"
    ov.save_model(model, str(xml))
    binf = xml.with_suffix(".bin")
    return {
        "xml": hashlib.sha256(xml.read_bytes()).hexdigest(),
        "bin": hashlib.sha256(binf.read_bytes()).hexdigest(),
    }


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
    ap.add_argument("--checkpoint",
                    help="local checkpoint directory containing config.json, "
                    "the safetensors, chat_template.jinja and tokenizer.json "
                    "(required unless --gguf-ir)")
    ap.add_argument("--out", required=True,
                    help="output directory for the arcint IR layout")
    ap.add_argument("--gguf-ir", action="store_true",
                    help="build the backbone IR from GGUF weights (q4e.gguf_feed "
                    "+ q4e.backbone). With --dry-run: emit the TINY config's IR "
                    "end to end on CPU and print its artifact hash (Phase C "
                    "windowless prep). Without --dry-run: full-size, refused on "
                    "CPU as window territory.")
    ap.add_argument("--gguf-shards",
                    help="GGUF shard directory or glob (with --gguf-ir). No "
                    "default location is baked in (this repo is public).")
    ap.add_argument("--seq-len", type=int, default=64,
                    help="fixed sequence length for the emitted static backbone "
                    "(with --gguf-ir).")
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

    if args.gguf_ir:
        if not args.gguf_shards:
            raise SystemExit("--gguf-ir requires --gguf-shards")
        digest = build_backbone_ir(args.out, geometry={}, shards=args.gguf_shards,
                                   seq_len=args.seq_len, tiny=args.dry_run)
        mode = "tiny dry-run" if args.dry_run else "full-size"
        print(f"gguf-ir ({mode}): openvino_language_model.xml under {args.out}")
        # graph AND weights: the .xml alone does not move when only weights do.
        print(f"artifact sha256 (.xml, graph):   {digest['xml']}")
        print(f"artifact sha256 (.bin, weights): {digest['bin']}")
        return 0

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --gguf-ir")
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
        # Full-size backbone emission is window territory (build_backbone_ir
        # refuses tiny=False on CPU); the weights come from GGUF via --gguf-shards
        # in a GPU window, not from the safetensors checkpoint on the export host.
        build_backbone_ir(out, geometry, shards=args.gguf_shards, tiny=False)

    write_output_layout(args.out, args.checkpoint, geo, options,
                        component_writer=writer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
