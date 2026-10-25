#!/usr/bin/env python3
"""C5 voice similarity: the seed-tts-eval SIM model (WavLM-Large + ECAPA-TDNN,
the fine-tuned speaker-verification checkpoint from `microsoft/UniSpeech`),
run as a one-number command behind `tools/tts_gate.py --embed-cmd`.

The model is the one the operator named (docs/milestone-1.2.0.md §6/D8):
`wavlm_large_finetune.pth`, sha256 `51f07e3b…`, the UniSpeech "WavLM large,
*Fix pre-train: No*" row; the mirrored copy is `bezzam/
wavlm_large_finetune_seed_tts_eval` (CC-BY-SA-3.0). Per the operator's
condition it runs in a container, and the file kept on the host is converted
to safetensors there. This script loads that safetensors copy.

Recipe (Apache-2.0, adapted from `huggingface/open-tts-leaderboard`
`transformers/score_similarity.py`): the WavLM-Large backbone is an HF
`WavLMModel` with the checkpoint's original-WavLM parameter names remapped
onto it (no s3prl / torch.hub). The ECAPA-TDNN head is `tools/speaker_head.py`
(Apache-2.0), written clean-room from a behavioural specification and the
paper (Desplanques, Thienpondt, Demuynck, Interspeech 2020), whose state_dict
carries the checkpoint's head keys, so the head loads strictly (a copy of this
tool needs `speaker_head.py` beside it). Only the 256-d
embedding is used; the training-time AM-softmax `loss_calculator.*` is
dropped. One clip per forward pass (instance norm and attentive pooling take
time-axis statistics, so batching would shift shorter clips). Cosine, as
upstream. See `THIRD_PARTY.md`.

Licence of this file: Apache-2.0, as the repository. The speaker model's
weights are CC BY-SA 3.0 (the `bezzam/wavlm_large_finetune_seed_tts_eval`
mirror of the UniSpeech checkpoint); they are staged operator-locally and not
part of this repository.

    tts_voice_sim.py pair --a A.wav --b B.wav [--model X.safetensors]
    tts_voice_sim.py mean-vs-ref --ref R.wav --wav W1.wav [W2.wav ...]
    tts_voice_sim.py stdin --ref R.wav [--model X.safetensors]

`stdin` reads the `cases` JSON that `tts_gate.py` pipes to `--embed-cmd` (a
list of objects with a `wav` path), embeds the reference once and prints the
mean cosine over the cases. `--model` (or `$ARCINT_SIM_MODEL`) is the staged
safetensors copy of the speaker model, an operator-local path;
`--config` is the backbone config (id or config.json path).
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import WavLMConfig, WavLMModel

from speaker_head import SpeakerHead

DEFAULT_MODEL = os.environ.get("ARCINT_SIM_MODEL", "")
DEFAULT_CONFIG = "microsoft/wavlm-large"
SR = 16000


def load_wav_16k_mono(path):
    """(samples,) f32 at 16 kHz mono from a RIFF WAVE; first channel only."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF WAVE: {path}")
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        sz = struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + sz]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            raw = body
        pos += 8 + sz + (sz & 1)
    if fmt is None or raw is None:
        raise ValueError(f"missing fmt/data chunk: {path}")
    tag, ch, rate, _byte_rate, _block, bits = struct.unpack_from("<HHIIHH", fmt, 0)
    if tag == 1 and bits == 16:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    elif tag == 3 and bits == 32:
        a = np.frombuffer(raw, dtype=np.float32).astype(np.float32)
    elif tag == 1 and bits == 8:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(f"unsupported WAVE fmt tag={tag} bits={bits}: {path}")
    if bits == 16:
        a = a / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch)[:, 0]
    if rate != SR:
        n = int(round(len(a) * SR / rate))
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
    return a


class WavLMSpeaker(nn.Module):
    """WavLM-Large's hidden states (the input embedding and every layer)
    stacked and handed to the ECAPA-TDNN head."""

    def __init__(self, wavlm, head):
        super().__init__()
        self.wavlm = wavlm
        self.head = head

    def forward(self, wav):
        with torch.no_grad():
            hidden = self.wavlm(wav, output_hidden_states=True).hidden_states
        return self.head(torch.stack(hidden, dim=0))


def convert_wavlm_backbone_state_dict(orig, target_keys):
    """Remap original-WavLM (`feature_extract.model.*`) names onto HF WavLMModel."""
    prefix = "feature_extract.model."
    parametrized = "encoder.pos_conv_embed.conv.parametrizations.weight.original0" in target_keys
    g_name = "parametrizations.weight.original0" if parametrized else "weight_g"
    v_name = "parametrizations.weight.original1" if parametrized else "weight_v"

    def convert(key):
        if key.startswith("feature_extractor.conv_layers."):
            layer_id, sub = key[len("feature_extractor.conv_layers."):].split(".", 1)
            if sub == "0.weight":
                return f"feature_extractor.conv_layers.{layer_id}.conv.weight"
            if sub.startswith("2.1."):
                return f"feature_extractor.conv_layers.{layer_id}.layer_norm.{sub[4:]}"
            return None
        if key.startswith("layer_norm."):
            return "feature_projection.layer_norm." + key[len("layer_norm."):]
        if key.startswith("post_extract_proj."):
            return "feature_projection.projection." + key[len("post_extract_proj."):]
        if key == "mask_emb":
            return "masked_spec_embed"
        if key.startswith("encoder.pos_conv.0."):
            suffix = key[len("encoder.pos_conv.0."):]
            mapped = {"bias": "bias", "weight_g": g_name, "weight_v": v_name}.get(suffix)
            return f"encoder.pos_conv_embed.conv.{mapped}" if mapped else None
        if key.startswith("encoder.layer_norm."):
            return key
        if key.startswith("encoder.layers."):
            layer_id, sub = key[len("encoder.layers."):].split(".", 1)
            out = f"encoder.layers.{layer_id}."
            for src, dst in (
                ("self_attn.q_proj.", "attention.q_proj."),
                ("self_attn.k_proj.", "attention.k_proj."),
                ("self_attn.v_proj.", "attention.v_proj."),
                ("self_attn.out_proj.", "attention.out_proj."),
                ("self_attn.grep_linear.", "attention.gru_rel_pos_linear."),
                ("self_attn.relative_attention_bias.", "attention.rel_attn_embed."),
                ("self_attn_layer_norm.", "layer_norm."),
                ("final_layer_norm.", "final_layer_norm."),
                ("fc1.", "feed_forward.intermediate_dense."),
                ("fc2.", "feed_forward.output_dense."),
            ):
                if sub.startswith(src):
                    return out + dst + sub[len(src):]
            if sub == "self_attn.grep_a":
                return out + "attention.gru_rel_pos_const"
            return None
        return None

    converted, unmapped = {}, []
    for key, value in orig.items():
        if not key.startswith(prefix):
            continue
        mapped = convert(key[len(prefix):])
        if mapped is None:
            unmapped.append(key)
        else:
            converted[mapped] = value
    if unmapped:
        raise RuntimeError(f"unmapped backbone keys: {sorted(unmapped)[:10]}")
    missing = sorted(target_keys - converted.keys())
    unexpected = sorted(converted.keys() - target_keys)
    if missing or unexpected:
        raise RuntimeError(f"backbone key mismatch: missing {missing[:10]}; unexpected {unexpected[:10]}")
    return converted


class Embedder:
    def __init__(self, model_path, config, device="cpu"):
        cfg = WavLMConfig.from_pretrained(config)
        cfg.apply_spec_augment = False
        cfg.layerdrop = 0.0
        backbone = WavLMModel(cfg)
        state = load_file(model_path)
        backbone.load_state_dict(convert_wavlm_backbone_state_dict(state, set(backbone.state_dict())), strict=True)
        head = SpeakerHead(num_states=1 + cfg.num_hidden_layers, feat_dim=cfg.hidden_size)
        head.load_state_dict({k: v for k, v in state.items()
                              if not k.startswith("feature_extract.") and not k.startswith("loss_calculator.")},
                             strict=True)
        self.model = WavLMSpeaker(backbone, head)
        self.model.to(device).eval()
        self.device = device

    def embed(self, path):
        a = load_wav_16k_mono(path)
        wav = torch.from_numpy(a).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(wav)[0]


def cosine(emb_a, emb_b):
    return float(F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0))[0])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pair", "mean-vs-ref", "stdin"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--config", default=DEFAULT_CONFIG)
        p.add_argument("--device", default="cpu")
        p.add_argument("--ref", default="")
        p.add_argument("--a", default="")
        p.add_argument("--b", default="")
        p.add_argument("--wav", nargs="*", default=[])
    args = ap.parse_args()
    if not args.model:
        ap.error("--model (or $ARCINT_SIM_MODEL): the speaker model's safetensors copy")
    emb = Embedder(args.model, args.config, args.device)
    if args.cmd == "pair":
        sim = cosine(emb.embed(args.a), emb.embed(args.b))
        print(f"{sim:.6f}")
        return 0
    if args.cmd == "mean-vs-ref":
        ref = emb.embed(args.ref)
        sims = [cosine(emb.embed(w), ref) for w in args.wav]
        print(f"mean {np.mean(sims):.6f} n={len(sims)} min {min(sims):.6f} max {max(sims):.6f}")
        return 0
    cases = json.load(sys.stdin)
    ref = emb.embed(args.ref)
    sims = [cosine(emb.embed(c["wav"]), ref) for c in cases]
    print(f"{np.mean(sims):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
