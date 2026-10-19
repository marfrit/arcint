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
onto it (no s3prl / torch.hub); the ECAPA-TDNN head is transcribed from
UniSpeech (MIT) with upstream's module names so the checkpoint head loads
strictly. Only the 256-d embedding is used; the training-time AM-softmax
`loss_calculator.*` is dropped. One clip per forward pass (instance norm and
attentive pooling take time-axis statistics, so batching would shift shorter
clips). Cosine, as upstream. See `THIRD_PARTY.md`.

    tts_voice_sim.py pair --a A.wav --b B.wav [--model X.safetensors]
    tts_voice_sim.py mean-vs-ref --ref R.wav --wav W1.wav [W2.wav ...]
    tts_voice_sim.py stdin --ref R.wav [--model X.safetensors]

`stdin` reads the `cases` JSON that `tts_gate.py` pipes to `--embed-cmd` (a
list of objects with a `wav` path), embeds the reference once and prints the
mean cosine over the cases. `--model` defaults to the staged safetensors copy;
`--config` is the backbone config (id or config.json path).
"""
import argparse
import json
import struct
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import WavLMConfig, WavLMModel

DEFAULT_MODEL = "/models/.sim/wavlm_large_finetune_seed_tts_eval.safetensors"
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


# ── ECAPA-TDNN head, transcribed from UniSpeech (MIT), names match upstream ──
class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0
        self.scale, self.width = scale, channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList(
            nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias)
            for _ in range(self.nums))
        self.bns = nn.ModuleList(nn.BatchNorm1d(self.width) for _ in range(self.nums))

    def forward(self, x):
        out, spx = [], torch.split(x, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.bns[i](F.relu(self.convs[i](sp)))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        return x * torch.sigmoid(self.linear2(F.relu(self.linear1(out)))).unsqueeze(2)


class SE_Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, scale, se_bottleneck_dim):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(out_channels, kernel_size, stride, padding, dilation, scale=scale)
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)
        self.shortcut = (nn.Conv1d(in_channels, out_channels, kernel_size=1)
                         if in_channels != out_channels else None)

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut else x
        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)
        return x + residual


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att
        in_ch = in_dim * 3 if global_context_att else in_dim
        self.linear1 = nn.Conv1d(in_ch, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)

    def forward(self, x):
        if self.global_context_att:
            cm = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            cs = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, cm, cs), dim=1)
        else:
            x_in = x
        alpha = torch.softmax(self.linear2(torch.tanh(self.linear1(x_in))), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        std = torch.sqrt((torch.sum(alpha * (x ** 2), dim=2) - mean ** 2).clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class WavLMEcapaTdnn(nn.Module):
    """ECAPA-TDNN over a softmax-weighted sum of WavLM-Large's 25 hidden states."""

    def __init__(self, wavlm, feat_dim=1024, channels=512, emb_dim=256):
        super().__init__()
        self.wavlm = wavlm
        self.feat_num = 1 + wavlm.config.num_hidden_layers
        self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))
        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.channels = [channels] * 4 + [1536]
        self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, channels, 3, 1, 2, 2, 8, 128)
        self.layer3 = SE_Res2Block(channels, channels, 3, 1, 3, 3, 8, 128)
        self.layer4 = SE_Res2Block(channels, channels, 3, 1, 4, 4, 8, 128)
        self.conv = nn.Conv1d(channels * 3, self.channels[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(self.channels[-1], attention_channels=128)
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

    def get_feat(self, wav):
        with torch.no_grad():
            hidden = self.wavlm(wav, output_hidden_states=True).hidden_states
        x = torch.stack(hidden, dim=0)
        x = (F.softmax(self.feature_weight, dim=-1).view(-1, 1, 1, 1) * x).sum(dim=0)
        return self.instance_norm(torch.transpose(x, 1, 2) + 1e-6)

    def forward(self, wav):
        x = self.get_feat(wav)
        o2, o3, o4 = self.layer2(self.layer1(x)), None, None
        o3 = self.layer3(o2)
        o4 = self.layer4(o3)
        out = F.relu(self.conv(torch.cat([o2, o3, o4], dim=1)))
        return self.linear(self.bn(self.pooling(out)))


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
        self.model = WavLMEcapaTdnn(backbone, feat_dim=cfg.hidden_size)
        head = {k: v for k, v in state.items()
                if not k.startswith("feature_extract.") and not k.startswith("loss_calculator.")}
        missing, unexpected = self.model.load_state_dict(head, strict=False)
        stray = [k for k in missing if not k.startswith("wavlm.")]
        if stray or unexpected:
            raise RuntimeError(f"head key mismatch: missing {stray[:10]}; unexpected {list(unexpected)[:10]}")
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
