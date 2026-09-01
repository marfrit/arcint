#!/usr/bin/env python3
"""DFlash2 pairing probe, part B (torch, CPU).

Teacher-forced replay of the DFlash2 draft against features recorded from
OUR int4 OV target (part A). Measures acceptance length (accepted+1 per
verification cycle, the model card's metric) on the target's own greedy
continuation. A shuffled-features arm is the null control: if the probe
cannot fail, it measures nothing.

Minimal reimplementation of z-lab/dflash dflash/model.py (transformers in
the venv is 5.0.0, the reference wants 5.15). Plain torch ops only,
float32 on CPU, teacher-forced (no incremental cache: context K/V are
recomputed per cycle; probe contexts are far below the 2048 window).

Inputs (from part A, .npz):
  tokens      [T]        prompt + greedy continuation token ids
  prompt_len  scalar
  feats       [T, 5, 5120]  residual stream after layers 5,19,33,47,61
                            (HF hidden_states[layer_id+1] convention)
  embeds      [T, 5120]     target input embeddings of `tokens`
  mask_embed  [5120]        target input embedding of mask_token_id
  lm_head applied in part A is NOT needed here; the draft's logits use the
  target lm_head via a callback -- part A saves lm_head as a matmul against
  a dequantized weight is too big, so part A instead dumps DRAFT-needed
  logits on the fly? No: we run the lm_head here via OV (CPU), passed as
  --lm-head path to the artifact dir.
"""
import argparse
import json
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
torch.manual_seed(0)


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        data = np.memmap(path, dtype=np.uint8, mode="r")
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            raw = torch.frombuffer(bytes(data[base + s:base + e]), dtype=torch.bfloat16)
            out[k] = raw.view(v["shape"]).to(torch.float32)
    return out


def rms_norm(x, w, eps=1e-6):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return v * w


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(positions, head_dim, theta):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = positions.float()[:, None] * inv[None, :]
    emb = torch.cat((ang, ang), dim=-1)
    return emb.cos(), emb.sin()


def grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    # hidden [1, L, H]; dynamic [1, L, K, groups]; base [K, H]
    b, length, h = hidden.shape
    groups = h // group_size
    blocks = hidden.view(b, length, groups, group_size)
    dyn = dynamic.reshape(b, length, base.shape[0], groups, 1)
    out = torch.zeros_like(blocks)
    for off in range(base.shape[0]):
        vals = blocks if off == 0 else F.pad(blocks[:, :-off], (0, 0, 0, 0, off, 0))
        kern = base[off].view(1, 1, groups, group_size)
        out = out + kern * vals + dyn[:, :, off] * vals
    return out.view(b, length, h)


class Conv:
    def __init__(self, w, prefix, group_size):
        self.base = w[f"{prefix}.base_kernel"]            # [2, K, H]
        self.proj = w[f"{prefix}.kernel_projection.weight"]  # [2*K*groups, H]
        self.gs = group_size
        self.k = self.base.shape[1]

    def prepare(self, hidden):
        groups = hidden.shape[-1] // self.gs
        dyn = (hidden @ self.proj.T).view(*hidden.shape[:-1], 2, self.k, groups)
        return grouped_dynamic_convolve(hidden, dyn[..., 0, :, :], self.base[0], self.gs), dyn[..., 1, :, :]

    def finish(self, hidden, dyn):
        return grouped_dynamic_convolve(hidden, dyn, self.base[1], self.gs)


class Draft:
    def __init__(self, path, cfg):
        w = load_safetensors(path)
        self.w = w
        self.cfg = cfg
        d = cfg["dflash_config"]
        self.block = d["block_size"]
        self.mask_id = d["mask_token_id"]
        self.layer_ids = d["target_layer_ids"]
        self.top_k = d["selector_top_k"]
        self.gs = d["conv_group_size"]
        self.n_layers = cfg["num_hidden_layers"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.theta = cfg["rope_parameters"]["rope_theta"]
        self.convs = [(Conv(w, f"layers.{i}.attention_conv", self.gs),
                       Conv(w, f"layers.{i}.mlp_conv", self.gs)) for i in range(self.n_layers)]

    def forward(self, target_hidden, noise_embedding, positions):
        """target_hidden [1, C, 25600]; noise_embedding [1, Q, 5120];
        positions: absolute ids for the C ctx rows then Q block rows."""
        w = self.w
        ctx = rms_norm(target_hidden @ w["fc.weight"].T, w["hidden_norm.weight"])
        h = noise_embedding
        C, Q = ctx.shape[1], h.shape[1]
        cos, sin = rope_cos_sin(positions, self.hd, self.theta)   # [C+Q, hd]
        for i in range(self.n_layers):
            p = f"layers.{i}"
            res = h
            x = rms_norm(h, w[f"{p}.input_layernorm.weight"])
            x, akern = self.convs[i][0].prepare(x)
            # attention: q from block; k/v from [ctx | block]
            q = (x @ w[f"{p}.self_attn.q_proj.weight"].T).view(1, Q, self.n_heads, self.hd)
            q = rms_norm(q, w[f"{p}.self_attn.q_norm.weight"]).transpose(1, 2)
            kin = torch.cat([ctx, x], dim=1)
            k = (kin @ w[f"{p}.self_attn.k_proj.weight"].T).view(1, C + Q, self.n_kv, self.hd)
            v = (kin @ w[f"{p}.self_attn.v_proj.weight"].T).view(1, C + Q, self.n_kv, self.hd)
            k = rms_norm(k, w[f"{p}.self_attn.k_norm.weight"]).transpose(1, 2)
            v = v.transpose(1, 2)
            qe = (q * cos[None, None, -Q:]) + (rotate_half(q) * sin[None, None, -Q:])
            ke = (k * cos[None, None, :]) + (rotate_half(k) * sin[None, None, :])
            ke = ke.repeat_interleave(self.n_heads // self.n_kv, dim=1)
            ve = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)
            # non-causal (config is_causal false), window >> probe ctx: no mask
            a = F.scaled_dot_product_attention(qe, ke, ve, scale=self.hd ** -0.5)
            a = a.transpose(1, 2).reshape(1, Q, -1)
            a = a @ w[f"{p}.self_attn.o_proj.weight"].T
            a = self.convs[i][0].finish(a, akern)
            h = res + a
            res = h
            x = rms_norm(h, w[f"{p}.post_attention_layernorm.weight"])
            x, mkern = self.convs[i][1].prepare(x)
            g = F.silu(x @ w[f"{p}.mlp.gate_proj.weight"].T) * (x @ w[f"{p}.mlp.up_proj.weight"].T)
            x = g @ w[f"{p}.mlp.down_proj.weight"].T
            x = self.convs[i][1].finish(x, mkern)
            h = res + x
        return rms_norm(h, self.w["norm.weight"])

    def propose(self, hidden, anchor_id, logits_fn):
        """Greedy selector path over top-k candidates per position."""
        w = self.w
        logits = logits_fn(hidden)                                  # [Q, V]
        unary, cand = torch.topk(logits, self.top_k, dim=-1)        # [Q, k]
        hp = hidden[0] @ w["candidate_selector.hidden_projection.weight"].T  # [Q, r]
        pred = torch.tensor([anchor_id])
        path = []
        for pos in range(hidden.shape[1]):
            s = unary[pos] + (w["candidate_selector.predecessor_codebook"][pred[0]] * hp[pos]) \
                @ w["candidate_selector.successor_codebook"][cand[pos]].T
            pred = cand[pos][torch.argmax(s)][None]
            path.append(int(pred[0]))
        return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--draft", default="/models/gptq/qwen38-dflash2")
    ap.add_argument("--artifact", default="/models/ov/qwen38-b7c1-ov")
    ap.add_argument("--shuffle", action="store_true", help="null control: shuffled features")
    args = ap.parse_args()

    cfg = json.load(open(f"{args.draft}/config.json"))
    draft = Draft(f"{args.draft}/model.safetensors", cfg)

    import openvino as ov
    core = ov.Core()
    lm_head = core.compile_model(f"{args.artifact}/openvino_mtp_lm_head.xml", "CPU")

    def logits_fn(hidden):
        out = lm_head(hidden.numpy().astype(np.float32))[0]
        return torch.from_numpy(out.reshape(hidden.shape[1], -1).astype(np.float32))

    d = np.load(args.npz)
    tokens = d["tokens"]
    p0 = int(d["prompt_len"])
    feats = torch.from_numpy(d["feats"].astype(np.float32))     # [T, 5, 5120]
    feats = feats.reshape(feats.shape[0], -1)                   # [T, 25600]
    embeds = torch.from_numpy(d["embeds"].astype(np.float32))   # [T, 5120]
    mask_embed = torch.from_numpy(d["mask_embed"].astype(np.float32))

    if args.shuffle:
        g = torch.Generator().manual_seed(7)
        feats = feats[torch.randperm(feats.shape[0], generator=g)]

    T = len(tokens)
    block = draft.block
    start = p0                     # tokens[p0] is the first generated token (the anchor)
    cycles, accepted_total = 0, 0
    while start + 1 < T - 1:
        q = min(block, T - start)
        if q < 2:
            break
        ctx_feats = feats[:start][None]              # accepted positions only
        noise = torch.cat([embeds[start][None], mask_embed[None].repeat(q - 1, 1)])[None]
        positions = torch.arange(0, start + q)       # ctx 0..start-1, block start..start+q-1
        hid = draft.forward(ctx_feats, noise, positions)[:, 1 - q:, :]
        path = draft.propose(hid, int(tokens[start]), logits_fn)
        truth = tokens[start + 1:start + q].tolist()
        acc = 0
        for a, b in zip(path, truth):
            if a == b:
                acc += 1
            else:
                break
        cycles += 1
        accepted_total += acc
        start += acc + 1
    arm = "SHUFFLED (null control)" if args.shuffle else "paired"
    print(f"{arm}: cycles {cycles}, drafted {(cycles * (block - 1))}, accepted {accepted_total}, "
          f"acceptance length {(accepted_total + cycles) / max(1, cycles):.2f} "
          f"(accepted+1 per cycle; card metric)")


if __name__ == "__main__":
    main()
