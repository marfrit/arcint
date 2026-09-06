#!/usr/bin/env python3
"""Compare a GGUF's projections with the served IR's, tensor by tensor, on the host.

Both are quantizations of the same base checkpoint, so a correctly mapped and
correctly un-reordered tensor pair has a cosine similarity near 1 and a
mismapped or mispermuted one collapses (docs/design-gguf-native.md §3.4, the
host layer). Reads the IR's .bin directly through the XML's offsets (u4 weights,
u4 zero points, f16 scales, group 64; u8 per-channel for the head), so it needs
numpy and gguf-py only.

    python3 tools/gguf_ir_compare.py <file.gguf> <ir-dir> [--layers 0,3] [--all]

Prints one line per module: cosine, max |diff| / max |ir|, and for the V-head
tensors the cosine with and without the un-reorder, which is the check that the
inverse of the converter's permutation is the right one.
"""
import argparse, json, re, sys
import xml.etree.ElementTree as ET
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

LAYER_MODULES = [
    ("linear_attn.in_proj_qkv", "attn_qkv", "rows_qkv"),
    ("linear_attn.in_proj_z", "attn_gate", "rows_v"),
    ("linear_attn.in_proj_a", "ssm_alpha", "rows_v"),
    ("linear_attn.in_proj_b", "ssm_beta", "rows_v"),
    ("linear_attn.out_proj", "ssm_out", "columns"),
    ("mlp.gate_proj", "ffn_gate", None),
    ("mlp.up_proj", "ffn_up", None),
    ("mlp.down_proj", "ffn_down", None),
    ("self_attn.q_proj", "attn_q", None),
    ("self_attn.k_proj", "attn_k", None),
    ("self_attn.v_proj", "attn_v", None),
    ("self_attn.o_proj", "attn_output", None),
]


def ir_constants(xml_path):
    root = ET.parse(xml_path).getroot()
    out = {}
    for l in root.iter("layer"):
        if l.get("type") != "Const":
            continue
        d = l.find("data")
        out[l.get("name")] = (d.get("element_type"), [int(x) for x in d.get("shape").split(",") if x], int(d.get("offset")), int(d.get("size")))
    return out


def ir_dequant(bin_mm, consts, name):
    et, shape, off, size = consts[name]
    if et == "u4":
        raw = np.frombuffer(bin_mm[off:off + size], dtype=np.uint8)
        lo = (raw & 0xF).astype(np.float32)
        hi = (raw >> 4).astype(np.float32)
        w = np.stack([lo, hi], axis=-1).reshape(-1)[: int(np.prod(shape))].reshape(shape)
        zet, zshape, zoff, zsize = consts[name + "/zero_point"]
        zraw = np.frombuffer(bin_mm[zoff:zoff + zsize], dtype=np.uint8)
        zlo = (zraw & 0xF).astype(np.float32); zhi = (zraw >> 4).astype(np.float32)
        z = np.stack([zlo, zhi], axis=-1).reshape(-1)[: int(np.prod(zshape))].reshape(zshape)
        set_, sshape, soff, ssize = consts[name + "/scale"]
        s = np.frombuffer(bin_mm[soff:soff + ssize], dtype=np.float16).reshape(sshape).astype(np.float32)
        return ((w - z) * s).reshape(shape[0], -1)
    if et == "u8":
        raw = np.frombuffer(bin_mm[off:off + size], dtype=np.uint8).astype(np.float32).reshape(shape)
        zet, zshape, zoff, zsize = consts[name + "/zero_point"]
        z = np.frombuffer(bin_mm[zoff:zoff + zsize], dtype=np.uint8).astype(np.float32).reshape(zshape)
        set_, sshape, soff, ssize = consts[name + "/scale"]
        s = np.frombuffer(bin_mm[soff:soff + ssize], dtype=np.float16).reshape(sshape).astype(np.float32)
        return ((raw - z) * s).reshape(shape[0], -1)
    raise SystemExit(f"unhandled IR element type {et} for {name}")


def to_file_heads(k_heads, v_heads):
    r = v_heads // k_heads
    return [ (h % r) * k_heads + h // r for h in range(v_heads) ]


def unreorder_rows(w, first, v_heads, to_file, head_rows):
    out = w.copy()
    for h in range(v_heads):
        src = first + to_file[h] * head_rows
        dst = first + h * head_rows
        out[dst:dst + head_rows] = w[src:src + head_rows]
    return out


def cosine(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf"); ap.add_argument("ir_dir")
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--all", action="store_true", help="every layer")
    a = ap.parse_args()
    reader = GGUFReader(a.gguf)
    tensors = {t.name: t for t in reader.tensors}
    consts = ir_constants(f"{a.ir_dir}/openvino_language_model.xml")
    bin_mm = np.memmap(f"{a.ir_dir}/openvino_language_model.bin", dtype=np.uint8, mode="r")
    cfg = json.load(open(f"{a.ir_dir}/config.json")); tc = cfg.get("text_config", cfg)
    kh, vh = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    vdim = tc["linear_value_head_dim"]; kdim = tc["linear_key_head_dim"]
    to_file = to_file_heads(kh, vh)
    layers = range(tc["num_hidden_layers"]) if a.all else [int(x) for x in a.layers.split(",")]
    prefix = "self.model.model.language_model.layers."

    def gg(name):
        t = tensors[name]
        w = dequantize(t.data, t.tensor_type)
        return np.asarray(w, dtype=np.float32).reshape(-1, int(t.shape[0]))  # [out, in] as numpy from gguf-py

    for i in layers:
        for ir_mod, gname, reorder in LAYER_MODULES:
            ir_name = f"{prefix}{i}.{ir_mod}._openvino_orig_weight"
            if ir_name not in consts:
                continue
            g = gg(f"blk.{i}.{gname}.weight")
            w = ir_dequant(bin_mm, consts, ir_name)
            if g.shape != w.shape:
                print(f"layer {i:2d} {ir_mod:26s} SHAPE gguf {g.shape} ir {w.shape}"); continue
            line = f"layer {i:2d} {ir_mod:26s} {tensors[f'blk.{i}.{gname}.weight'].tensor_type.name:5s}"
            if reorder == "rows_qkv":
                fixed = unreorder_rows(g, 2 * kh * kdim, vh, to_file, vdim)
                line += f" cos raw {cosine(g, w):.4f} unreordered {cosine(fixed, w):.4f}"
            elif reorder == "rows_v":
                head_rows = g.shape[0] // vh
                fixed = unreorder_rows(g, 0, vh, to_file, head_rows)
                line += f" cos raw {cosine(g, w):.4f} unreordered {cosine(fixed, w):.4f}"
            elif reorder == "columns":
                fixed = unreorder_rows(g.T.copy(), 0, vh, to_file, vdim).T
                line += f" cos raw {cosine(g, w):.4f} unreordered {cosine(fixed, w):.4f}"
            else:
                line += f" cos {cosine(g, w):.4f}"
            print(line, flush=True)
    if "self.model.lm_head._openvino_orig_weight" in consts:
        g = gg("output.weight"); w = ir_dequant(bin_mm, consts, "self.model.lm_head._openvino_orig_weight")
        print(f"lm_head {'':30s} {tensors['output.weight'].tensor_type.name:5s} cos {cosine(g, w):.4f}" if g.shape == w.shape else f"lm_head SHAPE {g.shape} {w.shape}")


if __name__ == "__main__":
    main()
