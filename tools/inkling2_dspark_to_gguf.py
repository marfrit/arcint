#!/usr/bin/env python3
"""Inkling 2 DSpark drafter (mtp.safetensors) -> a DFLASH GGUF for the pinned llama.cpp.

A numpy-only port of the Inkling2DSparkModel converter of llama.cpp PR #32218
(conversion/inkling2.py at 9f57b4453570c6a5fef01fec8e65a44d783a63e0; authors Xuan Son Nguyen, Piotr Wilkin, ruixiang63):
the same tensor names, shapes and metadata, without torch, transformers or the
safetensors package. It reads the files itself, opens no network connection, starts
no process and writes one file, the output.

Licence of the ported code: MIT, as llama.cpp.
  Copyright (c) 2023-2026 The ggml authors
  Permission is hereby granted, free of charge, to any person obtaining a copy of this
  software and associated documentation files (the "Software"), to deal in the Software
  without restriction, including without limitation the rights to use, copy, modify,
  merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to the following
  conditions: The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
  PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
  CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
  OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Inputs
  --checkpoint DIR    the publisher's checkpoint: config.json, mtp.safetensors and
                      the model shard holding model.llm.unembed.weight and
                      model.llm.embed_norm.weight (both plain bfloat16)
  --target-gguf FILE  an Inkling 2 GGUF of the same model; its tokenizer.* metadata
                      is copied (the pinned converter has no Inkling 2 tokenizer)
  --outfile FILE      the drafter GGUF
  --outtype           bf16 (the source's bits, default), f16 or f32 for the 2-D weights
  --markov-w1         f32 (default) or as-outtype: markov_w1's type. F32 is the bfloat16 widened (exact,
                      2 x the bytes: 204,859,392 B for Inkling 2): the OpenCL backend takes an F32 GET_ROWS
                      on the card, while a BF16 weight it declines stays in host memory and its row is a CPU
                      island every Markov step (patch 0240's measurement). The reference keeps it bfloat16.

What is computed and not copied (as in #32218)
  markov_w2.weight / .bias = unembed[:unpadded] @ hidden_out.weight / .bias, in float64
  blk.N.ctx_mix, markov_ctx_mix = softmax(fuse_weight) by row (the last row is the head's)
  mask_embd = the one mask embedding, one row per draft slot
  markov_w1 is cut to the unpadded vocabulary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np


class SafeTensors:
    """Read-only view of a .safetensors file: a JSON header, then raw little-endian data."""

    DTYPES = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "F64": np.float64}

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 256 * 1024 * 1024:
                raise ValueError(f"{path}: header of {n} bytes")
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.header.pop("__metadata__", None)

    def __contains__(self, name: str) -> bool:
        return name in self.header

    def names(self) -> list[str]:
        return list(self.header)

    def raw(self, name: str) -> tuple[np.ndarray, str]:
        info = self.header[name]
        dt = self.DTYPES.get(info["dtype"])
        if dt is None:
            raise ValueError(f"{name}: unsupported dtype {info['dtype']}")
        beg, end = info["data_offsets"]
        count = int(np.prod(info["shape"], dtype=np.int64)) if info["shape"] else 1
        if end - beg != count * np.dtype(dt).itemsize:
            raise ValueError(f"{name}: {end - beg} bytes for shape {info['shape']}")
        arr = np.memmap(self.path, dtype=dt, mode="r", offset=self.base + beg, shape=tuple(info["shape"]))
        return arr, info["dtype"]

    def f32(self, name: str) -> np.ndarray:
        arr, dtype = self.raw(name)
        return bf16_to_f32(arr) if dtype == "BF16" else np.asarray(arr, dtype=np.float32)


def bf16_to_f32(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.uint32) << 16).view(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


LAYER = {
    "input_layernorm.weight":           ("attn_norm.weight", 1),
    "post_attention_layernorm.weight":  ("ffn_norm.weight", 1),
    "dflare_norm.weight":               ("ctx_norm.weight", 1),
    "self_attn.q_norm.weight":          ("attn_q_norm.weight", 1),
    "self_attn.k_norm.weight":          ("attn_k_norm.weight", 1),
    "self_attn.k_norm_ctx.weight":      ("attn_k_ctx_norm.weight", 1),
    "self_attn.q_proj.weight":          ("attn_q.weight", 2),
    "self_attn.k_proj.weight":          ("attn_k.weight", 2),
    "self_attn.v_proj.weight":          ("attn_v.weight", 2),
    "self_attn.o_proj.weight":          ("attn_output.weight", 2),
    "self_attn.k_proj_ctx.weight":      ("attn_k_ctx.weight", 2),
    "self_attn.v_proj_ctx.weight":      ("attn_v_ctx.weight", 2),
    "mlp.gate_proj.weight":             ("ffn_gate.weight", 2),
    "mlp.up_proj.weight":               ("ffn_up.weight", 2),
    "mlp.down_proj.weight":             ("ffn_down.weight", 2),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--target-gguf", type=Path, required=True)
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--outtype", choices=("bf16", "f16", "f32"), default="bf16")
    ap.add_argument("--markov-w1", choices=("f32", "as-outtype"), default="f32")
    ap.add_argument("--gguf-py", type=Path, help="the pinned llama.cpp's gguf-py directory, when gguf is not importable")
    args = ap.parse_args()

    if args.gguf_py:
        sys.path.insert(0, str(args.gguf_py))
    import gguf

    ckpt = args.checkpoint
    out_dir = args.outfile.resolve().parent
    if out_dir == ckpt.resolve() or ckpt.resolve() in out_dir.parents:
        raise SystemExit("the output must not be written into the checkpoint directory")

    cfg = json.loads((ckpt / "config.json").read_text())
    hp = cfg.get("text_config", cfg)
    mtp = cfg.get("mtp_config")
    if not mtp:
        raise SystemExit("Inkling2 DSpark export needs the mtp_config")
    if mtp.get("markov_head_type") != "rnn_hidden" or mtp.get("q_bias") or mtp.get("o_bias") or mtp["hidden_size"] != hp["hidden_size"]:
        raise SystemExit(f"unsupported mtp_config {mtp}")

    # the ids count layer visits (loop passes included), the last id is the final norm output
    n_visits = hp["num_hidden_layers"]
    loop = hp.get("loop_config")
    if loop is not None:
        n_visits += (loop["num_passes"] - 1) * (loop["core_end"] - loop["core_start"])
    targets = mtp["target_layer_ids"]
    if len(targets) < 2 or targets != sorted(set(targets)) or targets[-1] != n_visits:
        raise SystemExit(f"target_layer_ids {targets} must be sorted and end at {n_visits}")

    n_layer = mtp["num_hidden_layers"]
    n_embd  = mtp["hidden_size"]
    n_vocab = hp["unpadded_vocab_size"]
    block   = mtp["block_size"]
    rank    = mtp["markov_rank"]

    st = SafeTensors(ckpt / "mtp.safetensors")

    # the target's lm head and embedding norm come from the model shards
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    shards: dict[str, SafeTensors] = {}

    def target(*names: str) -> np.ndarray:
        for name in names:
            if name in index:
                shard = shards.setdefault(index[name], SafeTensors(ckpt / index[name]))
                arr, dtype = shard.raw(name)
                if dtype != "BF16":
                    raise SystemExit(f"{name}: {dtype}, expected plain bfloat16")
                return arr
        raise SystemExit(f"none of {names} in the checkpoint")

    unembed    = target("model.llm.unembed.weight", "model.language_model.unembed.weight", "lm_head.weight")
    embed_norm = target("model.llm.embed_norm.weight", "model.language_model.embed_norm.weight",
                        "model.language_model.embed_tokens.embed_norm.weight")
    if unembed.shape[1] != n_embd or unembed.shape[0] < n_vocab:
        raise SystemExit(f"unembed {unembed.shape} does not match ({n_vocab}+, {n_embd})")

    qtype = {"bf16": gguf.GGMLQuantizationType.BF16, "f16": gguf.GGMLQuantizationType.F16, "f32": gguf.GGMLQuantizationType.F32}[args.outtype]
    ftype = {"bf16": gguf.LlamaFileType.MOSTLY_BF16, "f16": gguf.LlamaFileType.MOSTLY_F16, "f32": gguf.LlamaFileType.ALL_F32}[args.outtype]

    w = gguf.GGUFWriter(args.outfile, "dflash")
    used: set[str] = set()

    def add_f32(name: str, arr: np.ndarray) -> None:
        w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))

    def add_2d(name: str, raw: np.ndarray, dtype: str = "BF16") -> None:
        # raw: the source's bfloat16 bits (uint16) or float32 values
        if dtype == "BF16" and args.outtype == "bf16":
            w.add_tensor(name, np.ascontiguousarray(raw), raw_shape=raw.shape, raw_dtype=qtype)
            return
        vals = bf16_to_f32(raw) if dtype == "BF16" else np.asarray(raw, dtype=np.float32)
        if args.outtype == "bf16":
            # round to nearest even, like ggml's conversion
            u = np.ascontiguousarray(vals).view(np.uint32)
            bits = ((u + (0x7fff + ((u >> 16) & 1))) >> 16).astype(np.uint16)
            w.add_tensor(name, bits, raw_shape=bits.shape, raw_dtype=qtype)
        elif args.outtype == "f16":
            w.add_tensor(name, vals.astype(np.float16))
        else:
            w.add_tensor(name, np.ascontiguousarray(vals))

    def src(name: str) -> tuple[np.ndarray, str]:
        used.add(name)
        return st.raw(name)

    # ---- metadata (set_gguf_parameters of #32218)
    w.add_type("model")
    w.add_name(cfg.get("_name_or_path") or "Inkling-2-Mini DSpark")
    w.add_block_count(n_layer)
    w.add_context_length(mtp["max_position_embeddings"])
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(mtp["intermediate_size"])
    w.add_head_count(mtp["num_attention_heads"])
    w.add_head_count_kv(mtp["num_key_value_heads"])
    w.add_key_length(mtp["head_dim"])
    w.add_value_length(mtp["head_dim"])
    w.add_rope_dimension_count(mtp["head_dim"])
    w.add_rope_freq_base(float(mtp["rope_theta"]))
    w.add_layer_norm_rms_eps(mtp["rms_norm_eps"])
    # the block sees W positions before the anchor, llama.cpp's sliding window sees n_swa - 1
    w.add_sliding_window(mtp["sliding_window_size"] + 1)
    w.add_sliding_window_pattern([i in mtp["local_layer_ids"] for i in range(n_layer)])
    w.add_final_logit_softcapping(hp["final_logit_softcapping"])
    w.add_block_size(block)
    w.add_target_layers(targets)
    # slot 0 is the anchor, it does not predict a token
    w.add_sample_from_anchor(False)
    w.add_has_confidence_head(False)
    w.add_bool("dflash.has_markov_context", True)
    w.add_file_type(ftype)
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)

    # ---- tokenizer: copied from the target GGUF
    reader = gguf.GGUFReader(args.target_gguf)
    n_tok_keys = 0
    for key, field in reader.fields.items():
        if not key.startswith("tokenizer."):
            continue
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            w.add_key_value(key, field.contents(), vtype, sub_type=field.types[-1])
        else:
            w.add_key_value(key, field.contents(), vtype)
        n_tok_keys += 1
    tokens = reader.fields.get("tokenizer.ggml.tokens")
    if tokens is None or len(tokens.data) != n_vocab:
        raise SystemExit(f"the target GGUF's vocabulary does not have {n_vocab} tokens")
    if "tokenizer.ggml.mask_token_id" not in reader.fields:
        # the block uses mask_embd, the mask token id is not used
        w.add_mask_token_id(0)
    del reader

    # ---- root tensors
    add_f32("token_embd_norm.weight", bf16_to_f32(embed_norm))
    add_f32("output_norm.weight", st.f32("norm.weight")); used.add("norm.weight")

    raw, dt = src("markov_head.markov_w1.weight")
    if raw.shape != (unembed.shape[0], rank) and raw.shape != (n_vocab, rank):
        raise SystemExit(f"markov_w1 {raw.shape}")
    if args.markov_w1 == "f32":
        # widened, exact: bf16_to_f32 is a 16-bit shift (an F32 source is copied as is)
        add_f32("markov_w1.weight", bf16_to_f32(raw[:n_vocab]) if dt == "BF16" else np.asarray(raw[:n_vocab], dtype=np.float32))
    else:
        add_2d("markov_w1.weight", raw[:n_vocab], dt)

    raw, dt = src("markov_head.rnn_proj.weight")
    if raw.shape != (3 * rank, 2 * n_embd + 2 * rank):
        raise SystemExit(f"rnn_proj {raw.shape}")
    add_2d("joint_proj.weight", raw, dt)
    add_f32("joint_proj.bias", st.f32("markov_head.rnn_proj.bias")); used.add("markov_head.rnn_proj.bias")
    add_f32("markov_ctx_norm.weight", st.f32("markov_head.dflare_norm.weight")); used.add("markov_head.dflare_norm.weight")

    # markov_w2 = unembed @ hidden_out in float64 like the reference, one row chunk at a time;
    # padded tokens are never sampled, so only the tokenizer's rows are kept
    h_w = st.f32("markov_head.hidden_out.weight").astype(np.float64); used.add("markov_head.hidden_out.weight")
    h_b = st.f32("markov_head.hidden_out.bias").astype(np.float64);   used.add("markov_head.hidden_out.bias")
    if h_w.shape != (n_embd, rank) or h_b.shape != (n_embd,):
        raise SystemExit(f"hidden_out {h_w.shape} does not match the unembed {(n_vocab, n_embd)}")
    w2   = np.empty((n_vocab, rank), dtype=np.float32)
    w2_b = np.empty((n_vocab,), dtype=np.float32)
    for i in range(0, n_vocab, 4096):
        head = bf16_to_f32(unembed[i:min(i + 4096, n_vocab)]).astype(np.float64)
        w2[i:i + head.shape[0]]   = (head @ h_w).astype(np.float32)
        w2_b[i:i + head.shape[0]] = (head @ h_b).astype(np.float32)
    # a product has no bfloat16 source: f32 unless another type was asked for
    if args.outtype == "bf16":
        w.add_tensor("markov_w2.weight", w2)
    else:
        add_2d("markov_w2.weight", w2, "F32")
    add_f32("markov_w2.bias", w2_b)

    fuse = st.f32("fuse_weight"); used.add("fuse_weight")
    if fuse.shape != (n_layer + 1, len(targets)):
        raise SystemExit(f"unexpected fuse_weight shape {fuse.shape}")
    # one row per draft layer, the last row is for the markov head
    alpha = softmax(fuse)
    add_f32("markov_ctx_mix.weight", alpha[-1])

    mask = st.f32("mask_embed"); used.add("mask_embed")
    if mask.shape != (n_embd,):
        raise SystemExit(f"mask_embed {mask.shape}")
    # the one mask embedding, copied to each mask slot
    add_f32("mask_embd.weight", np.broadcast_to(mask, (block - 1, n_embd)))

    # ---- layers
    for i in range(n_layer):
        add_f32(f"blk.{i}.ctx_mix.weight", alpha[i])
        for key, (dst, ndim) in LAYER.items():
            raw, dt = src(f"layers.{i}.{key}")
            if raw.ndim != ndim:
                raise SystemExit(f"layers.{i}.{key}: shape {raw.shape}")
            if ndim == 1:
                add_f32(f"blk.{i}.{dst}", bf16_to_f32(raw) if dt == "BF16" else raw)
            else:
                add_2d(f"blk.{i}.{dst}", raw, dt)

    left = sorted(set(st.names()) - used)
    if left:
        raise SystemExit(f"unexpected tensors in mtp.safetensors: {left}")

    n_tensors = sum(len(t) for t in w.tensors)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    h = hashlib.sha256()
    with open(args.outfile, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    print(f"{args.outfile.name}: {args.outfile.stat().st_size} bytes, sha256 {h.hexdigest()}, "
          f"{n_tensors} tensors, {n_tok_keys} tokenizer keys copied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
