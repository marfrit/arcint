#!/usr/bin/env python3
"""DFlash2 pairing probe, part A (OpenVINO, CPU — no card is touched).

Drives the b7c1 stateful IR greedily for N tokens and records, per input
position, the residual stream entering layers {6,20,34,48,62}'s
input_layernorm — HF's hidden_states[layer_id+1] for layer_ids
{5,19,33,47,61}, the DFlash2 context features — plus the input embeddings.

Output npz: tokens [T], prompt_len, feats [T-1, 5, 5120], embeds [T, 5120],
mask_embed [5120]. feats[t] belongs to position t (token t as input).
"""
import argparse
import sys

import numpy as np
import openvino as ov
import openvino_tokenizers  # noqa: F401  (registers SpecialTokensSplit etc.)

LAYERS = [6, 20, 34, 48, 62]   # tap = input of these layers' input_layernorm
MASK_TOKEN_ID = 248070


def tap_points(model):
    taps = {}
    for op in model.get_ops():
        n = op.get_friendly_name()
        for L in LAYERS:
            key = f"language_model.layers.{L}.input_layernorm"
            if key in n and L not in taps:
                # walk to the producer outside the layernorm namespace
                src = op
                guard = 0
                while key in src.get_friendly_name() and guard < 8:
                    src = src.input(0).get_source_output().get_node()
                    guard += 1
                taps[L] = src.output(0)
    assert len(taps) == len(LAYERS), f"taps found: {sorted(taps)}"
    return [taps[L] for L in LAYERS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt_file")
    ap.add_argument("out_npz")
    ap.add_argument("--artifact", default="/models/ov/qwen38-b7c1-ov")
    ap.add_argument("--n", type=int, default=192)
    args = ap.parse_args()

    core = ov.Core()
    text = open(args.prompt_file).read()
    prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

    tok = core.compile_model(f"{args.artifact}/openvino_tokenizer.xml", "CPU")
    ids = tok([prompt])["input_ids"].reshape(-1).astype(np.int64)
    print("prompt tokens:", len(ids), flush=True)

    emb = core.compile_model(f"{args.artifact}/openvino_text_embeddings_model.xml", "CPU")

    def embed(token_ids):
        return emb(np.asarray(token_ids, dtype=np.int64).reshape(1, -1))[0]

    lm = core.read_model(f"{args.artifact}/openvino_language_model.xml")
    for out in tap_points(lm):
        lm.add_outputs([out])
    comp = core.compile_model(lm, "CPU")
    req = comp.create_infer_request()
    print("LM compiled on CPU", flush=True)

    n_out = len(comp.outputs)
    logits_i, feat_i = 0, list(range(n_out - 5, n_out))

    tokens = list(ids)
    feats, embeds = [], []
    past = 0

    def forward(token_ids):
        nonlocal past
        n = len(token_ids)
        e = embed(token_ids)
        embeds.append(e.reshape(n, -1))
        pos = np.tile(np.arange(past, past + n, dtype=np.int64), (4, 1, 1)).reshape(4, 1, n)
        req.set_tensor("inputs_embeds", ov.Tensor(e.reshape(1, n, -1)))
        req.set_tensor("attention_mask", ov.Tensor(np.ones((1, past + n), dtype=np.int64)))
        req.set_tensor("position_ids", ov.Tensor(pos))
        req.set_tensor("beam_idx", ov.Tensor(np.zeros(1, dtype=np.int32)))
        req.infer()
        past += n
        f = [req.get_output_tensor(i).data.reshape(n, -1) for i in feat_i]
        feats.append(np.stack(f, axis=1).astype(np.float16))     # [n, 5, 5120]
        lg = req.get_output_tensor(logits_i).data
        return int(np.argmax(lg.reshape(n, -1)[-1]))

    req.reset_state()
    nxt = forward(tokens)
    tokens.append(nxt)
    for i in range(args.n - 1):
        nxt = forward([tokens[-1]])
        tokens.append(nxt)
        if (i + 1) % 32 == 0:
            print(f"  {i + 1} tokens", flush=True)

    mask_embed = embed([MASK_TOKEN_ID]).reshape(-1)
    np.savez_compressed(
        args.out_npz,
        tokens=np.asarray(tokens, dtype=np.int64),
        prompt_len=np.int64(len(ids)),
        feats=np.concatenate(feats, axis=0),
        embeds=np.concatenate(embeds, axis=0).astype(np.float16),
        mask_embed=mask_embed.astype(np.float16),
    )
    print("saved", args.out_npz, "T =", len(tokens), flush=True)


if __name__ == "__main__":
    main()
