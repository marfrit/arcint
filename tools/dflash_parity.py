#!/usr/bin/env python3
"""Part C: parity of the OV-exported DFlash2 draft against the torch probe.

Replays the teacher-forced acceptance loop of partB_probe.py, but the draft
forward runs through the compiled OV graph (CPU), logits through the
artifact's extracted lm_head (OV), and the selector in numpy from the
sidecar. Gate: acceptance within noise of the torch arms (3.39 code / 3.76
prose), plus a one-cycle max-abs check of the draft hidden against torch.
"""
import argparse
import json
import sys

import numpy as np
import openvino as ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--draft", default="/models/gptq/qwen38-dflash2")
    ap.add_argument("--artifact", default="/models/ov/qwen38-b7c1-ov")
    ap.add_argument("--torch-check", action="store_true",
                    help="compare cycle-1 hidden against the torch reimpl")
    ap.add_argument("--stateful", action="store_true",
                    help="drive the stateful export incrementally")
    args = ap.parse_args()

    core = ov.Core()
    import os
    dev = os.environ.get("DFLASH_DEV", "CPU")
    xml = "openvino_dflash_draft_stateful.xml" if args.stateful else "openvino_dflash_draft.xml"
    props = {}
    if os.environ.get("DFLASH_F32"):
        props["INFERENCE_PRECISION_HINT"] = "f32"
    draft = core.compile_model(f"{args.draft}/{xml}", dev, props)
    lm_head = core.compile_model(f"{args.artifact}/openvino_mtp_lm_head.xml", "CPU")
    sel = np.load(f"{args.draft}/dflash_selector.npz")
    hp_w = sel["hidden_projection"].astype(np.float32)
    pred_cb = sel["predecessor_codebook"].astype(np.float32)
    succ_cb = sel["successor_codebook"].astype(np.float32)
    top_k = int(sel["top_k"])
    block = int(sel["block_size"])

    d = np.load(args.npz)
    tokens = d["tokens"]
    p0 = int(d["prompt_len"])
    feats = d["feats"].astype(np.float32).reshape(d["feats"].shape[0], -1)
    scale = float(os.environ.get("DFLASH_SCALE", "1"))
    if scale != 1.0:
        feats = feats / scale   # RMSNorm after fc is scale-invariant; this only
                                 # moves the values out of f16 overflow range
    embeds = d["embeds"].astype(np.float32)
    mask_embed = d["mask_embed"].astype(np.float32)

    req = draft.create_infer_request()

    def draft_hidden(ctx, noise, positions):
        req.set_tensor("ctx_feats", ov.Tensor(ctx[None]))
        req.set_tensor("noise", ov.Tensor(noise[None]))
        req.set_tensor("positions", ov.Tensor(positions))
        req.infer()
        return req.get_output_tensor(0).data[0].copy()          # [Q, 5120]

    def propose(rows, anchor):
        lg = lm_head(rows[None].astype(np.float32))[0].reshape(rows.shape[0], -1)
        cand = np.argpartition(-lg, top_k, axis=-1)[:, :top_k]  # [Q-1, k]
        unary = np.take_along_axis(lg, cand, axis=-1)
        hp = rows @ hp_w.T                                       # [Q-1, r]
        pred = anchor
        path = []
        for pos in range(rows.shape[0]):
            s = unary[pos] + (pred_cb[pred] * hp[pos]) @ succ_cb[cand[pos]].T
            pred = int(cand[pos][np.argmax(s)])
            path.append(pred)
        return path

    def draft_hidden_stateful(new_feats, noise, positions):
        req.set_tensor("new_feats", ov.Tensor(new_feats[None]))
        req.set_tensor("noise", ov.Tensor(noise[None]))
        req.set_tensor("positions", ov.Tensor(positions))
        req.infer()
        return req.get_output_tensor(0).data[0].copy()

    T = len(tokens)
    start, cycles, accepted_total = p0, 0, 0
    fed = 0
    if args.stateful:
        req.reset_state()
    first_cycle_rows = None
    while start + 1 < T - 1:
        q = min(block, T - start)
        if q < 2:
            break
        noise = np.concatenate([embeds[start][None],
                                np.tile(mask_embed[None], (q - 1, 1))])
        if args.stateful:
            new = feats[fed:start]
            positions = np.concatenate([np.arange(fed, start), np.arange(start, start + q)]).astype(np.int64)
            hid = draft_hidden_stateful(new, noise, positions)[1 - q:]
            fed = start
        else:
            ctx = feats[:start]
            positions = np.arange(0, start + q, dtype=np.int64)
            hid = draft_hidden(ctx, noise, positions)[1 - q:]
        if first_cycle_rows is None and not args.stateful:
            first_cycle_rows = (ctx.copy(), noise.copy(), positions.copy(), hid.copy())
        path = propose(hid, int(tokens[start]))
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
    print(f"OV parity: cycles {cycles}, accepted {accepted_total}, "
          f"acceptance length {(accepted_total + cycles) / max(1, cycles):.2f}")

    if args.torch_check and first_cycle_rows is not None:
        sys.path.insert(0, "/tmp")
        import torch
        from partB_probe import Draft
        cfg = json.load(open(f"{args.draft}/config.json"))
        td = Draft(f"{args.draft}/model.safetensors", cfg)
        ctx, noise, positions, hid = first_cycle_rows
        th = td.forward(torch.from_numpy(ctx)[None], torch.from_numpy(noise)[None],
                        torch.from_numpy(positions))[0].numpy()[1:]
        diff = np.abs(th - hid)
        print(f"cycle-1 hidden vs torch: max abs {diff.max():.4f}, "
              f"mean abs {diff.mean():.5f} (f16-compressed export vs f32 torch)")


if __name__ == "__main__":
    main()
