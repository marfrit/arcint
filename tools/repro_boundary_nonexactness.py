#!/usr/bin/env python3
"""Minimal reproducer: on the OpenVINO GPU stateful path, advancing a hybrid
GDN model's state by k tokens in one forward is not the same computation as
advancing it k times by one.

This is the measurement behind three decisions in DESIGN.md:

  §3.2  chunked prefill is not bit-exact, and prefill therefore chunks on an
        absolute grid so that every path splits a sequence the same way
  §3.4  a prefix-cache hit is a boundary, so checkpoints are restricted to that
        same grid -- without it, warm-vs-cold equality held on argmax margin
  §3.5  speculative decoding cannot be byte-identical to plain greedy, because
        its verify pass is a multi-token forward by definition

Run against any of the three target artifacts:

    python3 tools/repro_boundary_nonexactness.py /models/ov/qwen38-b7c1-ov GPU.0

Measured 2026-08-28 on an Arc Pro B60, OpenVINO 2026.4, xe KMD, with
qwen38-b7c1-ov:

    row 0 of forward([x, y])  vs forward([x])                  identical
    last row, batched vs sequential                            differs, 0.013
    cold (one 235-token forward) vs warm (224 then 11)         differs, 0.210
      ... with a top-2 margin at that position of              0.145
    both paths split at the same boundary                      identical

The last line is the point: the backend is allowed to be chunk-sensitive, and
equality comes back by construction as soon as no two paths hand the model a
different split of the same tokens.
"""
import sys
import numpy as np
import openvino as ov


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "/models/ov/qwen38-b7c1-ov"
    dev = sys.argv[2] if len(sys.argv) > 2 else "GPU.0"

    core = ov.Core()
    lm = core.compile_model(core.read_model(base + "/openvino_language_model.xml"), dev)
    emb = core.compile_model(base + "/openvino_text_embeddings_model.xml", dev)
    r, e = lm.create_infer_request(), emb.create_infer_request()

    def embed(ids):
        return e.infer({0: np.array(ids, dtype=np.int64).reshape(1, -1)})[0]

    def fwd(ids, past):
        n = len(ids)
        pos = np.tile(np.arange(past, past + n, dtype=np.int64).reshape(1, 1, n), (4, 1, 1))
        return r.infer({"inputs_embeds": embed(ids), "position_ids": pos,
                        "attention_mask": np.ones((1, past + n), dtype=np.int64),
                        "beam_idx": np.zeros(1, dtype=np.int32)})["logits"]

    rng = np.random.default_rng(7)
    toks = [int(t) for t in rng.integers(1000, 40000, size=235)]

    # ---- 1. same position, one pass vs two: is it the batching or the split? --
    r.reset_state()
    fwd(toks[:15], 0)
    snap = [s.state.data.copy() for s in r.query_state()]

    def rewind():
        for s, v in zip(r.query_state(), snap):
            s.state = ov.Tensor(v)

    rewind(); a0 = fwd([toks[15]], 15)[0, -1].copy()
    y = int(a0.argmax())
    rewind(); row0 = fwd([toks[15], y], 15)[0, 0].copy()
    print(f"  row 0, batched vs alone      : max diff {np.abs(a0 - row0).max():.6f}")

    rewind(); fwd([toks[15]], 15); seq = fwd([y], 16)[0, -1].copy()
    rewind(); bat = fwd([toks[15], y], 15)[0, -1].copy()
    print(f"  last row, batched vs sequential: max diff {np.abs(seq - bat).max():.6f}")

    # ---- 2. the case that matters: a cache hit is a boundary -----------------
    HIT = 224
    r.reset_state(); cold = fwd(toks, 0)[0, -1].copy()
    r.reset_state(); fwd(toks[:HIT], 0); warm = fwd(toks[HIT:], HIT)[0, -1].copy()
    top = np.argsort(cold)[::-1][:2]
    print(f"  cold vs warm                 : max diff {np.abs(cold - warm).max():.6f}"
          f"  (top-2 margin {float(cold[top[0]] - cold[top[1]]):.4f})")

    # ---- 3. the fix: put both paths on the same grid -------------------------
    r.reset_state(); fwd(toks[:HIT], 0); g1 = fwd(toks[HIT:], HIT)[0, -1].copy()
    r.reset_state(); fwd(toks[:HIT], 0); g2 = fwd(toks[HIT:], HIT)[0, -1].copy()
    print(f"  same boundary on both paths  : max diff {np.abs(g1 - g2).max():.6f}")


if __name__ == "__main__":
    main()
