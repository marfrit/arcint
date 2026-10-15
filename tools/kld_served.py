#!/usr/bin/env python3
"""The llama.cpp reference-capture reader, and the per-row KL against it.

`tools/kld_capture.py` builds a capture with `llama-perplexity
--kl-divergence-base`; `tools/capture_vs_capture.py` compares two captures and
`tools/ref_forward_stream.py` writes one in the same format. This module is
the shared reader: `read_capture` (the file), `reference_log_probs` (one
recorded row to log-probs; the writer is `log_softmax(int, const float*,
uint16_t*, int)` in llama.cpp's tools/perplexity/perplexity.cpp, transcribed
here as `llama_row`), `kl_ref_vs_served` (KL(P_ref || P_other) for one row).

The cells that enforce the claims above are tests/python/test_kld_served.py
(reconstruction to the quantisation step; KL zero for the same row and
positive for a moved one; the reader refusing a file that is not a capture).
The half that read a served-logits dump left with the retired engine
(docs/history.md).
"""
import struct

import numpy as np

# Rows at 0-based index < 2051 and >= 2051 were reported apart for Flash-Next
# (its 12 full-attention layers' boundary); capture_vs_capture.py splits there.
QSA_BOUNDARY_TOKENS = 2051


# ---------------------------------------------------------------- the capture
def read_capture(path):
    """The llama.cpp `--kl-divergence-base` file. Returns (n_ctx, n_vocab,
    n_chunk, tokens [n_chunk, n_ctx] int32, rows [n_chunk, n_rows, nv]
    uint16 memmap) with nv = 2 * ((n_vocab + 1) // 2) + 4 and n_rows =
    n_ctx - 1 - n_ctx // 2."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"_logits_":
            raise ValueError(f"{path}: not a llama.cpp logits capture (magic {magic!r})")
        n_ctx = struct.unpack("<i", f.read(4))[0]
        n_vocab = struct.unpack("<i", f.read(4))[0]
        n_chunk = struct.unpack("<i", f.read(4))[0]
        tokens = np.fromfile(f, dtype=np.int32, count=n_chunk * n_ctx).reshape(n_chunk, n_ctx)
        off = f.tell()
    first = n_ctx // 2
    n_rows = n_ctx - 1 - first
    nv = 2 * ((n_vocab + 1) // 2) + 4
    rows = np.memmap(path, dtype=np.uint16, mode="r", offset=off,
                     shape=(n_chunk, n_rows, nv))
    return n_ctx, n_vocab, n_chunk, tokens, rows


def llama_row(logits):
    """llama.cpp perplexity.cpp `log_softmax(int, const float*, uint16_t*, int)`,
    transcribed: the writer of every recorded row."""
    logits = np.asarray(logits, dtype=np.float32)
    max_logit = float(logits.max())
    min_logit = max(float(logits.min()), max_logit - 16)
    log_sum_exp = float(np.log(np.exp(logits - max_logit).sum()))
    min_log_prob = min_logit - max_logit - log_sum_exp
    scale = (max_logit - min_logit) / 65535.0
    row = np.zeros(2 * ((len(logits) + 1) // 2) + 4, dtype=np.uint16)
    row[:4] = np.array([scale, min_log_prob], dtype=np.float32).view(np.uint16)
    if scale:
        q = np.where(logits > min_logit, np.rint((logits - min_logit) / scale), 0)
        row[4:4 + len(logits)] = q.astype(np.uint16)
    return row


def reference_log_probs(row_u16, n_vocab):
    """One recorded row -> log P_ref over the vocab, f64. The writer stores
    scale (f32), min_log_prob (f32), then q[i] = round((logit_i - min_logit)
    / scale) for logit_i > min_logit else 0, with min_logit clamped to
    max_logit - 16; so log P(i) = min_log_prob + q[i] * scale, and a q of 0
    is the floor (16 nats below the max, or lower)."""
    hdr = np.asarray(row_u16[:4]).view(np.float32)
    scale, min_log_prob = float(hdr[0]), float(hdr[1])
    q = np.asarray(row_u16[4:4 + n_vocab], dtype=np.float64)
    return min_log_prob + q * scale


def log_softmax(x):
    z = x - x.max()
    return z - np.log(np.exp(z).sum())


def kl_ref_vs_served(ref_logp, served_logits):
    """KL(P_ref || P_served) in nats for one row. P_ref from the capture's
    reconstructed log-probs (renormalised: the uint16 floor makes them sum to
    slightly under 1), P_served from a log-softmax of the served f32 row."""
    ref = log_softmax(np.asarray(ref_logp, dtype=np.float64))
    cand = log_softmax(np.asarray(served_logits, dtype=np.float64))
    p = np.exp(ref)
    return float((p * (ref - cand)).sum())
