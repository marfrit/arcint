"""KL between two captures of the same window in llama's format: the f32
reference (window 0) against llama.cpp's own capture (window W) -- the floor
between two implementations of the same checkpoint, per bucket. 2026-09-19:
llama.cpp reads mean 0.34 / median 0.065 nats against the model's own f32
arithmetic on window 0; the native artifact 0.37 / 0.18.
  capture_vs_capture.py REF_CAPTURE OTHER_CAPTURE OTHER_WINDOW
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
from kld_served import read_capture, reference_log_probs, log_softmax, QSA_BOUNDARY_TOKENS
ref, lla, w = sys.argv[1], sys.argv[2], int(sys.argv[3])
n_ctx, n_vocab, _, tok_r, rows_r = read_capture(ref)
n2, v2, _, tok_l, rows_l = read_capture(lla)
assert (n_ctx, n_vocab) == (n2, v2) and np.array_equal(tok_r[0], tok_l[w]), "captures differ in tokens"
first = n_ctx // 2; n_rows = n_ctx - 1 - first
kl = np.empty(n_rows); agree = np.empty(n_rows, dtype=bool)
for i in range(n_rows):
    r = log_softmax(reference_log_probs(rows_r[0, i], n_vocab)); l = log_softmax(reference_log_probs(rows_l[w, i], n_vocab))
    kl[i] = float((np.exp(r) * (r - l)).sum()); agree[i] = np.argmax(r) == np.argmax(l)
pos = np.arange(n_rows) + first; below = pos < QSA_BOUNDARY_TOKENS
print(f"llama.cpp capture window {w} vs the f32 reference: mean KL {kl.mean():.4f} (below {kl[below].mean():.4f} / above {kl[~below].mean():.4f}), median {np.median(kl):.4f}, max {kl.max():.3f}, argmax agreement {agree.mean():.4f}")
for lo in range(first - first % 128, n_ctx, 128):
    m = (pos >= lo) & (pos < lo + 128)
    if m.any(): print(f"   {lo:>5}-{lo+127:<5} rows {int(m.sum()):>3} median {np.median(kl[m]):.4f} mean {kl[m].mean():.4f} argmax {agree[m].mean():.3f}")
