"""A served logits dump against ANY capture in llama's format -- the model's own
f32 reference (tools/ref_forward_stream.py --write-capture) or llama.cpp's --
one window at a time, the dump's window chosen explicitly (kld_served.py's
compare takes the LAST n_chunk replays, which is wrong for a one-window
reference against a two-window dump): mean/median KL, argmax, per 128-token
bucket, split at the sparse-attention boundary. The 2026-09-19 gate re-read.
  kld_vs_capture.py CAPTURE DUMP DUMP_WINDOW_INDEX LABEL
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
from kld_served import read_capture, read_dump, dump_windows, window_rows, reference_log_probs, kl_ref_vs_served, QSA_BOUNDARY_TOKENS

ref, dump, wsel, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
n_ctx, n_vocab, n_chunk, tokens, rows = read_capture(ref)
recs = read_dump(dump)
replays = [w for w in dump_windows(recs) if sum(r[3] for r in w if r[1] < n_ctx and r[1] + r[3] <= n_ctx) == n_ctx]
win = replays[wsel]
served = window_rows(dump, win, n_ctx, n_vocab)
first = n_ctx // 2
n_rows = n_ctx - 1 - first
kl = np.empty(n_rows); agree = np.empty(n_rows, dtype=bool)
for i in range(n_rows):
    p = first + i
    ref_lp = reference_log_probs(rows[0, i], n_vocab)          # the reference capture holds ONE window
    kl[i] = kl_ref_vs_served(ref_lp, served[p])
    agree[i] = np.argmax(ref_lp) == np.argmax(served[p])
pos = np.arange(n_rows) + first
below = pos < QSA_BOUNDARY_TOKENS
print(f"{label}: dump window {wsel} of {len(replays)} vs the f32 reference: mean KL {kl.mean():.4f} (below {kl[below].mean():.4f} / above {kl[~below].mean():.4f}), "
      f"median {np.median(kl):.4f}, max {kl.max():.3f}, argmax agreement {agree.mean():.4f}")
for lo in range(first - first % 128, n_ctx, 128):
    m = (pos >= lo) & (pos < lo + 128)
    if m.any():
        print(f"   {lo:>5}-{lo+127:<5} rows {int(m.sum()):>3} median {np.median(kl[m]):.4f} mean {kl[m].mean():.4f} argmax {agree[m].mean():.3f}")
