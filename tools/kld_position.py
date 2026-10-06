"""Position-resolved KL of a served replay against the model's own capture:
the per-row KL of `kld_served.py --compare`, bucketed by position, so a
residual that grows with context (attention, KV precision, chunk seams) can
be told from one that is flat in position (per-token depth accumulation).
The 2026-09-18 reading of the u4 artifact was done inline from the dump;
this makes it a tool.

  kld_position.py --ref CAPTURE --dump DUMP [--bucket 128] [--window W]

Prints, per bucket: rows, median KL, mean KL, argmax agreement. Positions
are 0-based row indices in the window; only the scored half (the capture
scores rows from n_ctx // 2 on) is present, as in the compare.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kld_served import (QSA_BOUNDARY_TOKENS, dump_windows, kl_ref_vs_served,  # noqa: E402
                        read_capture, read_dump, reference_log_probs, window_rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--bucket", type=int, default=128)
    ap.add_argument("--window", type=int, default=0, help="which replayed window (capture order)")
    args = ap.parse_args(argv)
    n_ctx, n_vocab, n_chunk, tokens, rows = read_capture(args.ref)
    recs = read_dump(args.dump)
    replays = [w for w in dump_windows(recs)
               if sum(r[3] for r in w if r[1] < n_ctx and r[1] + r[3] <= n_ctx) == n_ctx]
    if args.window >= len(replays):
        raise SystemExit(f"dump holds {len(replays)} replayed window(s); --window {args.window} is out of range")
    win = replays[args.window]
    served = window_rows(args.dump, win, n_ctx, n_vocab)
    first = n_ctx // 2
    n_rows = n_ctx - 1 - first
    kl = np.empty(n_rows)
    agree = np.empty(n_rows, dtype=bool)
    for i in range(n_rows):
        p = first + i
        ref_lp = reference_log_probs(rows[args.window, i], n_vocab)
        kl[i] = kl_ref_vs_served(ref_lp, served[p])
        agree[i] = np.argmax(ref_lp) == np.argmax(served[p])
    print(f"window {args.window}: n_ctx {n_ctx}, scored rows {n_rows} from position {first}; "
          f"QSA boundary {QSA_BOUNDARY_TOKENS}; mean KL {kl.mean():.4f} median {np.median(kl):.4f} "
          f"argmax agreement {agree.mean():.4f}")
    print(f"{'positions':<14} {'rows':>5} {'median KL':>10} {'mean KL':>9} {'max KL':>8} {'argmax':>7}")
    b = args.bucket
    for lo in range(first - first % b, n_ctx, b):
        m = (np.arange(n_rows) + first >= lo) & (np.arange(n_rows) + first < lo + b)
        if not m.any():
            continue
        print(f"{lo:>6}-{lo + b - 1:<6} {int(m.sum()):>5} {np.median(kl[m]):>10.4f} {kl[m].mean():>9.4f} "
              f"{kl[m].max():>8.3f} {agree[m].mean():>7.3f}")
    # the chunk seams: rows just after each 512-token boundary vs the rest
    seam = ((np.arange(n_rows) + first) % 512) < 8
    print(f"chunk-seam rows (first 8 after each 512 boundary): {int(seam.sum())} rows, median KL "
          f"{np.median(kl[seam]):.4f} vs {np.median(kl[~seam]):.4f} elsewhere")
    return 0


if __name__ == "__main__":
    sys.exit(main())
