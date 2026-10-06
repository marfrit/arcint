#!/usr/bin/env python3
"""The 0.5.1 instrument-resolution bound, derived from THIS model's own
reference round-trip (docs/window-051.md clause (d)) -- a stated multiple of
the reference writer's own uint16 rounding error, never an imported number.

    F_ref           mean over real f32 logit rows of KL(P_row || P_roundtrip),
                    P_roundtrip = the row written by the llama.cpp perplexity
                    writer's transcription (`kld_served.llama_row`) and read back
                    by `kld_served.reference_log_probs` -- the capture's own
                    resolution, measured on rows of the served path's width.
                    Its PER-ROW MAX is printed too: the clamp tail can exceed
                    the bound (the mean bound is not a per-row bound).
    bar_below       = multiple x F_ref                      (rows below 2051)
    bar_above       = bar_below + the measured QSA price     (rows at/above 2051)
    floor pair KL(A||B) between two forwards of the SAME rows, printed beside
                    the bound: a bound under the floor is UNREADABLE, never PASS

WHAT THIS IS NOT. `bar_below`/`bar_above` are an INSTRUMENT-RESOLUTION BOUND,
not an acceptance bar: they ask the served path to reproduce the reference
within 100 x the capture's own transcription error, and the reference
implementation itself (llama.cpp) reads ~110 x above `bar_below` (mean 0.3387
vs 3.0905e-03), while F_ref's own per-row max (1.0533e-02) also exceeds it.
The ACCEPTANCE candidate, printed in the report, is the floor between
implementations: llama.cpp's error against the same f32 reference,
median 0.0649 (w0) / 0.0283 (w1). No verdict is quotable without `F_served`.
[2026-09-28] Those medians are whole-window figures against the f32 reference,
whose rows at or above 2,051 ran a garbage indexer (a gguf_feed BF16 defect,
fixed): void as quoted until the re-capture (docs/campaigns/qsa.md).

    kld_bar.py --rows forward_1.npy [--pair forward_2.npy] [--multiple 100]
               [--qsa-price 2.385560e-02] [--out bar.json]

The rows come from `tools/boot_serving_shape.py --dump-logits DIR` ([T, vocab]
f32, one file per forward). The report is PROVISIONAL until the leg that
produced the rows is dated in the window document with its card, depth,
precision and token window.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kld_served as ks  # noqa: E402

INHERITED_BAR_PROVISIONAL = 0.0599      # 1.5 x another model's R0 (window-050 §7)
QSA_PRICE_DEFAULT = 2.385560e-02        # window-050 §8, over 29/2080 rows
# [2026-09-28] a max-abs block-output figure, not nats, drawn with the q/k gamma
# folds applied twice; bar_above built from it is withdrawn (docs/campaigns/qsa.md).


def f_ref(rows):
    """KL(P_row || P_roundtrip) per row through the writer's transcription."""
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError(f"rows must be [T, vocab], got {rows.shape}")
    vocab = rows.shape[1]
    kls = []
    for r in rows:
        back = ks.reference_log_probs(ks.llama_row(r), vocab)
        kls.append(ks.kl_ref_vs_served(ks.log_softmax(r.astype(np.float64)), back))
    kls = np.asarray(kls, dtype=np.float64)
    return {"rows": int(rows.shape[0]), "vocab": int(vocab),
            "f_ref_mean": float(kls.mean()), "f_ref_max": float(kls.max()),
            "f_ref_min": float(kls.min())}


def floor_pair(a, b):
    """KL(A||B) per row between two forwards of the same rows, mean and max,
    the argmax agreement and the count of rows whose argmax moved."""
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"pair shapes differ: {a.shape} vs {b.shape}")
    kls = np.asarray([ks.kl_ref_vs_served(ks.log_softmax(x.astype(np.float64)), y)
                      for x, y in zip(a, b)], dtype=np.float64)
    moved = int((a.argmax(axis=1) != b.argmax(axis=1)).sum())
    return {"rows": int(a.shape[0]), "kl_ab_mean": float(kls.mean()),
            "kl_ab_max": float(kls.max()), "rows_moved": moved,
            "argmax_agreement": float(1.0 - moved / a.shape[0]),
            "bit_identical": bool(np.array_equal(a, b)),
            "max_abs_diff": float(np.abs(a - b).max())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="forward_K.npy, [T, vocab] f32")
    ap.add_argument("--pair", default=None, help="a second forward of the same rows")
    ap.add_argument("--multiple", type=float, default=100.0)
    ap.add_argument("--qsa-price", type=float, default=QSA_PRICE_DEFAULT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rows = np.load(args.rows, mmap_mode="r")
    rep = f_ref(rows)
    rep["multiple"] = args.multiple if args.multiple != int(args.multiple) else int(args.multiple)
    rep["bar_below_2051"] = args.multiple * rep["f_ref_mean"]
    rep["qsa_price"] = args.qsa_price
    rep["bar_at_or_above_2051"] = rep["bar_below_2051"] + args.qsa_price
    rep["inherited_bar_provisional"] = INHERITED_BAR_PROVISIONAL
    rep["bound_kind"] = ("instrument-resolution bound (window-051 clause (d)); "
                         "NOT an acceptance bar, and a MEAN bound only")
    rep["bound_is_acceptance"] = False
    rep["per_row_clamp_tail_exceeds_bound"] = rep["f_ref_max"] > rep["bar_below_2051"]
    rep["acceptance_candidate"] = {
        "kind": "between-implementations floor (llama.cpp vs the same f32 reference)",
        "median_w0_nats": 0.0649, "median_w1_nats": 0.0283,
        "mean_w0_nats": 0.3387, "mean_w1_nats": 0.4415,
        "provenance": ("window-051.md record, 2026-09-19 08:36: llama.cpp build "
                       "56b9eb28 against the model's own f32 reference capture; "
                       "void as quoted since 2026-09-28 (the reference's rows at or "
                       "above 2,051 ran a garbage indexer; docs/campaigns/qsa.md)")}
    rep["floor_pair"] = floor_pair(rows, np.load(args.pair, mmap_mode="r")) if args.pair else None
    rep["rows_file"] = str(args.rows)
    rep["pair_file"] = str(args.pair) if args.pair else None
    rep["status"] = "PROVISIONAL until the dated leg names its rows"
    text = json.dumps(rep, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
