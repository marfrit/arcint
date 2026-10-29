#!/usr/bin/env python3
"""KL between captures of the same window in llama's format: the f32 reference
(window 0) against a served capture (window W). A comparison is judged against
the card's own run-to-run floor, so the twin mode takes two baseline captures
(same binary and configuration, two runs) beside one candidate and reports the
baseline twin's mean KL, the candidate's and the difference. An optional mask
restricts both to [start, end) token ranges, the assistant spans of a rendered
chat, and the masked and unmasked numbers are printed side by side.

  capture_vs_capture.py REF_CAPTURE OTHER_CAPTURE OTHER_WINDOW [--mask FILE]
  capture_vs_capture.py REF_CAPTURE --baseline BASE_A BASE_B \\
      --candidate CANDIDATE --window W [--mask FILE]

The reference is window 0 (llama.cpp's `--kl-divergence-base` file); the
baseline and candidate files may carry several windows, `--window` picks the
one to read. 2026-09-19: llama.cpp reads mean 0.34 / median 0.065 nats against
the model's own f32 arithmetic on window 0; the native artifact 0.37 / 0.18.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kld_served import read_capture, kl_rows, QSA_BOUNDARY_TOKENS  # noqa: E402


# ------------------------------------------------------------ position mask
def read_mask(path):
    """[start, end) token ranges, one per line; '#' starts a comment. Returns
    the list of pairs. An empty mask is refused: it would silently select no
    rows."""
    ranges = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            body = line.split("#", 1)[0].strip()
            if not body:
                continue
            parts = body.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{lineno}: a range is 'START END', got {line.strip()!r}")
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: not an integer: {e}") from e
            if start < 0 or end <= start:
                raise ValueError(f"{path}:{lineno}: [{start}, {end}) is not a non-empty range")
            ranges.append((start, end))
    if not ranges:
        raise ValueError(f"{path}: the mask has no ranges")
    return ranges


def row_positions(n_ctx):
    """The token position of each recorded row: llama.cpp records the logits of
    positions n_ctx // 2 through n_ctx - 2, one row each, so row i is token
    position (n_ctx // 2) + i."""
    first = n_ctx // 2
    return np.arange(n_ctx - 1 - first) + first


def position_mask(positions, ranges):
    """The rows whose token position lies in one of the [start, end) ranges."""
    mask = np.zeros(len(positions), dtype=bool)
    for start, end in ranges:
        mask |= (positions >= start) & (positions < end)
    if not mask.any():
        raise ValueError("the mask selects no recorded row: every range lies outside "
                         f"the capture's positions [{positions[0]}, {positions[-1]}]")
    return mask


# ---------------------------------------------------------------- summaries
def summarise(kl, agree, select=None):
    """Mean KL and argmax agreement over the rows `select` picks (None: all).
    Median and max accompany the mean because one row can carry the mean."""
    if select is None:
        sel = np.ones(len(kl), dtype=bool)
    else:
        sel = select
    if not sel.any():
        raise ValueError("the selection holds no rows")
    return {"rows": int(sel.sum()),
            "mean_kl": float(kl[sel].mean()),
            "median_kl": float(np.median(kl[sel])),
            "max_kl": float(kl[sel].max()),
            "argmax": float(agree[sel].mean())}


def load_arm(ref_path, other_path, window, mask_ranges=None):
    """One capture's KL against the reference, over all rows and (when a mask
    is given) over the masked rows."""
    n_ctx, n_vocab, _, tok_ref, rows_ref = read_capture(ref_path)
    n_oth, v_oth, n_chunk, tok_oth, rows_oth = read_capture(other_path)
    if (n_ctx, n_vocab) != (n_oth, v_oth):
        raise ValueError(f"{other_path}: {n_oth} x {v_oth} does not match the reference's "
                         f"{n_ctx} x {n_vocab}")
    if window < 0 or window >= n_chunk:
        raise ValueError(f"{other_path}: window {window} of {n_chunk}")
    if not np.array_equal(tok_ref[0], tok_oth[window]):
        raise ValueError(f"{other_path}: window {window} holds different tokens than the reference")
    kl, agree = kl_rows(rows_ref[0], rows_oth[window], n_vocab)
    positions = row_positions(n_ctx)
    out = {"unmasked": summarise(kl, agree)}
    if mask_ranges is not None:
        out["masked"] = summarise(kl, agree, position_mask(positions, mask_ranges))
    return out


def twin_report(ref_path, baseline_a, baseline_b, candidate, window, mask_ranges=None):
    """The noise twin: two baseline runs against the reference give the card's
    own run-to-run floor, the candidate's mean KL is read beside it, and the
    difference is candidate - twin (positive: the candidate is worse than the
    floor)."""
    a = load_arm(ref_path, baseline_a, window, mask_ranges)
    b = load_arm(ref_path, baseline_b, window, mask_ranges)
    c = load_arm(ref_path, candidate, window, mask_ranges)

    keys = [k for k in ("unmasked", "masked") if k in a and k in b and k in c]
    twin = {k: {field: (a[k][field] + b[k][field]) / 2 for field in ("mean_kl", "argmax")} for k in keys}
    diff = {k: {field: c[k][field] - twin[k][field] for field in ("mean_kl", "argmax")} for k in keys}
    return {"a": a, "b": b, "twin": twin, "candidate": c, "difference": diff}


# ------------------------------------------------------------------- output
def _side_fields(s):
    """A summary's numbers as they exist: the twin holds only mean and argmax,
    an arm also holds median, max and its row count."""
    parts = [f"mean KL {s['mean_kl']:.4f}"]
    if "median_kl" in s:
        parts.append(f"median {s['median_kl']:.4f}")
    if "max_kl" in s:
        parts.append(f"max {s['max_kl']:.3f}")
    parts.append(f"argmax agreement {s['argmax']:.4f}")
    if "rows" in s:
        parts.append(f"{s['rows']} rows")
    return ", ".join(parts)


def print_side(label, arm):
    un, ma = arm["unmasked"], arm.get("masked")
    line = f"{label}: {_side_fields(un)}"
    if ma is not None:
        line += f" | masked {_side_fields(ma)}"
    print(line)


def print_legacy(ref_path, other_path, window, mask_ranges):
    n_ctx, n_vocab, _, tok_ref, rows_ref = read_capture(ref_path)
    _, _, _, tok_oth, rows_oth = read_capture(other_path)
    if not np.array_equal(tok_ref[0], tok_oth[window]):
        raise ValueError(f"{other_path}: window {window} holds different tokens than the reference")
    kl, agree = kl_rows(rows_ref[0], rows_oth[window], n_vocab)
    un = summarise(kl, agree)
    ma = summarise(kl, agree, position_mask(row_positions(n_ctx), mask_ranges)) if mask_ranges else None
    print(f"llama.cpp capture window {window} vs the f32 reference: mean KL {un['mean_kl']:.4f}, "
          f"median {un['median_kl']:.4f}, max {un['max_kl']:.3f}, argmax agreement {un['argmax']:.4f}")
    if ma is not None:
        print(f"   masked to {ma['rows']} rows: mean KL {ma['mean_kl']:.4f}, median {ma['median_kl']:.4f}, "
              f"max {ma['max_kl']:.3f}, argmax agreement {ma['argmax']:.4f}")
    positions = row_positions(n_ctx)
    for lo in range(n_ctx // 2 - (n_ctx // 2) % 128, n_ctx, 128):
        m = (positions >= lo) & (positions < lo + 128)
        if m.any():
            print(f"   {lo:>5}-{lo+127:<5} rows {int(m.sum()):>3} median {np.median(kl[m]):.4f} "
                  f"mean {kl[m].mean():.4f} argmax {agree[m].mean():.3f}")


def print_twin(ref_path, baseline_a, baseline_b, candidate, window, mask_ranges):
    rep = twin_report(ref_path, baseline_a, baseline_b, candidate, window, mask_ranges)
    print_side(f"baseline a (window {window})", rep["a"])
    print_side(f"baseline b (window {window})", rep["b"])
    print_side("baseline twin (mean of the two runs)", {"unmasked": rep["twin"]["unmasked"],
                                                       "masked": rep["twin"]["masked"]})
    print_side("candidate", rep["candidate"])
    d = rep["difference"]
    print(f"difference candidate - twin: mean KL {d['unmasked']['mean_kl']:+.4f}, "
          f"argmax {d['unmasked']['argmax']:+.4f} | masked {d['masked']['mean_kl']:+.4f}, "
          f"argmax {d['masked']['argmax']:+.4f}"
          + (" (positive: worse than the card's own floor)" if d["unmasked"]["mean_kl"] >= 0 else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", help="the f32 reference capture (window 0)")
    ap.add_argument("other", nargs="?", help="the capture compared against it (legacy single mode)")
    ap.add_argument("legacy_window", nargs="?", type=int, metavar="WINDOW",
                    help="the other capture's window (legacy single mode)")
    ap.add_argument("--baseline", nargs=2, metavar=("A", "B"),
                    help="twin mode: two captures of the baseline arm, same binary and configuration")
    ap.add_argument("--candidate", help="twin mode: the candidate arm's capture")
    ap.add_argument("--window", type=int, help="twin mode: the window the baseline and candidate captures hold")
    ap.add_argument("--mask", help="a file of [start, end) token ranges (the assistant spans)")
    args = ap.parse_args(argv)

    mask_ranges = read_mask(args.mask) if args.mask else None
    if args.baseline or args.candidate:
        if args.other is not None or args.legacy_window is not None:
            ap.error("twin mode takes the window through --window, not positionally")
        if not args.baseline or not args.candidate or args.window is None:
            ap.error("twin mode needs --baseline A B, --candidate and --window")
        print_twin(args.ref, args.baseline[0], args.baseline[1], args.candidate,
                   args.window, mask_ranges)
    else:
        if args.other is None or args.legacy_window is None:
            ap.error("single mode needs REF OTHER WINDOW (or use --baseline/--candidate)")
        print_legacy(args.ref, args.other, args.legacy_window, mask_ranges)


if __name__ == "__main__":
    main()
