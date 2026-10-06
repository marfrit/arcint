"""tools/kld_bar.py, device-free: the 0.5.1 KLD bar derived from THIS model's
own reference round-trip (docs/window-051.md §3(d)).

F_ref is the uint16 reconstruction error of the reference writer's own
transcription (`kld_served.llama_row`, perplexity.cpp `log_softmax(int, const
float*, uint16_t*, int)`) measured on real f32 rows: KL(P_row || P_roundtrip)
per row, mean over rows. bar_below = 100 x F_ref, bar_above = bar_below +
the measured QSA price. The floor pair is KL(A||B) between two forwards of
the same rows, printed beside the bar so no bar is quoted without it.

Red first: the cells were run before the module existed (ModuleNotFoundError),
then the step-bound cell went red on 40-nat rows: the clamp tail (its own cell now).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import kld_bar  # noqa: E402
import kld_served as ks  # noqa: E402


def _rows(n, vocab, seed, sharp=4.0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, vocab)) * sharp).astype(np.float32)


def test_f_ref_is_positive_and_under_the_half_step_bound_when_no_logit_is_clamped():
    """A row whose range stays under 16 nats meets no clamp: every logit is
    rounded to the nearest uint16 step, at most half a step off (8/65535
    nats at the widest step). KL is a probability-weighted second-order
    quantity of that error, so F_ref sits under the per-logit bound and
    above zero (the rows are not on the writer's grid)."""
    rows = _rows(6, 248_320, seed=3, sharp=1.2)       # max-min ~ 12 nats at this width
    assert float((rows.max(axis=1) - rows.min(axis=1)).max()) < 16.0
    rep = kld_bar.f_ref(rows)
    assert rep["rows"] == 6 and rep["vocab"] == 248_320
    assert 0.0 < rep["f_ref_mean"] < 8.0 / 65535.0
    assert rep["f_ref_max"] >= rep["f_ref_mean"]


def test_the_clamp_tail_dominates_f_ref_when_the_row_spans_more_than_16_nats():
    """THE TERM window-051 C6 DID NOT PRICE, found by the first run of this
    file (2026-09-13): the writer clamps the row's minimum at max-16, so
    every logit further down reconstructs to exactly max-16 -- and at
    248,320-wide those tens of thousands of entries carry e^-16 each, a
    tail mass the rounding-step bound never counts. F_ref then exceeds the
    half-step bound by more than a decade. Rows of a real language model span
    30-40 nats; the dated leg measures which regime the served rows are in."""
    wide = _rows(4, 248_320, seed=5, sharp=4.0)        # max-min ~ 40 nats
    assert float((wide.max(axis=1) - wide.min(axis=1)).min()) > 16.0
    rep = kld_bar.f_ref(wide)
    assert rep["f_ref_mean"] > 10.0 * 8.0 / 65535.0
    # the same rows with their tail lifted to within 16 nats of the max:
    # no clamp, and the error collapses back under the step bound
    lifted = np.maximum(wide, wide.max(axis=1, keepdims=True) - 15.9)
    assert kld_bar.f_ref(lifted)["f_ref_mean"] < 8.0 / 65535.0


def test_a_row_already_on_the_grid_round_trips_to_zero():
    """Logits that are exact multiples of the writer's step above the clamped
    minimum reconstruct bit-for-bit: F_ref 0 -- the cell that fails if the
    transcription or its inverse drifts from perplexity.cpp."""
    vocab = 257
    q = np.arange(vocab, dtype=np.float32) % 100      # grid indices 0..99
    # the max sits exactly 16 above the min, so the clamp is inert and the
    # writer's step is 16/65535; every logit is an integer number of steps
    # (99 x 662 = 65538 -> use 65535/99 is not integral; take 655 steps per
    # index instead: 99 x 655 = 64845 steps = 15.83 nats, under the clamp)
    scale = 16.0 / 65535.0
    logits = (q * 655.0 * scale).astype(np.float64)
    logits[-1] = 16.0                                 # the max, 65535 steps up
    logits = logits.astype(np.float32)
    rep = kld_bar.f_ref(logits[None, :])
    assert rep["f_ref_max"] < 1e-9


def test_the_floor_pair_is_zero_for_identical_rows_and_counts_moved_argmaxes():
    a = _rows(5, 1000, seed=7)
    b = a.copy()
    pair = kld_bar.floor_pair(a, b)
    assert pair["kl_ab_mean"] == 0.0 and pair["argmax_agreement"] == 1.0
    b[2, 0] += 50.0                                     # one row's argmax moves
    pair = kld_bar.floor_pair(a, b)
    assert pair["kl_ab_mean"] > 0.0 and pair["argmax_agreement"] == 0.8
    assert pair["rows_moved"] == 1


def test_the_bar_is_a_stated_multiple_of_f_ref_with_the_qsa_price_above(tmp_path):
    rows = _rows(4, 4096, seed=1)
    np.save(tmp_path / "forward_1.npy", rows)
    np.save(tmp_path / "forward_2.npy", rows)
    out = tmp_path / "bar.json"
    kld_bar.main(["--rows", str(tmp_path / "forward_1.npy"),
                  "--pair", str(tmp_path / "forward_2.npy"),
                  "--multiple", "100", "--qsa-price", "2.385560e-02",
                  "--out", str(out)])
    rep = json.loads(out.read_text())
    assert rep["bar_below_2051"] == pytest.approx(100.0 * rep["f_ref_mean"])
    assert rep["bar_at_or_above_2051"] == pytest.approx(rep["bar_below_2051"] + 2.385560e-02)
    assert rep["floor_pair"]["kl_ab_mean"] == 0.0
    assert rep["multiple"] == 100 and rep["inherited_bar_provisional"] == 0.0599
    assert rep["status"] == "PROVISIONAL until the dated leg names its rows"


def test_the_bound_is_marked_not_an_acceptance_bar_and_the_tail_is_flagged(tmp_path):
    """The 2026-09-19 honest fix: bar_below/bar_above are an
    instrument-resolution bound, not an acceptance bar; the acceptance
    candidate (the between-implementations floor) is carried; and the
    per-row clamp tail is flagged against the bound.

    The documented fact is pinned from the HARNESS constants, because it is a
    measured property of the REAL served rows (F_ref mean 3.0905e-05, per-row
    max 1.0533e-02 -> the max exceeds bar_below = 100 x mean by ~3.4x); a
    synthetic all-wide fixture does not reproduce that ratio, since its own
    tail inflates the mean. The flag's arithmetic is pinned on the report.
    Red first: the previous report had none of these fields."""
    import json
    import math
    import kld_harness as kh
    wide = _rows(4, 248_320, seed=5, sharp=4.0)      # spans >16 nats -> clamp tail
    x = tmp_path / "rows.npy"
    np.save(x, wide)
    out = tmp_path / "bar.json"
    assert kld_bar.main(["--rows", str(x), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["bound_is_acceptance"] is False
    assert "instrument-resolution" in rep["bound_kind"]
    assert rep["acceptance_candidate"]["median_w0_nats"] == 0.0649
    assert rep["acceptance_candidate"]["median_w1_nats"] == 0.0283
    assert rep["inherited_bar_provisional"] == kld_bar.INHERITED_BAR_PROVISIONAL
    # the flag is the report's own arithmetic ...
    assert rep["per_row_clamp_tail_exceeds_bound"] == (
        rep["f_ref_max"] > rep["bar_below_2051"])
    # ... and on the REAL rows the tail DOES exceed the bound (documented fact)
    assert kh.F_REF_PER_ROW_MAX_NATS > kh.RESOLUTION_BOUND_BELOW_NATS
    # the bound is 100 x F_ref to float precision (window-051's own decimal
    # 3.0905e-03 is 100 x 3.0905e-05 rounded to five significant figures)
    assert math.isclose(kh.RESOLUTION_BOUND_BELOW_NATS,
                        kh.RESOLUTION_BOUND_MULTIPLE * kh.F_REF_MEAN_NATS,
                        rel_tol=1e-6)
