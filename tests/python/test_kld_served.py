"""tools/kld_served.py, device-free: the reference capture reader reconstructs
llama.cpp's uint16 log-prob rows to the quantisation step, the dump reader
stitches a window from prefill records by `past` and refuses a sliced record,
and the per-row KL reads 0 for an identical distribution and > 0 for a moved
one.

Red first: the reconstruction cell was written against `min_log_prob + q`
(no scale) and read a KL of 3.1 nats against the row's own log-softmax; the
stitching cell against a dump whose second record was sliced to one row
(rows != n) and the reader accepted it.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import kld_served as ks  # noqa: E402


llama_row = ks.llama_row     # the writer transcription, one source (tools/kld_served.py)


def write_capture(path, n_ctx, n_vocab, windows_logits, tokens):
    n_chunk = len(windows_logits)
    first = n_ctx // 2
    with open(path, "wb") as f:
        f.write(b"_logits_")
        f.write(struct.pack("<iii", n_ctx, n_vocab, n_chunk))
        np.asarray(tokens, dtype=np.int32).reshape(n_chunk, n_ctx).tofile(f)
        for w in range(n_chunk):
            for p in range(first, n_ctx - 1):
                llama_row(windows_logits[w][p]).tofile(f)


def write_dump(path, records):
    """records: [(lane, past, n, rows_array[rows, vocab])]"""
    with open(path, "wb") as f:
        for lane, past, n, arr in records:
            arr = np.asarray(arr, dtype=np.float32)
            f.write(ks.DUMP_MAGIC)
            f.write(struct.pack("<5Q", lane, past, n, arr.shape[0], arr.shape[1]))
            arr.tofile(f)


def test_the_reference_row_reconstructs_to_the_quantisation_step():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=257).astype(np.float32) * 4
    row = llama_row(logits)
    ref_lp = ks.reference_log_probs(row, 257)
    direct = ks.log_softmax(logits.astype(np.float64))
    # the quantisation step is (max - min) / 65535 <= 16 / 65535 nats
    kept = logits > max(logits.min(), logits.max() - 16)
    assert np.abs(ref_lp[kept] - direct[kept]).max() < 16 / 65535 + 1e-6
    # the quantisation floor: measured 2.9e-6 nats for a 257-wide row whose
    # tail sits below max - 16 (the writer's clamp); 1e-4 is two decades
    # above it and three below anything the gate would read
    assert ks.kl_ref_vs_served(ref_lp, logits) < 1e-4


def test_kl_is_zero_for_the_same_row_and_positive_for_a_moved_one():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=64).astype(np.float32)
    row = llama_row(logits)
    ref_lp = ks.reference_log_probs(row, 64)
    assert ks.kl_ref_vs_served(ref_lp, logits) < 1e-4
    moved = logits.copy()
    moved[0] += 5.0
    assert ks.kl_ref_vs_served(ref_lp, moved) > 0.01


def test_windows_are_stitched_by_past_and_a_sliced_record_is_refused(tmp_path):
    n_ctx, vocab = 12, 9
    rng = np.random.default_rng(3)
    win = rng.normal(size=(n_ctx, vocab)).astype(np.float32)
    dump = tmp_path / "dump.bin"
    write_dump(dump, [(0, 0, 8, win[:8]), (0, 8, 4, win[8:]), (0, 12, 1, win[11:12]),
                      (0, 0, 12, win)])
    recs = ks.read_dump(dump)
    wins = ks.dump_windows(recs)
    assert [len(w) for w in wins] == [3, 1]
    np.testing.assert_array_equal(ks.window_rows(dump, wins[0], n_ctx, vocab), win)
    np.testing.assert_array_equal(ks.window_rows(dump, wins[1], n_ctx, vocab), win)
    sliced = tmp_path / "sliced.bin"
    write_dump(sliced, [(0, 0, 8, win[:8]), (0, 8, 4, win[11:12])])   # rows 1 for n 4
    with pytest.raises(ValueError, match="sliced"):
        ks.window_rows(sliced, ks.dump_windows(ks.read_dump(sliced))[0], n_ctx, vocab)


def test_compare_reads_zero_for_a_dump_equal_to_the_capture(tmp_path, capsys):
    n_ctx, vocab = 10, 7
    rng = np.random.default_rng(4)
    w0 = rng.normal(size=(n_ctx, vocab)).astype(np.float32)
    w1 = rng.normal(size=(n_ctx, vocab)).astype(np.float32)
    cap = tmp_path / "ref.dat"
    write_capture(cap, n_ctx, vocab, [w0, w1], np.arange(2 * n_ctx))
    n_ctx_r, n_vocab_r, n_chunk, tokens, rows = ks.read_capture(cap)
    assert (n_ctx_r, n_vocab_r, n_chunk) == (n_ctx, vocab, 2)
    assert tokens.tolist() == [list(range(10)), list(range(10, 20))]
    dump = tmp_path / "dump.bin"
    # two load-time probe windows first (128 and 4 tokens, not the capture's
    # n_ctx), then the two replays -- the compare must skip the probes
    write_dump(dump, [(0, 0, 4, w0[:4]), (0, 0, 3, w1[:3]),
                      (0, 0, n_ctx, w0), (0, n_ctx, 1, w0[-1:]),
                      (0, 0, n_ctx, w1), (0, n_ctx, 1, w1[-1:])])
    out = tmp_path / "rep.json"
    assert ks.main(["--ref", str(cap), "--compare", "--dump", str(dump),
                    "--out", str(out)]) == 0
    import json
    rep = json.loads(out.read_text())
    assert rep["mean_kl_below"] < 1e-6 and rep["mean_kl_above"] is None
    assert rep["windows"][0]["argmax_agreement"] == 1.0
    # and a moved dump reads positive
    write_dump(dump, [(0, 0, n_ctx, w0 + rng.normal(size=w0.shape) * 3),
                      (0, 0, n_ctx, w1)])
    assert ks.main(["--ref", str(cap), "--compare", "--dump", str(dump),
                    "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["mean_kl_below"] > 0.1


def test_the_floor_is_reported_from_the_last_two_replays(tmp_path):
    """REVIEW ba2d5de F1: two replays of the same windows give KL(A||B); zero
    for identical replays, positive for a moved second one. Written with the
    change, not before it: against the previous compare (no "floor" key in
    the report) the first assertion fails by construction, and that is the
    only red this cell has seen."""
    import json
    n_ctx, vocab = 10, 7
    rng = np.random.default_rng(5)
    w0 = rng.normal(size=(n_ctx, vocab)).astype(np.float32)
    w1 = rng.normal(size=(n_ctx, vocab)).astype(np.float32)
    cap = tmp_path / "ref.dat"
    write_capture(cap, n_ctx, vocab, [w0, w1], np.arange(2 * n_ctx))
    dump = tmp_path / "dump.bin"
    out = tmp_path / "rep.json"
    # replay A == replay B: floor 0
    write_dump(dump, [(0, 0, n_ctx, w0), (0, 0, n_ctx, w1), (0, 0, n_ctx, w0), (0, 0, n_ctx, w1)])
    assert ks.main(["--ref", str(cap), "--compare", "--dump", str(dump), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["floor"] is not None
    assert rep["floor"]["mean_kl_a_b"] < 1e-9 and rep["floor"]["max_kl_a_b"] < 1e-9
    assert all(f["argmax_agreement"] == 1.0 for f in rep["floor"]["windows"])
    # replay B moved: floor positive, and the means are against B (the last)
    w0b = w0 + rng.normal(size=w0.shape).astype(np.float32) * 2
    write_dump(dump, [(0, 0, n_ctx, w0), (0, 0, n_ctx, w1), (0, 0, n_ctx, w0b), (0, 0, n_ctx, w1)])
    assert ks.main(["--ref", str(cap), "--compare", "--dump", str(dump), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["floor"]["windows"][0]["mean_kl_a_b"] > 0.05
    assert rep["floor"]["windows"][1]["mean_kl_a_b"] < 1e-9
    assert rep["windows"][0]["mean_kl_below"] > 0.05      # ref vs B, B moved
    # one replay only: no floor, said so
    write_dump(dump, [(0, 0, n_ctx, w0), (0, 0, n_ctx, w1)])
    assert ks.main(["--ref", str(cap), "--compare", "--dump", str(dump), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["floor"] is None


def test_three_replays_give_two_pairs_and_count_the_moved_windows(tmp_path):
    """The floor over N replays: every earlier replay against the last, and
    the count of window pairs that moved. Red first: against the two-replay
    compare the report has no "pairs" key and the count is not there."""
    import json
    n_ctx, vocab = 10, 7
    rng = np.random.default_rng(6)
    w0 = rng.normal(size=(n_ctx, vocab)).astype(np.float32)
    w1 = rng.normal(size=(n_ctx, vocab)).astype(np.float32)
    cap = tmp_path / "ref.dat"
    write_capture(cap, n_ctx, vocab, [w0, w1], np.arange(2 * n_ctx))
    dump = tmp_path / "dump.bin"
    out = tmp_path / "rep.json"
    w0b = w0 + rng.normal(size=w0.shape).astype(np.float32) * 2
    # replay 0: moved w0; replay 1: identical to the last; replay 2: the last
    write_dump(dump, [(0, 0, n_ctx, w0b), (0, 0, n_ctx, w1),
                      (0, 0, n_ctx, w0), (0, 0, n_ctx, w1),
                      (0, 0, n_ctx, w0), (0, 0, n_ctx, w1)])
    assert ks.main(["--ref", str(cap), "--compare", "--dump", str(dump), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    fl = rep["floor"]
    assert fl["replays"] == 3 and len(fl["pairs"]) == 2
    assert fl["pairs"][0]["windows_moved"] == 1 and fl["pairs"][1]["windows_moved"] == 0
    assert fl["window_pairs_moved"] == 1 and fl["window_pairs"] == 4
    assert fl["mean_kl_a_b"] < 1e-9                    # the last pair is identical
    assert fl["pairs"][0]["windows"][0]["mean_kl_a_b"] > 0.05


def test_the_report_carries_the_bar_as_provisional_with_its_provenance(tmp_path):
    """REVIEW ba2d5de F2 + the frontier's conditions: the bar is printed as
    PROVISIONAL with its provenance beside it, and it is kld_harness's, not a
    second literal. Red first: the previous report had neither key."""
    import json
    import kld_harness as kh
    n_ctx, vocab = 10, 7
    rng = np.random.default_rng(7)
    w0 = rng.normal(size=(n_ctx, vocab)).astype(np.float32)
    cap = tmp_path / "ref.dat"
    write_capture(cap, n_ctx, vocab, [w0], np.arange(n_ctx))
    dump = tmp_path / "dump.bin"
    out = tmp_path / "rep.json"
    write_dump(dump, [(0, 0, n_ctx, w0)])
    assert ks.main(["--ref", str(cap), "--compare", "--dump", str(dump), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["threshold_nats"] == kh.THRESHOLD_NATS
    assert rep["threshold_status"] == "PROVISIONAL"
    assert "0.0399" in rep["threshold_provenance"] and "2026-08-11" in rep["threshold_provenance"]
    assert ks.THRESHOLD_NATS is kh.THRESHOLD_NATS


def test_warmup_posts_extra_windows_before_the_counted_replays(monkeypatch, tmp_path):
    """--warmup N posts every window N more times before the --repeat replays
    (the kernel set settles first, window-050 §4.11); the request order is
    warmups, then replays. Red first: --warmup was an unknown option."""
    import urllib.request
    n_ctx, vocab = 6, 5
    cap = tmp_path / "ref.dat"
    w = np.zeros((n_ctx, vocab), np.float32)
    write_capture(cap, n_ctx, vocab, [w, w], np.arange(2 * n_ctx))
    posted = []

    class _Resp:
        def __init__(self, body): self._b = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._b

    def fake_urlopen(req, timeout=None):
        import json as _j
        posted.append(_j.loads(req.data)["prompt"])
        return _Resp(b'{"usage": {}, "choices": [{"text": "x"}]}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert ks.main(["--ref", str(cap), "--replay", "--repeat", "2", "--warmup", "3",
                    "--url", "http://x"]) == 0
    assert len(posted) == (3 + 2) * 2                    # 5 passes x 2 windows
    assert posted[0] == list(range(6)) and posted[1] == list(range(6, 12))


def test_the_report_also_carries_the_decided_bound_and_the_readable_candidate(tmp_path):
    """The 2026-09-19 honest fix: no reader may see only the SUPERSEDED
    inherited literal. The report carries the DECIDED instrument-resolution
    bound (window-051 clause (d)) with its MEAN/per-row caveat, and the
    between-implementations floor as the acceptance candidate. Red first:
    the previous report had none of these keys."""
    import json
    import kld_harness as kh
    n_ctx, vocab = 10, 7
    rng = np.random.default_rng(11)
    w0 = rng.normal(size=(n_ctx, vocab)).astype(np.float32)
    cap = tmp_path / "ref.dat"
    write_capture(cap, n_ctx, vocab, [w0], np.arange(n_ctx))
    dump = tmp_path / "dump.bin"
    out = tmp_path / "rep.json"
    write_dump(dump, [(0, 0, n_ctx, w0)])
    assert ks.main(["--ref", str(cap), "--compare", "--dump", str(dump), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())

    # the decided bound is present and equals the harness's own constants
    assert rep["resolution_bound_below_nats"] == kh.RESOLUTION_BOUND_BELOW_NATS
    assert rep["resolution_bound_above_nats"] == kh.RESOLUTION_BOUND_ABOVE_NATS
    # and it is priced as a MEAN bound: its own per-row tail exceeds it
    assert rep["f_ref_per_row_max_nats"] > kh.RESOLUTION_BOUND_BELOW_NATS
    # the acceptance candidate is the between-implementations floor
    assert rep["implementation_floor_median_w0_nats"] == kh.IMPLEMENTATION_FLOOR_MEDIAN_W0_NATS
    assert rep["implementation_floor_median_w1_nats"] == kh.IMPLEMENTATION_FLOOR_MEDIAN_W1_NATS
    assert "not an acceptance bar" in rep["resolution_bound_provenance"].lower() or \
           "instrument" in rep["resolution_bound_provenance"].lower()
    # continuity: the inherited literal is still carried, and is marked SUPERSEDED
    assert rep["threshold_nats"] == kh.THRESHOLD_NATS
    assert "SUPERSEDED" in kh.BAR_PROVENANCE
