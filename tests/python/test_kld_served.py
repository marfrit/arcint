"""tools/kld_served.py, device-free: the reference capture reader reconstructs
llama.cpp's uint16 log-prob rows to the quantisation step, the per-row KL reads
0 for an identical distribution and > 0 for a moved one, and a file without the
capture magic is refused.

Red first: the reconstruction cell was written against `min_log_prob + q`
(no scale) and read a KL of 3.1 nats against the row's own log-softmax.
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


def test_a_file_that_is_not_a_capture_is_refused(tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"NOTALOGS" + b"\0" * 64)
    with pytest.raises(ValueError, match="not a llama.cpp logits capture"):
        ks.read_capture(str(bad))


def test_the_capture_reader_round_trips_a_written_capture(tmp_path):
    rng = np.random.default_rng(3)
    n_ctx, n_vocab = 8, 33
    logits = [rng.normal(size=(n_ctx, n_vocab)).astype(np.float32) for _ in range(2)]
    toks = rng.integers(0, n_vocab, size=(2, n_ctx))
    path = tmp_path / "cap.bin"
    write_capture(path, n_ctx, n_vocab, logits, toks)
    c_ctx, c_vocab, c_chunk, tokens, rows = ks.read_capture(str(path))
    assert (c_ctx, c_vocab, c_chunk) == (n_ctx, n_vocab, 2)
    assert (tokens == toks).all()
    first = n_ctx // 2
    assert rows.shape[1] == n_ctx - 1 - first
    ref = ks.reference_log_probs(rows[1][0], n_vocab)
    assert ks.kl_ref_vs_served(ref, logits[1][first]) < 1e-3
