"""tools/capture_vs_capture.py, device-free: the noise-twin comparison and the
position mask over synthetic captures.

Twin mode judges an A/B against the card's own run-to-run floor: two baseline
captures give the twin's mean KL against the reference, and the candidate's
difference is read beside it. Identical arms read a difference of 0; a moved
candidate reads a positive one. The mask restricts the mean KL and the argmax
agreement to [start, end) token ranges, so a large KL placed outside them is
ignored.

Red first:
- twin: `twin_report` taking the candidate's own mean for the twin (one
  substituted line) reads a difference of 0 for the moved candidate.
- mask: `position_mask` returning all rows (the ranges ignored) reads the
  masked mean as the unmasked one, red for the large KL outside the ranges.
- twin without a mask through the CLI: `print_twin` passing
  `{"unmasked": ..., "masked": rep["twin"]["masked"]}` (the line that was
  there) raises KeyError 'masked' on `main(argv)`, so the twin mode without
  `--mask` never completed.
- a vocabulary mismatch in the single mode: `print_legacy` reading the
  captures itself and passing `n_vocab` to `kl_rows` (the shape before
  load_rows) accepts an n_vocab mismatch instead of refusing it.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import capture_vs_capture as cvc  # noqa: E402
import kld_served as ks  # noqa: E402


def write_capture(path, windows_logits, tokens):
    """A capture in llama.cpp's format: magic, context, vocabulary, chunk count,
    the tokens, then one uint16 row a recorded position."""
    n_chunk = len(windows_logits)
    n_ctx, n_vocab = windows_logits[0].shape
    first = n_ctx // 2
    with open(path, "wb") as f:
        f.write(b"_logits_")
        f.write(struct.pack("<iii", n_ctx, n_vocab, n_chunk))
        np.asarray(tokens, dtype=np.int32).reshape(n_chunk, n_ctx).tofile(f)
        for w in range(n_chunk):
            for p in range(first, n_ctx - 1):
                ks.llama_row(windows_logits[w][p]).tofile(f)


def fixture(tmp_path, n_ctx=32, n_vocab=17, seed=7):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_ctx, n_vocab)).astype(np.float32)
    tokens = rng.integers(0, n_vocab, size=n_ctx)
    ref = tmp_path / "ref.bin"
    write_capture(ref, [base], tokens)
    return ref, base, tokens, rng


def write_arm(tmp_path, name, logits, tokens):
    p = tmp_path / name
    write_capture(p, [logits], tokens)
    return p


def test_twin_difference_is_zero_for_identical_arms(tmp_path):
    ref, base, tokens, _ = fixture(tmp_path)
    a = write_arm(tmp_path, "a.bin", base, tokens)
    b = write_arm(tmp_path, "b.bin", base, tokens)
    cand = write_arm(tmp_path, "cand.bin", base, tokens)

    rep = cvc.twin_report(str(ref), str(a), str(b), str(cand), 0)
    assert abs(rep["difference"]["unmasked"]["mean_kl"]) < 1e-4
    assert rep["twin"]["unmasked"]["mean_kl"] < 1e-4
    assert rep["candidate"]["unmasked"]["mean_kl"] < 1e-4


def test_twin_difference_is_positive_for_a_moved_candidate(tmp_path):
    ref, base, tokens, _ = fixture(tmp_path)
    a = write_arm(tmp_path, "a.bin", base, tokens)
    b = write_arm(tmp_path, "b.bin", base, tokens)
    moved = base.copy()
    moved[20, 0] += 4.0   # a recorded position's distribution moved
    cand = write_arm(tmp_path, "cand.bin", moved, tokens)

    rep = cvc.twin_report(str(ref), str(a), str(b), str(cand), 0)
    assert rep["difference"]["unmasked"]["mean_kl"] > 0.01
    assert rep["candidate"]["unmasked"]["mean_kl"] > rep["twin"]["unmasked"]["mean_kl"]


def test_the_masked_mean_ignores_a_large_kl_outside_the_ranges(tmp_path):
    ref, base, tokens, _ = fixture(tmp_path)
    first = 32 // 2   # rows stand at token positions 16 through 30
    outside = base.copy()
    outside[25, 0] += 6.0   # token position 25, outside the mask (rows start at position 16)
    cand = write_arm(tmp_path, "cand.bin", outside, tokens)
    mask = tmp_path / "mask.txt"
    mask.write_text(f"# the assistant span\n{first} {first + 2}\n")

    ranges = cvc.read_mask(str(mask))
    arm = cvc.load_arm(str(ref), str(cand), 0, ranges)
    assert arm["unmasked"]["mean_kl"] > 0.01
    assert arm["masked"]["mean_kl"] < 1e-4
    assert arm["masked"]["rows"] == 2


def test_twin_mode_runs_without_a_mask_through_the_cli(tmp_path, capsys):
    """The twinning path with no --mask: the twin holds no 'masked' key, so the
    printer must not read one. Red: the unconditional rep["twin"]["masked"]
    line raises KeyError and main() never returns."""
    ref, base, tokens, _ = fixture(tmp_path)
    a = write_arm(tmp_path, "a.bin", base, tokens)
    b = write_arm(tmp_path, "b.bin", base, tokens)
    cand = write_arm(tmp_path, "cand.bin", base, tokens)

    cvc.main([str(ref), "--baseline", str(a), str(b), "--candidate", str(cand), "--window", "0"])
    out = capsys.readouterr().out
    assert "baseline twin" in out and "difference candidate - twin" in out
    assert "masked" not in out


def test_single_mode_refuses_a_vocabulary_mismatch(tmp_path):
    """A capture for a different vocabulary: kl_rows truncates to the shorter
    row silently and reads garbage. The single mode must refuse it, as the
    twin mode's load_arm does. Red: print_legacy passing the reference's
    n_vocab to kl_rows without the equality check returns numbers instead."""
    ref, base, tokens, _ = fixture(tmp_path)
    other_n_vocab = 19
    other = np.random.default_rng(3).normal(size=(32, other_n_vocab)).astype(np.float32)
    p = tmp_path / "other.bin"
    write_capture(p, [other], tokens)

    with pytest.raises(ValueError, match="does not match the reference"):
        cvc.main([str(ref), str(p), "0"])
    with pytest.raises(ValueError, match="does not match the reference"):
        cvc.load_arm(str(ref), str(p), 0)


def test_single_mode_prints_the_boundary_split(tmp_path, capsys):
    """The legacy print keeps the below/above QSA_BOUNDARY_TOKENS split its
    docstring describes."""
    ref, base, tokens, _ = fixture(tmp_path)
    cand = write_arm(tmp_path, "cand.bin", base, tokens)
    cvc.main([str(ref), str(cand), "0"])
    out = capsys.readouterr().out
    assert "(below " in out and "/ above " in out


def test_a_mask_without_a_range_is_refused(tmp_path):
    mask = tmp_path / "empty.txt"
    mask.write_text("# nothing here\n\n")
    with pytest.raises(ValueError, match="no ranges"):
        cvc.read_mask(str(mask))
