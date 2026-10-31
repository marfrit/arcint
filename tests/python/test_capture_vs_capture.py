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

The assistant mask, the twin floor's spread and the floor verdict, red first,
each mutation seen failing on its own:
- the mask off by one token: `spans.append((pos, pos + len(ids)))` ->
  `(pos + 1, pos + len(ids) + 1)`; the high KL planted at position 22 lands in
  the mask and test_the_mask_is_derived_from_a_transcripts_assistant_turns
  FAILS (the masked mean no longer < 1e-4).
- the twin's spread read as zero: `abs(a[k][f] - b[k][f])` -> `0.0` (one
  metric line); test_the_twin_reports_both_numbers_and_its_own_spread FAILS
  on `sp["mean_kl"] > 0`.
- the floor verdict's boundary strictly `<` instead of `<=`; identical arms
  (difference 0, spread 0) become "outside" and
  test_the_cli_twin_prints_the_floor_and_the_floor_verdict FAILS (no
  "inside the floor" in the output).
- the role filter inverted: `if t["role"] != "assistant"` -> `==`; the
  multi-turn test FAILS with spans [(0, 0), (0, 0)] against [(1, 3), (5, 8)].

Round 2, the row convention and the merge, red first and each seen failing:
- `token_rows` returning `(start, end)` (the one-row shift): the first-token
  test FAILS (position 19 left out of the mask) and the token-after test FAILS
  (position 23 inside it).
- `reply_end` dropping the marker's full count (`len(tokens) - m` only): the
  period test FAILS, the merged last token's row leaves the mask.
- the mask's convention removed from the single mode's output: the CLI test
  FAILS on `"row p scores token p+1"`.
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
    # a TOKEN range [first + 1, first + 3) is the rows [first, first + 2): the
    # convention row p scores token p + 1 (round 2), so two rows again
    mask.write_text(f"# the assistant span\n{first + 1} {first + 3}\n")

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


# ------------------------------------------------- the assistant mask from role ids
def write_transcript(path, turns):
    import json
    path.write_text(json.dumps({"turns": turns}) + "\n")


def test_the_mask_is_derived_from_a_transcripts_assistant_turns(tmp_path):
    """The audit's lack: the served arm scores only the positions the model
    writes. Here the mask comes from the request's own turn ids, not a
    hand-written range. A high KL planted just past the assistant span (token
    position 22) must not move the masked mean. Red: the span shifted one token
    right (start 21) selects position 22 and the masked mean jumps."""
    ref, base, tokens, _ = fixture(tmp_path)
    moved = base.copy()
    moved[22, 0] += 6.0          # position 22, just outside the span [20, 22)
    cand = write_arm(tmp_path, "cand.bin", moved, tokens)
    tr = tmp_path / "request.json"
    write_transcript(tr, [{"role": "user"},
                          {"role": "assistant", "tokens": [int(tokens[20]), int(tokens[21])]}])

    turns = ks.read_transcript(str(tr))
    arm = cvc.load_arm(str(ref), str(cand), 0, transcript=turns)
    assert arm["unmasked"]["mean_kl"] > 0.01
    assert arm["masked"]["rows"] == 2
    assert arm["masked"]["mean_kl"] < 1e-4


def test_the_cli_prints_the_masked_and_unmasked_means_side_by_side(tmp_path, capsys):
    """`--spans`: the reply's positions beside the whole-window number, so a
    reader can see which one the bar is judged on."""
    ref, base, tokens, _ = fixture(tmp_path)
    cand = write_arm(tmp_path, "cand.bin", base, tokens)
    tr = tmp_path / "request.json"
    write_transcript(tr, [{"role": "user"},
                          {"role": "assistant",
                           "tokens": [int(tokens[20]), int(tokens[21]), int(tokens[22])]}])
    cvc.main([str(ref), str(cand), "0", "--spans", str(tr)])
    out = capsys.readouterr().out
    assert "unmasked:" in out and "masked:" in out
    # the single mode states the row convention it scored by (round 2)
    assert "row p scores token p+1" in out


def test_a_run_without_spans_says_a_corpus_arm_has_no_roles(tmp_path, capsys):
    """A plain-text corpus capture has no roles and must not be masked by
    invention: the output says so and names the chat path."""
    ref, base, tokens, _ = fixture(tmp_path)
    cand = write_arm(tmp_path, "cand.bin", base, tokens)
    cvc.main([str(ref), str(cand), "0"])
    out = capsys.readouterr().out
    assert "no roles" in out and "--spans" in out


def test_mask_and_spans_together_are_refused(tmp_path):
    ref, base, tokens, _ = fixture(tmp_path)
    cand = write_arm(tmp_path, "cand.bin", base, tokens)
    mask = tmp_path / "mask.txt"
    mask.write_text("16 18\n")
    tr = tmp_path / "request.json"
    write_transcript(tr, [{"role": "assistant", "tokens": [int(tokens[20])]}])
    with pytest.raises(SystemExit):
        cvc.main([str(ref), str(cand), "0", "--mask", str(mask), "--spans", str(tr)])


# ------------------------------------------------------- the twin floor, two numbers
def test_the_twin_reports_both_numbers_and_its_own_spread(tmp_path):
    """The floor is two numbers, not one: mean KL and argmax agreement, each
    with the two baseline runs' own spread. The candidate's difference is
    flagged inside or outside that spread, and the answer-level bar is read
    beside it unchanged."""
    ref, base, tokens, _ = fixture(tmp_path)
    a_shift = base.copy()
    a_shift[20, 0] += 0.8       # baseline a: a stray row, so the twin has a spread
    a = write_arm(tmp_path, "a.bin", a_shift, tokens)
    b = write_arm(tmp_path, "b.bin", base, tokens)
    cand = write_arm(tmp_path, "cand.bin", base, tokens)

    rep = cvc.twin_report(str(ref), str(a), str(b), str(cand), 0)
    sp = rep["spread"]["unmasked"]
    assert sp["mean_kl"] > 0 and sp["argmax"] >= 0
    assert set(rep["twin"]["unmasked"]) == {"mean_kl", "argmax"}
    # the candidate sits at baseline b, inside the two runs' spread
    assert rep["inside_floor"]["unmasked"]["inside"]
    assert abs(rep["difference"]["unmasked"]["mean_kl"]) <= sp["mean_kl"]
    assert rep["bar"]["unmasked"]["pass"]


def test_a_candidate_outside_the_twins_spread_is_flagged(tmp_path):
    """The floor is per window and per metric: a difference larger than the two
    runs' own difference is outside it, even when it still passes the bar."""
    ref, base, tokens, _ = fixture(tmp_path)
    a = write_arm(tmp_path, "a.bin", base, tokens)
    b = write_arm(tmp_path, "b.bin", base, tokens)
    moved = base.copy()
    moved[20, 0] += 6.0        # a zero twin spread: any moved row is beyond it
    cand = write_arm(tmp_path, "cand.bin", moved, tokens)

    rep = cvc.twin_report(str(ref), str(a), str(b), str(cand), 0)
    assert rep["spread"]["unmasked"]["mean_kl"] == 0.0
    assert not rep["inside_floor"]["unmasked"]["inside"]
    assert rep["difference"]["unmasked"]["mean_kl"] > 0.01
    # the bar is a separate verdict and it still says the candidate is worse
    assert not rep["bar"]["unmasked"]["pass"]


def test_the_bar_stays_at_003_nats_and_one_point(tmp_path):
    """CLAUDE.md's answer-level bar, stated and unchanged: 0.03 nats worse than
    the baseline arm and argmax at most 1 point (0.01 of the fraction) below."""
    ref, base, tokens, _ = fixture(tmp_path)
    a = write_arm(tmp_path, "a.bin", base, tokens)
    b = write_arm(tmp_path, "b.bin", base, tokens)
    cand = write_arm(tmp_path, "cand.bin", base, tokens)
    rep = cvc.twin_report(str(ref), str(a), str(b), str(cand), 0)
    assert cvc.BAR_KL_NATS == 0.03
    assert cvc.BAR_ARGMAX_POINTS == 0.01


def test_the_cli_twin_prints_the_floor_and_the_floor_verdict(tmp_path, capsys):
    ref, base, tokens, _ = fixture(tmp_path)
    a = write_arm(tmp_path, "a.bin", base, tokens)
    b = write_arm(tmp_path, "b.bin", base, tokens)
    cand = write_arm(tmp_path, "cand.bin", base, tokens)
    cvc.main([str(ref), "--baseline", str(a), str(b), "--candidate", str(cand), "--window", "0"])
    out = capsys.readouterr().out
    assert "spread" in out and "inside the floor" in out
    assert "bar" in out and "0.03" in out and "1 point" in out


# ------------------------------------------------- the row convention in the mask
def test_the_masked_mean_holds_the_replys_first_token_prediction(tmp_path):
    """The reply's FIRST token is predicted by the row at token position
    start - 1, the one right after the assistant header -- usually the
    highest-entropy token of the reply. Plant a high KL there: it must be inside
    the masked mean. Red: the one-row shift (token_rows -> (start, end), or
    position_mask over the raw token range) leaves position 19 out of
    [20, 24) and the masked mean stays near zero."""
    ref, base, tokens, _ = fixture(tmp_path)
    first_token = base.copy()
    first_token[19, 0] += 6.0     # position 19 predicts token 20, the reply's first
    cand = write_arm(tmp_path, "cand.bin", first_token, tokens)
    tr = tmp_path / "request.json"
    write_transcript(tr, [{"role": "user"},
                          {"role": "assistant",
                           "tokens": [int(tokens[20]), int(tokens[21]), int(tokens[22]), int(tokens[23])]}])
    arm = cvc.load_arm(str(ref), str(cand), 0, transcript=ks.read_transcript(str(tr)))
    assert arm["masked"]["rows"] == 4      # rows 19..22, one per reply token
    assert arm["masked"]["mean_kl"] > 0.0005, "the first token's prediction is not in the mask"


def test_the_masked_mean_excludes_the_token_after_the_reply(tmp_path):
    """The row at token position end - 1 predicts the token after the reply; it
    is outside the mask. Red: the shift selects [20, 24) and position 23 lands
    inside, moving the masked mean."""
    ref, base, tokens, _ = fixture(tmp_path)
    after = base.copy()
    after[23, 0] += 6.0           # position 23 predicts token 24, outside the reply
    cand = write_arm(tmp_path, "cand.bin", after, tokens)
    tr = tmp_path / "request.json"
    write_transcript(tr, [{"role": "user"},
                          {"role": "assistant",
                           "tokens": [int(tokens[20]), int(tokens[21]), int(tokens[22]), int(tokens[23])]}])
    arm = cvc.load_arm(str(ref), str(cand), 0, transcript=ks.read_transcript(str(tr)))
    assert arm["unmasked"]["mean_kl"] > 0.01
    assert arm["masked"]["mean_kl"] < 1e-4


def test_a_reply_ending_in_a_period_keeps_its_last_token_in_the_mask(tmp_path):
    """The marker form end to end: the window's tokens 22 and 23 are the merged
    last reply token and the end id, and the turn's ids are text + merged token +
    end id. The reply's span keeps the merged token, so the row predicting it
    (position 21, the row at token 21) is inside the mask. Red: reply_end
    dropping the marker's full count -- 3 -> 2 -- makes the span (20, 22) and
    puts position 21 outside [19, 21), so the planted KL leaves the mean."""
    ref, base, tokens, _ = fixture(tmp_path)
    tokens = tokens.copy()
    tokens[22], tokens[23] = 602, 601      # the merged '.'+newline token, the end id
    write_capture(ref, [base], tokens)     # the reference's window holds the same ids
    overall = base.copy()
    overall[21, 0] += 6.0      # position 21 predicts token 22, the merged last reply token
    cand = write_arm(tmp_path, "cand.bin", overall, tokens)
    tr = tmp_path / "request.json"
    write_transcript(tr, [{"role": "user"},
                          {"role": "assistant",
                           "tokens": [int(tokens[20]), int(tokens[21]), 602, 601],
                           "marker_tokens": [600, 601]}])
    arm = cvc.load_arm(str(ref), str(cand), 0, transcript=ks.read_transcript(str(tr)))
    assert arm["masked"]["rows"] == 3      # rows 19..21: text, '.', the merged token
    assert arm["masked"]["mean_kl"] > 0.0005
