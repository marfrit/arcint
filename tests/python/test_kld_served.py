"""tools/kld_served.py, device-free: the reference capture reader reconstructs
llama.cpp's uint16 log-prob rows to the quantisation step, the per-row KL reads
0 for an identical distribution and > 0 for a moved one, and a file without the
capture magic is refused.

Red first: the reconstruction cell was written against `min_log_prob + q`
(no scale) and read a KL of 3.1 nats against the row's own log-softmax.

The assistant spans, red first and the mutations each seen failing:
- the role filter inverted (`if t["role"] != "assistant"` -> `==`) makes
  assistant_spans read every non-assistant turn's empty id list;
  test_assistant_spans_locate_a_multi_turn_transcript_in_order FAILS with
  spans [(0, 0), (0, 0)] against [(1, 3), (5, 8)].
- the assistant-id requirement removed (`if not ids:` -> `if False:`);
  test_an_assistant_turn_without_ids_is_refused FAILS (a mask with no ids
  would be a guess).

Round 2, red first and each mutation seen failing:
- `token_rows` returning `(start, end)` instead of `(start - 1, end - 1)`
  FAILS test_a_token_range_is_scored_by_the_rows_predicting_it.
- `reply_end`'s merge branch killed (`if False:`): the merged last token test
  FAILS, 2 against 3.
- `assistant_spans` locating only the reply's text (needle `ids[:reply_end]`)
  FAILS test_a_user_quote_of_the_reply_does_not_capture_the_span: the quote
  is taken for the turn.
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


# ---------------------------------------------------------- the assistant mask
def test_a_transcript_reads_the_turns_and_their_ids(tmp_path):
    """A chat request as the server saw it: roles in order, the assistant
    turns' token ids (the server's tokenizer produced them; this tool has no
    tokenizer). A non-assistant turn may omit ids -- the mask only needs the
    assistant spans. Red: read_transcript returning bare strings, or taking
    `tokens` from the JSON without validating them, before it existed."""
    p = tmp_path / "request.json"
    p.write_text('{"turns": [{"role": "user", "tokens": [1, 2, 3]},'
                 ' {"role": "assistant", "tokens": [4, 5]}]}\n')
    turns = ks.read_transcript(str(p))
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[1]["tokens"] == [4, 5]


def test_an_assistant_turn_without_ids_is_refused(tmp_path):
    """Without the ids the span can only be guessed, and a guess is worse than
    a refusal: there is no tokenizer here to recompute them from text."""
    p = tmp_path / "request.json"
    p.write_text('[{"role": "assistant"}]')
    with pytest.raises(ValueError, match="no token ids"):
        ks.read_transcript(str(p))


def test_a_transcript_without_an_assistant_turn_is_refused(tmp_path):
    p = tmp_path / "request.json"
    p.write_text('[{"role": "system"}, {"role": "user"}]')
    turns = ks.read_transcript(str(p))
    with pytest.raises(ValueError, match="no assistant turn"):
        ks.assistant_spans(np.arange(8), turns)


def test_assistant_spans_locate_a_multi_turn_transcript_in_order(tmp_path):
    """The window is the rendered prompt plus the reply: the earlier assistant
    turns sit inside the prompt, the reply at its end. Locating each turn's id
    sequence forward from the previous one keeps a repeated phrase from
    capturing a later turn."""
    p = tmp_path / "request.json"
    p.write_text('{"turns": [{"role": "user"},'
                 ' {"role": "assistant", "tokens": [7, 8]},'
                 ' {"role": "user"},'
                 ' {"role": "assistant", "tokens": [11, 12, 13]}]}')
    window = np.array([0, 7, 8, 9, 10, 11, 12, 13, 0], dtype=np.int32)
    spans = ks.assistant_spans(window, ks.read_transcript(str(p)))
    assert spans == [(1, 3), (5, 8)]


def test_a_turn_absent_from_the_window_is_refused(tmp_path):
    """A reply that is not in the captured window would silently mask the wrong
    rows; the span search refuses it by name."""
    p = tmp_path / "request.json"
    p.write_text('{"turns": [{"role": "assistant", "tokens": [99, 98]}]}')
    with pytest.raises(ValueError, match="not in the capture's window"):
        ks.assistant_spans(np.array([1, 2, 3], dtype=np.int32), ks.read_transcript(str(p)))


# -------------------------------------------------- the row convention (round 2)
def test_a_token_range_is_scored_by_the_rows_predicting_it():
    """A capture row holds the distribution over the NEXT token: llama.cpp
    writes row i as the row scoring `tokens[i+1]`
    (tools/perplexity/perplexity.cpp:157-160), with the recorded rows starting at
    n_ctx/2. So a token range [start, end) is the row positions
    [start - 1, end - 1): the first token's prediction at start - 1 is in, the
    token after the range's at end - 1 is out. Red: token_rows returning
    (start, end) -- the one-row shift -- reads [20, 24) as rows 20..23."""
    assert ks.token_rows(20, 24) == (19, 23)
    assert ks.token_rows(1, 2) == (0, 1)   # token 1 is scored by row 0


def test_a_range_starting_at_token_zero_is_refused():
    """No row predicts token 0: refusing is better than masking one token short."""
    with pytest.raises(ValueError, match="predicted by no row"):
        ks.token_rows(0, 3)


def test_the_reply_ends_at_a_merged_last_token():
    """A reply ending in punctuation: the Qwen2 pretokenizer folds the period and
    the marker's leading newline into ONE token, so the turn's ids tail is that
    merged token followed by marker[1:]. The reply's own ids therefore END at the
    merged token -- dropping only the marker's ids from the tail, not its count.
    Red: reply_end returning len(tokens) - len(marker): 2 instead of 3, and the
    prediction of the reply's last token falls outside the mask."""
    full = [500, 13, 602, 601]        # text, '.', the merged '.'+newline, the end id
    marker = [600, 601]               # the marker alone: newline, end id
    assert ks.reply_end(full, marker) == 3


def test_an_unmerged_reply_ends_before_the_marker():
    full = [500, 13, 600, 601]
    marker = [600, 601]
    assert ks.reply_end(full, marker) == 2


def test_a_transcript_carries_the_marker_and_its_own_ids(tmp_path):
    p = tmp_path / "req.json"
    p.write_text('[{"role": "assistant", "tokens": [500, 602, 601],'
                 ' "marker_tokens": [600, 601]}]')
    turns = ks.read_transcript(str(p))
    assert turns[0]["marker_tokens"] == [600, 601]


def test_a_user_quote_of_the_reply_does_not_capture_the_span(tmp_path):
    """Anchoring: the located sequence includes the turn's closing marker, so a
    user turn quoting the reply's bare text cannot match it. The window holds
    the quote first (ids 500, 601 -- no marker), then the real turn. Red: the
    reply-alone form locates the quote."""
    p = tmp_path / "req.json"
    p.write_text('[{"role": "assistant", "tokens": [500, 602, 601],'
                 ' "marker_tokens": [600, 601]}]')
    window = np.array([9, 500, 601, 9, 500, 602, 601, 9], dtype=np.int32)
    spans = ks.assistant_spans(window, ks.read_transcript(str(p)))
    assert spans == [(4, 6)]    # text and the merged last token, the end id out


def test_a_reply_ending_in_a_period_lists_a_reply_span_inside_the_turn(tmp_path):
    """The end-to-end form of the merge: transcript ids as the recipe writes
    them, the reply's span ends at the merged token and excludes the end id."""
    p = tmp_path / "req.json"
    p.write_text('[{"role": "assistant", "tokens": [500, 602, 601],'
                 ' "marker_tokens": [600, 601]}]')
    window = np.array([1, 2, 500, 602, 601, 3], dtype=np.int32)
    spans = ks.assistant_spans(window, ks.read_transcript(str(p)))
    assert spans == [(2, 4)]     # text and the merged token, the end id out
