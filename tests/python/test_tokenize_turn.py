"""tools/tokenize_turn.py: the transcript's ids at the capture's own settings.

A capture tokenises its prompt with `parse_special = false`: perplexity's
`common_tokenize(ctx, prompt, true)` omits it and the default is false
(common/common.h:1159-1163; the calls at tools/perplexity/perplexity.cpp:310 and
the KL mode's :473), so a closing marker inside a prompt is TEXT there.
The pin's `llama-tokenize` defaults the other way (`parse_special = true` for
LLAMA_EXAMPLE_TOKENIZE, common/arg.cpp:1408) and the affirmative `--parse-special`
is imatrix-only (:3227-3233), while `--no-parse-special` is the tokenize
example's own switch (:3255-3261, passed to common_tokenize at
tools/tokenize/tokenize.cpp:185). The helper is what pins that setting: its argv
always carries `--no-parse-special` and `--no-bos`, so a marker is tokenised as
text and the transcript matches the window.

Red first, each mutation seen failing:
- `--no-parse-special` dropped from the argv: test_the_argv_pins_the_capture_
  tokenisation FAILS (the marker would then be one special id).
- the text and the marker written as two files instead of one concatenated
  stream: test_the_turn_is_tokenised_as_one_stream FAILS, the tokenizer sees a
  trailing newline the rendered turn does not have.
- a non-zero exit read as ids (the raise replaced by `return []`):
  test_a_failing_tokenizer_is_refused FAILS.
"""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

import tokenize_turn as tt


FAKE = """#!/usr/bin/env python3
import json, sys
argv = sys.argv[1:]
_p = LOG
log = {"calls": json.load(open(_p))["calls"] if __import__("os").path.exists(_p) else []}
i = argv.index("-f")
path = argv[i + 1]
data = open(path, "rb").read().decode("utf-8")
# the fake stands in for a tokenizer: a stream with text gets the turn's ids, the
# marker alone (or any stream without the reply's text) gets the marker's
log["calls"].append({"argv": argv, "file": data,
                     "ids": (FULL if "." in data else MARK)})
if "-m" in argv and argv[argv.index("-m") + 1].endswith("bad.gguf"):
    open(LOG, "w").write(json.dumps(log))
    sys.exit(4)
open(LOG, "w").write(json.dumps(log))
print(json.dumps(FULL if "." in data else MARK))
"""


def runner(tmp_path, full=(500, 602, 601), mark=(600, 601)):
    """A fake llama-tokenize that logs every call: the argv, the file's bytes, the ids."""
    p = tmp_path / "llama-tokenize"
    log = tmp_path / "log.json"
    p.write_text(FAKE.replace("LOG", repr(str(log)))
                 .replace("FULL", repr(list(full))).replace("MARK", repr(list(mark))))
    p.chmod(0o755)
    return p


def log_of(tmp_path):
    p = tmp_path / "log.json"
    return json.loads(p.read_text()) if p.exists() else None


def test_the_argv_pins_the_capture_tokenisation(tmp_path):
    """The marker must be TEXT, as the capture sees it. Red: without
    --no-parse-special llama-tokenize (default parse_special=true) turns the end
    marker into one special id and the located span is refused downstream."""
    r = runner(tmp_path)
    text = tmp_path / "text.txt"
    text.write_text("a reply.")
    tt.tokenize_ids(str(r), "model.gguf", str(text))
    argv = log_of(tmp_path)["calls"][0]["argv"]
    assert "--no-parse-special" in argv
    assert "--no-bos" in argv
    assert "--ids" in argv


def test_the_turn_is_tokenised_as_one_stream(tmp_path):
    """The turn's text and its closing marker are ONE tokenisation, exactly the
    bytes the rendered window holds -- no added newline between them. Red:
    tokenising the two separately, or joining with "\n", gives the pretokenizer
    a boundary the window does not have."""
    r = runner(tmp_path)
    text = tmp_path / "text.txt"
    marker = tmp_path / "marker.txt"
    text.write_text("a reply.")
    marker.write_text(" END")
    tokens, marker_tokens = tt.turn_ids(str(r), "model.gguf", str(text), str(marker))
    calls = log_of(tmp_path)["calls"]
    assert calls[0]["file"] == "a reply. END"
    assert tokens == [500, 602, 601]
    assert marker_tokens == [600, 601]


def test_the_marker_is_tokenised_alone_at_the_same_setting(tmp_path):
    """marker_tokens is the marker's own ids, at the same flags: the reader needs
    it to tell a merged last token from an unmerged one."""
    r = runner(tmp_path)
    text = tmp_path / "text.txt"
    marker = tmp_path / "marker.txt"
    text.write_text("a reply.")
    marker.write_text(" END")
    tt.turn_ids(str(r), "model.gguf", str(text), str(marker))
    calls = log_of(tmp_path)["calls"]
    assert len(calls) == 2
    assert "--no-parse-special" in calls[1]["argv"]
    assert calls[1]["file"] == " END"


def test_a_failing_tokenizer_is_refused(tmp_path):
    """A missing model or an unknown flag is not an empty id list: red under
    'treat a non-zero exit as ids', which would write a transcript of nothing."""
    r = runner(tmp_path)
    text = tmp_path / "text.txt"
    text.write_text("a reply.")
    with pytest.raises(RuntimeError, match="llama-tokenize"):
        tt.tokenize_ids(str(r), "bad.gguf", str(text))


def test_a_transcript_carries_the_turn_and_the_marker(tmp_path):
    r = runner(tmp_path)
    text = tmp_path / "text.txt"
    marker = tmp_path / "marker.txt"
    text.write_text("a reply.")
    marker.write_text(" END")
    tr = tmp_path / "request.json"
    tt.append_turn(str(tr), "user", [1, 2])
    tt.append_turn(str(tr), "assistant",
                   *tt.turn_ids(str(r), "model.gguf", str(text), str(marker)))
    turns = json.loads(tr.read_text())
    assert turns[0] == {"role": "user", "tokens": [1, 2]}
    assert turns[1]["role"] == "assistant"
    assert turns[1]["tokens"] == [500, 602, 601]
    assert turns[1]["marker_tokens"] == [600, 601]


def test_the_appended_transcript_locates_the_span(tmp_path):
    """End to end with the reader: a transcript the helper writes is one
    assistant_spans can locate (the fake's ids stand in for the tokenizer's
    answer; the merge rule then drops the marker's ids from the tail)."""
    import numpy as np
    import kld_served as ks

    r = runner(tmp_path)
    text = tmp_path / "text.txt"
    marker = tmp_path / "marker.txt"
    text.write_text("a reply.")
    marker.write_text(" END")
    tr = tmp_path / "request.json"
    tt.append_turn(str(tr), "assistant",
                   *tt.turn_ids(str(r), "model.gguf", str(text), str(marker)))
    window = np.array([9, 500, 602, 601, 9], dtype=np.int32)
    spans = ks.assistant_spans(window, ks.read_transcript(str(tr)))
    # the turn's 3 ids, its tail matching the marker's last id: merge assumed,
    # so the reply's own ids are the first 2 and the span is (1, 3)
    assert spans == [(1, 3)]
