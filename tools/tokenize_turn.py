#!/usr/bin/env python3
"""Write the transcript's ids at the CAPTURE's tokenisation settings.

`tools/kld_served.py`'s `assistant_spans` locates a turn by the ids the capture's
window holds, so those ids have to be produced at the setting the capture used.
llama-perplexity tokenises its prompt with `parse_special = false`: its
`common_tokenize(ctx, prompt, true)` omits the flag, which defaults to `false`
(`common/common.h:1159-1163`; the calls at `tools/perplexity/perplexity.cpp:310`
and, in the KL mode, `:473`). The pin's `llama-tokenize` defaults
the other way for its own example (`parse_special = true`,
`common/arg.cpp:1408`) -- the end marker would then be one SPECIAL id, not the
text the capture holds -- and the affirmative `--parse-special` is scoped to the
imatrix example (`:3227-3233`). The tokenize example's switch is
`--no-parse-special` (`common/arg.cpp:3255-3261`, passed through to
common_tokenize at `tools/tokenize/tokenize.cpp:185`).

So this helper's argv always carries `--no-parse-special` and `--no-bos`; it is
the one place the flag is written, so no caller can forget it.

A turn is tokenised as ONE stream: its text followed by the template's closing
marker, byte for byte, because that is what the rendered prompt holds. The
marker is also tokenised alone at the same setting -- the transcript's
`marker_tokens` -- so `kld_served.reply_end` can tell a merged last token from
an unmerged one. The concatenation goes through a temporary file rather than
`-p`, so no shell quoting or escape processing can reach the bytes.

Usage (per assistant turn of the rendered request):

    python3 tools/tokenize_turn.py --tokenize <tree>/build-tools/bin/llama-tokenize \\
        --model <model>.gguf --text reply.txt --marker marker.txt \\
        --transcript request.json

`marker.txt` holds the template's closing marker alone -- built from parts,
never typed literally in this repository -- and `reply.txt` the reply's content
byte for byte (`enable_thinking:false`, so it equals the served `content`).
Repeat for earlier assistant turns and for the user turns (`--text` only), then
pass the transcript to `tools/capture_vs_capture.py --spans request.json`.

The merge rule `reply_end` implements is a heuristic over the pretokenizer's
regex, not a measured case (see `kld_served.py`): a reply ending in punctuation
may share its last token with the marker's leading characters. This helper
supplies `marker_tokens` so that case is caught rather than guessed.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


def tokenize_ids(tokenizer, model, path):
    """One llama-tokenize call at the capture's settings; the id list or raise.

    `--no-parse-special` is not optional here: without it the end marker is a
    special token and the located span would be refused by the reader."""
    argv = [tokenizer, "-m", model, "--no-bos", "--no-parse-special",
            "-f", path, "--ids"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"llama-tokenize not found at {tokenizer}: {e}") from e
    out = proc.stdout.strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-tokenize failed ({proc.returncode}) on {model}: {proc.stderr.strip() or out}. "
            "The flag this helper needs is --no-parse-special (common/arg.cpp:3255-3261 at "
            "the pin); a tree without it cannot produce the capture's tokenisation.")
    try:
        ids = json.loads(out)
    except ValueError as e:
        raise RuntimeError(f"llama-tokenize printed no id list: {out[:200]!r}") from e
    if not isinstance(ids, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        raise RuntimeError(f"llama-tokenize printed no id list: {out[:200]!r}")
    return ids


def _one_stream(text, marker):
    """Write text + marker to a temporary file, exactly the bytes, and return the path."""
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.write(marker)
    return path


def turn_ids(tokenizer, model, text_path, marker_path=None):
    """(tokens, marker_tokens) for one turn; `tokens` is text+marker as one stream.

    Without a marker (a user turn, or a reply whose template adds nothing), the
    ids are the text alone and `marker_tokens` is empty."""
    with open(text_path, encoding="utf-8", newline="") as f:
        text = f.read()
    marker = ""
    if marker_path:
        with open(marker_path, encoding="utf-8", newline="") as f:
            marker = f.read()
    merged = _one_stream(text, marker)
    try:
        tokens = tokenize_ids(tokenizer, model, merged)
    finally:
        os.unlink(merged)
    marker_tokens = tokenize_ids(tokenizer, model, marker_path) if marker_path else []
    return tokens, marker_tokens


def append_turn(transcript, role, tokens, marker_tokens=None):
    """Add a turn to the transcript JSON, keeping any turns already there."""
    turns = []
    if os.path.exists(transcript):
        with open(transcript, encoding="utf-8") as f:
            text = f.read().strip()
        turns = json.loads(text) if text else []
    turn = {"role": role, "tokens": list(tokens)}
    if marker_tokens:
        turn["marker_tokens"] = list(marker_tokens)
    turns.append(turn)
    with open(transcript, "w", encoding="utf-8") as f:
        json.dump(turns, f, indent=2)
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenize", required=True, help="the pin's llama-tokenize binary")
    ap.add_argument("--model", required=True, help="the GGUF the capture ran on")
    ap.add_argument("--text", required=True, help="the turn's text, byte for byte")
    ap.add_argument("--marker", default=None,
                    help="the template's closing marker alone (assistant turns); omit for a user turn")
    ap.add_argument("--role", default="assistant")
    ap.add_argument("--transcript", required=True, help="the request.json to append to")
    args = ap.parse_args(argv)
    tokens, marker_tokens = turn_ids(args.tokenize, args.model, args.text, args.marker)
    append_turn(args.transcript, args.role, tokens, marker_tokens)
    print(f"{args.role}: {len(tokens)} ids (text+marker), {len(marker_tokens)} marker ids -> {args.transcript}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
