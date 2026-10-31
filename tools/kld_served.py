#!/usr/bin/env python3
r"""The llama.cpp reference-capture reader, and the per-row KL against it.

`tools/kld_capture.py` builds a capture with `llama-perplexity
--kl-divergence-base`; `tools/capture_vs_capture.py` compares two captures and
`tools/ref_forward_stream.py` writes one in the same format. This module is
the shared reader: `read_capture` (the file), `reference_log_probs` (one
recorded row to log-probs; the writer is `log_softmax(int, const float*,
uint16_t*, int)` in llama.cpp's tools/perplexity/perplexity.cpp, transcribed
here as `llama_row`), `kl_ref_vs_served` (KL(P_ref || P_other) for one row).

The cells that enforce the claims above are tests/python/test_kld_served.py
(reconstruction to the quantisation step; KL zero for the same row and
positive for a moved one; the reader refusing a file that is not a capture;
the assistant spans from a transcript's role ids; the row/token convention).
The half that read a served-logits dump left with the retired engine
(docs/history.md).

--------------------------------------------------------------------------
A ROW SCORES THE NEXT TOKEN
--------------------------------------------------------------------------

A capture row is a DISTRIBUTION OVER THE NEXT TOKEN, not over the token it sits
at. llama.cpp records at row `i` the logits of token position `first + i` and
scores `tokens[i+1]` there (`tools/perplexity/perplexity.cpp:157-160`, the
`log_softmax(..., tokens[i+1])` that writes every recorded row; `first =
n_ctx/2` at `:540`, the row's token base at `:610`). So row position `p` holds
the prediction of token `p + 1`, and a token range `[start, end)` maps to the
row position range `[start - 1, end - 1)` (`token_rows`). The off-by-one is the
point: the naive mapping drops the prediction of the range's FIRST token -- for
a reply, the one right after the assistant header, usually the highest-entropy
token in it -- and adds the prediction of the token after the reply.

--------------------------------------------------------------------------
THE ASSISTANT MASK, AND WHAT CANNOT HAVE ONE
--------------------------------------------------------------------------

A capture scores every recorded position, and a served chat arm's quality
lives in the positions the model itself writes: the reply's tokens, and the
earlier assistant turns of a multi-turn transcript. `assistant_spans` derives
those token ranges from the request's roles and the turns' token ids -- the
ids the capture's window HOLDS, not text re-tokenised here: this module has no
tokenizer, and a guess would be worse than a refusal.

THE TURN'S IDS ARE TOKENISED WITH ITS CLOSING MARKER, NOT ALONE. With specials
unparsed -- what the capture does -- the end marker is TEXT, so it is subject to
the pretokenizer: the Qwen2 pre-type's regex alternative
` ?[^\s\p{L}\p{N}]+[\r\n]*` (`src/llama-vocab.cpp:389` in the pinned
llama.cpp; the STABLELM2, QWEN2, HUNYUAN and SOLAR_OPEN pre-types share that
block) lets a run of non-space, non-letter, non-digit characters absorb the
newlines that follow it. A reply ending in punctuation can therefore share its
last token with the marker's leading characters, and a standalone tokenisation
of the reply reads a different last id. Whether it does is MODEL- AND
TEXT-DEPENDENT: this module has no tokenizer and no measured case, so the merge
is a HEURISTIC, not an observation (the fixture tests use synthetic ids).

A transcript therefore carries, per assistant turn, `tokens` = the ids of the
reply's text PLUS its closing marker tokenised together as one stream -- the
sequence the window holds -- and `marker_tokens` = the ids of the closing marker
alone at the same tokenisation settings (`tools/tokenize_turn.py` writes both
with `--no-parse-special`, the pin's own tokenize switch
(`common/arg.cpp:3255-3261`, forwarded to common_tokenize at
`tools/tokenize/tokenize.cpp:185`)); `reply_end` drops the marker's ids from the tail
and keeps a merged last token, trying the exact marker first and `marker_tokens`
minus its first id second. A turn whose ids do not occur in the window is
refused by name. A transcript formed without `marker_tokens` is the reply alone,
exact only when its last token does not merge (stated in the tool's output); the
marker form is the one the recipe documents.

The capture's window is the rendered prompt plus the reply, so the earlier
assistant turns sit inside the prompt and the reply at its end. Each turn's id
sequence is located forward from the previous turn's end, and the sequence
includes the turn's closing marker: a user turn quoting the reply's text
without the marker cannot be taken for the reply.

A plain-text corpus capture (`llama-perplexity --kl-divergence-base` over a
corpus file) has NO roles. There is nothing to mask there and nothing is
invented: every recorded position is scored, and the caller says so.

--------------------------------------------------------------------------
THE FP32 LOGIT DIAGNOSTIC: NOT NEEDED, AND THE LIMIT OF WHAT WAS CHECKED
--------------------------------------------------------------------------

The audit sketched an optional `LLAMA_LOGITS_F32=1` at the LM head. The graph
DECLARES f32: the head's `ggml_mul_mat` result tensor is `GGML_TYPE_F32`
(`ggml/src/ggml.c:3370` in the pinned llama.cpp), and qwen4exp takes that tensor
unmodified as `res->t_logits` (`src/models/qwen4exp.cpp:519-521`, the MTP head
at `:613-615`). The MoE router asks for f32 accumulation explicitly too, but
CONDITIONALLY -- `ggml_prec_set_acc(logits, GGML_PREC_F32)` sits inside `if
(gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS)`
(`src/llama-graph.cpp:2170-2171` in the served-series tree; `:2155` is the
unpatched pin's line of the same call), so on a gating function other than
SOFTMAX it does not apply. A switch asking the head for f32 would be a no-op on
the declared result type.

What a declaration does NOT settle is the KERNEL's arithmetic, and the
reference's BF16-logit finding is about exactly that. Read here, at the
dot-product accumulator level, in kernels whose result tensor this series
deposits as f32: `float sumf[N_R0_MXFP4]`
(`ggml/src/ggml-opencl/kernels/mul_mv_mxfp4_f32_flat.cl:127`), `float8 sumf`
(`kernels/mul_mv_iq4_nl_f32_flat.cl:114`), `float4 sumf` in the expert-routed
GEMV (`kernels/mul_mv_id_q8_0_f32_flat.cl:136`), and the router kernel this
tree adds (`float wsum`,
`contrib/llama.cpp/patches/0064-opencl-intel-topk-moe-subgroup.patch`).

That is NOT a claim about the LM head. The head is a dense matmul, neither
expert-routed nor one of those three weight types on the served GGUFs, and
which kernel a given shape is dispatched to was NOT examined -- nor which
Adreno/Intel path wins, nor the f16 GEMM accumulator forms. The claim is only:
the accumulator forms read are `float`, and the LM head's f32 path was not
audited beyond its declared result type.
"""
import json
import struct

import numpy as np

# Rows at 0-based index < 2051 and >= 2051 were reported apart for Flash-Next
# (its 12 full-attention layers' boundary); capture_vs_capture.py splits there.
QSA_BOUNDARY_TOKENS = 2051


# ---------------------------------------------------------------- the capture
def read_capture(path):
    """The llama.cpp `--kl-divergence-base` file. Returns (n_ctx, n_vocab,
    n_chunk, tokens [n_chunk, n_ctx] int32, rows [n_chunk, n_rows, nv]
    uint16 memmap) with nv = 2 * ((n_vocab + 1) // 2) + 4 and n_rows =
    n_ctx - 1 - n_ctx // 2."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"_logits_":
            raise ValueError(f"{path}: not a llama.cpp logits capture (magic {magic!r})")
        n_ctx = struct.unpack("<i", f.read(4))[0]
        n_vocab = struct.unpack("<i", f.read(4))[0]
        n_chunk = struct.unpack("<i", f.read(4))[0]
        tokens = np.fromfile(f, dtype=np.int32, count=n_chunk * n_ctx).reshape(n_chunk, n_ctx)
        off = f.tell()
    first = n_ctx // 2
    n_rows = n_ctx - 1 - first
    nv = 2 * ((n_vocab + 1) // 2) + 4
    rows = np.memmap(path, dtype=np.uint16, mode="r", offset=off,
                     shape=(n_chunk, n_rows, nv))
    return n_ctx, n_vocab, n_chunk, tokens, rows


def llama_row(logits):
    """llama.cpp perplexity.cpp `log_softmax(int, const float*, uint16_t*, int)`,
    transcribed: the writer of every recorded row."""
    logits = np.asarray(logits, dtype=np.float32)
    max_logit = float(logits.max())
    min_logit = max(float(logits.min()), max_logit - 16)
    log_sum_exp = float(np.log(np.exp(logits - max_logit).sum()))
    min_log_prob = min_logit - max_logit - log_sum_exp
    scale = (max_logit - min_logit) / 65535.0
    row = np.zeros(2 * ((len(logits) + 1) // 2) + 4, dtype=np.uint16)
    row[:4] = np.array([scale, min_log_prob], dtype=np.float32).view(np.uint16)
    if scale:
        q = np.where(logits > min_logit, np.rint((logits - min_logit) / scale), 0)
        row[4:4 + len(logits)] = q.astype(np.uint16)
    return row


def reference_log_probs(row_u16, n_vocab):
    """One recorded row -> log P_ref over the vocab, f64. The writer stores
    scale (f32), min_log_prob (f32), then q[i] = round((logit_i - min_logit)
    / scale) for logit_i > min_logit else 0, with min_logit clamped to
    max_logit - 16; so log P(i) = min_log_prob + q[i] * scale, and a q of 0
    is the floor (16 nats below the max, or lower)."""
    hdr = np.asarray(row_u16[:4]).view(np.float32)
    scale, min_log_prob = float(hdr[0]), float(hdr[1])
    q = np.asarray(row_u16[4:4 + n_vocab], dtype=np.float64)
    return min_log_prob + q * scale


def log_softmax(x):
    z = x - x.max()
    return z - np.log(np.exp(z).sum())


def kl_ref_vs_served(ref_logp, served_logits):
    """KL(P_ref || P_served) in nats for one row. P_ref from the capture's
    reconstructed log-probs (renormalised: the uint16 floor makes them sum to
    slightly under 1), P_served from a log-softmax of the served f32 row."""
    ref = log_softmax(np.asarray(ref_logp, dtype=np.float64))
    cand = log_softmax(np.asarray(served_logits, dtype=np.float64))
    p = np.exp(ref)
    return float((p * (ref - cand)).sum())


# ------------------------------------------------- row and token conventions
def token_rows(start, end):
    """A token range [start, end) as the row position range it is scored in:
    [start - 1, end - 1). Row position `p` scores token `p + 1` (see the module
    docstring), so the row predicting the range's first token is at `start - 1`
    and the row predicting the token after the range is at `end - 1`, outside.
    A range starting at token 0 has no row for its first token -- that token is
    in no context -- so it is refused rather than silently scored short."""
    if start < 1:
        raise ValueError(f"token range [{start}, {end}): token {start} is predicted by no row "
                         "(row p scores token p+1), so it cannot be masked")
    return start - 1, end - 1


# ------------------------------------------------- assistant spans from a request
def read_transcript(path):
    """A chat request's turns in order, each with a `role` and -- an assistant
    turn -- `tokens`: the ids of the turn's text PLUS its closing marker,
    tokenised together (the sequence the capture's window holds), and optional
    `marker_tokens`: the ids of the closing marker alone at the same settings,
    which say where the reply's own ids end (`reply_end`). Without
    `marker_tokens` the turn's `tokens` are taken as the reply alone -- exact
    only when the reply's last token does not merge with the marker. JSON is
    either {"turns": [...]} or a bare list; a non-assistant turn may omit its
    ids."""
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    turns = obj.get("turns") if isinstance(obj, dict) else obj
    if not isinstance(turns, list):
        raise ValueError(f"{path}: expected a list of turns or {{'turns': [...]}}")
    out = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or not isinstance(t.get("role"), str):
            raise ValueError(f"{path}: turn {i} has no role")
        ids = t.get("tokens") or []
        mark = t.get("marker_tokens") or []
        if t["role"] == "assistant":
            if not ids:
                raise ValueError(f"{path}: assistant turn {i} carries no token ids; the ids the "
                                 "capture's window holds are what the mask is located by, and this "
                                 "tool cannot recompute them from text")
            if not all(isinstance(v, int) for v in ids + mark):
                raise ValueError(f"{path}: assistant turn {i} has non-integer token ids")
            out.append({"role": t["role"], "tokens": [int(v) for v in ids],
                        "marker_tokens": [int(v) for v in mark]})
        else:
            out.append({"role": t["role"], "tokens": [int(v) for v in ids], "marker_tokens": []})
    return out


def reply_end(tokens, marker_tokens):
    """How many of `tokens` belong to the reply: the marker's ids are dropped
    from the tail. The marker is tokenised alone at the capture's settings, so
    its first id is the one a trailing punctuation token could merge with --
    the turn's tail then holds that merged token and `marker_tokens[1:]`. The
    exact marker is tried first, so a merge is read only when the exact marker
    does not fit; a reply whose own tail spells the marker exactly is the case
    the reply-alone form exists for.

    The merge is a HEURISTIC over the pretokenizer's regex (see the module
    docstring), not a measured case: no tokenizer runs here, so nothing says a
    given reply merges. What the rule guarantees is that a tail matching
    neither form is REFUSED rather than read as a longer reply."""
    if not marker_tokens:
        return len(tokens)
    m = len(marker_tokens)
    if tokens[-m:] == marker_tokens:
        return len(tokens) - m
    if m > 1 and tokens[-(m - 1):] == marker_tokens[1:]:
        return len(tokens) - (m - 1)
    raise ValueError(f"the turn's ids ({len(tokens)}) do not end with the marker's ids "
                     f"({m}): neither the marker nor the marker minus its merged first token "
                     "is the tail, so where the reply ends cannot be read -- re-tokenise the "
                     "turn together with its closing marker")


def find_ids(tokens, needle, start=0):
    """The first index >= `start` where `needle` occurs contiguously in
    `tokens`, or None. One window and a few hundred ids a turn, so a linear
    scan is the whole cost."""
    win = np.asarray(tokens, dtype=np.int64).tolist()
    nd = [int(v) for v in needle]
    for i in range(start, len(win) - len(nd) + 1):
        if win[i:i + len(nd)] == nd:
            return i
    return None


def assistant_spans(window_tokens, turns):
    """[(start, end)] TOKEN ranges, in turn order, of the assistant turns inside
    a capture's window -- the reply's own tokens, the closing marker excluded
    (`reply_end`). The window is the request's rendered prompt plus the reply,
    so the earlier assistant turns sit in the prompt and the reply at its end.
    Each turn's full id sequence (reply plus marker) is located forward from the
    previous turn's end; because that sequence includes the closing marker, a
    user turn quoting the reply's text without the marker cannot be taken for
    it. Refuses a transcript with no assistant turn and a turn whose ids do not
    occur in the window -- a reply not in the capture is a real mismatch, not a
    reason to mask the wrong rows."""
    cursor = 0
    spans = []
    for i, t in enumerate(turns):
        if t["role"] != "assistant":
            continue
        ids = t["tokens"]
        pos = find_ids(window_tokens, ids, cursor)
        if pos is None:
            raise ValueError(f"assistant turn {i} ({len(ids)} tokens) is not in the capture's window: "
                             "a span located by ids cannot be guessed for text that the "
                             "window does not hold")
        spans.append((pos, pos + reply_end(ids, t.get("marker_tokens") or [])))
        cursor = pos + len(ids)
    if not spans:
        raise ValueError("the transcript has no assistant turn; nothing can be masked")
    return spans


# ---------------------------------------------------------------- the KL rows
def kl_rows(ref_rows, other_rows, n_vocab):
    """Per-row KL and argmax agreement between two captures' row arrays (the
    reference's window 0 and another capture's window W). Both arrays are
    uint16 records, shape (n_rows, nv); the returned arrays are one entry per
    row, so a position mask can select a subset without recomputing."""
    n = min(len(ref_rows), len(other_rows))
    kl = np.empty(n, dtype=np.float64)
    agree = np.empty(n, dtype=bool)
    for i in range(n):
        r = log_softmax(reference_log_probs(ref_rows[i], n_vocab))
        o = log_softmax(reference_log_probs(other_rows[i], n_vocab))
        kl[i] = float((np.exp(r) * (r - o)).sum())
        agree[i] = np.argmax(r) == np.argmax(o)
    return kl, agree
