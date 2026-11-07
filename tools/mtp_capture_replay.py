#!/usr/bin/env python3
"""Replay a corpus through a server running ARCINT_MTP_CAPTURE.

Campaign mtp-head-fit, experiment E1. The corpus is JSON lines made of the
coder's own generated text (never session content); a record has `kind`
(code, prose, think, tool), `tokens` (the completion's token count), `q` (the
task string), `think` (the thinking text or null), `text` (the answer or null)
and `calls`. This tool sends each text as the PROMPT of a completion with
max_tokens 1, so the server decodes every token of it once and the capture
(src/exec/mtp_capture.h) writes the target's row and the head's row at every
position. It writes the record index tools/mtp_fit.py reads.

    mtp_capture_replay.py PORT CORPUS.jsonl OUT.idx [--host H] [--with-question]

Text classes. A record's answer text is the class of its `kind`; its thinking
text is its own class, `think`, whatever the kind (a record of kind think has
thinking text and an empty answer). The tool class (calls only; 374 tokens in
all) is dropped.

Ids. The text is tokenised by the server itself (POST /tokenize) and sent as
the OpenAI token-array prompt, so the ids the capture holds are the ids this
tool counted, with no second tokenisation. The corpus stores text, not ids:
each re-tokenised count is checked against the record's `tokens` and every
mismatch is reported (a thinking-only record's count is the thinking text's; a
record with both texts is checked on the sum). The text is replayed alone, with
no chat template around it, so the first positions of a text have no context:
tools/mtp_fit.py drops the first --skip-first of them. --with-question sends
the task string's ids first (as plain text, not the chat template) and records
how many rows that context took, so the fit skips them too.

Fresh prefix. The server's prefix reuse would skip the rows of a prompt that
shares a start with the lane's previous one, and those rows would be missing
from the capture. This tool does not try to defeat it from the outside: a
server running with ARCINT_MTP_CAPTURE starts every request on an empty lane
(src/exec/backend_llama.cpp), which is the one way that cannot miss a position
whatever the previous request was. Anything else that sends a request to the
same server while a capture runs adds its rows too; the fit finds each record's
rows by the hash of its token ids, so strays are harmless.

The index (JSON, format mtp-capture-records-1) holds, per replayed text: its
position `i`, `kind`, `cls`, `task`, the corpus `tokens` and the counted `n_ids`,
`ctx` (question rows), and `ids_sha1`, the SHA-1 of the int32 little-endian ids
without the last (the last token's rows are not captured: its next token and
its rows come from the verify path), which is how the fit finds the rows. It
is private data about private text; it is never committed. Any HTTP failure
stops the run with the index of what was done, `complete` false.
"""
import argparse
import hashlib
import json
import struct
import sys
import time
import urllib.error
import urllib.request

DROP_KINDS = ("tool",)


def segments(rec):
    """[(cls, text)] of one corpus record: the thinking text as class think,
    the answer text as the record's kind."""
    out = []
    if rec.get("think"):
        out.append(("think", rec["think"]))
    if rec.get("text") and rec["kind"] not in DROP_KINDS:
        out.append((rec["kind"], rec["text"]))
    return out


def ids_sha1(ids):
    """SHA-1 of the ids without the last, int32 little endian: what the capture's
    token column holds for the request."""
    return hashlib.sha1(struct.pack("<%di" % (len(ids) - 1), *ids[:-1])).hexdigest()


def post(base, path, body, key=None, timeout=3600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    # no proxy: the server is the one named on the command line
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tokenize(base, text, key=None):
    r = post(base, "/tokenize", {"content": text, "add_special": False}, key, 600)
    toks = r["tokens"]
    if toks and isinstance(toks[0], dict):
        toks = [t["id"] for t in toks]
    return [int(t) for t in toks]


def plan(corpus_lines):
    """The replay units, in corpus order: (corpus position, record, cls, text)."""
    units = []
    for n, line in enumerate(corpus_lines):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        for cls, text in segments(rec):
            units.append((n, rec, cls, text))
    return units


def count_check(units_by_rec, rec_tokens):
    """Mismatches between the re-tokenised counts and the corpus' `tokens`:
    [(corpus position, kind, counted, stored)]. A record is checked on the sum
    of its replayed texts."""
    out = []
    for n, (kind, counted) in sorted(units_by_rec.items()):
        if counted != rec_tokens[n]:
            out.append((n, kind, counted, rec_tokens[n]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", type=int)
    ap.add_argument("corpus")
    ap.add_argument("out")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--with-question", action="store_true", help="send the task string's ids before each text and record the rows they took")
    ap.add_argument("--key", help="API key (bearer), if the server wants one")
    ap.add_argument("--limit", type=int, default=0, help="only the first N texts (a dry run)")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds a request may take (the first compiles kernels)")
    args = ap.parse_args(argv)

    base = "http://%s:%d" % (args.host, args.port)
    with open(args.corpus) as f:
        lines = f.readlines()
    recs = [json.loads(l) for l in lines if l.strip()]
    units = plan(lines)
    if args.limit:
        units = units[:args.limit]
    print("replaying %d texts of %d corpus records through %s" % (len(units), len(recs), base))

    out = {"format": "mtp-capture-records-1", "complete": False, "with_question": args.with_question, "records": []}
    counts = {}
    rec_tokens = {}
    t0 = time.time()
    rows_total = 0
    try:
        for i, (n, rec, cls, text) in enumerate(units):
            ids = tokenize(base, text, args.key)
            if len(ids) < 2:
                print("text %d (%s): %d token(s), skipped" % (i, cls, len(ids)))
                continue
            ctx = 0
            send = ids
            if args.with_question:
                q = tokenize(base, rec["q"], args.key)
                send = q + ids
                ctx = len(q)
            body = {"prompt": send, "max_tokens": 1, "temperature": 0, "stream": False}
            try:
                resp = post(base, "/v1/completions", body, args.key, args.timeout)
            except urllib.error.HTTPError as e:
                raise RuntimeError("text %d: HTTP %d %s" % (i, e.code, e.read().decode("utf-8", "replace")[:300]))
            used = (resp.get("usage") or {}).get("prompt_tokens")
            if used is not None and used != len(send):
                raise RuntimeError("text %d: the server counted %d prompt tokens, %d were sent" % (i, used, len(send)))
            k, c = counts.get(n, (rec["kind"], 0))
            counts[n] = (k, c + len(ids))
            rec_tokens[n] = rec["tokens"]
            out["records"].append({
                "i": i, "corpus": n, "kind": rec["kind"], "cls": cls, "task": rec["q"],
                "tokens": rec["tokens"], "n_ids": len(ids), "ctx": ctx, "ids_sha1": ids_sha1(send),
            })
            rows_total += len(send) - 1
            if (i + 1) % 20 == 0 or i + 1 == len(units):
                print("  %d/%d texts, %d rows, %.0f s" % (i + 1, len(units), rows_total, time.time() - t0), flush=True)
        out["complete"] = True
    except (RuntimeError, urllib.error.URLError, OSError) as e:
        print("STOPPED: %s" % e, file=sys.stderr)
    with open(args.out, "w") as f:
        json.dump(out, f)

    # a record is checked once both its texts are done
    done = {r["corpus"] for r in out["records"]}
    full = {n: v for n, v in counts.items() if n in done}
    mism = count_check(full, rec_tokens)
    print("rows %d expected in the capture (one less than the ids of each text)" % rows_total)
    if mism:
        print("%d of %d records re-tokenise to another count than the corpus' `tokens`:" % (len(mism), len(full)))
        for n, kind, got, was in mism[:20]:
            print("  record %d (%s): %d counted, %d stored (%+d)" % (n, kind, got, was, got - was))
        if len(mism) > 20:
            print("  ... %d more" % (len(mism) - 20))
    else:
        print("every re-tokenised count equals the corpus' `tokens`")
    print("index: %s (complete: %s)" % (args.out, out["complete"]))
    return 0 if out["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
