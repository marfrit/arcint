#!/usr/bin/env python3
"""The STT gate: word error rate on a fixed German test set against the same
model's CPU run (`docs/milestone-1.2.0.md` C3, B2).

WER = (substitutions + deletions + insertions) / reference words, corpus-wide
over the set. The comparison is comparative, as the language services' KL bar:
the candidate (the card run) may be at most `--tol` worse than the CPU run of
the *same model on the same clips*; identical inputs are the gate's premise, so
`compare` refuses two arms whose clip list or model differ. An absolute floor
(`--floor`) may also be set; by default only the CPU comparison gates.

Transcription comes from an OpenAI `/v1/audio/transcriptions` server
(`multipart/form-data`, `language`); that is the route the STT unit is to
serve (`docs/speech-service.md`). The manifest is a list of
`{"audio": path, "text": reference, "sha256": optional}`.

Exit status: 0 inside the bar; 1 outside; 2 a usage or decode error.

    stt_gate.py run --url URL --model NAME --manifest CLIPS.json --out DIR \
                    [--language de --return text --tol 0.005 --floor none]
    stt_gate.py compare --cpu DIR --card DIR --tol 0.005
"""
import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request

TOL_DEFAULT = 0.005   # candidate WER may exceed the CPU arm's by this (proposal, D8)


def norm_words(text):
    text = text.lower()
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if w]


def wer(ref, hyp):
    """(S+D+I)/N over the whole sentence, at word level."""
    r = norm_words(ref)
    h = norm_words(hyp)
    # Levenshtein over word tokens; substitutions, deletions, insertions each 1
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cost = 0 if rw == hw else 1
            cur.append(min(prev[j] + 1,        # deletion (ref word missing in hyp)
                           cur[j - 1] + 1,     # insertion (extra hyp word)
                           prev[j - 1] + cost))  # substitution/match
        prev = cur
    dist = prev[-1]
    n = max(1, len(r))
    return dist / n, dist, len(r)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def multipart(audio_path, model, language, response_format):
    boundary = "----arcintstt"
    with open(audio_path, "rb") as f:
        data = f.read()
    parts = []
    def field(name, value):
        parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                      % (boundary, name, value)).encode())
    field("model", model)
    if language:
        field("language", language)
    field("response_format", response_format)
    parts.append(("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
                  "Content-Type: audio/wav\r\n\r\n" % (boundary, os.path.basename(audio_path))).encode())
    parts.append(data)
    parts.append(b"\r\n")   # CRLF before the closing delimiter, per RFC 2046
    parts.append(("--%s--\r\n" % boundary).encode())
    return b"".join(parts), "multipart/form-data; boundary=%s" % boundary


def transcribe(url, model, audio_path, language, timeout):
    body, ctype = multipart(audio_path, model, language, "json")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    return out.get("text", "")


def run(args):
    manifest = json.load(open(args.manifest))
    if not isinstance(manifest, list) or not manifest:
        print("manifest must be a non-empty list", file=sys.stderr)
        return 2
    msha = sha256_file(args.manifest)
    os.makedirs(args.out, exist_ok=True)
    cases, tot_d, tot_n = [], 0, 0
    for case in manifest:
        path = case["audio"]
        if not os.path.exists(path):
            print("clip missing: %s" % path, file=sys.stderr)
            return 2
        sha = sha256_file(path)
        if "sha256" in case and case["sha256"] != sha:
            print("clip sha256 mismatch: %s" % path, file=sys.stderr)
            return 2
        hyp = transcribe(args.url, args.model, path, args.language, args.timeout)
        w, d, n = wer(case["text"], hyp)
        tot_d += d
        tot_n += n
        cases.append({"audio": path, "sha256": sha, "reference": case["text"],
                      "hypothesis": hyp, "wer": round(w, 4), "errors": d, "words": n})
        print("%-24s wer %.3f  %s" % (os.path.basename(path), w, hyp[:60]))
        sys.stdout.flush()
    corpus = tot_d / max(1, tot_n)
    ok = args.floor is None or corpus <= args.floor
    result = {"n": len(cases), "corpus_wer": corpus, "errors": tot_d, "ref_words": tot_n,
              "floor": args.floor, "verdict": "PASS" if ok else "FAIL",
              "manifest_sha256": msha, "cases": cases}
    json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=1)
    json.dump({"url": args.url, "model": args.model, "language": args.language,
               "manifest": args.manifest, "manifest_sha256": msha},
              open(os.path.join(args.out, "arm.json"), "w"), indent=1)
    print("STT gate %s: corpus WER %.4f (%d errors / %d ref words)%s"
          % (result["verdict"], corpus, tot_d, tot_n,
             " floor %.4f" % args.floor if args.floor is not None else ""))
    return 0 if ok else 1


def compare(args):
    a = json.load(open(os.path.join(args.cpu, "result.json")))
    b = json.load(open(os.path.join(args.card, "result.json")))
    if a.get("manifest_sha256") != b.get("manifest_sha256"):
        print("compare: different clip sets (%s vs %s)"
              % (a.get("manifest_sha256"), b.get("manifest_sha256")), file=sys.stderr)
        return 2
    ma = json.load(open(os.path.join(args.cpu, "arm.json")))
    mb = json.load(open(os.path.join(args.card, "arm.json")))
    if ma.get("model") != mb.get("model"):
        print("compare: different models (%s vs %s)" % (ma.get("model"), mb.get("model")),
              file=sys.stderr)
        return 2
    ok = b["corpus_wer"] <= a["corpus_wer"] + args.tol
    print("cpu : %.4f\ncard: %.4f (%+.4f, allowance +%.4f) %s"
          % (a["corpus_wer"], b["corpus_wer"], b["corpus_wer"] - a["corpus_wer"], args.tol,
             "PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--language", default="de")
    r.add_argument("--return", dest="response_format", default="json")
    r.add_argument("--floor", type=float, default=None)
    r.add_argument("--timeout", type=float, default=600)
    r.set_defaults(fn=run)
    c = sub.add_parser("compare")
    c.add_argument("--cpu", required=True)
    c.add_argument("--card", required=True)
    c.add_argument("--tol", type=float, default=TOL_DEFAULT)
    c.set_defaults(fn=compare)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
