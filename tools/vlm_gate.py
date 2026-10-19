#!/usr/bin/env python3
"""The VLM gate: answer quality on a fixed set of images with checkable content
(`docs/milestone-1.2.0.md` C1/C2).

The answer-level bar for a vision model is the answer: each image has a
question whose truth can be checked offline (an object, a count, a colour, a
short lettering), and the gate is the set's pass rate. There are no logits and
no denoising, so the image gate's step curve does not apply. A candidate is
accepted only against the same manifest (the images and questions are the arm;
`compare` refuses a changed case list, as `imggen_gate.py` refuses changed
image-defining flags).

Exit status: 0 at or above the pass-rate floor; 1 below; 2 a usage or decode
error.

    vlm_gate.py run --url URL --model NAME --manifest CASES.json --out DIR \
                    [--floor 1.0 --max-tokens 64 --timeout 600]
    vlm_gate.py compare --ref DIR --cand DIR

`run` writes `result.json` (per case: the image's sha256, the question, the
answer, matched/not, the seconds and tokens) and `arm.json` (the url, the
model, the manifest's sha256, the floor). A manifest case is:

    {"image": "path/to.png",        # a local file (read and sent as a data URI)
     "sha256": "...",               # optional: checked when present
     "question": "What animal ...?",
     "expect": "cat",               # a whole word or phrase, case-insensitive
     "kind": "object"}              # free label, for the report

The gate voice/format is the OpenAI chat `image_url` content part, which is
what arcint's `/v1/chat/completions` takes (docs/milestone-1.2.0.md D2). Only
the standard library is used.

`expect` may be one word (or phrase) or a list of accepted ones (a count
answer can be `["one", "1"]`). It matches as a whole word, so "cat" does not
pass "catalog", "one" does not pass "none" and "1" does not pass "10"; a case
without an `expect` is refused, never counted as a pass; `image` resolves against the manifest's own
directory, so the committed manifest carries only filenames and the images
stay operator-local beside it (1.2.0 charter §4's rule).
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import re
import time
import urllib.request

DEFAULT_FLOOR = 1.0     # every answer right: the operator's C1 floor (2026-10-19)


def matches(answer, alts):
    """Whether any accepted word or phrase is in the answer as a whole word."""
    return any(re.search(r"(?<!\w)" + re.escape(a.lower()) + r"(?!\w)", answer.lower())
               for a in alts)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _data_uri(path):
    raw = open(path, "rb").read()
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp",
            "bmp": "bmp", "gif": "gif", "tga": "tga"}.get(ext, "png")
    return "data:image/%s;base64,%s" % (mime, base64.b64encode(raw).decode())


def _ask(url, model, question, uri, max_tokens, timeout):
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": uri}}]}],
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    dt = time.time() - t0
    msg = out.get("choices", [{}])[0].get("message", {})
    text = msg.get("content")
    if text is None:
        text = json.dumps(msg, ensure_ascii=False)
    usage = out.get("usage", {})
    return text, dt, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def run(args):
    manifest = json.load(open(args.manifest))
    if not isinstance(manifest, list) or not manifest:
        print("manifest must be a non-empty list of cases", file=sys.stderr)
        return 2
    manifest_sha = _sha256_file(args.manifest)
    os.makedirs(args.out, exist_ok=True)
    cases = []
    n_pass = 0
    for case in manifest:
        path = case["image"]
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(args.manifest)), path)
        if not os.path.exists(path):
            print("case image missing: %s" % path, file=sys.stderr)
            return 2
        exp = case.get("expect", "")
        if not [e for e in ([exp] if isinstance(exp, str) else exp) if e]:
            print("case without an expect: %s" % case.get("question", path), file=sys.stderr)
            return 2
        sha = _sha256_file(path)
        if "sha256" in case and case["sha256"] != sha:
            print("case image sha256 mismatch: %s" % path, file=sys.stderr)
            return 2
        uri = _data_uri(path)
        answer, dt, ptoks, ctoks = _ask(args.url, args.model, case["question"],
                                        uri, args.max_tokens, args.timeout)
        expect = case.get("expect", "")
        alts = [expect] if isinstance(expect, str) else list(expect)
        alts = [a for a in alts if a]
        hit = matches(answer, alts)
        n_pass += hit
        cases.append({"image": path, "sha256": sha, "kind": case.get("kind", ""),
                      "question": case["question"], "expect": expect,
                      "answer": answer, "matched": hit,
                      "seconds": round(dt, 3), "prompt_tokens": ptoks,
                      "completion_tokens": ctoks})
        print("%-12s %-38s -> %-30s %s" % (case.get("kind", ""), case["question"][:38],
                                           answer[:30], "PASS" if hit else "FAIL"))
        sys.stdout.flush()
    rate = n_pass / len(cases)
    verdict = rate >= args.floor
    result = {"n": len(cases), "pass": n_pass, "pass_rate": rate,
              "floor": args.floor, "verdict": "PASS" if verdict else "FAIL",
              "manifest_sha256": manifest_sha, "cases": cases}
    json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=1)
    json.dump({"url": args.url, "model": args.model, "floor": args.floor,
               "max_tokens": args.max_tokens, "manifest": args.manifest,
               "manifest_sha256": manifest_sha},
              open(os.path.join(args.out, "arm.json"), "w"), indent=1)
    print("VLM gate %s: %d/%d = %.3f (floor %.3f)" % (result["verdict"], n_pass,
                                                      len(cases), rate, args.floor))
    return 0 if verdict else 1


def compare(args):
    a = json.load(open(os.path.join(args.ref, "result.json")))
    b = json.load(open(os.path.join(args.cand, "result.json")))
    if a.get("manifest_sha256") != b.get("manifest_sha256"):
        print("compare: the two arms ran different manifests (%s vs %s)"
              % (a.get("manifest_sha256"), b.get("manifest_sha256")), file=sys.stderr)
        return 2
    print("reference: %d/%d %.3f\ncandidate: %d/%d %.3f"
          % (a["pass"], a["n"], a["pass_rate"], b["pass"], b["n"], b["pass_rate"]))
    return 0 if b["pass"] >= a["pass"] else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    r.add_argument("--max-tokens", type=int, default=64)
    r.add_argument("--timeout", type=float, default=600)
    r.set_defaults(fn=run)
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--cand", required=True)
    c.set_defaults(fn=compare)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
