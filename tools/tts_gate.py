#!/usr/bin/env python3
"""The TTS gate: intelligibility of synthesized German, by round-trip
(`docs/milestone-1.2.0.md` C4).

There is no reference waveform: TTS is stochastic, and the honest answer-level
gate for "does it say the words" is the STT round trip. Each fixed sentence is
synthesized through `/v1/audio/speech` (OpenAI shape, `voice` a name from the
server's enrolled list, D5), saved as WAV, transcribed by the STT unit, and
scored word-error-rate against the written sentence. The corpus WER over the
set is the gate. Voice similarity (C5) needs a named speaker-embedding model
on the record; when `--embed-cmd` is given it is computed, otherwise it is
reported as not measured (never omitted silently).

Identical inputs are the premise: `compare` refuses two arms whose sentence
list differs, as the image and VLM gates do. The sampling (temperature, seed)
is in `arm.json`; a reproducible arm fixes them and `--seed` is passed on.

Exit status: 0 inside the floor; 1 outside; 2 a usage or decode error.

    tts_gate.py run --tts-url URL --model NAME --voice NAME --stt-url URL \
                    --stt-model NAME --manifest SENTENCES.json --out DIR \
                    [--floor 0.20 --seed 42 --lang de --embed-cmd CMD]
    tts_gate.py compare --ref DIR --cand DIR
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request

FLOOR_DEFAULT = 0.20   # corpus WER of the round trip (proposal, D8)


def norm_words(text):
    text = text.lower()
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    return [w for w in re.sub(r"[^\w\s]", " ", text).split() if w]


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if rw == hw else 1)))
        prev = cur
    return prev[-1] / max(1, len(r)), prev[-1], len(r)


def synthesize(tts_url, model, voice, text, seed, timeout):
    body = {"model": model, "input": text, "voice": voice, "response_format": "wav",
            "temperature": 0}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(tts_url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()   # the WAV bytes


def transcribe(stt_url, stt_model, wav_bytes, language, timeout):
    boundary = "----arcintttsgate"
    parts = []
    def field(n, v):
        parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                      % (boundary, n, v)).encode())
    field("model", stt_model)
    field("language", language)
    field("response_format", "json")
    parts.append(("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
                  "Content-Type: audio/wav\r\n\r\n" % boundary).encode())
    parts.append(wav_bytes)
    parts.append(b"\r\n")   # CRLF before the closing delimiter, per RFC 2046
    parts.append(("--%s--\r\n" % boundary).encode())
    req = urllib.request.Request(stt_url, data=b"".join(parts),
                                 headers={"Content-Type": "multipart/form-data; boundary=%s" % boundary})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()).get("text", "")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_texts(args):
    manifest = json.load(open(args.manifest))
    texts = manifest if isinstance(manifest, list) else manifest.get("sentences", [])
    if not texts or not all(isinstance(t, str) for t in texts):
        print("manifest must be a non-empty list of sentences", file=sys.stderr)
        raise SystemExit(2)
    return texts, sha256_file(args.manifest)


def speech(args):
    """Phase A: synthesize every sentence to WAV (TTS up; no STT needed)."""
    texts, msha = load_texts(args)
    os.makedirs(args.out, exist_ok=True)
    cases = []
    for i, text in enumerate(texts):
        wav = synthesize(args.tts_url, args.model, args.voice, text, args.seed, args.timeout)
        wav_path = os.path.join(args.out, "sent_%03d.wav" % i)
        open(wav_path, "wb").write(wav)
        cases.append({"text": text, "wav": wav_path, "wav_bytes": len(wav)})
        print("wav %5d bytes  %s" % (len(wav), text[:50]))
        sys.stdout.flush()
    json.dump({"n": len(cases), "cases": cases, "manifest_sha256": msha,
               "manifest": args.manifest, "voice": args.voice, "seed": args.seed,
               "tts_url": args.tts_url, "model": args.model, "lang": args.lang,
               "floor": args.floor},
              open(os.path.join(args.out, "speech.json"), "w"), indent=1)
    print("TTS speech phase: %d sentences -> %s" % (len(cases), args.out))
    return 0


def transcribe_phase(args):
    """Phase B: transcribe the saved WAVs (STT up; no TTS needed) and score.

    The two phases exist because one card holds one non-service process at a
    time (docs/milestone-1.2.0.md §9.1): the TTS unit and the STT unit never
    run beside each other on the same card, so a leg synthesizes first, stops
    the TTS server, then transcribes."""
    sp = json.load(open(os.path.join(args.speech, "speech.json")))
    cases, tot_d, tot_n = [], 0, 0
    for c in sp["cases"]:
        wav = open(c["wav"], "rb").read()
        hyp = transcribe(args.stt_url, args.stt_model, wav, sp["lang"], args.timeout)
        w, d, n = wer(c["text"], hyp)
        tot_d += d
        tot_n += n
        cases.append({"text": c["text"], "wav": c["wav"], "wav_bytes": c["wav_bytes"],
                      "hypothesis": hyp, "wer": round(w, 4), "errors": d, "words": n})
        print("wer %.3f  ref %-40s -> %s" % (w, c["text"][:40], hyp[:50]))
        sys.stdout.flush()
    corpus = tot_d / max(1, tot_n)
    verdict = corpus <= sp["floor"]
    result = {"n": len(cases), "corpus_wer": corpus, "errors": tot_d, "ref_words": tot_n,
              "floor": sp["floor"], "verdict": "PASS" if verdict else "FAIL",
              "manifest_sha256": sp["manifest_sha256"], "voice": sp["voice"],
              "seed": sp["seed"], "cases": cases}
    if args.embed_cmd:
        try:
            sim = float(subprocess.check_output(args.embed_cmd.split(),
                                                input=json.dumps(cases).encode()).decode().strip())
            result["similarity"] = sim
        except Exception as e:
            print("embed-cmd failed: %s" % e, file=sys.stderr)
    os.makedirs(args.out, exist_ok=True)
    json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=1)
    json.dump({"tts_url": sp["tts_url"], "model": sp["model"], "voice": sp["voice"],
               "stt_url": args.stt_url, "stt_model": args.stt_model, "lang": sp["lang"],
               "seed": sp["seed"], "floor": sp["floor"], "manifest": sp["manifest"],
               "manifest_sha256": sp["manifest_sha256"]},
              open(os.path.join(args.out, "arm.json"), "w"), indent=1)
    print("TTS gate %s: round-trip corpus WER %.4f (floor %.4f)%s"
          % (result["verdict"], corpus, sp["floor"],
             "; similarity not measured (no --embed-cmd)" if "similarity" not in result else
             " similarity %.3f" % result["similarity"]))
    return 0 if verdict else 1


def run(args):
    """Both phases at once (both services up; convenient, not for a single-card
    leg)."""
    args.speech = args.out
    rc = speech(args)
    if rc:
        return rc
    return transcribe_phase(args)


def compare(args):
    a = json.load(open(os.path.join(args.ref, "result.json")))
    b = json.load(open(os.path.join(args.cand, "result.json")))
    if a.get("manifest_sha256") != b.get("manifest_sha256"):
        print("compare: different sentence lists", file=sys.stderr)
        return 2
    ok = b["corpus_wer"] <= a["corpus_wer"]
    print("reference: WER %.4f\ncandidate: WER %.4f %s"
          % (a["corpus_wer"], b["corpus_wer"], "PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tts-url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--voice", required=True)
    r.add_argument("--stt-url", required=True)
    r.add_argument("--stt-model", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--floor", type=float, default=FLOOR_DEFAULT)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--lang", default="de")
    r.add_argument("--embed-cmd", default="")
    r.add_argument("--timeout", type=float, default=600)
    r.set_defaults(fn=run)
    s = sub.add_parser("speech")
    s.add_argument("--tts-url", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--voice", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--floor", type=float, default=FLOOR_DEFAULT)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--lang", default="de")
    s.add_argument("--timeout", type=float, default=600)
    s.set_defaults(fn=speech)
    t = sub.add_parser("transcribe")
    t.add_argument("--speech", required=True, help="the speech phase's out dir (speech.json)")
    t.add_argument("--stt-url", required=True)
    t.add_argument("--stt-model", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--embed-cmd", default="")
    t.add_argument("--timeout", type=float, default=600)
    t.set_defaults(fn=transcribe_phase)
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--cand", required=True)
    c.set_defaults(fn=compare)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
