#!/usr/bin/env python3
"""The TTS gate: intelligibility of synthesized German, by round-trip
(`docs/milestone-1.2.0.md` C4).

There is no reference waveform: TTS is stochastic, and the honest answer-level
gate for "does it say the words" is the STT round trip. Each fixed sentence is
synthesized through `/v1/audio/speech` (OpenAI shape, `voice` a name from the
server's enrolled list, D5), saved as WAV, transcribed by the STT unit, and
scored word-error-rate against the written sentence. The C4 floor (operator, D8 2026-10-19) is the **corpus WER <= 0.10 AND no single sentence above 0.5**; the verdict prints the corpus WER, the worst sentence and the share of sentences with WER <= 0.10. Voice similarity (C5) needs a named speaker-embedding model
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
import time
import urllib.request
import wave

FLOOR_DEFAULT = 0.10        # corpus round-trip WER (operator floor, D8, 2026-10-19)
MAX_SENT_WER_DEFAULT = 0.5  # no single sentence above this (operator floor, D8)


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


def synthesize(tts_url, model, voice, text, sampling, timeout):
    # The reference tts.cpp runs the backbone's codec-token sampling stochastically
    # (common_params defaults: temperature 0.8, top_k 40, top_p 0.95, repetition
    # 1.0). A greedy temperature 0 made the codec stream derail past the first
    # frames and never reach end-of-speech (measured-here, 2026-10-18). Fixed seed
    # keeps the arm reproducible; the sampling is recorded in speech.json.
    body = {"model": model, "input": text, "voice": voice, "response_format": "wav"}
    body.update({k: v for k, v in sampling.items() if v is not None})
    req = urllib.request.Request(tts_url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        wav = r.read()
    return wav, time.time() - t0   # the WAV bytes and the process seconds


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
    sampling = {"temperature": args.temperature, "top_p": args.top_p,
                "top_k": args.top_k, "repetition_penalty": args.repetition_penalty,
                "seed": args.seed}
    cases, proc_tot, audio_tot = [], 0.0, 0.0
    for i, text in enumerate(texts):
        wav, dt = synthesize(args.tts_url, args.model, args.voice, text, sampling, args.timeout)
        wav_path = os.path.join(args.out, "sent_%03d.wav" % i)
        open(wav_path, "wb").write(wav)
        with wave.open(wav_path, "rb") as w:
            asec = w.getnframes() / float(w.getframerate())
        proc_tot += dt
        audio_tot += asec
        cases.append({"text": text, "wav": wav_path, "wav_bytes": len(wav),
                      "process_s": round(dt, 3), "audio_s": round(asec, 3)})
        print("timed %8.3f s process  %7.3f s audio  %5d bytes  %s"
              % (dt, asec, len(wav), text[:46]))
        sys.stdout.flush()
    json.dump({"n": len(cases), "cases": cases, "manifest_sha256": msha,
               "manifest": args.manifest, "voice": args.voice, "seed": args.seed,
               "sampling": {k: v for k, v in sampling.items() if v is not None},
               "tts_url": args.tts_url, "model": args.model, "lang": args.lang,
               "floor": args.floor, "max_sentence_wer": args.max_sentence_wer,
               "process_s_total": round(proc_tot, 3), "audio_s_total": round(audio_tot, 3),
               "rtf": round(audio_tot / proc_tot, 4) if proc_tot else None},
              open(os.path.join(args.out, "speech.json"), "w"), indent=1)
    print("TTS speech phase: %d sentences -> %s (audio %.3f s / process %.3f s, rtf %.4f)"
          % (len(cases), args.out, audio_tot, proc_tot, audio_tot / max(proc_tot, 1e-9)))
    return 0


def transcribe_phase(args):
    """Phase B: transcribe the saved WAVs (STT up; no TTS needed) and score.

    The two phases exist because one card holds one non-service process at a
    time (docs/milestone-1.2.0.md §9.1): the TTS unit and the STT unit never
    run beside each other on the same card, so a leg synthesizes first, stops
    the TTS server, then transcribes."""
    sp = json.load(open(os.path.join(args.speech, "speech.json")))
    max_sent = sp.get("max_sentence_wer", MAX_SENT_WER_DEFAULT)
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
    worst = max((c["wer"] for c in cases), default=0.0)
    le010 = sum(1 for c in cases if c["wer"] <= 0.10) / max(1, len(cases))
    verdict = corpus <= sp["floor"] and worst <= max_sent
    result = {"n": len(cases), "corpus_wer": corpus, "errors": tot_d, "ref_words": tot_n,
              "floor": sp["floor"], "max_sentence_wer": max_sent,
              "worst_sentence_wer": worst, "share_le_010": le010,
              "verdict": "PASS" if verdict else "FAIL",
              "manifest_sha256": sp["manifest_sha256"], "voice": sp["voice"],
              "seed": sp["seed"], "rtf": sp.get("rtf"),
              "process_s_total": sp.get("process_s_total"),
              "audio_s_total": sp.get("audio_s_total"), "cases": cases}
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
               "sampling": sp.get("sampling"),
               "stt_url": args.stt_url, "stt_model": args.stt_model, "lang": sp["lang"],
               "seed": sp["seed"], "floor": sp["floor"], "manifest": sp["manifest"],
               "manifest_sha256": sp["manifest_sha256"]},
              open(os.path.join(args.out, "arm.json"), "w"), indent=1)
    print("TTS gate %s: round-trip corpus WER %.4f (floor %.4f), worst sentence %.3f "
          "(bar %.3f), sentences WER<=0.10: %.1f%%, rtf %.3f%s"
          % (result["verdict"], corpus, sp["floor"], worst, max_sent, 100 * le010,
             sp.get("rtf") or 0.0,
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
    ok = (b["corpus_wer"] <= a["corpus_wer"]
          and b.get("worst_sentence_wer", 0.0) <= a.get("worst_sentence_wer", 1.0))
    print("reference: WER %.4f worst %.3f\ncandidate: WER %.4f worst %.3f %s"
          % (a["corpus_wer"], a.get("worst_sentence_wer", 0.0),
             b["corpus_wer"], b.get("worst_sentence_wer", 0.0),
             "PASS" if ok else "FAIL"))
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
    r.add_argument("--max-sentence-wer", type=float, default=MAX_SENT_WER_DEFAULT)
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
    s.add_argument("--max-sentence-wer", type=float, default=MAX_SENT_WER_DEFAULT)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--temperature", type=float, default=0.8)   # tts.cpp/common_params
    s.add_argument("--top-p", type=float, default=0.95)
    s.add_argument("--top-k", type=int, default=40)
    s.add_argument("--repetition-penalty", type=float, default=1.0)
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
