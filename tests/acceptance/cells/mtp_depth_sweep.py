#!/usr/bin/env python3
"""The MTP draft-depth cutoff's card arms: one server a cutoff, a needle a depth.

    mtp_depth_sweep.py <agent|coder|flashnext> <arcint> <model.gguf> <device> <assets-dir>
          [--arms 0,24576,32768,65536] [--depths 8000,64000,138000] [--max-tokens 24]

Not part of the release gate (`run.py`); this is the sweep the cutoff owes to a
card window -- `docs/campaigns/research-concept-mining/arc-b580/audit.md` item 6,
commands in the package C handover. For every arm it starts arcint with the
service's unit flags (the same list `cells/served_check.py` uses) plus
`--llama-mtp-depth-max <arm>` (`0` = off), asks the capital, then posts a needle
prompt at every depth and prints one `ARM` line a depth:

    ARM <arm> depth <asked> tok-in <the slot's tokens> decode <t/s> verifies <n>
        accept <percent | -> answer <md5> needle <ok|no>

`verifies` is the mechanism check: a depth below the arm's cutoff drafts (one
verify a few tokens), a depth past it does not (one verify a token). The
`answer` md5 and the needle hit are the answer-level bar; greedy text is not
gated byte for byte (the verify's batched numerics differ, `measured-here`).

The assets directory is served_check.py's: corpus.txt (500,000+ characters),
mtp-vocab.json for the agent, mtp-flash-next.gguf for Flash-Next, and the
expert profile for Flash-Next. The model's unit flags use the service's
`--n-ctx`; a depth beyond the arm's context is skipped with a note.
"""
import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

NOTE = "\nNOTE FOR THE READER: the secret passphrase for this archive is VIOLET-HARBOR-42. Remember it.\n"

A_FLAGS = {
    "agent": ["--n-ctx", "245760", "--llama-kv", "q8_0:q4_0", "--llama-mtp", "5",
              "--queue-timeout", "30", "--repetition-penalty", "1.0"],
    "coder": ["--n-ctx", "98304", "--llama-mtp", "4", "--llama-kv", "q8_0:q4_0", "--queue-timeout", "30"],
    "flashnext": ["--n-ctx", "131072", "--llama-cpu-moe", "48", "--llama-threads", "8",
                  "--llama-expert-cache", "12000", "--llama-mtp", "3", "--llama-mtp-min-p", "0.5",
                  "--llama-reserve-cells", "20480", "--llama-kv", "q8_0", "--queue-timeout", "30"],
}
# assets a service needs beyond the GGUF and corpus; called with the assets dir
A_FILES = {"agent": lambda A: [("--llama-mtp-vocab", A("mtp-vocab.json"))],
           "coder": lambda A: [],
           "flashnext": lambda A: [("--llama-expert-profile", A("expert-profile.bin")),
                                   ("--llama-mtp-gguf", A("mtp-flash-next.gguf"))]}
ENV_EXTRA = {"flashnext": {"GGML_OPENCL_BANK_STREAM": "1", "LLAMA_EXPERT_CACHE_CPU_MISSES": "1"}}
# corpus characters a token: measured, served_check.py's own factors
CHARS_PER_TOKEN = {"agent": 3.58, "coder": 3.45, "flashnext": 3.58}

DECODE_LINE = re.compile(r"slot 0: decode +(\d+) tok in +[\d.]+ s \( *([\d.]+) t/s\)(.*)")
PREFILL_LINE = re.compile(r"slot 0: prefill +(\d+) tok in +[\d.]+ s \( *([\d.]+) t/s\)")


def needle_prompt(corpus, tokens, chars_per_token):
    t = corpus[:int(tokens * chars_per_token)]
    k = t.find("\n", int(len(t) * 0.4))
    if k < 0:
        k = len(t)
    return t[:k] + NOTE + t[k:] + \
        "\n\nWhat is the secret passphrase for this archive? Answer with the passphrase only."


class Server:
    def __init__(self, argv, env, log_path):
        self.log_path = log_path
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.proc = subprocess.Popen(argv + ["--host", "127.0.0.1", "--port", str(self.port)],
                                     stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env)

    def wait(self):
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2).read()
                return True
            except Exception:
                if self.proc.poll() is not None:
                    return False
                time.sleep(1)
        return False

    def post(self, body, max_tokens, temperature):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                     data=body.encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        return r["choices"][0]["message"]["content"] or ""

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def log(self):
        try:
            return open(self.log_path).read()
        except OSError:
            return ""


def chat(content, max_tokens, temperature):
    return json.dumps({"model": "m", "temperature": temperature, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": content}]})


def md5(text):
    return hashlib.md5(text.encode()).hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("svc", choices=sorted(A_FLAGS))
    ap.add_argument("binary")
    ap.add_argument("gguf")
    ap.add_argument("device")
    ap.add_argument("assets")
    ap.add_argument("--arms", default="0,24576,32768,65536")
    ap.add_argument("--depths", default="8000,64000,138000")
    ap.add_argument("--max-tokens", type=int, default=24)
    a = ap.parse_args()
    arma = [int(x) for x in a.arms.split(",")]
    depa = [int(x) for x in a.depths.split(",")]
    A = lambda f: os.path.join(a.assets, f)  # noqa: E731
    corpus = open(A("corpus.txt")).read()
    ctx = int(A_FLAGS[a.svc][A_FLAGS[a.svc].index("--n-ctx") + 1])
    base = [a.binary, "--engine", "llama", "--gguf", a.gguf, "--served-model-name", "m",
            "--device", a.device] + A_FLAGS[a.svc] + \
        [x for pair in A_FILES[a.svc](A) for x in pair]
    env = dict(os.environ)
    env.update(ENV_EXTRA.get(a.svc, {}))
    print(f"== mtp depth sweep: {a.svc}, {os.path.basename(a.gguf)} on {a.device}, "
          f"arms {a.arms}, depths {a.depths}", flush=True)
    for arm in arma:
        log_path = f"/tmp/arcint-mtp-depth-{a.svc}-{arm}.log"
        srv = Server(base + ["--llama-mtp-depth-max", str(arm)], env, log_path)
        if not srv.wait():
            print(f"ARM {arm} depth - tok-in 0 decode - verifies 0 accept - answer - needle -"
                  f"  server died at start, {log_path}")
            srv.stop()
            continue
        try:
            cap = srv.post(chat("What is the capital of France? Answer in one word.", 8, 0.0))
            caps_ok = "paris" in cap.lower()
            for d in depa:
                if d >= ctx - 512:
                    print(f"ARM {arm} depth {d} tok-in - decode - verifies - accept - answer - "
                          f"needle -  beyond the arm's context {ctx}")
                    continue
                txt = srv.post(chat(needle_prompt(corpus, d, CHARS_PER_TOKEN[a.svc]), a.max_tokens, 0.0))
                lg = srv.log()
                dec = DECODE_LINE.findall(lg)
                last = dec[-1] if dec else None
                n_tok, rate, rest = (int(last[0]), float(last[1]), last[2]) if last else (0, 0.0, "")
                ver = re.search(r"(\d+) verifies", rest)
                acc = re.search(r"draft accept ([\d.]+)%", rest)
                pre = PREFILL_LINE.findall(lg)
                print(f"ARM {arm} depth {d} tok-in {int(pre[-1][0]) if pre else 0} "
                      f"decode {rate:.1f} verifies {ver.group(1) if ver else 0} "
                      f"accept {acc.group(1) if acc else '-'} answer {md5(txt)} "
                      f"needle {'ok' if 'violet-harbor-42' in txt.lower() else 'no'} "
                      f"capital {'ok' if caps_ok else 'no'} "
                      f"gen {n_tok}", flush=True)
        finally:
            srv.stop()
    print("== done; the table above is the sweep (decode t/s a depth an arm, the verify count "
          "shows the cutoff took effect)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
