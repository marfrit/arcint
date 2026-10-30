#!/usr/bin/env python3
"""One prefill arm of any unit: a fresh process with the given flags, one corpus prompt, the prefill line and peaks.

    prefill_arm.py <arcint> <model.gguf> <device> <corpus.txt> <tag> <prompt-tokens> -- <arcint flags...>

Starts arcint with exactly the flags after `--` (the unit's own, plus the arm's change), asks the capital, then one
prompt of about <prompt-tokens> tokens cut from the corpus, and prints the slot's prefill line and the process's peak
VRAM and GTT from its DRM fdinfo (sampled every 0.5 s). For pricing a static trade (KV cells against a larger prefill
ubatch) before building a mechanism. Last line: `PREFILL <tag> <tokens> tok <t/s> t/s vram <MiB> gtt <MiB>`.
"""
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request

binary, gguf, device, corpus, tag, ntok = sys.argv[1:7]
flags = sys.argv[sys.argv.index("--") + 1:]
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
log_path = f"/tmp/prefill-arm-{tag}.log"
proc = subprocess.Popen([binary, "--engine", "llama", "--gguf", gguf, "--served-model-name", "m", "--device", device,
                         "--host", "127.0.0.1", "--port", str(port)] + flags,
                        stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
peak = [0, 0]


def sampler():
    while proc.poll() is None:
        v = g = 0
        try:
            for fd in os.listdir(f"/proc/{proc.pid}/fdinfo"):
                t = open(f"/proc/{proc.pid}/fdinfo/{fd}").read()
                a = re.search(r"drm-resident-vram0:\s+(\d+)", t); b = re.search(r"drm-resident-gtt:\s+(\d+)", t)
                v = max(v, int(a.group(1)) if a else 0); g = max(g, int(b.group(1)) if b else 0)
        except OSError:
            pass
        peak[0] = max(peak[0], v); peak[1] = max(peak[1], g)
        time.sleep(0.5)


def post(text, n):
    body = {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": n, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))["choices"][0]["message"]["content"] or ""


threading.Thread(target=sampler, daemon=True).start()
try:
    while not any("listening on" in l for l in open(log_path, errors="replace")):
        if proc.poll() is not None:
            sys.exit(f"arcint exited during load: {proc.returncode}\n" + open(log_path).read()[-1500:])
        time.sleep(2)
    cap = post("What is the capital of France? Answer in one word.", 8)
    post(open(corpus).read()[:int(int(ntok) * 3.5)] + "\n\nSummarize the last paragraph above in one sentence.", 24)
    time.sleep(0.5)
    pf = [l for l in open(log_path, errors="replace") if "prefill" in l and "tok in" in l][-1]
    m = re.search(r"prefill\s+(\d+) tok in\s+([\d.]+) s \(\s*([\d.]+) t/s\)", pf)
    print(f"  capital {cap.strip()[:12]!r}; {pf.strip()[:140]}", flush=True)
    print(f"PREFILL {tag} {m.group(1)} tok {m.group(3)} t/s vram {peak[0] // 1024} MiB gtt {peak[1] // 1024} MiB",
          flush=True)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
