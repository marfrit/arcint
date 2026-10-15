#!/usr/bin/env python3
"""The served check of one service (DESIGN.md §5, §7.1): the release floor.

    served_check.py <agent|coder|flashnext> <arcint> <model.gguf> <device> <assets-dir>

Starts arcint with the service's unit flags (contrib/systemd/), asks the
capital, a needle at depth and (coder, Flash-Next) 300-token answers, and
prints `ACCEPTANCE-METRIC <metric> <value> <unit>` for tests/acceptance/run.py
to compare against the cell's references: the last release's pre-tag numbers
minus their run-to-run spread (operator, 2026-10-15). The answers are gated
here; the rates are gated by run.py.

The assets directory holds what the repository cannot ship:
  corpus.txt              plain text, at least 500,000 characters (the agent's
                          and the coder's long prompts are cut from it)
  needle-20k.json         Flash-Next's 20,045-token needle request (an OpenAI
                          chat body whose answer contains ORANGE-FALCON-77)
  expert-profile.bin      Strata's decode profile (Flash-Next's cache fill)
  mtp-flash-next.gguf     Flash-Next's MTP-only GGUF (tools/flash_next_mtp_gguf.sh)
  mtp-vocab.json          the agent's draft vocabulary (--llama-mtp-vocab)
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

svc, binary, gguf, device, assets = sys.argv[1:6]
A = lambda f: os.path.join(assets, f)   # noqa: E731

FLAGS = {
    "agent": ["--n-ctx", "245760", "--llama-kv", "q8_0:q4_0", "--llama-mtp", "5",
              "--llama-mtp-vocab", A("mtp-vocab.json"), "--queue-timeout", "30", "--repetition-penalty", "1.0"],
    "coder": ["--n-ctx", "98304", "--llama-mtp", "4", "--llama-kv", "q8_0:q4_0", "--queue-timeout", "30"],
    "flashnext": ["--n-ctx", "131072", "--llama-cpu-moe", "48", "--llama-threads", "8", "--llama-expert-cache", "12000",
                  "--llama-expert-profile", A("expert-profile.bin"), "--llama-mtp", "3", "--llama-mtp-min-p", "0.5",
                  "--llama-reserve-cells", "20480", "--llama-mtp-gguf", A("mtp-flash-next.gguf"), "--llama-kv", "q8_0",
                  "--queue-timeout", "30"],
}[svc]
ENV = dict(os.environ)
if svc == "flashnext":
    ENV.update(GGML_OPENCL_BANK_STREAM="1", LLAMA_EXPERT_CACHE_CPU_MISSES="1")

failed = []
s = socket.socket()
s.bind(("127.0.0.1", 0))
port = s.getsockname()[1]
s.close()
log_path = f"/tmp/arcint-served-{svc}.log"
proc = subprocess.Popen([binary, "--engine", "llama", "--gguf", gguf, "--served-model-name", "m", "--device", device,
                         "--host", "127.0.0.1", "--port", str(port)] + FLAGS,
                        stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=ENV)


def resident():
    v = g = 0
    try:
        for fd in os.listdir(f"/proc/{proc.pid}/fdinfo"):
            try:
                t = open(f"/proc/{proc.pid}/fdinfo/{fd}").read()
            except OSError:
                continue
            if "drm-driver" not in t:
                continue
            v += sum(int(x) for x in re.findall(r"drm-resident-vram0:\s+(\d+)", t))
            g += sum(int(x) for x in re.findall(r"drm-resident-gtt:\s+(\d+)", t))
    except OSError:
        pass
    return v // 1024, g // 1024      # KiB -> MiB


peak = [0, 0]


def sampler():
    while proc.poll() is None:
        v, g = resident()
        peak[0], peak[1] = max(peak[0], v), max(peak[1], g)
        time.sleep(1)


def gate(ok, msg):
    print(f"  {'ok' if ok else 'FAIL'}   {msg}", flush=True)
    if not ok:
        failed.append(msg)


def post(content, n):
    b = {"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": n, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(b).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=7200))


def post_body(body):
    body = dict(body, model="m", stream=False, max_tokens=24, temperature=0)
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=7200))


def text(r):
    return r["choices"][0]["message"]["content"] or ""


def slot_lines(kind):
    return re.findall(rf"slot 0: {kind} +(\d+) tok in +([\d.]+) s \( *([\d.]+) t/s\)", open(log_path).read())


NOTE = "\nNOTE FOR THE READER: the secret passphrase for this archive is VIOLET-HARBOR-42. Remember it.\n"


def needle_prompt(tokens, chars_per_token):
    corpus = open(A("corpus.txt")).read()
    t = corpus[:int(tokens * chars_per_token)]
    k = t.find("\n", int(len(t) * 0.4))
    return t[:k] + NOTE + t[k:] + "\n\nWhat is the secret passphrase for this archive? Answer with the passphrase only."


def metric(name, value, unit="t/s"):
    print(f"ACCEPTANCE-METRIC {name} {value:.1f} {unit}", flush=True)


def prefill(name, line, lo, hi):
    # the depth actually read, from the slot line: the floor holds only at the reference's depth
    n = int(line[0])
    gate(lo <= n <= hi, f"{name} read {n} tokens (band {lo}-{hi})")
    metric(name, float(line[2]))


def decodes(names, min_tok=100):
    d = [x for x in slot_lines("decode") if int(x[0]) >= min_tok][-len(names):]
    gate(len(d) == len(names), f"{len(names)} decodes of at least {min_tok} tokens (got {len(d)})")
    for name, x in zip(names, d):
        metric(name, float(x[2]))


try:
    for _ in range(3600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            break
        except Exception:
            if proc.poll() is not None:
                print(open(log_path).read()[-3000:])
                raise SystemExit("the server died at start")
            time.sleep(1)
    threading.Thread(target=sampler, daemon=True).start()
    print(f"== served check: {svc}, {os.path.basename(gguf)} on {device}", flush=True)
    gate("paris" in text(post("What is the capital of France? Answer in one word.", 8)).lower(), "the capital")

    if svc == "agent":
        corpus = open(A("corpus.txt")).read()
        post(corpus[:int(58000 * 3.58)] + "\n\nSummarize the last paragraph above in one sentence.", 24)
        a = text(post(needle_prompt(133000, 3.58), 24))
        gate("violet-harbor-42" in a.lower(), f"the needle at 138k ({a.strip()[:40]!r})")
        p = slot_lines("prefill")
        prefill("prefill-55k", p[-2], 52000, 58000)
        prefill("prefill-138k", p[-1], 131000, 145000)
    elif svc == "coder":
        for q in ("Explain in about 250 words how a hash table handles collisions.",
                  "Describe in about 250 words how a compiler turns source code into machine code.",
                  "Explain in about 250 words how TCP congestion control works."):
            gate(len(text(post(q, 300))) > 200, f"an answer to {q[:40]!r}")
        decodes(("decode-a", "decode-b", "decode-c"))
        a = text(post(needle_prompt(44000, 3.45), 24))
        gate("violet-harbor-42" in a.lower(), f"the needle at 41k ({a.strip()[:40]!r})")
        prefill("prefill-41k", slot_lines("prefill")[-1], 38000, 44000)
    else:
        a = text(post_body(json.load(open(A("needle-20k.json")))))
        gate("orange-falcon-77" in a.lower(), f"the 20k needle ({a.strip()[:40]!r})")
        prefill("prefill-20k", slot_lines("prefill")[-1], 19500, 20500)
        gate(len(text(post("Explain in about 250 words how a hash table handles collisions.", 300))) > 200,
             "the long answer")
        decodes(("decode",))
finally:
    proc.terminate()
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
print(f"  peak VRAM {peak[0]} MiB, GTT {peak[1]} MiB", flush=True)
sys.exit(1 if failed else 0)
