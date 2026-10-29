#!/usr/bin/env python3
"""One arm of a served Flash-Next sweep: a fresh process with the unit's flags plus the arm's overrides.

    fn_sweep.py <arcint> <model.gguf> <assets-dir> <tag> [--env K=V]... [--set --flag VALUE]... [--port N]
                [--prompt-tokens N] [--no-answers]

Starts arcint with the Flash-Next unit's flags (contrib/systemd/, the 11,600 MiB slot budget, the CPU misses,
the bank stream), an `--env` or `--set` per override (a flag the unit has gets its value replaced, a new one
is appended), asks the capital, then Strata's request (the RFC 4180 Lua parser, 500 tokens) and five
300-token answers, greedy, without thinking. Per answer it prints the slot's decode line (tokens, t/s, draft
acceptance), the expert cache's share of routed experts on the card and its swaps, and the answer's md5 (a
byte tripwire between arms, not a gate). Last line: `SWEEP <tag> decode-mean <t/s> accept-mean <%> hits-mean
<%>` over the six answers.

The assets directory: expert-profile.bin (Strata's profile), mtp-flash-next.gguf (the MTP head),
mtp-vocab.json (the draft vocabulary), as for tests/acceptance/cells/served_check.py.
"""
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

args = sys.argv[1:]
binary, gguf, assets, tag = args[:4]
env_over, set_over, port, prompt_tokens, no_answers = {}, [], 0, 0, False
i = 4
while i < len(args):
    if args[i] == "--env":
        k, v = args[i + 1].split("=", 1); env_over[k] = v; i += 2
    elif args[i] == "--set":
        set_over.append((args[i + 1], args[i + 2])); i += 3
    elif args[i] == "--port":
        port = int(args[i + 1]); i += 2
    elif args[i] == "--prompt-tokens":
        prompt_tokens = int(args[i + 1]); i += 2
    elif args[i] == "--no-answers":
        no_answers = True; i += 1
    else:
        sys.exit(f"unknown argument {args[i]!r}")
A = lambda n: os.path.join(assets, n)

FLAGS = ["--n-ctx", "131072", "--llama-cpu-moe", "48", "--llama-threads", "8", "--llama-expert-cache", "11600",
         "--llama-kv", "q8_0", "--llama-expert-profile", A("expert-profile.bin"), "--llama-mtp", "3",
         "--llama-mtp-min-p", "0.5", "--llama-reserve-cells", "20480", "--llama-mtp-gguf", A("mtp-flash-next.gguf"),
         "--queue-timeout", "30", "--llama-mtp-vocab", A("mtp-vocab.json")]
for flag, value in set_over:
    if flag in FLAGS:
        FLAGS[FLAGS.index(flag) + 1] = value
    else:
        FLAGS += [flag, value]
ENV = dict(os.environ, GGML_OPENCL_BANK_STREAM="1", LLAMA_EXPERT_CACHE_CPU_MISSES="1")
ENV.update(env_over)

PROMPTS = [
    ("strata", 500, "Write a Lua function parse_csv(s) that parses CSV text according to RFC 4180 (quoted fields, "
                    "escaped quotes, CRLF and LF line endings, embedded newlines in quoted fields) and returns a table "
                    "of rows. Explain the design briefly, then give the code."),
    ("hash", 300, "Explain in about 250 words how a hash table handles collisions."),
    ("compiler", 300, "Describe in about 250 words how a compiler turns source code into machine code."),
    ("tcp", 300, "Explain in about 250 words how TCP congestion control works."),
    ("btree", 300, "Explain in about 250 words how a B-tree index speeds up database lookups."),
    ("gc", 300, "Describe in about 250 words how a generational garbage collector works."),
]

if not port:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
log_path = f"/tmp/fn-sweep-{tag}.log"
cmd = [binary, "--engine", "llama", "--gguf", gguf, "--served-model-name", "m", "--device", "GPU.0",
       "--host", "127.0.0.1", "--port", str(port)] + FLAGS
print(f"== {tag}: {' '.join(f'{k}={v}' for k, v in env_over.items())} {' '.join(f'{f} {v}' for f, v in set_over)}",
      flush=True)
t0 = time.time()
proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=ENV)


def post(text, n):
    body = {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": n, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))["choices"][0]["message"]["content"] or ""


def log_lines():
    return open(log_path, errors="replace").read().splitlines()


DEC = re.compile(r"decode\s+(\d+) tok in\s+([\d.]+) s \(\s*([\d.]+) t/s\).*?draft accept ([\d.]+)%")
EC = re.compile(r"expert cache: ([\d.]+) % of decode's routed experts on the card \((\d+) / (\d+)\), (\d+) swaps")
rc = 1
try:
    while True:
        if proc.poll() is not None:
            print("\n".join(log_lines()[-40:])); sys.exit(f"arcint exited with {proc.returncode} during load")
        if any("listening on" in l for l in log_lines()):
            break
        time.sleep(2)
    print(f"  load {time.time() - t0:.0f} s", flush=True)
    paris = post("What is the capital of France? Answer in one word.", 8)
    print(f"  capital: {paris.strip()[:20]!r} {'ok' if 'paris' in paris.lower() else 'WRONG'}", flush=True)
    if prompt_tokens:
        # a long prompt first: its prefill line, and the grouped GEMM's tile padding if the probe is on
        before = len(log_lines())
        post(open(A("corpus.txt")).read()[:int(prompt_tokens * 3.6)] +
             "\n\nSummarize the last paragraph above in one sentence.", 24)
        time.sleep(0.5)
        new = log_lines()[before:]
        pf = next((l for l in reversed(new) if "prefill" in l and "tok in" in l), "")
        print(f"  prompt    {pf.split('prefill', 1)[-1].strip()[:120]}", flush=True)
        PAD = re.compile(r"\[KQ_MM_ID_PAD\] (\S+) experts=\d+ used=(\d+) routed=(\d+) tile=(\d+) slots=(\d+)")
        agg = {}
        for l in new:
            m = PAD.search(l)
            if m:
                kind = re.sub(r"blk\.\d+\.", "", m.group(1))
                a = agg.setdefault(kind, [0, 0, 0, m.group(4)])
                a[0] += 1; a[1] += int(m.group(3)); a[2] += int(m.group(5))
        for kind, (n, r, s, tile) in sorted(agg.items()):
            print(f"  pad {kind:28s} calls {n:4d} tile {tile:>3} routed {r:9d} slots {s:9d} "
                  f"pad {100.0 * (s - r) / max(r, 1):5.1f} %", flush=True)
        if agg:
            r = sum(a[1] for a in agg.values()); s = sum(a[2] for a in agg.values())
            print(f"PAD {tag} routed {r} slots {s} pad {100.0 * (s - r) / max(r, 1):.1f} %", flush=True)
    dec, acc, hit = [], [], []
    for name, n, q in ([] if no_answers else PROMPTS):
        before = len(log_lines())
        t = post(q, n)
        time.sleep(0.5)
        new = log_lines()[before:]
        d = next((DEC.search(l) for l in reversed(new) if DEC.search(l)), None)
        e = next((EC.search(l) for l in reversed(new) if EC.search(l)), None)
        md5 = hashlib.md5(t.encode()).hexdigest()[:8]
        if d:
            dec.append(float(d.group(3))); acc.append(float(d.group(4)))
        if e:
            hit.append(float(e.group(1)))
        print(f"  {name:9s} {d.group(1) if d else '?':>4} tok {d.group(3) if d else '?':>6} t/s accept "
              f"{d.group(4) if d else '?':>5} %  cache {e.group(1) if e else '?':>5} % swaps {e.group(4) if e else '?':>4}"
              f"  md5 {md5}", flush=True)
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    print(f"SWEEP {tag} decode-mean {mean(dec):.2f} accept-mean {mean(acc):.1f} hits-mean {mean(hit):.1f}", flush=True)
    rc = 0 if "paris" in paris.lower() and (no_answers or len(dec) == len(PROMPTS)) else 1
finally:
    proc.terminate()
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
sys.exit(rc)
