#!/usr/bin/env python3
"""The equivalence suite (DESIGN.md §5, §3.4) on the libllama engine.

Every arm serves the same questions under a configuration change that must not
change the answers. The gate is the answer-level bar's first clause: the
answers stay right in every arm (a fact, a needle at depth). Byte equality of
the text between two arms is a tripwire (§3.4 point 5): reported, never a
failure. The arms:

  determinism    two fresh processes, the same request
  warm vs cold   a lane's reused prefix (the second request of a process,
                 which must report a cache hit) against a fresh process
  checkpoint     an edited last question resumed from the lane's recurrent
                 checkpoint (the log must say so) against a full re-prefill
  MTP on / off   when ARCINT_EXTRA_ARGS drafts (--llama-mtp N): the drafts
                 must be accepted at all, and the answers stay right without
  two lanes      named lanes, both busy at once, against one lane

Usage: run.sh <arcint> <model.gguf> [device]
ARCINT_EXTRA_ARGS is appended to every start (the unit's KV and MTP flags).
"""
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

BIN, GGUF = sys.argv[1], sys.argv[2]
DEV = sys.argv[3] if len(sys.argv) > 3 else "GPU.0"
EXTRA = shlex.split(os.environ.get("ARCINT_EXTRA_ARGS", ""))
WORK = tempfile.mkdtemp(prefix="arcint-eq-")
N_CTX = 16384

FACT = ("What is the capital of France? Answer in one word.", "paris")
WORD = "ORANGE-FALCON-77"


def filler(n_sentences, seed):
    # deterministic, varied text: a repeated sentence lets a model shortcut
    topics = ["the river", "the old mill", "a ledger", "the harbour", "a lantern", "the archive",
              "a bridge", "the orchard", "a telescope", "the market"]
    verbs = ["was rebuilt", "was measured", "was painted", "was moved", "was catalogued", "was repaired"]
    out = []
    x = seed
    for i in range(n_sentences):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(f"In year {1700 + x % 300}, {topics[x % len(topics)]} {verbs[(x >> 8) % len(verbs)]} "
                   f"by worker number {x % 997}.")
    return " ".join(out)


CONTEXT = (filler(90, 7) + f" The secret code word is {WORD}. Remember it. " + filler(140, 11))
Q1 = "What is the secret code word? Answer with the code word only."
Q2 = "Repeat the secret code word exactly, and say nothing else."

failed = 0
srv = None
log_path = None
port = None


def say(tag, msg):
    print(f"  {tag:<4} {msg}", flush=True)


def gate(ok, msg):
    global failed
    if ok:
        say("ok", msg)
    else:
        say("FAIL", msg)
        failed += 1


def tripwire(a, b, msg):
    say("ok" if a == b else "--", f"{msg}: {'byte-identical' if a == b else 'DIFFERS (tripwire, reported, not gated)'}")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def stop():
    global srv
    if srv is not None:
        srv.terminate()
        try:
            srv.wait(timeout=90)
        except subprocess.TimeoutExpired:
            srv.kill()
            srv.wait()
        srv = None


def start(name, args, n_ctx=True):
    global srv, log_path, port
    stop()
    port = free_port()
    log_path = os.path.join(WORK, name + ".log")
    cmd = [BIN, "--engine", "llama", "--gguf", GGUF, "--device", DEV, "--host", "127.0.0.1",
           "--port", str(port)] + (["--n-ctx", str(N_CTX)] if n_ctx else []) + EXTRA + args
    srv = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    for _ in range(3600):   # a cold kernel cache and a large GGUF take minutes
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return
        except Exception:
            if srv.poll() is not None:
                print(open(log_path).read()[-3000:])
                raise SystemExit(f"server died at start ({name})")
            time.sleep(1)
    raise SystemExit(f"server never became ready ({name})")


def ask(messages, model="", max_tokens=48):
    body = {"messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    if model:
        body["model"] = model
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    o = json.load(urllib.request.urlopen(r, timeout=3600))
    return o["choices"][0]["message"]["content"] or "", o.get("usage", {})


def right(text, want):
    return want.lower() in text.lower()


def log_text():
    return open(log_path).read()


def needle(q):
    return [{"role": "user", "content": CONTEXT + "\n\n" + q}]


def fact():
    return [{"role": "user", "content": FACT[0]}]


print(f"== arcint equivalence suite: {os.path.basename(GGUF)} on {DEV}"
      + (f" ({' '.join(EXTRA)})" if EXTRA else ""), flush=True)

# ------------------------------------------------ determinism, warm vs cold
start("a", [])
fa, _ = ask(fact())
gate(right(fa, FACT[1]), f"the fact is right ({fa.strip()[:40]!r})")
cold_a, _ = ask(needle(Q1))
gate(right(cold_a, WORD), f"the needle is right, cold ({cold_a.strip()[:40]!r})")
warm, _ = ask(needle(Q1))
gate(right(warm, WORD), f"the needle is right, warm ({warm.strip()[:40]!r})")
hits = re.findall(r"cache hit (\d+) tok", log_text())
gate(len(hits) > 0 and int(hits[-1]) > 1000, f"the warm request reused the lane's prefix (cache hit {hits[-1] if hits else 'none'} tok)")
tripwire(cold_a, warm, "warm against cold, one process")

# checkpoint: the last question edited; the recurrent state cannot be cut back,
# so the lane resumes from its newest checkpoint inside the shared prefix. The
# warm request above may already have resumed (its one re-run token is a cut on
# a hybrid model): gate on a resume by this request, not on any in the log
resumes_before = log_text().count("resumes from its checkpoint")
edited, _ = ask(needle(Q2))
gate(right(edited, WORD), f"the needle is right after an edited question ({edited.strip()[:40]!r})")
gate(log_text().count("resumes from its checkpoint") > resumes_before, "the edited request resumed from a checkpoint")

start("b", [])
cold_b, _ = ask(needle(Q1))
gate(right(cold_b, WORD), "the needle is right in a second fresh process")
tripwire(cold_a, cold_b, "two fresh processes")
tripwire(warm, cold_b, "warm against a fresh process")
full, _ = ask(needle(Q2))
gate(right(full, WORD), "the edited question is right after a full prefill")
tripwire(edited, full, "checkpoint resume against a full re-prefill")

# ------------------------------------------------------------- MTP on / off
mtp = 0
for i, a in enumerate(EXTRA):
    if a == "--llama-mtp" and i + 1 < len(EXTRA):
        mtp = int(EXTRA[i + 1])
if mtp > 0:
    copy = [{"role": "user", "content": "Repeat the following text back exactly, nothing else:\n\n" + filler(12, 3)}]
    _, u = ask(copy, max_tokens=200)
    det = u.get("completion_tokens_details", {})
    acc, rej = det.get("accepted_prediction_tokens", 0), det.get("rejected_prediction_tokens", 0)
    rate = acc / max(1, acc + rej)
    gate(rate > 0.10, f"MTP drafts are accepted ({acc} of {acc + rej}, {100 * rate:.0f} % on a copy prompt)")
    start("nomtp", ["--llama-mtp", "0"])
    plain, _ = ask(needle(Q1))
    gate(right(plain, WORD), "the needle is right without MTP")
    tripwire(cold_b, plain, "MTP against plain decoding")
else:
    print("  --   no --llama-mtp in ARCINT_EXTRA_ARGS; skipping the MTP arm")
    print("ACCEPTANCE-SKIP mtp-section no-llama-mtp-in-extra-args", flush=True)

# --------------------------------------------------------- one lane vs two
start("lanes", ["--served-model-name", "eq-a,eq-b", "--lane-ctx", f"{N_CTX},{N_CTX // 2}"], n_ctx=False)
res = {}


def lane(name, msgs):
    res[name] = ask(msgs, model=name)[0]


t = threading.Thread(target=lane, args=("eq-b", [{"role": "user", "content": filler(60, 5) + "\n\nIn which year was the river first rebuilt? Answer briefly."}]))
t.start()
lane("eq-a", needle(Q1))
t.join()
gate(right(res["eq-a"], WORD), "the needle is right on lane a while lane b decodes")
gate(len(res.get("eq-b", "")) > 0, "lane b answered")
tripwire(cold_b, res["eq-a"], "two lanes against one")

stop()
print()
if failed == 0:
    print("equivalence: all gated checks passed")
    sys.exit(0)
print(f"equivalence: {failed} gated check(s) failed")
sys.exit(1)
