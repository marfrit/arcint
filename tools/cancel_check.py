#!/usr/bin/env python3
"""Served check of request cancellation: a client that leaves during a long prefill, and a stop signal.

    cancel_check.py <arcint> <model.gguf> <device> <corpus.txt> <tag> [extra arcint flags...]

Starts arcint (the coder's flags unless extra flags follow), then:
  1. streams a ~40k-token prompt with a needle in its middle, closes the connection 8 s in (inside the prefill),
     and times how long the slot stays busy afterwards (GET /health, slots_free);
  2. sends the same prompt again to completion and checks the needle in the answer: the request after an
     aborted prefill must still be right on top of whatever the aborted one left in the cache;
  3. starts the prompt once more and sends SIGTERM 8 s in, timing the process's exit.
Prints `CANCEL <tag> slot-free-after-close <s> needle <ok|WRONG> exit-after-sigterm <s>`.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

binary, gguf, device, corpus, tag = sys.argv[1:6]
extra = sys.argv[6:] or ["--n-ctx", "98304", "--llama-mtp", "4", "--llama-kv", "q8_0:q4_0", "--queue-timeout", "30"]
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
log_path = f"/tmp/cancel-{tag}.log"


def listening():
    try:
        return sum("listening on" in l for l in open(log_path, errors="replace"))
    except OSError:
        return 0


def start():
    seen = listening()   # the log is appended to: wait for THIS process's line
    p = subprocess.Popen([binary, "--engine", "llama", "--gguf", gguf, "--served-model-name", "m", "--device", device,
                          "--host", "127.0.0.1", "--port", str(port)] + extra,
                         stdout=open(log_path, "a"), stderr=subprocess.STDOUT)
    while listening() <= seen:
        if p.poll() is not None: sys.exit(f"arcint exited during load: {p.returncode}")
        time.sleep(2)
    return p


def health():
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2))
    except Exception:
        return {}


text = open(corpus).read()[:int(40000 * 3.5)]
mid = len(text) // 2
prompt = (text[:mid] + "\n\nThe secret code for the archive is VIOLET-HARBOR-42.\n\n" + text[mid:] +
          "\n\nWhat is the secret code for the archive? Answer with the code only.")
body = json.dumps({"model": "m", "messages": [{"role": "user", "content": prompt}], "max_tokens": 24,
                   "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}).encode()


def raw_stream(close_after, payload=None):
    payload = payload or body
    c = socket.create_connection(("127.0.0.1", port))
    c.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
    c.settimeout(0.5)
    t0 = time.time()
    while time.time() - t0 < close_after:
        try:
            if not c.recv(65536): break
        except socket.timeout:
            pass
    c.close()


sigterm_only = os.environ.get("CANCEL_SIGTERM_ONLY") == "1"
free_after, needle, free_after_ns = float("nan"), "skipped", float("nan")
proc = start()
try:
    if not sigterm_only:
        raw_stream(8.0)
        t_close = time.time()
        while health().get("slots_free") != 1:
            if time.time() - t_close > 900: break
            time.sleep(0.2)
        free_after = time.time() - t_close
        print(f"  slot free {free_after:.1f} s after the client closed", flush=True)

        r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                   json.dumps({**json.loads(body), "stream": False}).encode(),
                                   {"Content-Type": "application/json"})
        a = json.load(urllib.request.urlopen(r, timeout=1800))["choices"][0]["message"]["content"] or ""
        needle = "ok" if "violet-harbor-42" in a.lower() else "WRONG"
        print(f"  follow-up needle {needle}: {a.strip()[:40]!r}", flush=True)

        # the same with stream:false and a second needle: the first half of the prompt is cached, the
        # second half (~20k tokens) is prefilled while the client leaves
        prompt2 = prompt.replace("VIOLET-HARBOR-42", "AMBER-COAST-17")
        body2 = json.dumps({**json.loads(body), "stream": False,
                            "messages": [{"role": "user", "content": prompt2}]}).encode()
        raw_stream(8.0, body2)
        t_close = time.time()
        while health().get("slots_free") != 1:
            if time.time() - t_close > 900: break
            time.sleep(0.2)
        free_after_ns = time.time() - t_close
        print(f"  non-streamed: slot free {free_after_ns:.1f} s after the client closed", flush=True)

        # a fresh process for the stop signal, so the prompt is not cached
        proc.terminate(); proc.wait(timeout=300)
        proc = start()
    import threading
    threading.Thread(target=raw_stream, args=(600.0,), daemon=True).start()
    time.sleep(8.0)
    t_sig = time.time()
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=900)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
    exit_after = time.time() - t_sig
    print(f"  exit {exit_after:.1f} s after SIGTERM (rc {proc.returncode})", flush=True)
    gone = sum("aborted: " in l for l in open(log_path, errors="replace"))
    print(f"CANCEL {tag} slot-free-after-close {free_after:.1f} non-streamed {free_after_ns:.1f} needle {needle} exit-after-sigterm {exit_after:.1f} "
          f"aborted-lines {gone}", flush=True)
finally:
    if proc.poll() is None:
        proc.kill()
