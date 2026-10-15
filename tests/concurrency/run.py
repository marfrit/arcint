#!/usr/bin/env python3
"""The lane gates (DESIGN.md §4.1, §4.2, §5) on the libllama engine. Needs a
real card: what is gated is what the card does with two sequences at once.

    run.py <arcint> <model.gguf> [device]

ARCINT_EXTRA_ARGS is appended to every start (the unit's KV and MTP flags).
The server runs two named lanes. The gates, each able to fail:

  no cross-lane bleed    two lanes asked at once, each about its own context
                         with its own code word, in both start orders: each
                         answers its own word. A lane that read the other's KV,
                         recurrent state or logits answers a plausible and
                         different thing. Byte equality with the solo run is a
                         tripwire (§3.4 point 5): with both lanes busy the
                         step combiner verifies them in one batch, which may
                         move a near-tie.
  both lanes used        the console shows lane 1 working
  routing                an unknown model name is a 404, not another lane
  cancellation           one client walking away mid-stream leaves the other
                         lane's answer right and gives its lane back
  a busy lane            a second request for a busy lane is a 503 naming the
                         lane (--queue-timeout 0), never another lane's slot
  warm under load        a lane's reused prefix while the other lane decodes:
                         the answer right and the cache hit reported
"""
import json
import os
import tempfile
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

FAILED = []


def ok(msg):
    print(f"  ok   {msg}", flush=True)


def fail(msg):
    print(f"  FAIL {msg}", flush=True)
    FAILED.append(msg)


def note(msg):
    print(f"  --   {msg}", flush=True)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


LANES = ("cc-a", "cc-b")


class Server:
    def __init__(self, binary, model, device, log_path):
        self.port = free_port()
        self.log_path = log_path
        self.log = open(log_path, "wb")
        extra = shlex.split(os.environ.get("ARCINT_EXTRA_ARGS", ""))
        self.proc = subprocess.Popen(
            [binary, "--engine", "llama", "--gguf", model, "--device", device, "--host", "127.0.0.1",
             "--port", str(self.port), "--served-model-name", ",".join(LANES), "--lane-ctx", "12288,12288",
             "--queue-timeout", "0", *extra],
            stdout=self.log, stderr=subprocess.STDOUT)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def wait_ready(self, seconds=3600):
        for _ in range(seconds):
            try:
                urllib.request.urlopen(self.url + "/health", timeout=2).read()
                return True
            except Exception:                                    # noqa: BLE001
                if self.proc.poll() is not None:
                    print("server died:")
                    print(self.text()[-3000:])
                    return False
                time.sleep(1)
        print("server never became ready")
        return False

    def text(self):
        self.log.flush()
        with open(self.log_path, "r", errors="replace") as f:
            return f.read()

    def health(self):
        return json.load(urllib.request.urlopen(self.url + "/health", timeout=10))

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def body(prompt, model, max_tokens, stream=False):
    return {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}}


def ask(url, prompt, model, max_tokens=32):
    """One non-streaming greedy answer, or ('HTTP', status, body) on refusal."""
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(body(prompt, model, max_tokens)).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            o = json.load(r)
            return o["choices"][0]["message"]["content"] or ""
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return ("HTTP", e.code, json.loads(raw))
        except ValueError:
            return ("HTTP", e.code, {"raw": raw})


def ask_async(url, prompt, model, out, key, max_tokens=32, delay=0.0):
    def run():
        if delay:
            time.sleep(delay)
        out[key] = ask(url, prompt, model, max_tokens)
    t = threading.Thread(target=run)
    t.start()
    return t


def context(seed, word):
    lines = []
    x = seed
    for _ in range(160):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        lines.append(f"Entry {x % 10007}: crate {x % 97} shipped to dock {x % 13} on day {x % 365}.")
    lines.insert(70, f"The secret code word is {word}. Remember it.")
    return " ".join(lines) + "\n\nWhat is the secret code word? Answer with the code word only."


WORD_A, WORD_B = "VIOLET-HARBOR-42", "ORANGE-FALCON-77"
PROMPT_A, PROMPT_B = context(3, WORD_A), context(5, WORD_B)


def right(text, word):
    return isinstance(text, str) and word.lower() in text.lower()


def check_no_cross_lane_bleed(srv):
    solo_a = ask(srv.url, PROMPT_A, LANES[0])
    solo_b = ask(srv.url, PROMPT_B, LANES[1])
    if not right(solo_a, WORD_A) or not right(solo_b, WORD_B):
        fail(f"solo baselines answer their own word (got {str(solo_a)[:40]!r} / {str(solo_b)[:40]!r})")
        return
    ok("solo baselines answer their own word")
    for order, first in (("a then b", 0), ("b then a", 1)):
        got = {}
        prompts = {0: (PROMPT_A, LANES[0], WORD_A, solo_a), 1: (PROMPT_B, LANES[1], WORD_B, solo_b)}
        p1, p2 = prompts[first], prompts[1 - first]
        t1 = ask_async(srv.url, p1[0], p1[1], got, p1[1], max_tokens=64)
        t2 = ask_async(srv.url, p2[0], p2[1], got, p2[1], max_tokens=64, delay=0.5)
        t1.join()
        t2.join()
        for prompt, lane, word, solo in (p1, p2):
            if right(got.get(lane), word):
                ok(f"concurrent ({order}): lane {lane} answers its own word")
            else:
                fail(f"concurrent ({order}): lane {lane} answers its own word (got {str(got.get(lane))[:60]!r})")
            if got.get(lane) == solo:
                ok(f"concurrent ({order}): lane {lane} byte-identical to its solo run")
            else:
                note(f"concurrent ({order}): lane {lane} DIFFERS from its solo run (tripwire, reported, not gated)")
    if re.search(r"slot 1: (prefill|decode)", srv.text()):
        ok("both lanes were used (the console shows slot 1 working)")
    else:
        fail("both lanes were used -- everything ran on slot 0, so the check proved nothing")


def check_routing(srv):
    r = ask(srv.url, "Say hello.", "no-such-lane", 8)
    if isinstance(r, tuple) and r[1] == 404:
        ok("an unknown model name is a 404")
    else:
        fail(f"an unknown model name is a 404 (got {r if isinstance(r, tuple) else 'an answer'})")


def check_cancel_leaves_the_other_alone(srv):
    got = {}
    keep = ask_async(srv.url, PROMPT_B, LANES[1], got, "kept", delay=0.5)
    req = urllib.request.Request(srv.url + "/v1/chat/completions",
                                 data=json.dumps(body(PROMPT_A + " Then explain it at length.", LANES[0], 512,
                                                      stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=3600)
    seen = 0
    for raw in resp:
        if raw.startswith(b"data: "):
            seen += 1
        if seen > 4:
            break
    resp.close()                      # the client is gone
    keep.join()
    if right(got.get("kept"), WORD_B):
        ok("cancelling one lane leaves the other's answer right")
    else:
        fail(f"cancelling one lane leaves the other's answer right (got {str(got.get('kept'))[:60]!r})")
    h = {}
    for _ in range(120):
        h = srv.health()
        if h.get("slots_free") == h.get("slots_total"):
            break
        time.sleep(0.5)
    if h.get("slots_free") == h.get("slots_total"):
        ok(f"the cancelled lane came back ({h['slots_free']}/{h['slots_total']} free)")
    else:
        fail(f"the cancelled lane came back (only {h.get('slots_free')}/{h.get('slots_total')} free)")


def check_busy_lane_is_a_503(srv):
    got = {}
    # a long generation, so the lane stays busy while the second request arrives
    busy = ask_async(srv.url, "Write a story of at least 600 words about a lighthouse keeper.", LANES[0], got, "busy", 600)
    for _ in range(200):              # until /health shows lane a taken
        lanes = {l.get("name"): l for l in srv.health().get("lanes", [])}
        if lanes.get(LANES[0], {}).get("slots_free", 1) == 0:
            break
        time.sleep(0.1)
    second = ask(srv.url, "Say hello.", LANES[0], 8)
    other = ask(srv.url, "What is the capital of France? Answer in one word.", LANES[1], 8)
    busy.join()
    if isinstance(second, tuple) and second[1] == 503:
        ok("a second request for a busy lane is a 503")
        sl = second[2].get("slots", {})
        if sl.get("lane") == LANES[0] and sl.get("free") == 0 and sl.get("total") == 1:
            ok(f"the 503 carries the lane's numbers ({json.dumps(sl)[:120]})")
        else:
            fail(f"the 503 carries the lane's numbers (slots {json.dumps(sl)[:200]})")
    else:
        fail(f"a second request for a busy lane is a 503 (got {second if isinstance(second, tuple) else 'an answer'})")
    if right(other, "paris"):
        ok("the other lane serves meanwhile")
    else:
        fail(f"the other lane serves meanwhile (got {other!r:.60})")


def check_warm_under_load(srv):
    cold = ask(srv.url, PROMPT_A, LANES[0])
    got = {}
    noise = ask_async(srv.url, PROMPT_B, LANES[1], got, "noise", 256)
    time.sleep(0.5)                   # the other lane is busy
    warm = ask(srv.url, PROMPT_A, LANES[0])
    noise.join()
    if right(warm, WORD_A):
        ok("a lane's reused prefix answers right with the other lane busy")
    else:
        fail(f"a lane's reused prefix answers right with the other lane busy (got {str(warm)[:60]!r})")
    hits = re.findall(r"slot 0: prefill .*cache hit (\d+) tok", srv.text())
    if hits and int(hits[-1]) > 1000:
        ok(f"the lane reused its prefix (cache hit {hits[-1]} tok)")
    else:
        fail("the lane reused its prefix (no cache hit on slot 0, so the check proved nothing)")
    if warm == cold:
        ok("warm byte-identical to cold under load")
    else:
        note("warm DIFFERS from cold under load (tripwire, reported, not gated)")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    binary, model = sys.argv[1], sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "GPU.0"
    work = tempfile.mkdtemp(prefix="arcint-cc-")   # per run: two cells may run at once on two cards
    print(f"== arcint concurrency suite: {os.path.basename(model)} on {device}", flush=True)
    srv = Server(binary, model, device, f"{work}/main.log")
    try:
        if not srv.wait_ready():
            fail("the two-lane server started")
        else:
            check_no_cross_lane_bleed(srv)
            check_routing(srv)
            check_cancel_leaves_the_other_alone(srv)
            check_busy_lane_is_a_503(srv)
            check_warm_under_load(srv)
    finally:
        srv.close()
    print()
    if FAILED:
        print(f"concurrency: {len(FAILED)} check(s) failed")
        return 1
    print("concurrency: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
