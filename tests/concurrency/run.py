#!/usr/bin/env python3
"""The M6 gates (DESIGN.md §4.1, §5). Needs a real card, like the equivalence
suite, and for the same reason: what is being gated is what the GPU does with
two sequences on it at once.

    run.py <arcint> <model-dir> [device]

Every check here can fail, and each was run against a deliberately broken build
before being trusted:

  no cross-slot bleed    two prompts interleaved produce exactly the bytes each
                         produces alone, in both start orders. This is the gate
                         the whole milestone rests on: a lane that reads another
                         lane's KV pages, GDN rows or logits buffer answers a
                         plausible and different thing, which is the failure
                         mode this engine exists to refuse.
  cold/warm per lane     §3.4's invariant, held while the other lane is busy
  cancellation           one lane's client disappearing leaves the other's bytes
                         alone and gives the lane back
  admission              a third concurrent request is a 503 carrying the
                         reservation numbers, not an allocation failure
"""
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

FAILED = []


def ok(msg):
    print(f"  ok   {msg}")


def fail(msg):
    print(f"  FAIL {msg}")
    FAILED.append(msg)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Server:
    def __init__(self, binary, model, device, log_path, *extra):
        self.port = free_port()
        self.log_path = log_path
        self.log = open(log_path, "wb")
        # ARCINT_EXTRA_ARGS: the same channel tests/equivalence/run.sh honours.
        # The acceptance cells (tests/acceptance/cells.json) pass their served
        # configuration this way; the positional list ends at the device, and
        # anything after it is dropped, which a cell must never rely on.
        extra_env = os.environ.get("ARCINT_EXTRA_ARGS", "").split()
        self.proc = subprocess.Popen(
            [binary, "--model", model, "--device", device, "--host", "127.0.0.1",
             "--port", str(self.port), *extra_env, *extra],
            stdout=self.log, stderr=subprocess.STDOUT)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def wait_ready(self, seconds=2400):
        for _ in range(seconds):
            try:
                urllib.request.urlopen(self.url + "/health", timeout=2).read()
                return True
            except Exception:                                    # noqa: BLE001
                if self.proc.poll() is not None:
                    print("server died:")
                    print(self.tail(30))
                    return False
                time.sleep(1)
        print("server never became ready")
        return False

    def tail(self, n=20):
        self.log.flush()
        with open(self.log_path, "r", errors="replace") as f:
            return "".join(f.readlines()[-n:])

    def text(self):
        self.log.flush()
        with open(self.log_path, "r", errors="replace") as f:
            return f.read()

    def health(self):
        return json.load(urllib.request.urlopen(self.url + "/health", timeout=10))

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def body(prompt, max_tokens, stream=False):
    return {"messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}}


def ask(url, prompt, max_tokens=48):
    """One non-streaming greedy answer, or ('HTTP', status, body) on refusal."""
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(body(prompt, max_tokens)).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            o = json.load(r)
            return o["choices"][0]["message"]["content"] or ""
    except urllib.error.HTTPError as e:
        return ("HTTP", e.code, json.loads(e.read().decode("utf-8", "replace")))


def ask_async(url, prompt, out, key, max_tokens=48, delay=0.0):
    def run():
        if delay:
            time.sleep(delay)
        out[key] = ask(url, prompt, max_tokens)
    t = threading.Thread(target=run)
    t.start()
    return t


FILLER = ("A gated delta network folds every token it has seen into a fixed-size "
          "recurrent state. ")
PROMPT_A = ("Explain in one paragraph, without bullet points, why a recurrent "
            "linear-attention state cannot be truncated the way an attention KV "
            "cache can. " + FILLER * 40)
PROMPT_B = ("Write a short Python function that merges two sorted lists, then "
            "explain its complexity in one sentence. " + FILLER * 40)


def check_no_cross_slot_bleed(srv):
    """Two prompts at once must be byte-identical to each one alone."""
    solo_a = ask(srv.url, PROMPT_A)
    solo_b = ask(srv.url, PROMPT_B)
    if not isinstance(solo_a, str) or not isinstance(solo_b, str) or not solo_a or not solo_b:
        fail(f"solo baselines produced answers (got {solo_a!r:.80} / {solo_b!r:.80})")
        return
    ok(f"solo baselines: {len(solo_a)} and {len(solo_b)} chars")

    for order, (first, second) in (("A then B", (PROMPT_A, PROMPT_B)),
                                   ("B then A", (PROMPT_B, PROMPT_A))):
        got = {}
        t1 = ask_async(srv.url, first, got, "first")
        # Long enough that the first request is in decode, short enough that it
        # is nowhere near finished: the interleaving has to be real.
        t2 = ask_async(srv.url, second, got, "second", delay=0.7)
        t1.join()
        t2.join()
        want = {"first": solo_a if first is PROMPT_A else solo_b,
                "second": solo_a if second is PROMPT_A else solo_b}
        for key in ("first", "second"):
            if got.get(key) == want[key]:
                ok(f"interleaved ({order}), {key} request is byte-identical to its solo run")
            else:
                fail(f"interleaved ({order}), {key} request is byte-identical to its solo run")
                print(f"       solo: {want[key][:160]!r}")
                print(f"       conc: {str(got.get(key))[:160]!r}")

    # Equality proves nothing if the two never actually shared the card.
    if re.search(r"slot 1: (prefill|decode)", srv.text()):
        ok("both lanes were used (the console shows slot 1 working)")
    else:
        fail("both lanes were used -- everything ran on slot 0, so equality proved nothing")


def check_stall_is_reported(srv):
    """The console must say what the other lane cost, or the bound is a claim."""
    if re.search(r"stall p95 +\d+ ms", srv.text()):
        line = [l for l in srv.text().splitlines() if "stall p95" in l][-1]
        ok(f"the console reports the stall ({line.split('|')[-1].strip()})")
    else:
        fail("the console reports the stall a lane suffered (no 'stall p95' line)")


def check_cancel_leaves_the_other_alone(srv):
    """One client walking away must not touch the other's bytes or its lane."""
    solo_b = ask(srv.url, PROMPT_B)

    got = {}
    keep = ask_async(srv.url, PROMPT_B, got, "kept", delay=0.5)

    # A streaming request that is abandoned mid-decode: the socket closes, and
    # §3.7 says the work stops at the next boundary.
    req = urllib.request.Request(srv.url + "/v1/chat/completions",
                                 data=json.dumps(body(PROMPT_A, 512, stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=600)
    seen = 0
    for raw in resp:
        if raw.startswith(b"data: "):
            seen += 1
        if seen > 4:
            break
    resp.close()                      # the client is gone

    keep.join()
    if got.get("kept") == solo_b:
        ok("cancelling one lane leaves the other's bytes untouched")
    else:
        fail("cancelling one lane leaves the other's bytes untouched")
        print(f"       solo: {solo_b[:160]!r}")
        print(f"       with: {str(got.get('kept'))[:160]!r}")

    for _ in range(60):
        h = srv.health()
        if h["slots_free"] == h["slots_total"]:
            break
        time.sleep(0.5)
    h = srv.health()
    if h["slots_free"] == h["slots_total"]:
        ok(f"the cancelled lane came back ({h['slots_free']}/{h['slots_total']} free)")
    else:
        fail(f"the cancelled lane came back (only {h['slots_free']}/{h['slots_total']} free)")

    if h["kv_blocks_free"] == h["kv_blocks_total"]:
        ok(f"its KV pages came back too ({h['kv_blocks_free']} of {h['kv_blocks_total']})")
    else:
        fail(f"its KV pages came back too ({h['kv_blocks_free']} of {h['kv_blocks_total']} "
             f"-- pages leaked)")


def check_third_request_is_a_clean_503(srv):
    """A lane is a memory reservation; a third one is refused with the numbers."""
    got = {}
    busy = [ask_async(srv.url, PROMPT_A, got, f"busy{i}", 256) for i in range(2)]
    time.sleep(1.0)                    # both lanes in flight
    third = ask(srv.url, "Say hello.", 16)
    for t in busy:
        t.join()

    if not (isinstance(third, tuple) and third[0] == "HTTP"):
        fail("a third concurrent request is refused rather than served (it was served)")
        return
    status, payload = third[1], third[2]
    if status != 503:
        fail(f"a third concurrent request is refused with 503 (got {status})")
        return
    ok("a third concurrent request is refused with 503")

    message = payload.get("error", {}).get("message", "")
    if payload.get("reservation") and re.search(r"\d+\.\d+ GiB", message):
        ok(f"the refusal carries the numbers ({message[:120]}...)")
    else:
        fail("the refusal carries the reservation numbers")
        print(f"       body: {json.dumps(payload)[:300]}")

    if "CL_OUT_OF_RESOURCES" not in srv.text():
        ok("nothing hit an allocation failure on the card")
    else:
        fail("nothing hit an allocation failure on the card (CL_OUT_OF_RESOURCES in the log)")


def check_cold_warm_under_load(binary, model, device, work):
    """§3.4's invariant, held while the other lane is busy."""
    srv = Server(binary, model, device, f"{work}/cache.log", "--n-ctx", "8192",
                 "--parallel", "2", "--prefix-cache-mib", "4096",
                 "--kv-block-size", "32", "--prefill-chunk", "64")
    try:
        if not srv.wait_ready():
            fail("the prefix-cache server started")
            return
        cold = ask(srv.url, PROMPT_A)          # populates
        got = {}
        noise = ask_async(srv.url, PROMPT_B, got, "noise", 256)
        time.sleep(0.5)                        # the other lane is decoding
        warm = ask(srv.url, PROMPT_A)          # must hit, on the other lane
        noise.join()

        if warm == cold:
            ok("warm output is byte-identical to cold with the other lane busy")
        else:
            fail("warm output is byte-identical to cold with the other lane busy")
            print(f"       cold: {cold[:160]!r}")
            print(f"       warm: {warm[:160]!r}")

        if "cache hit" in srv.text():
            hit = [l for l in srv.text().splitlines() if "cache hit" in l][-1]
            ok(f"the cache actually hit ({hit.split('|')[-1].strip()})")
        else:
            fail("the cache actually hit (no hit, so the equality above proved nothing)")

        h = srv.health()
        if h["kv_blocks_free"] < h["kv_blocks_total"]:
            ok(f"the cache is holding KV pages ({h['kv_blocks_total'] - h['kv_blocks_free']} "
               f"of {h['kv_blocks_total']})")
        else:
            fail("the cache is holding KV pages (it holds none, so nothing was shared)")
    finally:
        srv.close()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    binary, model = sys.argv[1], sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "GPU.0"
    work = "/tmp/arcint-concurrency"
    subprocess.run(["mkdir", "-p", work], check=True)

    print(f"== arcint concurrency suite on {device}")
    srv = Server(binary, model, device, f"{work}/main.log", "--n-ctx", "8192", "--parallel", "2")
    try:
        if not srv.wait_ready():
            fail("the two-lane server started")
        else:
            check_no_cross_slot_bleed(srv)
            check_stall_is_reported(srv)
            check_cancel_leaves_the_other_alone(srv)
            check_third_request_is_a_clean_503(srv)
    finally:
        srv.close()

    check_cold_warm_under_load(binary, model, device, work)

    print()
    if FAILED:
        print(f"concurrency: {len(FAILED)} check(s) failed")
        return 1
    print("concurrency: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
