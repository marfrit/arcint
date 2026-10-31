#!/usr/bin/env python3
"""Cancel measurement for Open Responses (docs/campaigns/feyntalk.md W4/W5).

Two arms at a given prompt length, against a running arcint (`--stub` now, a
card unit later):

  close   the client closes its socket mid-stream (today's control path);
  cancel  a second connection POSTs /v1/responses/{id}/cancel as soon as
          `response.created` gives the id (the arcint extension).

Printed per arm: issue/close -> the server's abort (the slot line in the
server log), issue -> the cancel reply and issue -> the terminal SSE event, and
the abort's prefilled-token count.

The server does not report its prefill progress while a request runs, only at
the abort (its slot line), so `tokens_at_issue` cannot be read live. Pass the
value you measured out of band with --tokens-at-issue and the script prints
`prompt_tokens_prefilled - tokens_at_issue`; without it the delta is `n/a`.
The campaign's bounded expectation is one ubatch (--prefill-chunk).

    tools/feyntalk_cancel.py --base-url http://127.0.0.1:8197 \\
        --log /tmp/stub.log --prompt-words 400 [--arm both|close|cancel]
"""
import argparse
import http.client
import json
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse


def prompt_of(words: int) -> str:
    return " ".join(f"w{i}" for i in range(words))


def log_since(path: str, offset: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            return f.read()
    except OSError:
        return ""


def wait_for_line(path: str, offset: int, pattern: str, timeout: float) -> float | None:
    """Seconds until `pattern` appears in the log appended after `offset`."""
    rx = re.compile(pattern)
    start = time.time()
    while time.time() - start < timeout:
        if rx.search(log_since(path, offset)):
            return time.time() - start
        time.sleep(0.02)
    return None


class Watcher:
    """Records the wall time the server's abort line first appears; it polls
    while the arm is in flight, so issue->abort is not read after the fact."""

    def __init__(self, path: str, offset: int, pattern: str):
        self.path, self.offset, self.pattern = path, offset, pattern
        self._stop = threading.Event()
        self.at: float | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        rx = re.compile(self.pattern)
        while not self._stop.is_set():
            if rx.search(log_since(self.path, self.offset)):
                self.at = time.time()
                return
            time.sleep(0.02)

    def start(self) -> "Watcher":
        self.thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(timeout=5)


def prefilled_from(text: str) -> int | None:
    m = re.findall(r"slot \d+: prefill\s+(\d+) tok", text)
    return int(m[-1]) if m else None


def start_stream(base: str, prompt: str, max_tokens: int):
    u = urlparse(base)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=60)
    body = json.dumps({"model": "stub", "input": prompt, "stream": True,
                       "max_output_tokens": max_tokens})
    conn.request("POST", "/v1/responses", body, {"Content-Type": "application/json"})
    return conn, conn.getresponse()


def arm_close(base: str, prompt: str, log: str, max_tokens: int, after: float) -> dict:
    offset = len(log_since(log, 0))
    watcher = Watcher(log, offset, r"aborted: client gone").start()
    conn, resp = start_stream(base, prompt, max_tokens)
    types = 0
    t0 = time.time()
    while time.time() - t0 < after:
        line = resp.fp.readline()
        if not line:
            break
        if line.decode("utf-8", errors="replace").startswith("event: "):
            types += 1
    t_close = time.time()
    conn.close()   # the client's connection is gone; the server sees it at a boundary
    for _ in range(500):
        if watcher.at is not None:
            break
        time.sleep(0.02)
    watcher.stop()
    text = log_since(log, offset)
    return {"arm": "close", "events_before_close": types,
            "close_to_abort_s": None if watcher.at is None else round(watcher.at - t_close, 3),
            "prefilled": prefilled_from(text)}


def arm_cancel(base: str, prompt: str, log: str, max_tokens: int,
               tokens_at_issue: int | None) -> dict:
    offset = len(log_since(log, 0))
    out: dict = {"arm": "cancel"}
    q: queue.Queue[str] = queue.Queue()

    def canceller():
        rid = q.get()
        t_issue = time.time()
        try:
            req = urllib.request.Request(f"{base}/v1/responses/{rid}/cancel", method="POST")
            r = urllib.request.urlopen(req, timeout=30)
            out["cancel_status"] = r.status
            out["cancel_reply_s"] = round(time.time() - t_issue, 3)
            out["cancelling"] = json.loads(r.read()).get("arcint")
        except urllib.error.HTTPError as e:
            out["cancel_status"] = e.code
            out["cancelling"] = None

    th = threading.Thread(target=canceller)
    th.start()
    conn, resp = start_stream(base, prompt, max_tokens)
    types = []
    done = False
    t_issue = None
    while True:
        line = resp.fp.readline()
        if not line:
            break
        s = line.decode("utf-8", errors="replace").strip()
        if s.startswith("event: "):
            types.append(s[7:])
        if s.startswith("data: {"):
            d = json.loads(s[6:])
            if d.get("type") == "response.created" and t_issue is None:
                t_issue = time.time()
                out["id"] = d["response"]["id"]
                q.put(out["id"])
            if d.get("type") == "response.incomplete":
                out["terminal"] = "response.incomplete"
                out["reason"] = d["response"]["incomplete_details"]["reason"]
                out["terminal_s"] = round(time.time() - t_issue, 3)
        if s == "data: [DONE]":
            done = True
            break
    th.join(timeout=30)
    out["done"] = done
    out["cancelled_event"] = "arcint:response.cancelled" in types
    # Issue -> the abort as the client sees it: the terminal SSE event. The
    # server's slot line is written in the same moment (log_stats precedes the
    # terminal events) but carries no timestamp, so it is not a clock here.
    out["issue_to_abort_s"] = out.get("terminal_s")
    out["issue_to_abort_source"] = "terminal event"
    text = log_since(log, offset)
    prefilled = prefilled_from(text)
    out["prefilled"] = prefilled
    out["tokens_at_issue"] = tokens_at_issue
    out["prefilled_minus_tokens_at_issue"] = (
        prefilled - tokens_at_issue if prefilled is not None and tokens_at_issue is not None else None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8197")
    ap.add_argument("--log", required=True, help="the server's -v log file (slot lines)")
    ap.add_argument("--prompt-words", type=int, default=400)
    ap.add_argument("--max-output-tokens", type=int, default=500)
    ap.add_argument("--arm", choices=["both", "close", "cancel"], default="both")
    ap.add_argument("--close-after-s", type=float, default=0.2)
    ap.add_argument("--tokens-at-issue", type=int, default=None,
                    help="prefill tokens measured out of band at issue (card run)")
    args = ap.parse_args()

    prompt = prompt_of(args.prompt_words)
    arms = []
    if args.arm in ("both", "close"):
        arms.append(arm_close(args.base_url, prompt, args.log,
                              args.max_output_tokens, args.close_after_s))
    if args.arm in ("both", "cancel"):
        arms.append(arm_cancel(args.base_url, prompt, args.log,
                               args.max_output_tokens, args.tokens_at_issue))
    for a in arms:
        print(json.dumps(a, sort_keys=True))
    return 0 if all(a.get("prefilled") is not None for a in arms) else 1


if __name__ == "__main__":
    sys.exit(main())
