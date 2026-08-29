"""The M6 measurement client: what two lanes do to each other, from outside.

Everything here is measured at the endpoint that matters -- the SSE stream a
client actually reads -- so a number this prints is a number a caller would
have felt. The server prints its own view on the console (per-slot rates and
the stall p95); the two are meant to be compared, not to substitute for each
other.

    single          one stream, nothing else running: the regression baseline
    agent-subagent  one long-context session mid-decode + bursts of short
                    requests at the same model, which is the use case M6 is
                    scoped to

The interesting quantity is the *inter-token gap* on the long session while
the short one prefills. A mean decode rate hides it (one 300 ms stall inside
40 tokens of 15 ms each is a 12% rate change and a 20x latency spike), so the
gap distribution is reported, never averaged away.

    python3 tools/bench_slots.py --url http://127.0.0.1:8090 single --depth 2000
    python3 tools/bench_slots.py --url http://127.0.0.1:8090 agent-subagent \
        --depth 30000 --bursts 6
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

FILLER = ("The recurrent state of a gated delta network integrates every token it "
          "has seen, which is why a context shift cannot be implemented honestly. ")


# Measured against the coder's own tokenizer, not guessed: a 12-rep prompt plus
# the chat template renders 332 prompt tokens, so one repetition is ~27.
TOKENS_PER_REP = 27


def filler_prompt(approx_tokens, tail):
    reps = max(1, approx_tokens // TOKENS_PER_REP)
    return FILLER * reps + tail


class Stream:
    """One streaming request, timed at every token boundary."""

    def __init__(self, name, url, prompt, max_tokens, temperature=0.0, seed=None,
                 ignore_eos=False):
        self.name = name
        self.ignore_eos = ignore_eos
        self.url = url
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.text = ""
        self.gaps = []          # seconds between consecutive token arrivals
        self.ttft = None        # seconds to the first token
        self.total = None       # seconds to the last token
        self.tokens = 0
        self.error = None
        self.t_start = None
        self.first_token_at = None

    def body(self):
        d = {"messages": [{"role": "user", "content": self.prompt}],
             "temperature": self.temperature,
             "max_tokens": self.max_tokens,
             "stream": True,
             "chat_template_kwargs": {"enable_thinking": False}}
        if self.seed is not None:
            d["seed"] = self.seed
        if self.ignore_eos:
            # A decode of exactly max_tokens, so a rate is a rate and not a
            # measurement of how chatty the model felt.
            d["ignore_eos"] = True
        return json.dumps(d).encode()

    def run(self):
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=self.body(),
                                     headers={"Content-Type": "application/json"})
        self.t_start = time.perf_counter()
        prev = self.t_start
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    now = time.perf_counter()
                    obj = json.loads(payload)
                    if not obj.get("choices"):
                        continue          # a usage-only or role-only frame
                    delta = obj["choices"][0].get("delta", {})
                    piece = delta.get("content")
                    if piece is None:
                        continue
                    self.text += piece
                    self.tokens += 1
                    if self.ttft is None:
                        self.ttft = now - self.t_start
                        self.first_token_at = now
                    else:
                        self.gaps.append(now - prev)
                    prev = now
        except Exception as e:                       # noqa: BLE001 - reported, not raised
            self.error = f"{type(e).__name__}: {e}"
        self.total = time.perf_counter() - self.t_start

    def decode_rate(self):
        """Pieces per second over the decode phase only (TTFT excluded).

        Pieces, not tokens: streaming holds back an incomplete UTF-8 sequence
        until it is whole (§3.7), so a frame is not a token. The server console
        is the authority on t/s; what this client is authoritative about is
        *when* bytes reached a reader, which is the stall.
        """
        if self.first_token_at is None or self.tokens < 2:
            return 0.0
        span = (self.first_token_at + sum(self.gaps)) - self.first_token_at
        return (self.tokens - 1) / span if span > 0 else 0.0

    def report(self, prefix=""):
        if self.error:
            print(f"{prefix}{self.name}: ERROR {self.error}")
            return
        g = sorted(self.gaps)
        def pct(p):
            if not g:
                return 0.0
            return g[min(len(g) - 1, int(p * len(g)))]
        print(f"{prefix}{self.name}: {self.tokens} pieces | ttft {self.ttft:6.2f} s | "
              f"decode {self.decode_rate():6.1f} p/s | gap p50 {pct(0.5)*1000:6.1f} ms "
              f"p95 {pct(0.95)*1000:7.1f} ms max {(g[-1] if g else 0)*1000:7.1f} ms")


def http_json(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8", "replace"))


def scenario_single(args):
    s = Stream("solo", args.url, filler_prompt(args.depth, "\n\nSummarise the paragraph above."),
               args.max_tokens, seed=args.seed, ignore_eos=args.ignore_eos)
    s.run()
    s.report()
    return 0 if s.error is None else 1


def scenario_agent_subagent(args):
    """One long session mid-decode; short requests fired at it from the side.

    The long session is started first and given time to reach steady-state
    decode, so every burst lands during decode rather than during prefill --
    that is the case the milestone is about.
    """
    agent = Stream("agent  ", args.url,
                   filler_prompt(args.depth, "\n\nWrite a long, detailed explanation."),
                   args.agent_tokens, seed=args.seed, ignore_eos=args.ignore_eos)
    t = threading.Thread(target=agent.run)
    t.start()

    # Wait for the long prefill to finish and decode to be under way.
    deadline = time.perf_counter() + args.prefill_budget
    while agent.tokens < args.settle_tokens and time.perf_counter() < deadline:
        if agent.total is not None:
            break
        time.sleep(0.1)
    if agent.tokens == 0:
        print("agent never produced a token; aborting the scenario", file=sys.stderr)
        t.join()
        agent.report()
        return 1
    print(f"  agent decoding after {time.perf_counter() - agent.t_start:.1f} s "
          f"({agent.tokens} tok); firing {args.bursts} subagent request(s)")

    marks = []   # (start, end) of every burst, in the agent's clock
    subs = []
    for i in range(args.bursts):
        if agent.total is not None:
            break
        sub = Stream(f"sub-{i:02d}", args.url,
                     filler_prompt(args.sub_depth, f"\n\nAnswer in one sentence. (probe {i})"),
                     args.sub_tokens, seed=args.seed, ignore_eos=args.ignore_eos)
        t0 = time.perf_counter()
        sub.run()
        marks.append((t0, time.perf_counter()))
        subs.append(sub)
        time.sleep(args.gap)
    t.join()

    print()
    agent.report("  ")
    for s in subs:
        s.report("  ")
    if subs:
        oks = [s for s in subs if s.error is None and s.tokens > 0]
        if oks:
            print(f"  subagent aggregate: {len(oks)}/{len(subs)} ok | "
                  f"ttft median {statistics.median(s.ttft for s in oks):.2f} s | "
                  f"decode median {statistics.median(s.decode_rate() for s in oks):.1f} t/s")

    # The number the milestone asks for: what the agent felt while a subagent
    # was on the card, separated from what it felt when it had the card alone.
    if agent.gaps and marks:
        times = []
        acc = agent.first_token_at
        for g in agent.gaps:
            acc += g
            times.append(acc)
        during, alone = [], []
        for gap, at in zip(agent.gaps, times):
            hit = any(a <= at <= b for a, b in marks)
            (during if hit else alone).append(gap)
        def line(label, xs):
            if not xs:
                print(f"  {label}: no samples")
                return
            xs = sorted(xs)
            print(f"  {label}: n={len(xs):4d} p50 {xs[len(xs)//2]*1000:6.1f} ms "
                  f"p95 {xs[min(len(xs)-1, int(0.95*len(xs)))]*1000:7.1f} ms "
                  f"max {xs[-1]*1000:7.1f} ms")
        print()
        line("agent gaps, card to itself ", alone)
        line("agent gaps, subagent active", during)
    return 0 if agent.error is None else 1


def scenario_admission(args):
    """Fire `--bursts` requests at once and report every status code.

    A refusal here must be a 503 carrying the numbers, not a hang and not an
    allocation failure on the card.
    """
    results = []
    lock = threading.Lock()

    def one(i):
        st, body = http_json(args.url, "/v1/chat/completions",
                             {"messages": [{"role": "user", "content": f"Say hello. ({i})"}],
                              "temperature": 0, "max_tokens": args.sub_tokens,
                              "chat_template_kwargs": {"enable_thinking": False}})
        with lock:
            results.append((i, st, body))

    ts = [threading.Thread(target=one, args=(i,)) for i in range(args.bursts)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(f"  {len(results)} request(s) in {time.perf_counter() - t0:.1f} s")
    for i, st, body in sorted(results):
        if st == 200:
            print(f"    {i}: 200")
        else:
            print(f"    {i}: {st} {json.dumps(body)[:300]}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--depth", type=int, default=2000, help="approximate prompt tokens")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--agent-tokens", type=int, default=400)
    ap.add_argument("--sub-depth", type=int, default=200)
    ap.add_argument("--sub-tokens", type=int, default=32)
    ap.add_argument("--bursts", type=int, default=4)
    ap.add_argument("--gap", type=float, default=1.0, help="seconds between bursts")
    ap.add_argument("--settle-tokens", type=int, default=8)
    ap.add_argument("--ignore-eos", action="store_true",
                    help="decode exactly --max-tokens, so a rate is a rate")
    ap.add_argument("--prefill-budget", type=float, default=1800.0)
    ap.add_argument("scenario", choices=["single", "agent-subagent", "admission"])
    args = ap.parse_args()

    return {"single": scenario_single,
            "agent-subagent": scenario_agent_subagent,
            "admission": scenario_admission}[args.scenario](args)


if __name__ == "__main__":
    sys.exit(main())
