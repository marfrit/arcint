#!/usr/bin/env python3
"""arcint measured the way llama-bench measures, for the community GPU tables.

The tables compare cards on one file: Llama 2 7B at Q4_0 (TheBloke/Llama-2-7B-GGUF,
llama-2-7b.Q4_0.gguf), every layer on the card, pp512 and tg128 in t/s with their
spread over 5 repetitions. This tool puts arcint in llama-bench's place:

    llama-bench                                  here
    warm-up decode, not counted                  one pp and one tg request, not counted
    pp: 512 random tokens, memory cleared        a token-id prompt of 512 random ids,
                                                 a new one each repetition; a run the
                                                 prompt cache helped is refused
    tg: 128 tokens from an empty context,        a one-token prompt, 128 tokens, greedy,
    one at a time, no sampler                    ignore_eos, no MTP (the sampler is
                                                 included, ~0.1 ms a token, measured)
    t/s from its own timers                      the engine's own /metrics counters
                                                 around each request, so no HTTP time
    n_ctx 4096 (the model's), F16 KV             --n-ctx 4096, arcint's default F16 KV

Differences that stay: arcint asks for flash attention on auto, which is flash attention
on this backend, so a row compares with the tables' FA rows; and the path is the served
one (HTTP, slot, sampler), which is what a client gets. `--kv` adds an arm at another KV precision, named in its row:
only the f16 row is the tables' configuration.

One process per card and arm; the card must be free (stop the unit that holds it).

    python3 tools/llama_scoreboard.py --arcint /usr/bin/arcint \\
        --model llama-2-7b.Q4_0.gguf --card "Arc A770=16 GB=GPU.1" \\
        --card "Arc Pro B60=24 GB=GPU.0" --kv f16 --kv q8_0:q4_0 --out out/

    --repetitions 1 is the smoke; --dry-run prints the server command lines.
"""
import argparse
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.request

MODEL_SHA256 = "78b8f9777dd620ad29cd2cffb6653b17fa8a5b1fddc1b8821180d60eedd24d48"
N_VOCAB = 32000
HEADER = "| Chip | Memory | pp512 t/s | tg128 t/s | Engine | KV |\n|---|---|---|---|---|---|"
COUNTERS = ("arcint_prompt_tokens_total", "arcint_completion_tokens_total",
            "arcint_prompt_cache_hit_tokens_total", "arcint_prefill_seconds_total", "arcint_decode_seconds_total")


class NotComparable(ValueError):
    """The run would not make a row the tables can hold."""


def server_argv(arcint, model, device, port, kv):
    cmd = [arcint, "--engine", "llama", "--gguf", model, "--served-model-name", "bench", "--device", device,
           "--host", "127.0.0.1", "--port", str(port), "--n-ctx", "4096", "--metrics"]
    if kv != "f16":
        cmd += ["--llama-kv", kv]
    return cmd


def parse_metrics(text):
    """The engine's counters from a Prometheus exposition; a missing one is an error."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", line)
        if m and m.group(1) in COUNTERS:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    missing = [c for c in COUNTERS if c not in out]
    if missing:
        raise NotComparable(f"/metrics lacks {missing}")
    return out


def pp_rate(before, after, n_prompt=512):
    """t/s of one prefill, from the counters around it."""
    d = {k: after[k] - before[k] for k in COUNTERS}
    if d["arcint_prompt_cache_hit_tokens_total"] != 0:
        raise NotComparable(f"the prompt cache served {d['arcint_prompt_cache_hit_tokens_total']:.0f} tokens")
    if d["arcint_prompt_tokens_total"] != n_prompt:
        raise NotComparable(f"prefilled {d['arcint_prompt_tokens_total']:.0f} tokens, not {n_prompt}")
    if d["arcint_prefill_seconds_total"] <= 0:
        raise NotComparable("no prefill time on the counters")
    return n_prompt / d["arcint_prefill_seconds_total"]


def tg_rate(before, after, n_gen=128):
    """t/s of one generation, from the counters around it."""
    d = {k: after[k] - before[k] for k in COUNTERS}
    if d["arcint_completion_tokens_total"] != n_gen:
        raise NotComparable(f"generated {d['arcint_completion_tokens_total']:.0f} tokens, not {n_gen}")
    if d["arcint_decode_seconds_total"] <= 0:
        raise NotComparable("no decode time on the counters")
    return n_gen / d["arcint_decode_seconds_total"]


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def row(chip, memory, pp, tg, engine, kv):
    return f"| {chip} | {memory} | {pp[0]:.2f} ± {pp[1]:.2f} | {tg[0]:.2f} ± {tg[1]:.2f} | {engine} | {kv} |"


def random_prompt(rng, n):
    return [rng.randrange(1000, N_VOCAB) for _ in range(n)]


def parse_card(text):
    parts = text.split("=")
    if len(parts) != 3 or not re.fullmatch(r"GPU\.\d+", parts[2]):
        raise argparse.ArgumentTypeError("a card is name=memory=GPU.N")
    return parts


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
        return r.read().decode()


def complete(port, prompt, max_tokens):
    body = {"model": "bench", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def measure(port, repetitions, rng):
    complete(port, random_prompt(rng, 512), 1)          # warm-up: kernels build on first use
    complete(port, random_prompt(rng, 1), 16)
    pp, tg = [], []
    for _ in range(repetitions):
        before = parse_metrics(get(port, "/metrics"))
        complete(port, random_prompt(rng, 512), 1)
        pp.append(pp_rate(before, parse_metrics(get(port, "/metrics"))))
    for _ in range(repetitions):
        before = parse_metrics(get(port, "/metrics"))
        complete(port, random_prompt(rng, 1), 128)
        tg.append(tg_rate(before, parse_metrics(get(port, "/metrics"))))
    return pp, tg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arcint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--card", action="append", type=parse_card, required=True)
    p.add_argument("--kv", action="append", help="KV precision of an arm; default f16, the tables' configuration")
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--port", type=int, default=8095)
    p.add_argument("--out", default=".")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--any-model", action="store_true", help="another file, measured the same way; not a row of the tables' model")
    a = p.parse_args()
    kvs = a.kv or ["f16"]
    if a.dry_run:
        for chip, memory, device in a.card:
            for kv in kvs:
                print(" ".join(server_argv(a.arcint, a.model, device, a.port, kv)))
        return
    if not a.any_model and sha256(a.model) != MODEL_SHA256:
        sys.exit("not the tables' file: llama-2-7b.Q4_0.gguf's sha256 differs")
    os.makedirs(a.out, exist_ok=True)
    version = subprocess.run([a.arcint, "--version"], capture_output=True, text=True).stdout.strip()
    engine = " ".join(version.split()[:3])
    rng = random.Random(42)
    rows = []
    for chip, memory, device in a.card:
        for kv in kvs:
            stem = os.path.join(a.out, re.sub(r"\W+", "-", f"{chip}-{kv}"))
            with open(stem + ".log", "w") as log:
                srv = subprocess.Popen(server_argv(a.arcint, a.model, device, a.port, kv), stdout=log, stderr=log)
            try:
                for _ in range(300):
                    if srv.poll() is not None:
                        sys.exit(f"{chip} {kv}: arcint exited {srv.returncode}, see {stem}.log")
                    try:
                        get(a.port, "/health")
                        break
                    except OSError:
                        time.sleep(2)
                pp, tg = measure(a.port, a.repetitions, rng)
            finally:
                srv.terminate()
                srv.wait()
            time.sleep(8)
            with open(stem + ".json", "w") as f:
                json.dump({"chip": chip, "kv": kv, "engine": version, "pp512": pp, "tg128": tg}, f)
            rows.append(row(chip, memory, mean_sd(pp), mean_sd(tg), engine, kv))
            print(rows[-1], flush=True)
    print(HEADER)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
