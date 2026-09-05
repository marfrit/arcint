#!/usr/bin/env python3
"""Build a prompt of a target token count against a running server's own
tokenizer -- never assumed, always read back from the response's
usage.prompt_tokens field (docs/design-0.3.1-test-ladder.md, Increment 2:
tier_reference.sh and depth_ladder.sh both need a prompt sized to a specific
depth, and a runner "must never claim a token count you did not read back").

Tiles a committed seed paragraph (tests/acceptance/prompts/*.txt) enough
times to approximate the target by a characters-per-token heuristic, asks
the server (max_tokens=1, so the generated token itself is thrown away) how
many tokens it actually counted, and rescales the tiling proportionally.
Prints what was measured on every round, and the final count on stdout.

    size_prompt.py --seed FILE --target N [--tol PCT] --url BASE_URL --out FILE
"""
import argparse
import json
import sys
import urllib.error
import urllib.request


def measure(url, prompt):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": 1, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        o = json.load(r)
    return o.get("usage", {}).get("prompt_tokens")


def build(seed, repeats):
    unit = seed.strip() + " "
    return (unit * repeats).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", required=True, help="path to the committed seed paragraph")
    p.add_argument("--target", type=int, required=True, help="target prompt_tokens")
    p.add_argument("--tol", type=float, default=3.0,
                   help="percent tolerance to report against (informational, not gated)")
    p.add_argument("--url", required=True, help="base URL of the already-running server")
    p.add_argument("--out", required=True, help="where to write the resolved prompt")
    p.add_argument("--max-rounds", type=int, default=5)
    args = p.parse_args()

    with open(args.seed, encoding="utf-8") as f:
        seed = f.read()

    # First guess only: ~4 characters per token is a common rough estimate for
    # English text. It is corrected below by the server's own count and never
    # trusted on its own.
    unit_chars = len(seed.strip()) + 1
    repeats = max(1, round(args.target * 4 / unit_chars))

    prompt = None
    measured = None
    try:
        for round_ in range(1, args.max_rounds + 1):
            prompt = build(seed, repeats)
            measured = measure(args.url, prompt)
            if not isinstance(measured, int) or measured <= 0:
                print(f"size_prompt: server did not report a usable usage.prompt_tokens "
                      f"(round {round_}, got {measured!r})", file=sys.stderr)
                return 1
            deviation = abs(measured - args.target) / args.target * 100
            print(f"size_prompt: round {round_}: repeats={repeats} measured={measured} "
                  f"target={args.target} deviation={deviation:.1f}%", file=sys.stderr)
            if deviation <= args.tol:
                break
            repeats = max(1, round(repeats * args.target / measured))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"size_prompt: could not reach {args.url}: {e}", file=sys.stderr)
        return 1

    deviation = abs(measured - args.target) / args.target * 100
    if deviation > args.tol:
        # A mis-sized prompt must not gate whatever depth-specific claim the
        # caller is about to test against it -- exit non-zero rather than
        # write out a prompt that silently misses the depth it was asked
        # for. Callers already treat a non-zero exit here as a FAIL with a
        # reason, which is the right place for this to land, not a printed
        # warning nobody gates on.
        print(f"size_prompt: measured {measured} tokens after {args.max_rounds} round(s), "
              f"outside the +/-{args.tol:g}% tolerance around target {args.target} "
              f"(deviation {deviation:.1f}%)", file=sys.stderr)
        return 1

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"size_prompt: final prompt_tokens={measured} (target {args.target} "
          f"+/-{args.tol:g}%, deviation {deviation:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
