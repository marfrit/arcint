#!/usr/bin/env python3
"""REFERENCE-LOGIT CAPTURE for the 0.5.0 KLD gate, and only the capture.

The KLD gate is the quality adjudicator for Flash-Next serving: arcint's served
logits against a reference implementation's, below AND above the QSA boundary.
THIS TOOL BUILDS THE REFERENCE HALF. The comparison is deliberately not here --
it needs served logits, and there are none yet.

--------------------------------------------------------------------------
THE REFERENCE IS NOT A FORK ANY MORE
--------------------------------------------------------------------------

The roadmap recorded the reference as "the llama.cpp qwen4exp fork", carrying a
companion PR. That is superseded: `qwen4exp` is IN UPSTREAM MASTER. Verified
2026-09-13 -- `LLM_ARCH_QWEN4EXP` is registered in `src/llama-arch.cpp` and the
graph is its own translation unit, `src/models/qwen4exp.cpp`; the shipped GGUF
declares `general.architecture = "qwen4exp"`, so they are the same arch string.
Pin a master commit, not a PR branch.

Upstream's own `llama-perplexity --kl-divergence-base FNAME` (alias
`--save-all-logits`) IS the capture, and its `--kl-divergence` is the
comparison half this tool does not invoke. Nothing here reimplements either;
what this adds is the part a bare invocation cannot carry:

  * THE REFUSAL BELOW, which is the whole reason the tool exists rather than a
    shell line;
  * a MANIFEST beside the capture -- every input hashed, so a capture found on
    disk in three weeks can be attributed to a reference, a model and a corpus
    rather than to a memory.

--------------------------------------------------------------------------
WHY A CONTEXT THAT DOES NOT CROSS THE BOUNDARY IS REFUSED
--------------------------------------------------------------------------

The gate's whole shape is a comparison on BOTH sides of the QSA boundary:
below it the served logits must sit inside the noise floor, above it they must
diverge by the measured QSA price and no more. The boundary and the price are
measured facts, not settings:

    T <= 2051   price 0.0, exact          (the boundary is DERIVED, and it is
                                           not the 2048 budget)
    T == 2052   2.307817e-06 over 1/2052 rows
    T == 2080   2.385560e-02 over 29/2080 rows = T - 2051

[DATED 2026-09-28: the magnitudes above were drawn at 692c0a6 with the (1 + w)
fold applied twice to the q/k gammas; with the feed's gamma1 the pin reads
5.082879e-05 over 1/2052 and 1.064551e-03 over 29/2080 rows. The above-2051
bar built on it is withdrawn.]

The count of touched rows is `T - 2051`, and it is DERIVED FROM THOSE TWO
MEASURED POINTS rather than from reasoning about position indices -- which is
how the first draft of this file got it wrong, by one, in both the arithmetic
and the test that was supposed to pin it. `T - 1 - 2051` gives 0 and 28 where
the measurement says 1 and 29. So the smallest window the price touches at all
is 2052, not 2053.

A capture taken at a context of 2051 or less touches no row. It is not a worse
capture; it is a capture the gate CANNOT USE for half its job, and it looks
exactly like a usable one on disk. So it is refused by name here rather than
discovered later -- arcint's standing pattern for the class.

--------------------------------------------------------------------------
WHAT THIS TOOL DOES NOT KNOW
--------------------------------------------------------------------------

Every path is an argument. The reference binary, the model and the corpus live
on the measurement host and none of their locations belong in a public
repository; the tool hashes whatever it is handed and records the hashes.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

# THE QSA BOUNDARY, measured, not chosen: the largest T at which the
# QSA-to-dense price is exactly 0.0. The budget is 2048; the boundary is 2051,
# and the difference is the point -- see this module's docstring for the three
# measured rows.
QSA_BOUNDARY_TOKENS = 2051

# CORRECTED 2026-09-13. This read "`llama-perplexity` evaluates `--chunks`
# windows of `--ctx-size` tokens each and needs at least two windows' worth of
# tokens before it will start", and a refusal of `--chunks 1` was billed on it.
# That conflated two different requirements:
#
#   * the corpus must tokenise to at least 2 * n_ctx tokens, WHATEVER --chunks
#     says -- that is the failure a first attempt actually hit ("you need at
#     least 1024 tokens to evaluate perplexity with a context of 512");
#   * --chunks 1 over a long enough corpus produces a perfectly good
#     one-window capture, and refusing it was unfounded.
#
# So the chunk floor is 1, and the corpus is checked instead -- as far as it
# CAN be checked here, which is stated rather than overstated below.
MIN_CHUNKS = 1

# A token is at least one byte, so a corpus of N bytes tokenises to AT MOST N
# tokens. That makes `bytes < 2 * n_ctx` a refusal that cannot be wrong. It is
# a weak bound on purpose: the tool has no tokenizer, and the real ratio is
# text-dependent (measured on two corpora the same day: 4.56 B/token on
# repetitive prose, at most 11.14 B/token on the fleet corpus). Anything
# tighter would be a guess wearing a refusal's clothes, so a likely-short
# corpus gets a WARNING and only an impossible one gets refused.
LIKELY_BYTES_PER_TOKEN = 4.0


class CaptureRefusal(RuntimeError):
    """A capture that would be taken but could not serve the gate."""


def rows_past_boundary(n_ctx, boundary=QSA_BOUNDARY_TOKENS):
    """How many rows of a T-token window the QSA price touches.

    DERIVED FROM THE TWO MEASURED POINTS, not from reasoning about position
    indices -- which is how this got written wrong once. The recorded rows are

        T = 2052   1 / 2052 rows
        T = 2080  29 / 2080 rows = T - 2051

    so the count is `T - boundary`, and the first draft's `T - 1 - boundary`
    gives 0 and 28 against a measured 1 and 29.

    THIS DESCRIBES THE PRICE, NOT THE CAPTURE. A row at 0-based index `p`
    carries the price when `p >= boundary`. Which of those rows a capture
    actually CONTAINS is a different question, and getting the two confused is
    the correction below.
    """
    return max(0, n_ctx - boundary)


# WHAT A CAPTURE ACTUALLY CONTAINS -- CORRECTION, 2026-09-13, and the first
# version of this file shipped without it.
#
# `llama-perplexity` does not record a row per token of the window. It records
# THE SECOND HALF ONLY, and says why in its own comment: "calculate the
# perplexity over the last half of the window (so the model always has some
# context to predict the token)". In the source, `first = n_ctx/2` and
# `n_ctx - 1 - first` rows are written per chunk, so the recorded 0-based row
# indices are `n_ctx/2 .. n_ctx-2`.
#
# MEASURED, on a real capture rather than read off the source alone: a
# `-c 4096 --chunks 2` run wrote 2,033,309,700 B. The file's own header is
# `"_logits_"` + n_ctx + n_vocab + n_chunk + the token ids, and each recorded
# row is `2*((n_vocab+1)/2)+4` uint16. At n_vocab 248,320 that is 496,648 B a
# row, and (2,033,309,700 - 32,788) / 496,648 = 4,094 rows EXACTLY -- which is
# 2 chunks x (4096 - 1 - 2048). The model is confirmed by the artifact.
#
# THE CONSEQUENCE, and it is why this correction is not cosmetic: at
# `-c 4096` the recorded rows are 2048..4094, of which exactly THREE (2048,
# 2049, 2050) sit below the boundary against 2044 at or above it. That capture
# passes the old refusal -- the window certainly crosses the boundary -- and is
# useless for the half of the gate that judges the served logits against the
# noise floor BELOW it. The refusal checked the wrong range, which made it a
# refusal that could not refuse the thing it exists to refuse.


def recorded_rows(n_ctx):
    """(first, last, count) of the 0-based row indices a capture contains."""
    first = n_ctx // 2
    last = n_ctx - 2
    return first, last, max(0, last - first + 1)


def rows_by_side(n_ctx, boundary=QSA_BOUNDARY_TOKENS):
    """(below, at_or_above) counts among the rows a capture actually contains.

    A row at index `p` carries the QSA price when `p >= boundary`.
    """
    first, last, count = recorded_rows(n_ctx)
    if count <= 0:
        return 0, 0
    below = max(0, min(last + 1, boundary) - first)
    at_or_above = max(0, last - max(first, boundary) + 1)
    return below, at_or_above


def balanced_n_ctx(boundary=QSA_BOUNDARY_TOKENS):
    """The window that puts the most rows on the SCARCER side.

    below = boundary - n/2 and at_or_above = n - 1 - boundary, so the two are
    equal at n = (2*boundary + 1) / 1.5. Searched rather than rounded off that
    algebra, because the floor division in `recorded_rows` makes the exact
    solution not quite the best integer.
    """
    best, best_n = -1, None
    for n in range(boundary + 2, 2 * boundary + 4):
        low, high = rows_by_side(n, boundary)
        worst = min(low, high)
        if worst > best:
            best, best_n = worst, n
    return best_n, best


def check_capture_serves_both_sides(n_ctx, min_rows_each_side,
                                    boundary=QSA_BOUNDARY_TOKENS):
    """Refuse a window whose RECORDED rows do not populate both sides.

    Not "does the window cross the boundary" -- that was the first version's
    question and it is the wrong one. The question is whether the rows the
    capture CONTAINS land on both sides of it, with enough of them on each to
    say anything.
    """
    below, at_or_above = rows_by_side(n_ctx, boundary)
    if below >= min_rows_each_side and at_or_above >= min_rows_each_side:
        return below, at_or_above
    first, last, count = recorded_rows(n_ctx)
    best_n, best_rows = balanced_n_ctx(boundary)
    raise CaptureRefusal(
        f"--n-ctx {n_ctx} cannot serve the KLD gate. A capture records only "
        f"the SECOND HALF of each window -- rows {first}..{last}, {count} of "
        f"them -- so of the rows it would contain, {below} sit below the QSA "
        f"boundary {boundary} and {at_or_above} at or above it, against the "
        f"{min_rows_each_side} each side this run asked for.\n"
        f"The gate needs both: below the boundary the served logits are judged "
        f"against the noise floor, above it against the measured price (0.0 at "
        f"T<={boundary}, 1/2052 rows at T=2052, 2.385560e-02 over 29/2080 rows "
        f"at T=2080).\n"
        f"--n-ctx {best_n} is the balanced width ({best_rows} rows on the "
        f"scarcer side); lower --min-rows-each-side if fewer will do.")


def check_corpus_long_enough(corpus, n_ctx):
    """Refuse a corpus that CANNOT hold 2 * n_ctx tokens; warn if it looks short.

    `llama-perplexity` needs at least two windows' worth of tokens before it
    evaluates anything, whatever `--chunks` asks for, and it discovers that
    AFTER loading the model -- which on the shipped artifact is a minute of
    streaming before it tells you. Catching it here costs nothing.

    The tool has no tokenizer, so the refusal uses the one bound that cannot be
    wrong: a token is at least one byte, so N bytes is at most N tokens.
    """
    try:
        size = os.path.getsize(corpus)
    except OSError:
        return None                       # a missing corpus is the runner's error
    needed = 2 * n_ctx
    if size < needed:
        raise CaptureRefusal(
            f"the corpus is {size:,} B and `llama-perplexity` needs at least "
            f"{needed:,} tokens (2 x --n-ctx) before it evaluates anything. A "
            f"token is at least one byte, so this corpus cannot reach that "
            f"however it tokenises -- and the run would only say so after "
            f"loading the model.")
    if size < needed * LIKELY_BYTES_PER_TOKEN:
        print(f"[kld-capture] WARNING: corpus {size:,} B against "
              f"{needed:,} tokens needed; at a typical ~"
              f"{LIKELY_BYTES_PER_TOKEN:g} B/token that is close. If the run "
              f"reports too few tokens, this is why.")
    return size


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_head(path):
    """The commit of the checkout `path` sits in, or None. A capture whose
    reference cannot be pinned is still taken -- the manifest says `null` and
    the reader knows what they have."""
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def build_command(args):
    """The `llama-perplexity` invocation, as a list. Separate from running it
    so the tests can read the command without a model on disk."""
    return [
        args.reference_bin,
        "-m", args.model,
        "-f", args.corpus,
        "-c", str(args.n_ctx),
        "-b", str(args.batch),
        "--chunks", str(args.chunks),
        "-t", str(args.threads),
        "--kl-divergence-base", args.out,
    ]


def manifest(args, sides, started, elapsed, returncode, stderr_tail):
    """Everything needed to attribute this capture later, hashes included.

    The capture file itself is hashed too. It is the artifact; a manifest that
    describes inputs and not the output cannot detect a truncated run.
    """
    return {
        "tool": "tools/kld_capture.py",
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                      time.gmtime(started)),
        "elapsed_seconds": round(elapsed, 1),
        "returncode": returncode,
        "qsa_boundary_tokens": QSA_BOUNDARY_TOKENS,
        "n_ctx": args.n_ctx,
        "chunks": args.chunks,
        "batch": args.batch,
        "threads": args.threads,
        "recorded_rows_per_window": recorded_rows(args.n_ctx)[2],
        "recorded_rows_below_boundary": sides[0],
        "recorded_rows_at_or_above_boundary": sides[1],
        "reference": {
            "binary": os.path.abspath(args.reference_bin),
            "binary_sha256": sha256_file(args.reference_bin),
            "checkout_head": git_head(args.reference_checkout)
                             if args.reference_checkout else None,
        },
        "model": {
            "path": os.path.abspath(args.model),
            "sha256": sha256_file(args.model) if args.hash_model else None,
        },
        "corpus": {
            "path": os.path.abspath(args.corpus),
            "sha256": sha256_file(args.corpus),
            "bytes": os.path.getsize(args.corpus),
        },
        "capture": {
            "path": os.path.abspath(args.out),
            "bytes": os.path.getsize(args.out) if os.path.exists(args.out) else 0,
            "sha256": sha256_file(args.out) if os.path.exists(args.out) else None,
        },
        "stderr_tail": stderr_tail,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reference-bin", required=True,
                   help="upstream llama.cpp `llama-perplexity`")
    p.add_argument("--reference-checkout",
                   help="the checkout it was built from, to pin its commit")
    p.add_argument("--model", required=True, help="the qwen4exp GGUF (shard 1)")
    p.add_argument("--corpus", required=True, help="the fixed prompt set")
    p.add_argument("--out", required=True, help="capture file to write")
    p.add_argument("--n-ctx", type=int, default=4096,
                   help="evaluated window; must cross the QSA boundary")
    p.add_argument("--chunks", type=int, default=MIN_CHUNKS)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--min-rows-each-side", type=int, default=256,
                   help="refuse unless the capture's RECORDED rows put at "
                        "least this many on each side of the QSA boundary")
    p.add_argument("--hash-model", action="store_true",
                   help="sha256 the model too; it is tens of GiB, so off by "
                        "default and the manifest records null rather than "
                        "a figure nobody waited for")
    p.add_argument("--dry-run", action="store_true",
                   help="run every check and print the command, run nothing")
    args = p.parse_args(argv)

    below, at_or_above = check_capture_serves_both_sides(
        args.n_ctx, args.min_rows_each_side)
    if args.chunks < MIN_CHUNKS:
        raise CaptureRefusal(
            f"--chunks {args.chunks}: at least {MIN_CHUNKS} window is needed "
            f"for a capture to contain anything.")
    check_corpus_long_enough(args.corpus, args.n_ctx)

    cmd = build_command(args)
    first, last, count = recorded_rows(args.n_ctx)
    print(f"[kld-capture] window {args.n_ctx}; a capture records rows "
          f"{first}..{last} ({count} of them, the second half only): "
          f"{below} below the QSA boundary {QSA_BOUNDARY_TOKENS}, "
          f"{at_or_above} at or above it")
    print("[kld-capture] " + " ".join(cmd))
    if args.dry_run:
        return 0

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-12:])
    print(tail)

    man = manifest(args, (below, at_or_above), started, elapsed,
                   proc.returncode, tail)
    with open(args.out + ".manifest.json", "w") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"[kld-capture] {man['capture']['bytes']:,} B in "
          f"{man['elapsed_seconds']} s -> {args.out}")
    print(f"[kld-capture] manifest -> {args.out}.manifest.json")
    return proc.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CaptureRefusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
