#!/usr/bin/env python3
"""Count the turn-shape defects a served unit causes in real agent sessions.

Campaign: docs/campaigns/clean-turns.md. Four classes, per model:

  (a) a turn that ends with NEITHER text nor a tool call  - the client got an
      empty answer with a normal finish;
  (b) a turn that ends INSIDE an open think block - the whole output stayed in
      reasoning_content, so the unit never saw the block's closer;

CAVEAT (2026-10-21, clean-turns full battery): class (b) is counted from
`has_think` - that a thinking block EXISTS - not that it CLOSED. The block's
closed state was not recorded in the sessions this script reads, so (b) cannot
distinguish a mid-block stop from a closed-block empty answer. On the served
coder artifact the full battery (20 arms) found zero mid-block stops: the
"empty stops" are closed-block empty answers, i.e. class (a). A unit that
reports `reasoning_closed` (the clean-turns build) can split the two; these
logs cannot.  (c) aborted or errored turns;
  (d) tool calls that failed at the receiving end: the tool result reports an
      argument/JSON/validation error, or the call carries arguments that are
      not parseable JSON.

Session content never leaves the machine: the script prints counts only.

Marker strings are built from parts, never spelled as a literal - a chat
template's control token written into a prompt or a log line is the defect
this campaign is about.

Two input modes:

  tools/turn_defect_audit.py SESSION_DIR [SESSION_DIR ...] [--exclude WORD]
      Scans Pi session JSONL files (one JSON object per line). --exclude skips
      WHOLE FILES whose text contains WORD; the file is never opened beyond the
      grep that decides it. Repeatable.

  journalctl --user -u <unit> | tools/turn_defect_audit.py --journal
      Counts the served unit's own tool-call parse warnings against the number
      of completions it logged. Reads stdin; no unit name in this file.
"""

import argparse
import collections
import glob
import json
import os
import re
import sys


def markers():
    """The template's control tokens, assembled so this file never contains one."""
    lt, gt = chr(60), chr(62)
    return {
        "think_close": lt + "/think" + gt,
        "think_open": lt + "think" + gt,
        "thinking_open": lt + "thinking" + gt,
        "tool_open": lt + "tool_call" + gt,
        "tool_close": lt + "/tool_call" + gt,
        "eot": lt + "|endoftext|" + gt,
        "im_end": lt + "|im_end|" + gt,
    }


M = markers()

# Argument-level failures at the tool: the call reached the tool and the tool
# refused the arguments. Distinguished from "the tool ran and its command
# failed", which is not a tool-call defect.
ARG_ERROR = re.compile(
    r"(invalid json|json parse|parse error|unexpected eof|invalid arguments?"
    r"|missing required|missing argument|unrecognized argument|unexpected argument"
    r"|validation error|input validation|failed to match schema|schema validation"
    r"|expected .*argument|no such option|unknown option|must be specified"
    r"|unable to parse|malformed|not a valid|pydantic"
    r"|cannot deserialize|expected string or bytes"
    r"|did not match the expected format|required parameter)",
    re.I,
)
# Shell-level fallout of a badly quoted argument string.
SHELL_ARG = re.compile(r"(unexpected eof while looking for matching"
                       r"|here-document .* delimited by end-of-file)", re.I)

# Served-unit log lines.
JOURNAL_WARN = re.compile(r"unterminated .*block; returned as raw content")
JOURNAL_COMPLETION = re.compile(r"^\s*lgc\s+slot \d+: prefill\s+\d+ tok")
JOURNAL_ABORT = re.compile(r"abort", re.I)


def blocks(msg, kind):
    return [b for b in msg.get("content", []) if isinstance(b, dict) and b.get("type") == kind]


def result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def scan_sessions(paths, excludes):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)))
        else:
            files.append(p)

    kept, skipped = [], 0
    for f in files:
        bad = False
        if excludes:
            try:
                with open(f, "r", errors="replace") as fh:
                    blob = fh.read()
            except OSError:
                bad = True
            for w in excludes:
                if w in blob:
                    bad = True
                    break
        if bad:
            skipped += 1
        else:
            kept.append(f)

    per_model = collections.defaultdict(lambda: collections.Counter())
    marker_hits = collections.Counter()
    turns = 0
    # A toolResult line carries no model; remember which model made each call.
    call_owner = {}

    for f in kept:
        for line in open(f, "r", errors="replace"):
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if o.get("type") != "message":
                continue
            m = o.get("message") or {}
            role = m.get("role")

            if role == "assistant":
                turns += 1
                model = m.get("model") or m.get("responseModel") or "?"
                pm = per_model[model]
                pm["turns"] += 1
                stop = m.get("stopReason")
                has_text = any(b.get("text", "").strip() for b in blocks(m, "text"))
                has_call = bool(blocks(m, "toolCall"))
                has_think = bool(blocks(m, "thinking"))

                if stop in ("error", "aborted"):
                    pm["c_aborted_or_error"] += 1
                    if m.get("errorMessage"):
                        pm["c_error"] += 1
                    else:
                        pm["c_aborted"] += 1
                elif not has_text and not has_call:
                    pm["a_empty_turn"] += 1
                    # finish "stop" = the model's own end-of-generation token won;
                    # "length" = the token cap. Only the first is an unwanted stop.
                    pm["a_empty_" + (stop or "none")] += 1
                    if has_think:
                        pm["b_open_think_block"] += 1
                    else:
                        pm["a_totally_empty"] += 1
                elif stop == "length" and has_think and not has_call and not has_text:
                    pm["b_open_think_block"] += 1

                # The mention signature: a control token that reached the text
                # the client was shown, i.e. the model wrote it as content.
                for b in blocks(m, "text"):
                    t = b.get("text", "")
                    for name, tok in M.items():
                        if tok in t:
                            marker_hits[name] += 1
                            pm["d_marker_in_text"] += 1

                for b in blocks(m, "toolCall"):
                    pm["d_tool_calls"] += 1
                    if b.get("id"):
                        call_owner[b["id"]] = model
                    a = b.get("arguments")
                    if isinstance(a, str):
                        try:
                            json.loads(a)
                        except ValueError:
                            pm["d_arguments_not_json"] += 1
                    elif not isinstance(a, dict):
                        pm["d_arguments_not_json"] += 1

            elif role == "toolResult" and m.get("isError"):
                model = call_owner.get(m.get("toolCallId")) or "?"
                pm = per_model[model]
                pm["tool_results_errored"] += 1
                t = result_text(m.get("content"))
                if ARG_ERROR.search(t) or SHELL_ARG.search(t):
                    pm["d_argument_errors_at_tool"] += 1

    return per_model, marker_hits, skipped, len(kept), turns


def scan_journal(stream):
    warns = completions = aborts = 0
    for line in stream:
        if JOURNAL_WARN.search(line):
            warns += 1
        if JOURNAL_COMPLETION.search(line):
            completions += 1
        if JOURNAL_ABORT.search(line):
            aborts += 1
    return warns, completions, aborts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="session dirs or JSONL files")
    ap.add_argument("--exclude", action="append", default=[],
                    help="skip whole files containing WORD (repeatable)")
    ap.add_argument("--journal", action="store_true", help="read a unit's journal from stdin")
    args = ap.parse_args()

    if args.journal:
        warns, completions, aborts = scan_journal(sys.stdin)
        print("journal: completions logged %d | unterminated tool-call blocks %d "
              "| lines mentioning abort %d" % (completions, warns, aborts))
        return 0

    if not args.paths:
        ap.error("session paths required (or --journal)")
    per_model, hits, skipped, kept, turns = scan_sessions(args.paths, args.exclude)

    print("files scanned %d (skipped by --exclude: %d) | assistant turns %d"
          % (kept, skipped, turns))
    cols = ["turns", "a_empty_turn", "a_empty_stop", "a_empty_length", "a_totally_empty",
            "b_open_think_block",
            "c_aborted_or_error", "c_error", "c_aborted",
            "d_tool_calls", "d_arguments_not_json", "d_argument_errors_at_tool",
            "tool_results_errored", "d_marker_in_text"]
    print("\n%-34s %s" % ("model", " ".join("%10s" % c.replace("_", " ") for c in cols)))
    totals = collections.Counter()
    for model, c in sorted(per_model.items(), key=lambda kv: -kv[1]["turns"]):
        totals.update(c)
        print("%-34s %s" % (model[:34], " ".join("%10d" % c[col] for col in cols)))
    print("%-34s %s" % ("TOTAL", " ".join("%10d" % totals[col] for col in cols)))
    if hits:
        print("\ncontrol token found in client-visible text (count of occurrences):")
        for k, v in hits.most_common():
            print("  %-14s %d" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
