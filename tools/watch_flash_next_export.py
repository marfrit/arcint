#!/usr/bin/env python3
"""Weekly watcher for the Qwen Flash Next export blocker (FIX A).

Runs export_flash_next.py's transformers-support check and optimum-intel
compatibility check (not the full export -- no need to touch the network
or a GPU to know whether the blockers have cleared) and appends one line
to a log, so the refusal is a standing, dated record instead of something
re-discovered by hand every time someone asks "has upstream shipped it
yet". The log never says who ran it or from where -- see
docs/design-qwen-flash-next.md, FIX A.

Exit code is always 0: this is meant to run unattended (cron) and a
"still blocked" result is the expected, non-alarming outcome. Read the
log, don't watch for a failing exit code.

Suggested weekly cron line (operator's crontab, not this repo -- host and
account details belong in CLAUDE.local.md, not here):

    0 6 * * 1  /usr/bin/python3 /path/to/ligence/tools/watch_flash_next_export.py

Log path defaults to a git-ignored file next to this script; override with
--log or FLASH_NEXT_WATCH_LOG for a different location (e.g. a persistent
path on whichever host actually carries the venv this checks).
"""
import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_flash_next import check_transformers_support, check_optimum_intel_compat  # noqa: E402

DEFAULT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "flash-next-watch.log")


def run_once(log_path):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    try:
        version = check_transformers_support()
    except AssertionError as exc:
        line = f"{stamp} still blocked (transformers): {exc}"
        with open(log_path, "a") as f:
            f.write(line + "\n")
        print(line)
        return line

    try:
        oi_version, tf_version = check_optimum_intel_compat()
        line = (f"{stamp} OPEN->CLOSED transformers {tf_version} carries "
                f"qwen4_exp and optimum-intel {oi_version} imports cleanly "
                f"-- re-run export_flash_next.py for real")
    except AssertionError as exc:
        line = (f"{stamp} transformers {version} carries qwen4_exp but "
                f"optimum-intel blocked: {exc}")

    with open(log_path, "a") as f:
        f.write(line + "\n")
    print(line)
    return line


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=os.environ.get("FLASH_NEXT_WATCH_LOG", DEFAULT_LOG))
    args = ap.parse_args()
    run_once(args.log)


if __name__ == "__main__":
    main()
