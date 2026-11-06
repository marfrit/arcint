#!/usr/bin/env python3
"""A unit's /props `downgrades` against what is accepted for it.

The release gate runs this after a unit has served long prompts and answers, and fails on
a key that is not accepted. A unit that does not report the field fails too: an old binary
must not read as a clean one, and neither must a unit that served nothing.

    downgrade_check.py URL UNIT [--expected FILE]

FILE maps a unit's name to its accepted keys (tests/acceptance/expected-downgrades.json).
An accepted key that did not occur is said, and is no failure: a key depends on the prompts.
Exit 0: nothing new. Exit 1: new keys, or not reporting. Exit 2: usage or an unknown unit.
"""
import argparse
import json
import os
import sys
import urllib.request


def compare(props, accepted):
    """(ok, new keys with counts, accepted keys that did not occur). `ok` is False when the
    field is absent."""
    seen = props.get("downgrades")
    if not isinstance(seen, list):
        return False, None, list(accepted)
    keys = {d["key"]: d["count"] for d in seen}
    new = sorted((k, n) for k, n in keys.items() if k not in accepted)
    return not new, new, sorted(k for k in accepted if k not in keys)


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("unit")
    ap.add_argument("--expected", default=os.path.join(here, "..", "tests", "acceptance", "expected-downgrades.json"))
    a = ap.parse_args(argv)
    with open(a.expected) as f:
        table = json.load(f)
    if a.unit not in table:
        print("downgrades %s: UNKNOWN UNIT (known: %s)" % (a.unit, ", ".join(sorted(table))))
        return 2
    with urllib.request.urlopen(a.url.rstrip("/") + "/props", timeout=30) as r:
        props = json.load(r)
    ok, new, absent = compare(props, set(table[a.unit]))
    if new is None:
        print("downgrades %s: FAIL, not reporting" % a.unit)
        return 1
    for k, n in new:
        print("downgrades %s: NEW %d %s" % (a.unit, n, k))
    for k in absent:
        print("downgrades %s: accepted, did not occur: %s" % (a.unit, k))
    print("downgrades %s: %s, %d new, %d accepted did not occur" % (a.unit, "ok" if ok else "FAIL", len(new), len(absent)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
