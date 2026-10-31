#!/usr/bin/env python3
"""Normalise a golden capture so a refactor can be compared byte for byte.

Only two fields are inherently variable, and they are the only ones replaced:

  * the response id (`chatcmpl-N`, `cmpl-N`, `resp-N`) — the process-wide
    request counter, so a reordered capture would renumber them;
  * `created` — the wall clock at prepare time.

Everything else stays as it is on the wire (whitespace, key order, SSE
framing), because `prepare_chat` must not change the bytes it serves.

Usage: normalise.py <capture-dir>
"""
import re
import sys
from pathlib import Path

ID = re.compile(rb'"id":"(?:chatcmpl|cmpl|resp)-\d+"')
CREATED = re.compile(rb'"created":\d+')


def normalise(path: Path) -> None:
    raw = path.read_bytes()
    raw = ID.sub(b'"id":"<id>"', raw)
    raw = CREATED.sub(b'"created":<created>', raw)
    path.write_bytes(raw)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    for path in sorted(root.glob("*.body")):
        normalise(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
