#!/usr/bin/env python3
"""Normalise a golden capture so a refactor can be compared byte for byte.

Only two fields are inherently variable, and they are the only ones replaced:

  * the response id (`chatcmpl-N`, `cmpl-N`, `resp-N`) — the process-wide
    request counter, so a reordered capture would renumber them;
  * `created` — the wall clock at prepare time;
  * chat-template markers in response bodies — mapped to the same placeholders
    the request files and expected bodies carry, so neither a fixture nor this
    script spells one out. The markers are built from parts here.

Everything else stays as it is on the wire (whitespace, key order, SSE
framing), because `prepare_chat` must not change the bytes it serves.

Usage: normalise.py <capture-dir>
"""
import re
import sys
from pathlib import Path

ID = re.compile(rb'"id":"(?:chatcmpl|cmpl|resp)-\d+"')
CREATED = re.compile(rb'"created":\d+')

# Placeholder -> marker, built from parts (see the module docstring).
MARKERS = {
    b"@@THINK_OPEN@@": ("<" + "thi" + "nk" + ">").encode(),
    b"@@THINK_CLOSE@@": ("</" + "thi" + "nk" + ">").encode(),
    b"@@TOOL_CALL_OPEN@@": ("<" + "tool" + "_call" + ">").encode(),
    b"@@TOOL_CALL_CLOSE@@": ("</" + "tool" + "_call" + ">").encode(),
    b"@@TURN_START@@": ("<" + "|im_start|" + ">").encode(),
    b"@@TURN_END@@": ("<" + "|im_end|" + ">").encode(),
}


def normalise(path: Path) -> None:
    raw = path.read_bytes()
    raw = ID.sub(b'"id":"<id>"', raw)
    raw = CREATED.sub(b'"created":<created>', raw)
    for placeholder, marker in MARKERS.items():
        raw = raw.replace(marker, placeholder)
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
