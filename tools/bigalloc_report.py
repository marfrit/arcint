#!/usr/bin/env python3
"""Symbolise and rank a tools/bigalloc.c dump.

Frames arrive as `module+0xoffset sym`. For each module the ELF symbol table is
read once with `nm` (the unstripped relink of the same objects when the served
library is stripped: pass `--sym stripped_path=unstripped_path`), and each
offset is mapped to the enclosing function by bisection. Stacks are ranked by
their live bytes at the peak snapshot (`--key at_peak`, the default), by bytes
live at the dump (`live`) or by bytes ever allocated (`total`).

The report prints, per stack, its bytes and the first N symbolised frames that
are not the allocator itself, and a roll-up by the first frame inside the
libraries named with `--own` (default: the OpenVINO core and GPU plugin), which
is the frame that decided to allocate.
"""

import argparse
import bisect
import re
import subprocess
from collections import defaultdict

GiB = float(1 << 30)
ALLOC_NOISE = re.compile(
    r"^(malloc|calloc|realloc|posix_memalign|aligned_alloc|memalign|mmap|"
    r"operator new.*|__libc_.*|_Znwm|_Znam|_ZnwmSt11align_val_t|record|-)$"
)


def parse(path):
    text = open(path).read()
    head, *_ = text.splitlines()
    fields = dict(re.findall(r"(\w+)=([\d.]+)", head))
    stacks = []
    for block in re.split(r"^S ", text, flags=re.M)[1:]:
        lines = block.splitlines()
        rec = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", lines[0])}
        frames = []
        for ln in lines[1:]:
            ln = ln.strip()
            if not ln.startswith("F "):
                continue
            m = re.match(r"F (\S+)\+0x([0-9a-f]+)(?: (\S+))?", ln)
            if m:
                frames.append((m.group(1), int(m.group(2), 16), m.group(3) or "-"))
            else:
                frames.append(("?", 0, ln[2:]))
        rec["frames"] = frames
        stacks.append(rec)
    return fields, stacks


class Symtab:
    def __init__(self, sym_map):
        self.sym_map = sym_map
        self.cache = {}

    def _load(self, module):
        path = self.sym_map.get(module, module)
        addrs, names = [], []
        try:
            out = subprocess.run(["nm", "-C", "--defined-only", "-n", path], capture_output=True,
                                 text=True, check=False).stdout
        except FileNotFoundError:
            out = ""
        for ln in out.splitlines():
            parts = ln.split(" ", 2)
            if len(parts) == 3 and parts[1] in "tTwW":
                addrs.append(int(parts[0], 16))
                names.append(parts[2])
        return addrs, names

    def name(self, module, off, dl_name):
        if module not in self.cache:
            self.cache[module] = self._load(module)
        addrs, names = self.cache[module]
        # return addresses point one past the call; step back into the caller
        i = bisect.bisect_right(addrs, off - 1) - 1
        if i >= 0:
            return names[i]
        return dl_name


def short(module):
    return module.rsplit("/", 1)[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("--sym", action="append", default=[], help="served_path=unstripped_path")
    ap.add_argument("--key", default="at_peak", choices=["at_peak", "live", "total"])
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--own", action="append", default=[], help="library basename prefix that owns a decision")
    args = ap.parse_args()
    own = args.own or ["libopenvino_intel_gpu_plugin", "libopenvino.so", "arcint"]
    sym_map = dict(s.split("=", 1) for s in args.sym)
    st = Symtab(sym_map)
    fields, stacks = parse(args.dump)
    print(f"# {args.dump}: live {int(fields['live'])/GiB:.3f} GiB, peak {int(fields['peak'])/GiB:.3f} GiB "
          f"at t={fields.get('t_peak')} s, stacks {fields['stacks']}, dropped {fields['dropped']}")
    ranked = sorted(stacks, key=lambda s: s[args.key], reverse=True)
    rollup = defaultdict(lambda: [0, 0, 0])
    for s in stacks:
        owner = "?"
        for mod, off, dl in s["frames"]:
            if any(short(mod).startswith(o) for o in own):
                owner = f"{short(mod)}:{st.name(mod, off, dl)}"
                break
        r = rollup[owner]
        r[0] += s["at_peak"]
        r[1] += s["live"]
        r[2] += s["total"]
    print(f"\n## roll-up by first own frame (at_peak / live / total, GiB)")
    for owner, (p, l, t) in sorted(rollup.items(), key=lambda kv: kv[1][0], reverse=True)[: args.top]:
        if p == 0 and l == 0 and t < (1 << 30):
            continue
        print(f"{p/GiB:9.3f} {l/GiB:9.3f} {t/GiB:9.3f}  {owner[:200]}")
    print(f"\n## top stacks by {args.key}")
    for s in ranked[: args.top]:
        if s[args.key] == 0:
            break
        print(f"\nS at_peak {s['at_peak']/GiB:.3f} live {s['live']/GiB:.3f} total {s['total']/GiB:.3f} GiB "
              f"count {s['count']} max {s['max']/(1<<20):.1f} MiB")
        shown = 0
        for mod, off, dl in s["frames"]:
            nm = st.name(mod, off, dl)
            if shown == 0 and ALLOC_NOISE.match(nm.split("(")[0]):
                continue
            print(f"    {short(mod)}+{off:#x} {nm[:180]}")
            shown += 1
            if shown >= args.frames:
                break


if __name__ == "__main__":
    main()
