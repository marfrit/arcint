#!/usr/bin/env python3
# Write a GGUF without its tokenizer.chat_template key, for the template-less "runs" start.
# The metadata is rewritten; the tensor data and its offsets are copied unchanged (offsets are
# relative to the data section). No model loader, no Python package: the GGUF header is parsed.
import struct
import sys


def rd_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n)


def skip_value(f, t):
    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in fixed:
        f.read(fixed[t])
    elif t == 8:
        rd_str(f)
    elif t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            skip_value(f, et)
    else:
        raise ValueError(f"GGUF value type {t}")


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: gguf_strip_chat_template.py IN.gguf OUT.gguf")
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit("not a GGUF")
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        keys = []
        for _ in range(n_kv):
            kstart = f.tell()
            name = rd_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            vstart = f.tell()
            skip_value(f, t)
            keys.append((name, kstart, vstart, f.tell()))
        tstart = f.tell()
        for _ in range(n_tensors):
            rd_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            f.read(8 * nd)
            f.read(4)
            f.read(8)
        tend = f.tell()
        data_start = (tend + 31) // 32 * 32
        hit = [k for k in keys if k[0] == b"tokenizer.chat_template"]
        if len(hit) != 1:
            sys.exit("expected exactly one tokenizer.chat_template key")
        _, kstart, _, vend = hit[0]
        f.seek(0)
        before = f.read(kstart)
        f.seek(vend)
        after = f.read(tend - vend)
        f.seek(data_start)
        data = f.read()
    out = bytearray()
    out += b"GGUF" + struct.pack("<I", version) + struct.pack("<Q", n_tensors) + struct.pack("<Q", n_kv - 1)
    out += before[24:] + after
    new_data = (len(out) + 31) // 32 * 32
    out += b"\0" * (new_data - len(out))
    out += data
    with open(dst, "wb") as f:
        f.write(out)
    print(f"{dst}: removed tokenizer.chat_template ({n_kv} -> {n_kv - 1} keys, {len(out)} bytes)")


if __name__ == "__main__":
    main()
