#!/usr/bin/env python3
"""What the think-block mask will actually do on a given GGUF, without a GPU.

The mask blocks, by ID, the tokens that end generation and a second opening
think-tag, while a think block is open. Whether the opener can be blocked at
all depends on the vocabulary: a ChatML vocabulary holds the think-tags as
single added control tokens (so there is one logit to block), while a
vocabulary that spells the same tag as ordinary text has nothing to block and
is handled by matching the emitted text instead. Which one a served artifact
is is a fact about its file, and the campaign's design depends on it.

This reads it out of the GGUF's own metadata: the added-token list, the token
type table (gguf's CONTROL type), and the eos/eot ids. No tensor is read, so
it runs on a vocab-only file and on a laptop.

The tag strings are assembled from parts at run time, exactly as
tools/turn_defect_audit.py and tools/think_eog_probe.py do: a control token
written as a literal in a source file is the defect this campaign tracks. The
authoritative table is the C++ one, src/core/reasoning.cpp; this file only
needs the two shapes to ask the question.

    tools/think_vocab_probe.py <file.gguf> [<file2.gguf> ...]
"""
import struct
import sys

LT, GT, SLASH = chr(60), chr(62), chr(47)

# gguf's LLAMA_TOKEN_TYPE_CONTROL (gguf-py GGUFTokenype / llama.cpp's
# LLAMA_TOKEN_TYPE_CONTROL); the added markers carry it.
CONTROL = 3


def tag(word, closing=False):
    return LT + (SLASH if closing else "") + word + GT


def read_metadata(path):
    """GGUF v2/v3 metadata only: name -> value, tensors never touched."""
    TYPES = {0: "<B", 1: "<c", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
             6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    with open(path, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError("%s is not a GGUF file" % path)
        version = struct.unpack("<I", fh.read(4))[0]
        if version < 2:
            raise ValueError("%s: GGUF v%d is not readable" % (path, version))
        n_tensors = struct.unpack("<Q", fh.read(8))[0]
        n_kv = struct.unpack("<Q", fh.read(8))[0]

        def string():
            n = struct.unpack("<Q", fh.read(8))[0]
            return fh.read(n).decode("utf-8", "replace")

        def value(depth=0):
            t = struct.unpack("<I", fh.read(4))[0]
            if t == 8:
                return string()
            if t == 9:
                # An array declares its ELEMENT type once; the elements then
                # carry no type byte of their own. Reading one per element is
                # the classic way to desync a GGUF reader on the token_type
                # array, which is the one this probe needs.
                et = struct.unpack("<I", fh.read(4))[0]
                n = struct.unpack("<Q", fh.read(8))[0]
                if et == 8:
                    return [string() for _ in range(n)]
                if et in TYPES:
                    fmt = TYPES[et]
                    size = struct.calcsize(fmt)
                    return list(struct.unpack("<%d%s" % (n, fmt[1]),
                                              fh.read(size * n)))
                if depth > 1:
                    raise ValueError("nested array at depth %d" % depth)
                return [value(depth + 1) for _ in range(n)]
            if t in TYPES:
                return struct.unpack(TYPES[t], fh.read(struct.calcsize(TYPES[t])))[0]
            raise ValueError("unknown GGUF value type %d at byte %d" % (t, fh.tell() - 4))

        meta = {}
        for _ in range(n_kv):
            # Order matters: in `meta[string()] = value()` CPython evaluates the
            # RIGHT-HAND SIDE first, so value() would read the key's length as
            # a type byte and desync the whole stream. Read the key first.
            key = string()
            meta[key] = value()
        return meta, n_tensors


def report(path):
    meta, n_tensors = read_metadata(path)
    tokens = meta.get("tokenizer.ggml.tokens") or []
    types = meta.get("tokenizer.ggml.token_type") or []
    added = meta.get("tokenizer.ggml.added_tokens") or []
    added = [a.decode("utf-8", "replace") if isinstance(a, bytes) else a for a in added]
    index = {t: i for i, t in enumerate(tokens)}

    print("%s" % path)
    print("  architecture %s, %d tokens in the vocabulary, %d added, "
          "%d tensor(s) in the file"
          % (meta.get("general.architecture", "?"), len(tokens), len(added), n_tensors))

    for word in ("think", "thinking"):
        for closing in (False, True):
            t = tag(word, closing)
            where = index.get(t)
            kind = "-"
            if where is not None:
                kind = "CONTROL" if (where < len(types) and int(types[where]) == CONTROL) \
                    else "type %s" % types[where]
            single = where is not None and t in added
            print("  %-11s %-12s id=%-7s %s %s"
                  % (word + ("/" if closing else ""), "closer" if closing else "opener",
                     where, kind,
                     "ONE added token: blockable by id" if single
                     else "not one added token: matched as text"))

    eog = []
    for key in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"):
        v = meta.get(key)
        if v is not None:
            eog.append((key.split(".")[-1], int(v), tokens[int(v)] if int(v) < len(tokens) else "?"))
    print("  turn-ending ids the mask will block (arcint takes every id "
          "llama_vocab_is_eog answers for, which is this set plus any "
          "control-typed id it also marks):")
    for name, i, txt in eog:
        print("     %-4s id=%-7d %s" % (name, i, txt))
    if not eog:
        print("     none in the metadata: the loader decides")
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    rc = 0
    for path in argv[1:]:
        try:
            rc |= report(path)
        except Exception as exc:  # a bad file is a line, not a traceback
            print("%s: %s" % (path, exc))
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
