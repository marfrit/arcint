#!/usr/bin/env python3
"""Offline scoring and a linear refit of the MTP draft head from a capture.

Campaign mtp-head-fit, experiment E1. The capture is arcint's own
(ARCINT_MTP_CAPTURE, src/exec/mtp_capture.h, off by default): for every
position of a replayed text the target's row h and the head's row z, both
post-norm vectors that the output head multiplies and that both predict the
token one position on. tools/mtp_capture_replay.py drives the replay and writes
the record index this tool reads.

    mtp_fit.py eval   --capture BASE --records OUT.idx --head GGUF --vocab IDS
    mtp_fit.py linear --capture BASE --records OUT.idx --head GGUF --vocab IDS
    mtp_fit.py gguf-info FILE [--tensor NAME]
    mtp_fit.py rows-check --head SERVED.gguf --other F16.gguf [--vocab IDS] [--tensor NAME]

`eval`, per text class on one split (default: the held-out test split):

  * top-1 of the head (argmax of W_d z) against the actual next token and
    against the target's argmax (argmax of W_d h), and the target's own top-1
    against the actual token as the sanity line: if h were not the vector the
    output head multiplies, it would collapse;
  * the same two at simulated round starts (a greedy chain of --width drafts
    a round; a position starts a round after a rejection or after the width's
    drafts), with kept-given-reached by depth and tokens a verify, printed
    beside the served first-draft table of the campaign document, which the
    first-draft column must reproduce (the check that this reimplementation is
    right). Only depth 1 is captured: a deeper draft of the served chain takes
    the head's own row as input, the capture holds the target's, so depth 2 and
    on are the head with a teacher-forced input, an upper bound, and are
    printed as such;
  * E[max p_T] at the deployed sampling (top-k, then temperature, then top-p,
    the order of src/core/sampler.cpp; the draft vocabulary stands for the
    vocabulary, where the top-k of the target almost always lies) and
    E[p_T(head's pick)], the expected first-draft acceptance when the walk
    samples;
  * --shift K pairs the head's row of position p + K with position p's target
    row and token: the red case. Top-1 must collapse; it is a flag so a
    capture whose rows are misaligned by one position is told apart from a
    head that is bad.

`linear` fits H ~ Z A in closed form, A = (Z'Z + lambda I)^-1 Z'H over the
training split (all classes pooled; row convention, so the fitted head is
logits = z A W_d^T, i.e. W' = W_d A^T for column vectors), lambda chosen on the
validation split by KL, at several training sizes (a learning curve), and
reports top-1 and KL(p_T || p_head) over the draft vocabulary per class on the
training rows and on the held-out test rows, with the unfitted head (A = I) as
the baseline.

The split holds out by TASK, never by answer: the corpus crosses the same
tasks with several languages, so answers of one task share their content. A
task is the question with its language removed (code class); the split is
stratified by class, ordered by a hash of the task so it is the same on every
run and every host, and independent of the order of the records.

W_d, the draft head's rows, is read from the GGUF's output.weight for the
--llama-mtp-vocab ids (sorted, unique, as the drafter keeps them) in the
file's own type, as src/exec/llama_spec.cpp DraftHead does; F32, F16, BF16,
Q8_0, Q4_0, Q4_K and Q6_K are decoded here. `rows-check` compares a served
file's rows with another file's (the F16 coder) so the decoder is itself
checked on the real tensor. numpy is imported where it is needed only: the
split, the task key, the chain simulation and the format are pure Python.

The corpus, the captured rows and any fitted matrix are private data and are
never committed: this file is the method.
"""
import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys

HEADER_BYTES = 64
MAGIC = b"ARCMTPC1"
KIND_NAMES = {1: "h", 2: "z", 3: "tok"}

# the campaign document's served first-draft table (greedy, 4 drafts a verify,
# the coder's file, arcint 1.5.4 + 0101): kept given reached by draft position,
# and tokens a verify
SERVED = {
    "code": {"kept": [0.90, 0.81, 0.78, 0.68], "tokens": 3.59},
    "think": {"kept": [0.80, 0.74, 0.75, 0.72], "tokens": 3.15},
    "prose": {"kept": [0.69, 0.56, 0.53, 0.43], "tokens": 2.36},
}

# deployed sampling: temperature, top-p, top-k (the coder unit's flags)
SAMPLING_THINK = (0.6, 0.95, 20)
SAMPLING_OTHER = (0.7, 0.8, 20)

SPLITS = ("train", "val", "test")


def _np():
    import numpy as np  # numpy only where numbers are computed
    return np


# --------------------------------------------------------------- the format

def parse_header(b):
    """The 64-byte header of a capture file: a dict, or ValueError."""
    if len(b) < HEADER_BYTES or bytes(b[:8]) != MAGIC:
        raise ValueError("not an ARCINT_MTP_CAPTURE file (magic)")
    version, kind, dim, dtype, count = struct.unpack("<IIIIQ", bytes(b[8:32]))
    if version != 1:
        raise ValueError("capture format version %d, this tool reads 1" % version)
    if kind not in KIND_NAMES:
        raise ValueError("capture file kind %d" % kind)
    return {"kind": kind, "dim": dim, "dtype": dtype, "count": count}


def render_header(kind, dim, dtype, count):
    b = MAGIC + struct.pack("<IIIIQ", 1, kind, dim, dtype, count)
    return b + b"\0" * (HEADER_BYTES - len(b))


class Capture:
    """The three files of a capture, as memmaps. h and z: f32 [n, dim];
    req, pos, tok: i32 [n]. Only the first `count` records are valid; a file
    read while the server still runs is cut there."""

    def __init__(self, base):
        np = _np()
        self.base = base
        parts = {}
        for ext, kind in (("h", 1), ("z", 2), ("tok", 3)):
            path = "%s.%s" % (base, ext)
            with open(path, "rb") as f:
                head = parse_header(f.read(HEADER_BYTES))
            if head["kind"] != kind:
                raise ValueError("%s: kind %d, expected %d" % (path, head["kind"], kind))
            size = os.path.getsize(path)
            need = HEADER_BYTES + head["count"] * head["dim"] * 4
            if size < need:
                raise ValueError("%s: %d bytes, the header counts %d records (%d bytes)" % (path, size, head["count"], need))
            parts[ext] = (path, head)
        n = {parts[e][1]["count"] for e in parts}
        if len(n) != 1:
            raise ValueError("the three files hold different record counts: %s" % sorted(n))
        self.n = n.pop()
        self.dim = parts["h"][1]["dim"]
        if parts["z"][1]["dim"] != self.dim:
            raise ValueError("h and z rows differ in width")
        if parts["tok"][1]["dim"] != 3:
            raise ValueError("the token index is not 3 values a record")
        if self.n == 0:
            raise ValueError("the capture holds no records")
        self.h = np.memmap(parts["h"][0], dtype="<f4", mode="r", offset=HEADER_BYTES, shape=(self.n, self.dim))
        self.z = np.memmap(parts["z"][0], dtype="<f4", mode="r", offset=HEADER_BYTES, shape=(self.n, self.dim))
        t = np.memmap(parts["tok"][0], dtype="<i4", mode="r", offset=HEADER_BYTES, shape=(self.n, 3))
        self.req, self.pos, self.tok = t[:, 0], t[:, 1], t[:, 2]


# --------------------------------------------------------- tasks and splits

_LANG = re.compile(r"\s+in\s+[A-Z][\w#+.\-]*(?:\s+[A-Z][\w#+.\-]*)*(?=\s*[,.](?:\s|$))")


def task_key(q, cls):
    """The task of a question. The code class crosses each task with several
    languages (' ... in TypeScript, and explain ...' / ' ... in C#. Only the
    code.'): the key is the question up to its last ' in <Language>', so every
    language and every wording of the same task falls in one split. The other
    classes' questions are their own keys."""
    if cls == "code":
        last = None
        for m in _LANG.finditer(q):
            last = m
        if last is not None:
            return q[:last.start()].strip()
    return q.strip()


def assign_splits(keys, seed="mtpfit1", test_frac=0.2, val_frac=0.15):
    """{key: 'train'|'val'|'test'} for the distinct task keys of ONE class.
    Ordered by sha256(seed|key), the first n_test to test, the next n_val to
    val, the rest to train; at least one task each while three or more exist.
    Deterministic: no random state, no dependence on the order of `keys`."""
    ks = sorted(set(keys), key=lambda k: (hashlib.sha256((seed + "|" + k).encode()).hexdigest(), k))
    n = len(ks)
    n_test = max(1, int(round(n * test_frac))) if n >= 3 else 0
    n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
    if n_test + n_val >= n and n >= 3:
        n_test, n_val = 1, 1
    out = {}
    for i, k in enumerate(ks):
        out[k] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def split_records(records, seed="mtpfit1", test_frac=0.2, val_frac=0.15):
    """Adds 'task_key' and 'split' to every record of the replay's index (each
    has 'kind', the corpus record's kind, 'cls', its text class, and 'task');
    the split is per kind, by task, so the thinking text and the answer of one
    corpus record always land together. Returns the same list."""
    by_kind = {}
    for r in records:
        r["task_key"] = task_key(r["task"], r["kind"])
        by_kind.setdefault(r["kind"], []).append(r["task_key"])
    maps = {k: assign_splits(ks, seed, test_frac, val_frac) for k, ks in by_kind.items()}
    for r in records:
        r["split"] = maps[r["kind"]][r["task_key"]]
    return records


# --------------------------------------------------- the round-start simulation

def simulate_rounds(match, width):
    """Greedy chain rounds over one text. match[p] says whether the head's
    draft from position p's row is right. A round starts at s; draft j (0-based)
    is kept while match[s + j] holds, up to `width` drafts; the verify adds one
    token, so a round yields accepted + 1 tokens and the next round starts at
    s + accepted + 1: a position is a round start after a rejection or after the
    width's drafts. The last position (no token after it) starts no round.
    Returns [(start, accepted)]."""
    out = []
    s = 0
    n = len(match)
    while s < n:
        a = 0
        while a < width and s + a < n and match[s + a]:
            a += 1
        out.append((s, a))
        s += a + 1
    return out


def chain_table(rounds, width):
    """Pooled over rounds [(start, accepted)]: {'rounds', 'kept': [kept given
    reached by depth 1..width], 'tokens': tokens a verify}."""
    n = len(rounds)
    kept, reached = [], []
    for j in range(1, width + 1):
        r = sum(1 for _, a in rounds if a >= j - 1)
        k = sum(1 for _, a in rounds if a >= j)
        reached.append(r)
        kept.append(k / r if r else float("nan"))
    tokens = sum(a + 1 for _, a in rounds) / n if n else float("nan")
    return {"rounds": n, "kept": kept, "reached": reached, "tokens": tokens}


# ------------------------------------------------------------------- GGUF

# ggml type id -> (name, values a block, bytes a block)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    30: ("BF16", 1, 2),
}


class _Reader:
    def __init__(self, f):
        self.f = f

    def u32(self): return struct.unpack("<I", self.f.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.f.read(8))[0]
    def s(self):
        n = self.u64()
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, t):
        f = self.f
        fixed = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if t in fixed:
            return struct.unpack(fixed[t], f.read(struct.calcsize(fixed[t])))[0]
        if t == 8:
            return self.s()
        if t == 9:
            et = self.u32()
            n = self.u64()
            return [self.value(et) for _ in range(n)]
        raise ValueError("gguf value type %d" % t)


def read_gguf_header(path):
    """{'kv': {...without large arrays...}, 'tensors': {name: (dims, type, offset)}, 'data': absolute data offset}."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("%s: not a GGUF file" % path)
        r = _Reader(f)
        version = r.u32()
        if version < 2:
            raise ValueError("GGUF version %d" % version)
        n_t, n_kv = r.u64(), r.u64()
        kv = {}
        for _ in range(n_kv):
            k = r.s()
            t = r.u32()
            if t == 9:  # an array: keep only the short ones (the vocabulary is read nowhere here)
                pos = f.tell()
                et = r.u32()
                n = r.u64()
                if n > 64:
                    f.seek(pos)
                    r.value(9)
                    kv[k] = "<array of %d>" % n
                    continue
                f.seek(pos)
            kv[k] = r.value(t)
        tensors = {}
        for _ in range(n_t):
            name = r.s()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            ty = r.u32()
            off = r.u64()
            tensors[name] = (dims, ty, off)
        align = int(kv.get("general.alignment", 32))
        pos = f.tell()
        data = (pos + align - 1) // align * align
    return {"kv": kv, "tensors": tensors, "data": data}


def find_gguf_tensor(path, name):
    """(file, header, dims, type, absolute offset) of `name` in `path` or, for a
    split GGUF (-00001-of-0000N), in the shard that holds it."""
    h = read_gguf_header(path)
    if name in h["tensors"]:
        d, t, o = h["tensors"][name]
        return path, h, d, t, h["data"] + o
    n = int(h["kv"].get("split.count", 1))
    m = re.search(r"-00001-of-(\d+)", path)
    if n > 1 and m:
        for k in range(2, n + 1):
            shard = path[:m.start()] + "-%05d-of-%s" % (k, m.group(1)) + path[m.end():]
            if os.path.exists(shard):
                hs = read_gguf_header(shard)
                if name in hs["tensors"]:
                    d, t, o = hs["tensors"][name]
                    return shard, hs, d, t, hs["data"] + o
    raise KeyError("no %s in %s" % (name, path))


def dequant_rows(raw, ty, ncols):
    """raw: uint8 array [nrows, row_bytes] -> float32 [nrows, ncols]."""
    np = _np()
    nrows = raw.shape[0]
    if ty not in GGML_TYPES:
        raise ValueError("ggml type %d is not known to this reader" % ty)
    name, blk, bsz = GGML_TYPES[ty]
    if ncols % blk or raw.shape[1] != ncols // blk * bsz:
        raise ValueError("%s row of %d bytes for %d columns" % (name, raw.shape[1], ncols))
    nb = ncols // blk
    if name == "F32":
        return raw.copy().view("<f4").reshape(nrows, ncols).astype(np.float32)
    if name == "F16":
        return raw.copy().view("<f2").reshape(nrows, ncols).astype(np.float32)
    if name == "BF16":
        u = raw.copy().view("<u2").reshape(nrows, ncols).astype(np.uint32) << 16
        return u.view("<f4").astype(np.float32)
    b = raw.reshape(nrows, nb, bsz)

    def f16(x):
        return np.ascontiguousarray(x).view("<f2").astype(np.float32)

    if name == "Q8_0":
        d = f16(b[:, :, 0:2])                                  # [r, nb, 1]
        q = np.ascontiguousarray(b[:, :, 2:34]).view(np.int8).astype(np.float32)
        return (q * d).reshape(nrows, ncols)
    if name == "Q4_0":
        d = f16(b[:, :, 0:2])
        qs = b[:, :, 2:18]
        lo = (qs & 0xF).astype(np.float32) - 8.0
        hi = (qs >> 4).astype(np.float32) - 8.0
        return (np.concatenate([lo, hi], axis=2) * d).reshape(nrows, ncols)
    if name == "Q4_K":
        d = f16(b[:, :, 0:2])
        dmin = f16(b[:, :, 2:4])
        sc = b[:, :, 4:16].astype(np.uint8)
        qs = b[:, :, 16:144]
        scl = np.empty(b.shape[:2] + (8,), np.float32)
        mn = np.empty(b.shape[:2] + (8,), np.float32)
        for j in range(8):
            if j < 4:
                scl[:, :, j] = sc[:, :, j] & 63
                mn[:, :, j] = sc[:, :, j + 4] & 63
            else:
                scl[:, :, j] = (sc[:, :, j + 4] & 0xF) | ((sc[:, :, j - 4] >> 6) << 4)
                mn[:, :, j] = (sc[:, :, j + 4] >> 4) | ((sc[:, :, j] >> 6) << 4)
        out = np.empty(b.shape[:2] + (256,), np.float32)
        for c in range(4):                                       # 64 values from 32 bytes: low then high nibbles
            q = qs[:, :, 32 * c:32 * c + 32]
            out[:, :, 64 * c:64 * c + 32] = (d * scl[:, :, 2 * c:2 * c + 1]) * (q & 0xF) - dmin * mn[:, :, 2 * c:2 * c + 1]
            out[:, :, 64 * c + 32:64 * c + 64] = (d * scl[:, :, 2 * c + 1:2 * c + 2]) * (q >> 4) - dmin * mn[:, :, 2 * c + 1:2 * c + 2]
        return out.reshape(nrows, ncols)
    if name == "Q6_K":
        ql = b[:, :, 0:128]
        qh = b[:, :, 128:192]
        sc = np.ascontiguousarray(b[:, :, 192:208]).view(np.int8).astype(np.float32)
        d = f16(b[:, :, 208:210])
        out = np.empty(b.shape[:2] + (256,), np.float32)
        for half in range(2):                                    # 128 values per half
            qlh = ql[:, :, 64 * half:64 * half + 64]
            qhh = qh[:, :, 32 * half:32 * half + 32]
            for l4 in range(4):
                if l4 == 0:
                    q = (qlh[:, :, 0:32] & 0xF) | (((qhh >> 0) & 3) << 4)
                elif l4 == 1:
                    q = (qlh[:, :, 32:64] & 0xF) | (((qhh >> 2) & 3) << 4)
                elif l4 == 2:
                    q = (qlh[:, :, 0:32] >> 4) | (((qhh >> 4) & 3) << 4)
                else:
                    q = (qlh[:, :, 32:64] >> 4) | (((qhh >> 6) & 3) << 4)
                q = q.astype(np.float32) - 32.0
                # 16 scales, one per 16 values: scale index = 8*half + 2*l4 + (l // 16)
                s0 = sc[:, :, 8 * half + 2 * l4:8 * half + 2 * l4 + 1]
                s1 = sc[:, :, 8 * half + 2 * l4 + 1:8 * half + 2 * l4 + 2]
                base = 128 * half + 32 * l4
                out[:, :, base:base + 16] = d * s0 * q[:, :, 0:16]
                out[:, :, base + 16:base + 32] = d * s1 * q[:, :, 16:32]
        return out.reshape(nrows, ncols)
    raise ValueError("%s is not decoded by this reader (take the F16 file)" % name)


def read_gguf_rows(path, name, ids, chunk=4096):
    """Rows `ids` (None: all) of the 2-D tensor `name`, float32 [len(ids), ncols],
    decoded from the file's own type."""
    np = _np()
    f, h, dims, ty, off = find_gguf_tensor(path, name)
    if len(dims) != 2:
        raise ValueError("%s is not a matrix" % name)
    ncols, nrows = dims[0], dims[1]
    bname, blk, bsz = GGML_TYPES.get(ty, ("?", 1, 0))
    rb = ncols // blk * bsz
    if ids is None:
        ids = np.arange(nrows)
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) and (ids.min() < 0 or ids.max() >= nrows):
        raise ValueError("a row id outside [0, %d)" % nrows)
    mm = np.memmap(f, dtype=np.uint8, mode="r", offset=off, shape=(nrows, rb))
    out = np.empty((len(ids), ncols), np.float32)
    for s in range(0, len(ids), chunk):
        sel = ids[s:s + chunk]
        out[s:s + len(sel)] = dequant_rows(np.asarray(mm[sel]), ty, ncols)
    return out


def load_vocab(path):
    """The draft vocabulary as the drafter keeps it: sorted, unique ids. A
    .json file is a JSON list of integers, anything else little-endian int32."""
    if path.endswith(".json"):
        with open(path) as f:
            ids = json.load(f)
    else:
        np = _np()
        ids = np.fromfile(path, dtype="<i4").tolist()
    return sorted(set(int(i) for i in ids))


# --------------------------------------------------------------- the dataset

def load_records(path):
    with open(path) as f:
        d = json.load(f)
    if d.get("format") != "mtp-capture-records-1":
        raise ValueError("%s: not a record index of tools/mtp_capture_replay.py" % path)
    return d


def request_spans(cap):
    """[(start, end)] row ranges of the capture's requests: a request starts
    where the position is 0 and its rows run in position order."""
    np = _np()
    pos = np.asarray(cap.pos)
    starts = np.flatnonzero(pos == 0)
    ends = list(starts[1:]) + [cap.n]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def resolve_rows(cap, records):
    """Finds each record's rows in the capture by the hash of its token ids (the
    replay stored it): sets 'row0' and 'n_rows' and returns the problems (empty:
    every record found and aligned). By content, not by counting requests, so a
    warm-up request, a retry or a second pass in the same capture does not shift
    anything; a request is checked to be positions 0..n-1 of one request. Equal
    texts take the captured requests in order."""
    np = _np()
    by_sha = {}
    bad = []
    for a, b in request_spans(cap):
        pos = np.asarray(cap.pos[a:b])
        req = np.asarray(cap.req[a:b])
        if not (pos == np.arange(b - a)).all() or not (req == req[0]).all():
            continue          # not one request in position order: no record can match it
        sha = hashlib.sha1(np.asarray(cap.tok[a:b]).astype("<i4").tobytes()).hexdigest()
        by_sha.setdefault(sha, []).append((a, b))
    for r in records:
        spans = by_sha.get(r["ids_sha1"])
        if not spans:
            bad.append("record %d (%s): no captured request holds its token ids" % (r["i"], r["cls"]))
            continue
        a, b = spans.pop(0)
        r["row0"], r["n_rows"] = a, b - a
    return bad


def usable(rec, skip_first, shift=0):
    """[a, b) rows of a record whose next token is known, minus the question
    context and the first `skip_first` answer positions; `shift` shortens the
    range for the red case."""
    a = rec["row0"] + rec.get("ctx", 0) + skip_first
    b = rec["row0"] + rec["n_rows"] - 1 - max(shift, 0)
    return a, max(a, b)


def gather(cap, recs, skip_first, shift=0, cap_rows=None):
    """Concatenated rows of the records: (Zrows, Hrows, y, seg, stride), y the
    actual next token id, seg the row count of each record. `cap_rows`: at most
    that many rows, an even stride over each record's rows (deterministic; the
    chain simulation needs stride 1)."""
    np = _np()
    spans = [usable(r, skip_first, shift) for r in recs]
    total = sum(b - a for a, b in spans)
    stride = 1
    if cap_rows is not None and total > cap_rows:
        stride = int(math.ceil(total / cap_rows))
    zs, hs, ys, seg = [], [], [], []
    for a, b in spans:
        idx = np.arange(a, b, stride)
        seg.append(len(idx))
        if not len(idx):
            continue
        zs.append(np.asarray(cap.z[idx + shift]))
        hs.append(np.asarray(cap.h[idx]))
        ys.append(np.asarray(cap.tok[idx + 1]))
    if not zs:
        d = cap.dim
        return np.zeros((0, d), np.float32), np.zeros((0, d), np.float32), np.zeros(0, np.int32), seg, stride
    return np.concatenate(zs), np.concatenate(hs), np.concatenate(ys).astype(np.int64), seg, stride


# ------------------------------------------------------- the row statistics

def sampled_stats(lh, k, temp, top_p, pick):
    """From the target's logits lh [c, V] (draft vocabulary): the deployed
    sampler's probabilities of src/core/sampler.cpp, top-k then temperature then
    top-p, renormalised over the kept set. Returns (pmax [c], p_pick [c]), the
    probability of the most likely token and of column pick[c]."""
    np = _np()
    c, v = lh.shape
    k = min(k, v)
    idx = np.argpartition(-lh, k - 1, axis=1)[:, :k]
    top = np.take_along_axis(lh, idx, 1)
    order = np.argsort(-top, axis=1)
    top = np.take_along_axis(top, order, 1)
    idx = np.take_along_axis(idx, order, 1)
    pr = np.exp((top - top[:, :1]).astype(np.float64) / temp)
    pr /= pr.sum(1, keepdims=True)
    keep = np.ones((c, k), bool)
    if top_p < 1.0:
        cum = np.cumsum(pr, 1)
        hit = cum >= top_p
        cut = np.where(hit.any(1), hit.argmax(1) + 1, k)          # at least one survives
        keep = np.arange(k)[None, :] < cut[:, None]
    renorm = (pr * keep).sum(1)
    pmax = pr[:, 0] / renorm
    ppick = ((pr * keep) * (idx == pick[:, None])).sum(1) / renorm
    return pmax, ppick


def row_stats(Z, H, W, sampling=None, want_kl=False, chunk=512):
    """Per row: dz (the head's argmax column of W), dh (the target's), and with
    `sampling` (temp, top_p, top_k) pmax and p_pick, with want_kl the
    KL(p_T || p_head) at temperature 1 over the draft vocabulary."""
    np = _np()
    n = len(Z)
    dz = np.empty(n, np.int64)
    dh = np.empty(n, np.int64)
    pmax = np.empty(n)
    ppick = np.empty(n)
    kl = np.empty(n)
    WT = W.T   # a view: BLAS takes the transposed operand as it is
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        lz = Z[s:e] @ WT
        lh = H[s:e] @ WT
        dz[s:e] = lz.argmax(1)
        dh[s:e] = lh.argmax(1)
        if sampling is not None:
            t, p, k = sampling
            pmax[s:e], ppick[s:e] = sampled_stats(lh, k, t, p, dz[s:e])
        if want_kl:
            a = lh.astype(np.float64)
            b = lz.astype(np.float64)
            a -= a.max(1, keepdims=True)
            b -= b.max(1, keepdims=True)
            la = a - np.log(np.exp(a).sum(1, keepdims=True))
            lb = b - np.log(np.exp(b).sum(1, keepdims=True))
            kl[s:e] = (np.exp(la) * (la - lb)).sum(1)
    return {"dz": dz, "dh": dh, "pmax": pmax, "ppick": ppick, "kl": kl}


def next_index(y, ids):
    """Column of the draft vocabulary for each actual next token id, -1 when the
    token is not in it (the head cannot say it)."""
    np = _np()
    ids = np.asarray(ids, dtype=np.int64)
    pos = np.searchsorted(ids, y)
    pos = np.minimum(pos, len(ids) - 1)
    return np.where(ids[pos] == y, pos, -1)


def summarize(st, yi, seg, width, sampling):
    """One class' numbers from its rows' statistics: top-1s, the chain
    simulation per record, the sampled expectations."""
    np = _np()
    n = len(yi)
    m_a = st["dz"] == yi                       # head against the actual token
    m_t = st["dz"] == st["dh"]                 # head against the target's argmax
    out = {
        "rows": int(n),
        "coverage": float((yi >= 0).mean()) if n else float("nan"),
        "top1_actual": float(m_a.mean()) if n else float("nan"),
        "top1_target": float(m_t.mean()) if n else float("nan"),
        "target_top1_actual": float((st["dh"] == yi).mean()) if n else float("nan"),
    }
    for tag, m in (("greedy", m_t), ("actual", m_a)):
        rounds = []
        o = 0
        for c in seg:
            rounds += [(o + s, a) for s, a in simulate_rounds(m[o:o + c].tolist(), width)]
            o += c
        tab = chain_table(rounds, width)
        starts = np.array([s for s, _ in rounds], dtype=np.int64)
        tab["start_top1_actual"] = float(m_a[starts].mean()) if len(starts) else float("nan")
        tab["start_top1_target"] = float(m_t[starts].mean()) if len(starts) else float("nan")
        out["rounds_" + tag] = tab
    if sampling is not None:
        out["E_max_pT"] = float(st["pmax"].mean()) if n else float("nan")
        out["E_pT_pick"] = float(st["ppick"].mean()) if n else float("nan")
    return out


def pct(x):
    return "  n/a" if x != x else "%5.1f" % (100 * x)


def print_eval(res, width, shift):
    print("eval: split=%s width=%d skip_first=%d shift=%d%s" % (res["split"], width, res["skip_first"], shift,
                                                               "   <<< RED CASE: head rows paired one position off" if shift else ""))
    for cls, r in sorted(res["classes"].items()):
        print("\n[%s]  rows %d, next token inside the draft vocabulary %s %%" % (cls, r["rows"], pct(r["coverage"]).strip()))
        print("  top-1 of the head: against the actual token %s %%, against the target's argmax %s %%" %
              (pct(r["top1_actual"]), pct(r["top1_target"])))
        print("  sanity: the target's own top-1 against the actual token %s %% (sampled text: well below 100, far above 0)" %
              pct(r["target_top1_actual"]))
        g = r["rounds_greedy"]
        print("  greedy chain, %d rounds, %.2f tokens a verify (served %s)" %
              (g["rounds"], g["tokens"], ("%.2f" % SERVED[cls]["tokens"]) if cls in SERVED else "n/a"))
        print("    kept given reached by depth: " + "  ".join(pct(k) for k in g["kept"]) + "   (reached: " +
              " ".join(str(x) for x in g["reached"]) + ")")
        if cls in SERVED:
            print("    served table (campaign):     " + "  ".join(pct(k) for k in SERVED[cls]["kept"]))
            d1 = 100 * (g["kept"][0] - SERVED[cls]["kept"][0])
            flag = "ok" if abs(d1) <= 5 else "CHECK: the reimplementation or the text differs"
            print("    first draft vs served: %+.1f points (%s); depths 2+ use the target's row as input: an upper bound" % (d1, flag))
        print("    at round starts: head against target argmax %s %%, against actual %s %%" %
              (pct(g["start_top1_target"]), pct(g["start_top1_actual"])))
        a = r["rounds_actual"]
        print("  chain against the actual token: first draft %s %%, %.2f tokens a verify" % (pct(a["kept"][0]), a["tokens"]))
        if "E_max_pT" in r:
            print("  deployed sampling (T=%g, top-p %g, top-k %d): E[max p_T] %s %%, E[p_T(head's pick)] %s %%" %
                  (res["sampling"][cls][0], res["sampling"][cls][1], res["sampling"][cls][2], pct(r["E_max_pT"]), pct(r["E_pT_pick"])))


# --------------------------------------------------------------------- eval

def class_sampling(cls, args):
    return args.sampling_think if cls == "think" else args.sampling_other


def load_w(args):
    np = _np()
    ids = load_vocab(args.vocab)
    if args.w_npy:
        W = np.load(args.w_npy).astype(np.float32)
        if W.shape[0] != len(ids):
            raise ValueError("--w-npy has %d rows, the vocabulary %d ids" % (W.shape[0], len(ids)))
    else:
        W = read_gguf_rows(args.head, "output.weight", ids)
    return ids, W


def pick_records(index, split):
    recs = [r for r in index["records"] if split == "all" or r["split"] == split]
    by = {}
    for r in recs:
        by.setdefault(r["cls"], []).append(r)
    return by


def prepare(args):
    cap = Capture(args.capture)
    index = load_records(args.records)
    split_records(index["records"], args.seed, args.test_frac, args.val_frac)
    bad = resolve_rows(cap, index["records"])
    if bad:
        raise SystemExit("rows and tokens do not align with the replay's record index:\n  " +
                         "\n  ".join(bad[:10]) + ("\n  ... %d more" % (len(bad) - 10) if len(bad) > 10 else ""))
    return cap, index


def cmd_eval(args):
    np = _np()
    cap, index = prepare(args)
    ids, W = load_w(args)
    by = pick_records(index, args.split)
    res = {"split": args.split, "skip_first": args.skip_first, "shift": args.shift, "classes": {}, "sampling": {}}
    for cls, recs in sorted(by.items()):
        samp = class_sampling(cls, args)
        res["sampling"][cls] = samp
        Z, H, y, seg, stride = gather(cap, recs, args.skip_first, args.shift)
        if not len(Z):
            continue
        st = row_stats(Z, H, W, samp)
        yi = next_index(y, ids)
        res["classes"][cls] = summarize(st, yi, seg, args.width, samp)
        res["classes"][cls]["stride"] = stride
    print_eval(res, args.width, args.shift)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


# ------------------------------------------------------------------- linear

def train_order(recs, seed):
    return sorted(recs, key=lambda r: hashlib.sha256((seed + "|train|" + str(r["i"])).encode()).hexdigest())


def cmd_linear(args):
    np = _np()
    cap, index = prepare(args)
    ids, W = load_w(args)
    d = cap.dim
    sizes = [int(s) for s in args.sizes.split(",")]
    trains = train_order([r for rs in pick_records(index, "train").values() for r in rs], args.seed)
    # pooled over classes, ordered by the hash of the record: the curve's subsets nest
    if not trains:
        raise SystemExit("no training records")
    val = [r for rs in pick_records(index, "val").values() for r in rs]
    tests = pick_records(index, "test")
    lam_c = [float(x) for x in args.lambdas.split(",")]

    # Gram matrices at the sizes, accumulated over the nested subsets of the training rows
    G = np.zeros((d, d))
    C = np.zeros((d, d))
    rows_seen = 0
    snaps = {}
    want = sorted(set(sizes))

    def add(s0, e0):
        nonlocal rows_seen
        Zc = np.asarray(cap.z[s0:e0]).astype(np.float64)
        Hc = np.asarray(cap.h[s0:e0]).astype(np.float64)
        G.__iadd__(Zc.T @ Zc)
        C.__iadd__(Zc.T @ Hc)
        rows_seen += e0 - s0

    for r in trains:
        a0, b0 = usable(r, args.skip_first)
        s0 = a0
        while s0 < b0:
            e0 = min(b0, s0 + 4096)
            while want and rows_seen + (e0 - s0) >= want[0]:
                take = want[0] - rows_seen
                if take > 0:
                    add(s0, s0 + take)
                    s0 += take
                snaps[want[0]] = (G.copy(), C.copy(), rows_seen)
                want.pop(0)
            if e0 > s0:
                add(s0, e0)
            s0 = e0
    avail = rows_seen
    # sizes the data cannot reach collapse to what exists
    if avail not in [v[2] for v in snaps.values()]:
        snaps[avail] = (G.copy(), C.copy(), avail)
    print("linear: %d training rows available, curve at %s" % (avail, ",".join(str(v[2]) for _, v in sorted(snaps.items()))))

    Zv, Hv, yv, segv, _ = gather(cap, val, args.skip_first, 0, args.val_rows)
    base_v = row_stats(Zv, Hv, W, None, True)
    print("validation: %d rows, baseline (unfitted head) KL %.4f, top-1 against target %.3f" %
          (len(Zv), base_v["kl"].mean(), (base_v["dz"] == base_v["dh"]).mean()))

    def evaluate(A, recs, cap_rows):
        Z, H, y, seg, _ = gather(cap, recs, args.skip_first, 0, cap_rows)
        Zf = (Z @ A.astype(np.float32)) if A is not None else Z
        st = row_stats(Zf, H, W, None, True)
        yi = next_index(y, ids)
        return {"rows": len(Z), "top1_actual": float((st["dz"] == yi).mean()), "top1_target": float((st["dz"] == st["dh"]).mean()),
                "target_top1_actual": float((st["dh"] == yi).mean()), "kl": float(st["kl"].mean())}

    report = {"avail": avail, "baseline": {}, "curve": []}
    classes = sorted(set(r["cls"] for r in index["records"]))
    train_by = pick_records(index, "train")
    for cls in classes:
        report["baseline"][cls] = {"train": evaluate(None, train_by.get(cls, []), args.eval_rows) if cls in train_by else None,
                                   "test": evaluate(None, tests.get(cls, []), args.eval_rows) if cls in tests else None}
    last_A = None
    for key in sorted(snaps):
        G, C, n_rows = snaps[key]
        e, Q = np.linalg.eigh(G)
        QtC = Q.T @ C
        scale = float(np.mean(np.diag(G)))
        best = None
        tried = []
        for c in lam_c:
            lam = c * scale
            A = Q @ (QtC / (e + lam)[:, None])
            Zf = Zv @ A.astype(np.float32)
            st = row_stats(Zf, Hv, W, None, True)
            kl = float(st["kl"].mean())
            tried.append((c, kl, float((st["dz"] == st["dh"]).mean())))
            if best is None or kl < best[0]:
                best = (kl, c, A)
        kl_b, c_b, A = best
        last_A = A
        print("\nsize %d rows: lambda = %g x mean(diag Z'Z) chosen on validation (KL %.4f); tried %s" %
              (n_rows, c_b, kl_b, "  ".join("%g:%.4f" % (c, k) for c, k, _ in tried)))
        entry = {"rows": n_rows, "lambda_rel": c_b, "val_kl": kl_b, "classes": {}}
        for cls in classes:
            tr = evaluate(A, train_by.get(cls, []), args.eval_rows) if cls in train_by else None
            te = evaluate(A, tests.get(cls, []), args.eval_rows) if cls in tests else None
            entry["classes"][cls] = {"train": tr, "test": te}
            b = report["baseline"][cls]
            print("  [%s]" % cls)
            for tag, now, was in (("train", tr, b["train"]), ("held-out", te, b["test"])):
                if now is None:
                    continue
                print("    %-8s rows %5d  top-1 actual %s -> %s %%   top-1 target %s -> %s %%   KL %.4f -> %.4f" %
                      (tag, now["rows"], pct(was["top1_actual"]), pct(now["top1_actual"]), pct(was["top1_target"]),
                       pct(now["top1_target"]), was["kl"], now["kl"]))
        report["curve"].append(entry)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    if args.save_a and last_A is not None:
        np.save(args.save_a, last_A.astype(np.float32))
        print("\nA of the largest size saved to %s (private data: never commit it)" % args.save_a)
    return 0


# ---------------------------------------------------------- gguf-info / rows

def distill_grad(Z, H, W, A, tau, chunk=512):
    """Loss and d(loss)/dA of KL(q || softmax(Z A W^T)), q the target's softmax sharpened by tau
    (1: the target's distribution; small: its argmax). Row convention: a row's logits are z A W^T."""
    np = _np()
    WT = W.T
    G = np.zeros_like(A, dtype=np.float64)
    loss = 0.0
    n = len(Z)
    for s0 in range(0, n, chunk):
        z, h = Z[s0:s0 + chunk], H[s0:s0 + chunk]
        lp = (z @ A) @ WT
        lq = (h @ WT) / tau
        lp = lp - lp.max(1, keepdims=True)
        lq = lq - lq.max(1, keepdims=True)
        p_ = np.exp(lp)
        p_ /= p_.sum(1, keepdims=True)
        q = np.exp(lq)
        q /= q.sum(1, keepdims=True)
        loss += float((q * (np.log(np.maximum(q, 1e-30)) - np.log(np.maximum(p_, 1e-30)))).sum())
        G += z.T @ ((p_ - q) @ W)
    return loss / n, G / n


def fit_distill(Z, H, W, Zv, Hv, steps, batch, lr, tau, every, seed=1, log=None):
    """Adam on A from the identity, keeping the A with the best top-1 against the target's argmax on
    the validation rows; the identity competes, so the fit can only match or beat the head there."""
    np = _np()
    d = Z.shape[1]
    A = np.eye(d, dtype=np.float32)
    best, best_step = A.copy(), 0
    base = row_stats(Zv, Hv, W)
    best_top = float((base["dz"] == base["dh"]).mean())
    m1 = np.zeros((d, d))
    m2 = np.zeros((d, d))
    rng = np.random.RandomState(seed)
    for t in range(1, steps + 1):
        idx = np.sort(rng.choice(len(Z), size=min(batch, len(Z)), replace=False))
        loss, g = distill_grad(Z[idx], H[idx], W, A, tau)
        m1 = 0.9 * m1 + 0.1 * g
        m2 = 0.999 * m2 + 0.001 * g * g
        A = A - (lr * (m1 / (1 - 0.9 ** t)) / (np.sqrt(m2 / (1 - 0.999 ** t)) + 1e-8)).astype(np.float32)
        if t % every == 0 or t == steps:
            st = row_stats(Zv @ A, Hv, W)
            top = float((st["dz"] == st["dh"]).mean())
            if log:
                log("  step %5d train KL %.4f, validation top-1 against the target %.2f %%%s" %
                    (t, loss, 100 * top, " (best)" if top > best_top else ""))
            if top > best_top:
                best, best_top, best_step = A.copy(), top, t
    return best, best_top, best_step


def cmd_distill(args):
    np = _np()
    cap, index = prepare(args)
    ids, W = load_w(args)
    W = W.astype(np.float32)
    trains = train_order([r for rs in pick_records(index, "train").values() for r in rs], args.seed)
    val = [r for rs in pick_records(index, "val").values() for r in rs]
    tests = pick_records(index, "test")
    if not trains:
        raise SystemExit("no training records")
    Z, H, _, _, _ = gather(cap, trains, args.skip_first, 0, args.train_rows)
    Zv, Hv, _, _, _ = gather(cap, val, args.skip_first, 0, args.val_rows)
    print("train %d rows, validation %d rows, lr %g, tau %g, %d steps of %d" % (len(Z), len(Zv), args.lr, args.tau, args.steps, args.batch), flush=True)
    A, top, step = fit_distill(Z.astype(np.float32), H.astype(np.float32), W, Zv.astype(np.float32), Hv.astype(np.float32),
                               args.steps, args.batch, args.lr, args.tau, args.every,
                               log=lambda m: print(m, flush=True))
    print("best validation top-1 against the target %.2f %% at step %d (0: the identity stayed best)" % (100 * top, step))
    print("|A - I| / |I| = %.4f" % (np.linalg.norm(A - np.eye(cap.dim, dtype=np.float32)) / np.sqrt(cap.dim)))
    report = {"tau": args.tau, "lr": args.lr, "best_step": step, "classes": {}}
    sampling = {}
    for cls, recs in sorted(tests.items()):
        z, h, y, _, _ = gather(cap, recs, args.skip_first)
        z = z.astype(np.float32)
        h = h.astype(np.float32)
        yi = next_index(y, ids)
        smp = class_sampling(cls, args)
        b = row_stats(z, h, W, smp, True)
        f = row_stats(z @ A, h, W, smp, True)
        row = {"rows": len(z)}
        for k, st in (("base", b), ("fit", f)):
            row[k] = {"top1_target": float((st["dz"] == st["dh"]).mean()), "top1_actual": float((st["dz"] == yi).mean()),
                      "kl": float(st["kl"].mean()), "p_pick": float(st["ppick"].mean())}
        row["pmax"] = float(b["pmax"].mean())
        report["classes"][cls] = row
        print("[%s] held out, %d rows: top-1 target %.1f -> %.1f %%, actual %.1f -> %.1f %%, KL %.4f -> %.4f, sampled pick %.1f -> %.1f %% (target's best %.1f)" %
              (cls, len(z), 100 * row["base"]["top1_target"], 100 * row["fit"]["top1_target"], 100 * row["base"]["top1_actual"], 100 * row["fit"]["top1_actual"],
               row["base"]["kl"], row["fit"]["kl"], 100 * row["base"]["p_pick"], 100 * row["fit"]["p_pick"], 100 * row["pmax"]), flush=True)
    if args.json:
        with open(args.json, "w") as f_:
            json.dump(report, f_, indent=1)
    if args.save_a:
        np.save(args.save_a, A)
    return 0


def cmd_gguf_info(args):
    h = read_gguf_header(args.file)
    names = [args.tensor] if args.tensor else [n for n in h["tensors"] if n.startswith("blk.4") or n in ("output.weight", "token_embd.weight", "output_norm.weight")]
    for n in sorted(names):
        if n not in h["tensors"]:
            print("%s: not in this file (a split GGUF keeps it in another shard)" % n)
            continue
        dims, ty, off = h["tensors"][n]
        print("%-48s %-6s %s" % (n, GGML_TYPES.get(ty, ("type%d" % ty,))[0], dims))
    return 0


def cmd_rows_check(args):
    """The decoder against another file's rows of the same tensor: output.weight
    at random draft-vocabulary ids, or another tensor (--tensor, e.g. a Q4_K
    attention matrix of the head) at random rows."""
    np = _np()
    rng = np.random.RandomState(1)
    if args.tensor == "output.weight":
        if not args.vocab:
            raise SystemExit("--vocab for output.weight")
        pool = load_vocab(args.vocab)
    else:
        _, _, dims, _, _ = find_gguf_tensor(args.head, args.tensor)
        pool = list(range(dims[1]))
    pick = sorted(rng.choice(pool, size=min(args.n, len(pool)), replace=False).tolist())
    a = read_gguf_rows(args.head, args.tensor, pick)
    b = read_gguf_rows(args.other, args.tensor, pick)
    rel = np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-12)
    ta = GGML_TYPES[find_gguf_tensor(args.head, args.tensor)[3]][0]
    tb = GGML_TYPES[find_gguf_tensor(args.other, args.tensor)[3]][0]
    print("%s, %d rows: relative L2 difference %s (%s) vs %s (%s): mean %.4f, max %.4f" %
          (args.tensor, len(pick), os.path.basename(args.head), ta, os.path.basename(args.other), tb, rel.mean(), rel.max()))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--capture", required=True, help="the capture's base path (ARCINT_MTP_CAPTURE)")
        p.add_argument("--records", required=True, help="the record index written by tools/mtp_capture_replay.py")
        p.add_argument("--head", help="the GGUF whose output.weight is the draft head's source")
        p.add_argument("--w-npy", help="W_d as a .npy [V', d] instead of --head (tests)")
        p.add_argument("--vocab", required=True, help="the --llama-mtp-vocab file")
        p.add_argument("--seed", default="mtpfit1", help="the split's seed (part of the data's identity)")
        p.add_argument("--test-frac", type=float, default=0.2)
        p.add_argument("--val-frac", type=float, default=0.15)
        p.add_argument("--skip-first", type=int, default=8, help="answer positions dropped at each text's start (no context yet)")
        p.add_argument("--json", help="also write the numbers here")

    pe = sub.add_parser("eval", help="top-1, round starts and sampled expectations per class")
    common(pe)
    pe.add_argument("--split", default="test", choices=SPLITS + ("all",))
    pe.add_argument("--width", type=int, default=4, help="drafts a verify")
    pe.add_argument("--shift", type=int, default=0, help="RED CASE: pair the head's row of position p+K with position p")
    pe.add_argument("--sampling-think", type=lambda s: tuple(float(x) if i < 2 else int(x) for i, x in enumerate(s.split(","))),
                    default=SAMPLING_THINK, help="temperature,top-p,top-k of the thinking class")
    pe.add_argument("--sampling-other", type=lambda s: tuple(float(x) if i < 2 else int(x) for i, x in enumerate(s.split(","))),
                    default=SAMPLING_OTHER, help="temperature,top-p,top-k of the other classes")
    pe.set_defaults(fn=cmd_eval)

    pl = sub.add_parser("linear", help="closed-form A, learning curve")
    common(pl)
    pl.add_argument("--sizes", default="23000,46000,92000", help="training rows at each point of the curve")
    pl.add_argument("--lambdas", default="0.0001,0.001,0.01,0.1,1,10", help="lambda as a multiple of mean(diag Z'Z)")
    pl.add_argument("--val-rows", type=int, default=4096)
    pl.add_argument("--eval-rows", type=int, default=4096, help="rows a class and split the tables use")
    pl.add_argument("--save-a", help="save the largest size's A here (private)")
    pl.set_defaults(fn=cmd_linear)

    pd = sub.add_parser("distill", help="a correction fit to the target's distribution from the identity, stopped on validation")
    common(pd)
    pd.add_argument("--train-rows", type=int, default=None)
    pd.add_argument("--val-rows", type=int, default=4096)
    pd.add_argument("--steps", type=int, default=400)
    pd.add_argument("--batch", type=int, default=1024)
    pd.add_argument("--lr", type=float, default=1e-4)
    pd.add_argument("--tau", type=float, default=1.0, help="sharpens the target's distribution; small is its argmax")
    pd.add_argument("--every", type=int, default=50)
    pd.add_argument("--save-a", help="save the best A here (private)")
    pd.add_argument("--sampling-think", type=lambda s: tuple(float(x) if i < 2 else int(x) for i, x in enumerate(s.split(","))),
                    default=SAMPLING_THINK)
    pd.add_argument("--sampling-other", type=lambda s: tuple(float(x) if i < 2 else int(x) for i, x in enumerate(s.split(","))),
                    default=SAMPLING_OTHER)
    pd.set_defaults(fn=cmd_distill)

    pg = sub.add_parser("gguf-info", help="type and shape of the head's tensors (no numpy)")
    pg.add_argument("file")
    pg.add_argument("--tensor")
    pg.set_defaults(fn=cmd_gguf_info)

    pr = sub.add_parser("rows-check", help="the decoder against another file's rows of the same tensor")
    pr.add_argument("--head", required=True)
    pr.add_argument("--other", required=True)
    pr.add_argument("--vocab", help="the draft vocabulary (for output.weight)")
    pr.add_argument("--tensor", default="output.weight")
    pr.add_argument("--n", type=int, default=256)
    pr.set_defaults(fn=cmd_rows_check)

    args = ap.parse_args(argv)
    if args.cmd in ("eval", "linear", "distill") and not (args.head or args.w_npy):
        ap.error("--head or --w-npy")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
