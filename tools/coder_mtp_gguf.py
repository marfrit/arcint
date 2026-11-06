#!/usr/bin/env python3
"""Head-only MTP GGUFs for the coder (campaign mtp-head-fit, experiment E2).

The arms differ only in the head file given to arcint through --llama-mtp-gguf;
the served path does not change. The loader takes a file as MTP-only when the
trunk's probe tensor (blk.0.attn_norm.weight) is absent, so a head file holds
the nextn layer (blk.<block_count-1>.*), the final norm and the token embedding
and LM head the draft reads, and the metadata of the full model (the
tokenizer included: the draft's vocabulary must equal the target's). No card,
no model load: files are read and written byte for byte.

  build      F16 head-only GGUF, from the coder's F16 GGUF (H0, H1, H2) or from
             the base model's safetensors (H3, the unpruned 256-expert head);
             --experts-used N sets expert_used_count (H2: 8, instead of 10).
  quantize   llama-quantize on that file with the rules of the coder's .ttf for
             the head's layer, no importance matrix (it has no entry for the
             head); refuses a float matrix no rule covers.
  splice     take (or replace) named tensors, raw, from another GGUF, e.g. the
             served token embedding and LM head, so every arm shares them.
  check      the file is head-only, names and shapes are the spec's.
  compare    tensor by tensor against another file: type, shape, sha256, and
             with --rel the relative difference of the decoded values.
  verify-base  H3's non-expert tensors and the kept experts against the coder's
             F16 (the pruned head's rows are the base's: E0), which also fixes
             the gate/up split and the norms' +1.

Tools and method are published; head files are private and never committed.
"""
import argparse
import hashlib
import os
import re
import struct
import subprocess
import sys
import tempfile

ALIGN = 32
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
_FIXED = {T_U8: 1, T_I8: 1, T_U16: 2, T_I16: 2, T_U32: 4, T_I32: 4, T_F32: 4, T_BOOL: 1, T_U64: 8, T_I64: 8, T_F64: 8}
_FMT = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I", T_I32: "<i", T_F32: "<f", T_BOOL: "<?",
        T_U64: "<Q", T_I64: "<q", T_F64: "<d"}
F32, F16 = 0, 1
# ggml type id -> (name, values a block, bytes a block)
GGML = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
        7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
        12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
        16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
        20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
        30: ("BF16", 1, 2)}
CHUNK = 64 << 20


class SpecError(Exception):
    """A tensor the tool cannot place, a shape or type that is not the spec's."""


def _np():
    import numpy
    return numpy


def nbytes(dims, ty):
    if ty not in GGML:
        raise SpecError("ggml type %d is not known to this tool" % ty)
    _, blk, bsz = GGML[ty]
    n = 1
    for d in dims:
        n *= d
    if dims[0] % blk:
        raise SpecError("row of %d values is not a multiple of the type's block %d" % (dims[0], blk))
    return n // blk * bsz


def norm_dims(d):
    """Trailing dimensions of 1 dropped (the F16 file writes [2048, 1] where the served file has [2048])."""
    d = list(d)
    while len(d) > 1 and d[-1] == 1:
        d.pop()
    return d


# ------------------------------------------------------------------- GGUF read


def _u32(f): return struct.unpack("<I", f.read(4))[0]
def _u64(f): return struct.unpack("<Q", f.read(8))[0]


def _str(f):
    return f.read(_u64(f)).decode("utf-8")


def _skip_value(f, t):
    if t in _FIXED:
        f.seek(_FIXED[t], 1)
    elif t == T_STR:
        f.seek(_u64(f), 1)
    elif t == T_ARR:
        et = _u32(f)
        n = _u64(f)
        if et in _FIXED:
            f.seek(_FIXED[et] * n, 1)
        else:
            for _ in range(n):
                _skip_value(f, et)
    else:
        raise SpecError("gguf value type %d" % t)


class Tensor:
    def __init__(self, name, dims, ty, chunks, size=None):
        self.name, self.dims, self.type = name, list(dims), ty
        self.chunks = chunks            # callable -> iterator of bytes-like
        self.size = nbytes(self.dims, ty) if size is None else size

    def data(self):
        return b"".join(bytes(c) for c in self.chunks())


class Gguf:
    """Header of a GGUF: the metadata as raw bytes (kept verbatim, the tokenizer included) and the tensor table."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise SpecError("%s: not a GGUF file" % path)
            self.version = _u32(f)
            if self.version < 2:
                raise SpecError("GGUF version %d" % self.version)
            n_t, n_kv = _u64(f), _u64(f)
            self.kv = []                # [key, type, raw value bytes]
            for _ in range(n_kv):
                key = _str(f)
                t = _u32(f)
                a = f.tell()
                _skip_value(f, t)
                b = f.tell()
                f.seek(a)
                self.kv.append([key, t, f.read(b - a)])
            self.infos = []             # (name, dims, type, offset)
            for _ in range(n_t):
                name = _str(f)
                nd = _u32(f)
                dims = [_u64(f) for _ in range(nd)]
                ty = _u32(f)
                off = _u64(f)
                self.infos.append((name, dims, ty, off))
            pos = f.tell()
        align = self.scalar("general.alignment", ALIGN)
        self.data_start = (pos + align - 1) // align * align
        self.names = [i[0] for i in self.infos]

    def scalar(self, key, default=None):
        for k, t, raw in self.kv:
            if k == key:
                if t not in _FMT:
                    raise SpecError("%s is not a scalar" % key)
                return struct.unpack(_FMT[t], raw)[0]
        if default is None:
            raise SpecError("%s: no metadata key %s" % (self.path, key))
        return default

    def info(self, name):
        for i in self.infos:
            if i[0] == name:
                return i
        raise SpecError("%s: no tensor %s" % (self.path, name))

    def tensor(self, name):
        _, dims, ty, off = self.info(name)
        size = nbytes(dims, ty)
        path, start = self.path, self.data_start + off

        def chunks():
            with open(path, "rb") as f:
                f.seek(start)
                left = size
                while left:
                    b = f.read(min(CHUNK, left))
                    if not b:
                        raise SpecError("%s: %s is cut short" % (path, name))
                    left -= len(b)
                    yield b
        return Tensor(name, dims, ty, chunks, size)


def set_u32(kv, key, value):
    for e in kv:
        if e[0] == key:
            if e[1] != T_U32:
                raise SpecError("%s is not a uint32 in the metadata" % key)
            e[2] = struct.pack("<I", value)
            return
    raise SpecError("no metadata key %s to set" % key)


# ------------------------------------------------------------------ GGUF write


def _wstr(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, kv, tensors):
    """kv: [key, type, raw]; tensors: [Tensor]. Written to path.part, then renamed."""
    names = [t.name for t in tensors]
    if len(set(names)) != len(names):
        raise SpecError("a tensor name twice in the output")
    head = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(kv))]
    for k, t, raw in kv:
        head += [_wstr(k), struct.pack("<I", t), raw]
    off = 0
    for t in tensors:
        head += [_wstr(t.name), struct.pack("<I", len(t.dims)), b"".join(struct.pack("<Q", d) for d in t.dims),
                 struct.pack("<I", t.type), struct.pack("<Q", off)]
        off += (t.size + ALIGN - 1) // ALIGN * ALIGN
    blob = b"".join(head)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(blob)
        f.write(b"\0" * (-len(blob) % ALIGN))
        for t in tensors:
            n = 0
            for c in t.chunks():
                f.write(c)
                n += len(c)
            if n != t.size:
                raise SpecError("%s: wrote %d bytes of %d" % (t.name, n, t.size))
            f.write(b"\0" * (-n % ALIGN))
    os.replace(tmp, path)


# ------------------------------------------------------------------------ spec


def head_layer(g):
    return g.scalar("qwen35moe.block_count") - g.scalar("qwen35moe.nextn_predict_layers")


def head_spec(g, n_expert):
    """{gguf name: (dims, 'mat' | 'f32')} of a head-only file: the nextn layer, the final norm, the embedding
    and LM head. The embedding and head are the file's own (vocabulary from the embedding's table)."""
    n = g.scalar("qwen35moe.embedding_length")
    hd = g.scalar("qwen35moe.attention.key_length")
    if g.scalar("qwen35moe.attention.value_length") != hd:
        raise SpecError("key and value lengths differ; the spec is for equal ones")
    hq = g.scalar("qwen35moe.attention.head_count") * hd
    hkv = g.scalar("qwen35moe.attention.head_count_kv") * hd
    ff = g.scalar("qwen35moe.expert_feed_forward_length")
    ffs = g.scalar("qwen35moe.expert_shared_feed_forward_length")
    b = "blk.%d." % head_layer(g)
    s = {
        b + "attn_norm.weight": ([n], "f32"),
        b + "post_attention_norm.weight": ([n], "f32"),
        b + "attn_q.weight": ([n, 2 * hq], "mat"),
        b + "attn_k.weight": ([n, hkv], "mat"),
        b + "attn_v.weight": ([n, hkv], "mat"),
        b + "attn_output.weight": ([hq, n], "mat"),
        b + "attn_q_norm.weight": ([hd], "f32"),
        b + "attn_k_norm.weight": ([hd], "f32"),
        b + "ffn_gate_inp.weight": ([n, n_expert], "f32"),
        b + "ffn_gate_exps.weight": ([n, ff, n_expert], "mat"),
        b + "ffn_up_exps.weight": ([n, ff, n_expert], "mat"),
        b + "ffn_down_exps.weight": ([ff, n, n_expert], "mat"),
        b + "ffn_gate_inp_shexp.weight": ([n], "f32"),
        b + "ffn_gate_shexp.weight": ([n, ffs], "mat"),
        b + "ffn_up_shexp.weight": ([n, ffs], "mat"),
        b + "ffn_down_shexp.weight": ([ffs, n], "mat"),
        b + "nextn.eh_proj.weight": ([2 * n, n], "mat"),
        b + "nextn.enorm.weight": ([n], "f32"),
        b + "nextn.hnorm.weight": ([n], "f32"),
        b + "nextn.shared_head_norm.weight": ([n], "f32"),
    }
    return s


TOP = ("token_embd.weight", "output.weight", "output_norm.weight")


def check_file(path, n_expert=None, floats_only=False):
    """Head-only and exactly the spec's names and shapes. Returns the Gguf."""
    g = Gguf(path)
    ne = g.scalar("qwen35moe.expert_count")
    if n_expert is not None and ne != n_expert:
        raise SpecError("%s: expert_count %d, expected %d" % (path, ne, n_expert))
    spec = head_spec(g, ne)
    layer = head_layer(g)
    for x in g.names:
        m = re.match(r"blk\.(\d+)\.", x)
        if m and int(m.group(1)) != layer:
            raise SpecError("%s: carries trunk tensor %s; the loader would take the file for a full model" % (path, x))
    seen = {}
    for name, dims, ty, _ in g.infos:
        if name in TOP:
            continue
        if name not in spec:
            raise SpecError("%s: tensor %s is not in the head's spec" % (path, name))
        seen[name] = 1
        if norm_dims(dims) != norm_dims(spec[name][0]):
            raise SpecError("%s: %s has shape %s, the spec says %s" % (path, name, dims, spec[name][0]))
        if spec[name][1] == "f32" and ty != F32:
            raise SpecError("%s: %s is %s, a norm or router is F32" % (path, name, GGML.get(ty, ("?",))[0]))
        if floats_only and spec[name][1] == "mat" and ty != F16:
            raise SpecError("%s: %s is %s, the quantizer's input is F16" % (path, name, GGML.get(ty, ("?",))[0]))
    miss = [x for x in spec if x not in seen]
    if miss:
        raise SpecError("%s: missing %s" % (path, ", ".join(miss)))
    return g


# ----------------------------------------------------------------------- build


def _kv_for_head(src, experts, experts_used):
    kv = [list(e) for e in src.kv]
    if experts is not None:
        set_u32(kv, "qwen35moe.expert_count", experts)
    if experts_used is not None:
        n = experts if experts is not None else src.scalar("qwen35moe.expert_count")
        if not 1 <= experts_used <= n:
            raise SpecError("expert_used_count %d outside 1..%d" % (experts_used, n))
        set_u32(kv, "qwen35moe.expert_used_count", experts_used)
    return kv


def build_from_gguf(src_path, out, experts_used=None):
    src = Gguf(src_path)
    ne = src.scalar("qwen35moe.expert_count")
    spec = head_spec(src, ne)
    layer = "blk.%d." % head_layer(src)
    own = [x for x in src.names if x.startswith(layer)]
    for x in own:
        if x not in spec:
            raise SpecError("%s: tensor %s cannot be placed (not in the head's spec)" % (src_path, x))
    ts = []
    for name, (dims, kind) in spec.items():
        if name not in own:
            raise SpecError("%s: tensor %s is missing" % (src_path, name))
        t = src.tensor(name)
        if norm_dims(t.dims) != norm_dims(dims):
            raise SpecError("%s: %s has shape %s, the spec says %s" % (src_path, name, t.dims, dims))
        want = F32 if kind == "f32" else F16
        if t.type != want:
            raise SpecError("%s: %s is %s, expected %s" % (src_path, name, GGML.get(t.type, ("?",))[0], GGML[want][0]))
        t.dims = list(dims)
        t.size = nbytes(t.dims, t.type)
        ts.append(t)
    write_gguf(out, _kv_for_head(src, None, experts_used), ts)


# safetensors -> gguf: (hf name under mtp., gguf suffix, kind)
ST_MAP = [
    ("fc.weight", "nextn.eh_proj.weight", "mat"),
    ("layers.0.input_layernorm.weight", "attn_norm.weight", "norm"),
    ("layers.0.post_attention_layernorm.weight", "post_attention_norm.weight", "norm"),
    ("layers.0.self_attn.q_proj.weight", "attn_q.weight", "mat"),
    ("layers.0.self_attn.k_proj.weight", "attn_k.weight", "mat"),
    ("layers.0.self_attn.v_proj.weight", "attn_v.weight", "mat"),
    ("layers.0.self_attn.o_proj.weight", "attn_output.weight", "mat"),
    ("layers.0.self_attn.q_norm.weight", "attn_q_norm.weight", "norm"),
    ("layers.0.self_attn.k_norm.weight", "attn_k_norm.weight", "norm"),
    ("layers.0.mlp.gate.weight", "ffn_gate_inp.weight", "f32"),
    ("layers.0.mlp.experts.down_proj", "ffn_down_exps.weight", "mat"),
    ("layers.0.mlp.shared_expert.gate_proj.weight", "ffn_gate_shexp.weight", "mat"),
    ("layers.0.mlp.shared_expert.up_proj.weight", "ffn_up_shexp.weight", "mat"),
    ("layers.0.mlp.shared_expert.down_proj.weight", "ffn_down_shexp.weight", "mat"),
    ("layers.0.mlp.shared_expert_gate.weight", "ffn_gate_inp_shexp.weight", "f32"),
    ("norm.weight", "nextn.shared_head_norm.weight", "norm"),
    ("pre_fc_norm_embedding.weight", "nextn.enorm.weight", "norm"),
    ("pre_fc_norm_hidden.weight", "nextn.hnorm.weight", "norm"),
]
ST_GATE_UP = "layers.0.mlp.experts.gate_up_proj"


class Safetensors:
    """Tensors of one or more shards: name -> (path, dtype, shape, absolute start, size)."""

    def __init__(self, paths):
        import json
        self.t = {}
        for p in paths:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            for k, v in h.items():
                if k == "__metadata__":
                    continue
                if k in self.t:
                    raise SpecError("%s: %s is in two shards" % (p, k))
                a, b = v["data_offsets"]
                self.t[k] = (p, v["dtype"], v["shape"], 8 + n + a, b - a)

    def array(self, name):
        np = _np()
        p, dt, shape, start, size = self.t[name]
        if dt != "BF16":
            raise SpecError("%s is %s; the tool reads the base's BF16" % (name, dt))
        return np.memmap(p, dtype=np.uint16, mode="r", offset=start, shape=tuple(shape))


def _bf16_f32(a):
    np = _np()
    return (np.ascontiguousarray(a).astype(np.uint32) << 16).view(np.float32)


def _to_f16_bytes(x, name):
    np = _np()
    if x.size and not np.isfinite(x).all():
        raise SpecError("%s: a non-finite value" % name)
    if x.size and np.abs(x).max() > 65504.0:
        raise SpecError("%s: a value outside the F16 range" % name)
    return x.astype("<f2").tobytes()


def build_from_safetensors(shards, kv_from, out, experts_used=8):
    st = Safetensors(shards)
    src = Gguf(kv_from)
    np = _np()
    pre = "mtp."
    extra = [k for k in st.t if k.startswith(pre) and k[len(pre):] not in {m[0] for m in ST_MAP} | {ST_GATE_UP}]
    if extra:
        raise SpecError("tensors %s cannot be placed (not in the head's map)" % ", ".join(sorted(extra)))
    need = [pre + m[0] for m in ST_MAP] + [pre + ST_GATE_UP]
    miss = [k for k in need if k not in st.t]
    if miss:
        raise SpecError("missing from the shards: %s" % ", ".join(miss))
    gate = st.t[pre + "layers.0.mlp.gate.weight"][2]
    ne = gate[0]
    spec = head_spec(src, ne)
    layer = "blk.%d." % head_layer(src)

    def slabs(name, fn):
        def chunks():
            a = st.array(name)
            step = 1 if a.ndim == 3 else max(1, 4096)
            for i in range(0, a.shape[0], step):
                yield fn(a[i:i + step])
        return chunks

    ts = {}
    for hf, suffix, kind in ST_MAP:
        name = layer + suffix
        dims, want = spec[name]
        shape = st.t[pre + hf][2]
        if norm_dims(list(reversed(shape))) != norm_dims(dims):
            raise SpecError("%s: shape %s does not fit %s %s" % (pre + hf, shape, name, dims))
        full = pre + hf
        if kind == "mat":
            ts[name] = Tensor(name, dims, F16, slabs(full, lambda a, n=name: _to_f16_bytes(_bf16_f32(a), n)))
        elif kind == "norm":      # the converter's RMS-norm form: the base's weight plus 1
            ts[name] = Tensor(name, dims, F32, slabs(full, lambda a: (_bf16_f32(a) + np.float32(1.0)).astype("<f4").tobytes()))
        else:
            ts[name] = Tensor(name, dims, F32, slabs(full, lambda a: _bf16_f32(a).astype("<f4").tobytes()))
    gu = st.t[pre + ST_GATE_UP][2]
    n_ff = src.scalar("qwen35moe.expert_feed_forward_length")
    n = src.scalar("qwen35moe.embedding_length")
    if gu != [ne, 2 * n_ff, n]:
        raise SpecError("%s: shape %s, expected %s" % (pre + ST_GATE_UP, gu, [ne, 2 * n_ff, n]))
    dn = st.t[pre + "layers.0.mlp.experts.down_proj"][2]
    if dn != [ne, n, n_ff]:
        raise SpecError("down_proj shape %s, expected %s" % (dn, [ne, n, n_ff]))
    for suffix, lo in (("ffn_gate_exps.weight", 0), ("ffn_up_exps.weight", n_ff)):
        name = layer + suffix
        ts[name] = Tensor(name, spec[name][0], F16,
                          slabs(pre + ST_GATE_UP, lambda a, lo=lo, n_=name: _to_f16_bytes(_bf16_f32(a[:, lo:lo + n_ff, :]), n_)))
    ordered = [ts[k] for k in spec]
    write_gguf(out, _kv_for_head(src, ne, experts_used), ordered)


# -------------------------------------------------------------- splice, quantize


def splice(into, donor, names, out):
    a, b = Gguf(into), Gguf(donor)
    ts = {n: a.tensor(n) for n in a.names}
    for n in names:
        t = b.tensor(n)
        if n in ts and norm_dims(ts[n].dims) != norm_dims(t.dims):
            raise SpecError("%s: shape %s in %s, %s in %s" % (n, ts[n].dims, into, t.dims, donor))
        ts[n] = t
    write_gguf(out, a.kv, list(ts.values()))


def quantize(inp, out, ttf, binary, threads=4):
    g = check_file(inp, floats_only=True)
    layer = head_layer(g)
    rules = [l.strip() for l in open(ttf) if l.strip()]
    mine = [l for l in rules if l.startswith("^blk\\.%d\\." % layer)]
    pats = [re.compile(l.rsplit("=", 1)[0]) for l in mine]
    for name, dims, ty, _ in g.infos:
        if ty == F16 and not any(p.search(name) for p in pats):
            raise SpecError("%s has no rule in %s; it would get the default type" % (name, ttf))
    if not os.path.exists(binary):
        raise SpecError("no llama-quantize at %s" % binary)
    with tempfile.NamedTemporaryFile("w", suffix=".ttf", delete=False) as f:
        f.write("\n".join(mine) + "\n")
        rf = f.name
    try:
        r = subprocess.run(["nice", "-n", "19", binary, "--tensor-type-file", rf, inp, out, "Q4_K_M", str(threads)],
                           capture_output=True, text=True)
    finally:
        os.unlink(rf)
    if r.returncode:
        raise SpecError("llama-quantize failed (%d): %s" % (r.returncode, (r.stdout + r.stderr)[-1500:]))
    return mine


# --------------------------------------------------------------------- compare


def _sha(t):
    h = hashlib.sha256()
    for c in t.chunks():
        h.update(c)
    return h.hexdigest()


def _decode_rows(t, lo, hi):
    """Decoded float32 rows [lo, hi) of the tensor viewed as [prod(dims[1:]), dims[0]]."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mtp_fit
    np = _np()
    cols = t.dims[0]
    _, blk, bsz = GGML[t.type]
    rb = cols // blk * bsz
    buf = bytearray()
    pos = 0
    for c in t.chunks():                    # a streaming slice of the rows
        a, b = max(lo * rb - pos, 0), min(hi * rb - pos, len(c))
        if b > a:
            buf += c[a:b]
        pos += len(c)
        if pos >= hi * rb:
            break
    raw = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(hi - lo, rb)
    return mtp_fit.dequant_rows(raw, t.type, cols)


def rel_diff(ta, tb):
    np = _np()
    rows = 1
    for d in ta.dims[1:]:
        rows *= d
    num = den = 0.0
    step = 8192
    for lo in range(0, rows, step):
        hi = min(rows, lo + step)
        x, y = _decode_rows(ta, lo, hi), _decode_rows(tb, lo, hi)
        num += float(((x - y).astype(np.float64) ** 2).sum())
        den += float((y.astype(np.float64) ** 2).sum())
    return (num / den) ** 0.5 if den else 0.0


def compare(pa, pb, prefix, rel, out=print):
    a, b = Gguf(pa), Gguf(pb)
    names = sorted({n for n in a.names + b.names if n.startswith(prefix)})
    same = 0
    rows = []
    for n in names:
        if n not in a.names or n not in b.names:
            rows.append((n, "only in " + (pa if n in a.names else pb)))
            continue
        ta, tb = a.tensor(n), b.tensor(n)
        if norm_dims(ta.dims) != norm_dims(tb.dims):
            rows.append((n, "shape %s vs %s" % (ta.dims, tb.dims)))
            continue
        if ta.type != tb.type:
            rows.append((n, "type %s vs %s" % (GGML[ta.type][0], GGML[tb.type][0])))
            continue
        if _sha(ta) == _sha(tb):
            same += 1
            rows.append((n, "identical %s" % GGML[ta.type][0]))
            continue
        msg = "DIFFERS %s" % GGML[ta.type][0]
        if rel:
            msg += " rel %.3g" % rel_diff(ta, tb)
        rows.append((n, msg))
    for n, m in rows:
        out("%-48s %s" % (n, m))
    out("%d of %d tensors identical" % (same, len(names)))
    return same, len(names)


def verify_base(h3, coder, out=print):
    """H3 (256 experts, F16) against the coder's F16 head: non-expert tensors, and the kept experts matched by their
    router rows. Returns the worst relative difference."""
    np = _np()
    a, c = Gguf(h3), Gguf(coder)
    layer = "blk.%d." % head_layer(c)
    gate = lambda g: np.frombuffer(g.tensor(layer + "ffn_gate_inp.weight").data(), "<f4")
    ga = gate(a).reshape(a.scalar("qwen35moe.expert_count"), -1)
    gc = gate(c).reshape(c.scalar("qwen35moe.expert_count"), -1)
    index = {ga[i].tobytes(): i for i in range(ga.shape[0])}
    kept = []
    for j in range(gc.shape[0]):
        if gc[j].tobytes() not in index:
            raise SpecError("coder router row %d is not a row of the base's router" % j)
        kept.append(index[gc[j].tobytes()])
    out("router rows matched: %d of %d kept, ascending %s" % (len(kept), ga.shape[0], kept == sorted(kept)))
    worst = 0.0
    for n in sorted(x for x in c.names if x.startswith(layer)):
        ta, tc = a.tensor(n), c.tensor(n)
        ty = "<f4" if tc.type == F32 else "<f2"
        xa = np.frombuffer(ta.data(), ty).astype(np.float64)
        xc = np.frombuffer(tc.data(), ty).astype(np.float64)
        if "_exps" in n:
            per = xa.size // ga.shape[0]
            xa = xa.reshape(ga.shape[0], per)[kept].ravel()
        if xa.shape != xc.shape:
            raise SpecError("%s: %s vs %s values" % (n, xa.shape, xc.shape))
        d = np.sqrt(((xa - xc) ** 2).sum() / max((xc ** 2).sum(), 1e-300))
        worst = max(worst, float(d))
        out("%-48s rel %.3g  equal %.4f" % (n, d, float((xa == xc).mean())))
    out("worst relative difference %.3g" % worst)
    return worst


# ------------------------------------------------------------------------- CLI


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="F16 head-only GGUF")
    p.add_argument("--out", required=True)
    p.add_argument("--from-gguf", help="the coder's F16 GGUF (H0, H1, H2)")
    p.add_argument("--from-safetensors", nargs="+", help="the base model's shards holding mtp.* (H3)")
    p.add_argument("--kv-from", help="metadata source for --from-safetensors (the coder's F16 GGUF)")
    p.add_argument("--experts-used", type=int)
    p = sub.add_parser("quantize", help="llama-quantize with the .ttf's rules for the head layer")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ttf", required=True)
    p.add_argument("--llama-quantize", required=True)
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("splice", help="take or replace raw tensors from another GGUF")
    p.add_argument("--into", required=True)
    p.add_argument("--from", dest="donor", required=True)
    p.add_argument("--tensors", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("check")
    p.add_argument("file")
    p.add_argument("--experts", type=int)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--prefix", default="blk.40.")
    p.add_argument("--rel", action="store_true")
    p = sub.add_parser("verify-base")
    p.add_argument("h3")
    p.add_argument("coder")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            if bool(a.from_gguf) == bool(a.from_safetensors):
                ap.error("exactly one of --from-gguf, --from-safetensors")
            if a.from_gguf:
                build_from_gguf(a.from_gguf, a.out, a.experts_used)
            else:
                if not a.kv_from:
                    ap.error("--from-safetensors needs --kv-from")
                build_from_safetensors(a.from_safetensors, a.kv_from, a.out, a.experts_used or 8)
            check_file(a.out)
        elif a.cmd == "quantize":
            quantize(a.inp, a.out, a.ttf, a.llama_quantize, a.threads)
            check_file(a.out)
        elif a.cmd == "splice":
            splice(a.into, a.donor, a.tensors, a.out)
        elif a.cmd == "check":
            g = check_file(a.file, a.experts)
            print("%s: head-only, %d tensors, expert_count %d, expert_used_count %d" % (
                a.file, len(g.names), g.scalar("qwen35moe.expert_count"), g.scalar("qwen35moe.expert_used_count")))
        elif a.cmd == "compare":
            same, n = compare(a.a, a.b, a.prefix, a.rel)
            return 0 if same == n else 1
        elif a.cmd == "verify-base":
            verify_base(a.h3, a.coder)
    except SpecError as e:
        print("coder_mtp_gguf: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
