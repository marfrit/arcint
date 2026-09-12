"""THE SERVING-SHAPE IR: the full 48-layer qwen4_exp backbone at REAL geometry,
emitted as STRUCTURE, with no materialised expert constants.

This module exists to retire blocker (3) of `tools/export_qwen4_exp.py`'s
refusal, which reads:

    "(3) RESIDENCY, and only then: this emitter materialises every weight as an
     f32 ov Constant. Measured over the shipped tensor list at that assumption,
     the mapped set is 659.1 GiB ... No local card holds that and neither does
     the export host's RAM -- so full-size needs a different weight strategy
     (quantised constants, and a gather for the n-gram table), not a bigger
     window."

The refusal is right about the premise and the premise is a CHOICE. The
serving shape is the other choice, and it is not a smaller model: it is the
same graph with the weights carried the way the served runtime carries them.

--------------------------------------------------------------------------
1. WHY A STRUCTURE EMISSION IS A REAL ARTIFACT, NOT A MOCK
--------------------------------------------------------------------------

arcint's own load path sizes the expert slot pool from the IR without reading
one byte of weight data. `src/exec/backend_ov.cpp:552-556`:

    // M7 §2 Phase B, analytic route ...: sizes the expert slot pool from the
    // `read_model` the load path already holds, without materialising any
    // weight data -- a weightless IR (ov::weights_path) still carries every
    // constant's shape and element type in the XML.

So "shape and element type, no data" is not a degraded artifact from the
serving runtime's point of view -- for the slot-pool decision it is the WHOLE
artifact. This module emits exactly that, and the contract test
(`tests/python/test_serving_shape.py`) checks it against a Python transcription
of `slot_pool_from_ir` (`backend_ov.cpp:577-623`) rather than against a
description of it.

--------------------------------------------------------------------------
2. THE EXPERT-SLOT (OTD) CONTRACT, BOTH SIDES, CITED
--------------------------------------------------------------------------

C++ side, `src/exec/backend_ov.cpp:577-623` `slot_pool_from_ir`:

  * a MoE op is any node whose OpenVINO TYPE NAME contains "moe",
    case-insensitively                                    (backend_ov.cpp:581-585)
  * its expert-weight inputs are the Constant operands, or a Constant behind
    exactly ONE Convert, whose LEADING DIMENSION equals `num_expert`
                                                          (backend_ov.cpp:588-603)
  * per-expert bytes = product of dims[1:] x element_type().size()
                                                          (backend_ov.cpp:600-603)
  * an unmatched graph returns nullopt and the caller falls back to the
    plateau probe -- "this function never guesses"        (backend_ov.cpp:566-569)

Export side, the shape that was MEASURED to fuse on the card is the TILED
lowering, described in `tools/verify_moe_lowering.py:26-45` and emitted by
`tools/export_mtp.py:401-507` `moe_block_tiled`:

    Reshape(hidden -> [M,H]) -> Tile([E,1]) -> Reshape -> [E,M,H]
    expert weights as rank-4 [E, out, groups, group_size] Constants in a
    compressed integer type (u4/i4/u8/i8) behind
    Convert -> Subtract(zero_point) -> Multiply(scale) -> Reshape(4 -> 3)
    router at rank 2 over the same [M,H] flatten:
    Softmax -> TopK -> Divide -> Slice -> ScatterElementsUpdate -> ...

`verify_moe_lowering.py:33-42` records why the trailing Reshape is
load-bearing: an earlier flat rank-3 weight "passed every check in this file
(and CPU compiles) but crashed a real GPU compile inside the fusing pass's own
rewrite, because the pass's matcher anchors on that Reshape node."

This module emits that shape at real geometry (E=512, I=640, H=2560) with the
expert weights declared u4 and NEVER materialised.

--------------------------------------------------------------------------
3. HOW "NO MATERIALISED CONSTANTS" IS ACHIEVED -- and it is measured
--------------------------------------------------------------------------

`SparseArena` hands out numpy memmaps over ONE sparse file. A page that is
never written is never allocated -- on disk or in RAM. An ov Constant is then
built over that buffer with `shared_memory=True`, so OpenVINO wraps the
mapping instead of copying it.

Measured on the dev host before this module was written, one real-width expert
tensor:

    [512, 640, 2560] u4  ->  byte_size 419,430,400
    du -k of the backing file: 1 KiB      (apparent size 419,430,400)
    process maxrss: 47 MiB

That is the whole mechanism, and it is why a 48-layer real-geometry graph can
be built on a host with 48 GiB of RAM at all.

The emitters themselves are UNCHANGED. `shared_constants()` swaps the `_c`
constant factory that every q4e emitter imports from `q4e.gdn` for a
zero-copy one, for the duration of one build, and restores it in a finally.
Swapping the factory rather than editing six emitters keeps the graphs this
module emits byte-identical in STRUCTURE to the ones the parity suites gate.

--------------------------------------------------------------------------
4. WHAT THIS IS NOT
--------------------------------------------------------------------------

* It is not a numeric artifact. Every weight is an unwritten (zero) page.
  Nothing here may be used to make a parity claim, and the contract test
  asserts structure and shape only. The numeric gates live in the per-piece
  suites, on real fed tensors.
* It is not the paged serving graph. The served forward's port contract
  (`conv_state_table.N`, `gated_delta_state_table.N`, `key_cache.N`,
  `value_cache.N`, `position_ids`, `la.block_indices`,
  `la.block_indices_begins`, `la.past_lens`, `la.cache_interval` --
  `backend_ov.cpp:3191-3199` and `backend_ov.cpp:6141-6151`) is NOT emitted
  here; this is the static full-sequence shape the piecewise work validates.
  The contract test names that gap with a STRICT xfail so it fails loudly the
  day it closes rather than passing silently while it is open.
* The PLE n-gram table is declared as a Constant here so that the gather has
  something to index. In SERVING it is the host-mmap tier, read through
  `src/exec/ngram_table.h` (Link 3) and `src/exec/ngram_gather.h`, never an
  emitted constant. That divergence is deliberate and is recorded rather than
  hidden: what this module emits is the kernel-side gather's STRUCTURE, and
  which of the two the 0.5.0 artifact ships is a frontier decision, not one
  this file makes.
"""
import contextlib
import os
import tempfile

import numpy as np
import openvino as ov
from openvino import Model, Type
from openvino import opset13 as op

from . import attention as qattn
from . import gdn as qgdn
from . import hc as qhc
from . import ple as qple
from . import piecewise_export as pwe

# The n-gram table's element type as DECLARED in the serving-shape IR. The
# shipped tensor is IQ4_NL, which OpenVINO has no element type for; u4 carries
# the same nibble width, which is what the residency arithmetic depends on.
NGRAM_DECLARED_TYPE = Type.u4
# The expert bodies' declared type: the tiled lowering's compressed-weight
# family (verify_moe_lowering.py:30 "u4/i4/u8/i8"); u4 matches the shipped
# int4 slice width that exec/flash_next_offload.h:45 sizes the slot pool from
# (kFlashNextSliceBytes = 2,457,600 B per expert-layer for gate+up+down).
EXPERT_DECLARED_TYPE = Type.u4
# The tiled pass wants rank-4 [E, out, groups, group_size]; 128 is the group
# size the compressed-weight chain in export_mtp.py uses when a checkpoint does
# not declare its own (see moe_block_tiled's docstring: "this exporter still
# cannot reproduce production's own group_size ... only the grouping *shape*").
EXPERT_GROUP_SIZE = 128


class SparseArena:
    """Bump allocator over one sparse file. Every buffer it returns is a
    writeable numpy memmap; untouched pages cost nothing on disk or in RAM.

    `hold` keeps a reference to every buffer, because an ov Constant built with
    `shared_memory=True` does NOT own the mapping -- if the memmap were
    collected the Constant would point at unmapped memory.
    """

    def __init__(self, path=None, capacity_bytes=1 << 40):     # 1 TiB of address space
        self.dir = None
        if path is None:
            self.dir = tempfile.mkdtemp(prefix="q4e-serving-arena-")
            path = os.path.join(self.dir, "arena.bin")
        self.path = path
        with open(self.path, "wb") as f:
            f.truncate(capacity_bytes)
        self.capacity = capacity_bytes
        # ONE mapping for the whole arena; every allocation is a slice of it.
        # The first draft mmap'd the file once PER TENSOR and the 48-layer
        # build died at layer 3 with `OSError: [Errno 24] Too many open files`
        # -- np.memmap holds an fd per mapping. A view costs neither an fd nor
        # a page.
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r+",
                             shape=(capacity_bytes,))
        self.offset = 0
        self.hold = []
        self.declared_bytes = 0

    def alloc(self, nbytes):
        nbytes = int(nbytes)
        # page-align so two buffers never share a page (a write to one would
        # otherwise fault in the other's page and quietly cost real memory)
        base = (self.offset + 4095) & ~4095
        if base + nbytes > self.capacity:
            raise MemoryError(
                f"arena exhausted: {base + nbytes} > {self.capacity}; raise "
                f"capacity_bytes (it is address space, not disk)")
        buf = self._mm[base:base + max(nbytes, 1)]
        self.offset = base + nbytes
        self.declared_bytes += nbytes
        self.hold.append(buf)
        return buf

    def f32(self, shape):
        """A zero-filled f32 view of the requested shape, backed by sparse
        pages. Returned as numpy so an emitter can consume it unchanged."""
        n = int(np.prod(shape)) if len(shape) else 1
        buf = self.alloc(n * 4)
        return buf.view(np.float32)[:n].reshape(shape)

    def i64(self, shape):
        n = int(np.prod(shape)) if len(shape) else 1
        buf = self.alloc(n * 8)
        return buf.view(np.int64)[:n].reshape(shape)

    def constant(self, shape, ov_type):
        """An ov Constant of `shape` and `ov_type` over sparse pages, zero-copy.

        Sub-byte types go through the (array, shape, type) Tensor overload,
        which reinterprets a byte buffer -- that is the only way to declare a
        u4 tensor without allocating its dense form.
        """
        elems = int(np.prod(shape)) if len(shape) else 1
        nbytes = (elems * ov_type.bitwidth + 7) // 8
        buf = self.alloc(nbytes)
        tensor = ov.Tensor(buf, ov.Shape([int(d) for d in shape]), ov_type)
        return op.constant(tensor)

    def disk_kib(self):
        """Actual blocks on disk -- the measurement that proves the claim."""
        return os.stat(self.path).st_blocks * 512 // 1024

    def close(self):
        self.hold.clear()
        self._mm = None
        try:
            os.unlink(self.path)
        except OSError:
            pass
        if self.dir:
            try:
                os.rmdir(self.dir)
            except OSError:
                pass


# Every module that imported `_c` from q4e.gdn holds its OWN binding, so all of
# them have to be swapped. Missing one would silently materialise that family's
# weights -- which is why the contract test measures resident bytes rather than
# trusting this list.
_C_MODULES = (qgdn, qattn, qhc, qple, pwe)
try:                                              # moe is imported lazily by pwe
    from . import moe as qmoe
    _C_MODULES = _C_MODULES + (qmoe,)
except Exception:                                 # pragma: no cover
    pass


@contextlib.contextmanager
def shared_constants():
    """Swap every emitter's `_c` for a zero-copy constant factory.

    The replacement is the smallest possible change to `q4e.gdn._c`
    (`op.constant(np.ascontiguousarray(v, np.float32))`): the same call with
    `shared_memory=True`, taken only when the array is already a contiguous,
    writeable f32 -- which is exactly what SparseArena hands out. Anything else
    (a derived rope table, a reshaped norm weight) falls through to the
    original factory and is copied, as it should be: those are small and real.
    """
    saved = [(m, getattr(m, "_c")) for m in _C_MODULES if hasattr(m, "_c")]
    original = qgdn._c

    def _c_shared(v):
        arr = np.asarray(v)
        if (arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]
                and arr.flags["WRITEABLE"] and arr.size > 4096):
            return op.constant(arr, shared_memory=True)
        return original(v)

    try:
        for m, _ in saved:
            setattr(m, "_c", _c_shared)
        yield _c_shared
    finally:
        for m, old in saved:
            setattr(m, "_c", old)


# --------------------------------------------------------------------------
# The tiled MoE layer, expert bodies slot-referenced
# --------------------------------------------------------------------------

def _compressed_expert(arena, e, out, inn, name):
    """One expert-stacked weight in the tiled lowering's shape.

    rank-4 [E, out, groups, group_size] u4 Constant
      -> Convert(f32) -> Subtract(zero_point) -> Multiply(scale)
      -> Reshape(rank 4 -> 3)  [E, out, inn]

    The trailing Reshape is not cosmetic: verify_moe_lowering.py:33-42 records
    a real GPU compile crashing inside the fusing pass's own rewrite when it
    was absent, "because the pass's matcher anchors on that Reshape node".
    """
    gs = EXPERT_GROUP_SIZE
    assert inn % gs == 0, f"{name}: inner {inn} is not a multiple of group {gs}"
    groups = inn // gs
    w = arena.constant([e, out, groups, gs], EXPERT_DECLARED_TYPE)
    w.set_friendly_name(name + "/weight_u4")
    zp = arena.constant([e, out, groups, 1], EXPERT_DECLARED_TYPE)
    zp.set_friendly_name(name + "/zero_point")
    scale = op.constant(np.ones((e, out, groups, 1), np.float32))
    scale.set_friendly_name(name + "/scale")
    x = op.convert(w, Type.f32)
    x = op.subtract(x, op.convert(zp, Type.f32))
    x = op.multiply(x, scale)
    x = op.reshape(x, op.constant(np.array([e, out, inn], np.int64)),
                   special_zero=False)
    x.set_friendly_name(name + "/dequant_reshape")
    return x


def emit_moe_tiled(hidden_bth, config, state, arena, T, tag):
    """The MoE layer in the shape measured to fuse on the card
    (export_mtp.py:401 moe_block_tiled), at real geometry, expert bodies
    slot-referenced. Returns a [1,T,H] node."""
    H = config.hidden_size
    E = config.num_experts
    I = config.moe_intermediate_size
    k = config.num_experts_per_tok

    i32 = lambda v: op.constant(np.array(v, np.int32))
    i32v = lambda v: op.constant(np.array([v], np.int32))

    # the shared rank-2 flatten both the router and the Tile entry read
    y_flat = op.reshape(hidden_bth, op.constant(np.array([-1, H], np.int32)),
                        special_zero=False)                            # [M,H]

    # router, rank 2 throughout (export_mtp.py:472-494)
    logits = op.matmul(y_flat, qgdn._c(state["mlp.gate.weight"]),
                       transpose_a=False, transpose_b=True)            # [M,E]
    probs = op.softmax(logits, axis=-1)
    tk = op.topk(probs, i32(k), axis=-1, mode="max", sort="value",
                 index_element_type="i32")
    vals, idx = tk.output(0), tk.output(1)
    if getattr(config, "norm_topk_prob", True):
        vals = op.divide(vals, op.reduce_sum(vals, i32v(-1), keep_dims=True))
    vals = op.slice(vals, op.constant(np.array([0, 0], np.int32)),
                    op.shape_of(vals, output_type="i32"),
                    op.constant(np.array([1, 1], np.int32)),
                    op.constant(np.array([0, 1], np.int32)))
    zeros = op.multiply(probs, op.constant(np.array([0.0], np.float32)))
    weights = op.scatter_elements_update(zeros, idx, vals, i32(-1))     # [M,E]

    # entry: Tile -> Reshape
    tiled = op.tile(y_flat, op.constant(np.array([E, 1], np.int32)))
    m_h3 = op.reshape(tiled, op.constant(np.array([E, -1, H], np.int32)),
                      special_zero=False)                              # [E,M,H]

    gate_w = _compressed_expert(arena, E, I, H, f"{tag}/experts_gate")
    up_w = _compressed_expert(arena, E, I, H, f"{tag}/experts_up")
    down_w = _compressed_expert(arena, E, H, I, f"{tag}/experts_down")

    g = op.swish(op.matmul(m_h3, gate_w, transpose_a=False, transpose_b=True))
    u = op.matmul(m_h3, up_w, transpose_a=False, transpose_b=True)
    outs = op.matmul(op.multiply(g, u), down_w,
                     transpose_a=False, transpose_b=True)              # [E,M,H]

    wt = op.transpose(weights, op.constant(np.array([1, 0], np.int32)))  # [E,M]
    wt = op.unsqueeze(wt, i32(-1))                                       # [E,M,1]
    mixed = op.reduce_sum(op.multiply(outs, wt), i32v(0), keep_dims=False)  # [M,H]

    # the shared expert (pin 986-996) stays dense f32 -- it is one MLP per
    # layer, 0.0183 GiB, and it is CARD tier in the size ledger
    shared_state = {kk[len("mlp."):]: v for kk, v in state.items()
                    if kk.startswith("mlp.shared_expert")}
    from . import moe as _moe
    sh = _moe.emit_shared_expert(
        op.reshape(y_flat, op.constant(np.array([1, T, H], np.int64)),
                   special_zero=False),
        config, shared_state, T)                                       # [T,H]

    out2d = op.add(mixed, sh)
    return op.reshape(out2d, op.constant(np.array([1, T, H], np.int64)),
                      special_zero=False)


# --------------------------------------------------------------------------
# The full-geometry serving-shape backbone
# --------------------------------------------------------------------------

def _layer_state(arena, config, kind):
    """Sparse-declared state for one decoder layer, at the pin's own
    module-relative keys, real shapes from `config`."""
    H = config.hidden_size
    hc = config.hc_count
    lr = config.hc_lowrank
    st = {}

    # SHAPES READ FROM THE CHECKPOINT, not derived from a guess. Measured
    # 2026-09-12 through gguf_feed.pin_tensor at blk.0/blk.1/blk.3:
    #   hc_norm [10240]  down [320,10240]  up [10240,320]  inject [4,10240]
    # The first draft had up as [hc*hc, lr] and inject as [hc, H]; both are
    # wrong and both were caught by `emit_combine` refusing to reshape.
    def mix(prefix):
        st[prefix + "hc_norm.weight"] = arena.f32([hc * H])
        st[prefix + "input_mix_weight_down.weight"] = arena.f32([lr, hc * H])
        st[prefix + "input_mix_weight_up.weight"] = arena.f32([hc * H, lr])
        st[prefix + "block_inject_weight.weight"] = arena.f32([hc, hc * H])

    mix("attn_hyper_connection.")
    mix("mlp_hyper_connection.")

    if kind == "gdn":
        kd, kh = config.linear_key_head_dim, config.linear_num_key_heads
        vd, vh = config.linear_value_head_dim, config.linear_num_value_heads
        conv_dim = kd * kh * 2 + vd * vh
        st["linear_attn.in_proj_qkv.weight"] = arena.f32([conv_dim, H])
        st["linear_attn.in_proj_z.weight"] = arena.f32([vd * vh, H])
        st["linear_attn.in_proj_a.weight"] = arena.f32([vh, H])
        st["linear_attn.in_proj_b.weight"] = arena.f32([vh, H])
        st["linear_attn.A_log"] = arena.f32([vh])
        st["linear_attn.dt_bias"] = arena.f32([vh])
        st["linear_attn.conv1d.weight"] = arena.f32(
            [conv_dim, 1, config.linear_conv_kernel_dim])
        st["linear_attn.norm.weight"] = arena.f32([vd])
        st["linear_attn.out_proj.weight"] = arena.f32([H, vd * vh])
    else:
        heads, kv, d = (config.num_attention_heads,
                        config.num_key_value_heads, config.head_dim)
        st["self_attn.q_proj.weight"] = arena.f32([heads * 2 * d, H])
        st["self_attn.k_proj.weight"] = arena.f32([kv * d, H])
        st["self_attn.v_proj.weight"] = arena.f32([kv * d, H])
        st["self_attn.o_proj.weight"] = arena.f32([H, heads * d])
        st["self_attn.q_norm.weight"] = arena.f32([d])
        st["self_attn.k_norm.weight"] = arena.f32([d])

    Is = config.shared_expert_intermediate_size
    st["mlp.gate.weight"] = arena.f32([config.num_experts, H])
    st["mlp.shared_expert.gate_proj.weight"] = arena.f32([Is, H])
    st["mlp.shared_expert.up_proj.weight"] = arena.f32([Is, H])
    st["mlp.shared_expert.down_proj.weight"] = arena.f32([H, Is])
    st["mlp.shared_expert_gate.weight"] = arena.f32([1, H])
    return st


def _ple_state(arena, config):
    H = config.hidden_size
    hc = config.hc_count
    Hn = (config.ngram_size - 1) * config.heads_per_ngram
    head_dim = config.ple_embed_dim // Hn
    return {
        "key_proj.weight": arena.f32([hc * H, config.ple_embed_dim]),
        "value_proj.weight": arena.f32([H, config.ple_embed_dim]),
        # [hc*H] = 10240, measured; the group RMS norms span the whole
        # hyper-connection width, not one hidden width
        "norm_key.weight": arena.f32([hc * H]),
        "norm_query.weight": arena.f32([hc * H]),
        "norm_conv.weight": arena.f32([hc * H]),
        "conv1d.weight": arena.f32([hc * H, 1, config.ple_conv_kernel_size]),
    }, head_dim, Hn


def build_serving_shape_ir(config=None, seq_len=64, arena=None, n_layers=None):
    """The full-geometry serving-shape backbone as an ov::Model.

    Inputs
        input_ids        [1, T]      i64
        position_ids     [1, T]      i64
        ngram_row_ids    [1, T, 16]  i64   -- the hashed n-gram row ids, one per
                                             n-gram head; 16 = (ngram_size-1) *
                                             heads_per_ngram, exactly
                                             `HashParams::num_ngram_heads()`
                                             (src/exec/ngram_row_ids.h:59) and
                                             the "8 x 2-gram + 8 x 3-gram" the
                                             same file's header states at :21.
                                             The hash itself is computed by
                                             `lgc::ngram::row_ids`
                                             (ngram_row_ids.h:117); the graph
                                             consumes the ids, never the tokens.
    Output
        logits           [1, T, vocab] f32

    Returns (model, report) where `report` carries the measured structure.
    """
    cfg = config if config is not None else pwe.real_config()
    T = int(seq_len)
    own_arena = arena is None
    ar = arena if arena is not None else SparseArena()
    nl = int(n_layers if n_layers is not None else cfg.num_hidden_layers)

    H = cfg.hidden_size
    hc = cfg.hc_count
    V = cfg.vocab_size
    ple_layer_idx = 1                      # GGUF ple.layers [1]; pwe REAL_GEOMETRY

    try:
        with shared_constants():
            input_ids = op.parameter([1, T], Type.i64)
            input_ids.set_friendly_name("input_ids")
            pid = op.parameter([1, T], Type.i64)
            pid.set_friendly_name("position_ids")
            ple_state, head_dim, Hn = _ple_state(ar, cfg)
            row_ids = op.parameter([1, T, Hn], Type.i64)
            row_ids.set_friendly_name("ngram_row_ids")
            # the GDN + PLE padding mask, same port q4e.backbone declares
            # (backbone.py:105-106); ones = full sequence
            conv_mask = op.parameter([1, T], Type.f32)
            conv_mask.set_friendly_name("conv_mask")

            # embed -> repeat to the hyper-connection width (backbone.py)
            embed_w = ar.f32([V, H])
            emb = op.gather(qgdn._c(embed_w), input_ids, qgdn._i(0))    # [1,T,H]
            hidden = op.tile(emb, op.constant(np.array([1, 1, hc], np.int64)))

            # the n-gram table: declared, never materialised. In SERVING this is
            # the host-mmap tier (src/exec/ngram_table.h Link 3); here it is the
            # kernel-side gather's structure.
            ngram_table = ar.constant([cfg.ngram_total_vocab
                                       if hasattr(cfg, "ngram_total_vocab")
                                       else pwe.REAL_GEOMETRY["ngram_total_vocab"],
                                       head_dim], NGRAM_DECLARED_TYPE)
            ngram_table.set_friendly_name("ple/ngram_table_u4")

            kinds = []
            for i in range(nl):
                kind = "attn" if (i % 4) == 3 else "gdn"
                kinds.append(kind)
                st = _layer_state(ar, cfg, kind)

                if i == ple_layer_idx:
                    # pin 1283: hidden = hidden + ple(...), ADDITIVE
                    gathered = op.gather(op.multiply(
                        op.convert(ngram_table, Type.f32),
                        op.constant(np.ones((1, 1), np.float32))),
                        row_ids, qgdn._i(0))                # [1,T,Hn,head_dim]
                    emb_ple = op.reshape(
                        gathered,
                        op.constant(np.array([1, T, Hn * head_dim], np.int64)),
                        special_zero=False)
                    hidden = op.add(hidden, _ple_tail(
                        hidden, emb_ple, cfg, ple_state, T, conv_mask))

                # attn_hyper_connection (use_combine=True) -> mixer -> combine
                h, hyper, inj = _split_combine(hidden, cfg, st,
                                               "attn_hyper_connection.", T)
                if kind == "gdn":
                    g = qgdn.emit_gdn(h, conv_mask, cfg,
                                      _strip(st, "linear_attn."), T)
                else:
                    g = qattn.emit_dense_attention(h, pid, cfg,
                                                   _strip(st, "self_attn."), T)
                hidden = _recombine(hyper, inj, g, cfg, T)

                h, hyper, inj = _split_combine(hidden, cfg, st,
                                               "mlp_hyper_connection.", T)
                m = emit_moe_tiled(h, cfg, st, ar, T, f"layer{i}/moe")
                hidden = _recombine(hyper, inj, m, cfg, T)

            # the final mixer, use_combine=False (pin 1493-1496), then lm_head
            fin = {
                "hc_norm.weight": ar.f32([hc * H]),
                "input_mix_weight_down.weight": ar.f32([cfg.hc_lowrank, hc * H]),
                "input_mix_weight_up.weight": ar.f32([hc * H, cfg.hc_lowrank]),
            }
            last = qhc.emit_hc(hidden, cfg, fin, T)                     # [1,T,H]
            head_w = ar.f32([V, H])
            logits = op.matmul(last, qgdn._c(head_w),
                               transpose_a=False, transpose_b=True)
            res = op.result(logits)
            res.set_friendly_name("logits")

            model = Model([res], [input_ids, pid, row_ids, conv_mask],
                          "qwen4_exp_serving_shape")

        nodes, const_bytes, counts = pwe.graph_measures(model)
        report = {
            "n_layers": nl,
            "gdn_layers": kinds.count("gdn"),
            "attn_layers": kinds.count("attn"),
            "seq_len": T,
            "nodes": nodes,
            "graph_const_bytes": const_bytes,
            "op_histogram": counts,
            "arena_declared_bytes": ar.declared_bytes,
            "arena_disk_kib": ar.disk_kib(),
            "inputs": [(p.get_node().get_friendly_name(),
                        list(p.get_shape()), str(p.get_element_type()))
                       for p in model.inputs],
            "outputs": [(r.get_node().get_friendly_name(),
                         list(r.get_shape()), str(r.get_element_type()))
                        for r in model.outputs],
        }
        return model, report
    except BaseException:
        if own_arena:
            ar.close()
        raise


def _strip(state, prefix):
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def _split_combine(hidden, config, state, prefix, T):
    """backbone.py's use_combine=True mixer: returns (h, hyper, inject)."""
    sub = _strip(state, prefix)
    return qhc.emit_combine(hidden, config, sub, T)


def _recombine(hyper, inj, block_out, config, T):
    """pin 1302-3 / 1308-9: hidden = hyper + (out.unsqueeze(-2) *
    inj.unsqueeze(-1)).flatten(-2)."""
    H = config.hidden_size
    hc = config.hc_count
    o4 = op.reshape(block_out, op.constant(np.array([1, T, 1, H], np.int64)),
                    special_zero=False)
    i4 = op.reshape(inj, op.constant(np.array([1, T, hc, 1], np.int64)),
                    special_zero=False)
    prod = op.reshape(op.multiply(o4, i4),
                      op.constant(np.array([1, T, hc * H], np.int64)),
                      special_zero=False)
    return op.add(hyper, prod)


def _ple_tail(hidden, emb, config, state, T, conv_mask=None):
    """The PLE body after the gather -- pin 1243-1254, the same op sequence
    q4e.ple._ple_subgraph runs, entered from a pre-gathered [1,T,ple_embed_dim].
    Kept here rather than added to ple.py so the parity-gated emitter is not
    touched by a structure-only change."""
    import math
    H = config.hidden_size
    hc = config.hc_count
    eps = config.rms_norm_eps
    K = config.ple_conv_kernel_size
    dilation = config.ngram_size

    key = qgdn._mm(emb, qgdn._c(state["key_proj.weight"]), tb=True)
    key_n = qple._group_rms(key, T, hc, H, state["norm_key.weight"], eps)
    key_n4 = qgdn._reshape(key_n, [1, T, hc, H])
    value = qgdn._mm(emb, qgdn._c(state["value_proj.weight"]), tb=True)
    q_n = qple._group_rms(hidden, T, hc, H, state["norm_query.weight"], eps)
    q_n4 = qgdn._reshape(q_n, [1, T, hc, H])
    gate = qgdn._rsum(qgdn._mul(key_n4, q_n4), 3)
    gate = qgdn._mul(gate, qgdn._c(np.float32(1.0 / math.sqrt(H))))
    ag = op.maximum(op.abs(gate), qgdn._c(np.float32(1e-6)))
    gate = qgdn._mul(op.sqrt(ag), op.sign(gate))
    sg = op.sigmoid(gate)
    value4 = qgdn._reshape(value, [1, T, 1, H])
    gv = qgdn._mul(sg, value4)
    gv_flat = qgdn._reshape(gv, [1, T, hc * H])
    gv_normed = qple._group_rms(gv_flat, T, hc, H, state["norm_conv.weight"], eps)
    if conv_mask is not None:                    # pin 1251-1253
        m = qgdn._reshape(conv_mask, [1, T, 1])
        gv_flat = qgdn._mul(gv_flat, m)
        gv_normed = qgdn._mul(gv_normed, m)
    conv_out = qple._short_conv(gv_normed, state["conv1d.weight"], T,
                                hc * H, K, dilation)
    return qgdn._add(gv_flat, conv_out)


# --------------------------------------------------------------------------
# The OTD contract, transcribed from the C++ so the test checks code, not prose
# --------------------------------------------------------------------------

def slot_pool_from_ir(model, num_expert, ratio_pct):
    """Python transcription of `slot_pool_from_ir`, src/exec/backend_ov.cpp:577-623.

    Line-for-line, with the C++ line numbers on each step. Returns None where
    the C++ returns nullopt.
    """
    if num_expert <= 0:                                        # :578
        return None
    out = {"total_bytes": 0, "per_expert_bytes": 0, "slots": 0, "moe_layers": 0}
    for node in model.get_ordered_ops():                       # :580
        tname = node.get_type_name().lower()                   # :581-584
        if "moe" not in tname:                                 # :585
            continue
        per_expert_bytes = 0
        for i in range(node.get_input_size()):                 # :588
            src = node.input_value(i).get_node()
            if src.get_type_name() == "Convert":               # :594-596
                src = src.input_value(0).get_node()
            if src.get_type_name() != "Constant":              # :597-598
                continue
            sh = list(src.get_output_shape(0))                 # :599
            if not sh or sh[0] != num_expert:                  # :600
                continue
            elems = 1
            for d in sh[1:]:                                   # :602
                elems *= d
            # :603 -- element_type().size(), which CEILS a sub-byte width to a
            # whole byte. That is deliberate over-reservation, per the comment
            # at :610-615, and the contract test measures the factor.
            et = src.get_output_element_type(0)
            per_expert_bytes += elems * ((et.bitwidth + 7) // 8)
        if per_expert_bytes == 0:                              # :605
            continue
        if out["moe_layers"] == 0:                             # :607
            out["per_expert_bytes"] = per_expert_bytes
        out["total_bytes"] += _expert_slot_bytes(num_expert, ratio_pct,
                                                 per_expert_bytes, 1)   # :618
        out["moe_layers"] += 1
    if out["moe_layers"] == 0:                                 # :621
        return None
    out["slots"] = _expert_slot_bytes(num_expert, ratio_pct, 1, 1)
    return out


def _expert_slot_bytes(num_expert, ratio_pct, per_expert_bytes, moe_layers):
    """src/exec/fit.h:95 -- ceil(num_expert * (100 - ratio) / 100) slots per
    layer, times per-expert bytes, times layers."""
    slots = -((-num_expert * (100 - ratio_pct)) // 100)
    return slots * per_expert_bytes * moe_layers


__all__ = [
    "SparseArena", "shared_constants", "build_serving_shape_ir",
    "emit_moe_tiled", "slot_pool_from_ir",
    "EXPERT_DECLARED_TYPE", "NGRAM_DECLARED_TYPE", "EXPERT_GROUP_SIZE",
]
