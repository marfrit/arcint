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
one byte of weight data. `src/exec/backend_ov.cpp:553-557`:

    // M7 §2 Phase B, analytic route ...: sizes the expert slot pool from the
    // `read_model` the load path already holds, without materialising any
    // weight data -- a weightless IR (ov::weights_path) still carries every
    // constant's shape and element type in the XML.

So "shape and element type, no data" is not a degraded artifact from the
serving runtime's point of view -- for the slot-pool decision it is the WHOLE
artifact. This module emits exactly that, and the contract test
(`tests/python/test_serving_shape.py`) checks it against a Python transcription
of `slot_pool_from_ir` (`backend_ov.cpp:578-624`) rather than against a
description of it.

--------------------------------------------------------------------------
2. THE EXPERT-SLOT (OTD) CONTRACT, BOTH SIDES, CITED
--------------------------------------------------------------------------

C++ side, `src/exec/backend_ov.cpp:578-624` `slot_pool_from_ir`:

  * a MoE op is any node whose OpenVINO TYPE NAME contains "moe",
    case-insensitively                                    (backend_ov.cpp:582-586)
  * its expert-weight inputs are the Constant operands, or a Constant behind
    exactly ONE Convert, whose LEADING DIMENSION equals `num_expert`
                                                          (backend_ov.cpp:589-605)
  * per-expert bytes = product of dims[1:] x element_type().size()
                                                          (backend_ov.cpp:601-605)
  * an unmatched graph returns nullopt and the caller falls back to the
    plateau probe -- "this function never guesses"        (backend_ov.cpp:567-570)

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
* It is not the paged serving graph, and the reason is not that the paged
  ports are unwritten. NOBODY WRITES THEM: `load_paged` runs
  `ov::pass::SDPAToPagedAttention` over the artifact it has just read
  (`backend_ov.cpp:2582`) and that pass produces the whole port contract
  (`conv_state_table.N`, `gated_delta_state_table.N`, `key_cache.N`,
  `value_cache.N`, `la.*` and the index ports -- `backend_ov.cpp:3199-3207`
  and `:6141-6151`) out of three constructs of the STATEFUL graph: a rank-3
  Variable per GDN short conv, a rank-4 Variable per GDN recurrent state, and
  a ScaledDotProductAttention over a rank-4 KV Variable. Measured both ways
  2026-09-13; the reading is in `tests/python/test_serving_shape.py` beside
  the ports table.

  ALL THREE ARE EMITTED (2026-09-13): the full-attention layers carry a KV
  Variable and a ScaledDotProductAttention (`emit_stateful_attention`); every
  GDN layer's short conv carries its K-column state in a rank-3 Variable
  (`stateful_short_conv`); and the GDN recurrence is the token-sequential
  `v5::Loop` that `FuseGDNLoop` fuses into the op `PagedGatedDeltaNetFusion`
  matches (`stateful_gdn_core`). Both GDN constructs reach the parity-gated
  emitter through `emit_gdn` hooks, which default to the unrolled conv and
  the chunked core, so that module emits no different op when nobody passes
  them. The pass therefore produces EVERY port in the contract, and the
  strict xfail that carried the gap RETIRED when the last row of the table
  flipped -- which is what it was written strict for.
* The PLE n-gram table is NOT a constant of this graph any more (2026-09-13,
  increment 5). It used to be declared as one u4 Constant so the gather had
  something to index, and the device wrote the refusal: ONE object of
  25,600,122,880 B, above the A770's 4,294,959,104 B per-object cap and above
  the B60's whole 24,385,683,456 B (window-050 §4.6, P3/P3b/P6). Chunking it
  into sub-cap CONSTANTS was falsified before any card was touched, by three
  measured numbers and one inequality: 25,600,122,880 B of table exceeds the
  A770's 16,225,243,136 B and the B60's 24,385,683,456 B of VRAM whether it is
  one object or six, so a chunked-constant graph cannot be resident on
  either card. The table is now carried as PARAMETER PORTS `ngram_table.K`,
  u8 `[rows_K, row_bytes]`, each under the cap (`ngram_table_chunks`), fed at
  request time from host memory -- the host-mmap tier the served runtime
  already reads through `src/exec/ngram_table.h`, handed to the graph instead
  of gathered beside it. The gather stays IN the graph (`ngram_chunked_gather`:
  chunk id and local row FED by the host -- the in-graph decomposition was
  measured wrong above 2**24 on the A770, see that function -- one Gather per
  port, a Select to pick the chunk's row). Why u8 and not the declared u4: the GPU
  plugin rewrites a u4 Parameter to u8 anyway (`transformations_pipeline.cpp`
  `int_convert_precision_map`, pinned source), and a USM-host tensor is shared
  with the graph WITHOUT a device copy only when its element type is the
  port's own (`sync_infer_request.cpp` `prepare_input`, the `is_usm_host_tensor
  && !convert_needed` branch). The nibble unpack after the gather is this
  graph's stand-in for the row format, low nibble first; the shipped rows are
  block-quantised and dequantised host-side today (`ngram_gather.h`), and which
  side dequantises in the 0.5.0 artifact is still the frontier's decision.
"""
import contextlib
import os
import tempfile
import types

import numpy as np
import openvino as ov
import openvino.op.util as ovutil
from openvino import Model, Type
from openvino import opset13 as op

from . import attention as qattn
from . import backbone as qbb
from . import gdn as qgdn
from . import hc as qhc
from . import ple as qple
from . import piecewise_export as pwe

# The n-gram table's row format. It used to be DECLARED u4 (the same nibble
# width as the shipped tensor, for the residency arithmetic). The
# shipped tensor is IQ4_NL, which OpenVINO has no element type for; since
# feed-the-ports the rows travel over the `ngram_table.K` ports as their OWN
# bytes -- ggml's block_iq4_nl: a little-endian f16 scale `d` followed by 16
# bytes of nibbles per 32 elements (element j is the low nibble of byte j,
# element 16+j the high one), so a 160-wide row is 5 x 18 = 90 bytes -- and
# are dequantised AFTER the gather (`ngram_dequant_iq4nl`: d x kvalues[q]).
# That is what lets the real table (28.8 GB) bind to the ports unchanged
# from the GGUF, with no host-side conversion and no second copy.
NGRAM_PORT_TYPE = Type.u8
NGRAM_BLOCK_ELEMS = 32
NGRAM_BLOCK_BYTES = 18
# ggml's kvalues_iq4nl, the 16-entry codebook every IQ4_NL nibble indexes
NGRAM_IQ4NL_KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10,
                       1, 13, 25, 38, 53, 69, 89, 113)


def ngram_row_bytes(head_dim):
    """Bytes of one table row of `head_dim` elements in IQ4_NL."""
    assert head_dim % NGRAM_BLOCK_ELEMS == 0, head_dim
    return head_dim // NGRAM_BLOCK_ELEMS * NGRAM_BLOCK_BYTES
# The per-object allocation cap the table is chunked under: the A770's, as the
# GPU plugin reported it at engine.cpp:319 (`RUN@be57428`, window-050 §4.4,
# re-read verbatim by `RUN@8a84598` §4.6). The tighter of the two cards, and
# the check applies to USM-host allocations too (engine.cpp `check_allocatable`
# runs before the allocation type is looked at). A build for a card with a
# wider cap passes its own; nothing here reads the device.
NGRAM_CHUNK_CAP_BYTES = 4_294_959_104


def tiny_config(n_layers, ngram_vocab_size_base=200):
    """The suite's REDUCED GEOMETRY at a chosen depth (a multiple of 4, so
    every 4-aligned segment holds one full-attention layer): hidden 256,
    vocab 512, 8 experts of 128, one IQ4_NL block per PLE table row. The
    n-gram table is sized from the hash constants the same rule derives for
    the real model (`ngram_ids.derive`), so `ngram_ids.gen_row_ids` on this
    config addresses rows the table has -- a forward on the CPU plugin is
    then a device-free cell (tests/python, the boot driver's `--tiny`).
    Never the served path; the real geometry is `piecewise_export.real_config`.
    """
    from . import ngram_ids as nid
    cfg = pwe.real_config()
    small = type(cfg)(
        hidden_size=256, num_hidden_layers=n_layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        hc_count=cfg.hc_count, hc_lowrank=32,
        ple_embed_dim=512, ple_conv_kernel_size=cfg.ple_conv_kernel_size,
        ngram_size=cfg.ngram_size, heads_per_ngram=cfg.heads_per_ngram,
        ngram_vocab_size_base=ngram_vocab_size_base,
        vocab_size=512, rms_norm_eps=cfg.rms_norm_eps,
        linear_key_head_dim=32, linear_num_key_heads=2,
        linear_value_head_dim=32, linear_num_value_heads=4,
        linear_conv_kernel_dim=cfg.linear_conv_kernel_dim,
        hidden_act="silu",
        layer_types=["qwen_sparse_attention" if i % 4 == 3 else "linear_attention"
                     for i in range(n_layers)],
        ple_layer_ids=[2],                       # decoder layer 1, as the real model
        eos_token_id=511,
    )
    _mult, sizes, _offs = nid.derive(small.vocab_size, small.ngram_size,
                                     small.heads_per_ngram, ngram_vocab_size_base, 0)
    small.ngram_total_vocab = int(sum(sizes))    # every derived row id has a row
    return small
# Chunk row counts are rounded down to a multiple of this so a chunk boundary
# never falls inside a page of the host mapping that serves it.
NGRAM_CHUNK_ROW_ALIGN = 4096


def ngram_table_chunks(n_rows, row_bytes, cap_bytes=NGRAM_CHUNK_CAP_BYTES,
                       align=NGRAM_CHUNK_ROW_ALIGN):
    """The row partition of an `[n_rows, row_bytes]` table into objects that
    each stay under `cap_bytes`: a list of row counts, in port order, summing
    to `n_rows`. Every chunk but the last has `rows_per_chunk` rows, the
    largest multiple of `align` whose byte size fits the cap; the last takes
    the remainder. A table that fits is one chunk of `n_rows`.

    This is the ONE place the partition is computed. The contract cell
    transcribes the arithmetic independently rather than importing it, and the
    served runtime reads the partition off the compiled model's port shapes
    rather than off any config -- so the artifact carries its own cap.
    """
    n_rows, row_bytes = int(n_rows), int(row_bytes)
    assert n_rows > 0 and row_bytes > 0, (n_rows, row_bytes)
    if n_rows * row_bytes <= cap_bytes:
        return [n_rows]
    per = (cap_bytes // row_bytes) // align * align
    assert per > 0, (
        f"a single aligned row block of {align} x {row_bytes} B does not fit "
        f"the {cap_bytes} B cap")
    full, rest = divmod(n_rows, per)
    return [per] * full + ([rest] if rest else [])
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
        # Bytes this arena was asked to WRITE (the fill). Distinct from
        # `declared_bytes`, which counts address space handed out: an unfilled
        # build declares tens of GiB and writes nothing at all.
        self.written_bytes = 0
        # Where each NAMED constant landed: (name, base, nbytes). The content
        # acceptance reads a filled body back out of these pages -- the bytes
        # the ov Constant actually wraps -- rather than out of the filler's
        # return value, which would be checking the fill against itself.
        self.placements = []
        self._last_base = 0
        # The f32 scales, by constant name. Kept as objects rather than read
        # back from pages because they ARE f32 pages -- there is no packing to
        # verify, and the read-back acceptance is about the u4 nibbles.
        self.scales = {}

    def alloc(self, nbytes):
        nbytes = int(nbytes)
        # page-align so two buffers never share a page (a write to one would
        # otherwise fault in the other's page and quietly cost real memory)
        base = (self.offset + 4095) & ~4095
        self._last_base = base
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

    def constant(self, shape, ov_type, fill=None, name=None):
        """An ov Constant of `shape` and `ov_type` over sparse pages, zero-copy.

        Sub-byte types go through the (array, shape, type) Tensor overload,
        which reinterprets a byte buffer -- that is the only way to declare a
        u4 tensor without allocating its dense form.

        `fill`, when given, is a byte buffer written into the arena pages
        BEFORE the Constant is built over them. Writing first and wrapping
        after is not a style choice: `op.constant(tensor)` wraps the mapping
        rather than copying it (that is the whole keystone mechanism), so a
        write afterwards would be a mutation of a live Constant. Pages this
        touches become real pages -- on disk and in RAM -- which is exactly
        what a FILLED artifact is and exactly why the unfilled path must keep
        passing `fill=None`.
        """
        elems = int(np.prod(shape)) if len(shape) else 1
        nbytes = (elems * ov_type.bitwidth + 7) // 8
        buf = self.alloc(nbytes)
        if fill is not None:
            src = np.frombuffer(np.ascontiguousarray(fill), dtype=np.uint8)
            assert src.size == nbytes, (
                f"fill is {src.size} B, the {list(shape)} {ov_type} constant "
                f"is {nbytes} B")
            buf[:nbytes] = src
            self.written_bytes += nbytes
        if name is not None:
            self.placements.append(
                {"name": name, "base": self._last_base, "nbytes": nbytes,
                 "shape": [int(d) for d in shape], "type": str(ov_type)})
        tensor = ov.Tensor(buf, ov.Shape([int(d) for d in shape]), ov_type)
        return op.constant(tensor)

    def f32_filled(self, values):
        """An f32 Constant over arena pages carrying `values`, zero-copy.

        Used only by the FILL: the unfilled build's scales are a materialised
        `np.ones`, and moving those into the arena would write 78.6 MB of
        pages per layer and break the `0 KiB on disk` measurement that is the
        keystone's whole point.
        """
        arr = np.ascontiguousarray(values, dtype=np.float32)
        buf = self.f32(arr.shape)
        buf[...] = arr
        self.written_bytes += arr.nbytes
        return op.constant(buf, shared_memory=True)

    def read_back(self, name):
        """The raw bytes currently in the arena pages of a named constant.

        Read from the mapping, so what comes back is what the ov Constant is
        looking at -- including, if something went wrong, whatever overwrote
        it."""
        hits = [p for p in self.placements if p["name"] == name]
        assert len(hits) == 1, (
            f"{name!r} matches {len(hits)} placements; "
            f"have {[p['name'] for p in self.placements][:8]}")
        p = hits[0]
        return np.array(self._mm[p["base"]:p["base"] + p["nbytes"]],
                        dtype=np.uint8), p

    def disk_kib(self):
        """Blocks the filesystem has allocated for the arena file.

        READ THE LIMITS BEFORE USING THIS AS EVIDENCE. It is reported by
        198b736 as "arena blocks on disk 0 KiB" beside "183.07 GiB declared",
        and on the dev host's filesystem it CANNOT DISTINGUISH an unwritten
        arena from a written one. Measured 2026-09-12, dev host, ZFS
        (recordsize 131072), one 4 GiB sparse file per probe:

            written            st_blocks after msync   after `sync` + 12 s
            nothing                        512 B                    512 B
            512 MiB of zeros               512 B                    512 B
            512 MiB of random            512 B              439,174,656 B

        Two separate effects, and each defeats the check on its own:

          * ZFS allocates on TRANSACTION-GROUP COMMIT, not on msync. Until the
            txg syncs, a fully written file reports 512 B. The `flush()` below
            is msync and is NOT sufficient; only a system `sync` plus a wait
            is, which is not a thing a test should do.
          * An all-zero record is stored as a HOLE. A arena deliberately
            written full of zeros would report 0 KiB forever, correctly.

        So `disk_kib() <= 64` is true of the unfilled build, and would ALSO be
        true of a build that had materialised every constant as zeros, and of
        one that had just written the real weights. It is reported here
        because it is cheap and it is one more sign; it is not the guard. The
        guard is `test_the_full_48_layer_stack_emits_at_real_geometry`'s peak
        RSS (CF-RESIDENT) and, for a filled build, reading the pages back and
        finding them non-zero.

        This was found while accepting the fill: 2,693,529,600 B of expert
        bodies written, 1,899,503,616 B reported even after msync.
        """
        if self._mm is not None:
            self._mm.flush()            # msync: necessary, and not sufficient
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
# them have to be swapped. Missing one silently materialises that family's
# weights: the build still emits the same node count, still declares the same
# bytes, still occupies 0 KiB on disk, and costs GiB of anonymous memory that
# nothing observes until the host OOMs. (The node count used to be written out
# here as 84,372, which the stateful attention layers moved to 84,374 -- the
# argument never needed the number, and a number in a comment is a number that
# rots, so it is gone rather than corrected.)
#
# CF-RESIDENT (REVIEW 2a45349 F2, closed 2026-09-12). This comment used to end
# "-- which is why the contract test measures resident bytes rather than
# trusting this list". IT DID NOT MEASURE RESIDENT BYTES. The reviewer dropped
# `qattn` from the tuple and the whole suite stayed green while peak RSS went
# 4.52 -> 8.98 GiB. Two cells now carry the claim, and between them they cover
# what neither covers alone:
#
#   tests/python/test_serving_shape.py
#     ::test_the_full_48_layer_stack_emits_at_real_geometry
#        asserts peak RSS of the 48-layer build against a ceiling derived from
#        both sides (authored 4.52 GiB, cheapest single-module defect 6.23).
#        It cannot see `qple` or `pwe`: dropping either leaves peak RSS at
#        4.52 GiB exactly, because neither module's `_c` is reached with a
#        large arena array during this build. Measured, not assumed. `qbb` is
#        in the same position for a structural reason (below) and is the one
#        row of that derivation which is NOT probed.
#        Which modules that cell sees and which it does not is generated from
#        `PEAK_RSS_GIB_WHEN_DROPPED` and printed by
#        ::test_the_rss_derivation_accounts_for_every_swapped_module, which
#        fails if a module joins this tuple without a row (REVIEW 23938c1 F2:
#        `qbb` joined it without one, and a docstring then counted six).
#     ::test_every_module_binding_the_constant_factory_is_swapped
#        closes that gap structurally -- an ast scan over tools/q4e asserting
#        every module that BINDS `_c` appears below, whether or not this
#        build happens to call it.
#
# `backbone` was the one the scan found on its first run: it binds `_c`
# (backbone.py:70) and spends it on `embed_w` and `head_w`, 2.37 GiB each at
# real geometry -- the largest pair in the model. `build_serving_shape_ir`
# does not call it today, so nothing leaked; it is listed because the next
# path that does must not have to remember.
_C_MODULES = (qgdn, qattn, qhc, qple, pwe, qbb)
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
# The n-gram table as PORTS, and the gather that indexes them
# --------------------------------------------------------------------------

def ngram_table_ports(n_rows, row_bytes, cap_bytes=NGRAM_CHUNK_CAP_BYTES):
    """One u8 Parameter per chunk of `ngram_table_chunks`, named
    `ngram_table.K` in row order, static `[rows_K, row_bytes]`.

    STATIC on purpose, both dims. A static input is what the GPU plugin defers
    allocating until a tensor is set (`allocate_inputs`: "Reserve a null slot;
    materialized lazily or replaced by set_tensor()") -- a dynamic one is
    allocated eagerly at request creation. And the static row count is the
    contract the served runtime reads the partition off: the port shapes ARE
    the chunk table, no config carries it.
    """
    ports = []
    for k, rows in enumerate(ngram_table_chunks(n_rows, row_bytes, cap_bytes)):
        p = op.parameter([int(rows), int(row_bytes)], NGRAM_PORT_TYPE)
        p.set_friendly_name(f"ngram_table.{k}")
        p.output(0).set_names({f"ngram_table.{k}"})
        ports.append(p)
    return ports


def ngram_chunked_gather(chunk_ids, local_ids, ports):
    """`[1, T, Hn]` i32 chunk ids + `[1, T, Hn]` i64 local row ids over the
    chunked table -> `[1, T, Hn, row_bytes]` f32: the gathered rows' BYTES
    (0..255, exact in f32), the same rows a Gather over the whole
    `[n_rows, row_bytes]` table would produce. The format is decoded by
    `ngram_dequant_iq4nl`, kept separate so the gather can be probed on a
    card with arbitrary sentinel bytes and the decode gated on CPU against
    the gguf package's own dequantiser.

    NO ARITHMETIC ON THE INDEX PATH. The first form of this function took the
    GLOBAL row id and decomposed it in the graph with i32 Divide / Multiply /
    Subtract. On the A770 that gathered the WRONG ROW for every id that is
    not exactly representable in f32 -- 240 of 240 probed rows agreed with
    that predicate, none with any other (window-050 §4.7, Q2): the GPU plugin
    runs integer eltwise arithmetic in f32, exact only below 2**24, and the
    table has 320,001,536 rows. Measured the same day on the same card: a
    Gather with i64 indices and a Select carrying them are EXACT at every
    row of a 4 GiB chunk (probe modes `select` and `direct`, 0 wrong of 80,
    at one and at four chunks). So the decomposition moved to the host, where
    the hash already lives (src/exec/ngram_row_ids.h; the same boundary
    q4e.ple's header drew for the hash after the CPU's i64 arithmetic was
    measured broken), and the ids pass through Equal, Select and Gather only
    -- the ops measured exact. (The fallback zero is a Broadcast over
    ShapeOf(local_ids): it reads the port's shape, not one id.)

    Every chunk is gathered, at its own local row where the id lands in it and
    at row 0 otherwise, so no Gather ever sees an out-of-range index
    (OpenVINO's Gather does not throw for one; it reads something). One Select
    per chunk then keeps the row of the chunk the id named. The cost is
    `len(ports)` gathers of T x Hn rows each -- bytes, not the table.
    """
    i32 = lambda v: op.constant(np.array(v, np.int32))

    # the fallback index for the chunks the id does not name: a zero
    # BROADCAST to the id port's shape -- ShapeOf reads the shape, never the
    # ids. The first form multiplied the id tensor by zero: inert, but an
    # eltwise op on the index path that the record said carried none (review
    # of d30db36, F1 rider); the second was a static constant, which a graph
    # dynamic in T cannot have.
    zero = op.broadcast(op.constant(np.array(0, np.int64)),
                        op.shape_of(local_ids, output_type="i64"))
    picked = None
    for k, port in enumerate(ports):
        here = op.equal(chunk_ids, i32(k))                     # [1,T,Hn] bool
        idx = op.select(here, local_ids, zero)
        rows = op.gather(port, idx, op.constant(np.array(0, np.int64)))
        rows.set_friendly_name(f"ple/ngram_gather.{k}")       # [1,T,Hn,row_bytes]
        if picked is None:
            picked = rows
        else:
            picked = op.select(op.unsqueeze(here, i32(-1)), rows, picked)

    return op.convert(picked, Type.f32)                        # bytes as f32


def ngram_dequant_iq4nl(row_bytes_f32, head_dim):
    """`[1, T, Hn, row_bytes]` f32 (bytes 0..255) -> `[1, T, Hn, head_dim]`
    f32: ggml's IQ4_NL dequantisation, `d * kvalues[q]` per element, done
    with ops that are EXACT on this card class (window-050 §4.7: integer
    eltwise runs in f32, so every intermediate here is an integer below 2**24
    or a power of two, and the codebook and the exponent are Gathers on
    constants rather than arithmetic):

      * the block splits into d_lo, d_hi (the f16 scale's two bytes) and 16
        nibble bytes; nibbles come out as floor(b/16) and b - 16*floor(b/16);
      * the f16 is rebuilt from its bit fields -- sign, 5-bit exponent, 10-bit
        mantissa, subnormals included -- as sign * 2**(e-15) * (1 + m/1024),
        the power of two gathered from a 32-entry table, so `d` is bit-exact;
      * kvalues[q] is a Gather on the 16-entry codebook with the nibble
        converted to i32 (Convert is exact for 0..15).

    The product of an f16-exact scale and a codebook integer is exact in
    f32, so the result equals gguf-py's `dequantize(raw, IQ4_NL)` bit for
    bit; the contract cell asserts equality, not tolerance.
    """
    f32 = lambda v: op.constant(np.array(v, np.float32))
    i64 = lambda v: op.constant(np.array(v, np.int64))
    nb = head_dim // NGRAM_BLOCK_ELEMS
    # [..., nb, 18]: 18 = 2 scale bytes + 16 nibble bytes
    blocks = op.reshape(row_bytes_f32, i64([0, 0, 0, nb, NGRAM_BLOCK_BYTES]),
                        special_zero=True)
    d_lo = op.gather(blocks, i64(0), i64(-1))                  # [..., nb]
    d_hi = op.gather(blocks, i64(1), i64(-1))
    qs = op.slice(blocks, i64([2]), i64([NGRAM_BLOCK_BYTES]), i64([1]), i64([-1]))
    hi = op.floor(op.divide(qs, f32(16.0)))
    lo = op.subtract(qs, op.multiply(hi, f32(16.0)))
    nibbles = op.concat([lo, hi], axis=-1)                     # [..., nb, 32]
    kv = op.gather(f32(NGRAM_IQ4NL_KVALUES), op.convert(nibbles, Type.i32),
                   i64(0))                                     # codebook
    # the f16 scale from its bits: bits = lo + 256*hi (< 65536, exact)
    bits = op.add(d_lo, op.multiply(d_hi, f32(256.0)))
    sign_bit = op.floor(op.divide(bits, f32(32768.0)))         # 0 or 1
    rest = op.subtract(bits, op.multiply(sign_bit, f32(32768.0)))
    exp = op.floor(op.divide(rest, f32(1024.0)))               # 0..31
    mant = op.subtract(rest, op.multiply(exp, f32(1024.0)))    # 0..1023
    # 2**(e-15) for e = 0..31, with e = 0 (subnormal) mapped to 2**-14 and
    # the mantissa then taken WITHOUT the implicit one
    pow_table = [2.0 ** (e - 15) if e > 0 else 2.0 ** -14 for e in range(32)]
    scale = op.gather(f32(pow_table), op.convert(exp, Type.i32), i64(0))
    normal = op.convert(op.greater(exp, f32(0.0)), Type.f32)   # 1 if e > 0
    frac = op.add(normal, op.divide(mant, f32(1024.0)))        # 1.m or 0.m
    sign = op.subtract(f32(1.0), op.multiply(sign_bit, f32(2.0)))
    d = op.multiply(op.multiply(sign, scale), frac)            # [..., nb]
    vals = op.multiply(kv, op.unsqueeze(d, i64(-1)))           # [..., nb, 32]
    return op.reshape(vals, i64([0, 0, 0, head_dim]), special_zero=True)


# --------------------------------------------------------------------------
# The tiled MoE layer, expert bodies slot-referenced
# --------------------------------------------------------------------------

class ExpertPortSink:
    """SEGMENTED FORWARD (0.5.1): the expert bodies as PORTS instead of
    Constants. Every `_compressed_expert` built with a sink declares its
    packed u4 codes as a u8 Parameter `<name>/weight_u8` of shape
    [E, out, groups, group_size/2] -- the same bytes `pack_u4` writes, two
    codes a byte, even index in the low nibble -- and unpacks them in-graph
    (`_unpack_u8_to_u4_f32`). The sink collects the Parameters (the model
    declares them) and the bytes the artifact writer streams to disk, so the
    runtime can bind one segment's bodies at a time from host memory instead
    of the compile staging all 144 of them at once (window-050 §4.10).

    The zero-points and scales stay Constants: 3.3 MB and 26 MB a body
    against 400 MiB of codes."""

    def __init__(self, writer=None):
        self.params = []                  # the Parameters, in emission order
        self.bodies = []                  # (port name, shape, np.uint8 bytes or None)
        # `writer(name, packed_u8)` streams a body to disk as it is produced
        # and returns whatever the artifact wants recorded (an offset); with
        # a writer the bytes are NOT retained -- 36 bodies of 400 MiB would
        # be 14 GiB of host memory for one 12-layer segment otherwise
        self.writer = writer
        self.written = []                 # (port name, shape, nbytes, writer's return)

    def declare(self, name, shape, packed):
        p = op.parameter(list(shape), Type.u8)
        p.set_friendly_name(name)
        p.output(0).set_names({name})
        self.params.append(p)
        shp = tuple(int(s) for s in shape)
        if packed is not None:
            packed = np.ascontiguousarray(packed, dtype=np.uint8)
            assert packed.size == int(np.prod(shp)), (
                f"{name}: {packed.size} packed bytes for a port of {shp}")
        if self.writer is not None and packed is not None:
            self.written.append((name, shp, int(packed.size), self.writer(name, packed)))
            self.bodies.append((name, shp, None))
        else:
            self.bodies.append((name, shp, packed))
        return p


def _unpack_u8_to_u4_f32(packed, e, out, groups, gs):
    """[E, out, groups, gs/2] u8 -> [E, out, groups, gs] f32 of the codes,
    even index from the low nibble, odd from the high -- `pack_u4`'s layout,
    which is also the C++ unpack (expert_fill.unpack_u4, gguf_repack.cpp:225).
    Arithmetic in f32, no bitwise op: the GPU plugin runs integer eltwise in
    f32 anyway (memory: GPU plugin graph contracts), and 0..255 is exact."""
    x = op.convert(packed, Type.f32)
    hi = op.floor(op.multiply(x, op.constant(np.array(1.0 / 16.0, np.float32))))
    lo = op.subtract(x, op.multiply(hi, op.constant(np.array(16.0, np.float32))))
    both = op.concat([op.unsqueeze(lo, op.constant(np.array(-1, np.int64))),
                      op.unsqueeze(hi, op.constant(np.array(-1, np.int64)))], axis=-1)
    return op.reshape(both, op.constant(np.array([e, out, groups, gs], np.int64)),
                      special_zero=False)


def _compressed_expert(arena, e, out, inn, name, filler=None,
                       layer=None, kind=None, port_sink=None):
    """One expert-stacked weight in the tiled lowering's shape.

    rank-4 [E, out, groups, group_size] u4 Constant
      -> Convert(f16) -> Subtract(Convert(u4 zero_point -> f16))
      -> Multiply(f16 scale Constant) -> Reshape(rank 4 -> 3) [E, out, inn]
      -> Convert(f32)
    (the fusing control's chain, since 2026-09-17; the PORTED route below
    keeps the f32 arithmetic it was measured with, and no trailing Convert)

    The trailing Reshape is not cosmetic: verify_moe_lowering.py:33-42 records
    a real GPU compile crashing inside the fusing pass's own rewrite when it
    was absent, "because the pass's matcher anchors on that Reshape node".

    WITHOUT a `filler` this is the STRUCTURE emission: unwritten (zero) pages
    behind the u4 constants and a materialised `np.ones` scale -- the artifact
    198b736 measured at 183.07 GiB declared over 0 KiB of disk.

    WITH a `filler` (q4e.expert_fill.ExpertFiller) the same shapes carry the
    real checkpoint: the filler returns packed u4 codes, packed u4
    zero-points and f32 scales for (layer, kind) -- the scales are carried
    as f16 Constants, the exact f32 stays in `arena.scales` -- and each of
    the three constants is built OVER pages written first. The graph is structurally
    identical either way -- same ops, shapes and element types -- which is
    what lets the empty build's contract test speak for the filled one.
    """
    gs = EXPERT_GROUP_SIZE
    assert inn % gs == 0, f"{name}: inner {inn} is not a multiple of group {gs}"
    groups = inn // gs
    pw = pzp = sc = None
    if filler is not None:
        pw, pzp, sc = filler.body(layer, kind, e, out, inn)
        assert sc.shape == (e, out, groups, 1), (
            f"{name}: filler returned scales {sc.shape}, the constant is "
            f"{(e, out, groups, 1)}")
    # THE DEQUANT CHAIN'S TYPE IS f16 WITH A TRAILING Convert TO f32 -- the
    # fusing 35B control's exact shape (walked node by node 2026-09-17), and
    # not a cosmetic choice: under the plugin's f16 inference precision an
    # f32 scale Constant feeding the fused MOECompressed gets a Convert
    # inserted by KeepConstantsPrecisionAndAddConverts, and the offload
    # series' OTD resolver (moe.cpp, patch 0005 on) demands direct Constants:
    # "Expected constant input for MOE3GemmFusedCompressed, got: Convert"
    # (census 2, B60, 2026-09-17). An f16 scale needs no Convert. The PORTED
    # chain (dead route) keeps its f32 arithmetic; the unpack cell compares
    # each against its own reference.
    ct = Type.f32 if port_sink is not None else Type.f16
    if port_sink is not None:
        # SEGMENTED: the codes are a u8 PORT, bound by the runtime; only the
        # zero-points and scales are constants of this segment's graph
        w = port_sink.declare(name + "/weight_u8", [e, out, groups, gs // 2], pw)
        x = _unpack_u8_to_u4_f32(w, e, out, groups, gs)
    elif filler is None:
        w = arena.constant([e, out, groups, gs], EXPERT_DECLARED_TYPE)
        w.set_friendly_name(name + "/weight_u4")
        x = op.convert(w, ct)
    else:
        w = arena.constant([e, out, groups, gs], EXPERT_DECLARED_TYPE,
                           fill=pw, name=name + "/weight_u4")
        w.set_friendly_name(name + "/weight_u4")
        x = op.convert(w, ct)
    if filler is None:
        zp = arena.constant([e, out, groups, 1], EXPERT_DECLARED_TYPE)
        scale = op.constant(np.ones((e, out, groups, 1),
                                    np.float32 if ct == Type.f32 else np.float16))
    else:
        zp = arena.constant([e, out, groups, 1], EXPERT_DECLARED_TYPE,
                            fill=pzp, name=name + "/zero_point")
        if ct == Type.f32:
            scale = arena.f32_filled(sc)
            arena.scales[name + "/scale"] = sc
        else:
            sc16 = np.ascontiguousarray(sc, dtype=np.float16)
            scale = arena.constant(list(sc16.shape), Type.f16, fill=sc16,
                                   name=name + "/scale")
            # the EXACT scale the filler quantised with (tests dequantise the
            # codes against it); the artifact carries its f16 rounding
            arena.scales[name + "/scale"] = sc
    zp.set_friendly_name(name + "/zero_point")
    scale.set_friendly_name(name + "/scale")
    x = op.subtract(x, op.convert(zp, ct))
    x = op.multiply(x, scale)
    x = op.reshape(x, op.constant(np.array([e, out, inn], np.int64)),
                   special_zero=False)
    x.set_friendly_name(name + "/dequant_reshape")
    if ct != Type.f32:
        x = op.convert(x, Type.f32)
        x.set_friendly_name(name + "/dequant_f32")
    return x


def swish1(x):
    """Swish with ONE input. The Python binding's `op.swish(x)` appends a
    beta Constant (1.0) as a second input; the plugin's tiled MoE matcher
    declares `Swish({gate_matmul})` with one input and the C++ Matcher
    rejects a node whose argument count differs (measured 2026-09-17: the
    fusing 35B control carries Swish/opset4 in=1, this emitter carried in=2,
    and the census stayed at 0 MoE primitives with the Reshapes in place)."""
    s = op.swish(x)
    s.set_arguments([s.input_value(0)])          # drop the beta the binding added
    s.validate_and_infer_types()
    return s


def _native_expert(arena, e, out, inn, name, fmt, parts):
    """One expert-stacked weight as the CHECKPOINT'S OWN BLOCKS, decoded in
    standard ops (design-routing-aware-expert-execution 2.3a/2.3b, DESIGN
    7.0.2bz): the u4 grouped-affine repack of the IQ3_XXS / IQ4_NL experts
    costs 0.10-0.13 relative RMS per tensor and 0.73 nats at depth 48, so the
    experts are carried as q4e.native_blocks lays them out per role, and the
    decode is expressed in ops the CPU plugin runs exactly (the suite's
    oracle) and the GPU plugin's matcher will lower to native kernels:

      IQ4_NL   codes u4 [E,out,inn/32,32] -> Convert(i32) -> Gather(table[16] f32)
               * Convert(f32)(scales f16 [E,out,inn/32,1]) -> Reshape [E,out,inn]
      IQ4_XS   the IQ4_NL chain over the IQ4_NL layout (sub-block scales folded
               into the f32 scale by the split)
      Q8_0     codes i8 [E,out,inn/32,32] -> Convert(f32)
               * Convert(f32)(scales f16) -> Reshape [E,out,inn]
      IQ3_XXS  gridix u8 [E,out,inn/32,8] -> Convert(i32) -> Gather(grid[256,4])
               -> Reshape [E,out,inn/32,32]  (the magnitudes)
               signix u8 [E,out,inn/32,4] -> Convert(i32) -> Gather(ksigns[128])
               -> Unsqueeze -> BitwiseAnd(masks[8]) -> Greater(0)
               -> Select(-1, +1) -> Reshape [E,out,inn/32,32]  (the signs)
               magnitudes * signs * Convert(f32)(scales f16) -> Reshape [E,out,inn]

    THE CONSTANTS' SHAPES ARE THE FUSED OP'S OWN (design note 2.3c): the
    plugin's MOECompressed takes every expert weight as rank-4 [E, out,
    groups, group_size] with a scale [E, out, groups, 1] and an optional
    zero-point [E, out, groups, 1]. Both formats fit at group 32 -- IQ4_NL
    as u4 codes plus a table and no zero-point; IQ3_XXS as 8 grid indices
    per 32 values in the weight slot, 4 sign indices per 32 values in the
    zero-point slot, and the per-32 scale -- so the plugin patch lowers
    these Constants as they are, and the offload path copies one expert's
    bytes exactly as it does for the u4 route.

    The arithmetic is f32 (exact against numpy's decode); the fusing u4 chain
    is f16 with a trailing Convert because its matcher demands direct f16
    Constants, and the native matcher variants are the plugin patch's to
    define. Until that patch, the GPU plugin would constant-fold these
    chains into dense f16 weights (10 GiB per layer), so a native artifact
    is measured on the CPU plugin at small geometry and, at depth, through
    the served binary once the patch exists.

    `parts` are q4e.native_blocks' per-role arrays over [e*out, ...] rows.
    Returns the [E,out,inn] f32 node.
    """
    from q4e import native_blocks as nb
    from q4e.expert_fill import pack_u4
    rows = e * out
    groups = inn // 32
    g4 = op.constant(np.array([e, out, groups, 32], np.int64))
    if fmt == "IQ2_S_PACKED":
        w80, d = parts
        return _native_packed_expert(arena, e, out, inn, name, w80, d)
    if fmt in ("IQ4_NL", "IQ4_XS"):
        # IQ4_XS splits to the IQ4_NL layout (its 6-bit sub-block scales are
        # folded into the per-32 f32 scale, native_blocks.iq4_xs_split), so
        # the chain -- and the plugin's lowering -- are the IQ4_NL ones
        codes, scales = parts
        assert codes.shape == (rows, inn) and scales.shape == (rows, groups), (codes.shape, scales.shape)
        w = arena.constant([e, out, groups, 32], Type.u4, fill=pack_u4(codes), name=name + "/codes_u4")
        w.set_friendly_name(name + "/codes_u4")
        table = op.constant(nb.KVALUES_IQ4NL.astype(np.float32))
        x = op.gather(table, op.convert(w, Type.i32), op.constant(np.int64(0)))     # [E,out,groups,32]
        x.set_friendly_name(name + "/iq4nl_table")
    elif fmt == "IQ3_XXS":
        gridix, signix, scales = parts
        assert gridix.shape == (rows, inn // 4) and signix.shape == (rows, inn // 8), (gridix.shape, signix.shape)
        assert scales.shape == (rows, groups), scales.shape
        gi = arena.constant([e, out, groups, 8], Type.u8, fill=np.ascontiguousarray(gridix, np.uint8),
                            name=name + "/gridix_u8")
        gi.set_friendly_name(name + "/gridix_u8")
        grid = op.constant(nb.IQ3XXS_GRID.astype(np.float32))                      # [256, 4]
        mag = op.gather(grid, op.convert(gi, Type.i32), op.constant(np.int64(0)))    # [E,out,groups,8,4]
        mag = op.reshape(mag, g4, special_zero=False)
        mag.set_friendly_name(name + "/iq3xxs_grid")
        si = arena.constant([e, out, groups, 4], Type.u8, fill=np.ascontiguousarray(signix, np.uint8),
                            name=name + "/signix_u8")
        si.set_friendly_name(name + "/signix_u8")
        ks = op.constant(nb.KSIGNS_IQ2XS.astype(np.int32))                          # [128]
        masks = op.gather(ks, op.convert(si, Type.i32), op.constant(np.int64(0)))    # [E,out,groups,4]
        bits = op.bitwise_and(op.unsqueeze(masks, op.constant(np.int64(-1))),
                              op.constant(nb.KMASK_IQ2XS.astype(np.int32)))         # [E,out,groups,4,8]
        neg = op.greater(bits, op.constant(np.int32(0)))
        sign = op.select(neg, op.constant(np.float32(-1.0)), op.constant(np.float32(1.0)))
        sign = op.reshape(sign, g4, special_zero=False)
        sign.set_friendly_name(name + "/iq3xxs_sign")
        x = op.multiply(mag, sign)
    elif fmt == "IQ2_S":
        # IQ2_S (ggml type 22): four 10-bit grid indices per 32 values, one raw
        # sign byte per 8 values, and TWO 4-bit sub-block scales per 32 (the low
        # nibble serves values 0..15, the high nibble 16..31). The weight slot is
        # the u8 [E,out,K/32,8] patch 0043/0045's fourth format reads: the four
        # indices as little-endian u16, two bytes each. The scale slot is the
        # compact [E,out,K/32,2] and is EXPANDED in-graph to [E,out,K/32,32] (lo
        # repeated 16, hi repeated 16) -- the constant stays 4 B per 32 values,
        # not the 64 B an expanded constant would cost.
        gridix, signix, scales8 = parts
        assert gridix.shape == (rows, inn // 8) and gridix.dtype == np.uint16, (
            gridix.shape, gridix.dtype)
        assert signix.shape == (rows, inn // 8) and scales8.shape == (rows, inn // 8), (
            signix.shape, scales8.shape)
        gi = np.ascontiguousarray(
            gridix.reshape(e, out, groups, 4).astype("<u2")).view(np.uint8).reshape(
            e, out, groups, 8)
        w = arena.constant([e, out, groups, 8], Type.u8, fill=gi, name=name + "/gridix_u8")
        w.set_friendly_name(name + "/gridix_u8")
        w5 = op.reshape(w, op.constant(np.array([e, out, groups, 4, 2], np.int64)),
                        special_zero=False)
        lo = op.gather(w5, op.constant(np.array(0, np.int64)), op.constant(np.int64(-1)))
        hi = op.gather(w5, op.constant(np.array(1, np.int64)), op.constant(np.int64(-1)))
        idx = op.add(op.convert(lo, Type.i32),
                     op.multiply(op.convert(hi, Type.i32),
                                 op.constant(np.array(256, np.int32))))
        idx.set_friendly_name(name + "/iq2s_index")
        grid = op.constant(nb.IQ2S_GRID.astype(np.float32))                        # [1024, 8]
        mag = op.gather(grid, idx, op.constant(np.int64(0)))                       # [E,out,g,4,8]
        mag = op.reshape(mag, g4, special_zero=False)
        mag.set_friendly_name(name + "/iq2s_grid")
        # IQ2_S signs are the RAW byte: bit j flips value j (no 7-bit table
        # index, unlike IQ3_XXS)
        si = arena.constant([e, out, groups, 4], Type.u8,
                            fill=np.ascontiguousarray(signix.reshape(e, out, groups, 4)),
                            name=name + "/signix_u8")
        si.set_friendly_name(name + "/signix_u8")
        bits = op.bitwise_and(op.unsqueeze(op.convert(si, Type.i32), op.constant(np.int64(-1))),
                              op.constant(nb.KMASK_IQ2XS.astype(np.int32)))        # [E,out,g,4,8]
        neg = op.greater(bits, op.constant(np.int32(0)))
        sign = op.reshape(op.select(neg, op.constant(np.float32(-1.0)),
                                    op.constant(np.float32(1.0))),
                          g4, special_zero=False)
        sign.set_friendly_name(name + "/iq2s_sign")
        x = op.multiply(mag, sign)
        s4 = scales8.reshape(e, out, groups, 4)
        sc2_f32 = np.stack([s4[..., 0], s4[..., 2]], axis=-1).astype(np.float32)   # lo, hi
        sc2 = arena.constant([e, out, groups, 2], Type.f16,
                             fill=np.ascontiguousarray(sc2_f32, np.float16),
                             name=name + "/block_scale")
        sc2.set_friendly_name(name + "/block_scale")
        arena.scales[name + "/block_scale"] = sc2_f32
        sc5 = op.reshape(op.convert(sc2, Type.f32),
                         op.constant(np.array([e, out, groups, 1, 2], np.int64)),
                         special_zero=False)
        sc_rep = op.broadcast(sc5,
                              op.constant(np.array([e, out, groups, 16, 2], np.int64)))
        sc32 = op.reshape(sc_rep, g4, special_zero=False)
        sc32.set_friendly_name(name + "/iq2s_scale32")
        x = op.multiply(x, sc32)
        x = op.reshape(x, op.constant(np.array([e, out, inn], np.int64)), special_zero=False)
        x.set_friendly_name(name + "/native_f32")
        return x
    elif fmt == "Q8_0":
        codes, scales = parts
        assert codes.shape == (rows, inn) and codes.dtype == np.int8 and scales.shape == (rows, groups), (
            codes.shape, codes.dtype, scales.shape)
        w = arena.constant([e, out, groups, 32], Type.i8, fill=np.ascontiguousarray(codes, np.int8),
                           name=name + "/codes_i8")
        w.set_friendly_name(name + "/codes_i8")
        x = op.convert(w, Type.f32)
    else:
        raise ValueError(f"{name}: unsupported native expert format {fmt!r} "
                         f"({sorted(nb.SPLIT)})")
    # the block scale is an f16 Constant like every stock scale: an f32 scale
    # Constant is wrapped in a Convert(f16) by the plugin's precision pass,
    # which the offload series cannot fold (file-backed Constants), and the
    # op translation then refuses the non-Constant input (measured on GPU.0,
    # 2026-09-18). IQ4_NL's and Q8_0's d IS an f16, so those stay exact;
    # IQ3_XXS's d*(0.5+s)*0.5 and IQ4_XS's d*(ls-32) round once to f16
    # (<= 2^-11 relative, test_native_expert_chain). The exact f32 stays in
    # arena.scales; the chain's arithmetic stays f32 (Convert after the
    # Constant), which the plugin's pattern accepts as an optional Convert.
    sc16 = np.ascontiguousarray(scales, np.float16).reshape(e, out, groups, 1)
    sc = arena.constant([e, out, groups, 1], Type.f16, fill=sc16, name=name + "/block_scale")
    sc.set_friendly_name(name + "/block_scale")
    arena.scales[name + "/block_scale"] = np.ascontiguousarray(scales, np.float32)
    x = op.multiply(x, op.convert(sc, Type.f32))
    x = op.reshape(x, op.constant(np.array([e, out, inn], np.int64)), special_zero=False)
    x.set_friendly_name(name + "/native_f32")
    return x


def _native_packed_expert(arena, e, out, inn, name, w80, d):
    """IQ2_S-PACKED (patch 0052). The checkpoint's own 82-byte ggml block per
    256 values, kept VERBATIM: 32 B qs (four 2-bit low index bytes per ib32) |
    32 B RAW sign masks | 8 B qh (two high index bits per l) | 8 B four-bit
    sub-block scales, with the f16 d lifted into the scale slot. The weight
    Constant is [E, out, K/256, 80] u8 -- 82 B per 256 against the re-laid
    IQ2_S's 128 B and the u4 repack's 144 B, the GGUF's own size (design note
    12.4: 10.35 GiB of experts against 14.47 re-laid). The decode is
    arithmetic in f32 (the same style as _unpack_u8_to_u4_f32): idx =
    qs + 256*((qh >> 2l) & 3), y = d*(0.5+nib)*0.25 * grid[idx] * sign, the
    low nibble for l<2 and the high for l>=2. Bit-exact against
    native_blocks.iq2_s_decode.

    The arithmetic is the CPU oracle; the plugin's matcher replaces the whole
    chain, reading only the two Constants (the packed u8 weight and the f16 d).
    """
    from q4e import native_blocks as nb
    rows = e * out
    nblk = inn // 256
    assert inn % 256 == 0, (inn, "IQ2_S-packed needs a multiple of 256")
    assert w80.shape == (rows, nblk * 80) and d.shape == (rows, nblk), (w80.shape, d.shape)
    f32 = lambda v: op.constant(np.array(v, np.float32))
    i64 = lambda v: op.constant(np.array(v, np.int64))

    w = arena.constant([e, out, nblk, 80], Type.u8,
                       fill=np.ascontiguousarray(w80.reshape(e, out, nblk, 80)),
                       name=name + "/iq2s_packed_u8")
    w.set_friendly_name(name + "/iq2s_packed_u8")

    def sl(a, b):
        return op.slice(w, i64([a]), i64([b]), i64([1]), i64([-1]))

    qs = op.convert(op.reshape(sl(0, 32), i64([e, out, nblk, 8, 4]), special_zero=False), Type.f32)
    # the sign bytes carry (nblk, ib32) fused into ONE axis: the IQ2_S chain's
    # signs Select is rank 5 [.,4,8] and the GPU plugin's layout optimizer has
    # no layout for a rank-6 one (measured 2026-09-26, add_required_reorders
    # :342 on packed shape [256,512,8,8,4,8]). The fused axis flattens to the
    # SAME element order (nblk*256 + ib32*32 + l*8 + m), so the bytes and the
    # decode are unchanged.
    sg = op.convert(op.reshape(sl(32, 64), i64([e, out, nblk * 8, 4]), special_zero=False), Type.f32)
    qh = op.convert(op.reshape(sl(64, 72), i64([e, out, nblk, 8]), special_zero=False), Type.f32)
    scb = op.convert(op.reshape(sl(72, 80), i64([e, out, nblk, 8]), special_zero=False), Type.f32)

    # high_l = (qh >> 2l) & 3 in f32 arithmetic: a per-l divisor broadcasts
    # over the last axis, so no shifts and no Concat -- floor(qh/[1,4,16,64])
    # then mod 4 (floor(f/4) subtracted back)
    qh_u = op.unsqueeze(qh, i64([-1]))                              # [E,out,nblk,8,1]
    qh_f = op.floor(op.divide(qh_u, f32([1.0, 4.0, 16.0, 64.0])))
    high = op.subtract(qh_f, op.multiply(op.floor(op.divide(qh_f, f32(4.0))), f32(4.0)))
    idx = op.add(qs, op.multiply(high, f32(256.0)))
    idx.set_friendly_name(name + "/iq2s_packed_index")

    grid = op.constant(nb.IQ2S_GRID.astype(np.float32))            # [1024,8]
    mag = op.gather(grid, op.convert(idx, Type.i32), i64(0))       # [E,out,nblk,8,4,8]
    mag = op.reshape(mag, i64([e, out, nblk, 256]), special_zero=False)
    mag.set_friendly_name(name + "/iq2s_packed_grid")

    # IQ2_S signs are the RAW byte: bit j flips value j
    bits = op.bitwise_and(op.unsqueeze(op.convert(sg, Type.i32), i64([-1])),
                          op.constant(nb.KMASK_IQ2XS.astype(np.int32)))   # [E,out,nblk*8,4,8] rank 5
    neg = op.greater(bits, op.constant(np.int32(0)))
    sign = op.reshape(op.select(neg, f32(-1.0), f32(1.0)),
                      i64([e, out, nblk, 256]), special_zero=False)
    sign.set_friendly_name(name + "/iq2s_packed_sign")
    x = op.multiply(mag, sign)

    # a 32-value group carries two 4-bit scales: the low nibble for l=0,1 and
    # the high for l=2,3 (ggml-quants.c dequantize_row_iq2_s). Same trick: the
    # per-l divisor [1,1,16,16] broadcasts, then mod 16.
    scb_u = op.unsqueeze(scb, i64([-1]))                            # [E,out,nblk,8,1]
    sc_f = op.floor(op.divide(scb_u, f32([1.0, 1.0, 16.0, 16.0])))
    nib = op.subtract(sc_f, op.multiply(op.floor(op.divide(sc_f, f32(16.0))), f32(16.0)))
    sc32 = op.reshape(op.broadcast(op.reshape(nib, i64([e, out, nblk, 8, 4, 1]), special_zero=False),
                                   i64([e, out, nblk, 8, 4, 8])),
                      i64([e, out, nblk, 256]), special_zero=False)
    sc32.set_friendly_name(name + "/iq2s_packed_scale32")

    dd16 = np.ascontiguousarray(d, np.float16).reshape(e, out, nblk, 1)
    dd = arena.constant([e, out, nblk, 1], Type.f16, fill=dd16, name=name + "/block_scale")
    dd.set_friendly_name(name + "/block_scale")
    arena.scales[name + "/block_scale"] = np.ascontiguousarray(d, np.float32).reshape(e, out, nblk)
    # ggml's own association: d * (0.5 + s) * 0.25, left to right
    scale = op.multiply(op.multiply(op.convert(dd, Type.f32), op.add(f32(0.5), sc32)),
                        f32(0.25))
    x = op.multiply(x, scale)
    x = op.reshape(x, i64([e, out, inn]), special_zero=False)
    x.set_friendly_name(name + "/native_f32")
    return x


def emit_moe_tiled(hidden_bth, config, state, arena, T, tag, filler=None,
                   layer=None, port_sink=None):
    """The MoE layer in the shape the GPU plugin's
    ConvertTiledMoeBlockTo3GatherMatmuls matcher accepts (export_mtp.py:401
    moe_block_tiled, walked node by node against the pattern source), at real
    geometry, expert bodies slot-referenced. Returns a [1,T,H] node.

    "Measured to fuse" was inherited from the MTP exporter, not re-measured
    here, and until 2026-09-17 this function deviated from it in the two
    Reshapes the matcher anchors on (see the comment at the mixing stage).
    The contract cell is `test_every_moe_layer_walks_the_plugins_tiled_3gemm_pattern`;
    the compile that proves it is a card window.

    Contract: `hidden_bth` is rank 3 with dim 0 the batch, statically 1 (the
    mixing stage reads B off its ShapeOf and the shared-expert Add relies on
    it). With a `port_sink` the expert bodies are Parameters, and the matcher's
    CompressedWeightsBlock anchors on a Constant: a ported build cannot fuse
    by construction, and the contract cell covers the Constant build only."""
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

    def _expert(kind, out_, inn_):
        nm = f"{tag}/experts_{kind}"
        if filler is not None and hasattr(filler, "native"):
            # the checkpoint's own blocks, decoded in ops (2026-09-18)
            fmt, parts = filler.native(layer, kind, E, out_, inn_)
            return _native_expert(arena, E, out_, inn_, nm, fmt, parts)
        return _compressed_expert(arena, E, out_, inn_, nm, filler, layer, kind, port_sink)

    gate_w = _expert("gate", I, H)
    up_w = _expert("up", I, H)
    down_w = _expert("down", H, I)

    g = swish1(op.matmul(m_h3, gate_w, transpose_a=False, transpose_b=True))
    u = op.matmul(m_h3, up_w, transpose_a=False, transpose_b=True)
    outs = op.matmul(op.multiply(g, u), down_w,
                     transpose_a=False, transpose_b=True)              # [E,M,H]

    # THE TWO RESHAPES THE MATCHER ANCHORS ON. build_3gemm_pattern() in the
    # plugin's convert_tiled_moe_block_to_gather_matmuls.cpp wants
    # `end_reshape` = Reshape(down_matmul) and `router_reshape` =
    # Reshape(Transpose(scatter)) -> optional Unsqueeze, both feeding the
    # router-weight Multiply. This emitter dropped both until 2026-09-17: the
    # constraint walker (tools/check_tiled_pattern.py) failed every MoE
    # candidate of the depth-12 artifact at R4.router_reshape.type and its
    # compiled graph carried 0 MoE-typed primitives, 230 FullyConnected, every
    # expert computed for every token. Construction as export_mtp.py:532-537:
    # B read from ShapeOf, S a runtime -1 -- a target whose every dim is known
    # is folded away at validate/save on 2026.4.0 (export_mtp.py:515-531).
    # Here B is statically 1 (asserted), so the targets are the LITERALS
    # [E,1,-1,H] and [E,1,-1] -- the same construction tools/
    # moe_tiled_rewrite.py used for every card census on the record (12
    # fused primitives, 3.00 GiB, the served legs of 2026-09-17); the -1
    # alone keeps the Reshape alive (M stays dynamic). export_mtp's B is a
    # genuine runtime value and stays a ShapeOf there.
    ps = hidden_bth.output(0).get_partial_shape()
    assert (ps.rank.is_static and ps.rank.get_length() == 3
            and ps[0].is_static and ps[0].get_length() == 1), (
        f"{tag}: emit_moe_tiled wants [1,T,H], got {ps}")
    outs4 = op.reshape(outs, op.constant(np.array([E, 1, -1, H], np.int32)),
                       special_zero=False)                               # [E,1,S,H]
    wt = op.transpose(weights, op.constant(np.array([1, 0], np.int32)))  # [E,M]
    wr = op.reshape(wt, op.constant(np.array([E, 1, -1], np.int32)),
                    special_zero=False)                                  # [E,1,S]
    wu = op.unsqueeze(wr, i32v(-1))                                      # [E,B,S,1]
    mixed = op.reduce_sum(op.multiply(outs4, wu), i32v(0), keep_dims=False)  # [B,S,H]
    mixed.set_friendly_name(f"{tag}/mix")      # the matcher's root, addressable

    # the shared expert (pin 986-996) stays dense f32 -- it is one MLP per
    # layer, 0.0183 GiB, and it is CARD tier in the size ledger
    shared_state = {kk[len("mlp."):]: v for kk, v in state.items()
                    if kk.startswith("mlp.shared_expert")}
    from . import moe as _moe
    sh = _moe.emit_shared_expert(
        op.reshape(y_flat, op.constant(np.array([1, -1, H], np.int64)),
                   special_zero=False),
        config, shared_state, None)                                    # [T,H]

    out2d = op.add(mixed, sh)
    return op.reshape(out2d, op.constant(np.array([1, -1, H], np.int64)),
                      special_zero=False)


# --------------------------------------------------------------------------
# The full-geometry serving-shape backbone
# --------------------------------------------------------------------------

def _fill_dense(st, feed, prefix, census):
    """Write REAL weights into the arena buffers `st` holds (feed-the-ports):
    each key becomes the pin key `prefix + key` and `feed.fitted` returns the
    dequantised GGUF tensor cut to the buffer's shape (or raises by name).
    The buffers are the arena's memmaps, so writing them is what turns an
    unwritten page into a real one -- the same mechanism as the expert fill.
    `census` collects (pin_key, bytes) for the report."""
    if feed is None:
        return
    layer = None
    if prefix.startswith("layers."):
        layer = int(prefix.split(".")[1])
    for key, buf in st.items():
        arr = feed.fitted(prefix + key, tuple(buf.shape), gguf_layer=layer)
        buf[...] = arr
        census.append((prefix + key, int(buf.nbytes)))


def _layer_state(arena, config, kind, layer=None, feed=None, census=None):
    """Sparse-declared state for one decoder layer, at the pin's own
    module-relative keys, real shapes from `config`. With `feed` the buffers
    are written from the real shards (`_fill_dense`)."""
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
    if feed is not None:
        _fill_dense(st, feed, f"layers.{layer}.", census)
    return st


def _ple_state(arena, config, layer=None, feed=None, census=None):
    H = config.hidden_size
    hc = config.hc_count
    Hn = (config.ngram_size - 1) * config.heads_per_ngram
    head_dim = config.ple_embed_dim // Hn
    st = {
        "key_proj.weight": arena.f32([hc * H, config.ple_embed_dim]),
        "value_proj.weight": arena.f32([H, config.ple_embed_dim]),
        # [hc*H] = 10240, measured; the group RMS norms span the whole
        # hyper-connection width, not one hidden width
        "norm_key.weight": arena.f32([hc * H]),
        "norm_query.weight": arena.f32([hc * H]),
        "norm_conv.weight": arena.f32([hc * H]),
        "conv1d.weight": arena.f32([hc * H, 1, config.ple_conv_kernel_size]),
    }
    if feed is not None:
        _fill_dense(st, feed, f"layers.{layer}.ple.", census)
    return st, head_dim, Hn


def build_serving_shape_ir(config=None, arena=None, n_layers=None,
                           filler=None, feed=None,
                           ngram_chunk_cap_bytes=NGRAM_CHUNK_CAP_BYTES,
                           rope_span=None, layer_range=None, expert_ports=None,
                           ngram_staging_rows=None):
    """The full-geometry serving-shape backbone as an ov::Model, DYNAMIC IN T
    (feed-the-ports increment): no port, reshape or slice carries the block
    length. `T` below is the runtime token count of a forward.

    Inputs
        inputs_embeds    [1, T, H]   f32   -- the embedded tokens. The served
                                             runtime embeds on the host
                                             (`embed_paged`) and feeds this
                                             name (backend_ov.cpp:6153); the
                                             pass rewrites the port to [-1, -1]
                                             + Unsqueeze(1), so it is fed as
                                             [T, H]. The embedding weight is
                                             NOT in this graph any more --
                                             `pwe.build_embed_piece` is the
                                             separate model, as for the served
                                             artifact.
        position_ids     [1, T]      i64
        ngram_table.K    [rows_K, 90] u8   -- the n-gram table, one port per
                                             chunk under `ngram_chunk_cap_bytes`
                                             (ngram_table_ports), bound once
                                             per request from host memory;
                                             90 = one 160-wide IQ4_NL row as
                                             the GGUF stores it (5 blocks of
                                             18 bytes), decoded in-graph
        ngram_chunk_ids  [1, T, 16]  i32   -- which `ngram_table.K` holds the
        ngram_local_ids  [1, T, 16]  i64      hashed row, and the row inside
                                             it: the global row id split by
                                             the host at the port partition
                                             (row // rows_0, row % rows_0).
                                             16 = (ngram_size-1) *
                                             heads_per_ngram, exactly
                                             `HashParams::num_ngram_heads()`
                                             (src/exec/ngram_row_ids.h:59) and
                                             the "8 x 2-gram + 8 x 3-gram" the
                                             same file's header states at :21.
                                             The hash itself is computed by
                                             `lgc::ngram::row_ids`
                                             (ngram_row_ids.h:117); the graph
                                             consumes ids, never tokens, and
                                             does NO arithmetic on them (see
                                             ngram_chunked_gather).
    Output
        logits           [1, T, vocab] f32

    `rope_span` sizes the ONE cos/sin table pair every full-attention layer
    gathers from (default `max_position_embeddings`, 262,144 at real
    geometry: ~67 MB a side, once, shared -- not per layer).

    `feed` (a `q4e.gguf_feed.GgufFeed`) writes the REAL dense weights of the
    built layers, the PLE, the final mixer and the head into the arena
    (`_fill_dense`); with `filler` for the expert bodies that is the whole
    depth-`n_layers` model at real weights. The embedding is not in this
    graph (it is fed) and the n-gram table is bound to the ports at request
    time; both come from the same shards on the driver's side.

    SEGMENTED FORWARD (0.5.1, docs/window-051.md §2). `layer_range=(lo, hi)`
    emits GLOBAL layers lo..hi-1 as ONE segment of the 48-layer model:
        segment 0 (lo == 0)      takes `inputs_embeds` [1, T, H] as today
        a later segment          takes `inputs_embeds` [1, T, hc*H] -- the
                                 hyper-connection-width hidden state the
                                 previous segment emitted (the pass looks the
                                 port up by this name, so the name stays)
        the last segment         (hi == num_hidden_layers) carries the final
                                 mixer and the head and emits `logits`
        every other segment      emits `hidden_out` [1, T, hc*H] f32
    The PLE (global layer 1) and its n-gram ports exist only in the segment
    that holds layer 1; the rope tables, `conv_mask`, `attention_mask` and
    `beam_idx` in every segment; a segment must hold at least one
    full-attention layer (index 3 mod 4) or the paged transformation refuses
    it, and this function refuses first, by name. Without `layer_range` the
    depth cut 0..n_layers-1 with the head is emitted, as before.

    `expert_ports` (an `ExpertPortSink`) turns every expert body of the
    segment into a u8 PORT (the packed codes) with an in-graph unpack; the
    sink collects the Parameters and the bytes for the artifact writer.

    Returns (model, report) where `report` carries the measured structure.
    """
    cfg = config if config is not None else pwe.real_config()
    T = -1                                 # dynamic: every reshape uses -1
    own_arena = arena is None
    ar = arena if arena is not None else SparseArena()
    n_total = int(cfg.num_hidden_layers)
    depth = int(n_layers if n_layers is not None else n_total)   # the head sits after layer depth-1
    if layer_range is None:
        lo, hi = 0, depth
        first, last = True, True           # the depth cut: head on, whatever hi
    else:
        lo, hi = (int(layer_range[0]), int(layer_range[1]))
        if not (0 <= lo < hi <= depth <= n_total):
            raise ValueError(f"layer_range {layer_range!r} outside 0..{depth} "
                             f"(depth {depth} of {n_total})")
        first, last = lo == 0, hi == depth
    if layer_range is not None and not any((i % 4) == 3 for i in range(lo, hi)):
        # a depth cut of 1..3 layers is a legitimate structure build (the
        # round-trip cell uses one); a SEGMENT without an SDPA is not servable
        raise ValueError(f"layers {lo}..{hi - 1} hold no full-attention layer "
                         f"(index 3 mod 4): SDPAToPagedAttention would refuse "
                         f"the segment; the smallest 4-aligned range is 4 layers")
    nl = hi - lo

    H = cfg.hidden_size
    hc = cfg.hc_count
    V = cfg.vocab_size
    ple_layer_idx = 1                      # GGUF ple.layers [1]; pwe REAL_GEOMETRY
    has_ple = lo <= ple_layer_idx < hi
    in_width = H if first else hc * H
    # The Assign nodes of every stateful layer. They are the model's, not a
    # layer's: `ov::Model` takes them as its own argument and a graph whose
    # state is read and never written is not stateful, it is wrong.
    sinks = []
    dense_census = []

    try:
        with shared_constants():
            inputs_embeds = op.parameter([1, T, in_width], Type.f32)
            inputs_embeds.set_friendly_name("inputs_embeds")
            inputs_embeds.output(0).set_names({"inputs_embeds"})
            pid = op.parameter([1, T], Type.i64)
            pid.set_friendly_name("position_ids")
            pid.output(0).set_names({"position_ids"})
            Hn = (cfg.ngram_size - 1) * cfg.heads_per_ngram
            head_dim = cfg.ple_embed_dim // Hn
            ple_state = chunk_ids = local_ids = None
            if has_ple:
                ple_state, head_dim, Hn = _ple_state(ar, cfg, ple_layer_idx, feed,
                                                     dense_census)
                chunk_ids = op.parameter([1, T, Hn], Type.i32)
                chunk_ids.set_friendly_name("ngram_chunk_ids")
                local_ids = op.parameter([1, T, Hn], Type.i64)
                local_ids.set_friendly_name("ngram_local_ids")
            # the GDN + PLE padding mask, same port q4e.backbone declares
            # (backbone.py:105-106); ones = full sequence
            conv_mask = op.parameter([1, T], Type.f32)
            conv_mask.set_friendly_name("conv_mask")
            # The two ports the TRANSFORMATION consumes and removes. Neither
            # survives into the compiled model -- the served artifact's
            # transformed input list has neither -- but the pass looks them up
            # by name, and `attention_mask` carries the only honest statement
            # of the total key length in a graph whose query block is static:
            # it spans past + current, so its dim 1 is TOTAL.
            attn_mask = op.parameter([1, -1], Type.i64)
            attn_mask.set_friendly_name("attention_mask")
            attn_mask.output(0).set_names({"attention_mask"})
            beam = op.parameter([-1], Type.i32)
            beam.set_friendly_name("beam_idx")
            beam.output(0).set_names({"beam_idx"})

            # the fed embedding -> repeat to the hyper-connection width
            # (backbone.py). PIN THE LAYOUT where the pass's token axis enters:
            # after `SDPAToPagedAttention` `inputs_embeds` is [-1, -1] +
            # Unsqueeze(1), i.e. [tokens, 1, H], while this graph is
            # [1, T, ...]. Same bytes; a binary op between the two BROADCASTS
            # rather than refuses (the first forward on a card died at the
            # PLE's additive join with a [5, 5, 10240] hidden, window-050
            # §4.7). The reshape folds it to [1, T, H] -- the identity
            # pre-pass, the seam post-pass.
            emb = op.reshape(inputs_embeds,
                             op.constant(np.array([1, -1, in_width], np.int64)),
                             special_zero=False)
            # segment 0: the H-wide embedding repeated to the hc*H width; a
            # later segment receives the hc*H-wide state as it left the
            # previous one -- no tile, no projection, the same bytes
            hidden = (op.tile(emb, op.constant(np.array([1, 1, hc], np.int64)))
                      if first else emb)

            # the n-gram table: PORTS, one per sub-cap chunk, never a constant
            # (module docstring §4). The host-mmap tier serving reads through
            # src/exec/ngram_table.h is what gets bound to them.
            ngram_rows = (cfg.ngram_total_vocab
                          if hasattr(cfg, "ngram_total_vocab")
                          else pwe.REAL_GEOMETRY["ngram_total_vocab"])
            row_bytes = ngram_row_bytes(head_dim)
            # The port's row count is the STAGING BOUND when the caller asks for
            # the disk-backed path (campaign `ple-disk-backend`): one port of
            # `max_tokens x Hn` rows that the runtime fills per forward by
            # `pread`, instead of one port spanning the whole table. The source
            # tensor still has to be the full table -- that is admission's
            # business -- so the port is deliberately SMALLER than the source,
            # which is exactly how `bind_ngram_ports` recognises staging.
            port_rows = (int(ngram_staging_rows) if ngram_staging_rows
                         else ngram_rows)
            table_ports = (ngram_table_ports(port_rows, row_bytes,
                                             ngram_chunk_cap_bytes)
                           if has_ple else [])

            # ONE rope table pair for every full-attention layer, spanning
            # the whole context. It used to be baked per layer for positions
            # 0..T-1 -- right at position 0, silently wrong after, and pinned
            # as such by the suite until this increment. Shared, it costs the
            # table once (~67 MB a side at real geometry), not once per layer.
            span = int(rope_span if rope_span is not None
                       else cfg.max_position_embeddings)
            cos_np, sin_np = qattn._freqs_tables(cfg, span)
            rope_cos = op.constant(cos_np)
            rope_cos.set_friendly_name("rope/cos")
            rope_sin = op.constant(sin_np)
            rope_sin.set_friendly_name("rope/sin")

            kinds = []
            for i in range(lo, hi):        # GLOBAL layer indices
                kind = "attn" if (i % 4) == 3 else "gdn"
                kinds.append(kind)
                st = _layer_state(ar, cfg, kind, i, feed, dense_census)

                if has_ple and i == ple_layer_idx:
                    # pin 1283: hidden = hidden + ple(...), ADDITIVE
                    gathered = ngram_dequant_iq4nl(
                        ngram_chunked_gather(chunk_ids, local_ids, table_ports),
                        head_dim)                            # [1,T,Hn,hd] real values
                    gathered.set_friendly_name("ple/gathered")
                    emb_ple = op.reshape(
                        gathered,
                        op.constant(np.array([1, -1, Hn * head_dim], np.int64)),
                        special_zero=False)
                    hidden = op.add(hidden, _ple_tail(
                        hidden, emb_ple, cfg, ple_state, T, conv_mask))
                    hidden.set_friendly_name("ple/out")

                # attn_hyper_connection (use_combine=True) -> mixer -> combine
                h, hyper, inj = _split_combine(hidden, cfg, st,
                                               "attn_hyper_connection.", T)
                if kind == "gdn":
                    g = qgdn.emit_gdn(
                        h, conv_mask, cfg, _strip(st, "linear_attn."), None,
                        conv_emitter=stateful_short_conv(i, beam, sinks),
                        core_emitter=gdn_core_emitter(i, beam, sinks))
                else:
                    g = emit_stateful_attention(
                        h, pid, cfg, _strip(st, "self_attn."), i, beam,
                        attn_mask, sinks, rope_cos, rope_sin)
                hidden = _recombine(hyper, inj, g, cfg, T)
                # named so a card-side localiser can cut the graph here
                # (tools/boot_serving_shape.py --cut); names only, no op
                hidden.set_friendly_name(f"layer{i}/mixer_out")

                h, hyper, inj = _split_combine(hidden, cfg, st,
                                               "mlp_hyper_connection.", T)
                m = emit_moe_tiled(h, cfg, st, ar, T, f"layer{i}/moe",
                                   filler=filler, layer=i, port_sink=expert_ports)
                hidden = _recombine(hyper, inj, m, cfg, T)
                hidden.set_friendly_name(f"layer{i}/out")

            if last:
                # the final mixer, use_combine=False (pin 1493-1496), then lm_head
                fin = {
                    "hc_norm.weight": ar.f32([hc * H]),
                    "input_mix_weight_down.weight": ar.f32([cfg.hc_lowrank, hc * H]),
                    "input_mix_weight_up.weight": ar.f32([hc * H, cfg.hc_lowrank]),
                }
                if feed is not None:
                    _fill_dense(fin, feed, "hyper_connection_mixer.", dense_census)
                fin_h = qhc.emit_hc(hidden, cfg, fin, None)             # [1,T,H]
                head_w = ar.f32([V, H])
                if feed is not None:
                    _fill_dense({"lm_head.weight": head_w}, feed, "", dense_census)
                logits = op.matmul(fin_h, qgdn._c(head_w),
                                   transpose_a=False, transpose_b=True)
                res = op.result(logits)
                res.set_friendly_name("logits")
                res.output(0).set_names({"logits"})
            else:
                # the segment boundary: the hc*H-wide state, as is, for the
                # next segment's `inputs_embeds`
                res = op.result(hidden)
                res.set_friendly_name("hidden_out")
                res.output(0).set_names({"hidden_out"})

            params = [inputs_embeds, pid]
            if has_ple:
                params += [chunk_ids, local_ids]
            params += [conv_mask, attn_mask, beam] + table_ports
            if expert_ports is not None:
                params += expert_ports.params
            model = Model([res], sinks, params,
                          "qwen4_exp_serving_shape" if layer_range is None
                          else f"qwen4_exp_serving_shape_L{lo}_{hi}")

        nodes, const_bytes, counts = pwe.graph_measures(model)
        report = {
            "n_layers": nl,
            "layer_range": [lo, hi],
            "segment_first": bool(first),
            "segment_last": bool(last),
            "inputs_embeds_width": int(in_width),
            "has_ple": bool(has_ple),
            "expert_ports": ([(n, list(s), (int(np.prod(s)) if b is None else int(b.size)))
                              for n, s, b in expert_ports.bodies]
                             if expert_ports is not None else []),
            "gdn_layers": kinds.count("gdn"),
            "attn_layers": kinds.count("attn"),
            "seq_len": None,                   # dynamic in T since feed-the-ports
            "rope_span": span,
            "nodes": nodes,
            "graph_const_bytes": const_bytes,
            "op_histogram": counts,
            # the table as it now travels: rows per port, bytes per port, and
            # the cap they were cut under. Not counted in graph_const_bytes --
            # it is not a constant any more, which is the point.
            "ngram_table_rows": int(ngram_rows),
            "ngram_staging_rows": (int(ngram_staging_rows)
                                   if ngram_staging_rows else None),
            "ngram_row_bytes": int(row_bytes),
            "ngram_chunk_cap_bytes": int(ngram_chunk_cap_bytes),
            "ngram_table_ports": [
                (p.get_friendly_name(), int(_dims(p.output(0))[0]),
                 int(_dims(p.output(0))[0]) * int(row_bytes))
                for p in table_ports],
            "dense_fill_census": dense_census,
            "arena_declared_bytes": ar.declared_bytes,
            "arena_written_bytes": ar.written_bytes,
            "arena_disk_kib": ar.disk_kib(),
            "fill_census": filler.census() if filler is not None else None,
            "inputs": [(p.get_node().get_friendly_name(),
                        _dims(p), str(p.get_element_type()))
                       for p in model.inputs],
            "outputs": [(r.get_node().get_friendly_name(),
                         _dims(r), str(r.get_element_type()))
                        for r in model.outputs],
        }
        return model, report
    except BaseException:
        if own_arena:
            ar.close()
        raise


def _dims(port):
    """A port's shape as a list, with -1 for a dynamic dimension.

    `get_shape()` THROWS on a dynamic shape ("to_shape was called on a dynamic
    shape", partial_shape.cpp:261), and the serving shape stopped being fully
    static the moment it carried a KV Variable: `attention_mask` spans past +
    current and `beam_idx` is a batch of beams. Every static dimension still
    reports as the same int it did, so a reader of `report["inputs"]` sees no
    change where nothing changed.
    """
    ps = port.get_partial_shape()
    return [d.get_length() if d.is_static else -1 for d in ps]


def _kv_variable(layer, tag, kv_heads, head_dim):
    """One rank-4 KV Variable, the shape `load_paged` reads its KV prototypes
    from before the transformation runs (backend_ov.cpp:2565-2577: rank 4, the
    leading dim and the sequence dim dynamic, the tail static).

    The variable_id is the stateful-export convention the served artifact
    carries (`cache_params.past.key.N`); what it is called does not reach the
    paged port name -- the transformation numbers `key_cache.N` /
    `value_cache.N` by the order it meets the PagedAttentionExtension nodes --
    but a graph with two variables of one id is rejected, so it has to be
    unique per layer and side.
    """
    info = ovutil.VariableInfo()
    info.data_shape = ov.PartialShape([-1, kv_heads, -1, head_dim])
    info.data_type = Type.f32
    info.variable_id = f"cache_params.past.{tag}.{layer}"
    return ovutil.Variable(info)


def _repeat_kv_broadcast(x, kv_heads, heads, head_dim):
    """[1, kv, S, d] -> [1, heads, S, d] with each kv head's copies adjacent.

    `q4e.attention._repeat_heads_h` does the same thing with a Concat of r
    copies. It is not reusable HERE, and the reason is measured rather than
    stylistic: it reshapes to `[1, kv, 1, T, d]` with T written in, and the
    length of the joined KV is `past + T`, which is dynamic. Substituting it
    fails at BUILD time, before the transformation is reached -- so what this
    function replaces is a helper that cannot express the shape, not a helper
    the pass would reject. Whether the pass also prefers one form over the
    other is NOT established here and is not claimed.

    This is instead the SERVED ARTIFACT'S OWN shape for the operation --
    Unsqueeze -> Broadcast -> Reshape, read off its first attention layer
    2026-09-13 -- which is shape-agnostic in the sequence dim. Numerically the
    two are the same tensor.
    """
    r = heads // kv_heads
    if r == 1:
        return x
    shape = op.shape_of(x, output_type="i64")
    b = op.gather(shape, op.constant(np.array([0], np.int64)),
                  op.constant(np.array(0, np.int64)))
    s = op.gather(shape, op.constant(np.array([2], np.int64)),
                  op.constant(np.array(0, np.int64)))
    up = op.unsqueeze(x, op.constant(np.array(2, np.int64)))
    # the batch is read off the input too: it is the TOKEN axis at the SDPA
    # (increment 5, emit_stateful_attention) and not 1
    wide = op.broadcast(up, op.concat(
        [b, op.constant(np.array([kv_heads, r], np.int64)), s,
         op.constant(np.array([head_dim], np.int64))], axis=0))
    return op.reshape(wide,
                      op.constant(np.array([0, heads, -1, head_dim], np.int64)),
                      special_zero=True)


def _additive_causal_mask(n_tok, total, past):
    """The [1, 1, T, TOTAL] additive mask for a query block of `n_tok` tokens
    (a scalar i64 node) that starts at `past` inside a sequence of `total`
    keys: 0 where the key is visible,
    finfo(f32).min above the diagonal -- the value `q4e.attention` bakes for
    the same purpose (pin 809, which converts torch's bool mask with
    `torch.finfo(dtype).min`).

    It is built from Ranges rather than baked as a Constant because TOTAL is
    dynamic here: the key length is past + T and the past comes out of the KV
    Variable. The transformation REPLACES this input -- PagedAttention derives
    visibility from `past_lens` and `subsequence_begins` instead -- so it does
    not survive into the compiled graph. It is emitted correctly anyway,
    because a graph that is only right after a pass has run is a graph nobody
    can check.
    """
    i64 = lambda v: op.constant(np.array(v, np.int64))
    rows = op.add(op.range(i64(0), n_tok, i64(1), Type.i64), past)   # [T]
    cols = op.range(i64(0), op.squeeze(total, i64(0)), i64(1), Type.i64)
    visible = op.less_equal(op.unsqueeze(cols, i64(0)),
                            op.unsqueeze(rows, i64(1)))             # [T, TOT]
    blocked = op.select(
        visible, op.constant(np.array(0.0, np.float32)),
        op.constant(np.finfo(np.float32).min.astype(np.float32)))
    return op.unsqueeze(blocked, i64([0, 1]))                       # [1,1,T,TOT]


def stateful_short_conv(layer, beam, sinks):
    """The GDN depthwise causal conv with its K-column state in a Variable.

    Returns a drop-in for `q4e.gdn._causal_conv_silu` -- same arguments, same
    [1, conv_dim, T] result -- so `emit_gdn`'s `conv_emitter` hook is the only
    thing that changes in the parity-gated emitter.

    The shape is not a design; it is what `PagedCausalConv1DFusion` matches,
    read out of the pass's own source at the pinned OpenVINO commit:

        ReadValue (rank 3)  ->  optional Gather  ->  Concat(axis = -1)
          ->  GroupConvolution (rank-4 STATIC weights)  ->  optional Add
          ->  Slice                                      <- the match root

    Four of those are load-bearing in a way that is easy to get wrong, and each
    cost a refusal before it was read rather than guessed:

      * the Concat's `axis` ATTRIBUTE must be -1. On a rank-3 tensor axis 2 is
        the same tensor and a different attribute, and the matcher compares the
        attribute. That single value was the difference between no match and
        twelve of the thirteen ports.
      * the conv weights must be rank 4 and STATICALLY shaped. The checkpoint
        carries [conv_dim, 1, K]; GroupConvolution wants
        [groups, out/groups, in/groups, kernel], which for a depthwise conv is
        [conv_dim, 1, 1, K].
      * the state's dim 1 must equal the weights' dim 0 and its dim 2 the
        kernel; the callback returns false rather than matching otherwise.
      * the root is the Slice on the conv OUTPUT. The Slice that cuts the new
        state out of the joined sequence is not part of the pattern -- the pass
        drops the Variable by id.

    The arithmetic, which is also why the result is causally identical to the
    unrolled default: the state holds the last K inputs, so the joined sequence
    is K + T long and the valid convolution over it is T + 1 long. Output j
    covers joined[j .. j+K-1], so the output for x_i is at j = i + 1 and the
    slice starts at 1. On the first forward the state is zeros, which is the
    same left-pad `_causal_conv_silu` bakes as K-1 explicit zero columns.
    """
    def emit(x, conv_w, T, conv_dim, K):
        i64 = lambda v: op.constant(np.array(v, np.int64))
        info = ovutil.VariableInfo()
        info.data_shape = ov.PartialShape([-1, conv_dim, K])
        info.data_type = Type.f32
        info.variable_id = f"cache_params.past.conv.{layer}"
        var = ovutil.Variable(info)

        init = op.broadcast(op.constant(np.array(0.0, np.float32)),
                            i64([1, conv_dim, K]))
        past = op.gather(op.read_value(init, var), beam, i64(0))
        joined = op.concat([past, x], axis=-1)          # [1, conv_dim, K+T]

        # NO ShapeOf on this construct (feed-the-ports). The new state is the
        # last K columns and the conv output is everything past its first
        # column: both are negative-index / to-the-end Slices. The first form
        # read the joined length off ShapeOf -> Gather, and once T went
        # dynamic the pass's `TotalSequenceLengthPattern` matched that chain
        # as if it were an attention's KV length and threw
        # ("failed to determine the dimension value after the Gather").
        INT_MAX = 9223372036854775807
        sinks.append(op.assign(
            op.slice(joined, i64([-K]), i64([INT_MAX]), i64([1]), i64([2])),
            var))

        weights = qgdn._c(np.ascontiguousarray(conv_w, np.float32)
                          .reshape(conv_dim, 1, 1, K))
        conv = op.group_convolution(joined, weights, strides=[1],
                                    pads_begin=[0], pads_end=[0], dilations=[1])
        body = op.slice(conv, i64([1]), i64([INT_MAX]), i64([1]), i64([2]))
        return qgdn._silu(body)

    return emit


def _gdn_loop_body(HV, Dk, Dv):
    """One timestep of the delta rule, as the Loop body `FuseGDNLoop` requires.

    Transcribed op for op from `matches_linear_attention_loop`
    (fuse_gated_delta_net.cpp). The body's results must be, IN THIS ORDER,
    [execution condition, updated state, scattered output] -- the matcher reads
    `body_results[1]` and `body_results[2]` by index -- and query, key and value
    must be rank 4 with a sequence extent of ONE, which is what makes this the
    token-sequential rule rather than the chunked one.

        gated_state   = state * Unsqueeze(Exp(g), -1)
        key_unsq      = Unsqueeze(Squeeze(key, 2), -1)
        projected     = ReduceSum(gated_state * key_unsq, -2, keep_dims=False)
        delta         = Squeeze(value, 2) - projected
        updated       = gated_state + key_unsq * Unsqueeze(delta * beta, -2)
        out           = ReduceSum(updated * Unsqueeze(Squeeze(q,2), -1), -2,
                                  keep_dims=True)
        result[2]     = ScatterUpdate(buffer, Unsqueeze(step, 0), out, 2)
        result[1]     = updated
    """
    i64 = lambda v: op.constant(np.array(v, np.int64))
    step = op.parameter([], Type.i64)                    # current iteration
    state = op.parameter([1, HV, Dk, Dv], Type.f32)      # recurrent state
    buf = op.parameter([1, HV, -1, Dv], Type.f32)        # output buffer, T dynamic
    q = op.parameter([1, HV, 1, Dk], Type.f32)
    k = op.parameter([1, HV, 1, Dk], Type.f32)
    v = op.parameter([1, HV, 1, Dv], Type.f32)
    g = op.parameter([1, HV, 1], Type.f32)
    beta = op.parameter([1, HV, 1], Type.f32)

    gated_state = op.multiply(state, op.unsqueeze(op.exp(g), i64([-1])))
    key_unsq = op.unsqueeze(op.squeeze(k, i64([2])), i64([-1]))
    projected = op.reduce_sum(op.multiply(gated_state, key_unsq), i64([-2]),
                              keep_dims=False)
    delta = op.subtract(op.squeeze(v, i64([2])), projected)
    updated = op.add(gated_state,
                     op.multiply(key_unsq,
                                 op.unsqueeze(op.multiply(delta, beta),
                                              i64([-2]))))
    out = op.reduce_sum(
        op.multiply(updated,
                    op.unsqueeze(op.squeeze(q, i64(2)), i64([-1]))),
        i64([-2]), keep_dims=True)
    scattered = op.scatter_update(buf, op.unsqueeze(step, i64(0)), out, i64(2))

    params = [step, state, buf, q, k, v, g, beta]
    body = Model([op.result(op.constant(np.array(True))),
                  op.result(updated), op.result(scattered)],
                 params, "gdn_delta_rule_step")
    return body, params


def _gdn_chunk_body(HV, Dk, Dv, chunk):
    """One CHUNK of the gated delta rule, as a Loop body: the SAME algebra the
    chunked core emits (`q4e.gdn`'s `perchunk` branch, `gdn.py`:470-500), with
    the recurrent state a merged Loop input instead of a Python variable.

    This is what makes multi-block prefill possible: the token-sequential body
    (`_gdn_loop_body`) advances ONE token per iteration, so a 32k prefill pays
    32k iterations; this one advances `chunk` tokens, so it pays `ceil(T/chunk)`
    -- 512 iterations at 32k instead of 32,768.

    The algebra is reused, not re-derived. `last` is the body's `state`
    parameter (the token-sequential core's `[1, HV, Dk, Dv]`), and the chunk's
    output is written into the buffer at rows `step*chunk .. +chunk`.

    Body results, IN THIS ORDER (the token-sequential matcher's contract, kept
    even though a chunked Loop is NOT fused -- see `stateful_gdn_core_chunked`):
    [execution condition, updated state, scattered output].
    """
    i64 = lambda v: op.constant(np.array(v, np.int64))
    step = op.parameter([], Type.i64)
    state = op.parameter([1, HV, Dk, Dv], Type.f32)
    buf = op.parameter([1, HV, -1, Dv], Type.f32)
    q = op.parameter([1, HV, chunk, Dk], Type.f32)
    k = op.parameter([1, HV, chunk, Dk], Type.f32)
    v = op.parameter([1, HV, chunk, Dv], Type.f32)
    g = op.parameter([1, HV, chunk], Type.f32)
    beta = op.parameter([1, HV, chunk], Type.f32)

    beta_u = qgdn._reshape(beta, [1, HV, chunk, 1])
    v_beta = qgdn._mul(v, beta_u)
    k_beta = qgdn._mul(k, beta_u)
    add_mask = qgdn._c(np.where(np.triu(np.ones((chunk, chunk), np.float32), 1) > 0,
                                -1e30, 0.0).astype(np.float32))

    cum = op.cumsum(g, i64(2))                       # [1,HV,chunk]
    expc4 = qgdn._reshape(op.exp(cum), [1, HV, chunk, 1])
    pd = op.exp(qgdn._add(
        qgdn._sub(qgdn._reshape(cum, [1, HV, chunk, 1]),
                  qgdn._reshape(cum, [1, HV, 1, chunk])), add_mask))
    ut = qgdn._mul(qgdn._mm(k_beta, k, tb=True), pd)     # [1,HV,chunk,chunk]
    it = qgdn._mul(qgdn._mm(q, k, tb=True), pd)
    dkb = qgdn._mul(k_beta, expc4)
    inv = qgdn._ut_inverse(ut, chunk, [1, HV])           # pin 355-365
    nv = qgdn._mm(inv, v_beta)                           # pin 366
    kcd = qgdn._mm(inv, dkb)

    qd = qgdn._mul(q, expc4)
    cum_last = qgdn._slice(cum, chunk - 1, chunk, 1, 2)  # [1,HV,1]
    kd = qgdn._mul(k, qgdn._reshape(op.exp(qgdn._sub(cum_last, cum)),
                                    [1, HV, chunk, 1]))
    cd = qgdn._reshape(op.exp(cum_last), [1, HV, 1, 1])

    v_new = qgdn._sub(nv, qgdn._mm(kcd, state))
    inter = qgdn._mm(qd, state)
    core = qgdn._add(inter, qgdn._mm(it, v_new))         # [1,HV,chunk,Dv]
    updated = qgdn._add(qgdn._mul(state, cd),
                        qgdn._mm(kd, v_new, ta=True))    # [1,HV,Dk,Dv]

    rows = op.add(op.multiply(step, i64(chunk)),
                  op.constant(np.arange(chunk, dtype=np.int64)))   # [chunk]
    scattered = op.scatter_update(buf, rows, core, i64(2))

    params = [step, state, buf, q, k, v, g, beta]
    body = Model([op.result(op.constant(np.array(True))),
                  op.result(updated), op.result(scattered)],
                 params, "gdn_delta_rule_chunk")
    return body, params


def stateful_gdn_core_chunked(layer, beam, sinks, chunk=None):
    """The gated delta rule as a CHUNKED Loop -- `stateful_gdn_core`'s drop-in
    that advances `chunk` tokens per iteration instead of one.

    THE MATCHER QUESTION, DECIDED: `FuseGDNLoop` matches only the
    token-sequential Loop (`_gdn_loop_body`'s docstring: query/key/value rank 4
    with a sequence extent of ONE). A chunked body carries a sequence extent of
    `chunk`, so it is NOT rewritten into `ov::op::internal::GatedDeltaNet` and
    `PagedGatedDeltaNetFusion` never sees it. **Decision: keep the chunked body
    IN-GRAPH** -- the cost is that this prefill path loses the fused kernel,
    which is the decode path's fast route; the benefit is that it still beats
    one iteration per token by `chunk`, which is the whole point. Extending the
    matcher to a chunked Loop is a separate, larger change (the fusion's
    `matches_linear_attention_loop` reads a rank-4 seq-1 body and its state
    update is the token rule) and is NOT done here. Stated, not left implicit.

    The dynamic-T contract is kept: the trip count is `ceil(T/chunk)`, computed
    from `ShapeOf`, so the graph is T-independent (LYON's compile-once property
    survives). The inputs are zero-padded to a multiple of `chunk` before the
    Loop and the buffer is sliced back to T after it, so a prompt whose length
    is not a multiple of `chunk` is exact.
    """
    C = int(chunk or qgdn.CHUNK)

    def emit(q, k, v, beta_t, decay_t, T, HV, Dk, Dv):
        i64 = lambda val: op.constant(np.array(val, np.int64))

        head_size = op.convert(
            op.gather(op.shape_of(q, output_type="i64"), i64(3), i64(0)),
            Type.f32)
        q_scaled = op.divide(
            q, op.power(head_size, op.constant(np.array(0.5, np.float32))))

        info = ovutil.VariableInfo()
        info.data_shape = ov.PartialShape([1, HV, Dk, Dv])
        info.data_type = Type.f32
        info.variable_id = f"cache_params.past.ssm.{layer}"
        var = ovutil.Variable(info)
        init = op.broadcast(op.constant(np.array(0.0, np.float32)),
                            i64([1, HV, Dk, Dv]))
        # BEAM-FREE (arcint 0.5.4 LYON, 2026-09-26). The token-sequential core
        # gathers the state with the `beam_idx` PARAMETER, and the fusion
        # CONSUMES that chain -- so the parameter's declaration and its use
        # disappear together. An unfused chunked Loop leaves the chain alive,
        # and `SDPAToPagedAttention` then drops the declaration while the
        # Gather still references it (`backend_ov.cpp`:2637), refusing the
        # artifact: "Model references undeclared parameters: beam_idx". The
        # served path is ONE LANE, so the gather is a constant row 0 -- no
        # `beam_idx` reference survives to dangle. (`beam` is kept in the
        # signature for the hook contract and deliberately unused.)
        past = op.gather(op.read_value(init, var),
                         op.constant(np.array([0], np.int64)), i64(0))

        # ceil(T / chunk) and the pad to a whole chunk, from ShapeOf
        n_tok = op.squeeze(op.gather(op.shape_of(q, output_type="i64"),
                                     i64([2]), i64(0)), i64([0]))
        n_chunks = op.divide(op.add(n_tok, i64(C - 1)), i64(C))
        n_pad = op.subtract(op.multiply(n_chunks, i64(C)), n_tok)
        pad_shape = op.concat([i64([1, HV]), op.reshape(n_pad, i64([1]), False),
                               i64([Dk])], axis=0)
        buf_shape = op.concat([i64([1, HV]),
                               op.reshape(op.multiply(n_chunks, i64(C)),
                                          i64([1]), False),
                               i64([Dv])], axis=0)

        def _pad(x, shape):
            return op.concat([x, op.broadcast(op.constant(np.array(0.0, np.float32)),
                                              shape)], axis=2)

        q_p = _pad(q_scaled, pad_shape)
        k_p = _pad(k, pad_shape)
        pad_v = op.concat([i64([1, HV]), op.reshape(n_pad, i64([1]), False),
                           i64([Dv])], axis=0)
        v_p = _pad(v, pad_v)
        pad_1 = op.concat([i64([1, HV]), op.reshape(n_pad, i64([1]), False)], axis=0)
        g_p = _pad(decay_t, pad_1)
        b_p = _pad(beta_t, pad_1)

        body, (p_step, p_state, p_buf, p_q, p_k, p_v, p_g,
               p_beta) = _gdn_chunk_body(HV, Dk, Dv, C)
        loop = op.loop(n_chunks, op.constant(np.array(True)))
        loop.set_function(body)
        loop.set_special_body_ports([0, 0])
        for param, src in ((p_q, q_p), (p_k, k_p), (p_v, v_p),
                           (p_g, g_p), (p_beta, b_p)):
            loop.set_sliced_input(param, src.output(0), 0, C, C, -1, 2)
        loop.set_merged_input(p_state, past.output(0),
                              body.get_results()[1].output(0))
        loop.set_merged_input(
            p_buf,
            op.broadcast(op.constant(np.array(0.0, np.float32)),
                         buf_shape).output(0),
            body.get_results()[2].output(0))
        attn_out = loop.get_iter_value(body.get_results()[2].output(0), -1)
        state_out = loop.get_iter_value(body.get_results()[1].output(0), -1)
        loop.validate_and_infer_types()

        sinks.append(op.assign(
            op.reshape(state_out, i64([1, HV, Dk, Dv]), special_zero=False),
            var))
        # the buffer is chunk-padded: cut it back to T, then the served order
        kept = op.slice(attn_out, i64([0]), op.reshape(n_tok, i64([1]), False),
                        i64([1]), i64([2]))
        return op.transpose(kept,
                            op.constant(np.array([0, 2, 1, 3], np.int32)))

    return emit


# arcint (0.5.4 LYON). The GDN core the served backbone emits.
#
#   "sequential"  the token-sequential Loop (`stateful_gdn_core`): one token per
#                 iteration. This is the body `FuseGDNLoop` rewrites into
#                 `ov::op::internal::GatedDeltaNet` and `PagedGatedDeltaNetFusion`
#                 matches -- the decode path's fused kernel.
#   "chunked"     the multi-block Loop (`stateful_gdn_core_chunked`): CHUNK
#                 tokens per iteration, so a 32k prefill pays 512 iterations
#                 instead of 32,768. NOT fused (the fusion reads a seq-1 body);
#                 kept IN-GRAPH deliberately -- see that emitter's docstring.
#
# BOTH are T-independent (the trip count comes off `ShapeOf`), so LYON's
# compile-once property holds under either. `Q4E_GDN_CORE` selects one for a
# whole export without editing code, the discipline `Q4E_GDN_UT_MODE` uses; a
# typo is REFUSED, so a silent default cannot report the wrong core's graph.
GDN_CORE = os.environ.get("Q4E_GDN_CORE", "sequential").strip() or "sequential"
GDN_CORE_CHUNK = int(os.environ.get("Q4E_GDN_CHUNK", qgdn.CHUNK))


def gdn_core_emitter(layer, beam, sinks):
    if GDN_CORE == "sequential":
        return stateful_gdn_core(layer, beam, sinks)
    if GDN_CORE == "chunked":
        return stateful_gdn_core_chunked(layer, beam, sinks, chunk=GDN_CORE_CHUNK)
    raise ValueError(f"unknown Q4E_GDN_CORE {GDN_CORE!r}; expected sequential or chunked")


def stateful_gdn_core(layer, beam, sinks):
    """The gated delta rule as the token-sequential Loop the fusion chain wants.

    Returns a drop-in for `emit_gdn`'s `core_emitter` hook: the same
    [1, T, HV, Dv] the chunked core produces at the same point, with the
    recurrent state carried in a rank-4 ov Variable.

    TWO passes stand between this Loop and `gated_delta_state_table.N`, and
    neither is optional. `FuseGDNLoop` rewrites the Loop into one
    `ov::op::internal::GatedDeltaNet` node; `PagedGatedDeltaNetFusion` then
    matches THAT node over a ReadValue. Nothing can emit the internal op
    directly, which is why this is a Loop and not a call.

    What the first pass pins, all of it read out of the matcher:

      * the Loop's EXTERNAL INPUT ORDER, positions 2 through 8:
        q_scaled, key, value, gate, beta, init_state, output_buffer -- after
        trip count and execution condition. Input order here is the order the
        set_*_input calls are made in, so the call order below IS the contract.
      * at least 9 inputs and EXACTLY 2 outputs; output 0 the attention output,
        output 1 the final state, so `get_iter_value` is called in that order.
      * the query must arrive already DIVIDED by `head_size ** 0.5` where the
        head size is read from a `ShapeOf` -- `Divide(q, Power(Convert(
        Gather(ShapeOf(q), 3, 0)), 0.5))`. A folded constant does not match,
        which is why the chunked path's `q * Dk**-0.5` is skipped and this
        does its own scaling.

    The recurrent state is declared `[1, HV, Dk, Dv]` rather than the served
    artifact's `[?, ...]`. It reaches `load_paged`'s rank-4 prototype scan
    either way -- that code replaces dim 0 with 1 regardless
    (backend_ov.cpp:2565-2577) -- and this is the form that was measured to
    fuse. The attention KV Variables stay out of that scan on their own, by
    the dynamic sequence dim the same code excludes.
    """
    def emit(q, k, v, beta_t, decay_t, T, HV, Dk, Dv):
        i64 = lambda val: op.constant(np.array(val, np.int64))

        # the head-size scale, in the matcher's shape and not a folded constant
        head_size = op.convert(
            op.gather(op.shape_of(q, output_type="i64"), i64(3), i64(0)),
            Type.f32)
        q_scaled = op.divide(
            q, op.power(head_size, op.constant(np.array(0.5, np.float32))))

        info = ovutil.VariableInfo()
        info.data_shape = ov.PartialShape([1, HV, Dk, Dv])
        info.data_type = Type.f32
        info.variable_id = f"cache_params.past.ssm.{layer}"
        var = ovutil.Variable(info)
        init = op.broadcast(op.constant(np.array(0.0, np.float32)),
                            i64([1, HV, Dk, Dv]))
        past = op.gather(op.read_value(init, var), beam, i64(0))

        body, (p_step, p_state, p_buf, p_q, p_k, p_v, p_g,
               p_beta) = _gdn_loop_body(HV, Dk, Dv)
        # trip count and output buffer from the SHAPE, as the served
        # artifact's Loop has them (in[0] a Convert off ShapeOf, in[8] a
        # Broadcast [?, HV, ?, Dv]): dynamic in T
        n_tok = op.squeeze(op.gather(op.shape_of(q, output_type="i64"),
                                     i64([2]), i64(0)), i64([0]))
        loop = op.loop(n_tok, op.constant(np.array(True)))
        loop.set_function(body)
        # body parameter 0 is the iteration counter, body result 0 the condition
        loop.set_special_body_ports([0, 0])
        # ORDER IS THE CONTRACT (see the docstring): q, k, v, gate, beta,
        # state, buffer -- every one sliced on the sequence axis except the two
        # merged ones, which carry across iterations.
        for param, src in ((p_q, q_scaled), (p_k, k), (p_v, v),
                           (p_g, decay_t), (p_beta, beta_t)):
            loop.set_sliced_input(param, src.output(0), 0, 1, 1, -1, 2)
        loop.set_merged_input(p_state, past.output(0),
                              body.get_results()[1].output(0))
        loop.set_merged_input(
            p_buf,
            op.broadcast(op.constant(np.array(0.0, np.float32)),
                         op.concat([i64([1, HV]),
                                    op.gather(op.shape_of(q, output_type="i64"),
                                              i64([2]), i64(0)),
                                    i64([Dv])], axis=0)).output(0),
            body.get_results()[2].output(0))
        attn_out = loop.get_iter_value(body.get_results()[2].output(0), -1)
        state_out = loop.get_iter_value(body.get_results()[1].output(0), -1)
        loop.validate_and_infer_types()

        # `get_iter_value` hands back an Output and Assign wants a Node; a
        # same-shape Reshape is the cheapest node that carries it. The paged
        # fusion does not look at the Assign side -- it drops the Variable by
        # id -- so nothing about this reshape is load-bearing for the match.
        sinks.append(op.assign(
            op.reshape(state_out, i64([1, HV, Dk, Dv]), special_zero=False),
            var))
        return op.transpose(attn_out,
                            op.constant(np.array([0, 2, 1, 3], np.int32)))

    return emit


def emit_stateful_attention(hidden, pid, config, state, layer, beam,
                            attn_mask, sinks, rope_cos, rope_sin):
    """The full-attention layer in the STATEFUL shape the serving path's own
    transformation converts, at real geometry.

    Everything outside the attention core is `q4e.attention`'s, op for op and
    pin line for pin line -- the fused q_proj and its per-head [query | gate]
    chunk, q_norm / k_norm over head_dim, v_proj, the gather-and-apply rope,
    the sigmoid gate and o_proj. What is replaced is the eager core (repeat_kv,
    scaled q@k^T, the baked causal mask, the f32 softmax and @V): here K and V
    are carried in a Variable and the core is ONE ScaledDotProductAttention, so

        ReadValue(init = Broadcast(0.0 -> [1, kv, 0, d]))
          -> Gather(beam_idx, axis 0)
          -> Concat(past, current, axis 2)        <- Assign takes THIS
          -> repeat_kv -> SDPA(q, k, v, mask, scale), causal=False

    is the chain `ov::pass::SDPAToPagedAttention` rewrites into a
    PagedAttentionExtension with `key_cache.N` and `value_cache.N` ports. The
    five-input SDPA with an explicit scale and `causal=False` is the served
    artifact's own form, not a preference: read off it 2026-09-13.

    `sinks` is appended to rather than returned because the Assign nodes belong
    to the MODEL, not to the layer -- `ov::Model` takes them as a separate
    argument and a layer that dropped one would emit a graph whose state is
    read and never written.

    THE ROPE TABLE SPANS THE FULL CONTEXT AND IS SHARED (feed-the-ports
    increment; until then it was baked per layer for positions 0..T-1 and a
    position past T gathered off its end without a throw). `rope_cos` /
    `rope_sin` are the ONE constant pair `build_serving_shape_ir` makes from
    `_freqs_tables(config, rope_span)`, gathered here by `position_ids`; one
    table for the twelve layers is what keeps it out of the residency
    keystone (~67 MB a side once, not 1.6 GiB).

    DYNAMIC IN T: every reshape here uses -1, the mask and the loop bounds
    come from ShapeOf. The block length is nowhere in this layer.
    """
    H = config.hidden_size
    heads = config.num_attention_heads
    kv = config.num_key_value_heads
    d = getattr(config, "head_dim", None) or H // heads
    eps = config.rms_norm_eps
    rotary = int(rope_cos.get_output_shape(0)[-1])
    T = -1
    i64 = lambda v: op.constant(np.array(v, np.int64))

    # pin 867-870, unchanged from q4e.attention: the split is PER HEAD on the
    # last axis, and the gate is the second half of each head's 2*d block.
    qg = qgdn._reshape(
        qgdn._mm(hidden, qattn._c(state["q_proj.weight"]), tb=True),
        [1, T, heads, 2 * d])
    q = qgdn._slice(qg, 0, d, 1, 3)
    gate = qgdn._reshape(qgdn._slice(qg, d, 2 * d, 1, 3), [1, T, heads * d])

    q = qgdn._transpose(
        qattn._rmsnorm_hd(q, state["q_norm.weight"], eps, d,
                          getattr(config, "norm_plus_one", True)), [0, 2, 1, 3])
    k = qgdn._reshape(
        qgdn._mm(hidden, qattn._c(state["k_proj.weight"]), tb=True), [1, T, kv, d])
    k = qgdn._transpose(
        qattn._rmsnorm_hd(k, state["k_norm.weight"], eps, d,
                          getattr(config, "norm_plus_one", True)), [0, 2, 1, 3])
    v = qgdn._transpose(
        qgdn._reshape(qgdn._mm(hidden, qattn._c(state["v_proj.weight"]), tb=True),
                      [1, T, kv, d]), [0, 2, 1, 3])
    q, k = qattn._apply_rope(q, k, rope_cos, rope_sin, pid, rotary, T)
    q.set_friendly_name(f"attn{layer}/q_rope")            # localiser cut points
    k.set_friendly_name(f"attn{layer}/k_rope")

    # TOKEN-MAJOR AT THE SDPA, and only here (increment 5, read off the
    # device and then off the pass). `SDPAToPagedAttention` flattens the SDPA's
    # q/k/v for the PagedAttentionExtension with `Reshape [0, -1]`
    # (state_management_pattern.cpp, `q_reshape`): dimension 0 is KEPT as the
    # token axis and everything else is folded into the feature axis. On the
    # served artifact that is [?, ?] -> [tokens, heads*d], because the pass
    # also forces `input_ids` to [-1] + Unsqueeze(1), so at runtime the batch
    # axis carries the tokens and the sequence axis is 1. On this emitter's
    # [1, heads, T, d] it produced [1, T*heads*d] -- one token of 30,720
    # features -- and the GPU plugin died on it with a bare `map::at`
    # (window-050 §4.7). The two linear-attention fusions do NOT have this
    # asymmetry: PagedGatedDeltaNetFusion (`flatten_batch_length`) and
    # PagedCausalConv1DFusion (`Reshape [-1, hidden]`) flatten B*L into
    # tokens themselves, so the GDN and conv constructs stay as they are.
    #
    # So q, k and v are presented to the SDPA as [T, heads, 1, d]: the same
    # bytes, the token axis first. The unfused SDPA over that layout is a
    # batch of one-token queries -- exactly the served stateful graph's own
    # status after the pass's reinterpretation -- and is not a numeric
    # witness; the served path never runs it. The PA output comes back
    # [T, heads, 1, dv] (the pass's `pa_shape [0, 1, -1, dv]` + transpose)
    # and is transposed back to [1, heads, T, dv] for the rest of the layer.
    tm = lambda x: op.transpose(x, op.constant(np.array([2, 1, 0, 3], np.int32)))
    # ... AND DYNAMIC. With the operands token-major but STATIC ([5, 6144]
    # after the pass's flatten) the plugin still died with `map::at`, on the
    # cut just after this node and nowhere before it; every served
    # PagedAttentionExtension runs with a dynamic token axis ([?, ?]). The
    # token count is read off `position_ids`, which the pass rewrites to
    # [-1] (so post-pass it is dynamic and pre-pass it folds to T:
    # ReduceProd covers both [1, T] and [tokens] / [tokens, 1]), and the
    # operands are gathered along the token axis by Range(0, n) -- an
    # identity permutation whose only effect is that the axis is no longer
    # a compile-time constant. Measured, not argued: window-050 §4.7.
    n_tok = op.reduce_prod(op.shape_of(pid, output_type="i64"), i64([0]),
                           keep_dims=False)
    tok_idx = op.range(i64(0), n_tok, i64(1), Type.i64)
    dyn = lambda x: op.gather(tm(x), tok_idx, i64(0))
    q = dyn(q)                                               # [T?, heads, 1, d]
    k = dyn(k)                                               # [T?, kv, 1, d]
    v = dyn(v)

    # the two state reads, and the two writes that make them state
    full = []
    for tag, cur in (("key", k), ("value", v)):
        var = _kv_variable(layer, tag, kv, d)
        init = op.broadcast(op.constant(np.array(0.0, np.float32)),
                            i64([1, kv, 0, d]))
        past = op.gather(op.read_value(init, var), beam, i64(0))
        joined = op.concat([past, cur], axis=2)          # [T?, kv, past+1, d]
        sinks.append(op.assign(joined, var))
        full.append(joined)

    # TOTAL is the key length after the join; PAST is what was there before.
    # The mask is [T, 1, 1, TOTAL] in this layout: one row per token-batch.
    total = op.gather(op.shape_of(full[0], output_type="i64"), i64([2]), i64(0))
    n_vec = op.unsqueeze(n_tok, i64(0))                      # [1]: the token count
    past_len = op.subtract(total, n_vec)
    mask = op.reshape(_additive_causal_mask(n_tok, total, past_len),
                      op.concat([n_vec, i64([1, 1]), total], axis=0),
                      special_zero=False)
    att = op.scaled_dot_product_attention(
        q,
        _repeat_kv_broadcast(full[0], kv, heads, d),
        _repeat_kv_broadcast(full[1], kv, heads, d),
        mask,
        op.constant(np.array(d ** -0.5, np.float32)),
        causal=False)                                        # [T, heads, 1, d]

    out = qgdn._reshape(qgdn._transpose(tm(att), [0, 2, 1, 3]), [1, T, heads * d])
    out.set_friendly_name(f"attn{layer}/att_out")
    out = qgdn._mul(out, op.sigmoid(gate))                           # pin 898
    return qgdn._mm(out, qattn._c(state["o_proj.weight"]), tb=True)   # pin 900


def _strip(state, prefix):
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def _split_combine(hidden, config, state, prefix, T):
    """backbone.py's use_combine=True mixer: returns (h, hyper, inject)."""
    sub = _strip(state, prefix)
    return qhc.emit_combine(hidden, config, sub, None if T is None or T < 0 else T)


def _recombine(hyper, inj, block_out, config, T):
    """pin 1302-3 / 1308-9: hidden = hyper + (out.unsqueeze(-2) *
    inj.unsqueeze(-1)).flatten(-2)."""
    H = config.hidden_size
    hc = config.hc_count
    Td = -1 if T is None or T < 0 else int(T)
    o4 = op.reshape(block_out, op.constant(np.array([1, Td, 1, H], np.int64)),
                    special_zero=False)
    i4 = op.reshape(inj, op.constant(np.array([1, Td, hc, 1], np.int64)),
                    special_zero=False)
    prod = op.reshape(op.multiply(o4, i4),
                      op.constant(np.array([1, Td, hc * H], np.int64)),
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

    Td = -1 if T is None or T < 0 else int(T)
    key = qgdn._mm(emb, qgdn._c(state["key_proj.weight"]), tb=True)
    key_n = qple._group_rms(key, Td, hc, H, state["norm_key.weight"], eps)
    key_n4 = qgdn._reshape(key_n, [1, Td, hc, H])
    value = qgdn._mm(emb, qgdn._c(state["value_proj.weight"]), tb=True)
    q_n = qple._group_rms(hidden, Td, hc, H, state["norm_query.weight"], eps)
    q_n4 = qgdn._reshape(q_n, [1, Td, hc, H])
    gate = qgdn._rsum(qgdn._mul(key_n4, q_n4), 3)
    gate = qgdn._mul(gate, qgdn._c(np.float32(1.0 / math.sqrt(H))))
    ag = op.maximum(op.abs(gate), qgdn._c(np.float32(1e-6)))
    gate = qgdn._mul(op.sqrt(ag), op.sign(gate))
    sg = op.sigmoid(gate)
    value4 = qgdn._reshape(value, [1, Td, 1, H])
    gv = qgdn._mul(sg, value4)
    gv_flat = qgdn._reshape(gv, [1, Td, hc * H])
    gv_normed = qple._group_rms(gv_flat, Td, hc, H, state["norm_conv.weight"], eps)
    if conv_mask is not None:                    # pin 1251-1253
        m = qgdn._reshape(conv_mask, [1, Td, 1])
        gv_flat = qgdn._mul(gv_flat, m)
        gv_normed = qgdn._mul(gv_normed, m)
    conv_out = qple._short_conv(gv_normed, state["conv1d.weight"], Td,
                                hc * H, K, dilation)
    return qgdn._add(gv_flat, conv_out)



# --------------------------------------------------------------------------
# THE `qwen3_5_moe` SERVING SHAPE (design-qwen35moe-serving-shape, 2026-09-24)
# --------------------------------------------------------------------------
#
# The native route emits ONE graph family today -- the Flash-Next `qwen4_exp`
# backbone, whose layer is a HYPER-CONNECTION mixer around every sublayer and
# whose PLE gathers an n-gram table. `Qwen3.6-35B-A3B` (`general.architecture
# = qwen35moe`) has neither: 40 layers, 256 experts top-8, one shared expert,
# and a PLAIN PRE-NORM RESIDUAL layer. The four conventions below are measured
# (design note §5), not assumed; each cites its oracle.
#
#   1. GDN output gate = SILU, applied as a gated RMSNorm on the z projection
#      (`code`: llama.cpp src/models/qwen35moe.cpp build_norm_gated ->
#      ggml_silu; `measured-here`: the served int4 IR's linear_attn.norm chain
#      carries `aten::silu/Swish`, the only Sigmoid in linear_attn sits on
#      in_proj_b/beta).
#   2. value/key head map = TILED (`measured-here`: every value-head-indexed
#      GDN tensor in the GGUF is the HF interleave order re-laid into llama's
#      tiled order -- qkv v sigma-map max|diff| 0.012 vs identity 0.395;
#      attn_gate 0.012 vs 0.297; ssm_out 0.036 vs 0.413; ssm_alpha 0.0069 vs
#      0.171; ssm_beta 0.0043 vs 0.085; `code`: llama.cpp ggml_repeat_4d).
#   3. norms = PLAIN RMSNorm, no (1 + w), pre-norm residual (`measured-here`:
#      GGUF attn_norm / post_attention_norm / ssm_norm / attn_q_norm /
#      attn_k_norm all equal the served IR's Constants to max|diff| 0.0; the IR
#      chain is Power -> ReduceMean -> Add(eps) -> Sqrt -> Divide -> Multiply(x)
#      -> Multiply(weight), no +1). `_rmsnorm_hd` therefore takes
#      `norm_plus_one=False` for this family.
#   4. the tiled MoE lowering is E-agnostic (`measured-here`: a 256-expert
#      top-8 tiled block compiles to 3 GatherMatmul primitives on the CPU
#      plugin exactly as 512/top-10 does).

QWEN35MOE_LM = {
    "vocab_size": 248320,
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_value_head_dim": 128,
    "linear_num_value_heads": 32,
    "linear_conv_kernel_dim": 4,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
}


def qwen35moe_real_config(n_layers=None):
    """The `Qwen3.6-35B-A3B` geometry as a config object for the emitters,
    read from the shard's own metadata (gguf-py) and the served int4 IR's
    config.json. `rope_parameters` carries the text-degenerate mrope keys
    `q4e.attention._freqs_tables` reads: partial rotary 0.25 of head_dim 256 =
    64, theta 1e7, sections [11, 11, 10]. For TEXT all three mrope axes carry
    the same position, so the recomposition is the identity and the tables are
    the standard rope ones (`code`: qwen35moe.cpp ggml_rope_multi).

    `norm_plus_one=False`: this converter's gammas are PLAIN (measured, §3).
    `gdn_key_head_map="tiled"`: the GGUF's value heads are in llama's tiled
    order (measured, §2). `output_gate_type="silu"` (measured, §1)."""
    g = dict(QWEN35MOE_LM)
    nl = int(n_layers if n_layers is not None else g["num_hidden_layers"])
    c = types.SimpleNamespace(
        vocab_size=g["vocab_size"], hidden_size=g["hidden_size"],
        num_hidden_layers=nl, num_attention_heads=g["num_attention_heads"],
        num_key_value_heads=g["num_key_value_heads"], head_dim=g["head_dim"],
        max_position_embeddings=g["max_position_embeddings"],
        rms_norm_eps=g["rms_norm_eps"],
        linear_key_head_dim=g["linear_key_head_dim"],
        linear_num_key_heads=g["linear_num_key_heads"],
        linear_value_head_dim=g["linear_value_head_dim"],
        linear_num_value_heads=g["linear_num_value_heads"],
        linear_conv_kernel_dim=g["linear_conv_kernel_dim"],
        num_experts=g["num_experts"], num_experts_per_tok=g["num_experts_per_tok"],
        moe_intermediate_size=g["moe_intermediate_size"],
        shared_expert_intermediate_size=g["shared_expert_intermediate_size"],
        hidden_act="silu", output_gate_type="silu", norm_plus_one=False,
        gdn_key_head_map="tiled", norm_topk_prob=True, full_attention_interval=4,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                         "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]},
        layer_types=["full_attention" if (i % 4) == 3 else "linear_attention"
                     for i in range(nl)],
    )
    return c


# module-relative key -> (GGUF tensor suffix under blk.{i}., reshape kind).
# Kinds are `q4e.gguf_feed._materialise`'s; every gamma is `vec` (plain, §3),
# `ssm_a` is `neglog` (stored -exp(A_log), the GDN computes -exp(A_log)), and
# `ssm_conv1d` is `conv` ([K, C] -> [C, 1, K]).
_QWEN35MOE_TENSORS = {
    "input_norm.weight": ("attn_norm.weight", "vec"),
    "post_attention_norm.weight": ("post_attention_norm.weight", "vec"),
    "linear_attn.in_proj_qkv.weight": ("attn_qkv.weight", "direct2d"),
    "linear_attn.in_proj_z.weight": ("attn_gate.weight", "direct2d"),
    "linear_attn.in_proj_a.weight": ("ssm_alpha.weight", "direct2d"),
    "linear_attn.in_proj_b.weight": ("ssm_beta.weight", "direct2d"),
    "linear_attn.A_log": ("ssm_a", "neglog"),
    "linear_attn.dt_bias": ("ssm_dt.bias", "vec"),
    "linear_attn.conv1d.weight": ("ssm_conv1d.weight", "conv"),
    "linear_attn.norm.weight": ("ssm_norm.weight", "vec"),
    "linear_attn.out_proj.weight": ("ssm_out.weight", "direct2d"),
    "self_attn.q_proj.weight": ("attn_q.weight", "direct2d"),
    "self_attn.k_proj.weight": ("attn_k.weight", "direct2d"),
    "self_attn.v_proj.weight": ("attn_v.weight", "direct2d"),
    "self_attn.o_proj.weight": ("attn_output.weight", "direct2d"),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", "vec"),
    "self_attn.k_norm.weight": ("attn_k_norm.weight", "vec"),
    "mlp.gate.weight": ("ffn_gate_inp.weight", "direct2d"),
    "mlp.shared_expert.gate_proj.weight": ("ffn_gate_shexp.weight", "direct2d"),
    "mlp.shared_expert.up_proj.weight": ("ffn_up_shexp.weight", "direct2d"),
    "mlp.shared_expert.down_proj.weight": ("ffn_down_shexp.weight", "direct2d"),
    "mlp.shared_expert_gate.weight": ("ffn_gate_inp_shexp.weight", "row"),
}


def _qwen35_rmsnorm(x, weight, eps):
    """Plain RMSNorm over the LAST axis, no (1 + w) -- the qwen35moe
    convention (measured, §3). `x` [1, T, H]; `weight` the f32 memmap the
    state dict holds."""
    var = qgdn._rmean(qgdn._mul(x, x), -1)
    xn = qgdn._mul(x, qgdn._rsqrt_eps(var, eps))
    return qgdn._mul(qgdn._c(np.ascontiguousarray(weight, np.float32).reshape(1, 1, -1)), xn)


def _qwen35_layer_state(ar, cfg, kind, layer, feed=None, census=None):
    """Sparse-declared state for one `qwen35moe` decoder layer at the module-
    relative keys `q4e.gdn` / `emit_stateful_attention` / `emit_moe_tiled`
    consume. With `feed` the buffers are written from the real shards."""
    H = cfg.hidden_size
    st = {"input_norm.weight": ar.f32([H]),
          "post_attention_norm.weight": ar.f32([H])}
    if kind == "gdn":
        kd, kh = cfg.linear_key_head_dim, cfg.linear_num_key_heads
        vd, vh = cfg.linear_value_head_dim, cfg.linear_num_value_heads
        conv_dim = kd * kh * 2 + vd * vh
        st["linear_attn.in_proj_qkv.weight"] = ar.f32([conv_dim, H])
        st["linear_attn.in_proj_z.weight"] = ar.f32([vd * vh, H])
        st["linear_attn.in_proj_a.weight"] = ar.f32([vh, H])
        st["linear_attn.in_proj_b.weight"] = ar.f32([vh, H])
        st["linear_attn.A_log"] = ar.f32([vh])
        st["linear_attn.dt_bias"] = ar.f32([vh])
        st["linear_attn.conv1d.weight"] = ar.f32([conv_dim, 1, cfg.linear_conv_kernel_dim])
        st["linear_attn.norm.weight"] = ar.f32([vd])
        st["linear_attn.out_proj.weight"] = ar.f32([H, vd * vh])
    else:
        heads, kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        st["self_attn.q_proj.weight"] = ar.f32([heads * 2 * d, H])
        st["self_attn.k_proj.weight"] = ar.f32([kv * d, H])
        st["self_attn.v_proj.weight"] = ar.f32([kv * d, H])
        st["self_attn.o_proj.weight"] = ar.f32([H, heads * d])
        st["self_attn.q_norm.weight"] = ar.f32([d])
        st["self_attn.k_norm.weight"] = ar.f32([d])
    Is = cfg.shared_expert_intermediate_size
    st["mlp.gate.weight"] = ar.f32([cfg.num_experts, H])
    st["mlp.shared_expert.gate_proj.weight"] = ar.f32([Is, H])
    st["mlp.shared_expert.up_proj.weight"] = ar.f32([Is, H])
    st["mlp.shared_expert.down_proj.weight"] = ar.f32([H, Is])
    st["mlp.shared_expert_gate.weight"] = ar.f32([1, H])
    if feed is not None:
        _fill_qwen35_layer(st, feed, layer, census)
    return st


def _fill_qwen35_layer(st, feed, layer, census):
    """Write the real GGUF rows into the layer's arena buffers. Each axis is a
    leading slice to the buffer's shape; no fused axis is narrowed here (`FIX
    D` applies only to the fused `gate_up_proj`, which this emitter does not
    use -- the routed experts are filled by `NativeExpertFiller`)."""
    for key, buf in st.items():
        suffix, kind = _QWEN35MOE_TENSORS[key]
        gname = f"blk.{layer}.{suffix}"
        arr = feed.mapped(gname, kind)
        arr = np.ascontiguousarray(arr[tuple(slice(0, int(s)) for s in buf.shape)])
        assert arr.shape == tuple(int(s) for s in buf.shape), (
            f"{gname}: fed {arr.shape} != buffer {tuple(buf.shape)}")
        buf[...] = arr
        if census is not None:
            census.append((gname, int(buf.nbytes)))


def build_qwen35moe_serving_shape_ir(config=None, arena=None, n_layers=None,
                                     filler=None, feed=None, rope_span=None):
    """The `Qwen3.6-35B-A3B` (`qwen35moe`) serving-shape backbone as an
    ov::Model, DYNAMIC IN T, with a PLAIN PRE-NORM RESIDUAL layer (no
    hyper-connection, no PLE):

        hidden = hidden + mixer(rmsnorm(hidden, attn_norm))
        hidden = hidden + moe(rmsnorm(hidden, post_attention_norm))

    Mixers: `q4e.gdn` (30 linear-attention layers, tiled key-head map,
    stateful conv + token-sequential core) and `emit_stateful_attention` (10
    full-attention layers at i % 4 == 3, fused q|gate split, silu output gate,
    shared rope tables). The MoE layer is `emit_moe_tiled` with the native
    filler, so the expert bodies carry the checkpoint's own IQ2_S (gate/up) and
    IQ3_XXS / IQ4_XS (down) blocks.

    Ports: `inputs_embeds`, `position_ids`, `conv_mask`, `attention_mask`,
    `beam_idx`; state via the same Variables `stateful_short_conv` /
    `stateful_gdn_core` / `emit_stateful_attention` carry. The embedding is NOT
    in this graph (`tools/export_serving_artifact.py` emits it separately).

    Returns (model, report); the report keys mirror `build_serving_shape_ir`'s
    so the exporter can share its manifest path.
    """
    cfg = config if config is not None else qwen35moe_real_config()
    T = -1
    own_arena = arena is None
    ar = arena if arena is not None else SparseArena()
    n_total = int(cfg.num_hidden_layers)
    depth = int(n_layers if n_layers is not None else n_total)
    if not (1 <= depth <= n_total):
        raise ValueError(f"n_layers {depth} outside 1..{n_total}")
    nl = depth
    H = cfg.hidden_size
    V = cfg.vocab_size
    sinks = []
    dense_census = []
    try:
        with shared_constants():
            inputs_embeds = op.parameter([1, T, H], Type.f32)
            inputs_embeds.set_friendly_name("inputs_embeds")
            inputs_embeds.output(0).set_names({"inputs_embeds"})
            pid = op.parameter([1, T], Type.i64)
            pid.set_friendly_name("position_ids")
            pid.output(0).set_names({"position_ids"})
            conv_mask = op.parameter([1, T], Type.f32)
            conv_mask.set_friendly_name("conv_mask")
            attn_mask = op.parameter([1, -1], Type.i64)
            attn_mask.set_friendly_name("attention_mask")
            attn_mask.output(0).set_names({"attention_mask"})
            beam = op.parameter([-1], Type.i32)
            beam.set_friendly_name("beam_idx")
            beam.output(0).set_names({"beam_idx"})

            span = int(rope_span if rope_span is not None
                       else cfg.max_position_embeddings)
            cos_np, sin_np = qattn._freqs_tables(cfg, span)
            rope_cos = op.constant(cos_np)
            rope_cos.set_friendly_name("rope/cos")
            rope_sin = op.constant(sin_np)
            rope_sin.set_friendly_name("rope/sin")

            hidden = op.reshape(inputs_embeds,
                                op.constant(np.array([1, -1, H], np.int64)),
                                special_zero=False)
            hidden.set_friendly_name("embed/out")
            kinds = []
            for i in range(depth):
                kind = "attn" if (i % 4) == 3 else "gdn"
                kinds.append(kind)
                st = _qwen35_layer_state(ar, cfg, kind, i, feed, dense_census)
                h = _qwen35_rmsnorm(hidden, st["input_norm.weight"], cfg.rms_norm_eps)
                h.set_friendly_name(f"layer{i}/attn_norm")
                if kind == "gdn":
                    g = qgdn.emit_gdn(
                        h, conv_mask, cfg, _strip(st, "linear_attn."), None,
                        conv_emitter=stateful_short_conv(i, beam, sinks),
                        core_emitter=gdn_core_emitter(i, beam, sinks))
                else:
                    g = emit_stateful_attention(
                        h, pid, cfg, _strip(st, "self_attn."), i, beam,
                        attn_mask, sinks, rope_cos, rope_sin)
                hidden = op.add(hidden, g)
                hidden.set_friendly_name(f"layer{i}/mixer_out")
                h2 = _qwen35_rmsnorm(hidden, st["post_attention_norm.weight"],
                                     cfg.rms_norm_eps)
                h2.set_friendly_name(f"layer{i}/post_norm")
                m = emit_moe_tiled(h2, cfg, st, ar, T, f"layer{i}/moe",
                                   filler=filler, layer=i)
                hidden = op.add(hidden, m)
                hidden.set_friendly_name(f"layer{i}/out")

            final_norm_w = ar.f32([H])
            if feed is not None:
                arr = feed.mapped("output_norm.weight", "vec")[:H]
                final_norm_w[...] = np.ascontiguousarray(arr, np.float32)
                dense_census.append(("output_norm.weight", int(final_norm_w.nbytes)))
            fin = _qwen35_rmsnorm(hidden, final_norm_w, cfg.rms_norm_eps)
            fin.set_friendly_name("final_norm")
            head_w = ar.f32([V, H])
            if feed is not None:
                arr = feed.mapped("output.weight", "direct2d")[:V, :H]
                head_w[...] = np.ascontiguousarray(arr, np.float32)
                dense_census.append(("output.weight", int(head_w.nbytes)))
            logits = op.matmul(fin, qgdn._c(head_w), transpose_a=False, transpose_b=True)
            res = op.result(logits)
            res.set_friendly_name("logits")
            res.output(0).set_names({"logits"})
            model = Model([res], sinks, [inputs_embeds, pid, conv_mask, attn_mask, beam],
                          "qwen3_5_moe_serving_shape")

        nodes, const_bytes, counts = pwe.graph_measures(model)
        report = {
            "n_layers": nl,
            "layer_range": [0, depth],
            "segment_first": True,
            "segment_last": True,
            "inputs_embeds_width": int(H),
            "has_ple": False,
            "expert_ports": [],
            "gdn_layers": kinds.count("gdn"),
            "attn_layers": kinds.count("attn"),
            "seq_len": None,
            "rope_span": span,
            "nodes": nodes,
            "graph_const_bytes": const_bytes,
            "op_histogram": counts,
            "ngram_table_rows": 0,
            "ngram_staging_rows": None,
            "ngram_row_bytes": 0,
            "ngram_chunk_cap_bytes": int(NGRAM_CHUNK_CAP_BYTES),
            "ngram_table_ports": [],
            "dense_fill_census": dense_census,
            "arena_declared_bytes": ar.declared_bytes,
            "arena_written_bytes": ar.written_bytes,
            "arena_disk_kib": ar.disk_kib(),
            "fill_census": filler.census() if filler is not None else None,
            "inputs": [(p.get_node().get_friendly_name(), _dims(p),
                        str(p.get_element_type())) for p in model.inputs],
            "outputs": [(r.get_node().get_friendly_name(), _dims(r),
                         str(r.get_element_type())) for r in model.outputs],
        }
        return model, report
    except BaseException:
        if own_arena:
            ar.close()
        raise


# --------------------------------------------------------------------------
# The OTD contract, transcribed from the C++ so the test checks code, not prose
# --------------------------------------------------------------------------

def slot_pool_from_ir(model, num_expert, ratio_pct):
    """Python transcription of `slot_pool_from_ir`, src/exec/backend_ov.cpp:578-624.

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
            # :604 -- element_type().size(), which CEILS a sub-byte width to a
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
    """src/exec/fit.h:96 -- ceil(num_expert * (100 - ratio) / 100) slots per
    layer, times per-expert bytes, times layers."""
    slots = -((-num_expert * (100 - ratio_pct)) // 100)
    return slots * per_expert_bytes * moe_layers


__all__ = [
    "SparseArena", "shared_constants", "build_serving_shape_ir",
    "build_qwen35moe_serving_shape_ir", "qwen35moe_real_config", "QWEN35MOE_LM",
    "emit_moe_tiled", "emit_stateful_attention", "stateful_short_conv",
    "stateful_gdn_core", "slot_pool_from_ir",
    "ngram_table_chunks", "ngram_table_ports", "ngram_chunked_gather",
    "ngram_dequant_iq4nl", "ngram_row_bytes",
    "EXPERT_DECLARED_TYPE", "NGRAM_PORT_TYPE", "NGRAM_BLOCK_ELEMS",
    "NGRAM_BLOCK_BYTES", "NGRAM_IQ4NL_KVALUES",
    "NGRAM_CHUNK_CAP_BYTES", "NGRAM_CHUNK_ROW_ALIGN", "EXPERT_GROUP_SIZE",
]
