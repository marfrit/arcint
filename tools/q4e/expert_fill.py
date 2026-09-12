"""THE FILL: real Q3_K_XL GGUF rows into the u4 expert bodies of the
serving-shape IR.

`q4e.serving_shape` emits the 48-layer backbone at real geometry with every
expert body DECLARED and never materialised -- 183 GiB of constants over zero
blocks on disk. That artifact proves the shape. It carries no weights, so
nothing numeric can be asked of it.

This module carries the weights. It reads the shipped UD-Q3_K_XL shards through
`q4e.gguf_feed`, quantises each expert row to the u4 grouped-affine form the
tiled lowering declares, packs it the way the C++ unpacks it, and writes it
into the arena pages the Constants are built over. The result is an ov::Model
whose expert bodies hold the real checkpoint.

--------------------------------------------------------------------------
1. THE PACKING CONTRACT, BOTH SIDES, AND IT WAS MEASURED RATHER THAN ASSUMED
--------------------------------------------------------------------------

arcint's own u4 convention is stated at `src/core/gguf_repack.h:90`:

    std::vector<uint8_t> weights;   // [n][(k + k_aug) values]; u4: two per
                                    // byte, even index low nibble

and executed at `src/core/gguf_repack.cpp:225` (`main_value`, the host
emulation of the plugin's arithmetic) and again at `:476`:

    if (r.weights_type == RepackWeights::U4)
        q = static_cast<float>((r.weights[idx / 2] >> ((idx & 1) * 4)) & 0xF);

so element `idx` is the LOW nibble of byte `idx/2` when `idx` is even and the
HIGH nibble when it is odd.

That is arcint's convention. Whether OPENVINO agrees is a different question,
and the whole fill rests on it: a mismatch swaps every pair of weights and
produces a model that loads, runs, and is wrong. Production already bets on it
-- `src/exec/gguf_graph.cpp:222-223` hands a `RepackedTensor::weights` buffer
straight to an `ov::op::v0::Constant(ov::element::u4, ...)` -- but the bet was
never measured. It is now, on the dev host, OV 2026.4.0-22849:

    bytes 0x21 0x43 as u4[4] -> [1, 2, 3, 4]     (the C++ contract's prediction;
                                                  the other order gives [2,1,4,3])
    4096-element random round trip: exact, 0 mismatches

`tests/python/test_expert_fill.py::test_openvino_u4_element_order_is_the_cpp_contract`
keeps it measured.

--------------------------------------------------------------------------
2. THE QUANTISATION, AND THE ROUNDING IT IS ALLOWED TO COST
--------------------------------------------------------------------------

The tiled lowering's dequant chain is
`Convert(f32) -> Subtract(zero_point) -> Multiply(scale) -> Reshape(4->3)`
(serving_shape._compressed_expert), so the value the graph computes for code
`q` in group `g` is

    v_hat = (q - zp_g) * s_g          with q, zp_g integers in [0, 15]

a GROUPED AFFINE code, one (scale, zero-point) pair per `EXPERT_GROUP_SIZE`
consecutive elements of the CONTRACTION axis -- which is what the rank-4
[E, out, groups, group_size] shape means.

THE ROUNDING, NAMED, because the acceptance may not assert below it: with 16
levels spanning a group's range, the step is

    s_g = (max_g - min_g) / 15

and round-to-nearest puts every reconstructed value within HALF A STEP of the
original:

    |v - v_hat| <= s_g / 2 = (max_g - min_g) / 30

`quantise_group_affine` below guarantees that bound BY CONSTRUCTION rather
than by hope, and the guarantee needs one step most implementations skip. The
zero-point is itself an integer in [0, 15], so `round(-min/s)` generally does
not land on `-min/s` exactly, and the representable interval
`[-zp*s, (15-zp)*s]` then fails to cover `[min, max]` -- values outside it
CLIP, and a clipped value can be a whole step out instead of half. So after
rounding the zero-point the scale is ENLARGED to whatever covers the group's
real range at that zero-point. That costs a little resolution and buys an
exact, provable bound; the test asserts the bound AND that it is tight, since
a bound nothing approaches would pass vacuously.

Degenerate groups are handled explicitly rather than by an epsilon: a group
whose values are all equal has range 0 and no step, and is represented
EXACTLY (`q - zp = +/-15` under `s = |v|/15`, or the all-zero group at
`s = 0`).

--------------------------------------------------------------------------
3. WHAT THIS MODULE IS NOT
--------------------------------------------------------------------------

It is not a re-quantisation strategy and makes no accuracy claim about the
model. The shipped checkpoint is Q3_K_XL -- a mixed k-quant -- and
`q4e.gguf_feed` hands back its DEQUANTISED f32. Going f32 -> u4 grouped-affine
here is a SECOND quantisation, and it loses more than the file did. Whether
the serving artifact should instead carry the file's own k-quant blocks
through `src/core/gguf_repack.cpp`'s repack path -- which is what
`gguf_apply_to_template` does for the dense models arcint serves today -- is a
real question and it is NOT decided here. It is written down as a frontier
item in docs/window-050.md, with what would have to be measured to decide it.
What this module establishes is that the serving-shape IR's expert bodies can
be FILLED, byte-addressably, from the real checkpoint, and that what comes
back out through the documented unpack is what went in.
"""
import numpy as np

# The u4 packing contract, from src/core/gguf_repack.h:90 and executed at
# src/core/gguf_repack.cpp:225. Written as a named constant so a reader meets
# the convention before the shifts.
U4_EVEN_INDEX_IS_LOW_NIBBLE = True

# 16 levels, so 15 steps span a group's range.
U4_LEVELS = 16
U4_STEPS = U4_LEVELS - 1


def pack_u4(q):
    """Pack integer codes 0..15 into bytes, even index in the low nibble.

    `q` is flattened in C order, which is the order an `ov.Tensor(buf, shape,
    Type.u4)` reads its buffer in. An odd element count is not accepted: every
    shape this module packs has an even trailing dimension (the group size),
    so an odd total would mean the caller has the layout wrong, and silently
    padding would hide that.
    """
    q = np.ascontiguousarray(q).ravel()
    assert q.size % 2 == 0, (
        f"{q.size} u4 codes: odd counts do not pack into whole bytes, and "
        f"every expert-body shape here has an even group size")
    assert q.min() >= 0 and q.max() <= 15, (
        f"u4 codes out of range: [{q.min()}, {q.max()}], expected [0, 15]")
    q = q.astype(np.uint8)
    return (q[0::2] | (q[1::2] << 4)).astype(np.uint8)


def unpack_u4(buf, count):
    """THE C++ UNPACK, replicated: `(weights[idx/2] >> ((idx & 1) * 4)) & 0xF`.

    Transcribed from `src/core/gguf_repack.cpp:225` rather than derived from
    `pack_u4`, so the acceptance reads the bytes back the way the C++ would
    and not the way this file wrote them. Inverting the packer with the packer
    would measure nothing.
    """
    buf = np.ascontiguousarray(buf, dtype=np.uint8).ravel()
    idx = np.arange(int(count), dtype=np.int64)
    return ((buf[idx // 2] >> ((idx & 1) * 4)) & 0xF).astype(np.uint8)


def quantise_group_affine(x, group_size):
    """f32 -> (codes, zero_points, scales) with |v - v_hat| <= scale/2 exactly.

    `x` is [..., inn]; `inn` must be a whole number of groups. Returns
        q      [..., groups, group_size]  uint8, 0..15
        zp     [..., groups, 1]           uint8, 0..15
        scale  [..., groups, 1]           float32
    laid out as the rank-4 [E, out, groups, group_size] the tiled lowering
    declares, so the caller packs and writes without reshaping again.

    See this module's header for why the scale is enlarged after the
    zero-point is rounded: without that step a value at the end of a group's
    range CLIPS, and clipping costs a whole step where the bound promises
    half.
    """
    x = np.asarray(x, dtype=np.float32)
    inn = x.shape[-1]
    assert inn % group_size == 0, (
        f"inner dimension {inn} is not a multiple of group {group_size}")
    g = x.reshape(*x.shape[:-1], inn // group_size, group_size)

    # The per-group extrema and the scale are derived in f64 -- they are one
    # value per group, so the cost is negligible and the rounding of the scale
    # itself should not eat into the budget. The CODES are then derived in f32
    # against the ALREADY-ROUNDED f32 scale, because that is the number the
    # graph's Multiply will use: quantising against an f64 scale the graph
    # never sees would put the error slightly outside the bound it promises.
    lo = g.min(-1, keepdims=True).astype(np.float64)
    hi = g.max(-1, keepdims=True).astype(np.float64)

    scale = (hi - lo) / U4_STEPS
    degenerate = scale <= 0.0                       # every value in the group equal

    # a non-degenerate group: round the zero-point, then widen the scale until
    # the rounded zero-point's representable interval covers [lo, hi].
    safe = np.where(degenerate, 1.0, scale)
    zp = np.clip(np.round(-lo / safe), 0, U4_STEPS)
    need_lo = np.where(zp > 0, -lo / np.where(zp > 0, zp, 1.0), 0.0)
    need_hi = np.where(zp < U4_STEPS, hi / np.where(zp < U4_STEPS,
                                                   U4_STEPS - zp, 1.0), 0.0)
    scale = np.maximum(safe, np.maximum(need_lo, need_hi))

    # a degenerate group: represent the single value EXACTLY.
    #   v == 0  -> scale 0, zp 0, q 0
    #   v  > 0  -> scale v/15,   zp 0,  q 15   -> (15 - 0) * v/15  == v
    #   v  < 0  -> scale |v|/15, zp 15, q 0    -> (0 - 15) * |v|/15 == v
    dv = lo
    scale = np.where(degenerate, np.abs(dv) / U4_STEPS, scale)
    zp = np.where(degenerate, np.where(dv < 0, U4_STEPS, 0), zp)

    scale = scale.astype(np.float32)
    zp = zp.astype(np.float32)
    nz = scale > 0.0
    q = np.where(nz,
                 np.round(np.divide(g, np.where(nz, scale, np.float32(1.0)),
                                    dtype=np.float32) + zp),
                 np.where(degenerate & (dv > 0), np.float32(U4_STEPS),
                          np.float32(0.0)))
    q = np.clip(q, 0, U4_STEPS)

    return (q.astype(np.uint8), zp.astype(np.uint8), scale)


def dequantise_affine(q, zp, scale):
    """The dequant the IR's own chain computes: `(q - zp) * scale`, f32.

    Convert -> Subtract(zero_point) -> Multiply(scale), in f32, which is what
    `serving_shape._compressed_expert` emits. Note this is NOT
    `gguf_repack.cpp`'s `main_value`, which rounds each intermediate to f16
    because the repack path's scales are f16 (`gguf_repack.h:138-141`); the
    serving-shape chain carries f32 scales and does not.
    """
    return ((q.astype(np.float32) - zp.astype(np.float32))
            * scale.astype(np.float32))


def quantisation_step_bound(scale):
    """Half a step -- the rounding the u4 representation is entitled to, and
    the floor an acceptance may not assert below. Elementwise, broadcast over
    a group."""
    return np.asarray(scale, dtype=np.float32) / 2.0


class ExpertFiller:
    """Serves quantised expert bodies to `serving_shape._compressed_expert`
    and RECORDS WHAT IT SERVED.

    The census is the structural acceptance: a body served twice, or a body
    never served, is a defect the numbers would not show -- a double-written
    body still dequantises to something plausible, and an unwritten one is a
    block of zeros that a parity leg on OTHER experts would never reach. So
    every request is logged by (layer, kind) and the totals are GENERATED from
    that log rather than written down.

    `source(layer, kind)` returns the f32 rows for one expert-stacked weight,
    shaped [E, out, inn]. `ExpertFiller` does not know about GGUF; the caller
    supplies that, which keeps this class testable against synthetic rows.
    """

    def __init__(self, source, group_size, expert_chunk=32):
        self._source = source
        self.group_size = int(group_size)
        # Experts quantised per pass. At real width one expert-stacked weight
        # is 3.12 GiB of f32 and the quantiser's temporaries are the same size
        # again; chunking the EXPERT axis bounds that without changing a byte
        # of the result, because a group never spans two experts (it is a run
        # of `group_size` along the contraction axis, and the expert axis is
        # the leading one). Measured at real geometry: 512 experts in one pass
        # peaked past 8 GiB, in chunks of 32 it does not.
        self.expert_chunk = int(expert_chunk)
        self.served = []            # (layer, kind, shape, nbytes)
        self._seen = {}             # (layer, kind) -> how many times

    def body(self, layer, kind, e, out, inn):
        """(packed_weights, packed_zero_points, scales) for one expert body."""
        key = (layer, kind)
        self._seen[key] = self._seen.get(key, 0) + 1
        rows = np.asarray(self._source(layer, kind), dtype=np.float32)
        assert rows.shape == (e, out, inn), (
            f"{key}: source returned {rows.shape}, the IR declares "
            f"{(e, out, inn)} -- the fill must not reshape silently")
        wparts, zparts, sparts = [], [], []
        for lo in range(0, e, self.expert_chunk):
            hi = min(lo + self.expert_chunk, e)
            q, zp, scale = quantise_group_affine(rows[lo:hi], self.group_size)
            wparts.append(pack_u4(q))
            zparts.append(pack_u4(zp))
            sparts.append(scale)
        packed_w = np.concatenate(wparts)
        packed_zp = np.concatenate(zparts)
        scales = np.concatenate(sparts, axis=0)
        self.served.append((layer, kind, (e, out, inn),
                            int(packed_w.size + packed_zp.size
                                + scales.nbytes)))
        return packed_w, packed_zp, scales

    def census(self):
        """Generated, not recited: what was filled, and whether any body was
        served twice."""
        doubles = {k: n for k, n in self._seen.items() if n != 1}
        layers = sorted({l for l, _ in self._seen})
        kinds = sorted({k for _, k in self._seen})
        return {
            "bodies": len(self.served),
            "layers": layers,
            "kinds": kinds,
            "double_written": doubles,
            "filled_bytes": sum(n for _, _, _, n in self.served),
            "experts": sorted({s[0] for _, _, s, _ in self.served}),
        }


def gguf_expert_source(feed, layer_of=None):
    """An `ExpertFiller` source reading the shipped shards.

    The serving-shape IR wants gate and up SEPARATELY; `gguf_feed`'s
    `mlp.experts.gate_up_proj` is the pin's FUSED [E, 2*ff, in]. The halves are
    taken at the real ff by construction here (`arr[:, :ff]` / `arr[:, ff:]`
    with ff = shape[1] // 2), which is the split `fitted()`'s own `fuse_ff`
    path performs -- never a leading slice of the concatenation, which is FIX D
    (REVIEW 2cd2b2f finding D): a leading slice takes BOTH halves out of gate.

    `layer_of` maps an IR layer index to a GGUF block index, for the case
    where the IR emits fewer layers than the checkpoint ships.
    """
    def _source(layer, kind):
        blk = layer_of(layer) if layer_of else layer
        if kind == "down":
            return feed.pin_tensor(f"layers.{blk}.mlp.experts.down_proj")
        fused = feed.pin_tensor(f"layers.{blk}.mlp.experts.gate_up_proj")
        ff = fused.shape[1] // 2
        return fused[:, :ff] if kind == "gate" else fused[:, ff:]
    return _source


__all__ = [
    "U4_EVEN_INDEX_IS_LOW_NIBBLE", "U4_LEVELS", "U4_STEPS",
    "pack_u4", "unpack_u4", "quantise_group_affine", "dequantise_affine",
    "quantisation_step_bound", "ExpertFiller", "gguf_expert_source",
]
