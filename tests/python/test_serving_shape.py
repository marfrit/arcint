"""THE CONTRACT TEST AS HANDSHAKE: the serving-shape IR against what the C++
serving path actually expects, tensor-for-tensor-name.

A green contract turns the 0.5.0 window from an experiment into a boot: if the
IR the exporter emits satisfies, name for name and shape for shape, what
`src/exec/backend_ov.cpp` reads at load time, then the remaining risk is the
card and not the artifact. Everything here is device-free and needs no shards:
it reads an ov::Model built over sparse pages.

TWO KINDS OF CELL, and the distinction is the point:

  * MET -- the export side satisfies the C++ side, asserted directly.
  * NOT MET AND NAMED -- a strict xfail that carries the exact file:line where
    the handshake dies. `strict=True` means the day someone closes the gap the
    cell FAILS, which is how a known gap gets retired instead of forgotten.

WHAT THIS FILE FOUND, on its first run (2026-09-12), and it is a finding about
the C++ and not about the export:

`slot_pool_from_ir` (backend_ov.cpp:577-623) identifies a MoE layer by
    std::string tname = node->get_type_name();  ... tolower ...
    if (tname.find("moe") == std::string::npos) continue;      // :585
NO ARCINT-EXPORTED IR CARRIES AN OP WHOSE TYPE NAME CONTAINS "moe". Measured
over the whole model store on the dev host, 2026-09-12, widened by
REVIEW 2a45349 and re-run independently here:

    find /models/ov -name '*.xml' -size +100k | wc -l   ->  52
    find /models/ov -name '*.xml'             | wc -l   -> 172   (no filter)
    ... of 172, carrying a <layer type="...moe...">     ->   0
    ... of 172, carrying the string "moe" at all        ->   0

The negative is stronger than this file first stated it: 0 of 172, not 0 of 52.
The population includes `qwen36-35b-a3b-int4-ov/openvino_language_model.xml`,
the 35B-A3B MoE checkpoint that is the ground truth `moe_block_tiled` was extracted from
(export_mtp.py:406-409). Its op histogram is Const/Convert/Multiply/Reshape/
Subtract/MatMul/Swish -- the tiled dequant chain -- and no fused MoE node,
because the fusion (`ConvertTiledMoeBlockToGatherMatmuls`) is a GPU-PLUGIN
COMPILE-TIME pass and `slot_pool_from_ir` runs on `read_model`, before it.

So the analytic IR route returns nullopt on every artifact this fleet serves,
and the `else` branch at backend_ov.cpp:3746+ (3 x hidden x moe_intermediate x
bytes-per-weight per expert, from config.json) is what has always run. This is
not a defect the serving-shape IR introduces and it is not one it can fix from
the export side: an exporter cannot give a node a different OpenVINO type name.
It is recorded here, with the line, because the mission's standard is that the
first failure named exactly is worth more than a success.

`slot_pool_from_tiled_ir` below is the matcher that WOULD work on the shape
arcint actually exports -- pattern, not type name -- and its figures are
cross-checked against `src/exec/flash_next_offload.h:45`
(kFlashNextSliceBytes = 2,457,600 B per expert-layer for gate+up+down).
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import openvino as ov  # noqa: E402

from q4e import piecewise_export as pwe  # noqa: E402
from q4e import serving_shape as ss  # noqa: E402

# A real-geometry build of the whole 48-layer stack is expensive; the contract
# is per-layer and the layer kinds repeat with period 4, so 8 layers exercises
# 6 GDN + 2 dense-causal-attention layers AND the PLE layer (index 1). The
# full-stack build is its own cell, opt-in via Q4E_SERVING_FULL=1, because it
# is the keystone claim and must be run deliberately.
_CONTRACT_LAYERS = 8
_T = 8

# A DATED HOST CENSUS, NOT A GENERATED COUNT -- and the distinction is the
# point of writing it this way. This suite cannot regenerate these figures: a
# staged tree has no /models/ov, and the population is the dev host's model
# store rather than anything this repository owns. So they are recorded with
# their date and their exact command, and the only thing asserted below is the
# ZERO, which is the load-bearing half.
#
# Measured 2026-09-12, twice -- first by 198b736 over the size-filtered
# population, then WIDER by REVIEW 2a45349 and re-run here independently:
#
#   A  find /models/ov -name '*.xml' -size +100k        -> 52 IRs
#   B  find /models/ov -name '*.xml'   (no size filter) -> 172 IRs
#   C  of B, carrying a <layer type="...moe..."> (case-insensitive) -> 0
#   D  of B, carrying the string "moe" ANYWHERE, names included     -> 0
#
# C is the right method because the C++ test is
# `lowercase(get_type_name()).find("moe")`, and in an IR an op's type name is
# exactly the `<layer ... type="...">` attribute. D is the widest form the
# claim could take and it also holds. The negative is stronger than 198b736
# stated it: 0 of 172, not 0 of 52.
FLEET_IRS_SIZE_FILTERED = 52          # population A, 198b736's own
FLEET_IRS_ALL = 172                   # population B, the wider one
FLEET_IRS_WITH_MOE_TYPED_OP = 0       # C, and D too


@pytest.fixture(scope="module")
def built():
    arena = ss.SparseArena()
    model, report = ss.build_serving_shape_ir(
        seq_len=_T, arena=arena, n_layers=_CONTRACT_LAYERS)
    yield model, report, arena
    arena.close()


# ---------------------------------------------------------------------------
# MET -- the port contract, tensor for tensor name
# ---------------------------------------------------------------------------

def test_the_input_ports_are_the_names_and_shapes_the_serving_path_feeds(built):
    """Names, shapes and element types, exactly.

    `ngram_row_ids` is [1, T, 16] i64. 16 is not a choice: it is
    `HashParams::num_ngram_heads() = (ngram_size - 1) * heads_per_ngram`
    (src/exec/ngram_row_ids.h:59), which the same file's header states at :21
    as "16 on Qwen3.8: 8 x 2-gram + 8 x 3-gram", and each head gathers one
    160-wide row (:22). `position_ids` is the name backend_ov.cpp:98 declares
    (`kPositionIds`). `conv_mask` is the port q4e.backbone already declares
    (backbone.py:105-106).
    """
    model, report, _ = built
    cfg = pwe.real_config()
    Hn = (cfg.ngram_size - 1) * cfg.heads_per_ngram
    assert Hn == 16, f"num_ngram_heads moved to {Hn}; ngram_row_ids' 16 is derived"

    got = {name: (tuple(shape), etype)
           for name, shape, etype in report["inputs"]}
    want = {
        "input_ids":     ((1, _T), "int64_t"),
        "position_ids":  ((1, _T), "int64_t"),
        "ngram_row_ids": ((1, _T, Hn), "int64_t"),
        "conv_mask":     ((1, _T), "float32"),
    }
    print("\n[contract-ports] inputs:")
    for k in sorted(got):
        print(f"  {k:16s} {got[k][0]}  {got[k][1]}")
    for name, (shape, et) in want.items():
        assert name in got, f"input port {name!r} missing; got {sorted(got)}"
        assert got[name][0] == shape, f"{name}: shape {got[name][0]} != {shape}"
        assert et in got[name][1], f"{name}: type {got[name][1]} != {et}"
    assert set(got) == set(want), (
        f"unexpected input ports: {sorted(set(got) - set(want))}")


def test_the_output_is_logits_at_the_real_vocabulary(built):
    model, report, _ = built
    cfg = pwe.real_config()
    outs = {n: (tuple(s), t) for n, s, t in report["outputs"]}
    print(f"[contract-ports] outputs: {outs}")
    assert list(outs) == ["logits"], outs
    assert outs["logits"][0] == (1, _T, cfg.vocab_size), outs["logits"]
    assert "float32" in outs["logits"][1]


def test_the_layer_kinds_follow_the_checkpoint_not_a_convenience(built):
    """12 of 48 blocks are full-attention at layer_idx % 4 == 3
    (piecewise_export.REAL_GEOMETRY:154-156). At 8 layers that is 6 GDN + 2."""
    _, report, _ = built
    print(f"[contract-layers] gdn={report['gdn_layers']} "
          f"attn={report['attn_layers']} of {report['n_layers']}")
    assert report["attn_layers"] == _CONTRACT_LAYERS // 4
    assert report["gdn_layers"] == _CONTRACT_LAYERS - report["attn_layers"]


# ---------------------------------------------------------------------------
# MET -- the expert bodies are declared, slot-referenced, never materialised
# ---------------------------------------------------------------------------

def _expert_constants(model, num_expert):
    """Every Constant whose leading dimension is the expert count -- the same
    selector backend_ov.cpp:600 uses, applied without the type-name gate."""
    out = []
    for node in model.get_ordered_ops():
        if node.get_type_name() != "Constant":
            continue
        sh = list(node.get_output_shape(0))
        if sh and sh[0] == num_expert:
            out.append((node.get_friendly_name(), tuple(sh),
                        node.get_output_element_type(0)))
    return out


def test_expert_bodies_are_rank4_u4_and_one_gate_up_down_triple_per_layer(built):
    """The tiled lowering's weight shape, at real geometry.

    verify_moe_lowering.py:29-31: "expert weights as rank-4
    [E,out,groups,group_size] Constants in a compressed integer type
    (u4/i4/u8/i8)". Three per MoE layer -- gate, up, down -- plus their
    zero-points, which carry the same leading dimension by construction.
    """
    model, report, _ = built
    cfg = pwe.real_config()
    E, I, H = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
    gs = ss.EXPERT_GROUP_SIZE
    consts = _expert_constants(model, E)
    weights = [c for c in consts if c[0].endswith("/weight_u4")]
    zps = [c for c in consts if c[0].endswith("/zero_point")]

    print(f"\n[contract-experts] E={E} I={I} H={H} group_size={gs}")
    print(f"  expert weight constants {len(weights)}, zero-points {len(zps)}, "
          f"layers {report['n_layers']}")
    for name, sh, et in weights[:3]:
        print(f"  {name:34s} {sh}  {et}")

    assert len(weights) == 3 * report["n_layers"], (
        f"{len(weights)} expert weight constants for {report['n_layers']} "
        f"layers; expected gate/up/down per layer")
    assert len(zps) == len(weights), "every expert weight needs its zero-point"

    want = {(E, I, H // gs, gs), (E, H, I // gs, gs)}
    for name, sh, et in weights:
        assert len(sh) == 4, f"{name}: rank {len(sh)}, the tiled pass wants 4"
        assert sh in want, f"{name}: {sh} not in {want}"
        assert et == ss.EXPERT_DECLARED_TYPE, f"{name}: {et}"


def test_the_dequant_chain_is_convert_subtract_multiply_reshape(built):
    """The chain the fusing pass matches, INCLUDING the trailing Reshape --
    verify_moe_lowering.py:33-42 records a real GPU compile crashing inside the
    pass's own rewrite when that Reshape was absent."""
    model, _, _ = built
    cfg = pwe.real_config()
    E = cfg.num_experts
    reshapes = [n for n in model.get_ordered_ops()
                if n.get_type_name() == "Reshape"
                and n.get_friendly_name().endswith("/dequant_reshape")]
    assert reshapes, "no dequant Reshape found; the matcher anchors on it"
    walked = 0
    for r in reshapes:
        mul = r.input_value(0).get_node()
        assert mul.get_type_name() == "Multiply", \
            f"{r.get_friendly_name()}: parent is {mul.get_type_name()}, want Multiply"
        sub = mul.input_value(0).get_node()
        assert sub.get_type_name() == "Subtract", \
            f"{r.get_friendly_name()}: want Subtract, got {sub.get_type_name()}"
        conv = sub.input_value(0).get_node()
        assert conv.get_type_name() == "Convert", \
            f"{r.get_friendly_name()}: want Convert, got {conv.get_type_name()}"
        const = conv.input_value(0).get_node()
        assert const.get_type_name() == "Constant"
        assert list(const.get_output_shape(0))[0] == E
        assert const.get_output_element_type(0) == ss.EXPERT_DECLARED_TYPE
        # rank 4 in, rank 3 out -- the collapse the pass looks for
        assert len(list(const.get_output_shape(0))) == 4
        assert len(list(r.get_output_shape(0))) == 3
        walked += 1
    print(f"[contract-dequant] walked {walked} Convert->Subtract->Multiply->"
          f"Reshape(4->3) chains, all anchored on a u4 [E,...] Constant")


def test_no_expert_constant_is_materialised(built):
    """The arena declares tens of GiB of constants and occupies zero blocks on
    disk, because no page is ever written.

    WHAT THIS CELL DOES AND DOES NOT PROVE -- corrected 2026-09-12, while
    accepting the fill. `disk_kib()` is NOT a discriminator on the dev host's
    filesystem: ZFS allocates on transaction-group commit rather than on
    msync, and stores an all-zero record as a hole. Measured, three rows, one
    4 GiB sparse file each: nothing written -> 512 B; 512 MiB of zeros -> 512
    B; 512 MiB of RANDOM bytes -> 512 B after msync and only 439,174,656 B
    after a system `sync` and twelve seconds. So `disk_kib <= 64` is true of
    this build, and would be equally true of one that had materialised every
    constant as zeros, and of one that had just written real weights.

    It is kept because it is cheap and it is one more sign, and because on a
    filesystem that accounts blocks eagerly it does discriminate. The GUARD is
    `test_the_full_48_layer_stack_emits_at_real_geometry`'s peak RSS
    (CF-RESIDENT) -- which is exactly the finding that cell exists for, one
    level further down than the reviewer found it.
    """
    _, report, arena = built
    declared = report["arena_declared_bytes"]
    disk_kib = arena.disk_kib()
    print(f"\n[contract-residency] declared {declared / 2**30:.2f} GiB of "
          f"constant storage, {disk_kib} KiB actually on disk")
    assert declared > 8 * 2**30, (
        f"only {declared / 2**30:.2f} GiB declared at {_CONTRACT_LAYERS} real "
        f"layers -- the geometry is not real")
    assert disk_kib <= 64, (
        f"{disk_kib} KiB on disk: something WROTE to the arena, so the "
        f"constants are materialising after all. NB the converse does not "
        f"follow -- see this cell's docstring; 0 KiB is not proof of an "
        f"unwritten arena on a copy-on-write filesystem.")


# ---------------------------------------------------------------------------
# THE HANDSHAKE with the C++ slot-pool arithmetic
# ---------------------------------------------------------------------------

def slot_pool_from_tiled_ir(model, num_expert, ratio_pct):
    """What `slot_pool_from_ir` would find if it matched the PATTERN arcint
    exports instead of a type name: the Constants with leading dim
    `num_expert` that feed a dequant chain, grouped per MoE layer.

    Same per-expert arithmetic as backend_ov.cpp:600-604 (product of dims[1:]
    times the CEILED element size) and the same slot ceiling as fit.h:95.
    """
    per_layer = {}
    for node in model.get_ordered_ops():
        if node.get_type_name() != "Reshape":
            continue
        fname = node.get_friendly_name()
        if not fname.endswith("/dequant_reshape"):
            continue
        layer = fname.split("/")[0]
        const = node.input_value(0).get_node()          # Multiply
        const = const.input_value(0).get_node()         # Subtract
        const = const.input_value(0).get_node()         # Convert
        const = const.input_value(0).get_node()         # Constant
        sh = list(const.get_output_shape(0))
        if not sh or sh[0] != num_expert:
            continue
        elems = 1
        for d in sh[1:]:
            elems *= d
        et = const.get_output_element_type(0)
        per_layer[layer] = per_layer.get(layer, 0) + elems * ((et.bitwidth + 7) // 8)
    if not per_layer:
        return None
    first = per_layer[sorted(per_layer)[0]]
    total = sum(ss._expert_slot_bytes(num_expert, ratio_pct, b, 1)
                for b in per_layer.values())
    return {"total_bytes": total, "per_expert_bytes": first,
            "slots": ss._expert_slot_bytes(num_expert, ratio_pct, 1, 1),
            "moe_layers": len(per_layer)}


def test_the_cpp_type_name_matcher_finds_nothing_and_the_line_is_named(built):
    """THE HANDSHAKE FAILURE, named exactly.

    `slot_pool_from_ir`'s gate is backend_ov.cpp:585

        if (tname.find("moe") == std::string::npos) continue;

    and no arcint-exported IR carries a moe-typed op -- not this one, and not
    any of the 52 IRs in the dev host's model store, including the 35B-A3B MoE
    language model that `moe_block_tiled` was extracted from. The fusion that
    creates such a node is a GPU-plugin COMPILE-time pass; this function runs
    on `read_model`.

    Asserted, not lamented: the analytic route is nullopt here, so
    backend_ov.cpp:3746+ config.json fallback is what prices the host ledger.
    """
    model, _, _ = built
    cfg = pwe.real_config()
    got = ss.slot_pool_from_ir(model, cfg.num_experts, 0)
    typed = sorted({n.get_type_name() for n in model.get_ordered_ops()
                    if "moe" in n.get_type_name().lower()})
    print(f"\n[contract-otd] moe-typed ops in the serving-shape IR: {typed}")
    print(f"[contract-otd] slot_pool_from_ir(backend_ov.cpp:577) -> {got}")
    print(f"[contract-otd] dev-host model store, 2026-09-12: "
          f"{FLEET_IRS_WITH_MOE_TYPED_OP} of {FLEET_IRS_ALL} IRs carry one "
          f"({FLEET_IRS_SIZE_FILTERED} of them over 100k)")
    assert typed == [], (
        "an op type now contains 'moe'; slot_pool_from_ir may match -- "
        "re-derive this cell instead of editing it")
    assert got is None, (
        "slot_pool_from_ir matched. That is GOOD NEWS and this cell must be "
        "rewritten to assert the figures rather than the gap.")
    assert FLEET_IRS_WITH_MOE_TYPED_OP == 0


def test_the_pattern_matcher_prices_the_expert_pool_and_lands_on_the_cpp_constant(built):
    """The arithmetic the contract WOULD produce, cross-checked against a
    number the C++ already holds.

    `src/exec/flash_next_offload.h:45`:
        constexpr uint64_t kFlashNextSliceBytes = 2'457'600;
        // one expert-layer int4 slice (gate/up/down)

    gate+up+down at real geometry = 2*(640*2560) + 2560*640 = 4,915,200 int4
    values = 2,457,600 bytes. The IR walk cannot reproduce that figure, and the
    reason is structural rather than a bug in either side:
    backend_ov.cpp:604 uses `element_type().size()`, which CEILS a 4-bit width
    to one whole byte -- so it reads 4,915,200 B per expert, EXACTLY 2x. The
    C++ comment at :610-615 anticipates over-reservation ("this over-reserves
    rather than under-reserves, pending an on-card audit"); this cell measures
    the factor and pins it at exactly 2.
    """
    model, report, _ = built
    cfg = pwe.real_config()
    E = cfg.num_experts
    got = slot_pool_from_tiled_ir(model, E, 0)
    assert got is not None, "the pattern matcher found no MoE layer either"

    slice_bytes_cpp = 2_457_600            # flash_next_offload.h:45
    true_bytes = (2 * cfg.moe_intermediate_size * cfg.hidden_size
                  + cfg.hidden_size * cfg.moe_intermediate_size) // 2
    print(f"\n[contract-slots] moe_layers {got['moe_layers']} "
          f"slots/layer {got['slots']} (ratio 0)")
    print(f"[contract-slots] per-expert bytes: IR walk {got['per_expert_bytes']:,} "
          f"| true int4 {true_bytes:,} | flash_next_offload.h:45 "
          f"{slice_bytes_cpp:,}")
    print(f"[contract-slots] total {got['total_bytes'] / 2**30:.2f} GiB over "
          f"{got['moe_layers']} layers")

    assert got["moe_layers"] == report["n_layers"], (
        f"{got['moe_layers']} MoE layers found, {report['n_layers']} emitted")
    assert true_bytes == slice_bytes_cpp, (
        f"the geometry no longer gives the C++ constant: {true_bytes} != "
        f"{slice_bytes_cpp} (flash_next_offload.h:45)")
    assert got["per_expert_bytes"] == 2 * slice_bytes_cpp, (
        f"the u4 ceiling factor moved: {got['per_expert_bytes']} is "
        f"{got['per_expert_bytes'] / slice_bytes_cpp:.3f}x the C++ slice, "
        f"expected exactly 2 (element_type().size() ceils 4 bits to 1 byte)")
    assert got["slots"] == E, "ratio 0 must keep every expert resident"


def test_the_slot_arithmetic_transcription_matches_the_cpp_ceiling():
    """fit.h:95, `ceil(num_expert * (100 - ratio) / 100)` slots per layer.
    Checked at the boundaries a ceiling gets wrong."""
    cases = [(512, 0, 512), (512, 50, 256), (512, 100, 0),
             (512, 1, 507), (512, 99, 6), (10, 33, 7), (3, 50, 2)]
    for E, ratio, want in cases:
        got = ss._expert_slot_bytes(E, ratio, 1, 1)
        assert got == want, f"E={E} ratio={ratio}: {got} != {want}"


# ---------------------------------------------------------------------------
# LOADABLE BY ov.Core -- at a geometry whose .bin fits, and stated as such
# ---------------------------------------------------------------------------

def test_the_serving_shape_survives_save_and_read_back(tmp_path):
    """`ov.save_model` -> `ov.Core().read_model` round-trip, asserting the u4
    expert constants and the dequant chain come back intact.

    THE GEOMETRY IS REDUCED ON PURPOSE and the reason is arithmetic, not
    convenience: `ov.save_model(model, path, compress_to_fp16)` is the only
    serialisation this OpenVINO build's Python API offers, and it writes the
    whole .bin. At real geometry that is 183 GiB against 27 GiB free on the dev
    host, so the full-geometry model is validated as a live ov::Model (which is
    OpenVINO's own shape inference accepting every op as it is built) and NOT
    round-tripped. A weightless / `ov::weights_path` write -- the form
    backend_ov.cpp:552-556 says the load path can consume -- has no Python
    entry point here; that is recorded as blocked, with the reason, rather than
    claimed.

    What this cell does prove is that the SHAPE serialises and re-reads: same
    ports, same u4 element types, same rank-4 -> rank-3 dequant collapse.
    """
    cfg = pwe.real_config()
    # one MoE layer's worth of structure at a width whose .bin is a few MiB
    small = type(cfg)(
        hidden_size=256, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        hc_count=cfg.hc_count, hc_lowrank=32,
        ple_embed_dim=256, ple_conv_kernel_size=cfg.ple_conv_kernel_size,
        ngram_size=cfg.ngram_size, heads_per_ngram=cfg.heads_per_ngram,
        vocab_size=512, rms_norm_eps=cfg.rms_norm_eps,
        linear_key_head_dim=32, linear_num_key_heads=2,
        linear_value_head_dim=32, linear_num_value_heads=4,
        linear_conv_kernel_dim=cfg.linear_conv_kernel_dim,
        hidden_act="silu", layer_types=["linear_attention"],
    )
    small.ngram_total_vocab = 4096
    arena = ss.SparseArena(capacity_bytes=1 << 32)
    try:
        model, report = ss.build_serving_shape_ir(
            config=small, seq_len=4, arena=arena, n_layers=1)
        xml = tmp_path / "serving_shape.xml"
        ov.save_model(model, str(xml), compress_to_fp16=False)
        size_mib = (xml.stat().st_size
                    + xml.with_suffix(".bin").stat().st_size) / 2**20
        back = ov.Core().read_model(str(xml))
        names_before = {n for n, _, _ in report["inputs"]}
        names_after = {p.get_node().get_friendly_name() for p in back.inputs}
        u4_before = sum(1 for n in model.get_ordered_ops()
                        if n.get_type_name() == "Constant"
                        and n.get_output_element_type(0) == ov.Type.u4)
        u4_after = sum(1 for n in back.get_ordered_ops()
                       if n.get_type_name() == "Constant"
                       and n.get_output_element_type(0) == ov.Type.u4)
        print(f"\n[contract-roundtrip] reduced geometry E={small.num_experts} "
              f"H={small.hidden_size} I={small.moe_intermediate_size}, "
              f"xml+bin {size_mib:.2f} MiB")
        print(f"  ports  before {sorted(names_before)}")
        print(f"  ports  after  {sorted(names_after)}")
        print(f"  u4 constants  before {u4_before}  after {u4_after}")
        assert names_after == names_before, (names_before, names_after)
        assert u4_before > 0 and u4_after == u4_before, (u4_before, u4_after)
        assert list(back.outputs[0].get_shape()) == [1, 4, small.vocab_size]
    finally:
        arena.close()


# ---------------------------------------------------------------------------
# NOT MET AND NAMED -- the paged serving ports
# ---------------------------------------------------------------------------

_BACKEND_OV = REPO_ROOT / "src" / "exec" / "backend_ov.cpp"


def cite(anchor, path=None):
    """`<file>:<line>` for the ONE line of `path` that contains `anchor`.

    CF-COUNTS (REVIEW 2a45349 F4). The table below used to write its line
    numbers out by hand, and three of the thirteen had drifted by one line:
    `gated_delta_state_table.` was written :3196 where the prefix test is at
    :3195, and `key_cache.` / `value_cache.` were both written :3200 where it
    is at :3199 (:3200 is the `is_value` line underneath). Nothing was wrong
    with the ports; the citations rotted because a line was inserted above
    them, in a repository whose standard is grep-verified citations.

    Hand-correcting them would be right until the next edit above them. This
    resolves an ANCHOR -- a substring of the cited code, which is the thing
    actually meant -- and derives the number, so the citation cannot drift at
    all. The PROSE-CLAIM LAW's clause 2 ("counts are generated by their
    defining gate, never recited") applied to line numbers.

    Uniqueness is asserted, not hoped for: an anchor that matches two lines
    names neither, and one that matches none has been edited away.
    """
    p = path or _BACKEND_OV
    hits = [i for i, line in enumerate(p.read_text().splitlines(), 1)
            if anchor in line]
    assert len(hits) == 1, (
        f"citation anchor {anchor!r} matches {len(hits)} lines of {p.name}"
        + (f" ({hits[:6]})" if hits else " -- it has been edited away")
        + ". An anchor must identify exactly one line or it cites nothing.")
    return f"{p.name}:{hits[0]}"


# (port prefix, the CODE that classifies or feeds it). The file:line in the
# printed inventory is derived from the second column by `cite`, never typed.
_PAGED_PORT_ANCHORS = (
    # classified by name prefix at load time
    ("conv_state_table.", 'name.rfind("conv_state_table.", 0) == 0'),
    ("gated_delta_state_table.", 'name.rfind("gated_delta_state_table.", 0) == 0'),
    ("key_cache.",
     'name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0'),
    ("value_cache.",
     'name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0'),
    # fed every forward
    ("past_lens", 'set_i32("past_lens"'),
    ("subsequence_begins", 'set_i32("subsequence_begins"'),
    ("block_indices", 'set_i32("block_indices"'),
    ("block_indices_begins", 'set_i32("block_indices_begins"'),
    ("max_context_len", 'set_i32("max_context_len"'),
    ("la.block_indices", 'set_i32("la.block_indices"'),
    ("la.block_indices_begins", 'set_i32("la.block_indices_begins"'),
    ("la.past_lens", 'set_i32("la.past_lens"'),
    ("la.cache_interval", 'set_i32("la.cache_interval"'),
)

_PAGED_PORTS = [(port, cite(anchor)) for port, anchor in _PAGED_PORT_ANCHORS]


def test_the_paged_port_citations_resolve_to_the_code_they_name():
    """The anchors above must each still identify exactly one line -- `cite`
    asserts that as it builds `_PAGED_PORTS`, so this cell mostly documents the
    result and prints it. It also pins the two facts the inventory depends on:
    thirteen ports, and the two classification sites they fall into."""
    assert len(_PAGED_PORTS) == len(_PAGED_PORT_ANCHORS)
    lines = sorted(int(c.split(":")[1]) for _, c in _PAGED_PORTS)
    print("\n[contract-cite] paged-port citations, resolved from anchors:")
    for port, c in _PAGED_PORTS:
        print(f"  {port:28s} {c}")
    classify = [ln for ln in lines if ln < 5000]
    feed = [ln for ln in lines if ln >= 5000]
    print(f"[contract-cite] classified at {min(classify)}-{max(classify)}, "
          f"fed at {min(feed)}-{max(feed)}")
    assert len(classify) == 4 and len(feed) == 9, (
        f"{len(classify)} classified / {len(feed)} fed; the inventory's two "
        f"sites moved and the split above is no longer the document's")


@pytest.mark.xfail(strict=True, reason=(
    "the serving-shape IR is the STATIC full-sequence shape, not the paged "
    "one. The paged port contract (conv_state_table.N / "
    "gated_delta_state_table.N / key_cache.N / value_cache.N / la.* -- "
    "backend_ov.cpp:3191-3199 and :6141-6151) is NOT emitted. strict=True: "
    "the day it is, this cell fails and gets promoted instead of forgotten."))
def test_the_paged_port_contract_is_satisfied(built):
    model, report, _ = built
    names = {n for n, _, _ in report["inputs"]}
    missing = [(p, cite) for p, cite in _PAGED_PORTS
               if not any(n == p or n.startswith(p) for n in names)]
    assert not missing, (
        "paged ports absent from the serving-shape IR:\n"
        + "\n".join(f"  {p:28s} fed at {cite}" for p, cite in missing))


def test_the_paged_gap_is_inventoried_precisely(built):
    """The xfail above proves the gap; this cell PRINTS it, so a reader of the
    log knows exactly which thirteen ports stand between this IR and a paged
    forward, and where each one is fed."""
    _, report, _ = built
    names = {n for n, _, _ in report["inputs"]}
    missing = [(p, c) for p, c in _PAGED_PORTS if p not in names]
    print("\n[contract-paged] ports the paged forward feeds that this IR "
          "does not declare:")
    for p, c in missing:
        print(f"  {p:28s} {c}")
    assert len(missing) == len(_PAGED_PORTS), (
        "some paged ports appeared; update the xfail above rather than this "
        "inventory")


# ---------------------------------------------------------------------------
# The keystone: the whole 48-layer stack. Opt-in, because it is expensive.
# ---------------------------------------------------------------------------

# CF-RESIDENT (REVIEW 2a45349 F2). The peak-RSS ceiling for the 48-layer build,
# DERIVED from measurement on both sides rather than chosen. Dev host,
# 2026-09-12, OV 2026.4.0-22849, one variable per run (`rssprobe`, one module
# dropped from `_C_MODULES`, nothing else).
#
# The derivation used to be a table in this comment, and REVIEW 23938c1 F2 is
# what a table in a comment costs: `backbone` entered `_C_MODULES` in the same
# commit, no row was added for it, and a sentence below counted "four of the
# six" over a population that had become seven. So the table is DATA now.
# Every module in `_C_MODULES` must appear here -- the cell below fails if one
# does not -- and every count in prose about it is generated from this dict.
#
# The ceiling must sit above the authored peak and below the CHEAPEST defect.
# It is the GEOMETRIC MEAN of that pair, because that is the value with the
# same RELATIVE margin on each side, and peak RSS moves multiplicatively with
# how much of the model a defect copies, not additively.
#
# NO MARGIN IS QUOTED HERE, and that is the fix rather than an omission.
# REVIEW 23938c1 F3: this comment used to say "17.2% of headroom above the
# authored peak, 17.5% below the cheapest defect". Neither figure matched any
# consistent definition of its side, and with the rounded constant they are
# ordered the other way round -- in the one comment whose job is to stop the
# next session raising the ceiling without re-deriving it.
# `test_the_peak_rss_ceiling_is_the_geometric_mean_of_its_bracket` PRINTS both
# margins from the three constants and asserts the derivation that justifies
# them, so there is nothing here to drift.
#
# The ceiling stays a TYPED constant, re-derived by that cell rather than
# computed at import: a ceiling computed from the pair would absorb a raise
# silently, and the act this guard exists to catch is someone raising it. Typed
# + re-derived, that act is a red.
#
# RAISING THIS TO MAKE A RUN PASS RE-OPENS THE DEFECT. If the authored peak
# genuinely moves (a different OpenVINO, a different allocator), re-run both
# sides and re-derive; the two figures are what the constant means.
PEAK_RSS_AUTHORED_GIB = 4.52
PEAK_RSS_CHEAPEST_DEFECT_GIB = 6.23
PEAK_RSS_CEILING_GIB = 5.31        # == round(sqrt(4.52 * 6.23), 2), asserted

# Value = peak RSS in GiB of the 48-layer build with exactly that module
# dropped. `None` = NOT PROBED, with the reason in the row; no row borrows a
# figure it did not measure. Keys are module names as `_C_MODULES` reports
# them (`ss._C_MODULES` holds them under the aliases in brackets).
#
# The rows at the authored figure are why the keystone cell is not the whole
# closure: `test_every_module_binding_the_constant_factory_is_swapped` is.
PEAK_RSS_GIB_WHEN_DROPPED = {
    "moe":               6.23,   # [qmoe]  <- the CHEAPEST defect
    "attention":         8.98,   # [qattn] the reviewer's probe, exactly
    "hc":                9.51,   # [qhc]
    "gdn":              30.10,   # [qgdn]
    "ple":               4.52,   # [qple]  == authored: invisible to the RSS leg
    "piecewise_export":  4.52,   # [pwe]   == authored: invisible to the RSS leg
    # [qbb] NOT PROBED, and deliberately not given 4.52 by analogy.
    # `build_serving_shape_ir` imports `backbone` so `shared_constants()` can
    # swap it, and then never calls it: it reaches for `qgdn._c` directly for
    # `embed_w` and `head_w`. So dropping it cannot move peak RSS in THIS
    # build, for a structural reason rather than a measured one -- and that
    # reason is asserted, not asserted-by-comment, in
    # `test_the_rss_derivation_accounts_for_every_swapped_module`. The day
    # serving_shape reaches into `qbb`, that assertion goes red and this row
    # needs a probe.
    "backbone":          None,
}


def _rss_ceiling_margins():
    """`(above, below)`, the ceiling's two RELATIVE margins, from the constants.

    `above` = how far the ceiling sits over the authored peak; `below` = how
    far the cheapest visible defect sits over the ceiling. Ratios rather than
    differences, because that is what "the same relative margin on each side"
    means and what makes the geometric mean the right midpoint.
    """
    return (PEAK_RSS_CEILING_GIB / PEAK_RSS_AUTHORED_GIB - 1.0,
            PEAK_RSS_CHEAPEST_DEFECT_GIB / PEAK_RSS_CEILING_GIB - 1.0)


def _build_48_in_a_child():
    """Run the keystone build in a FRESH interpreter and return its report plus
    its own peak RSS.

    `ru_maxrss` is a high-water mark for the whole process, so measuring it
    inside pytest would measure whatever ran before this cell -- torch, the
    other suites, the module-scope 8-layer fixture. A child process is the only
    way the number means "this build", and it is also what makes the ceiling
    above reproducible from a bare shell.
    """
    import json
    import subprocess
    src = r"""
import json, resource, sys, time
sys.path.insert(0, %r)
from q4e import serving_shape as ss
arena = ss.SparseArena()
try:
    t0 = time.time()
    model, report = ss.build_serving_shape_ir(seq_len=64, arena=arena)
    report["build_seconds"] = time.time() - t0
    report["disk_kib"] = arena.disk_kib()
    report["peak_rss_gib"] = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 2**20
    report.pop("op_histogram", None)
    sys.stdout.write("REPORT " + json.dumps(report) + "\n")
finally:
    arena.close()
""" % str(REPO_ROOT / "tools")
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=1800)
    line = [l for l in r.stdout.splitlines() if l.startswith("REPORT ")]
    assert line, (
        "the keystone child produced no report.\n"
        f"rc={r.returncode}\nstdout tail:\n{r.stdout[-2000:]}\n"
        f"stderr tail:\n{r.stderr[-2000:]}")
    return json.loads(line[-1][len("REPORT "):])


@pytest.mark.skipif(not os.environ.get("Q4E_SERVING_FULL"),
                    reason="Q4E_SERVING_FULL unset: the full 48-layer "
                           "real-geometry build is the keystone cell and is "
                           "run deliberately, not on every suite pass")
def test_the_full_48_layer_stack_emits_at_real_geometry():
    """FULL GEOMETRY STRUCTURE EMISSION -- the thing the refusal said was
    blocked. 48 layers, 36 GDN + 12 dense-causal, real widths, real vocabulary,
    experts slot-referenced, PLE table declared and never materialised.

    AND ITS RESIDENCY, which until CF-RESIDENT nothing measured. The headline
    is "183 GiB declared, built on a 48 GiB host"; the two assertions that
    carried it (`declared > 8 GiB`, `disk_kib <= 64`) are both INVARIANT to a
    module dropping out of `_C_MODULES`, because a copied constant lives in
    anonymous memory and never touches the arena file. The reviewer dropped
    `qattn` and every quantity this cell observed stayed bit-identical while
    peak RSS doubled. `peak_rss_gib` is the quantity that moves.
    """
    report = _build_48_in_a_child()
    cfg = pwe.real_config()
    rss = report["peak_rss_gib"]
    print(f"\n[serving-shape FULL] {report['n_layers']} layers "
          f"({report['gdn_layers']} GDN + {report['attn_layers']} attn), "
          f"T={report['seq_len']}")
    print(f"  nodes                 {report['nodes']:,}")
    print(f"  declared constants    "
          f"{report['arena_declared_bytes'] / 2**30:.2f} GiB")
    print(f"  graph const bytes     "
          f"{report['graph_const_bytes'] / 2**30:.2f} GiB")
    print(f"  arena blocks on disk  {report['disk_kib']} KiB")
    print(f"  build                 {report['build_seconds']:.1f} s")
    above, below = _rss_ceiling_margins()
    print(f"  PEAK RSS              {rss:.2f} GiB   "
          f"(authored {PEAK_RSS_AUTHORED_GIB}, ceiling "
          f"{PEAK_RSS_CEILING_GIB}, cheapest defect "
          f"{PEAK_RSS_CHEAPEST_DEFECT_GIB})")
    print(f"  ceiling margins       +{above * 100:.1f}% over the authored peak, "
          f"+{below * 100:.1f}% under the cheapest defect "
          f"(printed, never written down -- REVIEW 23938c1 F3)")
    assert report["n_layers"] == cfg.num_hidden_layers == 48
    assert report["gdn_layers"] == 36 and report["attn_layers"] == 12
    assert report["nodes"] > 50_000, report["nodes"]
    # one more sign, not the guard -- see
    # test_no_expert_constant_is_materialised for why `disk_kib` cannot
    # distinguish an unwritten arena from a written one on this filesystem
    assert report["disk_kib"] <= 64
    assert report["outputs"][0][1] == [1, 64, cfg.vocab_size]
    assert rss <= PEAK_RSS_CEILING_GIB, (
        f"peak RSS {rss:.2f} GiB exceeds the derived ceiling "
        f"{PEAK_RSS_CEILING_GIB} GiB. The build declared the same "
        f"{report['arena_declared_bytes'] / 2**30:.2f} GiB over the same "
        f"{report['nodes']:,} nodes and still wrote {report['disk_kib']} KiB "
        f"to disk, so every other assertion in this cell is satisfied -- that "
        f"is the CF-RESIDENT signature. Check `_C_MODULES` in "
        f"tools/q4e/serving_shape.py against "
        f"test_every_module_binding_the_constant_factory_is_swapped before "
        f"touching this number; the cheapest module to drop costs "
        f"{PEAK_RSS_CHEAPEST_DEFECT_GIB} GiB.")


# ---------------------------------------------------------------------------
# CF-RESIDENT leg 2 -- the structural one, which runs on EVERY leg
# ---------------------------------------------------------------------------

def _modules_binding_the_constant_factory():
    """Every module under tools/q4e that binds the name `_c` at module level,
    by ast rather than by import: `from .gdn import _c`, `import _c as ...`,
    or its own `def _c` / `_c = ...`.

    Read from source on purpose. Importing to ask would run each module, and a
    module that failed to import would silently drop out of the answer -- the
    same shape of invisibility this whole finding is about.
    """
    import ast
    src_dir = REPO_ROOT / "tools" / "q4e"
    found = {}
    for p in sorted(src_dir.glob("*.py")):
        if p.name == "__init__.py":
            continue
        tree = ast.parse(p.read_text(), filename=str(p))
        where = None
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    if (a.asname or a.name.split(".")[0]) == "_c":
                        where = node.lineno
            elif isinstance(node, ast.FunctionDef) and node.name == "_c":
                where = node.lineno
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "_c":
                        where = node.lineno
        if where is not None:
            found[p.stem] = where
    return found


def test_every_module_binding_the_constant_factory_is_swapped():
    """CF-RESIDENT, the leg the peak-RSS assertion cannot cover.

    `shared_constants()` swaps `_c` per module, and each importer holds its OWN
    binding, so a module missing from `_C_MODULES` keeps the copying factory
    and materialises its family's weights. The 48-layer cell catches that for
    SOME of the swapped modules and not others: dropping `qple` or `pwe` leaves
    peak RSS at the authored figure EXACTLY, because this particular build
    never reaches their `_c` with a large arena array. Measured, not inferred
    -- which is why a behavioural leg alone would be a guard with holes in it.

    HOW MANY of each is not written here (REVIEW 23938c1 F2: the count that
    stood in this docstring said "four of the six" on the day `_C_MODULES`
    became seven). The split is generated from `PEAK_RSS_GIB_WHEN_DROPPED` and
    printed by `test_the_rss_derivation_accounts_for_every_swapped_module`,
    which also fails if a module joins the swap list without a row.

    So the invariant is asserted structurally instead: a module that BINDS `_c`
    must be in `_C_MODULES`, whether or not today's build happens to call it.
    On its first run this found `q4e.backbone`, which binds `_c`
    (backbone.py:70) and spends it on `embed_w` and `head_w` -- 2.37 GiB each
    at real geometry, the largest pair in the model.

    The technique is `test_suite_guards.py`'s: read the source with `ast`,
    device-free, no shards, runs on every leg.
    """
    binders = _modules_binding_the_constant_factory()
    listed = {m.__name__.rsplit(".", 1)[-1] for m in ss._C_MODULES}
    print(f"\n[contract-cmodules] modules binding `_c`: "
          f"{', '.join(f'{k} (:{v})' for k, v in sorted(binders.items()))}")
    print(f"[contract-cmodules] _C_MODULES: {sorted(listed)}")

    assert binders, (
        "no module under tools/q4e binds `_c` -- the scanner is broken, not "
        "the tree (q4e.gdn defines it and at least five modules import it)")
    missing = sorted(set(binders) - listed)
    assert not missing, (
        "module(s) bind the `_c` constant factory but are absent from "
        "`_C_MODULES` in tools/q4e/serving_shape.py: "
        + ", ".join(f"{m} (q4e/{m}.py:{binders[m]})" for m in missing)
        + ". `shared_constants()` will not swap them, so every constant they "
          "build during a serving-shape build is COPIED into anonymous memory "
          "-- invisible to the node count, to the declared bytes and to "
          "`disk_kib`, and visible only as peak RSS or as an OOM.")
    # and the reverse: a name in _C_MODULES that no longer binds `_c` is a
    # stale entry whose swap is a no-op, which would make this gate weaker
    # than it reads.
    stale = sorted(listed - set(binders))
    assert not stale, (
        f"`_C_MODULES` lists {stale}, which bind no `_c`; the swap is a no-op "
        f"for them. Remove them, or this gate over-reports its own coverage.")


_SERVING_SHAPE_PY = REPO_ROOT / "tools" / "q4e" / "serving_shape.py"


def _serving_shape_attribute_bases():
    """The module aliases `serving_shape.py` actually REACHES INTO (`alias.x`),
    by ast. An alias that is imported and never dereferenced is in
    `_C_MODULES` for the swap alone -- which is the whole of `backbone`'s
    position in the RSS derivation.
    """
    import ast
    tree = ast.parse(_SERVING_SHAPE_PY.read_text())
    return {n.value.id for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}


def _serving_shape_import_alias(module):
    """The local name `serving_shape.py` binds for `q4e.<module>`, by ast.

    CF-GUARDALIAS (REVIEW 7c26cce F8). The deref guard below used to spell its
    subject as the literal string `"qbb"`. That string is not the module; it is
    one spelling of the alias the module happened to be imported under, and an
    assertion keyed on a spelling passes VACUOUSLY the moment the spelling
    changes. Measured on a mutated tree before this function existed: rename
    every `qbb` to `qback` and add one real `qback._c` dereference, and the
    cell reported `1 passed` while `backbone` WAS being reached into -- the
    `None` row in `PEAK_RSS_GIB_WHEN_DROPPED` kept a licence it had just lost.

    So the alias is DERIVED from the `ImportFrom` node instead of typed -- the
    same move `cite()` makes for line numbers one level down, and for the same
    reason (clause 2: the defining source generates it, nobody recites it).
    Uniqueness and existence are both asserted: a `backbone` import that has
    been removed, or bound twice, names nothing, and that is a red rather than
    a silent pass.
    """
    import ast
    tree = ast.parse(_SERVING_SHAPE_PY.read_text())
    bound = [a.asname or a.name.rsplit(".", 1)[-1]
             for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
             for a in n.names if a.name.rsplit(".", 1)[-1] == module]
    assert len(bound) == 1, (
        f"`serving_shape.py` binds {module!r} {len(bound)} times ({bound}); "
        f"this guard needs exactly one alias to track. If the import was "
        f"removed, {module!r} no longer belongs in `_C_MODULES` either; if it "
        f"is bound twice, the deref guard below would only watch one of them.")
    return bound[0]


def test_the_rss_derivation_accounts_for_every_swapped_module():
    """REVIEW 23938c1 F2. The ceiling is derived over a POPULATION, and this
    asserts the population is the one `shared_constants()` actually swaps.

    The defect being closed is not arithmetic: the derivation table lived in a
    comment, `backbone` joined `_C_MODULES` in the commit that wrote the table,
    no row was added, and a docstring counted "four of the six" over seven.
    Nothing was mismeasured and nothing went red. That is the shape.

    So: every swapped module has a row here, every row is a swapped module, the
    cheapest visible defect is MINIMISED OUT OF THE ROWS rather than copied
    into a constant, and the one unprobed row states its structural reason --
    `serving_shape` imports `backbone` for the swap and never dereferences it.
    That last one is asserted from source, so the day the build reaches into
    `backbone` this cell goes red and asks for a probe instead of letting the
    ceiling be derived over a population with a hole in it.

    CF-GUARDALIAS (REVIEW 7c26cce F8): that deref assertion names the module
    and resolves its alias through `_serving_shape_import_alias`, because the
    hardcoded spelling it used to carry passed vacuously under a rename -- see
    that function's docstring for the measured red.
    """
    listed = {m.__name__.rsplit(".", 1)[-1] for m in ss._C_MODULES}
    rows = set(PEAK_RSS_GIB_WHEN_DROPPED)
    probed = {m: v for m, v in PEAK_RSS_GIB_WHEN_DROPPED.items() if v is not None}
    visible = sorted(m for m, v in probed.items() if v > PEAK_RSS_AUTHORED_GIB)
    invisible = sorted(listed - set(visible))

    print(f"\n[rss-derivation] swapped modules: {len(listed)}  "
          f"probed: {len(probed)}  unprobed: {len(rows) - len(probed)}")
    print(f"[rss-derivation] the 48-layer cell SEES {len(visible)} of "
          f"{len(listed)}: {', '.join(visible)}")
    print(f"[rss-derivation] INVISIBLE to it ({len(invisible)}): "
          f"{', '.join(invisible)} -- covered by the structural leg only")
    for m in sorted(PEAK_RSS_GIB_WHEN_DROPPED):
        v = PEAK_RSS_GIB_WHEN_DROPPED[m]
        print(f"    {m:20s} {'not probed' if v is None else f'{v:6.2f} GiB'}")

    assert rows == listed, (
        f"the peak-RSS derivation and `_C_MODULES` describe different "
        f"populations. Rows with no swapped module: "
        f"{sorted(rows - listed) or 'none'}; swapped modules with no row: "
        f"{sorted(listed - rows) or 'none'}. A module that joins the swap list "
        f"without a row makes every count about the derivation wrong and the "
        f"ceiling's bracket unattributed -- add the row (probe it, or state "
        f"why it cannot move peak RSS) rather than this assertion.")
    assert visible, "no row exceeds the authored peak: the bracket has no upper end"
    cheapest = min(probed[m] for m in visible)
    assert PEAK_RSS_CHEAPEST_DEFECT_GIB == cheapest, (
        f"PEAK_RSS_CHEAPEST_DEFECT_GIB is {PEAK_RSS_CHEAPEST_DEFECT_GIB}, but "
        f"the cheapest visible defect in the rows is {cheapest} "
        f"({min(visible, key=lambda m: probed[m])}). The constant is the "
        f"ceiling's upper bracket; it must be the minimum of the measured "
        f"defects, not a figure that was true when it was typed.")
    alias = _serving_shape_import_alias("backbone")
    print(f"[rss-derivation] `backbone` is imported as `{alias}` (derived from "
          f"the ImportFrom node, not typed) and is dereferenced: "
          f"{alias in _serving_shape_attribute_bases()}")
    assert alias not in _serving_shape_attribute_bases(), (
        f"`serving_shape.py` now dereferences `{alias}` (its alias for "
        f"`backbone`), so dropping `backbone` from `_C_MODULES` may move peak "
        f"RSS. Its row in PEAK_RSS_GIB_WHEN_DROPPED is `None` on the grounds "
        f"that it cannot. Probe it (rssprobe, one module dropped) and record "
        f"the figure.")


def test_the_peak_rss_ceiling_is_the_geometric_mean_of_its_bracket():
    """REVIEW 23938c1 F3. The ceiling's justification, checked against the
    ceiling -- and the two margins PRINTED instead of quoted.

    What was wrong: the comment quoted "17.2% above / 17.5% below". The exact
    geometric mean gives the same figure on both sides by construction, and
    rounding it to 0.01 GiB splits them slightly the OTHER way (the margin
    above the authored peak becomes the larger one). So the sentence disagreed
    with its own argument, with the constant, and with itself, and nothing in
    the suite could notice.

    What is asserted now is the argument itself: the ceiling brackets its pair,
    and it IS the geometric mean rounded to 0.01 GiB. The constant stays typed
    on purpose -- a ceiling computed at import would quietly absorb a raise,
    and a raise is the act this whole guard exists to catch. Typed and
    re-derived here, raising `PEAK_RSS_CEILING_GIB` without re-recording a
    measured side is a red.

    CF-TOLGEN (REVIEW 7c26cce F9). The margin tolerance was the hand-typed
    `0.005`, and it was unfailable: with `c == round(gm, 2)` (the round-trip
    assertion below, which is the real gate) the error `|c - gm| <= 0.005`
    propagates to the gap with derivative `1/a + d/c**2`, so at 4.52 / 6.23 the
    widest gap that arithmetic ALLOWS is 0.44219 * 0.005 = 0.002211 -- inside
    0.005 by 2.3x, always, while its sibling passes. `0.005` also tracked
    nothing: the bound scales with the bracket, and at a = 1, d = 100 the same
    constant is a SPURIOUS red at 0.01.

    The tolerance is now GENERATED from the bracket it is a bound for, which
    fixes both halves. What that leaves is a check that is not dead weight, and
    the distinction is worth stating because F9 read it as dead: the round-trip
    assertion constrains the CEILING against the two measured sides, and
    nothing in it constrains `_rss_ceiling_margins()`, which computes the two
    margins by its own two formulas. Break either formula and the gap leaves
    the bound while the ceiling still round-trips. Measured, not argued: with
    `below` taken against the authored peak instead of the ceiling, this cell
    goes red at 30.8 pp against a 0.2211 pp bound while every other assertion
    here passes. So it can fail, for a defect its sibling cannot see.

    The bound is the exact one, not the derivative at `c`: `gap(gm) == 0` and
    `gap(c) = |integral from gm to c of (1/a + d/t**2) dt|`, so with
    `|c - gm| <= 0.005` and the integrand falling in `t`, the supremum is
    `0.005 * (1/a + d/(c - 0.005)**2)` -- 0.2213 pp here against the observed
    0.1521 pp. Taking the derivative at `c` instead gives 0.2211 pp, which is
    below the true supremum and would be a tolerance that can be exceeded
    without a defect; the 1.01 slack factor REVIEW 7c26cce F9 suggested is
    that same gap covered by a fudge instead of by the integral.
    """
    above, below = _rss_ceiling_margins()
    exact = math.sqrt(
        PEAK_RSS_CHEAPEST_DEFECT_GIB / PEAK_RSS_AUTHORED_GIB) - 1.0
    unrounded = math.sqrt(PEAK_RSS_AUTHORED_GIB * PEAK_RSS_CHEAPEST_DEFECT_GIB)

    print(f"\n[rss-ceiling] bracket {PEAK_RSS_AUTHORED_GIB} < "
          f"{PEAK_RSS_CEILING_GIB} < {PEAK_RSS_CHEAPEST_DEFECT_GIB} GiB "
          f"(authored < ceiling < cheapest defect)")
    print(f"[rss-ceiling] sqrt({PEAK_RSS_AUTHORED_GIB} * "
          f"{PEAK_RSS_CHEAPEST_DEFECT_GIB}) = {unrounded:.6f} -> "
          f"{PEAK_RSS_CEILING_GIB} after rounding to 0.01 GiB")
    print(f"[rss-ceiling] margins: +{above * 100:.3f}% above the authored "
          f"peak, +{below * 100:.3f}% below the cheapest defect "
          f"(exact mean: {exact * 100:.3f}% each side)")
    # CF-TOLGEN: the tolerance is GENERATED from the bracket, never typed. The
    # supremum of the gap over `|c - gm| <= 0.005`, integrand taken at the
    # worst end of the interval so this is an upper bound and not an estimate.
    _HALF_GRID = 0.005                     # half of the 0.01 GiB rounding grid
    gap_bound = _HALF_GRID * (1.0 / PEAK_RSS_AUTHORED_GIB
                              + PEAK_RSS_CHEAPEST_DEFECT_GIB
                              / (PEAK_RSS_CEILING_GIB - _HALF_GRID) ** 2)
    print(f"[rss-ceiling] margin gap {abs(above - below) * 100:.4f} pp <= "
          f"{gap_bound * 100:.4f} pp, the most the 0.01 GiB rounding can "
          f"explain at this bracket (generated: 0.005 * (1/{PEAK_RSS_AUTHORED_GIB} "
          f"+ {PEAK_RSS_CHEAPEST_DEFECT_GIB}/"
          f"{PEAK_RSS_CEILING_GIB - _HALF_GRID:.3f}^2))")

    assert (PEAK_RSS_AUTHORED_GIB < PEAK_RSS_CEILING_GIB
            < PEAK_RSS_CHEAPEST_DEFECT_GIB), (
        f"the ceiling {PEAK_RSS_CEILING_GIB} does not bracket: it must sit "
        f"above the authored peak {PEAK_RSS_AUTHORED_GIB} and below the "
        f"cheapest defect {PEAK_RSS_CHEAPEST_DEFECT_GIB}. One side of the "
        f"derivation was re-measured and the other was not.")
    assert PEAK_RSS_CEILING_GIB == round(unrounded, 2), (
        f"the ceiling {PEAK_RSS_CEILING_GIB} is not sqrt("
        f"{PEAK_RSS_AUTHORED_GIB} * {PEAK_RSS_CHEAPEST_DEFECT_GIB}) = "
        f"{unrounded:.6f} rounded to 0.01 GiB ({round(unrounded, 2)}). Either "
        f"the ceiling was raised without re-deriving it -- which re-opens the "
        f"defect it guards -- or a measured side moved and the ceiling was "
        f"not recomputed.")
    assert abs(above - below) <= gap_bound, (
        f"the ceiling's two relative margins differ by "
        f"{abs(above - below) * 100:.4f} percentage points "
        f"(+{above * 100:.3f}% / +{below * 100:.3f}%), which EXCEEDS the "
        f"{gap_bound * 100:.4f} pp the 0.01 GiB rounding can explain at this "
        f"bracket. The assertion above already forces the ceiling to be the "
        f"rounded geometric mean, so the gap cannot get here through the "
        f"ceiling: what is wrong is how a margin is COMPUTED "
        f"(`_rss_ceiling_margins`), not how wide this tolerance is. Do not "
        f"widen it -- it is derived from the bracket, and widening it would "
        f"only hide the formula that changed.")


def test_the_constant_factory_scanner_detects_a_missing_module():
    """The scanner's own red, in tree and permanent.

    A gate that can only pass is not a gate. `_C_MODULES` minus one entry must
    be rejected by the same comparison the cell above runs.
    """
    binders = _modules_binding_the_constant_factory()
    listed = {m.__name__.rsplit(".", 1)[-1] for m in ss._C_MODULES}
    assert not (set(binders) - listed), "precondition: the tree is clean"
    for drop in sorted(listed):
        mutated = listed - {drop}
        missing = set(binders) - mutated
        assert missing == {drop}, (
            f"dropping {drop!r} from _C_MODULES was not detected as missing: "
            f"{missing}")
    print(f"[contract-cmodules] scanner rejects each of {len(listed)} "
          f"single-module deletions")
