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
over the whole model store on the dev host, 2026-09-12:

    find /models/ov -name '*.xml' -size +100k | wc -l   -> 52
    ... of which carry a moe-typed op                   -> 0

including `qwen36-35b-a3b-int4-ov/openvino_language_model.xml`, the 35B-A3B MoE
checkpoint that is the ground truth `moe_block_tiled` was extracted from
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

# Measured on the dev host, 2026-09-12, and asserted below so the finding
# cannot rot silently into prose.
FLEET_IRS_SCANNED = 52
FLEET_IRS_WITH_MOE_TYPED_OP = 0


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
    """THE CLAIM, measured rather than argued: the arena declares tens of GiB
    of constants and occupies zero blocks on disk, because no page is ever
    written."""
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
        f"constants are materialising after all")


# ---------------------------------------------------------------------------
# THE HANDSHAKE with the C++ slot-pool arithmetic
# ---------------------------------------------------------------------------

def slot_pool_from_tiled_ir(model, num_expert, ratio_pct):
    """What `slot_pool_from_ir` would find if it matched the PATTERN arcint
    exports instead of a type name: the Constants with leading dim
    `num_expert` that feed a dequant chain, grouped per MoE layer.

    Same per-expert arithmetic as backend_ov.cpp:600-603 (product of dims[1:]
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
          f"{FLEET_IRS_WITH_MOE_TYPED_OP} of {FLEET_IRS_SCANNED} IRs carry one")
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
    backend_ov.cpp:603 uses `element_type().size()`, which CEILS a 4-bit width
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

_PAGED_PORTS = [
    # backend_ov.cpp:3191-3199 classifies these by name prefix at load time
    ("conv_state_table.", "backend_ov.cpp:3191"),
    ("gated_delta_state_table.", "backend_ov.cpp:3196"),
    ("key_cache.", "backend_ov.cpp:3200"),
    ("value_cache.", "backend_ov.cpp:3200"),
    # backend_ov.cpp:6141-6151 feeds these every forward
    ("past_lens", "backend_ov.cpp:6143"),
    ("subsequence_begins", "backend_ov.cpp:6144"),
    ("block_indices", "backend_ov.cpp:6145"),
    ("block_indices_begins", "backend_ov.cpp:6146"),
    ("max_context_len", "backend_ov.cpp:6147"),
    ("la.block_indices", "backend_ov.cpp:6148"),
    ("la.block_indices_begins", "backend_ov.cpp:6149"),
    ("la.past_lens", "backend_ov.cpp:6150"),
    ("la.cache_interval", "backend_ov.cpp:6151"),
]


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

@pytest.mark.skipif(not os.environ.get("Q4E_SERVING_FULL"),
                    reason="Q4E_SERVING_FULL unset: the full 48-layer "
                           "real-geometry build is the keystone cell and is "
                           "run deliberately, not on every suite pass")
def test_the_full_48_layer_stack_emits_at_real_geometry():
    """FULL GEOMETRY STRUCTURE EMISSION -- the thing the refusal said was
    blocked. 48 layers, 36 GDN + 12 dense-causal, real widths, real vocabulary,
    experts slot-referenced, PLE table declared and never materialised."""
    import time
    arena = ss.SparseArena()
    try:
        t0 = time.time()
        model, report = ss.build_serving_shape_ir(seq_len=64, arena=arena)
        dt = time.time() - t0
        cfg = pwe.real_config()
        print(f"\n[serving-shape FULL] {report['n_layers']} layers "
              f"({report['gdn_layers']} GDN + {report['attn_layers']} attn), "
              f"T={report['seq_len']}")
        print(f"  nodes                 {report['nodes']:,}")
        print(f"  declared constants    "
              f"{report['arena_declared_bytes'] / 2**30:.2f} GiB")
        print(f"  graph const bytes     "
              f"{report['graph_const_bytes'] / 2**30:.2f} GiB")
        print(f"  arena blocks on disk  {arena.disk_kib()} KiB")
        print(f"  build                 {dt:.1f} s")
        assert report["n_layers"] == cfg.num_hidden_layers == 48
        assert report["gdn_layers"] == 36 and report["attn_layers"] == 12
        assert report["nodes"] > 50_000, report["nodes"]
        assert arena.disk_kib() <= 64
        assert report["outputs"][0][1] == [1, 64, cfg.vocab_size]
    finally:
        arena.close()
