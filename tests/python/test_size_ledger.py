"""THE SIZE LEDGER: every real-width piece measured, totalled, and confronted
with the cards and with the WP6b study.

Nothing in this file is a hand-written product of config fields. Each row is
either BUILT at the checkpoint's real geometry and measured with
`graph_measures()` (source "graph"), or -- for the two pieces that cannot be
built, the 512-expert bodies at 9.4 GiB/layer and the n-gram table at 190 GiB --
computed from the SHIPPED TENSOR LIST through an independent GGUFReader (source
"file"). The distinction is printed per row, because a figure's provenance is
part of the figure.

The previous `size_ledger` computed everything from formulas and three of them
were wrong: the GDN row multiplied the projections by the conv kernel width and
omitted attn_gate; the expert row carried a `/ chunk_frac * chunk_frac` that
cancels to nothing; and the docstring's "~464 MB f32 (0.18 GiB)" disagreed with
itself by 2.5x. So this file also keeps a gate that the old design could not
have: for every buildable piece, the constant bytes the GRAPH actually carries
must equal the sum of `nbytes` over the piece's own fed real tensors. Formula and
graph have to agree, or the row is wrong in one of two places and the cell says
which.

Shapes are read from the checkpoint, never assumed. The weights used for the
BUILDS are random of those real shapes -- a graph's node count, constant
footprint and compile cost depend on shapes, not values, and dequantising 6.7 GB
of expert bodies to learn their shape would buy nothing.

Device-free except for the CPU compiles. Needs Q4E_GGUF_SHARDS.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import openvino as ov  # noqa: E402

from q4e import attention as qattn  # noqa: E402
from q4e import gdn as qgdn  # noqa: E402
from q4e import gguf_feed  # noqa: E402
from q4e import hc as qhc  # noqa: E402
from q4e import moe as qmoe  # noqa: E402
from q4e import piecewise_export as pwe  # noqa: E402
from q4e import ple as qple  # noqa: E402

_SHARDS = os.environ.get("Q4E_GGUF_SHARDS", "").strip()
_skip = pytest.mark.skipif(
    not _SHARDS, reason="Q4E_GGUF_SHARDS unset: the ledger measures real widths")

_T = int(os.environ.get("LEDGER_T", "64"))

# Per piece: (layer that carries the family, module-relative key prefix, keys).
# blk.0 is a GDN layer, blk.3 the first QSA layer, blk.1 the PLE layer.
_FAMILIES = {
    "gdn_block": (0, "linear_attn.", [
        "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight",
        "in_proj_b.weight", "A_log", "dt_bias", "conv1d.weight",
        "norm.weight", "out_proj.weight"]),
    "attention_block": (3, "self_attn.", [
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "q_norm.weight", "k_norm.weight"]),
    "moe_router": (0, "mlp.", ["gate.weight"]),
    "moe_shexp": (0, "mlp.", [
        "shared_expert.gate_proj.weight", "shared_expert.up_proj.weight",
        "shared_expert.down_proj.weight", "shared_expert_gate.weight"]),
    "gatedresidual": (0, "attn_hyper_connection.", [
        "hc_norm.weight", "input_mix_weight_down.weight",
        "input_mix_weight_up.weight", "block_inject_weight.weight"]),
    "ple_block": (1, "ple.", [
        "key_proj.weight", "value_proj.weight", "norm_key.weight",
        "norm_query.weight", "norm_conv.weight", "conv1d.weight"]),
}
# Globals carry no layer index. The FINAL mixer has three tensors, not four:
# use_combine=False means no block_inject, and the GGUF agrees -- there is no
# `output_hc_inject` tensor. An earlier orphan helper asked for the fourth and
# got `KeyError: no GGUF map entry for global key
# 'hyper_connection_mixer.block_inject_weight.weight'`.
_GLOBALS = {
    "hc_combine": ("hyper_connection_mixer.", [
        "hc_norm.weight", "input_mix_weight_down.weight",
        "input_mix_weight_up.weight"]),
    "embed": ("", ["embed_tokens.weight"]),
    "lm_head": ("", ["lm_head.weight"]),
}


@pytest.fixture(scope="module")
def cfg():
    return pwe.real_config()


@pytest.fixture(scope="module")
def real_shapes():
    """Every piece's real tensor shapes and f32 byte count, from the checkpoint
    through the name map under test."""
    feed = gguf_feed.GgufFeed(_SHARDS)
    out = {}
    for piece, (layer, prefix, keys) in _FAMILIES.items():
        out[piece] = {}
        for k in keys:
            a = feed.pin_tensor(f"layers.{layer}.{prefix}{k}")
            out[piece][k] = tuple(int(x) for x in a.shape)
    for piece, (prefix, keys) in _GLOBALS.items():
        out[piece] = {}
        for k in keys:
            a = feed.pin_tensor(f"{prefix}{k}")
            out[piece][k] = tuple(int(x) for x in a.shape)
    return out


@pytest.fixture(scope="module")
def file_bytes():
    """The two pieces that are never built, sized from the shipped tensor list
    through an INDEPENDENT GGUFReader -- no q4e code in this path."""
    from gguf import GGUFReader
    import glob
    idx = {}
    for p in sorted(glob.glob(os.path.join(_SHARDS, "*.gguf"))):
        for t in GGUFReader(p).tensors:
            idx[t.name] = t

    def f32_bytes(name):
        t = idx[name]
        return int(np.prod([int(x) for x in t.shape])) * 4

    experts = (f32_bytes("blk.0.ffn_gate_exps.weight")
               + f32_bytes("blk.0.ffn_up_exps.weight")
               + f32_bytes("blk.0.ffn_down_exps.weight"))
    return {
        "moe_experts": experts,
        "ple_ngram_table": f32_bytes("per_layer_token_embd.weight"),
        "_quant_total": sum(int(t.n_bytes) for t in idx.values()),
    }


def _rand(shapes, seed):
    rng = np.random.default_rng(seed)
    return {k: rng.standard_normal(v).astype(np.float32)
            for k, v in shapes.items()}


def _measure(build):
    t0 = time.time()
    model = build()
    nodes, const_bytes, counts = pwe.graph_measures(model)
    t0 = time.time()
    ov.Core().compile_model(model, "CPU")
    compile_s = time.time() - t0
    return {"nodes": nodes, "const_bytes": const_bytes,
            "compile_s": round(compile_s, 3), "source": "graph",
            "counts": counts}


@pytest.fixture(scope="module")
def measured(cfg, real_shapes, file_bytes):
    m = {}
    m["gdn_block"] = _measure(
        lambda: qgdn.build_gdn_model(cfg, _rand(real_shapes["gdn_block"], 1), _T))
    m["attention_block"] = _measure(
        lambda: qattn.build_dense_attention_model(
            cfg, _rand(real_shapes["attention_block"], 2), _T))
    m["moe_router"] = _measure(
        lambda: qmoe.build_router_model(cfg, _rand(real_shapes["moe_router"], 3), _T))
    m["moe_shexp"] = _measure(
        lambda: qmoe.build_shared_expert_model(
            cfg, _rand(real_shapes["moe_shexp"], 4), _T))
    # build_hc_model consumes hc_norm/down/up only; build_combine_model adds
    # block_inject. The PER-LAYER mixer is the one with block_inject.
    m["gatedresidual"] = _measure(
        lambda: qhc.build_combine_model(
            cfg, _rand(real_shapes["gatedresidual"], 6), _T))
    m["hc_combine"] = _measure(
        lambda: qhc.build_hc_model(cfg, _rand(real_shapes["hc_combine"], 7), _T))
    Hn = (cfg.ngram_size - 1) * cfg.heads_per_ngram
    hd = cfg.ple_embed_dim // Hn
    win = [np.random.default_rng(80 + i).standard_normal((64, hd)).astype(np.float32)
           for i in range(Hn)]
    m["ple_block"] = _measure(
        lambda: qple.build_ple_model(
            cfg, _rand(real_shapes["ple_block"], 8), _T, windows=win))
    m["embed"] = _measure(
        lambda: pwe.build_embed_piece(cfg, _rand(real_shapes["embed"], 9), _T))
    m["lm_head"] = _measure(
        lambda: pwe.build_lm_head_piece(cfg, _rand(real_shapes["lm_head"], 10), _T))
    # Never built; sized from the file.
    m["moe_experts"] = {"nodes": None, "const_bytes": file_bytes["moe_experts"],
                        "compile_s": None, "source": "file"}
    m["ple_ngram_table"] = {"nodes": None,
                            "const_bytes": file_bytes["ple_ngram_table"],
                            "compile_s": None, "source": "file"}
    return m


def test_graph_constants_match_the_fed_tensor_bytes(real_shapes, measured):
    """THE AGREEMENT GATE. For every buildable piece the constant bytes the graph
    carries must equal the sum of its fed real tensors' f32 bytes, plus only the
    small index/shape constants the lowering emits. If a builder silently drops a
    weight or a shape table is wrong, the two sides part company here -- and this
    is the gate the formula-based ledger could not have, because it compared a
    formula against itself."""
    rows = []
    for piece, shapes in real_shapes.items():
        want = sum(int(np.prod(s)) * 4 for s in shapes.values())
        got = measured[piece]["const_bytes"]
        overhead = got - want
        rows.append((piece, want, got, overhead))
        assert got >= want, (
            f"{piece}: the graph carries {got} constant bytes but its fed tensors "
            f"are {want} -- a weight is missing from the emission")
        # The lowering's own constants are index vectors, axis scalars, eps and
        # the causal mask. Generous but finite: anything above this is a weight.
        assert overhead <= max(64 * 1024 * 1024, want // 8), (
            f"{piece}: {overhead} bytes of constants beyond the fed tensors -- "
            "too much to be index/axis/eps scaffolding")
    sys.stdout.write("\n[ledger-agreement] piece / fed f32 bytes / graph const "
                     "bytes / scaffolding\n")
    for piece, want, got, over in sorted(rows):
        sys.stdout.write(f"[ledger-agreement] {piece:18s} {want:>14,d} "
                         f"{got:>14,d} {over:>+12,d}\n")


def test_size_ledger_is_complete_and_reconciles(measured, file_bytes, capsys):
    """The ledger itself: every piece present, totals per tier, and the two
    confrontations -- the cards, and WP6b's figure for the same checkpoint."""
    rows = pwe.size_ledger(measured)
    totals = pwe.ledger_totals(rows)
    table = pwe.format_ledger(rows, totals)
    sys.stdout.write("\n" + table + "\n")

    assert not totals["missing"], (
        f"the ledger has holes: {totals['missing']} -- a piece with no "
        "measurement is not a ledger row, it is a gap")
    assert set(r[0] for r in rows) == set(pwe.PIECE_PLAN), "row set drifted"

    # Every piece the 48-layer model instantiates is accounted for exactly once.
    assert sum(1 for r in rows if r[1] == "CARD") == 9
    assert sum(1 for r in rows if r[1] == "OFFLOAD") == 1
    assert sum(1 for r in rows if r[1] == "HOST-MMAP") == 1

    # The grand total must land on the refusal's independently recomputed
    # residency figure (659.1 GiB over the whole mapped tensor list). This ledger
    # reaches it by a completely different route -- per-piece graphs times
    # instance counts -- so agreement to ~1% is a real cross-check and a larger
    # gap means one of the two is wrong about the architecture.
    refusal_gib = 659.1
    rel = abs(totals["total_gib"] - refusal_gib) / refusal_gib
    sys.stdout.write(
        f"\n[ledger-reconcile] ledger grand total {totals['total_gib']:.1f} GiB "
        f"vs the refusal's recomputed residency {refusal_gib} GiB "
        f"-> {rel * 100:.2f}% apart\n"
        f"[ledger-reconcile] checkpoint as shipped (quantized, summed over the "
        f"tensor list) {file_bytes['_quant_total'] / 1024**3:.2f} GiB; WP6b "
        f"reports {pwe.WP6B_TOTAL_GIB} GiB\n")
    assert rel < 0.02, (
        f"the ledger and the refusal disagree by {rel*100:.1f}% about the same "
        "checkpoint; one of them has the architecture wrong")


def test_card_tier_does_not_fit_the_a770_reserve_at_f32(measured):
    """The verdict the window needs, stated with its residency assumption named.

    Every figure here assumes each weight becomes an f32 ov Constant. Under that
    assumption the card-resident-first set is what it is, and whether it fits is
    not an opinion. This cell asserts the DIRECTION of the answer so that a
    change in the architecture or in the plan has to come past it."""
    rows = pwe.size_ledger(measured)
    t = pwe.ledger_totals(rows)
    sys.stdout.write(
        f"\n[ledger-verdict] CARD tier {t['card_gib']:.3f} GiB at f32 "
        f"(assumption: every weight an f32 ov Constant)\n"
        f"[ledger-verdict]   B60 {pwe.CARD_B60_GIB} GiB -> "
        f"{'fits' if t['fits_b60'] else 'DOES NOT FIT'} "
        f"({t['headroom_b60_gib']:+.3f} GiB)\n"
        f"[ledger-verdict]   A770 reserve {pwe.CARD_A770_GIB} GiB -> "
        f"{'fits' if t['fits_a770_reserve'] else 'DOES NOT FIT'} "
        f"({t['headroom_a770_gib']:+.3f} GiB)\n"
        f"[ledger-verdict] f32 constants cost "
        f"{t['f32_over_quantized']:.2f}x the shipped checkpoint "
        f"({t['total_gib']:.1f} vs {pwe.WP6B_TOTAL_GIB} GiB)\n")
    assert t["card_gib"] > 0
    # Not a tuned bound: a CARD tier that suddenly fits the 15 GiB reserve means
    # either a weight strategy landed (good -- update this cell with the
    # measurement) or a piece fell out of the plan (bad). Either way, look.
    assert not t["fits_a770_reserve"], (
        f"the CARD tier now claims to fit the A770 reserve at "
        f"{t['card_gib']:.3f} GiB. At f32 it did not (that is why a weight "
        "strategy is blocker 3). Verify a piece did not go missing before "
        "believing this.")


def test_ledger_json_round_trip(measured):
    """The window manifest pastes this table; it has to be machine-readable too,
    so a later session can diff two nights instead of re-reading prose."""
    rows = pwe.size_ledger(measured)
    payload = {"T": _T, "rows": [list(r) for r in rows],
               "totals": pwe.ledger_totals(rows)}
    s = json.dumps(payload, default=str)
    back = json.loads(s)
    assert back["T"] == _T and len(back["rows"]) == len(pwe.PIECE_PLAN)
    sys.stdout.write(f"\n[ledger-json] {len(s)} bytes, "
                     f"{len(back['rows'])} rows\n")
