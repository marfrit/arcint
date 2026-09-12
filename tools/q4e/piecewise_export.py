"""E2 window-050 -- PER-COMPONENT, REAL-WIDTH IR pieces at real checkpoint
geometry, plus the SIZE LEDGER and per-piece compile measurements.

SCOPE -- CORRECTION, 2026-09-12 (frontier ruling; supersedes the earlier
CAUSAL-ONLY reading). The checkpoint is 48 layers = 36 `linear_attention`
(GDN) + 12 full-attention layers (layer_idx % 4 == 3). The ruling excused the
QSA INDEXER only, not the attention layers: dense causal attention IS the
semantics the indexer approximates, so the attention piece is emitted DENSE
CAUSAL from the pin's own `Qwen4ExpTextAttention` minus the selection branch
(tools/q4e/attention.py). mrope is emitted in its degenerate-for-text form (all
position rows equal; fixed gather-and-apply over baked tables). The indexer's
own weights are NEVER fed into an emitted graph.

THE DELIVERABLE HERE IS PIECES, NOT THE WHOLE: a single real-width full-model
graph is ~689 GiB of f32 constants (measured refusal -- build_backbone_ir
enumerates the blockers) and is never attempted. Each piece is an OVModel at
the checkpoint's REAL geometry (H=2560, 48-layer GDN/attention/MoE/hc/PLE
widths), carrying a fed subset of the real tensors, sized to fit the
residency tier the LEDGER assigns it.

PIECES (name: build |
  constant | resid tier):
  * gdn_block            -- one GDN block, real conv_dim 10240 / key 2048 /
    value 6144 (tools/q4e/gdn.py), attention_mask ones. const ~464 MB f32
    (0.18 GiB). CARD-RESIDENT-FIRST (per WP6b attn+GDN 1.40 GiB blended).
  * attention_block      -- one dense-causal full-attn block, real heads 24 /
    kv 2 / head_dim 256 / rotary 64 (tools/q4e/attention.py). const ~132 MB
    f32. CARD-RESIDENT-FIRST (own line).
  * moe_router           -- [1,T,H] -> [T,E] gate only (moe._router_gate).
    const ~5 MB f32.
  * moe_shexp            -- shared-expert piece (moe.emit_shared_expert).
    const ~6 MB f32.
  * moe_chunk_<e0>_<e1>  -- one expert chunk (64 experts default):
    gate_up+down const ~1.18 GiB f32 per chunk. OFFLOAD tier (host/NVMe;
    never a single 512-expert dense graph -- that is 9.4 GiB f32/layer).
  * gatedresidual_<k>    -- the hc mixer for a layer (attn/ffn; hc.py
    emit_hc): const ~16 MB f32. CARD-RESIDENT-FIRST.
  * hc_combine           -- the final hc mixer (hc.py emit_combine).
    const ~16 MB f32. CARD-RESIDENT-FIRST.
  * ple_block            -- the single real PLE layer (layer_idx 1): real
    projections/norms/conv at width hc*H=10240; the n-gram embedding table is
    emitted only in a LEG-WINDOWED row set [K, 160] (the full table is
    [320,001,536, 160]  f32 = ~205 GiB -- NOT emittable; serving keeps the
    quantized table host-mmap'd, WP6b PLE 26.82 GiB) with base-adjusted
    row_ids. const K*160*4 bytes + ~80 MB proj. HOST-MMAP (table) /
    card (proj).
  * embed                -- token cross: Gather over token_embd [248,320,2560]
    f32 2.4 GiB const (Q8_0 in GGUF) by input_ids. const 2.4 GiB. CARD.
  * lm_head              -- the SEPARATE served head (output.weight Q6_K):
    [1,T,H] -> logits over the full vocab, W const [V,H] f32 2.4 GiB.
    CARD.

MEASUREMENTS per piece (the fenceposts the gates hang on): node count (model
ops before compile), constant blob bytes, CPU compile time, and -- for the
numeric legs -- OV-vs-pin max-abs / equality per the standing floor rules.
The parser of the ledger is `size_ledger()`: every row is one PIECE (piece x
residency), and each row names the checkpoint-quantized GiB (read from GGUF
metadata), the emitted f32 const GiB, the residency tier, and the node count.

RESIDENCY TIERS (the ones the assembled window can actually hold):
  CARD -- GPU.0 (B60, 22.71 GiB) / GPU.1 (A770 16 GB, the reserve, ~15 GiB
  usable WP9). card-resident-first: attention, GDN, hc, router, PLE proj,
  embed, head.
  OFFLOAD -- expert bodies (host pinned, then NVMe), per miss the ~1.0986
  GiB/token tier of WP6b.
  HOST-MMAP -- the PLE n-gram table (26.82 GiB quantized), never an f32
  constant.
Totals confront BOTH the 22.71 GiB card figure and ~15 GiB (the actual
reserve) and the WP6b numbers (176.94 B params / 85.38 GiB; routed 120.80 B /
56.25 GiB offload).

PARITY LEGS (CORRECTED 2026-09-12). The superseded header named
`tests/python/test_piecewise_export.py` as the home of every leg below. That
file was DISCARDED in the 2026-09-12 triage (its test-side key convention was
`self_attn.`/`mlp.`/`linear_attn.`-prefixed against emitters that read
module-relative keys, and it asserted indexer_n_heads == 3 where the file says
4). The legs now live in the per-piece suites named per bullet; one bullet has
NO home and says so. Every piece has a numeric leg vs the pin/f64 reference on
FED REAL tensors at >= 2 sequence lengths and (GDN/attention/MoE) >= 2 input
shapes:
  * gdn_block: emitted GDN vs an f64 reference on fed real weights;
    max-abs at the OV-f32-vs-f64 floor.
  * attention_block (tests/python/test_attention_piece.py): emitted
    DENSE-CAUSAL vs the pin's own Qwen4ExpTextAttention WITH ITS INDEXER (QSA)
    on fed real weights and identical fed tensors. The superseded text read
    "ACCEPTANCE IS KLD-SHAPED, NOT EQUALITY-SHAPED (frontier ruling): the leg
    PASTES the small non-zero divergence (QSA legitimately prunes)". MEASURED,
    QSA does not prune below its budget and the price is EXACTLY 0.0 there
    (0.000000e+00 over 0 rows at T=64 and T=96; 2.385560e-02 over 29/2080 above
    it), so the leg is EQUALITY-shaped at serving prefill lengths and pastes the
    non-zero price only above the budget. See the corrected PRICE paragraph in
    q4e/attention.py. The emission MATH is separately floored against the pin
    with its indexer stubbed to an all-zero additive mask -- not against a
    hand-written reference, which is what shared the emitter's misreading.
  * moe_*: router gate equality with the pin's `router_gate` on fed real
    tensors (tests/python/test_moe_block.py, which also gates dense==sparse for
    the FULL graph). **THE CHUNK LEG HAS NO HOME AS OF 2026-09-12**: the only
    cell that ever ran sum-over-chunks + shared == the pin's SPARSE output went
    with the discarded file, so `build_experts_chunk_model`'s partition of the
    expert axis and its per-chunk gate slice are UNGATED -- test_size_ledger.py
    builds moe_router and moe_shexp but never an expert chunk. Treat the
    chunk-split theorem as asserted-by-construction until a cell exists.
  * gatedresidual_* / hc_combine (tests/python/test_hc_block.py,
    test_hc_combine_block.py): equality vs the pin's mixers on fed real tensors
    (the mixer classes already have transcription-vs-pin == 0).
  * ple_block (tests/python/test_ple_block.py): vs the pin's
    Qwen4ExpTextPLELayer on fed real weights. max-abs at the conv-floored
    bound. The leg-windowed table (`windows=`) is exercised for SIZE only, by
    test_size_ledger.py; `emit_ple` still does not forward `windows`
    (CARRY-FORWARD).
  * embed / lm_head: exactness (Gather/MatMul are exact) on real rows; sized in
    test_size_ledger.py.
"""
import math

import numpy as np
from openvino import Model, Type
from openvino import opset13 as op

from . import attention, gdn, hc, moe, ple
from .gdn import _c, _i, _mm, _reshape, _slice, _add, _mul  # noqa: F401


# ---------------------------------------------------------------------------
# The real checkpoint geometry, read from the shipped GGUF metadata on the dev
# host, 2026-09-12 (the run re-verifies against the feed when one is present).
# Nothing here is guessable from defaults: each constant has a GGUF provenance.
# ---------------------------------------------------------------------------
REAL_GEOMETRY = {
    "vocab_size": 248320,
    "hidden_size": 2560,
    "num_hidden_layers": 48,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,              # attention.key_length/attention.value_length
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,         # attention.layer_norm_rms_epsilon
    # GDN (from blk.0 tensor shapes: attn_qkv [2560,10240], attn_gate
    # [2560,6144], ssm_alpha/beta [2560,48], ssm_norm [128]):
    "linear_key_head_dim": 128, "linear_num_key_heads": 16,   # key_dim 2048
    "linear_value_head_dim": 128, "linear_num_value_heads": 48,  # value 6144
    "linear_conv_kernel_dim": 4,
    # MoE:
    "num_experts": 512, "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,          # expert_feed_forward_length
    "shared_expert_intermediate_size": 640,  # expert_shared_feed_forward_length
    # GatedResidual hyper-connection:
    "hc_count": 4, "hc_lowrank": 320,       # hyper_connection.count / low_rank
    # PLE (single layer, layer_idx 1 -> config ple_layer_ids [2]):
    "ple_embed_dim": 2560,                  # default -> hidden_size
    "ple_conv_kernel_size": 4, "ngram_size": 3, "heads_per_ngram": 8,
    "ngram_vocab_size_base": 20_000_000, "make_ngram_vocab_size_divisible_by": 128,
    "eos_token_id": 248044,
    # full-attention / sparse: 12 QSA layers at layer_idx % 4 == 3, from
    # `qwen4exp.attention.compress_ratios` = 4 at exactly blk 3,7,...,47.
    # The indexer: `qwen4exp.attention.indexer.head_count` = 4 QUERY heads
    # (corrected 2026-09-12 from 3, which mis-read head_count as n+kv),
    # indexer.key_length 128, indexer.top_k 2048. The shapes are the second
    # witness: the pin builds ONE fused `index_qk_proj` of
    # (n_heads + kv_heads) * head_dim (pin 685-689) and the GGUF ships it split
    # as indexer.q_proj [512, 2560] + indexer.k_proj [128, 2560] = 640 = 5*128,
    # so n_heads=4, kv_heads=1. At n_heads=3 the pin builds 512 and the 640-row
    # concatenation cannot load -- the test asserts exactly that.
    "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128,
    "indexer_budget": 2048, "indexer_compress_ratio": 4,
    # rope: theta 1e7, rotary dim 64 = head_dim * partial 0.25, mrope sections
    # [11,11,10,0]:
    "rope_theta": 10_000_000.0, "partial_rotary_factor": 0.25,
    "mrope_section": [11, 11, 10, 0],
    "tie_word_embeddings": False,           # served head is independent
    # ngram table, from per_layer_token_embd metadata:
    "ngram_total_vocab": 320_001_536,       # sum of the 16 head vocab sizes
    "ngram_head_dim": 160,                  # per_head embedding width
    "ple_layer_ids": [2],                   # GGUF ple.layers [1] -> 1-index
}


def real_config(transformers_config_cls=None):
    """A pin Qwen4ExpTextConfig at the REAL checkpoint geometry."""
    if transformers_config_cls is None:
        from transformers.models.qwen4_exp import \
            configuration_qwen4_exp as pin_cfg
        transformers_config_cls = pin_cfg.Qwen4ExpTextConfig
    g = REAL_GEOMETRY
    layer_types = [
        "qwen_sparse_attention" if i % 4 == 3 else "linear_attention"
        for i in range(g["num_hidden_layers"])
    ]
    return transformers_config_cls(
        vocab_size=g["vocab_size"],
        hidden_size=g["hidden_size"],
        num_hidden_layers=g["num_hidden_layers"],
        num_attention_heads=g["num_attention_heads"],
        num_key_value_heads=g["num_key_value_heads"],
        head_dim=g["head_dim"],
        max_position_embeddings=g["max_position_embeddings"],
        rms_norm_eps=g["rms_norm_eps"],
        linear_key_head_dim=g["linear_key_head_dim"],
        linear_num_key_heads=g["linear_num_key_heads"],
        linear_value_head_dim=g["linear_value_head_dim"],
        linear_num_value_heads=g["linear_num_value_heads"],
        linear_conv_kernel_dim=g["linear_conv_kernel_dim"],
        num_experts=g["num_experts"],
        num_experts_per_tok=g["num_experts_per_tok"],
        moe_intermediate_size=g["moe_intermediate_size"],
        shared_expert_intermediate_size=g["shared_expert_intermediate_size"],
        hc_count=g["hc_count"],
        hc_lowrank=g["hc_lowrank"],
        ple_embed_dim=g["ple_embed_dim"],
        ple_conv_kernel_size=g["ple_conv_kernel_size"],
        ngram_size=g["ngram_size"],
        heads_per_ngram=g["heads_per_ngram"],
        ngram_vocab_size_base=g["ngram_vocab_size_base"],
        make_ngram_vocab_size_divisible_by=g["make_ngram_vocab_size_divisible_by"],
        eos_token_id=g["eos_token_id"],
        indexer_n_heads=g["indexer_n_heads"],
        indexer_kv_heads=g["indexer_kv_heads"],
        indexer_head_dim=g["indexer_head_dim"],
        indexer_budget=g["indexer_budget"],
        indexer_compress_ratio=g["indexer_compress_ratio"],
        layer_types=layer_types,
        tie_word_embeddings=g["tie_word_embeddings"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": g["rope_theta"],
            "partial_rotary_factor": g["partial_rotary_factor"],
            "mrope_section": g["mrope_section"],
        },
    )


# ---------------------------------------------------------------------------
# Piece builds. Every builder takes (config, seq_len) + an optional feed state
# dict; the state dicts use the pin's own module-relative keys so a caller can
# either feed real tensors (gguf_feed) or drop random weights of the right
# shape (device-free unit).
# ---------------------------------------------------------------------------
def build_gdn_piece(config, state, seq_len, amask=None):
    """One real-width GDN block. amask: attention_mask values; None -> ones."""
    return gdn.build_gdn_model(config, state, seq_len)


def build_attention_piece(config, state, seq_len):
    return attention.build_dense_attention_model(config, state, seq_len)


def build_moe_router_piece(config, state, seq_len):
    return moe.build_router_model(config, state, seq_len)


def build_moe_shared_piece(config, state, seq_len):
    return moe.build_shared_expert_model(config, state, seq_len)


def build_moe_chunk_piece(config, state, seq_len, e0, e1):
    return moe.build_experts_chunk_model(config, state, seq_len, e0, e1)


def build_hc_piece(config, state, seq_len):
    return hc.build_hc_model(config, state, seq_len)


def build_hc_combine_piece(config, state, seq_len):
    return hc.build_combine_model(config, state, seq_len)


def build_ple_piece(config, state, seq_len, with_mask=False, windows=None):
    return ple.build_ple_model(config, state, seq_len, with_mask=with_mask,
                               windows=windows)


def build_embed_piece(config, state, seq_len):
    """token_embd lookup: Gather over [V, H] by input_ids [1,T] i64."""
    T = int(seq_len)
    tbl = _c(state["embed_tokens.weight"])  # [V, H] f32
    ids = op.parameter([1, T], Type.i64)
    ids.set_friendly_name("input_ids")
    out = op.gather(tbl, ids, op.constant(np.int64(0)))
    res = op.result(out)
    res.set_friendly_name("output")
    return Model([res], [ids], "qwen4_exp_embed")


def build_lm_head_piece(config, state, seq_len):
    """The served lm_head: [1,T,H] -> [1,T,V], W [V,H] f32 constant."""
    T = int(seq_len)
    H = config.hidden_size
    v = config.vocab_size
    hid = op.parameter([1, T, H], Type.f32)
    hid.set_friendly_name("hidden_states")
    logits = _mm(hid, _c(state["lm_head.weight"]), tb=True)  # [1,T,V]
    res = op.result(logits)
    res.set_friendly_name("output")
    return Model([res], [hid], "qwen4_exp_lm_head")


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------
def graph_measures(model):
    """(node_count, const_bytes, op_name_counts) before any compile: the fence
    the gate reports. Constants are counted from the graph the piece BUILDS
    (pre-fold); the folded size is what a compile consumes, noted separately
    only where materialisation matters."""
    ops = model.get_ordered_ops()
    const_bytes = 0
    nodes = 0
    counts = {}
    for o in ops:
        nodes += 1
        counts[o.get_type_name()] = counts.get(o.get_type_name(), 0) + 1
        if o.get_type_name() == "Constant":
            const_bytes += o.get_element_type().size * \
                int(np.prod(list(o.get_output_shape(0))))
    return nodes, int(const_bytes), counts


# ---------------------------------------------------------------------------
# The SIZE LEDGER. Every row: piece x residency tier x (quantized checkpoint
# GiB, emitted f32 const GiB, node count when measured).
# ---------------------------------------------------------------------------
def _gib(nbytes):
    return nbytes / (1024 ** 3)


# Card capacities. GPU.0 = Arc Pro B60, 24 GB with 22.71 GiB usable; GPU.1 =
# Arc A770 16 GB, the 0.5.0 reserve, ~15 GiB usable per WP9.
CARD_B60_GIB = 22.71
CARD_A770_GIB = 15.0

# WP6b's study of the same checkpoint, for the reconciliation row.
WP6B_TOTAL_PARAMS_B = 176.94
WP6B_TOTAL_GIB = 85.38          # quantized, as shipped
WP6B_ROUTED_PARAMS_B = 120.80
WP6B_ROUTED_GIB = 56.25         # the offload share
WP6B_PLE_GIB = 26.82            # n-gram table, quantized

# How many of each piece the 48-layer checkpoint actually instantiates. 36 GDN +
# 12 QSA comes from `qwen4exp.attention.compress_ratios` being 4 at exactly
# blk 3,7,...,47; every layer carries an MoE and TWO hyper-connection mixers
# (attn_hyper_connection + mlp_hyper_connection); the PLE layer and the final
# mixer and the two vocab tables are singletons.
PIECE_PLAN = {
    "gdn_block":        ("CARD",      36, "one per linear_attention layer"),
    "attention_block":  ("CARD",      12, "one per QSA layer, emitted dense-causal"),
    "moe_router":       ("CARD",      48, "gate only, [E,H] per layer"),
    "moe_shexp":        ("CARD",      48, "shared expert + its gate, per layer"),
    "gatedresidual":    ("CARD",      96, "2 per layer: attn_hyper + mlp_hyper"),
    "hc_combine":       ("CARD",       1, "final mixer, use_combine=False -> NO "
                                          "block_inject (the GGUF ships no "
                                          "output_hc_inject)"),
    "ple_block":        ("CARD",       1, "PLE projections/norms/conv only; the "
                                          "n-gram table is its own row"),
    "embed":            ("CARD",       1, "token_embd, Q8_0 in the file"),
    "lm_head":          ("CARD",       1, "output.weight, Q6_K -- NOT tied here"),
    "moe_experts":      ("OFFLOAD",   48, "512 experts per layer; emitted in "
                                          "chunks, never one dense graph"),
    "ple_ngram_table":  ("HOST-MMAP",  1, "[320001536, 160] IQ4_NL; never an "
                                          "emitted f32 constant"),
}

TIER_ORDER = ("CARD", "OFFLOAD", "HOST-MMAP")


def size_ledger(measured, quant_bytes=None):
    """THE SIZE LEDGER, as a report over MEASUREMENTS -- not a formula.

    Every earlier version of this function computed constant bytes from
    hand-written products of config fields. Three of those products were wrong
    (one multiplied the GDN projections by the conv kernel width and omitted
    attn_gate entirely; one had a `/ chunk_frac * chunk_frac` that cancels), and
    the docstring's "~464 MB f32 (0.18 GiB)" disagreed with itself by 2.5x. A
    ledger whose numbers nothing re-derives is the exact defect REVIEW 58e3e09
    finding 5 was raised about: literals that assert are facts, literals that sit
    are drift.

    So: `measured` maps piece name -> {"nodes", "const_bytes", "compile_s",
    "source"}, where source is "graph" when the figure came from
    `graph_measures()` on a really-built piece and "file" when the piece is too
    large to build and the figure was computed from the shipped tensor list (the
    512-expert bodies and the n-gram table). Nothing here invents a number;
    missing pieces are reported as missing rather than defaulted to zero.

    Returns rows of
        (piece, tier, count, note, nodes, const_bytes, compile_s, source)
    where const_bytes is PER INSTANCE. Use `ledger_totals` for the confrontation
    against the cards.
    """
    rows = []
    for piece, (tier, count, note) in PIECE_PLAN.items():
        m = measured.get(piece)
        if m is None:
            rows.append((piece, tier, count, note, None, None, None, "MISSING"))
            continue
        rows.append((piece, tier, count, note, m.get("nodes"),
                     m.get("const_bytes"), m.get("compile_s"),
                     m.get("source", "graph")))
    rows.sort(key=lambda r: (TIER_ORDER.index(r[1]), -(r[5] or 0) * r[2]))
    return rows


def ledger_totals(rows):
    """Per-tier totals (count x per-instance bytes) plus the grand total, and the
    confrontation against both cards. Returns a dict; asserts nothing."""
    per_tier = {t: 0 for t in TIER_ORDER}
    missing = []
    for piece, tier, count, _note, _nodes, cb, _cs, source in rows:
        if cb is None:
            missing.append(piece)
            continue
        per_tier[tier] += count * cb
    total = sum(per_tier.values())
    card = per_tier["CARD"]
    return {
        "per_tier_bytes": per_tier,
        "per_tier_gib": {t: _gib(b) for t, b in per_tier.items()},
        "total_bytes": total,
        "total_gib": _gib(total),
        "card_gib": _gib(card),
        "fits_b60": _gib(card) <= CARD_B60_GIB,
        "fits_a770_reserve": _gib(card) <= CARD_A770_GIB,
        "headroom_b60_gib": CARD_B60_GIB - _gib(card),
        "headroom_a770_gib": CARD_A770_GIB - _gib(card),
        "wp6b_total_gib": WP6B_TOTAL_GIB,
        "f32_over_quantized": (_gib(total) / WP6B_TOTAL_GIB
                               if WP6B_TOTAL_GIB else None),
        "missing": missing,
    }


def format_ledger(rows, totals):
    """The table the window manifest and RECONCILE paste."""
    out = [
        "piece              tier       xN   nodes  const/inst GiB   total GiB  "
        "compile s  source",
        "-" * 96,
    ]
    for piece, tier, count, _note, nodes, cb, cs, source in rows:
        if cb is None:
            out.append(f"{piece:18s} {tier:10s} {count:>3d}   "
                       f"{'--':>6s}  {'--':>13s}  {'--':>10s}  {'--':>9s}  {source}")
            continue
        # The "file"-sourced rows (the 512-expert bodies, the n-gram table) are
        # never built, so they have a byte count and no node count or compile
        # time. That is the honest shape of those rows, not a hole.
        n_s = f"{nodes:>6d}" if nodes is not None else f"{'n/a':>6s}"
        c_s = f"{cs:>9.2f}" if cs is not None else f"{'n/a':>9s}"
        out.append(f"{piece:18s} {tier:10s} {count:>3d}  {n_s}  "
                   f"{_gib(cb):>13.4f}  {_gib(cb) * count:>10.3f}  "
                   f"{c_s}  {source}")
    out.append("-" * 96)
    for t in TIER_ORDER:
        out.append(f"{'TOTAL ' + t:18s} {'':10s} {'':>3s}  {'':>6s}  "
                   f"{'':>13s}  {totals['per_tier_gib'][t]:>10.3f}")
    out.append(f"{'GRAND TOTAL':18s} {'':10s} {'':>3s}  {'':>6s}  {'':>13s}  "
               f"{totals['total_gib']:>10.3f}")
    out.append("")
    out.append(f"CARD tier {totals['card_gib']:.3f} GiB vs B60 "
               f"{CARD_B60_GIB} GiB -> "
               f"{'FITS' if totals['fits_b60'] else 'DOES NOT FIT'} "
               f"(headroom {totals['headroom_b60_gib']:+.3f} GiB)")
    out.append(f"CARD tier {totals['card_gib']:.3f} GiB vs A770 reserve "
               f"{CARD_A770_GIB} GiB -> "
               f"{'FITS' if totals['fits_a770_reserve'] else 'DOES NOT FIT'} "
               f"(headroom {totals['headroom_a770_gib']+0:+.3f} GiB)")
    out.append(f"GRAND TOTAL {totals['total_gib']:.1f} GiB of f32 constants vs "
               f"WP6b's {WP6B_TOTAL_GIB} GiB as shipped (quantized) -> "
               f"f32 costs {totals['f32_over_quantized']:.2f}x the checkpoint")
    if totals["missing"]:
        out.append(f"MISSING MEASUREMENTS: {totals['missing']}")
    return "\n".join(out)


def _geometry_dict(config):
    H = config.hidden_size
    return {
        "hidden_size": H,
        "num_experts": config.num_experts,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "moe_intermediate_size": config.moe_intermediate_size,
        "shared_expert_intermediate_size": config.shared_expert_intermediate_size,
        "linear_num_key_heads": config.linear_num_key_heads,
        "linear_key_head_dim": config.linear_key_head_dim,
        "linear_num_value_heads": config.linear_num_value_heads,
        "linear_value_head_dim": config.linear_value_head_dim,
        "linear_conv_kernel_dim": config.linear_conv_kernel_dim,
        "hc_count": config.hc_count,
        "hc_lowrank": config.hc_lowrank,
        "vocab_size": config.vocab_size,
        "ngram_total_vocab": REAL_GEOMETRY["ngram_total_vocab"],
        "ngram_head_dim": REAL_GEOMETRY["ngram_head_dim"],
    }


# ---------------------------------------------------------------------------
# The window base-adjusting row-id helper the PLE leg uses to canonicalise the
# emitted table to the leg's reachable rows (FIX D doctrine: narrow the fed
# tensor, never the semantics).
# ---------------------------------------------------------------------------
def windowed_row_ids(row_ids, base):
    """row_ids [1,T,Hn] int64 (or numpy) -> [1,T,Hn] offset by -base."""
    return np.asarray(row_ids, dtype=np.int64) - base


__all__ = [
    "REAL_GEOMETRY", "real_config", "graph_measures", "size_ledger",
    "build_gdn_piece", "build_attention_piece",
    "build_moe_router_piece", "build_moe_shared_piece", "build_moe_chunk_piece",
    "build_hc_piece", "build_hc_combine_piece", "build_ple_piece",
    "build_embed_piece", "build_lm_head_piece", "windowed_row_ids",
]