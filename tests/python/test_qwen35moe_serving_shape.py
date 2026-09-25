"""The `qwen3_5_moe` serving-shape emitter (Qwen3.6-35B-A3B) and the four
conventions design-qwen35moe-serving-shape §5 measures before any code:

  1. GDN output gate = SILU (`code`: llama.cpp qwen35moe.cpp build_norm_gated),
  2. value/key head map = TILED (the GGUF's value-head tensors are the HF
     interleave order re-laid into llama's tiled order),
  3. norms = PLAIN RMSNorm, no (1 + w), pre-norm residual,
  4. the tiled MoE lowering is E-agnostic (256/top-8 == 512/top-10 on the
     device-free CPU-plugin oracle).

Cells 2 and 3 are measured against the served int4 IR on the operator-local
host and recorded in the design note; the cells here pin what can be pinned
from THIS repository alone: the emitter's structure, the E-agnostic tiled
lowering, and -- when the shard is present -- the byte-exactness of the
emitted IQ2_S blocks against `q4e.native_blocks` (which is bit-exact vs
gguf-py on the same bytes).

`Q4E_GGUF_SHARDS` gates the real-shard cell, as in test_native_blocks.py.
"""
import os
import sys
import types
from pathlib import Path

import numpy as np
import openvino as ov
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from q4e import serving_shape as ss  # noqa: E402
from q4e import native_blocks as nb  # noqa: E402
from q4e import expert_fill as ef  # noqa: E402

_SHARDS = os.environ.get("Q4E_GGUF_SHARDS", "")
_skip = pytest.mark.skipif(not _SHARDS,
                           reason="Q4E_GGUF_SHARDS unset (real GGUF shards absent)")


def _compile_cpu(model):
    return ov.Core().compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})


def _cpu_gather_matmuls(compiled):
    n = 0
    for node in compiled.get_runtime_model().get_ops():
        ri = node.get_rt_info()
        lt = ri["layerType"].astype(str) if "layerType" in ri else ""
        if "gathermatmul" in lt.lower():
            n += 1
    return n


def _walk_tiled(model):
    import check_tiled_pattern as ctp
    matched, failures = [], {}
    for rs in model.get_ordered_ops():
        if rs.get_type_name() != "ReduceSum":
            continue
        try:
            ctp.check_3gemm_from_reduce_sum(rs, lambda s: None)
            matched.append(rs.get_friendly_name())
        except ctp.Fail as f:
            failures[rs.get_friendly_name()] = (f.constraint, f.observed, f.expected)
    return matched, failures


def test_qwen35moe_emits_a_plain_pre_norm_residual_layer_and_compiles_on_cpu():
    """Structure: 4 layers = 3 GDN + 1 full-attention (i % 4 == 3), NO
    hyper-connection mixer and NO PLE (this family has neither), and the
    graph compiles on the CPU plugin. The layer body is
    `hidden + mixer(rmsnorm(hidden))` / `hidden + moe(rmsnorm(hidden))`."""
    cfg = ss.qwen35moe_real_config(4)
    arena = ss.SparseArena(capacity_bytes=1 << 32)
    try:
        model, rep = ss.build_qwen35moe_serving_shape_ir(
            config=cfg, arena=arena, n_layers=4, rope_span=128)
    finally:
        arena.close()
    assert (rep["n_layers"], rep["gdn_layers"], rep["attn_layers"]) == (4, 3, 1)
    assert rep["has_ple"] is False
    names = {o.get_friendly_name() for o in model.get_ordered_ops()}
    assert not any("hyper_connection" in n for n in names), "qwen35moe has no hc mixer"
    assert not any(n.startswith("ple/") for n in names), "qwen35moe has no PLE"
    assert {"layer0/attn_norm", "layer0/post_norm", "layer0/mixer_out",
            "layer0/out"} <= names
    assert model.inputs[0].get_node().get_friendly_name() == "inputs_embeds"
    cm = _compile_cpu(model)
    assert cm is not None


def test_the_256_expert_moe_block_fuses_like_the_512_expert_block_device_free():
    """Design note §5 cell 4. The GPU fusion is a compile-time pass and cannot
    run here, but the CPU plugin runs the SAME
    ConvertTiledMoeBlockTo3GatherMatmuls lowering and emits three GatherMatmul
    primitives per matched block. At both geometries the walker matches the one
    MoE root and the lowering yields exactly three -- so the tiled constraint
    list is E-agnostic (the matcher reads E off the weight's leading dim; the
    fused kernels' E is a slot count). The GPU compile at E=256/IQ2_S stays
    OWED to the A770 window."""
    def build(E, k, H, I):
        cfg = types.SimpleNamespace(
            hidden_size=H, num_experts=E, num_experts_per_tok=k,
            moe_intermediate_size=I, shared_expert_intermediate_size=I,
            hidden_act="silu", norm_topk_prob=True)
        rng = np.random.default_rng(1)
        st = {
            "mlp.gate.weight": op.constant(rng.standard_normal((E, H)).astype(np.float32) * 0.1),
            "mlp.shared_expert.gate_proj.weight": op.constant(rng.standard_normal((I, H)).astype(np.float32) * 0.1),
            "mlp.shared_expert.up_proj.weight": op.constant(rng.standard_normal((I, H)).astype(np.float32) * 0.1),
            "mlp.shared_expert.down_proj.weight": op.constant(rng.standard_normal((H, I)).astype(np.float32) * 0.1),
            "mlp.shared_expert_gate.weight": op.constant(rng.standard_normal((1, H)).astype(np.float32) * 0.1),
        }
        arena = ss.SparseArena(capacity_bytes=1 << 32)
        hidden = op.parameter([1, -1, H], ov.Type.f32)
        hidden.set_friendly_name("hidden")
        with ss.shared_constants():
            y = ss.emit_moe_tiled(hidden, cfg, st, arena, -1, "moe")
        res = op.result(y)
        res.set_friendly_name("out")
        return ov.Model([res], [hidden], "moe_block"), arena

    from openvino import opset13 as op
    cases = [(512, 10, 2560, 640), (256, 8, 2048, 512), (256, 8, 256, 128)]
    for (E, k, H, I) in cases:
        model, arena = build(E, k, H, I)
        try:
            matched, failures = _walk_tiled(model)
            assert len(matched) == 1, (E, failures)
            gm = _cpu_gather_matmuls(_compile_cpu(model))
            assert gm == 3, f"E={E} top={k}: {gm} GatherMatmul primitives, expected 3"
        finally:
            arena.close()


@_skip
def test_emitted_iq2_s_blocks_decode_exactly_as_the_gguf_split(tmp_path):
    """The emitted IQ2_S gate/up blocks read back out of the arena decode
    exactly as the checkpoint's own bytes re-laid per role: the 10-bit grid
    indices and the RAW sign bytes byte-exact, the decode equal at the f16
    scale the artifact carries. The oracle is `q4e.native_blocks` (bit-exact vs
    gguf-py on the same shards). Red-first: reconstructing the index from the
    LOW BYTE ALONE (`gi & 0xFF`) changes the decode, because real indices
    exceed 255 -- the cell asserts the mutant differs."""
    from q4e import gguf_feed as gf
    feed = gf.GgufFeed(_SHARDS)
    if not feed.has("blk.0.ffn_gate_exps.weight"):
        pytest.skip("not the Qwen3.6-35B-A3B shard")
    if feed.gguf_type("blk.0.ffn_gate_exps.weight") != "IQ2_S":
        pytest.skip(f"gate experts are {feed.gguf_type('blk.0.ffn_gate_exps.weight')}")
    filler = ef.NativeExpertFiller(feed)
    cfg = ss.qwen35moe_real_config(1)
    arena = ss.SparseArena(path=str(tmp_path / "arena.bin"))
    try:
        model, rep = ss.build_qwen35moe_serving_shape_ir(
            config=cfg, arena=arena, n_layers=1, filler=filler, feed=feed)
        raw = feed.raw_rows("blk.0.ffn_gate_exps.weight", rows=256)
        gi2, si2, sc2 = nb.iq2_s_split(raw.reshape(256 * 512, -1))
        for kind in ("gate", "up"):
            base = f"layer0/moe/experts_{kind}"
            gb, _ = arena.read_back(base + "/gridix_u8")
            sb, _ = arena.read_back(base + "/signix_u8")
            scb, _ = arena.read_back(base + "/block_scale")
            gi = np.ascontiguousarray(gb).view("<u2").reshape(256 * 512, -1)
            si = np.ascontiguousarray(sb).reshape(256 * 512, -1)
            s16 = np.frombuffer(scb, dtype="<f2").astype(np.float32).reshape(256 * 512, -1)
            s8 = np.stack([s16.reshape(-1, 64, 2)[..., 0],
                           s16.reshape(-1, 64, 2)[..., 0],
                           s16.reshape(-1, 64, 2)[..., 1],
                           s16.reshape(-1, 64, 2)[..., 1]], axis=-1).reshape(256 * 512, -1)
            got = nb.iq2_s_decode(gi, si, s8)
            src = feed.raw_rows(f"blk.0.ffn_{kind}_exps.weight", rows=256) if kind == "up" \
                else raw
            wgi, wsi, wsc = nb.iq2_s_split(src.reshape(256 * 512, -1))
            want = nb.iq2_s_decode(wgi, wsi, wsc.astype(np.float16).astype(np.float32))
            assert np.array_equal(gi, wgi), f"{kind}: grid indices not byte-exact"
            assert np.array_equal(si, wsi.astype(np.uint8)), f"{kind}: sign bytes not byte-exact"
            assert np.abs(got - want).max() == 0.0, f"{kind}: decode differs"
            # MUTANT: low byte alone loses every index >= 256
            assert not np.array_equal(nb.iq2_s_decode(gi & np.uint16(0xFF), si, s8), want), \
                f"{kind}: the low-byte-only mutant was not caught"
    finally:
        arena.close()
