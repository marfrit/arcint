#!/usr/bin/env python3
"""Red-first check that moe_block_unrolled() and moe_block_tiled() in
export_mtp.py are re-expressions of moe_block(), not different functions:
build a tiny synthetic MoE through all three emission paths with the same
random weights and the same input, and require

  unrolled vs. batched:
  1. structurally, verify_moe_lowering.py's scan reports the batched graph
     as NOT matching the unrolled pattern (the red case: this is the check
     most likely to pass for the wrong reason -- a heuristic that flags any
     ScatterElementsUpdate, rather than specifically a per-expert
     NonZero + ScatterElementsUpdate(reduction="sum") chain, would wrongly
     call the batched graph "unrolled" too, since it already has one
     ScatterElementsUpdate for its router weights. That naive heuristic is
     demonstrated failing below before the real one is asked to pass.)
  2. structurally, the unrolled graph DOES match (green case);
  3. numerically, both graphs produce the same output on CPU for the same
     random input (np.allclose, atol=1e-3, f32) -- the lowering changes the
     graph shape, not the arithmetic.

  tiled vs. batched:
  4. structurally, a red case specific to this pair: moe_block_tiled()'s
     router is deliberately built the *same way* as moe_block()'s (single
     ScatterElementsUpdate(reduction="none"), no OneHot, no NonZero) -- so a
     classifier that looked only at the router would call tiled "batched"
     too. Demonstrated failing below (on both graphs) before requiring the
     real classifier -- which also looks for Tile and rank-3 compressed
     weight Constants -- to tell them apart correctly.
  5. structurally, the tiled graph DOES match (green case);
  6. numerically, batched (exact f32 reference) and tiled (u8-quantized
     experts) agree within a quantization-aware tolerance, not np.allclose's
     usual float slop -- see the tolerance comment at that check.

No model weights are loaded; everything here is synthetic and small
(E=4, H=32, I=64, k=2, M=8 tokens) so this runs in well under a second.
Standalone: `python3 test_moe_lowering.py` from anywhere.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_mtp as em          # noqa: E402
import verify_moe_lowering as vm  # noqa: E402

try:
    import openvino as ov
except ImportError:
    print("SKIP: openvino not importable in this environment")
    sys.exit(0)

E, H, I, K, B, S = 4, 32, 64, 2, 1, 8


def make_weights(rng):
    scale = 0.1
    gu = (rng.normal(size=(E, 2 * I, H)) * scale).astype(np.float32)
    dn = (rng.normal(size=(E, H, I)) * scale).astype(np.float32)
    w = {
        "mlp.experts.gate_up_proj": gu,
        "mlp.experts.down_proj": dn,
        "mlp.gate.weight": (rng.normal(size=(E, H)) * scale).astype(np.float32),
        "mlp.shared_expert.gate_proj.weight": (rng.normal(size=(I, H)) * scale).astype(np.float32),
        "mlp.shared_expert.up_proj.weight": (rng.normal(size=(I, H)) * scale).astype(np.float32),
        "mlp.shared_expert.down_proj.weight": (rng.normal(size=(H, I)) * scale).astype(np.float32),
        "mlp.shared_expert_gate.weight": (rng.normal(size=(1, H)) * scale).astype(np.float32),
    }
    return w


def build_model(fn, w):
    """Builds with a fully DYNAMIC [-1,-1,H] parameter -- the same
    PartialShape build_mtp_layer() gives the real MoE block (B, S are
    genuinely unknown at export time in production). An earlier version of
    this harness used a static-looking [1,8,H] parameter instead; diffing
    this file's graph against the ground-truth IR found that mismatch (flagged
    explicitly as worth checking after the second GPU crash) -- ground
    truth's entire subgraph, router included, is built around dynamic B, S.
    Whether the earlier static parameter itself contributed to a GPU-only
    failure was never confirmed (no GPU access here), but building this way
    is strictly more representative of what export_mtp.py actually emits, so
    it is now the default rather than an opt-in. compile_model() and run()
    still work normally: OpenVINO accepts concrete-shaped input arrays at
    inference time against a dynamic-shaped compiled model without needing
    the graph itself to be static."""
    y = em.op.parameter(ov.PartialShape([-1, -1, H]), ov.Type.f32, name="y")
    out = fn(y, w, "", K, True)
    res = em.op.result(out)
    model = ov.Model([res], [y], fn.__name__)
    model.validate_nodes_and_infer_types()
    return model


def naive_looks_unrolled(model):
    """The heuristic this test's red case rules out: 'contains any
    ScatterElementsUpdate at all'. moe_block() already has exactly one (for
    its dense router-weight tensor), so this naive check wrongly says yes."""
    return any(o.get_type_name() == "ScatterElementsUpdate" for o in model.get_ordered_ops())


def looks_batched_by_router_alone(model):
    """The heuristic this test's second red case rules out: 'no NonZero, no
    OneHot, and a ScatterElementsUpdate(reduction="none") is present' as a
    sufficient condition for "batched". moe_block_tiled()'s router is built
    identically to moe_block()'s on purpose (same Softmax/TopK/renorm/
    ScatterElementsUpdate shape, per the on-card measurement this mode
    imitates), so this heuristic says yes for both -- distinguishing them
    needs the Tile op and the rank-3 compressed weight Constants, which is
    what verify_moe_lowering.structural_scan() actually checks."""
    ops = model.get_ordered_ops()
    has_nonzero = any(o.get_type_name() == "NonZero" for o in ops)
    has_onehot = any(o.get_type_name() == "OneHot" for o in ops)
    has_scatter_none = any(
        o.get_type_name() == "ScatterElementsUpdate"
        and o.get_attributes().get("reduction", "none") == "none"
        for o in ops
    )
    return (not has_nonzero) and (not has_onehot) and has_scatter_none


def run(model, y_np):
    cm = ov.Core().compile_model(model, "CPU")
    return cm([y_np])[cm.output(0)]


def main():
    rng = np.random.default_rng(1234)
    w = make_weights(rng)
    y_np = (rng.normal(size=(B, S, H)) * 0.1).astype(np.float32)

    batched_model = build_model(em.moe_block, w)
    unrolled_model = build_model(em.moe_block_unrolled, w)

    # --- 1. red case: the naive "any ScatterElementsUpdate" heuristic must
    #        NOT be what verify_moe_lowering.py actually uses -- prove it
    #        would give the wrong answer on the batched graph, then prove the
    #        real check gets it right. ------------------------------------
    assert naive_looks_unrolled(batched_model), (
        "sanity check on the test itself failed: moe_block() was expected to "
        "contain a ScatterElementsUpdate (its router-weight scatter) -- if "
        "this assert fails the test's premise for the red case is stale, "
        "not the classifier"
    )
    print("red case: naive 'any ScatterElementsUpdate' heuristic would call "
          "the batched graph unrolled (as expected -- that's why it's the "
          "wrong check)")

    batched_scan = vm.structural_scan(batched_model)
    assert batched_scan["classification"] != "unrolled", (
        f"RED: verify_moe_lowering classified the batched (moe_block) graph as "
        f"{batched_scan['classification']!r} instead of 'batched' -- it is using "
        f"a heuristic too weak to tell the two emission paths apart: {batched_scan}"
    )
    assert batched_scan["classification"] == "batched", (
        f"batched graph classified as {batched_scan['classification']!r}, expected 'batched': "
        f"{batched_scan}"
    )
    print(f"PASS: batched graph correctly NOT classified as unrolled ({batched_scan})")

    # --- 2. green case: the unrolled emission must match. -----------------
    unrolled_scan = vm.structural_scan(unrolled_model)
    assert unrolled_scan["classification"] == "unrolled", (
        f"unrolled graph classified as {unrolled_scan['classification']!r}, expected 'unrolled': "
        f"{unrolled_scan}"
    )
    assert unrolled_scan["expert_weight_constants"] == 3 * E, (
        f"expected {3 * E} per-expert weight constants (gate/up/down x {E} experts), "
        f"found {unrolled_scan['expert_weight_constants']}"
    )
    print(f"PASS: unrolled graph correctly classified as unrolled ({unrolled_scan})")

    # --- 3. numeric equivalence: same arithmetic, different graph shape. --
    out_batched = run(batched_model, y_np)
    out_unrolled = run(unrolled_model, y_np)
    max_diff = float(np.abs(out_batched - out_unrolled).max())
    assert np.allclose(out_batched, out_unrolled, atol=1e-3), (
        f"batched and unrolled emissions diverge numerically: max abs diff {max_diff}"
    )
    print(f"PASS: batched and unrolled outputs match on CPU (max abs diff {max_diff:.3e})")

    # ======================================================================
    # tiled vs. batched
    # ======================================================================
    tiled_model = build_model(em.moe_block_tiled, w)

    # --- 4. red case: a router-only heuristic cannot tell tiled from
    #        batched -- their routers are built the same way on purpose. ---
    assert looks_batched_by_router_alone(batched_model), (
        "sanity check on the test itself failed: moe_block()'s router was "
        "expected to look 'batched' under the router-only heuristic"
    )
    assert looks_batched_by_router_alone(tiled_model), (
        "sanity check on the test itself failed: moe_block_tiled()'s router "
        "was expected to ALSO look 'batched' under the router-only heuristic "
        "-- if this no longer holds, the tiled emission's router no longer "
        "matches the on-card measurement this mode is imitating, and the red "
        "case below stops being meaningful"
    )
    print("red case: a router-only heuristic (no NonZero, no OneHot, a "
          "ScatterElementsUpdate(reduction=\"none\") present) says 'batched' "
          "for BOTH the batched and tiled graphs (as expected -- the router "
          "shape genuinely doesn't distinguish them; Tile and the compressed "
          "weights have to)")

    tiled_scan = vm.structural_scan(tiled_model)
    assert tiled_scan["classification"] != "batched", (
        f"RED: verify_moe_lowering classified the tiled (moe_block_tiled) graph as "
        f"{tiled_scan['classification']!r} instead of 'tiled' -- it is relying on the "
        f"router shape (which tiled shares with batched) rather than Tile + compressed "
        f"weights: {tiled_scan}"
    )
    assert batched_scan["classification"] != "tiled", (
        f"RED: verify_moe_lowering classified the batched (moe_block) graph as "
        f"'tiled' -- false positive on Tile/compressed-weight detection: {batched_scan}"
    )
    print(f"PASS: batched and tiled graphs correctly NOT confused with each other "
          f"(batched={batched_scan['classification']!r}, tiled={tiled_scan['classification']!r})")

    # --- 5. green case: the tiled emission must match. --------------------
    assert tiled_scan["classification"] == "tiled", (
        f"tiled graph classified as {tiled_scan['classification']!r}, expected 'tiled': {tiled_scan}"
    )
    assert tiled_scan["tile"] >= 1, f"expected at least one Tile op, found {tiled_scan['tile']}"
    # 3 experts (gate/up/down) x 2 compressed-dtype constants each (the
    # weight itself and its zero_point -- ground truth stores zero_point at
    # the same low precision as the weight, not as a plain f32 Constant, so
    # compressed_weight() does too; see its docstring).
    assert tiled_scan["compressed_weights"] == 6, (
        f"expected 6 compressed weight constants (gate/up/down x {{weight, zero_point}}), "
        f"found {tiled_scan['compressed_weights']}"
    )
    print(f"PASS: tiled graph correctly classified as tiled ({tiled_scan})")

    # --- 5b. red-then-green: the CompressedWeightsBlock Reshape node that a
    #         real GPU compile caught missing. ------------------------------
    # A first version of moe_block_tiled() built each expert weight as a
    # flat rank-3 Constant with no groups dimension and no Reshape between
    # the dequant Multiply and the MatMul. It classified as "tiled" above
    # (Tile present, compressed dtype present, no NonZero) and matched
    # numerically -- every check this file had was green -- but a real GPU
    # compile of that IR crashed inside ConvertTiledMoeBlockToGatherMatmuls'
    # own rewrite with a MatMul batch-dim merge failure, because the pass's
    # matcher anchors on a specific node this file was not checking for: a
    # Reshape immediately downstream of the dequant Multiply that collapses
    # a rank-4 grouped-quantization tensor down to rank 3 (confirmed against
    # byte offsets read out of the one IR on record that provably fuses).
    # This assertion is that missing check. Demonstrated red first against a
    # reconstruction of the old, flat-rank-3 shape (built inline below, not
    # by calling compressed_weight() -- that function is exactly what got
    # fixed, so this has to build the broken shape independently to still
    # be able to fail), then green against the real moe_block_tiled().
    def flat_rank3_compressed_weight(arr3d, name):
        """The old, buggy shape: skips the groups dimension and the
        Reshape entirely, dequantizing straight to rank 3. Quantizes
        per-(E,out) channel inline -- independent of export_mtp.py's
        current quantize_u8_grouped(), since this exists specifically to
        reconstruct a shape that function no longer produces."""
        amax = np.abs(arr3d).max(axis=-1, keepdims=True)
        scale = np.where(amax == 0, 1.0, amax / 127.0).astype(np.float32)
        q = np.clip(np.round(arr3d / scale) + 128.0, 0, 255).astype(np.uint8)
        q_c = em.op.constant(np.ascontiguousarray(q)); q_c.friendly_name = name
        zp = em.op.constant(np.full(scale.shape, 128.0, dtype=np.float32))
        sc = em.op.constant(scale)
        return em.op.multiply(em.op.subtract(em.op.convert(q_c, "f32"), zp), sc)

    def moe_block_tiled_flat_rank3(y, w, p, topk, norm_topk):
        """moe_block_tiled() with compressed_weight() swapped for the old,
        pre-fix flat-rank-3 builder above -- everything else identical."""
        old = em.compressed_weight
        em.compressed_weight = flat_rank3_compressed_weight
        try:
            return em.moe_block_tiled(y, w, p, topk, norm_topk)
        finally:
            em.compressed_weight = old

    broken_tiled_model = build_model(moe_block_tiled_flat_rank3, w)
    broken_scan = vm.structural_scan(broken_tiled_model)
    assert broken_scan["compressed_weight_reshapes"] == 0, (
        f"RED case setup is stale: the flat-rank-3 reconstruction was expected to have zero "
        f"CompressedWeightsBlock collapsing reshapes, found {broken_scan['compressed_weight_reshapes']}: "
        f"{broken_scan}"
    )
    print(f"red case: the old flat-rank-3 weight shape has zero CompressedWeightsBlock "
          f"collapsing reshapes ({broken_scan['compressed_weight_reshapes']}) -- this is the "
          f"IR that classified as 'tiled' and matched numerically yet crashed a real GPU "
          f"compile, so a check that only looks at classification/numerics would have stayed "
          f"green through that regression")

    assert tiled_scan["compressed_weight_reshapes"] == 3, (
        f"expected 3 CompressedWeightsBlock collapsing reshapes (gate/up/down), "
        f"found {tiled_scan['compressed_weight_reshapes']}: {tiled_scan}"
    )
    print(f"PASS: the real moe_block_tiled() has all 3 CompressedWeightsBlock collapsing "
          f"reshapes (compressed_weight_reshapes={tiled_scan['compressed_weight_reshapes']})")

    # --- 6. numeric equivalence within quantization tolerance. -------------
    # moe_block_tiled()'s expert weights are u8, 127 symmetric levels either
    # side of the zero_point -- roughly 1/127 (~0.8%) relative error per
    # weight. That error passes through three chained matmuls (gate, up,
    # down) each reducing over 32 or 64 elements; independent per-element
    # quantization error accumulates like a random walk, i.e. roughly
    # sqrt(reduction width) rather than linearly, so a few percent of the
    # reference output's own dynamic range is the right order of magnitude
    # to expect -- not float32 rounding noise (atol=1e-3 as used above), and
    # not exact equality. atol = 5% of the reference's max magnitude gives
    # about 6x headroom over what this configuration actually measures
    # (~0.8% observed under an earlier prototype at these same E/H/I/k/B/S),
    # which is tight enough that a real bug (wrong axis, wrong transpose,
    # dropped router weight) still trips it, but not so tight that ordinary
    # u8 quantization noise does.
    out_tiled = run(tiled_model, y_np)
    ref_max = float(np.abs(out_batched).max())
    tol = max(0.05 * ref_max, 1e-6)
    max_diff_q = float(np.abs(out_batched - out_tiled).max())
    assert max_diff_q <= tol, (
        f"batched (f32) and tiled (u8) emissions diverge beyond quantization tolerance: "
        f"max abs diff {max_diff_q} > tol {tol} (5% of reference max {ref_max})"
    )
    print(f"PASS: batched (f32) and tiled (u8) outputs agree within quantization tolerance "
          f"(max abs diff {max_diff_q:.3e}, tol {tol:.3e}, {100 * max_diff_q / ref_max:.2f}% of ref max)")

    # --- 7. static-shape compile check, as requested after the GPU crash. -
    # build_mtp_layer() gives the MoE block a fully dynamic [-1,-1,H]
    # parameter (real B, S are unknown at export time); the actual GPU
    # failure was a MatMul batch-dim merge error inside
    # ConvertTiledMoeBlockToGatherMatmuls' own rewrite, reported as
    # reproducible without a GPU by reshaping the model to a concrete input
    # shape and compiling for CPU, since static shape inference walks the
    # same class of check. Run that here, on both the broken flat-rank-3
    # shape and the real fix, and report honestly: on this OpenVINO build
    # (2026.3.1), neither the dynamic-to-static reshape nor a CPU compile
    # raises for the flat-rank-3 shape either -- the source graph itself was
    # shape-consistent (every MatMul operand pair here is confirmed
    # equal-rank below), which is *why* the crash was invisible on CPU in
    # the first place: it happened inside the GPU-only fusing pass's rewrite
    # of that graph, not in generic shape inference over it. Kept anyway
    # because it is real, cheap, GPU-free insurance against a genuine
    # source-graph shape bug (a different failure mode from the one that
    # actually hit), and because the coordinator asked for it explicitly.
    def static_shape_compile_check(fn, w, static_b=B, static_s=S):
        y = em.op.parameter(ov.PartialShape([-1, -1, H]), ov.Type.f32, name="y")
        out = fn(y, w, "", K, True)
        res = em.op.result(out)
        m = ov.Model([res], [y], fn.__name__ if hasattr(fn, "__name__") else "moe")
        m.reshape({"y": [static_b, static_s, H]})
        m.validate_nodes_and_infer_types()
        matmul_ranks = []
        for o in m.get_ordered_ops():
            if o.get_type_name() != "MatMul":
                continue
            ranks = [len(o.input_value(i).get_partial_shape()) for i in range(o.get_input_size())]
            statics = [o.input_value(i).get_partial_shape().is_static for i in range(o.get_input_size())]
            matmul_ranks.append((o.get_friendly_name(), ranks, statics))
        cm = ov.Core().compile_model(m, "CPU")
        return {"matmul_ranks": matmul_ranks, "compiled": cm is not None}

    for label, fn in (("broken (flat rank-3)", moe_block_tiled_flat_rank3),
                      ("fixed (moe_block_tiled)", em.moe_block_tiled)):
        result = static_shape_compile_check(fn, w)
        # only the per-expert-stacked GEMMs (both operands rank>=3, meant to
        # batch-multiply) are checked for equal rank; router/shared-expert
        # matmuls legitimately pair a rank-3 activation with a rank-2
        # weight (linear()'s normal broadcasting form, unrelated to this
        # fix) and are not flagged for that.
        bad = [(name, ranks) for name, ranks, statics in result["matmul_ranks"]
              if (min(ranks) >= 3 and len(set(ranks)) != 1) or not all(statics)]
        assert result["compiled"], f"{label}: static-shape CPU compile failed unexpectedly"
        assert not bad, f"{label}: found MatMul(s) with mismatched/dynamic operand ranks: {bad}"
        print(f"static-shape compile check [{label}]: compiled for CPU with fully static, "
              f"equal-rank MatMul operands ({len(result['matmul_ranks'])} MatMuls checked)")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
