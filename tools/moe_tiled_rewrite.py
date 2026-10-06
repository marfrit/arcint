#!/usr/bin/env python3
"""Rewrite a serving-shape IR emitted BEFORE 2026-09-17 into the shape the GPU
plugin's tiled MoE matcher accepts -- in memory, before compile_model, so an
artifact that took hours to export does not have to be exported again for a
compile-time census.

The defect (campaign sub4bit-vram-kernel, status 2026-09-17): `emit_moe_tiled`
fed the down-projection MatMul [E,M,H] straight into the router-weight
Multiply, and the router side went Transpose -> Unsqueeze. The plugin's
ConvertTiledMoeBlockTo3GatherMatmuls (`build_3gemm_pattern`) anchors on
`end_reshape` = Reshape(down MatMul) and `router_reshape` =
Reshape(Transpose(ScatterElementsUpdate)) -> optional Unsqueeze. Without them
the block compiles as 230 FullyConnected over the Tile, every expert for every
token, at full residency.

This pass inserts exactly those two Reshapes, with the construction the fixed
emitter uses: target [E, 1, -1, H] for the expert outputs and [E, 1, -1] for
the router weights. B is the literal 1 this family has (hidden [1,T,H]); the
runtime -1 is what keeps the Reshape from being folded away at validate/save
(export_mtp.py:515-531). The Multiply then runs at [E,1,S,H] x [E,1,S,1] and
the ReduceSum over axis 0 yields [1,S,H] -- value-identical to the old
[M,H] with M = S, and the same shape the fused rewrite produces.

Idempotent: a block whose Multiply already reads a Reshape is left alone, so
the pass is safe on artifacts from the fixed emitter.

What a rewritten model proves: the SAME thing the fixed emitter's artifact
proves at the census (primitive types off `get_runtime_model()`), on the
measured artifact. What it does not give: an on-disk artifact the C++ serve
path can load -- for that either re-export or `ov.save_model` the rewritten
model (a full .bin write).

Usage:
    python3 moe_tiled_rewrite.py <ir.xml>              # walk, rewrite, walk
    python3 moe_tiled_rewrite.py <ir.xml> --out <dir>  # ... and save_model
"""
import argparse
import sys

import numpy as np


def _type(node):
    return node.get_type_name()


def _find_blocks(model):
    """Yield (mul3, matmul_output, unsqueeze) for every old-style tiled MoE
    block: ReduceSum(keep_dims=false) <- Multiply(MatMul, Unsqueeze(Transpose(
    ScatterElementsUpdate))). Either operand order."""
    for rs in model.get_ordered_ops():
        if _type(rs) != "ReduceSum":
            continue
        attrs = rs.get_attributes()
        if str(attrs.get("keep_dims", "false")).lower() == "true":
            continue
        mul3 = rs.input_value(0).get_node()
        if _type(mul3) != "Multiply":
            continue
        a, b = mul3.input_value(0), mul3.input_value(1)
        for outs, uns in ((a, b), (b, a)):
            if _type(outs.get_node()) != "MatMul" or _type(uns.get_node()) != "Unsqueeze":
                continue
            tr = uns.get_node().input_value(0).get_node()
            if _type(tr) != "Transpose":
                continue
            if _type(tr.input_value(0).get_node()) != "ScatterElementsUpdate":
                continue
            r = outs.get_partial_shape().rank
            if not r.is_static or r.get_length() != 3:
                continue
            yield mul3, outs, uns.get_node()
            break


def _strip_swish_beta(model):
    """Swish(x, beta=Constant 1.0) -> Swish(x). The binding's `op.swish`
    appends the beta input; the matcher declares Swish with one input and
    the C++ Matcher rejects an argument-count mismatch (measured 2026-09-17:
    Reshapes in place, Swish in=2, census 0 MoE primitives). Returns the
    number of nodes changed."""
    n = 0
    for sw in model.get_ordered_ops():
        if _type(sw) != "Swish" or sw.get_input_size() != 2:
            continue
        beta = sw.input_value(1).get_node()
        if _type(beta) != "Constant":
            continue
        val = np.asarray(beta.get_data()).reshape(-1)
        if val.size != 1 or float(val[0]) != 1.0:
            continue                                  # a real beta: not ours
        sw.set_arguments([sw.input_value(0)])
        sw.validate_and_infer_types()
        n += 1
    return n


def _chain_to_f16(model):
    """Constant(u4) -> Convert(f32) -> Subtract(Convert(u4 zp -> f32)) ->
    Multiply(f32 scale Constant) -> Reshape  ==>  the same chain in f16 with
    a trailing Convert -> f32 before the MatMul: the fusing 35B control's
    shape. Not cosmetic: under f16 inference the plugin inserts a Convert on
    an f32 scale Constant feeding MOECompressed and the offload series' OTD
    resolver demands a direct Constant there ("Expected constant input for
    MOE3GemmFusedCompressed, got: Convert", B60 census 2, 2026-09-17).
    Returns the number of chains converted."""
    from openvino import Type, opset13 as op

    n = 0
    for mm in model.get_ordered_ops():
        if _type(mm) != "MatMul":
            continue
        rs = mm.input_value(1).get_node()
        if _type(rs) != "Reshape" or rs.get_output_element_type(0) != Type.f32:
            continue
        mul = rs.input_value(0).get_node()
        if _type(mul) != "Multiply":
            continue
        sub, scale = mul.input_value(0).get_node(), mul.input_value(1).get_node()
        if _type(sub) != "Subtract" or _type(scale) != "Constant":
            continue
        cw, cz = sub.input_value(0).get_node(), sub.input_value(1).get_node()
        if _type(cw) != "Convert" or _type(cz) != "Convert":
            continue
        w, zp = cw.input_value(0).get_node(), cz.input_value(0).get_node()
        if _type(w) != "Constant" or _type(zp) != "Constant":
            continue
        if w.get_output_element_type(0) not in (Type.u4, Type.i4, Type.u8, Type.i8):
            continue                                  # the matcher's own type list
        sc16 = op.constant(np.asarray(scale.get_data(), dtype=np.float32).astype(np.float16))
        sc16.set_friendly_name(scale.get_friendly_name())
        x = op.multiply(op.subtract(op.convert(w, Type.f16), op.convert(zp, Type.f16)), sc16)
        x = op.reshape(x, rs.input_value(1), special_zero=False)
        x.set_friendly_name(rs.get_friendly_name())
        x = op.convert(x, Type.f32)
        mm.input(1).replace_source_output(x.output(0))
        n += 1
    return n


def rewrite_tiled_moe(model):
    """Insert the two matcher-anchoring Reshapes into every old-style block,
    strip the beta input off every Swish(x, 1.0), and turn f32 dequant
    chains into the f16 form. Returns a dict with the THREE counts kept
    apart -- {"blocks", "swish", "chains"} -- all 0 on a conformant model.
    Validates the model afterwards so downstream shapes follow."""
    from openvino import opset13 as op

    n = 0
    n_sw = _strip_swish_beta(model)
    n_16 = _chain_to_f16(model)
    for mul3, outs, uns in list(_find_blocks(model)):
        ps = outs.get_partial_shape()
        E, H = ps[0], ps[2]
        if not (E.is_static and H.is_static):
            raise ValueError(f"{mul3.get_friendly_name()}: expert output shape {ps} "
                             f"is not [E static, M, H static]")
        E, H = E.get_length(), H.get_length()
        tag = mul3.get_friendly_name()
        # identity by instance id, checked BEFORE anything is spliced: a block
        # whose Multiply does not read both the expert output and the
        # Unsqueeze refuses here, with the model untouched (a reviewer's
        # catch: the earlier version raised after mutating)
        srcs = {mul3.input_value(i).get_node().get_instance_id() for i in range(2)}
        if srcs != {outs.get_node().get_instance_id(), uns.get_instance_id()}:
            raise RuntimeError(f"{tag}: the Multiply's operands are not the expert output "
                               f"and the router Unsqueeze; nothing rewritten")
        outs4 = op.reshape(outs, op.constant(np.array([E, 1, -1, H], np.int32)),
                           special_zero=False)
        outs4.set_friendly_name(f"{tag}/end_reshape")
        wt = uns.input_value(0)                                     # [E,M]
        wr = op.reshape(wt, op.constant(np.array([E, 1, -1], np.int32)),
                        special_zero=False)
        wr.set_friendly_name(f"{tag}/router_reshape")
        wu = op.unsqueeze(wr, op.constant(np.array([-1], np.int32)))
        # the Multiply keeps its operand order (pattern order: end_reshape
        # first when the emitter wrote it that way); identity by instance
        # id, and a block counts only when BOTH operands were replaced
        done = 0
        for i in range(2):
            src = mul3.input_value(i).get_node()
            if src.get_instance_id() == outs.get_node().get_instance_id():
                mul3.input(i).replace_source_output(outs4.output(0))
                done += 1
            elif src.get_instance_id() == uns.get_instance_id():
                mul3.input(i).replace_source_output(wu.output(0))
                done += 1
        assert done == 2, (tag, done)              # guaranteed by the check above
        n += 1
    if n or n_sw or n_16:
        model.validate_nodes_and_infer_types()
    return {"blocks": n, "swish": n_sw, "chains": n_16}


def walk(model):
    """(matched, failed-by-constraint) from tools/check_tiled_pattern.py."""
    import check_tiled_pattern as ctp
    ok, fail = [], {}
    for rs in model.get_ordered_ops():
        if _type(rs) != "ReduceSum":
            continue
        try:
            ctp.check_3gemm_from_reduce_sum(rs, lambda s: None)
            ok.append(rs.get_friendly_name())
        except ctp.Fail as f:
            fail[rs.get_friendly_name()] = f.constraint
    return ok, fail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir")
    ap.add_argument("--out", default=None, help="directory to save_model the rewritten IR into")
    a = ap.parse_args()
    import os
    import resource
    import openvino as ov
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    core = ov.Core()
    m = core.read_model(a.ir)
    before, _ = walk(m)
    r = rewrite_tiled_moe(m)
    after, fail = walk(m)
    print(f"blocks rewritten {r['blocks']} (swish {r['swish']}, chains {r['chains']}); "
          f"walker matched before {len(before)} after {len(after)}; "
          f"failing constraints after: {sorted(set(fail.values()))}; "
          f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GiB")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        xml = os.path.join(a.out, os.path.basename(a.ir))
        ov.save_model(m, xml, compress_to_fp16=False)
        print(f"saved {xml}")
    # success: every block whose Reshapes were inserted now walks; a chain or
    # Swish-only rewrite leaves the walker count where it was (it checks
    # neither dtypes nor, before 2026-09-17, input counts)
    return 0 if len(after) >= len(before) + r["blocks"] else 1


if __name__ == "__main__":
    sys.exit(main())
