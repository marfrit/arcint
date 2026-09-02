#!/usr/bin/env python3
"""Walk an IR against the EXACT constraint list OpenVINO's GPU-plugin pass
ov::pass::ConvertTiledMoeBlockTo3GatherMatmuls (and the CompressedWeightsBlock
pattern block it uses for each expert weight) requires, in match order, and
report PASS/FAIL per constraint with the observed value at the first failure.

Source (read on the dev host, read-only, not reproduced verbatim here):
  transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.cpp
    -- build_3gemm_pattern() and ConvertTiledMoeBlockTo3GatherMatmuls's ctor
  transformations/pattern_blocks/compressed_weights_block.cpp
    -- CompressedWeightsBlock's ctor

The C++ matcher walks the graph backward from a ReduceSum root, matching a
tree of pattern nodes with per-edge shape/type/attribute/consumer-count
constraints; OpenVINO's Python bindings expose the finished pass but not its
internal pattern-matching primitives, so this script re-implements the same
backward walk in Python over ov.Model's op graph, using the plain node
introspection the bindings DO expose (get_type_name, get_attributes,
output(i).get_target_inputs() for consumer counts, get_partial_shape for
rank/static-ness). It is a re-implementation, not a call into the real
matcher.

Root nodes: every ReduceSum whose reduction matches the pattern's
keep_dims=true+Squeeze or keep_dims=false form is tried as a candidate match
start; the walk stops at the first failing constraint per candidate.

WHAT A PASS DOES NOT PROVE. Window H found a graph this walker reported FULL
MATCH on that the real matcher rejected -- proof the walk has blind spots,
not just a hedge. Known gaps, most to least likely to matter:

  (a) IR STAGE MISMATCH -- the live failure class. This walker inspects the
      SERIALIZED IR (what ov.save_model()/read_model() produces); the real
      matcher runs mid-pipeline, after earlier passes (MarkDequantization,
      ConvertGroupedMatMulToGroupedMatMulCompressed, ConvertPrecision, ...)
      have already rewritten the graph. A node arrangement that is correct
      in the serialized IR is not guaranteed to be what those earlier passes
      hand to ConvertTiledMoeBlockToGatherMatmuls -- the reshape-folding bug
      this tool caught (OpenVINO 2026.4.0 silently eliminating a rank-4
      Reshape during validate_nodes_and_infer_types(), invisible until
      re-inspecting the actual saved XML) is one instance of this whole
      class, not a one-off; another optimization in another pass, at a
      different pipeline stage, could just as easily do the same thing
      again without this walker ever seeing it.

  (b) NO rt_info / DEQUANTIZATION-MARK CHECKS. CompressedWeightsBlock's
      structural match may not be sufficient on its own -- MarkDequantization
      tags Convert nodes with rt_info the fusion passes downstream may
      require, and this walker has no access to (and does not check)
      per-node runtime info at all. A structurally identical chain missing
      that mark would still report PASS here.

  (c) OPERAND ORDER ON COMMUTATIVE OPS. The C++ pattern's wrap_type<Multiply>
      calls match either operand order; this walker used to assume a fixed
      position. Closed for the three Multiply sites this file checks
      (resolve_mul3_operands, resolve_swiglu_operands,
      resolve_weight_mul_operands below) -- trivial once named, since
      Swish/Gelu-vs-MatMul and Subtract-vs-anything are unambiguous by type.
      Two residual gaps: resolve_mul3_operands and resolve_weight_mul_operands
      fall back to the ORIGINAL (possibly wrong) order when both operands are
      type-ambiguous (e.g. optional_unsqueeze absent, or the no-zero-point
      mul_no_sub branch where both sides are Convert-shaped) -- a false FAIL
      or false PASS is still possible in exactly those configurations. And
      Subtract is NOT relaxed at all: the C++ pattern only defines
      Subtract(convert(weight), zero_point) in that order (no reversed
      alternative), so the fixed order this walker checks is believed
      correct, not a remaining gap -- flagged here only so a future subtract
      in the other order reads as a real FAIL, not a walker bug.

  (d) NO ATTRIBUTE/VALUE CHECKS beyond what is explicitly coded: Softmax's
      axis, TopK's mode/sort, the Transpose permutation constant's actual
      values, ScatterElementsUpdate's axis operand, ReduceSum's axes operand
      are none of them inspected -- only op TYPE and (for MatMul) the
      transpose_a/transpose_b attributes are checked. A graph with the right
      op skeleton but a wrong axis or permutation would still report PASS.

  (e) ELEMENT-TYPE PREDICATES only cover the weight Constant in
      CompressedWeightsBlock (W1's dtype-in-{u4,i4,i8,u8} check). No other
      typed input -- zero_point, scale, activations, indices -- has its
      dtype checked anywhere in this walker.

  (f) THE keep_dims=true REDUCESUM VARIANT IS NOT WALKED. R1 only follows
      the keep_dims=false direct path; a candidate whose ReduceSum has
      keep_dims=true (needing a following Squeeze, the ARM workaround shape
      build_3gemm_pattern() also accepts) is reported FAIL at R1 immediately,
      even if the rest of the chain would have matched.

Given all of the above: treat a PASS here as "this constraint looks
satisfied by direct inspection of the serialized IR", never as proof the
compiled pass would fire. Only a GPU compile proves that.

Usage:
    python3 check_tiled_pattern.py path/to/some_layer.xml
"""
import argparse
import sys

COMPRESSED_TYPES = {"u4", "i4", "i8", "u8"}


class Fail(Exception):
    def __init__(self, constraint, observed, expected):
        self.constraint = constraint
        self.observed = observed
        self.expected = expected
        super().__init__(f"{constraint}: observed={observed!r} expected={expected!r}")


def consumers(output):
    return len(output.get_target_inputs())


def type_of(node):
    return node.get_type_name()


def attrs_of(node):
    try:
        return node.get_attributes()
    except Exception:
        return {}


def require_type(node, allowed, constraint):
    t = type_of(node)
    if t not in allowed:
        raise Fail(constraint, t, "one of " + "/".join(allowed))
    return t


def require_consumers(output, n, constraint):
    c = consumers(output)
    if c != n:
        raise Fail(constraint, c, n)


def require_attrs(node, wanted, constraint):
    have = attrs_of(node)
    for k, v in wanted.items():
        got = have.get(k)
        # OpenVINO python attribute dicts stringify booleans; normalise.
        got_norm = str(got).lower() if isinstance(got, (bool, str)) else got
        want_norm = str(v).lower()
        if got_norm != want_norm:
            raise Fail(constraint, {k: got}, {k: v})


def matmul_weight_input(node):
    """The weight operand (input 1) producer node of a MatMul."""
    return node.input_value(1).get_node()


def resolve_mul3_operands(mul3):
    """mul3 = Multiply(end_reshape, optional_unsqueeze-or-router_reshape) is
    commutative in the C++ pattern (ov::pass::pattern's wrap_type<Multiply>
    matches either operand order), but this walker used to assume input(0)
    was always end_reshape. Classify by shape instead of position: the
    router side is an Unsqueeze, or (if optional_unsqueeze is absent) a
    Reshape fed directly by a Transpose; swap if input(0) looks like the
    router side instead."""
    a, b = mul3.input_value(0).get_node(), mul3.input_value(1).get_node()

    def looks_like_router_side(node):
        if type_of(node) == "Unsqueeze":
            return True
        if type_of(node) == "Reshape":
            try:
                return type_of(node.input_value(0).get_node()) == "Transpose"
            except Exception:
                return False
        return False

    if looks_like_router_side(a) and not looks_like_router_side(b):
        return b, a
    return a, b


def resolve_swiglu_operands(swiglu):
    """swiglu = Multiply(swish, up_matmul) is likewise commutative in the
    C++ pattern. Swish/Gelu vs MatMul are never ambiguous by type, so this
    one is fully, always resolvable (unlike mul3's heuristic above)."""
    a, b = swiglu.input_value(0).get_node(), swiglu.input_value(1).get_node()
    if type_of(a) in ("Swish", "Gelu"):
        return a, b
    if type_of(b) in ("Swish", "Gelu"):
        return b, a
    return a, b


def resolve_weight_mul_operands(mul_node):
    """CompressedWeightsBlock's mul = Multiply(subtract_or_convert, scale) is
    commutative too. Subtract is unambiguous, so swap if input(0) is the
    scale side and input(1) is the Subtract; if neither side is a Subtract
    (the no-zero-point mul_no_sub branch, both sides Convert-shaped) this
    cannot disambiguate by type alone and keeps the original order --
    same conservative limitation as resolve_mul3_operands."""
    a, b = mul_node.input_value(0).get_node(), mul_node.input_value(1).get_node()
    if type_of(b) == "Subtract" and type_of(a) != "Subtract":
        return b, a
    return a, b


def check_compressed_weights_block(weight_producer, label, log):
    """Walk the CompressedWeightsBlock chain backward from `weight_producer`
    (the node feeding a MatMul's weight input). Constraint order matches the
    C++ pattern's own construction order: Constant(dtype) -> Convert ->
    [Subtract(zero_point)] -> Multiply(scale) -> [Reshape(rank4->rank3)] ->
    [Transpose] -> [Convert]."""
    node = weight_producer
    trail = [f"{type_of(node)}"]

    # peel the optional trailing Convert (weights_input's outer optional Convert)
    if type_of(node) == "Convert":
        log(f"  [{label}] W8 optional final Convert: PRESENT ({node.get_friendly_name()})")
        node = node.input_value(0).get_node()
        trail.append(type_of(node))
    else:
        log(f"  [{label}] W8 optional final Convert: absent")

    # peel the optional Transpose
    if type_of(node) == "Transpose":
        log(f"  [{label}] W7 optional Transpose: PRESENT ({node.get_friendly_name()})")
        node = node.input_value(0).get_node()
        trail.append(type_of(node))
    else:
        log(f"  [{label}] W7 optional Transpose: absent")

    # the optional Reshape, with the rank predicate: BOTH input and output
    # rank must be static, input rank == 4, output rank == 3.
    reshape_node = None
    if type_of(node) == "Reshape":
        reshape_node = node
        in_ps = reshape_node.input_value(0).get_partial_shape()
        out_ps = reshape_node.output(0).get_partial_shape()
        in_rank = in_ps.rank
        out_rank = out_ps.rank
        in_static = bool(in_rank.is_static)
        out_static = bool(out_rank.is_static)
        log(f"  [{label}] W6 Reshape present: in_shape={in_ps} out_shape={out_ps} "
            f"in_rank_static={in_static} out_rank_static={out_static}")
        if not (in_static and out_static):
            raise Fail(f"{label}.W6.reshape_predicate.rank_static",
                      {"in_rank_static": in_static, "out_rank_static": out_static},
                      {"in_rank_static": True, "out_rank_static": True})
        in_len = in_ps.rank.get_length()
        out_len = out_ps.rank.get_length()
        if in_len != 4:
            raise Fail(f"{label}.W6.reshape_predicate.in_rank", in_len, 4)
        if out_len != 3:
            raise Fail(f"{label}.W6.reshape_predicate.out_rank", out_len, 3)
        log(f"  [{label}] W6 reshape_predicate: PASS (in_rank=4 out_rank=3, both static)")
        node = reshape_node.input_value(0).get_node()
        trail.append(type_of(node))
    else:
        log(f"  [{label}] W6 optional Reshape: absent (weight consumed directly at "
            f"{type_of(node)})")

    # `mul` -- Multiply(subtract_or_convert, scale)
    require_type(node, {"Multiply"}, f"{label}.W5.mul.type")
    mul_node = node
    a, scale_node = resolve_weight_mul_operands(mul_node)
    log(f"  [{label}] W5 Multiply (scale) present: lhs={type_of(a)} rhs={type_of(scale_node)}")
    # scale operand: Constant, or Convert(Constant)
    if type_of(scale_node) == "Convert":
        scale_const = scale_node.input_value(0).get_node()
        require_type(scale_const, {"Constant"}, f"{label}.W4.mul_scale.convert_const.type")
        log(f"  [{label}] W4 scale operand: Convert(Constant) -- PASS")
    else:
        require_type(scale_node, {"Constant"}, f"{label}.W4.mul_scale.type")
        log(f"  [{label}] W4 scale operand: Constant -- PASS")

    # `subtract` (optional): Subtract(convert, [Convert(]zero_point[)])
    if type_of(a) == "Subtract":
        sub_node = a
        convert_node = sub_node.input_value(0).get_node()
        zp_node = sub_node.input_value(1).get_node()
        log(f"  [{label}] W3 Subtract (zero_point) present: lhs={type_of(convert_node)} "
            f"rhs={type_of(zp_node)}")
        if type_of(zp_node) == "Convert":
            zp_const = zp_node.input_value(0).get_node()
            require_type(zp_const, {"Constant"}, f"{label}.W3.sub_with_convert.const.type")
            log(f"  [{label}] W3 zero_point operand: Convert(Constant) -- PASS")
        else:
            require_type(zp_node, {"Constant"}, f"{label}.W3.sub_no_convert.type")
            log(f"  [{label}] W3 zero_point operand: Constant -- PASS")
    else:
        convert_node = a
        log(f"  [{label}] W3 Subtract (zero_point): absent (mul_no_sub branch)")

    # `convert` -- required Convert right after the weight Constant
    require_type(convert_node, {"Convert"}, f"{label}.W2.convert.type")
    log(f"  [{label}] W2 Convert after weight Constant: PASS ({convert_node.get_friendly_name()})")
    weight_const = convert_node.input_value(0).get_node()

    # `weights` -- the raw Constant, dtype in the compressed set
    require_type(weight_const, {"Constant"}, f"{label}.W1.weights.type")
    try:
        et = weight_const.get_element_type().to_string()
    except Exception:
        et = "?"
    if et not in COMPRESSED_TYPES:
        raise Fail(f"{label}.W1.weights.element_type", et, "one of " + "/".join(sorted(COMPRESSED_TYPES)))
    log(f"  [{label}] W1 weight Constant dtype={et}: PASS ({weight_const.get_friendly_name()})")
    return True


def check_3gemm_from_reduce_sum(reduce_sum_node, log):
    """Walk the ConvertTiledMoeBlockTo3GatherMatmuls pattern backward from a
    candidate ReduceSum root, in the same order build_3gemm_pattern()
    constructs it (which is also match order, since C++ builds edges from
    the leaves up but the Matcher walks from p.reduce_sum outward through
    those same edges)."""
    # R1: ReduceSum, keep_dims false direct, or keep_dims true + Squeeze
    attrs = attrs_of(reduce_sum_node)
    keep_dims = str(attrs.get("keep_dims", "false")).lower() == "true"
    require_consumers(reduce_sum_node.output(0), 1, "R1.reduce_sum.consumers_count")
    if keep_dims:
        raise Fail("R1.reduce_sum.keep_dims_true_needs_squeeze_next",
                  "not walked (script only follows the common keep_dims=false path)", "n/a")
    log(f"R1 ReduceSum(keep_dims=false), consumers=1: PASS ({reduce_sum_node.get_friendly_name()})")

    mul3 = reduce_sum_node.input_value(0).get_node()
    require_type(mul3, {"Multiply"}, "R2.mul3.type")
    log(f"R2 mul3 = Multiply: PASS ({mul3.get_friendly_name()})")

    end_reshape, router_side = resolve_mul3_operands(mul3)

    # R3: optional Unsqueeze
    if type_of(router_side) == "Unsqueeze":
        log(f"R3 optional_unsqueeze: PRESENT ({router_side.get_friendly_name()})")
        router_reshape = router_side.input_value(0).get_node()
    else:
        log("R3 optional_unsqueeze: absent")
        router_reshape = router_side
    require_type(router_reshape, {"Reshape"}, "R4.router_reshape.type")
    log(f"R4 router_reshape = Reshape: PASS ({router_reshape.get_friendly_name()})")

    router_transpose = router_reshape.input_value(0).get_node()
    require_type(router_transpose, {"Transpose"}, "R5.router_transpose.type")
    log(f"R5 router_transpose = Transpose: PASS ({router_transpose.get_friendly_name()})")

    scatter = router_transpose.input_value(0).get_node()
    require_type(scatter, {"ScatterElementsUpdate"}, "R6.scatter_elements_update.type")
    log(f"R6 scatter_elements_update: PASS ({scatter.get_friendly_name()})")

    # E1: end_reshape
    require_type(end_reshape, {"Reshape"}, "E1.end_reshape.type")
    require_consumers(end_reshape.output(0), 1, "E1.end_reshape.consumers_count")
    end_ps = end_reshape.output(0).get_partial_shape()
    log(f"E1 end_reshape = Reshape, consumers=1, out_shape={end_ps}, "
        f"out_rank_static={bool(end_ps.rank.is_static)}: PASS ({end_reshape.get_friendly_name()})")

    down_matmul = end_reshape.input_value(0).get_node()
    require_type(down_matmul, {"MatMul"}, "E2.down_matmul.type")
    require_consumers(down_matmul.output(0), 1, "E2.down_matmul.consumers_count")
    require_attrs(down_matmul, {"transpose_a": False, "transpose_b": True}, "E2.down_matmul.attrs")
    log(f"E2 down_matmul: PASS ({down_matmul.get_friendly_name()})")
    check_compressed_weights_block(matmul_weight_input(down_matmul), "down_weight", log)

    swiglu = down_matmul.input_value(0).get_node()
    require_type(swiglu, {"Multiply"}, "E3.swiglu.type")
    require_consumers(swiglu.output(0), 1, "E3.swiglu.consumers_count")
    log(f"E3 swiglu = Multiply, consumers=1: PASS ({swiglu.get_friendly_name()})")

    swish, up_matmul = resolve_swiglu_operands(swiglu)
    require_type(swish, {"Swish", "Gelu"}, "E4.swish.type")
    require_consumers(swish.output(0), 1, "E4.swish.consumers_count")
    log(f"E4 swish/gelu, consumers=1: PASS ({swish.get_friendly_name()}, type={type_of(swish)})")

    require_type(up_matmul, {"MatMul"}, "E5.up_matmul.type")
    require_consumers(up_matmul.output(0), 1, "E5.up_matmul.consumers_count")
    require_attrs(up_matmul, {"transpose_a": False, "transpose_b": True}, "E5.up_matmul.attrs")
    log(f"E5 up_matmul: PASS ({up_matmul.get_friendly_name()})")
    check_compressed_weights_block(matmul_weight_input(up_matmul), "up_weight", log)

    gate_matmul = swish.input_value(0).get_node()
    require_type(gate_matmul, {"MatMul"}, "E6.gate_matmul.type")
    require_consumers(gate_matmul.output(0), 1, "E6.gate_matmul.consumers_count")
    require_attrs(gate_matmul, {"transpose_a": False, "transpose_b": True}, "E6.gate_matmul.attrs")
    log(f"E6 gate_matmul: PASS ({gate_matmul.get_friendly_name()})")
    check_compressed_weights_block(matmul_weight_input(gate_matmul), "gate_weight", log)

    after_tile_gate = gate_matmul.input_value(0).get_node()
    after_tile_up = up_matmul.input_value(0).get_node()
    if after_tile_gate.get_instance_id() != after_tile_up.get_instance_id():
        raise Fail("E7.after_tile_reshape.shared_by_gate_and_up",
                  f"gate reads {after_tile_gate.get_friendly_name()} ({type_of(after_tile_gate)}), "
                  f"up reads {after_tile_up.get_friendly_name()} ({type_of(after_tile_up)})",
                  "the SAME node feeding both gate_matmul and up_matmul")
    after_tile_reshape = after_tile_gate
    require_type(after_tile_reshape, {"Reshape"}, "E7.after_tile_reshape.type")
    require_consumers(after_tile_reshape.output(0), 2, "E7.after_tile_reshape.consumers_count")
    log(f"E7 after_tile_reshape = Reshape, consumers=2 (shared by gate+up): PASS "
        f"({after_tile_reshape.get_friendly_name()})")

    tile = after_tile_reshape.input_value(0).get_node()
    require_type(tile, {"Tile"}, "E8.tile.type")
    require_consumers(tile.output(0), 1, "E8.tile.consumers_count")
    log(f"E8 tile = Tile, consumers=1: PASS ({tile.get_friendly_name()})")

    experts_input = tile.input_value(0).get_node()
    log(f"E9 experts_input (any_input(), no constraint): {type_of(experts_input)} "
        f"({experts_input.get_friendly_name()})")

    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir", help="path to an OpenVINO IR .xml")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="only print the final PASS/FAIL summary per candidate, not every step")
    a = ap.parse_args()

    import openvino as ov
    model = ov.Core().read_model(a.ir)

    candidates = [o for o in model.get_ordered_ops() if type_of(o) == "ReduceSum"]
    if not candidates:
        print("no ReduceSum ops found in this IR -- nothing to check")
        sys.exit(1)

    print(f"{len(candidates)} ReduceSum candidate(s) found; walking each backward "
          f"against the ConvertTiledMoeBlockTo3GatherMatmuls + CompressedWeightsBlock "
          f"constraint list.\n")

    n_matched = 0
    first_failure_summary = None
    for i, rs in enumerate(candidates):
        lines = []
        log = (lambda s: None) if a.quiet else (lambda s: lines.append(s))
        print(f"=== candidate {i}: ReduceSum {rs.get_friendly_name()} ===")
        try:
            check_3gemm_from_reduce_sum(rs, log)
            for l in lines:
                print(l)
            print(f"--- candidate {i}: FULL MATCH (all constraints PASS) ---\n")
            n_matched += 1
        except Fail as f:
            for l in lines:
                print(l)
            print(f"--- candidate {i}: FAIL at {f.constraint}\n"
                  f"    observed: {f.observed}\n"
                  f"    expected: {f.expected} ---\n")
            if first_failure_summary is None:
                first_failure_summary = (i, f)

    print(f"SUMMARY: {n_matched}/{len(candidates)} ReduceSum candidate(s) fully matched "
          f"the 3GEMM+CompressedWeightsBlock constraint chain.")
    if first_failure_summary is not None:
        i, f = first_failure_summary
        print(f"First failure (candidate {i}): {f.constraint}\n"
              f"  observed: {f.observed}\n"
              f"  expected: {f.expected}")


if __name__ == "__main__":
    main()
