#!/usr/bin/env python3
"""Byte-level comparison of two OpenVINO IRs: every MatMul weight/scale/
zero-point constant, plus an op-type histogram diff of the two graphs.

Generalised from the one-off script that produced the 56/56 byte-identical
result recorded in docs/dflash-pairing-probe.md (2026-09-03) -- this is the
reproducible form of that check, not tied to the DFlash2 head or to any
particular directory layout.

Each MatMul's weight input is traced back through Convert/Reshape/
Transpose to its source Constant; a Multiply-by-scale and/or
Subtract-of-zero-point on that branch (the NNCF weight-compression pattern:
Constant -> Convert -> Subtract(zero_point) -> Multiply(scale) -> Reshape
-> Convert -> MatMul) is picked up as the scale/zero-point constants for
that weight, if present. A plain (uncompressed) weight has no scale or
zero-point and is compared as raw bytes only. Weights are matched between
the two graphs first by friendly name, then -- for the remainder, e.g.
compressed graphs where NNCF assigns most named weights a "_compressed"
suffix but leaves some anonymous -- by (shape, order-of-appearance) within
each shape group; anything left over is reported unresolved rather than
silently skipped.

When weight/zero-point are 4-bit (u4/i4), OpenVINO packs two elements per
byte (low nibble = even index, high nibble = odd index); this script
unpacks them before comparing, and reports a dequantised max-abs-diff
alongside the raw byte-equality result for anything that differs.

Usage:
    python3 tools/dflash_compare_ir.py IR_A.xml IR_B.xml [--json OUT.json]
    python3 tools/dflash_compare_ir.py --self-test

Exit status is 0 only for a clean, complete match: every weight in both
graphs was matched to a counterpart (n_a == n_b == n_pairs, nothing
unresolved), and every matched weight is byte-identical (0 differing, 0
errors). Anything short of that -- an unresolved weight, a count mismatch,
or a single differing constant -- is nonzero. Pass --lenient to relax
this to the old, permissive check (0 differing, 0 errors among whatever
got matched; unresolved and unmatched counts do not affect the exit code)
for exploratory comparisons where the two graphs are not expected to have
identical weight sets.
"""
import argparse
import json
import sys
from collections import Counter

import numpy as np
import openvino as ov


def trace_weight_branch(matmul):
    inp = matmul.input(1).get_source_output().get_node()
    cur = inp
    info = {"scale": None, "zero_point": None, "weight": None}
    visited = 0
    while cur is not None and visited < 12:
        visited += 1
        tname = cur.get_type_name()
        if tname == "Constant":
            info["weight"] = cur
            return info
        elif tname in ("Convert", "Reshape", "Transpose"):
            cur = cur.input(0).get_source_output().get_node()
        elif tname == "Multiply":
            n0 = cur.input(0).get_source_output().get_node()
            n1 = cur.input(1).get_source_output().get_node()
            # Exactly one of the two inputs continues the weight path (it
            # leads on, eventually, to the weight Constant); the other IS
            # the scale, directly a Constant or a Convert(Constant) one hop
            # away. Take the scale from the FIRST candidate that matches
            # that shape and stop there -- do not also test the other
            # candidate. A symmetric-compression graph (no Subtract stage)
            # is Multiply(Convert(Constant weight), Constant scale): here
            # BOTH candidates match "Constant, or Convert(Constant)" --
            # the scale directly, and the weight-path Convert one hop from
            # its own Constant -- so scanning both and keeping the last
            # match (the pre-2026-09-03 bug, found in review) silently
            # replaces the real scale with the weight itself. Scanning n1
            # before n0 and stopping at the first hit keeps the original
            # preference order for the common (asymmetric, Subtract-first)
            # case while no longer touching the second candidate once a
            # scale has been found.
            scale_node, weight_path = None, n0
            for cand, other in ((n1, n0), (n0, n1)):
                if cand.get_type_name() == "Constant":
                    scale_node, weight_path = cand, other
                    break
                elif cand.get_type_name() == "Convert":
                    src = cand.input(0).get_source_output().get_node()
                    if src.get_type_name() == "Constant":
                        scale_node, weight_path = src, other
                        break
            if scale_node is not None:
                info["scale"] = scale_node
            cur = weight_path
        elif tname == "Subtract":
            n0 = cur.input(0).get_source_output().get_node()
            n1 = cur.input(1).get_source_output().get_node()
            zcur = n1
            depth = 0
            while zcur is not None and depth < 5:
                if zcur.get_type_name() == "Constant":
                    info["zero_point"] = zcur
                    break
                elif zcur.get_type_name() == "Convert":
                    zcur = zcur.input(0).get_source_output().get_node()
                    depth += 1
                else:
                    break
            cur = n0
        else:
            return info
    return info


def _build_symmetric_matmul(weight_vals, scale_vals, name):
    """Constant(i8 weight) -> Convert(f32) -> Multiply(Constant f32 scale)
    -> Reshape -> Convert -> MatMul, built with op.multiply(weight_branch,
    scale_const) so the scale lands in Multiply's SECOND input and the
    weight-path Convert lands in the FIRST -- the exact ordering F2
    (2026-09-03 review) found trace_weight_branch corrupting: the old code
    scanned both Multiply inputs unconditionally and let the second
    candidate (the weight-path Convert(Constant)) overwrite an
    already-correct scale with the weight constant itself."""
    from openvino import opset13 as op

    out_dim, in_dim = weight_vals.shape
    x = op.parameter(ov.PartialShape([1, in_dim]), ov.Type.f32, name=f"{name}_x")
    w_const = op.constant(weight_vals.astype(np.int8))
    w_const.set_friendly_name(f"{name}_weight")
    w_f32 = op.convert(w_const, ov.Type.f32)
    scale_const = op.constant(scale_vals.astype(np.float32).reshape(out_dim, 1))
    scale_const.set_friendly_name(f"{name}_scale")
    dequant = op.multiply(w_f32, scale_const)  # weight-path first, scale second
    reshaped = op.reshape(dequant, op.constant(np.array([out_dim, in_dim], dtype=np.int32)),
                          special_zero=False)
    reshaped = op.convert(reshaped, ov.Type.f32)
    result = op.matmul(x, reshaped, transpose_a=False, transpose_b=True)
    res = op.result(result)
    return ov.Model([res], [x], name)


def self_test():
    """Symmetric-quantization regression test for the F2 (2026-09-03) fix:
    build two tiny synthetic IRs with the SAME weight but DIFFERENT scale,
    and assert (1) trace_weight_branch resolves the scale to the scale
    constant, not the weight constant, and (2) compare() reports the pair
    as differing -- i.e. a real scale difference cannot pass silently by
    being compared against itself. Returns True/prints and raises
    AssertionError on failure; no IR files or --dir needed."""
    w = np.array([[1, -2, 3, -4], [5, -6, 7, -8]], dtype=np.int8)
    model_a = _build_symmetric_matmul(w, np.array([0.5, 0.5]), "sym_a")
    model_b = _build_symmetric_matmul(w, np.array([0.5, 0.75]), "sym_b")  # scale differs

    matmul_a = next(n for n in model_a.get_ordered_ops() if n.get_type_name() == "MatMul")
    info_a = trace_weight_branch(matmul_a)
    assert info_a["weight"] is not None, "self-test: weight not resolved"
    assert info_a["scale"] is not None, "self-test: scale not resolved"
    assert info_a["scale"].get_friendly_name() == "sym_a_scale", (
        f"self-test: scale resolved to {info_a['scale'].get_friendly_name()!r}, "
        f"expected 'sym_a_scale' -- the weight-path Convert(Constant) is "
        f"overwriting the scale again (F2 regression)")
    assert np.array_equal(info_a["scale"].get_data(), np.array([0.5, 0.5], dtype=np.float32).reshape(2, 1)), (
        "self-test: scale resolved to the wrong data -- looks like the weight "
        "constant, not the scale (F2 regression)")

    out = compare(model_a, model_b, "a", "b")
    assert out["n_diff"] == 1 and out["n_identical"] == 0, (
        f"self-test: a real scale difference did not surface as a diff "
        f"(n_diff={out['n_diff']}, n_identical={out['n_identical']}) -- "
        f"if the scale had been resolved to the (identical) weight instead "
        f"of the (differing) scale, this would pass silently")

    print("self-test PASSED: symmetric-compression scale resolved correctly, "
          "differing scale detected (F2, 2026-09-03)")
    return True


def collect_matmul_weights(model):
    out = []
    for idx, op_ in enumerate(model.get_ordered_ops()):
        if op_.get_type_name() == "MatMul":
            info = trace_weight_branch(op_)
            if info["weight"] is None:
                continue
            w = info["weight"]
            out.append({
                "order": idx,
                "matmul_name": op_.get_friendly_name(),
                "weight_name": w.get_friendly_name(),
                "shape": tuple(w.get_output_shape(0)),
                "weight_node": w,
                "scale_node": info["scale"],
                "zp_node": info["zero_point"],
            })
    return out


def is_4bit(node):
    return node.get_element_type().get_type_name() in ("u4", "i4")


def unpack_4bit(packed_bytes, n_elems):
    # OpenVINO 4-bit packing convention: low nibble = even-index element,
    # high nibble = odd-index element.
    b = packed_bytes.astype(np.uint16)
    lo = (b & 0x0F).astype(np.uint8)
    hi = ((b >> 4) & 0x0F).astype(np.uint8)
    out = np.empty(n_elems, dtype=np.uint8)
    out[0::2] = lo[: (n_elems + 1) // 2]
    out[1::2] = hi[: n_elems // 2]
    return out


def get_array(node, n_elems=None, shape=None):
    """Constant -> numpy array, unpacking 4-bit weights/zero-points."""
    data = node.get_data()
    if is_4bit(node):
        assert n_elems is not None and shape is not None
        return unpack_4bit(data, n_elems).astype(np.float32).reshape(shape)
    return data


def dequant_max_abs_diff(a, b, shape):
    """a, b: dicts with weight_node/scale_node/zp_node. Returns
    (max_abs_diff, max_abs_a, max_abs_b) of the dequantised tensors, or
    None if either side has no scale (nothing to dequantise against)."""
    if a["scale_node"] is None or b["scale_node"] is None:
        return None
    n = int(np.prod(shape))
    wa = get_array(a["weight_node"], n, shape)
    wb = get_array(b["weight_node"], n, shape)
    zp_shape = shape[:-1] + (1,)
    if a["zp_node"] is not None and b["zp_node"] is not None:
        nz = int(np.prod(zp_shape))
        zpa = get_array(a["zp_node"], nz, zp_shape)
        zpb = get_array(b["zp_node"], nz, zp_shape)
    else:
        zpa = zpb = 0.0
    sca = np.asarray(a["scale_node"].get_data(), dtype=np.float32).reshape(zp_shape)
    scb = np.asarray(b["scale_node"].get_data(), dtype=np.float32).reshape(zp_shape)
    da = (wa - zpa) * sca
    db = (wb - zpb) * scb
    return float(np.max(np.abs(da - db))), float(np.max(np.abs(da))), float(np.max(np.abs(db)))


def bytes_equal(node_a, node_b):
    if node_a is None or node_b is None:
        return None
    return bool(np.array_equal(node_a.get_data(), node_b.get_data()))


def compare(model_a, model_b, label_a="a", label_b="b"):
    list_a = collect_matmul_weights(model_a)
    list_b = collect_matmul_weights(model_b)

    by_name_a = {e["weight_name"]: e for e in list_a}
    by_name_b = {e["weight_name"]: e for e in list_b}

    matched_by_name = sorted(set(by_name_a) & set(by_name_b))
    unmatched_a = [e for e in list_a if e["weight_name"] not in by_name_b]
    unmatched_b = [e for e in list_b if e["weight_name"] not in by_name_a]

    pairs = [(by_name_a[n], by_name_b[n]) for n in matched_by_name]

    def group_by_shape(lst):
        g = {}
        for e in sorted(lst, key=lambda x: x["order"]):
            g.setdefault(e["shape"], []).append(e)
        return g

    ga = group_by_shape(unmatched_a)
    gb = group_by_shape(unmatched_b)
    order_matched = 0
    unresolved = []
    for shape, slist in ga.items():
        blist = gb.get(shape, [])
        if len(blist) != len(slist):
            unresolved.extend(slist)
            continue
        for x, y in zip(slist, blist):
            pairs.append((x, y))
            order_matched += 1
    for shape, blist in gb.items():
        if shape not in ga:
            unresolved.extend(blist)

    results = []
    n_identical = n_diff = n_error = 0
    for a, b in pairs:
        rec = {f"{label_a}_name": a["weight_name"], f"{label_b}_name": b["weight_name"],
               "shape": list(a["shape"])}
        try:
            w_eq = bytes_equal(a["weight_node"], b["weight_node"])
            sc_eq = bytes_equal(a["scale_node"], b["scale_node"])
            zp_eq = bytes_equal(a["zp_node"], b["zp_node"])
            rec["weight_bytes_equal"] = w_eq
            rec["scale_bytes_equal"] = sc_eq
            rec["zero_point_bytes_equal"] = zp_eq

            all_eq = bool(w_eq) and (sc_eq is None or sc_eq) and (zp_eq is None or zp_eq)
            rec["all_equal"] = all_eq
            if all_eq:
                n_identical += 1
            else:
                n_diff += 1
                diff = dequant_max_abs_diff(a, b, a["shape"])
                if diff is not None:
                    rec["dequant_max_abs_diff"], rec["dequant_max_abs_a"], rec["dequant_max_abs_b"] = diff
        except Exception as e:  # noqa: BLE001 -- surfaced in the report, not swallowed
            n_error += 1
            rec["error"] = str(e)
        results.append(rec)

    hist_a = Counter(op_.get_type_name() for op_ in model_a.get_ordered_ops())
    hist_b = Counter(op_.get_type_name() for op_ in model_b.get_ordered_ops())
    all_types = sorted(set(hist_a) | set(hist_b))
    struct_diff = {}
    for t in all_types:
        na, nb = hist_a.get(t, 0), hist_b.get(t, 0)
        if na != nb:
            struct_diff[t] = {label_a: na, label_b: nb, "delta": nb - na}

    return {
        "n_a": len(list_a),
        "n_b": len(list_b),
        "n_matched_by_name": len(matched_by_name),
        "n_matched_by_shape_order": order_matched,
        "n_unresolved": len(unresolved),
        "unresolved": [{"weight_name": e["weight_name"], "shape": list(e["shape"])} for e in unresolved],
        "n_pairs": len(pairs),
        "n_identical": n_identical,
        "n_diff": n_diff,
        "n_error": n_error,
        "struct_diff": struct_diff,
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir_a", nargs="?", help="path to the first IR's .xml")
    ap.add_argument("ir_b", nargs="?", help="path to the second IR's .xml")
    ap.add_argument("--label-a", default="a", help="label for ir_a in the report (default: a)")
    ap.add_argument("--label-b", default="b", help="label for ir_b in the report (default: b)")
    ap.add_argument("--json", default=None, help="write the full per-weight report to this path")
    ap.add_argument("--lenient", action="store_true",
                     help="relax the exit code to the old, permissive check: 0 differing and "
                          "0 errors among whatever got matched, ignoring unresolved/unmatched "
                          "weights and any n_a != n_b count mismatch (default: strict -- exit "
                          "0 only if every weight in both graphs matched and was identical)")
    ap.add_argument("--self-test", action="store_true",
                     help="run the built-in symmetric-compression regression test (F2, "
                          "2026-09-03) and exit; ir_a/ir_b are not needed")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        sys.exit(0)
    if not args.ir_a or not args.ir_b:
        ap.error("ir_a and ir_b are required unless --self-test is given")

    core = ov.Core()
    model_a = core.read_model(args.ir_a)
    model_b = core.read_model(args.ir_b)

    out = compare(model_a, model_b, args.label_a, args.label_b)

    print(f"{args.label_a} ({args.ir_a}): {out['n_a']} MatMul-with-weight ops", file=sys.stderr)
    print(f"{args.label_b} ({args.ir_b}): {out['n_b']} MatMul-with-weight ops", file=sys.stderr)
    print(f"matched by exact name: {out['n_matched_by_name']}", file=sys.stderr)
    print(f"matched by shape+order (residual, unnamed constants): {out['n_matched_by_shape_order']}",
          file=sys.stderr)
    print(f"unresolved (no counterpart found): {out['n_unresolved']}", file=sys.stderr)
    for e in out["unresolved"]:
        print("  UNRESOLVED:", e["weight_name"], e["shape"], file=sys.stderr)

    print("\n=== SUMMARY ===", file=sys.stderr)
    print(f"total matched weight constants: {out['n_pairs']}", file=sys.stderr)
    print(f"byte-identical (weight+scale+zp): {out['n_identical']}", file=sys.stderr)
    print(f"differing: {out['n_diff']}", file=sys.stderr)
    print(f"errors: {out['n_error']}", file=sys.stderr)

    print("\n=== STRUCTURAL DIFF (op-type histogram) ===", file=sys.stderr)
    print(json.dumps(out["struct_diff"], indent=1), file=sys.stderr)

    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nfull per-weight report written to {args.json}", file=sys.stderr)

    if args.lenient:
        ok = out["n_diff"] == 0 and out["n_error"] == 0
    else:
        ok = (out["n_diff"] == 0 and out["n_error"] == 0 and out["n_unresolved"] == 0
              and out["n_a"] == out["n_b"] == out["n_pairs"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
