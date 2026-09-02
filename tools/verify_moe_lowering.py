#!/usr/bin/env python3
"""Check whether an MTP MoE IR is in a shape OpenVINO's GPU plugin can fuse,
and whether it actually fired.

Two independent checks, because a graph can be shaped right and still not get
fused (opset mismatch, an extra op the matcher does not expect, a pass that is
missing from this build), and the reverse is not informative either -- a
"batched" export was never meant to match.

  (a) structural scan: walks the *source* graph (before compilation) and
      classifies it as one of the three export_mtp.py emission paths by
      counting the ops that distinguish them:

        batched  (moe_block(), the default) -- a single, dense
                 ScatterElementsUpdate(reduction="none") builds the per-token
                 routing weights; no NonZero, no OneHot, no Tile, no rank-3
                 compressed weight constants.

        unrolled (moe_block_unrolled(), --moe-lowering unrolled) -- targets
                 ov::pass::FuseMOE's per-expert-gather rewrite: one OneHot
                 plus, per expert, one NonZero and one
                 ScatterElementsUpdate(reduction="sum").

        tiled    (moe_block_tiled(), --moe-lowering tiled) -- targets the
                 GPU plugin's ConvertTiledMoeBlockToGatherMatmuls pass instead
                 (this is the one measured actually fusing on the card): a
                 Reshape->Tile->Reshape entry, expert weights as rank-4
                 [E,out,groups,group_size] Constants in a compressed integer
                 type (u4/i4/u8/i8) behind a
                 Convert->Subtract(zero_point)->Multiply(scale)->Reshape(rank
                 4->3) decompression chain -- the trailing Reshape matters:
                 an earlier version of moe_block_tiled() stored the weight
                 flat at rank 3 with no groups dimension and no Reshape, and
                 that shape passed every check in this file (and CPU
                 compiles) but crashed a real GPU compile inside the fusing
                 pass's own rewrite, because the pass's matcher anchors on
                 that Reshape node -- and a router that is structurally the
                 same as the batched form's (single
                 ScatterElementsUpdate(reduction="none"), no OneHot, no
                 NonZero) -- so tiled is distinguished from batched only by
                 the Tile op and the compressed weights, not by the router
                 shape.

  (b) fusion check: compiles the model for CPU only (never GPU -- this script
      must be safe to run on a machine with no GPU driver at all) and walks
      compiled_model.get_runtime_model() for any op whose type name or
      friendly name contains "MOE". Requires openvino to be importable; skips
      itself cleanly if not, so it can run before or after (a) without
      needing a GPU-carrying host. ConvertTiledMoeBlockToGatherMatmuls is a
      GPU-plugin pass, so "not fused" on CPU is the expected, honest result
      for the tiled mode too -- this check only tells you whether *some*
      fusion happened on whatever device compiled it, CPU included.

Usage:
    python3 verify_moe_lowering.py path/to/openvino_mtp_layer.xml
    python3 verify_moe_lowering.py path/to/openvino_mtp_layer.xml --no-compile
"""
import argparse
import sys


_COMPRESSED_TYPES = ("u8", "u4", "i8", "i4")


def structural_scan(model):
    """Classify a source ov.Model as 'batched', 'unrolled', 'tiled', or
    'unknown' by counting the ops each emission path is expected to leave
    behind. Returns a dict with the counts and the classification, never
    raises on a model that matches none of the three shapes -- that is
    reported as 'unknown', not an error."""
    ops = model.get_ordered_ops()
    type_counts = {}
    for o in ops:
        t = o.get_type_name()
        type_counts[t] = type_counts.get(t, 0) + 1

    n_nonzero = type_counts.get("NonZero", 0)
    n_onehot = type_counts.get("OneHot", 0)
    n_tile = type_counts.get("Tile", 0)
    n_scatter_total = type_counts.get("ScatterElementsUpdate", 0)
    n_scatter_sum = 0
    for o in ops:
        if o.get_type_name() != "ScatterElementsUpdate":
            continue
        try:
            reduction = o.get_attributes().get("reduction", "none")
        except Exception:
            reduction = "none"
        if str(reduction).lower() == "sum":
            n_scatter_sum += 1

    # per-expert weight constants named "expert{i}_gate_proj" etc. by
    # moe_block_unrolled() -- present only in the unrolled emission.
    n_expert_constants = sum(
        1 for o in ops
        if o.get_type_name() == "Constant" and o.get_friendly_name().startswith("expert")
        and "_proj" in o.get_friendly_name()
    )

    # weight Constants in a compressed integer type, rank 3 (a flat
    # [E,out,in] quantization) or rank 4 (grouped, [E,out,groups,
    # group_size] -- what production and the current moe_block_tiled() both
    # emit): CompressedWeightsBlock, the dtype ConvertTiledMoeBlockToGather-
    # Matmuls' matcher requires; a plain f32/f16 Constant never counts here,
    # which is exactly why the batched and unrolled emissions never trip
    # this.
    n_compressed_weights = 0
    for o in ops:
        if o.get_type_name() != "Constant":
            continue
        try:
            et = o.get_element_type().to_string()
            rank = len(o.get_shape())
        except Exception:
            continue
        if et in _COMPRESSED_TYPES and rank in (3, 4):
            n_compressed_weights += 1

    # the Reshape that collapses a rank-4 grouped-quantization dequant
    # straight down to rank 3, immediately downstream of the dequant
    # Multiply -- the specific node the fusing pass's matcher anchors on and
    # whose absence (an earlier, rank-3-throughout version of
    # compressed_weight()) crashed a real GPU compile even though it looked
    # fine to every check that does not require rank-4 storage.
    n_compressed_weight_reshapes = 0
    for o in ops:
        if o.get_type_name() != "Reshape":
            continue
        src = o.input_value(0).get_node()
        if src.get_type_name() != "Multiply":
            continue
        if len(o.input_value(0).get_partial_shape()) == 4 and len(o.output(0).get_partial_shape()) == 3:
            n_compressed_weight_reshapes += 1

    is_unrolled = n_nonzero > 0 and n_onehot > 0 and n_scatter_sum > 0
    is_tiled = n_tile > 0 and n_compressed_weights > 0 and n_nonzero == 0
    is_batched = (n_nonzero == 0 and n_onehot == 0 and n_tile == 0
                 and n_compressed_weights == 0 and n_scatter_total > 0 and n_scatter_sum == 0)

    if is_unrolled:
        classification = "unrolled"
    elif is_tiled:
        classification = "tiled"
    elif is_batched:
        classification = "batched"
    else:
        classification = "unknown"

    return {
        "nonzero": n_nonzero,
        "onehot": n_onehot,
        "tile": n_tile,
        "compressed_weights": n_compressed_weights,
        "compressed_weight_reshapes": n_compressed_weight_reshapes,
        "scatter_total": n_scatter_total,
        "scatter_sum": n_scatter_sum,
        "expert_weight_constants": n_expert_constants,
        "classification": classification,
    }


def fusion_check(model, device="CPU"):
    """Compile `model` for CPU only and look for MOE ops in the runtime
    graph. Never pass anything but 'CPU' as device -- this script has no
    business touching a GPU, and the assert below is not decorative."""
    assert device == "CPU", "verify_moe_lowering only ever compiles for CPU"
    import openvino as ov

    core = ov.Core()
    compiled = core.compile_model(model, device)
    runtime = compiled.get_runtime_model()
    moe_ops = []
    for o in runtime.get_ordered_ops():
        t = o.get_type_name()
        fname = o.get_friendly_name()
        if "moe" in t.lower() or "moe" in fname.lower():
            moe_ops.append((t, fname))
    return {"fused": len(moe_ops) > 0, "moe_ops": moe_ops}


def _load(path):
    import openvino as ov
    return ov.Core().read_model(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir", help="path to an OpenVINO IR .xml (e.g. openvino_mtp_layer.xml)")
    ap.add_argument("--no-compile", action="store_true",
                    help="skip the fusion check (structural scan only, no openvino import needed "
                    "beyond reading the IR)")
    a = ap.parse_args()

    try:
        import openvino  # noqa: F401
    except ImportError:
        print("openvino not importable -- cannot load or scan the IR", file=sys.stderr)
        sys.exit(1)

    model = _load(a.ir)

    scan = structural_scan(model)
    print(f"structural scan: classification={scan['classification']}")
    print(f"  NonZero={scan['nonzero']}  OneHot={scan['onehot']}  Tile={scan['tile']}  "
          f"ScatterElementsUpdate total={scan['scatter_total']} sum={scan['scatter_sum']}  "
          f"expert-weight constants={scan['expert_weight_constants']}  "
          f"compressed weights={scan['compressed_weights']}  "
          f"compressed-weight collapsing reshapes={scan['compressed_weight_reshapes']}")

    if a.no_compile:
        print("fusion check: skipped (--no-compile)")
        return

    fused = fusion_check(model, device="CPU")
    if fused["fused"]:
        print(f"fusion check (CPU): FUSED -- {len(fused['moe_ops'])} MOE op(s) in the runtime graph")
        for t, fname in fused["moe_ops"]:
            print(f"    {t}  {fname}")
    else:
        print("fusion check (CPU): not fused -- no MOE op found in compiled_model.get_runtime_model()")


if __name__ == "__main__":
    main()
