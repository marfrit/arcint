#!/usr/bin/env python3
"""FRONTIER EXPERIMENT, PREPARED ON CPU: does the GPU plugin keep the filled u4
expert bodies COMPRESSED, and on which kernel?

WHY THIS EXISTS. `q4e.expert_fill` puts real Q3_K_XL rows into the
serving-shape IR's u4 expert bodies, and the entire residency story rests on
those weights STAYING u4 once the card compiles the graph. If the plugin
decompresses them to f16 at compile time the declared 4-bit slice becomes a
16-bit one and every figure in the size ledger is out by 4x. Nothing on the
CPU side can answer that: it is a GPU-plugin pass-selection question, and this
session spends no card. So the experiment is BUILT and CHARACTERISED here, and
the window runs it.

THE QUESTION IS SHARPER THAN "DOES IT WORK", because this repository already
measured the failure mode next door. `docs/prefill-baseline.md` §M2: at a
2048-token prefill, 40 of 371 `FullyConnectedCompressed` nodes fall off
`jit:gemm:any__i8` onto `ocl:ref:any__i8` and consume a THIRD of the chunk --
68.3 ms against 38.9 ms for the other 331 -- and the 40 are
`mlp.shared_expert_gate`, the one projection whose output width is 1. At M=1
and M=2 all 371 are on the jit kernel. So the plugin's compressed path is real,
it is shape-sensitive, and it has a reference-kernel cliff.

That evidence is all at i8. The expert bodies are u4, grouped, and rank-4
before the collapsing Reshape. Two things are therefore unknown and both
matter:

  (Q1) SELECTION. Does a compressed primitive appear in the runtime graph at
       all for this shape -- FullyConnectedCompressed, MOECompressed, or the
       3GEMM MoE fusion -- or does the graph arrive as plain MatMuls over
       decompressed f16 weights?
  (Q2) IMPLEMENTATION. If it is selected, is it the jit kernel or the
       `ocl:ref` fallback that already costs a third of a prefill chunk at i8?

--------------------------------------------------------------------------
THE CANDIDATES, and why each is in the list
--------------------------------------------------------------------------

  as_emitted      What `serving_shape._compressed_expert` emits TODAY: rank-4
                  [E, out, groups, gs] u4 -> Convert(f32) -> Subtract(u4
                  per-group zp) -> Multiply(f32 per-group scale) ->
                  Reshape(4->3), into a rank-3 MatMul(transpose_b=True).
                  This is the thing under test.

  f16_scale       Identical but the scale is f16. `src/exec/gguf_graph.cpp:229`
                  builds its scale as an f16 Constant, so production's
                  compressed weights differ from ours here, and scale
                  precision is a plausible selector input.

  u8_scalar_zp    Identical but the zero-point is a SCALAR u8 [1,1,1] rather
                  than a per-group u4. `gguf_graph.cpp:225-228`
                  (`RepackZeroPoint::U8Scalar`) is exactly that shape -- again
                  a difference between what arcint already serves on the cards
                  and what the serving-shape IR emits.

  production_2d   THE POSITIVE CONTROL, and the reason this experiment can
                  produce a usable answer rather than a shrug: the rank-3
                  [n, groups, gs] u4 -> Convert(f16) -> Subtract(u8 scalar) ->
                  Multiply(f16 [n,groups,1]) -> Reshape -> rank-2 MatMul that
                  `gguf_graph.cpp:216-235` builds for the dense models arcint
                  SERVES ON THESE CARDS TODAY. If this one does not select a
                  compressed primitive on the window's card, the harness is
                  wrong and no verdict about the others is worth anything.

  no_reshape      THE NEGATIVE CONTROL. The same chain with the trailing
                  rank-4->3 Reshape removed. `verify_moe_lowering.py:33-42`
                  records a real GPU compile CRASHING inside the fusing pass's
                  own rewrite without it. Run it last, expect a failure, and
                  if it passes then the pass being matched is not the one that
                  record is about.

--------------------------------------------------------------------------
WHAT THE WINDOW RUNS, IN ORDER
--------------------------------------------------------------------------

  1.  python3 tools/repro_fc_compressed_selection.py --device CPU --save /tmp/fc
      Confirms the hashes below still describe the tree being run, and that
      every candidate builds and compiles. It does NOT answer Q1 or Q2 and
      does not pretend to: `FullyConnectedCompressed` and `MOECompressed` are
      GPU-plugin primitives and the CPU plugin neither names nor runs them.
      If a hash moved, the IR moved, and the characterisation has to be
      redone before any GPU number means anything.

  2.  ... --device GPU.1 --only production_2d
      The positive control, on the A770 (the coder's card). Expect a
      compressed primitive. If absent -> STOP, the detector is wrong.

  3.  ... --device GPU.1 --only as_emitted
      THE ANSWER TO Q1 AND Q2, and the row the manifest is waiting for.

  4.  ... --device GPU.1 --only f16_scale,u8_scalar_zp
      Only if step 3 says "not compressed": these isolate which single
      attribute -- scale precision or zero-point shape -- differs from the
      production shape that works.

  5.  ... --device GPU.1 --only no_reshape
      Expect a crash. Run it LAST because it may take the process down.

This script never touches a card on its own: `--device` defaults to CPU and
the GPU values are typed by a window operator. It imports openvino and numpy
and nothing else -- no torch, no transformers, no shards -- so it runs in a
window with no setup.

Usage:
    python3 tools/repro_fc_compressed_selection.py [--device CPU|GPU.0|GPU.1]
        [--save DIR] [--only name[,name...]] [--experts N] [--tokens N]
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Real expert geometry (piecewise_export.REAL_GEOMETRY): E=512 I=640 H=2560.
# The default here is a REDUCED expert count, because the question is which
# primitive the plugin selects for a shape, and selection does not depend on
# how many experts are stacked -- while 512 of them at real width is 1.26 GiB
# per body and would make the experiment cost a card-hour instead of a minute.
# `--experts 512` runs it at full width when the window wants that.
DEFAULT_EXPERTS = 8
DEFAULT_TOKENS = 64
REAL_I = 640
REAL_H = 2560
GROUP_SIZE = 128

# Runtime-graph type names that mean "the weights stayed compressed". Matched
# case-insensitively as substrings, the same way verify_moe_lowering.py's
# fusion_check looks for MoE ops.
COMPRESSED_MARKERS = ("compressed", "moe")

# An `ov.Tensor(numpy_buffer, shape, Type.u4)` WRAPS the buffer; it does not
# own it. Letting the numpy array go out of scope leaves the Constant pointing
# at freed memory and the process dies on SIGSEGV the moment anything reads
# it -- which is what the first run of this file did, inside `save_model`.
# `SparseArena` solves the same problem with its `hold` list; this is that
# list.
_KEEP_ALIVE = []


def _u4_const(shape, seed):
    """A u4 Constant of `shape` over pseudo-random codes.

    VALUES ARE IRRELEVANT to pass selection and are seeded only so the IR
    hashes below are reproducible. Packing is arcint's own contract -- two per
    byte, even index low nibble (src/core/gguf_repack.h:90), verified against
    OpenVINO's reader by
    tests/python/test_expert_fill.py::test_openvino_u4_element_order_is_the_cpp_contract.
    """
    import openvino as ov
    from openvino import opset13 as op

    n = int(np.prod(shape))
    q = np.random.default_rng(seed).integers(0, 16, size=n, dtype=np.uint8)
    packed = np.ascontiguousarray((q[0::2] | (q[1::2] << 4)).astype(np.uint8))
    _KEEP_ALIVE.append(packed)          # the Tensor wraps it, it does not own it
    return op.constant(ov.Tensor(packed, ov.Shape([int(d) for d in shape]),
                                 ov.Type.u4))


def _candidates(E, I, H, M):
    """name -> (build a Model). Every entry is one VARIABLE away from
    `as_emitted` except `production_2d`, which is a different shape on
    purpose."""
    import openvino as ov
    from openvino import opset13 as op

    gs = GROUP_SIZE
    groups = H // gs

    def chain(scale_type, zp_scalar, reshape):
        """The tiled compressed-weight chain, parameterised on the two
        attributes that differ from production."""
        w = _u4_const([E, I, groups, gs], 1)
        x = op.convert(w, ov.Type.f32)
        if zp_scalar:
            zp = op.constant(np.array([[[[8]]]], np.uint8))
            zp = op.convert(zp, ov.Type.f32)
        else:
            zp = op.convert(_u4_const([E, I, groups, 1], 2), ov.Type.f32)
        x = op.subtract(x, zp)
        sc = np.full((E, I, groups, 1), 0.01,
                     np.float16 if scale_type == ov.Type.f16 else np.float32)
        s = op.constant(sc)
        if scale_type == ov.Type.f16:
            s = op.convert(s, ov.Type.f32)
        x = op.multiply(x, s)
        if reshape:
            x = op.reshape(x, op.constant(np.array([E, I, H], np.int64)),
                           special_zero=False)
            x.set_friendly_name("dequant_reshape")
        return x

    def flat_chain():
        """The NEGATIVE CONTROL, and it is NOT `tiled` with the Reshape
        deleted -- that does not even build, because a rank-4 [E,out,groups,gs]
        weight cannot MatMul a rank-3 activation (the first run of this file
        got `Incompatible MatMul matrix dimension ... 2560 ... 128` and that is
        a shape error, not the defect).

        What `verify_moe_lowering.py:33-42` records is a FLAT RANK-3 weight
        with no groups dimension and no Reshape: shape-valid, "passed every
        check in this file (and CPU compiles)", and crashed a real GPU compile
        inside the fusing pass's own rewrite "because the pass's matcher
        anchors on that Reshape node". That is what this builds."""
        w = _u4_const([E, I, H], 4)
        x = op.convert(w, ov.Type.f32)
        x = op.subtract(x, op.convert(_u4_const([E, I, 1], 5), ov.Type.f32))
        return op.multiply(x, op.constant(np.full((E, I, 1), 0.01, np.float32)))

    def tiled(scale_type=None, zp_scalar=False, reshape=True):
        st = scale_type or ov.Type.f32
        act = op.parameter([E, M, H], ov.Type.f32)
        act.set_friendly_name("x")
        wt = chain(st, zp_scalar, True) if reshape else flat_chain()
        y = op.matmul(act, wt, transpose_a=False, transpose_b=True)
        r = op.result(y)
        r.set_friendly_name("y")
        return ov.Model([r], [act], "tiled")

    def production_2d():
        """gguf_graph.cpp:216-235, the shape arcint serves on these cards."""
        n, width = I, H
        g = width // gs
        act = op.parameter([M, width], ov.Type.f32)
        act.set_friendly_name("x")
        w = _u4_const([n, g, gs], 3)
        x = op.convert(w, ov.Type.f16)
        zp = op.constant(np.array([[[8]]], np.uint8))
        x = op.subtract(x, op.convert(zp, ov.Type.f16))
        x = op.multiply(x, op.constant(np.full((n, g, 1), 0.01, np.float16)))
        x = op.reshape(x, op.constant(np.array([n, width], np.int64)),
                       special_zero=False)
        x.set_friendly_name("dequant_reshape")
        x = op.convert(x, ov.Type.f32)
        y = op.matmul(act, x, transpose_a=False, transpose_b=True)
        r = op.result(y)
        r.set_friendly_name("y")
        return ov.Model([r], [act], "production_2d")

    return {
        "as_emitted":   lambda: tiled(),
        "f16_scale":    lambda: tiled(scale_type=ov.Type.f16),
        "u8_scalar_zp": lambda: tiled(zp_scalar=True),
        "production_2d": production_2d,
        "no_reshape":   lambda: tiled(reshape=False),
    }


def ir_sha256(model, tmpdir):
    """sha256 of the SERIALISED IR (xml then bin), so the window can prove it
    ran the candidate this file characterised. A model held in memory has no
    stable bytes; a saved one does."""
    import openvino as ov

    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    xml = tmpdir / "m.xml"
    ov.save_model(model, str(xml), compress_to_fp16=False)
    h = hashlib.sha256()
    for p in (xml, xml.with_suffix(".bin")):
        h.update(p.read_bytes())
    size = sum(p.stat().st_size for p in (xml, xml.with_suffix(".bin")))
    return h.hexdigest(), size


def runtime_histogram(model, device):
    """(op type, kernel implementation) -> count, from the COMPILED graph.

    `get_runtime_model()` is what verify_moe_lowering.py:171 reads and what
    docs/prefill-baseline.md's kernel tables are built from; `execTimeMcs` and
    the implementation name live in each node's rt_info.
    """
    import openvino as ov

    compiled = ov.Core().compile_model(model, device)
    rt = compiled.get_runtime_model()

    def rt_get(info, key, default):
        # rt_info values are OVAny wrappers: `str(v)` gives "<OVAny class>",
        # which is how the first run of this file printed a histogram of
        # nothing. `.get()` (or `.astype(str)`) is what unwraps them.
        if key not in info:
            return default
        v = info[key]
        for how in ("get", "astype"):
            fn = getattr(v, how, None)
            if fn is None:
                continue
            try:
                return str(fn(str)) if how == "astype" else str(fn())
            except Exception:
                continue
        return default

    hist = {}
    for node in rt.get_ordered_ops():
        info = node.get_rt_info()
        t = rt_get(info, "layerType", node.get_type_name())
        impl = rt_get(info, "primitiveType",
                      rt_get(info, "implementation", "?"))
        hist[(t, impl)] = hist.get((t, impl), 0) + 1
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="CPU",
                    help="CPU (default) or GPU.0 / GPU.1 -- typed by a window "
                         "operator, never defaulted to a card")
    ap.add_argument("--save", default=None,
                    help="directory for the serialised candidate IRs")
    ap.add_argument("--only", default=None,
                    help="comma-separated candidate names")
    ap.add_argument("--experts", type=int, default=DEFAULT_EXPERTS)
    ap.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    a = ap.parse_args()

    import openvino as ov
    import tempfile

    print(f"openvino {ov.__version__}  device {a.device}  "
          f"E={a.experts} M={a.tokens} I={REAL_I} H={REAL_H} group={GROUP_SIZE}")
    cands = _candidates(a.experts, REAL_I, REAL_H, a.tokens)
    names = ([n.strip() for n in a.only.split(",")] if a.only
             else list(cands))
    unknown = [n for n in names if n not in cands]
    assert not unknown, f"unknown candidate(s) {unknown}; have {list(cands)}"

    tmp = a.save or tempfile.mkdtemp(prefix="fc-compressed-")
    for name in names:
        print(f"\n=== {name} " + "=" * (60 - len(name)))
        model = cands[name]()
        sha, size = ir_sha256(model, Path(tmp) / name)
        print(f"  ops {len(model.get_ordered_ops()):>4}   "
              f"IR {size:,} B   sha256 {sha}")
        try:
            hist = runtime_histogram(model, a.device)
        except Exception as e:                       # a GPU compile may refuse
            print(f"  COMPILE FAILED on {a.device}: {type(e).__name__}: "
                  f"{str(e)[:300]}")
            continue
        compressed = [(t, impl, n) for (t, impl), n in hist.items()
                      if any(m in t.lower() for m in COMPRESSED_MARKERS)]
        for (t, impl), n in sorted(hist.items(), key=lambda kv: -kv[1]):
            mark = " <-- COMPRESSED" if any(
                m in t.lower() for m in COMPRESSED_MARKERS) else ""
            print(f"  {n:>4}  {t:<30} {impl}{mark}")
        if a.device.upper().startswith("CPU"):
            # FullyConnectedCompressed / MOECompressed are GPU-PLUGIN
            # primitives. The CPU plugin names its nodes differently and does
            # not run those passes at all, so a CPU histogram cannot answer
            # Q1 either way -- reporting "NOT COMPRESSED" here would be a
            # verdict about the wrong plugin. What the CPU step is FOR is the
            # IR hashes and the fact that each candidate builds and compiles.
            print("  Q1/Q2: NOT ANSWERABLE ON CPU -- the compressed "
                  "primitives are GPU-plugin passes. This step fixes the IR "
                  "hash and proves the candidate compiles.")
        else:
            print(f"  Q1 selection: "
                  + ("COMPRESSED -- " + ", ".join(f"{t}({n})"
                                                  for t, _, n in compressed)
                     if compressed else
                     "NOT COMPRESSED: no compressed primitive in the runtime "
                     "graph"))
            if compressed:
                refs = [f"{t}/{impl}" for t, impl, _ in compressed
                        if "ref" in impl.lower()]
                print(f"  Q2 implementation: "
                      + (f"REFERENCE KERNEL {refs} -- the "
                         f"docs/prefill-baseline.md M2 cliff"
                         if refs else "no reference-kernel fallback"))
    print(f"\nIRs under {tmp}")


if __name__ == "__main__":
    main()
