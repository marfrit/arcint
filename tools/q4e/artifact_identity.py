"""THE ARTIFACT CONTRACT, in code.

Endorsed by REVIEW 57b1952 (F1) and recorded as a supersession in RECONCILE:

    ARTIFACT CONTRACT (endorsed): identity := (op-graph topology, constant-blob
    multiset, bit-exact outputs), never .bin sha256.

WHY A BYTE HASH IS NOT AN IDENTITY HERE, measured rather than argued. The
reviewer emitted the same tree 32 times (16 + 8 + 8 with ASLR disabled) and got
exactly TWO artifacts, split 16/16 -- a coin flip, not a rare event. Both were
the same model: 10,193 nodes each, an identical constant-blob multiset (5,944
constants, 284 distinct values), and bit-exact CPU outputs on the same input
(|out1-out2| = 0.000e+00). The whole difference was one f16 [1,4,1,1,1] 8-byte
constant landing at offset 50084 or 8480, shifting everything after it: .bin
111,128 vs 111,120 bytes. It was localised to `compress_to_fp16`'s constant dedup
-- one fixed in-memory model serialised 10x is identical, so save_model IS
deterministic given a graph; it is the rebuild that moves. The obvious
address-layout story was REFUTED by measurement (disabling ASLR does not remove
the flip). Which allocation property the dedup keys on was not identified and is
not asserted here.

So `sha256(.xml)` / `sha256(.bin)` answers "did these two files come out of the
same process run", which is not a question anyone needs. The triple answers "are
these the same model", which is.

THE THREE COMPONENTS, and what each one would catch:

  topology   -- the op graph in topological order: per node its type, its output
                element types and shapes, and WHICH earlier node each input comes
                from. Friendly names are excluded EXCEPT on Parameter and Result,
                where they are the IR's declared input/output names and are part
                of the interface. Catches: a dropped or added op, a rewired
                input, a changed shape, a renamed input/output port.
  constants  -- a multiset (a Counter) over (element type name, shape, byte
                length, sha256(raw bytes)) of every Constant. A MULTISET,
                not a sequence, so it is indifferent to serialisation order and
                to dedup -- which is exactly the axis the flip moves along --
                while still counting multiplicity, so losing one of two identical
                constants is visible. Catches: a changed weight, a weight of the
                right shape with wrong contents, a missing constant.
  outputs    -- bit-exact outputs on a FIXED, deterministic input, compiled on
                CPU. Catches everything the other two miss by construction: any
                difference that is structural-but-equivalent is allowed to pass
                topology and constants, and this is the component that says
                whether it mattered.

A pair is THE SAME ARTIFACT iff all three agree. Two artifacts may legitimately
differ on disk byte-for-byte and be identical under all three; that is the normal
case for this exporter and it is not a defect.

The input generator is deterministic and shape-derived, never random at call
time: f32 parameters get a fixed seeded pattern keyed on the parameter's index
and shape, and i64 parameters get `arange(numel) % shape[-1]`, which is a valid
index for any table whose leading dimension is at least shape[-1] (for the
[1, T] position_ids / input_ids this is exactly arange(T)). An identity is only
comparable against another identity computed with the SAME generator, so the
report carries the generator's version.
"""
import hashlib
from collections import Counter

import numpy as np

# Bump when the input generator or the digest recipe changes: identities from
# different recipes are not comparable and must refuse to compare.
CONTRACT_VERSION = "q4e-artifact-identity/1"


def _h(*parts):
    d = hashlib.sha256()
    for p in parts:
        d.update(str(p).encode() if not isinstance(p, bytes) else p)
        d.update(b"\x1f")
    return d.hexdigest()


def topology_digest(model):
    """Hash of the op graph's structure. Order-stable because
    `get_ordered_ops()` is a topological order of the same graph, and
    name-independent except at the declared interface."""
    ops = model.get_ordered_ops()
    index = {op: i for i, op in enumerate(ops)}
    d = hashlib.sha256()
    d.update(CONTRACT_VERSION.encode())
    for i, op in enumerate(ops):
        tn = op.get_type_name()
        parts = [str(i), tn]
        # Parameter / Result names are the IR's interface, not cosmetics.
        if tn in ("Parameter", "Result"):
            parts.append(op.get_friendly_name())
        for o in range(op.get_output_size()):
            parts.append(op.get_output_element_type(o).get_type_name())
            parts.append(str(op.get_output_partial_shape(o)))
        for inp in op.inputs():
            src = inp.get_source_output()
            parts.append("%d:%d" % (index[src.get_node()], src.get_index()))
        d.update(("|".join(parts) + "\n").encode())
    return d.hexdigest()


def constant_multiset(model):
    """Counter over (type name, shape, nbytes, sha256(bytes)) per Constant."""
    c = Counter()
    for op in model.get_ordered_ops():
        if op.get_type_name() != "Constant":
            continue
        arr = np.ascontiguousarray(op.data)
        # `get_type_name()`, never str(): str(ov.Type.f32) is "<Type: 'float32'>"
        # and a digest keyed on a repr is a digest keyed on a library's __str__.
        c[(op.get_output_element_type(0).get_type_name(),
           tuple(int(x) for x in arr.shape),
           int(arr.nbytes),
           hashlib.sha256(arr.tobytes()).hexdigest())] += 1
    return c


def constants_digest(model, multiset=None):
    ms = constant_multiset(model) if multiset is None else multiset
    # Sorted so the digest is a property of the multiset, not of traversal order.
    return _h(*[f"{k}x{n}" for k, n in sorted(ms.items(), key=lambda kv: str(kv[0]))])


def fixed_inputs(model):
    """Deterministic inputs derived from the declared parameters alone."""
    args = []
    for i, p in enumerate(model.inputs):
        shape = [int(x) for x in p.get_partial_shape().to_shape()]
        et = p.get_element_type().get_type_name()
        n = int(np.prod(shape)) if shape else 1
        if et in ("i64", "i32"):
            bound = max(1, shape[-1] if shape else 1)
            a = (np.arange(n, dtype=np.int64) % bound).reshape(shape)
            args.append(a.astype(np.int64 if "64" in et else np.int32))
        else:
            rng = np.random.default_rng(0xA27 + i * 1009 + n)
            args.append((rng.standard_normal(shape) * 0.02).astype(np.float32))
    return args


def output_arrays(model, device="CPU"):
    """Compile and run once on the fixed inputs; return the raw output arrays."""
    import openvino as ov
    compiled = ov.Core().compile_model(model, device)
    res = compiled(fixed_inputs(model))
    return [np.asarray(res[i]) for i in range(len(compiled.outputs))]


def outputs_digest(model, device="CPU", arrays=None):
    outs = output_arrays(model, device) if arrays is None else arrays
    return _h(*[np.ascontiguousarray(a).tobytes() for a in outs])


def artifact_identity(model, device="CPU", with_outputs=True):
    """The triple, as a dict. `with_outputs=False` skips the compile for callers
    that only want the two static components (a size census, say) -- such a
    report is marked partial and refuses to compare as equal."""
    ms = constant_multiset(model)
    rep = {
        "contract": CONTRACT_VERSION,
        "topology": topology_digest(model),
        "constants": constants_digest(model, ms),
        "n_nodes": len(model.get_ordered_ops()),
        "n_constants": sum(ms.values()),
        "n_distinct_constants": len(ms),
        # Byte counts come from the arrays themselves (element 2 of the key),
        # never from a dtype table that would silently default f16 to f32.
        "const_bytes": sum(k[2] * n for k, n in ms.items()),
        "outputs": None,
        "device": device if with_outputs else None,
    }
    if with_outputs:
        rep["outputs"] = outputs_digest(model, device)
    return rep


def compare_identity(a, b):
    """(same, reasons). `same` is True only when all three components agree and
    neither report is partial."""
    reasons = []
    if a.get("contract") != b.get("contract"):
        return False, [f"contract mismatch: {a.get('contract')} vs {b.get('contract')} "
                       "-- identities from different recipes are not comparable"]
    for key, label in (("topology", "op-graph topology"),
                       ("constants", "constant-blob multiset"),
                       ("outputs", "bit-exact outputs")):
        av, bv = a.get(key), b.get(key)
        if av is None or bv is None:
            reasons.append(f"{label}: not computed on one side (partial report)")
        elif av != bv:
            reasons.append(f"{label} DIFFERS ({av[:16]} vs {bv[:16]})")
    return (not reasons), reasons


def assert_same_artifact(a, b):
    same, reasons = compare_identity(a, b)
    if not same:
        raise AssertionError(
            "artifacts are not the same model under the contract "
            "(topology, constant multiset, bit-exact outputs):\n  "
            + "\n  ".join(reasons))


def format_identity(rep, label=""):
    return (f"[artifact-identity]{' ' + label if label else ''} "
            f"nodes={rep['n_nodes']} constants={rep['n_constants']}"
            f"/{rep['n_distinct_constants']} distinct "
            f"const_bytes={rep['const_bytes']} "
            f"topology={rep['topology'][:16]} constants={rep['constants'][:16]} "
            f"outputs={(rep['outputs'] or 'skipped')[:16]}")


__all__ = [
    "CONTRACT_VERSION", "artifact_identity", "compare_identity",
    "assert_same_artifact", "topology_digest", "constant_multiset",
    "constants_digest", "outputs_digest", "output_arrays", "fixed_inputs",
    "format_identity",
]
