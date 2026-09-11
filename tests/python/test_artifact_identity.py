"""The ARTIFACT CONTRACT, as a gate.

identity := (op-graph topology, constant-blob multiset, bit-exact outputs),
never .bin sha256 -- endorsed by REVIEW 57b1952 finding F1 and recorded as a
supersession in RECONCILE. `tools/q4e/artifact_identity.py` carries the reasoning
and the reviewer's 32-emit measurement; this file is what stops the contract from
being prose.

THE BYTE-VARIANT PAIR IS CONSTRUCTED, NOT OBSERVED. The real phenomenon is a
coin flip -- the reviewer got exactly two artifacts over 32 emits of one tree,
split 16/16 -- and a coin flip is no basis for a gate: it would pass half the
time for the wrong reason. So the pair here is made deterministically by building
the SAME graph with its two Constant nodes CREATED in opposite order. Measured
lever selection, on the dev host, before this file was written:

    A) constant creation order   xml 7de3fda32ba9a4cf -> 66895ee91400b8cc   DIFFERS
    B) .xml round-trip           xml 3397e29fafe259da -> 3397e29fafe259da   same
    C) same model saved twice    xml dd343700eb5623ef -> dd343700eb5623ef   same

B and C confirm the reviewer's localisation from the other side: save_model is
deterministic GIVEN a graph, so neither re-serialising nor round-tripping moves a
byte. It is the rebuild that moves, and (A) is a rebuild. That makes (A) both the
honest analogue of the real flip and a reliable one.

The cells are device-free and run in well under a second; they gate every emit,
not just a window.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import openvino as ov  # noqa: E402
from openvino import opset13 as op  # noqa: E402

from q4e import artifact_identity as aid  # noqa: E402


def _build(order, cb_value=-3.0, n=8, extra=None):
    """y = concat(x*ca, x*cb). `order` decides which Constant is CREATED first;
    the graph, its values and its outputs are identical either way."""
    p = op.parameter([1, n], ov.Type.f32)
    p.set_friendly_name("x")
    va = np.arange(n, dtype=np.float32).reshape(1, n) * 0.5 + 1.0
    vb = np.arange(n, dtype=np.float32).reshape(1, n) * 0.25 + cb_value
    if order == "ab":
        ca, cc = op.constant(va), op.constant(vb)
    else:
        cc, ca = op.constant(vb), op.constant(va)
    parts = [op.multiply(p, ca), op.multiply(p, cc)]
    if extra is not None:
        parts.append(op.multiply(p, op.constant(
            np.full((1, n), extra, dtype=np.float32))))
    r = op.result(op.concat(parts, axis=-1))
    r.set_friendly_name("y")
    return ov.Model([r], [p], "contract_probe")


def _sha_files(model, tmp_path, tag):
    import hashlib
    x = os.path.join(str(tmp_path), f"{tag}.xml")
    ov.save_model(model, x, compress_to_fp16=True)
    b = x[:-4] + ".bin"
    h = (hashlib.sha256(open(x, "rb").read()).hexdigest(),
         hashlib.sha256(open(b, "rb").read()).hexdigest())
    return h, os.path.getsize(b)


def test_byte_variant_pair_is_the_same_artifact(tmp_path, capsys):
    """THE CENTRAL CELL. Two emits that differ on disk are the same model under
    the triple. If this ever fails, either the contract is wrong or the two
    graphs really are different -- both worth stopping for."""
    ma, mb = _build("ab"), _build("ba")
    (xa, ba_), sa = _sha_files(ma, tmp_path, "variant_a")
    (xb, bb_), sb = _sha_files(mb, tmp_path, "variant_b")

    ia = aid.artifact_identity(ma)
    ib = aid.artifact_identity(mb)
    sys.stdout.write(
        f"\n[contract] A .xml {xa[:16]} .bin {ba_[:16]} ({sa} B)\n"
        f"[contract] B .xml {xb[:16]} .bin {bb_[:16]} ({sb} B)\n"
        f"{aid.format_identity(ia, 'A')}\n{aid.format_identity(ib, 'B')}\n")

    # The premise: they really are byte-different somewhere on disk.
    assert (xa, ba_) != (xb, bb_), (
        "the constructed pair is byte-IDENTICAL on disk, so this cell proves "
        "nothing -- the creation-order lever has stopped working and the pair "
        "must be rebuilt before the assertions below mean anything")

    # The contract: and yet they are the same artifact.
    aid.assert_same_artifact(ia, ib)
    same, reasons = aid.compare_identity(ia, ib)
    assert same and not reasons, reasons


def test_each_component_can_fail(tmp_path):
    """Three corruptions, one per component, each of which MUST be caught -- so
    no leg of the triple is decoration. Without this the contract could be three
    constants compared against themselves."""
    base = aid.artifact_identity(_build("ab"))

    # (1) constants: same topology, same shapes, ONE value changed.
    changed = aid.artifact_identity(_build("ab", cb_value=-3.5))
    same, reasons = aid.compare_identity(base, changed)
    assert not same
    joined = " ".join(reasons)
    assert "constant-blob multiset DIFFERS" in joined, reasons
    assert "bit-exact outputs DIFFERS" in joined, reasons
    assert "topology" not in joined, (
        "a pure value change must NOT move the topology digest: " + joined)

    # (2) topology: an extra op and constant -> a different graph.
    wider = aid.artifact_identity(_build("ab", extra=2.0))
    same, reasons = aid.compare_identity(base, wider)
    assert not same
    assert "op-graph topology DIFFERS" in " ".join(reasons), reasons

    # (3) outputs: a report whose outputs were never computed must refuse to
    # compare equal rather than quietly pass on two of three.
    partial = aid.artifact_identity(_build("ab"), with_outputs=False)
    same, reasons = aid.compare_identity(base, partial)
    assert not same
    assert "partial report" in " ".join(reasons), reasons


def test_interface_names_are_part_of_the_identity():
    """Renaming a declared input is an interface change, not cosmetics: a
    consumer feeding by name breaks. Interior node names are NOT part of it."""
    m1 = _build("ab")
    m2 = _build("ab")
    m2.inputs[0].get_node().set_friendly_name("input_ids")
    same, reasons = aid.compare_identity(aid.artifact_identity(m1),
                                         aid.artifact_identity(m2))
    assert not same and "topology DIFFERS" in " ".join(reasons), reasons

    m3 = _build("ab")
    for o in m3.get_ordered_ops():
        if o.get_type_name() == "Multiply":
            o.set_friendly_name("renamed_interior_op")
    same, reasons = aid.compare_identity(aid.artifact_identity(m1),
                                         aid.artifact_identity(m3))
    assert same, f"interior renames must not move the identity: {reasons}"


def test_constant_multiset_counts_multiplicity():
    """A multiset, not a set: losing one of two identical constants is a real
    change even though the DISTINCT values are unchanged. This is the property
    that lets the digest be indifferent to dedup ORDER without becoming
    indifferent to dedup itself."""
    n = 8
    v = np.full((1, n), 1.5, dtype=np.float32)

    def two_separate():
        p = op.parameter([1, n], ov.Type.f32)
        p.set_friendly_name("x")
        y = op.concat([op.multiply(p, op.constant(v)),
                       op.multiply(p, op.constant(v))], axis=-1)
        return ov.Model([op.result(y)], [p], "m")

    def one_shared():
        p = op.parameter([1, n], ov.Type.f32)
        p.set_friendly_name("x")
        c = op.constant(v)
        y = op.concat([op.multiply(p, c), op.multiply(p, c)], axis=-1)
        return ov.Model([op.result(y)], [p], "m")

    ms2 = aid.constant_multiset(two_separate())
    ms1 = aid.constant_multiset(one_shared())
    # The f32 value constant appears twice vs once; the Concat/axis constants
    # are the same either way.
    vals2 = {k: n_ for k, n_ in ms2.items() if k[0] == "f32"}
    vals1 = {k: n_ for k, n_ in ms1.items() if k[0] == "f32"}
    assert sum(vals2.values()) == sum(vals1.values()) + 1, (vals1, vals2)
    assert set(vals2) == set(vals1), "distinct VALUES are unchanged by design"
    assert aid.constants_digest(two_separate()) != aid.constants_digest(one_shared())


def test_fixed_inputs_are_deterministic_and_index_safe():
    """The generator must give the same bytes every call (or two identities are
    not comparable) and must produce in-range indices for i64 ports."""
    T = 12
    p0 = op.parameter([1, T, 4], ov.Type.f32)
    p0.set_friendly_name("hidden_states")
    p1 = op.parameter([1, T], ov.Type.i64)
    p1.set_friendly_name("position_ids")
    tbl = op.constant(np.arange(T * 4, dtype=np.float32).reshape(T, 4))
    g = op.gather(tbl, p1, op.constant(np.int64(0)))
    m = ov.Model([op.result(op.add(p0, g))], [p0, p1], "m")

    a1, a2 = aid.fixed_inputs(m), aid.fixed_inputs(m)
    for x, y in zip(a1, a2):
        assert np.array_equal(x, y), "fixed_inputs is not deterministic"
    idx = a1[1]
    assert idx.dtype == np.int64
    assert int(idx.min()) >= 0 and int(idx.max()) < T, (
        f"i64 inputs must be valid indices: got [{idx.min()}, {idx.max()}) for T={T}")
    # and it actually runs (a generator that produces an out-of-range index
    # would raise here, which is the point of the bound)
    assert aid.outputs_digest(m)


@pytest.mark.parametrize("order", ["ab", "ba"])
def test_identity_is_stable_across_repeated_computation(order):
    """Two identities of the same freshly built model agree. If this is flaky the
    contract cannot be used as a gate at all."""
    reps = [aid.artifact_identity(_build(order)) for _ in range(3)]
    for r in reps[1:]:
        aid.assert_same_artifact(reps[0], r)
