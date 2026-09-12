"""CF-CHUNKCOV: the PARTITION THEOREM that `build_experts_chunk_model` leans on,
measured instead of read.

WHY THIS FILE EXISTS (REVIEW e78812d, finding F1, 2026-09-12). The orphan
triage discarded `tests/python/test_piecewise_export.py` for good reasons -- its
key convention was wrong in every helper and it asserted `indexer_n_heads == 3`
against a file that says 4 -- but the triage log's conclusion, "Replaced by
test_size_ledger.py + test_attention_piece.py, which cover its intent", is false
for one leg. The review's words:

    "`build_experts_chunk_model` now has **zero coverage in tree**. The only
    cell that ever exercised it was the discarded `test_moe_piece_parity`,
    which asserted exactly the claim b929924 leans on: sum-over-8-chunks +
    shared == the pin's SPARSE MoE output. ... This matters beyond tidiness:
    the MOE-GPU-FUSION localisation ('the expert chunks RUN, |dev-CPU|
    2.766e-04') is a measurement taken through an ungated builder. I read the
    code and it is **correct** ... but 'I read it and it looked right' is the
    thing this repository does not accept."

And the emitter's own docstring names the missing gate: "The caller verifies
sum-over-chunks == pin sparse outside the graph (the theorem is measured in the
test, not asserted by construction)." There was no such test. There is now.

WHAT IS MEASURED, AND WHY MORE THAN ONE GEOMETRY

The theorem is about the PARTITION of the expert axis, so a single partition
cannot measure it: an emitter that ignored `e0` entirely would pass a one-chunk
test, and an emitter that mis-indexed the gate slice would pass any test whose
chunks all start at 0. Every leg here therefore sweeps at least four partitions
of the same expert set, including an UNEVEN one (a tail chunk narrower than the
others -- the shape an off-by-one in `e0:e1` survives) and the degenerate
one-expert-per-chunk partition:

  GEOMETRY A  E=16, H=256, I=64,  k=4   partitions [16] [8,8] [4,4,4,4] [1]x16
  GEOMETRY B  E=12, H=128, I=48,  k=3   partitions [12] [5,5,2] [7,5]   [1]x12
  GEOMETRY C  REAL width, 16 real experts fed from the checkpoint,
              H=2560, I=640        partitions [16] [8,8] [5,5,6]  [1]x16

A and B measure the FULL theorem: sum over chunks + shared expert == the pin's
own SPARSE MoE block (`ref_moe`, whose leaves are the pin's `Qwen4ExpTextExperts`
loop with `index_add_`). That is the claim b929924 leans on, restored verbatim.

C is the other thing this suite was missing. RECONCILE's close-out records that
"gdn, hc, moe and ple still have no isolated real-weights leg of their own".
C is the MoE family's: real dequantised expert bodies at real width, fetched
with `rows=` so 16 experts cost ~300 MiB instead of the full ~10 GiB, checked
against a FLOAT64 reference recomputed from the same fed tensors inside the
cell. The full 512-expert sparse comparison is not reachable on this host and
that is stated rather than skipped around: C measures the partition, not the
routing, and A/B measure the routing against the pin.

THE DISCRIMINATORS -- each one must BITE, and each has a red

A parity cell that only ever sees the correct emitter measures nothing, so the
two indexing claims the review had to take on trust are gated directly:

  * `gate_up[e0:e1]` really is the chunk's own experts, not experts 0..C:
    two DIFFERENT chunks of the same width, given the SAME gate slice, must
    produce DIFFERENT output (asserted large, > 1e-3).
  * the local `j` indexes the weight slice and the gate column CONSISTENTLY:
    permuting the experts inside a chunk and permuting that chunk's gate
    columns by the SAME permutation must leave the output invariant
    (relabel invariance, < 1e-5), while permuting only the weights must move
    it (> 1e-3). An emitter whose `j` drifted between the two would fail the
    first of those.
  * the gate slice is aligned to the chunk: rotating the gate slice by one
    column must move the output (> 1e-3).

Device-free except for the CPU compiles; geometry C needs Q4E_GGUF_SHARDS.

Run (dev-host venv):
    Q4E_GPU= Q4E_GGUF_SHARDS=<shards> \\
      ~/openarc-venv/bin/python3 -m pytest tests/python/test_moe_chunk_partition.py -s
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import openvino as ov  # noqa: E402
from q4e_device import compile_for, device_params  # noqa: E402
from transformers.models.qwen4_exp import configuration_qwen4_exp as pin_cfg  # noqa: E402

from q4e import gguf_feed  # noqa: E402
from q4e import moe as qmoe  # noqa: E402
from q4e import piecewise_export as pwe  # noqa: E402
from q4e import ref_moe  # noqa: E402

_SHARDS = os.environ.get("Q4E_GGUF_SHARDS", "").strip()
_skip_shards = pytest.mark.skipif(
    not _SHARDS,
    reason="Q4E_GGUF_SHARDS unset: geometry C feeds real expert bodies")

PIN_SHA256 = {
    "modeling_qwen4_exp.py": (
        "ca9f00bbd73cfcfbad7ba6073b5ecc23ca27bdace58b746fe6ead6ab175efc0c"
    ),
    "configuration_qwen4_exp.py": (
        "b78132d8cd935437208ee281fa4569b771a63fcb58ebffe84f3e62f5b86235ca"
    ),
}


def _assert_pin():
    import transformers.models.qwen4_exp as pkg
    pkg_dir = Path(pkg.__file__).resolve().parent
    for name, want in PIN_SHA256.items():
        got = hashlib.sha256((pkg_dir / name).read_bytes()).hexdigest()
        assert got == want, (
            f"oracle drift: {name} sha256 {got} != pin {want} ({pkg_dir / name})")


# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------
# name -> (E, H, I, top_k, T, [partitions])
_GEOMETRIES = {
    "A-even":   (16, 256, 64, 4, 8,
                 [[16], [8, 8], [4, 4, 4, 4], [1] * 16]),
    "B-uneven": (12, 128, 48, 3, 6,
                 [[12], [5, 5, 2], [7, 5], [1] * 12]),
}


def _chunk_bounds(widths):
    """[5,5,2] -> [(0,5),(5,10),(10,12)]. The partition itself, explicit."""
    out, e0 = [], 0
    for w in widths:
        out.append((e0, e0 + w))
        e0 += w
    return out


def _make_config(E, H, I, k):
    return pin_cfg.Qwen4ExpTextConfig(
        hidden_size=H,
        num_hidden_layers=1,
        num_experts=E,
        num_experts_per_tok=k,
        norm_topk_prob=True,
        moe_intermediate_size=I,
        shared_expert_intermediate_size=I,
        hidden_act="silu",
        layer_types=["linear_attention"],
    )


@pytest.fixture(scope="module")
def geoms():
    """Per geometry: config, the pin's own sparse block, its state as numpy, and
    a fixed input. Seeded random over EVERY parameter -- the pin zero-inits the
    router (pin 967) and leaves the expert Parameters as `torch.empty`
    (pin 929-930), so a synthetic fixture must stand in a whole checkpoint or
    the router ties 512 ways."""
    _assert_pin()
    out = {}
    for name, (E, H, I, k, T, parts) in _GEOMETRIES.items():
        cfg = _make_config(E, H, I, k)
        torch.manual_seed(hash(name) % (2 ** 31))
        blk = ref_moe.Qwen4ExpTextSparseMoeBlock(cfg).eval()
        with torch.no_grad():
            for p in blk.parameters():
                p.normal_(0.0, 0.05)
        state = {kk: v.detach().cpu().float().numpy()
                 for kk, v in blk.state_dict().items()}
        rng = np.random.default_rng(1234 + E)
        x = rng.standard_normal((1, T, H)).astype(np.float32)
        out[name] = {"cfg": cfg, "blk": blk, "state": state, "x": x,
                     "T": T, "E": E, "parts": parts}
    return out


def _run(model, feed, device="CPU"):
    core = ov.Core()
    compiled = compile_for(core, model, device)
    res = compiled(feed)
    return res[compiled.outputs[0]]


def _chunk_sum(cfg, state, x, T, widths, gate, device="CPU"):
    """Build one chunk model per (e0,e1) of the partition, feed it that chunk's
    OWN gate columns, and sum the outputs. This is the thing under test."""
    acc = None
    for e0, e1 in _chunk_bounds(widths):
        m = qmoe.build_experts_chunk_model(cfg, state, T, e0, e1)
        out = _run(m, {"hidden_states": x,
                       "gate_chunk": np.ascontiguousarray(gate[:, e0:e1])},
                   device)
        acc = out if acc is None else acc + out
    return acc


def _router_gate(cfg, state, x, T, device="CPU"):
    m = qmoe.build_router_model(cfg, state, T)
    return _run(m, {"hidden_states": x}, device)


def _shared(cfg, state, x, T, device="CPU"):
    m = qmoe.build_shared_expert_model(cfg, state, T)
    return _run(m, {"hidden_states": x}, device)


def _maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64)
                               - np.asarray(b, np.float64))))


# ---------------------------------------------------------------------------
# THE THEOREM -- geometries A and B, against the pin's own SPARSE block
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("geom", list(_GEOMETRIES))
@pytest.mark.parametrize("device", device_params())
def test_chunk_sum_plus_shared_equals_the_pin_sparse_moe(geoms, geom, device):
    """sum over chunks of `build_experts_chunk_model` + `build_shared_expert_model`
    == the pin's SPARSE MoE forward, at every partition of the expert axis.

    This is the claim b929924 leans on and the claim the discarded
    `test_moe_piece_parity` was the only cell to make."""
    g = geoms[geom]
    cfg, state, x, T = g["cfg"], g["state"], g["x"], g["T"]

    with torch.no_grad():
        ref = g["blk"](torch.from_numpy(x)).cpu().numpy()          # pin sparse

    gate = _router_gate(cfg, state, x, T, device)                  # [T, E]
    sh = _shared(cfg, state, x, T, device)                         # [T, H]

    print(f"\n[chunk-partition {geom}] device={device} E={g['E']} T={T} "
          f"ref |.|max={np.max(np.abs(ref)):.6e}")
    rows = []
    for widths in g["parts"]:
        got = _chunk_sum(cfg, state, x, T, widths, gate, device) + sh
        got = got.reshape(ref.shape)
        d = _maxabs(got, ref)
        label = f"{len(widths)}x{widths[0]}" if len(set(widths)) == 1 \
            else "+".join(str(w) for w in widths)
        rows.append((label, d))
        print(f"  partition {label:<20s} |chunks+shared - pin sparse| {d:.6e}")
        assert d < 1e-5, (
            f"{geom} partition {widths}: the chunk split is NOT the pin's MoE "
            f"({d:.6e}); the sum-over-chunks theorem does not hold")

    # every partition must land on the SAME answer, not merely within tolerance
    spread = max(d for _, d in rows) - min(d for _, d in rows)
    print(f"  spread across partitions {spread:.6e}")
    assert spread < 1e-5, f"{geom}: partitions disagree by {spread:.6e}"


@pytest.mark.parametrize("geom", list(_GEOMETRIES))
def test_partitions_agree_with_the_single_chunk_bitwise_close(geoms, geom):
    """A tighter statement than the parity above, and independent of the pin:
    however the expert axis is cut, the summed result must be the SAME graph
    result up to float re-association only (chunking changes the order of the
    adds and nothing else)."""
    g = geoms[geom]
    cfg, state, x, T = g["cfg"], g["state"], g["x"], g["T"]
    gate = _router_gate(cfg, state, x, T)
    whole = _chunk_sum(cfg, state, x, T, [g["E"]], gate)
    print(f"\n[partition-invariance {geom}]")
    for widths in g["parts"][1:]:
        got = _chunk_sum(cfg, state, x, T, widths, gate)
        d = _maxabs(got, whole)
        print(f"  {str(widths):<28s} |split - whole| {d:.6e}")
        assert d < 1e-5, f"{geom} {widths}: re-association moved the sum {d:.6e}"


# ---------------------------------------------------------------------------
# THE DISCRIMINATORS -- each must bite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("geom", list(_GEOMETRIES))
def test_a_chunk_uses_its_OWN_experts_not_the_first_C(geoms, geom):
    """`gate_up[e0:e1]`, not `gate_up[0:C]`. Two different chunks of the same
    width, the same gate columns fed to both, must disagree LARGELY. An emitter
    that ignored `e0` would return identical tensors and this cell would read
    0.0."""
    g = geoms[geom]
    cfg, state, x, T, E = g["cfg"], g["state"], g["x"], g["T"], g["E"]
    C = 4 if E >= 8 else 2
    rng = np.random.default_rng(7)
    gslice = rng.standard_normal((T, C)).astype(np.float32)

    a = _run(qmoe.build_experts_chunk_model(cfg, state, T, 0, C),
             {"hidden_states": x, "gate_chunk": gslice})
    b = _run(qmoe.build_experts_chunk_model(cfg, state, T, C, 2 * C),
             {"hidden_states": x, "gate_chunk": gslice})
    d = _maxabs(a, b)
    print(f"\n[chunk-offset {geom}] |chunk(0,{C}) - chunk({C},{2*C})| {d:.6e}")
    assert d > 1e-3, (
        f"{geom}: two different expert chunks produced the same output "
        f"({d:.6e}) -- the builder is ignoring e0")


@pytest.mark.parametrize("geom", list(_GEOMETRIES))
def test_the_local_index_j_addresses_weights_and_gate_consistently(geoms, geom):
    """The review's exact open question: "`gate_up[e0:e1]` with a local `j`
    indexing both the local weight slice and the local gate column".

    Permute the experts INSIDE a chunk and permute that chunk's gate columns by
    the SAME permutation: the sum is invariant (relabel invariance, < 1e-5).
    Permute the weights ONLY: it must move (> 1e-3). An emitter whose `j`
    drifted between weight and gate would fail the first leg."""
    g = geoms[geom]
    cfg, state, x, T, E = g["cfg"], g["state"], g["x"], g["T"], g["E"]
    C = 4 if E >= 8 else 2
    rng = np.random.default_rng(11)
    gslice = rng.standard_normal((T, C)).astype(np.float32)
    perm = rng.permutation(C)

    base = _run(qmoe.build_experts_chunk_model(cfg, state, T, 0, C),
                {"hidden_states": x, "gate_chunk": gslice})

    permuted = dict(state)
    gu = state["experts.gate_up_proj"].copy()
    dn = state["experts.down_proj"].copy()
    gu[0:C] = gu[0:C][perm]
    dn[0:C] = dn[0:C][perm]
    permuted["experts.gate_up_proj"] = gu
    permuted["experts.down_proj"] = dn

    weights_only = _run(qmoe.build_experts_chunk_model(cfg, permuted, T, 0, C),
                        {"hidden_states": x, "gate_chunk": gslice})
    both = _run(qmoe.build_experts_chunk_model(cfg, permuted, T, 0, C),
                {"hidden_states": x, "gate_chunk":
                    np.ascontiguousarray(gslice[:, perm])})

    d_move = _maxabs(base, weights_only)
    d_inv = _maxabs(base, both)
    print(f"\n[relabel {geom}] perm={list(perm)}  weights-only moves {d_move:.6e}  "
          f"weights+gate invariant {d_inv:.6e}")
    assert d_move > 1e-3, (
        f"{geom}: permuting the chunk's experts did not move the output "
        f"({d_move:.6e}) -- the gate is not selecting per expert")
    assert d_inv < 1e-5, (
        f"{geom}: relabel invariance broken ({d_inv:.6e}) -- the local j does "
        f"not address the weight slice and the gate column consistently")


@pytest.mark.parametrize("geom", list(_GEOMETRIES))
def test_the_gate_slice_must_be_aligned_to_the_chunk(geoms, geom):
    """Rotating the chunk's gate columns by one must move the output. This is
    the failure a caller makes when it hands every chunk `gate[:, :C]`."""
    g = geoms[geom]
    cfg, state, x, T, E = g["cfg"], g["state"], g["x"], g["T"], g["E"]
    gate = _router_gate(cfg, state, x, T)
    C = 4 if E >= 8 else 2
    m = qmoe.build_experts_chunk_model(cfg, state, T, 0, C)
    ok = _run(m, {"hidden_states": x,
                  "gate_chunk": np.ascontiguousarray(gate[:, 0:C])})
    rot = _run(m, {"hidden_states": x,
                   "gate_chunk": np.ascontiguousarray(
                       np.roll(gate[:, 0:C], 1, axis=1))})
    d = _maxabs(ok, rot)
    print(f"\n[gate-alignment {geom}] |aligned - rotated| {d:.6e}")
    assert d > 1e-6, (
        f"{geom}: rotating the gate slice changed nothing ({d:.6e}) -- the "
        f"chunk is not reading its gate columns per expert")


# ---------------------------------------------------------------------------
# GEOMETRY C -- the MoE family's first ISOLATED REAL-WEIGHTS leg
# ---------------------------------------------------------------------------

_REAL_EXPERTS = 16          # of 512; ~300 MiB of f32 expert bodies
_REAL_LAYER = 0             # blk.0 carries the MoE families (size-ledger _FAMILIES)
_REAL_PARTS = [[16], [8, 8], [5, 5, 6], [1] * 16]


@pytest.fixture(scope="module")
def real_chunk():
    """The first 16 real expert bodies at real width, dequantised from the
    shipped checkpoint through the name map under test.

    `rows=` slices the LEADING (expert) axis before dequant, so this costs
    16/512 of the layer -- roughly 300 MiB f32 -- instead of ~10 GiB. The full
    512-expert sparse comparison is NOT reachable on this host and is not
    attempted: geometry C measures the PARTITION on real bodies; geometries A
    and B measure the routing against the pin."""
    feed = gguf_feed.GgufFeed(_SHARDS)
    cfg = pwe.real_config()
    gu = feed.pin_tensor(f"layers.{_REAL_LAYER}.mlp.experts.gate_up_proj",
                         rows=_REAL_EXPERTS)
    dn = feed.pin_tensor(f"layers.{_REAL_LAYER}.mlp.experts.down_proj",
                         rows=_REAL_EXPERTS)
    return {"cfg": cfg, "gate_up": gu, "down": dn}


def _ref_experts(gate_up, down, x2d, gate, dtype):
    """The pin's per-expert body (pin 951-954) recomputed from the SAME fed
    tensors at `dtype`, summed over the experts present. No OV in this path.

    Run at both f64 and f32 so the gate can be RELATIVE to the reference's own
    rounding rather than to a constant somebody picked: the FIX-GDN-UTINV
    doctrine, "<= 20x the reference's own rounding, so a drifting yardstick
    cannot satisfy it"."""
    x = x2d.astype(dtype)
    I = down.shape[-1]
    acc = np.zeros((x.shape[0], down.shape[1]), dtype)
    for e in range(gate_up.shape[0]):
        gu = x @ gate_up[e].astype(dtype).T               # [T, 2I]  pin 951
        g, u = gu[:, :I], gu[:, I:]                       # chunk(2)
        inter = (g / (1.0 + np.exp(-g))) * u              # pin 952 silu*up
        acc += (inter @ down[e].astype(dtype).T) \
            * gate[:, e:e + 1].astype(dtype)              # pin 953/954
    return acc


def _ref_experts_f64(gate_up, down, x2d, gate):
    return _ref_experts(gate_up, down, x2d, gate, np.float64)


@_skip_shards
def test_real_expert_bodies_partition_and_match_a_float64_reference(real_chunk):
    """GEOMETRY C. Real dequantised expert bodies, real width (H=2560, I=640),
    four partitions of the same 16 experts, all against a float64 recomputation
    of the pin's own per-expert lines from the same fed tensors.

    This is the MoE family's first isolated real-weights leg -- RECONCILE's
    close-out records that gdn, hc, moe and ple had none."""
    cfg = real_chunk["cfg"]
    gu, dn = real_chunk["gate_up"], real_chunk["down"]
    E, H, I = gu.shape[0], gu.shape[2], dn.shape[2]
    assert (E, H, I) == (_REAL_EXPERTS, cfg.hidden_size, cfg.moe_intermediate_size), \
        f"real shapes moved: gate_up {gu.shape} down {dn.shape}"
    assert gu.shape[1] == 2 * I, f"gate_up is not [E, 2I, H]: {gu.shape}"

    T = 4                                    # real width is expensive per token
    rng = np.random.default_rng(2026)
    x = rng.standard_normal((1, T, H)).astype(np.float32) * 0.05
    gate = rng.standard_normal((T, E)).astype(np.float32)
    # the sparsity the real router produces: most experts off for a given token
    mask = np.zeros((T, E), np.float32)
    for t in range(T):
        mask[t, rng.choice(E, size=3, replace=False)] = 1.0
    gate = gate * mask

    state = {"experts.gate_up_proj": gu, "experts.down_proj": dn}
    x2d = x.reshape(T, H)
    ref64 = _ref_experts(gu, dn, x2d, gate, np.float64)
    ref32 = _ref_experts(gu, dn, x2d, gate, np.float32)
    floor = _maxabs(ref32, ref64)             # the reference's OWN f32 rounding
    gate_at = 20.0 * floor                    # FIX-GDN-UTINV doctrine

    print(f"\n[real-chunk] E={E} of {cfg.num_experts} real experts, H={H} I={I} "
          f"T={T}, f32 bytes fed {(gu.nbytes + dn.nbytes) / 2**20:.1f} MiB")
    print(f"  |ref64|max {np.max(np.abs(ref64)):.6e}   "
          f"|ref32-ref64| {floor:.6e}   gate = 20x floor = {gate_at:.6e}")
    assert floor > 0.0, "the f32 and f64 references are bit-identical; no yardstick"
    got_by_part = {}
    for widths in _REAL_PARTS:
        acc = None
        for e0, e1 in _chunk_bounds(widths):
            m = qmoe.build_experts_chunk_model(cfg, state, T, e0, e1)
            out = _run(m, {"hidden_states": x,
                           "gate_chunk": np.ascontiguousarray(gate[:, e0:e1])})
            acc = out if acc is None else acc + out
        d = _maxabs(acc, ref64)
        label = "+".join(str(w) for w in widths) if len(set(widths)) > 1 \
            else f"{len(widths)}x{widths[0]}"
        got_by_part[label] = acc
        print(f"  partition {label:<12s} |ov - f64| {d:.6e}   "
              f"{d / floor:6.2f}x the reference's own rounding")
        assert d <= gate_at, (
            f"real experts, partition {widths}: |ov - f64| {d:.6e} is "
            f"{d / floor:.1f}x the reference's own f32 rounding ({floor:.6e}); "
            f"the chunk split does not reproduce the pin's per-expert body on "
            f"real weights")

    labels = list(got_by_part)
    for lab in labels[1:]:
        d = _maxabs(got_by_part[lab], got_by_part[labels[0]])
        print(f"  partition {lab:<12s} vs {labels[0]}: {d:.6e}")
        assert d <= gate_at, \
            f"real partitions disagree: {lab} vs {labels[0]} {d:.6e}"


@_skip_shards
def test_the_real_reference_can_fail(real_chunk):
    """The float64 reference above must not be a tautology: corrupt ONE real
    expert body and the same comparison has to fire."""
    cfg = real_chunk["cfg"]
    gu, dn = real_chunk["gate_up"], real_chunk["down"]
    E, H = gu.shape[0], gu.shape[2]
    T = 4
    rng = np.random.default_rng(99)
    x = rng.standard_normal((1, T, H)).astype(np.float32) * 0.05
    gate = np.ones((T, E), np.float32)

    x2d = x.reshape(T, H)
    ref64 = _ref_experts(gu, dn, x2d, gate, np.float64)
    ref32 = _ref_experts(gu, dn, x2d, gate, np.float32)
    floor = _maxabs(ref32, ref64)
    state = {"experts.gate_up_proj": gu, "experts.down_proj": dn}
    clean = _maxabs(_run(qmoe.build_experts_chunk_model(cfg, state, T, 0, E),
                         {"hidden_states": x, "gate_chunk": gate}), ref64)

    bad = gu.copy()
    bad[3] *= 1.01                                    # 1% on expert 3 only
    got = _run(qmoe.build_experts_chunk_model(
        cfg, {"experts.gate_up_proj": bad, "experts.down_proj": dn}, T, 0, E),
        {"hidden_states": x, "gate_chunk": gate})
    d = _maxabs(got, ref64)
    print(f"\n[real-red] |ref32-ref64| floor {floor:.6e}   clean |ov-f64| "
          f"{clean:.6e} ({clean / floor:.2f}x)   one expert body x1.01 -> "
          f"{d:.6e} ({d / floor:.1f}x, {d / max(clean, 1e-30):.0f}x the clean leg)")
    assert d > 20.0 * floor, (
        f"corrupting a real expert body left the result inside the acceptance "
        f"gate ({d:.6e} <= 20x{floor:.6e}) -- the cell above would not have "
        f"noticed a wrong weight")
    assert d > 100.0 * clean, (
        f"corrupting a real expert body moved the result only {d / clean:.1f}x "
        f"the clean leg's own distance -- not a discriminating reference")
