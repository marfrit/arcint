"""E2 increment 4 -- the qwen4_exp SparseMoeBlock (MoE-512 gemv), parity for
`q4e.moe.build_moe_model`.

RED state (precise): this file does `from q4e.moe import build_moe_model`; the
`q4e.moe` MODULE does not exist yet (this increment's GREEN emits
tools/q4e/moe.py and adds `from . import moe` to q4e/__init__.py). Collecting
THIS file therefore fails with `ModuleNotFoundError: No module named 'q4e.moe'`.
`q4e/__init__.py` is untouched at RED, so test_gdn_block / test_hc_block /
test_hc_combine_block still collect and pass. No numeric leg here runs until
the emitter lands.

GREEN: `q4e.moe.build_moe_model(config, state, seq_len)` emits the block as a
STATIC DENSE opset-13 graph with TWO results, in this order:
    result 0  `output`       [1, T, H]            -- routed + shared expert sum
    result 1  `router_gate`  [T, num_experts]     -- the dense top-k routing
                                                     weights (0 for non-selected
                                                     experts, renormalized over
                                                     the selected when
                                                     norm_topk_prob) -- exposed
                                                     so the top-k selection is a
                                                     first-class, checkable output
The dense graph equals the pin's sparse loop because a non-selected (token,
expert) pair carries gate 0 and contributes `f_e(x) * 0.0 == 0.0`.

Config fixture: faithful where the math depends on geometry. `num_experts` is
16 (not the real 512 -- CPU tractability), but `num_experts_per_tok` = 4
(top-k >= 2, so the renorm and the multi-expert sum are exercised),
`norm_topk_prob` = True (the renormalize path), and the shared expert +
shared_expert_gate are present. hidden_size 256, moe_intermediate_size 64,
shared_expert_intermediate_size 64 (the real 512s shrunk for CPU; the gemv
shapes are otherwise the pin's).

Legs:
  * transcription-vs-pin: ref_moe (uses the pin's leaves) vs the pin's own
    SparseMoeBlock -- max-abs EXACTLY 0.0 (guards ref against oracle drift).
  * ov-parity: the emitted DENSE graph vs ref -- max-abs < 1e-5 + a KLD on a
    small logits head.
  * routing sensitivity (discriminator a, must BITE): permuting the expert
    weight tensors along the expert axis (WITHOUT the router) MUST move the
    output under the non-uniform random router -- asserted LARGE (> 1e-2), not
    quietly; permuting the router rows by the SAME permutation restores the
    output (relabel invariance, < 1e-5). A graph that averaged all experts or
    dropped the gate would fail the first assertion.
  * top-k argmax-consistency (discriminator b): the OV `router_gate`'s nonzero
    set per row equals the pin's `torch.topk` index set EXACTLY (and the count
    is exactly top_k -- a tie would fail loudly), and the nonzero gate values
    match the pin's renormalized `router_scores` < 1e-5.

Oracle discipline: the installed pinned transformers reference, re-hashed
before every numeric table (same pin the GDN/hc tests assert).

Run (dev-host venv):
    Q4E_GPU=  ~/openarc-venv/bin/python3 -m pytest tests/python/test_moe_block.py -s --continue-on-collection-errors
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# tools/ on the path so `import q4e` resolves to tools/q4e/.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import openvino as ov  # noqa: E402
from q4e_device import compile_for, device_params  # noqa: E402
from transformers.models.qwen4_exp import configuration_qwen4_exp as pin_cfg  # noqa: E402
from transformers.models.qwen4_exp import modeling_qwen4_exp as pin_mod  # noqa: E402

from q4e import ref_moe  # noqa: E402
# The RED import: the `q4e.moe` module does not exist until this increment's
# GREEN lands it. `q4e/__init__.py` is untouched, so only THIS file's
# collection fails -- the inc1/inc2/inc3 greens keep collecting.
from q4e.moe import build_moe_model  # noqa: E402  (RED: module absent)


# ---------------------------------------------------------------------------
# Oracle pin (same pin the GDN/hc tests assert -- identical installed files)
# ---------------------------------------------------------------------------
PIN_SHA256 = {
    "modeling_qwen4_exp.py": (
        "ca9f00bbd73cfcfbad7ba6073b5ecc23ca27bdace58b746fe6ead6ab175efc0c"
    ),
    "configuration_qwen4_exp.py": (
        "b78132d8cd935437208ee281fa4569b771a63fcb58ebffe84f3e62f5b86235ca"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_pin() -> None:
    """The installed qwen4_exp files must be the pin the transcription was
    validated against. Called before every numeric table."""
    import transformers.models.qwen4_exp as pkg

    pkg_dir = Path(pkg.__file__).resolve().parent
    for name, want in PIN_SHA256.items():
        got = _sha256(pkg_dir / name)
        assert got == want, (
            f"oracle drift: {name} sha256 {got} != pin {want} "
            f"({pkg_dir / name}); the transcription is only valid against the pin"
        )


# ---------------------------------------------------------------------------
# Config fixture (faithful where the math depends on geometry)
# ---------------------------------------------------------------------------
def _make_config():
    return pin_cfg.Qwen4ExpTextConfig(
        hidden_size=256,
        num_hidden_layers=1,
        num_experts=16,             # real 512 shrunk for CPU tractability
        num_experts_per_tok=4,      # top-k >= 2: renorm + multi-expert sum bite
        norm_topk_prob=True,        # the renormalize path
        moe_intermediate_size=64,   # real 512 shrunk
        shared_expert_intermediate_size=64,  # real 512 shrunk
        hidden_act="silu",
        layer_types=["linear_attention"],
    )


def _state_keys():
    """The seven state_dict keys the SparseMoeBlock carries (the fixture stages
    all of them; the GREEN emitter must consume exactly these)."""
    return [
        "gate.weight",                    # router, pin 967 (Linear-less Parameter)
        "experts.gate_up_proj",           # pin 929  [E, 2I, H]
        "experts.down_proj",              # pin 930  [E, H, I]
        "shared_expert.gate_proj.weight",
        "shared_expert.up_proj.weight",
        "shared_expert.down_proj.weight",
        "shared_expert_gate.weight",      # pin 987
    ]


def _ref_and_pin(config, seed: int = 0):
    """ref_moe.Qwen4ExpTextSparseMoeBlock (forward transcribed, leaves are the
    pin's) and the pin's own SparseMoeBlock, identical random weights, eval.

    The pin ZERO-initializes the router weight (pin 967) and leaves the expert
    Parameters as `torch.empty` (pin 929-930, uninitialized memory) -- the real
    values come from the trained checkpoint. A synthetic fixture must stand in a
    seeded random checkpoint over EVERY parameter: a NON-ZERO router so the
    logits are non-uniform and the top-k selection is unambiguous (a zero router
    gives uniform probabilities -- an all-way tie that would over-select and
    make OV/torch topk disagree on which experts), and finite expert weights.
    Scale 0.05 keeps every gemv output O(1) at fp32 (atol 1e-5)."""
    torch.manual_seed(seed)
    ref = ref_moe.Qwen4ExpTextSparseMoeBlock(config).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0.0, 0.05)
    pin = pin_mod.Qwen4ExpTextSparseMoeBlock(config).eval()
    pin.load_state_dict(ref.state_dict())
    return ref, pin


def _state_np(module) -> dict:
    return {k: v.detach().cpu().float().numpy() for k, v in module.state_dict().items()}


def _device_params():
    # Shared with every other q4e suite; see tests/python/q4e_device.py for why
    # the GPU legs must also carry INFERENCE_PRECISION_HINT f32.
    return device_params()


def _kld(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(softmax(p) || softmax(q)) averaged over rows, fp64."""
    def _softmax(x):
        x = x.astype(np.float64)
        x = x - x.max(-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(-1, keepdims=True)

    p = _softmax(p_logits)
    q = _softmax(q_logits)
    return float(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=-1)))


def _ov_outputs(model, feed, device):
    core = ov.Core()
    compiled = compile_for(core, model, device)
    out = compiled(feed)
    by_name = {}
    for i, port in enumerate(compiled.outputs):
        # friendly name is on the result's own node; fall back to any tensor name
        names = list(port.get_names())
        key = names[0] if names else str(i)
        by_name[key] = out[port]
    return by_name, [list(p.get_names()) for p in compiled.outputs]


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------
def test_transcription_matches_pin():
    """ref_moe (leaves = the pin's) reproduces the pin's SparseMoeBlock
    forward EXACTLY (max-abs 0.0) -- guards the transcription against oracle
    drift."""
    _assert_pin()
    config = _make_config()
    ref, pin = _ref_and_pin(config)
    print("\n[moe-transcription-vs-pin] pin sha OK; max-abs by T")
    for T in (64, 96):
        x = torch.randn(1, T, config.hidden_size)
        with torch.no_grad():
            yr = ref(x)
            yp = pin(x)
        md = float((yr - yp).abs().max())
        print(f"  T={T:>3}  max-abs(ref - pin) = {md:.3e}")
        assert md == 0.0, f"transcription drifted from pin at T={T}: {md:.3e}"


@pytest.mark.parametrize("T", [64, 96])
@pytest.mark.parametrize("device", _device_params())
def test_moe_ov_parity(device, T):
    """The emitted dense graph equals the transcription (== pin) at atol=1e-5
    on `output`, plus a KLD on a small logits head. Also asserts the two-result
    contract (`output`, `router_gate`)."""
    _assert_pin()
    config = _make_config()
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)
    assert sorted(state) == sorted(_state_keys()), (
        f"fixture staged the wrong state keys: {sorted(state)}"
    )
    H = config.hidden_size

    x = torch.randn(1, T, H)
    with torch.no_grad():
        y_ref = ref(x).float().numpy()

    model = build_moe_model(config, state, seq_len=T)
    outs, names = _ov_outputs(model, {"hidden_states": x.float().numpy()}, device)
    assert "output" in outs and "router_gate" in outs, (
        f"expected results named output/router_gate, got {names}"
    )
    y_ov = outs["output"]
    max_abs = float(np.max(np.abs(y_ref - y_ov)))
    rng = np.random.default_rng(0)
    head = rng.standard_normal((H, 128)).astype(np.float32)
    kld = _kld(y_ref.reshape(-1, H) @ head, y_ov.reshape(-1, H) @ head)
    print(f"\n[moe-ov-parity] device={device:<6} T={T:>3}  max-abs={max_abs:.3e}  KLD={kld:.3e}")
    assert max_abs < 1e-5, f"OV MoE parity failed device={device} T={T}: {max_abs:.3e}"


@pytest.mark.parametrize("device", _device_params())
def test_moe_routing_sensitivity(device):
    """Discriminator (a) -- routing must BITE. Under the non-uniform random
    router, permuting the EXPERT weight tensors along the expert axis (router
    unchanged) MUST move the output (asserted LARGE); permuting the router rows
    by the SAME permutation restores the output (relabel invariance). A graph
    that averaged all experts or dropped the routing gate would leave the first
    case unchanged and so fail here."""
    _assert_pin()
    config = _make_config()
    T = 64
    E = config.num_experts
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)
    H = config.hidden_size

    x = torch.randn(1, T, H).float().numpy()
    base_model = build_moe_model(config, state, seq_len=T)
    base = _ov_outputs(base_model, {"hidden_states": x}, device)[0]["output"]

    perm = np.roll(np.arange(E), 1)  # a full derangement: every expert moves

    # (i) permute expert weights ONLY -> the same indices now hit different
    #     physical experts -> output must change.
    st_exp = dict(state)
    st_exp["experts.gate_up_proj"] = state["experts.gate_up_proj"][perm]
    st_exp["experts.down_proj"] = state["experts.down_proj"][perm]
    y_exp = _ov_outputs(build_moe_model(config, st_exp, seq_len=T),
                        {"hidden_states": x}, device)[0]["output"]
    drift_exp = float(np.max(np.abs(y_exp - base)))

    # (ii) permute expert weights AND router rows by the same perm -> pure
    #      relabeling of the experts -> output unchanged.
    st_both = dict(st_exp)
    st_both["gate.weight"] = state["gate.weight"][perm]
    y_both = _ov_outputs(build_moe_model(config, st_both, seq_len=T),
                         {"hidden_states": x}, device)[0]["output"]
    drift_both = float(np.max(np.abs(y_both - base)))

    print(f"\n[moe-routing-sensitivity] device={device:<6}  "
          f"perm-experts-only drift={drift_exp:.3e} (must be LARGE)  "
          f"perm-experts+router drift={drift_both:.3e} (must be ~0)")
    assert drift_exp > 1e-2, (
        f"routing does not bite: permuting expert weights moved the output by "
        f"only {drift_exp:.3e} -- the graph is not selecting/gating experts by index"
    )
    assert drift_both < 1e-5, (
        f"relabel invariance broken: permuting experts+router by the same perm "
        f"changed the output by {drift_both:.3e}"
    )


@pytest.mark.parametrize("device", _device_params())
def test_moe_topk_argmax_consistency(device):
    """Discriminator (b) -- the emitted `router_gate`'s selected set (nonzero
    columns per row) equals the pin router's `torch.topk` indices EXACTLY, the
    count is exactly top_k (a tie would fail loudly), and the nonzero gate
    values match the pin's renormalized `router_scores` < 1e-5."""
    _assert_pin()
    config = _make_config()
    T = 96
    k = config.num_experts_per_tok
    ref, pin = _ref_and_pin(config)
    state = _state_np(ref)
    H = config.hidden_size

    x = torch.randn(1, T, H)
    # pin router on the same reshaped input (pin 969-978)
    with torch.no_grad():
        _, pin_scores, pin_idx = pin.gate(x.view(-1, H))
    pin_idx = pin_idx.cpu().numpy()           # [T, k]
    pin_scores = pin_scores.float().cpu().numpy()  # [T, k] renormalized

    model = build_moe_model(config, state, seq_len=T)
    gate = _ov_outputs(model, {"hidden_states": x.float().numpy()}, device)[0]["router_gate"]
    assert gate.shape == (T, config.num_experts), f"router_gate shape {gate.shape}"

    # exactly top_k selected per row (no ties)
    nsel = (gate > 0.0).sum(axis=-1)
    assert np.all(nsel == k), (
        f"router_gate selects {sorted(set(nsel.tolist()))} experts/row, expected exactly {k} "
        f"(a tie at the k-th probability would show here)"
    )
    # set-equality of the selected experts, per row
    ov_sets = [set(np.nonzero(gate[t])[0].tolist()) for t in range(T)]
    pin_sets = [set(pin_idx[t].tolist()) for t in range(T)]
    mismatch = [t for t in range(T) if ov_sets[t] != pin_sets[t]]
    # value agreement at the selected indices
    max_val_dev = 0.0
    for t in range(T):
        for j in range(k):
            e = int(pin_idx[t, j])
            max_val_dev = max(max_val_dev, abs(float(gate[t, e]) - float(pin_scores[t, j])))
    print(f"\n[moe-topk-argmax] device={device:<6} T={T:>3}  set-mismatch rows={len(mismatch)}  "
          f"max gate-vs-pin-score dev={max_val_dev:.3e}")
    assert not mismatch, f"top-k selection disagrees with pin on rows {mismatch[:8]}..."
    assert max_val_dev < 1e-5, f"renormalized gate values drift from pin scores: {max_val_dev:.3e}"


# ---------------------------------------------------------------------------
# CF-TIESENT (REVIEW 2a45349 F3) -- the premise under the scatter swap, and the
# one place the emitter's selection does NOT reproduce the pin.
#
# be57428 replaced the pin's one_hot+reduce_sum gate layout with
# ScatterElementsUpdate because the one_hot shape makes the Intel GPU plugin's
# router fusion fire and then fail. The commit argued the swap was safe because
# "both forms scatter torch.topk's OWN indices, so the selection reproduces the
# pin EXACTLY, tie-break included". The reviewer measured both halves false.
# These two cells replace the argument with what is actually true, and each
# carries its own negative so it cannot pass vacuously.
# ---------------------------------------------------------------------------

def _scatter_layout_model(T, E, k):
    """The SHIPPED layout, isolated: ScatterElementsUpdate into a zeros [T,E]
    along the last axis -- `_router_gate`'s final two ops, nothing else."""
    from openvino import opset13 as op
    zeros = op.parameter([T, E], ov.Type.f32)
    zeros.set_friendly_name("zeros")
    idx = op.parameter([T, k], ov.Type.i32)
    idx.set_friendly_name("idx")
    sc = op.parameter([T, k], ov.Type.f32)
    sc.set_friendly_name("scores")
    g = op.scatter_elements_update(zeros, idx, sc,
                                   op.constant(np.array(-1, np.int32)))
    res = op.result(g)
    res.set_friendly_name("gate")
    return ov.Model([res], [zeros, idx, sc], "scatter_layout")


def _one_hot_layout_model(T, E, k):
    """The SUPERSEDED layout, isolated: the pin's own construct (pin 941),
    `one_hot(top_k_index)` weighted by the scores and summed over the k axis."""
    from openvino import opset13 as op
    idx = op.parameter([T, k], ov.Type.i32)
    idx.set_friendly_name("idx")
    sc = op.parameter([T, k], ov.Type.f32)
    sc.set_friendly_name("scores")
    oh = op.one_hot(idx, op.constant(np.array(E, np.int32)),
                    op.constant(np.array(1.0, np.float32)),
                    op.constant(np.array(0.0, np.float32)), -1)       # [T,k,E]
    w = op.unsqueeze(sc, op.constant(np.array(-1, np.int32)))         # [T,k,1]
    g = op.reduce_sum(op.multiply(oh, w),
                      op.constant(np.array([1], np.int32)), keep_dims=False)
    res = op.result(g)
    res.set_friendly_name("gate")
    return ov.Model([res], [idx, sc], "one_hot_layout")


def test_scatter_equals_one_hot_exactly_when_the_indices_are_distinct():
    """THE LOAD-BEARING PREMISE, asserted directly rather than inherited.

    `one_hot` + reduce_sum SUMS a repeated index; `ScatterElementsUpdate`
    ASSIGNS it, last write wins. So the two layouts are NOT interchangeable in
    general, and the swap be57428 made is legitimate for exactly one reason:
    TopK's indices along a row are DISTINCT. Until this cell nothing in tree
    asserted that -- the swap was gated only transitively, through the
    chunk-partition cells at k=3 and k=4.

    Both legs run on CPU only, and that is not a shortcut: the one_hot layout
    is the shape that FAILS to compile on both Arc cards
    (`program_builder.cpp:268 ... MoERouterFused ... hasn't been found in
    primitive_ids map`), which is why it was replaced. The equivalence under
    test is layout arithmetic, and the CPU plugin executes both.
    """
    T, E, k = 4, 16, 4
    core = ov.Core()
    scat = compile_for(core, _scatter_layout_model(T, E, k), "CPU")
    oneh = compile_for(core, _one_hot_layout_model(T, E, k), "CPU")

    def _both(idx, sc):
        zeros = np.zeros((T, E), np.float32)
        a = scat({"zeros": zeros, "idx": idx, "scores": sc})[scat.outputs[0]]
        b = oneh({"idx": idx, "scores": sc})[oneh.outputs[0]]
        return np.asarray(a), np.asarray(b)

    # LEG 1 -- DISTINCT indices: the layouts agree, exactly, over many draws.
    rng = np.random.default_rng(20260912)
    worst = 0.0
    for _ in range(64):
        idx = np.stack([rng.permutation(E)[:k] for _ in range(T)]).astype(np.int32)
        assert all(len(set(r.tolist())) == k for r in idx), "draw was not distinct"
        sc = rng.random((T, k)).astype(np.float32)
        sc /= sc.sum(-1, keepdims=True)
        a, b = _both(idx, sc)
        worst = max(worst, float(np.max(np.abs(a - b))))
    print(f"\n[moe-scatter-premise] 64 distinct-index draws, "
          f"|scatter - one_hot| max {worst:.6e}")
    assert worst == 0.0, (
        f"the two layouts disagree at {worst:.3e} on DISTINCT indices -- the "
        f"swap's equivalence is broken, not merely its premise")

    # LEG 2 -- THE NEGATIVE. A repeated index, and they must differ, by the
    # amount the arithmetic predicts. Without this leg LEG 1 would be
    # consistent with the two layouts being identical ops, and the premise
    # would be guarding nothing.
    idx = np.tile(np.array([[3, 3, 5, 7]], np.int32), (T, 1))
    sc = np.tile(np.array([[0.4, 0.3, 0.2, 0.1]], np.float32), (T, 1))
    a, b = _both(idx, sc)
    print(f"[moe-scatter-premise] repeated index [3,3,5,7] scores [.4,.3,.2,.1]:"
          f" column 3 -> scatter {a[0, 3]:.3f}  one_hot {b[0, 3]:.3f}")
    assert abs(float(b[0, 3]) - 0.7) < 1e-6, (
        f"one_hot must SUM the repeat: column 3 = {b[0, 3]}, expected 0.7")
    assert abs(float(a[0, 3]) - 0.3) < 1e-6, (
        f"scatter must ASSIGN (last write wins): column 3 = {a[0, 3]}, "
        f"expected 0.3")
    assert float(np.max(np.abs(a - b))) > 0.3, (
        "the layouts agreed on a REPEATED index -- then distinctness is not "
        "what makes the swap sound and this cell's premise is wrong")

    # LEG 3 -- and TopK, the actual producer of those indices, never emits a
    # repeat. Asserted on the emitter's own router output, not on the op's
    # documentation.
    config = _make_config()
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)
    Tr, H = 96, config.hidden_size
    x = torch.randn(1, Tr, H).float().numpy()
    model = build_moe_model(config, state, seq_len=Tr)
    gate = _ov_outputs(model, {"hidden_states": x}, "CPU")[0]["router_gate"]
    nz = (gate != 0.0).sum(axis=-1)
    print(f"[moe-scatter-premise] emitted router_gate: nonzero cols per row "
          f"{sorted(set(nz.tolist()))} over {Tr} rows (k={config.num_experts_per_tok})")
    assert np.all(nz == config.num_experts_per_tok), (
        f"a row carries {sorted(set(nz.tolist()))} nonzero gate columns, not "
        f"{config.num_experts_per_tok}: TopK repeated an index, or a scored "
        f"expert landed on exactly 0.0, and either way the premise above no "
        f"longer holds")


def test_a_degenerate_row_may_select_differently_and_still_emits_zero():
    """WHERE THE EMITTER DOES NOT REPRODUCE THE PIN, measured.

    `op.topk` (OpenVINO) and `torch.topk` (the pin) break ties differently. The
    reachable degenerate case is a ZEROED hidden row -- a masked or padding
    position -- because `logits = x @ W^T` is then 0 for every expert whatever
    the router weight is, and softmax of a constant vector is uniform: an
    all-way tie.

    The mitigation is measured rather than assumed: a bias-free expert gives
    f_e(0) = 0, so whichever experts a zeroed row selects, the block emits
    exactly 0.0 there. The divergence is real and invisible, and this cell
    asserts BOTH halves -- the strict rows still agree, the zeroed row's output
    is exactly 0.0 on both sides -- so a future change that makes the
    divergence visible fails here instead of in a number nobody attributes.
    """
    _assert_pin()
    config = _make_config()
    # T=16, and the value is NOT free: this graph SEGFAULTS the OpenVINO
    # plugin at compile time at T=6 and T=8, which is what the first draft of
    # this cell picked for a readable per-row table. See
    # `test_the_block_compiles_at_every_short_prefill_length` below and
    # tools/repro_moe_compile_short_T.py. 16 is the smallest power of two that
    # compiles and still prints a table a reader can check by eye.
    T, k, H = 16, config.num_experts_per_tok, config.hidden_size
    ref, pin = _ref_and_pin(config)
    state = _state_np(ref)

    torch.manual_seed(3)
    x = torch.randn(1, T, H)
    zero_row = 2
    x[0, zero_row] = 0.0                      # the padded / masked position

    with torch.no_grad():
        _, pin_scores, pin_idx = pin.gate(x.view(-1, H))
        pin_out = pin(x)                      # ref_moe block == pin, cell above
    pin_idx = pin_idx.cpu().numpy()

    model = build_moe_model(config, state, seq_len=T)
    outs = _ov_outputs(model, {"hidden_states": x.float().numpy()}, "CPU")[0]
    gate, ov_out = outs["router_gate"], outs["output"]

    print(f"\n[moe-tie] E={config.num_experts} k={k}, row {zero_row} zeroed")
    for t in range(T):
        p = sorted(pin_idx[t].tolist())
        o = sorted(np.nonzero(gate[t])[0].tolist())
        mark = "  <- zeroed row" if t == zero_row else ""
        print(f"  row {t}  pin {p}  emitter {o}  same={p == o}{mark}")

    # the strict rows agree -- the positive half of the corrected sentence
    for t in range(T):
        if t == zero_row:
            continue
        assert sorted(pin_idx[t].tolist()) == sorted(np.nonzero(gate[t])[0].tolist()), (
            f"row {t} is not degenerate and the selections differ: "
            f"{sorted(pin_idx[t].tolist())} vs "
            f"{sorted(np.nonzero(gate[t])[0].tolist())}")

    # the degenerate row is genuinely degenerate: a uniform softmax
    assert float(np.max(np.abs(gate[zero_row][np.nonzero(gate[zero_row])]
                               - 1.0 / k))) < 1e-6, (
        f"the zeroed row's gate is not uniform 1/k: {gate[zero_row]}")
    assert len(np.nonzero(gate[zero_row])[0]) == k

    # THE MITIGATION, measured on both sides: output exactly 0.0 on that row.
    ov_zero = float(np.max(np.abs(ov_out[0, zero_row])))
    pin_zero = float(pin_out[0, zero_row].abs().max())
    row_dev = [float(np.max(np.abs(ov_out[0, t] - pin_out[0, t].numpy())))
               for t in range(T)]
    print(f"[moe-tie] zeroed-row output: emitter {ov_zero:.6e}  pin {pin_zero:.6e}")
    print(f"[moe-tie] per-row |ov - pin|: "
          + " ".join(f"{d:.2e}" for d in row_dev))
    assert ov_zero == 0.0, (
        f"the emitter's zeroed row is {ov_zero:.3e}, not exactly 0.0 -- the "
        f"tie divergence is no longer invisible and the docstring's mitigation "
        f"is void")
    assert pin_zero == 0.0, (
        f"the pin's zeroed row is {pin_zero:.3e}, not exactly 0.0 -- an expert "
        f"gained a bias and f_e(0) != 0")
    assert max(row_dev) < 1e-5, f"live rows drifted: {max(row_dev):.3e}"


# ---------------------------------------------------------------------------
# NOT MET AND NAMED -- a plugin compile crash this suite could not see
# ---------------------------------------------------------------------------

_SHORT_T_SWEEP = (6, 7, 8)


def _compile_in_a_child(T, device="CPU"):
    """Compile the block at `T` in a FRESH interpreter; return its exit code.

    A SIGSEGV takes the whole process with it, so a crash cannot be asserted
    in-process: pytest would die mid-run and report nothing. A child turns the
    crash into an exit code, which is a fact a cell can hold.
    """
    import subprocess
    repro = REPO_ROOT / "tools" / "repro_moe_compile_short_T.py"
    r = subprocess.run([sys.executable, str(repro), str(T), device],
                       capture_output=True, text=True, timeout=900)
    return r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]


@pytest.mark.xfail(strict=True, reason=(
    "the emitted MoE block SEGFAULTS the OpenVINO plugin at compile time at "
    "T=6 and T=8 (SIGSEGV, OV 2026.4.0-22849, CPU). Found 2026-09-12; every "
    "other numeric cell here runs at T=64/96, so nothing in tree could see "
    "it. Reproducer: tools/repro_moe_compile_short_T.py. strict=True: the day "
    "a plugin or an emitter change fixes it, this cell PASSES, xfail-strict "
    "turns that into an error, and the gap gets retired instead of forgotten."))
def test_the_block_compiles_at_every_short_prefill_length():
    """Every T in the sweep must compile. Two of them do not.

    The crash is in COMPILE and not inference, it is deterministic, and it is
    the ASSEMBLED block rather than any piece: at T=8 `build_router_model`,
    `build_shared_expert_model` and `build_experts_chunk_model` each compile
    cleanly on their own. The full sweep and the piece-wise control are in the
    reproducer's header.

    This matters past tidiness. T is the emitted graph's build-time sequence
    length, so a short prefill -- or any piecewise export that picks a small
    window -- lands on it, and the failure mode is a segfault rather than an
    exception: no traceback, no partial result, the serving process is simply
    gone. It is recorded as a frontier item in docs/window-050.md rather than
    chased here, because the question it raises is which shapes the CARD's
    plugin refuses, and this session spends no card.
    """
    bad = []
    for T in _SHORT_T_SWEEP:
        rc, tail = _compile_in_a_child(T)
        print(f"[moe-short-T] T={T:>3}  rc={rc:<4} {tail[0][:70]}")
        if rc != 0:
            bad.append((T, rc))
    assert not bad, (
        "the MoE block fails to compile at "
        + ", ".join(f"T={T} (rc {rc})" for T, rc in bad)
        + ". A negative rc is the signal that killed the child -- -11 is SIGSEGV "
          "inside the plugin's compile, not an exception this graph could catch.")


@pytest.mark.parametrize("device", _device_params())
def test_moe_row_locality(device):
    """The block is row-local: a garbage probe in trailing rows leaves every
    earlier row of `output` exactly unchanged (no op mixes positions)."""
    _assert_pin()
    config = _make_config()
    T = 64
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)
    H = config.hidden_size
    live = T // 2

    x = torch.randn(1, T, H).float().numpy()
    model = build_moe_model(config, state, seq_len=T)
    base = _ov_outputs(model, {"hidden_states": x}, device)[0]["output"]

    probe = x.copy()
    probe[:, live:] = (np.random.default_rng(1).standard_normal((1, T - live, H)) * 1000.0).astype(np.float32)
    y2 = _ov_outputs(model, {"hidden_states": probe}, device)[0]["output"]
    drift = float(np.max(np.abs(y2[:, :live] - base[:, :live])))
    print(f"\n[moe-row-locality] device={device:<6} T={T:>3}  live-row drift={drift:.3e}")
    assert drift == 0.0, f"row-locality broken: masked-row garbage moved live rows by {drift:.3e}"
