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

import openvino as ov  # noqa: E402
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
    pin's) and the pin's own SparseMoeBlock, identical random weights, eval."""
    torch.manual_seed(seed)
    ref = ref_moe.Qwen4ExpTextSparseMoeBlock(config).eval()
    pin = pin_mod.Qwen4ExpTextSparseMoeBlock(config).eval()
    pin.load_state_dict(ref.state_dict())
    return ref, pin


def _state_np(module) -> dict:
    return {k: v.detach().cpu().float().numpy() for k, v in module.state_dict().items()}


def _device_params():
    devs = ["CPU"]
    extra = os.environ.get("Q4E_GPU", "").strip()
    if extra:
        devs += [d for d in (s.strip() for s in extra.split(",")) if d]
    return devs


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
    compiled = core.compile_model(model, device)
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
