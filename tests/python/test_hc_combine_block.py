"""E2 increment 3 -- per-layer `use_combine=True` GatedResidual mixer,
RED-first parity leg for `q4e.hc.build_combine_model` (NOT YET WRITTEN --
that emitter is this increment's GREEN, next session; this file is the RED).
The `use_combine=True` graph lands as a new entry point IN the existing
`tools/q4e/hc.py` module (same norm/gate body as `build_hc_model`, plus the
injection leg), NOT a new module -- so `q4e/__init__.py` stays untouched.

The `use_combine=True` form is the DECODER-LAYER mixer (pin 1270-1271, the
layer's `attn_hyper_connection` / `mlp_hyper_connection`): `forward` returns
the 3-tuple `(mixed_input, hyper_input, injection_weights)` where
`injection_weights = 2 * sigmoid(block_inject(hyper_input_normed) /
hc_count)` (pin 1030-1031) and the caller (the decoder layer) does
    injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    hidden_states = hyper_input + injection.flatten(-2)
    (the combine: pin 1302-1303 after the attn call 1288, pin 1308-1309
    after the mlp call 1305) so the block's state grows one weight: `block_inject_weight` is present
(pin 1012), unlike the final mixer's `use_combine=False` form (hc.py, pin
1393). The first two legs of the forward body are identical to hc.py's
(pin 1021-1026; the (1 + w) scale keeps hc.py's f64-roundtrip lowering,
160a64a review section 4); the third (pin 1030) is new.

Red state (precise): this file does `from q4e.hc import build_combine_model`;
the `q4e.hc` module IS present (the inc2 emitter), but the NAME
`build_combine_model` is ABSENT, so collecting THIS file fails with
`ImportError: cannot import name 'build_combine_model' from 'q4e.hc'`.
Only this file's collection fails -- `q4e/__init__.py` is untouched, so
test_gdn_block.py and test_hc_block.py (the 8 greens) still collect and
pass. No numeric leg here can run until the emitter lands. GREEN (next
session): `q4e.hc.build_combine_model` emits the `use_combine=True` graph
-- SAME norm/gate body as `build_hc_model` plus the injection leg -- with
THREE results (`mixed` [1,T,H], `hyper_input` [1,T,4H] -- the unchanged
input passthrough, pin 1031 -- `injection` [1,T,4]), state = the fixture's
four keys, opset-13, reusing the hc.py/gdn.py wrappers.

Masked semantics (the layer form, not the final mixer's -- the file
header of tests/python/test_hc_block.py carries the model-path warning
verbatim). The per-layer mixers have NO mask input by design: this harness
masks their input (apply_mask_to_padding_states, pin 199) before the
block; the block is row-local (group RMSNorm group_size=hidden, row-wise
projections, per-row stream mean, row-wise inject projection; pin
1021-1031), so under the masked input all three outputs are exactly zero
on the masked rows and a garbage probe in the masked rows must leave
every live row of all three outputs exactly unchanged. That is the
harness-side contract this leg asserts. (In the pin the per-layer mixers
are called at 1288/1305 on the layer's stream -- the model-path masking
question for the FULL layer stack is the final-mixer story, pin 1493,
tracked in test_hc_block.py's warning; this leg anchors to the masked
input, as the layer's own path does.)

Oracle discipline: same as tests/python/test_hc_block.py -- the installed
pinned transformers reference, re-hashed before every numeric table.

Config fixture: the same small-but-faithful geometry the final-mixer leg
uses (hidden_size 256, hc_count 4, hc_lowrank 32); `use_combine=True`
adds the block_inject Linear (hc_hidden_size -> hc_count, bias=False, pin
1012), so the fixture stages FOUR state keys:
    hc_norm.weight
    input_mix_weight_down.weight
    input_mix_weight_up.weight
    block_inject_weight.weight

Run (dev-host venv):
    Q4E_GPU=        ~/openarc-venv/bin/python3 -m pytest tests/python/test_hc_combine_block.py -s
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

# The RED import: the NAME `build_combine_model` is absent from the
# (present) q4e.hc module until the GREEN of this increment lands it there
# (next session). `q4e/__init__.py` is untouched, so only THIS file's
# collection fails -- the inc1/inc2 greens keep collecting.
from q4e import ref_hc  # noqa: E402
from q4e.hc import build_combine_model  # noqa: E402  (RED: name absent)


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
# Config fixture (use_combine=True, the per-layer mixer geometry)
# ---------------------------------------------------------------------------
def _make_config():
    """A single per-layer mixer config: same small-but-faithful geometry as
    test_hc_block (hidden_size 256, hc_count 4 -- the real model's stream
    count, hc_lowrank 32 -- the real 320 shrunk 10x for CPU cost).
    `use_combine=True` is the CALLER's choice (pin 1004: the class default
    is True); it selects the block_inject_weight presence (pin 1012) and
    the 3-tuple return (pin 1030-1031)."""
    return pin_cfg.Qwen4ExpTextConfig(
        hidden_size=256,
        num_hidden_layers=1,
        hc_count=4,
        hc_lowrank=32,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"],
    )


def _state_keys():
    """The four state_dict keys the use_combine=True mixer carries (the
    fixture stages all of them; the GREEN emitter must consume exactly
    these, the fourth -- block_inject_weight -- being new vs hc.py)."""
    return [
        "hc_norm.weight",
        "input_mix_weight_down.weight",
        "input_mix_weight_up.weight",
        "block_inject_weight.weight",
    ]


def _ref_and_pin(config, seed: int = 0):
    """Build the transcription (ref_hc.Qwen4ExpTextGatedResidual, whose
    use_combine branch is line-for-line the pin's, pin 1028-1031) and the
    pin's own class with identical random weights, eval mode,
    use_combine=True (the per-layer form, pin 1270-1271)."""
    torch.manual_seed(seed)
    ref = ref_hc.Qwen4ExpTextGatedResidual(config, use_combine=True).eval()
    pin = pin_mod.Qwen4ExpTextGatedResidual(config, use_combine=True).eval()
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


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------
def test_transcription_matches_pin():
    """The transcription's use_combine branch mirrors the pin's (pin
    1028-1031): a 3-tuple (mixed [1,T,H], hyper_input [1,T,4H]
    UNCHANGED, injection [1,T,4] in (0, 2))."""
    _assert_pin()
    config = _make_config()
    ref, pin = _ref_and_pin(config)
    print("\n[combine-transcription-vs-pin] pin sha OK; max-abs by T")
    for T in (64, 96):
        x = torch.randn(1, T, config.hc_count * config.hidden_size)
        with torch.no_grad():
            yr = ref(x)
            yp = pin(x)
        assert isinstance(yr, (tuple, list)) and len(yr) == 3, "ref must return the 3-tuple"
        assert isinstance(yp, (tuple, list)) and len(yp) == 3, (
            "use_combine=True must return the 3-tuple (pin 1030-1031)"
        )
        md = max(float((a - b).abs().max()) for a, b in zip(yr, yp))
        print(f"  T={T:>3}  max-abs(ref - pin) = {md:.3e}")
        assert md < 1e-5, f"transcription drifted from pin at T={T}: {md:.3e}"


@pytest.mark.parametrize("T", [64, 96])
@pytest.mark.parametrize("device", _device_params())
def test_combine_ov_parity(device, T):
    """The emitted OV mixer (mask = ones, full rows) equals the
    transcription's 3-tuple at atol=1e-5 on all three outputs (max-abs
    table + KLD on the mixed stream). The GREEN's graph must expose
    THREE results: `mixed`, `hyper_input`, `injection` (the pin's return
    order, pin 1031)."""
    _assert_pin()
    config = _make_config()
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)
    assert sorted(state) == sorted(_state_keys()), (
        f"fixture staged the wrong state keys: {sorted(state)}"
    )
    H = config.hidden_size
    C = config.hc_count

    x = torch.randn(1, T, C * H)
    with torch.no_grad():
        mixed_r, hyper_r, inj_r = ref(x)
    y_ref = (
        mixed_r.float().numpy(),
        hyper_r.float().numpy(),
        inj_r.float().numpy(),
    )

    model = build_combine_model(config, state, seq_len=T)
    core = ov.Core()
    compiled = core.compile_model(model, device)
    out = compiled({"hyper_input": x.float().numpy()})
    y_ov = tuple(
        out[compiled.outputs()[i]] for i in range(len(compiled.outputs()))
    )
    assert len(y_ov) == 3, f"expected 3 results (mixed/hyper_input/injection), got {len(y_ov)}"

    max_abs = max(float(np.max(np.abs(r - o))) for r, o in zip(y_ref, y_ov))
    rng = np.random.default_rng(0)
    head = rng.standard_normal((H, 128)).astype(np.float32)
    kld = _kld(
        y_ref[0].reshape(-1, H) @ head,
        y_ov[0].reshape(-1, H) @ head,
    )
    print(f"\n[combine-ov-parity] device={device:<6} T={T:>3}  max-abs(3-tuple)={max_abs:.3e}  "
          f"KLD(mixed)={kld:.3e}")
    assert max_abs < 1e-5, (
        f"OV combine parity failed device={device} T={T}: max-abs {max_abs:.3e} "
        f"(per-output: "
        + ", ".join(f"{float(np.max(np.abs(r - o))):.3e}" for r, o in zip(y_ref, y_ov))
        + ")"
    )


@pytest.mark.parametrize("T", [64, 96])
@pytest.mark.parametrize("device", _device_params())
def test_combine_ov_parity_masked(device, T):
    """Trailing-zeros mask (the per-layer form's masked semantics). The
    harness zeroes the masked rows before the block; the row-local block
    then emits EXACTLY zero on the masked rows for all three outputs
    (zero row -> group norm 0 -> gate sigmoid(0)=0.5 -> mean of zero
    streams 0; the block_inject projection of the zero normed row ->
    sigmoid(0)=0.5 -> 2*0.5=1. The injection is CONSUMED downstream in
    the layer (pin 1302-1303/1308-1309), so this block's own masked-row
    output for the injection leg is the gate row, exactly 1.0).
    The exact contract below is asserted on what this block itself
    returns: (a) ref emits masked rows exactly 0.0 for mixed AND
    hyper_input; the injection weight rows on masked input are exactly
    2*sigmoid(0)=1.0 (the gate, not zero) -- asserted like-for-like;
    (b) a garbage probe in the masked rows leaves every live row of all
    three outputs exactly unchanged."""
    _assert_pin()
    config = _make_config()
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)
    H = config.hidden_size
    C = config.hc_count

    live = T // 2
    mask = torch.ones(1, T, dtype=torch.long)
    mask[:, live:] = 0

    x = torch.randn(1, T, C * H)
    x_masked = ref_hc.apply_mask_to_padding_states(x, mask)
    with torch.no_grad():
        mixed_r, hyper_r, inj_r = ref(x_masked)
    y_ref = (
        mixed_r.float().numpy(),
        hyper_r.float().numpy(),
        inj_r.float().numpy(),
    )

    # (a) like-for-like ref contract on masked rows:
    #     mixed and hyper_input exactly 0 (row-local + masked-zero input);
    #     injection weight rows = 2*sigmoid(0)=1.0 EXACTLY (zero normed row
    #     -> block_inject matmul of zeros -> 0 -> sigmoid 0.5 -> *2).
    assert float(np.max(np.abs(y_ref[0][:, live:]))) == 0.0, "ref: masked mixed rows not 0"
    assert float(np.max(np.abs(y_ref[1][:, live:]))) == 0.0, "ref: masked hyper_input rows not 0"
    assert float(np.max(np.abs(y_ref[2][:, live:] - 1.0))) == 0.0, (
        "ref: masked injection rows not exactly 2*sigmoid(0)=1"
    )

    model = build_combine_model(config, state, seq_len=T)
    core = ov.Core()
    compiled = core.compile_model(model, device)
    out = compiled({"hyper_input": x_masked.float().numpy()})
    y_ov = tuple(out[compiled.outputs()[i]] for i in range(len(compiled.outputs())))

    masked_max = float(max(np.max(np.abs(o[:, live:])) for o in (y_ov[0], y_ov[1])))
    assert masked_max == 0.0, (
        f"OV: masked mixed/hyper_input rows not exactly 0 at device={device} T={T} "
        f"(max {masked_max:.3e}) -- a zero row must stay exactly zero"
    )
    inj_dev = float(np.max(np.abs(y_ov[2][:, live:] - 1.0)))
    assert inj_dev < 1e-5, (
        f"OV: masked injection rows deviate from 2*sigmoid(0)=1 by {inj_dev:.3e} "
        f"at device={device} T={T}"
    )
    max_abs = float(
        max(
            np.max(np.abs(r[:, :live] - o[:, :live]))
            for r, o in zip(y_ref, y_ov)
        )
    )
    print(f"[combine-ov-parity-masked] device={device:<6} T={T:>3} live={live:>2}  "
          f"max-abs(live, 3-tuple)={max_abs:.3e}  max-abs(masked mixed/hyper)={masked_max:.3e}  "
          f"inj(masked)-1={inj_dev:.3e}")
    assert max_abs < 1e-5, f"OV combine masked parity failed device={device} T={T}: {max_abs:.3e}"

    # (b) garbage probe in masked rows: every live row of all three
    # outputs must move by exactly 0.0 (row-locality proof).
    probe = x_masked.clone()
    probe[:, live:] = torch.randn(1, T - live, C * H) * 1000.0
    out2 = compiled({"hyper_input": probe.float().numpy()})
    drift = float(
        max(
            np.max(np.abs(out2[compiled.outputs()[i]][:, :live] - o[:, :live]))
            for i, o in enumerate(y_ov)
        )
    )
    print(f"[combine-mask-garbage-probe] device={device:<6} T={T:>3}  live-row drift(3-tuple)={drift:.3e}")
    assert drift == 0.0, (
        f"mask isolation broken: masked-row garbage moved live rows by {drift:.3e} "
        f"(device={device} T={T}) -- the block must be row-local"
    )
