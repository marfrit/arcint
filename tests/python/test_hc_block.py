"""E2 increment 2 -- GatedResidual (hc-4 hyper-connection mixer) OV-emission
parity test.

Red-first harness for `tools/q4e/hc.py` (the opset-13 emission of the
qwen4_exp `use_combine=False` GatedResidual, the text model's final
`hyper_connection_mixer`, pin line 1393). It is RED until `q4e.hc` exists:
the package `q4e/__init__.py` does `from . import hc`, so importing anything
from `q4e` raises ImportError until the emitter lands. That import failure
is this file's intended red.

Oracle discipline (same as tests/python/test_gdn_block.py): the numeric
oracle is the *installed* pinned transformers reference
(`transformers.models.qwen4_exp`). `_assert_pin()` hashes the installed
modeling/configuration files against the pin sha256 BEFORE every numeric
table, so a silently-swapped reference goes red rather than reporting a
number against the wrong math.

Legs:
  * transcription-vs-pin: `ref_hc.Qwen4ExpTextGatedResidual` (the
    transcription the OV graph mirrors) must equal the pin's own class on
    identical random weights/inputs at 1e-5. Validated against the pin, not
    against itself.
  * OV parity: the emitted OV model vs the transcription at atol=1e-5, with
    a max-abs table and a KLD on a small softmax head, at T=64 and T=96
    (row-local block: T only sizes the graph, no chunk/pad structure).
  * masked-parity: trailing-zeros mask. Graph contract: the OV graph takes
    the ALREADY-masked tensor -- the harness zeroes masked rows before
    feeding the block (ref_hc.apply_mask_to_padding_states). The block is
    row-local, so this leg asserts:
      (a) OV on the zeroed input emits the masked rows EXACTLY 0.0 (the
          reference's masked output, asserted like-for-like), and the live
          rows agree at atol=1e-5;
      (b) a garbage probe in the masked rows leaves every live row EXACTLY
          unchanged (the row-locality proof -- in GDN the same probe WOULD
          leak through the recurrent state; here it must move nothing).
    WHY masked positions do not matter in THIS module (the reviewer 6e93b8f
    asked, per module): they cannot. In GDN, a masked position still
    evolves the recurrent state (beta = sigmoid(0) = 0.5, g != 0), so a
    graph that ignored the mask would diverge -- the mask carries signal
    there. Here every op is row-local (group RMSNorm group_size=hidden,
    row-wise projections, per-row stream mean; pin 1021-1026), so a zeroed
    row is exactly inert for ANY content and nothing leaks between rows.

WARNING -- model-path contract for the per-layer `use_combine=True` mixers
(E2 increment 3, and any KLD instrument over the full layer stack). Two
mask paths, only one is the model's own:
  * the per-layer mixers (the decoder layer's `attn_hyper_connection` /
    `mlp_hyper_connection`, called at pin 1288/1305) have NO mask input by
    design -- this harness masks their input (apply_mask_to_padding_states,
    pin 199) before the block; being row-local, the masked rows of their
    3-tuple output (mixed, hyper_input, injection) are then exactly zero
    under that masked input. That is a HARNESS-side property of the graph,
    not a path the model executes.
  * the model's own masked application (pin 1252-1253) lives in the PLE
    layer's conv path (Qwen4ExpTextPLELayer.forward), not in front of the
    decoder layer and not in front of the final mixer. The TextModel's
    final `hyper_connection_mixer` is called at pin 1493 on the UNMASKED
    layer-stack output: in the real model the padded rows carry live,
    layer-generated values INTO the final mixer and come out NON-zero.
  * a future KLD instrument that averages over ALL rows of the final-mixer
    output must therefore NOT assume zero pads: score live rows only, or
    subtract the reference's pad rows like-for-like.
(Reviewer 160a64a finding: the earlier "exactly what the model does (pin
1252-1253)" anchor for this leg was a wrong-path claim; it is retracted.)

Devices:
  CPU is always run (host RAM, touches no card). GPU legs (GPU.0/GPU.1) run
  ONLY when Q4E_GPU is set to a comma list of device names, because loading a
  graph onto a card requires stopping the resident service -- a GPU window,
  which is operator/frontier territory, not the engineer seat.

Run (dev-host venv):
    Q4E_GPU=        ~/openarc-venv/bin/python3 -m pytest tests/python/test_hc_block.py -s
    Q4E_GPU=GPU.0   ...   # operator/frontier, inside a window with the card free
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

# The RED import: q4e/__init__.py -> `from . import hc`; fails until hc.py is
# emitted. ref_hc is the (present) transcription but is unreachable through
# the package until the same import succeeds.
from q4e import hc, ref_hc  # noqa: E402


# ---------------------------------------------------------------------------
# Oracle pin (same pin the GDN test asserts -- identical installed files)
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
# Small but architecturally faithful mixer config
# ---------------------------------------------------------------------------
def _make_config():
    """A single-mixer config small enough for CPU parity but faithful where
    the math depends on geometry: hc_count 4 (the real model's stream count,
    which the config validation requires to be > 1) and hc_lowrank 32 (the
    real 320 shrunk 10x for CPU cost -- the low-rank shape is the only
    non-normalized dimension the gate has)."""
    return pin_cfg.Qwen4ExpTextConfig(
        hidden_size=256,
        num_hidden_layers=1,
        hc_count=4,
        hc_lowrank=32,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"],
    )


def _ref_and_pin(config, seed: int = 0):
    """Build the transcription and the pin's own GatedResidual class with
    identical random weights, in eval mode, use_combine=False (the form the
    text model's final mixer is built in, pin line 1393)."""
    torch.manual_seed(seed)
    ref = ref_hc.Qwen4ExpTextGatedResidual(config, use_combine=False).eval()
    pin = pin_mod.Qwen4ExpTextGatedResidual(config, use_combine=False).eval()
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


def _run_ref(ref, x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """The transcription with the pin's apply_mask_to_padding_states as
    entry, then the masked ref output (what the OV graph must match)."""
    masked = ref_hc.apply_mask_to_padding_states(x, mask)
    with torch.no_grad():
        return ref(masked).float().numpy()


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------
def test_transcription_matches_pin():
    """The transcription mirrors the pin's own class (validated against the
    pin, not against itself), use_combine=False -> a bare [1, T, H] tensor."""
    _assert_pin()
    config = _make_config()
    ref, pin = _ref_and_pin(config)
    print("\n[transcription-vs-pin] pin sha OK; max-abs by T")
    for T in (64, 96):
        x = torch.randn(1, T, config.hc_count * config.hidden_size)
        with torch.no_grad():
            yr = ref(x)
            yp = pin(x)
        assert isinstance(yp, torch.Tensor), (
            "use_combine=False must return a bare tensor (pin 1028-1029)"
        )
        md = (yr - yp).abs().max().item()
        print(f"  T={T:>3}  max-abs(ref - pin) = {md:.3e}")
        assert md < 1e-5, f"transcription drifted from pin at T={T}: {md:.3e}"


@pytest.mark.parametrize("T", [64, 96])
@pytest.mark.parametrize("device", _device_params())
def test_hc_ov_parity(device, T):
    """The emitted OV mixer equals the transcription at atol=1e-5 (max-abs
    table + KLD on a small softmax head). Mask = ones (full rows)."""
    _assert_pin()
    config = _make_config()
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)

    x = torch.randn(1, T, config.hc_count * config.hidden_size)
    mask = torch.ones(1, T, dtype=torch.long)
    y_ref = _run_ref(ref, x, mask)

    model = hc.build_hc_model(config, state, seq_len=T)
    core = ov.Core()
    compiled = core.compile_model(model, device)
    out = compiled({"hyper_input": x.float().numpy()})
    y_ov = out[compiled.output(0)]

    max_abs = float(np.max(np.abs(y_ref - y_ov)))

    # KLD on a small fixed softmax head over the mixed hidden dim.
    rng = np.random.default_rng(0)
    head = rng.standard_normal((config.hidden_size, 128)).astype(np.float32)
    kld = _kld(
        y_ref.reshape(-1, config.hidden_size) @ head,
        y_ov.reshape(-1, config.hidden_size) @ head,
    )
    print(f"\n[ov-parity] device={device:<6} T={T:>3}  max-abs={max_abs:.3e}  KLD={kld:.3e}")
    assert max_abs < 1e-5, f"OV hc parity failed device={device} T={T}: {max_abs:.3e}"


@pytest.mark.parametrize("T", [64, 96])
@pytest.mark.parametrize("device", _device_params())
def test_hc_ov_parity_masked(device, T):
    """Partial mask (trailing padding): the masked output must agree with the
    masked transcription at atol=1e-5, the masked rows must be EXACTLY zero,
    and a garbage probe in the masked rows must not leak into any live row.

    WHY this is the discriminator for this module (the reviewer 6e93b8f asked,
    per module, why masked positions matter): they do NOT, and this test
    proves it rather than asserting it. In GDN, masked positions still evolve
    the recurrent state (beta = sigmoid(0) = 0.5, g = -exp(A_log)*softplus(dt_bias)
    != 0), so a graph that ignored the mask would diverge at randn scale --
    the mask carries signal there. Here every op is row-local (group RMSNorm
    group_size=hidden, row-wise projections, per-row stream mean; pin
    1021-1026), so:
      (a) a zeroed row is normed to a zero group, gates to sigmoid(0)=0.5,
          and the weighted mean of zero streams is exactly 0.0 (not
          ~1e-8) -- assert with tolerance 0, i.e. max over masked rows == 0;
      (b) garbage in a masked row cannot change any other row -- the probe
          must move the live rows by exactly 0.0.
    If the OV graph ever gains cross-position structure (it must not), this
    test is the one that goes red."""
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
    # The harness masks the input here: apply_mask_to_padding_states zeroes
    # the masked rows before the block, and the OV graph -- row-local, no
    # mask input by design -- takes that already-masked tensor, exactly as
    # the GDN leg does. (The final mixer in the pin is called on the
    # UNMASKED layer-stack output, pin 1493 -- see the file-header warning;
    # that is a different path and is not what this leg anchors to.)
    x_masked = ref_hc.apply_mask_to_padding_states(x, mask)
    y_ref = _run_ref(ref, x, mask)

    # (a) the reference emits the masked rows exactly 0.0 (the contract the
    # OV graph must reproduce; asserted here so the OV assertion below
    # compares like with like).
    assert float(np.max(np.abs(y_ref[:, live:]))) == 0.0, "ref: masked rows not exactly 0"

    model = hc.build_hc_model(config, state, seq_len=T)
    core = ov.Core()
    compiled = core.compile_model(model, device)
    out = compiled({"hyper_input": x_masked.float().numpy()})
    y_ov = out[compiled.output(0)]

    # (a) OV on the zeroed input emits the masked rows exactly 0.0: a zero
    # row is normed to a zero group (rsqrt(eps) is finite, 0*finite=0),
    # gates to sigmoid(0)=0.5, and the weighted mean of zero streams is 0
    # -- exact, not ~1e-8.
    masked_max = float(np.max(np.abs(y_ov[:, live:])))
    assert masked_max == 0.0, (
        f"OV: masked rows not exactly 0 at device={device} T={T} "
        f"(max {masked_max:.3e}) -- a zero row must stay exactly zero"
    )
    # live rows agree with the transcription
    max_abs = float(np.max(np.abs(y_ref[:, :live] - y_ov[:, :live])))
    print(f"[ov-parity-masked] device={device:<6} T={T:>3} live={live:>2}  "
          f"max-abs(live)={max_abs:.3e}  max-abs(masked)={masked_max:.3e}")
    assert max_abs < 1e-5, f"OV hc masked parity failed device={device} T={T}: {max_abs:.3e}"

    # (b) garbage probe: corrupt ONLY masked rows (they are already zero --
    # this is a DIFFERENT row content, not a re-zeroing) and re-run. If any
    # op mixed positions, a live row would move; the block is row-local, so
    # every live row must move by exactly 0.0.
    probe = x_masked.clone()
    probe[:, live:] = torch.randn(1, T - live, C * H) * 1000.0
    out2 = compiled({"hyper_input": probe.float().numpy()})
    y_probe = out2[compiled.output(0)]
    drift = float(np.max(np.abs(y_probe[:, :live] - y_ov[:, :live])))
    print(f"[mask-garbage-probe] device={device:<6} T={T:>3}  live-row drift={drift:.3e}")
    assert drift == 0.0, (
        f"mask isolation broken: masked-row garbage moved live rows by {drift:.3e} "
        f"(device={device} T={T}) -- the block must be row-local"
    )
