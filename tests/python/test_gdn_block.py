"""E2 increment 1 — GatedDeltaNet (GDN) block OV-emission parity test.

Red-first harness for `tools/q4e/gdn.py` (the opset-13 emission of the
qwen4_exp full-sequence GDN block). It is RED until `q4e.gdn` exists: the
package `q4e/__init__.py` does `from . import gdn`, so importing anything from
`q4e` raises `ModuleNotFoundError: No module named 'q4e.gdn'` until the emitter
lands. That import failure is this file's intended red.

Oracle discipline (constitution: receipts lie, stores don't):
  The numeric oracle is the *installed* pinned transformers reference
  (`transformers.models.qwen4_exp`, transcribed no-cache branch in
  `tools/q4e/ref_gdn.py`). `_assert_pin()` hashes the installed
  modeling/configuration files against the pin sha256 BEFORE every numeric
  table, so a silently-swapped reference goes red rather than reporting a
  number against the wrong math.

Legs:
  * transcription-vs-pin: `ref_gdn.Qwen4ExpTextGatedDeltaNet` (the transcription
    the OV graph mirrors) must equal the pin's own class on identical random
    weights/inputs at 1e-5. This validates the transcription against the pin,
    not against itself.
  * OV parity: the emitted OV model vs the transcription at atol=1e-5, with a
    max-abs table and a KLD on a small softmax head, at T=64 (1 chunk) and
    T=96 (2 chunks, 32 pad rows — the pad-row KLD no-op is checked here).

Devices:
  CPU is always run (host RAM, touches no card). GPU legs (GPU.0/GPU.1) run
  ONLY when Q4E_GPU is set to a comma list of device names, because loading a
  graph onto a card requires stopping the resident service — a GPU window,
  which is operator/frontier territory, not the engineer seat. The GPU column
  is therefore filled in a window, not by this seat.

Run (dev-host venv):
    Q4E_GPU=        ~/openarc-venv/bin/python3 -m pytest tests/python/test_gdn_block.py -s
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

# The RED import: q4e/__init__.py -> `from . import gdn`; fails until gdn.py
# is emitted. ref_gdn is the (present) transcription but is unreachable through
# the package until the same import succeeds.
from q4e import gdn, ref_gdn  # noqa: E402


# ---------------------------------------------------------------------------
# Oracle pin
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
# Small but architecturally faithful GDN config
# ---------------------------------------------------------------------------
def _make_config():
    """A single-layer linear-attention config small enough for CPU parity but
    faithful where the GDN math depends on geometry: value/key head ratio 2
    exercises the repeat_interleave path, conv kernel 4 the depthwise conv."""
    return pin_cfg.Qwen4ExpTextConfig(
        hidden_size=256,
        num_hidden_layers=1,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        output_gate_type=None,
        layer_types=["linear_attention"],
    )


def _ref_and_pin(config, seed: int = 0):
    """Build the transcription and the pin's own GDN class with identical
    random weights, in eval mode."""
    torch.manual_seed(seed)
    ref = ref_gdn.Qwen4ExpTextGatedDeltaNet(config, layer_idx=0).eval()
    pin = pin_mod.Qwen4ExpTextGatedDeltaNet(config, layer_idx=0).eval()
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
    """The transcription mirrors the pin's own class (validated against the
    pin, not against itself)."""
    _assert_pin()
    config = _make_config()
    ref, pin = _ref_and_pin(config)
    print("\n[transcription-vs-pin] pin sha OK; max-abs by T")
    for T in (64, 96):
        x = torch.randn(1, T, config.hidden_size)
        mask = torch.ones(1, T, dtype=torch.long)
        with torch.no_grad():
            yr = ref(x, mask)
            yp = pin(x, cache_params=None, cache_position=None, attention_mask=mask)
        if isinstance(yp, tuple):
            yp = yp[0]
        md = (yr - yp).abs().max().item()
        print(f"  T={T:>3}  max-abs(ref - pin) = {md:.3e}")
        assert md < 1e-5, f"transcription drifted from pin at T={T}: {md:.3e}"


@pytest.mark.parametrize("T", [64, 96])
@pytest.mark.parametrize("device", _device_params())
def test_gdn_ov_parity(device, T):
    """The emitted OV GDN model equals the transcription at atol=1e-5, with a
    max-abs + KLD table, at T=64 (1 chunk) and T=96 (2 chunks / pad rows)."""
    _assert_pin()
    config = _make_config()
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)

    x = torch.randn(1, T, config.hidden_size)
    mask = torch.ones(1, T, dtype=torch.long)
    with torch.no_grad():
        y_ref = ref(x, mask).float().numpy()

    model = gdn.build_gdn_model(config, state, seq_len=T)
    core = ov.Core()
    compiled = core.compile_model(model, device)
    out = compiled({
        "hidden_states": x.float().numpy(),
        "attention_mask": mask.float().numpy(),
    })
    y_ov = out[compiled.output(0)]

    max_abs = float(np.max(np.abs(y_ref - y_ov)))

    # KLD on a small fixed softmax head over the hidden dim.
    rng = np.random.default_rng(0)
    head = rng.standard_normal((config.hidden_size, 128)).astype(np.float32)
    kld = _kld(
        y_ref.reshape(-1, config.hidden_size) @ head,
        y_ov.reshape(-1, config.hidden_size) @ head,
    )
    print(f"\n[ov-parity] device={device:<6} T={T:>3}  max-abs={max_abs:.3e}  KLD={kld:.3e}")
    assert max_abs < 1e-5, f"OV GDN parity failed device={device} T={T}: {max_abs:.3e}"
