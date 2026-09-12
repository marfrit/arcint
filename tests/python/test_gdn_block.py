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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import openvino as ov  # noqa: E402
from q4e_device import compile_for, device_params  # noqa: E402
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


@pytest.mark.parametrize("T", [64, 65, 66, 96])
@pytest.mark.parametrize("device", _device_params())
def test_gdn_ov_parity(device, T):
    """The emitted OV GDN model against the f64 TRUTH, gated RELATIVELY against
    the f32 reference's own rounding.

    THE GPU ACCEPTANCE DOCTRINE, decided 2026-09-12 by measurement (window-050
    §4.2). The question was whether a GPU leg should be judged by
    equality-to-reference (|ov - ref_f32|, what this cell used to assert at a
    flat 1e-5) or by distance-to-truth (|ov - ref_f64|). Both sides were
    measured against an f64 recomputation at T=64/96/128/256 on CPU and both
    cards:

        device  T     |ov-r64|    |r32-r64|    |ov-r32|   ov/floor
        CPU     64   5.7251e-07  4.4179e-07  5.6624e-07       1.30
        CPU     96   8.3250e-07  9.4986e-07  9.3132e-07       0.88
        CPU    128   7.5987e-07  8.0206e-07  9.9838e-07       0.95
        CPU    256   1.6980e-06  9.1336e-07  1.4603e-06       1.86
        GPU.0   64   7.1151e-07  4.4179e-07  6.8545e-07       1.61
        GPU.0   96   7.4504e-02  9.4986e-07  7.4504e-02   78437.26
        GPU.0  128   7.5063e-02  8.0206e-07  7.5063e-02   93588.29
        GPU.0  256   1.2177e-01  9.1336e-07  1.2177e-01  133325.58
        GPU.1  96   7.4505e-02  9.4986e-07  7.4505e-02   78437.33   (etc)

    The numbers force the choice, three ways:

      1. The f32 REFERENCE IS SOUND -- 4.42e-07 to 9.50e-07 from f64 at every
         T, on every device (it is torch on CPU, so device-independent). It is
         a legitimate yardstick, which is what makes a relative gate possible
         at all.
      2. WHERE THE DEFECT LIVES THE TWO CRITERIA ARE INDISTINGUISHABLE:
         |ov-r32| and |ov-r64| agree to four significant figures at T >= 96,
         because the error dwarfs both floors. So the defect cannot decide it.
      3. WHERE THEY DIFFER, THE ABSOLUTE GATE IS THE UNANCHORED ONE. The floor
         itself MOVES with T (4.42e-07 -> 9.50e-07), so a flat `< 1e-5` is a
         drifting standard: it would pass a result 20x the floor at T=64 and
         10x at T=96 and call both the same thing.

    So: DISTANCE TO TRUTH, gated at 20x the reference's own f32 rounding --
    the FIX-GDN-UTINV doctrine (commit 1f075f0), now applied to the device
    legs. The gate has a denominator that is itself measured every run, so a
    drifting yardstick cannot satisfy it, and a second assert fails outright if
    the f32 reference leaves the float floor.

    T=65 and T=66 are parametrised because they are the boundary pair that
    pins the failure. Measured on GPU.0:

        T    pad   max-abs      first bad row   n bad
        64     0   6.8545e-07        -1            0
        65    63   1.1735e-06        -1            0
        66    62   3.6024e-02        65            1
        67    61   2.5580e-02        65            2
        96    32   7.4504e-02        65           31

    Row 64 -- the FIRST row of the second chunk -- is always correct
    (1.341e-07). Row 65 is always the first wrong one, and the count is exactly
    T-65. So the inter-chunk state ARRIVES correct and the corruption begins at
    the first row that mixes the carried state with the in-chunk accumulation.
    That refines, and does not confirm, the earlier "the carry is the
    divergence point" reading. The op responsible is NOT identified; what is
    established is the row, the shape threshold, and that CPU is at the float
    floor at every one of these shapes.
    """
    _assert_pin()
    config = _make_config()
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)

    x = torch.randn(1, T, config.hidden_size)
    mask = torch.ones(1, T, dtype=torch.long)
    with torch.no_grad():
        y_ref = ref(x, mask).float().numpy()
        # the f64 TRUTH, from the SAME weights -- the doctrine's denominator
        ref64 = ref_gdn.Qwen4ExpTextGatedDeltaNet(config, layer_idx=0).eval().double()
        ref64.load_state_dict({k: v.double() for k, v in ref.state_dict().items()})
        y_64 = ref64(x.double(), mask).numpy()

    model = gdn.build_gdn_model(config, state, seq_len=T)
    core = ov.Core()
    compiled = compile_for(core, model, device)
    out = compiled({
        "hidden_states": x.float().numpy(),
        "attention_mask": mask.float().numpy(),
    })
    y_ov = out[compiled.output(0)]

    max_abs = float(np.max(np.abs(y_ref - y_ov)))
    d_ov64 = float(np.max(np.abs(np.asarray(y_ov, np.float64) - y_64)))
    floor = float(np.max(np.abs(y_ref.astype(np.float64) - y_64)))
    rows_bad = int((np.max(np.abs(np.asarray(y_ov, np.float64) - y_64),
                           axis=-1)[0] > 1e-4).sum())
    first_bad = int(np.argmax(np.max(np.abs(np.asarray(y_ov, np.float64) - y_64),
                                     axis=-1)[0] > 1e-4)) if rows_bad else -1

    # KLD on a small fixed softmax head over the hidden dim.
    rng = np.random.default_rng(0)
    head = rng.standard_normal((config.hidden_size, 128)).astype(np.float32)
    kld = _kld(
        y_ref.reshape(-1, config.hidden_size) @ head,
        y_ov.reshape(-1, config.hidden_size) @ head,
    )
    ratio = d_ov64 / floor if floor > 0 else float("inf")
    print(f"\n[ov-parity] device={device:<6} T={T:>3}  |ov-r64|={d_ov64:.4e}  "
          f"|r32-r64|={floor:.4e}  |ov-r32|={max_abs:.4e}  ratio={ratio:.2f}x  "
          f"KLD={kld:.3e}  rows>1e-4 {rows_bad}/{T} first={first_bad}")
    assert floor < 1e-5, (
        f"the f32 REFERENCE itself left the float floor ({floor:.3e}) -- the "
        f"denominator of this gate is broken, stop before reading the emitter")
    assert d_ov64 <= 20.0 * floor, (
        f"OV GDN on {device} at T={T} is {ratio:.0f}x the reference's own f32 "
        f"rounding ({d_ov64:.4e} vs {floor:.4e}); {rows_bad}/{T} rows past "
        f"1e-4, first at row {first_bad}. GPU acceptance is distance-to-truth "
        f"at 20x the floor -- see this cell's docstring for the measurement "
        f"that forced the doctrine")
