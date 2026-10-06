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


def test_the_emitter_follows_the_configured_output_gate(device="CPU", T=64):
    """The output gate's activation is the config's `output_gate_type` (the
    pin: `output_gate_type or hidden_act`, RMSNormGated line 14). The shipped
    Flash-Next checkpoint gates with a SIGMOID -- llama.cpp hard-codes it for
    this architecture ("the one numerical difference from Qwen3.5's GDN:
    sigmoid output gate, not silu", `src/models/qwen4exp.cpp`, `code`) and
    the HF config carries no `output_gate_type` in the GGUF's metadata, so our
    real-geometry config had defaulted to silu. Measured on the dev host
    (France ids, depth 1, campaign serving-shape-logits): with the silu gate
    the pin's gated-norm output correlates 0.81 with llama.cpp's; see the
    campaign record for the sigmoid reading. Red first: the emitter emitted
    silu unconditionally, so a sigmoid-configured reference differs from it."""
    config = _make_config()
    config.output_gate_type = "sigmoid"
    ref, pin = _ref_and_pin(config)
    state = _state_np(ref)
    x = torch.randn(1, T, config.hidden_size)
    mask = torch.ones(1, T, dtype=torch.long)
    with torch.no_grad():
        y_ref = ref(x, mask).float().numpy()
        y_pin = pin(x, cache_params=None, attention_mask=mask.float()).float().numpy()
    assert float(np.max(np.abs(y_ref - y_pin))) == 0.0, "transcription != pin under sigmoid"
    model = gdn.build_gdn_model(config, state, seq_len=T)
    compiled = compile_for(ov.Core(), model, device)
    y_ov = compiled({"hidden_states": x.float().numpy(),
                     "attention_mask": mask.float().numpy()})[compiled.output(0)]
    max_abs = float(np.max(np.abs(y_ref - y_ov)))
    print(f"\n[gate-sigmoid] T={T} |ov-ref|={max_abs:.3e}")
    assert max_abs < 1e-5, f"emitter vs sigmoid-gated reference: {max_abs:.3e}"
    config.output_gate_type = "swish"
    with pytest.raises(ValueError):
        gdn.build_gdn_model(config, state, seq_len=T)


def test_the_emitter_follows_the_configured_key_head_pairing(device="CPU", T=64):
    """`gdn_key_head_map` (2026-09-18, campaign serving-shape-logits): how the
    HK key heads serve the HV value heads. The pin interleaves (value head h
    <- key head h // r); the shipped GGUF is computed TILED by llama.cpp
    (h <- h % HK), and only tiled does the real model's layer 0 agree with
    llama.cpp's whole tensors (48/48 core heads; interleaved, 4/48). The
    transcription carries the option as a documented deviation; the emitter
    must follow it. Red first: the emitter interleaved regardless, so a
    tiled reference differed from it by O(1)."""
    config = _make_config()
    config.gdn_key_head_map = "tiled"
    ref, pin = _ref_and_pin(config)
    state = _state_np(ref)
    x = torch.randn(1, T, config.hidden_size)
    mask = torch.ones(1, T, dtype=torch.long)
    with torch.no_grad():
        y_ref = ref(x, mask).float().numpy()
        y_pin = pin(x, cache_params=None, attention_mask=mask.float()).float().numpy()
    # the option changes the computation (r = 2 at this geometry)
    assert float(np.max(np.abs(y_ref - y_pin))) > 1e-3, "tiled == interleaved: the option is inert"
    model = gdn.build_gdn_model(config, state, seq_len=T)
    compiled = compile_for(ov.Core(), model, device)
    y_ov = compiled({"hidden_states": x.float().numpy(),
                     "attention_mask": mask.float().numpy()})[compiled.output(0)]
    max_abs = float(np.max(np.abs(y_ref - y_ov)))
    print(f"\n[key-head-map tiled] T={T} |ov-ref|={max_abs:.3e} |ref-pin|={float(np.max(np.abs(y_ref - y_pin))):.3e}")
    assert max_abs < 1e-5, f"emitter vs tiled reference: {max_abs:.3e}"
    config.gdn_key_head_map = "shuffled"
    with pytest.raises(ValueError):
        gdn.build_gdn_model(config, state, seq_len=T)


# ---------------------------------------------------------------------------
# THE CHUNK-AXIS EMISSION MODES (the de-batch spike, 2026-09-12)
# ---------------------------------------------------------------------------

def test_the_chunk_emission_modes_agree_on_cpu_and_differ_in_node_count():
    """All four `ut_mode`s are the same arithmetic; only the shapes ops see
    differ. Both halves of that are asserted, because both are load-bearing.

    WHY THIS EXISTS. On both Arc cards every multi-chunk static-T GDN graph is
    corrupt from global row 65 onward, exactly T-65 rows, while T=64 is clean
    at the f32 floor (frontier card pass, ~/win-050/FINDINGS). The spike that
    chased it needed four emissions of one algebra, so the card could be asked
    which SHAPE it mishandles rather than which arithmetic. That question only
    means anything if the four really are one algebra, which is this cell.

    The node counts are GENERATED here, not recited: "perchunk costs nodes" is
    a claim about the graph and it dies here if it stops being true.
    """
    config = _make_config()
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)
    modes = ("batched", "folded", "debatched", "perchunk")

    core = ov.Core()
    print("\n[ut-modes] node count by mode, generated from the emitted graph")
    for T in (64, 96, 128):
        x = torch.randn(1, T, config.hidden_size)
        mask = torch.ones(1, T, dtype=torch.long)
        with torch.no_grad():
            y_ref = ref(x, mask).float().numpy().astype(np.float64)
        nodes, outs = {}, {}
        for mode in modes:
            m = gdn.build_gdn_model(config, state, seq_len=T, ut_mode=mode)
            nodes[mode] = len(m.get_ordered_ops())
            c = compile_for(core, m, "CPU")
            outs[mode] = np.asarray(
                c({"hidden_states": x.float().numpy(),
                   "attention_mask": mask.float().numpy()})[c.output(0)], np.float64)
        chunks = (T + (gdn.CHUNK - T % gdn.CHUNK) % gdn.CHUNK) // gdn.CHUNK
        print(f"  T={T:>3} C={chunks}  "
              + "  ".join(f"{m}={nodes[m]}" for m in modes)
              + "   max|mode-batched| "
              + " ".join(f"{m}={np.max(np.abs(outs[m] - outs['batched'])):.2e}"
                         for m in modes[1:]))

        for mode in modes[1:]:
            spread = float(np.max(np.abs(outs[mode] - outs["batched"])))
            assert spread < 1e-4, (
                f"ut_mode={mode!r} at T={T} disagrees with 'batched' by "
                f"{spread:.3e} on CPU. The modes must be one algebra emitted "
                f"four ways -- if they are not, the card comparison they exist "
                f"for compares two different computations and means nothing.")
        if chunks == 1:
            assert nodes["debatched"] == nodes["batched"] + 9, (
                f"at C=1 'debatched' is one unroll like 'batched', plus the "
                f"slice/reshape pair: expected {nodes['batched'] + 9} nodes, "
                f"got {nodes['debatched']}")
            assert nodes["perchunk"] <= nodes["batched"], (
                f"at C=1 'perchunk' has no chunk axis to hoist and must not "
                f"cost more than 'batched': {nodes['perchunk']} vs "
                f"{nodes['batched']}")
        else:
            assert nodes["perchunk"] > nodes["batched"], (
                f"'perchunk' emits C unrolls and must cost more nodes than the "
                f"single batched one at C={chunks}: {nodes['perchunk']} vs "
                f"{nodes['batched']}. If they are equal the mode is not taking "
                f"effect and every green it reports is the batched path's.")

    # THE GROWTH LAW, generated, because it is the number that says where this
    # form stops being usable. 'perchunk' is the default and it emits one
    # ~1,900-op unroll PER CHUNK, so the block grows linearly in C; projected
    # over the 48-layer stack's 36 GDN blocks it reaches millions of nodes at
    # serving prefill lengths, and the route there is the stateful-prefill
    # increment rather than a bigger static graph. gdn.py's header carries that
    # table; this keeps its shape honest.
    counts = {}
    for T in (64, 128, 192, 256):
        c_of_t = (T + (gdn.CHUNK - T % gdn.CHUNK) % gdn.CHUNK) // gdn.CHUNK
        counts[c_of_t] = len(gdn.build_gdn_model(
            config, state, seq_len=T, ut_mode="perchunk").get_ordered_ops())
    cs = sorted(counts)
    steps = [counts[b] - counts[a] for a, b in zip(cs, cs[1:])]
    print("[ut-modes] perchunk nodes by C: "
          + " ".join(f"C{c}={counts[c]}" for c in cs)
          + f"  -> per-chunk increments {steps}")
    # Linear from C=2 on. The C1 -> C2 step is ONE node larger than the rest
    # and that is not noise: at C == 1 the emitter returns `cores[0]` directly,
    # so the `concat` over chunks does not exist yet and appears exactly once,
    # at C == 2. Asserting all increments equal is what this cell did first,
    # and it went red on [1905, 1904, 1904] -- the graph was right and the
    # claim was too strong.
    steps_from_two = steps[1:]
    assert len(set(steps_from_two)) == 1, (
        f"'perchunk' node growth is not linear in C for C >= 2 (increments "
        f"{steps_from_two}). The cost table in gdn.py's header, and the "
        f"projection that says this form is not viable at prefill lengths, "
        f"both assume exactly one unroll per chunk -- re-derive them before "
        f"changing this.")
    assert steps[0] == steps_from_two[0] + 1, (
        f"the C1 -> C2 step is {steps[0]} against {steps_from_two[0]} for every "
        f"later chunk; it should be exactly one larger (the chunk `concat` that "
        f"C == 1 does not emit). A different gap means the C == 1 path and the "
        f"C > 1 path have diverged by more than that concat.")


def test_the_served_prefill_graph_does_not_grow_with_the_prompt_length():
    """LYON's compile-once property (0.5.4). The served prefill graph carries
    its GDN state in Variables and its chunk axis is dynamic, so ONE compile is
    replayed across every prompt length. The static `perchunk` form unrolls
    ~1,900 ops PER CHUNK (`code`: docs/window-050.md §4.2) -- 2,198,232 nodes at
    T=2048 over the 48-layer stack -- which is the 2.2M-node cost LYON retires.

    Red-first: the `perchunk` control asserts the static form GROWS with T (the
    incumbent red); the served path asserts it does not. This cell fails if the
    served emitter stops being dynamic/stateful, or if `perchunk` stops
    unrolling (i.e. the defect's reproducer silently changed)."""
    from openvino import opset13 as ovop
    from q4e import serving_shape as ss

    config = _make_config()
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)
    H = config.hidden_size
    beam = ovop.constant(np.array([0], np.int64))

    # the served path: dynamic in T, stateful conv + stateful (Loop) core
    sinks = []
    hidden = ovop.parameter([1, -1, H], ov.Type.f32)
    amask = ovop.parameter([1, -1], ov.Type.f32)
    out = gdn.emit_gdn(hidden, amask, config, state, None,
                       conv_emitter=ss.stateful_short_conv(0, beam, sinks),
                       core_emitter=ss.stateful_gdn_core(0, beam, sinks))
    served = ov.Model([ovop.result(out)], list(sinks), [hidden, amask], "served")
    n_served = len(served.get_ordered_ops())
    types = {n.get_type_name() for n in served.get_ordered_ops()}
    assert "Loop" in types and "Assign" in types and "ReadValue" in types, types
    assert len(sinks) == 2, f"{len(sinks)} Assign(s); the two stateful hooks carry two states"

    # the incumbent static form: grows ~1,900 ops per chunk
    static = {T: len(gdn.build_gdn_model(config, state, seq_len=T,
                                         ut_mode="perchunk").get_ordered_ops())
              for T in (64, 256, 512)}
    print(f"\n[lyon] served(dynamic T) nodes={n_served}  "
          f"static perchunk nodes={static}  "
          f"per-chunk growth={static[512] - static[64]} over 448 tokens")
    assert static[512] > static[256] > static[64], (
        f"the perchunk control no longer unrolls per chunk: {static}")
    assert n_served < static[64], (
        f"the served path ({n_served} nodes) is not smaller than one perchunk "
        f"chunk ({static[64]}); the stateful route did not retire the unroll")

    # 0.5.4 LYON: the MULTI-BLOCK served core must be T-invariant too -- chunk-
    # count shaped, NOT token-count shaped. Build it dynamic and assert the
    # node count does not move with the length asked for, and that neither
    # served core grows one node per token (the unroll would).
    def served_with(core_emitter):
        s2 = []
        h = ovop.parameter([1, -1, H], ov.Type.f32)
        a = ovop.parameter([1, -1], ov.Type.f32)
        o = gdn.emit_gdn(h, a, config, state, None,
                         conv_emitter=ss.stateful_short_conv(0, beam, s2),
                         core_emitter=core_emitter(0, beam, s2))
        return len(ov.Model([ovop.result(o)], list(s2), [h, a], "t2").get_ordered_ops())

    n_chunked = served_with(ss.stateful_gdn_core_chunked)
    print(f"[lyon] served sequential={n_served} chunked={n_chunked} nodes "
          f"(both T-invariant; perchunk would add {static[512] - static[64]} per 448 tokens)")
    assert n_chunked < static[64], (
        f"the chunked served path ({n_chunked}) must stay smaller than one "
        f"perchunk chunk ({static[64]}) -- it must NOT be token-count shaped")
    # neither served core may grow with prompt length: both are dynamic in T,
    # so a node added per token is impossible unless the emitter regressed
    for n in (n_served, n_chunked):
        assert n < static[64] + 64, (
            f"a served core read {n} nodes, near the per-chunk unroll's own "
            f"size -- the stateful route is not holding")


def test_the_chunked_stateful_core_is_the_chunked_algebra_across_boundaries():
    """LYON: the multi-block stateful core (`stateful_gdn_core_chunked`) must
    be the CHUNKED algebra (`gdn.py`'s `perchunk` core) with the state carried
    across chunks -- so it is byte-exact against it -- and only f32-bounded
    against the token-sequential core (different summation order; byte-exact is
    impossible there and that is recorded, not weakened).

    Lengths chosen to cross MULTIPLE chunk boundaries and to include a
    non-multiple (T=192 is 3 chunks exactly; T=224 is 3.5 -> the zero-pad
    path), not one token and not one chunk."""
    import openvino as ov
    from openvino import opset13 as ovop
    from q4e import serving_shape as ss

    config = _make_config()
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)
    beam = ovop.constant(np.array([0], np.int64))
    core = ov.Core()

    for T in (128, 192, 224, 256):
        x = torch.randn(1, T, config.hidden_size)
        mask = torch.ones(1, T, dtype=torch.long)
        with torch.no_grad():
            ref64 = ref_gdn.Qwen4ExpTextGatedDeltaNet(config, layer_idx=0).eval().double()
            ref64.load_state_dict({k: v.double() for k, v in ref.state_dict().items()})
            y_64 = ref64(x.double(), mask).numpy().astype(np.float64)
        with torch.no_grad():
            y_ref_inst = ref(x, mask).float().numpy().astype(np.float64)
        floor = float(np.max(np.abs(y_ref_inst - y_64)))
        feed = {"hidden_states": x.float().numpy(),
                "attention_mask": mask.float().numpy()}

        y_ref = np.asarray(compile_for(core, gdn.build_gdn_model(
            config, state, seq_len=T), "CPU")(feed)[0], np.float64)

        def _stateful(emitter):
            # build_gdn_model wires only the CORE hook (its conv stays the
            # default unroll), so one state is carried -- the same contract
            # the existing sequential-vs-chunked parity cell uses.
            sinks = []
            m = gdn.build_gdn_model(
                config, state, seq_len=T, sinks=sinks,
                core_emitter=emitter(0, beam, sinks))
            assert len(sinks) == 1, f"{len(sinks)} Assign(s); the core carries one state"
            assert len(m.get_variables()) == 1, m.get_variables()
            return np.asarray(compile_for(core, m, "CPU")(feed)[0], np.float64)

        y_seq = _stateful(ss.stateful_gdn_core)
        y_chunk = _stateful(ss.stateful_gdn_core_chunked)

        d_chunk = float(np.max(np.abs(y_chunk - y_ref)))
        d_seq = float(np.max(np.abs(y_seq - y_ref)))
        print(f"\n[lyon-chunk] T={T:4d}  |chunked-stateful - chunked|={d_chunk:.4e}  "
              f"|sequential - chunked|={d_seq:.4e}  floor={floor:.4e}")

        # 1. the chunked stateful core IS the chunked algebra: byte-exact
        assert np.array_equal(y_chunk, y_ref), (
            f"T={T}: the chunked stateful core is NOT byte-exact against the "
            f"chunked algebra (max |diff| {d_chunk:.4e}); the body reused the "
            f"wrong ops or the state merge lost precision")
        # 2. against the token-sequential core it is f32-bounded, NOT byte-exact
        #    -- different summation order; recorded as the finding, not weakened
        assert d_seq <= 20.0 * floor, (
            f"T={T}: the token-sequential core left the bound ({d_seq:.4e} vs "
            f"{20.0 * floor:.4e}); floor {floor:.4e}")
        assert d_seq > 0.0, (
            f"T={T}: the sequential and chunked cores are bitwise identical, "
            f"which they cannot be -- the parity cell is comparing one graph "
            f"with itself")


def test_the_chunked_served_core_leaves_no_dangling_beam_idx():
    """LYON beam-free fix, artifact-level. Measured on the A770 (2026-09-26):
    the chunked artifact was REFUSED with `Model references undeclared
    parameters: beam_idx` -- the unfused chunked Loop kept its
    `ReadValue -> Gather(beam_idx)` chain into the paged-attention rewrite,
    which drops the declaration (`backend_ov.cpp`:2637). The chunked served
    core must therefore reference `beam_idx` NOWHERE, so the rewrite has
    nothing to dangle. Red-first: fails if the reference returns.

    Built with the default (beam-free) conv so the `beam_idx` Parameter has no
    OTHER consumer: any consumer found here is the GDN core's own."""
    from openvino import opset13 as ovop
    from q4e import serving_shape as ss

    config = _make_config()
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)
    beam = ovop.parameter([-1], ov.Type.i32)
    beam.set_friendly_name("beam_idx")

    def consumers(emitter):
        # static T + CORE hook only: the default conv carries no beam, so every
        # consumer found is the core's own, and `beam_idx` can be declared
        # without the model becoming unregistered either way
        sinks = []
        h = ovop.parameter([1, 128, config.hidden_size], ov.Type.f32)
        a = ovop.parameter([1, 128], ov.Type.f32)
        o = gdn.emit_gdn(h, a, config, state, 128,
                         core_emitter=emitter(0, beam, sinks))
        m = ov.Model([ovop.result(o)], list(sinks), [h, a, beam], "gdn_core")
        # the consumers of the beam OUTPUT are the ops that take it as input
        return sorted(i.get_node().get_type_name()
                      for i in beam.output(0).get_target_inputs())

    c_cons = consumers(ss.stateful_gdn_core_chunked)
    s_cons = consumers(ss.stateful_gdn_core)
    print(f"\n[lyon-beam] chunked beam_idx consumers={c_cons}  "
          f"sequential beam_idx consumers={s_cons}")
    assert c_cons == [], (
        f"the chunked served core still references beam_idx via {c_cons}; the "
        f"unfused Loop leaves that chain alive and the paged-attention rewrite "
        f"drops the declaration, refusing the artifact")
    assert s_cons == ["Gather"], (
        f"the sequential control's beam_idx consumers moved to {s_cons}; the "
        f"cell can no longer tell the two paths apart and is vacuous")


def test_an_unknown_chunk_emission_mode_is_refused():
    """The mode string reaches a comparison, not a silent default. A typo that
    fell through to 'batched' would report the defect as fixed."""
    config = _make_config()
    ref, _ = _ref_and_pin(config)
    with pytest.raises(ValueError, match="unknown ut emit mode"):
        gdn.build_gdn_model(config, _state_np(ref), seq_len=96,
                            ut_mode="per-chunk")


# The two sequence lengths the chunk-axis census is taken at, and the reason
# they are this large. A "live chunk axis" is detected by looking for an axis
# whose extent is C -- so C must be a value that occurs in the graph ONLY
# because it is the chunk count. Every integer from 1 to CHUNK-1 occurs inside
# `_ut_inverse` in EVERY mode (the forward substitution slices row i at width i
# and pads it at width chunk-i, for i = 1..chunk-1), CHUNK itself occurs
# everywhere, and the small head counts occur as leading axes (HK=2 and HV=4
# collide with C=2 and C=4, the first chunk counts anyone reaches for). So the
# smallest usable C is CHUNK+1, and the cell PROVES that rather than trusting
# it: each value must be absent from the other's build.
_CHUNK_SIG_TS = (4160, 4224)      # C = 65 and C = 66 at CHUNK = 64


def _output_shapes(model):
    """(op type, static output shape) for every output of every op in `model`."""
    out = []
    for node in model.get_ordered_ops():
        for o in range(node.get_output_size()):
            ps = node.get_output_partial_shape(o)
            if ps.rank.is_dynamic:
                continue
            out.append((node.get_type_name(),
                        tuple(int(d.get_length()) for d in ps if d.is_static)))
    return out


def _chunk_axis_census(shapes, c):
    """How many axes in the graph have extent exactly `c`, plus the rank-5 set.

    Counts EVERY axis at EVERY position, not a chosen one: the point is that
    the chunk axis is nowhere, so a walk that skipped positions could pass the
    claim by not looking.
    """
    hits = sum(s.count(c) for _, s in shapes)
    rank5 = [s for _, s in shapes if len(s) == 5]
    return hits, rank5, [s for s in rank5 if c in s]


def test_no_emitted_op_sees_a_live_chunk_axis_under_perchunk():
    """K0 (REVIEW 9162ac9). gdn.py's header claims, of the DEFAULT emission,
    that "no emitted op ever sees a live chunk axis". That is the whole
    mechanism of the row-65 fix and nothing held it. This does.

    THE DISCRIMINATOR AND ITS TRAP. Asking "is C in this shape" is not a test:
    at C=2 it collides with HK, at C=4 with HV, and at any C below CHUNK with
    the forward substitution's own row sweep, which emits every extent from 1
    to CHUNK-1 in every mode. The census therefore runs at C = 65 and C = 66,
    and the first thing it asserts is that 65 does not occur in the C=66 graph
    and 66 does not occur in the C=65 graph -- which is what makes each of them
    a C-BUILD SIGNATURE rather than a number that happens to be in a shape.

    BOTH DIRECTIONS, in one cell, because a one-directional probe is vacuous --
    a walk that found nothing at all would pass "perchunk has zero". So
    `batched` must come back NON-zero from the same walk, and the rank-5
    positive control below must come back non-empty.

    THE FOUR RANK-5 OUTPUTS IN `perchunk` ARE NOT CHUNK AXES. They are the
    GQA repeat's ([1, T, HK, 1, Dk] and [1, T, HK, ratio, Dk], query and key),
    and that is asserted here from their shapes rather than asserted in prose.
    """
    config = _make_config()
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)
    HK = config.linear_num_key_heads
    Dk = config.linear_key_head_dim
    ratio = config.linear_num_value_heads // HK

    cs, census = [], {}
    for T in _CHUNK_SIG_TS:
        c = (T + (gdn.CHUNK - T % gdn.CHUNK) % gdn.CHUNK) // gdn.CHUNK
        cs.append(c)
        for mode in ("batched", "perchunk"):
            shapes = _output_shapes(
                gdn.build_gdn_model(config, state, seq_len=T, ut_mode=mode))
            census[(c, mode)] = (shapes,) + _chunk_axis_census(shapes, c)

    c1, c2 = cs
    assert c1 != c2 and min(cs) > gdn.CHUNK, (
        f"the census needs two DIFFERENT chunk counts, both above CHUNK="
        f"{gdn.CHUNK}: got {cs}. Below CHUNK the value collides with the "
        f"forward substitution's row sweep and the discriminator is void.")

    print(f"\n[chunk-axis] CHUNK={gdn.CHUNK} HK={HK} HV="
          f"{config.linear_num_value_heads} ratio={ratio}; signatures C={c1}, C={c2}")
    for c in cs:
        for mode in ("batched", "perchunk"):
            shapes, hits, rank5, rank5_c = census[(c, mode)]
            other = c2 if c == c1 else c1
            cross = sum(s.count(other) for _, s in shapes)
            print(f"  C={c:>3} {mode:>9}  outputs={len(shapes):>6}  "
                  f"axes==C {hits:>5}  rank5 {len(rank5):>5}  rank5-with-C "
                  f"{len(rank5_c):>5}  axes=={other} (cross) {cross:>3}")

    # 1. THE TRAP: each signature must be a property of ITS OWN build only.
    for c in cs:
        other = c2 if c == c1 else c1
        for mode in ("batched", "perchunk"):
            shapes = census[(c, mode)][0]
            cross = sum(s.count(other) for _, s in shapes)
            assert cross == 0, (
                f"C={other} occurs {cross} times in the C={c} {mode} graph, so "
                f"it is NOT a chunk-axis signature -- it collides with a "
                f"config dimension or an unroll slice width. Pick chunk counts "
                f"that occur nowhere else before reading anything below.")

    # 2. BOTH DIRECTIONS from the same walk.
    for c in cs:
        b_hits = census[(c, "batched")][1]
        p_hits, p_r5, p_r5c = census[(c, "perchunk")][1:]
        assert b_hits > 0, (
            f"the 'batched' graph at C={c} shows NO axis of extent {c}. That "
            f"emission is defined by carrying the chunk axis, so a zero here "
            f"means this walk cannot see chunk axes at all and the perchunk "
            f"result below is vacuous.")
        assert p_hits == 0, (
            f"'perchunk' at C={c} has {p_hits} axes of extent {c}: an emitted "
            f"op DOES see a live chunk axis, and gdn.py's header claim -- the "
            f"whole mechanism of the row-65 fix -- is false. Shapes: "
            f"{sorted({s for _, s in census[(c, 'perchunk')][0] if c in s})}")
        # 3. ANTI-VACUITY on the same walk: rank-5 tensors still EXIST under
        #    perchunk, so the zero above is "no chunk axis", not "no shapes
        #    found". These four are the GQA repeat and nothing else.
        assert len(p_r5) == 4 and not p_r5c, (
            f"expected exactly 4 rank-5 outputs under perchunk (the GQA "
            f"repeat's two per projection) and none carrying the chunk axis; "
            f"got {len(p_r5)} rank-5 outputs, {len(p_r5c)} with C. If rank-5 "
            f"outputs vanished entirely this cell's zero proves nothing.")
        for s in p_r5:
            assert s[2] == HK and s[3] in (1, ratio) and s[4] == Dk, (
                f"rank-5 output {s} under perchunk is not a GQA repeat "
                f"([1, T, HK={HK}, 1 or ratio={ratio}, Dk={Dk}]). A NEW rank-5 "
                f"shape appeared and this cell cannot vouch for it being "
                f"chunk-free by position -- identify it before trusting the "
                f"census.")

    # 4. The batched count is STRUCTURAL, not a function of how many chunks
    #    there are: one unroll is emitted whatever C is, so the number of ops
    #    carrying the chunk axis must not move between the two builds. (The
    #    literal is generated above; the reviewer's 256 was at C=4, where the
    #    graph is smaller -- it is not an invariant and is not asserted.)
    b1, b2 = census[(c1, "batched")][1], census[(c2, "batched")][1]
    assert b1 == b2, (
        f"'batched' carries the chunk axis on {b1} axes at C={c1} but {b2} at "
        f"C={c2}. One unroll is emitted regardless of C, so this count is "
        f"structural; if it moved, the batched emission changed shape and the "
        f"comparison this cell draws is no longer between the same two things.")


# ---------------------------------------------------------------------------
# THE SERVING CORE, against the same truth the chunked one answers to
# ---------------------------------------------------------------------------

def test_the_sequential_serving_core_is_the_same_gdn_as_the_chunked_one():
    """CF-GDNSEQ. The serving-shape emitter does NOT run this module's chunked
    delta rule: it runs a token-sequential `v5::Loop`, because that is the only
    form the serving path's fusion chain turns into
    `gated_delta_state_table.N`. Two forms of one recurrence, and the commit
    that introduced the second one wrote that the divergence "needs a gate of
    its own". This is that gate.

    It is NOT a comparison of the two emitters against each other. Judging the
    new core by the old one would make the pair self-consistent and say nothing
    about either: the sequential core is measured against the SAME f64 TRUTH,
    at the SAME 20x-the-f32-floor doctrine, that `test_gdn_ov_parity` holds the
    chunked core to. The chunked output is computed in the same run and printed
    beside it, as context rather than as the yardstick.

    Bitwise equality is NOT expected and is not asserted. DESIGN 3.2 already
    records that a k-token pass computes bitwise-different state from k
    one-token passes; the two differ in float association order and agree in
    arithmetic, which is exactly what "distance to truth" measures and what a
    comparison between them could not distinguish from both being wrong.
    """
    config = _make_config()
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)
    # T=64 is exactly CHUNK, so the chunked side never crosses a chunk
    # boundary there and the comparison row is weaker than it looks. T=96 is
    # two chunks with 32 pad rows -- the shape the chunk-mode cells use for the
    # same reason.
    for T in (64, 96):
        _one_sequential_core_leg(config, ref, state, T)


def _one_sequential_core_leg(config, ref, state, T):
    import openvino as ov
    from openvino import opset13 as ovop

    from q4e import serving_shape as ss

    x = torch.randn(1, T, config.hidden_size)
    mask = torch.ones(1, T, dtype=torch.long)
    with torch.no_grad():
        y_ref = ref(x, mask).float().numpy()
        ref64 = ref_gdn.Qwen4ExpTextGatedDeltaNet(config, layer_idx=0).eval().double()
        ref64.load_state_dict({k: v.double() for k, v in ref.state_dict().items()})
        y_64 = ref64(x.double(), mask).numpy()
    floor = float(np.max(np.abs(y_ref.astype(np.float64) - y_64)))

    core = ov.Core()
    feed = {"hidden_states": x.float().numpy(),
            "attention_mask": mask.float().numpy()}

    chunked = compile_for(core, gdn.build_gdn_model(config, state, seq_len=T),
                          "CPU")
    y_chunked = np.asarray(chunked(feed)[chunked.output(0)], np.float64)

    # The serving core wants a beam index and somewhere to put its Assign. A
    # constant beam is the single-lane case the serving path runs anyway, and
    # the fusion's own pattern has the Gather as optional.
    sinks = []
    seq_model = gdn.build_gdn_model(
        config, state, seq_len=T, sinks=sinks,
        core_emitter=ss.stateful_gdn_core(
            0, ovop.constant(np.array([0], np.int64)), sinks))
    assert len(sinks) == 1, f"{len(sinks)} Assign(s); the core carries one state"
    assert len(seq_model.get_variables()) == 1, seq_model.get_variables()
    sequential = compile_for(core, seq_model, "CPU")
    y_seq = np.asarray(sequential(feed)[sequential.output(0)], np.float64)

    d_seq = float(np.max(np.abs(y_seq - y_64)))
    d_chunked = float(np.max(np.abs(y_chunked - y_64)))
    spread = float(np.max(np.abs(y_seq - y_chunked)))
    print(f"\n[gdn-seq] T={T}  |sequential-r64|={d_seq:.4e}  "
          f"|chunked-r64|={d_chunked:.4e}  |r32-r64|={floor:.4e}  "
          f"ratio={d_seq / floor if floor else float('inf'):.2f}x  "
          f"|sequential-chunked|={spread:.4e}")
    print(f"[gdn-seq] nodes: chunked "
          f"{len(gdn.build_gdn_model(config, state, seq_len=T).get_ordered_ops())}"
          f", sequential {len(seq_model.get_ordered_ops())} "
          f"(+ a Loop body of "
          f"{len([n for n in seq_model.get_ordered_ops() if n.get_type_name() == 'Loop'][0].get_function().get_ordered_ops())})")

    assert floor < 1e-5, (
        f"the f32 REFERENCE itself left the float floor ({floor:.3e}) -- the "
        f"denominator of this gate is broken, stop before reading either core")
    assert d_seq <= 20.0 * floor, (
        f"the SEQUENTIAL serving core is {d_seq / floor:.0f}x the reference's "
        f"own f32 rounding ({d_seq:.4e} vs {floor:.4e}). It is the recurrence "
        f"the serving graph actually runs, so this is not a lesser gate than "
        f"the chunked core's -- same doctrine, same denominator.")
    assert spread > 0.0, (
        "the two cores are bitwise identical, which they cannot be: one sums "
        "the recurrence per chunk and the other per token. A zero here means "
        "this cell built the same graph twice and is measuring nothing.")

    # THE STATE MUST ACTUALLY CARRY, which the rows above cannot see: they read
    # one forward, and a Loop whose Assign went nowhere would produce exactly
    # the same first forward. A second infer on the SAME input must differ,
    # because the recurrent state it starts from is the one the first infer
    # wrote -- and the chunked core, which has no state, must repeat itself
    # exactly. Both halves are needed: the first alone would also pass on a
    # graph that was merely non-deterministic.
    y_seq2 = np.asarray(sequential(feed)[sequential.output(0)], np.float64)
    y_chunked2 = np.asarray(chunked(feed)[chunked.output(0)], np.float64)
    carry = float(np.max(np.abs(y_seq2 - y_seq)))
    print(f"[gdn-seq] T={T} second forward: |seq2-seq1|={carry:.4e}  "
          f"|chunked2-chunked1|={np.max(np.abs(y_chunked2 - y_chunked)):.4e}")
    assert carry > 0.0, (
        "a second forward on the same input reproduced the first exactly, so "
        "the recurrent state did not carry: the Assign is not reaching the "
        "Variable the ReadValue starts from. That is the whole reason this "
        "core is a Loop over a state rather than a function of its inputs.")
    assert np.array_equal(y_chunked2, y_chunked), (
        "the CHUNKED core changed between two infers on identical input. It "
        "carries no state, so this is non-determinism, and it would make the "
        "assertion above meaningless -- a difference there would no longer be "
        "evidence of a state carry.")
