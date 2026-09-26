"""The plugin's native lowering (marfrit-openvino patch 0043), on the card.

A tiled MoE block at a small REAL-format geometry (hidden 512, inter 256,
4 experts, top-2 -- IQ3_XXS needs 256-wide rows) with RANDOM native blocks
(not alike across rows or pages), saved as IR so the offload path has
file-backed Constants, compiled on the GPU with the served properties
(OFFLOAD_RATIO, MOE_CPU_TIER, WEIGHTS_PATH) and compared against the CPU
plugin's exact run of the same IR. The census asserts the fused primitive
exists (the lowering fired) -- a silent fall-through to generic ops would
still compute the right numbers here and would be the wrong mechanism.

Runs only with the suite's recorded GPU gate set (`Q4E_GPU=GPU.0`, see
tests/python/q4e_device.py; a new gate would be a new axis in the count
space, test_suite_guards) and the patched runtime on PYTHONPATH; the SOP
card window applies (docs/sop-card-window.md).
"""
import os
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import pytest
from openvino import Type, opset13 as op

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from q4e import native_blocks as nb  # noqa: E402
from q4e import serving_shape as ss  # noqa: E402
from q4e_device import device_params  # noqa: E402

_GPUS = [d for d in device_params() if d != "CPU"]
_skip = pytest.mark.skipif(not _GPUS, reason="Q4E_GPU unset (a card window)")


class _RandomNativeFiller:
    """A native filler serving random valid blocks in the checkpoint's two
    per-layer combinations: IQ3_XXS gate/up over an IQ4_NL down (43 layers)
    and IQ4_XS gate/up over a Q8_0 down (layer 2)."""

    def __init__(self, gate_up_fmt, down_fmt, seed=0):
        self.fmts = {"gate": gate_up_fmt, "up": gate_up_fmt, "down": down_fmt}
        self.rng = np.random.default_rng(seed)

    def native(self, layer, kind, e, out, inn):
        fmt = self.fmts[kind]
        # "<fmt>_PACKED" serves the checkpoint's own block verbatim (patch 0052)
        packed = fmt.endswith("_PACKED")
        base = fmt[: -len("_PACKED")] if packed else fmt
        block, nbytes = nb.BLOCK_BYTES[base]
        rows, nblk = e * out, inn // block
        raw = self.rng.integers(0, 256, size=(rows, nblk, nbytes), dtype=np.uint8)
        # the block scale keeps every decoded weight below ~0.3 whatever the
        # format's code range (IQ4_NL table 127, IQ3_XXS grid 62 x sub-scale
        # 7.75, IQ4_XS 127 x 32, Q8_0 128): the fused block runs in f16 on the
        # card, and random 0.2-scale blocks overflowed it (NaN on GPU.1,
        # 2026-09-18) -- the same magnitude the affine control's 0.001..0.02
        # scales over 0..15 codes give
        max_mag = {"IQ4_NL": 127.0, "IQ3_XXS": 62.0 * 7.75, "IQ4_XS": 127.0 * 32.0, "Q8_0": 128.0,
                   "IQ2_S": 43.0 * 3.875}[base]
        d = (self.rng.uniform(0.05, 1.0, size=(rows, nblk)).astype(np.float32) * (0.3 / max_mag))
        raw[:, :, 0:2] = d.astype("<f2").view(np.uint8).reshape(rows, nblk, 2)
        if packed:
            return fmt, nb.PACKED[base](raw.reshape(rows, nblk * nbytes))
        return fmt, nb.SPLIT[fmt](raw.reshape(rows, nblk * nbytes))


class _RandomAffineFiller:
    """The CONTROL: the stock u4 grouped-affine bodies (serving_shape's
    fusing chain, group 128) at the same geometry, through the same harness
    and properties. A failure here is the harness or the environment, not
    the native lowering."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def body(self, layer, kind, e, out, inn):
        from q4e.expert_fill import pack_u4
        gs = ss.EXPERT_GROUP_SIZE
        groups = inn // gs
        codes = self.rng.integers(0, 16, size=(e, out, groups, gs), dtype=np.uint8)
        zp = self.rng.integers(0, 16, size=(e, out, groups, 1), dtype=np.uint8)
        sc = self.rng.uniform(0.001, 0.02, size=(e, out, groups, 1)).astype(np.float32)
        return pack_u4(codes), pack_u4(zp), sc


def _config():
    from transformers.models.qwen4_exp import configuration_qwen4_exp as pin_cfg
    return pin_cfg.Qwen4ExpTextConfig(
        hidden_size=512, num_hidden_layers=1, num_experts=4, num_experts_per_tok=2,
        norm_topk_prob=True, moe_intermediate_size=256, shared_expert_intermediate_size=64,
        hidden_act="silu", hc_count=4, hc_lowrank=8, rms_norm_eps=1e-6,
        layer_types=["linear_attention"], vocab_size=257, eos_token_id=0, pad_token_id=0,
    )


def _build(tmp_path, T, gate_up_fmt, down_fmt):
    cfg = _config()
    H, I, E, Is = cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts, cfg.shared_expert_intermediate_size
    rng = np.random.default_rng(1)
    arena = ss.SparseArena(path=str(tmp_path / "arena.bin"))
    st = {
        "mlp.gate.weight": (rng.standard_normal((E, H)) * 0.05).astype(np.float32),
        "mlp.shared_expert.gate_proj.weight": (rng.standard_normal((Is, H)) * 0.05).astype(np.float32),
        "mlp.shared_expert.up_proj.weight": (rng.standard_normal((Is, H)) * 0.05).astype(np.float32),
        "mlp.shared_expert.down_proj.weight": (rng.standard_normal((H, Is)) * 0.05).astype(np.float32),
        "mlp.shared_expert_gate.weight": (rng.standard_normal((1, H)) * 0.05).astype(np.float32),
    }
    # T dynamic as in every served artifact: the emitter's Reshape targets keep
    # M as the runtime -1, and the plugin's router/MoE lowering is only ever
    # exercised with a dynamic token dim (the first form of this cell declared
    # T static and the stock control failed at compile on a missing router
    # primitive, 2026-09-18 GPU.1)
    x = op.parameter([1, -1, H], Type.f32, name="x")
    filler = _RandomAffineFiller() if gate_up_fmt == "affine" else _RandomNativeFiller(gate_up_fmt, down_fmt)
    y = ss.emit_moe_tiled(x, cfg, st, arena, T, "layer0/moe", filler=filler, layer=0)
    model = ov.Model([op.result(y)], [x], "native_moe_block")
    ov.save_model(model, str(tmp_path / "moe.xml"), compress_to_fp16=False)
    return arena, cfg


# the served routes: the tiered arm (ratio 50 + host tier) and the all-resident
# native arm (ratio 0 + per-expert dispatch, patch 0051; the tier auto-enabled
# as arcint's config does). The all-resident arm is the one the A770 serves.
_ROUTES = {
    "tier50": {"OFFLOAD_RATIO": "50", "MOE_CPU_TIER": "YES"},
    "resident": {"OFFLOAD_RATIO": "0", "MOE_CPU_TIER": "YES", "MOE_PER_EXPERT_DISPATCH": "YES"},
}


@_skip
@pytest.mark.parametrize("dev", _GPUS)
@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize("gate_up_fmt,down_fmt", [("affine", "affine"), ("IQ3_XXS", "IQ4_NL"), ("IQ4_XS", "Q8_0"),
                                                  ("IQ2_S", "IQ3_XXS"), ("IQ2_S_PACKED", "IQ3_XXS")])
def test_the_native_block_lowers_to_the_fused_primitive_and_matches_the_cpu_plugin(tmp_path, dev, route, gate_up_fmt,
                                                                                   down_fmt):
    if route == "resident" and gate_up_fmt == "affine":
        pytest.skip("ratio 0 is the native formats' all-resident configuration (patch 0051); the affine "
                    "control at ratio 0 takes the stock resident provider, a different path")
    _DEV = dev
    T = 6
    arena, cfg = _build(tmp_path, T, gate_up_fmt, down_fmt)
    xml = str(tmp_path / "moe.xml")
    props = dict(_ROUTES[route], WEIGHTS_PATH=str(tmp_path / "moe.bin"), INFERENCE_PRECISION_HINT="f16")
    try:
        if _AB:
            # a patched runtime the Python binding refuses: the C++ runner
            # (tools/native_moe_block_ab.cpp) does the same CPU-vs-GPU run
            moe_typed, native_nodes, st = _run_ab(xml, _DEV, T, props)
        else:
            moe_typed, native_nodes, st = _run_inprocess(xml, _DEV, T, cfg, props)
    finally:
        arena.close()
    print(f"\n[native-lowering] {_DEV} {route} {gate_up_fmt}/{down_fmt}: moe-typed primitives {moe_typed}; "
          f"native nodes {native_nodes}; max|diff| {st['max_abs']:.4e} vs max|want| {st['max_want']:.4e}; "
          f"max diff/band {st['max_over_band']:.3f}; corr {st['corr']:.6f}")
    assert moe_typed, "no MoE-typed primitive in the runtime graph: the lowering did not fire"
    if gate_up_fmt != "affine":
        # the native pass names its op; the stock fusion never produces this name
        assert native_nodes, "a MoE primitive exists but none carries the native pass's name: the stock fusion took it"
    assert st["max_over_band"] <= 1.0, f"max diff/band {st['max_over_band']:.3f}, max|diff| {st['max_abs']:.4e}"


# The band, both runners: per token the block's output is a top-2 sum of
# expert rows, so a wrong expert (or a wrong sign table in one) moves elements
# by the order of the row's RMS. It is 2% of the element plus 1% of its row's
# RMS -- not a fraction of the tensor's peak applied everywhere (review,
# 2026-09-18). Calibrated on the stock control (A770, f16 path): its max diff
# was 0.6% of the row RMS at 1e-2 + 5e-3, so this band sits at 2x the measured
# f16 noise and 1/100 of a wrong expert.
_AB = os.environ.get("ARCINT_NATIVE_BLOCK_AB", "")
_AB_LIB = os.environ.get("ARCINT_NATIVE_BLOCK_LIB", "")


def _run_inprocess(xml, dev, T, cfg, props):
    core = ov.Core()
    x = np.random.default_rng(2).standard_normal((1, T, cfg.hidden_size)).astype(np.float32)
    ref = core.compile_model(core.read_model(xml), "CPU")
    want = ref({"x": x})[ref.output(0)]
    gpu = core.compile_model(core.read_model(xml), dev, props)
    got = gpu({"x": x})[gpu.output(0)]
    moe_typed, native_nodes = 0, 0
    for n in gpu.get_runtime_model().get_ordered_ops():
        t = n.get_rt_info()["layerType"].astype(str) if "layerType" in n.get_rt_info() else n.get_type_name()
        moe_typed += "moe" in t.lower()
        native_nodes += "MOECompressedNative" in n.get_friendly_name()
    d = np.abs(got.astype(np.float64) - want.astype(np.float64))
    rms_row = np.sqrt((want.astype(np.float64) ** 2).mean(axis=-1, keepdims=True))
    band = 2e-2 * np.abs(want) + 1e-2 * rms_row
    return moe_typed, native_nodes, {"max_abs": d.max(), "max_want": np.abs(want).max(),
                                     "max_over_band": (d / band).max(),
                                     "corr": np.corrcoef(got.ravel(), want.ravel())[0, 1]}


def _run_ab(xml, dev, T, props, extra_env=None):
    import subprocess
    env = dict(os.environ, LD_LIBRARY_PATH=_AB_LIB + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
    env.update(extra_env or {})
    out = subprocess.run([_AB, xml, dev, str(T), "2"] + [f"{k}={v}" for k, v in props.items()],
                         capture_output=True, text=True, env=env, check=True).stdout
    moe_typed = native_nodes = 0
    st = {}
    got_hash = None
    for ln in out.splitlines():
        f = ln.split()
        kv = dict(x.split("=") for x in f[1:] if "=" in x)
        if f and f[0] == "RUNTIME":
            moe_typed, native_nodes = int(kv["moe_typed"]), int(kv["native_nodes"])
        elif f and f[0] == "DIFF":
            st = {k: float(v) for k, v in kv.items()}
        elif f and f[0] == "GOT":
            got_hash = kv["fnv1a64"]
    assert st, out
    st["got_hash"] = got_hash
    return moe_typed, native_nodes, st


@_skip
@pytest.mark.skipif(not _AB, reason="needs the C++ runner (ARCINT_NATIVE_BLOCK_AB): two GPU runs compared by bytes")
@pytest.mark.parametrize("dev", _GPUS)
@pytest.mark.parametrize("gate_up_fmt,down_fmt", [("IQ2_S_PACKED", "IQ3_XXS"), ("IQ3_XXS", "IQ4_NL"),
                                                  ("IQ4_XS", "Q8_0")])
# T=17: 34 pairs over _config()'s 4 experts, so one slot holds 9 or more --
# full tiles of NATIVE_TILE_M and a split one (M = 8 in 0060, 4 since 0061);
# T in {1, 6} never filled an 8-tile.
@pytest.mark.parametrize("T", [1, 6, 17])
@pytest.mark.parametrize("mode", ["batched", "grouped"])
def test_batched_dispatch_is_bit_identical_to_per_pair(tmp_path, dev, gate_up_fmt, down_fmt, T, mode):
    """Patch 0059: every (token, expert) pair of a call in one launch per
    stage; patch 0060: the pairs grouped by expert into tiles that decode the
    weights once. Each pair keeps the per-pair body's own arithmetic, and the
    output must be the SAME BYTES as the per-pair launches
    (MOE_DISPATCH_MODE=pair), not only inside the band -- and inside it too. Measured on the A770 (GPU.1), whose
    served path is run-to-run bit-identical; on the B60 two processes differ by
    f16 ulps on their own (DESIGN 7.0.2cb), so there this cell cannot read."""
    arena, cfg = _build(tmp_path, T, gate_up_fmt, down_fmt)
    xml = str(tmp_path / "moe.xml")
    props = dict(_ROUTES["resident"], WEIGHTS_PATH=str(tmp_path / "moe.bin"), INFERENCE_PRECISION_HINT="f16")
    try:
        _, native_b, batched = _run_ab(xml, dev, T, props, {"MOE_DISPATCH_MODE": mode})
        _, native_p, per_pair = _run_ab(xml, dev, T, props, {"MOE_DISPATCH_MODE": "pair"})
    finally:
        arena.close()
    # both runs on the native route, or equal hashes would mean nothing
    assert native_b and native_p, "the native pass did not take the block"
    print(f"\n[{mode}-dispatch] {dev} {gate_up_fmt}/{down_fmt} T={T}: {batched['got_hash']} vs {per_pair['got_hash']}; "
          f"band {batched['max_over_band']:.3f}")
    assert batched["got_hash"] and batched["got_hash"] == per_pair["got_hash"]
    assert batched["max_over_band"] <= 1.0


@_skip
@pytest.mark.skipif(not _AB, reason="needs the C++ runner (ARCINT_NATIVE_BLOCK_AB): two GPU runs compared by bytes")
@pytest.mark.parametrize("dev", _GPUS)
@pytest.mark.parametrize("gate_up_fmt,down_fmt", [("IQ2_S_PACKED", "IQ3_XXS"), ("IQ3_XXS", "IQ4_NL"),
                                                  ("IQ4_XS", "Q8_0")])
@pytest.mark.parametrize("T", [1, 6, 17])
@pytest.mark.parametrize("mode", ["batched", "grouped"])
@pytest.mark.parametrize("tile_n", ["default", "2", "4"])
def test_row_blocked_decode_is_bit_identical_to_row_at_a_time(tmp_path, dev, gate_up_fmt, down_fmt, T, mode, tile_n):
    """Patch 0061: the native per-expert kernels decode several output rows
    per pass (IQ2_S-packed gate and up together, IQ3_XXS and IQ4_NL down), so
    one load of a token's activation feeds every row. "default" is the
    shipped pair (gate/up 2, down 4); 2 and 4 set both. MOE_NATIVE_TILE_N=1 is
    the 0060 row-at-a-time loop in the same build, the reference, and the
    bytes must be equal. The cell reads f16 output: it catches a gate/up
    reordering, but a rounding-level reordering of the down sum did not move
    it (the product order there rests on the code). IQ4_XS/Q8_0 has no
    row-blocked decode and is the control.
    Measured on the A770 (GPU.1), as the cell above."""
    arena, cfg = _build(tmp_path, T, gate_up_fmt, down_fmt)
    xml = str(tmp_path / "moe.xml")
    props = dict(_ROUTES["resident"], WEIGHTS_PATH=str(tmp_path / "moe.bin"), INFERENCE_PRECISION_HINT="f16")
    try:
        env_r = {"MOE_DISPATCH_MODE": mode}
        if tile_n != "default":
            env_r["MOE_NATIVE_TILE_N"] = tile_n
        _, native_r, rows = _run_ab(xml, dev, T, props, env_r)
        _, native_1, one = _run_ab(xml, dev, T, props, {"MOE_DISPATCH_MODE": mode, "MOE_NATIVE_TILE_N": "1"})
    finally:
        arena.close()
    assert native_r and native_1, "the native pass did not take the block"
    print(f"\n[rows-{tile_n} {mode}] {dev} {gate_up_fmt}/{down_fmt} T={T}: {rows['got_hash']} vs {one['got_hash']}; "
          f"band {rows['max_over_band']:.3f}")
    assert rows["got_hash"] and rows["got_hash"] == one["got_hash"]
    assert rows["max_over_band"] <= 1.0
