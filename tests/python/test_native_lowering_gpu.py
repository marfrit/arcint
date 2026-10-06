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


def _config(hidden_size=512, inter=256, experts=4, top_k=2):
    from transformers.models.qwen4_exp import configuration_qwen4_exp as pin_cfg
    return pin_cfg.Qwen4ExpTextConfig(
        hidden_size=hidden_size, num_hidden_layers=1, num_experts=experts, num_experts_per_tok=top_k,
        norm_topk_prob=True, moe_intermediate_size=inter, shared_expert_intermediate_size=64,
        hidden_act="silu", hc_count=4, hc_lowrank=8, rms_norm_eps=1e-6,
        layer_types=["linear_attention"], vocab_size=257, eos_token_id=0, pad_token_id=0,
    )


def _build(tmp_path, T, gate_up_fmt, down_fmt, hidden_size=512, inter=256, experts=4, top_k=2):
    cfg = _config(hidden_size, inter, experts, top_k)
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
    for k, v in (extra_env or {}).items():
        if v is None:
            env.pop(k, None)            # an explicit None removes the caller's value
        else:
            env[k] = v
    run = subprocess.run([_AB, xml, dev, str(T), "2"] + [f"{k}={v}" for k, v in props.items()],
                         capture_output=True, text=True, env=env, check=True)
    out = run.stdout
    moe_typed = native_nodes = 0
    st = {}
    got_hash = None
    st_head = None
    for ln in out.splitlines():
        f = ln.split()
        kv = dict(x.split("=") for x in f[1:] if "=" in x)
        if f and f[0] == "RUNTIME":
            moe_typed, native_nodes = int(kv["moe_typed"]), int(kv["native_nodes"])
        elif f and f[0] == "DIFF":
            st = {k: float(v) for k, v in kv.items()}
        elif f and f[0] == "GOT":
            got_hash = kv["fnv1a64"]
        elif f and f[0] == "HEAD":
            st_head = kv["fnv1a64"]
    assert st, out
    # the plugin's exit counters (MOE_OTD_PERF_LOG=1), when the caller asked for them
    for ln in run.stderr.splitlines():
        if ln.startswith("[OTD_PERF]") and "device_routed_calls=" in ln:
            st["device_routed_calls"] = int(ln.split("device_routed_calls=")[1].split(",")[0])
    st["got_hash"] = got_hash
    st["head_hash"] = st_head
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


@_skip
@pytest.mark.skipif(not _AB, reason="needs the C++ runner (ARCINT_NATIVE_BLOCK_AB): GPU runs compared by bytes")
@pytest.mark.parametrize("dev", _GPUS)
@pytest.mark.parametrize("gate_up_fmt,down_fmt", [("IQ2_S_PACKED", "IQ3_XXS"), ("IQ3_XXS", "IQ4_NL")])
@pytest.mark.parametrize("mode", ["batched", "grouped"])
@pytest.mark.parametrize("hidden", [512, 2048])
def test_a_tokens_bytes_do_not_depend_on_its_call(tmp_path, dev, gate_up_fmt, down_fmt, mode, hidden):
    """Patch 0064: IQ2_S-packed gate/up runs on the matrix unit in tiles of up
    to 16 pairs of one expert, for every call size. A token's output must not
    depend on which other tokens share its call -- its tile-mates, its row in
    the tile, the call's size -- or the served greedy output would depend on
    how a prompt was chunked or cached (DESIGN §3's cold/warm invariant). The
    runner draws the input rows in order from one stream, so token 0 is the
    same at every T: its output bytes at T = 1 (alone), 17 and 40 (in tiles
    with others, 40 filling a 16-tile over the cell's 4 experts) must be equal.
    IQ3_XXS/IQ4_NL keeps the scalar kernels (the control). hidden 2048 is the
    35B's eight 256-value blocks per row: at 512 (two) a reversed block order
    in the one-pair kernel did not reach the f16 bytes and stayed green.
    Measured on the A770 (GPU.1); the B60 differs run to run on its own
    (DESIGN 7.0.2cb)."""
    heads = {}
    for T in (1, 17, 40):
        (tmp_path / f"T{T}").mkdir()
        arena, _ = _build(tmp_path / f"T{T}", T, gate_up_fmt, down_fmt, hidden)
        xml = str(tmp_path / f"T{T}" / "moe.xml")
        props = dict(_ROUTES["resident"], WEIGHTS_PATH=str(tmp_path / f"T{T}" / "moe.bin"),
                     INFERENCE_PRECISION_HINT="f16")
        try:
            _, native, st = _run_ab(xml, dev, T, props, {"MOE_DISPATCH_MODE": mode, "ARCINT_BLOCK_AB_HASH_ROWS": "1"})
        finally:
            arena.close()
        assert native, "the native pass did not take the block"
        heads[T] = st["head_hash"]
    print(f"\n[call-independence {mode} h{hidden}] {dev} {gate_up_fmt}/{down_fmt}: token 0 at T=1/17/40: {heads}")
    assert heads[1] and heads[1] == heads[17] == heads[40]


@_skip
@pytest.mark.skipif(not _AB, reason="needs the C++ runner (ARCINT_NATIVE_BLOCK_AB): two GPU runs compared by bytes")
@pytest.mark.parametrize("dev", _GPUS)
@pytest.mark.parametrize("gate_up_fmt,down_fmt", [("IQ2_S_PACKED", "IQ3_XXS"), ("IQ3_XXS", "IQ4_NL")])
@pytest.mark.parametrize("T", [1, 6, 17])
@pytest.mark.parametrize("hidden", [512, 2048])
def test_decode_routed_on_the_device_gives_the_host_routes_bytes(tmp_path, dev, gate_up_fmt, down_fmt, T, hidden):
    """Patch 0067: on the all-resident pool a call below the grouped threshold
    (decode, and the short calls up to 63 pairs) routes on the device -- a
    kernel writes the pair table from topk_id, and the host neither reads the
    ids back nor builds the table. The output bytes must be the host route's
    (MOE_DEVICE_ROUTE=0 in the same build). The plugin's exit counter must show
    the device route took the calls in the first arm and none in the second,
    or equal hashes would compare the host route with itself. IQ2_S-packed
    gate/up takes the matrix unit's one-pair kernel, IQ3_XXS the scalar
    batched one. T = 1, 6, 17 are 2, 12 and 34 pairs at the cell's top-2.
    Measured on the A770 (GPU.1), as the cells above."""
    arena, _ = _build(tmp_path, T, gate_up_fmt, down_fmt, hidden)
    xml = str(tmp_path / "moe.xml")
    props = dict(_ROUTES["resident"], WEIGHTS_PATH=str(tmp_path / "moe.bin"), INFERENCE_PRECISION_HINT="f16")
    try:
        _, native_d, on_dev = _run_ab(xml, dev, T, props, {"MOE_OTD_PERF_LOG": "1"})
        _, native_h, on_host = _run_ab(xml, dev, T, props, {"MOE_OTD_PERF_LOG": "1", "MOE_DEVICE_ROUTE": "0"})
    finally:
        arena.close()
    assert native_d and native_h, "the native pass did not take the block"
    print(f"\n[device-route h{hidden}] {dev} {gate_up_fmt}/{down_fmt} T={T}: {on_dev['got_hash']} vs "
          f"{on_host['got_hash']}; device-routed calls {on_dev.get('device_routed_calls')} / "
          f"{on_host.get('device_routed_calls')}")
    assert on_dev.get("device_routed_calls", 0) > 0, "the device route did not fire"
    assert on_host.get("device_routed_calls") == 0
    assert on_dev["got_hash"] and on_dev["got_hash"] == on_host["got_hash"]
    assert on_dev["max_over_band"] <= 1.0


@_skip
@pytest.mark.skipif(not _AB, reason="needs the C++ runner (ARCINT_NATIVE_BLOCK_AB): two GPU runs compared by bytes")
@pytest.mark.parametrize("dev", _GPUS)
@pytest.mark.parametrize("hidden,inter", [(2560, 640), (2304, 608)])
@pytest.mark.parametrize("T", [1, 8])
def test_flash_next_geometry_matches_the_cpu_plugin_and_the_per_pair_bytes(tmp_path, dev, hidden, inter, T):
    """Patch 0069: the IQ3_XXS gate/up and the IQ4_NL down decode a whole
    32-value block per lane -- gate/up lane i takes blocks i, i + 16, ... of a
    row; down gives each row a group of four lanes (SIMD16, N_BLOCK 4), lane q
    taking blocks q, q + 4, .... The cells above run at hidden 512/2048 and
    inter 256, where both splits come out even. This one runs Flash-Next's own
    geometry (hidden 2560: 80 blocks, 5 per lane; inter 640: 20 blocks, 5 per
    group lane) and a remainder one (hidden 2304: 72 blocks, the last trip
    half the lanes; inter 608: 19 blocks, the last group lane one short), at
    16 experts and Flash-Next's top-10: T = 1 is a decode step (10 pairs, the
    batched kernels, routed on the device), T = 8 is 80 pairs (the grouped
    kernels under the auto dispatch). The output must be the per-pair
    launches' bytes (MOE_DISPATCH_MODE=pair), and the gate/up's rows per
    subgroup (MOE_NATIVE_GU_ROWS, default 2) must not move them either (4
    rows: a different grid, the same per-row sums). Against the CPU oracle it
    must sit in the band -- or, where the per-element decoders it replaces
    (MOE_NATIVE_LEGACY_DECODE=1, the same build) already sit outside it, no
    further out than they do, within 5% of the band. That case is measured:
    at hidden 2560, T = 8 one element (token 3, column 1491: want -0.0058,
    got 0.0952, the row's RMS 6.4) reads 1.567 of the band under both
    decoders, with the same output bits, on the pre-0069 plugin too; its
    cause is not measured. A decode defect moves elements by the order of
    the row's RMS, about 100 bands (the calibration note above).
    Measured on the A770 (GPU.1), as the cells above."""
    arena, _ = _build(tmp_path, T, "IQ3_XXS", "IQ4_NL", hidden, inter, experts=16, top_k=10)
    xml = str(tmp_path / "moe.xml")
    props = dict(_ROUTES["resident"], WEIGHTS_PATH=str(tmp_path / "moe.bin"), INFERENCE_PRECISION_HINT="f16")
    # the block decode by name, whatever the caller's shell holds (review of 0069, F1)
    base = {"MOE_NATIVE_LEGACY_DECODE": "0", "MOE_NATIVE_W_ROUND": None, "MOE_NATIVE_GU_ROWS": "2"}
    try:
        _, native_a, auto = _run_ab(xml, dev, T, props, dict(base))
        _, native_p, pair = _run_ab(xml, dev, T, props, dict(base, MOE_DISPATCH_MODE="pair"))
        _, native_4, rows4 = _run_ab(xml, dev, T, props, dict(base, MOE_NATIVE_GU_ROWS="4"))
        _, native_l, legacy = _run_ab(xml, dev, T, props, dict(base, MOE_NATIVE_LEGACY_DECODE="1"))
    finally:
        arena.close()
    assert native_a and native_p and native_4 and native_l, "the native pass did not take the block"
    print(f"\n[flash-next-geometry h{hidden} i{inter}] {dev} T={T}: {auto['got_hash']} vs {pair['got_hash']} vs "
          f"{rows4['got_hash']}; band {auto['max_over_band']:.3f} / {pair['max_over_band']:.3f}, per-element "
          f"decoders {legacy['max_over_band']:.3f} ({legacy['got_hash']}); corr {auto['corr']:.6f}")
    # the block decode ran: its summation order differs from the per-element one
    assert auto["got_hash"] != legacy["got_hash"], "the block decode did not run (legacy bytes)"
    # the relaxation covers ONE measured case (hidden 2560, T = 8, 1.567 bands under
    # both decoders); a defect the two decoders share would sit near 100 bands
    # (review of 0069, F2)
    assert legacy["max_over_band"] < 2.0 and auto["corr"] > 0.999, (
        f"per-element decoders at {legacy['max_over_band']:.3f} bands, corr {auto['corr']:.6f}")
    allowed = max(1.0, legacy["max_over_band"] + 0.05)
    assert auto["max_over_band"] <= allowed, f"max diff/band {auto['max_over_band']:.3f} > {allowed:.3f}"
    assert pair["max_over_band"] <= allowed, f"max diff/band {pair['max_over_band']:.3f} > {allowed:.3f}"
    assert auto["got_hash"] and auto["got_hash"] == pair["got_hash"] == rows4["got_hash"]
