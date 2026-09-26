"""The GPU pipeline's native MoE matcher, asked at its OUTPUT (device-free).

`ConvertTiledMoeBlockNativeToMoeCompressed` (plugin patches 0043/0050/0052)
must turn the emitter's tiled MoE block into ONE `MOECompressed` whose expert
inputs are the artifact's own Constants. When it does not, the decode chain
stays in the graph and the GPU compile constant-folds it -- on the host
(`ConstantFolding` inside `ConvertPrecision`, `MultiplyMultiplyFusion` in
`CommonOptimizations`, 1 GiB f32 per expert tensor at the 35B's geometry) and
on the card (`propagate_constants`, where the packed d4 load died with
CL_OUT_OF_RESOURCES). Measured 2026-09-26 with tools/bigalloc.c; see
docs/design-fit-levers.md.

A print inside the matcher's `resolve()` read as "the matcher fires" while the
callback's Constant guard refused the packed block afterwards. So these cells
run the pass alone (tools/native_moe_match_probe.cpp, linked against the
patched libopenvino.so) and read what it produced.

The expert bytes come from a fake GGUF feed with RANDOM valid blocks through
the real `NativeExpertFiller` and `emit_moe_tiled` -- the served emitter path,
not a hand-built graph. Gated on ARCINT_NATIVE_MATCH_PROBE (the probe binary)
and ARCINT_NATIVE_MATCH_LIB (the runtime dir holding the libopenvino.so under
test); skipped without them.
"""
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ov = pytest.importorskip("openvino")
from openvino import opset13 as op  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from q4e import expert_fill as ef  # noqa: E402
from q4e import native_blocks as nb  # noqa: E402
from q4e import serving_shape as ss  # noqa: E402

PROBE = os.environ.get("ARCINT_NATIVE_MATCH_PROBE", "")
LIB = os.environ.get("ARCINT_NATIVE_MATCH_LIB", "")
_skip = pytest.mark.skipif(not (PROBE and LIB), reason="ARCINT_NATIVE_MATCH_PROBE / _LIB unset")

# MOECompressed::kWeightFormat* (moe_compressed.hpp, patches 0043/0050/0052)
IQ3_XXS, IQ2_S, IQ2_S_PACKED = 2, 4, 5
E, K, H, I = 8, 2, 512, 256


class _FakeFeed:
    """Random valid blocks per tensor: IQ2_S gate/up, IQ3_XXS down -- the
    35B-A3B checkpoint's own mix. Every byte random, the f16 d finite and
    non-zero (the blind-fill lesson: rows and blocks must differ)."""

    def __init__(self, seed=7):
        self._rng = np.random.default_rng(seed)

    def gguf_type(self, name):
        return "IQ3_XXS" if "down" in name else "IQ2_S"

    def raw_rows(self, name, rows):
        fmt = self.gguf_type(name)
        out, inn = (H, I) if "down" in name else (I, H)
        block, nbytes = nb.BLOCK_BYTES[fmt]
        nblk = inn // block
        raw = self._rng.integers(0, 256, size=(rows * out, nblk, nbytes), dtype=np.uint8)
        d = self._rng.uniform(0.01, 0.2, size=(rows * out, nblk)).astype("<f2")
        raw[:, :, 0:2] = d.view(np.uint8).reshape(rows * out, nblk, 2)
        return raw.reshape(rows, out, nblk * nbytes)


def _emit(tmp_path, packed, fp16=False):
    cfg = types.SimpleNamespace(hidden_size=H, num_experts=E, num_experts_per_tok=K,
                                moe_intermediate_size=I, shared_expert_intermediate_size=I,
                                hidden_act="silu", norm_topk_prob=True)
    rng = np.random.default_rng(1)
    st = {
        "mlp.gate.weight": op.constant(rng.standard_normal((E, H)).astype(np.float32) * 0.1),
        "mlp.shared_expert.gate_proj.weight": op.constant(rng.standard_normal((I, H)).astype(np.float32) * 0.1),
        "mlp.shared_expert.up_proj.weight": op.constant(rng.standard_normal((I, H)).astype(np.float32) * 0.1),
        "mlp.shared_expert.down_proj.weight": op.constant(rng.standard_normal((H, I)).astype(np.float32) * 0.1),
        "mlp.shared_expert_gate.weight": op.constant(rng.standard_normal((1, H)).astype(np.float32) * 0.1),
    }
    arena = ss.SparseArena(capacity_bytes=1 << 30)
    try:
        hidden = op.parameter([1, -1, H], ov.Type.f32)
        hidden.set_friendly_name("hidden")
        filler = ef.NativeExpertFiller(_FakeFeed(), packed=packed)
        with ss.shared_constants():
            y = ss.emit_moe_tiled(hidden, cfg, st, arena, -1, "moe", filler=filler, layer=0)
        res = op.result(y)
        res.set_friendly_name("out")
        xml = tmp_path / (("packed" if packed else "relaid") + ("-f16" if fp16 else "") + ".xml")
        # fp16 = the exporter's --dense-fp16: save_model compresses EVERY f32
        # Constant, the decode chain's grids/tables/divisors included
        ov.save_model(ov.Model([res], [hidden], "moe_block"), str(xml), compress_to_fp16=fp16)
        return xml
    finally:
        arena.close()


def _probe(xml):
    env = dict(os.environ, LD_LIBRARY_PATH=LIB + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
    out = subprocess.run([PROBE, str(xml), "1"], capture_output=True, text=True, env=env, check=True).stdout
    moes, ins, count = [], {}, None
    for ln in out.splitlines():
        f = ln.split()
        if f[0] == "MOE_COMPRESSED":
            count = int(f[1])
        elif f[0] == "MOE":
            moes.append(dict(kv.split("=") for kv in f[2:]))
        elif f[0] == "IN":
            ins[f[2]] = (f[3], f[4], " ".join(f[5:]))
    return count, moes, ins, out


@_skip
@pytest.mark.parametrize("fp16", [False, True], ids=["f32", "dense-fp16"])
def test_the_relaid_iq2s_block_fuses(tmp_path, fp16):
    """Control: the re-laid IQ2_S form (0050) served on the A770 all-resident
    (DESIGN record, 2026-09-25), so the matcher must take it -- also from a
    --dense-fp16 save, whose compressed chain constants no block matched
    before patch 0057 (the full-depth f16 artifact fused 0 of 40)."""
    count, moes, ins, out = _probe(_emit(tmp_path, packed=False, fp16=fp16))
    assert count == 1, out
    assert int(moes[0]["gate_up_format"]) == IQ2_S and int(moes[0]["down_format"]) == IQ3_XXS, out
    for slot in ("gate_w", "gate_s", "up_w", "up_s", "down_w", "down_s"):
        assert ins[slot][0] == "Constant", (slot, out)


@_skip
@pytest.mark.parametrize("fp16", [False, True], ids=["f32", "dense-fp16"])
def test_the_packed_iq2s_block_fuses_with_d_in_the_scale_slot(tmp_path, fp16):
    """The packed form (0052) must fuse too, and its scale slot must carry the
    checkpoint's f16 `d` Constant [E, out, K/256, 1] -- what the OCL decode, the
    OTD runtime and the CPU tier read for format 5 -- not a folded per-value
    f32 scale."""
    count, moes, ins, out = _probe(_emit(tmp_path, packed=True, fp16=fp16))
    assert count == 1, out
    assert int(moes[0]["gate_up_format"]) == IQ2_S_PACKED, out
    assert int(moes[0]["down_format"]) == IQ3_XXS, out
    for proj, inn in (("gate", H), ("up", H)):
        t, et, shape = ins[proj + "_w"]
        assert (t, et) == ("Constant", "u8") and shape == f"[{E},{I},{inn // 256},80]", (proj, out)
        t, et, shape = ins[proj + "_s"]
        assert (t, et) == ("Constant", "f16") and shape == f"[{E},{I},{inn // 256},1]", (proj, out)
