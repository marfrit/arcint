"""E2 Phase B -- feed REAL GGUF weights into the pin/transcription state-dict
keys and confirm the graph-semantics parity leg stays exactly 0.0.

The kickoff insight, kept separate on purpose: parity against the pin is about
GRAPH SEMANTICS, not weight provenance. The pin loaded with dequantised GGUF
tensors keeps transcription-vs-pin at 0.0 because BOTH sides then carry
byte-identical weights. So a non-0.0 here is a NAME-MAP or a RESHAPE/FUSION bug,
never a quantisation-accuracy story (that is a later, separate gate -- KLD).

Each slice cell feeds ONE module's weights (narrowed to the tiny config's shape)
into an otherwise-random tiny model and checks transcription-vs-pin == 0.0. Two
micro-cells additionally check the non-trivial reshapes against an INDEPENDENT
reference (a hand-written dequant+reshape), so a shape-preserving permutation bug
-- which the load leg alone cannot see -- is caught: the GDN conv1d [C,K]->[C,1,K]
and the MoE gate|up fusion.

WHAT THE `== 0.0` LEGS CANNOT GATE (measured, REVIEW 2cd2b2f finding A): the
NAME half of the map, for any entry that shares a shape with another. Ref and
pin both read the same mis-mapped tensor, so the difference cancels identically
and a swap stays invisible -- 21 of the 30 `_LAYER_MAP` entries sit in such a
class. The NAME-MAP GATE at the bottom of this file closes that: one row per
map entry, the expected GGUF name hardcoded test-side and the reference read by
an independent `GGUFReader`. Proven by swapping entries: the gate goes red on
exactly the swapped pair while every `== 0.0` leg stays green.

Real-weight source: the shipped UD-Q3_K_XL GGUF shards. Point Q4E_GGUF_SHARDS at
the directory (or a glob) holding them; absent, every test here skips by name
(the shards are ~90 GiB and live only on the dev host -- this is not a
device-free unit). gguf-py must be importable as a real `gguf.GGUFReader`.

Run (dev host):
    Q4E_GGUF_SHARDS=<shard-dir> PYTHONPATH=<gguf-py> \\
        Q4E_GPU=  <venv>/bin/python3 -m pytest tests/python/test_gguf_feed.py -s \\
        --continue-on-collection-errors
"""
import glob
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_backbone as tb  # noqa: E402  (tiny-config fixture + pin helpers)
from transformers.models.qwen4_exp import modeling_qwen4_exp as pin_mod  # noqa: E402

_SHARDS = os.environ.get("Q4E_GGUF_SHARDS", "").strip()

try:
    from q4e import gguf_feed  # noqa: E402
    _FEED_IMPORT_ERR = None
except Exception as e:  # module absent in the RED state
    gguf_feed = None
    _FEED_IMPORT_ERR = e

_skip = pytest.mark.skipif(
    not _SHARDS, reason="Q4E_GGUF_SHARDS unset (real GGUF shards absent)")

# How far above the REFERENCE's own f32-vs-f64 rounding the emitter may sit and
# still count as "at the float floor". Relative, so it cannot be satisfied by a
# reference that is itself drifting, and not a tuned constant: pre-fix the
# nilpotent-series emitter sat at ~4500x this ratio at T=96, post-fix at O(1).
_FLOOR_FACTOR = 20.0


@pytest.fixture(scope="module")
def feed():
    if gguf_feed is None:
        pytest.fail(f"q4e.gguf_feed did not import: {_FEED_IMPORT_ERR!r}")
    return gguf_feed.GgufFeed(_SHARDS)


def _fit(feed, pin_key, shape, gguf_layer=None):
    """The real tensor narrowed to a tiny-fixture `shape`.

    Delegates to `GgufFeed.fitted` -- NOT a leading-index slice of
    `pin_tensor`'s result, which is FIX D: the fused `gate_up_proj` ff axis is
    two stacked halves, so a leading slice of the concat takes BOTH halves out
    of gate whenever the target is narrower than the real ff (measured: real ff
    640 vs a 2*32 target; `ffn_up_exps` never reached the state dict)."""
    return feed.fitted(pin_key, shape, gguf_layer=gguf_layer)


def _feed_state(config, feed, keys, seed=0):
    """Tiny random ref state with `keys` overwritten by feed-narrowed real
    tensors. Returns (ref, pin) both loaded with that state."""
    ref = tb._build_ref(config, seed=seed)
    sd = ref.state_dict()
    for pin_key in keys:
        sd[pin_key] = torch.from_numpy(
            _fit(feed, pin_key, tuple(sd[pin_key].shape))
        ).to(sd[pin_key].dtype)
    ref.load_state_dict(sd)
    pin = pin_mod.Qwen4ExpTextModel(config).eval()
    pin.load_state_dict(sd)
    return ref, pin


def _transcription_vs_pin(ref, pin, T=64, seed=9):
    torch.manual_seed(seed)
    ids = torch.randint(1, ref.config.vocab_size, (1, T))
    with torch.no_grad():
        h_ref = ref(ids, torch.ones(1, T))
        h_pin = pin(input_ids=ids, attention_mask=None, use_cache=False).last_hidden_state
    return float((h_ref - h_pin).abs().max())


def _layer_keys(sd_keys, layer, leaf):
    pre = f"layers.{layer}.{leaf}."
    return [k for k in sd_keys if k.startswith(pre)]


# --------------------------------------------------------------------------- #
@_skip
def test_map_covers_causal_keys(feed):
    """Every tiny-config pin key the causal backbone consumes resolves to a
    GGUF tensor that exists in the file (or is a derived buffer we do not feed).
    A missing entry is a NAME-MAP hole, caught here before any forward."""
    tb._assert_pin()
    config = tb._make_config()
    ref = tb._build_ref(config)
    missing, derived = [], []
    for k in ref.state_dict():
        # blk.1 is a real GDN+MoE+hc+PLE block; feed every per-layer key from it
        # (the tiny fixture makes all 4 layers GDN, but the real model ships some
        # as QSA -- feeding from a real GDN block is what a slice cell does).
        gl = 1 if k.startswith("layers.") else None
        try:
            feed.pin_tensor(k, rows=1, gguf_layer=gl)
        except KeyError as e:
            (derived if "derived index buffer" in str(e) else missing).append(k)
    print(f"\n[map-coverage] derived(skipped)={len(derived)}  unmapped={missing}")
    assert not missing, f"unmapped causal keys: {missing}"


@_skip
def test_embed_feed_transcription(feed):
    """Slice cell: embed_tokens fed from token_embd (Q8_0). Name map + shape."""
    tb._assert_pin()
    config = tb._make_config()
    ref, pin = _feed_state(config, feed, ["embed_tokens.weight"])
    md = _transcription_vs_pin(ref, pin)
    print(f"\n[feed-embed] token_embd type={feed.gguf_type('token_embd.weight')}  "
          f"transcription-vs-pin={md:.3e}")
    assert md == 0.0, md


@_skip
def test_gdn_conv_reshape_reference(feed):
    """Micro-cell: GDN conv1d [C,K]->[C,1,K] vs an INDEPENDENT reshape. Bites a
    permutation bug the load leg cannot (shape-preserving)."""
    got = feed.pin_tensor("layers.0.linear_attn.conv1d.weight")
    raw = feed.dequant("blk.0.ssm_conv1d.weight")        # [C, K]
    ref = raw.reshape(raw.shape[0], 1, raw.shape[1])      # [C, 1, K]
    drift = float(np.max(np.abs(got - ref)))
    # RED guard: a wrong (but same-shape) reshape must differ.
    wrong = np.transpose(raw, (1, 0)).reshape(raw.shape[0], 1, raw.shape[1]) \
        if raw.shape[0] == raw.shape[1] else raw[:, ::-1].reshape(raw.shape[0], 1, raw.shape[1])
    red = float(np.max(np.abs(got - wrong)))
    print(f"\n[feed-gdn-conv] shape={list(got.shape)}  vs-reference={drift:.3e}  "
          f"wrong-reshape-drift={red:.3e}")
    assert drift == 0.0, drift
    assert red > 0.0, "reshape check is vacuous (wrong reshape did not differ)"


@_skip
def test_moe_gate_up_fuse_reference(feed):
    """Micro-cell: MoE gate|up fusion [E,ff,in]+[E,ff,in]->[E,2ff,in] vs an
    independent concat, and the ordering (gate first)."""
    got = feed.pin_tensor("layers.0.mlp.experts.gate_up_proj", rows=2)  # 2 experts
    gate = feed.dequant("blk.0.ffn_gate_exps.weight", rows=2)
    up = feed.dequant("blk.0.ffn_up_exps.weight", rows=2)
    ref = np.concatenate([gate, up], axis=1)
    drift = float(np.max(np.abs(got - ref)))
    ff = gate.shape[1]
    gate_half = float(np.max(np.abs(got[:, :ff] - gate)))
    up_half = float(np.max(np.abs(got[:, ff:] - up)))
    print(f"\n[feed-moe-fuse] shape={list(got.shape)}  vs-reference={drift:.3e}  "
          f"gate-half={gate_half:.3e}  up-half={up_half:.3e}")
    assert drift == 0.0 and gate_half == 0.0 and up_half == 0.0


@_skip
def test_fused_gate_up_narrowing_takes_true_halves(feed):
    """FIX D (REVIEW 2cd2b2f finding D): narrowing the fused `gate_up_proj` to a
    smaller fixture must take TRUE HALVES at the real ff.

    The old code sliced `pin_tensor`'s [E, 2*ff_real, in] result by leading
    index. With ff_real=640 and a tiny target of 2*32, that leading slice never
    reaches the up half at all -- the measured failure was

        tiny 'up' half == real up  [0:32]  : False
        tiny 'up' half == real GATE[32:64] : True

    so `ffn_up_exps` never entered the tiny fed state through this key at all.
    `GgufFeed.fitted` now cuts each half at the real ff before the concat, and
    the callers use it instead of slicing by hand -- which is what makes the old
    failure unreachable rather than merely repaired. Asserted here against the
    real geometry (ff_real) AND against the old wrong answer."""
    ff_real = feed.dequant("blk.0.ffn_gate_exps.weight", rows=1).shape[1]
    ff_t = 32                              # the tiny config's moe_intermediate_size
    E, IN = 2, 16
    assert ff_real == 640, f"real expert ff moved: {ff_real} (re-read finding D)"
    assert ff_t < ff_real, "this cell only bites when the target is NARROWER"

    got = feed.fitted("layers.0.mlp.experts.gate_up_proj", (E, 2 * ff_t, IN))
    gate = feed.dequant("blk.0.ffn_gate_exps.weight", rows=E)[:, :, :IN]
    up = feed.dequant("blk.0.ffn_up_exps.weight", rows=E)[:, :, :IN]

    gate_ok = np.array_equal(got[:, :ff_t], gate[:, :ff_t])
    up_ok = np.array_equal(got[:, ff_t:], up[:, :ff_t])
    # the OLD answer: a leading slice of the concat -> both halves out of gate
    old_wrong = np.array_equal(got[:, ff_t:], gate[:, ff_t:2 * ff_t])
    print(f"\n[fused-halves] ff_real={ff_real} tiny_ff={ff_t}  shape={list(got.shape)}  "
          f"gate-half-from-gate={gate_ok}  up-half-from-UP={up_ok}  "
          f"up-half-from-gate[{ff_t}:{2*ff_t}](the old bug)={old_wrong}")
    assert gate_ok, "gate half is not gate[:ff]"
    assert up_ok, "up half is not up[:ff] -- the fused narrowing is still wrong"
    assert not old_wrong, "up half still equals gate's second block (FIX D regressed)"

    # and the guard that makes an over-wide request a named error, not silence
    with pytest.raises(ValueError):
        feed.fitted("layers.0.mlp.experts.gate_up_proj", (E, 2 * (ff_real + 1), IN))


@_skip
def test_gdn_layer_feed_transcription(feed):
    """Slice cell: all GDN (linear_attn) weights of layer 0 fed from GGUF."""
    tb._assert_pin()
    config = tb._make_config()
    keys = _layer_keys(tb._build_ref(config).state_dict(), 0, "linear_attn")
    ref, pin = _feed_state(config, feed, keys)
    md = _transcription_vs_pin(ref, pin)
    print(f"\n[feed-gdn-layer] {len(keys)} keys  transcription-vs-pin={md:.3e}")
    assert md == 0.0, md


@_skip
def test_moe_trio_feed_transcription(feed):
    """Slice cell: one MoE block (router + experts gate_up/down + shared trio +
    shared gate) of layer 0 fed from GGUF."""
    tb._assert_pin()
    config = tb._make_config()
    keys = _layer_keys(tb._build_ref(config).state_dict(), 0, "mlp")
    ref, pin = _feed_state(config, feed, keys)
    md = _transcription_vs_pin(ref, pin)
    print(f"\n[feed-moe-trio] {len(keys)} keys  transcription-vs-pin={md:.3e}")
    assert md == 0.0, md


@_skip
def test_ple_feed_transcription(feed):
    """Slice cell: the PLE layer (layer 1 in the tiny config) fed from GGUF --
    projections, norms, conv, and the n-gram table's first rows."""
    tb._assert_pin()
    config = tb._make_config()
    all_keys = tb._build_ref(config).state_dict()
    keys = [k for k in all_keys if k.startswith("layers.1.ple.")
            and not any(k.endswith(s) for s in gguf_feed._DERIVED_SUFFIXES)]
    ref, pin = _feed_state(config, feed, keys)
    md = _transcription_vs_pin(ref, pin)
    print(f"\n[feed-ple] {len(keys)} keys  transcription-vs-pin={md:.3e}")
    assert md == 0.0, md


@_skip
def test_backbone_assembled_from_feed(feed):
    """Assembly cell (Phase B step 3): the WHOLE tiny backbone loaded from
    gguf_feed tensors.

    WHAT THIS ACTUALLY COVERS (reworded under FIX D; the old framing said "on
    real weights" and left the geometry implicit): every weight is a REAL GGUF
    tensor NARROWED to the tiny config's geometry -- a leading-index cut on each
    axis, and true gate|up halves on the fused expert tensor. That is real
    BYTES through the real name map, not the real model's geometry: 2 of 512
    experts, ff 32 of 640, hidden 16 of 2560, vocab 257 of 248320, 4 layers of
    48. What it gates is the name map, the reshapes/fusion and the assembly
    wiring; what it does NOT gate is anything that only appears at full width.

    GDN/MoE/hc keys are fed from a real GDN block (blk.0); PLE from the real PLE
    block (blk.1); globals from token_embd / output_hc_*. Two legs:
    transcription-vs-pin (ref vs pin on the SAME fed tensors) = 0.0 exactly (the
    required parity leg), plus OV build_backbone vs ref.

    THE OV LEG WAS A DEFECT, NOT A NUMERICS STORY (resolved 2026-09-11,
    FIX-GDN-UTINV). This cell used to report max-abs 2.320e-04 with 2 of 64
    spike rows and call it "a few-token OV-emission numerics effect" for the
    later KLD gate, with the MoE top-k boundary and a router tie recorded as
    REFUTED. The actual cause was `q4e.gdn._ut_inverse` building the UT solve as
    a nilpotent geometric series, which collapses in f32 on real weights. With
    the pin's export-branch forward substitution the figure is 8.941e-08 with
    ZERO spike rows -- the float floor. There is no residual few-token effect.

    THIS CELL RUNS ONE SHAPE, T=64, and says so: T=64 is a single chunk, which
    is the most favourable shape for exactly the class of bug that hid here (the
    old error concentrated at the END of a chunk and only propagated from two
    chunks on). The real gate is
    test_gdn_emitter_real_weights_parity_sweep at T=64/96/128/256 against an f64
    recomputation. Full 48-layer full-width residency is window territory."""
    from q4e.backbone import build_backbone
    tb._assert_pin()
    config = tb._make_config()
    ref = tb._build_ref(config)
    sd = ref.state_dict()
    fed = 0
    for k in list(sd):
        if any(k.endswith(s) for s in gguf_feed._DERIVED_SUFFIXES):
            continue  # derived index buffer -- the pin recomputes it
        if k.startswith("layers."):
            gl = 1 if ".ple." in k else 0  # ple from blk.1, GDN/MoE/hc from blk.0
        else:
            gl = None
        sd[k] = torch.from_numpy(
            _fit(feed, k, tuple(sd[k].shape), gguf_layer=gl)).to(sd[k].dtype)
        fed += 1
    ref.load_state_dict(sd)
    pin = pin_mod.Qwen4ExpTextModel(config).eval()
    pin.load_state_dict(sd)
    md = _transcription_vs_pin(ref, pin)

    T = 64
    torch.manual_seed(500 + T)
    ids = torch.randint(1, config.vocab_size, (1, T))
    mask = torch.ones(1, T)
    with torch.no_grad():
        y_ref = ref.logits(ids, mask).float().numpy()
    row_ids = tb._gen_row_ids(config, tb._ple_index(config), ids[0].tolist())
    state = {kk: vv.detach().cpu().numpy() for kk, vv in ref.state_dict().items()}
    model = build_backbone(config, state, seq_len=T)
    y_ov = tb._run_ov(model, {
        "input_ids": ids.numpy().astype(np.int64),
        "ngram_row_ids": row_ids,
        "conv_mask": mask.numpy().astype(np.float32),
    }, "CPU")
    d = np.abs(y_ref - y_ov)
    ov_max = float(d.max())
    per_row = d.reshape(-1, config.vocab_size).max(1)
    spikes = int((per_row > 1e-4).sum())
    am_ref = y_ref.reshape(-1, config.vocab_size).argmax(1)
    am_ov = y_ov.reshape(-1, config.vocab_size).argmax(1)
    argmax_mismatch = int((am_ref != am_ov).sum())
    print(f"\n[feed-assembly] {fed} keys fed from GGUF  transcription-vs-pin={md:.3e}")
    print(f"[feed-assembly] OV-vs-ref(logits): max-abs={ov_max:.3e}  median-row={np.median(per_row):.2e}  "
          f"rows>1e-4={spikes}/{T}  argmax-mismatch={argmax_mismatch}/{T}  max|logit|={np.abs(y_ref).max():.2e}")
    # The required parity (kickoff step 3): the WHOLE backbone assembled from the
    # real GGUF tensors reproduces the pin's own forward EXACTLY.
    assert md == 0.0, f"transcription-vs-pin on fed weights: {md:.3e}"
    # Gated at the float floor now that FIX-GDN-UTINV removed the real cause.
    # These bounds are set from the measurement (8.941e-08 max, 3.35e-08 median,
    # 0 spike rows at T=64), not tuned to pass: the pre-fix emitter produced
    # 2.320e-04 / 2.17e-07 / 2 spike rows and fails every one of them.
    assert ov_max < 1e-5, f"OV assembly diverges on fed weights: {ov_max:.3e}"
    assert np.median(per_row) < 1e-6, f"OV median-row off the floor: {np.median(per_row):.3e}"
    assert spikes == 0, f"{spikes} spike rows -- the UT-inverse class is back"


# --- the head: pin-vs-checkpoint divergence (FIX B) ------------------------- #
# --- the real-weights T-sweep: the standing law's acceptance leg ----------- #
#
# STANDING LAW (frontier, FIX-GDN-UTINV, 2026-09-11): every emitter acceptance
# requires at least one REAL-WEIGHTS leg at >= 2 sequence lengths. Random-weight
# parity is NECESSARY, NEVER SUFFICIENT.
#
# Why the law exists, measured (REVIEW 58e3e09 findings 1 and 3): on the random
# fixtures the GDN block's intermediate powers DECAY (peak |(-L0)^p| 3.69e-02),
# so `_ut_inverse`'s series was harmless and every 1e-5 floor in
# test_backbone.py passed. On the shipped weights the same powers EXPLODE (peak
# 7.38e+04 against an O(1) answer) and the block's numerics collapsed. A fixture
# that makes every case alike certifies nothing -- the same class as the blind
# PA fill of 2026-09-08.
#
# The reference here is recomputed in FLOAT64 from the SAME fed tensors, so the
# leg measures the EMITTER, not the reference's own rounding: a reference whose
# f32 and f64 agree to ~1e-7 cannot excuse an emitter that departs from both.


@pytest.fixture(scope="module")
def fed_state(feed):
    """The tiny config's state dict with every weight fed from the real GGUF,
    built once: (config, ref_f32, ref_f64, state_np). `ref_f64` is the SAME fed
    tensors in double precision -- the yardstick the T-sweep measures against."""
    tb._assert_pin()
    config = tb._make_config()
    ref = tb._build_ref(config)
    sd = ref.state_dict()
    for k in list(sd):
        if any(k.endswith(s) for s in gguf_feed._DERIVED_SUFFIXES):
            continue
        gl = (1 if ".ple." in k else 0) if k.startswith("layers.") else None
        sd[k] = torch.from_numpy(
            _fit(feed, k, tuple(sd[k].shape), gguf_layer=gl)).to(sd[k].dtype)
    ref.load_state_dict(sd)
    ref64 = tb._build_ref(config).double()
    ref64.load_state_dict({k: v.double() if v.is_floating_point() else v
                           for k, v in ref.state_dict().items()})
    ref64.eval()
    state = {k: v.detach().cpu().numpy() for k, v in ref.state_dict().items()}
    return config, ref, ref64, state


@_skip
@pytest.mark.parametrize("T", [64, 96, 128, 256])
def test_gdn_emitter_real_weights_parity_sweep(feed, fed_state, T):
    """FIX-GDN-UTINV acceptance: the assembled backbone on REAL fed weights at
    four sequence lengths, each against an f64 recomputation of the same graph
    from the same tensors.

    Three distances per T:
      |ov - r32|  the emitter vs the f32 reference (what the old cell reported)
      |ov - r64|  the emitter vs the f64 truth     (THE gate)
      |r32 - r64| the reference's OWN f32 rounding (the floor to beat)

    The gate is relative, not a tuned constant: the emitter must sit within
    `_FLOOR_FACTOR` of the reference's own f32 rounding. Pre-fix, the nilpotent
    doubling series in `q4e.gdn._ut_inverse` missed it by ~4500x at T=96 and
    worsened with T, because its error concentrates at the END of a chunk and
    from two chunks on those rows enter the recurrent state (2/64 spike rows at
    T=64 becomes 16/96, 51/128, 40/256). T=64 was the single most favourable
    shape the old cell could have run."""
    from q4e.backbone import build_backbone
    config, ref, ref64, state = fed_state

    torch.manual_seed(500 + T)
    ids = torch.randint(1, config.vocab_size, (1, T))
    mask = torch.ones(1, T)
    with torch.no_grad():
        y32 = ref.logits(ids, mask).float().numpy().astype(np.float64)
        y64 = ref64.logits(ids, mask.double()).numpy()
    row_ids = tb._gen_row_ids(config, tb._ple_index(config), ids[0].tolist())
    y_ov = tb._run_ov(build_backbone(config, state, seq_len=T), {
        "input_ids": ids.numpy().astype(np.int64),
        "ngram_row_ids": row_ids,
        "conv_mask": mask.numpy().astype(np.float32),
    }, "CPU").astype(np.float64)

    V = config.vocab_size
    d_ov64 = np.abs(y_ov - y64)
    ov_r32 = float(np.abs(y_ov - y32).max())
    ov_r64 = float(d_ov64.max())
    r32_r64 = float(np.abs(y32 - y64).max())
    per_row = d_ov64.reshape(-1, V).max(1)
    median = float(np.median(per_row))
    spikes = int((per_row > 1e-4).sum())
    am = int((y32.reshape(-1, V).argmax(1) != y_ov.reshape(-1, V).argmax(1)).sum())
    closer = "OV" if ov_r64 < r32_r64 else "ref-f32"
    chunks = (T + 63) // 64

    print(f"\n[gdn-sweep] T={T:>3} chunks={chunks}  |ov-r32|={ov_r32:.3e}  "
          f"|ov-r64|={ov_r64:.3e}  |r32-r64|={r32_r64:.3e}  "
          f"median-row={median:.3e}  rows>1e-4={spikes}/{T}  argmax!={am}/{T}  "
          f"ratio(ov-r64)/(r32-r64)={ov_r64 / max(r32_r64, 1e-30):.1f}  closer={closer}")

    assert r32_r64 < 1e-5, (
        f"the f32 REFERENCE itself drifted from f64 ({r32_r64:.3e}) -- this "
        "computation is supposed to be stable; the yardstick is broken, stop")
    assert ov_r64 <= _FLOOR_FACTOR * r32_r64, (
        f"T={T}: emitter is {ov_r64 / max(r32_r64, 1e-30):.0f}x the reference's "
        f"own f32 rounding ({ov_r64:.3e} vs {r32_r64:.3e}) -- not at the float "
        "floor")


@_skip
def test_lm_head_is_not_tied_in_this_checkpoint(feed):
    """THE PIN TIES THE HEAD; THIS CHECKPOINT DOES NOT (the measurement, run as
    a cell so the claim cannot rot).

    Pin 1593: `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.
    weight"}`. The shipped UD-Q3_K_XL carries `output.weight` (Q6_K) AND
    `token_embd.weight` (Q8_0) as INDEPENDENT tensors: row-band cosine sits at
    the random-vector floor where a tie would give 1.0. Reported per band; the
    gate is that no band looks tied."""
    assert feed.has_lm_head(), "this checkpoint declares no output.weight"
    print(f"\n[head-tie] token_embd={feed.gguf_type('token_embd.weight')}  "
          f"output={feed.gguf_type('output.weight')}")
    worst = 0.0
    for lo in (0, 1000, 100000, 248000):
        n = 64
        emb = feed.dequant("token_embd.weight", rows=lo + n)[lo:lo + n]
        out = feed.dequant("output.weight", rows=lo + n)[lo:lo + n]
        cos = (emb * out).sum(1) / np.maximum(
            np.linalg.norm(emb, axis=1) * np.linalg.norm(out, axis=1), 1e-30)
        print(f"[head-tie] rows {lo}:{lo+n}  |embd|={np.linalg.norm(emb, axis=1).mean():.4f}  "
              f"|out|={np.linalg.norm(out, axis=1).mean():.4f}  "
              f"mean-cos={cos.mean():+.5f}  max-cos={cos.max():+.3f}")
        worst = max(worst, float(np.abs(cos).max()))
    # Tied => every |cos| == 1. The gate is generous on purpose: anything near
    # 1 would mean the head IS a duplicate and the fallback is the right wiring.
    assert worst < 0.5, (
        f"max |cos| {worst:.3f} across bands looks TIED -- if this checkpoint "
        "changed, the head wiring (q4e.backbone) must be revisited")


@_skip
def test_declared_head_fixture_feeds_output_weight(feed):
    """DECLARED-HEAD FIXTURE (new with FIX B; the absence of one is what hid the
    wrong head). Every prior fixture declares no head at all, so no cell could
    see the head wiring -- the tiny config's parity legs were 0.0 *because* both
    sides tied, not because the head was right.

    This cell builds the tiny model WITH a head, feeds it from the real
    `output.weight`, and checks three things:
      1. the fed head is NOT the fed embedding (hash + max-abs divergence);
      2. OV `build_backbone` reproduces the ref's fed-head logits;
      3. RED GUARD -- dropping `lm_head.weight` from the state (the pin's tie)
         changes the OV logits materially. Without this leg, an emitter that
         ignored the fed head would still pass leg 2 only by accident."""
    from q4e.backbone import build_backbone
    tb._assert_pin()
    config = tb._make_config()
    ref = tb._build_ref(config, declare_lm_head=True)
    sd = ref.state_dict()
    assert "lm_head.weight" in sd, "fixture did not declare a head"

    fed = 0
    for k in list(sd):
        if any(k.endswith(s) for s in gguf_feed._DERIVED_SUFFIXES):
            continue
        gl = (1 if ".ple." in k else 0) if k.startswith("layers.") else None
        sd[k] = torch.from_numpy(
            _fit(feed, k, tuple(sd[k].shape), gguf_layer=gl)).to(sd[k].dtype)
        fed += 1
    ref.load_state_dict(sd)

    # 1. the served head is an independent tensor, not the embedding
    emb_np = sd["embed_tokens.weight"].numpy()
    head_np = sd["lm_head.weight"].numpy()
    h_emb = hashlib.sha256(np.ascontiguousarray(emb_np).tobytes()).hexdigest()
    h_head = hashlib.sha256(np.ascontiguousarray(head_np).tobytes()).hexdigest()
    div = float(np.max(np.abs(emb_np - head_np)))
    print(f"\n[declared-head] {fed} keys fed  embed-sha={h_emb[:16]}  "
          f"head-sha={h_head[:16]}  max-abs(head-embed)={div:.3e}")
    assert h_head != h_emb, "fed head is byte-identical to the embedding"
    assert div > 0.0, "fed head does not diverge from the embedding"

    # the transcription leg still holds on the shared (headless) keys
    pin = pin_mod.Qwen4ExpTextModel(config).eval()
    pin.load_state_dict({k: v for k, v in sd.items() if not k.startswith("lm_head.")})
    md = _transcription_vs_pin(ref, pin)
    print(f"[declared-head] transcription-vs-pin (head-free keys)={md:.3e}")
    assert md == 0.0, md

    # 2. OV emits the FED head
    T = 64
    torch.manual_seed(500 + T)
    ids = torch.randint(1, config.vocab_size, (1, T))
    mask = torch.ones(1, T)
    with torch.no_grad():
        y_ref = ref.logits(ids, mask).float().numpy()
    row_ids = tb._gen_row_ids(config, tb._ple_index(config), ids[0].tolist())
    state = {k: v.detach().cpu().numpy() for k, v in ref.state_dict().items()}
    feed_in = {
        "input_ids": ids.numpy().astype(np.int64),
        "ngram_row_ids": row_ids,
        "conv_mask": mask.numpy().astype(np.float32),
    }
    y_ov = tb._run_ov(build_backbone(config, state, seq_len=T), feed_in, "CPU")
    ov_max = float(np.max(np.abs(y_ref - y_ov)))

    # 3. RED GUARD: the tie is a DIFFERENT model on this checkpoint
    tied_state = {k: v for k, v in state.items() if k != "lm_head.weight"}
    y_tied = tb._run_ov(build_backbone(config, tied_state, seq_len=T), feed_in, "CPU")
    tie_gap = float(np.max(np.abs(y_ov - y_tied)))
    print(f"[declared-head] OV-vs-ref(fed head)={ov_max:.3e}   "
          f"OV(fed head)-vs-OV(tied fallback)={tie_gap:.3e}")
    # Gated at the measured floor (1.891e-07 at the tip), not at the old 1e-2
    # sanity bound. PROVENANCE of this figure, which moved twice without either
    # mover saying so: 1.214e-03 at FIX B (222ac82) -> 5.385e-04 at 58e3e09
    # (FIX D changed the fed state) -> 1.891e-07 now (FIX-GDN-UTINV took the
    # GDN UT-inverse defect out of it; ~99.97% of the 5.385e-04 was that).
    # Both superseded values fail this bound, which is the point of asserting it.
    assert ov_max < 1e-5, f"OV does not reproduce the fed-head logits: {ov_max:.3e}"
    assert tie_gap > 1e-3, (
        f"tied fallback gives the same logits as the fed head ({tie_gap:.3e}) -- "
        "the head wiring is not observable, so this cell gates nothing")

# --- FIX A: the name map's OWN gate (test-side names, the map bypassed) ----- #
#
# Measured problem (REVIEW 2cd2b2f finding A, reproduced by the census cell
# below): 21 of the 30 `_LAYER_MAP` entries sit in a same-shape class, so a
# swapped pair produces a state dict that LOADS -- and because ref and pin then
# both consume the same mis-mapped tensor, every transcription-vs-pin leg above
# stays at exactly 0.0. The `== 0.0` legs gate the SHAPE / RESHAPE / FUSION
# half; they cannot gate the NAME half for any degenerate pair.
#
# The gate below is the two micro-cells' pattern (hardcoded GGUF names
# test-side, `q4e.gguf_feed` bypassed on the reference side) extended to EVERY
# mapped entry, degenerate or not. The names here are transcribed from the
# shipped tensor list and are deliberately NOT imported from the map -- that is
# the whole mechanism: swap two entries inside `_LAYER_MAP` and the two affected
# rows go red while every parity leg stays green.
#
# Kept complete, not merely "the degenerate ones": the coverage cell asserts
# this table names every map entry, so a NEW map entry lands red until someone
# writes down, independently, which GGUF tensor it is supposed to read.

_EXPECTED_LAYER = {
    # GatedDeltaNet
    "linear_attn.in_proj_qkv.weight": ("attn_qkv.weight", "direct2d"),
    "linear_attn.in_proj_z.weight": ("attn_gate.weight", "direct2d"),
    "linear_attn.in_proj_a.weight": ("ssm_alpha.weight", "direct2d"),
    "linear_attn.in_proj_b.weight": ("ssm_beta.weight", "direct2d"),
    "linear_attn.A_log": ("ssm_a", "vec"),
    "linear_attn.dt_bias": ("ssm_dt.bias", "vec"),
    "linear_attn.conv1d.weight": ("ssm_conv1d.weight", "conv"),
    "linear_attn.norm.weight": ("ssm_norm.weight", "vec"),
    "linear_attn.out_proj.weight": ("ssm_out.weight", "direct2d"),
    # SparseMoeBlock
    "mlp.gate.weight": ("ffn_gate_inp.weight", "direct2d"),
    "mlp.experts.gate_up_proj": (("ffn_gate_exps.weight", "ffn_up_exps.weight"),
                                 "fuse_gate_up"),
    "mlp.experts.down_proj": ("ffn_down_exps.weight", "expert3d"),
    "mlp.shared_expert.gate_proj.weight": ("ffn_gate_shexp.weight", "direct2d"),
    "mlp.shared_expert.up_proj.weight": ("ffn_up_shexp.weight", "direct2d"),
    "mlp.shared_expert.down_proj.weight": ("ffn_down_shexp.weight", "direct2d"),
    "mlp.shared_expert_gate.weight": ("ffn_gate_inp_shexp.weight", "row"),
    # hyper-connection mixers -- attn and mlp members are pairwise degenerate
    "attn_hyper_connection.hc_norm.weight": ("hc_attn_norm.weight", "vec"),
    "attn_hyper_connection.input_mix_weight_down.weight": ("hc_attn_down.weight", "direct2d"),
    "attn_hyper_connection.input_mix_weight_up.weight": ("hc_attn_up.weight", "direct2d"),
    "attn_hyper_connection.block_inject_weight.weight": ("hc_attn_inject.weight", "direct2d"),
    "mlp_hyper_connection.hc_norm.weight": ("hc_ffn_norm.weight", "vec"),
    "mlp_hyper_connection.input_mix_weight_down.weight": ("hc_ffn_down.weight", "direct2d"),
    "mlp_hyper_connection.input_mix_weight_up.weight": ("hc_ffn_up.weight", "direct2d"),
    "mlp_hyper_connection.block_inject_weight.weight": ("hc_ffn_inject.weight", "direct2d"),
    # DENSE-CAUSAL full attention (the 12 QSA blocks; blk.3 is the witness).
    # `attn_q` is the FUSED [query|gate] projection -- 12288 = heads 24 x 2 x
    # head_dim 256 -- not a query projection; the emitter splits it per head.
    "self_attn.q_proj.weight": ("attn_q.weight", "direct2d"),
    "self_attn.k_proj.weight": ("attn_k.weight", "direct2d"),
    "self_attn.v_proj.weight": ("attn_v.weight", "direct2d"),
    "self_attn.o_proj.weight": ("attn_output.weight", "direct2d"),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", "vec"),
    "self_attn.k_norm.weight": ("attn_k_norm.weight", "vec"),
    # PLE
    "ple.key_proj.weight": ("ple_key.weight", "direct2d"),
    "ple.value_proj.weight": ("ple_value.weight", "direct2d"),
    "ple.norm_key.weight": ("ple_norm_key.weight", "vec"),
    "ple.norm_query.weight": ("ple_norm_query.weight", "vec"),
    "ple.norm_conv.weight": ("ple_norm_conv.weight", "vec"),
    "ple.conv1d.weight": ("ple_conv1d.weight", "conv"),
}

_EXPECTED_GLOBAL = {
    "embed_tokens.weight": ("token_embd.weight", "direct2d"),
    # After dequant this is shape-identical to embed_tokens -- the head the pin
    # ties and this checkpoint does not (FIX B). A swap between the two is
    # exactly the degeneracy this gate exists for.
    "lm_head.weight": ("output.weight", "direct2d"),
    "hyper_connection_mixer.hc_norm.weight": ("output_hc_norm.weight", "vec"),
    "hyper_connection_mixer.input_mix_weight_down.weight": ("output_hc_down.weight", "direct2d"),
    "hyper_connection_mixer.input_mix_weight_up.weight": ("output_hc_up.weight", "direct2d"),
}

# blk index each layer key is read from: PLE lives on the real PLE block (1),
# everything else on a real GDN block (0). Same convention as the slice cells.
def _blk_of(suffix):
    # Which real block carries a family. `ple.*` ships on blk.1 only; the
    # DENSE-ATTENTION families (`self_attn.*`) ship ONLY on the 12 QSA blocks
    # (blk 3,7,...,47 -- `qwen4exp.attention.compress_ratios` is 4 exactly
    # there and 0 on the 36 GDN blocks), so blk.0 has no attn_q/k/v at all and
    # asking for one is a KeyError, not a miss. Everything else is per-block
    # and blk.0 is the cheapest witness.
    if suffix.startswith("ple."):
        return 1
    if suffix.startswith("self_attn."):
        return 3
    return 0


_ROW_KINDS = ("direct2d", "expert3d", "fuse_gate_up")
_GATE_ROWS = 2   # leading-axis rows to compare; enough to separate any two tensors


@pytest.fixture(scope="module")
def raw_index():
    """An INDEPENDENT GGUF index: a plain `GGUFReader`, no q4e code anywhere in
    the path. The reference side of the content-hash gate must not share a line
    of name-resolution with the thing under test."""
    from gguf import GGUFReader
    if os.path.isdir(_SHARDS):
        paths = sorted(glob.glob(os.path.join(_SHARDS, "*.gguf")))
    else:
        paths = sorted(glob.glob(_SHARDS))
    assert paths, f"no shards for {_SHARDS!r}"
    idx = {}
    for p in paths:
        for t in GGUFReader(p).tensors:
            idx[t.name] = t
    return idx


def _raw_dequant(raw_index, name, rows=None):
    """Independent dequant: gguf.quants straight off the reader's block data."""
    from gguf import quants as _Q
    t = raw_index[name]
    data = t.data if rows is None else t.data[:rows]
    tn = t.tensor_type.name
    if tn in ("F32", "F16", "BF16"):
        return np.ascontiguousarray(np.asarray(data), dtype=np.float32)
    return _Q.dequantize(data, t.tensor_type).astype(np.float32)


def _expected_array(raw_index, gname, kind, rows):
    """The array the map SHOULD produce, built test-side: independent dequant +
    an independently written reshape for each kind."""
    r = rows if kind in _ROW_KINDS else None
    if kind == "fuse_gate_up":
        gate = _raw_dequant(raw_index, gname[0], rows=r)
        up = _raw_dequant(raw_index, gname[1], rows=r)
        return np.concatenate([gate, up], axis=1)
    arr = _raw_dequant(raw_index, gname, rows=r)
    if kind in ("direct2d", "vec", "expert3d"):
        return arr
    if kind == "row":
        return arr.reshape(1, -1)
    if kind == "conv":
        return arr.reshape(arr.shape[0], 1, arr.shape[1])
    raise AssertionError(f"unhandled kind {kind!r}")


def _sha(arr):
    return hashlib.sha256(
        np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()


@_skip
def test_name_map_gate_covers_every_entry():
    """The gate's own coverage: the test-side expectation table names EVERY map
    entry. A new `_LAYER_MAP` / `_GLOBAL_MAP` key lands red here until someone
    writes down -- independently of the map -- which GGUF tensor it reads."""
    missing_layer = sorted(set(gguf_feed._LAYER_MAP) - set(_EXPECTED_LAYER))
    extra_layer = sorted(set(_EXPECTED_LAYER) - set(gguf_feed._LAYER_MAP))
    missing_global = sorted(set(gguf_feed._GLOBAL_MAP) - set(_EXPECTED_GLOBAL))
    extra_global = sorted(set(_EXPECTED_GLOBAL) - set(gguf_feed._GLOBAL_MAP))
    print(f"\n[map-gate-coverage] layer={len(_EXPECTED_LAYER)}/{len(gguf_feed._LAYER_MAP)}  "
          f"global={len(_EXPECTED_GLOBAL)}/{len(gguf_feed._GLOBAL_MAP)}")
    assert not missing_layer, f"map entries with no test-side expectation: {missing_layer}"
    assert not missing_global, f"global entries with no test-side expectation: {missing_global}"
    assert not extra_layer, f"expectations for absent layer keys: {extra_layer}"
    assert not extra_global, f"expectations for absent global keys: {extra_global}"


@_skip
def test_name_map_degeneracy_census(feed, raw_index):
    """WHY the content-hash gate exists, measured rather than asserted.

    Group every mapped entry by the shape it has AFTER dequant + reshape (after
    dequant everything is f32, so shape alone decides whether a swap still
    loads). Entries sharing a class are interchangeable without any `== 0.0`
    leg noticing: ref and pin both read the same wrong tensor and the
    difference cancels identically. Reported, with the class membership, so the
    number in the header of this section cannot go stale.

    THE CENSUS IS TAKEN FROM THE MAP UNDER TEST, not from this file's
    expectation table (corrected 2026-09-11, REVIEW 58e3e09 finding 6). The
    coverage assertion at the end used to build `covered` from the very list it
    had just grouped, so it could not fail by construction -- it read as a gate
    and was not one, the exact shape the round before was called to fix. Now the
    entries come from `gguf_feed._LAYER_MAP` / `_GLOBAL_MAP` and the coverage
    target is `_EXPECTED_LAYER` / `_EXPECTED_GLOBAL`: a degenerate MAP entry
    with no test-side expectation behind it is a real hole and lands red here.
    (Shapes are read by name from the map, but a swapped name inside a
    degenerate class has the same shape by definition, so the census measures
    shape classes and is indifferent to a swap -- which is what the per-entry
    content-hash rows are for.)"""
    entries = []
    for suffix, (gname, kind) in gguf_feed._LAYER_MAP.items():
        blk = _blk_of(suffix)
        names = gname if isinstance(gname, tuple) else (gname,)
        entries.append((f"layers.{blk}.{suffix}",
                        tuple(f"blk.{blk}.{g}" for g in names), kind, suffix))
    for key, (gname, kind) in gguf_feed._GLOBAL_MAP.items():
        entries.append((key, (gname,), kind, key))

    cls = defaultdict(list)
    for key, names, kind, _ in entries:
        t = raw_index[names[0]]
        logical = tuple(int(x) for x in reversed(t.shape))
        if kind == "conv":
            logical = (logical[0], 1, logical[1])
        elif kind == "row":
            logical = (1, logical[0])
        elif kind == "fuse_gate_up":
            logical = (logical[0], 2 * logical[1], logical[2])
        cls[logical].append(key)

    degenerate = {k: v for k, v in cls.items() if len(v) > 1}
    n_deg = sum(len(v) for v in degenerate.values())
    print(f"\n[map-degeneracy] {len(entries)} mapped entries, {n_deg} of them in a "
          f"same-shape class ({len(degenerate)} classes) -- a swap inside such a "
          f"class LOADS and leaves every ==0.0 leg green")
    for shape, members in sorted(degenerate.items(), key=lambda kv: -len(kv[1])):
        print(f"[map-degeneracy]   {list(shape)}: {', '.join(sorted(members))}")
    assert n_deg >= 21, (
        f"only {n_deg} entries look degenerate -- if the checkpoint's geometry "
        "changed, re-read finding A before trusting the ==0.0 legs")
    # ... and every degenerate MAP entry must have a test-side expectation
    # behind it, i.e. be a row of the content-hash gate. Independent sources:
    # the census comes from the map, `gated` from this file's own table.
    gated = {f"layers.{_blk_of(s)}.{s}" for s in _EXPECTED_LAYER} | set(_EXPECTED_GLOBAL)
    holes = sorted({m for members in degenerate.values() for m in members} - gated)
    print(f"[map-degeneracy] gate rows={len(gated)}  degenerate entries outside "
          f"the gate={holes if holes else 'none'}")
    assert not holes, (
        f"degenerate map entries with no content-hash row behind them: {holes} "
        "-- a swap inside their class would pass every cell in this file")


@_skip
@pytest.mark.parametrize("suffix", sorted(_EXPECTED_LAYER))
def test_name_map_entry_content_layer(feed, raw_index, suffix):
    """THE NAME-MAP GATE, one row per per-layer entry: what `pin_tensor` returns
    for this pin key must be byte-identical to the tensor the test-side table
    says it is -- read by an independent `GGUFReader`, dequantised
    independently, reshaped independently. Swap this entry with its degenerate
    twin inside `_LAYER_MAP` and THIS row goes red; the parity legs do not."""
    gname, kind = _EXPECTED_LAYER[suffix]
    blk = _blk_of(suffix)
    names = gname if isinstance(gname, tuple) else (gname,)
    full = tuple(f"blk.{blk}.{g}" for g in names)
    got = feed.pin_tensor(f"layers.{blk}.{suffix}", rows=_GATE_ROWS)
    want = _expected_array(raw_index, full if len(full) > 1 else full[0],
                           kind, _GATE_ROWS)
    g_sha, w_sha = _sha(got), _sha(want)
    print(f"\n[map-gate] {suffix} -> {'|'.join(full)}  shape={list(got.shape)}  "
          f"sha={g_sha[:16]}  {'OK' if g_sha == w_sha else 'MISMATCH'}")
    assert got.shape == want.shape, (got.shape, want.shape)
    assert g_sha == w_sha, (
        f"{suffix}: map read a DIFFERENT tensor than {'|'.join(full)} "
        f"(got {g_sha[:16]}, want {w_sha[:16]})")


@_skip
@pytest.mark.parametrize("key", sorted(_EXPECTED_GLOBAL))
def test_name_map_entry_content_global(feed, raw_index, key):
    """The same gate for the non-per-layer entries -- including
    `embed_tokens.weight` vs `lm_head.weight`, which are shape-identical after
    dequant and would swap silently (FIX B)."""
    gname, kind = _EXPECTED_GLOBAL[key]
    got = feed.pin_tensor(key, rows=_GATE_ROWS)
    want = _expected_array(raw_index, gname, kind, _GATE_ROWS)
    g_sha, w_sha = _sha(got), _sha(want)
    print(f"\n[map-gate] {key} -> {gname}  shape={list(got.shape)}  "
          f"sha={g_sha[:16]}  {'OK' if g_sha == w_sha else 'MISMATCH'}")
    assert got.shape == want.shape, (got.shape, want.shape)
    assert g_sha == w_sha, (
        f"{key}: map read a DIFFERENT tensor than {gname} "
        f"(got {g_sha[:16]}, want {w_sha[:16]})")


# --- FIX C's residency figures, promoted from literals to facts ------------ #
@_skip
def test_refusal_residency_figures_are_recomputed_from_the_file(raw_index):
    """REVIEW 58e3e09 finding 5: the full-size refusal's residency numbers
    (then 656.9 / 461.4 / 195.5 / 190.7 GiB, "12 of 48", "120 tensors,
    2.30 GiB") were f-string literals with nothing behind them.

    THE CELL EARNED ITS KEEP, 2026-09-12: mapping the six dense-attention
    families moved four of those eight figures (total 656.9 -> 659.1, per-block
    461.4 -> 463.6, unmapped 120 -> 48 tensors and 2.30 -> 0.07 GiB, the 2.2 GiB
    delta being exactly the attn_q/k/v/output + q_norm/k_norm families crossing
    from unmapped to mapped) and this cell is what caught the stale text. An
    orphan branch had added the map entries and left the refusal saying 120. They were correct -- the
    reviewer recomputed every one -- but nothing re-derived them, so they would
    rot silently if the checkpoint changed. FIX B set the better precedent in
    that same round (`test_lm_head_is_not_tied_in_this_checkpoint` exists
    precisely "so the claim cannot rot"); C did not follow it. This cell does.

    Literals that assert are facts; literals that sit are drift.

    Everything here is recomputed from the shipped tensor list through an
    INDEPENDENT `GGUFReader` -- nelem x 4 bytes, the f32-ov-Constant assumption
    the refusal itself names -- and then the refusal's OWN text is required to
    contain each number as it is formatted here. Change the checkpoint and this
    goes red at the exact figure that moved."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from export_qwen4_exp import build_backbone_ir

    try:
        build_backbone_ir("/nonexistent", {"n_layer": 48, "n_embd": 2560,
                                           "num_experts": 512}, "/no/such/shards")
        raise AssertionError("full-size emission did not refuse")
    except NotImplementedError as e:
        msg = str(e)

    GB = 1024.0 ** 3

    def nbytes(t):
        n = 1
        for x in t.shape:
            n *= int(x)
        return n * 4                      # f32 ov Constant, the stated assumption

    # the per-block GGUF suffixes the causal map consumes
    mapped = set()
    for g, _kind in gguf_feed._LAYER_MAP.values():
        mapped.update(g if isinstance(g, tuple) else (g,))
    globals_ = {g for g, _ in gguf_feed._GLOBAL_MAP.values()} | {gguf_feed._PLE_TABLE}

    per_block = glob_bytes = 0
    unmapped_n = unmapped_bytes = 0
    blocks, qsa, gdn = set(), set(), set()
    for name, t in raw_index.items():
        if name.startswith("blk."):
            _, idx, suffix = name.split(".", 2)
            blocks.add(int(idx))
            if ".attn_q." in name:
                qsa.add(int(idx))
            if suffix.startswith("ssm_a"):
                gdn.add(int(idx))
            if suffix in mapped:
                per_block += nbytes(t)
            else:
                unmapped_n += 1
                unmapped_bytes += nbytes(t)
        elif name in globals_:
            glob_bytes += nbytes(t)

    per_block_gib = per_block / GB
    glob_gib = glob_bytes / GB
    total_gib = per_block_gib + glob_gib
    unmapped_gib = unmapped_bytes / GB
    ple = raw_index[gguf_feed._PLE_TABLE]
    ple_gib = nbytes(ple) / GB
    ple_shape = tuple(int(x) for x in ple.shape)

    print(f"\n[residency] blocks={len(blocks)}  QSA={len(qsa)} {sorted(qsa)[:4]}...  GDN={len(gdn)}")
    print(f"[residency] per-block mapped={per_block_gib:.1f} GiB  globals={glob_gib:.1f} GiB  "
          f"total={total_gib:.1f} GiB  (f32 ov Constants)")
    print(f"[residency] PLE n-gram table {list(ple_shape)} {ple.tensor_type.name} "
          f"-> {ple_gib:.2f} GiB f32")
    print(f"[residency] unmapped (ruled scope): {unmapped_n} tensors  {unmapped_gib:.2f} GiB")

    # every figure the refusal states must be the figure the file yields
    for label, text in (
        ("total", f"{total_gib:.1f} GiB"),
        ("per-block", f"{per_block_gib:.1f} GiB"),
        ("globals", f"{glob_gib:.1f} GiB"),
        ("PLE table", f"{ple_gib:.1f} GiB"),
        ("unmapped bytes", f"{unmapped_gib:.2f} GiB"),
        ("unmapped count", f"{unmapped_n} tensors"),
        ("QSA blocks", f"{len(qsa)} of its {len(blocks)} blocks"),
        ("PLE table shape", f"{ple_shape[1]:,} x {ple_shape[0]}"),
    ):
        assert text in msg, (
            f"refusal's {label} figure is stale: the file yields {text!r}, which "
            f"does not appear in the refusal text")

    # and the QSA block list the refusal names
    assert sorted(qsa) == list(range(3, 48, 4)), sorted(qsa)
    assert f"blk {min(qsa)},{sorted(qsa)[1]},...,{max(qsa)}" in msg.replace(", ", ",")
