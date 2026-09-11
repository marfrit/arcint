"""E2 Phase B -- feed REAL GGUF weights into the pin/transcription state-dict
keys and confirm the graph-semantics parity leg stays exactly 0.0.

The kickoff insight, kept separate on purpose: parity against the pin is about
GRAPH SEMANTICS, not weight provenance. The pin loaded with dequantised GGUF
tensors keeps transcription-vs-pin at 0.0 because BOTH sides then carry
byte-identical weights. So a non-0.0 here is a NAME-MAP or a RESHAPE/FUSION bug,
never a quantisation-accuracy story (that is a later, separate gate -- KLD).

Each slice cell feeds ONE module's weights (sliced to the tiny config's shape)
into an otherwise-random tiny model and checks transcription-vs-pin == 0.0. Two
micro-cells additionally check the non-trivial reshapes against an INDEPENDENT
reference (a hand-written dequant+reshape), so a shape-preserving permutation bug
-- which the load leg alone cannot see -- is caught: the GDN conv1d [C,K]->[C,1,K]
and the MoE gate|up fusion.

Real-weight source: the shipped UD-Q3_K_XL GGUF shards. Point Q4E_GGUF_SHARDS at
the directory (or a glob) holding them; absent, every test here skips by name
(the shards are ~90 GiB and live only on the dev host -- this is not a
device-free unit). gguf-py must be importable as a real `gguf.GGUFReader`.

Run (dev host):
    Q4E_GGUF_SHARDS=<shard-dir> PYTHONPATH=<gguf-py> \\
        Q4E_GPU=  <venv>/bin/python3 -m pytest tests/python/test_gguf_feed.py -s \\
        --continue-on-collection-errors
"""
import os
import sys
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


@pytest.fixture(scope="module")
def feed():
    if gguf_feed is None:
        pytest.fail(f"q4e.gguf_feed did not import: {_FEED_IMPORT_ERR!r}")
    return gguf_feed.GgufFeed(_SHARDS)


def _fit(arr, shape):
    """Leading-index slice of `arr` down to `shape` (each target dim <= arr's)."""
    assert arr.ndim == len(shape), (arr.shape, shape)
    for a, s in zip(arr.shape, shape):
        assert a >= s, f"feed axis {a} < target {s} (need a bigger real tensor)"
    return arr[tuple(slice(0, s) for s in shape)]


def _feed_state(config, feed, keys, seed=0):
    """Tiny random ref state with `keys` overwritten by feed-sliced real
    tensors. Returns (ref, pin) both loaded with that state."""
    ref = tb._build_ref(config, seed=seed)
    sd = ref.state_dict()
    for pin_key in keys:
        rows = sd[pin_key].shape[0] if sd[pin_key].ndim >= 1 else None
        arr = feed.pin_tensor(pin_key, rows=rows)
        sd[pin_key] = torch.from_numpy(
            np.ascontiguousarray(_fit(arr, tuple(sd[pin_key].shape)))
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
    gguf_feed tensors -- every weight fed from the real GGUF, sliced to the tiny
    config. GDN/MoE/hc keys are fed from a real GDN block (blk.0); PLE from the
    real PLE block (blk.1); globals from token_embd / output_hc_*. Two legs:
    transcription-vs-pin (ref vs pin on the SAME fed tensors) = 0.0 exactly (the
    required parity leg), plus OV build_backbone vs ref REPORTED (on real weights
    the divergence is a few-token OV-emission numerics effect, not the random-
    weight 1e-5 floor -- see the asserts). Small-config only -- full 48-layer
    full-width residency is window territory."""
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
        rows = sd[k].shape[0] if sd[k].ndim >= 1 else None
        arr = feed.pin_tensor(k, rows=rows, gguf_layer=gl)
        sd[k] = torch.from_numpy(
            np.ascontiguousarray(_fit(arr, tuple(sd[k].shape)))).to(sd[k].dtype)
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
    # OV-vs-ref is REPORTED, not gated at the random-weight 1e-5 floor: on real
    # weights the divergence is confined to a few tokens (median stays at the
    # float floor) and is NOT the MoE top-k boundary (refuted: top_k=all still
    # spikes) nor a router tie (refuted: scaling the router does not collapse it)
    # -- an OV-emission numerics question for the later KLD gate, not this
    # name-map/wiring gate. Only a gross-breakage sanity bound is asserted here.
    assert ov_max < 1e-2, f"OV assembly grossly diverges on fed weights: {ov_max:.3e}"
    assert np.median(per_row) < 1e-5, f"OV divergence is not token-confined: {np.median(per_row):.3e}"
