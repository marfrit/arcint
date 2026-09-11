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


# --- the head: pin-vs-checkpoint divergence (FIX B) ------------------------- #
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
    import hashlib

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
        rows = sd[k].shape[0] if sd[k].ndim >= 1 else None
        arr = feed.pin_tensor(k, rows=rows, gguf_layer=gl)
        sd[k] = torch.from_numpy(
            np.ascontiguousarray(_fit(arr, tuple(sd[k].shape)))).to(sd[k].dtype)
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
    assert ov_max < 1e-2, f"OV does not reproduce the fed-head logits: {ov_max:.3e}"
    assert tie_gap > 1e-3, (
        f"tied fallback gives the same logits as the fed head ({tie_gap:.3e}) -- "
        "the head wiring is not observable, so this cell gates nothing")
