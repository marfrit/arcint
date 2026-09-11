"""E2 increment 5b -- the assembled qwen4_exp text backbone, parity for
`q4e.backbone.build_backbone`.

RED state: `from q4e.backbone import build_backbone` -- the module does not
exist yet (GREEN emits tools/q4e/backbone.py + adds `from . import backbone` to
q4e/__init__.py). Collecting this file fails; the other test files still pass.

DENSE-CAUSAL SCOPE (CORRECTION, 2026-09-12 -- supersedes the CAUSAL-ONLY
paragraph below, which over-read the frontier ruling). The checkpoint is
48 layers = 36 GDN + 12 full-attention layers (layer_idx % 4 == 3). The
frontier ruled dense causal IS the semantics; only the QSA INDEXER is ruled
out (its per-query `nonzero` is not statically opset-13-emittable, E1.5
finding 7). The full-attention layers assemble as DENSE CAUSAL, emitted from
the pin's own Qwen4ExpTextAttention (pin 819-901) minus the selection branch,
with RoPE in its degenerate-for-text form -- tools/q4e/attention.py. This
suite's tiny fixtures stay all-linear (the parity floor for the harness
shape); the real-width dense-causal piece is exercised piecewise in
tests/python/test_piecewise_export.py.

Pin facts this increment establishes (pin re-verified on the dev host, sha ca9f00bb):
  * PLE is ADDITIVE (hidden += ple(...), pin 1283-1284), not a layer replacement.
  * NO final RMSNorm: TextModel returns hyper_connection_mixer(hidden) (pin 1493)
    and lm_head is applied to it directly (pin 1669, tied to embed_tokens). The
    self.norm at pin 1807 is the VISION patch merger (a trap), not assembled.

Config fixture: small-but-complete. 4 GDN layers, a PLE layer at ple_layer_ids
[2] (1-indexed -> layer_idx 1, ple_layer_index 0), every layer an MoE. hidden 16,
hc 4, ngram_size 3 / heads_per_ngram 2 (4 n-gram heads), num_experts 8 / top-k 2,
vocab 257.

Legs:
  * ov-parity: OV logits vs ref_backbone.logits (the transcription) -- max-abs
    table by T (64/96) + KLD on the logits head.
  * ov-parity (mask-partial): trailing-zeros conv_mask, both sides -- max-abs.
  * transcription-vs-pin: ref_backbone last_hidden vs the pin Qwen4ExpTextModel
    forward -- 0.0 (each block's transcription is already 0.0-validated in
    inc1-5a; this checks the composition wiring against the pin's own forward).

Row ids for the PLE layer are produced by the in-file numpy int64 generator (the
frontier-ruled test-side producer; serving uses src/exec/ngram_row_ids.h). No
threshold tuning anywhere: parity floors are reported as measured.

Standing cell (adopted from the REVIEWER seat, E2 stack re-review 2026-09-11):
test_backbone_ov_parity_interior_mask_hole -- T=48, seed 3, an INTERIOR conv_mask
hole [18:30). Frontier-suggested and first-run by the reviewer; adopted here as
authored. A trailing-zeros mask alone (test_backbone_ov_parity_masked) never
exercises a valid position AFTER a masked one, so it cannot catch a
prefix-assuming emitter; the interior hole does.

Run (dev-host venv):
    Q4E_GPU=  ~/openarc-venv/bin/python3 -m pytest tests/python/test_backbone.py -s --continue-on-collection-errors
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import openvino as ov  # noqa: E402
from transformers.models.qwen4_exp import configuration_qwen4_exp as pin_cfg  # noqa: E402
from transformers.models.qwen4_exp import modeling_qwen4_exp as pin_mod  # noqa: E402

from q4e import ref_backbone  # noqa: E402
from q4e.backbone import build_backbone  # noqa: E402  (RED: module absent)


PIN_SHA256 = {
    "modeling_qwen4_exp.py": "ca9f00bbd73cfcfbad7ba6073b5ecc23ca27bdace58b746fe6ead6ab175efc0c",
    "configuration_qwen4_exp.py": "b78132d8cd935437208ee281fa4569b771a63fcb58ebffe84f3e62f5b86235ca",
}


def _assert_pin():
    import transformers.models.qwen4_exp as pkg

    pkg_dir = Path(pkg.__file__).resolve().parent
    for name, want in PIN_SHA256.items():
        got = hashlib.sha256((pkg_dir / name).read_bytes()).hexdigest()
        assert got == want, f"oracle drift: {name} sha256 {got} != pin {want}"


def _make_config():
    return pin_cfg.Qwen4ExpTextConfig(
        hidden_size=16,
        num_hidden_layers=4,
        hc_count=4,
        hc_lowrank=8,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"] * 4,
        # MoE
        num_experts=8, num_experts_per_tok=2, norm_topk_prob=True,
        moe_intermediate_size=32, shared_expert_intermediate_size=32,
        hidden_act="silu",
        # GDN
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        # PLE
        ple_layer_ids=[2], ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=17, make_ngram_vocab_size_divisible_by=128,
        ple_embed_dim=32, ple_conv_kernel_size=4, seed=1234,
        vocab_size=257, eos_token_id=0, pad_token_id=0,
    )


def _build_ref(config, seed=0, declare_lm_head=False):
    """Random tiny reference. `declare_lm_head=True` registers the SEPARATE
    head (`lm_head.weight`) the shipped checkpoint ships as `output.weight` --
    the pin TextModel has no such key, so any leg that loads this state dict
    into the pin must drop it (see test_gguf_feed's declared-head cell)."""
    torch.manual_seed(seed)
    ref = ref_backbone.Qwen4ExpTextBackbone(
        config, declare_lm_head=declare_lm_head).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0.0, 0.05)
    return ref


def _state_np(module):
    return {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}


def _device_params():
    devs = ["CPU"]
    extra = os.environ.get("Q4E_GPU", "").strip()
    if extra:
        devs += [d for d in (s.strip() for s in extra.split(",")) if d]
    return devs


def _kld(p_logits, q_logits):
    def sm(x):
        x = x.astype(np.float64)
        x = x - x.max(-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(-1, keepdims=True)

    p, q = sm(p_logits), sm(q_logits)
    return float(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=-1)))


# --- in-file numpy int64 row-id generator (test-side producer) --------------
_MASK64 = (1 << 64) - 1


def _s64(x):
    x &= _MASK64
    return x - (1 << 64) if x >= (1 << 63) else x


def _derive(config, ple_idx):
    ng, hpn = config.ngram_size, config.heads_per_ngram
    mult = pin_mod._build_layer_multipliers(config.vocab_size, ng, ple_idx, config.seed).tolist()
    sizes, offs, tot = [], [], 0
    for h in range((ng - 1) * hpn):
        gh = ple_idx * (ng - 1) * hpn + h
        sz = pin_mod._find_nth_prime_after(config.ngram_vocab_size_base - 1, gh + 1)
        sizes.append(sz); offs.append(tot); tot += sz
    return mult, sizes, offs


def _gen_row_ids(config, ple_idx, tokens):
    ng, hpn, eos = config.ngram_size, config.heads_per_ngram, int(config.eos_token_id)
    mult, sizes, offs = _derive(config, ple_idx)
    packed = [eos] * (ng - 1) + [int(t) for t in tokens]
    W = len(packed)
    prev, last = [-1] * W, -1
    for p in range(W):
        prev[p] = last
        if packed[p] == eos:
            last = p
    in_seg = [p - prev[p] - 1 for p in range(W)]
    shifted = [list(packed)] + [
        [packed[p - s] if (p - s >= 0 and in_seg[p] >= s) else eos for p in range(W)]
        for s in range(1, ng)
    ]
    per = [[r[(ng - 1) + i] for i in range(len(tokens))] for r in shifted]
    out = []
    for i in range(len(tokens)):
        heads = []
        for n in range(2, ng + 1):
            st = (n - 2) * hpn
            mixed = _s64(per[0][i] * mult[0])
            for pos in range(1, n):
                mixed = _s64((mixed & _MASK64) ^ (_s64(per[pos][i] * mult[pos]) & _MASK64))
            for h in range(st, st + hpn):
                heads.append(mixed % sizes[h] + offs[h])
        out.append(heads)
    return np.array(out, dtype=np.int64)[None]  # [1,T,Hn]


def _ple_index(config):
    """The ple_layer_index of the (single) PLE layer -- position of its 1-indexed
    id in ple_layer_ids (pin 1268). For ple_layer_ids [2] that is 0."""
    return list(config.ple_layer_ids).index(config.ple_layer_ids[0])


def _run_ov(model, feed, device):
    compiled = ov.Core().compile_model(model, device)
    out = compiled(feed)
    return out[compiled.outputs[0]]


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T", [64, 96])
@pytest.mark.parametrize("device", _device_params())
def test_backbone_ov_parity(device, T):
    _assert_pin()
    config = _make_config()
    ref = _build_ref(config)
    state = _state_np(ref)

    torch.manual_seed(500 + T)
    ids = torch.randint(1, config.vocab_size, (1, T))
    mask = torch.ones(1, T)
    with torch.no_grad():
        y_ref = ref.logits(ids, mask).float().numpy()

    row_ids = _gen_row_ids(config, _ple_index(config), ids[0].tolist())
    model = build_backbone(config, state, seq_len=T)
    y_ov = _run_ov(model, {
        "input_ids": ids.numpy().astype(np.int64),
        "ngram_row_ids": row_ids,
        "conv_mask": mask.numpy().astype(np.float32),
    }, device)

    V = config.vocab_size
    max_abs = float(np.max(np.abs(y_ref - y_ov)))
    kld = _kld(y_ref.reshape(-1, V), y_ov.reshape(-1, V))
    print(f"\n[backbone-ov-parity] device={device:<6} T={T:>3}  max-abs(logits)={max_abs:.3e}  KLD={kld:.3e}")
    assert max_abs < 1e-5, f"backbone parity failed device={device} T={T}: {max_abs:.3e}"


@pytest.mark.parametrize("device", _device_params())
def test_backbone_ov_parity_masked(device):
    _assert_pin()
    config = _make_config()
    ref = _build_ref(config)
    state = _state_np(ref)
    T = 96
    live = T // 2

    torch.manual_seed(777)
    ids = torch.randint(1, config.vocab_size, (1, T))
    mask = torch.ones(1, T)
    mask[:, live:] = 0.0
    with torch.no_grad():
        y_ref = ref.logits(ids, mask).float().numpy()

    row_ids = _gen_row_ids(config, _ple_index(config), ids[0].tolist())
    model = build_backbone(config, state, seq_len=T)
    y_ov = _run_ov(model, {
        "input_ids": ids.numpy().astype(np.int64),
        "ngram_row_ids": row_ids,
        "conv_mask": mask.numpy().astype(np.float32),
    }, device)

    V = config.vocab_size
    max_abs = float(np.max(np.abs(y_ref - y_ov)))
    print(f"\n[backbone-ov-parity-masked] device={device:<6} T={T:>3} live={live}  max-abs(logits)={max_abs:.3e}")
    assert max_abs < 1e-5, f"backbone masked parity failed device={device}: {max_abs:.3e}"


@pytest.mark.parametrize("device", _device_params())
def test_backbone_ov_parity_interior_mask_hole(device):
    """Standing cell adopted from the REVIEWER seat (E2 stack re-review,
    2026-09-11): an INTERIOR conv_mask hole [18:30) at T=48, seed 3 -- a
    non-trailing mask. Rules out a prefix-assuming emitter: a trailing-zeros
    mask never exercises a valid position after a masked one, an interior hole
    does. Frontier-suggested, first-run by the reviewer; floor reported, no
    tuning. OV-vs-ref (both sides get the identical interior mask)."""
    _assert_pin()
    config = _make_config()
    ref = _build_ref(config, seed=3)
    state = _state_np(ref)
    T = 48

    torch.manual_seed(3)
    ids = torch.randint(1, config.vocab_size, (1, T))
    mask = torch.ones(1, T)
    mask[:, 18:30] = 0.0                                     # INTERIOR hole
    with torch.no_grad():
        y_ref = ref.logits(ids, mask).float().numpy()

    row_ids = _gen_row_ids(config, _ple_index(config), ids[0].tolist())
    model = build_backbone(config, state, seq_len=T)
    y_ov = _run_ov(model, {
        "input_ids": ids.numpy().astype(np.int64),
        "ngram_row_ids": row_ids,
        "conv_mask": mask.numpy().astype(np.float32),
    }, device)

    V = config.vocab_size
    max_abs = float(np.max(np.abs(y_ref - y_ov)))
    print(f"\n[backbone-ov-parity-interior-hole] device={device:<6} T={T}  hole=[18:30)  max-abs(logits)={max_abs:.3e}")
    assert max_abs < 1e-5, f"backbone interior-hole parity failed device={device}: {max_abs:.3e}"


def test_backbone_head_declared_vs_tied_fallback():
    """The head wiring, device-free (FIX B, the fallback half).

    `build_backbone` takes `state["lm_head.weight"]` when the state provides one
    and falls back to the pin's tie to `embed_tokens` (pin 1593) only when it is
    absent. Both halves are asserted here on random weights, so the wiring is
    covered without the real shards:
      * declared head -> OV logits match `ref.logits` (which applies the head,
        pin 1669) AND differ from the tied emission;
      * no head in the state -> OV logits match the tie exactly.
    The real-weights half (fed from the checkpoint's `output.weight`, which is
    NOT tied -- measured) is `test_declared_head_fixture_feeds_output_weight`
    in test_gguf_feed.py."""
    _assert_pin()
    config = _make_config()
    T = 48
    torch.manual_seed(11)
    ids = torch.randint(1, config.vocab_size, (1, T))
    mask = torch.ones(1, T)

    # -- declared head -------------------------------------------------------
    ref_h = _build_ref(config, seed=5, declare_lm_head=True)
    state_h = _state_np(ref_h)
    assert "lm_head.weight" in state_h
    with torch.no_grad():
        y_ref = ref_h.logits(ids, mask).float().numpy()
    row_ids = _gen_row_ids(config, _ple_index(config), ids[0].tolist())
    feed = {
        "input_ids": ids.numpy().astype(np.int64),
        "ngram_row_ids": row_ids,
        "conv_mask": mask.numpy().astype(np.float32),
    }
    y_ov = _run_ov(build_backbone(config, state_h, seq_len=T), feed, "CPU")
    head_err = float(np.max(np.abs(y_ref - y_ov)))

    # -- tied fallback (same weights, head key removed) ----------------------
    tied_state = {k: v for k, v in state_h.items() if k != "lm_head.weight"}
    y_tied = _run_ov(build_backbone(config, tied_state, seq_len=T), feed, "CPU")
    with torch.no_grad():
        h = ref_h(ids, mask)
        y_tie_ref = (h @ ref_h.embed_tokens.weight.t()).float().numpy()
    tie_err = float(np.max(np.abs(y_tie_ref - y_tied)))
    gap = float(np.max(np.abs(y_ov - y_tied)))

    print(f"\n[backbone-head] declared: OV-vs-ref={head_err:.3e}   "
          f"fallback: OV-vs-tied-ref={tie_err:.3e}   declared-vs-tied gap={gap:.3e}")
    assert head_err < 1e-5, f"declared head not emitted: {head_err:.3e}"
    assert tie_err < 1e-5, f"tied fallback drifted: {tie_err:.3e}"
    assert gap > 1e-3, (
        f"declared and tied emissions coincide ({gap:.3e}) -- the cell gates nothing")


def test_transcription_matches_pin():
    """ref_backbone (pin leaves, transcribed composition) reproduces the pin
    Qwen4ExpTextModel forward EXACTLY -- the composition-wiring check against the
    pin's own forward (each block's leaf math is already 0.0-validated in
    inc1-5a). Full sequence (no padding); conv_mask = all-valid."""
    _assert_pin()
    config = _make_config()
    ref = _build_ref(config)
    T = 64
    torch.manual_seed(9)
    ids = torch.randint(1, config.vocab_size, (1, T))
    mask = torch.ones(1, T)

    pin = pin_mod.Qwen4ExpTextModel(config).eval()
    pin.load_state_dict(ref.state_dict())
    with torch.no_grad():
        h_ref = ref(ids, mask)                                   # [1,T,H]
        out = pin(input_ids=ids, attention_mask=None, use_cache=False)
        h_pin = out.last_hidden_state
    md = float((h_ref - h_pin).abs().max())
    print(f"\n[backbone-transcription-vs-pin] T={T}  max-abs(ref last_hidden - pin)={md:.3e}")
    assert md == 0.0, f"backbone transcription drifted from pin: {md:.3e}"
