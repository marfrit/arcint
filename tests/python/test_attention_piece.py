"""The DENSE-CAUSAL full-attention piece, at the checkpoint's REAL width, on
fed real GGUF tensors, against the pin itself.

THE ORACLE IS THE PIN, NOT A TRANSCRIPTION (decided 2026-09-12, salvage triage).
An orphan branch shipped a hand-written f64 `ref_attention.DenseAttention`
alongside the emitter. Both had read the pin's fused [query|gate] projection the
same wrong way, so the parity leg between them was GREEN on a defect that put
heads 12-23's query block into the gate. That is the standing law's failure mode
in its purest form: a reference written by the author of the emitter certifies
the author's misreading. `Qwen4ExpTextAttention` runs on CPU at real width in
both float32 and float64 here (measured: 0.8-3.0 s per shape), so there is no
reason to transcribe it at all. ref_gdn.py exists because the GDN's chunked
kernel has no runnable pin path; attention has no such excuse.

The pin's own class is driven in three configurations, all from ONE class:
  DENSE  -- `self.indexer` replaced by a stub returning a zero additive mask.
            The pin then adds zero to the causal mask (pin 857-858) and its own
            code path IS dense causal. f32 and f64.
  QSA    -- the real indexer, its weights fed from the checkpoint. This is the
            thing the ruling approximates.

THE RULING'S PRICE IS ZERO BELOW THE INDEXER BUDGET -- measured, and it
supersedes the orphan's premise that the leg must be KLD-shaped because "QSA
legitimately prunes". It does not prune, below the budget. For query i the
indexer sees i+1 visible tokens, forms num_complete_blocks = (i+1)//ratio of
them and keeps topk(min(block_topk, num_complete_blocks)) (pin 734, 757) with
block_topk = budget//ratio = 2048//4 = 512 (the definition is pin 684; pin 734
and 757 are `num_complete_blocks` and the min). While (i+1)//4 <= 512 that min
is num_complete_blocks, i.e. EVERY complete block is selected, and the
incomplete tail is added unconditionally -- the selected set is every visible
token and the overlaid mask equals the causal mask exactly.

THE BOUNDARY IS 2051, NOT 2048 (CF-BOUNDS, corrected 2026-09-12; REVIEW e78812d
F6). Row i is dense iff (i+1)//ratio <= block_topk. With ratio 4 and block_topk
512 that is i+1 <= 2051, i.e. i <= 2050, because 2051//4 == 512 and 2052//4 ==
513. So EVERY row of a prefill is dense iff

    T <= block_topk * ratio + ratio - 1  ==  budget + ratio - 1  ==  2051

and the number of pruned rows at any T is exactly max(0, T - 2051). The prose
here said "up to ~2048" and the cell branched on `T <= budget`, which is wrong
in the interval {2049, 2050, 2051}: the price there is still exactly 0.0 while
the cell took the `else` branch and demanded `md > 0.0`. Not live at the time
(the parametrisation was 64/96/2080), but a gate that would fail on a correct
result is a defect in a file whose stated standard is derived-not-tuned bounds.
The boundary is now the gate: 2051 and 2052 are parametrised, the threshold is
DERIVED from the config rather than written down, and the row count is asserted
EXACTLY rather than as "> 0".

    T=  64   |dense - QSA| max-abs 0.000000e+00    rows differing    0/64
    T=  96   |dense - QSA| max-abs 0.000000e+00    rows differing    0/96
    T=2051   |dense - QSA| max-abs 0.000000e+00    rows differing    0/2051
    T=2052   |dense - QSA| max-abs <sampled>       rows differing    1/2052
    T=2080   |dense - QSA| max-abs 2.385560e-02    rows differing   29/2080

[DATED 2026-09-28: the magnitudes above (and the 1.720381e-02 second draw
below) were drawn at 692c0a6 with the (1 + w) fold applied twice to all four
q/k gammas -- the main attention's (undone by the feed since c41cb39) and the
indexer's (undone in the `indexer_state` fixture on 2026-09-28). Re-run with
both undone: 5.082879e-05 over 1/2052 and 1.064551e-03 over 29/2080 rows. The
row counts are structural and did not move.]

The zeros are exact and input-independent -- the masks are equal, so the
arithmetic is the same arithmetic. The T=2080 magnitude is NOT: a second draw
(seeded differently, same geometry and weights) gave 1.720381e-02 over the same
29 rows. The ROW COUNT is the structural quantity and the magnitude is a sample,
so the cell asserts the EXACT count and pastes the magnitude rather than
bounding it. 29 rows at T=2080 = 2080 - 2051, and 1 row at T=2052 -- the first
pruned row is i=2051, which is what makes 2051/2052 the pair that pins the
boundary from both sides.

That makes the parity leg at serving-relevant prefill lengths EQUALITY-shaped
against the pin's REAL QSA path -- a far stronger gate than a KLD-shaped one,
and the price figure is a measured 0.0 rather than a tolerated residue. The
non-zero price is real and is measured above the budget, where it belongs; the
end-to-end KLD gate decides small-enough for T > 2048, later.

THE INDEXER HAS FOUR QUERY HEADS, not three (corrected 2026-09-12). The file
says so -- `qwen4exp.attention.indexer.head_count` = 4 -- and the load is the
second, load-bearing witness: the pin builds ONE fused `index_qk_proj` of
(n_heads + kv_heads) * head_dim (pin 685-689) and the checkpoint ships it SPLIT
as indexer.q_proj [512, 2560] + indexer.k_proj [128, 2560]; 512 + 128 = 640 =
5 x 128 only at n_heads=4. `test_indexer_head_count_is_four_or_the_pin_cannot_
load_it` asserts both directions.

Every cell re-hashes the oracle before it measures. CPU by default and no card
is touched; `Q4E_GPU=GPU.0,GPU.1` adds the device legs, which require the card to
be FREE (a resident arcint service holds all of its VRAM) and which pin
INFERENCE_PRECISION_HINT f32 and print the precision the plugin actually used --
see tests/python/q4e_device.py for why an unconfigured GPU compile runs f16.
"""
import hashlib
import math
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
from q4e_device import compile_for, device_params, effective_precision  # noqa: E402
from transformers.models.qwen4_exp import modeling_qwen4_exp as pin_mod  # noqa: E402

from q4e import attention as qattn  # noqa: E402
from q4e import gguf_feed  # noqa: E402
from q4e import piecewise_export as pwe  # noqa: E402

PIN_SHA256 = {
    "modeling_qwen4_exp.py":
        "ca9f00bbd73cfcfbad7ba6073b5ecc23ca27bdace58b746fe6ead6ab175efc0c",
}

_SHARDS = os.environ.get("Q4E_GGUF_SHARDS", "").strip()
_skip_shards = pytest.mark.skipif(
    not _SHARDS,
    reason="Q4E_GGUF_SHARDS unset: the real-width attention legs need the "
           "shipped GGUF shards")

# The standing relative gate: how far above the reference's OWN f32-vs-f64
# rounding the emitter may sit. Identical to the GDN sweep's, so a drifting
# yardstick cannot buy a pass.
_FLOOR_FACTOR = 20.0
_YARDSTICK_CEILING = 1e-5   # if the f32 pin itself leaves this, stop

_ATTN_KEYS = ("q_proj.weight", "k_proj.weight", "v_proj.weight",
              "o_proj.weight", "q_norm.weight", "k_norm.weight")


def _assert_pin():
    import transformers.models.qwen4_exp as pkg
    d = Path(pkg.__file__).resolve().parent
    for name, want in PIN_SHA256.items():
        got = hashlib.sha256((d / name).read_bytes()).hexdigest()
        assert got == want, f"oracle drift: {name} sha256 {got} != pin {want}"


@pytest.fixture(scope="module")
def cfg():
    _assert_pin()
    c = pwe.real_config()
    c._attn_implementation = "eager"   # pin 882-884 interface lookup
    return c


@pytest.fixture(scope="module")
def feed():
    return gguf_feed.GgufFeed(_SHARDS)


@pytest.fixture(scope="module")
def attn_state(feed):
    """The six dense-attention tensors for the first QSA block (blk.3), through
    the name map under test."""
    return {k: feed.pin_tensor(f"layers.3.self_attn.{k}") for k in _ATTN_KEYS}


@pytest.fixture(scope="module")
def indexer_state(feed):
    """The indexer's OWN weights, through the name map under test. The fused
    `index_qk_proj` is the checkpoint's q_proj rows followed by its k_proj
    rows, which is what the pin's own split (pin 707-711) takes apart again.

    The two norm gammas are gguf_feed kind `gamma1`. The converter folds them,
    stored = 1 + w (llama.cpp `conversion/qwen4exp.py`, `data_torch + 1` on
    `.indexer.{q,k}_layernorm.weight`), and llama.cpp applies them as a plain
    RMSNorm (`qwen4exp.cpp` build_norm). The pin's RMSNorm adds the 1 itself
    (pin 171). Until 2026-09-28 this fixture read the stored values raw, so
    the pin applied the fold twice (a scale of ~1.96). Steps 1-2 held parity
    anyway, because both sides read the same state; the convention check below
    is what makes that regression failable."""
    pre = "layers.3.self_attn.indexer."
    state = {f"indexer.{k}": feed.pin_tensor(pre + k)
             for k in ("index_qk_proj.weight", "q_layernorm.weight",
                       "k_layernorm.weight")}
    for k in ("indexer.q_layernorm.weight", "indexer.k_layernorm.weight"):
        m = float(np.mean(state[k]))
        # w sits near 0 (blk.3: -0.04); the folded 1 + w near 1 (0.96)
        assert abs(m) < 0.5, (
            f"{k}: mean {m:.4f} -- the pin wants w, not the stored (1 + w)")
    return state


class _ZeroIndexer(torch.nn.Module):
    """A stand-in for `Qwen4ExpTextQSAIndexer` that selects everything: the pin
    adds its result to the attention mask (pin 857-858), so a zero additive mask
    leaves the causal mask untouched and the pin's own forward becomes dense
    causal. Nothing about the attention path is bypassed."""

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values):
        return torch.zeros_like(attention_mask)


def _pin_attention(cfg, attn_state, indexer_state, dense, dtype):
    m = pin_mod.Qwen4ExpTextAttention(cfg, layer_idx=3).eval()
    sd = {k: torch.from_numpy(np.ascontiguousarray(v).copy())
          for k, v in attn_state.items()}
    sd.update({k: torch.from_numpy(v.copy()) for k, v in indexer_state.items()})
    m.load_state_dict(sd, strict=True)      # strict: no silent shape escape
    if dense:
        m.indexer = _ZeroIndexer()
    return m.to(dtype)


def _causal(T, dtype):
    """The pin's eager-path additive mask: 0 where visible, finfo(f32).min
    strictly above the diagonal. f32's min in BOTH widths on purpose -- the f64
    leg must price the same mask the f32 one does, not a wider -inf."""
    m = np.triu(np.full((T, T), np.float32(np.finfo(np.float32).min),
                        np.float32), 1)
    return torch.from_numpy(m.reshape(1, 1, T, T)).to(dtype)


def _cos_sin(cfg, T, dtype):
    cosT, sinT = qattn._freqs_tables(cfg, T)
    return (torch.from_numpy(cosT).unsqueeze(0).to(dtype),
            torch.from_numpy(sinT).unsqueeze(0).to(dtype))


def _hidden(cfg, T):
    """Finite hidden states at the scale a real residual carries. Seeded per T
    so the shapes are not correlated samples of one draw."""
    g = torch.Generator().manual_seed(1000 + T)
    return (torch.randn(1, T, cfg.hidden_size, generator=g) * 0.02).float()


def _run_ov(model, args, device="CPU"):
    """compile_for, never a bare compile_model: on a GPU device the plugin
    defaults INFERENCE_PRECISION_HINT to f16 and the parity floors here live at
    1e-7 (see tests/python/q4e_device.py)."""
    compiled = compile_for(ov.Core(), model, device)
    return np.asarray(compiled(args)[0]), effective_precision(compiled)


# --------------------------------------------------------------------------- #
# 1. the geometry correction, proven by the load
# --------------------------------------------------------------------------- #
@_skip_shards
def test_indexer_head_count_is_four_or_the_pin_cannot_load_it(cfg, indexer_state):
    """`qwen4exp.attention.indexer.head_count` is 4 in the file. The shapes are
    the independent witness: only at n_heads=4 does the pin's fused projection
    have room for the checkpoint's q_proj|k_proj concatenation."""
    fused = indexer_state["indexer.index_qk_proj.weight"]
    assert fused.shape == (640, 2560), fused.shape
    assert cfg.indexer_n_heads == 4 and cfg.indexer_kv_heads == 1
    widths = {}
    for n in (3, 4):
        c = pwe.real_config()
        c.indexer_n_heads = n
        c._attn_implementation = "eager"
        w = tuple(pin_mod.Qwen4ExpTextAttention(
            c, layer_idx=3).indexer.index_qk_proj.weight.shape)
        widths[n] = w
    sys.stdout.write(
        f"\n[indexer-geometry] checkpoint fused {fused.shape}  "
        f"pin@n_heads=3 {widths[3]}  pin@n_heads=4 {widths[4]}\n")
    assert widths[4] == fused.shape, (
        "n_heads=4 must make the pin's fused projection match the checkpoint")
    assert widths[3] != fused.shape, (
        "n_heads=3 must NOT fit -- if it does, this assertion is vacuous and "
        "the head-count correction has lost its proof")


# --------------------------------------------------------------------------- #
# 2. the baked rope tables ARE the pin's rotary module
# --------------------------------------------------------------------------- #
def _rope_tables_f64(cfg, T):
    """The same spec as `_freqs_tables`, in float64, written here independently
    of the module under test. In f64 the last-bit noise of pow/cos/sin is ~1e-16
    and a STRUCTURAL error (wrong mrope offsets, a missing doubling, the
    transpose dropped) is O(1), so this is the oracle both f32 implementations
    are measured against."""
    theta = float(cfg.rope_parameters["rope_theta"])
    partial = float(cfg.rope_parameters.get("partial_rotary_factor", 1.0))
    dim = int(cfg.head_dim * partial)
    sec = list(cfg.rope_parameters["mrope_section"])
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    f = np.outer(np.arange(T, dtype=np.float64), inv)        # [T, dim//2]
    rows = np.stack([f, f, f], axis=0)                       # text: 3 equal rows
    thw = rows[0].copy()
    for d, off in ((1, 1), (2, 2)):
        idx = slice(off, sec[d] * 3, 3)
        thw[..., idx] = rows[d][..., idx]
    thw = np.concatenate([thw, thw], axis=-1)
    return np.cos(thw), np.sin(thw)


@pytest.mark.parametrize("T", [64, 96])
def test_baked_rope_tables_are_the_pin_rotary_module(cfg, T):
    """`_freqs_tables` bakes what `Qwen4ExpTextRotaryEmbedding` produces for the
    text-degenerate positions.

    TWO DIFFERENT FLOORS, and the first draft of this cell confused them
    (corrected 2026-09-12, measured):

      * the two f32 towers -- baked and the pin's -- agree to 5.960e-08 = 2^-24,
        one ulp at |cos| <= 1. That is numpy's and torch's libm differing in the
        last bit of pow/cos/sin. Asserting 0.0 here would be asserting that two
        libms round identically.
      * BOTH f32 towers sit 1.759e-06 (T=64) / 2.545e-06 (T=96) from an exact
        f64 computation -- thirty times further than they sit from each other.
        That is NOT libm noise and not a defect: the rope argument runs up to
        (T-1) * max(inv_freq) = T-1 (inv_freq[0] = theta^0 = 1), f32's ulp AT
        THAT ARGUMENT is 3.815e-06 at T=64 and 7.629e-06 at T=96, and cos/sin
        carry the argument's rounding through with |derivative| <= 1. So the
        f32-vs-f64 floor is the ARGUMENT's ulp, not the output's, and it grows
        with T exactly as measured.

    Hence the gate: each f32 tower within the DERIVED argument-ulp bound of f64
    (a structural error -- wrong mrope offsets, a dropped transpose, a missing
    doubling -- is O(1) and blows through it by five orders of magnitude), the
    two f32 towers within two output ulp of each other, neither tower
    systematically worse than the other, and the doubling the pin's
    `cat((freqs_thw, freqs_thw))` puts there (pin 149) exact."""
    out_ulp = 2.0 ** -24
    arg_ulp = float(np.spacing(np.float32(max(T - 1, 1))))
    cosT, sinT = qattn._freqs_tables(cfg, T)
    cos64, sin64 = _rope_tables_f64(cfg, T)
    rot = pin_mod.Qwen4ExpTextRotaryEmbedding(cfg).eval()
    # pin 1433-1434 / 1438-1440: four identical rows for text; rows 1-3 are what
    # reaches the rotary module.
    pos = torch.arange(T, dtype=torch.long).view(1, 1, -1).expand(4, 1, -1)
    with torch.no_grad():
        cos_p, sin_p = rot(torch.zeros(1, T, cfg.hidden_size), pos[1:])
    cos_p, sin_p = cos_p.numpy()[0], sin_p.numpy()[0]

    mine = max(float(np.max(np.abs(cosT - cos64))),
               float(np.max(np.abs(sinT - sin64))))
    pin = max(float(np.max(np.abs(cos_p - cos64))),
              float(np.max(np.abs(sin_p - sin64))))
    both = max(float(np.max(np.abs(cosT - cos_p))),
               float(np.max(np.abs(sinT - sin_p))))
    half = cosT.shape[-1] // 2
    dbl = max(float(np.max(np.abs(cosT[:, :half] - cosT[:, half:]))),
              float(np.max(np.abs(sinT[:, :half] - sinT[:, half:]))))
    sys.stdout.write(
        f"\n[rope] T={T} rotary={cosT.shape[-1]} shape={tuple(cos_p.shape)}  "
        f"|baked-f64| {mine:.3e}  |pin-f64| {pin:.3e}  |baked-pin| {both:.3e}  "
        f"arg-ulp(T-1) {arg_ulp:.3e}  out-ulp {out_ulp:.3e}  "
        f"doubling-residual {dbl:.3e}\n")
    assert cos_p.shape == (T, cosT.shape[-1])
    assert dbl == 0.0, (
        f"the baked tables are not the pin's cat(freqs, freqs): the two halves "
        f"differ by {dbl:.3e}")
    assert pin <= arg_ulp, (
        f"the PIN's own f32 tables left the f32 argument ulp ({pin:.3e} > "
        f"{arg_ulp:.3e}) -- the oracle for this cell is broken, stop")
    assert mine <= arg_ulp, (
        f"baked rope tables are {mine / arg_ulp:.2f}x the f32 argument ulp from "
        f"the f64 spec ({mine:.3e} > {arg_ulp:.3e}) -- a structural "
        "transcription error, not rounding")
    assert both <= 2 * out_ulp, (
        f"baked tables and the pin's differ by {both / out_ulp:.1f} output ulp "
        f"({both:.3e}), more than two libms can explain")
    assert mine <= 2.0 * pin and pin <= 2.0 * mine, (
        f"one f32 tower is systematically further from f64 than the other "
        f"(baked {mine:.3e} vs pin {pin:.3e}) -- they should round alike")


# --------------------------------------------------------------------------- #
# 3. the oracle's own property, and the ruling's PRICE
# --------------------------------------------------------------------------- #
@_skip_shards
@pytest.mark.parametrize("T", [64, 96, 2051, 2052, 2080])
def test_qsa_price_is_zero_below_the_budget(cfg, attn_state, indexer_state, T):
    """DENSE vs QSA, both the pin, real weights. Below the boundary the indexer
    selects every visible token and the two are bit-identical; above it, the
    divergence is the ruling's price and is PASTED, never tolerated silently.

    The boundary is DERIVED from the config here, not written down, and it is
    `block_topk * ratio + ratio - 1` = 2051, not the budget 2048 (CF-BOUNDS,
    REVIEW e78812d F6). 2051 and 2052 are parametrised so the boundary itself
    is the gate rather than a claim in the prose, and the count of pruned rows
    is asserted EXACTLY: max(0, T - 2051)."""
    hidden = _hidden(cfg, T)
    cs = _cos_sin(cfg, T, torch.float32)
    dense = _pin_attention(cfg, attn_state, indexer_state, True, torch.float32)
    qsa = _pin_attention(cfg, attn_state, indexer_state, False, torch.float32)
    with torch.no_grad():
        od, _ = dense(hidden, cs, _causal(T, torch.float32), None)
        oq, _ = qsa(hidden, cs, _causal(T, torch.float32), None)
    diff = (od - oq).abs()
    md = float(diff.max())
    rows = int(diff.amax(-1).squeeze(0).gt(0).sum())
    budget = cfg.indexer_budget
    ratio = cfg.indexer_compress_ratio
    block_topk = budget // ratio                     # pin 684
    # Row i is dense iff (i+1)//ratio <= block_topk (pin 734, 757), i.e.
    # i+1 <= block_topk*ratio + ratio - 1. Every row of a T-token prefill is
    # dense iff T <= that. Derived, never a literal -- a config change moves it.
    dense_max_T = block_topk * ratio + ratio - 1
    expected_rows = max(0, T - dense_max_T)
    sys.stdout.write(
        f"\n[qsa-price] T={T:5d} block_topk={block_topk} "
        f"dense_max_T={dense_max_T} |dense-QSA| max-abs {md:.6e}  "
        f"rows differing {rows}/{T}  (expected {expected_rows})\n")
    assert dense_max_T == 2051, (
        f"the derivation moved: block_topk {block_topk} x ratio {ratio} + "
        f"{ratio} - 1 = {dense_max_T}, but the pin's geometry gives 2051")
    assert rows == expected_rows, (
        f"T={T}: the indexer prunes exactly the rows with (i+1)//{ratio} > "
        f"{block_topk}, i.e. i >= {dense_max_T}, so {expected_rows} rows should "
        f"differ -- measured {rows}. The boundary is structural, not a sample")
    if expected_rows == 0:
        assert md == 0.0, (
            f"T={T} is at or below the dense boundary {dense_max_T}, where every "
            f"complete block is selected (topk(min(block_topk, nb)) == nb), so "
            f"dense and QSA must be bit-identical -- got max-abs {md:.3e}")
    else:
        assert md > 0.0, (
            f"T={T} exceeds the dense boundary {dense_max_T}; the indexer must "
            f"prune and the price must be visible, otherwise this measures nothing")


# --------------------------------------------------------------------------- #
# 4. THE PARITY LEG: the emitted piece against the pin's REAL QSA path
# --------------------------------------------------------------------------- #
@_skip_shards
@pytest.mark.parametrize("device", device_params())
@pytest.mark.parametrize("T", [64, 96])
def test_dense_attention_piece_parity_real_weights(cfg, attn_state,
                                                   indexer_state, T, device):
    """Real width (H=2560, heads 24, kv 2, head_dim 256, rotary 64), real fed
    tensors, the emitted OV piece vs the pin WITH ITS REAL QSA INDEXER.

    Because T is inside the indexer budget the QSA path is the dense path
    exactly (cell 3 proves it independently), so this is equality-shaped
    against the thing that actually ships, floored by the pin's own f32-vs-f64
    rounding. Two gates, in this order: the yardstick must be sound, and only
    then may the emitter be judged against it.

    The GPU legs run only when Q4E_GPU names a device AND the card is free (a
    resident service holds all of its VRAM). They carry
    INFERENCE_PRECISION_HINT f32 and PRINT the precision the compiled model
    reports, so a leg cannot claim f32 while the plugin ran f16."""
    hidden = _hidden(cfg, T)
    pid = np.arange(T, dtype=np.int64).reshape(1, T)

    model = qattn.build_dense_attention_model(cfg, attn_state, T)
    ov_out, prec = _run_ov(model, [hidden.numpy(), pid], device)

    qsa32 = _pin_attention(cfg, attn_state, indexer_state, False, torch.float32)
    dense64 = _pin_attention(cfg, attn_state, indexer_state, True, torch.float64)
    with torch.no_grad():
        p32, _ = qsa32(hidden, _cos_sin(cfg, T, torch.float32),
                       _causal(T, torch.float32), None)
        p64, _ = dense64(hidden.double(), _cos_sin(cfg, T, torch.float64),
                         _causal(T, torch.float64), None)
    p32 = p32.numpy()
    p64 = p64.numpy()

    yardstick = float(np.max(np.abs(p32 - p64.astype(np.float32))))
    d_ov64 = float(np.max(np.abs(ov_out - p64)))
    d_ov32 = float(np.max(np.abs(ov_out - p32)))
    med = float(np.median(np.max(np.abs(ov_out - p64), axis=-1)))
    ratio = d_ov64 / yardstick if yardstick > 0 else float("inf")
    closer = "OV" if d_ov64 < yardstick else "pin-f32"
    sys.stdout.write(
        f"\n[attn-parity] dev={device} prec={prec} T={T}  |ov-pinQSA32| "
        f"{d_ov32:.3e}  |ov-pin64| {d_ov64:.3e}  |pin32-pin64| "
        f"{yardstick:.3e}  median-row {med:.3e}  ratio {ratio:.1f}x  "
        f"closer-to-f64 {closer}\n")

    if device.upper().startswith("GPU"):
        assert "f32" in prec or "float32" in prec, (
            f"{device} compiled at {prec}, not f32 -- the floors below are "
            "meaningless at f16; fix the compile config before reading them")
    assert yardstick <= _YARDSTICK_CEILING, (
        f"the f32 PIN itself drifted from f64 ({yardstick:.3e}) -- the "
        "yardstick is broken, stop")
    assert d_ov64 <= _FLOOR_FACTOR * yardstick, (
        f"T={T} on {device}: emitter is {ratio:.0f}x the pin's own f32 rounding "
        f"({d_ov64:.3e} vs {yardstick:.3e}) -- not at the float floor")


# --------------------------------------------------------------------------- #
# 4b. CF-ROPEAB: ONE VARIABLE AT A TIME, and the variable is named
# --------------------------------------------------------------------------- #
# REVIEW e78812d F5: ab69ea9's record puts "4 failed, 5 passed" immediately
# above "same host, same invocation -> 9 passed", which reads as one transition
# with one variable. It is not: two of those four failures came from the
# suite's own first-draft rope bound, which the SAME commit corrected. Red
# against suite revision A, green against revision B.
#
# Suite revision A is NOT RECOVERABLE -- ab69ea9 added this file whole, with the
# corrected bound already in it, and the first draft existed only inside that
# session. Rather than reconstruct it from prose, the attribution is closed
# from the other side, by measurement:
#
#   * the emitter A/B was re-run at 692c0a6 with the archived orphan emitter
#     (sha256 ca262ca2...) swapped in, ONE variable, suite fixed:
#         orphan    -> 2 failed, 9 passed   (exactly the two parity legs)
#         committed -> 11 passed
#     and the two `[rope]` lines are BYTE-IDENTICAL across those two legs
#     (|baked-f64| 1.759e-06 / 2.545e-06, |baked-pin| 5.960e-08, arg-ulp
#     3.815e-06 / 7.629e-06, doubling 0.000e+00). The rope cell does not touch
#     the emitter's attention path at all, so the two rope failures in the
#     record CANNOT have been caused by the emitter. The variable was the bound.
#
#   * the cell below makes that permanent without depending on an untracked
#     archive file: each of ab69ea9's two numeric defects is INJECTED
#     SEPARATELY into the committed emitter and its own ratio measured, so the
#     964999x headline is attributed rather than asserted.

def _flat_split_weight(w, heads, d):
    """The orphan's DEFECT 1 expressed as a permutation of `q_proj.weight`.

    The committed emitter does the pin's thing: view [heads, 2d], take
    [:, 0:d] as query and [:, d:2d] as gate (pin 867-869). The orphan sliced
    the LEADING heads*d rows of the flat 2*heads*d axis as query and the
    trailing block as gate. Those two readings of the SAME graph differ only in
    which output rows land where, so feeding the committed emitter a permuted
    weight reproduces the orphan's tensor exactly -- no second emitter, no
    monkeypatch, and the equivalence is checkable (it is, against the archived
    orphan's own number; see the docstring below).

        correct picks, for head h:  query rows h*2d + [0..d)
                                    gate  rows h*2d + [d..2d)
        orphan  picks, for head h:  query rows      h*d + [0..d)
                                    gate  rows heads*d + h*d + [0..d)
    """
    out = np.empty_like(w)
    for h in range(heads):
        out[h * 2 * d: h * 2 * d + d] = w[h * d: h * d + d]
        out[h * 2 * d + d: (h + 1) * 2 * d] = \
            w[heads * d + h * d: heads * d + (h + 1) * d]
    return out


@_skip_shards
@pytest.mark.parametrize("T", [64, 96])
def test_each_ab69ea9_defect_is_attributed_on_its_own(cfg, attn_state,
                                                      indexer_state, T):
    """The 964999x headline, split into its two causes, one variable per row.

    Defect 1 -- the [query|gate] split read off the flat axis instead of per
    head -- is injected as the weight permutation above. Defect 2 -- RMSNorm
    summing where the pin means (pin 164) -- is injected by replacing the
    emitter's `_rmean` with a reduce_sum for the duration of one build, which
    is exactly what the orphan's line 120 did.

    The gate is that each defect ALONE is catastrophic and the clean emitter is
    at the floor: a defect that only shows up in combination would mean the
    parity leg is passing for a compensating reason.

    Cross-check on defect 1's reconstruction: with BOTH defects injected the
    ratio must land on the archived orphan's own measured figure (964999.3x at
    T=64, 469606.6x at T=96, re-measured at 692c0a6 with the orphan file
    swapped in), which is what makes the permutation a faithful stand-in for a
    file this repository does not track.

    ARCHIVAL CONVENTION (2026-09-18): those two ratios were measured while
    the feed handed the q/k norm gammas over as the GGUF stores them, (1 + w)
    -- a converter fold the feed undoes since gguf_feed's kind `gamma1`
    (DESIGN 7.0.2bz). The cross-check against the archived figures needs the
    same numbers, so this cell re-folds the two gammas for every row; the
    attribution itself (each defect alone catastrophic, the clean emitter at
    the floor) does not depend on the convention -- both the pin and the
    emitter read the same state."""
    attn_state = dict(attn_state)
    for k in ("q_norm.weight", "k_norm.weight"):
        attn_state[k] = np.asarray(attn_state[k], np.float32) + np.float32(1.0)
    hidden = _hidden(cfg, T)
    pid = np.arange(T, dtype=np.int64).reshape(1, T)
    heads = cfg.num_attention_heads
    d = cfg.head_dim

    dense64 = _pin_attention(cfg, attn_state, indexer_state, True, torch.float64)
    qsa32 = _pin_attention(cfg, attn_state, indexer_state, False, torch.float32)
    with torch.no_grad():
        p64 = dense64(hidden.double(), _cos_sin(cfg, T, torch.float64),
                      _causal(T, torch.float64), None)[0].numpy()
        p32 = qsa32(hidden, _cos_sin(cfg, T, torch.float32),
                    _causal(T, torch.float32), None)[0].numpy()
    yardstick = float(np.max(np.abs(p32 - p64.astype(np.float32))))
    assert yardstick > 0.0, "no yardstick: the f32 and f64 pins agree bitwise"

    def _ratio(state, sum_not_mean):
        saved = qattn._rmean
        if sum_not_mean:
            # the orphan's line 120: reduce_sum where pin 164 means -> a factor
            # sqrt(head_dim) = 16 on the normalised vector
            qattn._rmean = (lambda x, axis:
                            qattn.op.reduce_sum(x, qattn._i([axis]), True))
        try:
            model = qattn.build_dense_attention_model(cfg, state, T)
            out, _ = _run_ov(model, [hidden.numpy(), pid], "CPU")
        finally:
            qattn._rmean = saved
        return float(np.max(np.abs(out - p64))) / yardstick

    bad_w = dict(attn_state)
    bad_w["q_proj.weight"] = _flat_split_weight(
        np.ascontiguousarray(attn_state["q_proj.weight"]), heads, d)

    rows = [
        ("clean (committed emitter)",      attn_state, False),
        ("defect 1 only: flat [q|gate]",   bad_w,      False),
        ("defect 2 only: RMSNorm sum",     attn_state, True),
        ("both (the orphan)",              bad_w,      True),
    ]
    sys.stdout.write(f"\n[defect-attribution] T={T} yardstick |pin32-pin64| "
                     f"{yardstick:.3e}\n")
    got = {}
    for label, state, sm in rows:
        r = _ratio(state, sm)
        got[label] = r
        sys.stdout.write(f"[defect-attribution]   {label:<30s} "
                         f"{r:12.1f}x the pin's own f32 rounding\n")

    assert got["clean (committed emitter)"] <= _FLOOR_FACTOR, (
        "the clean emitter is not at the floor; the rest of this cell is moot")
    for label in ("defect 1 only: flat [q|gate]", "defect 2 only: RMSNorm sum"):
        assert got[label] > 1000.0, (
            f"{label} alone moved the result only {got[label]:.1f}x -- either "
            f"the injection is not reaching the emitter or the parity leg is "
            f"passing for a compensating reason")
    orphan_ref = {64: 964999.3, 96: 469606.6}[T]
    both = got["both (the orphan)"]
    assert 0.5 * orphan_ref <= both <= 2.0 * orphan_ref, (
        f"the reconstruction does not land on the archived orphan's measured "
        f"ratio ({both:.1f}x vs {orphan_ref}x) -- the permutation is not a "
        f"faithful stand-in for the orphan emitter and this attribution is void")


# --------------------------------------------------------------------------- #
# 5. graph cost of the piece, reported (no threshold: the window budgets it)
# --------------------------------------------------------------------------- #
@_skip_shards
@pytest.mark.parametrize("device", device_params())
def test_dense_attention_piece_graph_cost(cfg, attn_state, device):
    import time
    T = 64
    model = qattn.build_dense_attention_model(cfg, attn_state, T)
    nodes, const_bytes, counts = pwe.graph_measures(model)
    t0 = time.time()
    compile_for(ov.Core(), model, device)
    compile_s = time.time() - t0
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
    sys.stdout.write(
        f"\n[attn-cost] dev={device} T={T} nodes={nodes} "
        f"const_bytes={const_bytes} ({const_bytes / 1024**3:.3f} GiB) "
        f"compile={compile_s:.2f}s\n"
        f"[attn-cost]   {', '.join(f'{k} {v}' for k, v in top)}\n")
    assert nodes > 0 and const_bytes > 0


# --------------------------------------------------------------------------- #
# 6. QSA EMITTED: the indexer's selection against the pin's, above the boundary
# --------------------------------------------------------------------------- #
def _pin_block_scores(idx, hidden, cs, row, T):
    """The pin's own block scores for one query row (pin 707-755): its
    projections, norms, pooling, rope and relu-sum, for the complete blocks the
    row sees."""
    with torch.no_grad():
        qk = idx.index_qk_proj(hidden)
        q, tk = torch.split(qk, [idx.index_n_heads * idx.index_head_dim, idx.index_head_dim], dim=-1)
        q = q.reshape(1, T, -1, idx.index_head_dim)
        raw = tk.reshape(1, T, -1, idx.index_head_dim).squeeze(2)
        q = idx.q_layernorm(q)
        q = pin_mod.apply_rotary_pos_emb(q, cos=cs[0], sin=cs[1], unsqueeze_dim=2)
        nblk = (row + 1) // idx.compress_ratio
        if nblk == 0:
            return np.zeros(0, np.float32)
        blocks = torch.arange(nblk * idx.compress_ratio).view(nblk, idx.compress_ratio)
        kg = raw[0].index_select(0, blocks.flatten()).view(nblk, idx.compress_ratio, -1).float().mean(1)
        kg = idx.k_layernorm(kg)
        kb = pin_mod.apply_rotary_pos_emb(kg.unsqueeze(1), cos=cs[0][0].index_select(0, blocks[:, 0]),
                                          sin=cs[1][0].index_select(0, blocks[:, 0])).squeeze(1)
        sc = torch.matmul(q[0, row].float(), kb.float().transpose(-1, -2)).transpose(-1, -2)
        return (torch.relu(sc).sum(-1) / math.sqrt(idx.index_head_dim)).numpy()


@_skip_shards
@pytest.mark.parametrize("T", [2052, 2080, 4096])
def test_qsa_attention_piece_selects_what_the_pin_indexer_selects(cfg, attn_state,
                                                                  indexer_state, T):
    """The emitted indexer (`build_qsa_attention_model`, attention.py
    `_qsa_additive_mask`) against the pin's REAL `Qwen4ExpTextQSAIndexer` on the
    same fed blk.3 tensors, above the 2,051-token boundary where the indexer
    prunes (max(0, T - 2051) rows; cell 3). Two gates:

      * SELECTION, equality up to exact ties: the keys each query keeps -- the
        pin's indexer mask added to the causal one (pin 857-858) against the
        emitted mask -- differ only in rows where every differing block scores
        EXACTLY the cut value (relu makes zero scores common; the pin leaves
        the order among equal scores to torch.topk, unspecified; the emitter
        keeps the lower block index). The dense emitter differs on T - 2051
        rows with real score gaps (the red case: its mask is causal).
      * OUTPUT: on the rows without a tie the emitted block sits within the
        yardstick ceiling of the pin's own f32 QSA forward.
    CPU plugin (f32)."""
    hidden = _hidden(cfg, T)
    pid = np.arange(T, dtype=np.int64).reshape(1, T)
    state = dict(attn_state)
    state.update(indexer_state)
    model = qattn.build_qsa_attention_model(cfg, state, T, with_mask=True)
    compiled = compile_for(ov.Core(), model, "CPU")
    res = compiled([hidden.numpy(), pid])
    ov_out = np.asarray(res[0])
    ov_mask = np.asarray(res[1])[0, 0]

    m = _pin_attention(cfg, attn_state, indexer_state, False, torch.float32)
    cs = _cos_sin(cfg, T, torch.float32)
    causal = _causal(T, torch.float32)
    with torch.no_grad():
        pin_sel = m.indexer(hidden, cs, causal, None)
        p32, _ = m(hidden, cs, causal, None)
    p32 = p32.numpy()
    causal_ok = causal.numpy()[0, 0] == 0
    pin_allowed = (pin_sel.numpy()[0, 0] == 0) & causal_ok
    ov_allowed = ov_mask == 0
    pruned = int(np.any(pin_allowed != causal_ok, axis=1).sum())
    differ = np.nonzero(np.any(pin_allowed != ov_allowed, axis=1))[0]
    ratio = int(cfg.indexer_compress_ratio)
    block_topk = int(cfg.indexer_budget) // ratio
    tie_rows, real_rows = [], []
    for r in differ:
        sc = _pin_block_scores(m.indexer, hidden, cs, int(r), T)
        nblk = len(sc)
        # a tie needs differing BLOCKS, all at the cut score, and an identical
        # tail/remainder (tokens ratio*nblk..T-1); anything else is real
        same_tail = np.array_equal(pin_allowed[r, nblk * ratio:], ov_allowed[r, nblk * ratio:])
        if nblk == 0 or not same_tail:
            real_rows.append(int(r))
            continue
        cut = np.sort(sc)[::-1][min(block_topk, nblk) - 1]
        pb = pin_allowed[r, :nblk * ratio].reshape(nblk, ratio).all(1)
        ob = ov_allowed[r, :nblk * ratio].reshape(nblk, ratio).all(1)
        diff_blocks = np.nonzero(pb != ob)[0]
        is_tie = diff_blocks.size > 0 and bool(np.all(sc[diff_blocks] == cut))
        (tie_rows if is_tie else real_rows).append(int(r))
    # the kept block count per row is the pin's min(block_topk, complete blocks)
    # (pin 757) whatever the ties -- a mask that keeps an extra tied block (the
    # dense one, at the first pruned row) is not a tie-order difference
    nb_all = T // ratio
    kept = ov_allowed[:, :nb_all * ratio].reshape(T, nb_all, ratio).all(2)
    nblk_row = (np.arange(T) + 1) // ratio
    kept_ok = kept.sum(1) == np.minimum(block_topk, nblk_row)
    bad_count = np.nonzero(~kept_ok)[0]
    clean = np.ones(T, bool)
    clean[differ] = False
    d_clean = float(np.max(np.abs(ov_out[0, clean] - p32[0, clean])))
    sys.stdout.write(
        f"\n[qsa-emitted] T={T} rows pruned by the pin {pruned} (boundary predicts "
        f"{max(0, T - 2051)})  selection rows differing {len(differ)}/{T}: exact ties "
        f"at the cut {len(tie_rows)}, real {len(real_rows)}  |ov-pinQSA32| on the "
        f"other rows {d_clean:.3e}\n")

    assert pruned == max(0, T - 2051), "the pin's own boundary moved -- stop"
    assert bad_count.size == 0, (
        f"T={T}: {bad_count.size} rows keep a block count other than min({block_topk}, blocks)")
    assert not real_rows, (
        f"T={T}: the emitted selection differs from the pin's beyond exact ties on rows {real_rows[:8]}")
    assert d_clean <= _YARDSTICK_CEILING, (
        f"T={T}: the emitted QSA block is {d_clean:.3e} from the pin's f32 QSA forward")


# --------------------------------------------------------------------------- #
# 7. QSA STATEFUL: chunked prefill and decode across the boundary (qsa step 2)
# --------------------------------------------------------------------------- #
@_skip_shards
def test_qsa_stateful_piece_matches_the_pin_with_its_cache(cfg, attn_state, indexer_state):
    """`build_qsa_stateful_attention_model` (K, V and the indexer's raw keys in
    Variables, dynamic T) against the pin's attention with its REAL indexer and
    its own `DynamicCache` (the indexed layer, cache_utils 321-353), fed the
    same chunks: a 2,048-token prefill, a 40-token chunk, then 12 single-token
    decode steps -- 2,100 tokens, across the 2,051 boundary. Per chunk: the
    selection equal to the pin's up to exact ties at the top-k cut (the rule of
    cell 6), and the output within the yardstick ceiling on every row without
    a tie. CPU plugin (f32)."""
    from transformers.cache_utils import DynamicCache
    chunks = [2048, 40] + [1] * 12
    N_all = sum(chunks)
    hidden_all = _hidden(cfg, N_all)
    state = dict(attn_state)
    state.update(indexer_state)
    model = qattn.build_qsa_stateful_attention_model(cfg, state, N_all, with_mask=True)
    req = compile_for(ov.Core(), model, "CPU").create_infer_request()

    m = _pin_attention(cfg, attn_state, indexer_state, False, torch.float32)
    cache = DynamicCache(config=cfg)
    cos_all, sin_all = _cos_sin(cfg, N_all, torch.float32)
    ratio = int(cfg.indexer_compress_ratio)
    block_topk = int(cfg.indexer_budget) // ratio
    minf = np.float32(np.finfo(np.float32).min)
    past = 0
    ties_total = 0
    for T in chunks:
        N = past + T
        h = hidden_all[:, past:N]
        pid = np.arange(past, N, dtype=np.int64).reshape(1, T)
        res = req.infer({0: h.numpy(), 1: pid})
        ov_out = np.asarray(res[0])
        ov_mask = np.asarray(res[1])[0, 0]
        cm = np.where(np.arange(N)[None, :] <= np.arange(past, N)[:, None], np.float32(0), minf)
        causal = torch.from_numpy(cm.reshape(1, 1, T, N).astype(np.float32))
        cs = (cos_all[:, :N], sin_all[:, :N])
        with torch.no_grad():
            pin_sel = m.indexer(h, cs, causal, cache)
            p32, _ = m(h, cs, causal, cache)
        # the indexer call above appended this chunk's raw keys once; the full
        # forward appends them again -- undo the first append so the cache holds
        # each key once (the forward is the pin's own path, the direct call a probe)
        lay = cache.layers[3]
        lay.indexer_keys = torch.cat([lay.indexer_keys[:, :past], lay.indexer_keys[:, past + T:]], dim=1)
        p32 = p32.numpy()
        causal_ok = cm == 0
        pin_allowed = (pin_sel.numpy()[0, 0] == 0) & causal_ok
        ov_allowed = ov_mask == 0
        differ = np.nonzero(np.any(pin_allowed != ov_allowed, axis=1))[0]
        real = []
        for r in differ:
            a = past + int(r)
            sc = _pin_block_scores(m.indexer, hidden_all[:, :a + 1], (cos_all[:, :a + 1], sin_all[:, :a + 1]), a, a + 1)
            nblk = len(sc)
            same_tail = np.array_equal(pin_allowed[r, nblk * ratio:], ov_allowed[r, nblk * ratio:])
            if nblk == 0 or not same_tail:
                real.append(a)
                continue
            cut = np.sort(sc)[::-1][min(block_topk, nblk) - 1]
            pb = pin_allowed[r, :nblk * ratio].reshape(nblk, ratio).all(1)
            ob = ov_allowed[r, :nblk * ratio].reshape(nblk, ratio).all(1)
            dblk = np.nonzero(pb != ob)[0]
            if not (dblk.size > 0 and np.all(sc[dblk] == cut)):
                real.append(a)
        ties_total += len(differ) - len(real)
        clean = np.ones(T, bool)
        clean[differ] = False
        d_clean = float(np.max(np.abs(ov_out[0, clean] - p32[0, clean]))) if clean.any() else 0.0
        sys.stdout.write(f"\n[qsa-stateful] chunk past={past} T={T}: rows differing {len(differ)} "
                         f"(real {len(real)})  |ov-pin| on the other rows {d_clean:.3e}")
        assert not real, f"chunk past={past} T={T}: selection differs beyond ties at {real[:8]}"
        assert d_clean <= _YARDSTICK_CEILING, f"chunk past={past} T={T}: output {d_clean:.3e} from the pin"
        past = N
    sys.stdout.write(f"\n[qsa-stateful] {N_all} tokens, tie rows {ties_total}\n")


# --------------------------------------------------------------------------- #
# 8. QSA ON THE SERVED PATH: the served emitter's indexer (qsa step 3, T1a)
#
# `q4e.serving_shape.emit_stateful_attention(qsa=True)` calls the SAME
# `q4e.attention._qsa_mask_dynamic` the step-2 graph does, and carries the raw
# keys in a PLAIN state Variable (`cache_params.past.indexer_key.N`, NOT
# gathered by `beam_idx`). The served attention CORE cannot be run before the
# paged pass: its KV Concat joins a [1, kv, past, d] history with a token-major
# [T, kv, 1, d] current key and only `SDPAToPagedAttention` makes those leading
# axes agree (measured: CPU shape inference refuses the pre-pass graph). So the
# served side's own runtime witness here is the INDEXER'S SELECTION, which has
# no KV operand; the served attention core's byte-exactness is step 2's pass
# (T2) and the card window (T6).
# --------------------------------------------------------------------------- #
def _served_indexer_mask_model(cfg, attn_state, indexer_state, layer,
                               rope_span):
    """The served emitter's standalone indexer (`_qsa_indexer_mask_served`)
    as an ov.Model: inputs `hidden_states` [1, T, H] f32 and `position_ids`
    [1, T] i64, result `qsa_mask` [T, 1, 1, N], with the raw-key Assign in
    the model's sinks."""
    from openvino import opset13 as op
    from q4e import serving_shape as ss
    H = cfg.hidden_size
    hidden = op.parameter([1, -1, H], ov.Type.f32)
    hidden.set_friendly_name("hidden_states")
    pid = op.parameter([1, -1], ov.Type.i64)
    pid.set_friendly_name("position_ids")
    cos_np, sin_np = qattn._freqs_tables(cfg, rope_span)
    rope_cos, rope_sin = op.constant(cos_np), op.constant(sin_np)
    state = {}
    state.update({k: np.ascontiguousarray(v, np.float32)
                  for k, v in attn_state.items()})
    state.update({k: np.ascontiguousarray(v, np.float32)
                  for k, v in indexer_state.items()})
    sinks = []
    mask = ss._qsa_indexer_mask_served(hidden, pid, cfg, state, layer,
                                       sinks, rope_cos, rope_sin)
    res = op.result(mask)
    res.set_friendly_name("qsa_mask")
    return ov.Model([res], sinks, [hidden, pid], "qsa_served_indexer")


def _causal_additive(T, N, past):
    """The dense-causal additive mask a served chunk at `past` would carry:
    [1, 1, T, N], 0 where the key is at or before the query's absolute
    position, finfo(f32).min above."""
    rows = np.arange(past, past + T)[:, None]
    cols = np.arange(N)[None, :]
    vis = cols <= rows
    return np.where(vis, np.float32(0.0),
                    np.float32(np.finfo(np.float32).min))[None, None]


@_skip_shards
def test_qsa_served_indexer_selects_what_the_step2_stateful_graph_selects(
        cfg, attn_state, indexer_state):
    """T1a cell 1: the served emitter's indexer against the step-2 stateful
    reference, same chunks ([2048, 40] + [1] x 12), across the 2,051 boundary.
    The selection -- the additive per-query mask -- is BIT-IDENTICAL, because
    both sides call `q4e.attention._qsa_mask_dynamic`; the served side differs
    only in that it is token-major ([T, 1, 1, N]) and its raw keys ride
    `cache_params.past.indexer_key.3`. A served emitter that baked the causal
    mask, dropped the raw-key history, or transposed the wrong axes reds this
    cell. CPU plugin (f32).

    BLIND SPOT, stated rather than hidden: both sides call the same
    `_qsa_mask_dynamic`, so a mutant INSIDE that function is invisible here;
    it is guarded transitively by the step-1 pin cell and
    `test_qsa_stateful_piece_matches_the_pin_with_its_cache`. This cell is a
    wrapper check -- that the served emitter reaches the same algebra over the
    same history -- not an algebra check."""
    chunks = [2048, 40] + [1] * 12
    N_all = sum(chunks)
    hidden_all = _hidden(cfg, N_all)
    state = dict(attn_state)
    state.update(indexer_state)
    ref = qattn.build_qsa_stateful_attention_model(cfg, state, N_all, with_mask=True)
    rr = compile_for(ov.Core(), ref, "CPU").create_infer_request()
    served = _served_indexer_mask_model(cfg, attn_state, indexer_state, 3, N_all)
    sr = compile_for(ov.Core(), served, "CPU").create_infer_request()
    past = 0
    for T in chunks:
        N = past + T
        h = hidden_all[:, past:N]
        pid = np.arange(past, N, dtype=np.int64).reshape(1, T)
        ref_mask = np.asarray(rr.infer({0: h.numpy(), 1: pid})[1])       # [1,1,T,N]
        got = np.asarray(sr.infer({0: h.numpy(), 1: pid})[0])           # [T,1,1,N]
        got = np.transpose(got, (2, 1, 0, 3))                           # [1,1,T,N]
        assert got.shape == ref_mask.shape, (got.shape, ref_mask.shape)
        assert np.array_equal(got, ref_mask), (
            f"chunk past={past} T={T}: the served mask differs from the step-2 "
            f"stateful mask (max|d| {np.max(np.abs(got - ref_mask)):.3e})")
        past = N


def test_qsa_served_mask_is_exactly_causal_below_the_2051_boundary(cfg):
    """T1a cell 3: below 2,051 tokens every row keeps every visible key, so
    the served indexer's mask EQUALS the dense causal mask exactly -- which is
    why the served answer must not move there (DESIGN 3.4). At T=2052 the
    first pruned row (i=2051) appears and the masks differ. The boundary is
    derived, not tuned: block_topk * ratio + ratio - 1 = 2048 + 4 - 1."""
    from openvino import opset13 as op
    from q4e import serving_shape as ss
    cfg_dense = cfg
    # a shared, non-degenerate hidden is not needed: equality is structural
    g = np.random.default_rng(20260928)
    ratio = int(cfg_dense.indexer_compress_ratio)
    block_topk = int(cfg_dense.indexer_budget) // ratio
    boundary = block_topk * ratio + ratio - 1
    for T in (boundary, boundary + 1):
        hidden = (g.standard_normal((1, T, cfg_dense.hidden_size)) * 0.02).astype(np.float32)
        pid = np.arange(T, dtype=np.int64).reshape(1, T)
        H = cfg_dense.hidden_size
        hp = op.parameter([1, -1, H], ov.Type.f32)
        pp = op.parameter([1, -1], ov.Type.i64)
        cos_np, sin_np = qattn._freqs_tables(cfg_dense, T)
        rc, rs = op.constant(cos_np), op.constant(sin_np)
        # zero indexer weights are enough: the SELECTION's visibility is what
        # is asserted, and at/below the boundary it is every visible block
        st = {"indexer.index_qk_proj.weight": np.zeros((5 * 128, H), np.float32),
              "indexer.q_layernorm.weight": np.zeros((128,), np.float32),
              "indexer.k_layernorm.weight": np.zeros((128,), np.float32)}
        sinks = []
        mask = ss._qsa_indexer_mask_served(hp, pp, cfg_dense, st, 3, sinks, rc, rs)
        r = op.result(mask)
        m = ov.Model([r], sinks, [hp, pp], "boundary")
        got = np.asarray(compile_for(ov.Core(), m, "CPU")([hidden, pid])[0])
        got = np.transpose(got, (2, 1, 0, 3))
        causal = _causal_additive(T, T, 0)
        if T == boundary:
            assert np.array_equal(got, causal), (
                f"T={T} (the boundary): the served mask is not the causal mask")
        else:
            assert not np.array_equal(got, causal), (
                f"T={T}: the first pruned row did not appear -- boundary moved")


def test_qsa_route_gate_boundary_is_on_the_marked_mask(cfg):
    """T3b cell: the exporter also writes the route gate's boundary on the
    same marked mask -- block_topk * ratio + ratio - 1, 2051 for this config.
    The pass copies it to the PagedAttention node and the GPU impl keeps
    today's route (micro included) at or below it, reading the mask only
    above it. Red first: without the rt_info the pass leaves the PA node at
    boundary 0, which the impl treats as "above" and the byte-identity below
    the boundary fails (T3's control cell measured the floor there)."""
    from openvino import opset13 as op
    from q4e import serving_shape as ss
    ratio = int(cfg.indexer_compress_ratio)
    block_topk = int(cfg.indexer_budget) // ratio
    expected = block_topk * ratio + ratio - 1
    H = cfg.hidden_size
    hp = op.parameter([1, -1, H], ov.Type.f32)
    pp = op.parameter([1, -1], ov.Type.i64)
    cos_np, sin_np = qattn._freqs_tables(cfg, 2048)
    rc, rs = op.constant(cos_np), op.constant(sin_np)
    st = {"indexer.index_qk_proj.weight": np.zeros((5 * 128, H), np.float32),
          "indexer.q_layernorm.weight": np.zeros((128,), np.float32),
          "indexer.k_layernorm.weight": np.zeros((128,), np.float32)}
    sinks = []
    mask = ss._qsa_indexer_mask_served(hp, pp, cfg, st, 3, sinks, rc, rs)
    ri = mask.get_rt_info()
    assert "arcint" in ri and ri["arcint"].astype(str) == "qsa_selection", ri
    assert "qsa_boundary" in ri, ri
    assert ri["qsa_boundary"].astype(int) == expected == 2051, ri


def test_qsa_off_leaves_the_serving_shape_graph_unchanged():
    """T1a cell 2: the qsa flag is additive. With `qsa=False` (the default)
    the serving-shape backbone carries NO indexer Variable and NO
    `qsa_selection` rt_info and reports the same node count as the default
    build -- so every existing artifact and the arch hash are untouched. With
    `qsa=True` the indexer Variable appears per full-attention layer, the
    marker is set, and the graph grows. The DEFAULT build's byte-shape guard
    is the whole `test_serving_shape.py` contract suite (its structural counts
    and the saved-graph checks), which this change leaves green; the check
    that the marker reaches the ATTENTION CORE is
    `test_qsa_served_attention_consumes_the_marked_mask`."""
    from q4e import serving_shape as ss
    def build(qsa):
        arena = ss.SparseArena()
        try:
            return ss.build_serving_shape_ir(arena=arena, n_layers=4, qsa=qsa)
        finally:
            arena.close()
    off, r_off = build(False)
    default, r_def = build(False)
    on, r_on = build(True)
    assert r_off["nodes"] == r_def["nodes"]
    assert r_off["graph_const_bytes"] == r_def["graph_const_bytes"]
    assert r_off["inputs"] == r_def["inputs"]
    assert r_off["qsa"] is False and r_on["qsa"] is True

    def var_ids(model):
        return sorted(v.get_info().variable_id for v in model.get_variables())
    off_ids, on_ids = var_ids(off), var_ids(on)
    assert not any("indexer_key" in i for i in off_ids), off_ids
    assert sum("indexer_key" in i for i in on_ids) == r_on["attn_layers"] == 1, on_ids
    assert "cache_params.past.indexer_key.3" in on_ids

    def markers(model):
        n = 0
        for node in model.get_ops():
            ri = node.get_rt_info()
            if "arcint" in ri and ri["arcint"].astype(str) == "qsa_selection":
                n += 1
        return n
    assert markers(off) == 0 and markers(on) == 1
    assert r_on["nodes"] > r_off["nodes"]


def test_qsa_served_attention_consumes_the_marked_mask():
    """The served emitter must FEED the indexer's mask to the SDPA, not merely
    build it: without this cell a mutant that emits the marker but hands the
    SDPA the dense causal mask would pass every other T1a cell. Each
    `ScaledDotProductAttention` in a qsa=True backbone has the marked mask
    (`attn3/qsa_mask`, rt_info `arcint=qsa_selection`) as input 3 -- its
    `attention_mask` operand; with qsa=False no SDPA carries a marked mask.
    Depth 4 has exactly one attention layer."""
    from q4e import serving_shape as ss
    def build(qsa):
        arena = ss.SparseArena()
        try:
            return ss.build_serving_shape_ir(arena=arena, n_layers=4, qsa=qsa)[0]
        finally:
            arena.close()
    def marked_spda(model):
        n = 0
        for node in model.get_ops():
            if node.get_type_name() != "ScaledDotProductAttention":
                continue
            ri = node.input_value(3).get_node().get_rt_info()
            if "arcint" in ri and ri["arcint"].astype(str) == "qsa_selection":
                n += 1
        return n
    on, off = build(True), build(False)
    assert marked_spda(on) == 1, "the QSA mask is built but not consumed by the SDPA"
    assert marked_spda(off) == 0


def test_qsa_indexer_state_survives_the_paged_attention_pass():
    """T1b (option A): the pass `SDPAToPagedAttention` removes only the KV
    Assigns it matched (`var_ids_to_remove`); the indexer's ReadValue and its
    Assign must survive it, and the plain Variable must not be gathered by
    `beam_idx` (whose Parameter the pass deletes -- a gathered Variable would
    lose its input and the transformed graph would not be well-formed). The
    KV pair is consumed into the paged cache ports and `beam_idx` disappears;
    the indexer state stays. A mutant that gathers the indexer Variable by
    `beam_idx` fails this cell (and the pass)."""
    from openvino._offline_transformations import paged_attention_transformation
    from q4e import serving_shape as ss
    arena = ss.SparseArena()
    try:
        model, _ = ss.build_serving_shape_ir(arena=arena, n_layers=4, qsa=True)
        paged_attention_transformation(model)
        ops = model.get_ops()
        var = "cache_params.past.indexer_key.3"
        assert any(n.get_type_name() == "ReadValue" and n.get_variable_id() == var
                   for n in ops), "the indexer ReadValue did not survive the pass"
        assert any(n.get_type_name() == "Assign" and n.get_variable_id() == var
                   for n in ops), "the indexer Assign did not survive the pass"
        # the KV state is consumed into the paged caches, and beam_idx is gone
        live = {n.get_variable_id() for n in ops if n.get_type_name() == "ReadValue"}
        assert "cache_params.past.key.3" not in live
        assert "cache_params.past.value.3" not in live
        names = {p.get_node().get_friendly_name() for p in model.inputs}
        assert "beam_idx" not in names
        assert any(n.startswith("key_cache") for n in names)
    finally:
        arena.close()
