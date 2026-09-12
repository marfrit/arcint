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
import glob
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
def indexer_state():
    """The indexer's OWN weights, read through a plain GGUFReader -- they are
    deliberately NOT in the q4e name map (the selection branch is not emitted),
    and they are needed only to drive the pin-side QSA oracle. The fused
    `index_qk_proj` is the checkpoint's q_proj rows followed by its k_proj rows,
    which is what the pin's own split (pin 707-711) takes apart again."""
    from gguf import GGUFReader
    from gguf import quants as gq
    raw = {}
    for p in sorted(glob.glob(os.path.join(_SHARDS, "*.gguf"))):
        for t in GGUFReader(p).tensors:
            if t.name.startswith("blk.3.indexer."):
                raw[t.name] = t
    missing = {"blk.3.indexer.q_proj.weight", "blk.3.indexer.k_proj.weight",
               "blk.3.indexer.q_norm.weight", "blk.3.indexer.k_norm.weight"
               } - set(raw)
    assert not missing, f"indexer tensors absent from the shards: {sorted(missing)}"

    def deq(name):
        t = raw[name]
        return np.ascontiguousarray(gq.dequantize(t.data, t.tensor_type),
                                    dtype=np.float32)

    fused = np.concatenate([deq("blk.3.indexer.q_proj.weight"),
                            deq("blk.3.indexer.k_proj.weight")], axis=0)
    return {"indexer.index_qk_proj.weight": fused,
            "indexer.q_layernorm.weight": deq("blk.3.indexer.q_norm.weight"),
            "indexer.k_layernorm.weight": deq("blk.3.indexer.k_norm.weight")}


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
