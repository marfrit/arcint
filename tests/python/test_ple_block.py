"""E2 increment 5a -- the qwen4_exp PLELayer (n-gram PLE block), parity for
`q4e.ple.build_ple_model`.

RED state: `from q4e.ple import build_ple_model` -- the `q4e.ple` module does
not exist yet (GREEN emits tools/q4e/ple.py + adds `from . import ple` to
q4e/__init__.py). Collecting this file fails with ModuleNotFoundError; the
other test files still collect and pass.

THE N-GRAM ROW INDEX (bit-exact, integer). The row-index hash is validated in
python against the committed Link-3 vectors (tests/ngram_row_ids_vectors.h) --
the `test_ngram_row_ids_match_link3_vectors` leg -- BEFORE and independently of
any graph, using the pin's OWN _build_layer_multipliers/_find_nth_prime_after
to derive the constants. The index CANNOT be emitted as opset-13 integer ops on
the installed OpenVINO build (its CPU i64 Multiply/Add are 32-bit -- measured;
see tools/q4e/ple.py's header and the RECONCILE session block), so it is
produced by the validated derivation and FED to the graph as the int64 input
`ngram_row_ids`. `test_ids_reproduce_pin_gather` proves the fed ids reproduce
the pin NGramEmbedding's own gather EXACTLY.

Config fixture: faithful geometry, shrunk for CPU: ngram_size 3, heads_per_ngram
2 (num_ngram_heads = 4), ngram_vocab_size_base 17 (real 20M shrunk so the table
is tiny), ple_embed_dim 32 (4 heads x 8), hidden 16, hc_count 4, conv kernel 4 /
dilation 3 (= ngram_size), eos 0.

Legs:
  * ngram-row-ids byte-match: python derivation == the committed Link-3 vectors.
  * ids-reproduce-pin: fed ids gather == the pin NGramEmbedding's embeddings (0.0).
  * transcription-vs-pin: ref_ple (pin leaves) vs the pin PLELayer -- 0.0.
  * ov-parity (unmasked): the emitted graph vs ref_ple -- max-abs < 1e-5 + KLD.
  * ov-parity (masked): conv_mask trailing zeros, both sides -- max-abs < 1e-5
    (the conv mixes positions, so no exact-zero claim; this validates the
    apply_mask wiring, pin 1251-1253).

Oracle discipline: the installed pinned reference, re-hashed before every table.

Run (dev-host venv):
    Q4E_GPU=  ~/openarc-venv/bin/python3 -m pytest tests/python/test_ple_block.py -s --continue-on-collection-errors
"""
import hashlib
import math
import os
import re
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

from q4e import ref_ple  # noqa: E402
from q4e.ple import build_ple_model  # noqa: E402  (RED: module absent)


PIN_SHA256 = {
    "modeling_qwen4_exp.py": "ca9f00bbd73cfcfbad7ba6073b5ecc23ca27bdace58b746fe6ead6ab175efc0c",
    "configuration_qwen4_exp.py": "b78132d8cd935437208ee281fa4569b771a63fcb58ebffe84f3e62f5b86235ca",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_pin() -> None:
    import transformers.models.qwen4_exp as pkg

    pkg_dir = Path(pkg.__file__).resolve().parent
    for name, want in PIN_SHA256.items():
        got = _sha256(pkg_dir / name)
        assert got == want, f"oracle drift: {name} sha256 {got} != pin {want}"


def _make_config():
    return pin_cfg.Qwen4ExpTextConfig(
        hidden_size=16,
        num_hidden_layers=1,
        hc_count=4,
        rms_norm_eps=1e-6,
        ngram_size=3,
        heads_per_ngram=2,           # num_ngram_heads = 4
        ngram_vocab_size_base=17,    # real 20M shrunk so the table is tiny
        make_ngram_vocab_size_divisible_by=128,
        ple_embed_dim=32,            # 4 heads x 8
        ple_conv_kernel_size=4,
        seed=1234,
        vocab_size=257,
        eos_token_id=0,
        layer_types=["linear_attention"],
    )


def _ref_and_pin(config, seed: int = 0, ple_layer_index: int = 1):
    """ref_ple.Qwen4ExpTextPLELayer (forward transcribed, pin leaves) and the
    pin's own PLELayer, identical seeded random weights, eval."""
    torch.manual_seed(seed)
    ref = ref_ple.Qwen4ExpTextPLELayer(config, layer_idx=0, ple_layer_index=ple_layer_index).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0.0, 0.05)
    pin = pin_mod.Qwen4ExpTextPLELayer(config, layer_idx=0, ple_layer_index=ple_layer_index).eval()
    pin.load_state_dict(ref.state_dict())
    return ref, pin


def _state_np(module) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}


def _device_params():
    devs = ["CPU"]
    extra = os.environ.get("Q4E_GPU", "").strip()
    if extra:
        devs += [d for d in (s.strip() for s in extra.split(",")) if d]
    return devs


def _kld(p_logits, q_logits):
    def _softmax(x):
        x = x.astype(np.float64)
        x = x - x.max(-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(-1, keepdims=True)

    p, q = _softmax(p_logits), _softmax(q_logits)
    return float(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=-1)))


# --- the index derivation, transcribed (matches src/exec/ngram_row_ids.h) ----
_MASK64 = (1 << 64) - 1


def _s64(x):
    x &= _MASK64
    return x - (1 << 64) if x >= (1 << 63) else x


def _derive(vocab_size, ngram_size, heads_per_ngram, base, ple_idx, seed=1234):
    nh = (ngram_size - 1) * heads_per_ngram
    mult = pin_mod._build_layer_multipliers(vocab_size, ngram_size, ple_idx, seed).tolist()
    sizes, offs, tot = [], [], 0
    for h in range(nh):
        gh = ple_idx * nh + h
        sz = pin_mod._find_nth_prime_after(base - 1, gh + 1)
        sizes.append(sz)
        offs.append(tot)
        tot += sz
    return mult, sizes, offs


def _row_ids(mult, sizes, offs, ng, hpn, eos, context, tokens):
    ctx = ng - 1
    packed = list(context) + list(tokens)
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
    per = [[r[ctx + i] for i in range(len(tokens))] for r in shifted]
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
    return out


# ---------------------------------------------------------------------------
def test_ngram_row_ids_match_link3_vectors():
    """The python index derivation (pin-derived constants + the documented XOR
    mixing) reproduces the committed Link-3 vectors byte-for-byte -- the
    bit-exact index gate, BEFORE any graph. Params per gen_ngram_vectors.py."""
    _assert_pin()
    hdr = (REPO_ROOT / "tests" / "ngram_row_ids_vectors.h").read_text()
    fx = {  # name -> (vocab, ngram, heads_per_ngram, base, ple_idx) per gen_ngram_vectors.py FIXTURES
        "basic_2gram_only": (257, 2, 3, 17, 1),
        "trigram_no_eos": (257, 3, 2, 17, 1),
        "trigram_eos_midseq": (257, 3, 2, 17, 2),
        "trigram_fresh_context": (257, 3, 2, 17, 0),
        "decode_single_token": (257, 3, 4, 17, 1),
        "qwen_scale_16head": (151936, 3, 8, 8209, 3),
    }

    def nums(s):
        return [int(x) for x in re.findall(r"-?\d+", s)]

    chunks = hdr.split("RowIdsVector{")[1:]
    assert chunks, "no RowIdsVector fixtures parsed from the committed header"
    print("\n[ngram-row-ids-vs-link3] pin sha OK")
    for ch in chunks:
        body = ch.split("\n        },")[0]
        name = re.search(r'"([^"]+)"', body).group(1)
        nn = body.replace('"' + name + '"', "")
        scal = nums(nn[: nn.index("{")])
        br = re.findall(r"\{([^}]*)\}", nn)
        ng, hpn, eos = scal[0], scal[1], scal[2]
        mult, sizes, offs = nums(br[0]), nums(br[1]), nums(br[2])
        ctx, tok, rows = nums(br[3]), nums(br[4]), nums(br[5])
        got = [v for r in _row_ids(mult, sizes, offs, ng, hpn, eos, ctx, tok) for v in r]
        assert got == rows, f"row_ids mismatch on fixture {name}"
        # constants independently pin-derived match the committed ones
        vocab, ng2, hpn2, base, idx = fx[name]
        dmult, dsizes, doffs = _derive(vocab, ng, hpn, base, idx)
        assert (dmult, dsizes, doffs) == (mult, sizes, offs), f"constant drift on {name}"
        print(f"  {name:24s} {len(rows):3d} ids OK; pin-derived consts OK")


@pytest.mark.parametrize("with_eos", [False, True])
def test_ids_reproduce_pin_gather(with_eos):
    """The fed ids (ref_ple.row_ids_from_input) reproduce the pin
    NGramEmbedding's own gather EXACTLY -- so feeding them to the OV graph is
    the same table lookup the pin does (the index is not emitted in-graph;
    OV i64 arithmetic is 32-bit -- see tools/q4e/ple.py)."""
    _assert_pin()
    config = _make_config()
    ref, _ = _ref_and_pin(config)
    T = 64
    torch.manual_seed(7)
    ids = torch.randint(1, config.vocab_size, (1, T))
    if with_eos:
        ids[0, 20] = 0
        ids[0, 41] = 0
    emb_pin = ref.ple_embedding(ids, None)
    my = ref_ple.row_ids_from_input(ref.ple_embedding, ids[0])
    emb_mine = ref.ple_embedding.ngram_embedding.weight[my].flatten(-2).unsqueeze(0)
    md = float((emb_pin - emb_mine).abs().max())
    print(f"\n[ple-ids-reproduce-pin] with_eos={with_eos}  max-abs(pin-gather vs fed)={md:.3e}")
    assert md == 0.0, f"fed ids do not reproduce the pin's gather: {md:.3e}"


def test_transcription_matches_pin():
    """ref_ple (pin leaves) reproduces the pin PLELayer forward EXACTLY."""
    _assert_pin()
    config = _make_config()
    ref, pin = _ref_and_pin(config)
    print("\n[ple-transcription-vs-pin] pin sha OK; max-abs by T")
    for T in (64, 96):
        torch.manual_seed(100 + T)
        ids = torch.randint(1, config.vocab_size, (1, T))
        hs = torch.randn(1, T, config.hc_count * config.hidden_size)
        with torch.no_grad():
            yr, yp = ref(hs, ids), pin(hs, ids)
        md = float((yr - yp).abs().max())
        print(f"  T={T:>3}  max-abs(ref - pin) = {md:.3e}")
        assert md == 0.0, f"transcription drifted from pin at T={T}: {md:.3e}"


@pytest.mark.parametrize("T", [64, 96])
@pytest.mark.parametrize("device", _device_params())
def test_ple_ov_parity(device, T):
    """The emitted graph (fed the validated ids) equals ref_ple at atol=1e-5 +
    a KLD on a small logits head."""
    _assert_pin()
    config = _make_config()
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)
    C = config.hc_count * config.hidden_size

    torch.manual_seed(200 + T)
    ids = torch.randint(1, config.vocab_size, (1, T))
    ids[0, 30] = 0  # an eos to exercise the index boundary
    hs = torch.randn(1, T, C)
    with torch.no_grad():
        y_ref = ref(hs, ids).float().numpy()
    my = ref_ple.row_ids_from_input(ref.ple_embedding, ids[0]).numpy()[None]  # [1,T,Hn]

    model = build_ple_model(config, state, seq_len=T)
    compiled = ov.Core().compile_model(model, device)
    out = compiled({"hidden_states": hs.float().numpy(),
                    "ngram_row_ids": my.astype(np.int64)})
    y_ov = out[compiled.outputs[0]]
    max_abs = float(np.max(np.abs(y_ref - y_ov)))
    rng = np.random.default_rng(0)
    head = rng.standard_normal((C, 128)).astype(np.float32)
    kld = _kld(y_ref.reshape(-1, C) @ head, y_ov.reshape(-1, C) @ head)
    print(f"\n[ple-ov-parity] device={device:<6} T={T:>3}  max-abs={max_abs:.3e}  KLD={kld:.3e}")
    assert max_abs < 1e-5, f"OV PLE parity failed device={device} T={T}: {max_abs:.3e}"


@pytest.mark.parametrize("T", [64, 96])
@pytest.mark.parametrize("device", _device_params())
def test_ple_ov_parity_masked(device, T):
    """conv_mask trailing zeros (pin 1251-1253): both ref and OV zero the gated
    streams before the conv. The conv mixes positions, so no exact-zero claim --
    this validates the apply_mask wiring by full-sequence parity."""
    _assert_pin()
    config = _make_config()
    ref, _ = _ref_and_pin(config)
    state = _state_np(ref)
    C = config.hc_count * config.hidden_size
    live = T // 2

    torch.manual_seed(300 + T)
    ids = torch.randint(1, config.vocab_size, (1, T))
    hs = torch.randn(1, T, C)
    mask = torch.ones(1, T)
    mask[:, live:] = 0.0
    with torch.no_grad():
        y_ref = ref(hs, ids, conv_mask=mask).float().numpy()
    my = ref_ple.row_ids_from_input(ref.ple_embedding, ids[0]).numpy()[None]

    model = build_ple_model(config, state, seq_len=T, with_mask=True)
    compiled = ov.Core().compile_model(model, device)
    out = compiled({"hidden_states": hs.float().numpy(),
                    "ngram_row_ids": my.astype(np.int64),
                    "conv_mask": mask.float().numpy()})
    y_ov = out[compiled.outputs[0]]
    max_abs = float(np.max(np.abs(y_ref - y_ov)))
    print(f"\n[ple-ov-parity-masked] device={device:<6} T={T:>3} live={live}  max-abs={max_abs:.3e}")
    assert max_abs < 1e-5, f"OV PLE masked parity failed device={device} T={T}: {max_abs:.3e}"
