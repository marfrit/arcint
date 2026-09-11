"""Feed real GGUF weights into the q4e pin/transcription state-dict keys.

Phase B of the qwen4_exp export path: the parity work of E2 (inc1-5b) validated
GRAPH SEMANTICS on random weights. This module supplies the other half of the
provenance story -- it reads the shipped GGUF checkpoint, dequantises each
tensor to f32, and maps GGUF tensor names onto the pin `Qwen4ExpTextModel`
state-dict keys (the same keys `q4e.ref_backbone` / `q4e.backbone` consume).

WHAT THIS GATE IS, AND IS NOT (the kickoff insight, kept separate on purpose):
parity against the pin is about graph semantics, not weight provenance. The pin
class loaded with DEQUANTISED GGUF tensors keeps the transcription parity leg at
exactly 0.0, because both sides then carry byte-identical weights. So a cell that
feeds a real tensor and still reads 0.0 is validating the NAME MAP + the shape /
reshape / fusion, NOT quantisation error. The quantisation-vs-reference question
(is `ssm_a` really A_log? does the block quant lose accuracy?) is a DIFFERENT gate
-- KLD, the Mixpert protocol, a later era -- and is deliberately not blended in
here. A non-0.0 in a feed cell is a NAME-MAP or a RESHAPE bug, never a numerics
story.

Name map (authoritative source: llama.cpp gguf-py `TensorNameMap` for the model's
internal arch QWEN35 / QWEN35MOE -- the "qwen3.5" codename the converter uses;
`general.architecture` in the file is "qwen4exp", an Unsloth-fork addition the
mainline gguf-py does not carry, so the map is transcribed below rather than
resolved from the file's arch string). Resolved and cross-checked against the
shipped UD-Q3_K_XL tensor list; the qwen4exp-only families (hyper-connection
mixers, PLE) are not in QWEN35MOE and are mapped by name/shape below.

THE HEAD IS NOT TIED IN THIS CHECKPOINT (measured 2026-09-11; reviewer finding B
of REVIEW 2cd2b2f, reproduced independently). The pin declares the head tied to
the embedding (pin 1593 `_tied_weights_keys`); the shipped UD-Q3_K_XL ships
`output.weight` (Q6_K) AND `token_embd.weight` (Q8_0) as independent tensors --
row-band mean cosine -0.0016 / +0.0028 / +0.0263 / +0.0150, where tied would be
1.0 (full table in `q4e.ref_backbone`'s header). `lm_head.weight` is therefore a
mapped global key here, and `has_lm_head()` reports whether a source declares
one; the pin's tie is the fallback for sources that do not.

GGUF 2D tensors come back from gguf-py already in pytorch [out, in] / [vocab,
hidden] order (its `ReaderTensor.data` is shaped [ne1, ...], ne0 the inner
quant dim), so 2D linears/embeddings need NO transpose. Reshapes that DO apply
are noted per entry (`conv`: [C,K]->[C,1,K]; `fuse_gate_up`: concat gate|up
experts along the ff axis; `row`: 1D [H]->[1,H]).

gguf-py is llama.cpp's `gguf-py` package (or `pip install gguf`); it must be
importable as `gguf` with a real `GGUFReader`. A namespace-package shadow (e.g. a
`gguf` symlink to a model directory on sys.path) has no `GGUFReader` and is
rejected with a named error rather than a silent AttributeError later.
"""
import glob
import os
import re

import numpy as np

try:
    import gguf  # noqa: F401
    from gguf import GGUFReader
    from gguf import quants as _gguf_quants
    _HAVE_GGUF = hasattr(gguf, "GGUFReader")
except Exception:  # pragma: no cover - import-environment dependent
    _HAVE_GGUF = False


def _require_gguf():
    if not _HAVE_GGUF:
        raise ImportError(
            "gguf-py is not importable as a real package (need `gguf.GGUFReader`). "
            "Install llama.cpp's gguf-py (put its directory on PYTHONPATH) or "
            "`pip install gguf`. A `gguf` namespace-package shadow (a symlink to a "
            "model directory on sys.path) resolves without GGUFReader and is not it."
        )


# The separate LM head, when the source declares one (see _GLOBAL_MAP).
_LM_HEAD = "output.weight"

# --- pin-key-suffix -> (gguf name / names, reshape kind) ---------------------
# Per-decoder-layer keys, relative to "layers.{i}." ; {i} -> blk.{i} in GGUF.
_LAYER_MAP = {
    # GatedDeltaNet (linear_attn). in_proj_qkv/z land on the GGUF attn_qkv/attn_gate
    # names (QWEN35 convention -- these are the GDN input projections, NOT a QSA
    # full-attention layer); in_proj_a/b on ssm_alpha/ssm_beta.
    # NB: GGUF per-block tensor names keep their own trailing suffix -- most are
    # "<name>.weight", but ssm_a and ssm_dt.bias are not; the strings below are
    # the EXACT file names (verified against the shipped tensor list).
    "linear_attn.in_proj_qkv.weight": ("attn_qkv.weight", "direct2d"),
    "linear_attn.in_proj_z.weight": ("attn_gate.weight", "direct2d"),
    "linear_attn.in_proj_a.weight": ("ssm_alpha.weight", "direct2d"),
    "linear_attn.in_proj_b.weight": ("ssm_beta.weight", "direct2d"),
    "linear_attn.A_log": ("ssm_a", "vec"),
    "linear_attn.dt_bias": ("ssm_dt.bias", "vec"),
    "linear_attn.conv1d.weight": ("ssm_conv1d.weight", "conv"),
    "linear_attn.norm.weight": ("ssm_norm.weight", "vec"),
    "linear_attn.out_proj.weight": ("ssm_out.weight", "direct2d"),
    # SparseMoeBlock. The pin fuses gate+up into one [E, 2*ff, in] tensor; GGUF
    # ships them separately as [E, ff, in] each -> concat along the ff axis.
    "mlp.gate.weight": ("ffn_gate_inp.weight", "direct2d"),
    "mlp.experts.gate_up_proj": (("ffn_gate_exps.weight", "ffn_up_exps.weight"), "fuse_gate_up"),
    "mlp.experts.down_proj": ("ffn_down_exps.weight", "expert3d"),
    "mlp.shared_expert.gate_proj.weight": ("ffn_gate_shexp.weight", "direct2d"),
    "mlp.shared_expert.up_proj.weight": ("ffn_up_shexp.weight", "direct2d"),
    "mlp.shared_expert.down_proj.weight": ("ffn_down_shexp.weight", "direct2d"),
    "mlp.shared_expert_gate.weight": ("ffn_gate_inp_shexp.weight", "row"),
    # GatedResidual hyper-connection mixers (qwen4exp-only; mapped by name/shape).
    "attn_hyper_connection.hc_norm.weight": ("hc_attn_norm.weight", "vec"),
    "attn_hyper_connection.input_mix_weight_down.weight": ("hc_attn_down.weight", "direct2d"),
    "attn_hyper_connection.input_mix_weight_up.weight": ("hc_attn_up.weight", "direct2d"),
    "attn_hyper_connection.block_inject_weight.weight": ("hc_attn_inject.weight", "direct2d"),
    "mlp_hyper_connection.hc_norm.weight": ("hc_ffn_norm.weight", "vec"),
    "mlp_hyper_connection.input_mix_weight_down.weight": ("hc_ffn_down.weight", "direct2d"),
    "mlp_hyper_connection.input_mix_weight_up.weight": ("hc_ffn_up.weight", "direct2d"),
    "mlp_hyper_connection.block_inject_weight.weight": ("hc_ffn_inject.weight", "direct2d"),
    # PLE (qwen4exp-only). The n-gram embedding table is the GLOBAL
    # per_layer_token_embd (handled in _resolve, not per-block). The derived
    # index buffers (layer_multipliers / vocab_sizes / offsets) are NOT weights --
    # the pin recomputes them from config, so they are not fed.
    "ple.key_proj.weight": ("ple_key.weight", "direct2d"),
    "ple.value_proj.weight": ("ple_value.weight", "direct2d"),
    "ple.norm_key.weight": ("ple_norm_key.weight", "vec"),
    "ple.norm_query.weight": ("ple_norm_query.weight", "vec"),
    "ple.norm_conv.weight": ("ple_norm_conv.weight", "vec"),
    "ple.conv1d.weight": ("ple_conv1d.weight", "conv"),
}

# Global (non-per-layer) keys.
_GLOBAL_MAP = {
    "embed_tokens.weight": ("token_embd.weight", "direct2d"),
    "hyper_connection_mixer.hc_norm.weight": ("output_hc_norm.weight", "vec"),
    "hyper_connection_mixer.input_mix_weight_down.weight": ("output_hc_down.weight", "direct2d"),
    "hyper_connection_mixer.input_mix_weight_up.weight": ("output_hc_up.weight", "direct2d"),
    # The LM head. PIN-vs-CHECKPOINT DIVERGENCE (measured; module header): the
    # pin ties lm_head to embed_tokens, the shipped UD-Q3_K_XL does NOT -- it
    # carries an independent output.weight (Q6_K) next to token_embd (Q8_0).
    # Mapped here so the served head reaches the emitter; `has_lm_head()` says
    # whether a given source declares one at all.
    "lm_head.weight": (_LM_HEAD, "direct2d"),
}

# The pin ple_embedding.ngram_embedding.weight is a slice of this global table.
_PLE_TABLE = "per_layer_token_embd.weight"
# Non-persistent / derived index buffers: not weights, the pin recomputes them.
_DERIVED_SUFFIXES = (
    "ple.ple_embedding.layer_multipliers",
    "ple.ple_embedding.ngram_heads_vocab_sizes",
    "ple.ple_embedding.ngram_heads_offsets",
)


def _dequant(reader_tensor, rows=None):
    """Dequantise one GGUF tensor to f32 in pytorch axis order.

    `rows`, if given, slices the LEADING axis of the raw block data before
    dequant (the leading axis is the out / vocab / expert dimension, whose rows
    are independent quant blocks) -- so a cell can pull the first N rows of a
    248320-row embedding without materialising the whole 2.5 GiB f32 tensor.
    """
    qt = reader_tensor.tensor_type
    data = reader_tensor.data
    if rows is not None:
        data = data[:rows]
    name = qt.name
    if name in ("F32", "F16", "BF16"):
        arr = np.asarray(data)
        if name != "F32":
            arr = arr.astype(np.float32)
        return np.ascontiguousarray(arr, dtype=np.float32)
    return _gguf_quants.dequantize(data, qt).astype(np.float32)


class GgufFeed:
    """Index the GGUF shards once, then serve dequantised pin-keyed tensors."""

    def __init__(self, shards):
        """`shards`: a directory, a glob, or a list of shard paths. No default
        location is baked in (this repo is public; host paths live elsewhere)."""
        _require_gguf()
        if isinstance(shards, (list, tuple)):
            paths = list(shards)
        elif os.path.isdir(shards):
            paths = sorted(glob.glob(os.path.join(shards, "*.gguf")))
        else:
            paths = sorted(glob.glob(shards))
        if not paths:
            raise FileNotFoundError(f"no GGUF shards found for {shards!r}")
        self.paths = paths
        self._readers = [GGUFReader(p) for p in paths]
        self._index = {}
        for r in self._readers:
            for t in r.tensors:
                self._index[t.name] = t
        self.arch = None
        for f in self._readers[0].fields.values():
            if f.name == "general.architecture":
                try:
                    self.arch = f.contents()
                except Exception:
                    self.arch = None

    # -- low level -----------------------------------------------------------
    def has(self, gguf_name):
        return gguf_name in self._index

    def gguf_type(self, gguf_name):
        return self._index[gguf_name].tensor_type.name

    def has_lm_head(self):
        """Does this source ship a SEPARATE lm_head (GGUF `output.weight`)?

        The pin ties the head to the embedding (pin 1593); the shipped
        UD-Q3_K_XL does not (measured -- see the module header of
        `q4e.ref_backbone`). A caller building a state dict asks this to decide
        between feeding `lm_head.weight` and leaving the tie in place; there is
        no silent default, because guessing wrong ships the wrong head."""
        return self.has(_LM_HEAD)

    def dequant(self, gguf_name, rows=None):
        if gguf_name not in self._index:
            raise KeyError(f"GGUF tensor {gguf_name!r} not in any shard")
        return _dequant(self._index[gguf_name], rows=rows)

    # -- name-mapped ---------------------------------------------------------
    def _split_key(self, pin_key):
        m = re.match(r"layers\.(\d+)\.(.+)$", pin_key)
        if m:
            return int(m.group(1)), m.group(2)
        return None, pin_key

    def pin_tensor(self, pin_key, rows=None, gguf_layer=None):
        """Dequantised f32 numpy for a pin `Qwen4ExpTextModel` state-dict key,
        in the pin's own axis order.

        `rows` limits the LEADING axis (the out / vocab / expert dimension, whose
        rows are independent quant blocks) so a cheap slice cell need not
        materialise a full 2.5 GiB embedding. It is applied only where the pin's
        leading axis coincides with the GGUF tensor's (direct2d / expert3d /
        fuse / the n-gram table); it is ignored for the small vec / row / conv
        tensors, whose pin leading axis does not.

        `gguf_layer` overrides the block index read from `pin_key` -- needed when
        the tiny fixture declares a layer GDN that the real model ships as a QSA
        (full-attention) block: feed the GDN keys from a real GDN block instead.

        Raises for a derived / non-weight buffer -- the caller must not feed
        those (the pin recomputes them)."""
        layer, suffix = self._split_key(pin_key)
        if gguf_layer is not None:
            layer = gguf_layer
        if any(pin_key.endswith(s) for s in _DERIVED_SUFFIXES):
            raise KeyError(
                f"{pin_key!r} is a derived index buffer, not a fed weight "
                "(the pin recomputes layer_multipliers/vocab_sizes/offsets)"
            )
        if suffix.endswith("ple.ple_embedding.ngram_embedding.weight"):
            # The global n-gram table; the pin slices per ple-layer head range.
            return self.dequant(_PLE_TABLE, rows=rows)

        if layer is None:
            if pin_key not in _GLOBAL_MAP:
                raise KeyError(f"no GGUF map entry for global key {pin_key!r}")
            gname, kind = _GLOBAL_MAP[pin_key]
            return self._materialise(gname, kind, rows=rows)

        if suffix not in _LAYER_MAP:
            raise KeyError(f"no GGUF map entry for layer key suffix {suffix!r}")
        gname, kind = _LAYER_MAP[suffix]
        if isinstance(gname, tuple):
            gname = tuple(f"blk.{layer}.{g}" for g in gname)
        else:
            gname = f"blk.{layer}.{gname}"
        return self._materialise(gname, kind, rows=rows)

    def _materialise(self, gname, kind, rows=None):
        # rows slices the leading axis only for kinds whose pin leading axis IS
        # the GGUF leading axis; vec/row/conv are small and their leading axes
        # differ, so they dequant whole.
        row_kinds = ("direct2d", "expert3d", "fuse_gate_up")
        r = rows if kind in row_kinds else None
        if kind == "fuse_gate_up":
            gate = self.dequant(gname[0], rows=r)      # [E, ff, in]
            up = self.dequant(gname[1], rows=r)        # [E, ff, in]
            return np.concatenate([gate, up], axis=1)  # [E, 2*ff, in]
        arr = self.dequant(gname, rows=r)
        if kind in ("direct2d", "vec", "expert3d"):
            return arr
        if kind == "row":                              # 1D [H] -> [1, H]
            return arr.reshape(1, -1)
        if kind == "conv":                             # [C, K] -> [C, 1, K]
            return arr.reshape(arr.shape[0], 1, arr.shape[1])
        raise ValueError(f"unknown reshape kind {kind!r}")
