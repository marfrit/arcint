"""Reference transcription of the qwen4_exp PLELayer (n-gram PLE block), for
numeric parity tests.

The ORACLE is the pinned transformers reference (modeling_qwen4_exp.py sha256
`ca9f00bbd73cfcfbad7ba6073b5ecc23ca27bdace58b746fe6ead6ab175efc0c`, re-hashed
by the test before every numeric table). This module transcribes
`Qwen4ExpTextPLELayer.forward` (pin lines 1235-1255, no-cache branch) reusing
the pin's OWN leaves -- `Qwen4ExpTextNGramEmbedding` (pin 1080-1181, which
itself derives the hash constants and gathers the n-gram embedding),
`Qwen4ExpTextRMSNorm` (pin 152-172) and torch's nn.Linear/nn.Conv1d -- so a
forward-wiring error (a dropped signed-sqrt, a wrong residual, a mis-normed
stream) diverges from the pin while the leaf math stays the pin's. Validated
AGAINST the pin at 0.0 (test drives both classes on the same weights).

THE N-GRAM ROW INDEX. The row-index -> table-row function is a pure INTEGER
hash (splitmix-derived multipliers, an XOR-of-(token*multiplier) reduced modulo
a per-head prime vocab, eos-boundary shifting): pin 1080-1181, byte-identical
to arcint's vector-tested src/exec/ngram_row_ids.h. It requires exact int64
arithmetic on values up to ~2^63. The installed OpenVINO build's CPU integer
kernels are 32-bit (measured: i64 Multiply wraps at 2^32, i64 Add breaks past
int32 -- see tools/q4e/ple.py and the RECONCILE session block), so the index
CANNOT be emitted as opset-13 integer ops on this build. The index is therefore
produced by this validated derivation (row_ids below, proven bit-exact against
the committed Link-3 vectors, tests/ngram_row_ids_vectors.h) and FED to the OV
graph as an int64 input `ngram_row_ids`; tools/q4e/ple.py emits only the gather
+ the (float) PLE forward. The row_ids helper here mirrors the pin's own
NGramEmbedding path and the test asserts, per input, that its ids equal the ids
the pin's NGramEmbedding uses internally.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.qwen4_exp import modeling_qwen4_exp as _pin  # noqa: F401

_MASK64 = (1 << 64) - 1


def _s64(x: int) -> int:
    x &= _MASK64
    return x - (1 << 64) if x >= (1 << 63) else x


def row_ids_from_input(ngemb, input_ids_row):
    """Bit-exact n-gram row ids for one [T] sequence (fresh, all-eos context),
    using the pin NGramEmbedding module's OWN derived buffers (layer_multipliers
    / ngram_heads_vocab_sizes / ngram_heads_offsets, pin 1107-1111) and the
    documented shift/XOR/mod mixing (pin 1131-1181). Returns [T, num_heads]
    int64. Proven byte-equal to the committed Link-3 vectors and to the pin's
    internal ids (see test_ple_block)."""
    ngram = ngemb.ngram_size
    hpn = ngemb.heads_per_ngram
    eos = int(ngemb.eos_token_id)
    mult = ngemb.layer_multipliers.tolist()
    sizes = ngemb.ngram_heads_vocab_sizes.tolist()
    offs = ngemb.ngram_heads_offsets.tolist()

    tokens = [int(t) for t in input_ids_row]
    ctx = [eos] * (ngram - 1)                              # fresh sequence (pin 1123)
    packed = ctx + tokens
    W = len(packed)
    # in_segment (pin _shift_right_ignore_eos, 1121-1126): pos - prev_eos - 1
    prev_eos = [-1] * W
    last = -1
    for p in range(W):
        prev_eos[p] = last
        if packed[p] == eos:
            last = p
    in_seg = [p - prev_eos[p] - 1 for p in range(W)]
    shifted = [list(packed)]
    for s in range(1, ngram):
        shifted.append([packed[p - s] if (p - s >= 0 and in_seg[p] >= s) else eos
                        for p in range(W)])
    per = [[row[(ngram - 1) + i] for i in range(len(tokens))] for row in shifted]

    out = []
    for i in range(len(tokens)):
        heads = []
        for n in range(2, ngram + 1):                     # pin 1149
            start = (n - 2) * hpn
            mixed = _s64(per[0][i] * mult[0])             # pin 1152
            for pos in range(1, n):                       # pin 1153-1157
                mixed = _s64((mixed & _MASK64) ^ (_s64(per[pos][i] * mult[pos]) & _MASK64))
            for h in range(start, start + hpn):           # pin 1160-1161
                heads.append(mixed % sizes[h] + offs[h])
        out.append(heads)
    return torch.tensor(out, dtype=torch.long)


class Qwen4ExpTextPLELayer(nn.Module):
    """Transcribed from pin `Qwen4ExpTextPLELayer` (modeling_qwen4_exp.py lines
    1183-1255), no-cache branch. Leaves are the pin's; the forward body is
    line-for-line the pin's forward (pin 1242-1255)."""

    def __init__(self, config, layer_idx: int = 0, ple_layer_index: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        ple_embed_dim = config.ple_embed_dim
        hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = _pin.Qwen4ExpTextNGramEmbedding(       # pin 1198
            config, ple_embed_dim, layer_idx, ple_layer_index
        )
        conv_kernel_size = config.ple_conv_kernel_size
        conv_dilation = config.ngram_size
        self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation  # pin 1201
        self.key_proj = nn.Linear(ple_embed_dim, hc_hidden_size, bias=False)   # pin 1202
        self.value_proj = nn.Linear(ple_embed_dim, self.hidden_size, bias=False)  # 1203
        self.norm_key = _pin.Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.norm_query = _pin.Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.norm_conv = _pin.Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.conv1d = nn.Conv1d(hc_hidden_size, hc_hidden_size, kernel_size=conv_kernel_size,
                                groups=hc_hidden_size, dilation=conv_dilation, bias=False)  # pin 1207

    def _short_conv(self, hidden_states):
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.transpose(1, 2)                     # pin 1218
        hidden_states = F.pad(hidden_states, (self.short_conv_state_len, 0))  # pin 1228
        hidden_states = hidden_states[..., -(self.short_conv_state_len + seq_len):]  # 1229
        hidden_states = F.silu(self.conv1d(hidden_states))               # pin 1232
        hidden_states = hidden_states.transpose(1, 2)                     # pin 1234
        return hidden_states

    def forward(self, hidden_states, input_ids, conv_mask=None):
        embeddings = self.ple_embedding(input_ids, None)                 # pin 1242
        key_normed = self.norm_key(self.key_proj(embeddings)).unflatten(-1, (self.hc_count, self.hidden_size))  # 1243
        value = self.value_proj(embeddings)                             # pin 1244
        query_normed = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))  # 1245
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)  # 1246
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()          # pin 1247
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)        # pin 1248
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))    # pin 1249
        gated_value = gated_value.flatten(-2)                           # pin 1250
        if conv_mask is not None:                                       # pin 1251-1253
            gated_value = _pin.apply_mask_to_padding_states(gated_value, conv_mask)
            gated_value_normed = _pin.apply_mask_to_padding_states(gated_value_normed, conv_mask)
        output = gated_value + self._short_conv(gated_value_normed)     # pin 1254
        return output                                                  # pin 1255


__all__ = ["Qwen4ExpTextPLELayer", "row_ids_from_input"]
