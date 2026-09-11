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

THE N-GRAM ROW INDEX (the parity seam). The row-index -> table-row function is
a pure INTEGER hash (splitmix-derived multipliers, an XOR-of-(token*multiplier)
reduced modulo a per-head prime vocab, eos-boundary shifting): pin 1080-1181,
byte-identical to arcint's vector-tested src/exec/ngram_row_ids.h. It requires
exact int64 arithmetic on values up to ~2^63; the installed OpenVINO build's CPU
integer kernels are 32-bit (MEASURED -- see tools/q4e/ple.py's header), so the
index is NOT emitted in-graph. Per the frontier ruling the graph consumes
row_ids as a declared int64 input; the two PRODUCERS of that input, on the two
sides of the parity seam, are:
  * tests -- the in-file numpy int64 generator (test_ple_block._gen_row_ids),
    true 64-bit python ints, validated three-ways against the committed Link-3
    vectors (tests/ngram_row_ids_vectors.h) before any graph run;
  * serving -- arcint's src/exec/ngram_row_ids.h (AVX2-verified; the NEON twin
    is a separate queue item). This module does NOT reimplement or re-verify
    that kernel.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.qwen4_exp import modeling_qwen4_exp as _pin  # noqa: F401


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


__all__ = ["Qwen4ExpTextPLELayer"]
