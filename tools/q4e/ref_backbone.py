"""Reference transcription of the qwen4_exp text backbone (Qwen4ExpTextModel +
Qwen4ExpTextDecoderLayer forward), GDN-only no-cache, for the E2 inc5b assembly
parity test.

ORACLE: the pinned reference (modeling_qwen4_exp.py sha256 ca9f00bb...). This
transcribes the DecoderLayer forward (pin 1273-1310) and the TextModel forward
(pin 1400-1497), reusing the pin's OWN leaves (Qwen4ExpTextGatedDeltaNet,
Qwen4ExpTextSparseMoeBlock, Qwen4ExpTextGatedResidual, Qwen4ExpTextPLELayer,
nn.Embedding) so a composition-wiring error (a swapped hyper-connection, a
dropped PLE add, a wrong combine order) diverges from the pin while every leaf's
math stays the pin's. The tied lm_head (pin: lm_head.weight = embed_tokens.weight)
is applied as `hidden @ embed_tokens.weight.T`.

SCOPE (frontier ruling): causal-only, GDN layers only -- no QSA full-attention
layer, so rope / position_embeddings are unused (GDN ignores them) and are not
constructed here. conv_mask is taken explicitly (the pin builds it from
attention_mask via create_recurrent_attention_mask; for a full sequence it is
all-valid == ones). See tools/q4e/backbone.py.
"""
import torch
from torch import nn

from transformers.models.qwen4_exp import modeling_qwen4_exp as _pin  # noqa: F401


class Qwen4ExpTextBackbone(nn.Module):
    """Transcribed Qwen4ExpTextModel (GDN-only, no-cache). Its state_dict keys
    match the pin TextModel's exactly (same module names: embed_tokens, layers.i.
    {linear_attn,mlp,attn_hyper_connection,mlp_hyper_connection,ple},
    hyper_connection_mixer)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hc_count = config.hc_count
        self.ple_layer_ids = list(config.ple_layer_ids or [])
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = nn.Module()
            layer.linear_attn = _pin.Qwen4ExpTextGatedDeltaNet(config, i)          # pin 1263
            layer.mlp = _pin.Qwen4ExpTextSparseMoeBlock(config)                    # pin 1266
            layer.attn_hyper_connection = _pin.Qwen4ExpTextGatedResidual(config)   # pin 1270 (use_combine=True)
            layer.mlp_hyper_connection = _pin.Qwen4ExpTextGatedResidual(config)    # pin 1271
            if (i + 1) in self.ple_layer_ids:                                      # pin 1268-1269
                ple_index = self.ple_layer_ids.index(i + 1)
                layer.ple = _pin.Qwen4ExpTextPLELayer(config, i, ple_index)
            else:
                layer.ple = None
            self.layers.append(layer)
        self.hyper_connection_mixer = _pin.Qwen4ExpTextGatedResidual(config, use_combine=False)  # pin 1393

    def forward(self, input_ids, conv_mask):
        emb = self.embed_tokens(input_ids)                       # pin 1420  [1,T,H]
        ple_input_ids = input_ids                                # pin 1425 (no pad here)
        hidden = emb.repeat(1, 1, self.hc_count)                 # pin 1480  [1,T,hc*H]
        for layer in self.layers:
            if layer.ple is not None:                            # pin 1282-1285: PLE additive
                hidden = hidden + layer.ple(hidden, ple_input_ids, None, conv_mask=conv_mask)
            hidden, hyper, inj = layer.attn_hyper_connection(hidden)                # pin 1288
            g = layer.linear_attn(hidden, cache_params=None, attention_mask=conv_mask)  # pin 1289
            hidden = hyper + (g.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)      # pin 1302-1303
            hidden, hyper, inj = layer.mlp_hyper_connection(hidden)                 # pin 1305
            m = layer.mlp(hidden)                                                    # pin 1306
            hidden = hyper + (m.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)      # pin 1308-1309
        hidden = self.hyper_connection_mixer(hidden)             # pin 1493  [1,T,H]
        return hidden

    def logits(self, input_ids, conv_mask):
        hidden = self.forward(input_ids, conv_mask)
        return hidden @ self.embed_tokens.weight.t()             # tied lm_head


__all__ = ["Qwen4ExpTextBackbone"]
