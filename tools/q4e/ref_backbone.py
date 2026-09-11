"""Reference transcription of the qwen4_exp text backbone (Qwen4ExpTextModel +
Qwen4ExpTextDecoderLayer forward), GDN-only no-cache, for the E2 inc5b assembly
parity test.

ORACLE: the pinned reference (modeling_qwen4_exp.py sha256 ca9f00bb...). This
transcribes the DecoderLayer forward (pin 1273-1310) and the TextModel forward
(pin 1400-1498), reusing the pin's OWN leaves (Qwen4ExpTextGatedDeltaNet,
Qwen4ExpTextSparseMoeBlock, Qwen4ExpTextGatedResidual, Qwen4ExpTextPLELayer,
nn.Embedding) so a composition-wiring error (a swapped hyper-connection, a
dropped PLE add, a wrong combine order) diverges from the pin while every leaf's
math stays the pin's.

THE HEAD: PIN-vs-CHECKPOINT DIVERGENCE (measured 2026-09-11; reviewer finding B
of REVIEW 2cd2b2f, reproduced independently -- see the table below). The pin ties
the head to the embedding (pin 1593
`_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`; pin 1604
declares `self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)`, pin 1669
applies it). THE SHIPPED UD-Q3_K_XL CHECKPOINT DOES NOT: it carries a separate
`output.weight` (Q6_K) alongside `token_embd.weight` (Q8_0), and the two are
independent tensors, not converter duplicates:

    rows            |embd|    |out|    mean-cos   max-cos   min-cos
    0:64            0.1414   0.8949    -0.00156    +0.035    -0.049
    1000:1064       0.2823   0.7448    +0.00281    +0.052    -0.058
    100000:100064   0.3867   0.6521    +0.02634    +0.080    -0.063
    248000:248064   0.4477   0.6369    +0.01495    +0.063    -0.034
    max-abs(embd - out) over rows 0:256 = 4.132773e-01

Tied would give cos == 1. THE SERVED WEIGHTS ARE THE TRUTH: when a source
provides a head, it is fed and used; the pin's tied behaviour is the FALLBACK for
sources that ship no head (the tiny random fixtures, a genuinely tied
checkpoint). `declare_lm_head=True` registers the separate head
(`lm_head.weight` joins the state dict); the default False keeps the state dict
key-for-key identical to the pin TextModel's, which every transcription-vs-pin
parity leg depends on.

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
    hyper_connection_mixer).

    `declare_lm_head=True` adds ONE key the pin TextModel does not have --
    `lm_head.weight`, the separate head the shipped checkpoint ships as
    `output.weight` (module docstring: the pin ties, the checkpoint does not).
    Default False so the state dict stays key-for-key the pin's; any leg that
    loads this state dict into `pin_mod.Qwen4ExpTextModel` needs that."""

    def __init__(self, config, declare_lm_head=False):
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
        # pin 1604: ForCausalLM's own head. Registered only when the source
        # declares one; otherwise `logits` falls back to the pin's tie.
        self.lm_head = (
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            if declare_lm_head else None
        )

    def forward(self, input_ids, conv_mask):
        emb = self.embed_tokens(input_ids)                       # pin 1420  [1,T,H]
        ple_input_ids = input_ids                                # pin 1425 (no pad here)
        hidden = emb.repeat(1, 1, self.hc_count)                 # pin 1480  [1,T,hc*H]
        for layer in self.layers:
            if layer.ple is not None:                            # pin 1283-1286: PLE additive
                hidden = hidden + layer.ple(hidden, ple_input_ids, None, conv_mask=conv_mask)
            hidden, hyper, inj = layer.attn_hyper_connection(hidden)                # pin 1288
            g = layer.linear_attn(hidden, cache_params=None, attention_mask=conv_mask)  # pin 1290
            hidden = hyper + (g.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)      # pin 1302-1303
            hidden, hyper, inj = layer.mlp_hyper_connection(hidden)                 # pin 1305
            m = layer.mlp(hidden)                                                    # pin 1306
            hidden = hyper + (m.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)      # pin 1308-1309
        hidden = self.hyper_connection_mixer(hidden)             # pin 1493  [1,T,H]
        return hidden

    def logits(self, input_ids, conv_mask):
        hidden = self.forward(input_ids, conv_mask)
        if self.lm_head is not None:
            return self.lm_head(hidden)                          # pin 1669, fed head
        return hidden @ self.embed_tokens.weight.t()             # pin 1593 tie (fallback)


__all__ = ["Qwen4ExpTextBackbone"]
