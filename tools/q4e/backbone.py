"""OV opset-13 emission of a small-but-complete qwen4_exp text backbone (E2
inc5b, the assembly capstone): embed -> repeat(hc) -> N decoder layers ->
final hyper_connection_mixer (use_combine=False) -> lm_head -> logits.

THE HEAD: PIN-vs-CHECKPOINT DIVERGENCE (measured 2026-09-11; reviewer finding B
of REVIEW 2cd2b2f, reproduced independently -- the table is in
`q4e/ref_backbone`'s header). The pin TIES the head to the embedding (pin 1593
`_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`, applied
pin 1669). THE SHIPPED UD-Q3_K_XL CHECKPOINT DOES NOT: its `output.weight`
(Q6_K) and `token_embd.weight` (Q8_0) are independent tensors -- row-band mean
cosine -0.0016 / +0.0028 / +0.0263 / +0.0150 across the vocab, where tied would
be 1.0.

THE SERVED WEIGHTS ARE THE TRUTH, so this emitter takes the head from
`state["lm_head.weight"]` WHEN THE STATE PROVIDES IT, and falls back to the
pin's tie (`embed_tokens.weight`) only when it is absent -- which is the tiny
random fixtures, and a genuinely tied checkpoint. Emitting the tie against this
checkpoint would ship the wrong head; the fallback exists for sources that
really are tied, not as a default for this one.

Mirrors `tools/q4e/ref_backbone.Qwen4ExpTextBackbone.forward` (the transcription
of Qwen4ExpTextModel.forward + Qwen4ExpTextDecoderLayer.forward, pin 1258-1498,
GDN-only no-cache branch) reusing the already-validated per-block emitters:
  gdn.emit_gdn  hc.emit_combine / emit_hc  moe.emit_moe  ple.emit_ple.

DENSE-CAUSAL SCOPE -- CORRECTION, 2026-09-12 (frontier ruling; supersedes the
earlier CAUSAL-ONLY note, which wrongly read as excusing the attention LAYERS):
the checkpoint is 48 layers = 36 `linear_attention` (GDN) + 12 full-attention
layers (layer_idx % full_attention_interval == interval-1). The frontier rule
excused the INDEXER only, not the attention layers: dense causal attention IS
the semantics the QSA indexer approximates, so every full-attention layer
assembles as DENSE CAUSAL -- emitted from the pin's own Qwen4ExpTextAttention
(pin 819-901) MINUS the QSA selection branch (the per-batch/per-query Python
loops and their `torch.nonzero`, pin 729-731, NOT statically opset-13-emittable,
E1.5 finding 7) and with the pin's RoPE in its degenerate-for-text form (all
four mrope position rows equal the text positions: pin 1433-1436 expands
arange(T) to 4 identical rows, pin 1438-1440 hands row 0 to the QSA mask and
rows 1-3 to the rotary module, pin 79-149). That emission is
`tools/q4e/attention.py` (its layer piece is real-width parity-validated; this
module's tiny fixtures stay all-linear for the harness floor). The GDN branches
ignore position_embeddings and emit none.

Two pin corrections this increment establishes (pin wins; dated in RECONCILE):
  * PLE is ADDITIVE, not a layer replacement: `hidden = hidden + ple(...)` at
    the layer top (pin 1283) -- E1.5 finding 2 (PLE "replaces" the layer) was
    wrong.
  * There is NO final RMSNorm: TextModel returns `hyper_connection_mixer(hidden)`
    directly as last_hidden_state (pin 1493-1496) and ForCausalLM applies
    lm_head to it (no norm) -- the kickoff's "-> RMSNorm -> lm_head" is inexact.

The per-layer composition (pin 1273-1310):
  (PLE layer only) hidden = hidden + ple(hidden, row_ids, conv_mask)   pin 1283-4
  h, hyper, inj = attn_hyper_connection(hidden)      (use_combine=True) pin 1288
  g = linear_attn(h, conv_mask)                                        pin 1290
  hidden = hyper + (g.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)   pin 1302-3
  h, hyper, inj = mlp_hyper_connection(hidden)                          pin 1305
  m = mlp(h)                                          (MoE)             pin 1306
  hidden = hyper + (m.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)   pin 1308-9

Inputs (static, batch 1, fixed T):
  input_ids     [1, T]                    i64  (embedding lookup; head per above)
  ngram_row_ids [1, T, num_ngram_heads]   i64  (fed PLE index -- see ple.py)
  conv_mask     [1, T]                    f32  (GDN + PLE mask; ones = full seq)
Result: logits [1, T, vocab_size] f32.
"""
import numpy as np
from openvino import Model, Type
from openvino import opset13 as op

from .gdn import _c, _mm, _add, _reshape, emit_gdn
from .hc import emit_combine, emit_hc
from .moe import emit_moe
from .ple import emit_ple


def _sub(state, prefix):
    """The sub-state under `prefix`, prefix stripped (so each block emitter sees
    its own key names)."""
    n = len(prefix)
    return {k[n:]: v for k, v in state.items() if k.startswith(prefix)}


def _combine(hyper, mixer_out, inj, T, hc, H):
    """pin 1302-1303 / 1308-1309: hidden = hyper + (mixer.unsqueeze(-2) *
    inj.unsqueeze(-1)).flatten(-2). hyper [1,T,hc*H], mixer_out [1,T,H], inj
    [1,T,hc] -> [1,T,hc*H]."""
    m4 = _reshape(mixer_out, [1, T, 1, H])         # unsqueeze(-2)
    inj4 = _reshape(inj, [1, T, hc, 1])            # unsqueeze(-1)
    injection = op.multiply(m4, inj4)              # [1,T,hc,H] (broadcast)
    return _add(hyper, _reshape(injection, [1, T, hc * H]))


def build_backbone(config, state, seq_len):
    H = config.hidden_size
    hc = config.hc_count
    Hn = (config.ngram_size - 1) * config.heads_per_ngram
    T = int(seq_len)
    L = config.num_hidden_layers
    ple_layer_ids = list(config.ple_layer_ids or [])

    input_ids = op.parameter([1, T], Type.i64)
    input_ids.set_friendly_name("input_ids")
    ngram_row_ids = op.parameter([1, T, Hn], Type.i64)
    ngram_row_ids.set_friendly_name("ngram_row_ids")
    conv_mask = op.parameter([1, T], Type.f32)
    conv_mask.set_friendly_name("conv_mask")

    embed_w = state["embed_tokens.weight"]                    # [vocab, H]
    # pin 1420: inputs_embeds = embed_tokens(input_ids)
    emb = op.gather(_c(embed_w), input_ids, op.constant(np.int64(0)))  # [1,T,H]
    # pin 1480: inputs_embeds.repeat(1, 1, hc_count)
    hidden = op.concat([emb] * hc, axis=2)                    # [1,T,hc*H]

    for i in range(L):
        pfx = f"layers.{i}."
        # pin 1283-1284: PLE additive (only on the configured PLE layer)
        if (i + 1) in ple_layer_ids:
            ple_out = emit_ple(hidden, ngram_row_ids, config, _sub(state, pfx + "ple."), T, conv_mask)
            hidden = _add(hidden, ple_out)

        # pin 1288: attn_hyper_connection (use_combine=True)
        h, hyper, inj = emit_combine(hidden, config, _sub(state, pfx + "attn_hyper_connection."), T)
        # pin 1290: linear_attn (GDN) on the mixed stream
        g = emit_gdn(h, conv_mask, config, _sub(state, pfx + "linear_attn."), T)
        # pin 1302-1303: combine back into the hc stream
        hidden = _combine(hyper, g, inj, T, hc, H)

        # pin 1305: mlp_hyper_connection (use_combine=True)
        h2, hyper2, inj2 = emit_combine(hidden, config, _sub(state, pfx + "mlp_hyper_connection."), T)
        # pin 1306: mlp (MoE) on the mixed stream
        m = emit_moe(h2, config, _sub(state, pfx + "mlp."), T)
        # pin 1308-1309: combine back
        hidden = _combine(hyper2, m, inj2, T, hc, H)

    # pin 1493: final hyper_connection_mixer (use_combine=False) -> [1,T,H]
    final = emit_hc(hidden, config, _sub(state, "hyper_connection_mixer."), T)
    # pin 1669: logits = lm_head(final). The served weights are the truth -- a
    # fed `lm_head.weight` (GGUF output.weight) wins; the pin's tie to
    # embed_tokens (pin 1593) is the fallback for sources that ship no head.
    # See the module header: the shipped checkpoint is NOT tied.
    head_w = state.get("lm_head.weight", embed_w)             # [vocab, H]
    logits = _mm(final, _c(head_w), tb=True)                  # [1,T,vocab]

    res = op.result(logits)
    res.set_friendly_name("logits")
    return Model([res], [input_ids, ngram_row_ids, conv_mask], "qwen4_exp_backbone")


__all__ = ["build_backbone"]
