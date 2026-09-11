"""OV opset-13 emission of the qwen4_exp SparseMoeBlock (the MoE-512 gemv
layer), full-sequence, stateless.

Mirrors `tools/q4e/ref_moe.Qwen4ExpTextSparseMoeBlock.forward` (itself the
pinned transformers reference, modeling_qwen4_exp.py 981-1000) as a *static*
graph: batch fixed to 1 and sequence length fixed at build time.

The pin routes SPARSELY: the router does `torch.topk` (pin 973) and the expert
layer loops over the hit experts with `torch.where` / `index_add_` (pin
945-955) -- neither is statically emittable as opset-13. This graph computes
the SAME result DENSELY:

  * routing (pin 969-978): router_logits = x @ gate.weight^T (pin 971);
    router_probs = softmax(logits, fp32, dim=-1) (pin 972); op.topk gives the
    top-k probabilities and their expert indices (pin 973); the values are
    renormalized to sum 1 per row when norm_topk_prob (pin 974-975); the pin's
    own one-hot construct (pin 941: `one_hot(top_k_index)`) then scatters the
    renormalized scores into a dense `gate` [T, E] -- the renormalized weight
    at each selected expert, 0 elsewhere. That is byte-for-byte the
    `router_scores` the pin's loop multiplies in (pin 954), just laid out
    densely, and using torch.topk's own indices reproduces the pin's selection
    EXACTLY (including its tie-break on equal probabilities -- a value
    threshold would over-select on a tie).

  * experts (pin 921-955), dense: every expert is computed for every token and
    weighted by `gate[:, e]`. A non-selected (token, expert) pair has gate
    exactly 0.0, so its contribution `f_e(x) * 0.0 == 0.0` -- adding it changes
    nothing (x + 0.0 == x in fp32), and the sum over all E experts equals the
    pin's sum over the hit experts only. Per expert (pin 951-953):
      gate_up = x @ gate_up_proj[e]^T          -> [T, 2I]   (pin 951, chunk 2)
      inter   = silu(gate_up[:, :I]) * gate_up[:, I:]       (pin 952)
      out_e   = inter @ down_proj[e]^T          -> [T, H]    (pin 953)
    contribution_e = out_e * gate[:, e:e+1]                 (pin 954, dense)

  * shared expert (pin 986-996): a plain MLP (silu-gated) whose output is
    scaled by sigmoid(shared_expert_gate(x)) (pin 996), then added to the
    routed sum (pin 998).

Every op is ROW-LOCAL (router, each gemv and the shared expert act per token;
no op mixes positions), so no mask/pad machinery is needed -- the test's
garbage probe proves live rows are untouched by trailing-row content.

Op choices where opset-13 differs from torch:
  * F.silu (pin act_fn, config.hidden_act="silu") -> x * sigmoid(x) (opset-13
    has no silu node), the same decomposition gdn.py/hc.py use.
  * softmax(dtype=torch.float) -> op.softmax on the f32 stream (inputs are
    already f32, so the dtype cast is a no-op).
  * torch.topk (pin 973) -> op.topk (values + indices); the pin's
    `one_hot(top_k_index)` (pin 941) -> op.one_hot + reduce_sum scatters the
    scores into the dense [T, E] gate -- no dynamic gather of variable-length
    token lists, and no data-dependent loop.

Entry points:
  build_moe_model(config, state, seq_len) -> ov.Model, input `hidden_states`
    [1, T, H] f32, TWO results: `output` [1, T, H] (routed + shared sum) and
    `router_gate` [T, num_experts] (the dense top-k routing weights, exposed
    so the selection is a first-class checkable output).
  emit_moe(hidden_bth, config, state, seq_len) -> ov node [1, T, H]: the same
    subgraph for the assembled backbone (E2 inc5b), returning only `output`.

State keys (Qwen4ExpTextSparseMoeBlock.state_dict): gate.weight [E, H],
experts.gate_up_proj [E, 2I, H], experts.down_proj [E, H, I],
shared_expert.{gate_proj,up_proj,down_proj}.weight, shared_expert_gate.weight
[1, H].
"""
import numpy as np
from openvino import Model, Type
from openvino import opset13 as op

# Reused verbatim from tools/q4e/gdn.py (no op re-invented here).
from .gdn import _c, _i, _mm, _mul, _add, _slice, _reshape, _rsum, _silu  # noqa: F401


def _router_gate(h2d, config, state, T):
    """Dense top-k routing weights [T, E] (pin 969-978): the renormalized score
    at each selected expert, 0 elsewhere -- the pin's `router_scores` (pin 954)
    laid out densely via the pin's own one-hot construct (pin 941)."""
    E = config.num_experts
    k = config.num_experts_per_tok

    # pin 971: router_logits = F.linear(x, gate.weight) -> x @ W^T, W [E, H].
    logits = _mm(h2d, _c(state["gate.weight"]), tb=True)  # [T, E]
    # pin 972: softmax over the experts, fp32 (inputs already f32).
    probs = op.softmax(logits, -1)  # [T, E]

    # pin 973: router_top_value, router_indices = torch.topk(probs, k).
    # op.topk returns values (descending) and the selected expert indices.
    tk = op.topk(probs, op.constant(np.array(k, np.int64)), -1, "max", "value")
    vals = tk.output(0)                       # [T, k]  the top-k probabilities
    idx = op.convert(tk.output(1), Type.i64)  # [T, k]  the selected experts

    if config.norm_topk_prob:
        # pin 974-975: router_top_value /= router_top_value.sum(-1, keepdim).
        scores = op.divide(vals, _rsum(vals, 1))  # [T, k]
    else:
        scores = vals

    # pin 941: expert_mask = one_hot(top_k_index). Scatter the (renormalized)
    # scores into a dense [T, E] gate -- 0 at non-selected experts. Using
    # torch.topk's own indices (not a threshold on the value) reproduces the
    # pin's selection EXACTLY, including whatever tie-break torch.topk makes on
    # equal probabilities (a threshold mask would over-select on a tie).
    onehot = op.one_hot(
        idx, op.constant(np.array(E, np.int64)),
        op.constant(np.float32(1.0)), op.constant(np.float32(0.0)), -1,
    )  # [T, k, E]
    scores3 = _reshape(scores, [T, k, 1])
    gate = op.reduce_sum(_mul(onehot, scores3), _i([1]), False)  # [T, E]
    return gate


def _moe_subgraph(h2d, config, state, T):
    """h2d: [T, H] -> (output [T, H], router_gate [T, E])."""
    H = config.hidden_size
    E = config.num_experts
    I = config.moe_intermediate_size

    gate = _router_gate(h2d, config, state, T)  # [T, E]

    gate_up = state["experts.gate_up_proj"]  # [E, 2I, H]
    down = state["experts.down_proj"]        # [E, H, I]

    # pin 945-955, dense: every expert computed, weighted by gate[:, e]. A
    # non-selected pair carries gate 0 -> contributes exactly 0.
    acc = None
    for e in range(E):
        gu = _mm(h2d, _c(gate_up[e]), tb=True)  # [T, 2I]  (pin 951)
        g = _slice(gu, 0, I, 1, 1)              # [T, I]   chunk(2)[0]
        u = _slice(gu, I, 2 * I, 1, 1)          # [T, I]   chunk(2)[1]
        inter = _mul(_silu(g), u)               # pin 952: act(gate) * up
        de = _mm(inter, _c(down[e]), tb=True)   # [T, H]   (pin 953)
        ge = _slice(gate, e, e + 1, 1, 1)       # [T, 1]   the routed weight
        contrib = _mul(de, ge)                  # pin 954 (dense)
        acc = contrib if acc is None else _add(acc, contrib)

    # pin 986-996: shared expert (silu-gated MLP) scaled by
    # sigmoid(shared_expert_gate(x)).
    sg = _mm(h2d, _c(state["shared_expert.gate_proj.weight"]), tb=True)  # [T, Is]
    su = _mm(h2d, _c(state["shared_expert.up_proj.weight"]), tb=True)    # [T, Is]
    sinter = _mul(_silu(sg), su)                                         # pin 916
    sout = _mm(sinter, _c(state["shared_expert.down_proj.weight"]), tb=True)  # [T, H]
    sgate = op.sigmoid(_mm(h2d, _c(state["shared_expert_gate.weight"]), tb=True))  # [T,1]
    sout = _mul(sgate, sout)                                             # pin 996

    out2d = _add(acc, sout)  # pin 998: routed + shared
    return out2d, gate


def emit_moe(hidden_bth, config, state, seq_len):
    """The MoE subgraph for the assembled backbone (E2 inc5b): [1,T,H] node in,
    [1,T,H] node out (routed + shared sum). Reshapes to/from the [T,H] token
    layout the pin's `view(-1, H)` uses (pin 991/999)."""
    H = config.hidden_size
    T = int(seq_len)
    h2d = _reshape(hidden_bth, [T, H])       # pin 991: view(-1, H)
    out2d, _ = _moe_subgraph(h2d, config, state, T)
    return _reshape(out2d, [1, T, H])        # pin 999: reshape back


def build_moe_model(config, state, seq_len):
    H = config.hidden_size
    E = config.num_experts
    T = int(seq_len)

    hidden = op.parameter([1, T, H], Type.f32)
    hidden.set_friendly_name("hidden_states")

    h2d = _reshape(hidden, [T, H])           # pin 991: view(-1, H)
    out2d, gate = _moe_subgraph(h2d, config, state, T)
    out = _reshape(out2d, [1, T, H])         # pin 999: reshape back

    res_out = op.result(out)
    res_out.set_friendly_name("output")
    res_gate = op.result(gate)
    res_gate.set_friendly_name("router_gate")
    model = Model([res_out, res_gate], [hidden], "qwen4_exp_moe")
    return model


__all__ = ["build_moe_model", "emit_moe"]
