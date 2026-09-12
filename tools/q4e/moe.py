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
    laid out densely.

    MOE-GPU-FUSION, FIXED AND MEASURED 2026-09-12 (window-050 §4.3). The dense
    layout used to be built with the pin's own one-hot construct (pin 941):

        onehot = op.one_hot(idx, E, 1.0, 0.0, -1)            # [T, k, E]
        gate   = op.reduce_sum(onehot * scores[:, :, None], axis=1)

    That shape makes the Intel GPU plugin's own router fusion fire and then
    fail, on BOTH cards, at every width tried:

        program_builder.cpp:268  Input moerouterfused:MoERouterFused_1152.out1
                                 hasn't been found in primitive_ids map

    It is replaced by the ScatterElementsUpdate layout the production 35B-A3B
    export uses (tools/export_mtp.py:472-494 `moe_block_tiled`, extracted from
    the one IR on record that provably fuses). SAME ARITHMETIC -- measured
    byte-exact, not argued:

        |one_hot-router - scatter-router| on CPU, full MoE block  0.000000e+00

    and the before/after on the cards, one variable (this function):

        device   BEFORE one_hot (422 nodes)   AFTER scatter (421 nodes)
        CPU      OK                            OK    |dev-CPU| 0.000000e+00
        GPU.0    FAIL MoERouterFused           OK    |dev-CPU| 4.023314e-07
        GPU.1    FAIL MoERouterFused           OK    |dev-CPU| 4.451722e-07

    The 22-node isolated router reproduces the same split: one_hot FAILs on
    both cards, scatter runs and returns sum=64.0000, identical to CPU.

    The pin fidelity argument is unchanged and is why the swap is legitimate:
    both forms scatter torch.topk's OWN indices (not a threshold on the value),
    so the selection reproduces the pin EXACTLY, tie-break included. What
    changed is the ops that carry the scatter, not which experts are selected
    or with what weight -- and the 0.0 above is the proof rather than the
    claim.
    """
    E = config.num_experts
    k = config.num_experts_per_tok

    # pin 971: router_logits = F.linear(x, gate.weight) -> x @ W^T, W [E, H].
    logits = _mm(h2d, _c(state["gate.weight"]), tb=True)  # [T, E]
    # pin 972: softmax over the experts, fp32 (inputs already f32).
    probs = op.softmax(logits, -1)  # [T, E]

    # pin 973: router_top_value, router_indices = torch.topk(probs, k).
    # i32 indices: what ScatterElementsUpdate takes in the production shape.
    tk = op.topk(probs, op.constant(np.array(k, np.int32)), -1, "max", "value",
                 index_element_type="i32")
    vals = tk.output(0)                       # [T, k]  the top-k probabilities
    idx = tk.output(1)                        # [T, k]  the selected experts

    if config.norm_topk_prob:
        # pin 974-975: router_top_value /= router_top_value.sum(-1, keepdim).
        scores = op.divide(vals, _rsum(vals, 1))  # [T, k]
    else:
        scores = vals

    # A full-range Slice before the scatter. Numerically a no-op
    # (begin (0,0), end shape_of(scores), step (1,1)); the ground-truth IR has
    # it (export_mtp.py:483-492, "Slice411, between its Divide and its
    # ScatterElementsUpdate") and window-D fusion checking flagged its absence,
    # so it is reproduced literally rather than assumed irrelevant.
    scores = op.slice(scores,
                      op.constant(np.array([0, 0], np.int32)),
                      op.shape_of(scores, output_type="i32"),
                      op.constant(np.array([1, 1], np.int32)),
                      op.constant(np.array([0, 1], np.int32)))
    # zeros [T, E] derived from probs so the shape needs no ShapeOf plumbing
    zeros = _mul(probs, _c(np.float32(0.0)))
    gate = op.scatter_elements_update(zeros, idx, scores,
                                      op.constant(np.array(-1, np.int32)))
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


# ---------------------------------------------------------------------------
# PIECEWISE MoE pieces (window-050 / piecewise_export). The real checkpoint has
# E=512 experts; a single dense MoE graph at real width is ~6.7 GB (gate_up)
# + ~3.4 GB (down) of f32 constants for ONE layer, so the real-width MoE piece
# is split into ROUTER / EXPERT-CHUNK / SHARED pieces. The equality theorem
# (dense == sparse) makes the chunk split measurement-exact: a non-selected
# expert carries gate 0 and contributes exactly 0, so
# sum over chunks of chunk outputs + shared == the pin's sparse output on the
# same gate, and `router_gate` is exactly the pin's `router_gate` (same
# softmax/topk in f32). The theorem is MEASURED, not asserted by construction:
# `tests/python/test_moe_chunk_partition.py` (CF-CHUNKCOV, 2026-09-12) sweeps
# four partitions of the expert axis -- including an uneven one and the
# one-expert-per-chunk degenerate -- at two synthetic geometries against the
# pin's own SPARSE block, plus a third geometry of 16 REAL expert bodies at
# real width against a float64 recomputation. Between the discard of
# `test_piecewise_export.py` and that file, these builders had no gate at all
# (REVIEW e78812d F1).
# ---------------------------------------------------------------------------
def emit_router_gate(hidden_bth, config, state, seq_len):
    """[1,T,H] -> [T,E] dense top-k gate. The ROUTER is
    `Qwen4ExpTextTopKRouter.forward`, pin 969-978 (linear -> f32 softmax ->
    topk -> the norm_topk_prob renormalisation); pin 954 is where the expert
    loop multiplies the routed weight back in, which is a different line and
    was the citation this docstring carried until 2026-09-12."""
    H = config.hidden_size
    T = int(seq_len)
    return _router_gate(_reshape(hidden_bth, [T, H]), config, state, T)


def build_router_model(config, state, seq_len):
    T = int(seq_len)
    H = config.hidden_size
    hidden = op.parameter([1, T, H], Type.f32)
    hidden.set_friendly_name("hidden_states")
    gate = emit_router_gate(hidden, config, state, seq_len)
    res = op.result(gate)
    res.set_friendly_name("router_gate")
    return Model([res], [hidden], "qwen4_exp_moe_router")


def emit_experts_chunk(hidden_bth, gate_chunk, config, state, e0, e1, seq_len):
    """[1,T,H] x [T,C] (the gate slice for experts e0..e1) -> [T,H], the sum
    of the chunk's contributions -- pin 945-955 rolled over e in [e0, e1). The
    caller verifies sum-over-chunks == pin sparse outside the graph (the
    theorem is measured in the test, not asserted by construction):
    `tests/python/test_moe_chunk_partition.py`.

    The local `j` addresses the weight slice and the gate column TOGETHER --
    `gate_up[j]`/`down[j]` are the chunk's j-th expert and `gate_chunk[:, j]`
    is that expert's routed weight. That coupling is gated by the relabel-
    invariance cell (permute the chunk's experts and its gate columns by the
    same permutation -> output invariant; permute only the weights -> it
    moves), because reading it off the source is not measuring it."""
    H = config.hidden_size
    I = config.moe_intermediate_size
    T = int(seq_len)
    h2d = _reshape(hidden_bth, [T, H])
    gate_up = state["experts.gate_up_proj"][e0:e1]  # [C, 2I, H]
    down = state["experts.down_proj"][e0:e1]        # [C, H, I]
    acc = None
    for j in range(e1 - e0):
        gu = _mm(h2d, _c(gate_up[j]), tb=True)      # [T, 2I]  (pin 951)
        g = _slice(gu, 0, I, 1, 1)                  # [T, I]
        u = _slice(gu, I, 2 * I, 1, 1)              # [T, I]
        inter = _mul(_silu(g), u)                   # pin 952
        de = _mm(inter, _c(down[j]), tb=True)       # [T, H]   (pin 953)
        ge = _slice(gate_chunk, j, j + 1, 1, 1)     # [T, 1]   the routed weight
        contrib = _mul(de, ge)                      # pin 954 (dense)
        acc = contrib if acc is None else _add(acc, contrib)
    return acc


def build_experts_chunk_model(config, state, seq_len, e0, e1):
    T = int(seq_len)
    H = config.hidden_size
    C = e1 - e0
    hidden = op.parameter([1, T, H], Type.f32)
    hidden.set_friendly_name("hidden_states")
    gate = op.parameter([T, C], Type.f32)
    gate.set_friendly_name("gate_chunk")
    out = emit_experts_chunk(hidden, gate, config, state, e0, e1, T)
    res = op.result(out)
    res.set_friendly_name("output")
    return Model([res], [hidden, gate], f"qwen4_exp_moe_chunk_{e0}_{e1}")


def emit_shared_expert(hidden_bth, config, state, seq_len):
    """[1,T,H] -> [T,H], the shared-expert term scaled by sigmoid(shared
    expert gate) -- pin 986-996."""
    H = config.hidden_size
    I = config.shared_expert_intermediate_size
    T = int(seq_len)
    h2d = _reshape(hidden_bth, [T, H])
    sg = _mm(h2d, _c(state["shared_expert.gate_proj.weight"]), tb=True)
    su = _mm(h2d, _c(state["shared_expert.up_proj.weight"]), tb=True)
    sinter = _mul(_silu(sg), su)
    sout = _mm(sinter, _c(state["shared_expert.down_proj.weight"]), tb=True)
    sgate = op.sigmoid(_mm(h2d, _c(state["shared_expert_gate.weight"]), tb=True))
    return _mul(sgate, sout)


def build_shared_expert_model(config, state, seq_len):
    T = int(seq_len)
    H = config.hidden_size
    hidden = op.parameter([1, T, H], Type.f32)
    hidden.set_friendly_name("hidden_states")
    out = emit_shared_expert(hidden, config, state, T)
    res = op.result(out)
    res.set_friendly_name("output")
    return Model([res], [hidden], "qwen4_exp_moe_shared")


__all__ = [
    "build_moe_model", "emit_moe",
    "build_router_model", "build_experts_chunk_model",
    "build_shared_expert_model",
]
