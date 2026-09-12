"""Reference transcription of the qwen4_exp SparseMoeBlock (the MoE-512 gemv
layer), for numeric parity tests.

The ORACLE is the pinned transformers reference (tools/export_qwen4_exp.py
REFERENCE_COMMIT; modeling_qwen4_exp.py sha256
`ca9f00bbd73cfcfbad7ba6073b5ecc23ca27bdace58b746fe6ead6ab175efc0c`, re-hashed
by the test before every numeric table). This module transcribes
`Qwen4ExpTextSparseMoeBlock.forward` (pin lines 981-1000), reusing the pin's
OWN leaf submodules -- `Qwen4ExpTextTopKRouter` (pin 960-978),
`Qwen4ExpTextExperts` (pin 921-955) and `Qwen4ExpTextMLP` (pin 904-916) -- so
the leaf math is the pin's byte-for-byte while a wiring error in the block
forward (a dropped shared-expert-gate sigmoid, a missing routed/shared add, a
skipped renorm) still diverges from the pin. Validated AGAINST the pin at 0.0
(test drives both this class and the pin's on the same weights).

The block is STATELESS (no KV cache branch), so -- unlike the GDN block --
there is a single forward to mirror. Routing is data-dependent: the pin's
router does `torch.topk` and the expert layer loops over the hit experts with
`torch.where` / `index_add_`. tools/q4e/moe.py emits the SAME result as a
STATIC DENSE graph -- every expert computed, weighted by a dense gate built by
SCATTERING TopK's own indices, which is not the same thing as a threshold on
the value and the difference matters on a tie. That equals the sparse loop
exactly because a non-selected (token, expert) pair carries gate 0, so its
dense contribution is `f_e(x) * 0.0 == 0.0`. The equivalence argument, the
distinctness premise the scatter layout rests on, and the tie caveat are all
in tools/q4e/moe.py, each naming the cell that gates it.

Per-position-ness: the MoE block is fully ROW-LOCAL (the router, every expert
gemv and the shared expert act per token; no op mixes positions). A masked
(zeroed) row routes on all-zero logits: softmax is then uniform, an all-way
tie, and each selected expert carries gate exactly 1/top_k. WHICH experts is
NOT determined by index order -- that sentence stood here until 2026-09-12 and
a measurement contradicts it: at E=16, top_k=4 the pin's `torch.topk` selects
[9, 10, 11, 12] on such a row where OpenVINO's `op.topk` selects [0, 1, 2, 3]
(test_moe_block.py::test_a_degenerate_row_may_select_differently_and_still_emits_zero).
It does not matter here, and the reason is measured rather than assumed:
bias-free Linears give f_e(0) = 0, so whichever experts a zeroed row picks,
both sides emit exactly 0.0 on it. Because the block is row-local, a garbage
probe in masked rows leaves every live row untouched -- the same row-locality
proof the hc block carries.
"""
import torch
import torch.nn.functional as F
from torch import nn

# Pinned leaf submodules (modeling_qwen4_exp.py): the transcription below must
# use EXACTLY these, so its per-expert / router math cannot drift from the pin.
#   Qwen4ExpTextTopKRouter  (pin 960-978)
#   Qwen4ExpTextExperts     (pin 921-955)
#   Qwen4ExpTextMLP         (pin 904-916)
from transformers.models.qwen4_exp import modeling_qwen4_exp as _pin  # noqa: F401


class Qwen4ExpTextSparseMoeBlock(nn.Module):
    """Transcribed from pin `Qwen4ExpTextSparseMoeBlock` (modeling_qwen4_exp.py
    lines 981-1000). Leaves are the pin's; the forward body is line-for-line
    the pin's (pin 989-1000)."""

    def __init__(self, config):
        super().__init__()
        self.gate = _pin.Qwen4ExpTextTopKRouter(config)                      # pin 984
        self.experts = _pin.Qwen4ExpTextExperts(config)                      # pin 985
        self.shared_expert = _pin.Qwen4ExpTextMLP(                           # pin 986
            config, intermediate_size=config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)  # pin 987

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape           # pin 990
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)             # pin 991
        shared_expert_output = self.shared_expert(hidden_states_reshaped)       # pin 992
        # pin 993: router returns (logits, renormalized top-k scores, indices)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        # pin 994: the sparse expert layer (loop over hit experts, index_add_)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        # pin 996: sigmoid-gated shared expert
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        )
        expert_output = expert_output + shared_expert_output                    # pin 998
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)  # pin 999
        return expert_output                                                    # pin 1000


__all__ = ["Qwen4ExpTextSparseMoeBlock"]
