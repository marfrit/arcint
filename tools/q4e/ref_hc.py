"""Reference transcription of the qwen4_exp GatedResidual (hyper-connection
mixer), for numeric parity tests.

The ORACLE is the pinned transformers reference (tools/export_qwen4_exp.py
REFERENCE_COMMIT; modeling_qwen4_exp.py sha256
`ca9f00bbd73cfcfbad7ba6073b5ecc23ca27bdace58b746fe6ead6ab175efc0c`). The free
functions / norm class it uses are imported from the installed pinned module
(the dev-host venv holds transformers 5.17.0 whose models/qwen4_exp files are
byte-identical to the pin -- tests/python/test_gdn_block.py asserts that by
hash before every numeric table); the transcription below is the
`Qwen4ExpTextGatedResidual` class (pin lines 1003-1031) rewritten against
torch nn primitives.

The transcription is validated AGAINST the pin, not against itself: the
test drives BOTH this class and the pin's `Qwen4ExpTextGatedResidual` on
the same random weights/inputs and requires agreement at 1e-5. A
transcription error therefore goes red the same way an OV emission error
goes red.

The branch pinned here (what tools/q4e/hc.py emits as opset-13) is the
`use_combine=False` form -- the one the text model builds as its final
`hyper_connection_mixer` (pin line 1393): forward returns ONLY the mixed
[B, T, H]; no injection stream is produced. The `use_combine=True` form
(pin lines 1030-1031: the 2*sigmoid(...) injection stream) belongs to the
decoder-layer mixers (pin lines 1288, 1303, 1305, 1309) and is NOT this
increment's surface.

Per-position-ness (why the parity table's masked case is a real
discriminator, not a degeneracy): every op in the block is local to one
[B, *, 4H] row -- the group RMSNorm (group_size = hidden) normalizes within
a row, the two low-rank projections and the mean over the hc streams are
row-wise. No op mixes positions. A masked (zeroed) row is therefore exactly
inert: the block emits zero there for ANY row content, and garbage in a
masked row cannot change any other row's output. (Contrast the GDN block,
whose masked positions still evolve the recurrent state -- there the mask
carries signal; here it cannot.) The test proves this empirically with a
garbage-probe discriminator; see tests/python/test_hc_block.py.
"""
import torch
import torch.nn.functional as F
from torch import nn

# Pinned free function / class (modeling_qwen4_exp.py): the transcription
# below must use EXACTLY these, so it cannot drift from the pin's math.
#   apply_mask_to_padding_states  (pin line 199)
#   Qwen4ExpTextRMSNorm           (pin lines 152-172)
from transformers.models.qwen4_exp import modeling_qwen4_exp as _pin  # noqa: F401


class Qwen4ExpTextGatedResidual(nn.Module):
    """Transcribed from pin Qwen4ExpTextGatedResidual (modeling_qwen4_exp.py
    lines 1003-1031). The forward body is line-for-line the pin's, except
    the input-shape check (pin lines 1017-1020) is dropped -- the parity test
    feeds well-formed [B, T, 4H] streams, so it would be dead code here."""

    def __init__(self, config, use_combine: bool = True):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = _pin.Qwen4ExpTextRMSNorm(
            hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps
        )
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, hc_hidden_size, bias=False)
        self.block_inject_weight = (
            nn.Linear(hc_hidden_size, self.hc_count, bias=False) if use_combine else None
        )

    def forward(self, hyper_input: torch.Tensor) -> torch.Tensor:
        # pin 1021
        hyper_input_normed = self.hc_norm(hyper_input)
        # pin 1022: down projection, /hc_count, silu
        input_mix_weight = F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)
        # pin 1023: up projection, sigmoid
        input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))
        # pin 1024: unflatten the last dim into (hc_count, hidden_size)
        input_mix_weight = input_mix_weight.unflatten(-1, (self.hc_count, self.hidden_size))
        # pin 1025-1026: per-stream weights * normed stream, mean over the
        # hc streams (dim=-2) -> [B, T, hidden_size]
        mixed_input = (input_mix_weight * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(
            dim=-2
        )
        if self.block_inject_weight is None:
            return mixed_input
        injection_weights = 2 * torch.sigmoid(self.block_inject_weight(hyper_input_normed) / self.hc_count)
        return mixed_input, hyper_input, injection_weights


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mirror of pin line 199 (the pin's own function is called directly by
    the tests on both sides; kept here so this module documents the entry of
    the branch it transcribes)."""
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


__all__ = ["Qwen4ExpTextGatedResidual", "apply_mask_to_padding_states"]
