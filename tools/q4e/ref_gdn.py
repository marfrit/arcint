"""Reference transcription of the qwen4_exp GatedDeltaNet, full-sequence
no-cache branch, for numeric parity tests.

The ORACLE is the pinned transformers reference (tools/export_qwen4_exp.py
REFERENCE_COMMIT; modeling_qwen4_exp.py sha256
`ca9f00bbd73cfcfbad7ba6073b5ecc23ca27bdace58b746fe6ead6ab175efc0c`). The
free functions it uses are imported from the installed pinned module (the
dev-host venv holds transformers 5.17.0 whose models/qwen4_exp files are
byte-identical to the pin -- tests/python/test_gdn_block.py asserts that by
hash before every numeric table); the transcription below is the
`Qwen4ExpTextGatedDeltaNet` class (pin lines 465-628) rewritten against
torch nn primitives.

The transcription is validated AGAINST the pin, not against itself: the
test drives BOTH this class and the pin's `Qwen4ExpTextGatedDeltaNet` on
the same random weights/inputs and requires agreement at 1e-5. A
transcription error therefore goes red the same way an OV emission error
goes red.

The branch pinned here (what tools/q4e/gdn.py emits as opset-13):
forward with `cache_params=None` -> apply_mask_to_padding_states ->
in_proj_qkv -> causal_conv1d_fn (depthwise, bias=None) -> split q/k/v ->
torch_chunk_gated_delta_rule (fp32, l2norm inside, chunk 64) ->
RMSNormGated(o, z) -> out_proj.
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# Pinned free functions (modeling_qwen4_exp.py): the class below must use
# EXACTLY these, so the transcription cannot drift from the pin's math.
#   apply_mask_to_padding_states  (pin line 199)
#   causal_conv1d_fn              (pin line 232)
#   torch_chunk_gated_delta_rule  (pin line 263, export-safe branch 355-366)
#   torch_recurrent_gated_delta_rule (pin line 400)
#   Qwen4ExpTextRMSNormGated      (pin line 180)
from transformers.models.qwen4_exp import modeling_qwen4_exp as _pin  # noqa: F401


def _silu(x: torch.Tensor) -> torch.Tensor:
    """ACT2FN["silu"] (the config default `output_gate_type or hidden_act`)."""
    return F.silu(x)


class Qwen4ExpTextGatedDeltaNet(nn.Module):
    """Transcribed from pin Qwen4ExpTextGatedDeltaNet (modeling_qwen4_exp.py
    lines 465-628), the full-sequence branch (cache_params=None). The
    decode branch (single-token cached recurrent path) is not part of the
    stateless KLD surface and is not transcribed.

    __init__ and the no-cache forward body are line-for-line the pin's,
    except the kernel-forward/func decorators are dropped (the venv has no
    FLA/hub kernels, so the pin itself runs these exact fallback functions)
    and the cache bookkeeping is dropped (forward here takes
    attention_mask, not cache_params).
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # time step projection (discretization)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        # Lower bound kept away from 0 so log(A) never becomes -inf
        A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = _pin.Qwen4ExpTextRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=config.output_gate_type or config.hidden_act,
        )
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.layer_type = config.layer_types[layer_idx]

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = _pin.apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        mixed_qkv = _pin.causal_conv1d_fn(
            mixed_qkv,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            activation=self.activation,
        )

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, _ = _pin.torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output


__all__ = ["Qwen4ExpTextGatedDeltaNet"]
