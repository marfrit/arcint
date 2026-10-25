# SPDX-License-Identifier: Apache-2.0
"""ECAPA-TDNN speaker-embedding head.

Turns stacked WavLM-Large hidden states into a 256-dimensional speaker
embedding.  The architecture follows Desplanques, Thienpondt and Demuynck,
"ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
TDNN Based Speaker Verification", Interspeech 2020, with the deviations
required to reproduce the checkpoint described in the clean-room
specification (SE-Res2 blocks chained rather than summed).

Written from that specification alone, without reference to any existing
ECAPA-TDNN or WavLM speaker-head implementation.
"""

import torch
from torch import nn


class _ConvReLUBN(nn.Module):
    """Conv(k, d) with stride 1 and T-preserving zero padding, then ReLU, then BN."""

    def __init__(self, in_ch, out_ch, kernel, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel,
            stride=1,
            padding=dilation * (kernel - 1) // 2,
            dilation=dilation,
            bias=True,
        )
        self.bn = nn.BatchNorm1d(out_ch, eps=1e-5)

    def forward(self, x):
        return self.bn(torch.relu(self.conv(x)))


class _Res2(nn.Module):
    """Res2 stage: split into consecutive groups with a running sum."""

    def __init__(self, channels, scale, dilation):
        super().__init__()
        self.scale = scale
        self.width = channels // scale
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.width,
                    self.width,
                    3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=True,
                )
                for _ in range(scale - 1)
            ]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(self.width, eps=1e-5) for _ in range(scale - 1)]
        )

    def forward(self, x):
        groups = torch.split(x, self.width, dim=1)
        running = groups[0]
        collected = []
        for i in range(self.scale - 1):
            if i > 0:
                running = running + groups[i]
            running = self.bns[i](torch.relu(self.convs[i](running)))
            collected.append(running)
        collected.append(groups[self.scale - 1])
        return torch.cat(collected, dim=1)


class _SEConnect(nn.Module):
    """Squeeze-excitation gate from a time-averaged vector."""

    def __init__(self, channels, bottleneck):
        super().__init__()
        self.linear1 = nn.Linear(channels, bottleneck, bias=True)
        self.linear2 = nn.Linear(bottleneck, channels, bias=True)

    def forward(self, mean):
        return torch.sigmoid(self.linear2(torch.relu(self.linear1(mean))))


class _SERes2Block(nn.Module):
    """One SE-Res2 block with residual connection."""

    def __init__(self, channels, scale, bottleneck, dilation):
        super().__init__()
        self.Conv1dReluBn1 = _ConvReLUBN(channels, channels, 1, 1)
        self.Res2Conv1dReluBn = _Res2(channels, scale, dilation)
        self.Conv1dReluBn2 = _ConvReLUBN(channels, channels, 1, 1)
        self.SE_Connect = _SEConnect(channels, bottleneck)

    def forward(self, x):
        residual = x
        y = self.Conv1dReluBn1(x)
        y = self.Res2Conv1dReluBn(y)
        y = self.Conv1dReluBn2(y)
        gate = self.SE_Connect(y.mean(dim=2))
        return y * gate.unsqueeze(2) + residual


class _AttentiveStatsPooling(nn.Module):
    """Attentive statistics pooling over the time axis."""

    def __init__(self, features, attention):
        super().__init__()
        self.linear1 = nn.Conv1d(features, attention, 1, stride=1, bias=True)
        self.linear2 = nn.Conv1d(attention, features, 1, stride=1, bias=True)

    def forward(self, h):
        e = self.linear2(torch.tanh(self.linear1(h)))
        alpha = torch.softmax(e, dim=2)
        mean = torch.sum(alpha * h, dim=2)
        second = torch.sum(alpha * h * h, dim=2)
        variance = torch.clamp_min(second - mean * mean, 1e-9)
        std = torch.sqrt(variance)
        return torch.cat([mean, std], dim=1)


class SpeakerHead(nn.Module):
    """WavLM hidden states -> 256-d speaker embedding (inference only)."""

    def __init__(
        self,
        num_states=25,
        feat_dim=1024,
        channels=512,
        pooled_channels=1536,
        emb_dim=256,
    ):
        super().__init__()
        self.num_states = num_states
        self.feat_dim = feat_dim
        self.channels = channels
        self.pooled_channels = pooled_channels
        self.emb_dim = emb_dim

        self.feature_weight = nn.Parameter(torch.ones(num_states))
        self.layer1 = _ConvReLUBN(feat_dim, channels, 5, 1)
        self.layer2 = _SERes2Block(channels, 8, 128, 2)
        self.layer3 = _SERes2Block(channels, 8, 128, 3)
        self.layer4 = _SERes2Block(channels, 8, 128, 4)
        self.conv = nn.Conv1d(3 * channels, pooled_channels, 1, stride=1, bias=True)
        self.pooling = _AttentiveStatsPooling(pooled_channels, 128)
        self.bn = nn.BatchNorm1d(2 * pooled_channels, eps=1e-5)
        self.linear = nn.Linear(2 * pooled_channels, emb_dim, bias=True)

    def _layer_weight(self, states):
        weights = torch.softmax(self.feature_weight, dim=0)
        return torch.sum(states * weights.view(-1, 1, 1, 1), dim=0)

    def _instance_norm(self, x):
        x = x.transpose(1, 2) + 1e-6
        mean = x.mean(dim=2, keepdim=True)
        variance = x.var(dim=2, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(variance + 1e-5)

    def forward(self, states):
        x = self._instance_norm(self._layer_weight(states))
        x = self.layer1(x)
        block2 = self.layer2(x)
        block3 = self.layer3(block2)
        block4 = self.layer4(block3)
        x = torch.cat([block2, block3, block4], dim=1)
        x = torch.relu(self.conv(x))
        x = self.pooling(x)
        x = self.bn(x)
        return self.linear(x)
