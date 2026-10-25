# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SpeakerHead (tools/speaker_head.py), written independently
from the specification. They need torch: on a host without it they skip."""

import sys
import unittest
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from speaker_head import SpeakerHead  # noqa: E402


def _bn_keys(layout, prefix, n):
    layout[prefix + ".weight"] = (n,)
    layout[prefix + ".bias"] = (n,)
    layout[prefix + ".running_mean"] = (n,)
    layout[prefix + ".running_var"] = (n,)
    layout[prefix + ".num_batches_tracked"] = ()


def expected_layout(num_states=25, feat_dim=1024, channels=512, pooled=1536, emb=256):
    """The state_dict table transcribed from the specification."""
    layout = {}
    layout["feature_weight"] = (num_states,)
    layout["layer1.conv.weight"] = (channels, feat_dim, 5)
    layout["layer1.conv.bias"] = (channels,)
    _bn_keys(layout, "layer1.bn", channels)

    for layer in (2, 3, 4):
        p = "layer%d" % layer
        layout[p + ".Conv1dReluBn1.conv.weight"] = (channels, channels, 1)
        layout[p + ".Conv1dReluBn1.conv.bias"] = (channels,)
        _bn_keys(layout, p + ".Conv1dReluBn1.bn", channels)
        for i in range(7):
            layout[p + ".Res2Conv1dReluBn.convs.%d.weight" % i] = (64, 64, 3)
            layout[p + ".Res2Conv1dReluBn.convs.%d.bias" % i] = (64,)
            _bn_keys(layout, p + ".Res2Conv1dReluBn.bns.%d" % i, 64)
        layout[p + ".Conv1dReluBn2.conv.weight"] = (channels, channels, 1)
        layout[p + ".Conv1dReluBn2.conv.bias"] = (channels,)
        _bn_keys(layout, p + ".Conv1dReluBn2.bn", channels)
        layout[p + ".SE_Connect.linear1.weight"] = (128, channels)
        layout[p + ".SE_Connect.linear1.bias"] = (128,)
        layout[p + ".SE_Connect.linear2.weight"] = (channels, 128)
        layout[p + ".SE_Connect.linear2.bias"] = (channels,)

    layout["conv.weight"] = (pooled, 3 * channels, 1)
    layout["conv.bias"] = (pooled,)
    layout["pooling.linear1.weight"] = (128, pooled, 1)
    layout["pooling.linear1.bias"] = (128,)
    layout["pooling.linear2.weight"] = (pooled, 128, 1)
    layout["pooling.linear2.bias"] = (pooled,)
    _bn_keys(layout, "bn", 2 * pooled)
    layout["linear.weight"] = (emb, 2 * pooled)
    layout["linear.bias"] = (emb,)
    return layout


def _bn_reference(y, bn):
    shape = [1, -1] + [1] * (y.dim() - 2)
    mean = bn.running_mean.view(*shape)
    var = bn.running_var.view(*shape)
    scale = bn.weight.view(*shape)
    shift = bn.bias.view(*shape)
    return (y - mean) / torch.sqrt(var + 1e-5) * scale + shift


class StateDictTests(unittest.TestCase):
    def test_state_dict_keys_and_shapes(self):
        expected = expected_layout()
        actual = SpeakerHead().state_dict()
        self.assertEqual(set(actual), set(expected))
        for key, shape in expected.items():
            self.assertEqual(tuple(actual[key].shape), shape, key)

    def test_random_state_dict_loads_strictly(self):
        expected = expected_layout()
        values = {}
        for key, shape in expected.items():
            if key.endswith("num_batches_tracked"):
                values[key] = torch.randint(0, 100, shape, dtype=torch.long)
            else:
                values[key] = torch.randn(shape)
        head = SpeakerHead()
        head.load_state_dict(values, strict=True)
        loaded = head.state_dict()
        for key in expected:
            self.assertTrue(torch.equal(loaded[key], values[key]), key)


class ShapeTests(unittest.TestCase):
    def test_output_shape_and_time_preserved(self):
        head = SpeakerHead()
        head.eval()
        time_steps = 17
        states = torch.randn(25, 1, time_steps, 1024)
        seen = {}

        def make_hook(name):
            def hook(module, inputs, output):
                seen[name] = tuple(output.shape)

            return hook

        handles = []
        for name in ("layer1", "layer2", "layer3", "layer4", "conv", "pooling", "bn", "linear"):
            handles.append(getattr(head, name).register_forward_hook(make_hook(name)))
        with torch.no_grad():
            out = head(states)
        for handle in handles:
            handle.remove()

        self.assertEqual(tuple(out.shape), (1, 256))
        for name in ("layer1", "layer2", "layer3", "layer4", "conv"):
            self.assertEqual(seen[name][-1], time_steps, name)
        self.assertEqual(seen["pooling"], (1, 3072))
        self.assertEqual(seen["bn"], (1, 3072))
        self.assertEqual(seen["linear"], (1, 256))


class StageReferenceTests(unittest.TestCase):
    def test_instance_normalisation(self):
        head = SpeakerHead()
        x = torch.randn(2, 13, 1024)
        actual = head._instance_norm(x)

        shifted = x.transpose(1, 2) + 1e-6
        mean = shifted.mean(dim=2, keepdim=True)
        variance = ((shifted - mean) ** 2).mean(dim=2, keepdim=True)
        reference = (shifted - mean) / torch.sqrt(variance + 1e-5)
        self.assertTrue(torch.allclose(actual, reference, atol=1e-6))

    def test_layer_weighting(self):
        head = SpeakerHead()
        states = torch.randn(25, 2, 11, 1024)
        actual = head._layer_weight(states)

        weights = torch.softmax(head.feature_weight, dim=0)
        reference = torch.sum(states * weights.view(-1, 1, 1, 1), dim=0)
        self.assertTrue(torch.allclose(actual, reference, atol=1e-6))

    def test_res2_running_sum(self):
        head = SpeakerHead()
        head.eval()
        res2 = head.layer2.Res2Conv1dReluBn
        x = torch.randn(2, 512, 15)
        actual = res2(x)

        groups = torch.split(x, 64, dim=1)
        running = groups[0]
        collected = []
        for i in range(7):
            if i > 0:
                running = running + groups[i]
            conv = res2.convs[i]
            y = F.conv1d(
                running,
                conv.weight,
                conv.bias,
                padding=conv.dilation[0],
                dilation=conv.dilation[0],
            )
            running = _bn_reference(torch.relu(y), res2.bns[i])
            collected.append(running)
        collected.append(groups[7])
        reference = torch.cat(collected, dim=1)
        self.assertTrue(torch.allclose(actual, reference, atol=1e-5))

    def test_squeeze_excitation(self):
        head = SpeakerHead()
        se = head.layer2.SE_Connect
        mean = torch.randn(2, 512)
        actual = se(mean)

        hidden = torch.relu(F.linear(mean, se.linear1.weight, se.linear1.bias))
        reference = torch.sigmoid(F.linear(hidden, se.linear2.weight, se.linear2.bias))
        self.assertTrue(torch.allclose(actual, reference, atol=1e-6))

    def test_attentive_statistics_pooling_and_clamp(self):
        head = SpeakerHead()
        pooling = head.pooling
        h = torch.randn(2, 1536, 25)
        actual = pooling(h)

        e = torch.tanh(F.conv1d(h, pooling.linear1.weight, pooling.linear1.bias))
        e = F.conv1d(e, pooling.linear2.weight, pooling.linear2.bias)
        alpha = torch.softmax(e, dim=2)
        mean = torch.sum(alpha * h, dim=2)
        variance = torch.sum(alpha * h * h, dim=2) - mean * mean
        reference = torch.cat([mean, torch.sqrt(variance.clamp_min(1e-9))], dim=1)
        self.assertTrue(torch.allclose(actual, reference, atol=1e-5))

        # Constant frames: the raw variance is zero, so the 1e-9 clamp is visible.
        # 8 frames of 0.5 keep every step exact in float32 (uniform attention
        # 1/8, 0.5 squared): with 10 frames of 0.7 the second moment minus the
        # squared mean leaves a rounding residue up to ~1e-7 on some torch
        # builds, which the clamp does not hide.
        constant = torch.full((1, 1536, 8), 0.5)
        pooled = pooling(constant)
        self.assertTrue(torch.allclose(pooled[:, :1536], torch.full((1, 1536), 0.5), atol=1e-6))
        self.assertTrue(
            torch.allclose(
                pooled[:, 1536:],
                torch.full((1, 1536), (1e-9) ** 0.5),
                atol=1e-8,
            )
        )


class BatchAndInferenceTests(unittest.TestCase):
    def test_batch_two_matches_two_single_calls(self):
        head = SpeakerHead()
        head.eval()
        states = torch.randn(25, 2, 21, 1024)
        with torch.no_grad():
            batched = head(states)
            first = head(states[:, 0:1])
            second = head(states[:, 1:2])
        self.assertTrue(torch.allclose(batched, torch.cat([first, second], dim=0), atol=1e-5))

    def test_eval_mode_is_deterministic_and_keeps_running_statistics(self):
        head = SpeakerHead()
        head.eval()
        states = torch.randn(25, 1, 19, 1024)

        def snapshot():
            return {
                name: (module.running_mean.clone(), module.running_var.clone(),
                       module.num_batches_tracked.clone())
                for name, module in head.named_modules()
                if isinstance(module, torch.nn.BatchNorm1d)
            }

        before = snapshot()
        with torch.no_grad():
            first = head(states)
            second = head(states)
        after = snapshot()

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(set(before), set(after))
        for name in before:
            for old, new in zip(before[name], after[name]):
                self.assertTrue(torch.equal(old, new), name)


if __name__ == "__main__":
    unittest.main()
