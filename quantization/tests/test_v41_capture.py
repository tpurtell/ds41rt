"""Raw Hessian, natural routing, and non-mutating capture gates."""
import unittest

import torch

from gptqmodel.models.definitions.deepseek_v41 import DeepSeekV41Expert, DeepSeekV41Experts
from gptqmodel.utils.v41_capture import V41Capture


class V41CaptureTest(unittest.TestCase):
    def test_raw_hessians_route_mass_and_zero_expert(self):
        torch.manual_seed(55)
        block = torch.nn.Module()
        block.mlp = torch.nn.Module()
        block.mlp.experts = DeepSeekV41Experts([DeepSeekV41Expert(8, 12, 1.5) for _ in range(3)])
        hidden = torch.randn(7, 8)
        indices = torch.tensor([[0, 1]] * 7)
        weights = torch.rand(7, 2)
        down_inputs = []
        handle = block.mlp.experts[0].down_proj.register_forward_pre_hook(
            lambda module, args: down_inputs.append(args[0].detach().clone()))
        expected = block.mlp.experts(hidden, indices, weights)
        handle.remove()
        capture = V41Capture(block, [0, 2], device="cpu", chunk_rows=3)
        with capture:
            for _ in range(2):
                actual = block.mlp.experts(hidden, indices, weights)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gate, evidence = capture.projection(0, "w1")
        up, _ = capture.projection(0, "w3")
        down, _ = capture.projection(0, "w2")
        torch.testing.assert_close(gate["H"], 2 * hidden.T @ hidden)
        torch.testing.assert_close(up["H"], gate["H"], rtol=0, atol=0)
        torch.testing.assert_close(down["H"], 2 * down_inputs[0].T @ down_inputs[0])
        self.assertEqual(gate["count"], 14)
        self.assertAlmostEqual(evidence["expert_gate_squared_mass_fraction"],
                               (weights[:, 0].double().square().sum() / weights.double().square().sum()).item())
        zero, zero_evidence = capture.projection(2, "w2")
        self.assertEqual(zero["count"], 0)
        self.assertEqual(zero["H"].count_nonzero(), 0)
        self.assertEqual(zero_evidence["expert_gate_squared_mass_fraction"], 0)
        self.assertFalse(gate["finalized"])
        gate["H"].zero_()
        self.assertGreater(capture.projection(0, "w1")[0]["H"].count_nonzero(), 0)

    def test_failure_removes_all_hooks(self):
        block = torch.nn.Module()
        block.mlp = torch.nn.Module()
        block.mlp.experts = DeepSeekV41Experts([DeepSeekV41Expert(8, 12, 1.5)])
        capture = V41Capture(block, [0], device="cpu")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            with capture:
                block.mlp.experts(torch.ones(1, 8), torch.zeros(1, 1, dtype=torch.long), torch.full((1, 1), float("nan")))
        self.assertFalse(capture.handles)
        with self.assertRaisesRegex(RuntimeError, "failed capture"):
            capture.projection(0, "w1")
        for module in block.modules():
            self.assertFalse(module._forward_pre_hooks)
