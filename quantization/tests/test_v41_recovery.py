import unittest

import torch
from torch.nn import functional as F

from gptqmodel.models.definitions.deepseek_v41 import DeepSeekV41Expert, DeepSeekV41Experts
from gptqmodel.utils.v41_capture import V41Capture
from gptqmodel.utils.v41_recovery import V41Recovery
from gptqmodel.utils.v41_routed_batch import V41RoutedBatch


class Router(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k = 1
        self.score_fn = torch.sigmoid
        self.register_buffer("e_score_correction_bias", torch.zeros(4))

    def forward(self, hidden):
        logits = hidden.new_tensor([[4, 3, 2, 1], [4, 3, 2, 1], [4, 2, 3, 1], [4, 2, 3, 1]])
        scores = self.score_fn(logits)
        indices = scores.topk(1, dim=-1).indices
        return logits, scores.gather(1, indices), indices


class RecoveryTest(unittest.TestCase):
    def test_natural_output_candidate_identity_and_unit_down(self):
        torch.manual_seed(63)
        block = torch.nn.Module()
        block.mlp = torch.nn.Module()
        block.mlp.gate = Router()
        block.mlp.experts = DeepSeekV41Experts([DeepSeekV41Expert(8, 12, 1.5) for _ in range(4)])
        inputs = torch.randn(4, 8)
        _, weights, indices = block.mlp.gate(inputs)
        expected = block.mlp.experts(inputs, indices, weights)
        capture = V41Capture(block, [0, 1, 2, 3], device="cpu")
        recovery = V41Recovery(block, [0, 1, 2, 3], target_count=4)
        with capture, recovery:
            _, weights, indices = block.mlp.gate(inputs)
            actual = block.mlp.experts(inputs, indices, weights)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        h, evidence = recovery.projection(capture, 1, "w1")
        torch.testing.assert_close(h["H"], inputs[:2].T @ inputs[:2] + 2 * torch.eye(8))
        self.assertEqual((evidence["natural_rows"], evidence["augmented_rows"], evidence["identity_rows"]), (0, 2, 2))
        self.assertEqual(h["count"], 4)
        self.assertEqual(evidence["expert_gate_squared_mass_fraction"], 0)
        self.assertEqual(capture.projection(1, "w1")[0]["count"], 0)
        expert = block.mlp.experts[1]
        gate = expert.gate_proj(inputs[:2]).float().clamp(max=1.5)
        up = expert.up_proj(inputs[:2]).float().clamp(min=-1.5, max=1.5)
        down_inputs = F.silu(gate) * up
        h, _ = recovery.projection(capture, 1, "w2")
        torch.testing.assert_close(h["H"], down_inputs.T @ down_inputs + 2 * torch.eye(12))
        zero, evidence = recovery.projection(capture, 3, "w1")
        torch.testing.assert_close(zero["H"], 4 * torch.eye(8), rtol=0, atol=0)
        self.assertEqual(evidence["candidate_rows_observed"], 0)
        self.assertEqual(evidence["identity_rows"], 4)
        warm, evidence = recovery.projection(capture, 0, "w1")
        torch.testing.assert_close(warm["H"], inputs.T @ inputs)
        self.assertEqual(evidence["identity_rows"], 0)
        self.assertEqual(len(recovery.coordinates[1, 2]), 2)
        logits, weights, indices = block.mlp.gate(inputs)
        direct = V41Recovery(block, [0, 1, 2, 3], target_count=4)
        direct.observe_routed(V41RoutedBatch(inputs, logits, weights, indices))
        for expert in range(4):
            left, le = direct.projection(capture, expert, "w1")
            right, re = recovery.projection(capture, expert, "w1")
            torch.testing.assert_close(left["H"], right["H"], rtol=0, atol=0)
            self.assertEqual(le, re)
