import unittest
from unittest.mock import patch

import torch

from gptqmodel.models.definitions.deepseek_v41 import DeepSeekV41Expert, DeepSeekV41Experts
from gptqmodel.utils.v41_mixed_replay import install_projection


class MixedReplayTest(unittest.TestCase):
    def setUp(self):
        self.block = torch.nn.Module()
        self.block.mlp = torch.nn.Module()
        self.block.mlp.experts = DeepSeekV41Experts([DeepSeekV41Expert(16, 32, 1.5)])
        self.packed = dict(trellis=torch.zeros(1, 2, 48, dtype=torch.int16),
                           suh=torch.ones(16, dtype=torch.float16),
                           svh=torch.ones(32, dtype=torch.float16), mcg=torch.tensor(0, dtype=torch.int32))

    def test_replaces_only_target_from_packed_reconstruction(self):
        expert = self.block.mlp.experts[0]
        up, down = expert.up_proj, expert.down_proj
        weight = torch.randn(16, 32).to(torch.bfloat16)
        with patch("gptqmodel.utils.v41_mixed_replay.reconstruct_exl3_tensors", return_value=weight) as reconstruct:
            result = install_projection(self.block, 0, "w1", self.packed, device="cpu")
            reconstruct.assert_called_once_with(self.packed, device="cpu", dtype=torch.bfloat16)
        self.assertIs(expert.gate_proj, result)
        self.assertIs(expert.up_proj, up)
        self.assertIs(expert.down_proj, down)
        torch.testing.assert_close(result.weight, weight.T, rtol=0, atol=0)
        self.assertFalse(result.weight.requires_grad)

    def test_invalid_payload_leaves_source_unchanged(self):
        original = self.block.mlp.experts[0].gate_proj
        for payload in ({}, {**self.packed, "extra": torch.tensor(0)},
                        {**self.packed, "trellis": torch.zeros(1, 2, 32, dtype=torch.int16)},
                        {**self.packed, "suh": torch.full((16,), float("nan"), dtype=torch.float16)}):
            with self.assertRaises(ValueError):
                install_projection(self.block, 0, "w1", payload, device="cpu")
            self.assertIs(self.block.mlp.experts[0].gate_proj, original)
