"""dSpark teacher-forced replay boundary and visibility tests."""
from types import SimpleNamespace
import unittest

import torch

from gptqmodel.utils.v41_dspark import V41DSparkInput


class V41DSparkTest(unittest.TestCase):
    def test_visible_prefix_target_order_and_noise(self):
        config = SimpleNamespace(dspark_target_layer_ids=[7, 3], dspark_block_size=5,
                                 dspark_noise_token_id=15, hc_mult=2)
        embedding = torch.nn.Embedding(16, 8)
        projection = torch.nn.Linear(16, 8, bias=False)
        norm = torch.nn.LayerNorm(8)
        adapter = V41DSparkInput(embedding, projection, norm, config)
        features = {3: torch.randn(2, 9, 8), 7: torch.randn(2, 9, 8)}
        ids = torch.zeros(2, 10, dtype=torch.long)
        ids[:, 4] = torch.tensor([4, 8])
        state = adapter.prepare(features, ids, position=3)
        expected = norm(projection(torch.cat((features[7][:, :4], features[3][:, :4]), dim=-1)))
        torch.testing.assert_close(state.kwargs["main_x"], expected, rtol=0, atol=0)
        draft_ids = torch.tensor([[4, 15, 15, 15, 15], [8, 15, 15, 15, 15]])
        torch.testing.assert_close(state.hidden[:, :, 0], embedding(draft_ids), rtol=0, atol=0)
        self.assertTrue(torch.equal(state.hidden[:, :, 0], state.hidden[:, :, 1]))
        self.assertTrue(torch.equal(state.pre_mix[..., 0], torch.ones(2, 5)))
        self.assertTrue(torch.equal(state.pre_mix[..., 1], torch.zeros(2, 5)))
        for value in features.values():
            value[:, 4:] = float("nan")
        repeated = adapter.prepare(features, ids, position=3)
        torch.testing.assert_close(repeated.kwargs["main_x"], state.kwargs["main_x"], rtol=0, atol=0)
        for position in (0, 9):
            with self.assertRaises(ValueError):
                adapter.prepare(features, ids, position=position)


if __name__ == "__main__":
    unittest.main()
