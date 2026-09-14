"""dSpark teacher-forced replay boundary and visibility tests."""
from types import SimpleNamespace
import unittest

import torch

from gptqmodel.utils.v41_dspark import V41DSparkInput
from gptqmodel.utils.v41_native import dspark_joint_ring


class V41DSparkTest(unittest.TestCase):
    def test_joint_rings_match_literal_reference_cache_writes(self):
        main = torch.arange(300 * 8).reshape(1, 300, 8).float()
        anchors = torch.tensor([1, 7, 127, 128, 129, 256, 299])
        rings, indices = dspark_joint_ring(main, anchors, 128)
        for batch, anchor in enumerate(anchors.tolist()):
            expected = torch.zeros(128, 8)
            for position in range(anchor + 1):
                expected[position % 128] = main[0, position]
            torch.testing.assert_close(rings[batch], expected, rtol=0, atol=0)
            active = indices[batch][indices[batch] >= 0]
            self.assertEqual(active.tolist(), list(range(min(anchor + 1, 128))))
        changed = main.clone()
        changed[:, 130:] = float("nan")
        earlier, _ = dspark_joint_ring(changed, anchors[:5], 128)
        torch.testing.assert_close(earlier, rings[:5], rtol=0, atol=0)

    def test_joint_shared_prefix_and_anchor_visibility(self):
        config = SimpleNamespace(dspark_target_layer_ids=[7, 3], dspark_block_size=5,
                                 dspark_noise_token_id=15, hc_mult=2)
        adapter = V41DSparkInput(torch.nn.Embedding(16, 8), torch.nn.Linear(16, 8, bias=False),
                                 torch.nn.LayerNorm(8), config)
        features = {3: torch.randn(1, 12, 8), 7: torch.randn(1, 12, 8)}
        ids = torch.arange(12)[None]
        positions = torch.tensor([1, 4, 8])
        state = adapter.prepare_joint(features, ids, positions=positions)
        self.assertEqual(tuple(state.hidden.shape), (3, 5, 2, 8))
        self.assertEqual(tuple(state.kwargs["main_x"].shape), (1, 9, 8))
        expected = adapter.main_norm(adapter.main_proj(torch.cat((features[7][:, :9], features[3][:, :9]), -1)))
        torch.testing.assert_close(state.kwargs["main_x"], expected, rtol=0, atol=0)
        torch.testing.assert_close(state.hidden[:, 0, 0], adapter.embed(ids[0, positions + 1]), rtol=0, atol=0)
        for value in features.values():
            value[:, 9:] = float("nan")
        repeat = adapter.prepare_joint(features, ids, positions=positions)
        torch.testing.assert_close(state.kwargs["main_x"], repeat.kwargs["main_x"], rtol=0, atol=0)
        for bad in ([0, 2], [2, 2], [3, 1], [11]):
            with self.assertRaises(ValueError):
                adapter.prepare_joint(features, ids, positions=torch.tensor(bad))

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
