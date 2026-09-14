from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from block_driver import BlockDriver
from run_store import RunStore
from gptqmodel.utils.v41_checkpoint import save_routed_batch
from gptqmodel.utils.v41_routed_batch import V41RoutedBatch
from gptqmodel.models.definitions.deepseek_v41 import DeepSeekV41QModel
from transformers import DeepseekV41ForCausalLM
from test_v41_primitives import tiny_config


class BlockDriverTest(unittest.TestCase):
    @torch.no_grad()
    def test_interrupted_search_resumes_and_orders_phases(self):
        torch.manual_seed(73)
        model = DeepseekV41ForCausalLM(tiny_config()).eval()
        DeepSeekV41QModel.convert_model_structure(model)
        block = model.model.layers[0]
        hidden = torch.randn(6, 64)
        logits, weights, indices = block.mlp.gate(hidden)
        batch = V41RoutedBatch(hidden, logits, weights, indices)
        calls, installations = [], []
        fail = [True]
        def search(name, hessian, bits):
            if fail[0] and len(calls) == 3:
                raise RuntimeError("injected interruption")
            if name.endswith(".w2"):
                installed = {(e, p) for e, p, _ in installations}
                self.assertTrue({(e, p) for e in range(4) for p in ("w1", "w3")} <= installed)
            calls.append((name, bits))
            expert = int(name.split(".")[-2])
            return {"bits": torch.tensor(bits)}, {"hessian_weighted_relative_error": float(expert + 1),
                                                   "list_field": [1, {"nested": [2, 3]}]}
        def install(block, expert, projection, packed, *, device):
            installations.append((expert, projection, int(packed["bits"])))
        with tempfile.TemporaryDirectory() as directory:
            journal = RunStore(directory, {"test": "phase-driver"})
            self.addCleanup(journal.close)
            identity = {"test": "phase-driver"}
            provenance = {"batch": 0}
            save_routed_batch(batch, Path(directory) / "routed.safetensors", provenance=provenance)
            journal.record_file("routed", "routed", "routed.safetensors")
            driver = BlockDriver(None, journal, identity, device="cpu", subset_size=2, search=search)
            with patch("block_driver.NAMESPACES", {"base": ("layers", 1, 4)}), \
                 patch("block_driver.layer_quotas", return_value={"w1": 1, "w3": 1, "w2": 2}), \
                 patch("block_driver.install_projection", side_effect=install):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    driver.run(block, "base", 0, ["routed"], routed_provenance={"routed": provenance})
                self.assertIsNone(journal.get("blocks/base/000/gate_up/complete"))
                fail[0] = False
                key = driver.run(block, "base", 0, ["routed"], routed_provenance={"routed": provenance})
                self.assertEqual(key, "blocks/base/000/down/complete")
                self.assertEqual(len(calls), 16)  # 12 base candidates + 4 quota upgrades
                self.assertEqual(len(set(calls)), 16)  # committed candidates not searched twice
                before = list(calls)
                driver.run(block, "base", 0, ["routed"], routed_provenance={"routed": provenance})
                self.assertEqual(calls, before)
                self.assertEqual(len(installations), 24)  # reload installs all 12 selected weights
                self.assertIsNotNone(journal.get(key))
